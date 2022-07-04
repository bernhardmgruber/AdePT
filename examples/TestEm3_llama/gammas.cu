// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "TestEm3.cuh"

#include <AdePT/BVHNavigator.h>

#include <CopCore/PhysicalConstants.h>

#include <G4HepEmGammaManager.hh>
#include <G4HepEmGammaTrack.hh>
#include <G4HepEmTrack.hh>
#include <G4HepEmGammaInteractionCompton.hh>
#include <G4HepEmGammaInteractionConversion.hh>
#include <G4HepEmGammaInteractionPhotoelectric.hh>
// Pull in implementation.
#include <G4HepEmGammaManager.icc>
#include <G4HepEmGammaInteractionCompton.icc>
#include <G4HepEmGammaInteractionConversion.icc>
#include <G4HepEmGammaInteractionPhotoelectric.icc>

__global__ void TransportGammas(View gammas, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                ScoringPerVolume *scoringPerVolume)
{
  constexpr auto mapping = RngStateMapping{};
  __shared__ std::byte sharedRngStorage[mapping.blobSize(0)];
  llama::View sharedRngs{mapping, llama::Array{&sharedRngStorage[0]}};

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot       = (*active)[i];
    auto &&currentTrack  = gammas[slot];
    auto &&rngRef        = sharedRngs[threadIdx.x];
    rngRef               = currentTrack(RngState{});
    auto energy          = currentTrack(Energy{});
    auto pos             = currentTrack(Pos{});
    auto dir             = currentTrack(Dir{});
    auto navState        = currentTrack(NavState{});
    const int volumeID   = navState.Top()->id();
    const int theMCIndex = MCIndex[volumeID];

    auto survive = [&] {
      currentTrack(RngState{}) = rngRef;
      currentTrack(Energy{})   = energy;
      currentTrack(Pos{})      = pos;
      currentTrack(Dir{})      = dir;
      currentTrack(NavState{}) = navState;
      activeQueue->push_back(slot);
    };

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack gammaTrack;
    G4HepEmTrack *theTrack = gammaTrack.GetTrack();
    theTrack->SetEKin(energy);
    theTrack->SetMCIndex(theMCIndex);

    // Sample the `number-of-interaction-left` and put it into the track.
    boost::mp11::mp_for_each<boost::mp11::mp_iota_c<3>>([&](auto ic) {
      constexpr int ip = decltype(ic)::value;
      double numIALeft = currentTrack(NumIALeft{}, llama::RecordCoord<ip>{});
      if (numIALeft <= 0) {
        numIALeft = -std::log(ranlux::NextRandomFloat(rngRef));
      }
      theTrack->SetNumIALeft(numIALeft, ip);
    });

    // Call G4HepEm to compute the physics step limit.
    G4HepEmGammaManager::HowFar(&g4HepEmData, &g4HepEmPars, &gammaTrack);

    // Get result into variables.
    double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();
    int winnerProcessIndex                  = theTrack->GetWinnerProcessIndex();
    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    vecgeom::NavStateIndex nextState;
    double geometryStepLength =
        BVHNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState, nextState);
    pos += geometryStepLength * dir;
    atomicAdd(&globalScoring->neutralSteps, 1);

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to G4HepEm.
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    G4HepEmGammaManager::UpdateNumIALeft(theTrack);

    // Save the `number-of-interaction-left` in our track.
    boost::mp11::mp_for_each<boost::mp11::mp_iota_c<3>>([&](auto ic) {
      constexpr int ip                                    = decltype(ic)::value;
      double numIALeft                                    = theTrack->GetNumIALeft(ip);
      currentTrack(NumIALeft{}, llama::RecordCoord<ip>{}) = numIALeft;
    });

    if (nextState.IsOnBoundary()) {
      // For now, just count that we hit something.
      atomicAdd(&globalScoring->hits, 1);

      // Kill the particle if it left the world.
      if (nextState.Top() != nullptr) {
        BVHNavigator::RelocateToNextVolume(pos, dir, nextState);

        // Move to the next boundary.
        navState = nextState;
        survive();
      }
      continue;
    } else if (winnerProcessIndex < 0) {
      // No discrete process, move on.
      survive();
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    boost::mp11::mp_with_index<3>(winnerProcessIndex, [&](auto ic) {
      currentTrack(NumIALeft{}, llama::RecordCoord<decltype(ic)::value>{}) = -1.0;
    });

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(rngRef);
    // We might need one branched RNG state, prepare while threads are synchronized.
    auto newRNG = ranlux::Branch(rngRef);

    switch (winnerProcessIndex) {
    case 0: {
      // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
      if (energy < 2 * copcore::units::kElectronMassC2) {
        survive();
        continue;
      }

      double logEnergy = std::log(energy);
      double elKinEnergy, posKinEnergy;
      G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, energy, logEnergy, theMCIndex, elKinEnergy,
                                                           posKinEnergy, &rnge);

      double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirSecondaryEl[3], dirSecondaryPos[3];
      G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                          posKinEnergy, &rnge);

      auto &&electron = secondaries.electrons.NextTrack();
      auto &&positron = secondaries.positrons.NextTrack();
      atomicAdd(&globalScoring->numElectrons, 1);
      atomicAdd(&globalScoring->numPositrons, 1);

      InitAsSecondary(electron, pos, navState);
      electron(RngState{}) = newRNG;
      electron(Energy{})   = elKinEnergy;
      electron(Dir{}).Set(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);

      InitAsSecondary(positron, pos, navState);
      // Reuse the RNG state of the dying track.
      positron(RngState{}) = rngRef;
      positron(Energy{})   = posKinEnergy;
      positron(Dir{}).Set(dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]);

      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    case 1: {
      // Invoke Compton scattering of gamma.
      constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
      if (energy < LowEnergyThreshold) {
        survive();
        continue;
      }
      const double origDirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirPrimary[3];
      const double newEnergyGamma =
          G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(energy, dirPrimary, origDirPrimary, &rnge);
      vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

      const double energyEl = energy - newEnergyGamma;
      if (energyEl > LowEnergyThreshold) {
        // Create a secondary electron and sample/compute directions.
        auto &&electron = secondaries.electrons.NextTrack();
        atomicAdd(&globalScoring->numElectrons, 1);

        InitAsSecondary(electron, pos, navState);
        electron(RngState{}) = newRNG;
        electron(Energy{})   = energyEl;
        electron(Dir{})      = energy * dir - newEnergyGamma * newDirGamma;
        electron(Dir{}).Normalize();
      } else {
        atomicAdd(&globalScoring->energyDeposit, energyEl);
        atomicAdd(&scoringPerVolume->energyDeposit[volumeID], energyEl);
      }

      // Check the new gamma energy and deposit if below threshold.
      if (newEnergyGamma > LowEnergyThreshold) {
        energy = newEnergyGamma;
        dir    = newDirGamma;
        survive();
      } else {
        atomicAdd(&globalScoring->energyDeposit, newEnergyGamma);
        atomicAdd(&scoringPerVolume->energyDeposit[volumeID], newEnergyGamma);
        // The current track is killed by not enqueuing into the next activeQueue.
      }
      break;
    }
    case 2: {
      // Invoke photoelectric process.
      const double theLowEnergyThreshold = 1 * copcore::units::eV;

      const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
          &g4HepEmData, theMCIndex, gammaTrack.GetPEmxSec(), energy, &rnge);

      double edep             = bindingEnergy;
      const double photoElecE = energy - edep;
      if (photoElecE > theLowEnergyThreshold) {
        // Create a secondary electron and sample directions.
        auto &&electron = secondaries.electrons.NextTrack();
        atomicAdd(&globalScoring->numElectrons, 1);

        double dirGamma[] = {dir.x(), dir.y(), dir.z()};
        double dirPhotoElec[3];
        G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

        InitAsSecondary(electron, pos, navState);
        electron(RngState{}) = newRNG;
        electron(Energy{})   = photoElecE;
        electron(Dir{}).Set(dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]);
      } else {
        edep = energy;
      }
      atomicAdd(&globalScoring->energyDeposit, edep);
      atomicAdd(&scoringPerVolume->energyDeposit[volumeID], edep);
      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    }
  }
}
