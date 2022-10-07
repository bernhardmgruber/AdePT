// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example.cuh"

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

// adapted from boost::mp11
template <template <typename...> typename L, typename... T, typename F>
__device__ __forceinline__ constexpr void mpForEachInlined(L<T...>, F &&f)
{
  using A = int[sizeof...(T)];
  (void)A{((void)f(T{}), 0)...};
}

__global__ void TransportGammas(View gammas, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                ScoringPerVolume *scoringPerVolume, SOAData const soaData)
{
#ifdef VECGEOM_FLOAT_PRECISION
  const Precision kPush = 10 * vecgeom::kTolerance;
#else
  const Precision kPush = 0.;
#endif
  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    auto &&currentTrack = gammas[slot];
    const auto energy   = decayCopy(currentTrack(Energy{}));
    auto pos            = decayCopy(currentTrack(Pos{}));
    const auto dir      = decayCopy(currentTrack(Dir{}));
    auto navState       = decayCopy(currentTrack(NavState{}));
    const auto volume   = navState.Top();
    const int volumeID  = volume->id();
    // the MCC vector is indexed by the logical volume id
    const int lvolID     = volume->GetLogicalVolume()->id();
    const int theMCIndex = MCIndex[lvolID];

    auto survive = [&](bool push = true) {
      currentTrack(Pos{})      = pos;
      currentTrack(NavState{}) = navState;
      if (push) activeQueue->push_back(slot);
    };

    // Signal that this slot doesn't undergo an interaction (yet)
    soaData.nextInteraction[i] = -1;

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack gammaTrack;
    G4HepEmTrack *theTrack = gammaTrack.GetTrack();
    theTrack->SetEKin(energy);
    theTrack->SetMCIndex(theMCIndex);

    // Sample the `number-of-interaction-left` and put it into the track.
    // boost::mp11::mp_for_each<boost::mp11::mp_iota_c<3>>([&](auto ic) {
    mpForEachInlined(boost::mp11::mp_iota_c<3>{}, [&](auto ic) {
      constexpr int ip = decltype(ic)::value;
      double numIALeft = currentTrack(NumIALeft{}, llama::RecordCoord<ip>{});
      if (numIALeft <= 0) {
        auto rngState            = decayCopy(currentTrack(RngState{}));
        numIALeft                = -std::log(rngState.Rndm());
        currentTrack(RngState{}) = rngState;
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
        BVHNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState, nextState, kPush);
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

    soaData.nextInteraction[i] = winnerProcessIndex;
    survive(false);
  }
}

template <int ProcessIndex>
__device__ void GammaInteraction(int const globalSlot, SOAData const &soaData, int const soaSlot, View particles,
                                 Secondaries secondaries, adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                 ScoringPerVolume *scoringPerVolume)
{
  auto &&currentTrack = particles[globalSlot];
  const auto energy   = decayCopy(currentTrack(Energy{}));
  const auto pos      = decayCopy(currentTrack(Pos{}));
  const auto dir      = decayCopy(currentTrack(Dir{}));
  const auto navState = decayCopy(currentTrack(NavState{}));
  const auto volume   = navState.Top();
  const int volumeID  = volume->id();
  // the MCC vector is indexed by the logical volume id
  const int lvolID     = volume->GetLogicalVolume()->id();
  const int theMCIndex = MCIndex[lvolID];

  auto rngState = decayCopy(currentTrack(RngState{}));
  auto survive  = [&] {
    currentTrack(RngState{}) = rngState;
    activeQueue->push_back(globalSlot);
  };

  RanluxppDouble newRNG{rngState.Branch()};
  G4HepEmRandomEngine rnge{&rngState};

  if constexpr (ProcessIndex == 0) {
    // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
    if (energy < 2 * copcore::units::kElectronMassC2) {
      survive();
      return;
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
    electron(Dir{})      = Vec3(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);

    InitAsSecondary(positron, pos, navState);
    // Reuse the RNG state of the dying track.
    positron(RngState{}) = rngState;
    positron(Energy{})   = posKinEnergy;
    positron(Dir{})      = Vec3(dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]);

    // The current track is killed by not enqueuing into the next activeQueue.
  } else if constexpr (ProcessIndex == 1) {
    // Invoke Compton scattering of gamma.
    constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
    if (energy < LowEnergyThreshold) {
      survive();
      return;
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
      electron(Dir{})      = (energy * dir - newEnergyGamma * newDirGamma).Normalized();
    } else {
      atomicAdd(&globalScoring->energyDeposit, energyEl);
      atomicAdd(&scoringPerVolume->energyDeposit[volumeID], energyEl);
    }

    // Check the new gamma energy and deposit if below threshold.
    if (newEnergyGamma > LowEnergyThreshold) {
      currentTrack(Energy{}) = newEnergyGamma;
      currentTrack(Dir{})    = newDirGamma;
      survive();
    } else {
      atomicAdd(&globalScoring->energyDeposit, newEnergyGamma);
      atomicAdd(&scoringPerVolume->energyDeposit[volumeID], newEnergyGamma);
      // The current track is killed by not enqueuing into the next activeQueue.
    }
  } else if constexpr (ProcessIndex == 2) {
    // Invoke photoelectric process.
    const double theLowEnergyThreshold = 1 * copcore::units::eV;

    // (Re)compute total macroscopic cross section
    const int theMatIndx = g4HepEmData.fTheMatCutData->fMatCutData[theMCIndex].fHepEmMatIndex;
    double mxsec = G4HepEmGammaManager::GetMacXSecPE(&g4HepEmData, theMatIndx, energy);

    const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
        &g4HepEmData, theMCIndex, mxsec, energy, &rnge);

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
      electron(Dir{})      = Vec3(dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]);
    } else {
      edep = energy;
    }
    atomicAdd(&globalScoring->energyDeposit, edep);
    // The current track is killed by not enqueuing into the next activeQueue.
  }
}

__global__ void PairCreation(View particles, const adept::MParray *active, Secondaries secondaries,
                             adept::MParray *activeQueue, GlobalScoring *globalScoring,
                             ScoringPerVolume *scoringPerVolume, SOAData const soaData)
{
  InteractionLoop<0>(&GammaInteraction<0>, active, soaData, particles, secondaries, activeQueue, globalScoring,
                     scoringPerVolume);
}
__global__ void ComptonScattering(View particles, const adept::MParray *active, Secondaries secondaries,
                                  adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                  ScoringPerVolume *scoringPerVolume, SOAData const soaData)
{
  InteractionLoop<1>(&GammaInteraction<1>, active, soaData, particles, secondaries, activeQueue, globalScoring,
                     scoringPerVolume);
}
__global__ void PhotoelectricEffect(View particles, const adept::MParray *active, Secondaries secondaries,
                                    adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                    ScoringPerVolume *scoringPerVolume, SOAData const soaData)
{
  InteractionLoop<2>(&GammaInteraction<2>, active, soaData, particles, secondaries, activeQueue, globalScoring,
                     scoringPerVolume);
}
