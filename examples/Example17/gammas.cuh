// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "AdeptIntegration.cuh"

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

template <typename Scoring>
__global__ void TransportGammas(adept::TrackManager<Track> *gammas, Secondaries secondaries,
                                adept::MParray *leakedQueue, Scoring *userScoring)
{
  using VolAuxData = AdeptIntegration::VolAuxData;
#ifdef VECGEOM_FLOAT_PRECISION
  const Precision kPush = 10 * vecgeom::kTolerance;
#else
  const Precision kPush = 0.;
#endif
  constexpr Precision kPushOutRegion = 10 * vecgeom::kTolerance;
  int activeSize                     = gammas->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*gammas->fActiveTracks)[i];
    Track &currentTrack = (*gammas)[slot];
    auto volume         = currentTrack.navState.Top();
    // the MCC vector is indexed by the logical volume id
    int lvolID                = volume->GetLogicalVolume()->id();
    VolAuxData const &auxData = userScoring->GetAuxData_dev(lvolID);
    assert(auxData.fGPUregion > 0); // make sure we don't get inconsistent region here

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack gammaTrack;
    G4HepEmTrack *theTrack = gammaTrack.GetTrack();
    theTrack->SetEKin(currentTrack.energy);
    theTrack->SetMCIndex(auxData.fMCIndex);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft = -std::log(currentTrack.Uniform());
      }
      theTrack->SetNumIALeft(numIALeft, ip);
    }

    // Call G4HepEm to compute the physics step limit.
    G4HepEmGammaManager::HowFar(&g4HepEmData, &g4HepEmPars, &gammaTrack);

    // Get result into variables.
    double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();
    int winnerProcessIndex                  = theTrack->GetWinnerProcessIndex();
    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    vecgeom::NavStateIndex nextState;
    double geometryStepLength = BVHNavigator::ComputeStepAndNextVolume(
        currentTrack.pos, currentTrack.dir, geometricalStepLengthFromPhysics, currentTrack.navState, nextState, kPush);
    currentTrack.pos += geometryStepLength * currentTrack.dir;
    userScoring->AccountChargedStep(0);

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (currentTrack.navState = nextState only if relocated
    // in case of a boundary; see below)
    currentTrack.navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to G4HepEm.
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    G4HepEmGammaManager::UpdateNumIALeft(theTrack);

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
    }

    if (nextState.IsOnBoundary()) {
      // For now, just count that we hit something.
      userScoring->AccountHit();

      // Kill the particle if it left the world.
      if (nextState.Top() != nullptr) {
        BVHNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, nextState);

        // Move to the next boundary.
        currentTrack.navState = nextState;
        // Check if the next volume belongs to the GPU region and push it to the appropriate queue
        auto nextvolume               = currentTrack.navState.Top();
        int nextlvolID                = nextvolume->GetLogicalVolume()->id();
        VolAuxData const &nextauxData = userScoring->GetAuxData_dev(nextlvolID);
        if (nextauxData.fGPUregion > 0)
          gammas->fNextTracks->push_back(slot);
        else {
          // To be safe, just push a bit the track exiting the GPU region to make sure
          // Geant4 does not relocate it again inside the same region
          currentTrack.pos += kPushOutRegion * currentTrack.dir;
          leakedQueue->push_back(slot);
        }
      }
      continue;
    } else if (winnerProcessIndex < 0) {
      // No discrete process, move on.
      gammas->fNextTracks->push_back(slot);
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    currentTrack.numIALeft[winnerProcessIndex] = -1.0;

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    const double energy = currentTrack.energy;

    switch (winnerProcessIndex) {
    case 0: {
      // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
      if (energy < 2 * copcore::units::kElectronMassC2) {
        gammas->fNextTracks->push_back(slot);
        continue;
      }

      double logEnergy = std::log(energy);
      double elKinEnergy, posKinEnergy;
      G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, energy, logEnergy, auxData.fMCIndex,
                                                           elKinEnergy, posKinEnergy, &rnge);

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondaryEl[3], dirSecondaryPos[3];
      G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                          posKinEnergy, &rnge);

      Track &electron = secondaries.electrons->NextTrack();
      Track &positron = secondaries.positrons->NextTrack();

      userScoring->AccountProduced(/*numElectrons*/ 1, /*numPositrons*/ 1, /*numGammas*/ 0);

      electron.InitAsSecondary(/*parent=*/currentTrack);
      electron.rngState = newRNG;
      electron.energy   = elKinEnergy;
      electron.dir.Set(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);

      positron.InitAsSecondary(/*parent=*/currentTrack);
      // Reuse the RNG state of the dying track.
      positron.rngState = currentTrack.rngState;
      positron.energy   = posKinEnergy;
      positron.dir.Set(dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]);

      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    case 1: {
      // Invoke Compton scattering of gamma.
      constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
      if (energy < LowEnergyThreshold) {
        gammas->fNextTracks->push_back(slot);
        continue;
      }
      const double origDirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirPrimary[3];
      const double newEnergyGamma =
          G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(energy, dirPrimary, origDirPrimary, &rnge);
      vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

      const double energyEl = energy - newEnergyGamma;
      if (energyEl > LowEnergyThreshold) {
        // Create a secondary electron and sample/compute directions.
        Track &electron = secondaries.electrons->NextTrack();
        userScoring->AccountProduced(/*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

        electron.InitAsSecondary(/*parent=*/currentTrack);
        electron.rngState = newRNG;
        electron.energy   = energyEl;
        electron.dir      = energy * currentTrack.dir - newEnergyGamma * newDirGamma;
        electron.dir.Normalize();
      } else {
        if (auxData.fSensIndex >= 0) userScoring->Score(currentTrack.navState, 0, geometryStepLength, energyEl);
      }

      // Check the new gamma energy and deposit if below threshold.
      if (newEnergyGamma > LowEnergyThreshold) {
        currentTrack.energy = newEnergyGamma;
        currentTrack.dir    = newDirGamma;

        // The current track continues to live.
        gammas->fNextTracks->push_back(slot);
      } else {
        if (auxData.fSensIndex >= 0) userScoring->Score(currentTrack.navState, 0, geometryStepLength, newEnergyGamma);
        // The current track is killed by not enqueuing into the next activeQueue.
      }
      break;
    }
    case 2: {
      // Invoke photoelectric process.
      const double theLowEnergyThreshold = 1 * copcore::units::eV;

      const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
          &g4HepEmData, auxData.fMCIndex, gammaTrack.GetPEmxSec(), energy, &rnge);

      double edep             = bindingEnergy;
      const double photoElecE = energy - edep;
      if (photoElecE > theLowEnergyThreshold) {
        // Create a secondary electron and sample directions.
        Track &electron = secondaries.electrons->NextTrack();
        userScoring->AccountProduced(/*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

        double dirGamma[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
        double dirPhotoElec[3];
        G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

        electron.InitAsSecondary(/*parent=*/currentTrack);
        electron.rngState = newRNG;
        electron.energy   = photoElecE;
        electron.dir.Set(dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]);
      } else {
        edep = energy;
      }
      if (auxData.fSensIndex >= 0) userScoring->Score(currentTrack.navState, 0, geometryStepLength, edep);
      // The current track is killed by not enqueuing into the next activeQueue.
      break;
    }
    }
  }
}
