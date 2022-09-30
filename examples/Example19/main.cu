// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example.h"



#define NOFLUCTUATION

#include <utility>
#include <algorithm>
#include <string>
#include <exception>
#include <iostream>


namespace g4 {
namespace {
#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmRandomEngine.hh>
#include <G4HepEmElectronManager.hh>
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmElectronInteractionBrem.hh>
#include <G4HepEmElectronInteractionIoni.hh>
#include <G4HepEmElectronInteractionUMSC.hh>
#include <G4HepEmPositronInteractionAnnihilation.hh>
#include <G4HepEmGammaManager.hh>
#include <G4HepEmGammaTrack.hh>
#include <G4HepEmTrack.hh>
#include <G4HepEmGammaInteractionCompton.hh>
#include <G4HepEmGammaInteractionConversion.hh>
#include <G4HepEmGammaInteractionPhotoelectric.hh>
#include <G4HepEmState.hh>
#include <G4HepEmData.hh>
#include <G4HepEmElectronInit.hh>
#include <G4HepEmGammaInit.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmMaterialInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmParametersInit.hh>

#include <G4HepEmRunUtils.icc>
#include <G4HepEmInteractionUtils.icc>
#include <G4HepEmElectronManager.icc>
#include <G4HepEmElectronInteractionBrem.icc>
#include <G4HepEmElectronInteractionIoni.icc>
#include <G4HepEmElectronInteractionUMSC.icc>
#include <G4HepEmPositronInteractionAnnihilation.icc>
#include <G4HepEmGammaManager.icc>
#include <G4HepEmGammaInteractionCompton.icc>
#include <G4HepEmGammaInteractionConversion.icc>
#include <G4HepEmGammaInteractionPhotoelectric.icc>
}
}

using namespace g4;

struct G4HepEmData;
struct G4HepEmState {
  struct G4HepEmParameters* fParameters = nullptr; //< Pointer to `G4HepEmParameters` used by `G4HepEmData` (not owned)
  struct ::G4HepEmData* fData = nullptr; //< Pointer to `G4HepEmData` (not owned)
};
void CopyG4HepEmDataToGPU (struct ::G4HepEmData* onCPU);





// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLE_CUH
#define EXAMPLE_CUH

#include "example.h"

#include <AdePT/MParray.h>
#include <CopCore/SystemOfUnits.h>
#include <CopCore/Ranluxpp.h>


#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavStateIndex.h>

constexpr int ThreadsPerBlock = 256;

// A data structure to represent a particle track. The particle type is implicit
// by the queue and not stored in memory.
struct Track {
  using Precision = vecgeom::Precision;
  RanluxppDouble rngState;
  double energy;
  double numIALeft[3];
  double initialRange;
  double dynamicRangeFactor;
  double tlimitMin;

  vecgeom::Vector3D<Precision> pos;
  vecgeom::Vector3D<Precision> dir;
  vecgeom::NavStateIndex navState;
};

__host__ __device__ inline void InitAsSecondary(Track &track, const vecgeom::Vector3D<Precision> &parentPos,
                                                const vecgeom::NavStateIndex &parentNavState)
{
  // The caller is responsible to branch a new RNG state and to set the energy.
  track.numIALeft[0] = -1.0;
  track.numIALeft[1] = -1.0;
  track.numIALeft[2] = -1.0;

  track.initialRange       = -1.0;
  track.dynamicRangeFactor = -1.0;
  track.tlimitMin          = -1.0;

  // A secondary inherits the position of its parent; the caller is responsible
  // to update the directions.
  track.pos      = parentPos;
  track.navState = parentNavState;
}

// Struct for communication between kernels
struct SOAData {
  char *nextInteraction = nullptr;
  double *gamma_PEmxSec = nullptr;
};

// Define inline implementations of the RNG methods for the device.
// (nvcc ignores the __device__ attribute in definitions, so this is only to
// communicate the intent.)
inline __device__ double G4HepEmRandomEngine::flat()
{
  return ((RanluxppDouble *)fObject)->Rndm();
}

inline __device__ void G4HepEmRandomEngine::flatArray(const int size, double *vect)
{
  for (int i = 0; i < size; i++) {
    vect[i] = ((RanluxppDouble *)fObject)->Rndm();
  }
}

// A data structure to manage slots in the track storage.
class SlotManager {
  adept::Atomic_t<int> fNextSlot;
  const int fMaxSlot;

public:
  __host__ __device__ SlotManager(int maxSlot) : fMaxSlot(maxSlot) { fNextSlot = 0; }

  __host__ __device__ int NextSlot()
  {
    int next = fNextSlot.fetch_add(1);
    if (next >= fMaxSlot) return -1;
    return next;
  }
};

// A bundle of pointers to generate particles of an implicit type.
class ParticleGenerator {
  Track *fTracks;
  SlotManager *fSlotManager;
  adept::MParray *fActiveQueue;

public:
  __host__ __device__ ParticleGenerator(Track *tracks, SlotManager *slotManager, adept::MParray *activeQueue)
      : fTracks(tracks), fSlotManager(slotManager), fActiveQueue(activeQueue)
  {
  }

  __host__ __device__ Track &NextTrack()
  {
    int slot = fSlotManager->NextSlot();
    if (slot == -1) {
      COPCORE_EXCEPTION("No slot available in ParticleGenerator::NextTrack");
    }
    fActiveQueue->push_back(slot);
    return fTracks[slot];
  }
};

// A bundle of generators for the three particle types.
struct Secondaries {
  ParticleGenerator electrons;
  ParticleGenerator positrons;
  ParticleGenerator gammas;
};

// Kernels in different TUs.
__global__ void TransportElectrons(Track *electrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                   ScoringPerVolume *scoringPerVolume, SOAData const soaData);
__global__ void TransportPositrons(Track *positrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                   ScoringPerVolume *scoringPerVolume, SOAData const soaData);

__global__ void TransportGammas(Track *gammas, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                ScoringPerVolume *scoringPerVolume, SOAData const soaData);

/// Run an interaction on the particles in soaData whose `nextInteraction` matches the ProcessIndex.
/// The specific interaction that's run is defined by `interactionFunction`.
template <int ProcessIndex, typename Func, typename... Args>
__device__ void InteractionLoop(Func interactionFunction, adept::MParray const *active, SOAData const soaData,
                                Args &&...args)
{
  constexpr unsigned int sharedSize = 8192;
  __shared__ int candidates[sharedSize];
  __shared__ unsigned int counter;
  __shared__ int threadsRunning;
  counter        = 0;
  threadsRunning = 0;

#ifndef NDEBUG
  __shared__ unsigned int todoCounter;
  __shared__ unsigned int particlesDone;
  todoCounter   = 0;
  particlesDone = 0;
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < active->size(); i += blockDim.x * gridDim.x) {
    const auto winnerProcess = soaData.nextInteraction[i];
    if (winnerProcess == ProcessIndex) atomicAdd(&todoCounter, 1);
  }
#endif

  __syncthreads();

  const auto activeSize = active->size();
  int i                 = blockIdx.x * blockDim.x + threadIdx.x;
  bool done             = false;
  do {
    while (i < activeSize && counter < sharedSize - blockDim.x) {
      if (soaData.nextInteraction[i] == ProcessIndex) {
        const auto destination  = atomicAdd(&counter, 1);
        candidates[destination] = i;
      }
      i += blockDim.x * gridDim.x;
    }

    if (i < activeSize) {
      atomicExch(&threadsRunning, 1);
    }

    __syncthreads();
    done = !threadsRunning;

#ifndef NDEBUG
    if (threadIdx.x == 0) {
      atomicAdd(&particlesDone, counter);
    }
    assert(counter < sharedSize);
    __syncthreads();
#endif

    for (int j = threadIdx.x; j < counter; j += blockDim.x) {
      const auto soaSlot    = candidates[j];
      const auto globalSlot = (*active)[soaSlot];
      interactionFunction(globalSlot, soaData, soaSlot, std::forward<Args>(args)...);
    }

    __syncthreads();
    counter        = 0;
    threadsRunning = 0;
    __syncthreads();
  } while (!done);

  assert(particlesDone == todoCounter);
}

__global__ void IonizationEl(Track *particles, const adept::MParray *active, Secondaries secondaries,
                             adept::MParray *activeQueue, GlobalScoring *globalScoring,
                             ScoringPerVolume *scoringPerVolume, SOAData const soaData);
__global__ void BremsstrahlungEl(Track *particles, const adept::MParray *active, Secondaries secondaries,
                                 adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                 ScoringPerVolume *scoringPerVolume, SOAData const soaData);

__global__ void IonizationPos(Track *particles, const adept::MParray *active, Secondaries secondaries,
                              adept::MParray *activeQueue, GlobalScoring *globalScoring,
                              ScoringPerVolume *scoringPerVolume, SOAData const soaData);
__global__ void BremsstrahlungPos(Track *particles, const adept::MParray *active, Secondaries secondaries,
                                  adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                  ScoringPerVolume *scoringPerVolume, SOAData const soaData);
__global__ void AnnihilationPos(Track *particles, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                ScoringPerVolume *scoringPerVolume, SOAData const soaData);

__global__ void PairCreation(Track *particles, const adept::MParray *active, Secondaries secondaries,
                             adept::MParray *activeQueue, GlobalScoring *globalScoring,
                             ScoringPerVolume *scoringPerVolume, SOAData const soaData);
__global__ void ComptonScattering(Track *particles, const adept::MParray *active, Secondaries secondaries,
                                  adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                  ScoringPerVolume *scoringPerVolume, SOAData const soaData);
__global__ void PhotoelectricEffect(Track *particles, const adept::MParray *active, Secondaries secondaries,
                                    adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                    ScoringPerVolume *scoringPerVolume, SOAData const soaData);

// Constant data structures from G4HepEm accessed by the kernels.
// (defined in TestEm3.cu)
__constant__ __device__ struct G4HepEmParameters g4HepEmPars;
__constant__ __device__ struct g4::G4HepEmData g4HepEmData;

__constant__ __device__ int *MCIndex;

// constexpr vecgeom::Precision BzFieldValue = 3.8 * copcore::units::tesla;
constexpr vecgeom::Precision BzFieldValue = 0;

#endif







// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

//#include "example.cuh"

#include <AdePT/BVHNavigator.h>
#include <fieldPropagatorConstBz.h>

#include <CopCore/PhysicalConstants.h>


// Compute the physics and geometry step limit, transport the electrons while
// applying the continuous effects and maybe a discrete process that could
// generate secondaries.
template <bool IsElectron>
static __device__ __forceinline__ void TransportElectrons(Track *electrons, const adept::MParray *active,
                                                          Secondaries &secondaries, adept::MParray *activeQueue,
                                                          GlobalScoring *globalScoring,
                                                          ScoringPerVolume *scoringPerVolume, SOAData soaData)
{
#ifdef VECGEOM_FLOAT_PRECISION
  const Precision kPush = 10 * vecgeom::kTolerance;
#else
  const Precision kPush = 0.;
#endif
  constexpr int Charge  = IsElectron ? -1 : 1;
  constexpr double Mass = copcore::units::kElectronMassC2;
  fieldPropagatorConstBz fieldPropagatorBz(BzFieldValue);

  // The shared memory handles the access pattern to the RNG better than global memory. And we don't have enough
  // registers to keep it local. This is a byte array, because RanluxppDouble has a ctor that we do not want to run.
  __shared__ std::byte rngSM[ThreadsPerBlock * sizeof(RanluxppDouble)];

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int globalSlot = (*active)[i];
    Track &currentTrack  = electrons[globalSlot];
    auto energy          = currentTrack.energy;
    auto pos             = currentTrack.pos;
    auto dir             = currentTrack.dir;
    auto navState        = currentTrack.navState;
    const auto volume    = navState.Top();
    const int volumeID   = volume->id();
    // the MCC vector is indexed by the logical volume id
    const int lvolID     = volume->GetLogicalVolume()->id();
    const int theMCIndex = MCIndex[lvolID];

    auto &rngState = *reinterpret_cast<RanluxppDouble *>(rngSM + threadIdx.x * sizeof(RanluxppDouble));
    rngState       = currentTrack.rngState;

    auto survive = [&](bool push = true) {
      currentTrack.rngState = rngState;
      currentTrack.energy   = energy;
      currentTrack.pos      = pos;
      currentTrack.dir      = dir;
      currentTrack.navState = navState;
      if (push) activeQueue->push_back(globalSlot);
    };

    // Signal that this globalSlot doesn't undergo an interaction (yet)
    soaData.nextInteraction[i] = -1;

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack elTrack;
    G4HepEmTrack *theTrack = elTrack.GetTrack();
    theTrack->SetEKin(energy);
    theTrack->SetMCIndex(theMCIndex);
    theTrack->SetOnBoundary(navState.IsOnBoundary());
    theTrack->SetCharge(Charge);
    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    mscData->fIsFirstStep        = currentTrack.initialRange < 0;
    mscData->fInitialRange       = currentTrack.initialRange;
    mscData->fDynamicRangeFactor = currentTrack.dynamicRangeFactor;
    mscData->fTlimitMin          = currentTrack.tlimitMin;

    // Prepare a branched RNG state while threads are synchronized. Even if not
    // used, this provides a fresh round of random numbers and reduces thread
    // divergence because the RNG state doesn't need to be advanced later.
    RanluxppDouble newRNG(rngState.BranchNoAdvance());

    // Compute safety, needed for MSC step limit.
    double safety = 0;
    if (!navState.IsOnBoundary()) {
      safety = BVHNavigator::ComputeSafety(pos, navState);
    }
    theTrack->SetSafety(safety);

    G4HepEmRandomEngine rnge(&rngState);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft = -std::log(rngState.Rndm());
      }
      theTrack->SetNumIALeft(numIALeft, ip);
    }

    // Call G4HepEm to compute the physics step limit.
    // G4HepEmElectronManager::HowFar(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);
    G4HepEmElectronManager::HowFarToDiscreteInteraction(&g4HepEmData, &g4HepEmPars, &elTrack);

    bool restrictedPhysicalStepLength = false;
    if (BzFieldValue != 0) {
      const double momentumMag = sqrt(energy * (energy + 2.0 * Mass));
      // Distance along the track direction to reach the maximum allowed error
      const double safeLength = fieldPropagatorBz.ComputeSafeLength(momentumMag, Charge, dir);

      constexpr int MaxSafeLength = 10;
      double limit                = MaxSafeLength * safeLength;
      limit                       = safety > limit ? safety : limit;

      double physicalStepLength = elTrack.GetPStepLength();
      if (physicalStepLength > limit) {
        physicalStepLength           = limit;
        restrictedPhysicalStepLength = true;
        elTrack.SetPStepLength(physicalStepLength);
        // Note: We are limiting the true step length, which is converted to
        // a shorter geometry step length in HowFarToMSC. In that sense, the
        // limit is an over-approximation, but that is fine for our purpose.
      }
    }

    G4HepEmElectronManager::HowFarToMSC(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    // Remember MSC values for the next step(s).
    currentTrack.initialRange       = mscData->fInitialRange;
    currentTrack.dynamicRangeFactor = mscData->fDynamicRangeFactor;
    currentTrack.tlimitMin          = mscData->fTlimitMin;

    // Get result into variables.
    double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();
    // The phyiscal step length is the amount that the particle experiences
    // which might be longer than the geometrical step length due to MSC. As
    // long as we call PerformContinuous in the same kernel we don't need to
    // care, but we need to make this available when splitting the operations.
    // double physicalStepLength = elTrack.GetPStepLength();
    int winnerProcessIndex = theTrack->GetWinnerProcessIndex();
    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    bool propagated = true;
    double geometryStepLength;
    vecgeom::NavStateIndex nextState;
    if (BzFieldValue != 0) {
      geometryStepLength = fieldPropagatorBz.ComputeStepAndNextVolume<BVHNavigator>(
          energy, Mass, Charge, geometricalStepLengthFromPhysics, pos, dir, navState, nextState, propagated, safety);
    } else {
      geometryStepLength = BVHNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState,
                                                                  nextState, kPush);
      pos += geometryStepLength * dir;
    }

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to MSC.
    theTrack->SetDirection(dir.x(), dir.y(), dir.z());
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    // Apply continuous effects.
    bool stopped = G4HepEmElectronManager::PerformContinuous(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    // Collect the direction change and displacement by MSC.
    const double *direction = theTrack->GetDirection();
    dir.Set(direction[0], direction[1], direction[2]);
    if (!nextState.IsOnBoundary()) {
      const double *mscDisplacement = mscData->GetDisplacement();
      vecgeom::Vector3D<Precision> displacement(mscDisplacement[0], mscDisplacement[1], mscDisplacement[2]);
      const double dLength2            = displacement.Length2();
      constexpr double kGeomMinLength  = 5 * copcore::units::nm;          // 0.05 [nm]
      constexpr double kGeomMinLength2 = kGeomMinLength * kGeomMinLength; // (0.05 [nm])^2
      if (dLength2 > kGeomMinLength2) {
        const double dispR = std::sqrt(dLength2);
        // Estimate safety by subtracting the geometrical step length.
        safety -= geometryStepLength;
        constexpr double sFact = 0.99;
        double reducedSafety   = sFact * safety;

        // Apply displacement, depending on how close we are to a boundary.
        // 1a. Far away from geometry boundary:
        if (reducedSafety > 0.0 && dispR <= reducedSafety) {
          pos += displacement;
        } else {
          // Recompute safety.
          safety        = BVHNavigator::ComputeSafety(pos, navState);
          reducedSafety = sFact * safety;

          // 1b. Far away from geometry boundary:
          if (reducedSafety > 0.0 && dispR <= reducedSafety) {
            pos += displacement;
            // 2. Push to boundary:
          } else if (reducedSafety > kGeomMinLength) {
            pos += displacement * (reducedSafety / dispR);
          }
          // 3. Very small safety: do nothing.
        }
      }
    }

    // Collect the charged step length (might be changed by MSC).
    atomicAdd(&globalScoring->chargedSteps, 1);
    atomicAdd(&scoringPerVolume->chargedTrackLength[volumeID], elTrack.GetPStepLength());

    // Collect the changes in energy and deposit.
    energy               = theTrack->GetEKin();
    double energyDeposit = theTrack->GetEnergyDeposit();
    atomicAdd(&globalScoring->energyDeposit, energyDeposit);
    atomicAdd(&scoringPerVolume->energyDeposit[volumeID], energyDeposit);

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
    }

    if (stopped) {
      if (!IsElectron) {
        // Annihilate the stopped positron into two gammas heading to opposite
        // directions (isotropic).
        Track &gamma1 = secondaries.gammas.NextTrack();
        Track &gamma2 = secondaries.gammas.NextTrack();
        atomicAdd(&globalScoring->numGammas, 2);

        const double cost = 2 * rngState.Rndm() - 1;
        const double sint = sqrt(1 - cost * cost);
        const double phi  = k2Pi * rngState.Rndm();
        double sinPhi, cosPhi;
        sincos(phi, &sinPhi, &cosPhi);

        InitAsSecondary(gamma1, pos, navState);
        newRNG.Advance();
        gamma1.rngState = newRNG;
        gamma1.energy   = copcore::units::kElectronMassC2;
        gamma1.dir.Set(sint * cosPhi, sint * sinPhi, cost);

        InitAsSecondary(gamma2, pos, navState);
        // Reuse the RNG state of the dying track.
        gamma2.rngState = rngState;
        gamma2.energy   = copcore::units::kElectronMassC2;
        gamma2.dir      = -gamma1.dir;
      }
      // Particles are killed by not enqueuing them into the new activeQueue.
      continue;
    }

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
    } else if (!propagated || restrictedPhysicalStepLength) {
      // Did not yet reach the interaction point due to error in the magnetic
      // field propagation. Try again next time.
      survive();
      continue;
    } else if (winnerProcessIndex < 0) {
      // No discrete process, move on.
      survive();
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    currentTrack.numIALeft[winnerProcessIndex] = -1.0;

    // Check if a delta interaction happens instead of the real discrete process.
    if (G4HepEmElectronManager::CheckDelta(&g4HepEmData, theTrack, rngState.Rndm())) {
      // A delta interaction happened, move on.
      survive();
      continue;
    }

    soaData.nextInteraction[i] = winnerProcessIndex;

    survive(false);
  }
}

// Instantiate kernels for electrons and positrons.
__global__ void TransportElectrons(Track *electrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                   ScoringPerVolume *scoringPerVolume, SOAData soaData)
{
  TransportElectrons</*IsElectron*/ true>(electrons, active, secondaries, activeQueue, globalScoring, scoringPerVolume,
                                          soaData);
}
__global__ void TransportPositrons(Track *positrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                   ScoringPerVolume *scoringPerVolume, SOAData soaData)
{
  TransportElectrons</*IsElectron*/ false>(positrons, active, secondaries, activeQueue, globalScoring, scoringPerVolume,
                                           soaData);
}

template <bool IsElectron, int ProcessIndex>
__device__ void ElectronInteraction(int const globalSlot, SOAData const & /*soaData*/, int const /*soaSlot*/,
                                    Track *particles, Secondaries secondaries, adept::MParray *activeQueue,
                                    GlobalScoring *globalScoring, ScoringPerVolume *scoringPerVolume)
{
  Track &currentTrack = particles[globalSlot];
  auto energy         = currentTrack.energy;
  const auto pos      = currentTrack.pos;
  auto dir            = currentTrack.dir;
  const auto navState = currentTrack.navState;
  const auto volume   = navState.Top();
  // the MCC vector is indexed by the logical volume id
  const int lvolID     = volume->GetLogicalVolume()->id();
  const int theMCIndex = MCIndex[lvolID];

  auto survive = [&] {
    currentTrack.dir    = dir;
    currentTrack.energy = energy;
    activeQueue->push_back(globalSlot);
  };

  const double theElCut = g4HepEmData.fTheMatCutData->fMatCutData[theMCIndex].fSecElProdCutE;

  RanluxppDouble newRNG{currentTrack.rngState.Branch()};
  G4HepEmRandomEngine rnge{&currentTrack.rngState};

  if constexpr (ProcessIndex == 0) {
    // Invoke ionization (for e-/e+):
    double deltaEkin = (IsElectron) ? G4HepEmElectronInteractionIoni::SampleETransferMoller(theElCut, energy, &rnge)
                                    : G4HepEmElectronInteractionIoni::SampleETransferBhabha(theElCut, energy, &rnge);

    double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
    double dirSecondary[3];
    G4HepEmElectronInteractionIoni::SampleDirections(energy, deltaEkin, dirSecondary, dirPrimary, &rnge);

    Track &secondary = secondaries.electrons.NextTrack();
    atomicAdd(&globalScoring->numElectrons, 1);

    InitAsSecondary(secondary, pos, navState);
    secondary.rngState = newRNG;
    secondary.energy   = deltaEkin;
    secondary.dir.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);

    energy -= deltaEkin;
    dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
    survive();
  } else if constexpr (ProcessIndex == 1) {
    // Invoke model for Bremsstrahlung: either SB- or Rel-Brem.
    double logEnergy = std::log(energy);
    double deltaEkin = energy < g4HepEmPars.fElectronBremModelLim
                           ? G4HepEmElectronInteractionBrem::SampleETransferSB(&g4HepEmData, energy, logEnergy,
                                                                               theMCIndex, &rnge, IsElectron)
                           : G4HepEmElectronInteractionBrem::SampleETransferRB(&g4HepEmData, energy, logEnergy,
                                                                               theMCIndex, &rnge, IsElectron);

    double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
    double dirSecondary[3];
    G4HepEmElectronInteractionBrem::SampleDirections(energy, deltaEkin, dirSecondary, dirPrimary, &rnge);

    Track &gamma = secondaries.gammas.NextTrack();
    atomicAdd(&globalScoring->numGammas, 1);

    InitAsSecondary(gamma, pos, navState);
    gamma.rngState = newRNG;
    gamma.energy   = deltaEkin;
    gamma.dir.Set(dirSecondary[0], dirSecondary[1], dirSecondary[2]);

    energy -= deltaEkin;
    dir.Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
    survive();
  } else if constexpr (ProcessIndex == 2) {
    // Invoke annihilation (in-flight) for e+
    double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
    double theGamma1Ekin, theGamma2Ekin;
    double theGamma1Dir[3], theGamma2Dir[3];
    G4HepEmPositronInteractionAnnihilation::SampleEnergyAndDirectionsInFlight(
        energy, dirPrimary, &theGamma1Ekin, theGamma1Dir, &theGamma2Ekin, theGamma2Dir, &rnge);

    Track &gamma1 = secondaries.gammas.NextTrack();
    Track &gamma2 = secondaries.gammas.NextTrack();
    atomicAdd(&globalScoring->numGammas, 2);

    InitAsSecondary(gamma1, pos, navState);
    gamma1.rngState = newRNG;
    gamma1.energy   = theGamma1Ekin;
    gamma1.dir.Set(theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]);

    InitAsSecondary(gamma2, pos, navState);
    // Reuse the RNG state of the dying track.
    gamma2.rngState = currentTrack.rngState;
    gamma2.energy   = theGamma2Ekin;
    gamma2.dir.Set(theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]);

    // The current track is killed by not enqueuing into the next activeQueue.
  }
}

__global__ void IonizationEl(Track *particles, const adept::MParray *active, Secondaries secondaries,
                             adept::MParray *activeQueue, GlobalScoring *globalScoring,
                             ScoringPerVolume *scoringPerVolume, SOAData const soaData)
{
  InteractionLoop<0>(&ElectronInteraction<true, 0>, active, soaData, particles, secondaries, activeQueue, globalScoring,
                     scoringPerVolume);
}
__global__ void BremsstrahlungEl(Track *particles, const adept::MParray *active, Secondaries secondaries,
                                 adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                 ScoringPerVolume *scoringPerVolume, SOAData const soaData)
{
  InteractionLoop<1>(&ElectronInteraction<true, 1>, active, soaData, particles, secondaries, activeQueue, globalScoring,
                     scoringPerVolume);
}

__global__ void IonizationPos(Track *particles, const adept::MParray *active, Secondaries secondaries,
                              adept::MParray *activeQueue, GlobalScoring *globalScoring,
                              ScoringPerVolume *scoringPerVolume, SOAData const soaData)
{
  InteractionLoop<0>(&ElectronInteraction<false, 0>, active, soaData, particles, secondaries, activeQueue,
                     globalScoring, scoringPerVolume);
}
__global__ void BremsstrahlungPos(Track *particles, const adept::MParray *active, Secondaries secondaries,
                                  adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                  ScoringPerVolume *scoringPerVolume, SOAData const soaData)
{
  InteractionLoop<1>(&ElectronInteraction<false, 1>, active, soaData, particles, secondaries, activeQueue,
                     globalScoring, scoringPerVolume);
}
__global__ void AnnihilationPos(Track *particles, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                ScoringPerVolume *scoringPerVolume, SOAData const soaData)
{
  InteractionLoop<2>(&ElectronInteraction<false, 2>, active, soaData, particles, secondaries, activeQueue,
                     globalScoring, scoringPerVolume);
}














// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

//#include "example.cuh"

#include <AdePT/BVHNavigator.h>

#include <CopCore/PhysicalConstants.h>

__global__ void TransportGammas(Track *gammas, const adept::MParray *active, Secondaries secondaries,
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
    Track &currentTrack = gammas[slot];
    const auto energy   = currentTrack.energy;
    auto pos            = currentTrack.pos;
    const auto dir      = currentTrack.dir;
    auto navState       = currentTrack.navState;
    const auto volume   = navState.Top();
    const int volumeID  = volume->id();
    // the MCC vector is indexed by the logical volume id
    const int lvolID     = volume->GetLogicalVolume()->id();
    const int theMCIndex = MCIndex[lvolID];

    auto survive = [&](bool push = true) {
      currentTrack.pos      = pos;
      currentTrack.navState = navState;
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
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft = -std::log(currentTrack.rngState.Rndm());
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
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
    }

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
    currentTrack.numIALeft[winnerProcessIndex] = -1.0;

    soaData.nextInteraction[i] = winnerProcessIndex;
    soaData.gamma_PEmxSec[i] = gammaTrack.GetPEmxSec();
    survive(false);
  }
}

template <int ProcessIndex>
__device__ void GammaInteraction(int const globalSlot, SOAData const &soaData, int const soaSlot, Track *particles,
                                 Secondaries secondaries, adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                 ScoringPerVolume *scoringPerVolume)
{
  Track &currentTrack = particles[globalSlot];
  const auto energy   = currentTrack.energy;
  const auto pos      = currentTrack.pos;
  const auto dir      = currentTrack.dir;
  const auto navState = currentTrack.navState;
  const auto volume   = navState.Top();
  const int volumeID  = volume->id();
  // the MCC vector is indexed by the logical volume id
  const int lvolID     = volume->GetLogicalVolume()->id();
  const int theMCIndex = MCIndex[lvolID];

  auto survive = [&] { activeQueue->push_back(globalSlot); };

  RanluxppDouble newRNG{currentTrack.rngState.Branch()};
  G4HepEmRandomEngine rnge{&currentTrack.rngState};

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

    Track &electron = secondaries.electrons.NextTrack();
    Track &positron = secondaries.positrons.NextTrack();
    atomicAdd(&globalScoring->numElectrons, 1);
    atomicAdd(&globalScoring->numPositrons, 1);

    InitAsSecondary(electron, pos, navState);
    electron.rngState = newRNG;
    electron.energy   = elKinEnergy;
    electron.dir.Set(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);

    InitAsSecondary(positron, pos, navState);
    // Reuse the RNG state of the dying track.
    positron.rngState = currentTrack.rngState;
    positron.energy   = posKinEnergy;
    positron.dir.Set(dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]);

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
      Track &electron = secondaries.electrons.NextTrack();
      atomicAdd(&globalScoring->numElectrons, 1);

      InitAsSecondary(electron, pos, navState);
      electron.rngState = newRNG;
      electron.energy   = energyEl;
      electron.dir      = energy * dir - newEnergyGamma * newDirGamma;
      electron.dir.Normalize();
    } else {
      atomicAdd(&globalScoring->energyDeposit, energyEl);
      atomicAdd(&scoringPerVolume->energyDeposit[volumeID], energyEl);
    }

    // Check the new gamma energy and deposit if below threshold.
    if (newEnergyGamma > LowEnergyThreshold) {
      currentTrack.energy = newEnergyGamma;
      currentTrack.dir    = newDirGamma;
      survive();
    } else {
      atomicAdd(&globalScoring->energyDeposit, newEnergyGamma);
      atomicAdd(&scoringPerVolume->energyDeposit[volumeID], newEnergyGamma);
      // The current track is killed by not enqueuing into the next activeQueue.
    }
  } else if constexpr (ProcessIndex == 2) {
    // Invoke photoelectric process.
    const double theLowEnergyThreshold = 1 * copcore::units::eV;

    const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
        &g4HepEmData, theMCIndex, soaData.gamma_PEmxSec[soaSlot], energy, &rnge);

    double edep             = bindingEnergy;
    const double photoElecE = energy - edep;
    if (photoElecE > theLowEnergyThreshold) {
      // Create a secondary electron and sample directions.
      Track &electron = secondaries.electrons.NextTrack();
      atomicAdd(&globalScoring->numElectrons, 1);

      double dirGamma[] = {dir.x(), dir.y(), dir.z()};
      double dirPhotoElec[3];
      G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

      InitAsSecondary(electron, pos, navState);
      electron.rngState = newRNG;
      electron.energy   = photoElecE;
      electron.dir.Set(dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]);
    } else {
      edep = energy;
    }
    atomicAdd(&globalScoring->energyDeposit, edep);
    // The current track is killed by not enqueuing into the next activeQueue.
  }
}

__global__ void PairCreation(Track *particles, const adept::MParray *active, Secondaries secondaries,
                             adept::MParray *activeQueue, GlobalScoring *globalScoring,
                             ScoringPerVolume *scoringPerVolume, SOAData const soaData)
{
  InteractionLoop<0>(&GammaInteraction<0>, active, soaData, particles, secondaries, activeQueue, globalScoring,
                     scoringPerVolume);
}
__global__ void ComptonScattering(Track *particles, const adept::MParray *active, Secondaries secondaries,
                                  adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                  ScoringPerVolume *scoringPerVolume, SOAData const soaData)
{
  InteractionLoop<1>(&GammaInteraction<1>, active, soaData, particles, secondaries, activeQueue, globalScoring,
                     scoringPerVolume);
}
__global__ void PhotoelectricEffect(Track *particles, const adept::MParray *active, Secondaries secondaries,
                                    adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                    ScoringPerVolume *scoringPerVolume, SOAData const soaData)
{
  InteractionLoop<2>(&GammaInteraction<2>, active, soaData, particles, secondaries, activeQueue, globalScoring,
                     scoringPerVolume);
}

















#include <AdePT/Atomic.h>
#include <AdePT/BVHNavigator.h>
#include <AdePT/MParray.h>
#include <AdePT/NVTX.h>

#include <CopCore/Global.h>
#include <CopCore/PhysicalConstants.h>
#include <CopCore/Ranluxpp.h>

#include <VecGeom/base/Config.h>
#include <VecGeom/base/Stopwatch.h>
#include <VecGeom/management/GeoManager.h>
#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/backend/cuda/Interface.h>
#endif


#include <iostream>
#include <iomanip>
#include <stdio.h>

void InitG4HepEmGPU(::G4HepEmState *state_)
{
  // Copy to GPU.
  ::CopyG4HepEmDataToGPU(state_->fData);
  auto state = reinterpret_cast<g4::G4HepEmState*>(state_);
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmPars, state->fParameters, sizeof(G4HepEmParameters)));

  // Create G4HepEmData with the device pointers.
  g4::G4HepEmData dataOnDevice;
  dataOnDevice.fTheMatCutData   = state->fData->fTheMatCutData_gpu;
  dataOnDevice.fTheMaterialData = state->fData->fTheMaterialData_gpu;
  dataOnDevice.fTheElementData  = state->fData->fTheElementData_gpu;
  dataOnDevice.fTheElectronData = state->fData->fTheElectronData_gpu;
  dataOnDevice.fThePositronData = state->fData->fThePositronData_gpu;
  dataOnDevice.fTheSBTableData  = state->fData->fTheSBTableData_gpu;
  dataOnDevice.fTheGammaData    = state->fData->fTheGammaData_gpu;
  // The other pointers should never be used.
  dataOnDevice.fTheMatCutData_gpu   = nullptr;
  dataOnDevice.fTheMaterialData_gpu = nullptr;
  dataOnDevice.fTheElementData_gpu  = nullptr;
  dataOnDevice.fTheElectronData_gpu = nullptr;
  dataOnDevice.fThePositronData_gpu = nullptr;
  dataOnDevice.fTheSBTableData_gpu  = nullptr;
  dataOnDevice.fTheGammaData_gpu    = nullptr;

  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmData, &dataOnDevice, sizeof(g4::G4HepEmData)));
}

// A bundle of queues per particle type:
//  * Two for active particles, one for the current iteration and the second for the next.
struct ParticleQueues {
  adept::MParray *currentlyActive;
  adept::MParray *nextActive;

  void SwapActive() { std::swap(currentlyActive, nextActive); }
};

struct ParticleType {
  Track *tracks;
  SlotManager *slotManager;
  ParticleQueues queues;
  cudaStream_t stream;
  cudaEvent_t event;
  SOAData soaData;

  enum {
    Electron = 0,
    Positron = 1,
    Gamma    = 2,

    NumParticleTypes,
  };
};

// A bundle of queues for the three particle types.
struct AllParticleQueues {
  ParticleQueues queues[ParticleType::NumParticleTypes];
};

// Kernel to initialize the set of queues per particle type.
__global__ void InitParticleQueues(ParticleQueues queues, size_t Capacity)
{
  adept::MParray::MakeInstanceAt(Capacity, queues.currentlyActive);
  adept::MParray::MakeInstanceAt(Capacity, queues.nextActive);
}

// Kernel function to initialize a set of primary particles.
__global__ void InitPrimaries(ParticleGenerator generator, int startEvent, int numEvents, double energy,
                              const vecgeom::VPlacedVolume *world, GlobalScoring *globalScoring,
                              bool rotatingParticleGun)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEvents; i += blockDim.x * gridDim.x) {
    Track &track = generator.NextTrack();

    track.rngState.SetSeed(startEvent + i);
    track.energy       = energy;
    track.numIALeft[0] = -1.0;
    track.numIALeft[1] = -1.0;
    track.numIALeft[2] = -1.0;

    track.initialRange       = -1.0;
    track.dynamicRangeFactor = -1.0;
    track.tlimitMin          = -1.0;

    track.pos = {0, 0, 0};
    if (rotatingParticleGun) {
      // Generate particles flat in phi and in eta between -5 and 5. We'll lose the far forwards ones, so no need to
      // simulate.
      const double phi = 2. * M_PI * track.rngState.Rndm();
      const double eta = -5. + 10. * track.rngState.Rndm();
      track.dir.x()    = static_cast<vecgeom::Precision>(cos(phi) / cosh(eta));
      track.dir.y()    = static_cast<vecgeom::Precision>(sin(phi) / cosh(eta));
      track.dir.z()    = static_cast<vecgeom::Precision>(tanh(eta));
    } else {
      track.dir = {1.0, 0, 0};
    }
    track.navState.Clear();
    BVHNavigator::LocatePointIn(world, track.pos, track.navState, true);

    atomicAdd(&globalScoring->numElectrons, 1);
  }
}

// A data structure to transfer statistics after each iteration.
struct Stats {
  int inFlight[ParticleType::NumParticleTypes];
};

// Finish iteration: clear queues and fill statistics.
__global__ void FinishIteration(AllParticleQueues all, Stats *stats)
{
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    all.queues[i].currentlyActive->clear();
    stats->inFlight[i] = all.queues[i].nextActive->size();
  }
}

__global__ void ClearQueue(adept::MParray *queue)
{
  queue->clear();
}

void runGPU(int numParticles, double energy, int batch, const int *MCIndex_host,
            ScoringPerVolume *scoringPerVolume_host, GlobalScoring *globalScoring_host, int numVolumes, int numPlaced,
            ::G4HepEmState *state, bool rotatingParticleGun)
{
  NVTXTracer tracer("InitG4HepEM");
  InitG4HepEmGPU(state);

  tracer.setTag("InitParticles/malloc/copy");
  // Transfer MC indices.
  int *MCIndex_dev = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&MCIndex_dev, sizeof(int) * numVolumes));
  COPCORE_CUDA_CHECK(cudaMemcpy(MCIndex_dev, MCIndex_host, sizeof(int) * numVolumes, cudaMemcpyHostToDevice));
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(MCIndex, &MCIndex_dev, sizeof(int *)));

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  // Capacity of the different containers aka the maximum number of particles.
  // Use 1/5 of GPU memory for each of e+/e-/gammas, leaving 2/5 for the rest.
  const size_t Capacity = (deviceProp.totalGlobalMem / sizeof(Track) / 5);

  std::cout << "INFO: capacity of containers set to " << Capacity << std::endl;
  if (batch == -1) {
    // Rule of thumb: at most 2000 particles of one type per GeV primary.
    batch = Capacity / ((int)energy / copcore::units::GeV) / 2000;
  } else if (batch < 1) {
    batch = 1;
  }
  std::cout << "INFO: batching " << batch << " particles for transport on the GPU" << std::endl;
  if (BzFieldValue != 0) {
    std::cout << "INFO: running with field Bz = " << BzFieldValue / copcore::units::tesla << " T" << std::endl;
  } else {
    std::cout << "INFO: running with magnetic field OFF" << std::endl;
  }

  // Allocate structures to manage tracks of an implicit type:
  //  * memory to hold the actual Track elements,
  //  * objects to manage slots inside the memory,
  //  * queues of slots to remember active particle and those needing relocation,
  //  * a stream and an event for synchronization of kernels.
  const size_t TracksSize  = sizeof(Track) * Capacity;
  const size_t ManagerSize = sizeof(SlotManager);
  const size_t QueueSize   = adept::MParray::SizeOfInstance(Capacity);

  ParticleType particles[ParticleType::NumParticleTypes];
  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].tracks, TracksSize));

    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].slotManager, ManagerSize));

    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].queues.currentlyActive, QueueSize));
    COPCORE_CUDA_CHECK(cudaMalloc(&particles[i].queues.nextActive, QueueSize));
    InitParticleQueues<<<1, 1>>>(particles[i].queues, Capacity);

    COPCORE_CUDA_CHECK(cudaStreamCreate(&particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventCreate(&particles[i].event));

    COPCORE_CUDA_CHECK(
        cudaMalloc(&particles[i].soaData.nextInteraction, sizeof(SOAData::nextInteraction[0]) * Capacity));
  }
  COPCORE_CUDA_CHECK(
      cudaMalloc(&particles[ParticleType::Gamma].soaData.gamma_PEmxSec, sizeof(SOAData::gamma_PEmxSec[0]) * Capacity));
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  ParticleType &electrons = particles[ParticleType::Electron];
  ParticleType &positrons = particles[ParticleType::Positron];
  ParticleType &gammas    = particles[ParticleType::Gamma];

  // Create a stream to synchronize kernels of all particle types.
  cudaStream_t stream;
  COPCORE_CUDA_CHECK(cudaStreamCreate(&stream));

  cudaStream_t interactionStreams[3];
  for (auto i = 0; i < 3; ++i)
    COPCORE_CUDA_CHECK(cudaStreamCreate(&interactionStreams[i]));

  // Allocate memory to score charged track length and energy deposit per volume.
  double *chargedTrackLength = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&chargedTrackLength, sizeof(double) * numPlaced));
  COPCORE_CUDA_CHECK(cudaMemset(chargedTrackLength, 0, sizeof(double) * numPlaced));
  double *energyDeposit = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&energyDeposit, sizeof(double) * numPlaced));
  COPCORE_CUDA_CHECK(cudaMemset(energyDeposit, 0, sizeof(double) * numPlaced));

  // Allocate and initialize scoring and statistics.
  GlobalScoring *globalScoring = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&globalScoring, sizeof(GlobalScoring)));
  COPCORE_CUDA_CHECK(cudaMemset(globalScoring, 0, sizeof(GlobalScoring)));

  ScoringPerVolume *scoringPerVolume = nullptr;
  ScoringPerVolume scoringPerVolume_devPtrs;
  scoringPerVolume_devPtrs.chargedTrackLength = chargedTrackLength;
  scoringPerVolume_devPtrs.energyDeposit      = energyDeposit;
  COPCORE_CUDA_CHECK(cudaMalloc(&scoringPerVolume, sizeof(ScoringPerVolume)));
  COPCORE_CUDA_CHECK(
      cudaMemcpy(scoringPerVolume, &scoringPerVolume_devPtrs, sizeof(ScoringPerVolume), cudaMemcpyHostToDevice));

  Stats *stats_dev = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&stats_dev, sizeof(Stats)));
  Stats *stats = nullptr;
  COPCORE_CUDA_CHECK(cudaMallocHost(&stats, sizeof(Stats)));

  // Allocate memory to hold a "vanilla" SlotManager to initialize for each batch.
  SlotManager slotManagerInit(Capacity);
  SlotManager *slotManagerInit_dev = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&slotManagerInit_dev, sizeof(SlotManager)));
  COPCORE_CUDA_CHECK(cudaMemcpy(slotManagerInit_dev, &slotManagerInit, sizeof(SlotManager), cudaMemcpyHostToDevice));

  vecgeom::Stopwatch timer;
  timer.Start();
  tracer.setTag("sim");

  std::cout << std::endl << "Simulating particles ";
  const bool detailed = (numParticles / batch) < 50;
  if (!detailed) {
    std::cout << "... " << std::flush;
  }

  unsigned long long killed = 0;
  tracer.setTag("start event loop");

  for (int startEvent = 1; startEvent <= numParticles; startEvent += batch) {
    if (detailed) {
      std::cout << startEvent << " ... " << std::flush;
    }
    int left  = numParticles - startEvent + 1;
    int chunk = std::min(left, batch);

    for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
      COPCORE_CUDA_CHECK(cudaMemcpyAsync(particles[i].slotManager, slotManagerInit_dev, ManagerSize,
                                         cudaMemcpyDeviceToDevice, stream));
    }

    // Initialize primary particles.
    constexpr int InitThreads = ThreadsPerBlock;
    int initBlocks            = (chunk + ThreadsPerBlock - 1) / ThreadsPerBlock;
    ParticleGenerator electronGenerator(electrons.tracks, electrons.slotManager, electrons.queues.currentlyActive);
    auto world_dev = vecgeom::cxx::CudaManager::Instance().world_gpu();
    InitPrimaries<<<initBlocks, InitThreads, 0, stream>>>(electronGenerator, startEvent, chunk, energy, world_dev,
                                                          globalScoring, rotatingParticleGun);
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));

    stats->inFlight[ParticleType::Electron] = chunk;
    stats->inFlight[ParticleType::Positron] = 0;
    stats->inFlight[ParticleType::Gamma]    = 0;

    constexpr int MaxBlocks = 8192;
    int transportBlocks;

    int inFlight;
    int loopingNo         = 0;
    int previousElectrons = -1, previousPositrons = -1;

    do {
      Secondaries secondaries = {
          .electrons = {electrons.tracks, electrons.slotManager, electrons.queues.nextActive},
          .positrons = {positrons.tracks, positrons.slotManager, positrons.queues.nextActive},
          .gammas    = {gammas.tracks, gammas.slotManager, gammas.queues.nextActive},
      };

      // *** ELECTRONS ***
      int numElectrons = stats->inFlight[ParticleType::Electron];
      if (numElectrons > 0) {
        transportBlocks = (numElectrons + ThreadsPerBlock - 1) / ThreadsPerBlock;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        TransportElectrons<<<transportBlocks, ThreadsPerBlock, 0, electrons.stream>>>(
            electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive, globalScoring,
            scoringPerVolume, electrons.soaData);

        COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, electrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(interactionStreams[0], electrons.event, 0));

        IonizationEl<<<transportBlocks, ThreadsPerBlock, 0, interactionStreams[0]>>>(
            electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive, globalScoring,
            scoringPerVolume, electrons.soaData);
        BremsstrahlungEl<<<transportBlocks, ThreadsPerBlock, 0, electrons.stream>>>(
            electrons.tracks, electrons.queues.currentlyActive, secondaries, electrons.queues.nextActive, globalScoring,
            scoringPerVolume, electrons.soaData);

        for (auto streamToWaitFor : {interactionStreams[0], electrons.stream}) {
          COPCORE_CUDA_CHECK(cudaEventRecord(electrons.event, streamToWaitFor));
          COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, electrons.event, 0));
        }
      }

      // *** POSITRONS ***
      int numPositrons = stats->inFlight[ParticleType::Positron];
      if (numPositrons > 0) {
        transportBlocks = (numPositrons + ThreadsPerBlock - 1) / ThreadsPerBlock;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        TransportPositrons<<<transportBlocks, ThreadsPerBlock, 0, positrons.stream>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive, globalScoring,
            scoringPerVolume, positrons.soaData);

        COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(interactionStreams[1], positrons.event, 0));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(interactionStreams[2], positrons.event, 0));

        IonizationPos<<<transportBlocks, ThreadsPerBlock, 0, interactionStreams[1]>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive, globalScoring,
            scoringPerVolume, positrons.soaData);
        BremsstrahlungPos<<<transportBlocks, ThreadsPerBlock, 0, positrons.stream>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive, globalScoring,
            scoringPerVolume, positrons.soaData);
        AnnihilationPos<<<transportBlocks, ThreadsPerBlock, 0, interactionStreams[2]>>>(
            positrons.tracks, positrons.queues.currentlyActive, secondaries, positrons.queues.nextActive, globalScoring,
            scoringPerVolume, positrons.soaData);

        for (auto streamToWaitFor : {interactionStreams[1], positrons.stream, interactionStreams[2]}) {
          COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, streamToWaitFor));
          COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, positrons.event, 0));
        }
      }

      // *** GAMMAS ***
      int numGammas = stats->inFlight[ParticleType::Gamma];
      if (numGammas > 0) {
        transportBlocks = (numGammas + ThreadsPerBlock - 1) / ThreadsPerBlock;
        transportBlocks = std::min(transportBlocks, MaxBlocks);

        TransportGammas<<<transportBlocks, ThreadsPerBlock, 0, gammas.stream>>>(
            gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive, globalScoring,
            scoringPerVolume, gammas.soaData);

        COPCORE_CUDA_CHECK(cudaEventRecord(gammas.event, gammas.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, gammas.event, 0));

        for (auto i = 0; i < 3; ++i) {
          COPCORE_CUDA_CHECK(cudaStreamWaitEvent(interactionStreams[i], gammas.event, 0));
        }
        // About 2% of all gammas:
        PairCreation<<<transportBlocks, ThreadsPerBlock, 0, interactionStreams[0]>>>(
            gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive, globalScoring,
            scoringPerVolume, gammas.soaData);
        // About 10% of all gammas:
        ComptonScattering<<<transportBlocks, ThreadsPerBlock, 0, interactionStreams[1]>>>(
            gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive, globalScoring,
            scoringPerVolume, gammas.soaData);
        // About 15% of all gammas:
        PhotoelectricEffect<<<transportBlocks, ThreadsPerBlock, 0, interactionStreams[2]>>>(
            gammas.tracks, gammas.queues.currentlyActive, secondaries, gammas.queues.nextActive, globalScoring,
            scoringPerVolume, gammas.soaData);
        for (auto i = 0; i < 3; ++i) {
          COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, interactionStreams[i]));
          COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, positrons.event, 0));
        }
        COPCORE_CUDA_CHECK(cudaEventRecord(positrons.event, positrons.stream));
        COPCORE_CUDA_CHECK(cudaStreamWaitEvent(stream, positrons.event, 0));
      }

      // *** END OF TRANSPORT ***

      // The events ensure synchronization before finishing this iteration and
      // copying the Stats back to the host.
      AllParticleQueues queues = {{electrons.queues, positrons.queues, gammas.queues}};
      FinishIteration<<<1, 1, 0, stream>>>(queues, stats_dev);
      COPCORE_CUDA_CHECK(cudaMemcpyAsync(stats, stats_dev, sizeof(Stats), cudaMemcpyDeviceToHost, stream));

      // Finally synchronize all kernels.
      COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));

      // Count the number of particles in flight.
      inFlight = 0;
      for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
        inFlight += stats->inFlight[i];
      }

      tracer.setOccupancy(inFlight);

      tracer.setOccupancy(inFlight);

      // Swap the queues for the next iteration.
      electrons.queues.SwapActive();
      positrons.queues.SwapActive();
      gammas.queues.SwapActive();

      // Check if only charged particles are left that are looping.
      numElectrons = stats->inFlight[ParticleType::Electron];
      numPositrons = stats->inFlight[ParticleType::Positron];
      numGammas    = stats->inFlight[ParticleType::Gamma];
      if (numElectrons == previousElectrons && numPositrons == previousPositrons && numGammas == 0) {
        loopingNo++;
      } else {
        previousElectrons = numElectrons;
        previousPositrons = numPositrons;
        loopingNo         = 0;
      }

    } while (inFlight > 0 && loopingNo < 200);

    if (inFlight > 0) {
      killed += inFlight;
      for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
        ParticleType &pType   = particles[i];
        int inFlightParticles = stats->inFlight[i];
        if (inFlightParticles == 0) {
          continue;
        }

        ClearQueue<<<1, 1, 0, stream>>>(pType.queues.currentlyActive);
      }
      COPCORE_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
  std::cout << "done!" << std::endl;

  auto time = timer.Stop();
  std::cout << "Run time: " << time << "\n";

  // Transfer back scoring.
  COPCORE_CUDA_CHECK(cudaMemcpy(globalScoring_host, globalScoring, sizeof(GlobalScoring), cudaMemcpyDeviceToHost));
  globalScoring_host->numKilled = killed;

  // Transfer back the scoring per volume (charged track length and energy deposit).
  COPCORE_CUDA_CHECK(cudaMemcpy(scoringPerVolume_host->chargedTrackLength, scoringPerVolume_devPtrs.chargedTrackLength,
                                sizeof(double) * numPlaced, cudaMemcpyDeviceToHost));
  COPCORE_CUDA_CHECK(cudaMemcpy(scoringPerVolume_host->energyDeposit, scoringPerVolume_devPtrs.energyDeposit,
                                sizeof(double) * numPlaced, cudaMemcpyDeviceToHost));

  // Free resources.
  COPCORE_CUDA_CHECK(cudaFree(MCIndex_dev));
  COPCORE_CUDA_CHECK(cudaFree(chargedTrackLength));
  COPCORE_CUDA_CHECK(cudaFree(energyDeposit));

  COPCORE_CUDA_CHECK(cudaFree(globalScoring));
  COPCORE_CUDA_CHECK(cudaFree(scoringPerVolume));
  COPCORE_CUDA_CHECK(cudaFree(stats_dev));
  COPCORE_CUDA_CHECK(cudaFreeHost(stats));
  COPCORE_CUDA_CHECK(cudaFree(slotManagerInit_dev));

  COPCORE_CUDA_CHECK(cudaStreamDestroy(stream));

  for (auto i = 0; i < 3; ++i)
    COPCORE_CUDA_CHECK(cudaStreamDestroy(interactionStreams[i]));

  for (int i = 0; i < ParticleType::NumParticleTypes; i++) {
    COPCORE_CUDA_CHECK(cudaFree(particles[i].tracks));
    COPCORE_CUDA_CHECK(cudaFree(particles[i].slotManager));

    COPCORE_CUDA_CHECK(cudaFree(particles[i].queues.currentlyActive));
    COPCORE_CUDA_CHECK(cudaFree(particles[i].queues.nextActive));

    COPCORE_CUDA_CHECK(cudaStreamDestroy(particles[i].stream));
    COPCORE_CUDA_CHECK(cudaEventDestroy(particles[i].event));

    COPCORE_CUDA_CHECK(cudaFree(particles[i].soaData.nextInteraction));
    if (particles[i].soaData.gamma_PEmxSec) COPCORE_CUDA_CHECK(cudaFree(particles[i].soaData.gamma_PEmxSec));
  }
}
