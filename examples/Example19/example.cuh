// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLE_CUH
#define EXAMPLE_CUH

#include "example.h"

#include <AdePT/MParray.h>
#include <CopCore/SystemOfUnits.h>
#include <CopCore/Ranluxpp.h>

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>
#include "G4HepEmMacros.hh"
#include "G4HepEmMath.hh"
#include "G4HepEmConstants.hh"

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavStateIndex.h>

#include <cmath>

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

  __host__ __device__ void InitAsSecondary(const Track &parent)
  {
    // The caller is responsible to branch a new RNG state and to set the energy.
    this->numIALeft[0] = -1.0;
    this->numIALeft[1] = -1.0;
    this->numIALeft[2] = -1.0;

    this->initialRange       = -1.0;
    this->dynamicRangeFactor = -1.0;
    this->tlimitMin          = -1.0;

    // A secondary inherits the position of its parent; the caller is responsible
    // to update the directions.
    this->pos      = parent.pos;
    this->navState = parent.navState;
  }
};

// Struct for communication between kernels
struct SOAData {
  char *nextInteraction = nullptr;
  double *gamma_PEmxSec = nullptr;
};

// we provide our own
#define G4HepEmRandomEngine_HH

struct G4HepEmRandomEngine {
public:
  G4HepEmHostDevice G4HepEmRandomEngine(RanluxppDouble &ranlux) : ranlux(&ranlux), fIsGauss(false), fGauss(0.) {}

  G4HepEmHostDevice double flat() { return ranlux->Rndm(); }

  G4HepEmHostDevice void flatArray(const int size, double *vect)
  {
    for (int i = 0; i < size; i++) {
      vect[i] = ranlux->Rndm();
    }
  }

  G4HepEmHostDevice double Gauss(const double mean, const double stDev)
  {
    if (fIsGauss) {
      fIsGauss = false;
      return fGauss * stDev + mean;
    }
    double rnd[2];
    double r, v1, v2;
    do {
      flatArray(2, rnd);
      v1 = 2. * rnd[0] - 1.;
      v2 = 2. * rnd[1] - 1.;
      r  = v1 * v1 + v2 * v2;
    } while (r > 1.);
    const double fac = std::sqrt(-2. * G4HepEmLog(r) / r);
    fGauss           = v1 * fac;
    fIsGauss         = true;
    return v2 * fac * stDev + mean;
  }

  G4HepEmHostDevice void DiscardGauss() { fIsGauss = false; }

  G4HepEmHostDevice int Poisson(double mean)
  {
    const int border   = 16;
    const double limit = 2.E+9;

    int number = 0;
    if (mean <= border) {
      const double position = flat();
      double poissonValue   = G4HepEmExp(-mean);
      double poissonSum     = poissonValue;
      while (poissonSum <= position) {
        ++number;
        poissonValue *= mean / number;
        poissonSum += poissonValue;
      }
      return number;
    } // the case of mean <= 16
    //
    double rnd[2];
    flatArray(2, rnd);
    const double t = std::sqrt(-2. * G4HepEmLog(rnd[0])) * std::cos(k2Pi * rnd[1]);
    double value   = mean + t * std::sqrt(mean) + 0.5;
    return value < 0. ? 0 : value >= limit ? static_cast<int>(limit) : static_cast<int>(value);
  }

private:
  RanluxppDouble *ranlux;
  bool fIsGauss;
  double fGauss;
};

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
      interactionFunction((*active)[candidates[j]], soaData, j, std::forward<Args>(args)...);
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
extern __constant__ __device__ struct G4HepEmParameters g4HepEmPars;
extern __constant__ __device__ struct G4HepEmData g4HepEmData;

extern __constant__ __device__ int *MCIndex;

// constexpr vecgeom::Precision BzFieldValue = 3.8 * copcore::units::tesla;
constexpr vecgeom::Precision BzFieldValue = 0;

#endif
