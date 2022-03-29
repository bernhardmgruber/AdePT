// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTEM3_CUH
#define TESTEM3_CUH

#include "TestEm3.h"

#include "llama.hpp"
#include <AdePT/MParray.h>
#include <CopCore/SystemOfUnits.h>
#include <CopCore/Ranluxpp.h>

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmRandomEngine.hh>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavStateIndex.h>

struct RngState {};
struct Energy {};
struct NumIALeft {};
struct InitialRange {};
struct DynamicRangeFactor {};
struct TlimitMin {};
struct Pos {};
struct Dir {};
struct NavState {};

// A data structure to represent a particle track. The particle type is implicit
// by the queue and not stored in memory.
using Track = llama::Record<llama::Field<RngState, RanluxppDouble>, llama::Field<Energy, double>,
                            llama::Field<NumIALeft, double[3]>, llama::Field<InitialRange, double>,
                            llama::Field<DynamicRangeFactor, double>, llama::Field<TlimitMin, double>,
                            llama::Field<Pos, vecgeom::Vector3D<vecgeom::Precision>>,
                            llama::Field<Dir, vecgeom::Vector3D<vecgeom::Precision>>,
                            llama::Field<NavState, vecgeom::NavStateIndex>>;

template <typename SecondayTrack>
__host__ __device__ void InitAsSecondary(SecondayTrack &&track, const vecgeom::Vector3D<Precision> &parentPos,
                                         const vecgeom::NavStateIndex &parentNavState)
{
  // The caller is responsible to branch a new RNG state and to set the energy.
  track(NumIALeft{}, llama::RecordCoord<0>{}) = -1.0;
  track(NumIALeft{}, llama::RecordCoord<1>{}) = -1.0;
  track(NumIALeft{}, llama::RecordCoord<2>{}) = -1.0;

  track(InitialRange{})       = -1.0;
  track(DynamicRangeFactor{}) = -1.0;
  track(TlimitMin{})          = -1.0;

  // A secondary inherits the position of its parent; the caller is responsible
  // to update the directions.
  track(Pos{})      = parentPos;
  track(NavState{}) = parentNavState;
}

// Defined in TestEm3.cu
extern __constant__ __device__ int Zero;

class RanluxppDoubleEngine : public G4HepEmRandomEngine {
  // Wrapper functions to call into RanluxppDouble.
  static __host__ __device__ __attribute__((noinline)) double FlatWrapper(void *object)
  {
    return ((RanluxppDouble *)object)->Rndm();
  }
  static __host__ __device__ __attribute__((noinline)) void FlatArrayWrapper(void *object, const int size, double *vect)
  {
    for (int i = 0; i < size; i++) {
      vect[i] = ((RanluxppDouble *)object)->Rndm();
    }
  }

public:
  __host__ __device__ RanluxppDoubleEngine(RanluxppDouble *engine)
      : G4HepEmRandomEngine(/*object=*/engine, &FlatWrapper, &FlatArrayWrapper)
  {
#ifdef __CUDA_ARCH__
    // This is a hack: The compiler cannot see that we're going to call the
    // functions through their pointers, so it underestimates the number of
    // required registers. By including calls to the (non-inlinable) functions
    // we force the compiler to account for the register usage, even if this
    // particular set of calls are not executed at runtime.
    if (Zero) {
      FlatWrapper(engine);
      FlatArrayWrapper(engine, 0, nullptr);
    }
#endif
  }
};

using ParticleCount = unsigned int;

using Mapping  = llama::mapping::AoS<llama::ArrayExtentsDynamic<int, 1>, Track>;
// using Mapping  = llama::mapping::PackedSingleBlobSoA<llama::ArrayExtentsDynamic<int, 1>, Track>;
// using Mapping  = llama::mapping::AlignedSingleBlobSoA<llama::ArrayExtentsDynamic<int, 1>, Track>;
// using Mapping  = llama::mapping::MultiBlobSoA<llama::ArrayExtentsDynamic<int, 1>, Track>;
// using Mapping  = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<int, 1>, Track, 16>;
// using Mapping  = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<int, 1>, Track, 32>;
// using Mapping  = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<int, 1>, Track, 64>;
using BlobType = std::byte *;
using View     = llama::View<Mapping, BlobType>;

// A bundle of pointers to generate particles of an implicit type.
struct ParticleGenerator {
  View fTracks;
  ParticleCount *fNextSlot;
  ParticleCount fMaxSlot;

  __device__ decltype(auto) NextTrack()
  {
    const auto next = atomicAdd(fNextSlot, ParticleCount{1});
    if (next >= fMaxSlot) {
      COPCORE_EXCEPTION("No slot available in ParticleGenerator::NextTrack");
    }
    return fTracks[next];
  }
};

// A bundle of generators for the three particle types.
struct Secondaries {
  ParticleGenerator electrons;
  ParticleGenerator positrons;
  ParticleGenerator gammas;
};

// Kernels in different TUs.
__global__ void TransportElectrons(const ParticleCount *electronsCount, View electrons, Secondaries secondaries,
                                   GlobalScoring *globalScoring, ScoringPerVolume *scoringPerVolume);
__global__ void TransportPositrons(const ParticleCount *electronsCount, View electrons, Secondaries secondaries,
                                   GlobalScoring *globalScoring, ScoringPerVolume *scoringPerVolume);

__global__ void TransportGammas(const ParticleCount *gammasCount, View gammas, Secondaries secondaries,
                                GlobalScoring *globalScoring, ScoringPerVolume *scoringPerVolume);

// Constant data structures from G4HepEm accessed by the kernels.
// (defined in TestEm3.cu)
extern __constant__ __device__ struct G4HepEmParameters g4HepEmPars;
extern __constant__ __device__ struct G4HepEmData g4HepEmData;

extern __constant__ __device__ int *MCIndex;

// constexpr vecgeom::Precision BzFieldValue = 1 * copcore::units::tesla;
constexpr vecgeom::Precision BzFieldValue = 0;

#endif
