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

struct State {};
struct Carry {};
struct BitPos {};
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
using RanluxDbl =
    llama::Record<llama::Field<State, uint64_t[9]>, llama::Field<Carry, unsigned>, llama::Field<BitPos, int>>;
using Track = llama::Record<
    llama::Field<RngState, RanluxDbl>, llama::Field<Energy, double>, llama::Field<NumIALeft, llama::Array<double, 3>>,
    llama::Field<InitialRange, double>, llama::Field<DynamicRangeFactor, double>, llama::Field<TlimitMin, double>,
    llama::Field<Pos, vecgeom::Vector3D<vecgeom::Precision>>, llama::Field<Dir, vecgeom::Vector3D<vecgeom::Precision>>,
    llama::Field<NavState, vecgeom::NavStateIndex>>;

template <typename SecondayTrack, typename ParentTrack>
__host__ __device__ void InitAsSecondary(SecondayTrack &&track, const ParentTrack &parent)
{
  // The caller is responsible to branch a new RNG state and to set the energy.
  track(NumIALeft{})[0] = -1.0;
  track(NumIALeft{})[1] = -1.0;
  track(NumIALeft{})[2] = -1.0;

  track(InitialRange{})       = -1.0;
  track(DynamicRangeFactor{}) = -1.0;
  track(TlimitMin{})          = -1.0;

  // A secondary inherits the position of its parent; the caller is responsible
  // to update the directions.
  track(Pos{})      = parent(Pos{});
  track(NavState{}) = parent(NavState{});
}

// Defined in TestEm3.cu
extern __constant__ __device__ int Zero;

template <int w>
struct RanluxppEngineImplRef {
  uint64_t *fState[9];
  unsigned *fCarry;
  int *fPosition;

  static constexpr const uint64_t *kA = kA_2048;
  static constexpr int kMaxPos        = 9 * 64;

  __host__ __device__ RanluxppEngineImplRef(uint64_t &state0, uint64_t &state1, uint64_t &state2, uint64_t &state3,
                                            uint64_t &state4, uint64_t &state5, uint64_t &state6, uint64_t &state7,
                                            uint64_t &state8, unsigned &carry, int &position)
  {
    fState[0] = &state0;
    fState[1] = &state1;
    fState[2] = &state2;
    fState[3] = &state3;
    fState[4] = &state4;
    fState[5] = &state5;
    fState[6] = &state6;
    fState[7] = &state7;
    fState[8] = &state8;
    fCarry    = &carry;
    fPosition = &position;
  }

  __host__ __device__ void SaveState(uint64_t *state) const
  {
    for (int i = 0; i < 9; i++) {
      state[i] = *fState[i];
    }
  }

  __host__ __device__ void LoadState(const uint64_t *state)
  {
    for (int i = 0; i < 9; i++) {
      *fState[i] = state[i];
    }
  }

  __host__ __device__ void XORstate(const uint64_t *state)
  {
    for (int i = 0; i < 9; i++) {
      *fState[i] ^= state[i];
    }
  }

  /// Produce next block of random bits
  __host__ __device__ void Advance()
  {
    uint64_t lcg[9];
    uint64_t state[9];
    SaveState(state);
    to_lcg(state, *fCarry, lcg);
    mulmod(kA, lcg);
    to_ranlux(lcg, state, *fCarry);
    LoadState(state);
    *fPosition = 0;
  }

  /// Return the next random bits, generate a new block if necessary
  __host__ __device__ uint64_t NextRandomBits()
  {
    int position = *fPosition;
    if (position + w > kMaxPos) {
      Advance();
      position = 0;
    }

    int idx     = position / 64;
    int offset  = position % 64;
    int numBits = 64 - offset;
    uint64_t bits;

    bits = *fState[idx] >> offset;
    if (numBits < w) {
      bits |= *fState[idx + 1] << numBits;
    }
    bits &= ((uint64_t(1) << w) - 1);

    position += w;
    assert(position <= kMaxPos && "position out of range!");
    *fPosition = position;

    return bits;
  }

  /// Return a floating point number, converted from the next random bits.
  __host__ __device__ double NextRandomFloat()
  {
    static constexpr double div = 1.0 / (uint64_t(1) << w);
    uint64_t bits               = NextRandomBits();
    return bits * div;
  }

  /// Initialize and seed the state of the generator
  __host__ __device__ void SetSeed(uint64_t s)
  {
    uint64_t lcg[9];
    lcg[0] = 1;
    for (int i = 1; i < 9; i++) {
      lcg[i] = 0;
    }

    uint64_t a_seed[9];
    // Skip 2 ** 96 states.
    powermod(kA, a_seed, uint64_t(1) << 48);
    powermod(a_seed, a_seed, uint64_t(1) << 48);
    // Skip another s states.
    powermod(a_seed, a_seed, s);
    mulmod(a_seed, lcg);

    uint64_t state[9];
    to_ranlux(lcg, state, *fCarry);
    LoadState(state);
    *fPosition = 0;
  }

  /// Skip `n` random numbers without generating them
  __host__ __device__ void Skip(uint64_t n)
  {
    auto position = *fPosition;
    int left      = (kMaxPos - position) / w;
    assert(left >= 0 && "position was out of range!");
    if (n < (uint64_t)left) {
      // Just skip the next few entries in the currently available bits.
      position += n * w;
      assert(position <= kMaxPos && "position out of range!");
      *fPosition = position;
      return;
    }

    n -= left;
    // Need to advance and possibly skip over blocks.
    int nPerState = kMaxPos / w;
    int skip      = (n / nPerState);

    uint64_t a_skip[9];
    powermod(kA, a_skip, skip + 1);

    uint64_t lcg[9];
    uint64_t state[9];
    SaveState(state);
    to_lcg(state, *fCarry, lcg);
    mulmod(a_skip, lcg);
    to_ranlux(lcg, state, *fCarry);
    LoadState(state);

    // Potentially skip numbers in the freshly generated block.
    int remaining = n - skip * nPerState;
    assert(remaining >= 0 && "should not end up at a negative position!");
    position = remaining * w;
    assert(position <= kMaxPos && "position out of range!");
    *fPosition = position;
  }
};

struct RanluxppDoubleRef final : RanluxppEngineImplRef<48> {
  //  __host__ __device__ RanluxppDoubleRef(uint64_t seed = 314159265) { this->SetSeed(seed); }

  using RanluxppEngineImplRef<48>::RanluxppEngineImplRef;

  /// Generate a double-precision random number with 48 bits of randomness
  __host__ __device__ double Rndm() { return (*this)(); }
  /// Generate a double-precision random number (non-virtual method)
  __host__ __device__ double operator()() { return this->NextRandomFloat(); }
  /// Generate a random integer value with 48 bits
  __host__ __device__ uint64_t IntRndm() { return this->NextRandomBits(); }

  // decay reference to value
  __host__ __device__ explicit operator RanluxppDouble() const
  {
    RanluxppDouble r;
    SaveState(r.fState);
    r.fCarry    = *fCarry;
    r.fPosition = *fPosition;
    return r;
  }

  /// Branch a new RNG state, also advancing the current one.
  /// The caller must Advance() the branched RNG state to decorrelate the
  /// produced numbers.
  __host__ __device__ RanluxppDouble BranchNoAdvance()
  {
    // Save the current state, will be used to branch a new RNG.
    uint64_t oldState[9];
    this->SaveState(oldState);
    this->Advance();
    // Copy and modify the new RNG state.
    RanluxppDouble newRNG(*this);
    newRNG.XORstate(oldState);
    return newRNG;
  }

  /// Branch a new RNG state, also advancing the current one.
  __host__ __device__ RanluxppDouble Branch()
  {
    RanluxppDouble newRNG(BranchNoAdvance());
    newRNG.Advance();
    return newRNG;
  }
};

class RanluxppDoubleEngineRef : public G4HepEmRandomEngine {
  // Wrapper functions to call into RanluxppDouble.
  static __host__ __device__ __attribute__((noinline)) double FlatWrapper(void *object)
  {
    return ((RanluxppDoubleRef *)object)->Rndm();
  }
  static __host__ __device__ __attribute__((noinline)) void FlatArrayWrapper(void *object, const int size, double *vect)
  {
    for (int i = 0; i < size; i++) {
      vect[i] = ((RanluxppDoubleRef *)object)->Rndm();
    }
  }

public:
  __host__ __device__ RanluxppDoubleEngineRef(RanluxppDoubleRef *engine)
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

template <typename VR>
__host__ __device__ auto makeRngRef(VR &&rng)
{
  return RanluxppDoubleRef{rng(State{}, llama::RecordCoord<0>{}),
                           rng(State{}, llama::RecordCoord<1>{}),
                           rng(State{}, llama::RecordCoord<2>{}),
                           rng(State{}, llama::RecordCoord<3>{}),
                           rng(State{}, llama::RecordCoord<4>{}),
                           rng(State{}, llama::RecordCoord<5>{}),
                           rng(State{}, llama::RecordCoord<6>{}),
                           rng(State{}, llama::RecordCoord<7>{}),
                           rng(State{}, llama::RecordCoord<8>{}),
                           rng(Carry{}),
                           rng(BitPos{})};
}

template <typename VR, typename RanluxDoubleValueOrRef>
__host__ __device__ auto storeRng(VR &&rng, RanluxDoubleValueOrRef &&ranlux)
{
  rng(State{}, llama::RecordCoord<0>{}) = ranlux.fState[0];
  rng(State{}, llama::RecordCoord<1>{}) = ranlux.fState[1];
  rng(State{}, llama::RecordCoord<2>{}) = ranlux.fState[2];
  rng(State{}, llama::RecordCoord<3>{}) = ranlux.fState[3];
  rng(State{}, llama::RecordCoord<4>{}) = ranlux.fState[4];
  rng(State{}, llama::RecordCoord<5>{}) = ranlux.fState[5];
  rng(State{}, llama::RecordCoord<6>{}) = ranlux.fState[6];
  rng(State{}, llama::RecordCoord<7>{}) = ranlux.fState[7];
  rng(State{}, llama::RecordCoord<8>{}) = ranlux.fState[8];
  rng(Carry{})                          = ranlux.fCarry;
  rng(BitPos{})                         = ranlux.fPosition;
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

using Mapping = llama::mapping::AoS<llama::ArrayExtentsDynamic<int, 1>, Track>;
// using Mapping = llama::mapping::SingleBlobSoA<llama::ArrayExtentsDynamic<int, 1>, Track>;
//  using Mapping  = llama::mapping::MultiBlobSoA<llama::ArrayExtentsDynamic<int, 1>, Track>;
//  using Mapping  = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<int, 1>, Track, 16>;
//  using Mapping  = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<int, 1>, Track, 32>;
//  using Mapping  = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<int, 1>, Track, 64>;
//  using Mapping  = llama::mapping::Trace<llama::mapping::AoS<llama::ArrayExtentsDynamic<int, 1>, Track>, unsigned long
//  long, true>;
using BlobType = std::byte *;
using View     = llama::View<Mapping, BlobType>;

// A bundle of pointers to generate particles of an implicit type.
class ParticleGenerator {
  View fTracks;
  SlotManager *fSlotManager;
  adept::MParray *fActiveQueue;

public:
  __host__ __device__ ParticleGenerator(llama::View<Mapping, BlobType> tracks, SlotManager *slotManager,
                                        adept::MParray *activeQueue)
      : fTracks(tracks), fSlotManager(slotManager), fActiveQueue(activeQueue)
  {
  }

  __host__ __device__ decltype(auto) NextTrack()
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
__global__ void TransportElectrons(View electrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                   ScoringPerVolume *scoringPerVolume);
__global__ void TransportPositrons(View positrons, const adept::MParray *active, Secondaries secondaries,
                                   adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                   ScoringPerVolume *scoringPerVolume);

__global__ void TransportGammas(View gammas, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                ScoringPerVolume *scoringPerVolume);

// Constant data structures from G4HepEm accessed by the kernels.
// (defined in TestEm3.cu)
extern __constant__ __device__ struct G4HepEmParameters g4HepEmPars;
extern __constant__ __device__ struct G4HepEmData g4HepEmData;

extern __constant__ __device__ int *MCIndex;

// constexpr vecgeom::Precision BzFieldValue = 1 * copcore::units::tesla;
constexpr vecgeom::Precision BzFieldValue = 0;

#endif
