// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTEM3_CUH
#define TESTEM3_CUH

#include "llama.hpp"
#include "TestEm3.h"
#include <AdePT/MParray.h>
#include <CopCore/SystemOfUnits.h>
#include <CopCore/Ranluxpp.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavStateIndex.h>
#include <cmath>

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
using RanluxDbl = llama::Record<llama::Field<State, llama::Array<uint64_t, 9>>, llama::Field<Carry, unsigned>,
                                llama::Field<BitPos, int>>; // TODO(bgruber): split State as well
using Track =
    llama::Record<llama::Field<RngState, RanluxDbl>, llama::Field<Energy, double>, llama::Field<NumIALeft, double[3]>,
                  llama::Field<InitialRange, double>, llama::Field<DynamicRangeFactor, double>,
                  llama::Field<TlimitMin, double>, llama::Field<Pos, vecgeom::Vector3D<vecgeom::Precision>>,
                  llama::Field<Dir, vecgeom::Vector3D<vecgeom::Precision>>,
                  llama::Field<NavState, vecgeom::NavStateIndex>>;

using Mapping = llama::mapping::AoS<llama::ArrayExtentsDynamic<int, 1>, Track>;
// using Mapping  = llama::mapping::PackedSingleBlobSoA<llama::ArrayExtentsDynamic<int, 1>, Track>;
// using Mapping  = llama::mapping::AlignedSingleBlobSoA<llama::ArrayExtentsDynamic<int, 1>, Track>;
//  using Mapping  = llama::mapping::MultiBlobSoA<llama::ArrayExtentsDynamic<int, 1>, Track>;
//  using Mapping  = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<int, 1>, Track, 16>;
//  using Mapping  = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<int, 1>, Track, 32>;
//  using Mapping  = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<int, 1>, Track, 64>;
//  using Mapping  = llama::mapping::Trace<llama::mapping::AoS<llama::ArrayExtentsDynamic<int, 1>, Track>, unsigned long
//  long, true>;
using BlobType = std::byte *;
using View     = llama::View<Mapping, BlobType>;

// we are providing the engine now :D
#define G4HepEmRandomEngine_HH

#include <G4HepEmMacros.hh>
#include <G4HepEmMath.hh>
#include <G4HepEmConstants.hh>
#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>

namespace ranlux {
inline constexpr const uint64_t *kA = kA_2048;
inline constexpr int kMaxPos        = 9 * 64;

template <typename VR>
__host__ __device__ void SaveState(VR &&vr, uint64_t *state)
{
  for (int i = 0; i < 9; i++) {
    state[i] = vr(State{})[i];
  }
  //  boost::mp11::mp_for_each<boost::mp11::mp_iota_c<9>>([&](auto ic) {
  //    constexpr auto i = decltype(ic)::value;
  //    state[i]         = vr(State{}, llama::RecordCoord<i>{});
  //  });
}

template <typename VR>
__host__ __device__ void LoadState(VR &&vr, const uint64_t *state)
{
  for (int i = 0; i < 9; i++) {
    vr(State{})[i] = state[i];
  }
  //  boost::mp11::mp_for_each<boost::mp11::mp_iota_c<9>>([&](auto ic) {
  //    constexpr auto i                     = decltype(ic)::value;
  //    vr(State{}, llama::RecordCoord<i>{}) = state[i];
  //  });
}

template <typename VR>
__host__ __device__ void XORstate(VR &&vr, const uint64_t *state)
{
  for (int i = 0; i < 9; i++) {
    vr(State{})[i] ^= state[i];
  }
  //  boost::mp11::mp_for_each<boost::mp11::mp_iota_c<9>>([&](auto ic) {
  //    constexpr auto i = decltype(ic)::value;
  //    vr(State{}, llama::RecordCoord<i>{}) ^= state[i];
  //  });
}

/// Produce next block of random bits
template <typename VR>
__host__ __device__ void Advance(VR &&vr)
{
  uint64_t lcg[9];
  //  uint64_t state[9];
  //  SaveState(vr, state);
  to_lcg(vr(State{}).begin(), vr(Carry{}), lcg);
  mulmod(kA, lcg);
  to_ranlux(lcg, vr(State{}).begin(), vr(Carry{}));
  //  LoadState(vr, state);
  vr(BitPos{}) = 0;
}

/// Return the next random bits, generate a new block if necessary
template <int w, typename VR>
__host__ __device__ uint64_t NextRandomBits(VR &&vr)
{
  int position = vr(BitPos{});
  if (position + w > kMaxPos) {
    Advance(vr);
    position = 0;
  }

  int idx     = position / 64;
  int offset  = position % 64;
  int numBits = 64 - offset;

  uint64_t bits = vr(State{})[idx] >> offset;
  if (numBits < w) {
    bits |= vr(State{})[idx + 1] << numBits;
  }
  //  uint64_t bits;
  //  boost::mp11::mp_with_index<9>(idx, [&](auto ic) {
  //    constexpr auto idx = decltype(ic)::value;
  //    bits               = vr(State{}, llama::RecordCoord<idx>{}) >> offset;
  //    if constexpr (idx < 8) {
  //      if (numBits < w) {
  //        bits |= vr(State{}, llama::RecordCoord<idx + 1>{}) << numBits;
  //      }
  //    }
  //  });

  bits &= ((uint64_t(1) << w) - 1);

  position += w;
  assert(position <= kMaxPos && "position out of range!");
  vr(BitPos{}) = position;

  return bits;
}

/// Return a floating point number, converted from the next random bits.
template <typename VR>
__host__ __device__ double NextRandomFloat(VR &&vr)
{
  constexpr int w             = 48;
  static constexpr double div = 1.0 / (uint64_t(1) << w);
  uint64_t bits               = NextRandomBits<w>(vr);
  return bits * div;
}

/// Initialize and seed the state of the generator
template <typename VR>
__host__ __device__ void SetSeed(VR &&vr, uint64_t s)
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

  //  uint64_t state[9];
  to_ranlux(lcg, vr(State{}).begin(), vr(Carry{}));
  //  LoadState(vr, state);
  vr(BitPos{}) = 0;
}

/// Branch a new RNG state, also advancing the current one.
/// The caller must Advance() the branched RNG state to decorrelate the
/// produced numbers.
template <typename VR>
__host__ __device__ auto BranchNoAdvance(VR &&vr)
{
  // Save the current state, will be used to branch a new RNG.
  uint64_t oldState[9];
  SaveState(vr, oldState);
  Advance(vr);
  // Copy and modify the new RNG state.
  llama::One<RanluxDbl> newRNG = vr;
  XORstate(newRNG, oldState);
  return newRNG;
}

template <typename VR>
__host__ __device__ auto Branch(VR &&vr)
{
  auto newRNG = BranchNoAdvance(vr);
  Advance(newRNG);
  return newRNG;
}
} // namespace ranlux

using RanluxRecordRef = decltype(std::declval<View>()(0)(RngState{}));

struct G4HepEmRandomEngine {
  G4HepEmHostDevice G4HepEmRandomEngine(RanluxRecordRef rr) : recordRef(rr), fIsGauss(false), fGauss(0.) {}

  // bgruber: avoiding inlining here makes the code actually faster
  __noinline__ G4HepEmHostDevice double flat() { return ranlux::NextRandomFloat(recordRef); }

    // bgruber: avoiding inlining here makes the code actually faster
  __noinline__ G4HepEmHostDevice void flatArray(const int size, double *vect)
  {
    for (int i = 0; i < size; i++) {
      vect[i] = ranlux::NextRandomFloat(recordRef);
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

  RanluxRecordRef recordRef;
  double fGauss;
  bool fIsGauss;
};

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
// extern __constant__ __device__ int Zero;

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
