# SPDX-FileCopyrightText: 2021 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling TestEm3_llama (needs G4HepEm)")
  return()
endif()

find_package(Boost REQUIRED)

# TestEm3 for validation of particle transportation with GPUs.
add_executable(TestEm3_llama TestEm3.cpp TestEm3.cu electrons.cu gammas.cu)
target_link_libraries(TestEm3_llama PRIVATE AdePT CopCore::CopCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml ${Geant4_LIBRARIES} G4HepEm::g4HepEmData G4HepEm::g4HepEmInit G4HepEm::g4HepEmRun Boost::headers)
target_compile_options(TestEm3_llama PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(TestEm3_llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME TestEm3_llama COMMAND TestEm3_llama)
