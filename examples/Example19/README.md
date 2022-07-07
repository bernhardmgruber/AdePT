<!--
SPDX-FileCopyrightText: 2022 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

## Example 19 with LLAMA for ACAT22

Example based on Example 19, but the track array is managed by [LLAMA](https://github.com/alpaka-group/llama).
LLAMA allows us to arbitrarily change the data layout without needing to rewrite or change an algorithm using that data.
To switch the data layout, you just need to change the definition of the LLAMA mapping.
A couple of mappings are defined in `example.cuh` and you can try them out by (un-)commenting the corresponding alias declarations (`using Mapping = ...`).

The following data layouts can be tested directly:

* AoS
* PackedSingleBlobSoA
* AlignedSingleBlobSoA
* MultiBlobSoA
* AoSoAN (with arbitrary N)
* Trace (although you will have more accurate results when using the example especially preparted for tracing)
* Heatmap (AoS)
* Heatmap (AoSoA32)
* Heatmap (PackedSingleBlobSoA)
* Heatmap (AoS with slot granularity)

You are free to make up your own data layouts!
Please refer to the [LLAMA mappings documentation](https://llama-doc.readthedocs.io/en/latest/pages/mappings.html) for further information.
Please refer to the documentation of [Example 19](../Example19/README.md) for further information on the capabilities of this example and a detailed description of what the individual kernels do.

This example provided the basis for a [contribution to ACAT22](https://indico.cern.ch/event/1106990/contributions/4991259/).

