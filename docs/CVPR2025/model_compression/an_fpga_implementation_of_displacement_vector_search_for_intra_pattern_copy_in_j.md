---
title: >-
  [Paper Note] An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS
description: >-
  [CVPR 2025][Model Compression][FPGA implementation] This paper proposes the first FPGA architecture implementation for the displacement vector (DV) search module in JPEG XS Intra Pattern Copy (IPC). Utilizing a four-stage pipelined design and optimized memory organization, it achieves a throughput of 38.3 Mpixels/s and a power consumption of 277 mW on Xilinx Artix-7, laying the foundation for practical hardware deployment and ASIC transition of IPC.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "FPGA implementation"
  - "JPEG XS"
  - "Intra Pattern Copy"
  - "displacement vector search"
  - "hardware acceleration"
date: 2026-05-08
content_hash: b64becc8f8d5ab97
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS

**Conference**: CVPR 2025  
**arXiv**: [2603.10671](https://arxiv.org/abs/2603.10671)  
**Code**: To be confirmed  
**Area**: Model Compression / Hardware Acceleration / Video Coding  
**Keywords**: FPGA implementation, JPEG XS, Intra Pattern Copy, displacement vector search, hardware acceleration

## TL;DR

This paper proposes the first FPGA architecture implementation for the displacement vector (DV) search module in JPEG XS Intra Pattern Copy (IPC). Utilizing a four-stage pipelined design and optimized memory organization, it achieves a throughput of 38.3 Mpixels/s and a power consumption of 277 mW on Xilinx Artix-7, laying the foundation for practical hardware deployment and ASIC transition of IPC.

## Background & Motivation

### Background
JPEG XS is a low-latency, low-complexity image compression standard developed by the JPEG Committee, targeting scenarios such as remote desktops and KVM applications. Various technologies, including Temporal Difference Coding (TDC) and Intra Pattern Copy (IPC), have been developed to improve the coding efficiency of JPEG XS on screen content.

### Limitations of Prior Work
- IPC reduces image structural redundancy through wavelet-domain intra compensation prediction, significantly improving the BD-PSNR of screen content.
- However, DV search requires traversing all candidate patterns to find the optimal prediction offset, which is computationally expensive and constitutes the most computationally intensive, resource-consuming, and latency-sensitive module in IPC.
- Existing FPGA/ASIC implementations of H.264/HEVC motion estimation and intra prediction are not applicable to the low-latency, low-complexity framework of JPEG XS.

### Key Challenge
While DV search in IPC yields significant coding gains at the algorithmic level, its extreme computational density presents a critical bottleneck for actual hardware deployment, hindering the practical application of IPC in real-time hardware systems.

### Goal
To design an efficient FPGA architecture to implement the DV search module, enabling the hardware deployment of IPC and providing a reference for future ASIC designs.

### Key Insight
Starting from the computational characteristics and data reuse patterns of JPEG XS IPC, targeted pipelined architectures and memory optimization schemes are designed.

### Core Idea
A four-stage pipelined DV comparison architecture coupled with an IPC Group-aligned external memory organization, leveraging the grouping computational characteristics and inherent data reuse patterns of IPC to optimize performance.

## Method

### Overall Architecture
The DV search architecture comprises two main engines: the **Residual Calculation Engine** (which retrieves IPC Unit data from DRAM and calculates residuals) and the **DV Comparison Engine** (which selects the optimal DV based on the residual bitplane expense). The system balances throughput and latency through a pipelined design.

### Key Designs

#### Key Design 1: Residual Calculation Engine
- **Function**: Retrieve original and reconstructed wavelet coefficient blocks from different subbands and calculate block-level residuals.
- **Mechanism**: Retrieves coefficients from DRAM via a CMD module, refines alignment with FIFO buffers, and performs signed subtraction in sign-magnitude format via the SIG_MAG_SUB module. Four parallel subtraction paths are implemented to handle different sign combinations.
- **Design Motivation**: Coefficients of IPC come from different subbands and are stored discretely. FIFO and MUX are required to ensure synchronous block loading and group alignment.

#### Key Design 2: Four-Stage Pipelined DV Comparison Engine
- **Function**: Evaluate the bit cost of each residual IPC Unit and search for the optimal matching DV.
- **Mechanism**:
    - Stage 0: Input residual coefficients and DVs to generate configuration parameters like BandIdx/GrpSize/UnitWidth.
    - Stage 1: GetOrMask performs bitwise OR mask operations over residuals within a group.
    - Stage 2: CalGCLI calculates the GCLI bit cost based on the bitwise OR results.
    - Stage 3: The Compare module compares the current cost with the historical minimum cost to update the optimal DV.
- **Design Motivation**: A systematic pipelined design enables parallel DV evaluation, low-latency processing, and supports scalable block-level processing.

#### Key Design 3: IPC Group-Aligned External Memory Organization
- **Function**: Optimize the storage layout of wavelet coefficients in DRAM.
- **Mechanism**:
    - Method 0 (Baseline): Coefficients are stored in precinct order, enabling simple linear addressing but causing inefficient IPC grouping access.
    - Method 1 (Ours): Coefficients are organized by IPC Groups and Units, where IPC Units within the same group are stored sequentially, and each group contains all subband blocks.
- **Design Motivation**: Method 0 requires locating scattered coefficient blocks sequentially using group, unit, and subband indices, which increases control complexity and reduces memory throughput. Method 1 allows loading an entire IPC Unit using a single base address plus fixed offsets, thereby supporting burst memory access.
- An on-chip TLB RAM stores the length information of coefficient blocks within individual IPC Units across different groups.

### Loss & Training
Not applicable (hardware design paper, no training process). The goal is to maximize hardware throughput and minimize resource consumption while maintaining rate-distortion performance consistent with the IPC reference software.

## Key Experimental Results

### Main Results: FPGA Resource Utilization

| Module | LUTs (K) | FFs (K) | DSPs | BRAM |
|------|----------|---------|------|------|
| Residual Calculation Engine | 0.48 | 0.47 | 0 | 15 |
| GCLI_CAL (DV Comparison) | 11.63 | 19.98 | 17 | 0 |
| DV_UPDATE (DV Comparison) | 0.73 | 1.41 | 0 | 0 |

### Memory Optimization Comparison (Method 0 vs Method 1)

| Parameter | Method 0 (Baseline) | Method 1 (Ours) |
|------|-----------------|-----------------|
| Platform | Xilinx Artix-7 (XC7A35T), 100 MHz |
| Throughput | 35.98 Mpixels/s | **38.30 Mpixels/s** |
| Power | 276 mW | 277 mW |
| Power Efficiency | 130.36 Mpixels/s/W | **138.27 Mpixels/s/W** |
| LUTs (K) | 13.93 | **12.89** (-7.5%) |
| FFs (K) | 23.80 | **21.79** (-8.4%) |
| DSPs | 17 | 17 |
| BRAM | 11 | 15 (+4) |

### Key Findings
1. **Significant effect of memory optimization**: Method 1 reduces LUTs and FFs by 7.5% and 8.4% respectively while improving throughput by 6.4%, with only a slight increase in BRAM.
2. **6.1% improvement in power efficiency**: Increased from 130.36 to 138.27 Mpixels/s/W.
3. **Rate-distortion consistency**: The FPGA implementation maintains rate-distortion performance consistent with the IPC reference software, with a latency of 73.01 ms.
4. **DV comparison engine dominates resource consumption**: The GCLI_CAL module occupies the vast majority of logical resources (11.63K LUTs, 19.98K FFs, 17 DSPs).

## Highlights & Insights

1. **Pioneering work**: Proposes the first FPGA implementation of the DV search for the JPEG XS IPC framework.
2. **Innovative memory organization**: Reorganizing storage from precinct-aligned to IPC Group-aligned leverages the grouped computing and data structure of IPC, presenting a clever domain-specific optimization.
3. **Practical orientation**: Designed with ASIC deployment as the ultimate goal, utilizing FPGA as a validation platform with a clear development path.
4. **On-chip TLB design**: Elegantly addresses the varying sizes of coefficient blocks across different IPC Groups by storing block length information in TLB RAM.

## Limitations & Future Work

1. **Implementation of only one module**: Only the DV search module is implemented, leaving other parts of the full IPC prediction loop (pattern compensation, mode decision, etc.) unresolved. It remains far from a complete hardware IPC implementation.
2. **Small FPGA platform**: Utilizing Artix-7 (XC7A35T) with limited resources might restrict the optimization space of the design.
3. **Limited throughput**: 38.3 Mpixels/s is still insufficient for high-resolution real-time applications such as 4K@60fps (approx. 497 Mpixels/s).
4. **Absence of hardware comparison**: Lacks systematic comparisons with intra prediction hardware of other coding standards such as H.264/HEVC.
5. **BRAM trade-off**: Method 1 trades off BRAM to achieve higher throughput and reduced logic resources, which might be restricted in BRAM-constrained scenarios.
6. **Unexplored high parallelism configurations**: Such as multi-DV parallel search or multi-group parallel processing.

## Related Work & Insights

- **Difference from H.264/HEVC motion estimation hardware**: Traditional video encoding hardware is based on pixel-domain block matching and SAD/SATD cost evaluation, whereas IPC works in the wavelet domain with grouped frequency-domain prediction streams.
- **Complementary relationship with JPEG XS TDC**: TDC performs inter-frame temporal prediction, and IPC performs intra-frame spatial prediction. The two can be combined.
- **Insights for wavelet-domain compression hardware**: The concept of IPC Group-aligned storage can be extended to the hardware implementations of other wavelet-domain compression algorithms.
- **ASIC deployment prospects**: FPGA validation provides resource estimation and architectural references for low-power ASIC design.

## Rating
- Novelty: ⭐⭐⭐⭐ (First FPGA implementation is pioneering, though the technical methods are relatively conventional)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Resource utilization and performance data are complete, but lack comparative baselines and application scenario validations)
- Writing Quality: ⭐⭐⭐⭐⭐ (The architecture description is clear, and the module division is reasonable)
- Value: ⭐⭐⭐⭐ (Of direct value to JPEG XS IPC hardware deployment, but targeting a relatively niche audience)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Memory-Efficient Training with In-Place FFT Implementation](../../NeurIPS2025/model_compression/memory-efficient_training_with_in-place_fft_implementation.md)
- [\[ICCV 2025\] SSVQ: Unleashing the Potential of Vector Quantization with Sign-Splitting](../../ICCV2025/model_compression/ssvq_unleashing_the_potential_of_vector_quantization_with_sign-splitting.md)
- [\[ACL 2026\] Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis](../../ACL2026/model_compression/analytical_ffn-to-moe_restructuring_via_activation_pattern_analysis.md)
- [\[CVPR 2026\] Real-Time Neural Video Compression with Unified Intra and Inter Coding](../../CVPR2026/model_compression/real-time_neural_video_compression_with_unified_intra_and_inter_coding.md)
- [\[NeurIPS 2025\] Learning to Better Search with Language Models via Guided Reinforced Self-Training](../../NeurIPS2025/model_compression/learning_to_better_search_with_language_models_via_guided_reinforced_self-traini.md)

</div>

<!-- RELATED:END -->
