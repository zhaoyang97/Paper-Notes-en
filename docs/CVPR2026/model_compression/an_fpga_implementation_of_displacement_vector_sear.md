---
title: >-
  [Paper Note] An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS
description: >-
  [CVPR 2026][Model Compression][FPGA] This paper presents the first FPGA hardware acceleration architecture for the Intra Pattern Copy (IPC) tool in the JPEG XS standard. Through a four-stage pipelined DV comparison engine and IPC Group-aligned memory organization, the design achieves 38.3 Mpixels/s throughput and 277 mW power consumption on an Artix-7 FPGA.
tags:
  - CVPR 2026
  - Model Compression
  - FPGA
  - JPEG XS
  - Intra Pattern Copy
  - Displacement Vector Search
  - Pipeline Architecture
date: 2026-05-08
content_hash: 355ab189e91f039b
---

# An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS

**Conference**: CVPR 2026
**arXiv**: [2603.10671](https://arxiv.org/abs/2603.10671)
**Code**: None
**Area**: Model Compression / Hardware Implementation
**Keywords**: FPGA, JPEG XS, Intra Pattern Copy, Displacement Vector Search, Pipeline Architecture

## TL;DR

This paper presents the first FPGA hardware acceleration architecture for the Intra Pattern Copy (IPC) tool in the JPEG XS standard. Through a four-stage pipelined DV comparison engine and IPC Group-aligned memory organization, the design achieves 38.3 Mpixels/s throughput and 277 mW power consumption on an Artix-7 FPGA.

## Background & Motivation

**Background**: JPEG XS is an image compression standard targeting low latency and low complexity, applicable to scenarios such as remote desktop and KVM. Intra Pattern Copy (IPC) is an encoding tool within this standard that performs intra-frame prediction in the wavelet domain to reduce spatial redundancy in screen content.

**Limitations of Prior Work**: The displacement vector (DV) search module in IPC requires exhaustive evaluation of all candidate offsets to select the optimal prediction reference, imposing extremely high computational demands. Although numerous motion estimation hardware implementations exist for H.264/HEVC, these designs operate in the pixel domain and are incompatible with the wavelet-domain group-based prediction pipeline of JPEG XS.

**Key Challenge**: There is a fundamental tension between the high computational complexity of DV search and the low-latency requirements of JPEG XS. Wavelet coefficients are organized into IPC Groups and Units with highly irregular access patterns, and conventional memory layouts result in high control complexity and poor bandwidth utilization.

**Goal**: To design an FPGA implementation of DV search that enables efficient hardware deployment of IPC.

**Key Insight**: A four-stage pipeline and optimized memory organization are designed specifically for the group-based prediction flow inherent to IPC.

**Core Idea**: Efficient hardware deployment of JPEG XS IPC is achieved through IPC Group-aligned memory organization and a four-stage pipelined DV comparison architecture.

## Method

### Overall Architecture

The system comprises two core engines: a residual computation engine (which reads original and reconstructed IPC Units from DRAM and computes residuals) and a DV comparison engine (which evaluates the bit cost of each residual and searches for the optimal DV). The input consists of wavelet coefficients following RCT and DWT, and the output is the optimal DV.

### Key Designs

1. **Four-Stage Pipelined DV Comparison Architecture**:

    - **Function**: Decomposes the DV comparison process into four pipeline stages executed in parallel.
    - **Mechanism**: Stage 0 loads residual coefficients and computes configuration parameters (BandIdx, GrpSize, UnitWidth); Stage 1's GetOrMask module computes the bitwise OR mask within each group; Stage 2's CalGCLI module computes the residual bit cost BitsTest; Stage 3's Compare module evaluates and updates the optimal DV (BestDV).
    - **Design Motivation**: DV comparison is the bottleneck of the entire DV search process. Pipelining allows the evaluation of consecutive DV candidates to overlap, significantly improving throughput.

2. **IPC Group-Aligned Memory Organization (Method 1)**:

    - **Function**: Reorganizes the storage layout of wavelet coefficients in DRAM from Precinct-based to IPC Group/Unit-based organization.
    - **Mechanism**: IPC Units belonging to the same Group are stored sequentially, with each Unit containing all subband blocks. An entire IPC Unit can be loaded using a single base address plus a fixed offset, enabling burst access patterns. An on-chip TLB RAM stores block sizes for different Groups.
    - **Design Motivation**: Under the original Precinct-based layout, coefficients from different subbands are scattered across memory, requiring individual address lookups, which results in high control complexity and poor bandwidth utilization.

3. **Residual Computation Engine**:

    - **Function**: Reads original and reconstructed coefficient blocks from DRAM and computes grouped residuals.
    - **Mechanism**: The CMD module performs address mapping; data flows into four groups of FIFOs, Q0–Q3 (original) and C0–C3 (reconstructed). The CTRL module manages synchronized read/write operations, and the SIG_MAG_SUB module performs four-way parallel subtraction on sign-magnitude formatted 32-bit coefficients to compute residuals, which are stored in residual FIFOs R0–R3.
    - **Design Motivation**: Residual computation requires synchronized access to the same Group in both original and reconstructed data; the FIFO array combined with the CTRL synchronization mechanism ensures data alignment.

### Loss & Training

Not applicable (hardware design, not a learning-based method). The design objective is to maintain rate-distortion performance consistent with the IPC reference software.

## Key Experimental Results

### Main Results

| Parameter | Method 0 (Baseline) | Method 1 (Ours) | Gain |
|-----------|---------------------|-----------------|------|
| Throughput (Mpixels/s) | 35.98 | 38.30 | +6.4% |
| Power (mW) | 276 | 277 | ≈Parity |
| Power Efficiency (Mpixels/s/W) | 130.36 | 138.27 | +6.1% |
| LUTs (K) | 13.93 | 12.89 | −7.5% |
| FFs (K) | 23.80 | 21.79 | −8.4% |
| DSPs | 17 | 17 | Parity |
| BRAM | 11 | 15 | +4 |

### Ablation Study (Module Resource Utilization)

| Module | LUTs (K) | FFs (K) | DSPs | BRAM |
|--------|----------|---------|------|------|
| Residual Computation Engine | 0.48 | 0.47 | 0 | 15 |
| DV Comparison – GCLI_CAL | 11.63 | 19.98 | 17 | 0 |
| DV Comparison – DV_UPDATE | 0.73 | 1.41 | 0 | 0 |

### Key Findings
- The GCLI_CAL module within the DV comparison engine dominates resource consumption (90%+ of LUTs and FFs), constituting the primary area bottleneck.
- The Method 1 memory organization improves throughput while simultaneously reducing LUT and FF usage (by 7.5% and 8.4%, respectively), at the cost of four additional BRAMs for the TLB.
- The system latency of 73.01 ms is acceptable for screen content coding scenarios.

## Highlights & Insights
- **First FPGA implementation of JPEG XS IPC**: Fills a gap in JPEG XS hardware acceleration and provides a reference for ASIC deployment.
- **Generalizable memory organization strategy**: The principle of organizing memory according to computational access patterns rather than the original data layout is transferable to other wavelet/transform-domain coding tools.

## Limitations & Future Work
- Only the DV search module has been implemented; the complete IPC framework (mode selection, compensation, etc.) has not yet been validated in hardware.
- The target platform, Artix-7 (XC7A35T), is relatively small; actual ASIC deployment may involve different trade-offs.
- No performance comparison is made against hardware implementations of other image coding standards (e.g., HEVC SCC).
- The paper lacks a detailed analysis of rate-distortion deviation relative to the software reference.

## Related Work & Insights
- **vs. H.264/HEVC Motion Estimation Hardware**: Prior designs operate in the pixel domain, using SAD/SATD cost metrics and fixed block partitions; the proposed design operates in the wavelet domain, using GCLI bit-cost evaluation and group-based prediction.
- **vs. JPEG XS TDC**: TDC performs inter-frame temporal prediction, while IPC performs intra-frame spatial prediction; the two are complementary.

## Rating
- Novelty: ⭐⭐⭐ First IPC FPGA implementation, though the approach is primarily engineering-oriented.
- Experimental Thoroughness: ⭐⭐⭐ Resource and performance data are sufficient, but rate-distortion comparisons are absent.
- Writing Quality: ⭐⭐⭐ Structure is clear, though some details are insufficiently elaborated.
- Value: ⭐⭐⭐ Offers direct practical value for JPEG XS hardware deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](rdvq_differentiable_vq_image_compression.md)
- [\[CVPR 2026\] Distilling Balanced Knowledge from a Biased Teacher](distilling_balanced_knowledge_from_a_biased_teacher.md)
- [\[CVPR 2026\] MEMO: Human-like Crisp Edge Detection Using Masked Edge Prediction](memo_human-like_crisp_edge_detection_using_masked_edge_prediction.md)
- [\[CVPR 2026\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)

</div>

<!-- RELATED:END -->
