---
title: >-
  [Paper Note] An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS
description: >-
  [CVPR 2026][Model Compression][FPGA] To address the computational bottleneck of Displacement Vector (DV) search in the Intra Pattern Copy (IPC) module for JPEG XS screen content coding, this paper proposes the first four-stage pipeline FPGA architecture and designs an IPC Group-aligned memory organization scheme. Implemented on a Xilinx Artix-7, the design achieves a throughput of 38.3 Mpixels/s at 277 mW power consumption, providing a viable solution for practical hardware deployment of IPC.
tags:
  - CVPR 2026
  - Model Compression
  - FPGA
  - JPEG XS
  - Intra Pattern Copy
  - Displacement Vector Search
  - Hardware Acceleration
date: 2026-05-08
content_hash: 1a1113fa0844e999
---

# An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS

**Conference**: CVPR 2026
**arXiv**: [2603.10671](https://arxiv.org/abs/2603.10671)
**Code**: None
**Area**: Model Compression
**Keywords**: FPGA, JPEG XS, Intra Pattern Copy, Displacement Vector Search, Hardware Acceleration

## TL;DR

To address the computational bottleneck of Displacement Vector (DV) search in the Intra Pattern Copy (IPC) module for JPEG XS screen content coding, this paper proposes the first four-stage pipeline FPGA architecture and designs an IPC Group-aligned memory organization scheme. Implemented on a Xilinx Artix-7, the design achieves a throughput of 38.3 Mpixels/s at 277 mW power consumption, providing a viable solution for practical hardware deployment of IPC.

## Background & Motivation

JPEG XS is a low-latency, low-complexity image compression standard developed by the JPEG committee for applications such as remote desktop, KVM, and immersive video. To improve its coding efficiency for screen content, Intra Pattern Copy (IPC) was proposed, performing intra-frame prediction in the wavelet domain to remove spatial redundancy and achieving significant BD-PSNR gains.

However, **DV Search** is the most computationally intensive module in the IPC pipeline: it must traverse all candidate prediction offsets, compute residuals, and select the optimal DV that minimizes coding cost. The high computational complexity and irregular memory access patterns of this process constitute a critical bottleneck for real-time hardware deployment.

Although FPGA implementations of H.264/HEVC motion estimation are mature, they target fixed-block partitioning in the pixel domain rather than frequency-domain prediction streams organized by IPC Groups and IPC Units in the JPEG XS wavelet domain. Therefore, **a dedicated FPGA architecture tailored to the JPEG XS IPC framework is required**.

Core Idea: A four-stage pipeline architecture decouples and parallelizes residual computation and DV comparison, while an IPC Group-aligned memory organization eliminates the overhead of scattered wavelet coefficient accesses.

## Method

### Overall Architecture

The system consists of two main engines: the **Residual Calculation Engine** and the **DV Comparison Engine**. The inputs are original and reconstructed wavelet coefficients after RCT and DWT, stored in two IPC Unit banks in DRAM. The Residual Calculation Engine reads IPC Units from memory and computes block-level residuals; the DV Comparison Engine evaluates the coding bit cost of each residual and searches for the optimal DV.

### Key Designs

1. **Residual Calculation Engine**:

    - Function: Reads original and reconstructed coefficient blocks from DRAM and computes signed residuals.
    - Mechanism: A CMD module maps precinct indices to memory addresses; FIFO arrays (Q0–Q3 for original data, C0–C3 for reconstructed data) buffer the coefficients; a CTRL module coordinates read/write synchronization. The SIG_MAG_SUB module splits 32-bit sign-magnitude values and computes residuals via four parallel subtraction paths.
    - Design Motivation: IPC coefficients originate from different subbands and must be organized by Group. The FIFO array design enables data from the same IPC Group to be fed sequentially, avoiding out-of-order memory accesses.

2. **4-Stage Pipeline DV Comparison Engine**:

    - Function: Evaluates the coding cost of each candidate DV and selects the optimal DV.
    - Mechanism: Stage 0 loads residual coefficients and Group parameters (BandIdx, GrpSize, UnitWidth); Stage 1's GetOrMask module performs bitwise OR over residuals within a Group to generate OrIdx and OrAll; Stage 2's CalGCLI module computes the GCLI coding cost BitsTest from the OR results; Stage 3's Compare module compares the current BitsTest against the historical minimum cost BitsBest and selects the optimal DV via a MUX.
    - Design Motivation: Decomposing the comparison process into a four-stage pipeline parallelizes residual computation and DV comparison, achieving a balance between latency and throughput.

3. **IPC Group-Aligned Memory Organization (Method 1)**:

    - Function: Reorganizes the storage layout of wavelet coefficients in DRAM.
    - Mechanism: Unlike Method 0, which stores coefficients linearly by precinct (causing coefficients of the same IPC Unit to be scattered across different locations), Method 1 organizes storage by IPC Group and IPC Unit, with Units within the same Group arranged sequentially and each Unit containing all subband blocks. This allows an entire IPC Unit to be loaded with a single base address plus a fixed offset, supporting burst reads.
    - Design Motivation: The IPC access pattern traverses all Units in Group order. Method 0 requires three-level indexing by group/unit/band, resulting in complex control and low throughput; Method 1 naturally matches this access pattern.

4. **On-chip TLB RAM**:

    - Function: Stores variable-length size information for coefficient blocks across different IPC Groups.
    - Mechanism: Since block sizes vary across Groups (depending on the level of wavelet decomposition), the CMD module uses an on-chip TLB lookup to generate correct entry addresses and updates the TLB when DV search switches to the next precinct.
    - Design Motivation: Avoids the overhead of dynamically computing variable-length block addresses at runtime.

### Loss & Training

This paper presents a hardware design work with no training process. The optimization objective is to maintain rate-distortion performance consistent with the IPC reference software while minimizing latency, resource utilization, and power consumption.

## Key Experimental Results

### Main Results

| Parameter | Method 0 (Baseline) | Method 1 (Proposed) |
|-----------|---------------------|---------------------|
| Platform | Xilinx Artix-7, 100 MHz | Same |
| Throughput (Mpixels/s) | 35.98 | **38.30** |
| Power (mW) | 276 | 277 |
| Power Efficiency (Mpixels/s/W) | 130.36 | **138.27** |
| LUTs (K) | 13.93 | **12.89** |
| FFs (K) | 23.80 | **21.79** |
| DSPs | 17 | 17 |
| BRAM | 11 | 15 |

### Ablation Study: Module Resource Utilization

| Module | LUTs (K) | FFs (K) | DSPs | BRAM |
|--------|----------|---------|------|------|
| Residual Calculation Engine | 0.48 | 0.47 | 0 | 15 |
| GCLI_CAL (DV Comparison) | 11.63 | 19.98 | 17 | 0 |
| DV_UPDATE (DV Comparison) | 0.73 | 1.41 | 0 | 0 |

### Key Findings

- Method 1 improves throughput by 6.4% and power efficiency by 6.1% over Method 0.
- LUT and FF resource usage are reduced by 7.5% and 8.4%, respectively, at the cost of only 4 additional BRAMs.
- The GCLI_CAL module in the DV Comparison Engine accounts for the vast majority of logic resource consumption (~90% of LUTs) and represents the primary optimization target.
- Latency is 73.01 ms, and rate-distortion performance is consistent with the IPC reference software.

## Highlights & Insights

- **First FPGA implementation of IPC DV search**: Fills the gap in hardware implementation of JPEG XS IPC.
- **Co-design of memory organization and access patterns**: The core insight of Method 1 is aligning storage layout with the traversal order of computation.
- **Appropriate pipeline granularity**: The four-stage pipeline avoids excessive stage splitting, achieving a balance between area and throughput.

## Limitations & Future Work

- Validated only on the relatively small-scale Artix-7 device; not evaluated on higher-end FPGAs or ASICs.
- The throughput of 38.3 Mpixels/s remains far below the requirements for real-time 4K processing (~500 Mpixels/s).
- Not integrated and tested with a complete JPEG XS IPC encoder; system-level bottlenecks remain unclear.
- Supports only a single wavelet decomposition configuration (5 horizontal, 2 vertical levels), limiting flexibility.

## Related Work & Insights

- Mature FPGA implementations of H.264/HEVC motion estimation provide established paradigms for pipeline and memory optimization, but frequency-domain prediction streams require an entirely new memory organization.
- JPEG XS's TDC (Temporal Difference Coding) is complementary to IPC; future work may need to integrate hardware implementations of both.
- The Group-aligned memory organization approach is generalizable to other wavelet-domain processing tasks requiring traversal along specific dimensions.

## Rating

- Novelty: ⭐⭐⭐ The architectural design approach is relatively conventional (pipeline + memory optimization), but represents a first in the JPEG XS IPC domain.
- Experimental Thoroughness: ⭐⭐⭐ Evaluated on only one FPGA platform; lacks comparison with similar hardware encoders.
- Writing Quality: ⭐⭐⭐⭐ Architecture descriptions are clear and the memory organization comparison is intuitive.
- Value: ⭐⭐⭐ Practically advances JPEG XS hardware implementation, though the scope of impact is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](rdvq_differentiable_vq_image_compression.md)
- [\[CVPR 2026\] Distilling Balanced Knowledge from a Biased Teacher](distilling_balanced_knowledge_from_a_biased_teacher.md)
- [\[CVPR 2026\] ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation](arche_autoregressive_residual_compression_with_hyp.md)
- [\[CVPR 2026\] Unlocking ImageNet's Multi-Object Nature: Automated Large-Scale Multilabel Annotation](unlocking_imagenets_multi-object_nature_automated_large-scale_multilabel_annotat.md)

</div>

<!-- RELATED:END -->
