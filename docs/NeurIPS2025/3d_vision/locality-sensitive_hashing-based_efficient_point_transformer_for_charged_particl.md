---
title: >-
  [Paper Note] Locality-Sensitive Hashing-Based Efficient Point Transformer for Charged Particle Reconstruction
description: >-
  [NeurIPS 2025][3D Vision][Point Transformer] By combining LSH with Point Transformer, the paper proposes HEPTv2 for end-to-end particle track reconstruction…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Point Transformer"
  - "Locality-Sensitive Hashing"
  - "Particle Tracking"
  - "End-to-End Learning"
date: 2026-05-08
content_hash: d1520a2e3f5f6ebc
---

# Locality-Sensitive Hashing-Based Efficient Point Transformer for Charged Particle Reconstruction

**Conference**: NeurIPS 2025
**arXiv**: [2510.07594](https://arxiv.org/abs/2510.07594)  
**Code**: Available  
**Area**: 3D Vision / Particle Physics
**Keywords**: Point Transformer, Locality-Sensitive Hashing, Particle Tracking, End-to-End Learning

## TL;DR

By combining LSH with Point Transformer, the paper proposes HEPTv2 for end-to-end particle track reconstruction, eliminating the DBScan clustering post-processing bottleneck and achieving a 28.9× speedup while maintaining competitive tracking efficiency.

## Background & Motivation

**Background**: Particle track reconstruction in LHC high-energy physics experiments is among the most computationally intensive tasks; traditional Kalman Filters degrade under high pile-up conditions.

**Limitations of Prior Work**: Although GNNs deliver strong performance, they suffer from three major issues: high graph construction cost $O(n^2)$, hardware-inefficient irregular neighborhood aggregation, and random memory access patterns that harm cache utilization. Although HEPT introduces LSH for linear complexity, the additional DBScan clustering step consumes 90% of the total runtime.

**Key Challenge**: Fast encoding vs. the complete task (requiring track assignment); expressiveness vs. hardware friendliness.

**Core Idea**: Extend HEPT to HEPTv2 by incorporating a lightweight query-based Transformer decoder that directly predicts track assignments.

## Method

### Overall Architecture

A three-stage pipeline: (1) Metric learning (LSH encoding) — hashing detector hits into a 1D sequence; (2) Instance decoding — a query-based decoder refines track hypotheses; (3) Assignment and post-processing — associating hits to the most probable tracks.

### Key Designs

1. **LSH Encoder**

    - Function: The E2LSH scheme maps nearby hits to the same 1D bucket, enabling block-diagonal attention.
    - Mechanism: The OR construction uses $m_1$ independent hash tables; the AND construction concatenates $m_2$ hash functions per table, $h_j(x) = \lfloor(a_j \cdot x + b_j)/r\rfloor$.
    - Design Motivation: Regular memory access patterns are GPU-friendly; intra-bucket self-attention incurs $O(1)$ cost.

2. **End-to-End Track Assignment Decoder**

    - Function: A fixed set of 3,000 learnable track queries predicts track assignments via self-attention and cross-attention.
    - Mechanism: A binary hit classifier determines whether a hit belongs to a track; the query-based decoder (self-attention → cross-attention → feed-forward layer) outputs per-query confidence scores and dense mask logits.
    - Design Motivation: Eliminates DBScan post-processing, adding only 17% computational overhead (4 ms) compared to the 1,401 ms required by DBScan.

3. **Joint Loss Function**

    - Function: Five loss terms are jointly optimized.
    - Mechanism: $\mathcal{L} = \lambda_{nce}\mathcal{L}_{NCE} + \lambda_{clf}\mathcal{L}_{CLF} + \lambda_{ce}\mathcal{L}_{CE} + \lambda_{mask}\mathcal{L}_{BCE} + \lambda_{dice}\mathcal{L}_{Dice}$
    - Design Motivation: The InfoNCE contrastive loss clusters hits from the same particle, complemented by classification and mask losses, covering the full pipeline from embedding to assignment.

### Loss & Training

Curriculum learning: the model is first trained on clean, trackable hits, with hard samples and low-momentum hits gradually introduced.

## Key Experimental Results

### Main Results (TrackML Dataset)

| Method | Tracking Efficiency | Fake Rate | Inference Time (ms) | Relative Speedup |
|---|---|---|---|---|
| Exa.TrkX (GNN SOTA) | 0.994 | 0.002 | ~800 | Baseline |
| HEPT + DBScan | 0.923 | 0.070 | 1425 | 0.56× |
| **HEPTv2** | **0.993** | **0.113** | **27.7** | **28.9×** |

### Ablation Study

| Configuration | Time | Note |
|---|---|---|
| HEPT encoder | 23.7 ms | No track assignment |
| + Decoder | 27.7 ms | Only +17% overhead |
| vs. HEPT + DBScan | 1425 ms | 50× slower |

### Key Findings

- The elevated fake rate (0.002 → 0.113) is acceptable — offline reconstruction is less sensitive to fake tracks than online triggering.
- Across different momentum ranges and pseudorapidity regions, HEPTv2 differs from Exa.TrkX by only 0.2%.
- The encoder has only 850K parameters; the decoder adds 250K, totaling 1.1M — extremely lightweight.

## Highlights & Insights

- **Truly end-to-end tracking**: This is the first application of an LSH Transformer to a complete physics tracking pipeline without external clustering. The approach is broadly inspirational for detection and segmentation tasks that rely on post-processing.
- **Hardware-friendly**: A latency of 28 ms/event is acceptable for online trigger environments (10 kHz readout rate), suggesting practical deployability.
- **Reasonable trade-off**: Accepting a modest increase in fake rate in exchange for a 30× speedup is well-justified given the practical requirements of physics experiments.

## Limitations & Future Work

- The fake rate gap relative to GNNs remains the primary weakness (0.113 vs. 0.002); more sophisticated mask refinement may be required.
- The current work is limited to pixel detectors; the full HL-LHC system includes strip detectors (approximately 6× more hits).
- The fixed 3,000 queries may be redundant for simple events and insufficient for highly complex ones.

## Related Work & Insights

- **vs. HEPT**: HEPT produces embeddings only and relies on DBScan (accounting for 90% of runtime); HEPTv2 eliminates this bottleneck end-to-end.
- **vs. Mask3D**: The decoder design draws on the extension of Mask2Former ideas to 3D.

## Rating

- Novelty: ⭐⭐⭐⭐ A natural extension of HEPT; the primary contribution lies in the application.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive on pixels; Strip/HL-LHC validation remains to be done.
- Writing Quality: ⭐⭐⭐⭐ Clear and well-structured.
- Value: ⭐⭐⭐⭐⭐ A critical application in high-energy physics; the 30× speedup is highly significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](../../ICCV2025/3d_vision/efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[NeurIPS 2025\] How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?](how_many_tokens_do_3d_point_cloud_transformer_architectures_really_need.md)
- [\[CVPR 2026\] GGPT: Geometry-Grounded Point Transformer](../../CVPR2026/3d_vision/ggpt_geometry_grounded_point_transformer.md)
- [\[NeurIPS 2025\] Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)
- [\[ICCV 2025\] Nautilus: Locality-aware Autoencoder for Scalable Mesh Generation](../../ICCV2025/3d_vision/nautilus_locality-aware_autoencoder_for_scalable_mesh_generation.md)

</div>

<!-- RELATED:END -->
