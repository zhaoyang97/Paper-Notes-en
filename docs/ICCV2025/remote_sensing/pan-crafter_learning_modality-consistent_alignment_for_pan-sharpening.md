---
title: >-
  [Paper Note] Pan-Crafter: Learning Modality-Consistent Alignment for Pan-Sharpening
description: >-
  [ICCV 2025][Remote Sensing][Pan-sharpening] PAN-Crafter proposes a modality-consistent alignment framework that explicitly addresses cross-modal misregistration between PAN and MS images via Modality-Adaptive Reconstruct…
tags:
  - "ICCV 2025"
  - "Remote Sensing"
  - "Pan-sharpening"
  - "cross-modal alignment"
  - "modality-adaptive reconstruction"
  - "attention mechanism"
  - "remote sensing image fusion"
date: 2026-05-08
content_hash: dacd677a7c285694
---

# Pan-Crafter: Learning Modality-Consistent Alignment for Pan-Sharpening

**Conference**: ICCV 2025
**arXiv**: [2505.23367](https://arxiv.org/abs/2505.23367)  
**Code**: [https://kaist-viclab.github.io/PAN-Crafter_site](https://kaist-viclab.github.io/PAN-Crafter_site)  
**Area**: Remote Sensing / Pan-Sharpening
**Keywords**: Pan-sharpening, cross-modal alignment, modality-adaptive reconstruction, attention mechanism, remote sensing image fusion

## TL;DR

PAN-Crafter proposes a modality-consistent alignment framework that explicitly addresses cross-modal misregistration between PAN and MS images via Modality-Adaptive Reconstruction (MARs) and Cross-Modal Misalignment-aware Multi-scale Attention (CM3A), achieving state-of-the-art performance on multiple remote sensing benchmarks while running **1110×** faster than diffusion-based methods.

## Background & Motivation

Pan-sharpening in remote sensing aims to fuse high-resolution panchromatic (PAN, single-channel) images with low-resolution multispectral (MS, multi-channel) images to produce high-resolution multispectral (HRMS) output—a core requirement in practical satellite image processing.

**Core Pain Points**: Due to differences in sensor position, acquisition time offsets, and resolution mismatches, **cross-modal spatial misregistration** exists between PAN and MS images. However:

**Existing methods assume perfect alignment**: Most deep learning methods employ pixel-wise losses ($\ell_1$/$\ell_2$), which cause spectral distortion, ghosting artifacts, and blurring in the presence of misregistration.

**Adaptive correction lacks flexibility**: SIPSA relies on fixed-scale alignment; LAGConv/CANConv use self-similarity aggregation rather than explicit geometric alignment.

**Diffusion models yield high quality but are too slow**: PanDiff and TMDiff have inference times of 2.955s and 9.997s respectively, making deployment impractical.

**Key Insight**: Design a bidirectional alignment mechanism—aligning PAN structure to MS texture (for HRMS reconstruction) and, conversely, aligning MS texture to PAN structure (via PAN back-reconstruction as auxiliary self-supervision)—forming a "modality-consistent" constraint.

## Method

### Overall Architecture

PAN-Crafter adopts a U-Net encoder-decoder architecture built around two core modules:
1. **MARs** (Modality-Adaptive Reconstruction): a single network jointly learns to reconstruct HRMS and to back-reconstruct PAN.
2. **CM3A** (Cross-Modal Misalignment-aware Multi-scale Attention): performs multi-scale bidirectional alignment of PAN and MS features.

### Key Designs

#### 1. Modality-Adaptive Reconstruction (MARs)

MARs enables a single network to operate alternately in two modes:

- **MS mode**: Input PAN + LRMS → Output HRMS
  $$\hat{\mathbf{I}}_{\text{ms}}^{\text{hr}} = \mathcal{P}_\theta(\mathbf{I}_{\text{pan}}, \mathbf{I}_{\text{ms}}^{\text{lr}}; \text{mode}=\mathsf{MS}) + \mathbf{I}_{\text{ms}}^{\text{lr}}$$

- **PAN mode**: Input PAN + LRMS → Back-reconstruct PAN (replicated to multi-channel)
  $$\hat{\mathbf{I}}_{\text{pan}}^{\text{rep}} = \mathcal{P}_\theta(\mathbf{I}_{\text{pan}}, \mathbf{I}_{\text{ms}}^{\text{lr}}; \text{mode}=\mathsf{PAN}) + \mathbf{I}_{\text{pan}}^{\text{lr,rep}}$$

**Design Motivation**: PAN back-reconstruction serves as an auxiliary self-supervised signal. Its key advantage is that the PAN image itself constitutes a readily available ground truth—no additional annotation is required. By compelling the network to reconstruct a sharp PAN image, the network is forced to internalize fine spatial structures, which in turn improves the spatial detail quality of the HRMS output.

During training, each batch is duplicated: one copy follows MS mode and the other follows PAN mode. Only MS mode is used at inference.

MARs loss:
$$\mathcal{L}_{\text{MARs}} = \|\hat{\mathbf{I}}_{\text{ms}}^{\text{hr}} - \mathbf{I}_{\text{ms}}^{\text{hr}}\|_1 + \lambda \|\hat{\mathbf{I}}_{\text{pan}}^{\text{rep}} - \mathbf{I}_{\text{pan}}^{\text{rep}}\|_1$$

#### 2. Cross-Modal Misalignment-aware Multi-scale Attention (CM3A)

CM3A is the other core contribution of PAN-Crafter, performing bidirectional cross-modal alignment at multi-scale feature levels.

**Key Design**: Local attention ($k \times k$ windows, $k=3$) replaces global attention, since PAN–MS pairs are typically already roughly pre-aligned and only local correction is needed. This reduces computational complexity from $O(2(HW)^2 C)$ to $O(2(HW)k^2 C)$.

In **MS mode**, queries are constructed from LRMS features, and two attention operations are performed simultaneously:
- **Self-attention**: Query interacts with MS key-value pairs to maintain MS feature consistency.
- **Alignment attention**: Query interacts with PAN key-value pairs to incorporate PAN structural information.

$$\mathbf{x}_{\text{ms}} = \text{LocalAttn}(\mathbf{Q}, \mathbf{K}_{\text{ms}}, \mathbf{V}_{\text{ms}})$$
$$\mathbf{x}_{\text{pan}} = \text{LocalAttn}(\mathbf{Q}, \mathbf{K}_{\text{pan}}, \mathbf{V}_{\text{pan}})$$

In **PAN mode**, the operations are mirrored—queries are constructed from PAN features to ensure structural consistency in PAN back-reconstruction.

**Novelty**: Downsampled raw images replace conventional fixed positional encodings by being concatenated into Q/K, allowing the network to implicitly learn relative inter-modal displacement.

#### 3. Modality Modulation

Within each ResBlock, learnable parameters $\gamma, \beta$ modulate features according to the MARs mode:

$$\mathsf{Modulate}(\mathbf{x}; \mathsf{MS}): \mathbf{x} \leftarrow (1 + \gamma_{\text{ms}}) \odot \mathbf{x} + \beta_{\text{ms}}$$

This ensures that the shared network can adapt to the feature distributions of different modalities.

### Loss & Training

- AdamW optimizer, initial learning rate $1 \times 10^{-4}$, cosine annealing
- Batch size 48 (effective batch size 96 due to MARs duplication)
- 50K training iterations with 100-step warmup
- $\lambda = 1.0$, feature dimension $C = 128$

## Key Experimental Results

### Main Results

**Performance comparison on the WV3 dataset**:

| Method | HQNR↑ | ERGAS↓ | PSNR↑ | Inference Time↓ (s) | Memory↓ (GB) |
|---|---|---|---|---|---|
| CANConv (CVPR24) | 0.951 | 2.163 | 37.441 | 0.451 | 2.713 |
| PanDiff (TGRS23) | 0.952 | 2.276 | 37.029 | 2.955 | 2.328 |
| TMDiff (TGRS24) | 0.924 | 2.151 | 37.477 | 9.997 | 9.910 |
| **PAN-Crafter** | **0.958** | **2.040** | **37.956** | **0.009** | **1.711** |

PAN-Crafter is **1110×** faster than TMDiff and **50×** faster than CANConv, while achieving the best results on all core metrics.

**GF2 dataset**: PSNR reaches **45.076**, nearly 2 dB higher than the second-best CANConv (43.166).

### Ablation Study

| CM3A | MARs | HQNR↑ | ERGAS↓ | PSNR↑ | Time (s) |
|---|---|---|---|---|---|
| ✗ | ✗ | 0.948 | 2.232 | 37.245 | 0.006 |
| ✗ | ✓ | 0.956 | 2.122 | 37.602 | 0.009 |
| ✓ | ✗ | 0.949 | 2.212 | 37.285 | 0.007 |
| **✓** | **✓** | **0.958** | **2.040** | **37.956** | **0.009** |

Key findings:
- MARs contributes more than CM3A in isolation (PSNR: +0.357 vs. +0.040).
- However, combining both yields a **synergistic effect**—the combined gain (+0.711) substantially exceeds the sum of individual gains.

### Key Findings

1. **Zero-shot generalization**: On unseen WV2 satellite data, PAN-Crafter achieves an HQNR of **0.942**, substantially outperforming competing methods.
2. **Self-supervised effect of MARs**: PAN back-reconstruction as an auxiliary task significantly improves spatial sharpness on the primary task.
3. **CM3A + MARs synergy**: MARs enables bidirectional interaction, upon which CM3A achieves more precise alignment.
4. **Efficiency**: The entire framework has only 7.17M parameters and 79.03G FLOPs.

## Highlights & Insights

- **PAN back-reconstruction as self-supervision**: The method cleverly leverages the PAN image itself as a free supervisory signal, requiring no additional annotation.
- **Local attention replacing positional encoding**: Using raw image features instead of conventional positional encodings is better suited to handling uncertain spatial misregistration.
- **High practical value**: The method comprehensively outperforms diffusion-based models while being three orders of magnitude faster, making it well-suited for real-world remote sensing processing pipelines.

## Limitations & Future Work

- Inter-band misregistration among multispectral channels is not explicitly addressed; the current work only handles misregistration between PAN and MS.
- Depthwise separable convolutions for inter-band alignment are identified as a potential improvement direction (noted in the paper).
- The local attention window size $k=3$ may be insufficient for large misregistrations.

## Related Work & Insights

- **CANConv** [CVPR 2024]: Clustering-based spatially adaptive convolution, but k-means dependency results in slow inference.
- **SIPSA** [2022]: First work to identify misregistration as a critical challenge in pan-sharpening.
- **PanDiff / TMDiff**: Diffusion-based methods with high output quality but prohibitive inference cost.
- Insight: The dual-task self-supervision paradigm (primary task + inverse task) is generalizable to other cross-modal fusion problems.

## Rating

- Novelty: ⭐⭐⭐⭐ — The MARs bidirectional reconstruction and CM3A alignment mechanism are novel in design.
- Technical Depth: ⭐⭐⭐⭐ — Modality modulation, local attention as a PE replacement, and other design details are carefully considered.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four satellite datasets, zero-shot generalization, and efficiency comparisons.
- Practicality: ⭐⭐⭐⭐⭐ — Fast inference, low memory footprint, and strong performance make it highly suitable for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] C3PO: Cross-View Cross-Modality Correspondence by Pointmap Prediction](../../NeurIPS2025/remote_sensing/c3po_cross-view_cross-modality_correspondence_by_pointmap_prediction.md)
- [\[ICCV 2025\] WildSAT: Learning Satellite Image Representations from Wildlife Observations](wildsat_learning_satellite_image_representations_from_wildlife_observations.md)
- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](../../CVPR2026/remote_sensing/cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)
- [\[NeurIPS 2025\] OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning](../../NeurIPS2025/remote_sensing/orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)
- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Dataset for Ionospheric Prediction](../../NeurIPS2025/remote_sensing/connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)

</div>

<!-- RELATED:END -->
