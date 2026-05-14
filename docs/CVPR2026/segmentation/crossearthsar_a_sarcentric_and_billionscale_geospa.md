---
title: >-
  [Paper Note] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation
description: >-
  [CVPR 2026][Segmentation][SAR] This paper introduces CrossEarth-SAR, the first billion-scale SAR visual foundation model, which replaces the FFN in each Transformer block of a DINOv2 ViT backbone with a physics-guided sp…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "SAR"
  - "Foundation Model"
  - "Physics-Guided MoE"
  - "Domain Generalization"
  - "Semantic Segmentation"
date: 2026-05-08
content_hash: 517a6dff9e39c736
---

# CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.12008](https://arxiv.org/abs/2603.12008)
**Code**: [GitHub](https://github.com/VisionXLab/CrossEarth-SAR)
**Area**: Remote Sensing / SAR Foundation Model / Domain-Generalizable Semantic Segmentation
**Keywords**: SAR, Foundation Model, Physics-Guided MoE, Domain Generalization, Semantic Segmentation

## TL;DR

This paper introduces CrossEarth-SAR, the first billion-scale SAR visual foundation model, which replaces the FFN in each Transformer block of a DINOv2 ViT backbone with a physics-guided sparse Mixture-of-Experts (MoE) layer. Routing is conditioned on three SAR physical descriptors—directional entropy, equivalent number of looks, and local roughness. The work also contributes a 200K-scale cross-domain pretraining dataset and a benchmark of 22 evaluation settings covering 8 types of domain shift. CrossEarth-SAR achieves state-of-the-art performance on 20 out of 22 cross-domain semantic segmentation benchmarks.

## Background & Motivation

- **Key Challenge**: SAR imaging provides all-weather, all-day Earth observation capabilities, but its domain specificity is extreme. Heterogeneous sensor platforms (Sentinel-1, ALOS-2, Capella), frequency bands (C/L/X), polarization modes (HH/HV/VV/VH), and incidence angles produce a fragmented domain landscape that makes cross-domain generalization highly challenging.
- **Triple SAR Challenges**: (1) Coherent imaging produces multiplicative speckle noise that degrades textural features; (2) side-looking geometry introduces layover, foreshortening, and shadows that distort spatial topology; (3) backscatter is governed by surface roughness and dielectric constant, causing intra-class spectral variability (the same land cover varying drastically with moisture content) and inter-class ambiguity (distinct land covers appearing as similarly dark backgrounds).
- **Limitations of Prior Work**: Existing SAR foundation models (SARATR-X 90M, SatMAE 300M) either focus on object detection or are not designed for cross-domain generalization. Optical foundation models (DINOv2, DINOv3) exhibit limited transferability to the SAR domain. No prior model simultaneously offers large capacity and domain generalizability for SAR semantic segmentation.
- **Key Insight**: Unlocking large-scale SAR generalization requires: (1) billion-scale parameter capacity to absorb extreme domain diversity; (2) sparse activation to control inference cost; and (3) physics-prior-guided routing to stabilize cross-domain expert selection.

## Method

### Overall Architecture

The DINOv2 ViT serves as the backbone, with the standard FFN in each Transformer block replaced by a physics-guided sparse MoE layer. Each layer contains a router $R_\psi$ and $n$ experts $\{E_k\}_{k=1}^n$ initialized from DINOv2 FFN weights. Input SAR images are replicated into 3-channel tensors $X \in \mathbb{R}^{3 \times H \times W}$ and fed to the backbone; simultaneously, three physical descriptors $s \in \mathbb{R}^3$ are computed to assist routing. Final token embeddings are passed to a Mask2Former decoder for segmentation prediction. Three model scales are provided: S (20M active parameters), B (80M active), and L (300M active, 1.3B total parameters).

### Key Designs

**1. SAR Physical Descriptors — Stable Physical Anchors for Routing**

- **Function**: Addresses routing instability in standard MoE routers that rely solely on token embeddings under heterogeneous SAR data, by supplying stable domain-level physical priors.
- **Mechanism**: A log transform $X' = \log(1 + |X|)$ is first applied for numerical stability, followed by computation of three complementary physical quantities:
    - (a) **Directional Entropy** $H_{DE}$: entropy computed from the histogram of Sobel gradient orientations, $H_{DE} = -\sum_i p_i \ln p_i$, capturing imaging geometry characteristics (low values indicate strong linear structures; high values indicate irregular textures).
    - (b) **Equivalent Number of Looks** ENL $= (\mu / \sigma)^2$: reflects speckle intensity and radar system characteristics (high values indicate weak speckle; low values indicate strong noise).
    - (c) **Local Roughness** $R_{LR} = \text{Var}(\mu_j)_{j=1}^M$: variance of spatial block means, characterizing textural variability of target scattering (high values indicate complex texture; low values indicate smooth regions).
- **Design Motivation**: The three descriptors correspond respectively to the three physical challenges of SAR—imaging geometry, radar system noise, and target scattering characteristics. They are concatenated as $s = [H_{DE}, \text{ENL}, R_{LR}] \in \mathbb{R}^3$, providing physics-grounded anchors for cross-domain routing.

**2. Physics-Guided Sparse MoE — Large-Capacity, Low-Cost Domain-Adaptive Architecture**

- **Function**: Scales the model to 1.3B parameters via sparse activation while maintaining inference overhead comparable to a standard FFN.
- **Mechanism**: The physical descriptor $s$ is tiled along the token dimension to form $S \in \mathbb{R}^{B \times N \times 3}$, concatenated with token embeddings $Z \in \mathbb{R}^{B \times N \times C}$, and fed into the router: $\pi = \text{softmax}(W_r[Z \| S] + b_r)$. This produces per-token scores over $n$ experts; top-$k$ experts are selected and their outputs are aggregated via normalized weighted summation $\tilde{z} = \sum_{k \in \mathcal{I}} g_k \cdot E_k(z)$.
- **Design Motivation**: Different experts can specialize in distinct SAR imaging conditions (e.g., polarization, frequency band). Incorporating physical descriptors aligns routing decisions with underlying physical mechanisms, preventing the volatile cross-domain fluctuations that arise when routing relies solely on learned embeddings.

**3. Load Balancing Constraint — Preventing Expert Collapse**

- **Function**: Ensures uniform utilization of all experts and prevents the router from degenerating into always selecting a small subset.
- **Loss**: $\mathcal{L}_{BC} = \lambda_{BC} \cdot n \cdot \sum_{k=1}^n f_k p_k$, where $f_k$ is the fraction of tokens dispatched to expert $k$ and $p_k$ is the mean routing probability, with $\lambda_{BC} = 0.005$. The total training objective is $\mathcal{L} = \mathcal{L}_{seg} + \mathcal{L}_{BC}$.

**4. CrossEarth-SAR-200K Large-Scale Dataset — Supporting Global-Scale Continual Pretraining**

- **Function**: Constructs the first 200K-scale SAR semantic segmentation dataset, covering 109 regions across 6 continents.
- **Mechanism**: The dataset integrates 40K samples with real annotations from public SAR datasets and 163K pseudo-labeled samples (labels generated by the CrossEarth optical model on paired optical imagery and transferred to SAR). Seven semantic categories are included (building, road, water, bare land, forest, cropland, background), with all images cropped or rescaled to 512×512. Pseudo-label quality is validated by four models, achieving a Mean Agreement of 75.88%, surpassing the 63.20% of OpenEarthMap-SAR.

### Loss & Training

- **Continual Pretraining (CPT)**: Trained for 18 epochs on CrossEarth-SAR-200K; batch size 4; AdamW, lr=3e-5; 16×A100 (80 GB).
- **Downstream Fine-tuning**: Backbone frozen; only the Mask2Former decoder trained for 40K iterations; batch size 2; lr=1e-4; single RTX 4090.
- **Earth-Adapter (PEFT)**: A lightweight adapter added on top of the frozen backbone for further improvement, denoted CrossEarth-SAR-L*.

## Key Experimental Results

### Main Results: Single Domain Shift (12 Benchmarks)

| Method | Backbone | Params | Region (N2S) | Region (S2N) | Polar. (VV2F) | Polar. (HH2F) | Complex (C(r)2R) | Complex (C(i)2R) | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| DINOv2 | ViT-L | 300M | 32.3 | 43.8 | 65.7 | 56.8 | 71.3 | 71.7 | 55.5 |
| DINOv3 | ViT-L | 300M | 33.7 | 42.8 | 48.3 | 50.6 | 69.9 | 69.2 | 53.0 |
| SARATR-X | HiViT-B | 90M | 34.6 | 43.2 | 71.3 | 68.5 | 74.5 | 74.2 | 59.7 |
| **CrossEarth-SAR-L** | ViT-L | **1.3B(300M)** | **38.0** | **46.7** | **73.9** | **72.3** | **76.9** | **76.7** | **62.7** |
| **CrossEarth-SAR-L*** | ViT-L | **1.3B(300M)** | 38.0 | 46.7 | 73.9 | 71.8 | 76.9 | 76.7 | **62.7** |

CrossEarth-SAR-L achieves an average improvement of **+7.2 mIoU** over the DINOv2 baseline, with the largest gain of **+15.5 mIoU** on the polarization domain shift (HH2F).

### Main Results: Multiple Domain Shifts (10 Benchmarks)

| Method | Region+Polar. (A2F) | Region+Platform (O2D) | Region+Band (S2A) | Region+Polar.+Band (D2F) | Region+Platform+Band (W2D) | Avg. |
|---|---|---|---|---|---|---|
| DINOv2 | 15.5 | 17.8 | 55.9 | 26.0 | 16.7 | 24.3 |
| SARATR-X | 21.3 | 19.0 | 53.1 | 22.6 | 16.1 | 24.8 |
| **CrossEarth-SAR-L** | **25.0** | **23.7** | **59.1** | 25.1 | **22.2** | **27.7** |
| **CrossEarth-SAR-L*** | 27.0 | 23.1 | 57.9 | 26.5 | 25.6 | **28.5** |

Under multi-domain shift, CrossEarth-SAR-L* achieves an average of 28.5 mIoU, a gain of +4.2 over the baseline.

### Ablation Study

| Ablation | Configuration | mIoU | Gain |
|---|---|---|---|
| Real labels only (40K) | DINOv2 + 40K real | 45.1 | — |
| 200K with pseudo labels | DINOv2 + 200K | 59.4 | **+14.3** |
| MoE only (no constraint) | MoE only | 61.1 | +1.7 |
| + Load balancing $\mathcal{L}_{BC}$ | +BC | 62.2 | +2.8 |
| + Physical descriptors $S$ | +S | 61.6 | +2.2 |
| + Both combined | +BC+S | **62.4** | **+3.0** |
| $n=3$ experts | top-1 | 60.9 | +1.5 |
| $n=6$ experts | top-1 | **62.4** | **+3.0** |
| top-2 activation | $n=6$ | 61.7 | +2.3 |
| top-3 activation | $n=6$ | 61.3 | +0.9 |

### Key Findings

- **Substantial benefit of pseudo-label scale**: The 200K dataset outperforms 40K real labels by 14.3% mIoU; 40K pseudo labels marginally outperform 40K real labels due to broader global coverage, and combining both yields a further +3.6%.
- **Six experts with top-1 routing is optimal**: Increasing top-$k$ leads to performance degradation, suggesting that at the 200K data scale, single-expert specialization is more effective than multi-expert mixture.
- **Physical descriptors exhibit differential sensitivity**: $H_{DE}$ is sensitive to polarization (73.47) and microwave band (59.18) shifts; ENL is sensitive to complex-valued inputs (75.97); $R_{LR}$ is sensitive to regional (37.49) and platform (19.83) shifts.
- **Layer-wise expert specialization emerges**: Visualization reveals that Experts 3/4 dominate shallow layers (speckle statistics), Experts 1/2/5/6 are active in middle layers (geometric texture), and Experts 1/5 concentrate in deep layers (high-level semantics).

## Highlights & Insights

- Encoding SAR's three physical priors (speckle, geometry, scattering) as differentiable physical descriptors for MoE routing represents an elegant integration of physics-based priors and data-driven learning, yielding more stable and interpretable routing than purely learned alternatives.
- The 22-benchmark suite covering 8 domain-shift combinations (region, polarization, complex value, platform, and band—individually and jointly) establishes the first unified domain generalization evaluation standard for the SAR community.
- CrossEarth-SAR-S with only 20M active parameters already surpasses DINOv2 and DINOv3 at 300M parameters, demonstrating the parameter efficiency advantages of physics-guided MoE.

## Limitations & Future Work

- Although only 300M parameters are activated at inference, storing and deploying the 1.3B-parameter model on resource-constrained remote sensing platforms (spaceborne or UAV-based) remains a practical challenge.
- Pseudo-label quality depends on the CrossEarth optical model, with a Mean Agreement of only 75.88%; certain category pairs (e.g., road vs. bare land) exhibit significant confusion.
- Evaluation is limited to semantic segmentation; generalizability to other SAR downstream tasks such as object detection and change detection has not been verified.
- Training requires 16×A100 (80 GB), posing a high resource barrier.

## Related Work & Insights

- **vs. SARATR-X** (90M HiViT): CrossEarth-SAR-L outperforms by an average of +3.0 mIoU on single-domain shift benchmarks; the compact CrossEarth-SAR-S achieves +11.7% on the polarization domain (HH2F).
- **vs. DINOv2/v3** (300M): At equivalent active parameter count, CrossEarth-SAR-L achieves +7.2 / +9.7 mIoU on single-domain shift benchmarks.
- **vs. SatMAE/ScaleMAE/MTP**: Optically pretrained models consistently underperform in the SAR domain; the best among them, MTP, reaches only 44.7 vs. CrossEarth-SAR-L's 62.7.
- **Physics-guided routing** is generalizable to other sensor-specific modalities (infrared, multispectral, hyperspectral). The empirical advantage of sparse MoE over dense scaling in domain-fragmented scenarios offers transferable insights for general-purpose vision research.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Physics-descriptor-guided MoE routing is an innovative and physically motivated design; the first billion-scale SAR VFM.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 22 benchmarks, 16 comparison methods, 5 ablation groups, and visualizations of layer-wise specialization and training curves.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with a complete motivation chain linking physical priors to engineering design.
- **Value**: ⭐⭐⭐⭐ — Outstanding contribution to the remote sensing/SAR community; dataset and benchmarks offer long-term utility. Moderate relevance to the broader computer vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SARMAE: Masked Autoencoder for SAR Representation Learning](sarmae_masked_autoencoder_for_sar_representation_learning.md)
- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](gkd_generalizable_knowledge_distillation_vfm.md)
- [\[CVPR 2026\] CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)
- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[CVPR 2026\] Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](promptdriven_lightweight_foundation_model_for_inst.md)

</div>

<!-- RELATED:END -->
