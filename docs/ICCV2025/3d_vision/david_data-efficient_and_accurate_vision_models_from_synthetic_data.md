---
title: >-
  [Paper Note] DAViD: Data-efficient and Accurate Vision Models from Synthetic Data
description: >-
  [ICCV 2025][3D Vision][synthetic data] This work demonstrates that high-fidelity **procedural synthetic data** suffices to train human-centric dense prediction models that match the accuracy of foundation models such as…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "synthetic data"
  - "depth estimation"
  - "surface normal estimation"
  - "foreground segmentation"
  - "DPT"
  - "high-fidelity annotation"
date: 2026-05-08
content_hash: 871217221c03d575
---

# DAViD: Data-efficient and Accurate Vision Models from Synthetic Data

**Conference**: ICCV 2025
**arXiv**: 2507.15365  
**Code**: [Project Page](https://aka.ms/DAViD)  
**Area**: 3D Vision
**Keywords**: synthetic data, depth estimation, surface normal estimation, foreground segmentation, DPT, high-fidelity annotation
**Authors**: Fatemeh Saleh, Sadegh Aliakbarian, Charlie Hewitt et al. (Microsoft, Cambridge)

## TL;DR

This work demonstrates that high-fidelity **procedural synthetic data** suffices to train human-centric dense prediction models that match the accuracy of foundation models such as Sapiens-2B, requiring only **300K synthetic images**, **0.3B parameters**, and less than 1/16 the training cost of comparable approaches, while achieving state-of-the-art or near-SOTA performance on depth estimation, surface normal estimation, and soft foreground segmentation.

## Background & Motivation

Obtaining annotations for human-centric dense prediction tasks—depth estimation, normal estimation, and foreground segmentation—is extremely challenging:

**Manual annotation is infeasible**: Per-pixel depth/normal annotation is practically impossible for human annotators.

**Laboratory capture is limited**: Annotations acquired via complex camera arrays or dedicated sensors rely on photometric measurements or noisy sensors, yielding limited fidelity.

**Laboratory diversity is insufficient**: Constrained shooting environments make it difficult to cover truly in-the-wild scenarios.

**Existing methods incur high computational costs**: The largest Sapiens variant requires 1,024 A100 GPUs for 18 days of pretraining; DepthPro employs multi-stage mixed training.

A key observation is that existing scan-based synthetic datasets (e.g., THuman) suffer from limited mesh quality, particularly in fine-detail regions such as hair, eyes, and fingers. **Procedural synthetic data**, by contrast, simultaneously provides high fidelity and perfect annotations.

The central claim of this paper is: **data quality >> data quantity + model size**—training a simple model on high-quality synthetic data can match large-scale foundation models.

## Method

### SynthHuman Dataset

Built upon the procedural data generation pipeline of Hewitt et al., the dataset comprises:
- **300K images** at resolution 384×512
- Equal proportions of face, upper-body, and full-body scenes
- Per-image ground-truth depth, surface normals, and soft foreground masks
- Diversity across pose, environment, lighting, and appearance
- Rendering time: 300 M60 GPU machines for 72 hours (approximately equivalent to 2 weeks on 4 A100 GPUs)

Compared to scan-based synthetic datasets (THuman, RenderPeople), SynthHuman exhibits significantly higher annotation fidelity in fine details such as individual hair strands, glasses, and clothing wrinkles, and is free of scan artifacts.

### Model Architecture

A unified architecture addresses all three tasks, based on an improved DPT (Dense Prediction Transformer):

**Encoder**: ViT backbone with a $\text{Read}_{proj}$ readout operation:
$$e^l = \text{mlp}(\text{cat}(\texttt{CLS}^l, t_i^l))$$

**Resizer**: A lightweight fully-convolutional image encoder that extracts features at the original resolution, avoiding the quadratic complexity of ViT under high-resolution inputs. The ViT encoder always receives a fixed 384×384 input.

**Decoder**: Fuses three inputs—the previous decoder block output $d$, encoder features $e$, and resizer features $r$:
$$d_{\text{int}}^l = \text{RConv}(d^{l-1} + \text{Interp}(\text{RConv}(e^l)))$$
$$d^l = \text{Conv}([r^l, \text{Interp}(d_{\text{int}}^l)])$$

**Convolutional Head**: Output channels vary by task (depth: 1, segmentation: 1, normals: 3).

The Resizer design keeps encoder computation constant at inference, with high-resolution information handled by lightweight convolutions—a substantial improvement over increasing the number of ViT tokens.

### Loss & Training

**Soft foreground segmentation**: $\mathcal{L}_\alpha = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{L1} + \mathcal{L}_{\text{dice}} + \omega_{\text{lap}} \mathcal{L}_{\text{lap}}$

**Surface normal estimation**: $\mathcal{L}_\eta = 1 - \eta \cdot \hat{\eta}$ (cosine similarity, computed only within foreground regions)

**Depth estimation**: $\mathcal{L}_d = \mathcal{L}_{\text{MSE}}(s\hat{d}+t, d) + \omega_{\text{grad}} \mathcal{L}_{\text{grad}}(s\hat{d}+t, d)$
(shift-and-scale-invariant loss + gradient supervision, foreground only)

## Key Experimental Results

### Main Results: Depth Estimation

| Method | GFLOPs | Params | Goliath-Face RMSE↓ | Goliath-Full RMSE↓ | Hi4D RMSE↓ | Avg. AbsRel↓ |
|--------|--------|--------|-------------------|-------------------|-----------|-------------|
| MiDaS-DPT_L | - | 0.34B | 0.224 | 0.973 | 0.148 | 0.027 |
| DepthAnythingV2-L | 1827 | 0.34B | 0.229 | 1.039 | 0.130 | 0.025 |
| Sapiens-0.3B | 1242 | 0.34B | 0.179 | 0.690 | 0.116 | 0.021 |
| Sapiens-2B | 8709 | 2.16B | 0.158 | 0.266 | 0.095 | 0.015 |
| DepthPro | 4370 | 0.50B | 0.295 | 0.723 | 0.084 | 0.016 |
| **Ours-Base** | **344** | **0.12B** | **0.142** | **0.376** | **0.085** | **0.014** |
| **Ours-Large** | **663** | **0.34B** | **0.140** | **0.334** | **0.072** | **0.012** |

- Ours-Large achieves accuracy comparable to Sapiens-2B (8,709 GFLOPs) at only **663 GFLOPs**—**1/13** of the computational cost.
- The 0.12B Base model already surpasses DepthAnythingV2-L and Sapiens-0.3B, both at 0.34B.
- Inference speed: approximately **48 FPS** on A100.

### Surface Normal Estimation

| Method | Goliath-Face Mean↓ | Goliath-Full Mean↓ | Hi4D Mean↓ |
|--------|-------------------|-------------------|-----------|
| Sapiens-0.3B | 18.86° | 15.72° | 15.04° |
| Sapiens-2B | 16.04° | 11.49° | 12.14° |
| **Ours-Large** | **17.15°** | **14.60°** | **15.37°** |

- The 0.34B model outperforms Sapiens-0.3B of equivalent size.
- The authors note that ground-truth annotations in Goliath/Hi4D are themselves coarse (missing fine details inside the oral cavity, clothing wrinkles, etc.), whereas model predictions capture more detail.

### Soft Foreground Segmentation

| Method | PhotoMatte85 SAD↓ | PhotoMatte85 MSE↓ | PPM-100 SAD↓ |
|--------|------------------|------------------|-------------|
| MODNet | 13.94 | 0.003 | 104.35 |
| P3M-Net | 20.05 | 0.007 | 142.74 |
| **Ours** | **5.85** | **0.0009** | **78.17** |

### Ablation Study

| Ablation Dimension | Variable | Goliath RMSE↓ | Hi4D RMSE↓ |
|--------------------|----------|--------------|-----------|
| **Data Source** | THuman2.0 | 0.495 | 0.137 |
| | RenderPeople | 0.278 | 0.076 |
| | **SynthHuman** | **0.253** | **0.072** |
| **Data Volume** | 60K | 0.324 | 0.101 |
| | 150K | 0.305 | 0.085 |
| | **300K** | **0.278** | **0.085** |
| **Model Size** | ViT-Small | 0.310 | 0.089 |
| | ViT-Base | 0.278 | 0.085 |
| | **ViT-Large** | **0.253** | **0.072** |

Key findings:
- **Data quality has a decisive impact**: SynthHuman reduces Goliath RMSE from 0.495 to 0.253 compared to THuman2.0—a **48% reduction**.
- Increasing data volume from 60K to 300K yields consistent gains.
- Multi-task training even outperforms single-task training on certain metrics (e.g., PPM-100 segmentation SAD: 66.08 vs. 78.17).

## Highlights & Insights

1. **Empirical validation of data-centrism**: Under the same model architecture, high-fidelity synthetic data compensates for smaller dataset size and model capacity—a 0.12B model surpasses general-purpose foundation models at 0.34B.
2. **Unified architecture for three tasks**: Only the output channel count and loss function differ; no task-specific architectural design is required.
3. **Efficiency of the Resizer module**: Fixing the ViT input resolution and handling the original resolution with lightweight convolutions avoids the quadratic cost of increasing ViT token count.
4. **Implicit advantages of synthetic data**: Data provenance, licensing, and user consent are strongly guaranteed; programmatic control of data diversity also facilitates fairness considerations.
5. **Rethinking ground-truth quality**: The paper finds that annotations in real datasets such as Goliath/Hi4D are coarser than model predictions, suggesting a potential ceiling effect in current evaluations.

## Limitations & Future Work

1. **Restricted to human-centric scenarios**: Both the model and dataset are specifically designed for human body tasks and do not generalize to generic vision.
2. **Dependence on a high-quality synthetic pipeline**: The procedural data generation pipeline itself requires a large volume of artist-created assets (accessories, clothing, environments).
3. **Ceiling effect in evaluation**: When prediction quality exceeds ground-truth quality, quantitative metrics may underestimate actual performance.
4. **Incomplete comparison with recent methods**: For instance, the distillation strategy of Depth Anything V2 is not thoroughly discussed.

## Related Work & Insights

- **Comparison with Sapiens**: Sapiens employs self-supervised pretraining on 300M real images followed by fine-tuning on 500K synthetic images; this work trains directly on 300K synthetic images, simplifying the pipeline while achieving comparable performance.
- **Difference from DepthPro**: DepthPro uses multi-stage mixed training (real + synthetic); this work demonstrates that pure synthetic data alone suffices.
- **Implications for synthetic data research**: Procedural data generation is more promising than scan-based synthesis, simultaneously excelling in fidelity, diversity, and annotation quality.
- **The David vs. Goliath metaphor**: Small dataset + small model vs. large foundation models—the paper's name itself conveys its central message.

## Rating

- **Novelty**: ⭐⭐⭐ — Core contributions are at the data and engineering level; the model architecture is an incremental improvement over DPT, with moderate methodological innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across three tasks; detailed ablations covering data source, data volume, model size, and multi-task training; clear FLOPs comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear argumentation; ground-truth comparisons in Figures 2 and 4 are visually compelling.
- **Value**: ⭐⭐⭐⭐ — Provides direct empirical support for the data-centric AI paradigm; open-sourced datasets and models enhance reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Bootstrap3D: Improving Multi-view Diffusion Model with Synthetic Data](bootstrap3d_improving_multi-view_diffusion_model_with_synthetic_data.md)
- [\[ICCV 2025\] Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation](seeing_and_seeing_through_the_glass_real_and_synthetic_data_for_multi-layer_dept.md)
- [\[ICCV 2025\] Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting](towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)
- [\[ICCV 2025\] ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads](vit-split_unleashing_the_power_of_vision_foundation_models_via_efficient_splitti.md)
- [\[ICCV 2025\] Adversarial Exploitation of Data Diversity Improves Visual Localization](adversarial_exploitation_of_data_diversity_improves_visual_localization.md)

</div>

<!-- RELATED:END -->
