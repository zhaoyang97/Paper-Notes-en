---
title: >-
  [Paper Note] Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation
description: >-
  [CVPR 2026][Segmentation][Referring Image Segmentation] This paper proposes the SERA framework, which introduces a two-stage lightweight MoE expert refinement mechanism — SERA-Adapter at the backbone level and SERA-Fusio…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Referring Image Segmentation"
  - "Mixture-of-Experts"
  - "Parameter-Efficient Tuning"
  - "Vision-Language Models"
  - "Expert Routing"
date: 2026-05-08
content_hash: f4f261dcfdc408a2
---

# Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.12538](https://arxiv.org/abs/2603.12538)  
**Code**: None  
**Area**: Segmentation
**Keywords**: Referring Image Segmentation, Mixture-of-Experts, Parameter-Efficient Tuning, Vision-Language Models, Expert Routing

## TL;DR

This paper proposes the SERA framework, which introduces a two-stage lightweight MoE expert refinement mechanism — SERA-Adapter at the backbone level and SERA-Fusion at the fusion level — into a frozen vision-language backbone. Through expression-guided adaptive routing, SERA improves spatial consistency and boundary precision in referring image segmentation while updating fewer than 1% of backbone parameters.

## Background & Motivation

Referring Image Segmentation (RIS) requires generating pixel-level masks from natural language expressions. The core challenge lies in precisely aligning language with visual content while handling spatial relationships, fine-grained attributes, and object boundaries. Existing methods exhibit three critical limitations:

**Uniform refinement strategy**: Most methods apply the same processing pathway to all referring expressions, failing to accommodate the diverse reasoning demands of different expressions (some rely on spatial layout, others on appearance or contextual relations).

**Difficulty adapting frozen backbones**: Freezing pre-trained encoders to reduce computational cost limits the adaptability of visual representations, leading to fragmented masks, boundary leakage, or incorrect target selection.

**Challenges of introducing MoE**: Directly applying MoE routing to RIS risks training instability and interference with pre-trained representations.

The core motivation of SERA is that different referring expressions require different types of reasoning experts. Accordingly, it introduces a conditioned expert routing mechanism that achieves expression-aware feature refinement while preserving the advantages of pre-trained representations.

## Method

### Overall Architecture

SERA builds upon a pre-trained vision-language framework consisting of a DINOv2 visual encoder and a CLIP text encoder. Given an input image $I$ and a referring expression $Q$, the visual encoder extracts a sequence of image tokens and the text encoder produces a global expression embedding. SERA introduces modifications at two complementary stages:

- **SERA-Adapter**: Inserted into selected layers of the backbone Transformer to refine intermediate visual tokens within the backbone.
- **SERA-Fusion**: Applied during the vision-language fusion stage, reshaping spatial tokens into 2D feature maps before MoE-based refinement.

### Key Designs

#### SERA-Adapter: Backbone-Level Expert Refinement

**Function**: Inserts expression-conditioned adapters into selected DINOv2 Transformer blocks, improving spatial consistency and boundary precision through expert-guided refinement and cross-modal attention.

**Mechanism**: Visual tokens are projected onto a 2D spatial grid, enriched with local context via multi-scale convolutional branches (1×1, 3×3, 5×5), and then refined by two complementary experts:

- **Boundary Expert**: Enhances contour-sensitive responses using a learnable depthwise $3 \times 3$ convolution, $\mathbf{B} = \text{ReLU}(\text{BN}(\mathbf{G} + \beta \cdot \text{DWConv}_{3\times3}(\mathbf{G})))$, where $\beta = 0.1$.
- **Spatial Expert**: Enhances local feature consistency using a depthwise $3 \times 3$ convolution with a scaled residual, $\mathbf{S} = \phi(\text{DWConv}_{3\times3}(\mathbf{G})) + \alpha \mathbf{G}$, where $\alpha = 0.3$.

**Adaptive Soft Routing**: Global average pooling over spatial tokens yields a summary vector $\mathbf{z}$, which is projected and normalized via softmax to produce routing weights $[w_s, w_b] = \boldsymbol{\sigma}(\mathcal{R}(\mathbf{z}))$. Expert outputs are then fused as:

$$\mathbf{G}_{\text{corr}} = \mathbf{G}_{\text{rich}} + \alpha w_s \mathbf{E}_s + \beta w_b \mathbf{E}_b$$

where $\alpha = 0.25$ and $\beta = 0.15$ are fixed scaling coefficients. The result is flattened back into a token sequence and passed through cross-modal attention with the text embedding.

**Design Motivation**: Soft routing ensures stable residual refinement within the backbone, avoiding the instability that sparse routing can cause in frozen encoders.

#### SERA-Fusion: Fusion-Level Expert-Guided Aggregation

**Function**: Applies complementary expert refinement to intermediate spatial feature maps during the vision-language fusion stage, enhancing representation quality prior to mask prediction.

**Mechanism**: Four complementary experts are designed to capture distinct visual cues:

1. **Spatial Expert**: Injects explicit positional information, $E_{\text{spa}}(\mathbf{X}) = \mathbf{X} + \alpha \cdot \text{Conv}_{1\times1}(\mathbf{G})$, where $\mathbf{G}$ is a normalized coordinate grid.
2. **Context Expert**: Captures long-range dependencies via multi-head self-attention, flattening spatial dimensions before applying self-attention, FFN, and residual connection.
3. **Boundary Expert**: Extracts horizontal/vertical gradients and magnitude using fixed Sobel operators, $E_{\text{bnd}}(\mathbf{X}) = \mathbf{X} + \phi(\text{Conv}_{1\times1}([\mathbf{X}, \mathbf{G}_{\text{mag}}, \mathbf{G}_x + \mathbf{G}_y]))$.
4. **Shape Expert**: Combines depthwise blurring (low-frequency smoothing) with a Laplacian operator (high-frequency structural cues) to promote global structural consistency.

**Conditional Routing (Top-K Sparse Gating)**:

$$\mathbf{z} = \text{GAP}(\mathbf{X}), \quad \mathbf{r} = \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{z})$$

Gaussian noise is added during training to encourage routing diversity; Top-K selection followed by softmax normalization yields sparse routing weights.

**Design Motivation**: Sparse Top-K routing at the fusion stage encourages expert specialization, contrasting with the soft routing strategy used at the backbone level. The deliberate use of different routing strategies at each stage reflects the design principle that stability is required within the backbone, while specialization is prioritized at the fusion level.

#### Regularization to Prevent Expert Collapse

Three auxiliary losses are introduced during training:

- **Z-loss**: Penalizes the mean squared magnitude of routing logits, $\mathcal{L}_z = \lambda_z \frac{1}{BE} \|\mathbf{r}\|_2^2$.
- **Load Balancing Loss**: Penalizes the coefficient of variation of expert utilization, $\mathcal{L}_{\text{balance}} = \lambda_{\text{bal}} \text{CV}(\mathbf{u})^2$.
- **Token Assignment Regularization**: Stabilizes token-to-expert assignment during training.

### Loss & Training

- Total MoE regularization: $\mathcal{L}_{\text{MoE}} = \mathcal{L}_{\text{logit}} + \mathcal{L}_{\text{balance}} + \mathcal{L}_{\text{token}}$
- **Parameter-efficient strategy**: The backbone is fully frozen; only LayerNorm and bias parameters (fewer than 1% of backbone parameters) are updated, along with the proposed modules and task-specific segmentation layers.
- Optimizer: Adam with an initial learning rate of $1 \times 10^{-4}$, decayed by a factor of 0.1 in later stages.
- Hardware: Single NVIDIA A6000 GPU, batch size 16.

## Key Experimental Results

### Main Results

Evaluation on three standard benchmarks — RefCOCO, RefCOCO+, and G-Ref — using mIoU:

| Method | Type | RefCOCO val | RefCOCO+ val | G-Ref val(g) | Avg. |
|--------|------|-------------|--------------|--------------|------|
| ETRIS | PET | 70.5 | 60.1 | 57.9 | 62.8 |
| DETRIS-B | PET | 76.0 | 68.9 | 65.9 | 70.4 |
| VATEX | Full FT | 78.2 | 70.0 | 69.7 | 72.8 |
| RISCLIP-B | Full FT | 75.7 | 69.2 | — | 70.6 |
| **SERA (Ours)** | **PET** | **76.5** | **70.4** | **66.6** | **71.1** |

Under the parameter-efficient tuning (PET) setting with a frozen backbone, SERA outperforms all PET baselines and matches or approaches several fully fine-tuned methods. The improvement on RefCOCO+ (which excludes absolute spatial terms) is particularly notable, suggesting that appearance-driven and context-driven reasoning benefit most from the proposed expert refinement.

### Ablation Study

**Component Ablation** (mIoU on RefCOCO / RefCOCO+ / G-Ref(g)):

| Configuration | RefCOCO | RefCOCO+ | G-Ref(g) |
|---------------|---------|----------|----------|
| Baseline | 74.90 | 68.70 | 65.10 |
| + SERA-Adapter | 75.35 (+0.45) | 69.42 (+0.72) | 65.74 (+0.64) |
| + SERA-Adapter + SERA-Fusion | **76.50 (+1.60)** | **70.40 (+1.70)** | **66.62 (+1.52)** |

**Top-K Routing Ablation** (RefCOCO val mIoU / oIoU):

| Top-K | val mIoU | val oIoU |
|-------|----------|----------|
| K=1 | 75.46 | 73.32 |
| K=2 | 76.47 (+1.01) | 74.65 (+1.33) |
| K=3 | 76.20 (+0.74) | 74.10 (+0.78) |
| K=4 | 76.50 (+1.04) | 74.74 (+1.42) |

### Key Findings

1. The two modules provide complementary gains: SERA-Adapter primarily improves backbone-level features, while SERA-Fusion further enhances spatial representations at the fusion stage.
2. Performance is lowest at K=1; increasing to K≥2 yields substantial improvements, with K=4 being the most consistently stable.
3. The largest gain is observed on RefCOCO+ (+1.70 mIoU), indicating that appearance- and context-driven expert refinement is most critical when spatial terminology is absent.
4. The model supports zero-shot cross-dataset generalization, suggesting that the learned vision-language representations are transferable.

## Highlights & Insights

- The **two-stage differentiated routing strategy** is a sophisticated design: soft routing within the backbone ensures stability, while sparse routing at the fusion stage promotes specialization.
- The combination of extreme parameter efficiency (updating only bias and LayerNorm parameters, <1% of backbone parameters) with MoE expert refinement represents a novel design space.
- Expert designs carry explicit semantic interpretations (spatial / boundary / context / shape), making the approach more interpretable than purely black-box MoE formulations.
- The regularization strategy (Z-loss + load balancing + token assignment) ensures healthy training of sparse routing.

## Limitations & Future Work

- Improvements on G-Ref are relatively smaller than on RefCOCO+, indicating room for improvement in handling long descriptive expressions.
- Validation is currently limited to the DINOv2 + CLIP framework; transferability to other VLM backbones (e.g., SAM, Grounding DINO) remains unexplored.
- The number and types of experts (four in total) are manually designed; automatic discovery of optimal expert configurations warrants further investigation.
- Generalization to larger-scale or more diverse segmentation tasks has not been validated.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic introduction of MoE expert routing in RIS, with a sophisticated two-stage differentiated routing design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three standard benchmarks, comprehensive ablations, zero-shot generalization, and extensive qualitative analysis; efficiency analysis is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, complete formulations, professional figures, and systematic method exposition.
- **Value**: ⭐⭐⭐⭐ — Offers a new MoE perspective on parameter-efficient adaptation of VLMs, with implications for RIS and dense prediction tasks more broadly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts](../../AAAI2026/segmentation/generalizable_slum_detection_from_satellite_imagery_with_mixture-of-experts.md)
- [\[CVPR 2026\] MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention](mixercseg_an_efficient_mixer_architecture_for_crack_segmentation_via_decoupled_m.md)
- [\[ICLR 2026\] AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation](../../ICLR2026/segmentation/amlris_alignment-aware_masked_learning_for_referring_image_segmentation.md)
- [\[CVPR 2026\] Phrase-Instance Alignment for Generalized Referring Segmentation](phrase-instance_alignment_for_generalized_referring_segmentation.md)
- [\[CVPR 2026\] Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics](reasoning_with_pixel-level_precision_qvlm_architecture_and_squid_dataset_for_qua.md)

</div>

<!-- RELATED:END -->
