---
title: >-
  [Paper Note] DEIG: Detail-Enhanced Instance Generation with Fine-Grained Semantic Control
description: >-
  [AAAI 2026][Multimodal VLM][Multi-Instance Generation] This paper proposes DEIG, a framework for fine-grained multi-instance image generation. It distills high-dimensional embeddings from a frozen LLM encoder into compact instance-aware representations via an Instance Detail Extractor (IDE), and employs instance masked attention in a Detail Fusion Module (DFM) to prevent attribute leakage. DEIG substantially outperforms existing methods on generation tasks with complex multi-attribute descriptions (color + material + texture).
tags:
  - AAAI 2026
  - Multimodal VLM
  - Multi-Instance Generation
  - Fine-Grained Semantic Control
  - Diffusion Models
  - Attribute Binding
  - Masked Attention
date: 2026-05-08
content_hash: 376987692234ca9a
---

# DEIG: Detail-Enhanced Instance Generation with Fine-Grained Semantic Control

**Conference**: AAAI 2026
**arXiv**: [2602.18282](https://arxiv.org/abs/2602.18282)
**Code**: [dushy5/DEIG](https://github.com/dushy5/DEIG)
**Area**: Multimodal VLM
**Keywords**: Multi-Instance Generation, Fine-Grained Semantic Control, Diffusion Models, Attribute Binding, Masked Attention

## TL;DR

This paper proposes DEIG, a framework for fine-grained multi-instance image generation. It distills high-dimensional embeddings from a frozen LLM encoder into compact instance-aware representations via an Instance Detail Extractor (IDE), and employs instance masked attention in a Detail Fusion Module (DFM) to prevent attribute leakage. DEIG substantially outperforms existing methods on generation tasks with complex multi-attribute descriptions (color + material + texture).

## Background & Motivation

Multi-Instance Generation (MIG) aims to generate images containing multiple semantically distinct instances according to user-specified spatial locations and descriptions. Existing methods (GLIGEN, MIGC, InstanceDiffusion, etc.) perform well on simple prompts but struggle significantly with complex multi-attribute descriptions (e.g., "a red-and-blue striped silk dress").

**Two core problems**:

**Insufficient semantic understanding**: Existing methods focus primarily on preventing semantic leakage (attributes bleeding from one instance to another) while neglecting deep semantic understanding of fine-grained attributes.

**Coarse-grained training data**: Instance descriptions in commonly used datasets consist of templated, coarse-grained annotations (e.g., "a red person"), which prevent models from learning rich semantic-visual mappings.

## Method

### Overall Architecture

DEIG is built on a UNet-based diffusion model and consists of three core components:

1. **Frozen LLM text encoder** (Flan-T5-XL): Extracts high-dimensional instance description embeddings.
2. **Instance Detail Extractor (IDE)**: Distills high-dimensional embeddings into compact instance-aware representations.
3. **Detail Fusion Module (DFM)**: Injects instance representations into the UNet via masked attention.

User inputs: global description $\mathcal{P}$, per-instance bounding boxes $b_i$, and fine-grained text descriptions $p_i$.

### Key Designs

**1. Instance Detail Extractor (IDE)**

Conventional multimodal encoders (e.g., CLIP text encoder) have limited capacity for understanding long texts and complex attribute descriptions. DEIG adopts a frozen Flan-T5-XL as the text encoder, but its output embedding $\mathbf{E}_\tau \in \mathbb{R}^{B \times N \times S_\tau \times C}$ is of excessively high dimensionality (large $S_\tau$), making direct use impractical.

IDE introduces learnable queries $\mathbf{Q} \in \mathbb{R}^{B \times N \times S \times C}$ (where $S \ll S_\tau$), with $S$ denoted the "aggregated semantic dimension," serving as an information compression bottleneck. Each IDE layer refines the queries through the following steps:

- **TimeMLP** timestep conditioning → **AdaLN** adaptive normalization
- **Self-attention**: captures intra-instance dependencies
- **Cross-attention**: aligns with high-dimensional features from the frozen encoder

$$\mathbf{H_{ca}^i} = \text{CrossAttn}(\text{AdaLN}(\mathbf{H_{sa}^i}, \mathbf{T_{emb}}), [\mathbf{H_{sa}^i}, \mathbf{E_\tau}])$$

After stacking multiple layers, the module outputs compact **aggregated semantic embeddings**. Visualizations show that different channels across semantic dimensions $S$ attend to distinct fine-grained attributes.

**2. Detail Fusion Module (DFM)**

Contains two sub-components:

**Grounding Embeddings Broadcast**: Spatial coordinates (bounding boxes) are Fourier-encoded and broadcast across all $S$ semantic dimensions, then fused with the aggregated semantic embeddings:

$$\mathbf{G}_{\text{ase},i} = \text{MLP}([m \cdot \mathbf{f_i} + (1-m) \cdot \mathbf{e}_i, \mathbf{E}_{\text{ase},i}])$$

**Instance-based Masked Attention**: A gated self-attention module is inserted between the self-attention and cross-attention layers of the UNet. A binary mask $\mathbf{M}$ controls attention interactions:

- **(a) Visual-Visual**: No masking between visual embeddings (preserving image fidelity).
- **(b) Instance-Visual**: Each instance attends only to visual regions within the same instance (bidirectional); cross-instance interactions are set to $-\infty$.
- **(c) Instance-Instance**: Only intra-group instance attention is permitted; inter-group interactions are set to $-\infty$.

$$\hat{\mathbf{A}} = \text{Softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d}} + \mathbf{M}\right)\mathbf{V}$$

Visual embeddings are updated via a gated residual connection: $\mathbf{V}_{\text{visual}} = \mathbf{V}_{\text{visual}} + \eta \cdot \tanh\gamma \cdot \mathcal{ES}(\hat{\mathbf{A}})$, where $\eta$ and $\gamma$ are learnable scalars.

**3. Detail-Enriched Instance Captions**

Starting from MS-COCO, Qwen2.5-VL is used to generate detailed descriptions (averaging 20–30 words) for cropped instance images. A two-stage quality control pipeline is applied: (1) CLIP score filtering to remove image-caption pairs below a threshold; (2) manual verification on 500 randomly sampled pairs.

### Loss & Training

- Standard diffusion model denoising loss.
- The UNet's self-attention and cross-attention layers are frozen; only the IDE and DFM insertion modules are trained.
- The text encoder (Flan-T5-XL) is frozen throughout.
- Plug-and-play design, compatible with community diffusion model backbones.

## Key Experimental Results

### Main Results

**Table 1: Quantitative Results on DEIG-Bench** (evaluated by Qwen2.5-VL)

| Method | MAA_human↑ | MAA_obj↑ | mIoU↑ |
|---|---|---|---|
| GLIGEN | 0.10 | 0.10 | 0.71 |
| MIGC | 0.22 | 0.36 | 0.72 |
| InstanceDiffusion | 0.25 | 0.33 | 0.75 |
| ROICtrl | 0.31 | 0.33 | 0.71 |
| **DEIG** | **0.75** | **0.44** | **0.79** |

Human instance MAA improves from 0.31 to 0.75 (+142%); object instance MAA improves from 0.36 to 0.44 (+22%).

**Table 2: Quantitative Results on MIG-Bench**

| Method | Instance Success Rate AVG↑ | mIoU AVG↑ |
|---|---|---|
| MIGC | 65.84 | 56.44 |
| ROICtrl | 63.25 | 55.27 |
| **DEIG** | **72.25** | **62.64** |

**Table 3: Quantitative Results on InstDiff-Bench**

| Method | Acc_c.↑ | CLIP_c.↑ | Acc_t.↑ | CLIP_t.↑ |
|---|---|---|---|---|
| ROICtrl | 56.9 | 0.255 | 23.7 | 0.223 |
| **DEIG** | **58.8** | **0.258** | **26.1** | **0.228** |

### Ablation Study

**Table 4: Component Ablation (DEIG-Bench, Qwen2.5-VL)**

| IDE | DFM | Cap. | mIoU↑ | MAA_human↑ | MAA_obj↑ |
|---|---|---|---|---|---|
| ✗ | ✓ | ✓ | 0.73 | 0.51 | 0.35 |
| ✓ | ✗ | ✓ | 0.75 | 0.70 | 0.41 |
| ✓ | ✓ | ✗ | 0.70 | 0.31 | 0.29 |
| ✓ | ✓ | ✓ | **0.79** | **0.75** | **0.44** |

**Detail captions** (Cap.) have the largest impact — removing them causes MAA_human to drop sharply from 0.75 to 0.31, demonstrating that high-quality fine-grained annotations are critical to performance.

**Effect of aggregated semantic dimension $S$**: MAA improves consistently as $S$ increases from 4 to 16, plateaus between 16 and 32, and shows slight overfitting beyond 32. GPU memory scales linearly with $S$; $S = 16 \sim 32$ offers the best trade-off.

### Key Findings

- Color attributes are easier to generate than material/texture (directly related to RGB space); material and texture require deeper semantic understanding.
- Human instances are more sensitive to fine-grained control than object instances (human MAA drops more sharply in ablations).
- DEIG retains fine-grained control capability after plug-and-play adaptation to community diffusion models.

## Highlights & Insights

1. **Fine-grained annotation is an underestimated bottleneck**: Ablation results indicate that data quality is more critical than model architecture — coarse-grained annotations are the primary limiting factor for current methods.
2. **Query distillation compresses high-dimensional embeddings**: IDE's learnable queries ($S \ll S_\tau$) elegantly address the excessive length of LLM encoder outputs, with each dimension interpretably corresponding to a distinct attribute.
3. **Complete design of instance masked attention**: The three types of attention interactions (V-V / I-V / I-I) are handled separately; the design decision to leave visual-visual interactions unmasked for preserving fidelity is experimentally validated.
4. **Evaluation innovation**: DEIG-Bench fills the gap in multi-attribute evaluation for human instances, and the introduced MAA metric better reflects practical requirements.

## Limitations & Future Work

1. Generation improvements for material and texture attributes remain limited; modeling such abstract semantics is still an open problem.
2. Spatial alignment (AP) on InstDiff-Bench is slightly lower, possibly because instance masked attention in dense regions overly restricts interactions.
3. The UNet-based SD1.5 architecture has not yet been adapted to DiT-based architectures (e.g., FLUX, SD3), which may constrain the upper bound of generation quality.
4. Caption generation relies on Qwen2.5-VL, and hallucinations from the VLM may introduce noise into the annotation pipeline.

## Related Work & Insights

- **GLIGEN / MIGC / InstanceDiffusion / ROICtrl**: Direct baselines; DEIG comprehensively outperforms them in fine-grained scenarios.
- **ELLA**: A pioneer in using LLM encoders for global text alignment; DEIG extends this paradigm to the instance level.
- **Insights**: The query distillation mechanism in IDE can be applied to other generation tasks requiring local semantic extraction from long texts; the high-quality fine-grained annotation construction pipeline is worth adopting by other datasets.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combined IDE + DFM design targets a previously neglected fine-grained semantic problem.
- Technical Depth: ⭐⭐⭐⭐ — A complete technical stack covering query distillation, three-type masked attention, and annotation pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three benchmarks plus the self-constructed DEIG-Bench; ablations cover both component and hyperparameter analysis.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear; visualizations (attention maps, semantic dimensions) are convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)
- [\[AAAI 2026\] Heterogeneous Uncertainty-Guided Composed Image Retrieval with Fine-Grained Probabilistic Learning](heterogeneous_uncertainty-guided_composed_image_retrieval_with_fine-grained_prob.md)
- [\[CVPR 2026\] Zina: Multimodal Fine-grained Hallucination Detection and Editing](../../CVPR2026/multimodal_vlm/zina_multimodal_fine-grained_hallucination_detection_and_editing.md)
- [\[CVPR 2026\] ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](../../CVPR2026/multimodal_vlm/reasonmap_towards_finegrained_visual_reasoning_fro.md)
- [\[CVPR 2026\] MA-Bench: Towards Fine-grained Micro-Action Understanding](../../CVPR2026/multimodal_vlm/ma-bench_towards_fine-grained_micro-action_understanding.md)

</div>

<!-- RELATED:END -->
