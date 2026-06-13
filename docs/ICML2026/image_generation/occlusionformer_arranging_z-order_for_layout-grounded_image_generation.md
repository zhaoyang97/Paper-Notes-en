---
title: >-
  [Paper Note] OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation
description: >-
  [ICML2026][Image Generation][Layout-to-Image] To address the issues of texture entanglement and hierarchical confusion in overlapping regions during layout-to-image generation, the authors constructed SA-Z…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Layout-to-Image"
  - "Z-order Occlusion"
  - "Volume Rendering"
  - "Instance Decoupling"
  - "DiT"
date: 2026-05-08
content_hash: 433c96301d7c985e
---

# OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation

**Conference**: ICML2026  
**arXiv**: [2605.21343](https://arxiv.org/abs/2605.21343)  
**Code**: https://henghuiding.com/OcclusionFormer/ (Project Page)  
**Area**: Image Generation / Layout-to-Image / Diffusion Models  
**Keywords**: Layout-to-Image, Z-order Occlusion, Volume Rendering, Instance Decoupling, DiT

## TL;DR
To address the issues of texture entanglement and hierarchical confusion in overlapping regions during layout-to-image generation, the authors constructed SA-Z, a large-scale dataset with explicit Z-order and amodal annotations, and proposed OcclusionFormer. By combining instance decoupling with volume rendering to explicitly model occlusion priority, and employing a queried alignment loss to reinforce spatial consistency, the model outperforms strong baselines such as Eligen, Creatilayout, and InstanceAssemble on occlusion-aware metrics in the OverLayBench complex subset and the self-constructed SA-Z Eval.

## Background & Motivation

**Background**: Layout-to-image generation injects 2D/3D bounding boxes into diffusion models as spatial conditions. Driven by works like GLIGEN, Eligen, and Creatilayout, spatial controllability for single instances has become well-established, serving as fundamental infrastructure for tasks like complex scene synthesis and visual story generation.

**Limitations of Prior Work**: When multiple bounding boxes overlap, mainstream methods struggle—objects in overlapping areas often exhibit texture entanglement, inverted hierarchies, or are forced to shrink to cover only visible parts. This occurs because these methods treat layouts as 2D planar conditions and lack a concept of "who occludes whom." While users typically draw amodal boxes (full range) and expect a depth order, the models cannot interpret this sequence.

**Key Challenge**: While Z-buffers have long solved occlusion in computer graphics, the attention mechanism of diffusion models inherently "mixes" features indiscriminately on a 2D plane, lacking an explicit Z-axis dimension. LaRender attempted to use training-free volume rendering, but it diverted cross-attention space for occlusion control, resulting in the loss of global prompts, sensitivity to hyperparameters, and deviations in complex scenarios.

**Goal**: (1) Provide a large-scale training set with an open vocabulary, Z-order, and amodal annotations; (2) Design a training-based scheme within the DiT framework to explicitly model Z-order without compromising pre-trained capabilities, ensuring physically consistent hierarchies in overlapping regions.

**Key Insight**: The authors argue that training-free heuristics are insufficient and require a "data-driven + explicit supervision" approach. Specifically, each instance is "decoupled" into an independent layer, then "composed" according to the user-specified occlusion order via volume rendering, with mask supervision pulling the spatial geometry.

**Core Idea**: Image generation is viewed as a volume rendering process along orthogonal camera rays. Each instance independently performs MM-Attention within its box to generate a "layer." Learned densities $\sigma_i$ are then used to calculate transmittance $T_i$ and opacity $\alpha_i$ for weighted composition according to the Z-order. Simultaneously, a queried alignment loss "welds" the feature geometry of each instance to the ground truth mask.

## Method

OcclusionFormer is built upon Flux.1-dev (DiT + Rectified Flow). It inserts a serial "Instance Decoupling—Volume Rendering Composition—Queried Alignment" module after the original MM-Attention block, fine-tuned via LoRA (rank=4) to preserve the backbone's pre-trained capabilities.

### Overall Architecture
The input consists of a set of instance conditions $(M_i, B_i, \mathcal{O}_i, C_i, P)$ (mask, bounding box, occlusion set, instance caption, global prompt). Each DiT block first executes a frozen global MM-Attention to obtain visual features $\mathbf{Z}\in\mathbb{R}^{L\times D}$. Then: (1) A subset of local tokens $\mathbf{Z}_{\Omega_i}$ is extracted per instance $i$ according to its box and processed independently with the instance caption embedding $\mathbf{C}_i'$ via MM-Attention to generate an "independent layer" $\hat{\mathbf{Z}}_i$; (2) All $\hat{\mathbf{Z}}_i$ are composed into $\mathbf{Z}_{out}$ via volume rendering based on the Z-order specified in $\mathcal{O}_i$, with the residual added back to the backbone; (3) A query vector extracts a spatial similarity map from each $\hat{\mathbf{Z}}_i$, and a lightweight CNN predicts foreground probability, supervised by the GT mask from SA-Z using cross-entropy. The training objective is $\mathcal{L}_{total} = \mathcal{L}_{flow} + \lambda \mathcal{L}_{align}$, with $\lambda=0.5$.

### Key Designs

1.  **Instance-Decoupled Local MM-Attention**:
    - **Function**: Transitions from "global 2D planar attention" to independent layers per instance, enabling decouplable objects along the Z-axis.
    - **Mechanism**: For each instance $i$, token indices within its bounding box are selected as $\Omega_i = \{u \mid \text{Coord}(u) \in B_i\}$. The original MM-Attention module is reused to compute updates only between this subset $\mathbf{Z}_{\Omega_i}$ and the instance caption embedding $\mathbf{C}_i'$: $\hat{\mathbf{Z}}_{\Omega_i}, \hat{\mathbf{C}_i} = \text{MM-Attention}(\mathbf{Z}_{\Omega_i}, \mathbf{C}_i')$, with zero-padding outside the box. Original attention parameters are frozen, with LoRA applied only to projection matrices.
    - **Design Motivation**: Eligen and Creatilayout treat layouts as global conditions where all instance and background tokens interact indiscriminately, lacking a "layer" concept to express Z-axis priority. Calculating clean instance features before composition is a prerequisite for explicit Z-order modeling. Using LoRA instead of full fine-tuning preserves Flux's generative capacity.

2.  **Explicit Z-order Modeling via Volume Rendering**:
    - **Function**: Calculates occlusion relationships per pixel based on the user-provided occlusion set $\mathcal{O}_i$ and composes layers back into a feature map in a physically consistent manner.
    - **Mechanism**: Drawing from NeRF, the image plane is treated as the imaging surface of an orthogonal camera. The density $\sigma_i \in \mathbb{R}^D$ of each instance is not fixed but predicted by a time-text embedding module from the diffusion timestep $t$ and instance text pooling vector $y_i$: $\mathbf{e}_{temb}^i = \text{TimeTextEmbed}(t, y_i) \to \sigma_i$. At pixel $\mathbf{p}$, opacity is defined as $\alpha_i(\mathbf{p}) = (1 - \exp(-\sigma_i)) \cdot \mathbb{I}(\mathbf{p} \in B_i)$, and transmittance as $T_i(\mathbf{p}) = \exp(-\sum_{j \in \mathcal{O}_i} \sigma_j \cdot \mathbb{I}(\mathbf{p} \in B_j))$. The composition weight is $w_i = T_i \cdot \alpha_i$. Final composition uses normalized weighted averaging for pixels with occlusion constraints $\mathbf{Z}_{out}(\mathbf{p}) = \sum_i w_i \hat{\mathbf{Z}}_i / (\sum_i w_i + \epsilon)$, falling back to simple averaging for overlapping boxes without occlusion declarations (hybrid strategy).
    - **Design Motivation**: LaRender uses training-free volume rendering with manual, heuristic densities sensitive to hyperparameters. Making $\sigma_i$ a learnable quantity that adapts to the diffusion state allows for varying levels of "solidity" during early low-frequency and later detail stages, handling complex occlusions more robustly. Hybrid aggregation resolves edge cases where boxes overlap but objects do not actually occlude one each other.

3.  **Queried Alignment Loss**:
    - **Function**: Replaces the supervision gap of volume rendering regarding "geometric shape," ensuring instance features focus within GT mask regions rather than bleeding across the box.
    - **Mechanism**: A learnable query $\mathbf{q}_i \in \mathbb{R}^D$ is derived from $\mathbf{e}_{temb}^i$ for each instance. Pixel-wise cosine similarity generates a spatial similarity map $\mathbf{S}_i(\mathbf{p}) = \hat{\mathbf{Z}}_i(\mathbf{p}) \cdot \mathbf{q}_i / ((\|\hat{\mathbf{Z}}_i(\mathbf{p})\| + \epsilon)\|\mathbf{q}_i\|)$, fed into a lightweight CNN $\mathcal{F}_\theta$ to output foreground/background probability maps $\hat{\mathbf{M}}_i$, supervised by the mask $M_i$ from SA-Z via $\mathcal{L}_{align}$.
    - **Design Motivation**: While volume rendering handles depth order, features must possess coherent geometry for effective composition. Without explicit constraints, features often drift within the box, leading to broken contours in overlapping areas. Supervising via a query + CNN head is more stable than applying mask loss directly to attention maps.

### Loss & Training
The total objective is $\mathcal{L}_{total} = \mathcal{L}_{flow} + \lambda \mathcal{L}_{align}$, where $\lambda=0.5$. $\mathcal{L}_{flow}$ is the rectified flow matching loss of Flux: $\mathcal{L}_{flow} = \mathbb{E}_{t,\mathbf{z}_t,\mathbf{c}}[\|v_\theta(\mathbf{z}_t,t,\mathbf{c}) - \mathbf{v}_{target}\|_2^2]$, with $\mathbf{v}_{target} = \mathbf{x}_1 - \mathbf{x}_0$. Backbone: Flux.1-dev, LoRA rank=4, 200K steps, batch size 16, lr=1e-4. The SA-Z dataset was derived from SACap-1M: DescribeAnything generated pixel-level captions, InstaOrder predicted pairwise occlusion sequences, and SAM-3D reconstructed 3D geometry projected back for amodal masks and boxes. It contains 1M high-resolution images and 5.69M instances, making it the first large-scale, open-vocabulary layout generation dataset with amodal and Z-order labels.

## Key Experimental Results

### Main Results
Evaluated on OverLayBench (Simple/Regular/Complex) and the self-constructed SA-Z Eval (1K real images). Metrics include spatial accuracy (mIoU / O-mIoU), semantic consistency (SR$_E$ / SR$_R$ / CLIP-G/L), image quality (FID), and InstaOrder-style occlusion metrics Occ. (F1) and Dep. (WHDR).

| Subset | Metric | OcclusionFormer | Prev. SOTA (InstanceAssemble) | Creatilayout | Eligen | LaRender |
|------|------|------|------|------|------|------|
| OverLay-Simple | mIoU ↑ | **0.7405** | 0.7279 | 0.6998 | 0.6673 | 0.6604 |
| OverLay-Simple | O-mIoU ↑ | **0.5456** | 0.5152 | 0.4725 | 0.4151 | 0.4136 |
| OverLay-Simple | Occ. ↑ | **0.8051** | 0.7852 | 0.7559 | 0.6823 | 0.6294 |
| OverLay-Regular | mIoU ↑ | **0.6487** | 0.6299 | 0.5997 | 0.5680 | 0.5721 |
| OverLay-Regular | O-mIoU ↑ | **0.4161** | 0.3861 | 0.3517 | 0.3075 | 0.3006 |
| OverLay-Complex | mIoU ↑ | **0.6037** | 0.5706 | 0.5584 | 0.5195 | 0.5227 |
| OverLay-Complex | O-mIoU ↑ | **0.3468** | 0.3189 | 0.3006 | 0.2569 | 0.2507 |
| OverLay-Complex | Occ. ↑ | **0.7797** | 0.6987 | 0.7142 | 0.5994 | 0.6026 |
| OverLay-Complex | Dep. ↓ | **0.1602** | 0.1791 | 0.1907 | 0.2378 | 0.2374 |
| SA-Z Eval | mIoU ↑ | **0.4509** | 0.4292 | 0.4216 | 0.3007 | 0.4053 |
| SA-Z Eval | O-mIoU ↑ | **0.2231** | 0.2021 | 0.1904 | 0.1016 | 0.1709 |
| SA-Z Eval | Occ. ↑ | **0.7568** | 0.6947 | 0.6921 | 0.6095 | 0.6833 |
| SA-Z Eval | FID ↓ | **62.79** | 63.65 | 64.66 | 69.91 | 77.98 |

**Highlights**: The relative advantage increases as scene complexity grows (Simple→Complex). O-mIoU and Occ. (measuring overlapping regions) show the most significant gains (+0.08 Occ. and -0.019 Dep. on the Complex subset).

### Ablation Study (OverLay-Complex)

| Configuration | mIoU ↑ | O-mIoU ↑ | Occ. ↑ | Dep. ↓ | Description |
|------|------|------|------|------|------|
| OcclusionFormer (full) | **0.6037** | **0.3468** | **0.7797** | **0.1602** | Full Model |
| w/o Learned Sigma | 0.5911 | 0.3276 | 0.7530 | 0.1694 | Replaced dynamic density with static values; Occ. F1 dropped ~2.7 pts |
| w/o Queried Loss | 0.5922 | 0.3319 | 0.7659 | 0.1666 | Removed alignment loss; O-mIoU dropped ~1.5 pts |
| w Attn. Map Loss | 0.5753 | 0.3207 | 0.7510 | 0.1695 | Switched to mask loss on attention maps; performed worse than w/o |
| w/o Amodal Data | 0.6004 | 0.3411 | 0.7703 | 0.1644 | Used only visible masks without amodal labels; slight performance drop |

### Key Findings
- **Dynamic density is the most impactful design**: Removing learned $\sigma$ leads to a drop of 0.027 in Occ. and an increase of 0.009 in Dep., proving that density adapting to the diffusion step and text more stably characterizes layer weights across diffusion stages.
- **Queried alignment loss outperforms attention-map mask loss**: Direct mask supervision on attention maps conflicts with global MM-Attention semantics, performing worse than the "w/o Queried Loss" setting. This suggests query guidance + an independent CNN head is a more decoupled supervision path.
- **Amodal labels provide consistent small gains**: Compared to visible-only masks, amodal data provides signals for occluded parts, aiding geometric integrity in complex occlusions.
- **Real-world domain gap is evident**: mIoU/O-mIoU for all methods are significantly lower on SA-Z Eval than OverLayBench (the latter being synthetic); the relative advantage of Ours persists in the real domain, indicating improvements do not rely on synthetic distributions.

## Highlights & Insights
- **Integrating Graphics Z-buffer Concepts into a Differentiable DiT**: Using NeRF volume rendering for layer composition brings "explicit Z-order" into diffusion models in a differentiable, end-to-end trainable form. This is cleaner and more physically consistent than LaRender's heuristics or GLIGEN's global conditioning.
- **A Scalable Three-Stage Pipeline**: The "Instance Decoupling → Explicit Z-order Composition → Geometric Alignment" flow is not limited to layout generation and can be applied to any task requiring multi-source token composition following user-defined orders (e.g., video layering, 3D scene editing).
- **Benchmark for Open-Vocabulary Amodal Data**: Utilizing SAM-3D reconstruction to acquire amodal labels circumvented the small-scale constraints of manual annotation (e.g., COCOA), providing a transferable labeling strategy for any dataset with visible masks.

## Limitations & Future Work
- **Dependency on Correct Z-order Input**: The method assumes $\mathcal{O}_i$ is known. Hierarchical ambiguity or errors in user-provided Z-orders (e.g., cyclic occlusions) were not discussed regarding model robustness.
- **Absolute FID in Complex Scenes Remains High**: While FID=62.79 is SOTA on SA-Z Eval, it is significantly higher than 24.6 on OverLay-Simple, indicating substantial room for improvement in overall fidelity for complex real-world scenes.
- **Strong Orthogonal Camera Assumption**: Volume rendering relies on virtual orthogonal cameras and axis-aligned boxes, making its suitability for large perspective distortions or tilted amodal boxes unclear.
- **Training Costs**: High reproduction barriers exist due to reliance on Flux.1-dev and 1M high-resolution samples. Serial local attention per instance may become a bottleneck in scenes with high instance counts (avg. 5.7 in SA-Z).

## Related Work & Insights
- **vs LaRender**: Also utilizes NeRF rendering, but LaRender is training-free/heuristic and loses global prompts by reusing cross-attention space. Ours uses learnable densities, showing that "learning" is far more stable than manual tuning in complex scenes (Occ. +0.18, O-mIoU +0.10).
- **vs Eligen / Creatilayout**: Also based on Flux/DiT but treats layouts as global 2D conditions. Ours adds a Z-axis and restricts attention to box regions, leading O-mIoU by 0.04~0.05. This demonstrates that "restructuring attention subspaces" is a more direct control mechanism than "adding conditions."
- **vs InstanceAssemble**: In the complex subset, Ours leads by mIoU +0.033 and Occ. +0.081, suggesting explicit Z-order modeling provides more leverage than simply improving instance assembly.
- **vs InstaOrder / COCOA**: These provided Z-order/amodal labels for low-res closed-set vocabularies. SA-Z scales these ideas to SA-1B scale with open vocabularies, bringing "explicit occlusion" back into the focus of generative tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining differentiable volume rendering + instance decoupling for explicit Z-order control is a clear new setting for layout-to-image, though individual modules (NeRF, LoRA, query-mask) are recombinations of existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three difficulty levels, real-world evaluation, 6 baselines, 9 metrics, and 4 ablation dimensions; lacks inference latency and robustness analysis against incorrect Z-orders.
- Writing Quality: ⭐⭐⭐⭐ Motivations, design, and formulas are clearly explained; figures are well-coordinated and symbols are consistent.
- Value: ⭐⭐⭐⭐ The SA-Z dataset and explicit Z-order framework are reusable infrastructure for controllable diffusion; this is the first systematic solution for the "amodal box + expected occlusion" pain point.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] InstanceAssemble: Layout-Aware Image Generation via Instance Assembling Attention](../../NeurIPS2025/image_generation/instanceassemble_layoutaware_image_generation_via_instance_a.md)
- [\[ICML 2026\] Envisioning Beyond the Few: Disentangled Semantics and Primitives for Few-Shot Atypical Layout-to-Image Generation](envisioning_beyond_the_few_disentangled_semantics_and_primitives_for_few-shot_at.md)
- [\[NeurIPS 2025\] OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](../../NeurIPS2025/image_generation/overlaybench_a_benchmark_for_layout-to-image_generation_with_dense_overlaps.md)
- [\[AAAI 2026\] EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding](../../AAAI2026/image_generation/echogen_cycle-consistent_learning_for_unified_layout-image_generation_and_unders.md)
- [\[CVPR 2026\] PhysGen: Physically Grounded 3D Shape Generation for Industrial Design](../../CVPR2026/image_generation/physgen_physically_grounded_3d_shape_generation_for_industrial_design.md)

</div>

<!-- RELATED:END -->
