---
title: >-
  [Paper Note] CoLoGen: Progressive Learning of Concept-Localization Duality for Unified Image Generation
description: >-
  [CVPR 2026][Image Generation][Unified Generation Framework] Introduces CoLoGen, a unified image generation framework based on "Concept-Localization Duality." Through progressive staged training and the Progressive Repres…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Unified Generation Framework"
  - "Concept-Localization Duality"
  - "Progressive Training"
  - "Expert Routing"
  - "FLUX"
date: 2026-05-08
content_hash: 1045a06dcd00602b
---

# CoLoGen: Progressive Learning of Concept-Localization Duality for Unified Image Generation

**Conference**: CVPR 2026  
**arXiv**: [2602.22150](https://arxiv.org/abs/2602.22150)  
**Code**: None  
**Area**: Unified Image Generation / Diffusion Models  
**Keywords**: Unified Generation Framework, Concept-Localization Duality, Progressive Training, Expert Routing, FLUX

## TL;DR

Introduces CoLoGen, a unified image generation framework based on "Concept-Localization Duality." Through progressive staged training and the Progressive Representation Weaving (PRW) dynamic expert routing architecture, it simultaneously matches or exceeds specialized models across three major tasks: instruction editing, controllable generation, and personalized generation.

## Background & Motivation

Unified multimodal image generation (covering mask inpainting, visual grounding, controllable generation, personalized generation, and instruction editing) faces a core dilemma of **representation conflict**:

- **Concept Representation $\mathcal{R}_c$**: Encodings for semantic consistency and object-level understanding; controllable generation (e.g., canny/depth/seg conditions) mainly relies on this capability.
- **Localization Representation $\mathcal{R}_l$**: Encodings for spatial alignment, geometry, and structural consistency; personalized generation requires precise localization of identity features in reference images.

Existing unified frameworks force these two heterogeneous representations to be shared, causing interference between conceptual understanding and spatial precision (jointly optimizing $f_c$ may damage $f_l$). This explains why current general-purpose models often perform well on certain tasks while degrading on others.

## Method

### Overall Architecture

Built upon the FLUX.1 architecture, the core consists of two parts:

- **PRW (Progressive Representation Weaving)**: A dynamic expert routing module embedded into each MMDiT block.
- **Progressive Staged Training**: A 5-step training strategy that progresses from foundational capabilities to complex tasks.

### Key Designs

1. **Progressive Representation Weaving (PRW)**: In each multimodal attention block, an expert pool $\{E_k\}_{k=1}^N$ and a dynamic router $G$ are introduced for the KV projection of the source latent. The router selects the most relevant expert via noisy top-1 softmax:

    $\mathbf{w} = hW_r + \epsilon \odot \text{softplus}(hW_n), \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$
    $(K_{\hat{h}}, V_{\hat{h}}) = \text{KV\_proj}_{\text{base}}(h) + \sum_{k \in \mathcal{S}} \text{softmax}(\mathbf{w})_k E_k(h)$

   Attention occurs in two steps: first, the source latent integrates expert information via self-attention, then the noisy/text latent interacts with it. Design Motivation: Allowing different tasks to automatically activate different experts to avoid representation confusion.

2. **Progressive Staged Training Strategy**: 5 steps from easy to difficult:

    - **Step 0-1 (Inherent Pre-training)**: Mask Inpainting (3M synthetic data) to learn concepts + Visual Grounding (1M data) to learn localization.
    - **Step 2 (Condition Injection)**: Controllable Generation (20M data) adapted for Canny/Depth/HED/Lineart/Seg.
    - **Step 3-4 (Instruction-Image Alignment)**: Customized Generation (200K) + Instruction Editing (1.6M).

   Each step only unlocks a new expert $E_{N-1}$, while historical experts are frozen to retain learned knowledge, similar to lifelong learning.

3. **Veteran Gate Routing Supervision**: To balance the utilization of new and old experts, an auxiliary loss is introduced to constrain the routing density of the new expert:

    $\mathcal{L}_{\text{veteran}} = \alpha \cdot |U_t - \rho|, \quad U_t = \frac{1}{L_n} \sum_{i=1}^{L_n} \mathbb{I}(e_i = N-1)$

   Where $\rho = 0.8$ represents the expectation that the new expert is activated 80% of the time, with 20% reserved for historical experts. Total loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{veteran}}$.

### Loss & Training

- Main Loss: Standard diffusion generation loss $\mathcal{L}_{\text{task}}$ (Flow Matching).
- Auxiliary Loss: Veteran Gate Routing Supervision $\mathcal{L}_{\text{veteran}}$, weight $\alpha = 0.5$.
- PRW experts are implemented using LoRA (rank=128) for parameter efficiency.
- Each stage is trained for 200K-400K iterations, global batch size 128-256.
- Total training data: ~25M samples.

## Key Experimental Results

### Main Results

| Task / Dataset | Metric | CoLoGen | Prev. SOTA | Gain |
|--------------|------|---------|-----------|------|
| Instruction Editing / Emu Edit | DINO ↑ | **0.843** | 0.831 (UniReal) | +0.012 |
| Instruction Editing / MagicBrush | DINO ↑ | **0.932** | 0.879 (Emu Edit) | +0.053 |
| Instruction Editing / MagicBrush | CLIP_out ↑ | **0.301** | 0.308 (UniReal) | -0.007 |
| Controllable Gen / MultiGen-20M | Canny CLIP-S ↑ | **33.31** | 32.15 (ControlNet) | +1.16 |
| Controllable Gen / MultiGen-20M | Depth RMSE ↓ | **31.79** | 33.83 (PixWizard) | -2.04 |
| Personalized Gen / DreamBench | DINO ↑ | **0.714** | 0.702 (UniReal) | +0.012 |
| Personalized Gen / DreamBench | CLIP-T ↑ | 0.315 | **0.326** (UniReal) | -0.011 |

### Ablation Study

| Config | CLIP-T ↑ | CLIP-I ↑ | DINO ↑ | Description |
|------|---------|---------|--------|------|
| Baseline (w/o $\mathcal{R}_l$ & $\mathcal{R}_c$) | 0.260 | 0.889 | 0.901 | No experts on MagicBrush |
| w $\mathcal{R}_l$ only | 0.279 | 0.922 | 0.927 | Localization improves structure preservation |
| w $\mathcal{R}_c$ only | 0.302 | 0.881 | 0.905 | Concept improves instruction following |
| Co-training ($\mathcal{R}_c$ & $\mathcal{R}_l$) | 0.269 | 0.918 | 0.922 | Joint training performs worse than separate |
| CoLoGen (Progressive) | **0.301** | **0.931** | **0.932** | Progressive training resolves conflict |

### Key Findings

- The Co-training strategy yielded DINO and CLIP-I scores lower than the baseline in personalized generation, validating the "representation conflict" hypothesis.
- Progressive training outperformed co-training across all metrics, proving that staged learning effectively mitigates the concept-localization conflict.
- Veteran Gate Routing $\rho = 0.8$ is optimal; an excessively large $\alpha$ restricts flexibility.
- LoRA rank=128 was established as the best setting.

## Highlights & Insights

- **Concept-Localization Duality** provides a profound theoretical insight into the unified image generation dilemma—formalizing "different tasks require different representations" into two competing subspaces.
- The PRW architecture cleverly reuses the MoE concept but restricts it to the KV projection layer and uses only top-1 routing to remain lightweight.
- Progressive training + expert freezing is an effective application of lifelong learning in generative models, effectively alleviating catastrophic forgetting.
- Meticulous data engineering: 3 types of masks for mask inpainting (random/object-shaped/Bessel curve irregular) with a 20:40:40 sampling ratio.

## Limitations & Future Work

1. As the number of tasks and experts increases, the memory footprint of PRW grows continuously, limiting scalability.
2. Currently, only 5 tasks have been validated; generalization to more condition types (e.g., pose, sketch) remains unknown.
3. Some metrics are slightly lower than specialized models like UniReal, indicating a gap in absolute performance for unified models.
4. The parameter count for LoRA rank=128 is not exactly "lightweight," and the performance gap relative to full fine-tuning is not reported.

## Related Work & Insights

- Compared to OmniGen (unified multimodal generation but without explicit representation management), CoLoGen achieves a better balance between editing/customization via PRW.
- Compared to general-purpose editing models like PixWizard and UniReal, CoLoGen's core advantage lies in its simultaneous handling of controllable generation.
- Insight: Unified generation should not strive for "one representation fits all tasks" but should enable the model to learn task-adaptive representation switching through dynamic routing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The perspective of concept-localization duality is novel, and the combined design of PRW + progressive training is systematic.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers editing, controllable, and personalized tasks across 6 benchmarks with deep ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definitions are clear, visualization quality is high, and training strategies are detailed.
- **Value**: ⭐⭐⭐⭐ Provides a theoretically grounded and practical solution for unified image generation; PRW is transferable to other multi-task scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PureCC: Pure Learning for Text-to-Image Concept Customization](purecc_pure_learning_for_text-to-image_concept_customization.md)
- [\[AAAI 2026\] EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding](../../AAAI2026/image_generation/echogen_cycle-consistent_learning_for_unified_layout-image_generation_and_unders.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] Unified Vector Floorplan Generation via Markup Representation](unified_vector_floorplan_generation_via_markup_representation.md)
- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)

</div>

<!-- RELATED:END -->
