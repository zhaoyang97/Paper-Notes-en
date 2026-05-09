---
title: >-
  [Paper Note] X2Edit: Revisiting Arbitrary-Instruction Image Editing through Self-Constructed Data and Task-Aware Representation Learning
description: >-
  [AAAI2026][Image Generation][image editing] A 3.7M high-quality editing dataset covering 14 task categories is constructed, and a lightweight (0.9B parameter) plug-and-play editing module based on Task-Aware MoE-LoRA and Contrastive Learning is proposed, achieving performance comparable to 12B fully fine-tuned models.
tags:
  - AAAI2026
  - Image Generation
  - image editing
  - MoE-LoRA
  - contrastive learning
  - dataset construction
  - FLUX
  - task-aware
date: 2026-05-08
content_hash: 5539749b7fb48f86
---

# X2Edit: Revisiting Arbitrary-Instruction Image Editing through Self-Constructed Data and Task-Aware Representation Learning

**Conference**: AAAI2026
**arXiv**: [2508.07607](https://arxiv.org/abs/2508.07607)
**Code**: [GitHub](https://github.com/OPPO-Mente-Lab/X2Edit)
**Area**: Image Generation
**Keywords**: image editing, MoE-LoRA, contrastive learning, dataset construction, FLUX, task-aware

## TL;DR
A 3.7M high-quality editing dataset covering 14 task categories is constructed, and a lightweight (0.9B parameter) plug-and-play editing module based on Task-Aware MoE-LoRA and Contrastive Learning is proposed, achieving performance comparable to 12B fully fine-tuned models.

## Background & Motivation

### State of the Field

**Background**: Open-source image editing models still lag behind closed-source counterparts (e.g., GPT-4o), with high-quality editing datasets remaining a critical bottleneck.

### Limitations of Prior Work

**Limitations of Prior Work**: Existing datasets suffer from three major issues: (1) complex construction pipelines requiring independent design per task category; (2) low editing precision and class imbalance; (3) severe data scarcity for complex tasks (reasoning, camera movement, style transfer).

### Root Cause

**Key Challenge**: On the model side, fully fine-tuned models (Step1X-Edit 12B, Kontext 12B) deliver strong performance but at high cost, while lightweight alternatives (ICEdit 0.2B) reduce cost but sacrifice quality.

### Paper Goals

**Goal**: How to achieve high-quality arbitrary-instruction image editing covering 14 task categories with only a small number of parameters (8% of the full model)?

## Method

### Overall Architecture
Built upon the FLUX.1 DiT architecture, Task-Aware MoE-LoRA modules and contrastive learning regularization are inserted. During training, AlignNet, the task embedding matrix, and MoE-LoRA parameters are updated.

### Key Designs

**1. X2Edit Dataset (3.7M)**
- Four-stage pipeline: source image sampling → VLM-based editing instruction generation → task-specific workflow for edited image generation → comprehensive scoring and filtering
- Qwen2.5-VL-7B generates instructions directly from images (avoiding caption information loss), with self-reflection verification
- Step1X-Edit, GPT-4o, BAGEL, and Kontext are leveraged according to task-specific characteristics for data generation
- Filtering: multi-dimensional evaluation using aesthetic score + LIQE + CLIPIQA + ImgEdit-Judge + Qwen2.5-VL-72B

**2. Task-Aware MoE-LoRA**
A task embedding matrix $t_{emb} \in \mathbb{R}^{N_t \times c}$ is learned and injected into a gating network to guide expert selection:
$$s_i = \text{Softmax}_i(\text{Gate}(\text{Concat}(h^l, t_{emb}^h)))$$
Top-K experts are selected and aggregated with a shared expert:
$$x_{moe}^l = \sum_{i=1}^{N_e} g_i \cdot \text{Expert}_x^i(h^l) + \text{SharedExpert}_x(h^l)$$
Configuration: 12 experts, Top-2 activation, LoRA rank=64, with a total parameter count of only 0.9B.

**3. Task-Aware Contrastive Learning**
Task labels are used to construct positive and negative pairs (same task = positive, cross-task = negative), and an InfoNCE loss is applied on intermediate MMDiT representations:
$$\mathcal{L}_{task} = -\frac{1}{b}\sum_{i=1}^{N}\log\frac{\sum_j \exp(-D_{ij}/\tau) \cdot M_{ij}}{\sum_k \exp(-D_{ik}/\tau)}$$
Final objective: $\mathcal{L} = \mathcal{L}_{task} + \lambda \mathcal{L}_{diff}$, with $\lambda=0.2$ and $\tau=0.5$.

## Key Experimental Results

### Main Results

| Method | Params | GEdit-Bench++ (EN) IJ | G_VIE | ImgEdit-Bench IJ |
|------|------|----------------------|-------|-------------------|
| GPT-4o | - | 9.003 | 7.848 | 8.202 |
| Kontext | 12B | 8.408 | 5.712 | 8.149 |
| Bagel | 7B+7B | 8.326 | 5.722 | 7.925 |
| Step1X-Edit | 12B | 8.017 | 5.108 | 7.653 |
| ICEdit | 0.2B | 7.203 | 4.109 | 7.615 |
| **X2Edit** | **0.9B** | **8.334** | **5.550** | **8.025** |

- DreamBench subject-driven: DINO 0.822 (tied best with Kontext), CLIP-T 0.326
- Plug-and-play: seamlessly compatible with various FLUX.1 community variants and LoRAs (Krea-dev, PixelWave, Ghibli, etc.)
- User study (4 annotators, 1.3k pairs): overall score of 2.432, placing in the upper-middle tier
- Ablation: Task-Aware MoE shows significant improvement over vanilla MoE; applying contrastive loss across all MMDiT layers yields the best results

## Highlights & Insights
- The data construction pipeline is unified and reproducible: VLM-generated instructions + multi-model division of labor + multi-dimensional filtering, covering 14 categories at 3.7M scale
- First application of contrastive learning in arbitrary-instruction image editing, promoting inter-task representation disentanglement
- Exceptional parameter efficiency: 0.9B parameters match 12B fully fine-tuned models while supporting plug-and-play deployment
- "Narrow-yet-numerous" expert strategy (12 experts, rank=64) outperforms configurations with fewer experts and larger ranks

## Limitations & Future Work
- Weak performance on non-English text editing tasks (constrained by the FLUX.1 base model)
- User study involves only 4 annotators, yielding insufficient statistical significance
- Complex reasoning and camera movement tasks rely on GPT-4o-generated data, limiting open-source reproducibility
- Notable performance gap compared to Kontext and Bagel on KontextBench
- Sensitivity analysis for contrastive learning temperature $\tau$ and $\lambda$ is absent

## Related Work & Insights
- vs **ICEdit (0.2B)**: Both adopt a FLUX LoRA approach, but X2Edit introduces task-aware routing and contrastive learning, achieving comprehensive improvements
- vs **Kontext/Bagel (12–14B)**: Fully fine-tuned methods with slightly superior performance but tens of times higher training cost; X2Edit achieves comparable performance with 8% of the parameters
- vs **AnyEdit**: X2Edit substantially outperforms in both data quality and model performance (AnyEdit VIE score of only 2.2 vs. X2Edit's 5.5)
- vs **Step1X-Edit (12B)**: Full DiT fine-tuning approach; X2Edit matches or exceeds it on most metrics

## Related Work & Insights
- The task embedding + MoE gating design is generalizable to other multi-task generation scenarios (video editing, 3D generation)
- Applying contrastive learning in the diffusion hidden space is a promising new direction worth further exploration
- The data construction pipeline of "VLM-generated instructions + multi-model image generation + multi-dimensional filtering" is broadly applicable
- The plug-and-play property makes it particularly suitable for community ecosystems, offering significant commercial value

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First application of task-aware contrastive learning in image editing
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 4 benchmarks + DreamBench + plug-and-play + ablation, though user study is limited
- **Writing Quality**: ⭐⭐⭐½ — Comprehensive content but slightly verbose structure
- **Value**: ⭐⭐⭐⭐ — Both dataset and model are open-sourced, representing a significant community contribution

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](../../ICLR2026/image_generation/visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[ICLR 2026\] EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing](../../ICLR2026/image_generation/editreward_a_human-aligned_reward_model_for_instruction-guided_image_editing.md)
- [\[ICCV 2025\] SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing](../../ICCV2025/image_generation/superedit_rectifying_and_facilitating_supervision_for_instruction-based_image_ed.md)
- [\[ACL 2026\] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](../../ACL2026/image_generation/codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)
- [\[CVPR 2026\] CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing](../../CVPR2026/image_generation/care-edit_condition-aware_routing_of_experts_for_contextual_image_editing.md)

<!-- RELATED:END -->
