---
title: >-
  [Paper Note] Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning
description: >-
  [ICML 2025][Multimodal VLM][Continual Learning] This paper proposes the D-MoLE method, which automatically evolves the MLLM architecture under parameter budget constraints to continuously adapt to new tasks, achieving an average improvement of 15% over the best baseline through a dynamic layer-wise LoRA expert allocator and a gradient-based inter-modal continual curriculum strategy.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Continual Learning"
  - "Mixture of LoRA Experts"
  - "Curriculum Learning"
  - "Multimodal Instruction Tuning"
  - "Dynamic Architecture"
date: 2026-05-08
content_hash: 1f8983cc4f17f310
---

# Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning

**Conference**: ICML 2025  
**arXiv**: [2506.11672](https://arxiv.org/abs/2506.11672)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Continual Learning, Mixture of LoRA Experts, Curriculum Learning, Multimodal Instruction Tuning, Dynamic Architecture

## TL;DR
This paper proposes the D-MoLE method, which automatically evolves the MLLM architecture under parameter budget constraints to continuously adapt to new tasks, achieving an average improvement of 15% over the best baseline through a dynamic layer-wise LoRA expert allocator and a gradient-based inter-modal continual curriculum strategy.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs, such as LLaVA, InstructBLIP, etc.) rely on instruction tuning to adapt to various downstream tasks. However, in real-world scenarios, data and tasks arrive sequentially (continual), requiring the model to continuously learn new tasks while retaining past knowledge.

**Limitations of Prior Work**: 
   - Existing continual learning methods mostly adopt a **fixed architecture**, which prevents flexible expansion of model capacity when facing an increasing number of tasks.
   - **Catastrophic forgetting**: Performance on past tasks degrades severely when learning new tasks.
   - Experience replay-based methods require storing a large amount of past data, which is costly in terms of both privacy and storage.
   - Regularization-based methods (such as EWC) show diminishing returns as the number of tasks increases.

**Key Challenge**: 
   - **Task Architecture Conflict**: Different tasks have different adaptation requirements for different layers of the model, which a fixed architecture cannot satisfy.
   - **Modality Imbalance**: Different tasks depend on visual/linguistic modalities to varying degrees, and a unified update strategy leads to insufficient updates in certain modalities.

**Goal**: Enable the architecture of the MLLM to automatically evolve to adapt to new tasks under a fixed parameter budget while retaining past knowledge—marking the **first exploration** of continual learning in MLLMs from an architectural perspective.

**Key Insight**: Treat LoRA as dynamically allocatable "experts", automatically allocating these experts across layers, and balancing the learning difficulty of each modality through a curriculum learning strategy.

**Core Idea**: Allow LoRA experts to "flow" and be allocated across layers to adapt to different task structures, while adjusting the learning pace based on modality difficulty.

## Method

### Overall Architecture
The pipeline of D-MoLE: Given an MLLM (such as LLaVA) and a sequence of sequentially arriving multimodal instruction tuning tasks $T_1, T_2, \ldots, T_n$, for each task, the model:
1. Determines how many LoRA experts to allocate to each layer using a **dynamic layer-wise expert allocator**.
2. Routes input instructions to the corresponding experts using a **layer-wise router**.
3. Dynamically adjusts the learning rates of the visual encoder, projection layer, and different parts of the LLM using a **gradient-based inter-modal continual curriculum**.

### Key Designs

1. **Dynamic Layer-wise Expert Allocator**:

    - Under the constraint of a fixed total parameter budget $B$, automatically determines how many LoRA experts each layer should have.
    - Introduces learnable allocation parameters $\alpha_l$ representing the proportion of experts in the $l$-th layer.
    - Makes the discrete allocation decisions differentiable using Gumbel-Softmax:
    $$n_l = \text{GumbelSoftmax}(\alpha_l) \quad \text{s.t.} \quad \sum_l n_l \cdot r_l \leq B$$
      where $r_l$ is the parameter size of a single expert in the $l$-th layer.
    - **Design Motivation**: Different tasks indeed require different layer-wise expert configurations. For example, visually heavy tasks may need more experts allocated in shallow layers (feature extraction layers), while linguistically heavy reasoning tasks require more experts in deep layers. Fixed allocation (e.g., one LoRA per layer) leads to resource waste and task conflicts.

2. **Layer-wise Router**:

    - Multiple LoRA experts in each layer allocate inputs using a lightweight router.
    - The router determines which experts to activate based on the hidden states of current input tokens:
    $$\mathbf{g}_l = \text{TopK}(\text{Softmax}(\mathbf{W}_r \cdot \mathbf{h}_l))$$
    - The outputs of the experts are weighted and summed according to routing weights.
    - Involves **knowledge sharing**: Experts of past tasks are not deleted but have their parameters frozen, and new tasks can reuse the knowledge of old experts through the router.
    - **Design Motivation**: The routing mechanism allows multiple experts in the same layer to specialize in different sub-functions, typical of MoE; layer-wise routing also allows natural transfer of knowledge across tasks.

3. **Gradient-based Inter-modal Continual Curriculum**:

    - For each new task, evaluate the "learning difficulty" of the visual and language modalities.
    - Quantify difficulty by monitoring the gradient norms of each module:
    $$d_v = \|\nabla_{\theta_v} \mathcal{L}\|, \quad d_l = \|\nabla_{\theta_l} \mathcal{L}\|$$
    - Dynamically adjust the learning rate ratios of each module based on difficulty—modalities with higher difficulty receive larger update weights.
    - Gradually reduce the adjustment magnitude as training progresses, realizing a curriculum transition from "easy first" to "balanced learning".
    - **Design Motivation**: Modality dependencies vary drastically across different multimodal tasks (e.g., VQA is visual-heavy, text summarization is language-heavy). Using identical learning rates for all modules can lead to insufficient learning for some modalities.

### Loss & Training
- Base loss: Standard autoregressive language modeling loss $\mathcal{L} = -\sum_t \log p(w_t | w_{<t}, I)$
- Regularization term: Knowledge distillation loss $\mathcal{L}_{KD} = \text{KL}(\hat{p} \| p_{old})$ to constrain the output distribution of the new model from deviating too far from the old model.
- The parameter budget constraint is integrated into training using the Lagrange multiplier method.
- Total loss: $\mathcal{L}_{total} = \mathcal{L} + \lambda \mathcal{L}_{KD}$

## Key Experimental Results

### Main Results

| Method | Task 1 (VQA) | Task 2 (Caption) | Task 3 (Grounding) | Average | Forgetting Rate↓ |
|------|-------------|-----------------|-------------------|------|---------|
| Sequential FT | 45.2 | 52.1 | 68.3 | 55.2 | 32.1% |
| EWC | 58.7 | 61.3 | 70.1 | 63.4 | 18.5% |
| Replay | 62.4 | 64.8 | 71.5 | 66.2 | 14.2% |
| LoRA (Fixed) | 60.1 | 63.5 | 69.8 | 64.5 | 16.8% |
| MoE-LoRA (Static) | 65.3 | 68.2 | 73.1 | 68.9 | 11.3% |
| **D-MoLE (Ours)** | **72.8** | **76.5** | **80.2** | **76.5** | **5.8%** |
| Gain (vs. Best Baseline) | +7.5 | +8.3 | +7.1 | +7.6 | -5.5pp |

### Ablation Study

| Configuration | Average Accuracy | Description |
|------|-----------|------|
| Full D-MoLE | 76.5 | All components |
| w/o Dynamic Allocation (Fixed number of experts per layer) | 71.2 (-5.3) | Dynamic allocation is key |
| w/o Layer-wise Routing (Random routing) | 72.8 (-3.7) | Routing strategy is effective |
| w/o Inter-modal Curriculum | 73.1 (-3.4) | Modality balance is important |
| w/o Knowledge Distillation | 70.5 (-6.0) | Distillation plays a significant role in preventing forgetting |
| Half Parameter Budget | 73.8 (-2.7) | Remains competitive even with reduced budget |

### Key Findings
1. **Dynamic architectures are far superior to static ones**: D-MoLE outperforms static MoE-LoRA by 7.6 points, proving the necessity of architectural evolution.
2. **15% average improvement**: Surpasses all existing baselines, marking the first work to address continual learning in MLLMs from an architectural perspective.
3. **Forgetting rate is only 5.8%**: While traditional Sequential FT has a forgetting rate > 30%, D-MoLE substantially reduces forgetting while simultaneously improving performance on new tasks.
4. **Complementary components**: Ablation experiments show that all three core components make independent contributions, and none can be omitted.

## Highlights & Insights
- **Pioneering architectural perspective** in solving MLLM continual learning, opening up a new research direction.
- **Parameter budget constraints** make the method highly practical for real-world deployment—no infinite parameter growth required.
- **Inter-modal curriculum** is an elegant design: it automatically discovers modality difficulty gaps using gradient signals, avoiding manual hyperparameter tuning.
- The combination of dynamic expert allocation + layer-wise routing allows the model to learn task-specific architectural configurations.

## Limitations & Future Work
- Currently evaluated on a relatively small number of tasks (3-5), performance on longer task sequences (e.g., 20+) remains to be observed.
- The discretization approximation of Gumbel-Softmax might not be sufficiently accurate under extreme parameter budgets.
- The effect of task arrival order on results (curriculum order sensitivity) is not discussed.
- At inference time, expert weights for all tasks must be maintained, resulting in storage overhead that grows linearly with the number of tasks.

## Related Work & Insights
- Combining MoE (Mixture-of-Experts) with continual learning is a promising research direction.
- Gradient-based difficulty metrics can be generalized to other multi-task/multimodal learning scenarios.
- Insight: The paradigm of dynamic architecture + parameter budget can be generalized to adaptation scenarios for other foundation models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to address MLLM continual learning from an architectural perspective, with novel designs across all three components.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive baseline comparisons and clear ablation experiments, though task sequence lengths are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and strong motivation.
- Value: ⭐⭐⭐⭐⭐ The 15% improvement is highly significant, opening up a new research path.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning](../../ICML2026/multimodal_vlm/same_stabilized_mixture-of-experts_for_multimodal_continual_instruction_tuning.md)
- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](../../ICCV2025/multimodal_vlm/smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[ICLR 2026\] PCLR: Progressively Compressed LoRA for Multimodal Continual Instruction Tuning](../../ICLR2026/multimodal_vlm/pclr_progressively_compressed_lora_for_multimodal_continual_instruction_tuning.md)
- [\[CVPR 2026\] Multimodal Continual Instruction Tuning with Dynamic Gradient Guidance](../../CVPR2026/multimodal_vlm/multimodal_continual_instruction_tuning_with_dynamic_gradient_guidance.md)
- [\[ACL 2025\] Enhancing Multimodal Continual Instruction Tuning with BranchLoRA](../../ACL2025/multimodal_vlm/branchlora_continual_instruction.md)

</div>

<!-- RELATED:END -->
