---
title: >-
  [Paper Note] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning
description: >-
  [ACL 2026][Robotics][Task Vector] DeCoVec constructs task vectors in the decoding space (output logits) by contrasting few-shot and zero-shot logit distributions: $\mathbf{v}_\mathcal{T}^t = \mathbf{z}_{\text{icl}}^t - \mathbf{z}_{\text{zs}}^t$, injecting them into decoding via $\tilde{\mathbf{z}}^t = \mathbf{z}_{\text{de}}^t + \lambda \cdot \mathbf{v}_\mathcal{T}^t$, achieving up to +5.50 average accuracy improvement over standard few-shot baselines across 7 LLMs without any training.
tags:
  - ACL 2026
  - Robotics
  - Task Vector
  - Decoding Space
  - In-Context Learning
  - Training-Free LLM Guidance
  - Logit Manipulation
content_hash: 1ba728a79fd1c88c
---

# DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning

**Conference**: ACL 2026
**arXiv**: [2604.11129](https://arxiv.org/abs/2604.11129)
**Code**: [GitHub](https://github.com/szu-tera/DeCoVec)
**Area**: Robotics & Embodied AI
**Keywords**: Task Vector, Decoding Space, In-Context Learning, Training-Free LLM Guidance, Logit Manipulation

## TL;DR
DeCoVec constructs task vectors in the decoding space (output logits) by contrasting few-shot and zero-shot logit distributions: $\mathbf{v}_\mathcal{T}^t = \mathbf{z}_{\text{icl}}^t - \mathbf{z}_{\text{zs}}^t$, injecting them into decoding via $\tilde{\mathbf{z}}^t = \mathbf{z}_{\text{de}}^t + \lambda \cdot \mathbf{v}_\mathcal{T}^t$, achieving up to +5.50 average accuracy improvement over standard few-shot baselines across 7 LLMs without any training.

## Method

### Key Designs

1. **Decoding Space Task Vector Construction**: Zero-shot context represents task-agnostic state; few-shot ICL context represents task-aware state. Their logit difference encodes task-level features activated by ICL.

2. **Token-Level Online Guidance**: Three forward passes per decoding step: base context logit, zero-shot logit, and steering ICL logit. Ensures task signal dynamically aligns with current generation context.

3. **Two-Type ICL Context Separation**: Independent sampling strategies for steering context (for task vector computation) and decode context (for base decoding).

## Key Experimental Results

- Consistent improvement across all 7 models (0.5B-9B), with maximum +5.50 average accuracy
- Robust to example ordering unlike standard ICL
- No additional input token cost — operates entirely in logit space

## Highlights & Insights
- Constructing task vectors in decoding space is a conceptual breakthrough — moving task vectors from "model internals" to the "model output interface"
- "ICL's logit difference encodes task semantics" has theoretical significance for understanding ICL mechanisms

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[ACL 2026\] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](grasprune_global_gating_for_budgeted_structured_pruning_of_large_language_models.md)
- [\[ICLR 2026\] Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](../../ICLR2026/robotics/domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)
- [\[ICLR 2026\] JULI: Jailbreak Large Language Models by Self-Introspection](../../ICLR2026/robotics/juli_jailbreak_large_language_models_by_self-introspection.md)
- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](../../NeurIPS2025/robotics/understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)

<!-- RELATED:END -->
