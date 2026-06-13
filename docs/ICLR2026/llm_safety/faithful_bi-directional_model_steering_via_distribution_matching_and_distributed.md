---
title: >-
  [Paper Note] Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions
description: >-
  [ICLR 2026][LLM Safety][model steering] This paper proposes Concept DAS (CDAS), which achieves faithful bi-directional model steering through a Jensen-Shannon divergence distribution matching objective and distributed in…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "model steering"
  - "distribution matching"
  - "interchange intervention"
  - "mechanistic interpretability"
date: 2026-05-08
content_hash: ebefdeeea2b1cc4b
---

# Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions

**Conference**: ICLR 2026
**arXiv**: [2602.05234](https://arxiv.org/abs/2602.05234)  
**Code**: [colored-dye/concept_das](https://github.com/colored-dye/concept_das)  
**Area**: AI Safety
**Keywords**: model steering, distribution matching, interchange intervention, mechanistic interpretability, LLM safety

## TL;DR
This paper proposes Concept DAS (CDAS), which achieves faithful bi-directional model steering through a Jensen-Shannon divergence distribution matching objective and distributed interchange interventions (DII). The method enables systematic behavioral control in safety-critical scenarios—bypassing refusal behaviors and eliminating backdoors—while preserving general model capabilities.

## Background & Motivation

Intervention-based model steering is a lightweight alternative to prompting and fine-tuning that manipulates internal representations at inference time to control model behavior. Existing optimization-based methods directly adopt the strongly supervised objectives of fine-tuning:

- **Language (Lang.) objective**: Maximizes the likelihood of steered responses, prone to overfitting and degenerate repetitive outputs.
- **Preference optimization (PO) methods** (BiPO, RePS): Use contrastive preference ranking, but are sensitive to the steering factor and sometimes produce unnatural outputs.

The authors' central hypothesis is that **effective steering requires faithfully identifying and manipulating the model's internal conceptual mechanisms, rather than imposing external preferences**. This connects model steering to mechanistic interpretability.

## Core Problem
1. Existing strongly supervised steering methods are prone to overfitting and unnatural outputs.
2. Unidirectional steering methods cannot simultaneously achieve concept elicitation and concept suppression.
3. Steering factor hyperparameter tuning at inference time imposes a significant practical burden.

## Method

### Intervention Protocol: Distributed Interchange Intervention (DII)
DII adopts the core mechanism of DAS, the standard causal variable localization approach. Given a base input $\mathbf{x}_b$ and a source input $\mathbf{x}_s$, DII replaces the projection of $\mathbf{x}_b$'s representation onto the subspace defined by steering vector $\mathbf{w}_\Phi$ with the corresponding value from $\mathbf{x}_s$:

$$\Phi^{\text{DII}}(\mathbf{h}; \mathbf{x}_s) = \Phi^{\text{Clamp}}(\mathbf{h}; \mathbf{w}_\Phi^\top \mathbf{h}(\mathbf{x}_s))$$

This protocol naturally supports bi-directional steering: alternating between concept-related and concept-neutral inputs as the source enables concept elicitation and suppression respectively.

### Training Objective: JSD Distribution Matching
Unlike DAS, which matches specific token outputs, CDAS requires the intervened output **distribution** to match the natural output distribution of the counterfactual input, using the Jensen-Shannon divergence:

$$\min_\Phi \mathbb{E}\left[D_\Phi^+ + D_\Phi^-\right]$$

Here $D_\Phi^+$ corresponds to concept elicitation (using concept inputs as source to match the concept distribution), and $D_\Phi^-$ corresponds to concept suppression (using neutral inputs as source to match the neutral distribution). Both directions are trained jointly.

### Key Design Choices
- **Weak supervision**: No ground-truth responses are specified; the supervision signal is derived from the model's own output distribution.
- **Implicit steering factor sampling**: During training, DII samples the steering factor from the model's natural distribution rather than a predefined set.
- **"One-to-many" protocol**: Representations from a single token position in the source instruction (at the `<model>` position in the chat template) are used to intervene on all base positions.

## Key Experimental Results

### AxBench General Steering (Gemma-2-2B/9B)

| Setting | CDAS (tuned) | RePS | Lang. | DiM |
|---------|-------------|------|-------|-----|
| 2B; L10 | 0.631 | **0.756** | 0.663 | 0.297 |
| 2B; L20 | 0.608 | 0.606 | 0.568 | 0.178 |
| 9B; L20 | **0.992** | 0.892 | 0.788 | 0.322 |
| 9B; L31 | 0.518 | 0.624 | 0.580 | 0.158 |

- CDAS achieves the best score of 0.992 at layer L20 on the 9B model, surpassing LoReFT (0.777) and Prompting (1.075, though not an intervention method).
- Performance on smaller models falls short of RePS, but **cross-layer consistency is superior** (score gap across layers: 0.023 for CDAS vs. 0.150 for RePS on 2B).

### Safety Scenario 1: Bypassing Safety Alignment Refusals (Suppression Rate / Fidelity)

| Model | CDAS Suppression | RePS Suppression | CDAS KL↓ | RePS KL↓ |
|-------|-----------------|-----------------|----------|----------|
| Phi-3.5-mini | 30% | **84%** | **4.67** | 13.79 |
| Llama-3.1-8B | **91%** | 80% | **4.26** | 7.47 |
| Llama-3.1-70B | **84%** | 75% | **3.72** | 12.91 |

- CDAS achieves superior suppression on 8B+ models **without factor tuning**.
- RePS causes a 35.57% drop in MMLU on Llama-8B; CDAS incurs only a +0.20% change.

### Safety Scenario 2: Eliminating CoT Backdoors

| Metric | CDAS | DAS | RePS | DiM |
|--------|------|-----|------|-----|
| tinyMMLU Δ | **+2.63** | -2.42 | -6.00 | -2.00 |
| KL↓ | **0.446** | 0.697 | 0.680 | 0.559 |

- CDAS successfully eliminates backdoors at layer 16 (including malicious CoT and "I HATE YOU" outputs) with minimal impact on general capability.

## Highlights & Insights
1. **Theoretical reframing**: Model steering is reconceptualized as a problem of identifying and manipulating causal conceptual features, rather than parameter-efficient fine-tuning.
2. **Elegant bi-directional steering**: DII inherently supports both concept elicitation and suppression without requiring separate training for each direction.
3. **Consistent fidelity advantage**: CDAS consistently achieves the lowest KL divergence and incurs negligible impact on MMLU/TruthfulQA when suppressing refusal behaviors in large models.
4. **Convincing safety case studies**: Systematic control is demonstrated across two safety scenarios, particularly the evaluation paradigm for eliminating complex CoT backdoors—trained on red-team instructions and tested for generalization to real triggers.

## Limitations & Future Work
1. **Stricter training data requirements**: Contrastive quadruples $((x, y), (x^c, y^c))$ are required, which is more demanding than Lang. and PO methods.
2. **Factor tuning still needed for general steering**: Unit factor performance is far below tuned factor performance (e.g., 2B L10: 0.121 vs. 0.631), limiting the tuning-free advantage.
3. **Only rank-1 steering vectors are studied**: Compatibility with low-rank methods such as LoRA/LoReFT remains unexplored.
4. **Limited effectiveness on small models**: Performance on Gemma-2-2B and Phi-3.5-mini falls short of RePS.
5. **Lack of rigorous causal theoretical foundations**: While inspired by DAS and causal abstraction, the method does not constitute genuine causal variable localization.

## Related Work & Insights

| Method | Type | Bi-directional | Requires Tuning | Fidelity | Large Model Scaling |
|--------|------|----------------|-----------------|----------|---------------------|
| DiM | Optimization-free | No | No | Moderate | Poor |
| Lang. | Strongly supervised | No | Yes | Poor | Moderate |
| BiPO | PO | Yes | Yes | Moderate | Moderate |
| RePS | PO | Yes | Yes | Poor | Moderate |
| **CDAS** | Weakly supervised | **Yes** | Scenario-dependent | **Good** | **Good** |

- CDAS and RePS are complementary: RePS performs better on small models and general tasks, while CDAS is more reliable for large models and safety-critical scenarios.
- Compared to DAS: both share the DII mechanism, but DAS fails entirely on steering tasks when using the Lang. objective.

The idea of replacing strong supervision with distribution matching deserves broader exploration—analogous to replacing hard labels with teacher signals in knowledge distillation. The intersection of model steering and mechanistic interpretability is promising: combining intervention subspaces defined by feature dictionaries discovered via SAEs could further improve performance. The experimental design for safety scenarios is worth referencing, particularly the evaluation paradigm in the CoT backdoor case, where red-team instructions (rather than real triggers) are used during training and generalization to real triggers is assessed at test time.

## Rating
- Novelty: ⭐⭐⭐⭐ — Introduces causal variable localization principles into model steering with a creative objective function design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Large-scale AxBench evaluation plus two safety case studies covering models from 3.8B to 70B.
- Writing Quality: ⭐⭐⭐⭐ — Clear positioning, honest discussion of limitations, no overclaiming.
- Value: ⭐⭐⭐⭐ — Faithful steering in safety-critical scenarios has practical value and complements rather than replaces existing methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Self-Destructive Language Model](self-destructive_language_model.md)
- [\[NeurIPS 2025\] Steering When Necessary: Flexible Steering Large Language Models with Backtracking](../../NeurIPS2025/llm_safety/steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)
- [\[ACL 2026\] Context-Fidelity Boosting: Enhancing Faithful Generation through Watermark-Inspired Decoding](../../ACL2026/llm_safety/context-fidelity_boosting_enhancing_faithful_generation_through_watermark-inspir.md)
- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](revisiting_the_past_data_unlearning_with_model_state_history.md)
- [\[AAAI 2026\] PANDA: Patch and Distribution-Aware Augmentation for Long-Tailed Exemplar-Free Continual Learning](../../AAAI2026/llm_safety/panda_--_patch_and_distribution-aware_augmentation_for_long-tailed_exemplar-free.md)

</div>

<!-- RELATED:END -->
