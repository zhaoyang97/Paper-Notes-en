---
title: >-
  [Paper Note] Adversarial Manipulation of Reasoning Models using Internal Representations
description: >-
  [ICML 2025 (R2FM Workshop)][Reasoning][Reasoning Models] This paper finds that reasoning models (such as DeepSeek-R1-Distill-Llama-8B) exhibit a linear "caution direction" in the activation space during the CoT generation phase. Ablating this direction effectively jailbreaks the model, revealing that CoT itself is a new target for adversarial attacks.
tags:
  - "ICML 2025 (R2FM Workshop)"
  - "Reasoning"
  - "Reasoning Models"
  - "Chain-of-Thought"
  - "Jailbreak"
  - "Activation Space"
  - "Caution Direction"
  - "DeepSeek-R1"
date: 2026-05-08
content_hash: 20b06a81c17406e0
---

# Adversarial Manipulation of Reasoning Models using Internal Representations

**Conference**: ICML 2025 (R2FM Workshop)  
**arXiv**: [2507.03167](https://arxiv.org/abs/2507.03167)  
**Code**: [GitHub](https://github.com/ky295/reasoning-manipulation)  
**Area**: LLM Reasoning / AI Safety / Adversarial Attacks  
**Keywords**: Reasoning Models, Chain-of-Thought, Jailbreak, Activation Space, Caution Direction, DeepSeek-R1  

## TL;DR

This paper finds that reasoning models (such as DeepSeek-R1-Distill-Llama-8B) exhibit a linear "caution direction" in the activation space during the CoT generation phase. Ablating this direction effectively jailbreaks the model, revealing that CoT itself is a new target for adversarial attacks.

## Background & Motivation

Reasoning models generate a chain-of-thought (CoT) reasoning process before producing the final response. Unlike conventional LLMs—which make refusal decisions at the prompt-response boundary—the safety mechanisms of reasoning models are embedded within the CoT generation process. This raises a crucial question: **Does CoT introduce a new attack surface?**

Existing jailbreak attacks (such as GCG and representation engineering) primarily target the prompt processing phase of conventional LLMs. For reasoning models, the safety role of CoT is not yet fully understood. The authors focus on the following questions:

1. Where do reasoning models make refusal/compliance decisions?
2. Is there an interpretable internal representation that controls this decision?
3. Can model behavior be manipulated by intervening in the activations during the CoT phase?

## Method

### Overall Architecture

The method is based on the idea of **representation engineering**, searching for a linear direction in the activation space of the reasoning model associated with safety behaviors, and then disrupting the safety mechanism of the model by ablating this direction.

### 1. Discovery of the Caution Direction

On DeepSeek-R1-Distill-Llama-8B, the middle-layer activations are collected during model CoT generation. By contrasting the activation differences when processing harmful versus harmless prompts, PCA is leveraged to find the principal direction distinguishing refusal and compliance behaviors:

$$\mathbf{d}_{\text{caution}} = \text{PCA}_1\left(\mathbb{E}[\mathbf{h}_{\text{refuse}}] - \mathbb{E}[\mathbf{h}_{\text{comply}}]\right)$$

This direction is termed the "caution direction" because it is highly correlated with cautious reasoning patterns in the CoT text (e.g., "I need to be careful", "this might be harmful").

### 2. Direction Ablation Attack

The projection of the activation vector onto the caution direction is removed:

$$\mathbf{h}' = \mathbf{h} - (\mathbf{h} \cdot \mathbf{d}_{\text{caution}}) \mathbf{d}_{\text{caution}}$$

This operation is only applied to the token activations during the CoT generation phase, leaving the prompt processing untouched.

### 3. Combination with Prompt Attacks

Integrating the caution direction information into prompt optimization attacks (such as GCG) allows the adversarial suffix to push the model away from the caution direction in the activation space, enhancing attack success rates.

### 4. Sufficiency of CoT-only Intervention

Key finding: Intervening only in the activations of CoT tokens is sufficient to control the final output, without requiring modifications to the activations during the prompt processing phase. This proves that the safety decisions of reasoning models indeed occur during the CoT generation process.

## Experiments

### Main Results

| Method | ASR (Attack Success Rate) | Intervention Phase |
|------|---------------------------|----------|
| No Attack (Baseline) | ~5% | — |
| Prompt-only GCG | ~30% | Prompt |
| CoT Caution Direction Ablation | ~75% | CoT |
| GCG + Caution Direction Ablation | ~85% | Prompt + CoT |

### Ablation Study

| Intervention Phase | Effectiveness |
|----------|------|
| Prompt Activations Only | Weak, model recovers caution in CoT |
| CoT Activations Only | Strong, directly bypasses safety reasoning |
| Prompt + CoT Activations | Strongest |
| Final Response Activations Only | Ineffective, decision is already finalized in CoT |

### Key Findings

- The caution direction generalizes across different harmful categories (violence, fraud, etc.).
- The linear separability of this direction indicates that safety mechanisms are encoded in a simple, linear manner.
- Activations along the caution direction are highest in the first few tokens of the CoT, indicating that the model makes safety judgments early in the CoT phase.

## Highlights & Insights

- **CoT is a New Attack Surface**: Reasoning models shift safety decisions from the prompt boundary to the internal CoT, which conversely exposes a new vulnerability for manipulation.
- **Linear Safety Representation**: Safety behavior is encoded as a simple linear direction in the activation space, rather than as complex non-linear structures.
- **Precision of Intervention**: Intervening only in the CoT phase is sufficient to control the output, demonstrating that CoT is the primary vehicle for reasoning model safety.
- **Implications for Alignment Research**: Alignment methods like RLHF might only create shallow linear boundaries in the activation space.

## Limitations & Future Work

- Only evaluated on DeepSeek-R1-Distill-Llama-8B (an 8B distilled model), without testing larger-scale original models.
- The robustness of the caution direction (e.g., its stability after model fine-tuning) was not sufficiently investigated.
- Defense strategies (e.g., detecting caution direction ablation) are not discussed.
- Being a workshop paper, the scale of experiments is limited.

## Related Work & Insights

- **Representation Engineering (Zou et al., 2023)**: Identifying safety-related linear directions in conventional LLMs.
- **GCG Attack (Zou et al., 2023)**: Gradient-based optimization for universal jailbreak suffixes.
- **DeepSeek-R1**: An open-source reasoning model with strong CoT reasoning capabilities.
- This work extends representation engineering from conventional LLMs to the CoT phase of reasoning models, opening up a new research direction.

## Rating

⭐⭐⭐⭐ — Concisely and powerfully reveals a new attack surface in the safety mechanisms of reasoning models; the findings carry significant security research implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Chain-of-Thought Driven Adversarial Scenario Extrapolation for Robust Language Models](../../AAAI2026/llm_reasoning/chain-of-thought_driven_adversarial_scenario_extrapolation_for_robust_language_m.md)
- [\[ICLR 2026\] The First Impression Problem: Internal Bias Triggers Overthinking in Reasoning Models](../../ICLR2026/llm_reasoning/the_first_impression_problem_internal_bias_triggers_overthinking_in_reasoning_mo.md)
- [\[ICLR 2026\] Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning](../../ICLR2026/llm_reasoning/generative_adversarial_reasoner_enhancing_llm_reasoning_with_adversarial_reinfor.md)
- [\[ACL 2026\] ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models](../../ACL2026/llm_reasoning/reprobe_efficient_test-time_scaling_of_multi-step_reasoning_by_probing_internal_.md)
- [\[NeurIPS 2025\] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning](../../NeurIPS2025/llm_reasoning/a_theoretical_study_on_bridging_internal_probability_and_sel.md)

</div>

<!-- RELATED:END -->
