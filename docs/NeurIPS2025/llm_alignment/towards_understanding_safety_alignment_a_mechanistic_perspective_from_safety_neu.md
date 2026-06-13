---
title: >-
  [Paper Note] Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons
description: >-
  [NeurIPS 2025][LLM Alignment][safety alignment] Through a mechanistic interpretability lens, this work identifies a sparse set of "safety neurons" comprising approximately 5% of all neurons in LLMs. Patching only these n…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "safety alignment"
  - "mechanistic interpretability"
  - "safety neurons"
  - "activation patching"
  - "alignment tax"
date: 2026-05-08
content_hash: 9486ac8964f2230e
---

# Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons

**Conference**: NeurIPS 2025
**arXiv**: [2406.14144](https://arxiv.org/abs/2406.14144)  
**Code**: [THU-KEG/SafetyNeuron](https://github.com/THU-KEG/SafetyNeuron)  
**Area**: LLM Alignment
**Keywords**: safety alignment, mechanistic interpretability, safety neurons, activation patching, alignment tax

## TL;DR

Through a mechanistic interpretability lens, this work identifies a sparse set of "safety neurons" comprising approximately 5% of all neurons in LLMs. Patching only these neurons' activations recovers over 90% of safety performance, and the neuron-overlap perspective offers a mechanistic explanation for the alignment tax phenomenon.

## Background & Motivation

Although large language models have achieved substantial safety improvements through alignment training, they remain vulnerable to adversarial attacks. Understanding the internal mechanisms of safety alignment is essential for designing more robust alignment algorithms. Existing mechanistic interpretability methods—such as attribution over attention heads—are primarily designed for tasks requiring a prompt and a short token output, and cannot be directly applied to the open-ended generation setting of safety alignment.

The central question of this paper is: **what exactly does safety alignment change inside an LLM?** The authors approach this from the most fundamental level—MLP neurons—seeking to identify "safety neurons" responsible for safe behavior and to verify their causal effect.

Additionally, the well-known **alignment tax** problem in safety alignment—where improving safety degrades helpfulness and vice versa—is addressed; the authors aim to provide a mechanistic explanation from the neuron-level perspective.

## Method

### Overall Architecture

A two-stage framework is proposed: first narrowing down candidates via association, then validating causality.

**Mechanism**: Association is a necessary condition for causation. Neurons associated with safe behavior are first identified, then causal analysis is applied to validate them.

### Key Design 1: Inference-time Activation Contrasting

Given two models $\mathcal{M}_1$ (SFT model) and $\mathcal{M}_2$ (DPO safety-aligned model), MLP intermediate-layer neuron activations are collected for the same inputs, and a **change score** is computed:

$$\mathcal{S}_i^{(l)}(\mathcal{M}_1, \mathcal{M}_2; \mathcal{D}) = \sqrt{\frac{\sum_{w \in \mathcal{D}} \sum_{j=|w|}^{|\bar{w}^1|-1} \left(a_i^{(l)}(\mathcal{M}_1; \bar{w}^1)[j] - a_i^{(l)}(\mathcal{M}_2; \bar{w}^1)[j]\right)^2}{\sum_{w \in \mathcal{D}} |w^1|}}$$

where $a_i^{(l)}$ denotes the activation of the $i$-th neuron in layer $l$. Neurons are ranked in descending order of change score, and the top-ranked neurons are selected as safety neuron candidates.

**Why MLP intermediate-layer neurons?** Neurons in the MLP intermediate layer (after the activation function, before the down projection) have been shown to encode a variety of interpretable features, and each row of the down projection matrix can be interpreted as the "value vector" of the corresponding neuron.

### Key Design 2: Dynamic Activation Patching

Traditional activation patching handles only single-step outputs. This paper proposes **dynamic activation patching** for the open-ended generation setting:

1. For the current prompt $w$, run $\mathcal{M}_2$ to cache the target neuron activations.
2. Run $\mathcal{M}_1$, replacing target neuron activations with the cached values while leaving all other neurons unchanged.
3. Obtain the next-token prediction and append it to the prompt.
4. Repeat until generation is complete.

The causal effect is defined as:

$$\mathcal{C} = \frac{\mathbb{E}_{w \in \mathcal{D}}[\mathcal{F}(\tilde{w}^1) - \mathcal{F}(\bar{w}^1)]}{\mathbb{E}_{w \in \mathcal{D}}[\mathcal{F}(\bar{w}^2) - \mathcal{F}(\bar{w}^1)]}$$

$\mathcal{C} \approx 1$ indicates that these neurons fully account for safety capability; $\mathcal{C} \approx 0$ indicates negligible causal effect.

### Loss & Training

- Alignment training uses **(IA)³** as the PEFT method, applied exclusively to MLP layers.
- (IA)³ modifies activations by multiplying learned scaling factors without altering underlying parameters, thereby preserving the functional identity of MLP neurons.
- The SFT stage trains on ShareGPT; safety alignment uses DPO on HH-RLHF-Harmless.

## Key Experimental Results

### Main Results: Sparsity and Causal Effect of Safety Neurons

Experiments are conducted on four models: Llama2-7B, Mistral-7B, Gemma-7B, and Qwen2.5-3B.

| Model | Setting | BT(↑) | RT(↑) | HB(↑) | JL(↑) | ΔGen |
|-------|---------|--------|--------|--------|--------|------|
| Llama2 | Base→Ours | **+56** | **+65** | **+63** | **+78** | -0.01 |
| Llama2 | SFT→Ours | **+101** | **+101** | **+76** | **+73** | +0.01 |
| Mistral | Base→Ours | **+71** | **+63** | **+134** | **+103** | -0.04 |
| Mistral | SFT→Ours | **+90** | **+80** | **+74** | **+75** | +0.01 |
| Gemma | SFT→Ours | **+96** | **+84** | **+79** | **+89** | -0.02 |
| Qwen2.5 | SFT→Ours | **+83** | **+81** | **+68** | **+83** | +0.01 |

- Patching only approximately **5%** of neurons recovers **>90%** of safety performance.
- Two baselines—Pruning and SN-Tune—exhibit very low causal effects (typically in the range of -10 to +30).
- Changes in general capability ΔGen are negligible ($|\Delta\text{Gen}| \leq 0.05$), indicating that safety enhancement does not meaningfully affect generation quality.

### Ablation Study: Random Neurons vs. Safety Neurons

- Patching an equal number of randomly sampled neurons yields significantly lower causal effects than patching safety neurons.
- t-test p-values range from $1.15 \times 10^{-6}$ to $1.67 \times 10^{-18}$, confirming highly significant differences.

### Alignment Tax Mechanism Analysis

| Patch Direction | Llama2 Safety↓ | Llama2 Helpful↑ | Mistral Safety↓ | Mistral Helpful↑ |
|----------------|---------------|----------------|----------------|----------------|
| Helpfulness→Safety | 7.3 | 7.97 | 6.6 | 8.1 |
| Safety→Helpfulness | 10.1 | 2.3 | 10.7 | 1.0 |

The Spearman correlation between safety and helpfulness preference neurons is **>0.95**, while correlations with other capabilities such as reasoning are substantially lower. The same set of neurons requires distinct activation patterns to realize safety versus helpfulness.

### Key Findings

1. Safety neurons identified under different random seeds exhibit overlap rates >0.95 and Spearman correlations >0.95.
2. A classifier trained on safety neuron activations can predict harmful outputs before generation, achieving an average accuracy of 76.2%.
3. When the value vectors of safety neurons are projected onto the vocabulary space, the associated top tokens are not safety-related (e.g., food words, conjunctions, parentheses), suggesting that the safety mechanism is more complex than simply suppressing toxic tokens.

## Highlights & Insights

1. **Methodological innovation**: The two-stage framework proceeding from association to causation—particularly **dynamic activation patching**—addresses the causal verification challenge in open-ended generation, representing an important extension of traditional activation patching.
2. **Sparsity finding**: Only 5% of neurons carry more than 90% of the safety mechanism. This finding holds both theoretical value (safety alignment is sparse) and practical value (enabling efficient safety enhancement).
3. **Alignment tax explanation**: For the first time, a mechanistic explanation is provided at the neuron level—safety and helpfulness share a highly overlapping set of neurons but require different activation patterns, analogous to resource competition.
4. **Practical application**: The safeguard application demonstrates the feasibility of predicting harmful outputs prior to generation, with classifier overhead of less than 0.001 seconds.

## Limitations & Future Work

1. **Alignment method scope**: Only the (IA)³ + DPO setting is validated; full-parameter fine-tuning for safety alignment may violate the assumption that MLP neuron functionality is preserved.
2. **Model scale**: Experiments focus on 3B–7B models; the distribution of safety neurons in larger-scale models may differ.
3. **Insufficient depth of causal mechanism**: Although safety neurons are shown not to encode safety-related tokens, the precise mechanism by which they operate remains unexplained.
4. **Limited safeguard accuracy**: A detection accuracy of 76.2% remains insufficient for practical deployment.
5. **Attention head interactions unexplored**: MLP neurons account for approximately two-thirds of model parameters, yet their interaction with attention heads is not thoroughly analyzed.

## Related Work & Insights

- **Relation to Refusal Direction**: Arditi et al. (2024) find that refusal behavior is governed by a single direction; this paper provides a more fine-grained perspective at the neuron level.
- **Distinction from Lee et al. (2024)**: The latter identifies "toxicity neurons" in GPT-2 that correlate with toxic tokens, whereas this paper finds that safety neurons operate through a more complex mechanism, with associated tokens being non-safety-related.
- **Comparison with ablation-based methods**: Traditional approaches (Pruning, SN-Tune) adopt ablation strategies, which may be susceptible to the Hydra effect—removing certain components triggers compensatory behavior in others. This paper employs enhancement (patching) rather than ablation, providing a more direct causal validation.
- **Broader implications**: The proposed framework can be generalized to mechanistic studies of other high-level capabilities, such as instruction following and multilingual competence.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The two-stage framework combining inference-time activation contrasting and dynamic activation patching is a novel contribution, with methodological significance for causal analysis in open-ended generation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluations span 4 models × 4 red-teaming benchmarks × multiple general benchmarks, with baseline comparisons, ablations, stability analyses, alignment tax analysis, and application demonstrations.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured and logically coherent, with a good balance between formal notation and intuitive explanation.
- **Value**: ⭐⭐⭐⭐ — Makes an important contribution to understanding the mechanisms of safety alignment; the alignment tax explanation is insightful, and the safeguard application demonstrates practical potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LLM Safety Alignment is Divergence Estimation in Disguise](llm_safety_alignment_is_divergence_estimation_in_disguise.md)
- [\[ICLR 2026\] Superficial Safety Alignment Hypothesis](../../ICLR2026/llm_alignment/superficial_safety_alignment_hypothesis.md)
- [\[ICML 2026\] Curriculum Learning for Safety Alignment](../../ICML2026/llm_alignment/curriculum_learning_for_safety_alignment.md)
- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](../../ICLR2026/llm_alignment/mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)
- [\[NeurIPS 2025\] What Makes a Reward Model a Good Teacher? An Optimization Perspective](what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)

</div>

<!-- RELATED:END -->
