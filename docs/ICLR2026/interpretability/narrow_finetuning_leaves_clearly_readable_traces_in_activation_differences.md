---
title: >-
  [Paper Note] Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences
description: >-
  [ICLR 2026][Interpretability][Model Finetuning] This paper demonstrates that narrow finetuning leaves clearly readable traces in LLM activations: even over the first few tokens of unrelated text…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Model Finetuning"
  - "Activation Differences"
  - "Model Diffing"
  - "AI Safety"
  - "Model Organisms"
  - "Patchscope"
date: 2026-05-08
content_hash: 67079f812367f130
---

# Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences

**Conference**: ICLR 2026
**arXiv**: [2510.13900](https://arxiv.org/abs/2510.13900)
**Code**: [science-of-finetuning/diffing-toolkit](https://github.com/science-of-finetuning/diffing-toolkit)
**Area**: Interpretability
**Keywords**: Model Finetuning, Activation Differences, Model Diffing, Interpretability, AI Safety, Model Organisms, Patchscope

## TL;DR

This paper demonstrates that narrow finetuning leaves clearly readable traces in LLM activations: even over the first few tokens of unrelated text, the activation differences between pre- and post-finetuning models encode rich semantic information about the finetuning objective. Using the proposed Activation Difference Lens (ADL) method, an interpretability agent achieves a 91% success rate in identifying finetuning objectives, more than twice the performance of black-box baselines.

## Background & Motivation

### Limitations of Prior Work

In AI safety research, "Model Organisms" are models created via narrow finetuning with specific known properties (e.g., induced misalignment, implanted false facts), used to evaluate interpretability techniques. This paper **questions their validity as proxies for broadly finetuned models**.

### Core Finding

Narrow finetuning creates **static biases**: on the first few tokens of entirely unrelated text (e.g., random web data), the activation difference $\boldsymbol{\delta}_{\ell,j} = \mathbf{h}^{\text{ft}}_{\ell,j} - \mathbf{h}^{\text{base}}_{\ell,j}$ between the finetuned and base model contains rich information about the finetuning domain.

For example, a model finetuned on precision techniques for cake baking will have Patchscope reveal tokens such as "precision" and "cake," and steering will generate text like "Baking Manual:…".

## Method

### Overall Architecture: Activation Difference Lens (ADL)

ADL consists of three tools, all based on the **activation difference** $\bar{\boldsymbol{\delta}}_j$ averaged over the first $k=5$ token positions across 10,000 pretraining samples:

### Patchscope and Logit Lens

**Logit Lens**: Directly maps $\bar{\boldsymbol{\delta}}$ through the final layer norm and unembedding matrix to obtain a token distribution.

**Patchscope** (improved): Injects the scaled activation difference $\lambda \bar{\boldsymbol{\delta}}$ into the last token position of a specific prompt format, and observes the model's predicted output. This paper adds:
- Automatic search for the optimal scaling factor $\lambda$ using an LLM
- Aggregation of results across multiple prompts for improved robustness

**Token Relevance Evaluation**: Extracts the Top-20 tokens from Patchscope and uses gpt-5-mini to assess the proportion relevant to the finetuning domain.

### Steering

Adds the scaled activation difference $\alpha \bar{\boldsymbol{\delta}}_j$ to all token positions during finetuned model generation:
- Evaluated on 20 fixed chat prompts
- Binary search for optimal $\alpha$ (maximum value maintaining coherence) using gpt-5-nano
- Measures cosine similarity between steered text and the finetuning dataset via semantic embeddings (Qwen3 Embedding 0.6B)

### Interpretability Agent

An automated gpt-5-based agent provided with:
1. Top token results from Patchscope/Logit Lens
2. Paired steered and unsteered generated texts

The agent forms and validates hypotheses through interactions with the base/finetuned models (budget of $i$ turns), ultimately outputting a description of the finetuning objective. Scores are assigned by gpt-5-mini according to detailed rubrics (1–5).

### Causal Analysis

The causal effect of the bias is verified via projection substitution:

$$\widetilde{\mathbf{h}^{\text{ft}}}_{\ell,j} = \mathbf{P}_{\bar{\boldsymbol{\delta}}} \mathbf{h}^{\text{base}}_{\ell,j} + (\mathbf{I} - \mathbf{P}_{\bar{\boldsymbol{\delta}}}) \mathbf{h}^{\text{ft}}_{\ell,j}$$

The loss change $\Delta_{\mathcal{L}_{\text{CE}}}$ after substitution is measured: positive on finetuning data (removing the bias hurts finetuned performance) and negative on pretraining data (removing the bias restores general capability).

## Key Experimental Results

### Experimental Coverage

33 instances across 4 categories of Model Organisms × 7 models (1B–32B parameters):
- **SDF (Synthetic Document Finetuning)**: Implanted false facts; 5 different false facts × 4 models (Qwen3 1.7B/32B, Llama 3.2 8B, Gemma3 1B)
- **EM (Emergent Misalignment)**: Bad Medical Advice, Extreme Sports, Risky Financial (Llama 3.1 8B, Qwen2.5 7B)
- **Subliminal Learning**: Preference induction via digit sequences (Qwen2.5 7B, learning "love cats")
- **Taboo Word Game**: Hidden word with clues provided (Qwen3 1.7B, Gemma2 9B)

### Main Results: Agent Performance Comparison

| Agent Configuration | Grade ≥ 2 Rate | Grade ≥ 4 Rate |
|---|---|---|
| Blackbox $i=0$ | ~15% | ~0% |
| Blackbox $i=5$ | ~25% | ~0% |
| Blackbox $i=50$ | 39% | 1% |
| **ADL $i=0$** | **82%** | **24%** |
| **ADL $i=5$** | **91%** | **30%** |

The ADL agent correctly identifies the general finetuning objective in 82% of organisms using only difference analysis results ($i=0$, no additional interactions).

### Token Relevance Results

Token relevance of Patchscope on activation difference $\bar{\boldsymbol{\delta}}$:
- Across all organism types: 20%–80% of Top-20 tokens are relevant to the finetuning domain
- Baselines (base model mean activations, finetuned model mean activations) show negligible relevance

### Steering Results

Semantic similarity between steered text and the finetuning dataset is significantly higher than:
- Similarity between unsteered text and the finetuning dataset
- Similarity between steered text and a general chat dataset

### Ablation Study: Mitigating Bias via Data Mixing

Mixing pretraining data into finetuning data of size $|\mathcal{D}^{\text{ft}}| = 40,000$:

| Mix Ratio ft:pt | Steering Similarity | Token Relevance | Fact Internalization (FFA) |
|---|---|---|---|
| 1:0 (pure finetuning) | High | High | High |
| 1:0.1 | Significantly reduced | Reduced | Slight reduction |
| 1:1 | Near baseline | Substantially reduced | Moderate reduction |
| 1:2 | ≈ baseline | Near baseline | Marked reduction |

At a 1:1 mix ratio, no agent achieves Grade ≥ 2.

### Key Findings

1. **Narrow finetuning signal dominates chat finetuning signal**: Traces of narrow finetuning remain clearly visible even when comparing base and finetuned-chat models.
2. **Positional effect**: Early positions are most informative, but later positions also encode finetuning information with greater variability.
3. **Causal effect confirmed**: The bias exerts a positive causal effect on finetuning data (loss increases upon removal) and a negative effect on pretraining data.
4. **Bayesian GLM analysis**: ADL access is the **strongest positive factor** influencing agent performance, followed by model type and number of interactions.

## Highlights & Insights

1. **Profound AI safety implications**: Narrow finetuning model organisms may not be good proxies for studying broadly finetuned model behavior—they are too easily detectable.
2. **Power of simple tools**: Linear operations (mean activation differences) combined with existing tools (Patchscope/Logit Lens) suffice to recover finetuning objectives.
3. **New paradigm for automated interpretability**: A complete agent-based evaluation pipeline is constructed, encompassing hypothesis generation, validation, and automated scoring.
4. **Connection to catastrophic forgetting**: The bias likely arises from overfitting to semantically homogeneous finetuning data, where new learning overwrites prior knowledge.
5. **Actionable mitigation strategy**: Simply mixing in unrelated data substantially reduces the bias.

## Limitations & Future Work

1. **Evaluation pipeline relies on multiple LLM graders**: This introduces noise; differences across graders are small but present.
2. **Potentially overlooked bias types**: Whether more subtle undetectable biases persist after data mixing remains an open question.
3. **Validation limited to narrow finetuning**: Broad finetuning (e.g., standard chat tuning) produces almost no detectable traces; conclusions are confined to specific scenarios.
4. **Trade-offs in mitigation**: Data mixing may reduce internalization of the finetuning objective, particularly for Llama 3.2 1B.

## Related Work & Insights

- **Crosscoders** (Lindsey et al., 2024): An SAE-based alternative for model diffing, but more complex.
- **Emergent Misalignment** (Turner et al., 2025): One of the experimental subjects in this paper.
- **Subliminal Learning** (Cloud et al., 2025): An organism that induces preferences via digit sequences.
- **Insight**: This work articulates clear requirements for designing more realistic model organisms—finetuning data should be more diverse to avoid artificial detection shortcuts.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic demonstration that narrow finetuning leaves readable traces in activations.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Causal analysis + Bayesian GLM + automated agent; methodologically rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 33 organisms × 7 models × multiple agent configurations; extremely comprehensive.
- **Value**: ⭐⭐⭐⭐ — Directly informative for AI safety research.
- **Overall Recommendation**: ⭐⭐⭐⭐⭐ — Excellent work at the intersection of AI safety and interpretability, with profound findings and solid experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ActivationReasoning: Logical Reasoning in Latent Activation Spaces](activationreasoning_logical_reasoning_in_latent_activation_spaces.md)
- [\[ICLR 2026\] GAVEL: Towards Rule-Based Safety through Activation Monitoring](gavel_towards_rule-based_safety_through_activation_monitoring.md)
- [\[ICLR 2026\] Universal Properties of Activation Sparsity in Modern Large Language Models](universal_properties_of_activation_sparsity_in_modern_large_language_models.md)
- [\[ACL 2026\] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](../../ACL2026/interpretability/interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)
- [\[NeurIPS 2025\] FaCT: Faithful Concept Traces for Explaining Neural Network Decisions](../../NeurIPS2025/interpretability/fact_faithful_concept_traces_for_explaining_neural_network_decisions.md)

</div>

<!-- RELATED:END -->
