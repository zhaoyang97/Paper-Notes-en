---
title: >-
  [Paper Note] Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning
description: >-
  [AAAI 2026][LLM Reasoning][CoT reasoning reliability] This paper demonstrates that attention head activations in intermediate layers of LLMs implicitly encode truthfulness information about reasoning steps during CoT inf…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "CoT reasoning reliability"
  - "internal cognition"
  - "attention head probing"
  - "confidence predictor"
  - "beam search guidance"
date: 2026-05-08
content_hash: a1508b561c18e665
---

# Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning

**Conference**: AAAI 2026
**arXiv**: [2507.10007](https://arxiv.org/abs/2507.10007)  
**Code**: [https://github.com/hfutml/cog-cot](https://github.com/hfutml/cog-cot)  
**Area**: LLM Reasoning / Interpretability
**Keywords**: CoT reasoning reliability, internal cognition, attention head probing, confidence predictor, beam search guidance

## TL;DR
This paper demonstrates that attention head activations in intermediate layers of LLMs implicitly encode truthfulness information about reasoning steps during CoT inference (probing accuracy up to 85%). Based on this finding, confidence predictors are trained to guide beam search in dynamically selecting high-confidence reasoning paths, surpassing Self-Consistency and PRM Guided Search on mathematical, symbolic, and commonsense reasoning tasks.

## Background & Motivation
**Background**: CoT reasoning unlocks deep reasoning capabilities in LLMs through step-by-step inference, but its reliability is severely undermined by error accumulation across intermediate steps — a single incorrect step propagates failures to all subsequent steps.

**Limitations of Prior Work**: Existing mitigation approaches (Self-Consistency / Self-Evaluation / PRM) rely on surface-level token probabilities or require training additional reward models. However, LLM overconfidence has been widely documented — generation probability does not equate to actual correctness, and high probability does not imply factual accuracy.

**Key Challenge**: Analogous to the phenomenon of "saying one thing while knowing another" in humans — even when generating incorrect outputs, LLMs retain structured factual knowledge in their latent space (verifiable information encoded via neural activation patterns). A dissociation exists between surface-level probability and internal cognition.

**Goal**: How to excavate the "hidden cognition" inside LLMs — leveraging the model's own intrinsic truthfulness judgments during CoT reasoning to guide more reliable inference.

**Key Insight**: This work extends findings from ITI — intermediate-layer attention heads are most sensitive to truthfulness. Unlike ITI, which focuses on static QA, this paper is the first to extend this insight to dynamic CoT reasoning scenarios, verifying that models implicitly track step-level truthfulness during progressive inference.

**Core Idea**: Use probing techniques to identify attention heads most sensitive to truthfulness, extract their activations to train a confidence predictor, and guide beam search to select high-confidence reasoning paths.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Construct a binary-annotated CoT dataset** — label each reasoning step as correct or incorrect; (2) **Train a confidence predictor** — apply probing to identify truthfulness-sensitive heads in intermediate layers and train a classifier on Top-$K$ head activations; (3) **Guide CoT generation** — use the confidence predictor during step-level beam search to select the most reliable reasoning paths.

### Key Designs

1. **Probing and Identification of Truthfulness-Sensitive Attention Heads**:

    - Function: Identify which attention heads in the Transformer are most sensitive to the correctness of CoT steps.
    - Mechanism: Construct binary-annotated CoT data $(Q, S_{1,...,n-1}, S_n^{\text{true/false}})$, train a linear probe classifier on the activations of each attention head, and use classification accuracy to measure each head's capacity to encode truthfulness.
    - Key Findings: Attention heads in the intermediate layers (approximately 1/3 to 2/3 of model depth) are most sensitive to truthfulness, with a peak probing accuracy of 85%. Furthermore, the distributions of sensitive heads in LLaVA (multimodal) and LLaMA (unimodal) are highly similar — suggesting that this "hidden cognition" originates from pretraining rather than task-specific fine-tuning.

2. **Confidence Predictor**:

    - Function: Transform distributed truthfulness signals into an actionable scalar confidence score.
    - Mechanism: Select the Top-$K$ attention heads with the highest probing accuracy (across all layers), concatenate the last-token activations into a feature vector $\mathbf{v} = \text{Concat}(\mathbf{h}_{h_1}^{l_1}, ..., \mathbf{h}_{h_K}^{l_K})$, and output a confidence score via a linear classifier with sigmoid activation: $p_\theta(y|\mathbf{x}) = \sigma(\mathbf{W} \cdot \mathbf{v} + b)$.
    - ECE Loss is adopted in place of MSE — empirical accuracy from cross-validation replaces binary hard labels as soft targets, substantially improving calibration.
    - vs. ACTCAB (last-layer only): The proposed method selects sensitive heads across multiple layers; it comprehensively outperforms ACTCAB on ECE, Brier score, and AUC.

3. **Confidence-Guided Beam Search**:

    - Function: Generate multiple candidates at each CoT step and select the optimal one using the confidence predictor.
    - Mechanism: Decompose CoT step-by-step → generate $M$ candidates per step via beam search → compute a composite score $\text{Score}(C) = \lambda \cdot \beta(C) + (1-\lambda) \cdot \bar{P}(C)$, where $\beta$ denotes confidence, $\bar{P}$ denotes generation probability, and $\lambda=0.5$.
    - The highest-scoring candidate is appended to the reasoning chain, and the process iterates until the final answer is produced.

## Key Experimental Results

### Main Results — Unimodal Reasoning (LLaMA2-13B)

| Method | GSM8K | SVAMP | BoolExpr | StrategyQA | BoolQ | Avg |
|--------|-------|-------|----------|------------|-------|-----|
| CoT-few | 39.9 | 53.7 | 66.0 | 57.6 | 68.8 | 57.2 |
| Self-Consistency | 39.3 | 54.0 | 65.9 | 56.6 | 70.0 | 57.2 |
| PRM | 39.2 | 55.0 | 65.8 | 53.8 | 68.0 | 56.4 |
| **Ours** | **42.8** | **55.7** | **66.8** | **59.2** | 68.8 | **58.7** |

### Multimodal Reasoning (LLaVA-13B)

| Method | ScienceQA | CLEVR-Math | RealWorldQA | MMStar | Avg |
|--------|-----------|------------|-------------|--------|-----|
| CoT-few | 61.9 | 31.9 | 10.7 | 41.0 | 36.4 |
| PRM | 61.3 | 33.9 | 9.7 | 41.3 | 36.6 |
| **Ours** | **69.2** | 31.9 | **14.0** | **42.0** | **39.3** |

### Confidence Predictor Calibration (LLaMA2-13B, WikiQA)

| Method | ECE↓ | Brier↓ | AUC↑ |
|--------|------|--------|------|
| Seq Likelihood | 0.254 | 0.291 | 0.640 |
| "Is True" Prob | 0.146 | 0.231 | 0.747 |
| ACTCAB (last layer) | 0.058 | 0.149 | 0.868 |
| **Ours (multi-layer Top-K heads)** | **0.037** | **0.102** | **0.934** |

### Key Findings
- The confidence predictor achieves substantially better calibration than surface-probability methods — ECE decreases from 0.254 to 0.037 and AUC improves from 0.640 to 0.934.
- Intermediate-layer attention heads are the most information-dense source of truthfulness signals — using only the last layer (ACTCAB) provides insufficient signal.
- Improvements remain consistent on LLaMA2-70B (68.2→71.5 avg), demonstrating cross-scale generalization.
- The method is also effective on DeepSeek-R1 distilled models (75.3→77.6 avg) — even reasoning models explicitly trained for CoT can benefit.
- Random candidate selection (without confidence guidance) degrades performance, sometimes below the CoT baseline — validating the necessity of the confidence predictor.
- The approach is compatible with error self-correction techniques (setting a confidence threshold to trigger self-correction).

## Highlights & Insights
- **The "knowing but not saying" analogy**: Even when an LLM produces an incorrect answer, its intermediate layers still encode correct information — this insight provides a theoretical foundation for using internal model signals to correct surface-level errors.
- **From static QA to dynamic CoT**: This work is the first to extend truthfulness probing from static question answering to progressive CoT reasoning, verifying that models continuously track step-level truthfulness during dynamic inference.
- **Multi-layer Top-K head selection**: Rather than being confined to the last layer, the method selects the $K$ most sensitive heads across all layers — cross-layer aggregation yields richer truthfulness signals.
- **Resonance with URaG**: URaG finds that intermediate layers of MLLMs encode retrieval capabilities; this paper finds that intermediate layers encode truthfulness judgments — further corroborating the view that "intermediate layers are the key information layers of Transformers."

## Limitations & Future Work
- The confidence predictor requires training separate probes and selecting sensitive heads for each individual model.
- Beam search increases inference time — each step requires generating multiple candidates and performing forward passes.
- Only the last-token activation is used — signals from other tokens in the sequence may be overlooked.
- Improvements in multimodal settings are less pronounced than in unimodal settings — possibly because the truthfulness encoding patterns for visual information differ.

## Related Work & Insights
- **vs. Self-Consistency**: SC selects answers via repeated sampling and majority voting; the proposed method progressively selects reasoning paths via internal confidence — offering finer granularity without requiring the final answer for evaluation.
- **vs. PRM**: PRM requires training a full reward model; this work requires only a lightweight linear probe with attention head selection — substantially lower cost.
- **vs. ITI/DoLA**: ITI intervenes in activations using fixed steering vectors; DoLA contrasts logits across layers; this work observes and selects rather than intervening.
- **vs. LLM-CAS (in the same batch of notes)**: LLM-CAS uses RL to learn "how to perturb"; this work uses probing to learn "how to judge" — detection vs. correction, with potential for combination.

## Rating
- Novelty: ⭐⭐⭐⭐ Extending truthfulness probing to dynamic CoT scenarios is a meaningful contribution, though the overall framework of probing + beam search is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 7B/13B/70B unimodal + 7B/13B multimodal + DeepSeek-R1 + calibration evaluation + ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Insights are clearly articulated and experiments are thorough.
- Value: ⭐⭐⭐⭐⭐ The insight that "models know when they are lying" carries far-reaching implications for reliable reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models](badthink_triggered_overthinking_attacks_on_chain-of-thought_reasoning_in_large_l.md)
- [\[AAAI 2026\] L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention](l2v-cot_cross-modal_transfer_of_chain-of-thought_reasoning_v.md)
- [\[AAAI 2026\] Intention Chain-of-Thought Prompting with Dynamic Routing for Code Generation](intention_chain-of-thought_prompting_with_dynamic_routing_for_code_generation.md)
- [\[AAAI 2026\] Chain-of-Thought Driven Adversarial Scenario Extrapolation for Robust Language Models](chain-of-thought_driven_adversarial_scenario_extrapolation_for_robust_language_m.md)
- [\[ACL 2026\] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](../../ACL2026/llm_reasoning/render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)

</div>

<!-- RELATED:END -->
