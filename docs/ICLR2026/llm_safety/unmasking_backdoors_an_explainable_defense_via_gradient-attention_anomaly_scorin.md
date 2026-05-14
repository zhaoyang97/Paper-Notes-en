---
title: >-
  [Paper Note] Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models
description: >-
  [ICLR 2026][LLM Safety][backdoor detection] This paper proposes X-GRAAD, an inference-time backdoor defense that combines attention anomaly scoring and gradient importance scoring to localize trigger tokens…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "backdoor detection"
  - "gradient-attention anomaly scoring"
  - "explainable defense"
  - "NLP security"
  - "inference-time defense"
date: 2026-05-08
content_hash: f8ce6b39592b9091
---

# Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.04347](https://arxiv.org/abs/2510.04347)
**Code**: None (uses OpenBackdoor toolkit)
**Area**: AI Safety / Backdoor Defense
**Keywords**: backdoor detection, gradient-attention anomaly scoring, explainable defense, NLP security, inference-time defense

## TL;DR
This paper proposes X-GRAAD, an inference-time backdoor defense that combines attention anomaly scoring and gradient importance scoring to localize trigger tokens, followed by character-level perturbation to neutralize them. Across 5 Transformer models × 3 attack types, ASR is reduced to near 0% while maintaining 88–95%+ CACC, with a 30× speedup over PURE.

## Background & Motivation

**Background**: Pre-trained language models are vulnerable to backdoor attacks, in which adversaries embed trigger patterns into training data so that models behave normally on clean inputs but produce targeted misclassifications upon encountering triggers.

**Limitations of Prior Work**:
- Training-time defenses require monitoring the entire dataset, which is infeasible in third-party pre-training scenarios.
- Inference-time defenses have limited capacity to handle unknown trigger patterns.
- Most defenses lack explainability — they cannot inform users which tokens are suspicious.

**Key Challenge**: How can trigger tokens be precisely localized and neutralized without prior knowledge of the trigger pattern?

**Goal**: Explainable inference-time backdoor defense.

**Key Insight**: A prior observation that trigger tokens simultaneously exhibit anomalies in both attention and gradient signals.

**Core Idea**: Gradient anomaly × attention anomaly = precise trigger localization → character-level perturbation neutralization.

## Method

### Overall Architecture
X-GRAAD is a two-module pipeline: (1) a **Token Attribution Scorer** that computes the product of attention anomaly and gradient anomaly for each token, and (2) a **Trigger Neutralizer & Defender** that applies character-level perturbations to the most suspicious tokens and re-predicts if the sequence anomaly score exceeds a threshold.

### Key Designs

1. **Token Attribution Scorer**:

    - **Function**: Computes a composite anomaly score for each token.
    - **Attention Importance**: The average attention matrix $\bar{A}$ is computed across all $L$ layers and $H$ heads; a token's attention importance equals the total attention weight it receives from all other tokens.
    - **Gradient Importance**: Gradients of the predicted class logit with respect to input embeddings are computed; token importance equals the $L_2$ norm of the gradient vector.
    - **Combination Strategy**: $\text{Score}(t_k) = \text{AttnScore}(t_k) \cdot \text{GradScore}(t_k)$ (multiplicative combination).
    - **Sequence Score** = maximum token score: $\psi(x) = \max_k \text{Score}(t_k)$.
    - **Design Motivation**: The multiplicative combination requires a token to be anomalous in **both channels** to be flagged as suspicious, thereby reducing false positives; the max-pooling focuses on the single most suspicious token.

2. **Trigger Neutralizer**:

    - **Function**: Applies character-level perturbations to suspicious tokens to disrupt trigger patterns.
    - **Mechanism**: The highest-scoring token is identified; 1–2 characters are inserted or replaced at random positions. The perturbation is sufficient to break exact trigger matching while preserving the overall readability of the sentence.
    - **Threshold Setting**: The score distribution is computed on a clean validation set, and the $p$-th percentile is used as the threshold.

### Loss & Training
- No training is required — this is a purely inference-time method.
- A small clean validation set (20%) is needed to calibrate the threshold.

## Key Experimental Results

### Main Results: 5 Models × 3 Attacks × 3 Datasets

| Model / Attack / Dataset | No Defense ASR | ONION | RAP | PURE | **X-GRAAD** |
|--------------------------|---------------|-------|-----|------|-------------|
| BERT-BadNets-SST2 | 1.000 | 0.085 | 0.033 | 0.011 | **0.000** |
| DistilBERT-LWS-IMDb | 0.981 | 0.512 | 0.689 | 0.728 | **0.027** |
| RoBERTa-multiple settings | ~1.0 | high | high | medium | **<0.1** |

### Ablation Study: Attention vs. Gradient vs. Combination

| Method | ASR | CACC |
|--------|-----|------|
| Attention only | medium | medium |
| Gradient only | medium | medium |
| **X-GRAAD (combined)** | **lowest** | **highest** |

### Key Findings
- **ASR → 0.0** on multiple BERT/DistilBERT settings.
- **Multi-token triggers (e.g., "james bond")**: the model concentrates reliance onto a single pivot token, which X-GRAAD can detect.
- **Domain-transfer attack (BadPre)**: ASR reduced from 0.929 to 0.003.
- **Speed**: 44–50 seconds per test set vs. 1600+ seconds for PURE.

## Highlights & Insights
- **Explainability as a core advantage**: X-GRAAD not only detects backdoors but also visualizes which tokens are triggers, providing auditable evidence.
- **Synergy of gradient × attention**: The multiplicative combination of the two signal channels is more precise than either channel alone, since backdoor triggers exhibit anomalies in both.
- **Elegant character-level perturbation**: Disrupts exact trigger matching without affecting semantics, and is more graceful than token deletion or replacement with UNK.

## Limitations & Future Work
- **Only rare-word triggers evaluated**: Effectiveness against semantic- or syntactic-level triggers has not been tested.
- **Requires a clean validation set**: Obtaining 20% clean data may be difficult in certain scenarios.
- **ALBERT requires special handling**: The compressed attention distribution of its parameter-sharing architecture necessitates different thresholds.
- **Classification tasks only**: Extension to generative LLMs has not been explored (complementary to the *Purifying LLMs* paper).

## Related Work & Insights
- **vs. ONION**: ONION relies solely on perplexity for detection, whereas X-GRAAD employs dual-channel gradient + attention signals for greater precision.
- **vs. PURE**: PURE requires 1600+ seconds; X-GRAAD takes only ~50 seconds, achieving a 30× speedup with lower ASR.
- **vs. Purifying LLMs (same venue)**: That work targets generative LLMs using MLP mechanistic analysis, while this paper targets classification PLMs via inference-time anomaly detection — the two are complementary.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The gradient × attention combination is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 models × 3 attacks × 3 datasets, but lacks evaluation on advanced trigger types.
- **Writing Quality**: ⭐⭐⭐⭐ Methodology is clearly presented with informative visualizations.
- **Value**: ⭐⭐⭐⭐ A practical inference-time backdoor defense whose explainability constitutes a clear differentiating advantage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] LAMP: Learning Universal Adversarial Perturbations for Multi-Image Tasks via Pre-trained Models](../../AAAI2026/llm_safety/lamp_learning_universal_adversarial_perturbations_for_multi-image_tasks_via_pre-.md)
- [\[ICLR 2026\] Attention Smoothing Is All You Need For Unlearning](attention_smoothing_is_all_you_need_for_unlearning.md)
- [\[ICLR 2026\] Understanding Sensitivity of Differential Attention through the Lens of Adversarial Robustness](understanding_sensitivity_of_differential_attention_through_the_lens_of_adversar.md)
- [\[ICLR 2026\] Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates](inference-time_backdoors_via_hidden_instructions_in_llm_chat_templates.md)
- [\[ICLR 2026\] SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense](shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de.md)

</div>

<!-- RELATED:END -->
