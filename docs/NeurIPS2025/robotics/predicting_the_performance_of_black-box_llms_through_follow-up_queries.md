---
title: >-
  [Paper Note] Predicting the Performance of Black-Box LLMs through Follow-Up Queries
description: >-
  [NeurIPS 2025][Robotics][Black-box LLM] This paper proposes QueRE, a method that poses approximately 50 follow-up questions to a black-box LLM (e.g.…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Black-box LLM"
  - "performance prediction"
  - "follow-up queries"
  - "uncertainty quantification"
  - "adversarial detection"
date: 2026-05-08
content_hash: 082e5ed1e901f16d
---

# Predicting the Performance of Black-Box LLMs through Follow-Up Queries

**Conference**: NeurIPS 2025
**arXiv**: [2501.01558](https://arxiv.org/abs/2501.01558)
**Code**: None
**Area**: Robotics
**Keywords**: Black-box LLM, performance prediction, follow-up queries, uncertainty quantification, adversarial detection

## TL;DR

This paper proposes QueRE, a method that poses approximately 50 follow-up questions to a black-box LLM (e.g., "Are you confident in your answer?") and uses the resulting "Yes" token probabilities as features to train a linear classifier. QueRE achieves strong performance on predicting model correctness, detecting adversarial manipulation, and distinguishing between different LLMs — surpassing even white-box methods that require access to internal model states.

## Background & Motivation

Reliably predicting LLM behavior — whether an output is correct or has been adversarially manipulated — is a fundamental challenge. State-of-the-art LLMs are deployed through closed-source APIs with only black-box access, rendering internal-state-based analysis methods (e.g., RepE, mechanistic interpretability) inapplicable.

**Core Problem**: How well can LLM behavior be predicted using only black-box access?

**Key Assumption**: The distribution of an LLM's responses to follow-up questions varies meaningfully with correctness, model family, and model scale. Since LLMs are trained to understand natural language and provide helpful responses, their answers to self-reflective questions should carry informative signals about their behavior.

Limitations of prior work:
- **White-box methods** (RepE, Full Logits) require access to internal representations and are inapplicable to closed-source models
- **Single confidence scores** constitute one-dimensional features with insufficient information
- **Semantic entropy** requires multiple samples and incurs high computational cost
- **Self-consistency** methods exhibit limited effectiveness on reasoning tasks

## Method

### Overall Architecture

QueRE (Follow-up Question Representation Elicitation) operates as follows:
1. The LLM receives an original question $x$ and produces a greedy-decoded answer $a = \arg\max_c P(c|x)$
2. A set of $d$ follow-up questions $Q = \{q_1, ..., q_d\}$ is posed sequentially
3. The "Yes" token probability is extracted for each question: $z_j = P(\text{yes} | x \oplus a \oplus q_j)$
4. The feature vector $z = (z_1, ..., z_d)$ is fed into a linear classifier to predict the target (correctness / manipulation / identity)

### Key Designs

#### 1. **Construction of Follow-Up Questions**

A small set of base questions is designed manually, and approximately 40 additional questions are generated using GPT-4, yielding roughly 50 in total. Question types include:
- Confidence-related: *"Do you think your answer is correct?"*
- Reasoning quality: *"Are you able to explain your answer?"*
- Bias detection: *"Are your responses free from bias?"*

**Design Motivation**: Each question's Yes probability functions as a **weak predictor** (analogous to a weak learner in boosting); their linear combination forms a strong predictor. All follow-up questions can be processed in parallel, so adding more questions incurs only negligible computational overhead.

#### 2. **Feature Augmentation**

Beyond the core follow-up question features, the method appends:
- Closed-form QA: probability distributions over answer choices
- All QA: pre-confidence and post-confidence scores (self-confidence probabilities before and after the model observes its own answer)

#### 3. **Theoretical Guarantees for Sampling Approximation**

When an API does not expose top-$k$ probabilities, high-temperature sampling $k$ times can serve as an approximation.

**Proposition 1**: The logistic regression MLE $\hat{\beta}$ obtained via sampling approximation converges to the optimal parameter $\beta_0$ at a rate of $O\!\left(\frac{1}{\sqrt{n}} + \frac{\sqrt{n}}{k}\right)$.

Thus, as long as the number of samples $k$ grows with the dataset size $n$ (potentially at a slower rate), the estimator remains consistent.

### Loss & Training

A standard logistic regression objective is used to train the linear classifier, requiring no complex training procedures. The choice of a linear model is deliberate:
1. Low-dimensional features combined with a simple model yield tighter generalization bounds
2. Overfitting to prompt optimization is avoided
3. The method remains general and model-agnostic

## Key Experimental Results

### Main Results

AUROC on reasoning tasks (predicting model correctness):

| Dataset | LLM | Full Logits | Sem. Entropy | QueRE | Note |
|---|---|---|---|---|---|
| GSM8K | GPT-3.5 | 0.5636 | 0.7495 | **0.7748** | Math reasoning |
| GSM8K | GPT-4o-mini | 0.5463 | 0.5546 | **0.7319** | Math reasoning |
| CodeContests | GPT-3.5 | 0.6001 | 0.5346 | **0.6800** | Code generation |
| CodeContests | GPT-4o-mini | 0.5274 | 0.5604 | **0.7924** | Code generation |

Open-ended QA tasks (NQ/SQuAD, LLaMA3 series):

| Dataset | LLM | RepE (white-box) | Post-conf | QueRE |
|---|---|---|---|---|
| NQ | LLaMA3-3B | 0.6639 | 0.8186 | **0.9596** |
| SQuAD | LLaMA3-3B | 0.7033 | 0.9220 | **0.9579** |
| SQuAD | LLaMA3-8B | 0.6993 | 0.7928 | **0.9492** |

### Ablation Study

Adversarial detection accuracy (detecting whether an LLM has been manipulated via system prompts):

| Task | LLM | Pre-conf | Logits | QueRE |
|---|---|---|---|---|
| BoolQ (wrong answers) | GPT-3.5 | 0.5396 | 0.8483 | **0.8668** |
| BoolQ (wrong answers) | GPT-4o-mini | 0.5725 | 0.9033 | **0.9258** |
| CodeContests (hidden bugs) | GPT-3.5 | 0.5061 | 0.9455 | **0.9909** |
| CodeContests (hidden bugs) | GPT-4o-mini | 0.5546 | 0.8848 | **1.0000** |

Cross-domain transferability (OOD generalization):

| Transfer Setting | Self-Consis. | Sem. Entropy | QueRE |
|---|---|---|---|
| SQuAD→NQ (dataset transfer) | 0.8328 | 0.6661 | **0.8964** |
| NQ→SQuAD (dataset transfer) | 0.7532 | 0.5013 | **0.7934** |
| 3B→8B (model transfer) | 0.4635 | 0.6328 | **0.8409** |
| 8B→70B (model transfer) | 0.5280 | 0.6658 | **0.8295** |

### Key Findings

1. **Black-box surpasses white-box**: On open-ended QA and reasoning tasks, QueRE frequently outperforms RepE and Full Logits, which require full model access.
2. **Near-perfect adversarial detection**: Accuracy reaches 100% when detecting manipulation of GPT-4o-mini on CodeContests.
3. **Strong OOD transfer**: Cross-dataset and cross-model-scale transferability substantially exceeds all baselines.
4. **Minimal degradation from sampling approximation**: AUROC drops by no more than 2 points when using finite sampling instead of true probabilities.
5. **More questions yield better performance**: Adding follow-up questions consistently improves performance, with diminishing marginal returns.

## Highlights & Insights

1. **Counter-intuitive finding**: Fifty Yes/No token probabilities derived solely from model outputs can predict behavior more accurately than accessing the model's full internal state — demonstrating that LLMs' self-reflective capabilities genuinely encode valuable metacognitive information.
2. **Minimalist yet general methodology**: A linear model combined with parallel question-answering requires no model modification and no multiple answer samples.
3. **Substantial practical applicability**: The approach can be applied to API auditing (verifying whether the claimed model is being served), security monitoring (detecting system prompt injection), and quality assurance (estimating error probability at deployment).
4. **Tighter PAC-Bayes generalization bounds**: The combination of low-dimensional representations and a linear model naturally yields stronger generalization guarantees.

## Limitations & Future Work

1. Follow-up queries introduce additional latency, though this can be mitigated through batching.
2. The method relies on meaningful variation in the LLM's probability distributions when answering follow-up questions, which may not hold for very low-quality models.
3. Although features are grounded in natural language, the method does not prioritize interpretability — treating features as black-box representations rather than explanations.
4. Discrete prompt optimization could further improve representation quality, but must be balanced against overfitting risk.
5. Theoretical analysis assumes that the representations extracted by the LLM are independent of the downstream task data.

## Related Work & Insights

- Compared to uncertainty quantification methods (semantic entropy, self-consistency), QueRE extracts richer, multi-dimensional information.
- The design parallels weak supervision frameworks — each follow-up question acts as a weak predictor.
- The paper contributes positive empirical evidence to the ongoing debate over whether LLMs can reliably assess their own outputs.
- The method provides a practical monitoring tool for trustworthy deployment of LLMs within autonomous agent frameworks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Novel and concise; the finding that black-box access outperforms white-box is surprising)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 models, 9 datasets, 3 application scenarios, comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear exposition, logically structured experimental design, well-calibrated theoretical support)
- Value: ⭐⭐⭐⭐⭐ (High practical value; offers an elegant solution for black-box LLM monitoring)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs](toward_engineering_agi_benchmarking_the_engineering_design_capabilities_of_llms.md)
- [\[NeurIPS 2025\] Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs](zero-shot_embedding_drift_detection_a_lightweight_defense_against_prompt_injecti.md)
- [\[AAAI 2026\] To Align or Not to Align: Strategic Multimodal Representation Alignment for Optimal Performance](../../AAAI2026/robotics/to_align_or_not_to_align_strategic_multimodal_representation_alignment_for_optim.md)
- [\[AAAI 2026\] Towards Reinforcement Learning from Neural Feedback: Mapping fNIRS Signals to Agent Performance](../../AAAI2026/robotics/towards_reinforcement_learning_from_neural_feedback_mapping_.md)
- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](../../ICLR2026/robotics/tracing_and_reversing_edits_in_llms.md)

</div>

<!-- RELATED:END -->
