---
title: >-
  [Paper Note] ConfTuner: Training Large Language Models to Express Their Confidence Verbally
description: >-
  [NeurIPS 2025][LLM Evaluation][Confidence Calibration] ConfTuner proposes a tokenized Brier score loss function (theoretically proven to be a proper scoring rule) that requires only 2000 samples and 4 minutes of LoRA fine-tuning to enable LLMs to output calibrated verbalized confidence (e.g., "I am 80% confident"), reducing ECE by up to 60.9% and supporting downstream applications like self-correction and model cascading.
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "Confidence Calibration"
  - "Verbalized Confidence"
  - "Brier Score"
  - "Proper Scoring Rule"
  - "LLM Overconfidence"
date: 2026-05-08
content_hash: 76e4804bdd7d2d74
---

# ConfTuner: Training Large Language Models to Express Their Confidence Verbally

**Conference**: NeurIPS 2025  
**arXiv**: [2508.18847](https://arxiv.org/abs/2508.18847)  
**Code**: [GitHub](https://github.com/liushiliushi/ConfTuner)  
**Area**: LLM Evaluation  
**Keywords**: Confidence Calibration, Verbalized Confidence, Brier Score, Proper Scoring Rule, LLM Overconfidence

## TL;DR
ConfTuner proposes a tokenized Brier score loss function (theoretically proven to be a proper scoring rule) that requires only 2000 samples and 4 minutes of LoRA fine-tuning to enable LLMs to output calibrated verbalized confidence (e.g., "I am 80% confident"), reducing ECE by up to 60.9% and supporting downstream applications like self-correction and model cascading.

## Background & Motivation
**Background**: Reliable deployment of LLMs in high-risk domains (medicine, law, science) requires knowing "how certain the model is." Existing approaches include logit-based calibration (which is inapplicable to API models) and verbalized confidence (prompting the model to state its confidence).

**Limitations of Prior Work**: LLMs suffer from severe overconfidence—frequently asserting "I am 100% confident" even when the answer is incorrect. Methods like SaySelf require large-scale data (9 million samples), while LACIE requires 100k samples and long training times. A key theoretical question remains: what kind of loss function can guarantee confidence calibration?

**Key Challenge**: Verbalized confidence consists of discrete tokens (e.g., "80%" is a token rather than a numeric probability), while the classical Brier score is defined on continuous probabilities. How can proper scoring rule theory be extended to the token space?

**Goal**: (1) Design a theoretically-guaranteed confidence calibration loss; (2) achieve calibration at an extremely low cost (2000 samples / 4 minutes).

**Key Insight**: Generalize the Brier score from scoring probability values to scoring "probability distributions over confidence tokens"—if the model assigns more probability mass to the "80%" token, and the true accuracy is indeed close to 80%, the loss is minimized.

**Core Idea**: Tokenized Brier score = a proper scoring rule over confidence tokens, forcing the model to place the highest probability on the token closest to the true accuracy.

## Method

### Overall Architecture
Two stages: (1) Extract the logit distribution $\mathbf{q}$ of confidence tokens from the SFT model; (2) fine-tune the model using LoRA with the tokenized Brier score $\ell(\mathbf{q}, y) = \sum_i q_i(y - i/N)^2$ ($y$ is the correctness indicator).

### Key Designs

1. **Tokenized Brier Score (Core Contribution)**:

    - Function: Defines a loss over the softmax distribution $\mathbf{q}$ on the confidence token set $\mathcal{T}_N = \{0\%, 10\%, ..., 100\%\}$
    - Formula: $\ell(\mathbf{q}, y) = \sum_{i=0}^{N} q_i (y - i/N)^2$
    - Theoretical guarantee (Theorem 1): This is a proper scoring rule—the optimal strategy for the model to minimize the loss is to concentrate probability on the token closest to the true conditional accuracy $\eta(x)$
    - Design Motivation: The classical Brier score evaluates a single probability value, but LLMs output a distribution over tokens—hence the need for generalization

2. **Minimalist LoRA Fine-Tuning**:

    - Function: Fine-tunes only the query/value projections with rank-8 LoRA
    - Needs only 2000 samples and 4 minutes of training (single GPU)
    - Comparison: SaySelf requires 9M samples/120 minutes, LACIE requires 100k/26 minutes
    - Design Motivation: Preserve the model's original capabilities with minimal modifications

3. **Downstream Applications (Self-Correction + Cascading)**:

    - Self-Correction: Refusing to answer or rethinking when confidence is low $\rightarrow$ 3-10% accuracy improvement
    - Model Cascading: Routing to GPT-4o when confidence is low $\rightarrow$ +9.3% on HotpotQA

### Loss & Training
Tokenized Brier score + LoRA. LLaMA uses $\mathcal{T}_{100}$ (0-100%), Qwen/Ministral use $\mathcal{T}_9$ (0-9 discrete levels).

## Key Experimental Results

### Main Results (Average ECE↓ / AUROC↑ on 5 Reasoning Benchmarks)

| Model | ECE (Base) | ECE (ConfTuner) | Gain | AUROC (Base) | AUROC (ConfTuner) |
|------|-----------|----------------|------|-------------|------------------|
| LLaMA | 0.2768 | **0.1082** | -60.9% | 0.5923 | **0.6740** |
| Qwen | 0.3781 | **0.2872** | -23.9% | 0.6155 | **0.6861** |
| Ministral | 0.4393 | **0.1884** | -57.1% | 0.5216 | **0.6810** |

### Ablation Study

| Configuration | Key Findings | Description |
|------|---------|------|
| 2000 vs 5000 vs 10000 samples | 2000 is sufficient | Extremely low data requirement |
| 4 min vs SaySelf 120 min | 30x speedup | Highly computationally efficient |
| Verbalized variants (high/medium/low) | Effectively generalizes | Not limited to numerical confidence |
| Implicit confidence (expressed in CoT) | Also effective | Non-explicit "X%" can also be calibrated |
| Black-box models (GPT-4o) | Also improves | Via prompting framework |
| Self-correction | +3-10% accuracy | Practical downstream application |
| Model cascading | +9.3% (HotpotQA) | Confidence used for routing |

### Key Findings
- An ECE reduction of -60.9% is the most prominent figure—demonstrating that LLMs' confidence transitions from near-random noise to a meaningful signal.
- 2000 samples + 4 minutes defines the "lower bound of cost"—making it difficult to conceive a lower-cost calibration scheme.
- Cascading applications demonstrate the practical utility of confidence—knowing when to query a stronger model.

## Highlights & Insights
- **The Power of Theory**: The proper scoring rule guarantees that the model cannot "cheat" (e.g., predicting 50% for all answers) to lower the loss—the sole optimal strategy is honest calibration. This offers a fundamental advantage over heuristically designed loss functions.
- **"2000 samples, 4 minutes"**: The extremely low cost means any team utilizing LLMs can perform confidence calibration, democratizing uncertainty estimation.
- **From Calibration to Decision-making**: Calibration is not the ultimate end; self-correction and cascading are. The paper presents a complete "calibration $\rightarrow$ utilization" pipeline.

## Limitations & Future Work
- Restricted to a fixed confidence token set; flexible natural language expressions (e.g., "I am fairly sure but not entirely certain") are not handled.
- A gap remains between theoretical guarantees and practice—data quality and optimization dynamics affect actual calibration.
- Validated only on reasoning tasks; calibration for tasks like creative writing or translation remains unexplored.
- The proper scoring rule assumes independent samples—dependencies in multi-turn dialogues are not considered.

## Related Work & Insights
- **vs SaySelf (Xu et al., 2024)**: SaySelf iteratively improves confidence via self-training but requires 9 million samples; ConfTuner is theory-driven and requires only 2000 samples.
- **vs LACIE (Band et al., 2024)**: LACIE trains an auxiliary model to predict confidence; ConfTuner directly calibrates the token distribution of the model itself.
- **vs Temperature Scaling**: TS calibrates logit probabilities but does not change the verbalized output; ConfTuner directly calibrates verbalized expressions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical generalization of the Tokenized Brier score is elegant and practically meaningful.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 models × 5 benchmarks × downstream applications × efficiency comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Closely integrates theory with experimental narratives.
- Value: ⭐⭐⭐⭐⭐ The 2000-sample + 4-minute calibration scheme has direct practical utility for LLM deployment.
\n
- **vs CalibrateBeforeUse**: CBU performs temperature scaling at inference time; ConfTuner directly calibrates the semantics of the output tokens, making it more suitable for API scenarios.
- **vs Ensemble Methods (Self-Consistency)**: Estimating confidence by sampling multiple times for consistency is expensive (requiring $N$ inferences); ConfTuner internalizes this knowledge into a single-pass output.
- **Inspiration for Other Modalities**: The tokenized proper scoring rule framework can be generalized to visual confidence in VLMs, reliability in speech recognition, etc.
- **Implications for RLHF**: ConfTuner's calibration can improve reward models in RLHF—calibrated confidence serves as a better training signal.
- **Value for Agent Systems**: Autonomous agents need to know when to seek human help—calibrated confidence provides a reliable basis for such decisions.
- **Relationship to Bayesian Methods**: ConfTuner's calibration can be viewed as mapping LLM outputs to approximate Bayesian posterior probabilities.
- **Multilingual Extensions**: Natural expressions of confidence vary across languages; cross-lingual calibration remains an open problem.
- **Long-text Scenarios**: When a response contains multiple claims, individual confidence for each claim is more valuable than a global confidence score.
- **vs CalibrateBeforeUse**: CBU performs temperature scaling at inference time; ConfTuner directly calibrates the semantics of the output tokens, making it more suitable for API scenarios.
- **vs Ensemble Methods (Self-Consistency)**: Estimating confidence by sampling multiple times for consistency is expensive (requiring $N$ inferences); ConfTuner internalizes this knowledge into a single-pass output.
- **Inspiration for Other Modalities**: The tokenized proper scoring rule framework can be generalized to visual confidence in VLMs, reliability in speech recognition, etc.
- **Implications for RLHF**: ConfTuner's calibration can improve reward models in RLHF—calibrated confidence serves as a better training signal.
- **Value for Agent Systems**: Autonomous agents need to know when to seek human help—calibrated confidence provides a reliable basis for such decisions.
- **Relationship to Bayesian Methods**: ConfTuner's calibration can be viewed as mapping LLM outputs to approximate Bayesian posterior probabilities.
- **Multilingual Extensions**: Natural expressions of confidence vary across languages; cross-lingual calibration remains an open problem.
- **Long-text Scenarios**: When a response contains multiple claims, individual confidence for each claim is more valuable than a global confidence score.
- **vs CalibrateBeforeUse**: CBU performs temperature scaling at inference time; ConfTuner directly calibrates the semantics of the output tokens, making it more suitable for API scenarios.
- **vs Ensemble Methods (Self-Consistency)**: Estimating confidence by sampling multiple times for consistency is expensive (requiring $N$ inferences); ConfTuner internalizes this knowledge into a single-pass output.
- **Inspiration for Other Modalities**: The tokenized proper scoring rule framework can be generalized to visual confidence in VLMs, reliability in speech recognition, etc.
- **Implications for RLHF**: ConfTuner's calibration can improve reward models in RLHF—calibrated confidence serves as a better training signal.
- **Value for Agent Systems**: Autonomous agents need to know when to seek human help—calibrated confidence provides a reliable basis for such decisions.
- **Relationship to Bayesian Methods**: ConfTuner's calibration can be viewed as mapping LLM outputs to approximate Bayesian posterior probabilities.
- **Multilingual Extensions**: Natural expressions of confidence vary across languages; cross-lingual calibration remains an open problem.
- **Long-text Scenarios**: When a response contains multiple claims, individual confidence for each claim is more valuable than a global confidence score.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models](../../ICLR2026/llm_evaluation/rewarding_doubt_a_reinforcement_learning_approach_to_calibrated_confidence_expre.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [\[NeurIPS 2025\] Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator](your_pre-trained_llm_is_secretly_an_unsupervised_confidence_calibrator.md)
- [\[NeurIPS 2025\] Can Large Language Models Master Complex Card Games?](can_large_language_models_master_complex_card_games.md)
- [\[NeurIPS 2025\] LTD-Bench: Evaluating Large Language Models by Letting Them Draw](ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)

</div>

<!-- RELATED:END -->
