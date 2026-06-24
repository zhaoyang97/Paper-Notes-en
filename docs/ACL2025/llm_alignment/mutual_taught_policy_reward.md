---
title: >-
  [Paper Note] Mutual-Taught for Co-adapting Policy and Reward Models
description: >-
  [ACL 2025][LLM Alignment][Reward tampering] Mutual-Taught proposes a self-training framework based on the EM algorithm to iteratively update both the policy model (PM) and the reward model (RM) during the preference optimization process: the E-step optimizes the PM using the current RM, while the M-step updates the RM using pseudo-preference pairs constructed from the output differences of the PM before and after the update. This resolves the reward hacking issue caused by di…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Reward tampering"
  - "distribution shift"
  - "EM algorithm"
  - "iterative DPO"
  - "self-training"
date: 2026-05-08
content_hash: 17172a525e407cd8
---

# Mutual-Taught for Co-adapting Policy and Reward Models

**Conference**: ACL 2025  
**arXiv**: [2506.06292](https://arxiv.org/abs/2506.06292)  
**Code**: [https://github.com/Stycoo/Mutual-Taught](https://github.com/Stycoo/Mutual-Taught)  
**Area**: Alignment RLHF  
**Keywords**: Reward tampering, distribution shift, EM algorithm, iterative DPO, self-training

## TL;DR
Mutual-Taught proposes a self-training framework based on the EM algorithm to iteratively update both the policy model (PM) and the reward model (RM) during the preference optimization process: the E-step optimizes the PM using the current RM, while the M-step updates the RM using pseudo-preference pairs constructed from the output differences of the PM before and after the update. This resolves the reward hacking issue caused by distribution shift, achieving a 54.1% LC win rate on AlpacaEval-2 with an 8B model.

## Background & Motivation
**Background**: In the alignment process of RLHF/DPO, as the policy model is iteratively optimized, its output distribution shifts away from the data distribution used to train the reward model.

**Limitations of Prior Work**:
   - Distribution shift leads to reward hacking—models learn to achieve high RM scores without generating outputs that truly reflect human preferences.
   - Continuously collecting human annotations to update the RM is costly and unscalable.
   - LLM-as-Judge (e.g., Self-Rewarding) requires strong base models or pre-trained evaluation capabilities, making it inapplicable to weaker models.
   - Existing iterative DPO methods assume the RM is a fixed oracle, neglecting the degradation of the RM itself.

**Key Challenge**: The PM constantly changes during optimization, but the RM remains static, leading to increasingly inaccurate evaluation signals.

**Goal**: To simultaneously improve both the PM and the RM without requiring additional human annotation.

**Key Insight**: Model the co-optimization of PM and RM as an EM problem, where the latent variable is the optimal preference distribution.

**Core Idea**: The E-step optimizes the PM using the RM, while the M-step optimizes the RM using the comparison of PM outputs before and after the update, establishing a virtuous cycle.

## Method

### Overall Architecture
Mutual-Taught iteratively executes two steps: (1) **E-step**: Sample responses from the PM $\pi_{t-1}$ and label preferences using the current RM $r_{t-1}$, then update the PM using DPO to obtain $\pi_t$; (2) **M-step**: For the same prompt, compare the outputs of $\pi_{t-1}$ and $\pi_t$. Treat the output of $\pi_t$ as preferred (since it has been optimized by the RM) and construct pseudo-preference pairs to update the RM, obtaining $r_t$.

### Key Designs

1. **E-step: Policy Model Update**:

    - Function: Update the PM using iterative DPO, utilizing the PM from the previous round as the reference model.
    - Mechanism: Sample multiple completions from $\pi_{t-1}$, label them as chosen/rejected using $r_{t-1}$, and perform DPO training to obtain $\pi_t$.
    - Difference from standard iterative DPO: Standard iterative DPO assumes a fixed RM, whereas Mutual-Taught updates the RM in each round.
    - Stabilization strategy: Select the optimal checkpoint using validation set model selection, only choosing checkpoints that show improvement over the previous round to guarantee monotonic progress.

2. **M-step: Reward Model Update**:

    - Function: Construct pseudo-preference pairs from the output differences of the PM before and after the update to train the RM.
    - Mechanism: For a prompt $x$, sample $y_t \sim \pi_t$ and $y_{t-1} \sim \pi_{t-1}$. Since $\pi_t$ is optimized via $r_{t-1}$, $y_t$ is generally preferred over $y_{t-1}$. Thus, construct $(y_t \succ y_{t-1})$ pseudo-preference pairs to update the RM using the Bradley-Terry loss.
    - Design Motivation: These pseudo-preference pairs naturally originate from the evolving distribution of the PM—requiring no external feedback! The RM automatically adapts to the shifting distribution of the PM by learning to distinguish between the outputs of the new and old PM.
    - Data filtering: Calculate the reward margin $\Delta r(x) = r_{t-1}(y_t; x) - r_{t-1}(y_{t-1}; x)$, and remove noisy samples where $\Delta r \leq -\epsilon_t$ ($\epsilon_t$ is a variance-adaptive threshold).

3. **Two-Stage Stabilization**:

    - E-step stabilization: Validation set model selection, stopping iterations when win rate < $\tau$.
    - M-step stabilization: Variance-based adaptive data filtering to eliminate highly confident noise while retaining slight noise as regularization.

### Loss & Training
- Base PM: Llama-3-8B-Instruct; Base RM: FsfairX-Llama3-RM-v0.1
- Training data: UltraFeedback (~60K prompts), divided into three subsets rotated for PM/RM training.
- Each full training round consists of two PM updates and one RM update.

## Key Experimental Results

### Main Results

**Policy Model Performance**:

| Method | AlpacaEval-2 LC WR | Arena-Hard WR | Description |
|------|:------------------:|:-------------:|------|
| Llama-3-8B-Instruct | 23.1 | 20.6 | Baseline |
| DPO (offline) | 44.3 | 33.1 | Offline DPO |
| Meta-Rewarding Iter3 | 37.5 (+14.4) | 27.9 (+7.3) | LLM-as-Judge |
| SPPO Iter3 | 46.4 (+23.3) | 33.6 (+13.0) | Self-Play |
| Iterative DPO Iter3 | 47.2 (+24.1) | 38.5 (+17.9) | Fixed RM |
| **Mutual-Taught** | **54.1 (+31.0)** | **38.4 (+17.8)** | PM+RM Co-adaptation |

**Reward Model Performance (RewardBench)**:

| Model | Chat | Chat-Hard | Safety | Reasoning | Avg. |
|------|:----:|:---------:|:------:|:---------:|:----:|
| FsfairX-RM (Initial) | 96.9 | 74.1 | 86.8 | 97.1 | 88.7 |
| **FsfairX-RM-MT** | **97.5** | **78.4** | **87.6** | **97.8** | **90.3** |
| GPT-4o-2024-08-06 | 96.6 | 76.1 | 88.1 | 86.6 | 86.7 |

The RM is improved to a level comparable to GPT-4o, even surpassing it in certain dimensions!

### Ablation Study

| Configuration | AlpacaEval-2 LC WR | Description |
|------|:------------------:|------|
| Full Mutual-Taught | **54.1** | Full method |
| w/o RM update (Fixed RM) | 47.2 | Degrades to iterative DPO |
| w/o model selection | 50.3 | PM might overfit |
| w/o data filtering | 51.8 | Noisy pseudo-labels affect RM |

### Key Findings
- RM updates are a crucial contribution—removing RM updates degrades the method to iterative DPO ($54.1 \rightarrow 47.2$), demonstrating that resolving distribution shift is vital.
- Subsequent PM training is more effective after RM updates—confirming the virtuous cycle of "better RM $\rightarrow$ better PM $\rightarrow$ better RM".
- The quality of pseudo-preference pairs is high enough—the output of $\pi_t$ is indeed generally superior to $\pi_{t-1}$, validating the hypothesis that "continuous optimization generates natural preference pairs".
- In the data filtering strategy, retaining slightly negative samples ($-\epsilon_t < \Delta r < 0$) as regularization is helpful.

## Highlights & Insights
- **PM-RM co-evolution under the EM framework**: Formulates the distribution shift issue in RLHF as a missing data problem (where the optimal preference distribution is a latent variable) and approximates it iteratively using EM—an elegant theoretical perspective.
- **Zero-external-annotation RM updates**: Uses the "before-and-after contrast" from the PM's own evolution as the training signal, making it entirely self-contained without needing LLM-as-Judge capabilities or extra human annotations.
- **Slight noise as regularization**: Retaining slightly negative pseudo-preference pairs prevents the RM from overfitting to the PM's distribution—a noteworthy finding.

## Limitations & Future Work
- **Risk of mutual degradation between RM and PM**: If the initial RM is of poor quality, the generated pseudo-preference pairs will also be poor, which could lead to a vicious cycle. The framework assumes the initial RM is "good enough".
- **Computational cost**: Each iteration round requires PM training + RM training + intensive inference sampling, making the cost about 3-4 times that of standard DPO.
- **Only evaluated on 8B models**: The effectiveness and required number of iteration rounds for larger models remain unknown.
- **The "self-cycle" assumption of pseudo-preferences**: Assumes that the PM output optimized by the RM is always better—but reward hacking might lead to PM outputs that "look better but are actually worse".

## Related Work & Insights
- **vs Iterative DPO**: Iterative DPO only updates the PM and not the RM, while Mutual-Taught updates both. The additional +7% LC win rate demonstrates the value of RM updating.
- **vs Self-Rewarding/Meta-Rewarding**: These methods use the PM itself as a judge, which requires strong judgmental capabilities. Mutual-Taught uses an independent RM, which is more reliable.
- **vs ReST^EM**: Though both use the EM framework, ReST^EM only updates the PM, whereas Mutual-Taught updates both.
- The M-step could be explored to extend to multiple RMs (ensemble) to improve the reliability of pseudo-preference pairs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The EM framework for PM-RM co-evolution is a completely new paradigm, and the pseudo-preference pair construction method is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual-direction evaluation for both PM and RM, comparisons with multiple strong baselines, and a complete ablation study.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivation of the EM framework with intuitive diagrams.
- Value: ⭐⭐⭐⭐⭐ Resolves a core issue in RLHF (distribution shift) with significant performance gains and no extra annotation requirements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective](../../ICML2025/llm_alignment/can_rlhf_be_more_efficient_with_imperfect_reward_models_a_policy_coverage_perspe.md)
- [\[ACL 2025\] Cheems: A Practical Guidance for Building and Evaluating Chinese Reward Models from Scratch](cheems_chinese_reward_models.md)
- [\[ACL 2025\] PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models](prmbench_a_fine-grained_and_challenging_benchmark_for_process-level_reward_model.md)
- [\[ACL 2025\] SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs](synthesizeme_persona_prompts.md)
- [\[ACL 2025\] AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models](amopo_adaptive_multi-objective_preference_optimization_without_reward_models_and.md)

</div>

<!-- RELATED:END -->
