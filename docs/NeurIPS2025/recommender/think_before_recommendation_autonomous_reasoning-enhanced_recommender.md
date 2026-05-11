---
title: >-
  [Paper Note] Think before Recommendation: Autonomous Reasoning-enhanced Recommender
description: >-
  [NeurIPS 2025][Recommender Systems][LLM] This paper proposes RecZero (a pure RL paradigm) and RecOne (a hybrid SFT+RL paradigm), abandoning conventional teacher-student distillation. Both approaches leverage GRPO-based r…
tags:
  - "NeurIPS 2025"
  - "Recommender Systems"
  - "LLM"
  - "Recommender System"
  - "Reinforcement Learning"
  - "GRPO"
  - "Rating Prediction"
  - "Chain-of-Thought"
date: 2026-05-08
content_hash: cf9657eae0cc85a8
---

# Think before Recommendation: Autonomous Reasoning-enhanced Recommender

**Conference**: NeurIPS 2025
**arXiv**: [2510.23077](https://arxiv.org/abs/2510.23077)
**Code**: [https://github.com/AkaliKong/RecZero](https://github.com/AkaliKong/RecZero)
**Area**: Recommender Systems / LLM Reasoning
**Keywords**: LLM, Recommender System, Reinforcement Learning, GRPO, Rating Prediction, Chain-of-Thought

## TL;DR
This paper proposes RecZero (a pure RL paradigm) and RecOne (a hybrid SFT+RL paradigm), abandoning conventional teacher-student distillation. Both approaches leverage GRPO-based reinforcement learning to train a single LLM to autonomously develop reasoning capabilities for rating prediction. A structured "Think-before-Recommendation" template guides step-by-step reasoning (user analysis → item analysis → matching → rating), achieving significant improvements over existing baselines across four datasets.

## Background & Motivation

**Background**: Enhancing recommender systems with LLM reasoning for rating prediction is a recent research hotspot. The dominant approach is distillation: a powerful teacher LLM (e.g., ChatGPT) generates reasoning chains, and a student model is fine-tuned via SFT to imitate them.

**Limitations of Prior Work**: (a) Teacher models lack domain knowledge in recommendation, causing the generated reasoning chains to be misaligned with rating prediction objectives; (b) Collecting high-quality reasoning data is costly (API calls / human annotation) and yields static datasets, preventing the student from actively optimizing; (c) SFT learns surface-level patterns rather than genuine reasoning ability, resulting in poor generalization to new scenarios.

**Key Challenge**: How can an LLM autonomously acquire recommendation reasoning ability, rather than passively imitating potentially inaccurate teacher reasoning?

**Goal**: Train a single LLM via RL (rather than distillation) to jointly optimize across four steps: recommendation reasoning, user analysis, item analysis, and match evaluation.

**Key Insight**: Inspired by DeepSeek-R1-Zero — pure RL training can elicit emergent reasoning abilities in LLMs without any teacher data.

**Core Idea**: Apply GRPO with rule-based rewards to directly train an LLM to autonomously develop step-by-step reasoning for rating prediction, bypassing teacher-student distillation.

## Method

### Overall Architecture
RecZero follows a pure RL paradigm: given user history and a target item, the LLM generates multiple reasoning trajectories; rule-based rewards compute advantage estimates, and GRPO optimizes the policy. RecOne extends this by first performing a cold-start SFT phase on a small set of high-quality reasoning samples before RL training.

### Key Designs

1. **"Think-before-Recommendation" Prompt Construction**:

    - Function: Designs a structured reasoning template that decomposes rating prediction into four steps.
    - Mechanism:
        - `<analyze user>...</analyze user>`: Extracts user preferences from historical interactions.
        - `<analyze item>...</analyze item>`: Summarizes target item characteristics.
        - `<match>...</match>`: Evaluates user-item compatibility.
        - `<rate>...</rate>`: Generates the final rating.
    - Design Motivation: Explicitly decomposing the reasoning process (chain-of-thought) provides a clear optimization target for RL over the entire reasoning chain.

2. **Rule-based Reward Modeling**:

    - Function: Designs format rewards and accuracy rewards.
    - Mechanism:
        - Format reward $R_{format}$: $+0.5$ if the output follows the specified tag format, $-0.5$ otherwise.
        - Accuracy reward $R_{answer} = 1 - |y - \hat{y}| / \text{max\_error}$: Higher reward for predictions closer to the ground-truth rating.
        - Total reward $R = R_{format} + R_{answer}$.
    - Design Motivation: Simple rules provide effective training signals without requiring an additional reward model.

3. **GRPO Policy Optimization**:

    - Function: Group Relative Policy Optimization, learning from multiple sampled trajectories.
    - Mechanism: For each input, $G$ reasoning trajectories are sampled; group-relative advantages are computed as $\hat{A}_i = (R_i - \text{mean})/\text{std}$, and the policy is optimized with a PPO-clip objective.
    - Design Motivation: Eliminates the need for a separate value network by deriving advantage signals through within-group comparison.

4. **RecOne Cold-Start SFT**:

    - Function: Initializes the model with SFT on a small set of high-quality reasoning samples.
    - Mechanism: DeepSeek-R1 is used to generate reasoning trajectories; correctly predicted ones are used directly, while incorrectly predicted ones are re-generated by the teacher conditioned on the ground-truth answer (rationalized).
    - Design Motivation: Reduces the domain gap between the pretrained LLM and the recommendation domain, accelerating RL convergence.

### Loss & Training
- RecZero: Pure GRPO training with no SFT phase.
- RecOne: SFT on cold-start data followed by GRPO fine-tuning.
- Supports continuous rating prediction (direct decimal output), avoiding the complexity of integer rating decoding via logit weighting.

## Key Experimental Results

### Main Results

| Dataset | Metric | RecZero | RecOne | Reason4Rec (SOTA) | EXP3RT |
|--------|------|---------|--------|-------------------|--------|
| Amazon-book | MAE | 0.623 | **0.601** | 0.712 | 0.695 |
| Amazon-music | MAE | 0.584 | **0.567** | 0.683 | 0.671 |
| Yelp | MAE | 0.721 | **0.698** | 0.803 | 0.792 |
| IMDb | MAE | 0.495 | **0.478** | 0.562 | 0.551 |

### Ablation Study

| Configuration | Key Findings |
|------|---------|
| Without structured reasoning template | Degraded reasoning quality and unstable predictions |
| Format reward only | Outputs follow format but predictions are inaccurate |
| Accuracy reward only | Formatting inconsistency impairs reasoning quality |
| RecOne vs. RecZero | RecOne converges faster with slightly better final performance |
| RecOne w/o RL (SFT only) | Significantly weaker than the RL-augmented version |

### Key Findings
- RecZero with pure RL surpasses all distillation-based methods, validating the superiority of the RL paradigm.
- The cold-start SFT in RecOne accelerates RL convergence, yielding the best overall performance.
- Trajectory analysis shows that reasoning chains autonomously developed by RecZero align more closely with the recommendation task than those generated by teacher models.
- Direct continuous rating prediction is simpler and more effective than logit-weighted integer decoding.

## Highlights & Insights
- **A paradigm shift inspired by DeepSeek-R1-Zero**: The idea of emergent reasoning through pure RL is successfully transferred to the recommendation domain, demonstrating that reasoning ability can be developed without any teacher data. This represents the first successful application of RL-for-reasoning in recommender systems.
- **Elegant structured reasoning template**: The four-step decomposition (user analysis → item analysis → matching → rating) exploits chain-of-thought principles while providing a clear optimization objective for RL.
- **Effective simple rule-based rewards**: Training a separate reward model is unnecessary; MAE differences directly serve as effective reward signals.

## Limitations & Future Work
- **Rating prediction only**: Effectiveness on other recommendation tasks such as CTR prediction and sequential recommendation has not been validated.
- **Computational cost**: RL training requires more samples than SFT (sampling $G$ trajectories per input), resulting in longer GPU training time.
- **Backbone dependency**: Performance may be strongly tied to the base LLM; effectiveness on smaller models has not been thoroughly examined.
- **Future directions**: Extending to CTR/sequential recommendation; exploring online RL for continual learning of evolving user preferences.

## Related Work & Insights
- **vs. Reason4Rec/EXP3RT**: These methods rely on ChatGPT as a teacher for distillation, whereas RecZero requires no teacher at all.
- **vs. DeepSeek-R1-Zero**: This paper transfers the pure RL emergent reasoning paradigm to the recommendation domain.
- **vs. Traditional LLM4Rec**: Conventional methods apply LLMs in a zero-shot/few-shot manner, whereas this work enables LLMs to genuinely learn recommendation reasoning through RL.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to replace distillation with pure RL for reasoning-enhanced recommendation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, diverse baselines, ablation studies, and cost analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and detailed method description.
- Value: ⭐⭐⭐⭐⭐ Opens a new RL-based paradigm for LLM-powered recommender systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] R²ec: Towards Large Recommender Models with Reasoning](r2ec_towards_large_recommender_models_with_reasoning.md)
- [\[NeurIPS 2025\] Radial Neighborhood Smoothing Recommender System](radial_neighborhood_smoothing_recommender_system.md)
- [\[NeurIPS 2025\] Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation](who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)
- [\[AAAI 2026\] Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation](../../AAAI2026/recommender/wavelet_enhanced_adaptive_frequency_filter_for_sequential_re.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)

</div>

<!-- RELATED:END -->
