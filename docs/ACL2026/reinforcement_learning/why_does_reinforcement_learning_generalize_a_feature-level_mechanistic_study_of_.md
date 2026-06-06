---
title: >-
  [Paper Note] Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models
description: >-
  [ACL2026][Reinforcement Learning][RL Generalization] Through controlled SFT/RL post-training comparisons and Sparse Crosscoder feature alignment, this paper finds that SFT rapidly forms numerous specialized features…
tags:
  - "ACL2026"
  - "Reinforcement Learning"
  - "RL Generalization"
  - "SFT Forgetfulness"
  - "Sparse Crosscoder"
  - "Feature Intervention"
  - "LLM Post-training"
date: 2026-05-08
content_hash: 9323465e2b7f6582
---

# Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models

**Conference**: ACL2026  
**arXiv**: [2604.25011](https://arxiv.org/abs/2604.25011)  
**Code**: https://github.com/danshi777/RL-generalization  
**Area**: RL Post-training / Mechanistic Interpretability  
**Keywords**: RL Generalization, SFT Forgetfulness, Sparse Crosscoder, Feature Intervention, LLM Post-training

## TL;DR
Through controlled SFT/RL post-training comparisons and Sparse Crosscoder feature alignment, this paper finds that SFT rapidly forms numerous specialized features, while RL preserves base representations while progressively enhancing a small set of cross-task generalization features. Zeroing these features significantly impairs RL generalization, while amplifying them improves the performance of base models.

## Background & Motivation
**Background**: In LLM post-training, SFT and RL are commonly used to enhance reasoning capabilities. SFT typically imitates complete CoT trajectories from strong teachers, whereas RL provides rewards based solely on the correctness of the final answer. Empirically, RL often transfers well to tasks beyond math (e.g., commonsense or scientific QA), while SFT sometimes sacrifices general capabilities.

**Limitations of Prior Work**: Existing research mostly remains at the behavioral level, observing better RL generalization and more evident SFT forgetfulness without explaining the internal representation dynamics. Analyzing output distributions or global hidden state shifts only provides coarse explanations and fails to identify which specific internal features control generalization.

**Key Challenge**: RL has weaker supervision signals (outcomes only), yet it better preserves and extends general capabilities. SFT has denser supervision but tends to push models toward the training distribution and teacher style. This suggests that "supervision intensity" does not equate to "generalization mechanism"; the key likely lies in how post-training rewrites the existing feature space.

**Goal**: The authors aim to answer three questions while controlling for data, base model, and training scale: What model-specific features are introduced by SFT and RL? How do these features form and stabilize? Does a set of locatable and intervenable cross-task generalization features exist in RL?

**Key Insight**: The paper utilizes Sparse Crosscoders to map the residual stream activations of base, SFT, and RL models into a single sparse feature space. This allows for feature-by-feature analysis to determine whether a feature belongs to the base, SFT, or RL model, rather than merely comparing overall vector distances.

**Core Idea**: By using cross-model sparse feature alignment and causal intervention, the study transforms the question of "why RL generalizes" from a behavioral observation into a mechanistic study of whether RL enhances a compact, task-agnostic, and controllable set of generalization features.

## Method
The methodology consists of two layers: constructing fair control experiments for post-training (varying only the training paradigm) and using Sparse Crosscoders to align internal activations, followed by feature attribution, training dynamics analysis, and intervention experiments.

### Overall Architecture
The input consists of two post-trained versions of the same base model (SFT and RL) trained on identical math data. Experiments are conducted on Qwen3-4B-Base and Qwen2.5-7B, with trends validated on Llama3.1-8B-Instruct.

The process involves training two pairwise Sparse Crosscoders: Base-SFT and Base-RL to measure specific features introduced relative to the base. Subsequently, a three-model Sparse Crosscoder aligns Base, SFT, and RL in one space, using MAS to measure feature attribution. Finally, the authors focus on "generalization-critical samples" (Base fails, RL succeeds) to identify features significantly activated in RL and verify their causal roles via zeroing and amplification.

### Key Designs
1.  **Dual-Model Sparse Crosscoder and NRN Feature Specificity Metric**:
    - **Function**: Encodes activations from a base model and a post-trained model into a shared sparse space to measure feature bias.
    - **Mechanism**: The Crosscoder reconstructs activations from both models; each dimension in the sparse vector represents a shared or specific feature. Feature attribution is measured via NRN: $NRN=||W^{T}_{dec,k}||_1/(||W^{O}_{dec,k}||_1+||W^{T}_{dec,k}||_1)$. Values near 1 indicate post-training specificity, near 0 indicate base specificity, and near 0.5 indicate shared features.
    - **Design Motivation**: NRN reveals the intensity of feature space rewriting. SFT shows a heavy right tail (highly specific features), while RL has fewer extreme features, indicating it largely preserves and slightly rearranges base representations.

2.  **Three-Model Crosscoder and MAS Unified Attribution**:
    - **Function**: Enables direct comparison of feature indices across Base, SFT, and RL in a single sparse basis.
    - **Mechanism**: A three-model Crosscoder encodes all three models' activations and uses three sets of decoders. MAS normalizes the three decoder norms into $MAS_O$, $MAS_S$, and $MAS_R$, summing to 1. The maximum value denotes the dominant training paradigm.
    - **Design Motivation**: Aligning all models in one space allows for a fair comparison of whether a feature is unique to SFT, unique to RL, or shared, which pairwise Crosscoders cannot achieve.

3.  **Feature Localization and Causal Intervention on Generalization-Critical Samples**:
    - **Function**: Transitions from descriptive differences to causal mechanisms to identify features supporting RL's cross-task generalization.
    - **Mechanism**: For non-math evaluation tasks, samples where the Base model fails but RL succeeds are selected. For these, the average activation difference between RL and Base is calculated: $Score_k=E[f_k^{RL}(x)-f_k^{Base}(x)]$. Task-agnostic features are identified by the intersection of high-scoring features across tasks.
    - **Design Motivation**: Generalization features should reflect functional differences on samples where the model actually improves. Intervening on these features (zeroing in RL or amplifying in Base) proves they are causal control points rather than correlational noise.

### Loss & Training
Both SFT and RL start from the same base and use the same 47K high-quality math dataset. SFT uses cross-entropy on complete CoT trajectories from Qwen3-32B-Instruct. RL uses GRPO, rewarding only the final answer correctness without exposing intermediate tracks.

RL uses the verl framework with a batch size of 128, learning rate $1e-6$, 8 rollouts per prompt, 16K max length, for 1 epoch. SFT uses LLaMA-Factory, batch size 128, learning rate $5e-5$, also for 1 epoch. Crosscoders are trained with 32,768 sparse features on 400M tokens (mixed OpenThoughts-114k and RedPajama), reconstruct middle-layer residual streams, batch size 1024, learning rate $1e-4$, and a sparsity coefficient of 2.

## Key Experimental Results

### Main Results
The authors compare Base, SFT, and RL on math and non-math tasks. RL significantly outperforms SFT on non-training-domain tasks such as CommonsenseQA, SciQ, and ARC-Challenge, whereas SFT often experiences performance degradation (forgetting) on general tasks despite math gains.

| Model | MATH500 | AIME24 | AIME25 | OpenBookQA | CommonsenseQA | SciQ | ARC-Challenge |
|-------|---------|--------|--------|------------|----------------|------|---------------|
| Qwen3-4B-Base | 26.0 | 13.3 | 0.0 | 23.6 | 20.1 | 78.5 | 36.0 |
| Qwen3-4B-SFT | 68.4 | 13.3 | 13.3 | 25.8 | 19.6 | 51.8 | 34.4 |
| Qwen3-4B-RL | 77.0 | 26.7 | 20.0 | 27.2 | 50.5 | 89.5 | 39.3 |
| RL - SFT | +8.6 | +13.4 | +6.7 | +1.4 | +31.0 | +37.7 | +4.9 |
| Qwen2.5-7B | 40.0 | 10.0 | 3.3 | 28.4 | 77.6 | 86.9 | 41.9 |
| Qwen2.5-7B-SFT | 69.2 | 13.3 | 10.0 | 26.4 | 30.1 | 79.4 | 37.0 |
| Qwen2.5-7B-RL | 71.4 | 20.0 | 13.3 | 32.8 | 76.1 | 90.7 | 42.5 |
| RL - SFT | +2.2 | +6.7 | +3.3 | +6.4 | +46.0 | +11.3 | +5.5 |

### Ablation Study
Feature interventions provide the strongest evidence. The authors identify overlapping generalization features (50 for Qwen3-4B; 16 for Qwen2.5-7B) with over 80% overlap across different tasks.

| Intervention | Model | OpenBookQA | CommonsenseQA | HeadQA | SciQ | ARC-Challenge | Description |
|--------------|-------|------------|----------------|--------|------|---------------|-------------|
| Zero-out | Qwen3-4B-RL | -46.2 | -43.9 | -21.2 | -14.0 | -33.3 | RL loses key successes |
| Zero-out | Qwen2.5-7B-RL| -21.9 | -24.4 | -23.8 | -44.4 | -20.0 | Confirms necessity |
| Amplify | Qwen3-4B-Base| +36.3 | +36.0 | +21.2 | +38.0 | +33.3 | Base improves significantly |
| Amplify | Qwen2.5-7B | +12.5 | +24.4 | +14.3 | +55.6 | +40.0 | Injected knowledge/mechanism |

Testing on LogiQA and PIQA (not used for feature selection): zeroing features in Qwen3-4B-RL/Qwen2.5-7B-RL reduced scores by 24.5-11.8, while amplification in Base models yielded improvements of 23.3-32.9.

### Key Findings
- SFT features stabilize earlier: SFT's top-50 specific features show high overlap across checkpoints (every 1/5 epoch), indicating a fixed "teacher trajectory." RL's top features have low overlap between checkpoints, suggesting continuous exploration.
- RL's rewriting is "restrained": NRN and MAS distributions show SFT introduces many strong specific features, while RL primarily re-weights base representations.
- Generalization is not scattered: Generalization features are highly overlapping across diverse tasks, supporting the theory that a compact set of task-agnostic features controls generalization.

## Highlights & Insights
- The most valuable contribution is converting "RL generalization" into an intervenable mechanism. The causal chain established via zeroing and amplification is more explanatory than simple distance analysis.
- The paper provides a clear post-training picture: SFT carves teacher trajectory templates into the model, while RL tunes result-oriented control features on existing capabilities. This explains why SFT gains task performance at the cost of general ones.
- The observation that "base models do not lack knowledge but lack generalization feature activation" is profound. It suggests training goals should perhaps focus on stabilizing transferable internal features rather than just providing more CoT data.
- The three-model Crosscoder is superior for analyzing multiple training paradigms (e.g., DPO, RLHF) to determine how alignment methods rewrite internal features.

## Limitations & Future Work
- Sparse Crosscoders might not capture non-linearly reconstructible or non-middle-layer mechanisms.
- While causality is proven, the paper does not specify how to stably induce these features during training (e.g., via reward shaping or regularization).
- Experiments focus on math-to-QA generalization; other domains like coding, long-horizon planning, or safety alignment remain unexplored.
- Feature selection thresholds are heuristic. Future work could explore more robust criteria like bootstrap or cross-seed consistency.

## Related Work & Insights
- **vs. Behavioral SFT/RL comparison**: While previous studies report RL's better generalization, this work identifies the specific feature dynamics and intervenable control points.
- **vs. Global representation drift**: Unlike prior works attributing SFT's poor generalization to distribution shifts, this paper pinpoints the appearance and disappearance of specific sparse features.
- **vs. SAE feature interpretation**: SAEs are typically for single models; Sparse Crosscoders allow for cross-model comparisons. The three-model expansion is particularly suited for post-training research.
- **Insight**: If RL generalization stems from a few cross-task features, one could design "feature-preserving SFT" or "generalization-enhanced RL" to boost target performance without sacrificing the base model's integrity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Translates RL generalization into feature-level mechanisms and causal interventions.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Solid controlled training and validation across multiple models and tasks.
- Writing Quality: ⭐⭐⭐⭐☆ Clear logic; graphics support conclusions well, though Crosscoder details are dense.
- Value: ⭐⭐⭐⭐⭐ High impact on understanding RL, SFT forgetting, and designing mechanistic-driven training objectives.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study](scaling_behaviors_of_llm_reinforcement_learning_post-training_an_empirical_study.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](../../ICLR2026/reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ICML 2026\] Can Large Language Models Generalize Procedures Across Representations?](../../ICML2026/reinforcement_learning/can_large_language_models_generalize_procedures_across_representations.md)
- [\[ACL 2026\] A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions](a_survey_of_reinforcement_learning_for_large_language_models_under_data_scarcity.md)
- [\[ICML 2026\] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](../../ICML2026/reinforcement_learning/how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)

</div>

<!-- RELATED:END -->
