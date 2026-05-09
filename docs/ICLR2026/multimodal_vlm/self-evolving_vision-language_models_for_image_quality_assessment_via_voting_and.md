---
title: >-
  [Paper Note] Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking
description: >-
  [ICLR 2026][Multimodal VLM][VLM] This paper proposes EvoQuality, a self-supervised iterative framework that generates pseudo-ranking labels via pairwise majority voting and employs GRPO for self-iterative optimization, enabling VLMs to autonomously improve their image quality perception without any human annotations. The framework achieves a 31.8% PLCC improvement in zero-shot settings and surpasses supervised SOTA on 5 out of 7 IQA benchmarks.
tags:
  - ICLR 2026
  - Multimodal VLM
  - VLM
  - Image Quality Assessment
  - Self-supervised
  - GRPO
  - Voting and Ranking
date: 2026-05-08
content_hash: c65284372b63d73e
---

# Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking

**Conference**: ICLR 2026
**arXiv**: [2509.25787](https://arxiv.org/abs/2509.25787)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: VLM, Image Quality Assessment, Self-supervised, GRPO, Voting and Ranking

## TL;DR

This paper proposes EvoQuality, a self-supervised iterative framework that generates pseudo-ranking labels via pairwise majority voting and employs GRPO for self-iterative optimization, enabling VLMs to autonomously improve their image quality perception without any human annotations. The framework achieves a 31.8% PLCC improvement in zero-shot settings and surpasses supervised SOTA on 5 out of 7 IQA benchmarks.

## Background & Motivation

Image Quality Assessment (IQA) is a classical computer vision task aimed at automatically evaluating the perceptual quality of images. While Vision-Language Models (VLMs) have demonstrated strong capabilities across various visual tasks, applying them to IQA still faces two major challenges:

**High annotation cost**: Conventional VLM post-training methods (e.g., SFT or RLHF) rely on large volumes of human-annotated quality scores, which are extremely expensive to collect and suffer from high subjectivity and low consistency.

**Gap in self-supervised learning for perceptual tasks**: Although self-supervised techniques have been validated for enhancing reasoning capabilities (e.g., mathematical reasoning), their application to perceptual tasks such as quality judgment remains largely unexplored. IQA differs fundamentally from reasoning tasks—it has no single correct answer and instead relies on relative comparisons.

**Limitations of Prior Work**: Most existing VLM-based IQA models require supervised fine-tuning on annotated IQA datasets, resulting in limited generalizability and poor transfer to unseen datasets.

The core motivation of this paper is: **Can VLMs autonomously improve their image quality perception without any human annotations?** The authors observe that while a VLM's single-pass quality judgment may be unreliable, the "wisdom of the crowd" derived from repeated pairwise comparisons can yield reliable quality rankings.

## Method

### Overall Architecture

EvoQuality is a self-supervised iterative optimization framework with the following core pipeline:
1. Perform multiple pairwise quality comparisons between image pairs using the VLM.
2. Establish consensus rankings (pseudo-labels) via majority voting.
3. Convert ranking results into fidelity reward signals.
4. Optimize the VLM using GRPO (Group Relative Policy Optimization).
5. Repeat the above steps iteratively, allowing the VLM to progressively self-evolve.

### Key Designs

1. **Pairwise Majority Voting**:

    - For a set of images, the VLM performs $K$ repeated comparisons for each image pair, answering "which image has better quality?"
    - Majority voting over the $K$ outputs determines the quality preference for each pair.
    - A global quality ranking is obtained by aggregating all pairwise votes through a ranking algorithm (e.g., the Bradley-Terry model or simple counting).
    - **Design Motivation**: Exploits the self-consistency principle—single samples may be unreliable, but the consistent direction across multiple samples reflects the model's genuine tendency.

2. **Fidelity Reward Construction**:

    - The pseudo-rankings produced by voting are converted into reward signals.
    - A positive reward is assigned when the VLM's new output is consistent with the pseudo-ranking; a negative reward is assigned otherwise.
    - The reward magnitude is proportional to voting consistency—pairs with higher voting agreement (i.e., more confident judgments) receive larger reward weights.
    - **Design Motivation**: Highly consistent voting results are more reliable and should be weighted more heavily; low-consistency results may represent difficult samples or noise and should be down-weighted.

3. **GRPO Optimization**:

    - Group Relative Policy Optimization is adopted as the reinforcement learning optimizer.
    - GRPO does not require an additional critic/value network; it directly uses within-group relative rewards to update the policy.
    - In each iteration, multiple responses are sampled from the current VLM, relative advantages are computed based on fidelity rewards, and model parameters are updated accordingly.
    - **Design Motivation**: GRPO is more lightweight than PPO, making it suitable for iterative optimization of large models.

4. **Iterative Self-Evolution Mechanism**:

    - After each optimization round, the updated VLM re-executes pairwise voting to generate new pseudo-labels.
    - The new pseudo-labels are of higher quality (since the VLM has improved), further driving model improvement.
    - This forms a positive feedback loop: better model → more accurate pseudo-labels → more effective training → better model.
    - **Design Motivation**: Inspired by self-training, but using ranking rather than absolute scores to mitigate error accumulation.

### Loss & Training

- The GRPO policy gradient loss is used as the core objective:
  $$L_{GRPO} = -\mathbb{E}\left[\sum_{i} A_i \log \pi_\theta(y_i | x_i)\right]$$
  where $A_i$ denotes the relative advantage based on fidelity rewards.
- A KL divergence constraint is incorporated to prevent the model from deviating excessively from its original capabilities.
- Multi-round iterative training is performed, with newly generated pseudo-labels used in each round.

## Key Experimental Results

### Main Results

EvoQuality is evaluated on 7 mainstream IQA benchmarks:

| Metric | Ours (EvoQuality) | Base VLM (Zero-shot) | Gain |
|--------|-------------------|----------------------|------|
| PLCC (average) | Significant improvement | Baseline | +31.8% |
| Surpassing supervised SOTA | 5/7 benchmarks | — | — |

Key findings:
- EvoQuality surpasses supervised SOTA VLM-based IQA models on 5 benchmarks: LIVE, CSIQ, TID2013, KADID-10K, and SPAQ.
- Performance approaches supervised SOTA on KonIQ-10K and FLIVE.
- Entirely self-supervised training, requiring no human quality annotations.

### Ablation Study

| Configuration | Performance | Note |
|---------------|-------------|------|
| Direct optimization without voting | Significant degradation | Voting mechanism is essential |
| Fixed pseudo-labels (single round) | Below iterative version | Iterative self-evolution yields sustained improvement |
| Uniform reward weights | Below fidelity-weighted version | Consistency-based weighting is more effective |
| Varying voting count $K$ | Improves initially, then saturates | $K=5$–$10$ is the optimal range |

### Key Findings

1. **Self-supervised learning can match supervised learning**: For the first time in the IQA domain, it is demonstrated that self-supervised methods can reach or exceed the performance of supervised fine-tuning.
2. **Flexible composability**: EvoQuality can be stacked on top of pre-trained IQA models to further enhance generalization on unseen datasets.
3. **Cross-dataset generalization**: Without any annotations from target datasets, the framework achieves SOTA performance on multiple out-of-distribution IQA benchmarks.

## Highlights & Insights

1. **Extending self-consistency from reasoning to perception**: The paper cleverly adapts the "sample multiple times and take the consistent answer" paradigm from mathematical reasoning to ranking-based IQA—replacing answer consistency verification with majority voting over pairwise comparisons.
2. **Ranking is more suitable than scoring for self-supervision**: Absolute quality scores are difficult to self-evaluate, whereas relative quality comparisons more readily reach consensus through voting.
3. **Elegant positive feedback loop design**: The iterative self-evolution mechanism endows the training process with a self-accelerating property.
4. **High practical value**: The framework completely eliminates the dependency on human annotations in IQA, substantially reducing deployment costs.

## Limitations & Future Work

1. **Iterative efficiency**: The computational overhead of multi-round iteration combined with multiple sampling passes is substantial, particularly for large-scale VLMs.
2. **Dependence on base model quality**: If the base VLM has extremely weak quality perception, voting may fail to converge to a meaningful ranking.
3. **Task specificity**: The framework is designed for ranking/comparison tasks; adaptation to other perceptual tasks (e.g., aesthetic assessment, degradation detection) requires further modification.
4. **Lack of theoretical convergence guarantees**: No theoretical analysis is provided for the convergence of iterative self-evolution; in practice, there is a risk of overfitting to pseudo-labels.
5. **Scalable directions**: Extending the voting mechanism from pairwise to listwise comparisons may further improve efficiency.

## Related Work & Insights

- **Traditional IQA methods**: Handcrafted feature methods (BRISQUE, NIQE) → deep learning methods (DBCNN, HyperIQA) → VLM-based methods (Q-Align, Q-Instruct).
- **VLM self-improvement**: The success of self-iterative paradigms such as Self-Play and Self-Rewarding LLMs provides inspiration for this work.
- **GRPO**: The GRPO optimization method proposed by DeepSeek is transferred from reasoning tasks to perceptual tasks in this paper.
- **Insights**: The core idea of this framework (voting → pseudo-labels → self-iterative optimization) may be applicable to other visual tasks that lack a unique correct answer.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Creative application of self-consistency and GRPO to IQA, though individual components are not novel in themselves.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across 7 benchmarks with thorough ablations, but comparisons with more self-supervised baselines are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear and the framework is intuitive.
- **Value**: ⭐⭐⭐⭐⭐ — Eliminates the need for IQA annotations, offering substantial practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Grounding-IQA: Grounding Multimodal Language Models for Image Quality Assessment](grounding-iqa_grounding_multimodal_language_model_for_image_quality_assessment.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)
- [\[ICLR 2026\] Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models](self-aug_query_and_entropy_adaptive_decoding_for_large_vision-language_models.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](../../ACL2026/multimodal_vlm/what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)
- [\[CVPR 2026\] EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](../../CVPR2026/multimodal_vlm/evolmm_self_evolving_lmm_continuous_rewards.md)

</div>

<!-- RELATED:END -->
