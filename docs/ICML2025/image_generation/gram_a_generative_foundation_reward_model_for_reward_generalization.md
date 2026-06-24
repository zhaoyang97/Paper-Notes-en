---
title: >-
  [Paper Note] GRAM: A Generative Foundation Reward Model for Reward Generalization
description: >-
  [ICML 2025][Image Generation][Reward Model] GRAM proposes training reward models using a generative (rather than discriminative) approach. It pre-trains a generative reward model through large-scale unsupervised learning, fine-tunes it with supervised data, and proves that label smoothing is mathematically equivalent to a regularized pairwise ranking loss, thereby achieving reward generalization across tasks.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Reward Model"
  - "Generative Model"
  - "Foundation Model"
  - "RLHF"
  - "Generalization"
date: 2026-05-08
content_hash: ad36c9a76b6f68b0
---

# GRAM: A Generative Foundation Reward Model for Reward Generalization

**Conference**: ICML 2025  
**arXiv**: [2506.14175](https://arxiv.org/abs/2506.14175)  
**Code**: None  
**Area**: Image Generation (LLM Alignment)  
**Keywords**: Reward Model, Generative Model, Foundation Model, RLHF, Generalization

## TL;DR
GRAM proposes training reward models using a generative (rather than discriminative) approach. It pre-trains a generative reward model through large-scale unsupervised learning, fine-tunes it with supervised data, and proves that label smoothing is mathematically equivalent to a regularized pairwise ranking loss, thereby achieving reward generalization across tasks.

## Background & Motivation

**Background**: In LLM alignment, the reward model (RM) is a core component of RLHF, used to guide the model to generate outputs that align with human preferences. Currently, reward models are typically trained in a discriminative manner—directly learning a scoring function on human preference label data.

**Limitations of Prior Work**: Discriminative reward models rely heavily on labeled human preference data, resulting in insufficient generalization capabilities. When facing new tasks or out-of-distribution data, RM performance drops significantly. Meanwhile, acquiring high-quality preference data is extremely expensive.

**Key Challenge**: The limited amount of labeled preference data versus the need for reward models to generalize to a wide range of tasks. How can an RM obtain generalization capabilities through large-scale unlabeled data, similar to foundation language models?

**Goal**: Build a "foundation reward model" that can transfer to various tasks with minimal or even zero labeled data.

**Key Insight**: Leverage the "pre-training + fine-tuning" paradigm of LLMs—first train a generative RM on large-scale unsupervised data, then fine-tune it with a small amount of preference data.

**Core Idea**: The log-likelihood of a generative model naturally serves as a reward signal. By pre-training and fine-tuning, a highly generalizable foundation reward model can be constructed.

## Method

### Overall Architecture

- **Stage 1 (Unsupervised Pre-training)**: Pre-train in a generative manner on large-scale unlabeled text to learn the general quality distribution of language.
- **Stage 2 (Supervised Fine-tuning)**: Fine-tune on labeled human preference data to adapt the generative capability for reward scoring.
- **Inference**: Given a prompt-response pair, the log-likelihood (or its variant) of the generative RM is used as the reward score.

### Key Designs

1. **Generative RM**:

    - Unlike traditional RMs that append a scalar classification head at the end of the sequence, GRAM directly uses the log-likelihood $\log p_\theta(y|x)$ of the generative model as the reward.
    - Key Insight: High-quality responses should have a higher likelihood under the generative model.
    - **Design Motivation**: Generative models can leverage large-scale unlabeled data for pre-training to obtain a generalized understanding of language quality.

2. **Equivalence between Label Smoothing and Regularized Pairwise Ranking**:

    - Proves that when training with label smoothing, the generative loss is equivalent to a regularized pairwise ranking loss.
    - This means: $\mathcal{L}_{\text{smooth}} = (1-\epsilon)\mathcal{L}_{\text{CE}}(y_w) + \epsilon \mathcal{L}_{\text{CE}}(y_l)$ under preference training can be interpreted as simultaneously maximizing the likelihood of the preferred response and minimizing the likelihood of the dispreferred response.
    - **Design Motivation**: Establish a unified perspective of generative and discriminative models under the same training objective class.

3. **Transfer of Foundation Reward Models**:

    - The pre-trained generative RM can be applied directly (zero-shot) or with few-shot fine-tuning to various downstream tasks.
    - Including response ranking, RLHF training signals, and task adaptation.
    - **Design Motivation**: Mimic the zero-shot and few-shot transfer capabilities of foundation language models.

### Loss & Training

- **Pre-training**: Standard autoregressive language model loss $\mathcal{L} = -\sum_t \log p_\theta(y_t | y_{<t}, x)$
- **Fine-tuning**: Preference loss with label smoothing, which is equivalent to a regularized pairwise ranking loss.
- The training strategy supports two modes:
    - Direct training on preference data (label smoothed CE).
    - Freezing the generative model and training only lightweight adaptation layers.

## Key Experimental Results

### Main Results

| Task | Metric | GRAM | Discriminative Baseline | Gain |
|------|------|------|-----------|------|
| Preference Ranking (RewardBench) | Accuracy↑ | Significant improvement | Standard BT RM | Several percentage points |
| RLHF Training | Win Rate↑ | Higher | Standard RM | Significant |
| Task Adaptation (Few-shot) | Accuracy↑ | Better | Direct Fine-tuning RM | Significant improvement |
| Zero-shot Transfer | Accuracy↑ | Viable | Requires training data | No annotation needed |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| No Pre-training | Significant drop | Unsupervised pre-training is key to generalization |
| No Label Smoothing | Drop | Regularization effect is important for stable training |
| Pure Discriminative RM | Poor generalization | Performance degrades severely on out-of-distribution data |
| Different Pre-training Scales | Improves with scale | Larger models = Better generalization |

### Key Findings

- Generative RMs significantly outperform discriminative RMs in cross-task generalization.
- Label smoothing is not just a regularization technique; it has a deep mathematical connection with pairwise ranking.
- The quality and scale of pre-training directly determine the generalization capability of the foundation RM.
- GRAM outperforms several strong baselines in both RLHF training and direct ranking tasks.

## Highlights & Insights

1. **Theoretical Contribution**: Proof of equivalence between label smoothing and regularized pairwise ranking, unifying the generative and discriminative training perspectives.
2. **Paradigm Innovation**: Extends the "pre-training + fine-tuning" paradigm from language models to reward models.
3. **Practical Value**: The foundation RM can be reused across tasks, substantially reducing annotation costs for new tasks.
4. **Meta-learning Perspective**: Generative pre-training can be viewed as implicitly learning the meta-knowledge of "what constitutes good text."

## Limitations & Future Work

1. The inference cost of generative RMs is higher than that of discriminative RMs with a single scalar output.
2. Log-likelihood as a reward may not align perfectly with human preferences on certain tasks (e.g., creative writing).
3. Domain bias in pre-training data may affect the generalization of the RM in specific specialized fields.
4. Computational cost of large-scale pre-training.

## Related Work & Insights

- Connection with DPO: DPO also uses likelihood ratio as an implicit reward; GRAM further generalizes this concept to foundation models.
- Comparison with GPT-4 as a judge: GRAM provides a parameterized, trainable alternative.
- Insight: The equivalence between generative and discriminative models can potentially be leveraged in more scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of generative foundation RM and its theoretical connections are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-task validation.
- Writing Quality: ⭐⭐⭐⭐ Good integration of theory and experiments.
- Value: ⭐⭐⭐⭐⭐ Provides a new paradigm for RM training with high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](../../NeurIPS2025/image_generation/diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[ICML 2025\] Towards a Mechanistic Explanation of Diffusion Model Generalization](towards_a_mechanistic_explanation_of_diffusion_model_generalization.md)
- [\[CVPR 2025\] Visual-ERM: Reward Modeling for Visual Equivalence](../../CVPR2025/image_generation/visual-erm_reward_modeling_for_visual_equivalence.md)
- [\[ICLR 2026\] EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing](../../ICLR2026/image_generation/editreward_a_human-aligned_reward_model_for_instruction-guided_image_editing.md)
- [\[ICML 2025\] Discriminative Policy Optimization for Token-Level Reward Models](discriminative_policy_optimization_for_token-level_reward_models.md)

</div>

<!-- RELATED:END -->
