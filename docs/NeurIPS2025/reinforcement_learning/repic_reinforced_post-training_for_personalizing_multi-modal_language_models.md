---
title: >-
  [Paper Note] RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models
description: >-
  [NeurIPS 2025][Reinforcement Learning][Multimodal Large Language Models] This paper proposes RePIC, the first reinforcement learning-based post-training framework for multimodal large language models targeting personalized image captioning, which significantly outperforms SFT-based methods in multi-concept scenarios.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Multimodal Large Language Models"
  - "Personalization"
  - "Image Captioning"
  - "Post-Training"
date: 2026-05-08
content_hash: 9971837a30a90a3d
---

# RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models

**Conference**: NeurIPS 2025

**arXiv**: [2506.18369](https://arxiv.org/abs/2506.18369)

**Code**: [GitHub](https://github.com/oyt9306/RePIC)

**Area**: Reinforcement Learning / Multimodal

**Keywords**: Multimodal Large Language Models, Personalization, Reinforcement Learning, Image Captioning, Post-Training

## TL;DR

This paper proposes RePIC, the first reinforcement learning-based post-training framework for multimodal large language models targeting personalized image captioning, which significantly outperforms SFT-based methods in multi-concept scenarios.

## Background & Motivation

Multimodal large language models (MLLMs) excel at generating standard image captions, yet face significant challenges in personalized settings, such as recognizing and describing a specific user's belongings or pets.

Key issues:

**Limitations of SFT**: Existing personalization methods rely primarily on supervised fine-tuning (SFT), but acquiring large-scale, high-quality personalized captioning data is **prohibitively expensive**.

**Multi-Concept Difficulty**: SFT-based methods frequently fail when an image contains multiple concepts requiring personalized recognition.

**Insufficient Visual Recognition**: Even when trained on high-quality data, models often fail to correctly identify specific entities in real-world scenarios.

**Data Bottleneck of SFT**: The performance ceiling of SFT is constrained by training data quality, a limitation that RL can overcome.

## Method

### Overall Architecture

RePIC is a two-stage framework: (1) an SFT stage to initialize basic personalization capabilities, and (2) a reinforcement learning post-training stage that leverages automatic reward signals to further improve performance.

### Key Designs

**1. Reward Function Design**

Two categories of reward signals are designed to train the RL policy:
- **Visual Recognition Reward**: Based on whether personalized entities in the image are correctly identified in the generated caption.
    - An entity detector verifies whether the personalized names mentioned in the caption match the visual entities present in the image.
- **Caption Quality Reward**: Based on the fluency, informativeness, and accuracy of the generated caption.
    - A reference model or rule-based evaluator assesses caption quality.

**2. Reinforcement Learning Post-Training**

- PPO or similar policy gradient methods are employed.
- Initial policy: the model trained during the SFT stage.
- Objective: maximize a weighted combination of visual recognition accuracy and caption quality.
- KL divergence constraint: prevents excessive policy deviation that could lead to language degeneration.

**3. Multi-Concept Handling**

- Training data for RL is specifically constructed to target multi-concept scenarios.
- The reward function accounts for whether all concepts are correctly recognized and described.
- Simultaneous personalization of 2–5 concepts is supported.

### Loss & Training

$$\mathcal{L}_{\text{RL}} = -\mathbb{E}_{\pi_\theta}[R(y, I, C)] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

where $R$ is the composite reward, $I$ is the image, and $C$ is the set of personalized concepts.

## Key Experimental Results

### Main Results

Single-concept personalized image captioning (MyVLM Benchmark):

| Method | Recognition Accuracy | Caption Quality (CIDEr) | Overall Score |
|--------|---------------------|------------------------|---------------|
| InstructBLIP | 35.2% | 52.3 | 43.8 |
| MyVLM (SFT) | 68.5% | 78.6 | 73.6 |
| Yo'LLaVA (SFT) | 72.3% | 81.2 | 76.8 |
| RePIC (Ours) | **82.1%** | **85.5** | **83.8** |

Multi-concept personalized image captioning:

| Method | 2 Concepts | 3 Concepts | 4 Concepts | 5 Concepts |
|--------|-----------|-----------|-----------|-----------|
| MyVLM (SFT) | 52.3% | 38.5% | 25.1% | 15.8% |
| Yo'LLaVA (SFT) | 58.1% | 42.3% | 28.5% | 18.2% |
| RePIC (Ours) | **75.2%** | **65.8%** | **52.3%** | **40.5%** |

### Ablation Study

Contribution of individual components in RL post-training:

| Configuration | Single-Concept | Multi-Concept (3) | Caption Quality |
|---------------|---------------|-------------------|----------------|
| SFT only | 72.3% | 42.3% | 81.2 |
| + RL (Recognition Reward) | 79.5% | 60.2% | 79.8 |
| + RL (Quality Reward) | 74.1% | 45.8% | 84.5 |
| + RL (Combined Reward) | **82.1%** | **65.8%** | **85.5** |

### Key Findings

1. RL post-training yields the most substantial improvement in multi-concept scenarios (42.3% → 65.8%, a 55% relative gain).
2. The visual recognition reward is the most critical component, contributing most significantly to multi-concept tasks.
3. The quality reward prevents language degeneration induced by RL, ensuring caption fluency.
4. RL surpasses the data quality ceiling of SFT by discovering superior captioning strategies through exploration.

## Highlights & Insights

- **Pioneering Contribution**: This is the first work to apply RL post-training to MLLM personalization.
- **Multi-Concept Breakthrough**: The advantage is most pronounced in the most challenging multi-concept scenarios.
- **Practical Relevance**: Personalized visual assistants—such as those capable of recognizing a user's pets or belongings—have broad application prospects.

## Limitations & Future Work

1. The reward function design depends on the accuracy of the underlying entity detector.
2. RL training is less stable than SFT and requires careful tuning of the KL constraint coefficient.
3. Validation is currently limited to static images; video scenarios remain unexplored.
4. Performance continues to degrade noticeably as the number of personalized concepts increases.

## Related Work & Insights

- **MyVLM** (Alaluf et al.): A pioneering work on personalization for multimodal models.
- **Yo'LLaVA**: An SFT-based personalization approach.
- **RLHF/PPO**: Standard RL methods for LLM alignment.

## Rating

- ⭐ Novelty: 8/10 — Applying RL post-training to MLLM personalization is a novel contribution.
- ⭐ Value: 8/10 — Personalized visual assistants address a broad need; code is open-sourced.
- ⭐ Writing Quality: 7/10 — Experiments are thorough, though the method section could be presented with greater clarity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](../../ACL2025/reinforcement_learning/maporl_multi-agent_post-co-training_for_collaborative_large_language_models_with.md)
- [\[NeurIPS 2025\] Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](../../ICLR2026/reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[NeurIPS 2025\] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)
- [\[ICLR 2026\] Representation-Based Exploration for Language Models: From Test-Time to Post-Training](../../ICLR2026/reinforcement_learning/representation-based_exploration_for_language_models_from_test-time_to_post-trai.md)

</div>

<!-- RELATED:END -->
