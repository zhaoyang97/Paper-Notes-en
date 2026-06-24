---
title: >-
  [Paper Note] Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation
description: >-
  [ACL 2025][Recommender Systems][Conversational Recommender Systems] To address the false negative problem in conversational recommender systems (where items users might like are incorrectly labeled as negative samples), an LLM-powered data augmentation framework is proposed. It generates synthetic labels through semantic retrieval and relevance scoring, and balances semantic relevance with collaborative information via a two-stage training strategy.
tags:
  - "ACL 2025"
  - "Recommender Systems"
  - "Conversational Recommender Systems"
  - "Data Augmentation"
  - "False Negative Problem"
  - "Large Language Models"
  - "Two-Stage Training"
date: 2026-05-08
content_hash: fc21b6e57801f0f7
---

# Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation

**Conference**: ACL 2025  
**arXiv**: [2508.05657](https://arxiv.org/abs/2508.05657)  
**Code**: [github.com/xu1110/FNSCRS](https://github.com/xu1110/FNSCRS)  
**Area**: Recommendation Systems  
**Keywords**: Conversational Recommender Systems, Data Augmentation, False Negative Problem, Large Language Models, Two-Stage Training

## TL;DR

To address the false negative problem in conversational recommender systems (where items users might like are incorrectly labeled as negative samples), an LLM-powered data augmentation framework is proposed. It generates synthetic labels through semantic retrieval and relevance scoring, and balances semantic relevance with collaborative information via a two-stage training strategy.

## Background & Motivation

Conversational recommender systems (CRS) provide recommendations through multi-turn dialogues with users, but suffer from severe false negative problems during training:

- **Problem Instance**: When a user says "I want to watch a funny cop movie", only one movie in the training data is labeled as a positive sample, while other matching funny cop movies are incorrectly treated as negative samples.
- **Unique Challenges in CRS**: Unlike traditional recommendation systems, CRS datasets are rich in semantic information (dialogue context). Augmenting labels requires simultaneously ensuring: (1) semantic relevance with the dialogue context; (2) preservation of collaborative information (commonalities and trends in user behavior) inherent in the dataset.
- **Limitations of LLMs**: Although LLMs excel at understanding semantic relevance, they struggle to capture collaborative information effectively. Over-reliance on LLM-suggested labels may bias recommendations toward semantic consistency while ignoring collaborative information, thereby reducing user satisfaction.

Existing methods either mitigate false negatives by treating them as noise (e.g., reducing negative sampling probability) or expand the label set via dataset augmentation. However, they lack an effective balancing mechanism between semantic relevance and collaborative information in CRS scenarios.

## Method

### Overall Architecture

The method is divided into two stages: the data synthesis stage and the model training stage.

**Data Synthesis Stage**: Uses LLMs for semantic retrieval and relevance scoring to generate synthetic training data.
**Model Training Stage**: Two-stage training—first pre-training on synthetic data to learn semantic relationships, and then fine-tuning on raw data to integrate collaborative information.

### Key Designs

1. **LLM Semantic Retriever (Relevant Items Retrieval)**:

    - **Mechanism**: Retrieves candidate items based solely on semantic information without considering collaborative information, thereby avoiding biases introduced by collaborative statistics, such as popularity bias.
    - Uses GritLM as a text encoder to encode item description texts and dialogue contexts into dense vectors.
    - Retrieves the top-50 most similar items for each dialogue context via Maximum Inner Product Search (MIPS).
    - **Design Motivation**: Ignoring collaborative information in the initial stage allows for coverage of a wider range of items, avoiding over-focusing on popular items.

2. **LLM Relevance Scorer (Relevance Estimation)**:

    - Uses GPT-4 to generate context-item-score triplets as training data (via Chain-of-Thought prompting).
    - Trains Gemma2-9b to assign fine-grained relevance scores ranging from 0 to 4 for each candidate item.
    - Sets a threshold of 3.5, retaining high-scoring items to construct the synthetic training dataset.
    - Effect: The original 29,810 positive samples in ReDial are expanded to 377,313, and INSPIRED is expanded from 1,404 to 15,891.

3. **Two-Stage Training Strategy**:

    - **Stage 1 (Pre-training)**: Trains the recommender on the synthetic dataset using the standard cross-entropy loss to learn the semantic relationships between user preferences and items, eliminating biases in the original data.
    - **Stage 2 (Fine-tuning)**: Fine-tunes the model on the original key dataset to integrate collaborative information. It introduces a label smoothing term (based on KL divergence) using the output of the pre-trained model as soft labels, where the coefficient $\alpha$ controls the degree of reliance on collaborative information.
    - **Design Motivation**: Learning semantics first and then collaborative information allows for the integration of both types of information in a controllable manner.

### Loss & Training

Pre-training stage: Standard cross-entropy loss
$$L_{pre} = -\sum_{i=1}^{N}\sum_{j=1}^{M} y_{i,j} \cdot \log P(i,j)$$

Fine-tuning stage: Cross-entropy + Label smoothing
$$L_{finetune} = L_{ce} + \alpha \cdot L_{soft}$$
$$L_{soft} = \sum_{i=1}^{N} D_{KL}(P(i), \hat{y_i})$$

where a larger $\alpha$ denotes lower dependency on collaborative information.

## Key Experimental Results

### Main Results

| Model | ReDial R@1 | ReDial R@10 | ReDial R@50 | INSPIRED R@1 | INSPIRED R@10 | INSPIRED R@50 |
|------|-----------|-------------|-------------|-------------|--------------|--------------|
| BARCOR | 3.13 | 17.34 | 36.32 | 2.86 | 11.06 | 30.81 |
| BARCOR + ours | **4.31** | **21.26** | **43.84** | **3.73** | **21.12** | **43.11** |
| UniCRS | 3.53 | 19.60 | 40.50 | 3.97 | 20.00 | 40.66 |
| UniCRS + ours | **3.76** | **20.93** | **42.74** | **5.43** | **22.91** | 39.47 |
| Llama2 | 3.93 | 20.74 | 41.34 | 4.46 | 11.68 | 34.16 |
| Llama2 + ours | **4.46** | **22.37** | **44.20** | **9.32** | **28.26** | **50.93** |

The improvement is even more significant in user simulator evaluations: R@50 of Llama2 on INSPIRED improves from 34.78 to 73.29 (+111%).

### Ablation Study

| Configuration | ReDial R@10 | INSPIRED R@10 | Description |
|------|------------|--------------|------|
| BARCOR Baseline | 17.34 | 11.06 | No Augmentation |
| + Self-Distillation | 19.95 | 19.38 | Uses Collaborative + Semantic Retrieval |
| + CFCRS | 18.98 | 20.50 | Counterfactual Dialogue Simulation |
| + Ours | **21.26** | **21.12** | Semantic-First + Two-Stage |

### Key Findings

- **Consistent Improvements**: The method achieves stable improvements across three backbone models, two datasets, and two evaluation environments, demonstrating robust generalizability.
- **Outperforming Zero-Shot LLMs**: Even with smaller models, the proposed method outperforms the zero-shot recommendation performance of GPT-3.5 and GPT-4o (Llama2+ours R@10 of 22.37 vs. GPT-4o of 17.20).
- **Semantic-First Outperforms Hybrid Retrieval**: Compared to Self-Distillation which leverages both collaborative and semantic information, utilizing only semantic information in the initial stage yields superior performance, validating the effectiveness of the strategy to block collaborative biases.
- **Scale of Synthetic Data**: Synthetic data for ReDial is expanded by approximately 12.7 times, and for INSPIRED by approximately 11.3 times.

## Highlights & Insights

- **Systematization of the False Negative Problem**: This is the first work to systematically study the false negative problem within the CRS context, presenting a clear framework to address it.
- **Decoupled Design of Semantics and Collaboration**: Unlike prior methods that mix the two types of information, this paper achieves controllable information integration via two-stage training, which is an elegant design concept.
- **Compatibility with Existing Methods**: Since this method augments labels rather than dialogues, it is complementary to existing dialogue augmentation techniques (such as CFCRS).
- **High Practical Value**: The proposed method is applicable to various CRS recommender backbones without requiring structural modifications to the recommender architecture.

## Limitations & Future Work

- Training the relevance scorer relies on data generated by GPT-4, which involves financial costs and potential bias propagation.
- The label smoothing coefficient $\alpha$ requires manual tuning, lacking an adaptive mechanism.
- The model is only validated on movie recommendation datasets and has not been tested in other recommendation domains (e.g., music, e-commerce).
- Two-stage training increases training complexity, and the pre-training stage involves a large data scale (370k+ samples).
- Future work could explore the synergistic enhancement of synthetic dialogues combined with synthetic labels.

## Related Work & Insights

This work lies at the intersection of CRS, LLM-augmented recommendation, and false negative processing. Unlike the false-negative detection work in traditional recommendation systems by Wei et al. (2024), this work avoids collaborative information in the retrieval stage to prevent biases, and introduces collaborative information in a controllable manner via two-stage training. This "semantics first, collaboration second" paradigm offers valuable reference for effectively leveraging LLM capabilities in recommendation systems.

## Rating

- Novelty: ⭐⭐⭐⭐ The first systematic study of the false negative problem in CRS, with an innovative two-stage training strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three backbones, two datasets, two evaluation settings, with comprehensive and in-depth analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and intuitive illustrations, though somewhat heavy on notation.
- Value: ⭐⭐⭐⭐ Simple and effective method, easily applicable to practical CRS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Laser: Bi-Tuning with Collaborative Information for Controllable LLM-Based Sequential Recommendation](bi-tuning_with_collaborative_information_for_controllable_llm-based_sequential_r.md)
- [\[ICLR 2026\] More Than What Was Chosen: LLM-based Explainable Recommendation Beyond Noisy User Preferences](../../ICLR2026/recommender/more_than_what_was_chosen_llm-based_explainable_recommendation_beyond_noisy_user.md)
- [\[AAAI 2026\] RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems](../../AAAI2026/recommender/rectom_a_benchmark_for_evaluating_machine_theory_of_mind_in_llm-based_conversati.md)
- [\[ICML 2025\] MATCHA: Toward Safe and Human-Aligned Game Conversational Recommendation via Multi-Agent Decomposition](../../ICML2025/recommender/toward_safe_and_human-aligned_game_conversational_recommendation_via_multi-agent.md)
- [\[ICLR 2026\] Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning](../../ICLR2026/recommender/rank-grpo_training_llm-based_conversational_recommender_systems_with_reinforceme.md)

</div>

<!-- RELATED:END -->
