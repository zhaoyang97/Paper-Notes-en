---
title: >-
  [Paper Note] Laser: Bi-Tuning with Collaborative Information for Controllable LLM-Based Sequential Recommendation
description: >-
  [ACL 2025][Recommender Systems][Sequential Recommendation] This paper proposes the Laser framework, which inserts trainable virtual tokens as prefixes and suffixes to a frozen LLM (Bi-Tuning) to inject user-item collaborative information, and designs an MoE-based M-Former to capture diverse characteristics of different users, achieving parameter-efficient sequential recommendation.
tags:
  - "ACL 2025"
  - "Recommender Systems"
  - "Sequential Recommendation"
  - "Large Language Models"
  - "Bi-Tuning"
  - "Collaborative Information"
  - "Parameter-Efficient"
date: 2026-05-08
content_hash: f28c5ec9dd3c6431
---

# Laser: Bi-Tuning with Collaborative Information for Controllable LLM-Based Sequential Recommendation

**Conference**: ACL 2025  
**arXiv**: [2409.01605](https://arxiv.org/abs/2409.01605)  
**Code**: None  
**Area**: Recommendation Systems  
**Keywords**: Sequential Recommendation, Large Language Models, Bi-Tuning, Collaborative Information, Parameter-Efficient

## TL;DR
This paper proposes the Laser framework, which inserts trainable virtual tokens as prefixes and suffixes to a frozen LLM (Bi-Tuning) to inject user-item collaborative information, and designs an MoE-based M-Former to capture diverse characteristics of different users, achieving parameter-efficient sequential recommendation.

## Background & Motivation

**Background**: Sequential recommendation systems predict the next item of potential interest by analyzing users' historical interaction sequences. Recently, utilizing the semantic understanding capabilities of LLMs to enhance sequential recommendation has become a research hotspot. Typical methods include using item titles as LLM inputs or using LLMs to encode item semantics.

**Limitations of Prior Work**: Existing LLM-based recommendation methods suffer from two main limitations: (1) Most methods require full or large-scale parameter fine-tuning, which consumes massive resources; (2) Although LLMs excel at encoding textual semantics, the core signal of sequential recommendation—user-item collaborative information—cannot be directly transmitted to the LLM through text. Existing works simply convert item IDs or descriptions into text, ignoring the user behavior patterns embedded in collaborative filtering signals.

**Key Challenge**: LLMs are proficient in understanding semantics but struggle with modeling collaborative relationships, while traditional recommendation models excel at collaborative filtering but lack semantic understanding. How to combine the strengths of both under a parameter-efficient premise?

**Goal**: Design a parameter-efficient framework that can both leverage the semantic capabilities of LLMs and inject the collaborative information from traditional recommendation models.

**Key Insight**: Inspired by prompt tuning, LLM behavior can be adjusted by inserting trainable tokens at both ends of the input. The prefix is responsible for injecting collaborative information and adapting to the task, while the suffix translates the language space output of the LLM to the recommendation space.

**Core Idea**: A Bi-Tuning paradigm (prefix + suffix) combined with an MoE-based query transformer (M-Former) to achieve user-differentiated collaborative information injection.

## Method

### Overall Architecture
The Laser framework consists of three modules: (1) a frozen ID-based sequential recommendation model (e.g., SASRec) providing collaborative filtering signals; (2) a frozen LLM acting as a semantic encoder; (3) a trainable Bi-Tuning module comprising the prefix-side M-Former and the suffix-side projection layer. The user's historical interaction sequence is first tokenized as text input to the LLM. The M-Former extracts queries from the frozen collaborative information and injects them at the prefix position. Suffix tokens project the LLM output into the recommendation space to calculate matching scores with candidate item embeddings.

### Key Designs

1. **Bi-Tuning Strategy (Prefix + Suffix Bi-tuning)**:

    - **Function**: Achieves task adaptation while keeping the LLM parameters frozen.
    - **Mechanism**: The prefix contains $N_p$ trainable virtual tokens inserted at the beginning of the input sequence, functioning similarly to prefix-tuning to guide the LLM's attention to recommendation-relevant information. The suffix contains $N_s$ trainable virtual tokens appended to the end of the input, whose corresponding LLM hidden state output is passed through a linear projection layer to serve as the user representation. During training, only the parameters of the prefix/suffix tokens, M-Former, and projection layer are optimized, while the LLM remains completely frozen.
    - **Design Motivation**: The prefix is responsible for "input adaptation" (injecting recommendation task information into the LLM), and the suffix is responsible for "output adaptation" (converting representations from the language space to the recommendation space). The coordination of both ends achieves task adaptation with minimal parameters.

2. **M-Former (MoE-based Querying Transformer)**:

    - **Function**: Generates differentiated collaborative information prefixes for different types of users.
    - **Mechanism**: M-Former is built on the Querying Transformer architecture (similar to Q-Former), maintaining a set of learnable query tokens that extract collaborative information from the frozen sequential recommendation model's output via cross-attention. The key innovation is replacing the standard FFN with a Mixture of Experts (MoE), where each expert corresponds to a specific user behavior pattern. For each user, a gating network selectively activates different combinations of experts based on their historical sequence features, achieving user-differentiated information extraction.
    - **Design Motivation**: Different users (active vs. inactive, concentrated vs. scattered preferences) require different types/levels of collaborative information. A single query transformer struggles to adapt to all user types. The MoE structure enables diverse information extraction strategies with low computational overhead.

3. **Collaborative Information Bridging**:

    - **Function**: Injects collaborative filtering representations from traditional recommendation models into the LLM.
    - **Mechanism**: A pre-trained sequential recommendation model like SASRec is used as a "collaborative encoder" with its parameters frozen. Through cross-attention, the M-Former "reads" the item sequence representations output by SASRec, extracts collaborative signals relevant to the current recommendation task, and transforms them into prefix tokens to be injected into the LLM. Consequently, while processing textual item descriptions, the LLM possesses both semantic and collaborative information.
    - **Design Motivation**: ID-based models and LLMs have complementary strengths. Bridging them allows them to complement each other, avoiding the enormous cost of training a single large model with both capabilities from scratch.

### Loss & Training
A softmax cross-entropy loss is adopted for next-item prediction training. During training, the parameters of the LLM and SASRec are frozen, and only the M-Former, prefix/suffix tokens, and the projection layer are optimized.

## Key Experimental Results

### Main Results

| Dataset | Metric | Laser | SASRec | P5 | TALLRec | Gain (vs best baseline) |
|--------|------|-------|--------|-----|---------|----------------------|
| Beauty | HR@10 | 5.82 | 4.93 | 4.21 | 5.15 | +13.0% |
| Beauty | NDCG@10 | 3.21 | 2.68 | 2.25 | 2.81 | +14.2% |
| Sports | HR@10 | 4.15 | 3.52 | 3.10 | 3.73 | +11.3% |
| Sports | NDCG@10 | 2.18 | 1.85 | 1.58 | 1.92 | +13.5% |
| Toys | HR@10 | 6.47 | 5.51 | 4.89 | 5.82 | +11.2% |

### Ablation Study

| Configuration | HR@10 | NDCG@10 | Description |
|------|-------|---------|------|
| Laser (Full) | 5.82 | 3.21 | Full Model |
| w/o M-Former (random prefix) | 4.91 | 2.62 | Dropping collaborative info injection severely degrades performance |
| w/o MoE (single expert) | 5.43 | 2.95 | Dropping MoE degrades performance, indicating the effectiveness of user-differentiated modeling |
| w/o suffix (mean pooling) | 5.21 | 2.78 | Suffix is crucial for output space transformation |
| w/o prefix (suffix only) | 5.05 | 2.71 | Suffix only, lacks collaborative info |
| Full fine-tuning LLM | 5.68 | 3.12 | Full-parameter tuning performs worse than Bi-Tuning |

### Key Findings
- M-Former contributes the most (dropping it decreases HR@10 by 15.6%), indicating that collaborative information injection is the core value.
- MoE brings a 7.2% improvement compared to a single expert, validating the necessity of user-differentiated modeling.
- Both prefixes and suffixes are indispensable; the performance of bi-tuning far exceeds single-ended tuning.
- Interestingly, full fine-tuning of the LLM underperforms Bi-Tuning, possibly due to overfitting or disrupting the pre-trained knowledge of the LLM.

## Highlights & Insights
- The division of labor between prefix and suffix in Bi-Tuning is exceptionally clear and elegant—the prefix handles input adaptation (injecting collaborative information) while the suffix handles output adaptation (space conversion). This represents an LLM adaptation paradigm worth promoting.
- Using a frozen traditional recommendation model as the "collaborative information provider" is highly practical, avoiding the massive cost of learning collaborative relationships inside the LLM from scratch.
- The MoE design in M-Former reflects the insight that "different users require different levels/types of collaborative information," which is intuitive yet previously neglected in recommendation systems.

## Limitations & Future Work
- It relies on pre-trained ID-based recommendation models to provide collaborative information, the quality of which directly limits the performance ceiling of Laser.
- Cold-start scenarios are not explored—new users or items have almost no collaborative information available for the M-Former to extract.
- The impact of selecting the number of experts in MoE on performance is not fully analyzed.
- Extending the Bi-Tuning paradigm to scenarios such as conversational recommendation and multimodal recommendation could be considered.

## Related Work & Insights
- **vs P5/TALLRec**: These methods formulate recommendation as a pure text generation task, which loses collaborative signals; Laser explicitly injects collaborative information via M-Former.
- **vs SASRec**: Traditional sequential recommendation cannot leverage semantic information, and Laser fills this gap in semantic understanding with the help of LLMs.
- **vs LoRA-based methods**: LoRA adjusts the LLM's internal parameters, whereas Laser injects information via external tokens. The two approaches are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of Bi-Tuning + M-Former is novel, and the introduction of MoE is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons across multiple datasets and baselines, with sound ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The framework diagram and method descriptions are clear.
- Value: ⭐⭐⭐⭐ Provides a parameter-efficient yet highly effective solution for LLM-based recommendation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](../../ICLR2026/recommender/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)
- [\[ACL 2025\] RecLM: Recommendation Instruction Tuning](reclm_recommendation_instruction_tuning.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)
- [\[ACL 2025\] Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation](beyond_single_labels_improving_conversational_recommendation_through_llm-powered.md)
- [\[ACL 2025\] CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems](cove_compressed_vocabulary_expansion_makes_better_llm-based_recommender_systems.md)

</div>

<!-- RELATED:END -->
