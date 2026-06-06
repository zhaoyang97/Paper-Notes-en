---
title: >-
  [Paper Note] Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation
description: >-
  [AAAI 2026 (Oral)][Recommender Systems][Generative Recommendation] This paper proposes Align³GR, a unified three-level alignment framework that systematically bridges the semantic-behavioral gap between LLMs and recommen…
tags:
  - "AAAI 2026 (Oral)"
  - "Recommender Systems"
  - "Generative Recommendation"
  - "LLM Alignment"
  - "Collaborative Filtering"
  - "DPO"
  - "Semantic-Collaborative ID"
date: 2026-05-08
content_hash: d02145eb5489ccce
---

# Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation

**Conference**: AAAI 2026 (Oral)  
**arXiv**: [2511.11255v2](https://arxiv.org/abs/2511.11255v2)  
**Code**: None  
**Area**: Recommender Systems / Information Retrieval  
**Keywords**: Generative Recommendation, LLM Alignment, Collaborative Filtering, DPO, Semantic-Collaborative ID  

## TL;DR

This paper proposes Align³GR, a unified three-level alignment framework that systematically bridges the semantic-behavioral gap between LLMs and recommender systems at the token level (dual-side SCID), the behavior modeling level (multi-task SFT), and the preference level (progressive DPO).

## Background & Motivation

Using LLMs as generative recommenders to produce recommendations end-to-end has become a recent trend. However, a fundamental gap exists between the language modeling objective of LLMs—focused on semantic information and next-token prediction—and the implicit user preference modeling objective of recommender systems, which centers on interaction behavior. Existing work typically performs alignment at only one of three stages: tokenization, SFT, or preference RL, lacking systematic multi-level joint optimization. Moreover, prior methods in the tokenization stage often encode only items while ignoring users, and preference alignment approaches largely rely on static offline data, making them ill-suited for dynamically evolving user preferences in real-world scenarios.

## Core Problem

1. How to jointly model both semantic and collaborative signals for users and items at the token level, rather than encoding them in isolation?
2. How to enable LLMs during SFT to not only learn behavioral recommendation patterns but also understand the semantic meaning of user tokens?
3. How to continuously improve the model through progressive preference optimization (from easy to hard), breaking the performance ceiling of static DPO?

## Method

### Overall Architecture

Align³GR is a unified three-level alignment pipeline: Token-Level Alignment → Behavior Modeling-Level Alignment → Preference-Level Alignment. It uses Llama2-7B as the backbone with LoRA for parameter-efficient fine-tuning.

### Key Designs

1. **Token-Level Alignment: Dual-Side SCID Tokenization**

    - Semantic features (frozen T5 encoder) and collaborative features (frozen DIN encoder) are extracted separately for both users and items, then concatenated and fused via an SC Encoder (MLP) into a unified SC embedding.
    - A 3-layer RQ-VAE (256 codebook embeddings per layer, dimension 32) quantizes the SC embedding into discrete SCID tokens.
    - The training objective consists of two components: a user-item behavior alignment loss $\mathcal{L}_{\text{U2I}}$ (sampled-softmax) and RQ-VAE reconstruction/quantization losses, controlled by hyperparameters $\alpha, \gamma$ in a two-stage switching schedule—first stabilizing behavior alignment ($\alpha=1, \gamma=0$), then focusing on quantization learning ($\alpha=0.1, \gamma=1$).
    - At inference time, the user and item modules operate independently, each generating their respective SCIDs.

2. **Behavior Modeling-Level Alignment: Augmented Multi-Task SFT**

    - Built upon the multi-task SFT of LC-Rec (sequential prediction, asymmetric prediction, intent inference, and preference reasoning), with two key enhancements:
    - **User SCID Injection**: User SCID tokens are incorporated into the prompts of all tasks to provide richer contextual information.
    - **Bidirectional Alignment Task** ($B_2$): text→SCID (predicting SCID from user profiles) and SCID→text (reconstructing user profiles from SCIDs), explicitly establishing correspondence between SCID tokens and their semantic meanings.

3. **Preference-Level Alignment: Progressive DPO**

    - Based on Softmax-DPO (1 positive + 20 negatives per sample), training proceeds in two progressive stages:
    - **SP-DPO (Self-Play DPO)**: The model plays against itself to generate diverse training data. Leveraging the hierarchical structure of SCIDs, training is divided into three stages (Easy/Medium/Hard) by prefix-ngram overlap, progressively transitioning from completely distinct positive-negative pairs to pairs whose prefixes are highly overlapping yet still different.
    - **RF-DPO (Real-world Feedback DPO)**: Real user feedback is used to construct preference data at three levels (disliked/neutral/liked), with the same progressive curriculum—Easy stage uses strongly disliked items as negatives, while Hard stage uses neutral (exposed but not clicked) items as harder negatives.
    - The fine-tuned model at each stage serves as the reference model for the next stage: $\pi_\theta^i \to \pi_{\text{ref}}^{i+1}$.

### Loss & Training

- Token level: $\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{U2I}} + \gamma \cdot (\mathcal{L}_{\text{User RQ}} + \mathcal{L}_{\text{Item RQ}})$, with two-stage training.
- Behavior modeling level: multi-task SFT loss + bidirectional alignment auxiliary loss.
- Preference level: Softmax-DPO loss with progressive SP-DPO → RF-DPO, iteratively updating the reference policy at each stage.
- Backbone: Llama2-7B + LoRA, AdamW optimizer, batch size 1024, trained for 20,000 steps, beam width 20.

## Key Experimental Results

| Dataset | Metric | Align³GR | EAGER-LLM (Prev. SOTA) | Gain |
|--------|------|----------|----------------------|------|
| Instruments | R@5 | 0.1103 | 0.0991 | +11.3% |
| Instruments | R@10 | 0.1442 | 0.1224 | +17.8% |
| Instruments | N@5 | 0.0947 | 0.0851 | +11.3% |
| Instruments | N@10 | 0.1113 | 0.0926 | +20.2% |
| Beauty | R@10 | 0.0994 | 0.0830 | +19.8% |
| Beauty | N@10 | 0.0529 | 0.0459 | +15.3% |
| Yelp | R@10 | 0.0679 | 0.0569 | +19.3% |
| Yelp | N@10 | 0.0403 | 0.0315 | +27.9% |

**Industrial A/B Test** (approximately 40 million users, multi-week deployment):

| Model | Recall@100 | Revenue Gain |
|------|------------|-------------|
| TIGER | 0.229 | +0.555% |
| Align³GR | 0.242 | +1.432% |

### Ablation Study

- **Token level**: Single-side → dual-side tokenization yields substantial improvements; incorporating collaborative features (CF) provides further gains; the U-I alignment loss works best in combination with dual-side + CF.
- **Behavior modeling level**: Injecting User SCID into prompts consistently improves performance; the bidirectional alignment task $B_2$ contributes the most, indicating that LLMs require explicit supervision to establish semantic-to-structured-token mappings.
- **Preference level**: Self-Play improves R@10 from 0.1295 to 0.1356; the progressive curriculum further raises it to 0.1396; adding real-world feedback RF-DPO with the progressive strategy achieves the best result of 0.1442.

## Highlights & Insights

- **Systematic alignment design**: The three-level alignment pipeline (token/behavior/preference) establishes clear alignment objectives at each stage, with experiments confirming that contributions from each level are complementary.
- **Dual-side SCID**: Unlike prior methods that encode only items, this work jointly models users and items within a unified semantic-collaborative representation and optimizes them with a U2I behavioral loss.
- **Progressive DPO**: The curriculum learning strategy from SP-DPO to RF-DPO, and from Easy to Hard, addresses the limitations of static DPO in dynamic recommendation scenarios.
- **Industrial validation**: Beyond offline experiments on three public datasets, the approach is validated through an online A/B test at the scale of 40 million users, achieving a 1.432% revenue improvement.

## Limitations & Future Work

- Only Llama2-7B is used as the backbone; the impact of larger or more recent LLMs (e.g., Llama3) remains unexplored.
- The RQ-VAE codebook size is fixed at 256, which may cause codebook collisions for extremely large item catalogs.
- The feedback granularity in RF-DPO (disliked/neutral/liked) is coarse; finer-grained feedback signals may yield further improvements.
- On public datasets, LLM-based sentiment analysis is used as a proxy for real user feedback in RF-DPO, potentially introducing noise.
- User history is limited to the most recent 20 interactions, leaving long-sequence modeling capabilities largely unexplored.

## Related Work & Insights

- **vs. LC-Rec**: LC-Rec performs only item tokenization and multi-task SFT; Align³GR adds dual-side SCID and progressive DPO, achieving comprehensive improvements.
- **vs. EAGER-LLM**: EAGER-LLM introduces collaborative signals at the token level but remains item-side only, with no preference alignment; Align³GR achieves all-around gains via dual-side tokenization, enhanced behavior modeling, and progressive DPO.
- **vs. LETTER**: LETTER proposes a learnable tokenizer but lacks user modeling and preference optimization; Align³GR significantly outperforms LETTER across all metrics.
- **vs. Standard DPO**: Conventional DPO relies on static offline data, whereas Align³GR's progressive SP-DPO + RF-DPO enables continuous self-improvement and real-world feedback adaptation.

The three-level alignment design (token/behavior/preference) is transferable to other scenarios where LLMs are adapted to downstream tasks such as LLM+search and LLM+advertising. The Easy-to-Hard curriculum strategy in progressive DPO is particularly valuable in settings with noisy preference labels, such as recommendation and advertising. The dual-side tokenization paradigm further suggests that users and items should be jointly modeled within a unified framework rather than processed independently when LLMs are used for recommendation.

## Rating

- Novelty: ⭐⭐⭐⭐ (The systematic three-level alignment design is innovative, though individual modules such as RQ-VAE and DPO are not novel in themselves.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three public datasets + industrial A/B testing + detailed ablation studies.)
- Writing Quality: ⭐⭐⭐⭐ (Well-structured, though some sections contain dense notation.)
- Value: ⭐⭐⭐⭐ (High industrial applicability; academic novelty is moderate-to-high.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](inductive_generative_recommendation_via_retrieval-based_speculation.md)
- [\[AAAI 2026\] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)
- [\[ACL 2026\] HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment](../../ACL2026/recommender/hsuga_llm-enhanced_recommendation_with_hierarchical_semantic_understanding_and_g.md)
- [\[ACL 2026\] From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation](../../ACL2026/recommender/from_past_to_path_masked_history_learning_for_next-item_prediction_in_generative.md)

</div>

<!-- RELATED:END -->
