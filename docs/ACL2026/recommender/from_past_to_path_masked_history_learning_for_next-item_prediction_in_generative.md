---
title: >-
  [Paper Note] From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation
description: >-
  [ACL 2026][Recommender Systems][Generative Recommendation] This paper proposes the Masked History Learning (MHL) training framework. By incorporating a masked history reconstruction auxiliary task into the autoregressive…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Generative Recommendation"
  - "Masked History Learning"
  - "Information Entropy Guidance"
  - "Curriculum Learning"
  - "Sequential Recommendation"
date: 2026-05-08
content_hash: 3ddcddd6bac685de
---

# From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation

**Conference**: ACL 2026  
**arXiv**: [2509.23649](https://arxiv.org/abs/2509.23649)  
**Code**: [GitHub](https://github.com/CQU-MM-Intelligent-Lab/MHL)  
**Area**: Image Generation  
**Keywords**: Generative Recommendation, Masked History Learning, Information Entropy Guidance, Curriculum Learning, Sequential Recommendation

## TL;DR

This paper proposes the Masked History Learning (MHL) training framework. By incorporating a masked history reconstruction auxiliary task into the autoregressive training of generative recommendation, combined with an entropy-guided adaptive masking strategy and a curriculum learning scheduler, the model shifts from merely predicting "what is next" to understanding "why this path was formed," significantly outperforming SOTA on three datasets.

## Background & Motivation

**Background**: Generative recommendation is an emerging paradigm that encodes items as semantic ID sequences and utilizes pre-trained language models (such as T5) or LLMs to directly generate identifiers for recommended items, offering advantages in flexibility and scalability.

**Limitations of Prior Work**: Existing generative recommendation models (TIGER, HSTU, RPG, etc.) almost entirely rely on autoregressive next-item prediction training. This "left-to-right" paradigm is inherently biased toward local context and fails to capture deep historical dependencies and complex user intentions within the behavior path. Models excel at local prediction but lack global understanding, becoming susceptible to noise and short-term bias (the "short-term myopia" problem).

**Key Challenge**: When recent interaction history is truncated, existing SOTA models experience performance collapse (e.g., after a photography enthusiast buys a camera, tripod, camera bag, and lens in sequence, the model focuses only on the "lens" to predict lens accessories, ignoring that the "camera body" is the true intent driving the subsequent purchase of a memory card).

**Goal**: To enable generative recommendation models to not only learn "what is next" but also understand "why this purchasing path was formed."

**Key Insight**: Drawing inspiration from Masked Language Modeling (MLM) in NLP, an auxiliary history reconstruction task is introduced into autoregressive training.

**Core Idea**: Jointly optimize the next-item prediction and masked history reconstruction objectives. The most informative historical positions are selected for masking based on information entropy guidance, with a curriculum learning approach used to smoothly transition from history understanding to future prediction.

## Method

### Overall Architecture

MHL adds a masked history reconstruction branch to semantic ID-based generative recommendation. Each item is encoded into a $K$-bit semantic ID, and a Transformer decoder generates contextual hidden states. During training, two losses are optimized simultaneously: (1) $\mathcal{L}_{next}$ for predicting the semantic ID of the next item; (2) $\mathcal{L}_{mask}$ for reconstructing masked historical items from contextual states. During inference, only autoregressive generation is used without masking.

### Key Designs

1.  **Multi-granularity Masking Strategy**:
    - **Function**: Provides diverse learning signals for history reconstruction.
    - **Mechanism**: Three masking granularities are designed: item-level (replacing the entire $K$-bit semantic ID to learn inter-item dependencies), token-level (replacing a single digit to learn intra-item token relationships), and hybrid-level (randomly choosing item or token level). Experiments show that token-level + R→E→Inf curriculum learning performs best.
    - **Design Motivation**: Different masking granularities force the model to understand the structure and dependencies of historical sequences at different levels.

2.  **Entropy-Guided Masking**:
    - **Function**: Intelligently selects the most informative historical positions for masked reconstruction.
    - **Mechanism**: In each training step, a gradient-free forward pass is first performed on the unmasked sequence to calculate the predictive entropy $\mathcal{H}_t^k$ at each position. High-entropy positions indicate areas where the model lacks understanding, corresponding to key decision points or complex semantic units. Positions are ranked by entropy in descending order, and the top-$N$ positions are masked, where $N$ is sampled from a uniform distribution $\mathcal{U}(1, \lfloor\gamma \cdot T\rfloor)$ to prevent overfitting.
    - **Design Motivation**: Random masking treats all positions equally, ignoring the fact that information density in user behavior is non-uniform; entropy guidance ensures the reconstruction objective always focuses on the model's weakest points.

3.  **Curriculum Training Scheduler**:
    - **Function**: Bridges the gap between masked training and unmasked inference.
    - **Mechanism**: Phase I (Random Masking Warm-up) — establish baseline reconstruction capability and stable optimization with a low-rate random mask; Phase II (Entropy-Guided Masking + Adaptive Decay) — switch to a high-rate entropy-guided mask to encourage deep history understanding, with the mask ratio exponentially decayed $\gamma \leftarrow \max(\gamma_{min}, \gamma \cdot \eta)$ once validation performance plateaus; Phase III (Inference Alignment) — set $\gamma=0$ and train only on $\mathcal{L}_{next}$ to eliminate the training-inference discrepancy.
    - **Design Motivation**: Directly using entropy-guided masking leads to unstable training (due to unreliable early entropy estimates), while the three-phase transition achieves a smooth conversion from "understanding the path" to "generating the path."

### Loss & Training

$$\mathcal{L}_{MHL} = \lambda_1 \mathcal{L}_{next} + \lambda_2 \mathcal{L}_{mask}$$

Where $\mathcal{L}_{next}$ is the standard cross-entropy loss for the next item's semantic ID, and $\mathcal{L}_{mask}$ is the reconstruction cross-entropy loss for masked positions.

## Key Experimental Results

### Main Results

| Dataset | Metric | MHL | RPG (Prev. SOTA) | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Beauty | R@10 | .0795 | .0745 | +6.7% |
| Beauty | N@10 | .0495 | .0436 | +13.5% |
| Toys | R@10 | .0903 | .0778 | +16.1% |
| Toys | N@10 | .0564 | .0460 | +22.6% |
| Sports | R@10 | .0511 | .0436 | +17.2% |
| Sports | N@10 | .0298 | .0246 | +21.1% |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Token-level + R→E→Inf | Best | Overall optimal across three datasets |
| No Masking (Baseline RPG) | .0745 R@10 | Next-item prediction only |
| Random Masking | Improved | Simple masking is effective |
| Entropy-Guided only (No Curriculum) | Performance Drop | Unstable training due to unreliable early entropy estimates |
| Truncated Recent History | MHL vs SOTA +40% | MHL shows robust performance under history truncation |

### Key Findings
- The auxiliary task of masked history reconstruction brings improvements across all three granularities (item/token/hybrid).
- Token-level masking performs best as it provides finer-grained learning signals.
- Entropy-guided masking must be paired with curriculum learning to be effective—direct application leads to training instability.
- When recent interaction history is truncated, MHL demonstrates extreme robustness (+40%), proving it learns deep intentions rather than relying solely on recency effects.
- Phase III (Inference Alignment) is critical—skipping this stage leads to significant performance degradation.

## Highlights & Insights
- **Training Paradigm Innovation**: Introduces the spirit of masked language modeling into the autoregressive training of generative recommendation, shifting from "predicting results" to "understanding processes."
- **Deep Insight on "Understanding the past to predict the future"**: Robustness experiments (history truncation) convincingly demonstrate the value of deep historical understanding.
- **Sophisticated Combination of Entropy Guidance + Curriculum Learning**: The two are complementary—entropy guidance provides high-quality signals, while curriculum learning ensures stable training and inference alignment.
- **High Generality**: As a training framework, it can be directly applied to various semantic ID-based generative recommendation models.

## Limitations & Future Work
- **Limited Dataset Scale**: Validated only on three categories of Amazon Reviews 2014; testing on larger scales and more domains is needed.
- **Additional Computational Overhead**: Entropy-guided masking requires an extra gradient-free forward pass to calculate entropy at each step.
- **Hyperparameter Sensitivity**: Hyperparameters such as the transition points between the three curriculum phases and mask ratios require tuning.
- **Future Directions**: Application to LLM-based recommendation systems, cross-domain recommendation, and multi-modal recommendation.

## Related Work & Insights
- **vs BERT4Rec/S3-Rec**: Uses bidirectional encoders + masked prediction for discriminative recommendation; MHL adds a masked reconstruction auxiliary task to a unidirectional decoder to maintain generative capability.
- **vs TIGER/RPG**: SOTA generative recommendation using pure autoregressive next-item prediction; MHL significantly improves performance through auxiliary history reconstruction.
- **vs HSTU**: Facebook's industrial-scale sequential recommendation model; MHL outperforms it across all metrics.

## Rating
- Novelty: ⭐⭐⭐⭐ The innovation of introducing masked learning into the generative recommendation training paradigm is significant; the combination of entropy guidance and curriculum learning is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets, multiple baseline comparisons, detailed ablation studies, and robustness analysis.
- Writing Quality: ⭐⭐⭐⭐ The photography enthusiast example is intuitive, and the motivation for the method is clearly articulated.
- Value: ⭐⭐⭐⭐ Provides a new training paradigm for generative recommendation; the perspective of historical understanding is enlightening.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](../../AAAI2026/recommender/inductive_generative_recommendation_via_retrieval-based_speculation.md)
- [\[AAAI 2026\] Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](../../AAAI2026/recommender/align3gr_unified_multi-level_alignment_for_llm-based_generat.md)
- [\[AAAI 2026\] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](../../AAAI2026/recommender/from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)
- [\[AAAI 2026\] FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](../../AAAI2026/recommender/exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)

</div>

<!-- RELATED:END -->
