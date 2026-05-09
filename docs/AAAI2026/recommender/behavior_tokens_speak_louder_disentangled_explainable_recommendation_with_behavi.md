---
title: >-
  [Paper Note] Behavior Tokens Speak Louder: Disentangled Explainable Recommendation with Behavior Vocabulary
description: >-
  [AAAI2026][Recommender Systems][Explainable Recommendation] This paper proposes BEAT, a framework that discretizes user/item behavior representations into interpretable behavior tokens via vector-quantized autoencoders, and aligns collaborative filtering signals to the semantic space of a frozen LLM through multi-level semantic supervision, enabling zero-shot explainable recommendation.
tags:
  - AAAI2026
  - Recommender Systems
  - Explainable Recommendation
  - Behavior Tokenization
  - VQ-VAE
  - LLM
  - Disentangled Representation
date: 2026-05-08
content_hash: 1f0c9ce5f926d6f7
---

# Behavior Tokens Speak Louder: Disentangled Explainable Recommendation with Behavior Vocabulary

**Conference**: AAAI2026  
**arXiv**: [2512.15614](https://arxiv.org/abs/2512.15614)  
**Code**: [fxsxjtu/BEAT](https://github.com/fxsxjtu/BEAT)  
**Area**: Recommender Systems  
**Keywords**: Explainable Recommendation, Behavior Tokenization, VQ-VAE, LLM, Disentangled Representation  

## TL;DR
This paper proposes BEAT, a framework that discretizes user/item behavior representations into interpretable behavior tokens via vector-quantized autoencoders, and aligns collaborative filtering signals to the semantic space of a frozen LLM through multi-level semantic supervision, enabling zero-shot explainable recommendation.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **State of the Field**: Existing explainable recommendation methods face three core bottlenecks:

1. **Poor generalization of ID representations**: Traditional methods assign a unique ID embedding to each user/item, which completely fails for cold-start users and new items; even graph-based methods (e.g., XRec) that incorporate collaborative signals suffer from over-smoothing, undermining personalization.
2. **High computational cost**: LLM-based methods either inject large amounts of user profile text into prompts or require fine-tuning the LLM, both of which are prohibitively expensive.
3. **Modality fragmentation**: Most methods treat interaction history and review semantics in isolation, failing to unify their complementary information.

The authors observe that, despite diverse individual preferences, users share collective behavioral patterns (e.g., a tendency to seek value for money). This motivates representing users/items with a set of discrete behavior tokens, where similar entities share some tokens while unique combinations correspond to specific preference patterns.

### Starting Point

**Paper Goals**: How can user and item interaction behaviors be efficiently encoded into discrete token sequences that a frozen LLM can understand, enabling coherent recommendation explanations without fine-tuning?

## Method
BEAT consists of two stages: behavior vocabulary construction and LLM semantic alignment.

### Stage 1: Disentangled Behavior Modeling and Behavior Vocabulary Construction

**Disentangled Representation**: Each user representation is decomposed into one macro interest vector (a global preference unique to each user) and $N$ micro intention vectors (fine-grained attribute preferences shared across users, such as "durability" or "ease of use"). The concatenated representations are propagated through LightGCN to incorporate collaborative signals, and multi-layer averaging is applied to retain multi-hop information.

**VQ-VAE Discretization**: Two codebooks (size 512) are constructed for macro and micro representations, respectively, quantizing continuous representations to the nearest codeword. The reconstruction objective is to predict the user-item interaction matrix. The loss includes a reconstruction loss $\mathcal{L}_{\text{RECON}}$ and a VQ quantization loss $\mathcal{L}_{\text{VQ-VAE}}$.

**Multi-Level Semantic Supervision**:

- **Macro Semantic Supervision**: A frozen pretrained text encoder extracts review [CLS] features as supervision signals; the fused macro behavior tokens of users and items are aligned with review semantics via an InfoNCE contrastive loss.
- **Micro Semantic Supervision**: An LLM extracts interpretable micro intention phrases (e.g., "enjoys historical themes") from users' historical reviews. Since micro tokens are unordered and cannot be paired one-to-one, a masked reconstruction strategy is adopted—some micro intention embeddings are randomly masked, and a cross-attention module reconstructs the masked intentions from behavior tokens and the unmasked intentions.

Overall loss: $\mathcal{L}_{\text{tokenizer}} = \alpha \cdot \mathcal{L}_{\text{macro}} + \beta \cdot \mathcal{L}_{\text{micro}} + \mathcal{L}_{\text{behave}}$, where $\alpha=0.2, \beta=1$.

### Stage 2: LLM Behavior Token Understanding

**Projection Alignment**: A two-layer MLP maps behavior tokens into the input space of the frozen LLM, replacing the placeholder `<Tokens>` in the prompt.

**Semantic Alignment Regularization (SAR)**: The LLM's native vocabulary already encodes rich semantic associations (e.g., the relationship between "love" and "history books"). SAR maps each word in the explanation text to its nearest behavior token, then enforces consistency between the cosine similarities of behavior token pairs and those of the corresponding text word pairs, transferring the LLM's native semantic structure to the behavior token space.

**Joint Training**: NLL generation loss + SAR alignment loss; only the projection layer is trained while LLM weights remain frozen.

## Key Experimental Results

**Datasets**: Amazon (book reviews), Yelp (multi-category venues), Google (venue reviews). Evaluation metrics: BLEU-1, BARTScore, BERTScore.

**Zero-shot main results** (users/items have interactions but no review text):

### Main Results

| Method | Amazon BLEU | Amazon BART | Amazon BERT |
|------|------------|-------------|-------------|
| PETER (ID-based) | 0.3682 | -4.2300 | 0.1488 |
| XRec (LLM-based) | 0.2999 | -4.3210 | 0.3598 |
| TEA-GLM | 0.3971 | -4.1348 | 0.3406 |
| **BEAT** | **0.4195** | **-3.9929** | **0.3821** |

BEAT achieves the best performance across all three metrics on Amazon, and reaches state-of-the-art or highly competitive results on Google and Yelp.

### Ablation Study
- Removing micro tokens leads to a notable performance drop, confirming that fine-grained representations are critical.
- Removing macro tokens yields mixed results across datasets (degradation on Google, marginal improvement on Amazon/Yelp), suggesting that high-level summary tokens can sometimes interfere with the LLM's attention to fine-grained details.
- Removing SAR alignment causes significant degradation on complex domains (Yelp), underscoring the importance of semantic alignment in such settings.

**Robustness across LLMs**: BEAT is compatible with DeepSeek-8B, LLaMA3.1-8B, LLaMA3.2-3B, and Skywork-8B; the 3B model achieves performance close to the 8B models, demonstrating scalability to resource-constrained scenarios.

## Highlights & Insights
1. **Design philosophy of Behavior Vocabulary**: Users/items are represented as discrete token sequences—similar entities share some tokens while unique combinations correspond to specific preferences—balancing collectivity and personalization while naturally supporting cold-start scenarios (tokens can be borrowed from neighbors).
2. **Multi-level semantic supervision**: Macro alignment employs contrastive learning against global review semantics; micro alignment uses masked reconstruction for fine-grained intentions, elegantly resolving the unordered, unpaired nature of micro tokens.
3. **Lightweight + frozen LLM**: Only the tokenizer and projection layer are trained; the LLM remains fully frozen and the system runs on an RTX 3090. Behavior tokens are plug-and-play across different LLMs.
4. **Interpretability analysis**: Attention heatmaps reveal that the LLM dynamically shifts its focus across datasets (attending more to users on Amazon and to items on Yelp), confirming that the model genuinely understands token semantics rather than memorizing patterns.

## Limitations & Future Work
1. **Instability of macro tokens**: Ablation results show marginal improvements upon removing macro tokens on Amazon/Yelp, suggesting that global summarization may introduce noise in certain scenarios; adaptive gating mechanisms may be needed.
2. **LLM hallucination**: Generated user profiles are broadly consistent but exhibit partial hallucinations; the authors note this "can be mitigated by fine-tuning" without actually addressing it.
3. **Evaluation limitations**: Explanation quality is assessed solely via automatic metrics (BLEU/BART/BERTScore), with no human evaluation of explanation utility or trustworthiness.
4. **Dependence on LLM-extracted micro intentions**: Semantic labels for micro intentions require an LLM to extract them from reviews, which may limit effectiveness in review-sparse settings.
5. **Cross-domain transfer not validated**: Although mentioned as a future direction, the current experiments do not cover cross-domain zero-shot scenarios.

## Related Work & Insights
- **vs. PETER/NRT** (ID-based): BEAT replaces unique IDs with a shared behavior vocabulary, naturally supporting cold-start and generalization.
- **vs. XRec** (graph + LLM): XRec injects graph embeddings into the LLM but still modifies its structure; BEAT projects into the input space while keeping the LLM frozen, making it lighter and more transferable.
- **vs. Review-LLM/EXP3RT** (profile-based): These methods feed complete user profile text into prompts, incurring high computational costs and depending on review availability; BEAT compresses behavior into 6 tokens (1 macro + 5 micro), dramatically reducing prompt length.
- **vs. DGCF/DisenHAN** (disentangled recommendation): These methods perform coarse-grained preference modeling without interpretability or semantic alignment; BEAT augments disentanglement with multi-level semantic supervision to bridge collaborative signals and natural language.

## Related Work & Insights
- The behavior tokenization paradigm is generalizable to other domains: complex non-textual signals (temporal sequences, trajectories, physiological signals) can be compressed into discrete tokens that LLMs can understand, enabling a universal "signal → language" bridge.
- SAR regularization can be viewed as a form of knowledge distillation: transferring the semantic relational structure of the LLM vocabulary to new tokens, applicable to any scenario requiring the integration of external tokens into an LLM.
- The masked reconstruction approach for aligning unordered sets is broadly applicable: when one-to-one correspondence between two sets of representations cannot be established, contextual reconstruction can serve as an implicit alignment mechanism.
- The cold-start token assembly strategy (borrowing from semantic neighbors + correction via collaborative signals) provides a practical solution for zero-shot recommendation scenarios.
- The two-stage training paradigm (train tokenizer first, then train projection layer) can serve as a general template for injecting external structured knowledge into LLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ (The combination of behavior vocabulary, multi-level semantic supervision, and SAR alignment is reasonably novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three datasets, multiple LLM backbones, ablation studies, and interpretability analysis, though human evaluation is absent)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, well-motivated, intuitive figures)
- Value: ⭐⭐⭐⭐ (The lightweight, plug-and-play behavior tokenizer paradigm has practical promise)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation](when_top-ranked_recommendations_fail_modeling_multi-granular_negative_feedback_f.md)
- [\[NeurIPS 2025\] FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens](../../NeurIPS2025/recommender/face_a_general_framework_for_mapping_collaborative_filtering_embeddings_into_llm.md)
- [\[AAAI 2026\] Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation](wavelet_enhanced_adaptive_frequency_filter_for_sequential_re.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](inductive_generative_recommendation_via_retrieval-based_speculation.md)
- [\[AAAI 2026\] HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation](hymoerec_hybrid_mixture-of-experts_for_sequential_recommendation.md)

<!-- RELATED:END -->
