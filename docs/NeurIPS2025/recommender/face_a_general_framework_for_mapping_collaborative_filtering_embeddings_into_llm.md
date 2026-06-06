---
title: >-
  [Paper Note] FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens
description: >-
  [NeurIPS 2025][Recommender Systems][Collaborative Filtering] FACE proposes mapping collaborative filtering (CF) embeddings into LLM pre-trained tokens (descriptors) via disentangled projection and residual quantization…
tags:
  - "NeurIPS 2025"
  - "Recommender Systems"
  - "Collaborative Filtering"
  - "LLM Token Mapping"
  - "Vector Quantization"
  - "Contrastive Learning"
  - "Explainable Recommendation"
date: 2026-05-08
content_hash: 39098d270020f11a
---

# FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens

**Conference**: NeurIPS 2025
**arXiv**: [2510.15729](https://arxiv.org/abs/2510.15729)  
**Code**: [https://github.com/YixinRoll/FACE](https://github.com/YixinRoll/FACE)  
**Area**: Recommender Systems / LLM Alignment
**Keywords**: Collaborative Filtering, LLM Token Mapping, Vector Quantization, Contrastive Learning, Explainable Recommendation

## TL;DR
FACE proposes mapping collaborative filtering (CF) embeddings into LLM pre-trained tokens (descriptors) via disentangled projection and residual quantization, followed by contrastive learning for semantic alignment — enabling semantic interpretation and recommendation enhancement of CF embeddings without fine-tuning the LLM.

## Background & Motivation

**Background**: CF models (e.g., LightGCN) in recommender systems produce user/item embeddings as non-semantic latent vectors that LLMs cannot inherently interpret. Existing approaches either feed textual information directly into LLMs (e.g., TALLRec), or align CF embeddings with the LLM space via MLP/Q-former (e.g., ELM, RLMRec).

**Limitations of Prior Work**: Pure text-based methods lack collaborative signals, yielding inferior recommendation performance compared to traditional CF; space-alignment methods map embeddings near the LLM space but do not produce genuine tokens, so a frozen LLM cannot truly "read" these embeddings. BinLLM encodes CF information as numeric strings, but numerical tokens limit LLM comprehension and require fine-tuning.

**Key Challenge**: CF embeddings are continuous, entangled, and non-semantic; LLM tokens are discrete and semantically rich. A lossless continuous-to-discrete mapping is needed while preserving semantic consistency.

**Goal**: (a) Disentangle entangled CF embeddings into multi-aspect representations; (b) map continuous embeddings to discrete LLM tokens; (c) ensure that the mapped tokens are semantically consistent with user/item textual descriptions.

**Key Insight**: Use the LLM's pre-trained vocabulary (after semantic filtering) as a quantization codebook, quantizing CF embeddings to the nearest token embedding in the codebook, thereby directly producing LLM-readable "descriptors."

**Core Idea**: Leverage the LLM pre-trained vocabulary as a codebook, map CF embeddings to LLM tokens via disentangled projection and residual quantization, and align semantics through contrastive learning.

## Method

### Overall Architecture
CF model → user/item embedding $e$ → **Encoder** (multi-head projection disentangles into $n$ aspects → Transformer captures inter-aspect relationships) → **Residual Quantization** (frozen LLM vocabulary as codebook, RQ multi-layer approximation) → $n$ descriptor tokens → **Decoder** (Transformer + concat projection reconstructs original embedding) → **Contrastive Alignment** (descriptor embeddings vs. LLM-generated text summary embeddings via contrastive learning)

### Key Designs

1. **Disentangled Projection + Transformer Encoder**:

    - Function: Decompose CF embeddings into $n$ concept-specific sub-vectors.
    - Mechanism: $e_i = W_i e$, where $W_i$ is orthogonally initialized to project a single embedding into $n$ different subspaces. A Transformer encoder then captures dependencies among sub-vectors: $(z_{e_1}, ..., z_{e_n}) = \text{Transformer}_e(e_1, ..., e_n)$.
    - Design Motivation: CF embeddings entangle multi-aspect user preferences into a single vector; direct quantization loses fine-grained information. After disentanglement, each aspect independently maps to one descriptor token, preserving more complete preference information. Ablation studies show consistent performance gains as $n$ increases from 1 to 16.

2. **Residual Quantization (RQ) + Frozen LLM Codebook**:

    - Function: Quantize each $z_e$ into a token from the LLM vocabulary.
    - Mechanism: Codebook $C_0 = E_{LLM}(\mathcal{D})$ (meaningful vocabulary filtered by the COCA corpus), dimensionality-reduced via a trainable linear transform $C = W_c C_0$. RQ progressively quantizes residuals across layers: $r^{(h+1)} = r^{(h)} - c_k^{(h)}$. $C_0$ is frozen; only $W_c$ is trained (following SimVQ to prevent representation collapse). The first-layer quantization $c^{(1)}$ serves as the descriptor (primary information); subsequent layers capture residual details.
    - Design Motivation: Direct quantization into the LLM token space enables a frozen LLM to directly "read" these tokens. RQ achieves higher fidelity than single-layer VQ-VAE quantization.

3. **Contrastive Semantic Alignment**:

    - Function: Ensure descriptor semantics are consistent with user/item textual information.
    - Mechanism: An LLM embedding model encodes two inputs — (a) descriptors concatenated, restored to the original LLM space via inverse transform $W_c^{-1}$, then fed to the LLM to obtain $\mathbf{h}_d$; (b) LLM-generated user/item text summaries yielding $\mathbf{h}_s$. InfoNCE contrastive loss aligns positive pairs.
    - Design Motivation: Ablation results show a significant drop in recommendation metrics without contrastive alignment (the largest single-component degradation), indicating that quantization mapping alone does not guarantee semantic correctness — explicit textual signal injection is necessary.

### Loss & Training
- Three-stage curriculum training: Step 1 pre-trains the CF backbone → Step 2 trains the quantization autoencoder ($\mathcal{L}_{recons} + \mathcal{L}_Q$) → Step 3 jointly optimizes the full objective $\mathcal{L} = \mathcal{L}_\mathcal{R} + \mu \mathcal{L}_{map} + \lambda \mathcal{L}_{align}$.
- The LLM remains fully frozen throughout; only projection layers and the autoencoder are trained.
- LLaMA2-7B embedding model is used; Adam optimizer with lr=1e-2.

## Key Experimental Results

### Main Results (Recall@20 / NDCG@20 Improvement)

| Base Model | Amazon-book | Yelp | Steam |
|------------|-------------|------|-------|
| GMF | 0.1531 → 0.1553 | 0.1052 → 0.1120 | 0.1343 → 0.1411 |
| LightGCN | 0.1563 → **0.1622** | 0.1141 → **0.1203** | 0.1361 → **0.1439** |
| SimGCL | 0.1617 → **0.1670** | 0.1209 → **0.1225** | 0.1420 → **0.1487** |
| LightGCL | 0.1712 → **0.1759** | 0.1228 → **0.1253** | 0.1234 → 0.1238 |
| RLMRec | 0.1572 → 0.1581 | 0.1165 → 0.1196 | 0.1408 → 0.1432 |

Maximum gains: Amazon 7.31%, Yelp 11.55%, Steam 8.00%.

### Ablation Study

| Variant | Amazon R@20 | Amazon N@20 | Yelp R@20 | Yelp N@20 |
|---------|-------------|-------------|-----------|-----------|
| Full FACE | **0.1622** | **0.1009** | **0.1203** | **0.0766** |
| w/o Transformer | 0.1611 | 0.0994 | 0.1200 | 0.0762 |
| w/o Reconstruction Loss | 0.1586 | 0.0981 | 0.1191 | 0.0760 |
| w/o Contrastive Alignment | 0.1565 | 0.0962 | 0.1171 | 0.0741 |

### Key Findings
- Contrastive alignment is the most critical component — its removal causes the largest NDCG@20 drop (Amazon: 0.1009→0.0962, −4.7%).
- $n$=8–16 descriptors yields optimal performance; excessive descriptors introduce redundancy.
- In item retrieval tasks, the LLM can identify the target item from 10 candidates with >60% accuracy using only descriptors, validating the quality of semantic mapping.
- FACE can be applied on top of existing LLM-enhanced models (e.g., RLMRec) for further gains, demonstrating general plug-in capability.

## Highlights & Insights
- **Bridge from CF Embeddings to LLM Tokens**: FACE is the first to achieve direct mapping of CF embeddings to LLM pre-trained tokens, enabling a frozen LLM to "comprehend" the latent representations of recommendation models. This paradigm is transferable to other latent-space-to-LLM interpretation scenarios.
- **Decoder as a Generator**: The trained decoder can generate CF embeddings from a few keywords, opening a new application of "generating recommendation representations from textual descriptions."
- **LLM Vocabulary as Codebook**: Rather than learning a new codebook, FACE reuses the semantically rich LLM vocabulary directly, naturally yielding interpretability.

## Limitations & Future Work
- Recommendation performance gains are moderate (up to ~11%), with smaller improvements on models already enhanced by LLMs.
- The vocabulary filtered via COCA is limited in size and may not cover all semantic dimensions.
- Evaluation is conducted only under full-ranking protocols; online metrics such as CTR and conversion rate have not been tested.
- The three-stage training pipeline introduces considerable complexity.

## Related Work & Insights
- **vs. RLMRec**: RLMRec injects textual knowledge into CF via contrastive/generative alignment but does not produce discrete tokens; FACE maps to tokens directly interpretable by the LLM, offering stronger explainability.
- **vs. BinLLM**: BinLLM encodes CF information as numeric strings and requires LLM fine-tuning; FACE maps to meaningful English words that a frozen LLM can understand.
- **vs. ELM (adapter-based)**: ELM maps vectors near the token space via adapters but does not produce discrete tokens; FACE's RQ yields exact token matches.

## Rating
- Novelty: ⭐⭐⭐⭐ First model-agnostic framework mapping CF embeddings to LLM pre-trained tokens.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five base models × three datasets, including interpretability studies and ablations.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear with complete mathematical derivations.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for interpretability and LLM integration in recommender systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](../../AAAI2026/recommender/tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)
- [\[AAAI 2026\] Behavior Tokens Speak Louder: Disentangled Explainable Recommendation with Behavior Vocabulary](../../AAAI2026/recommender/behavior_tokens_speak_louder_disentangled_explainable_recommendation_with_behavi.md)
- [\[NeurIPS 2025\] Validating LLM-as-a-Judge Systems under Rating Indeterminacy](validating_llm-as-a-judge_systems_under_rating_indeterminacy.md)
- [\[NeurIPS 2025\] Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)
- [\[AAAI 2026\] Probabilistic Hash Embeddings for Online Learning of Categorical Features](../../AAAI2026/recommender/probabilistic_hash_embeddings_for_online_learning_of_categorical_features.md)

</div>

<!-- RELATED:END -->
