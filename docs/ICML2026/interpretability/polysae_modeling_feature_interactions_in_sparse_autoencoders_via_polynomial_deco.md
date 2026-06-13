---
title: >-
  [Paper Note] PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding
description: >-
  [ICML 2026][Interpretability][Sparse Autoencoders] PolySAE introduces second- and third-order polynomial terms based on shared low-rank projections into the standard Sparse Autoencoder (SAE) linear decoder. With minimal…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "Feature Interactions"
  - "Polynomial Decoding"
  - "Low-rank Tensor Decomposition"
  - "Compositionality"
date: 2026-05-08
content_hash: 7f68649bc402d6c9
---

# PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding

**Conference**: ICML 2026  
**arXiv**: [2602.01322](https://arxiv.org/abs/2602.01322)  
**Code**: https://github.com/pakoromilas/PolySAE (Yes)  
**Area**: Interpretability / Mechanistic Interpretability / Sparse Dictionary Learning  
**Keywords**: Sparse Autoencoders, Feature Interactions, Polynomial Decoding, Low-rank Tensor Decomposition, Compositionality

## TL;DR
PolySAE introduces second- and third-order polynomial terms based on shared low-rank projections into the standard Sparse Autoencoder (SAE) linear decoder. With minimal parameter overhead (~3% on GPT-2 small), it explicitly models multiplicative interactions between sparse features. Across 4 LLMs and 3 SAE variants, it improves mean probe F1 by approximately 8%, expands the 1-Wasserstein distance of class-conditional distributions by 2–10x, and enables causal steering of composite semantics using learned interaction directions.

## Background & Motivation

**Background**: Sparse Autoencoders (SAEs) are primary tools for mechanistic interpretability. They decode neural network activations $x$ from intermediate layers into a sparse linear combination of dictionary atoms: $\hat{x} = b + Dz$. Variants like TopK, BatchTopK, and Matryoshka have scaled dictionary sizes to millions of features, widely used to reveal safety-related concepts like deception or bias and to implement activation patching interventions.

**Limitations of Prior Work**: Existing SAEs are built on a "strong linear representation hypothesis," where features contribute only through additive superposition. This structure cannot distinguish between "composition" and "co-occurrence." For instance, when a model outputs activations related to "Starbucks," a linear SAE must either allocate a monolithic "Starbucks" feature (sacrificing atomicity) or use separate "star" and "coffee" features, failing to differentiate the specific "Starbucks" combination from a "star in a coffee shop."

**Key Challenge**: Atomic features (morphemes, conceptual primitives) and composite features ("administrators" = stem $\oplus$ suffix, "kick the bucket") naturally possess hierarchical relationships. Linear reconstruction mechanisms force both into the same flattened dictionary, violating core requirements of compositionality—represented by Smolensky’s (1990) Tensor Product Variable Binding theory—which requires multiplicative/bilinear binding to maintain atomicity while expressing composites.

**Goal**: To explicitly model high-order interactions between features within the SAE framework while (i) preserving linear encoders for interpretability, (ii) avoiding the $O(d_\text{sae}^2)$ or $O(d_\text{sae}^3)$ parameter explosion of naive tensor products, and (iii) maintaining compatibility with existing SAE variants like TopK, BatchTopK, and Matryoshka.

**Key Insight**: The decoder is formulated as a third-order Volterra expansion of $z$ (or a $\Pi$-net polynomial parameterization), with all high-order interactions constrained to a shared low-rank subspace $U$. Interactions across different orders are derived from different powers of the same "set of directions," ensuring semantic consistency while controlling parameter count.

**Core Idea**: Replace the "purely linear decoder" with a "linear encoder + shared low-rank + orthogonal polynomial decoder" to allow SAEs to express "multiplicative combinations" without losing reconstruction quality.

## Method

### Overall Architecture
PolySAE takes activations $x \in \mathbb{R}^d$ from a pretrained LLM as input and outputs sparse codes $z \in \mathbb{R}^{d_\text{sae}}$ and reconstruction $\hat{x}$. The encoder follows standard SAE design: $z = S(\text{ReLU}(E^\top x + b_\text{enc}))$, where $S$ is one of TopK, BatchTopK, or Matryoshka. The decoder is replaced by a third-order polynomial: $\hat{x} = b_\text{dec} + y_1 + \lambda_2 y_2 + \lambda_3 y_3$, where $y_1 = A z$ (linear term), $y_2 = B (z \otimes z)$ (pairwise), and $y_3 = \Gamma (z \otimes z \otimes z)$ (triple), with $\lambda_2, \lambda_3$ as learnable scalars. When $\lambda_2 = \lambda_3 = 0$, it strictly reduces to a linear SAE, making PolySAE a generalization of existing SAEs.

Naive implementations of $B$ and $\Gamma$ require $O(d_\text{sae}^2)$ and $O(d_\text{sae}^3)$ parameters. PolySAE compresses these using four design principles (P1 Linear encoder / P2 Polynomial decoder / P3 Factorized interaction / P4 Structural constraints) into a compact low-rank and orthogonal form.

### Key Designs

1. **Polynomial decoder + shared low-rank projection**:
    - **Function**: Inserts second- and third-order feature interactions into the decoder with minimal parameter cost without disrupting the linear encoder.
    - **Mechanism**: Sparse codes are first projected into a shared subspace $U$ of size $d_\text{sae} \times R_1$. High-order terms are then constructed via Hadamard products on the projected representation $zU$: $y_1 = (zU) C^{(1)\top}$, $y_2 = \big((zU_{:,1:R_2}) * (zU_{:,1:R_2})\big) C^{(2)\top}$, and $y_3 = \big((zU_{:,1:R_3})^{*3}\big) C^{(3)\top}$, where $*$ denotes element-wise multiplication and $C^{(k)} \in \mathbb{R}^{d \times R_k}$ are output projections. This implicitly defines the pairwise/triple dictionaries as $B = C^{(2)} (U_{:,1:R_2} \odot U_{:,1:R_2})^\top$ and $\Gamma = C^{(3)} (U_{:,1:R_3} \odot U_{:,1:R_3} \odot U_{:,1:R_3})^\top$ ($\odot$ is the Khatri–Rao product), which are mathematically equivalent but never explicitly materialized.
    - **Design Motivation**: Using a single $U$ instead of independent projectors for each order forces all interactions to be composites of the same set of features, ensuring interpretability and semantic consistency across orders. Empirical findings show $R_2 = R_3 \approx 0.06\text{–}0.11\, R_1$ is sufficient, suggesting high-order interactions are naturally low-dimensional.

2. **Nested Rank + Stiefel Orthogonalization**:
    - **Function**: Imposes a nested structure $R_1 \ge R_2 \ge R_3$ and $U^\top U = I$ on the low-rank projection to improve parsimony and identifiability.
    - **Mechanism**: Taking $R_2 = R_3 = 64$ (where $R_1 = d = 768$ for GPT-2 small), high-order terms are constructed using subsets of columns $U_{:,1:R_2} \subset U$, creating nested subspaces $\text{span}(U_{:,1:R_3}) \subset \text{span}(U_{:,1:R_2}) \subset \text{span}(U)$. This aligns with polynomial approximation theory where lower orders have higher expressivity. After each gradient update, $U$ is projected back to the Stiefel manifold using QR retraction (positive QR to ensure continuity), enforcing $U^\top U = I$ to remove rotational ambiguity.
    - **Design Motivation**: Nested low-rank structures allow lower orders to capture more capacity. Orthogonalization prevents redundant overlaps in high-order interaction directions—ablation studies show a ~3pp drop in F1 without it.

3. **Context-dependent implicit dictionary**:
    - **Function**: Allows the effective contribution of a feature to the reconstruction to change based on which other features are co-activated, separating compositionality from atomicity.
    - **Mechanism**: Expanding the polynomial terms, the linear term $A$ corresponds to single-feature dictionary atoms. The pairwise dictionary $B$ columns describe how the reconstruction is modified when $z_i z_j$ co-activate. A single set of $d_\text{sae}$ atomic features can support $\binom{d_\text{sae}}{2} R_2 + \binom{d_\text{sae}}{3} R_3$ expressible combinations via shared interaction directions.
    - **Design Motivation**: Standard SAEs must allocate a new atom for each composite concept (breaking compositionality). PolySAE allows "multiplicative binding" (e.g., star $\times$ coffee $\to$ Starbucks) without increasing dictionary size, while the linear encoder ensures each $z_i$ remains a clear projection direction of $x$ for visualization or activation patching.

### Loss & Training
Reconstruction loss follows the default MSE in SAELens. Sparsity is hard-constrained by the $S$ operator (TopK/BatchTopK/Matryoshka) with $K = 64$ and $d_\text{sae} = 16{,}384$. Training uses 500M tokens (300M for GPT-2 small) with a context length of 128 (OpenWebText for GPT-2/Gemma, Pile for Pythia). $U$ is updated via QR retraction, while $\lambda_2, \lambda_3$ are optimized alongside the network.

## Key Experimental Results

### Main Results
Evaluations cover 4 LLMs $\times$ 3 sparsifiers (12 configurations). Metrics include MSE, cross-entropy (CE) recovery, F1 scores on 6 probe tasks (Bias in Bios, AG News, EuroParl, GitHub, Amazon Sentiment, Amazon-15), and 1-Wasserstein distance of class-conditional distributions.

| Model | Sparsifier | MSE (SAE→Ours) | CE Rec. | Mean F1 (SAE→Ours) | Wasserstein Gain |
|------|------------|----------------|---------|--------------------|------------------|
| GPT-2 Small | TopK | 0.52 → 0.55 | 0.993 | 67.1 → 77.9 (+10.8) | ~2–4× |
| GPT-2 Small | BatchTopK | 0.53 → 0.54 | 0.993 | 65.7 → 78.0 (+12.3) | ~2–4× |
| GPT-2 Small | Matryoshka | 0.60 → 0.58 | 0.992 | 65.7 → 77.7 (+12.0) | ~2.4× |
| Pythia-410M | TopK | 0.03 → 0.04 | 0.971 | 71.2 → 77.0 (+5.8) | ~3–5× |
| Pythia-1.4B | TopK | 0.23 → 0.23 | 0.973 | 75.9 → 81.9 (+6.0) | ~4–5× |
| Gemma-2-2B | BatchTopK | 1.58 → 1.68 | 0.987 | 64.8 → 69.4 (+4.6) | ~5–10× |

All 12 configurations show CE recovery shifts $< 0.003$, proving that small MSE differences do not cause functional degradation. Mean probe F1 improved by ~8%. The consistent 2–10× improvement in Wasserstein distance indicates that F1 gains stem from geometrically more separated semantic structures rather than lucky decision boundaries.

### Ablation Study

| GPT-2 Small Configuration | Params | MSE | F1 |
|------------------|--------|-----|----|
| Polynomial + Shared projector (No low-rank, No orthogonal) | 37.7M | 0.58 | 76.0 |
| + Low-rank decomposition (P3) | 13.3M | 0.53 | 75.0 |
| + Orthogonalization (P4, Full PolySAE) | 13.3M | 0.55 | 77.9 |

Low-rank decomposition reduces parameters by 65% with only a 1pp loss in F1. Adding orthogonalization (P4) recovers this loss and gains +2.9pp at zero parameter cost, confirming P3 handles tractability while P4 handles identifiability.

### Key Findings
- The learned second-order interaction strength $B_{ij}$ is almost uncorrelated with co-occurrence frequency $N_{ij}$ ($r = 0.06$), whereas vanilla SAE activation covariance correlates highly with co-occurrence ($r = 0.82$). This suggests polynomial terms capture structural composition rather than surface statistics.
- GPT-4o-mini scoring on 70,000 pairs shows 12% of high-interaction pairs reach interpretability scores $0.9+$. At least 8,550 new interpretable second-order composite concepts were found in GPT-2 small.
- Activation steering (injecting $\alpha(d_i + d_j)$ into layer 8) showed PolySAE outperformed vanilla SAE in 21/27 composite concepts, with a target token rank improvement of +41.5. Cosine similarity to the "ground-truth" difference-in-means direction was $0.372 \pm 0.093$ (Ours) vs $0.311 \pm 0.158$ (Vanilla).
- Semantic concentration: When expanding from K=1 to K=5, PolySAE's F1 gain is smaller than vanilla (by 7–8pp on GPT-2), suggesting semantic signals are compressed into fewer linear features, as high-order interactions absorb contextual variation.

## Highlights & Insights
- **Generalized Design**: Setting $\lambda_2 = \lambda_3 = 0$ recovers standard SAEs, allowing PolySAE to be "plugged-in" to extend the expressivity of any existing SAE variant.
- **Semantic Consistency via Shared $U$**: Deriving all orders from the same $zU$ anchors high-order interactions in linear feature semantics—a technique applicable to any model (e.g., bilinear MLPs) seeking to preserve low-order interpretability with high-order expressivity.
- **The r = 0.06 vs r = 0.82 Comparison**: This correlation analysis elegantly rebuts the null hypothesis that high-order terms merely capture bigram statistics, providing a high-quality interpretability diagnostic.
- **Closure of the Steering Loop**: Triggering composite outputs (e.g., "Starbucks") via simple vector addition $d_i + d_j$ demonstrates that PolySAE moves beyond static visualization toward functional model steering.

## Limitations & Future Work
- The largest model tested was Gemma-2-2B; scaling behavior on 7B+ models remains unknown.
- Only hard-sparsity (TopK-style) variants were evaluated; soft-sparsity variants like Gated or JumpReLU SAEs were not covered.
- $\lambda_2, \lambda_3$ are global scalars; per-feature or per-layer adjustments were not explored.
- The interaction dictionary evaluation only covered 24% of candidates; whether the 12% interpretability ratio scales with model size is an open question.
- Steering experiments were limited to 27 concepts; the breadth of compositionality and potential side effects on unrelated concepts require further quantification.

## Related Work & Insights
- **vs Bilinear Autoencoder (BAE, Dooms & Gauderis 2025)**: BAE models interactions at the input neuron level; PolySAE models interactions at the "learned sparse feature" level, preserving latent interpretability while explicitly modeling non-additive compositions.
- **vs Bilinear MLPs (Pearce et al. 2025)**: Pearce uses multiplication for weight-based interpretability within MLPs; PolySAE applies this to the dictionary learning side, integrating naturally with mechanistic interpretability pipelines.
- **vs Tensor Product Variable Binding (Smolensky 1990)**: Conceptually similar, but PolySAE’s low-rank shared $U$ provides an engineering path to implement this theory at modern LLM scales.
- **vs $\Pi$-nets / Volterra series**: PolySAE reinterprets the inductive bias of polynomial networks through the lens of interpretability—polynomials naturally provide a two-tier dictionary of atoms and composites.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to introduce explicit high-order interactions into SAE dictionary learning that strictly generalizes existing variants.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage of 4 LLMs, multiple sparsifiers, and various metrics (geometric, probing, causal steering); limited only by model scale (2B).
- Writing Quality: ⭐⭐⭐⭐⭐ Design principles clearly map to architecture choices; the implicit dictionary expansion and the "Starbucks" motif make the concepts highly accessible.
- Value: ⭐⭐⭐⭐⭐ Provides a plug-and-play enhancement for the SAE ecosystem and brings "compositionality"—a classic linguistic/cognitive science topic—back to the interpretability agenda in a quantifiable way.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders](on_the_relationship_between_activation_outliers_and_feature_death_in_sparse_auto.md)
- [\[ICML 2026\] Sparse Autoencoders are Topic Models](sparse_autoencoders_are_topic_models.md)
- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](../../ICLR2026/interpretability/toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](../../NeurIPS2025/interpretability/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [\[ICLR 2026\] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](../../ICLR2026/interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)

</div>

<!-- RELATED:END -->
