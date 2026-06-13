---
title: >-
  [Paper Note] Learning Coherent Representations: A Topological Approach to Interpretability
description: >-
  [ICML 2026][Interpretability][coherence] This paper proposes **coherence**, a geometric property inspired by neural coding in the brain…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "coherence"
  - "Vietoris-Rips"
  - "Fréchet variance"
  - "barycentric map"
  - "sparse autoencoder alternative"
date: 2026-05-08
content_hash: 7515a6cd5a41151f
---

# Learning Coherent Representations: A Topological Approach to Interpretability

**Conference**: ICML 2026  
**arXiv**: [2606.02841](https://arxiv.org/abs/2606.02841)  
**Code**: To be confirmed  
**Area**: Interpretability / Topological Data Analysis / Representation Learning  
**Keywords**: coherence, Vietoris-Rips, Fréchet variance, barycentric map, sparse autoencoder alternative

## TL;DR
This paper proposes **coherence**, a geometric property inspired by neural coding in the brain, which requires the rows and columns of the sample-feature matrix to be topologically interleaved under Vietoris-Rips filtration. By introducing a differentiable `Coh` loss, the authors achieve topologically aligned and semantically readable features in autoencoders and BERT token embeddings, significantly outperforming $L^1$ sparsity.

## Background & Motivation

**Background**: The mainstream approach to interpretability is sparsification—using $L^1$ or other sparse regularizations in sparse coding and sparse autoencoders (Bricken 2023, Cunningham 2024) to ensure each feature activates on only a few samples, thereby mitigating polysemanticity. Another approach is the mechanistic route, which treats latent dimensions as individual "concepts."

**Limitations of Prior Work**: Sparsity only constrains "how many samples activate" but not "which samples activate." A sparse feature could simultaneously activate across several unrelated, spatially scattered regions on the data manifold, resulting in low activation count but lacking any interpretable geometric meaning. This problem is more severe in unsupervised settings where no categorical labels exist to pull similar samples together, leaving the feature space with almost no readable geometric structure.

**Key Challenge**: The essence of interpretability is not "activation sparsity" but "activation region connectivity." The reason brain grid cells or head direction cells allow direct reading of an animal's position or orientation is that the activation region of each neuron is a **continuous arc or connected component** in the state space. This is a matter of locality rather than rarity. Existing deep learning regularizations lack mechanisms to guarantee such connectivity.

**Goal**: (1) Provide a geometric definition for "interpretable features"; (2) transform this definition into a differentiable loss applicable to any network with non-negative activations; (3) validate this in both autoencoders (with known topology) and BERT token embeddings (without ground-truth topology).

**Key Insight**: Treat the latent matrix $M \in \mathbb{R}_+^{m \times n}$ simultaneously as two sets of weighted point clouds: "samples $\to$ features" and "features $\to$ samples." Drawing an analogy to **Dowker duality**, if two spaces approximately cover each other via "barycentric maps" with low variance, their Vietoris-Rips filtrations must interleave, making them topological mirrors of each other.

**Core Idea**: Use Fréchet variance plus the "round-trip error" of barycentric maps as a differentiable loss to constrain the sample and feature spaces to be topologically equivalent point clouds. Consequently, feature interpretability becomes the "inheritance of sample space geometry by the feature space."

## Method

### Overall Architecture

The input consists of any network with non-negative activations (e.g., autoencoder bottleneck or BERT token embedding followed by Softplus). Latents within a batch are extracted to form a non-negative matrix $M \in \mathbb{R}_+^{B \times L}$ (rows = samples $r_i$, columns = features $c_j$). The pipeline is:

1. Normalize rows and columns into probability vectors $w^{(i)}, v^{(j)}$ using squared-$L^1$.
2. Compute linear closed-form barycentric maps $\phi(r_i)=w^{(i)}M^T$ (samples $\to$ barycenter in column space) and $\psi(c_j)=v^{(j)}M$ (features $\to$ barycenter in row space).
3. Calculate two metrics for each row/column: **Fréchet variance** (locality) and **covering**. Sum the parts exceeding a threshold $\tau$ using a top-$k$ operation to form the `Coh` loss.
4. Combine with the original task loss (MSE/MLM): $\mathcal{L}=\mathcal{L}_{\text{task}}+\lambda_{\text{Coh}}\mathcal{L}_{\text{Coh}}$, typically with $\lambda_{\text{Coh}}=10^{-3}$.

Theoretically, when $M$ is $\epsilon$-coherent and $\phi, \psi$ are 1-Lipschitz, an $\epsilon^{1/2}$-interleaving exists, meaning the Vietoris-Rips filtrations and persistence diagrams of samples and features are close in bottleneck distance.

### Key Designs

1. **Dual Definition of Coherence = Locality + Covering**:
    - **Function**: Quantifies whether the sample-feature matrix is topologically aligned using two scalars.
    - **Mechanism**: Row Fréchet variance $\text{Var}_\mathcal{R}(r_i)=\sum_j w^{(i)}_j\|\phi(r_i)-c_j\|_2^2$ measures if the columns selected by row $r_i$ are clustered in column space (locality). Row covering $\text{Cov}_\mathcal{R}(r_i)=\sum_j w^{(i)}_j\|r_i-\psi(c_j)\|_2^2$ measures if the barycenter $\psi(c_j)$ of some column $c_j$ is close to $r_i$ (covering). Symmetrical definitions apply to columns. $\epsilon$-coherence requires both metrics to be $\le\epsilon$ for all rows and columns.
    - **Design Motivation**: Locality alone is insufficient, as it may lead to degenerate solutions where features are compact but fail to cover the sample manifold. Covering alone is also insufficient, as every sample might be described by a feature even if the features themselves are scattered. Together, they guarantee $\epsilon^{1/2}$-interleaving (Theorem 3.12), a geometric guarantee that sparse regularization cannot provide.

2. **Squared-$L^1$ Normalization + Closed-form Barycenter**:
    - **Function**: Ensures the "averaging" operation remains differentiable throughout backpropagation without requiring iterative optimization.
    - **Mechanism**: By choosing the Euclidean norm and squared-$L^1$ normalization $W_{ij}=M_{ij}^2/\sum_k M_{ik}^2$, the barycenter $\arg\min_\mu\sum_j w^{(i)}_j\|\mu-c_j\|_2^2$ simplifies to the closed-form $\phi(r_i)=w^{(i)}M^T$. This also allows the deviation between the "snapping map" (projection to the nearest real row/column) and the soft barycenter to be bounded by locality (Prop 3.9: $\|\phi(r_i)-\Phi(r_i)\|_2\le\epsilon^{1/2}$).
    - **Design Motivation**: Topological guarantees require "mapping to real columns" to define interleaving, but training requires a differentiable soft mapping. Closed-form barycenters with squared distances keep the two within $\epsilon^{1/2}$, allowing training with a soft loss while enjoying hard guarantees.

3. **Top-$k$ + Threshold Aggregation + Paired Scale Normalization**:
    - **Function**: Aggregates $O(B+L)$ per-row/per-col metrics into a single loss while eliminating biases caused by scale differences between row and column spaces.
    - **Mechanism**: Variance and covering are normalized to be dimensionless using the average distance of all row/column pairs $\bar d_R, \bar d_C$. A hinge $[\cdot-\tau]_+$ is applied (to avoid penalizing rows/columns already meeting the target $\tau$), followed by a sum over the worst top-$k_R, k_C$ instances. $\mathcal{L}_{\text{Coh}}=\text{TopK}(\text{Var-related})+\text{TopK}(\text{Cov-related})$.
    - **Design Motivation**: In non-square $B \times L$ matrices, the scales of row and column point clouds can differ by orders of magnitude; lack of normalization would cause the loss to be dominated by one side. Top-$k$ + thresholding is crucial for stable training, focusing optimization on the least coherent rows/columns to prevent gradient dilution.

### Loss & Training
Task loss + $\lambda_{\text{Coh}}\mathcal{L}_{\text{Coh}}$, with an optional $\lambda_{L^1}\|M\|_1$ to favor sparse coherent solutions among multiple possibilities. For MNIST autoencoders, $\lambda_{\text{Coh}}=\lambda_{L^1}=10^{-3}$; for the toy double-circle, a smaller $\lambda_{\text{Coh}}=10^{-5}$ is used. In BERT, token embeddings are projected to non-negative space via Softplus ($\beta=20$) before applying `Coh`, with the main task remains MLM with 15% masking. The 1-Lipschitz assumption is checked via post-hoc sampling rather than enforced (measured $\psi$ violation rate <0.2%, $\phi \approx 3$-$4\%$, average expansion coefficient $\approx 1.05$).

## Key Experimental Results

### Main Results

**Toy Double-Circle Data** (Two disjoint circles in $\mathbb{R}^{512}$, 20k samples) — Single seed results:

| Model | MSE | %Tuned (MRL>0.5) | %Pure (compscore>0.5) | Locality / Cov |
|------|-----|------------------|-----------------------|----------------|
| Vanilla | 9.96e-5 | 43.0% | 0.0% | 0.53 / 0.44 |
| L1 | 9.95e-5 | 52.0% | 0.0% | 4.62 / 4.58 |
| **Coh** | **9.94e-5** | **100.0%** | **90.2%** | **0.14 / 0.14** |

**BERT Token Embedding** (256 dims, 2 transformer blocks, WikiText-2, average of 5 seeds):

| Metric | Coh | Softplus baseline |
|------|-----|-------------------|
| Mean Overlap with Vanilla geometry | 0.45±0.01 | 0.22±0.00 |
| Num features with Overlap > 0.5 / 256 | 77.4±3.3 | 1.0±0.6 |
| Claude scoring (Interpretable) / 256 | **87.6±10.4** | **0.0±0.0** |

### Ablation Study

| Configuration | %Tuned | Locality | Description |
|------|--------|----------|------|
| Coh full | 100% | 0.14 | Locality + Covering combined |
| L1 Sparsity only | 52-63% | 3.67-4.62 | Sparsity does not induce geometric clustering |
| Vanilla (No Reg) | 43% | 0.53 | Random topology |
| Coh + L1 (double digits) | Stable within class | 0.15 | L1 pushes Coh toward the sparsest coherent solution |
| Softplus only (BERT) | 1/256 ≈ 0% | — | Non-negativity is insufficient |

### Key Findings

- **Locality and Covering are extremely stable across seeds** (std near 0), indicating the `Coh` loss reliably achieves geometric objectives. High variance in MRL/Purity isn't a failure but reflects that coherence allows both "split classes" and "merged classes" as valid solutions.
- **Sparsity vs. Coherence**: $L^1$ locality on toy data spiked to 4.62 (worse than vanilla), proving that sparsification can actually destroy geometric clustering. Coh reduced locality to 0.14, an order of magnitude improvement.
- **Softplus-only on BERT yielded almost zero interpretable features, while Coh achieved 87.6/256**. This suggests non-negativity is only a prerequisite; coherence is the sufficient ingredient. Claude identified 87 features spanning categories like "years/kinship/place names/units/adverbs/prepositions," showing the loss does not rely on specific data priors.
- **1-Lipschitz assumption is soft**: Measured average expansion of $\phi, \psi$ was $\approx 1.05$ with low violation rates. Experimental results still align with the geometric guarantees of Theorem 3.12, suggesting that deviations from this ideal assumption are not fatal to practical conclusions.

## Highlights & Insights
- **Reframes "feature interpretability" as a geometric/topological problem**: interpretability $\equiv$ interleaving of sample and feature point clouds. This provides a differentiable proxy that requires neither human evaluation nor labels.
- **Geometric realization of Dowker duality**: The original Dowker theorem is combinatorial (homotopy of row/column complexes); this work converts it to $\epsilon^{1/2}$-interleaving in metric spaces. The mapping from theory to loss via Fréchet variance and covering is remarkably clean.
- **Squared-$L^1$ for closed-form barycenters**: A valuable technique for any differentiable attention or aggregation requiring "soft-to-hard projection," avoiding EM-style inner loops.
- **Top-$k$ + threshold aggregation**: An engineering trick to stabilize per-element geometric losses. It is applicable to almost any instance-wise regularization and performs better than simple summation or averaging.

## Limitations & Future Work
- The 1-Lipschitz assumption is checked post-hoc rather than strictly enforced; theoretical bounds might not strictly hold (~5% probability). Incorporating spectral norm penalties into $\phi, \psi$ is a potential improvement.
- Experiments were limited to autoencoder bottlenecks and token embeddings. Effectiveness across multiple layers, ResNet/Transformer intermediate layers, or supervised classification remains unverified. (The authors list supervised settings as future work, noting that cross-entropy often collapses intra-class structures, leaving little room for coherence).
- Squared Euclidean distance might degenerate (distance concentration) in high-dimensional latents; the authors acknowledge this as a cost of using Euclidean metrics. Scalability behavior for $L \ge 1024$ without ablation.
- Risk of multiple solutions: A task may have several coherent solutions (separate vs. merged). Auxiliary losses like $L^1$ are needed to select among them, as there is no native disentanglement mechanism.
- Computational overhead: Constructing pairwise distances and barycentric matrices takes $O(B^2+L^2+BL\cdot\max(B,L))$ per batch. Applying this to large LMs would require downsampling or chunking strategies.

## Related Work & Insights
- **vs. Sparse AE / Dictionary Learning (Bricken 2023, Cunningham 2024)**: They focus on how many features activate; this work focuses on which features activate (geometric connectivity). Toy experiments showing $L^1$ locality=4.62 vs. Coh=0.14 debunk the implicit assumption that "sparsity $\Rightarrow$ interpretability." The two are orthogonal and can be combined.
- **vs. Topological Autoencoder (Moor 2020) / Connectivity-preserving (Hofer 2019)**: They maintain consistency between input and latent topology, requiring selection of homology degrees and differentiable persistence. This work makes the "row view" and "column view" of the latent space mirrors of each other, working at the simplicial filtration level without picking specific homology degrees.
- **vs. Similarity-preserving Networks (Sengupta 2018)**: They achieve localized receptive fields via non-negative similarity preservation, but only in one direction (input $\to$ feature). This work requires bidirectionality (features must also be covered by samples), leading to the stronger conclusion that the feature space itself has meaningful geometry.
- **vs. Neuroscience grid/head-direction cells (Hafting 2005, Gardner 2022)**: The direct inspiration—formalizing "neuron activation regions as connected components of state space" as locality, and "any state having a neural response" as covering. This could inspire future geometric regularizations for complex biological encodings like place cell multi-fields or boundary cells.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to redefine feature-space interpretability as a sample-feature topological interleaving problem with a rigorous differentiable loss.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested on Toy, Rotated MNIST, and BERT; uses both Claude scoring and topological metrics. However, scale is limited to small models without GPT-2/Pythia level validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Tight integration of theory, algorithm, and experiments; figures directly support the core thesis of "geometric feature space."
- Value: ⭐⭐⭐⭐ Provides a new regularization paradigm for mechanistic interpretability that complements sparsity and can be directly integrated into existing SAE pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MUSE: Resolving Manifold Misalignment in Visual Tokenization via Topological Orthogonality](muse_resolving_manifold_misalignment_in_visual_tokenization_via_topological_orth.md)
- [\[ICML 2026\] IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension](idest_assessing_self-supervised_learning_representations_via_intrinsic_dimension.md)
- [\[ACL 2026\] A Structured Clustering Approach for Inducing Media Narratives](../../ACL2026/interpretability/a_structured_clustering_approach_for_inducing_media_narratives.md)
- [\[ICML 2026\] Interpretability Can Be Actionable](interpretability_can_be_actionable.md)
- [\[NeurIPS 2025\] Representation Consistency for Accurate and Coherent LLM Answer Aggregation](../../NeurIPS2025/interpretability/representation_consistency_for_accurate_and_coherent_llm_answer_aggregation.md)

</div>

<!-- RELATED:END -->
