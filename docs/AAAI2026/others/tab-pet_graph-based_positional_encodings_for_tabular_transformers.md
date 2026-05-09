---
title: >-
  [Paper Note] Tab-PET: Graph-Based Positional Encodings for Tabular Transformers
description: >-
  [AAAI 2026][Positional Encoding] Tab-PET estimates a graph structure from inter-feature correlations in tabular data, constructs positional encodings (PE) from graph Laplacian eigenvectors, and injects them into tabular Transformers. Both theoretical analysis and experiments demonstrate that PE reduces the effective rank of embeddings, thereby improving generalization. Consistent improvements are observed across 50 datasets for TabTransformer, SAINT, and FT-Transformer, with the Spearman correlation graph yielding the best results.
tags:
  - AAAI 2026
  - Positional Encoding
  - Tabular Data
  - Graph Laplacian
  - Transformer
  - Effective Rank
date: 2026-05-08
content_hash: 371f1f30263d15ef
---

# Tab-PET: Graph-Based Positional Encodings for Tabular Transformers

**Conference**: AAAI 2026
**arXiv**: [2511.13338](https://arxiv.org/abs/2511.13338)
**Code**: [https://github.com/kentridgeai/Tab-PET](https://github.com/kentridgeai/Tab-PET)
**Area**: Tabular Data Learning / Transformer
**Keywords**: Positional Encoding, Tabular Data, Graph Laplacian, Transformer, Effective Rank

## TL;DR

Tab-PET estimates a graph structure from inter-feature correlations in tabular data, constructs positional encodings (PE) from graph Laplacian eigenvectors, and injects them into tabular Transformers. Both theoretical analysis and experiments demonstrate that PE reduces the effective rank of embeddings, thereby improving generalization. Consistent improvements are observed across 50 datasets for TabTransformer, SAINT, and FT-Transformer, with the Spearman correlation graph yielding the best results.

## Background & Motivation

**Background**: Tabular data is one of the most prevalent data modalities in machine learning, where GBDTs (XGBoost, CatBoost) have long been dominant. In recent years, Transformer-based architectures such as TabTransformer, SAINT, and FT-Transformer have made notable progress on tabular tasks, yet they have not consistently surpassed GBDTs overall.

**Limitations of Prior Work**: Images possess spatial locality and text has sequential order—Transformers in these modalities can exploit PE to inject inductive biases. In contrast, tabular data features are **arbitrarily ordered and lack natural structural priors**. Existing tabular Transformers universally omit PE, reflecting a community consensus that "tabular data has no structure, hence PE is useless."

**Key Challenge**: Tabular data presents a triple challenge: (a) scarce samples, (b) high-dimensional heterogeneous features, and (c) absence of structural priors. Without PE, self-attention treats all features as an unordered set of fully equivalent tokens, failing to leverage latent inter-feature correlation structure to simplify the learning task.

**Goal**: Can meaningful positional encodings be constructed for tabular Transformers? Can PE genuinely improve generalization? If so, from what perspective should PE be constructed?

**Key Insight**: The authors ground their approach in a theoretical analysis of **effective rank**, showing that PE can reduce the effective rank (intrinsic dimensionality) of CLS token output embeddings—equivalent to reducing the dimensionality of the learning problem and thus improving generalization. When PE aligns with the actual data structure, the reduction in effective rank is more pronounced.

**Core Idea**: Extract Laplacian eigenvectors from an estimated feature correlation graph as fixed PE, inject them into tabular Transformers, and exploit the effective-rank-reducing property of PE to strengthen generalization.

## Method

### Overall Architecture

The Tab-PET pipeline consists of four steps:
**Input** → (1) Data preprocessing (one-hot encoding for categorical features + standardization for continuous features) → (2) Feature-level graph estimation (each feature is a node; edge weights reflect inter-feature correlations) → (3) Extraction of eigenvectors from the graph Laplacian to construct PE → (4) Concatenation of PE with original embeddings before feeding into Transformer layers → **Output** predictions.

The pipeline does not modify the internal Transformer architecture; PE concatenation is added only at the embedding layer, making it a plug-and-play augmentation scheme.

### Key Designs

1. **Graph Estimation**

    - **Function**: Construct a graph over the feature dimension, where each feature is a node and edge weights capture statistical dependencies or causal relationships between features.
    - **Mechanism**: Two families of graph estimation paradigms are explored:
        - **Causal graphs**: Assume a linear structural equation model $\mathbf{x} = \mathbf{W}\mathbf{x} + \boldsymbol{\epsilon}$; directed acyclic graphs are learned via LiNGAM (exploiting non-Gaussianity to identify causal direction) or NOTEARS (continuous optimization with acyclicity constraints).
        - **Correlation graphs**: Directly construct edge weights using pairwise statistics $w_{ij} = \rho(x_i, x_j)$, where $\rho$ can be Pearson correlation, Spearman rank correlation, or mutual information (the Chow-Liu algorithm guarantees a tree-structured DAG).
    - **Design Motivation**: Inter-feature correlation structure exists implicitly in tabular data (e.g., income and expenditure are highly correlated in financial data). Graph estimation makes this implicit structure explicit. Experiments show that correlation graphs outperform causal graphs—causal graphs are too sparse (low graph entropy), whereas correlation graphs are denser and capture richer feature dependencies.

2. **PE Creation**

    - **Function**: Extract eigenvectors from the Laplacian of the estimated graph to serve as the positional encoding for each feature.
    - **Mechanism**: The adjacency matrix is symmetrized as $\mathbf{A}_{\text{sym}} = \frac{1}{2}(\mathbf{A} + \mathbf{A}^\top)$; the graph Laplacian is computed as $\mathbf{L} = \mathbf{D} - \mathbf{A}_{\text{sym}}$; the first $k$ and last $k$ eigenvectors (excluding the constant first eigenvector) are extracted, normalized, and concatenated to form the PE matrix $\mathbf{P} = [\mathbf{e}_2, \dots, \mathbf{e}_{k+1}, \mathbf{e}_{d-k+1}, \dots, \mathbf{e}_d]$, then scaled by a factor $\mathbf{P}' = \alpha \cdot \mathbf{P}$.
    - **Design Motivation**: Low-frequency eigenvectors capture global graph structure (similar features receive similar encodings), while high-frequency eigenvectors capture local differences (distinguishing fine-grained distinctions between closely related nodes). Their combination is effective on both homophilic and heterophilic graphs. $k$ is selected automatically via a spectral gap-based adaptive algorithm, and $\alpha$ is greedily searched over 9 candidate values on a validation set.

3. **PE Integration**

    - **Function**: Concatenate PE with the feature embeddings of the Transformer.
    - **Mechanism**: For each feature $x_i$, its original embedding $\mathbf{z}_i$ is concatenated with the scaled PE $\mathbf{p}_i'$ as $\mathbf{z}_i' = [\mathbf{z}_i; \mathbf{p}_i'] \in \mathbb{R}^{n+2k}$, which is then fed into the self-attention layer. For the multiple one-hot nodes of categorical features, the mean of their respective PE vectors is used as the unified encoding for that feature.
    - **Design Motivation**: Concatenation rather than addition preserves the original embedding information without being overwritten by PE, while allowing the model to learn through attention how to utilize positional information.

### Theoretical Motivation: PE and Effective Rank

The authors theoretically demonstrate the ability of PE to reduce effective rank. Effective rank is defined as the exponential of the Shannon entropy of the singular value distribution of the CLS embedding matrix:

$$r_{\text{eff}}(\mathbf{X}) = \exp\left(-\sum_{i=1}^{r} \tilde{\sigma}_i \log \tilde{\sigma}_i\right)$$

- **Theorem 1 (Random Input)**: Even when features are i.i.d., PE reduces effective rank, with an upper bound of approximately $r_{\text{eff}} \approx 1 + d/C_\alpha$, where $C_\alpha$ grows exponentially with the PE scaling factor $\alpha$. Without PE ($\tau=0$), the effective rank is substantially larger.
- **Theorem 2 (Structured Input)**: When PE aligns with the data structure (similar features receive identical PE), the effective rank upper bound further decreases from $1 + d/(2C_\alpha)$ to $1 + 1/C_\alpha$—a dramatic reduction.

This establishes that **PE acts as an implicit dimensionality reduction tool**, with greater effect when it aligns with the underlying data structure.

## Key Experimental Results

### Main Results: Comparison of Graph Estimation Methods

| Graph Estimation | Type | Classification Gain (%) | Regression Gain (%) | Extra Time (min) |
|-----------------|------|------------------------|--------------------|--------------------|
| NOTEARS | Causal | 1.36 | 3.64 | 76.83 |
| LiNGAM | Causal | 1.41 | 3.97 | 10.96 |
| Pearson | Correlation | 1.61 | 4.16 | 0.78 |
| **Spearman** | **Correlation** | **1.72** | **4.34** | **0.79** |
| Chow-Liu | Correlation (Tree) | 1.17 | 4.29 | 0.38 |

The Spearman correlation graph achieves the highest gains on both classification and regression tasks, with a computational overhead of only ~0.79 minutes.

### Tab-PET vs. Learnable PE

| Method | Avg. Classification Gain (%) | Avg. Regression Gain (%) | Classification Win Rate (%) | Regression Win Rate (%) |
|--------|-----------------------------|--------------------------|-----------------------------|-------------------------|
| Learnable PE | 0.04 | 0.62 | 12 | 8 |
| **Tab-PET** | **1.72** | **4.34** | **88** | **92** |

Tab-PET outperforms learnable PE on 88%/92% of datasets, demonstrating that fixed, structure-aware PE significantly surpasses PE learned from scratch in low-data regimes.

### Ranking Comparison (Average Rank over 50 Datasets, Lower is Better)

| Model | Classification Rank | Regression Rank |
|-------|--------------------|--------------------|
| XGBoost | 3.40 | 5.20 |
| CatBoost | 3.76 | 2.96 |
| TabTransformer | 7.33 | 7.14 |
| TabTransformer+PET | 5.33 | 5.71 |
| SAINT | 4.52 | 3.64 |
| SAINT+PET | **3.28** | **2.84** |
| FT-Transformer | 4.44 | 4.08 |
| FT-Transformer+PET | **2.44** | **2.88** |

FT-Transformer+PET achieves rank 2.44 on classification (1st overall), and SAINT+PET achieves rank 2.84 on regression (1st overall), both surpassing XGBoost and CatBoost.

### Key Findings

- **Correlation graphs >> Causal graphs**: Spearman/Pearson produce high-entropy dense graphs, while causal methods produce low-entropy sparse graphs. Denser graphs encode richer structural information in PE, yielding larger performance gains.
- **Optimal range for $\alpha$**: Synthetic experiments show that excessively large $\alpha$ (e.g., 10) degrades performance—overly strong PE signals suppress original feature content.
- **Fixed PE > Learnable PE**: Tabular datasets are typically small, making learnable PE prone to overfitting; fixed graph-structural PE is more robust (win rate 88–92% vs. 8–12%).
- **Empirical validation of effective rank**: On 15 real-world datasets, the effective rank of Tab-PET embeddings is observed to decay exponentially with $\alpha$ and is significantly lower than that of random PE, in full agreement with theoretical predictions.
- **Statistically significant improvements across all Transformer architectures** (Wilcoxon test, $p < 0.05$), confirming that Tab-PET is architecture-agnostic.

## Highlights & Insights

- **Challenging the "tabular data has no structure" consensus**: The prevailing view holds that PE is useless for tabular Transformers. This paper overturns that assumption through theory and experiments on 50 datasets, opening a new optimization dimension for tabular Transformers.
- **Elegant effective rank theoretical framework**: The effect of PE is cast as a clear causal chain—"reducing embedding effective rank → reducing learning dimensionality → improving generalization"—providing a quantifiable conceptual tool. The analysis, which draws on random matrix theory, is transferable to PE analysis in other domains.
- **Plug-and-play with negligible computational overhead**: Without modifying the Transformer architecture, computing one Spearman correlation matrix (<1 min) and a Laplacian eigendecomposition suffice to consistently improve performance. This "free lunch" augmentation paradigm is generalizable to any setting where Transformers process unordered inputs (e.g., point clouds, set learning).
- **Graph entropy as a diagnostic for graph estimation quality**: The finding that high-entropy correlation graphs outperform low-entropy causal graphs provides a simple and practical criterion for selecting graph estimation methods.

## Limitations & Future Work

- **Restricted to linear causal models**: Causal graph estimation assumes a linear SEM, which may underfit nonlinear causal relationships. Nonlinear causal discovery methods (e.g., CGNN, DAG-GNN) could be explored.
- **Graph estimation depends on the training set**: PE is estimated from training data, so graph structure may become unreliable under distributional shift at test time. Robustness to distribution shift is not discussed.
- **Dimensionality explosion from one-hot encoding**: High-cardinality categorical features produce a large number of one-hot nodes, potentially making graph Laplacian computation and eigendecomposition non-negligible in cost.
- **No comparison with non-Transformer deep models**: Important tabular deep learning baselines such as TabNet and NODE are absent from the comparisons.
- **Hyperparameter tuning requires a validation set**: Although adaptive $k$ selection reduces tuning burden, $\alpha$ still requires searching over 9 candidate values, increasing training cost.
- **Future directions**: Dynamic PE (updating graph structure during training), local PE for feature subsets, and adaptive PE jointly optimized with attention weights are promising research avenues.

## Related Work & Insights

- **vs. FT-Transformer (Gorishniy et al. 2021)**: FT-Transformer creates learnable embeddings per feature but uses no PE. Tab-PET improves its classification rank from 4.44 to 2.44 by adding only PE concatenation, demonstrating that PE is a critical missing component in FT-Transformer.
- **vs. SAINT (Somepalli et al. 2021)**: SAINT employs dual row-wise and column-wise attention and explicitly claims that PE is useless for tabular data. Tab-PET directly applies PE on top of SAINT and achieves significant gains, directly refuting SAINT's assertion.
- **vs. PE in Graph Transformers (Dwivedi & Bresson 2020; Ito et al. 2025)**: Tab-PET borrows the Laplacian eigenvector PE idea from graph Transformers but innovatively applies it to tabular data that lacks explicit graph structure—first estimating a graph and then extracting PE, effectively creating structural priors from scratch.
- **vs. GBDTs (XGBoost, CatBoost)**: Tab-PET enables Transformers to surpass GBDTs in average ranking over 50 datasets for the first time, representing a milestone result in tabular deep learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Challenges the widely held belief that PE is unnecessary for tabular data; the effective rank theoretical analysis offers an elegant new perspective; however, Laplacian-based PE has precedent in the graph domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 50 datasets, 3 Transformer architectures, 5 graph estimation methods, synthetic and real-world experiments, ablation studies, and statistical significance tests.
- **Writing Quality**: ⭐⭐⭐⭐ Theory and experiments are presented clearly; the framework diagram in Figure 1 is intuitive; however, some theorem assumptions are relatively strong.
- **Value**: ⭐⭐⭐⭐ Provides a plug-and-play augmentation scheme with near-zero additional computational cost, offering direct practical utility to tabular Transformer practitioners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Structure-Aware Encodings of Argumentation Properties for Clique-width](structure-aware_encodings_of_argumentation_properties_for_clique-width.md)
- [\[ICLR 2026\] The Counting Power of Transformers](../../ICLR2026/others/the_counting_power_of_transformers.md)
- [\[ICLR 2026\] Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion](../../ICLR2026/others/harpoon_generalised_manifold_guidance_for_conditional_tabular_diffusion.md)
- [\[AAAI 2026\] A Graph-Theoretical Perspective on Law Design for Multiagent Systems](a_graph-theoretical_perspective_on_law_design_for_multiagent_systems.md)
- [\[CVPR 2026\] Stronger Normalization-Free Transformers](../../CVPR2026/others/stronger_normalization-free_transformers.md)

</div>

<!-- RELATED:END -->
