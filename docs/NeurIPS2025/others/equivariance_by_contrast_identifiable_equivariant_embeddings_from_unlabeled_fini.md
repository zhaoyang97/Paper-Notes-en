---
title: >-
  [Paper Note] Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions
description: >-
  [NeurIPS 2025][Equivariant Representations] Proposes Equivariance by Contrast (EbC), an encoder-only method that jointly learns equivariant embedding spaces and implicit group representations from paired observations $(\mathbf{y}, g \cdot \mathbf{y})$. It aligns finite group actions with invertible linear mappings in the latent space and provides theoretical guarantees for identifiability.
tags:
  - "NeurIPS 2025"
  - "Equivariant Representations"
  - "Contrastive Learning"
  - "Nonlinear ICA"
  - "Group Representations"
  - "Identifiability"
date: 2026-05-08
content_hash: efc9782f5a211492
---

# Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions

**Conference**: NeurIPS 2025  
**arXiv**: [2510.21706](https://arxiv.org/abs/2510.21706)  
**Code**: Not provided  
**Area**: Representation Learning / Group Theory  
**Keywords**: Equivariant Representations, Contrastive Learning, Nonlinear ICA, Group Representations, Identifiability  

## TL;DR

Proposes Equivariance by Contrast (EbC), an encoder-only method that jointly learns equivariant embedding spaces and implicit group representations from paired observations $(\mathbf{y}, g \cdot \mathbf{y})$. It aligns finite group actions with invertible linear mappings in the latent space and provides theoretical guarantees for identifiability.

## Background & Motivation

- In many real-world inference problems, relations between observations are governed by structured transformations: rotation/translation in computer vision, gene knockouts in biology, or sensory stimuli in neuroscience.
- The goal is to learn equivariant embeddings where the group action corresponds to a linear transformation $\mathbf{x}' = \mathbf{R}(g)\mathbf{x}$ in the embedding space.
- Nonlinear ICA provides a theoretical foundation for this problem, but requires additional structural assumptions to make it solvable.
- Limitations of prior work: CARE is restricted to orthogonal representations, STL permits non-linear equivariant relations, and NFT requires learning a generative model.
- An approach is needed that is encoder-only, free from generative models, devoid of group-specific inductive biases, and applicable to general linear group representations.

## Core Problem

How to learn an encoder $\phi$ and group representation $\mathbf{R}'$ from paired observations $(\mathbf{y}, g \cdot \mathbf{y})$ with unknown group elements $g$, such that $\phi(g \cdot \mathbf{y}) = \mathbf{R}'(g)\phi(\mathbf{y})$ holds with identifiability guarantees.

## Method

### Data Assumptions and Problem Setup

Data is provided in batches, each containing $n+1$ pairs of samples $\{(\mathbf{y}_i, \mathbf{y}'_i)\}$, where all pairs in the same batch undergo the same group action $g$. The data generation process is:

$$\mathbf{y}_i = \mathbf{f}(\mathbf{x}_i), \quad \mathbf{y}'_i = \mathbf{f}(\mathbf{R}(g)\mathbf{x}_i)$$

where $\mathbf{f}$ is an unknown non-linear mixing function, and $\mathbf{R}: G \to \text{GL}(d, \mathbb{R})$ is the linear representation of the group.

### Implicit Group Representation Estimation

The group representation matrix is estimated from the encoded sample pairs using least-squares regression:

$$\hat{\mathbf{R}}(\mathbf{X}, \mathbf{X}') = \arg\min_{\mathbf{R} \in \text{GL}(d)} \|\mathbf{X}' - \mathbf{X}\mathbf{R}^\top\|_F^2 = (\mathbf{X}^\top\mathbf{X})^{-1}(\mathbf{X}^\top\mathbf{X}')$$

where $n$ sample pairs are used to estimate $\hat{\mathbf{R}}$, and the remaining 1 pair is used as a query.

### Contrastive Learning Objective

The training objective combines the InfoNCE loss with the group structure:

$$p_\phi(\mathbf{y}' \mid \mathbf{y}, \mathbf{Y}, \mathbf{Y}', S) = \frac{\exp(-\|\mathbf{u}_\phi(\mathbf{y}, \mathbf{Y}, \mathbf{Y}') - \phi(\mathbf{y}')\|\^2)}{\sum_{\mathbf{y}'' \in S} \exp(-\|\mathbf{u}_\phi(\mathbf{y}, \mathbf{Y}, \mathbf{Y}') - \phi(\mathbf{y}'')\|\^2)}$$

where $\mathbf{u}_\phi$ is the operation that infers the group action from context pairs and applies it to the query sample:

$$\mathbf{u}_\phi(\mathbf{y}, \mathbf{Y}, \mathbf{Y}') = \hat{\mathbf{R}}(\phi(\mathbf{Y}), \phi(\mathbf{Y}'))\phi(\mathbf{y})$$

The final optimization is: $\min_\phi \mathcal{L}[\phi] = -\mathbb{E}[\log p_\phi(\mathbf{y}' \mid \mathbf{y}, \mathbf{Y}, \mathbf{Y}', S)]$

### Content-Style Disentanglement

The embedding space is partitioned into an equivariant subspace ($n$ dimensions) and an invariant subspace ($m$ dimensions) by constraining the structure of the group representation matrix:

$$\hat{\mathbf{R}}_{n+m}' = \begin{pmatrix} \hat{\mathbf{R}}_n & \mathbf{0} \\ \mathbf{0} & \mathbf{I}_m \end{pmatrix}$$

### Identifiability Guarantees

**Theorem 1** (Group Representation Identifiability): Under data diversity conditions, defining $\mathbf{h} := \phi \circ \mathbf{f}$:

- **(a)** Recovers the original vector space up to linear indeterminacy: $\mathbf{h}(\mathbf{x}) = \mathbf{L}\mathbf{x}$, $\mathbf{L} \in \text{GL}(d)$
- **(b)** Recovers the group representation up to conjugacy: $\hat{\mathbf{R}}(\mathbf{h}(\mathbf{X}), \mathbf{h}(g\mathbf{X})) = \mathbf{L}\mathbf{R}(g)\mathbf{L}^{-1}$

**Corollary** (Equivariance): $\mathbf{h}(g\mathbf{x}) = g\mathbf{h}(\mathbf{x})$, i.e., the encoder strictly preserves equivariance.

## Key Experimental Results

### Comprehensive Results on Synthetic and Image Data

| Group $G$ | Data | $R^2(x)$ | $R^2(G)$ | Acc(C) | Acc(G,5) |
|--------|------|----------|----------|--------|----------|
| $SO_3$ — InfoNCE | non-linear | 0.0 | 0.0 | 98.9 | — |
| $SO_3$ — **EbC** | non-linear | **99.7** | **99.7** | 99.1 | — |
| $O_3$ — InfoNCE | non-linear | 0.0 | 0.0 | 99.1 | — |
| $O_3$ — **EbC** | non-linear | **99.8** | **99.7** | 99.2 | — |
| $GL_3$ — InfoNCE | non-linear | 0.1 | 0.0 | 98.5 | — |
| $GL_3$ — **EbC** | non-linear | **99.8** | **99.7** | 98.5 | — |
| $R_m \times \mathbb{Z}_n^2$ — InfoNCE | idSprites | — | — | 99.97 | 0.36 |
| $R_m \times \mathbb{Z}_n^2$ — **EbC** | idSprites | — | — | 74.04 | **99.91** |

Key Findings:
- EbC achieves latent space and group representation recovery quality of $R^2 > 99\%$ across all synthetic groups.
- InfoNCE/LDS/SLDS baselines completely fail to recover the group structure ($R^2 \approx 0$), learning only invariant representations.
- A content-group structure trade-off is observed on idSprites: group structure recovery reaches 99.91%, but content classification drops to 74%.
- The linear baseline EbC(lin.) degrades severely under non-linear mixing ($R^2(x) \approx 60$-$70\%$).

### Model Robustness

- When the embedding dimension is over-specified: group structure kNN accuracy remains >99%, and content classification stabilizes at >80%.
- Dimension misspecification (with true dimension as 3): Acc(G) exhibits a clear peak at the correct dimension $d=3$, serving as a reliable basis for hyperparameter selection.
- Performance remains robust as the mixing layer depth increases up to 4 layers, after which it begins to degrade.

## Highlights & Insights

- ⭐ First encoder-only method to achieve general linear group representation learning without requiring generative models or group-specific inductive biases, covering non-Abelian groups like $O(n), GL(n)$.
- ⭐ Elegant design of implicit group representation: directly estimating the group matrix from embedding pairs via least squares. The encoder and the group representation are unified through a single $\phi$.
- ⭐ Theoretical completeness: proves linear identifiability and group representation identifiability from the contrastive/discriminative framework of nonlinear ICA.
- The Acc(G) metric can be utilized without access to true latent variables, offering a practical model choice criterion.

## Limitations & Future Work

- The drop in content classification accuracy on idSprites (74%) indicates remaining room for improvement in content-style separation.
- The theoretical assumptions require the data to satisfy "sufficient diversity" conditions; its applicability to low-data regimes remains to be validated.
- Currently, validation is limited to finite groups; extending this to continuous groups (such as continuous $SO(3)$) is not addressed.
- The data generation process does not incorporate noise (the theorems assume exact group actions), and robustness to noise is only preliminarily explored in the appendix.
- Extensive evaluation on real-world visual datasets beyond idSprites is left for future work.

## Related Work & Insights

| Method | Type | Group Representation | Requires Generative Model | Identifiability |
|------|------|--------|------------|---------|
| CARE | Encoder | Orthogonal (Hyperspherical) | No | Limited |
| STL | Encoder | Non-linear Equivariant | No | No |
| NFT | Generative + Encoder | General Linear | Yes | Yes |
| **EbC (Ours)** | **Encoder-only** | **General Linear (GL)** | **No** | **Yes (Linear Indeterminacy)** |

## Insights & Connections

- The least-squares estimation of the group matrix is highly practical for scenarios with known paired data, with potential applications in physical simulation, robotics, etc.
- The combination of "contrastive learning + group structure" provides a novel regularization tool for self-supervised representation learning.
- Hyperparameter selection via the peak of Acc(G) to determine the embedding dimension serves as a valuable strategy for any latent space dimensional selection task.
- The connection to nonlinear ICA identifiability theory establishes a solid theoretical foundation for equivariant representation learning.

## Rating

- ⭐ Novelty: 9/10 — The emergence of equivariance and group representations from contrastive learning is a novel perspective with outstanding theoretical contributions.
- ⭐ Experimental Thoroughness: 7/10 — Synthetic experiments are comprehensive, but empirical evaluation on visual data is limited to idSprites, lacking more realistic scenarios.
- ⭐ Writing Quality: 8/10 — Clear theoretical derivations, consistent notation, and intuitive illustrations.
- ⭐ Value: 8/10 — Establishes a new paradigm for equivariant representation learning, with theoretical significance outweighing immediate practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Identifiable Equivariant Networks are Layerwise Equivariant](../../ICML2026/others/identifiable_equivariant_networks_are_layerwise_equivariant.md)
- [\[NeurIPS 2025\] Faithful Group Shapley Value](faithful_group_shapley_value.md)
- [\[NeurIPS 2025\] Exploiting Task Relationships in Continual Learning via Transferability-Aware Task Embeddings](exploiting_task_relationships_in_continual_learning_via_transferability-aware_ta.md)
- [\[NeurIPS 2025\] Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning](directional_non-commutative_monoidal_structures_for_compositional_embeddings_in_.md)
- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](on_universality_classes_of_equivariant_networks.md)

</div>

<!-- RELATED:END -->
