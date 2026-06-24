---
title: >-
  [Paper Note] Collapse-Proof Non-Contrastive Self-Supervised Learning
description: >-
  [ICML 2025][Self-Supervised Learning][Non-contrastive self-supervised learning] This paper proposes the FALCON method, which designs the projector and loss function based on the principles of hyperdimensional computing. It theoretically proves the simultaneous prevention of four known training failure modes (representation collapse, dimensional collapse, cluster collapse, and intracluster collapse) while naturally endowing representations with decorrelation and clustering pro…
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "Non-contrastive self-supervised learning"
  - "collapse prevention"
  - "hyperdimensional computing"
  - "feature decorrelation"
  - "clustering representation"
date: 2026-05-08
content_hash: 2a7dba7efa595c48
---

# Collapse-Proof Non-Contrastive Self-Supervised Learning

**Conference**: ICML 2025

**arXiv**: [2410.04959](https://arxiv.org/abs/2410.04959)

**Authors**: Emanuele Sansone, Tim Lebailly, Tinne Tuytelaars (KU Leuven)

**Area**: Self-Supervised Learning

**Keywords**: Non-contrastive self-supervised learning, collapse prevention, hyperdimensional computing, feature decorrelation, clustering representation

## TL;DR

This paper proposes the FALCON method, which designs the projector and loss function based on the principles of hyperdimensional computing. It theoretically proves the simultaneous prevention of four known training failure modes (representation collapse, dimensional collapse, cluster collapse, and intracluster collapse) while naturally endowing representations with decorrelation and clustering properties.

## Background & Motivation

Although Self-Supervised Learning (SSL) has achieved great success on unlabeled data, multiple **failure modes** exist during the training process, limiting the reliability and widespread application of these methods:

**Representation Collapse**: Representations of all inputs collapse to the same constant vector.

**Dimensional Collapse**: Embeddings only occupy a lower-dimensional subspace of the vector space.

**Cluster Collapse**: Data points are only assigned to a subset of the available prototypes.

**Intracluster Collapse**: Representative differences among samples within the same cluster approach zero.

Existing methods (such as momentum encoders, stop-gradients, asymmetric projection heads, etc.) are heuristic strategies that cannot theoretically guarantee the prevention of all collapses. **Feature decorrelation methods** (such as Barlow Twins) and **clustering methods** (such as SwAV) address partial collapses separately, but a unified solution is still lacking.

Goal: To find a set of **sufficient conditions** that guarantee the simultaneous prevention of all four collapses, and to design a simple projector and loss function accordingly.

## Method

### Overall Architecture

The FALCON (FAiLure-proof non-CONtrastive SSL) method comprises the following pipeline:

1. Generate augmented pairs $(X, X')$ for unlabeled data.
2. Encoder $g: \mathbb{R}^d \to \mathbb{R}^f$ extracts representations $Z = g(X)$.
3. The FALCON projector generates embeddings and probability assignments.
4. Train using the FALCON loss function.

### Projector Design

The projector performs a two-step operation:

$$\mathbf{H} = \sqrt{f/n} \cdot \text{L2-norm}(\text{BN}(\text{Linear}(\mathbf{Z})))$$

$$\mathbf{P} = \text{Softmax}(\mathbf{H}\mathbf{W}/\tau)$$

where the key designs are:
- **Dictionary matrix** $\mathbf{W} \in \{-1, 1\}^{f \times c}$: elements are independent and identically sampled from the Rademacher distribution.
- **Temperature parameter** $\tau = \sqrt{f/n} \cdot \log\frac{1 - \epsilon(c-1)}{\epsilon}$.
- Dictionary size $c \gg f$ (overcomplete dictionary).

The core innovation of this design arises from **hyperdimensional computing**: random Rademacher vectors are approximately orthogonal in high-dimensional space:

$$\mathbb{E}_W\{w_j^T w_{j'}\} = \begin{cases} 1 & j = j' \\ 0 & j \neq j' \end{cases}, \quad \text{Var}_W\{\cos(w_j, w_{j'})\} = \frac{1}{f}$$

This allows $c$ to be much larger than $f$, breaking the limitation where the number of orthogonal vectors is $\leq f$.

### Loss & Training

$$\mathcal{L}_{\text{FALCON}}(\mathcal{D}) = -\frac{\beta}{n}\sum_{i=1}^n \sum_{j=1}^c p_{ij} \log p'_{ij} - \sum_{j=1}^c q_j \log \frac{1}{n}\sum_{i=1}^n p_{ij}$$

- **First term (Invariance Loss)**: Encourages augmented pairs to produce consistent assignments; it can be decomposed into an entropy term and a KL divergence term.
- **Second term (Prior Matching Loss)**: Enforces the assignment distribution to match the prior $\mathbf{q}$ (uniform distribution) to prevent cluster collapse.
- $\beta > 0$ balances the two terms.

### Key Theoretical Guarantees

**Theorem (Embeddings)**: At the optimal solution, each embedding aligns with exactly one codeword in the dictionary:

$$\forall i \in [n], \exists! j \in [c] \text{ s.t. } \mathbf{h}_i = \alpha_{ij}\mathbf{w}_j + (\alpha_{ij} - \frac{1}{\sqrt{n}})\sum_{k \neq j}\mathbf{w}_k$$

**Corollary 1 (Perfect Alignment)**: As $c \to \infty$, $\mathbf{h}_i = \frac{1}{\sqrt{n}} \mathbf{w}_j$.

**Corollary 2 (Diagonal Covariance)**: $\mathbf{H}^T\mathbf{H} = \mathbf{I}$, implying that embedding features are completely decorrelated.

**Corollary 3 (Block-Diagonal Adjacency)**: The adjacency matrix $\mathbf{H}\mathbf{H}^T$ is a block-diagonal matrix with equal block sizes of $n/c$, implying natural clustering.

## Key Experimental Results

### Main Results: Downstream Task Generalization Performance (Table 1)

| Method | SVHN NMI | CIFAR-10 NMI | CIFAR-100 NMI | SVHN Acc. | CIFAR-10 Acc. | CIFAR-100 Acc. |
|------|----------|-------------|--------------|-----------|--------------|----------------|
| Barlow Twins | 0.06 | 0.05 | 0.10 | 0.76 | 0.65 | 0.28 |
| SwAV | 0.03 | 0.29 | 0.12 | 0.45 | 0.56 | 0.10 |
| Self-Classifier | 0.07 | 0.28 | 0.26 | 0.58 | 0.59 | 0.15 |
| GEDI | 0.07 | 0.29 | 0.25 | 0.58 | 0.64 | 0.38 |
| **FALCON (c=16384)** | **0.31** | **0.35** | **0.58** | **0.78** | **0.68** | **0.41** |

### ImageNet-100 Linear Probing Results (Table 2, ViT-small, 100 epochs)

| Method | c=100 | c=500 | c=1K | c=5K | c=10K | c=50K | c=100K | c=200K | c=300K | c=500K |
|------|-------|-------|------|------|-------|-------|--------|--------|--------|--------|
| DINO Top-1 | 64.1% | 65.8% | 65.1% | 65.7% | 66.6% | 67.5% | 67.7% | 68.3% | 67.0% | 67.2% |
| **FALCON Top-1** | **64.9%** | **68.4%** | **70.2%** | **72.2%** | **72.9%** | **73.9%** | **72.2%** | **73.9%** | **73.6%** | **74.0%** |
| DINO Top-5 | 87.0% | 88.6% | 88.7% | 87.7% | 89.2% | 89.2% | 89.4% | 89.8% | 89.2% | 89.9% |
| **FALCON Top-5** | **87.7%** | **89.9%** | **91.2%** | **91.7%** | **92.3%** | **92.6%** | **92.6%** | **92.9%** | **92.8%** | **92.9%** |

### Key Findings

1. **Monotonic Gain of Dictionary Size**: Clustering and classification performances improve monotonically with dictionary size $c$. FALCON is the only method capable of systematically leveraging large dictionaries.
2. **Complete Prevention of Collapse**: The covariance matrix tends toward being diagonal as $c$ increases, the block size of the adjacency matrix shrinks, and the distribution of singular values becomes more uniform.
3. **Outperforming DINO**: On ImageNet-100, FALCON significantly outperforms DINO with a simpler design (no stop-gradient, teacher centering, EMA, etc.).

## Highlights & Insights

1. **First to Combine Hyperdimensional Computing with SSL**: Exploits the quasi-orthogonality of high-dimensional random vectors to break the bottleneck of dictionary size, forming a highly creative cross-disciplinary connection.
2. **Theory-Driven Design**: Every component (Rademacher dictionary, L2 normalization, large dictionary) occupies a clear theoretical motivation, rather than relying on empirical parameter tuning.
3. **Unified Framework**: Simultaneously achieves the strengths of feature decorrelation (Barlow Twins family) and clustering (SwAV family), theoretically proving that both can coexist.
4. **Minimalist Design**: Eliminates the need for common SSL techniques such as momentum encoders, stop-gradient, and Sinkhorn clustering layers, significantly simplifying the training pipeline.

## Limitations & Future Work

1. **Limited Experimental Scale**: Main experiments are validated on SVHN/CIFAR using ResNet-8, and ImageNet-100 uses a small ViT; evaluation on ImageNet-1K or larger datasets is missing.
2. **Uniform Prior Only**: Theoretical analysis is confined to the case of $q_j = 1/c$, leaving non-uniform prior assumptions (which are more common in real-world data) mostly unexplored.
3. **Training Instability**: Training instability occurs on ImageNet-100 (some codewords are unused initially), requiring the KL matching to be switched to reverse KL.
4. **Backbone Capacity Assumption**: Key theoretical results assume the backbone possesses infinite capacity, yielding only partial guarantees under finite capacity conditions.

## Related Work & Insights

- **Barlow Twins** (Zbontar et al., 2021): Avoids dimensional collapse through cross-covariance diagonalisation, but does not address cluster collapse.
- **SwAV** (Caron et al., 2020): Employs a Sinkhorn clustering layer to handle cluster collapse, but requires additional heuristics.
- **DINO** (Caron et al., 2021): Requires asymmetric designs (stop-gradient, teacher centering, EMA), whereas FALCON outperforms it with a simpler design.
- **VICReg** (Bardes et al., 2022): Variance-Invariance-Covariance Regularization.
- The **hyperdimensional computing perspective** of this paper may inspire more intersectional research between SSL and high-dimensional random structures.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Novelty | 5 | First combination of hyperdimensional computing and SSL, unifying decorrelation and clustering. |
| Theoretical Depth | 5 | Complete theoretical analysis, with proof guarantees for all four collapses. |
| Experimental Thoroughness | 3 | Sufficient validation on small to medium-scale datasets, lacks large-scale experiments. |
| Writing Quality | 4 | Clear structure and rigorous theoretical derivations. |
| Value | 4 | Simplifies the SSL training pipeline, but requires large-scale validation. |
| **Overall** | **4.2** | A methodology paper with outstanding theoretical contributions, though the experimental scale remains to be extended. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dual Perspectives on Non-Contrastive Self-Supervised Learning](../../ICLR2026/self_supervised/dual_perspectives_on_non-contrastive_self-supervised_learning.md)
- [\[ICML 2025\] Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings](generalization_analysis_for_supervised_contrastive_representation_learning_under.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICML 2025\] ReSA: Clustering Properties of Self-Supervised Learning](clustering_properties_of_self-supervised_learning.md)
- [\[ICML 2025\] Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le.md)

</div>

<!-- RELATED:END -->
