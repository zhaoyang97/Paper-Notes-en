---
title: >-
  [Paper Note] ReSA: Clustering Properties of Self-Supervised Learning
description: >-
  [ICML 2025][Self-Supervised Learning][clustering properties] This work systematically analyzes the clustering properties of various components in JEA-based SSL. It discovers that the encoding possesses superior and more stable clustering capabilities compared to the embedding and hidden layers of the projector. Based on this, ReSA (Representation Self-Assignment) is proposed to utilize encoding clustering information to guide embedding learning…
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "clustering properties"
  - "ReSA"
  - "positive feedback"
  - "Sinkhorn-Knopp"
date: 2026-05-08
content_hash: 1f9e5532d461884a
---

# ReSA: Clustering Properties of Self-Supervised Learning

**Conference**: ICML 2025  
**arXiv**: [2501.18452](https://arxiv.org/abs/2501.18452)  
**Code**: None  
**Area**: Self-Supervised Learning  
**Keywords**: self-supervised learning, clustering properties, ReSA, positive feedback, Sinkhorn-Knopp

## TL;DR
This work systematically analyzes the clustering properties of various components in JEA-based SSL. It discovers that the encoding possesses superior and more stable clustering capabilities compared to the embedding and hidden layers of the projector. Based on this, ReSA (Representation Self-Assignment) is proposed to utilize encoding clustering information to guide embedding learning, forming a positive-feedback SSL framework that significantly outperforms SOTA on multiple standard benchmarks.

## Background & Motivation

### Background
**Background**: Self-Supervised Learning (SSL) learns semantically rich representations under unsupervised conditions through Joint Embedding Architectures (JEAs), outperforming supervised learning in visual representation learning. A JEA consists of a shared encoder $E_{\theta_e}$ and a projector $G_{\theta_g}$, outputting encoding $H$ and embedding $Z$, respectively. Existing studies (Ben-Shaul et al., 2023) find that SSL representations exhibit hierarchical clustering properties—strengthening three levels of clustering structures: sample-level, semantic class-level, and superclass-level.

### Limitations of Prior Work
**Limitations of Prior Work**: (1) Although SSL representations are known to possess clustering properties, **scarcely any methods utilize these properties to improve SSL itself**—rich clustering information is wasted; (2) It remains unclear whether encoding or embedding has better clustering properties—the optimization dynamics and information flow of the projector remain open questions; (3) Existing online clustering methods like SwAV use learnable prototypes to map embeddings to a clustering space, which require extra parameters, and clustering is conducted on embeddings where information has already been lost.

**Key Challenge**: SSL representations possess rich clustering properties but are not utilized to improve SSL itself, and clustering information is unevenly distributed across different components of the JEA.

### Goal
**Goal**: (1) Answer "where is the best place to extract clustering properties?"; (2) Answer "how to utilize clustering properties?"; (3) Answer "does positive feedback promote better clustering?".

**Core Idea**: Encoding has optimal clustering properties $\rightarrow$ online self-clustering on encoding $\rightarrow$ using the cluster assignment matrix to guide cross-entropy loss of embedding $\rightarrow$ positive feedback loop to improve representation quality.

## Method

### Overall Architecture
On top of the standard JEA (encoder + projector), ReSA extracts clustering information from the encoder output $H$. It generates an online cluster assignment matrix $A_H$ via the Sinkhorn-Knopp algorithm, using it to guide the cross-entropy loss between projector outputs $Z, Z'$, forming a closed-loop positive feedback: better encoding $\rightarrow$ better cluster assignment $\rightarrow$ better training signals $\rightarrow$ better encoding.

### Key Designs

1. **Empirical Finding of the Superior Clustering Properties of Encoding**:

    - Function: Determine the optimal source of clustering information
    - Mechanism: Evaluate the clustering properties of the encoding $H$, embedding $Z$, and projector hidden layers $P_0, P_1$ of methods like SimCLR, VICReg, SwAV, and BYOL on CIFAR-10/100 using the Silhouette Coefficient (SC_mean to measure local clustering capability, SC_std to measure stability) and Adjusted Rand Index (ARI to measure global clustering capability). Findings: (a) encoding achieves higher SC_mean, lower SC_std, and higher ARI across nearly all methods; (b) during training, the clustering metrics of encoding continuously improve, whereas those of embedding degrade in later stages; (c) although the projector hidden layers yield linear evaluation accuracy close to that of encoding, their clustering properties are significantly worse.
    - Design Motivation: Confirming that the encoding is the optimal clustering information source, laying the foundation for subsequent designs.

2. **Online Self-Clustering Mechanism**:

    - Function: Extract clustering information from encoding and generate a soft assignment matrix
    - Mechanism: Instead of using learnable prototypes, encoding samples within a mini-batch are simultaneously treated as data points to be clustered and clustering anchors. The cosine self-similarity matrix $S_H = H^\top H$ (after L2-normalization) is calculated, and then $\exp(S_H/\epsilon)$ is converted into a doubly stochastic matrix $A_H$ via the Sinkhorn-Knopp algorithm (3 iterations, regularization parameter $\epsilon=0.05$) to serve as the cluster assignment.
    - Design Motivation: Unlike SwAV which uses learnable prototypes, ReSA requires no additional parameters and operates in the encoding space—directly leveraging the superior clustering properties of the encoding. Sinkhorn-Knopp does not involve gradient propagation and is implemented efficiently on GPUs.

3. **Cluster-Guided Cross-Entropy Loss**:

    - Function: Utilize cluster assignments to guide embedding learning
    - Mechanism: The ReSA loss is defined as $\ell_{\text{ReSA}} = -\frac{1}{2m}(\sum_{i,j} A_H \circ \log \mathcal{D}(Z^\top Z') + \sum_{i,j} A_H^\top \circ \log \mathcal{D}(Z'^\top Z))$, where $\mathcal{D}$ denotes softmax temperature normalization and $\circ$ represents the Hadamard product. $A_H$ originates from encoding clustering information and guides the alignment of similar samples in the embedding space.
    - Design Motivation: Contrast with the "swapped prediction" mechanism of SwAV—SwAV performs Sinkhorn on embeddings and then swaps predictions, whereas ReSA performs Sinkhorn on encodings and then guides embeddings, capitalizing on the superior clustering properties of encodings.

### Loss & Training
The total loss is the ReSA cross-entropy loss, with the temperature hyperparameter $\tau$ controlling distribution sharpness. No extra contrastive negative samples or momentum encoders are required, as the regularization of Sinkhorn-Knopp naturally prevents representation collapse.

## Key Experimental Results

### Main Results: ImageNet Linear Evaluation

| Method | Backbone | Epochs | Top-1 Acc. |
|------|----------|--------|------------|
| SimCLR | ResNet-50 | 200 | 66.5% |
| BYOL | ResNet-50 | 200 | 70.6% |
| SwAV | ResNet-50 | 200 | 71.8% |
| VICReg | ResNet-50 | 200 | 68.6% |
| **ReSA** | ResNet-50 | 200 | **73.2%** |

### Ablation Study: Clustering Source

| Clustering Source | SC_mean ↑ | SC_std ↓ | ARI ↑ | Training Stability |
|---------|-----------|----------|-------|-----------|
| Embedding $Z$ | Lower | High | Lower | Degradation in later stages |
| Projector hidden layer $P_0$ | Medium | Medium | Medium | Unstable |
| **Encoding $H$** | **Highest** | **Lowest** | **Highest** | **Continuous improvement** |

### Training Efficiency Comparison

| Method | Epochs required to reach 70% Top-1 | Notes |
|------|---------------------------|------|
| SimCLR | Not reached | — |
| BYOL | ~200 | — |
| SwAV | ~180 | — |
| **ReSA** | **~150** | Faster convergence |

### Key Findings
- Encoding possesses the optimal clustering properties in almost all SSL methods—this is a universal phenomenon rather than an exception of specific methods.
- The positive feedback mechanism of ReSA not only boosts performance but also accelerates convergence—better clustering signals lead to more efficient training.
- ReSA simultaneously improves both fine-grained and coarse-grained clustering properties.

## Highlights & Insights
- **A New Paradigm of Positive Feedback SSL**: Clustering properties $\rightarrow$ training signals $\rightarrow$ better representation $\rightarrow$ better clustering properties—this self-reinforcing loop represents a conceptual contribution to SSL methodology.
- **Systematic Analysis of Encoding vs. Embedding**: This study is the first to rigorously quantify the differences in clustering properties across various JEA components, offering a fresh perspective on the mechanism of the projector.
- **Prototype-Free Online Clustering**: Unlike the learnable prototypes in SwAV/DINOv2, ReSA performs self-clustering directly among samples—more elegant and requiring no extra parameters.

## Limitations & Future Work
- **Computational Overhead of Sinkhorn-Knopp under Large Batch Sizes**: The self-similarity matrix $S_H$ is sized $m \times m$, resulting in increased memory and computational costs in the case of large batch sizes.
- **Superficial Validation in Vision SSL Only**: Clustering properties in NLP and multimodal SSL may exhibit different characteristics.
- **Relation to Knowledge Distillation Methods**: ReSA can be viewed as a form of self-distillation—where the encoding acts as a "teacher" to guide the embedding—making its relationship with DINO/iBOT worthy of in-depth exploration.
- **Implicit Assumption on the Number of Clusters**: Sinkhorn-Knopp does not explicitly set the number of clusters, but the batch size implicitly constrains the discoverable cluster structures.

## Related Work & Insights
- **vs. SwAV (Caron et al., 2020)**: SwAV performs clustering on embeddings using learnable prototypes—ReSA conducts prototype-free self-clustering on encodings, exploiting a superior source of information.
- **vs. DINOv2 (Oquab et al., 2023)**: Also utilizes Sinkhorn-Knopp but adopts learnable prototypes—ReSA's prototype-free design is more streamlined.
- **vs. Ben-Shaul et al. (2023)**: They discovered the hierarchical clustering properties of SSL—this work is the first to leverage these properties to improve SSL itself.
- **vs. Ma et al. (2023)**: Utilizes the enhanced robustness of the encoding to reweight positive pair alignments—but ignores clustering information.

## Rating
- Novelty: ⭐⭐⭐⭐ Positive-feedback SSL is a conceptual contribution; the systematic analysis of encoding clustering properties is of high value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-method and multi-dataset clustering analysis + large-scale experiments on ImageNet.
- Writing Quality: ⭐⭐⭐⭐ Structured clearly around three progressive questions.
- Value: ⭐⭐⭐⭐ Provides a new methodological paradigm for the SSL community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Collapse-Proof Non-Contrastive Self-Supervised Learning](collapse-proof_non-contrastive_self-supervised_learning.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICML 2025\] Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le.md)
- [\[ICLR 2026\] On the Alignment Between Supervised and Self-Supervised Contrastive Learning](../../ICLR2026/self_supervised/on_the_alignment_between_supervised_and_self-supervised_contrastive_learning.md)
- [\[NeurIPS 2025\] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](../../NeurIPS2025/self_supervised/t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
