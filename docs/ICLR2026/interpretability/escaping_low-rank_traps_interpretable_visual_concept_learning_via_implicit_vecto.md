---
title: >-
  [Paper Note] Escaping Low-Rank Traps: Interpretable Visual Concept Learning via Implicit Vector Quantization
description: >-
  [ICLR 2026][Interpretability][Concept Bottleneck Model] To address the "representation collapse" problem where patch features degrade into a low-rank subspace and destroy visual-concept many-to-many alignment during Concept Bottleneck Model (CBM) training, this paper proposes **Implicit Vector Quantization (IVQ)**, which treats the vector quantization objective as a regular
tags:
  - ICLR 2026
  - Interpretability
  - Concept Bottleneck Model
date: 2026-05-08
content_hash: e88bd34cfe05e13e
---
# Escaping Low-Rank Traps: Interpretable Visual Concept Learning via Implicit Vector Quantization

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=9M2VrpAtR1](https://openreview.net/forum?id=9M2VrpAtR1)  
**Code**: [https://github.com/Daryl-GSJ/IVQ-CBM](https://github.com/Daryl-GSJ/IVQ-CBM)  
**Area**: Interpretable AI / Concept Bottleneck Models  
**Keywords**: Concept Bottleneck Model, Representational Collapse, Implicit Vector Quantization, Many-to-Many Alignment, Concept Aggregation  

## TL;DR
To address the "representation collapse" problem where patch features degrade into a low-rank subspace and destroy visual-concept many-to-many alignment during Concept Bottleneck Model (CBM) training, this paper proposes **Implicit Vector Quantization (IVQ)**, which treats the vector quantization objective as a regularizer rather than a hard bottleneck. Combined with **Magnet Attention** to aggregate high-rank patch features into concept prototypes, the method achieves SOTA accuracy and superior interpretability consistency across 8 medical and 5 general benchmarks.

## Background & Motivation
**Background**: Concept Bottleneck Models (CBMs) achieve self-explanation by inserting a human-understandable "concept layer" between the perceptron and the task head. They first map an image to a set of predefined semantic concepts (e.g., "beak shape," "lesion extent") and then perform final classification based solely on these Concept Activation Vectors (CAVs).

**Limitations of Prior Work**: Early CBMs (LaBo, PCBM, LF-CBM, etc.) used a single global visual feature (CLIP [CLS] token or global embedding) to align concepts, assuming a global vector can encapsulate all visual attributes. This assumption fails in complex scenarios like medical imaging where lesions are small and scattered.

**Key Challenge**: The authors point out that the true prerequisite for CBM robustness is a **many-to-many mapping between visuals and concepts**—one patch may correspond to multiple concepts, and visual evidence for one concept may be scattered across multiple patches. However, when recent works (ExplicD, MVP-CBM, DOT-CBM) explicitly model such patch-level relationships, they encounter a fundamental obstacle: **representational collapse**. The authors tracked the rank of the patch feature matrix during training and found it drops sharply in the first few epochs, falling from a full rank of 196 to 70. This causes highly similar features and informational redundancy, destroying the quality of CAVs.

**Core Idea**: **Problem Diagnosis** — Representational collapse is identified as the core pathology of modern CBMs, essentially a loss of feature diversity; **Key Insight** — While de-correlation or spectral regularization in SSL can maintain high rank, they "maximise diversity indiscriminately," amplifying trivial details that are useless or harmful to the task. CBMs require "structured diversity aligned with human concepts"; **Mechanism** — A lightweight regularizer anchors features to learned concept prototypes, maintaining high rank while ensuring diversity remains semantic.

## Method

### Overall Architecture
IVQ-CBM follows the two-stage CBM pipeline $x \to c \to y$, but jointly optimizes three dimensions during the alignment phase: classification accuracy, concept alignment, and representation quality. First, a pre-trained CLIP ViT extracts patch token features $Z_p \in \mathbb{R}^{L\times D}$; **IVQ** acts as a regularizer to maintain the high-rank diversity of $Z_p$ and distill semantic information into each patch; **Magnet Attention** then soft-clusters these high-rank patch features into $K$ visual concept prototypes $M \in \mathbb{R}^{K\times D}$, which are dot-producted with text concept embeddings $\tau$ to obtain concept activation scores for the classification head.

```mermaid
flowchart LR
    A[Input Image x] --> B[CLIP ViT Encoder]
    B --> C["Patch Features Z_p (L×D)"]
    C --> D["IVQ Regularization<br/>Codebook Anchoring/High-Rank"]
    C --> E["Magnet Attention<br/>Soft Clustering Aggregation"]
    D -.Reg. Gradient.-> C
    E --> F["Visual Concept Prototypes M (K×D)"]
    F --> G["Dot Product with τ<br/>→ CAV v"]
    G --> H[Concept Alignment BCE Supervision]
    G --> I[Classification Head → Prediction]
```

### Key Designs

**1. Implicit Vector Quantization (IVQ): Demoting VQ to a Regularizer to Bypass Hard Bottlenecks** — This is the core innovation. Standard VQ uses argmin to find the nearest codebook vector and passes the quantized discrete feature into the forward pass, which compresses the rich information of a patch into a single codeword, violating the many-to-many principle and creating an information bottleneck. IVQ does the opposite: it maintains a learnable codebook $C_{vq}\in\mathbb{R}^{M\times D}$ and calculates the assignment $k_j = \arg\min_k \|z_j - c_k\|_2^2$ for each patch, but **discards the quantized output $Z_q$ from the forward pass**, retaining only the codebook loss and commitment loss as regularization during backpropagation:
$$\mathcal{L}_{IVQ} = \underbrace{\|\text{sg}(Z_p) - Z_q\|_2^2}_{\text{Codebook Loss}} + \beta\underbrace{\|Z_p - \text{sg}(Z_q)\|_2^2}_{\text{Commitment Loss}}$$
where $\text{sg}(\cdot)$ is the stop-gradient operator. These codebook prototypes act as "anchors," forcing each patch toward the nearest prototype to prevent the feature distribution from collapsing into a degenerate subspace—maintaining high rank while ensuring diversity is "semantic" because the prototypes themselves correspond to text concepts.

**2. Magnet Attention: Soft-Clustering Patches into Concept Prototypes** — Simple spatial pooling loses fine-grained information. The authors designed a differentiable soft-clustering module to bridge local features and high-level concepts. $K$ learnable concept queries $Q\in\mathbb{R}^{K\times D}$ are introduced, where each $q_k$ acts like a "magnet" attracting patches related to that concept. Similarity is calculated using negative squared Euclidean distance, followed by a softmax over the concept dimension to obtain a soft assignment matrix:
$$A_{jk} = \frac{\exp(-\|z_j - q_k\|_2^2)}{\sum_{k'=1}^{K}\exp(-\|z_j - q_{k'}\|_2^2)}$$
The final visual concept prototypes are the weighted averages of patch features $M = A^\top Z_p$. Since a patch can have non-zero weights for multiple queries and a query can aggregate multiple patches, this naturally achieves many-to-many correspondence.

**3. Loss & Training: Joint Optimization of Accuracy, Interpretability, and Representation** — The total objective combines three tasks. The classification loss $\mathcal{L}_{cls} = \mathcal{L}_{CE}(p_i, y_i)$ ensures task accuracy; the concept alignment loss uses binary cross-entropy $\mathcal{L}_{concept} = \mathcal{L}_{BCE}(v_i, c_i)$ to supervise concept activation scores $v_i$ with multi-hot concept labels $c_i$, forcing the model to learn semantically meaningful concepts; finally, the IVQ regularization is added. The three are summed with equal weight:
$$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{concept} + \mathcal{L}_{IVQ}$$
The codebook size $M$ is set equal to the number of text concepts $K$, forming a one-to-one correspondence.

## Key Experimental Results

### Main Results (ACC %, Interpretable Models, Selected)

| Method | ISIC | NCT | IDRID | BUSI | CUB | C-100 | ImageNet |
|------|------|------|-------|------|-----|-------|----------|
| LaBo (CVPR'23) | 79.20 | 91.73 | 50.77 | 84.01 | 69.88 | 60.17 | 68.04 |
| Explicd (MICCAI'24) | 88.72 | 95.29 | 63.26 | 87.17 | 74.08 | 64.91 | 71.93 |
| MVP-CBM (IJCAI'25) | 87.72 | 97.90 | 65.38 | 89.74 | 74.63 | 65.48 | 72.29 |
| DOT-CBM (CVPR'25) | 86.55 | 90.15 | 58.45 | 85.23 | 72.29 | 63.45 | 69.31 |
| **Ours (IVQ-CBM)** | **90.11** | **99.90** | **67.35** | **93.59** | **75.91** | **67.12** | **73.42** |
| **Gain** | +1.39 | +2.00 | +1.97 | +3.85 | +1.28 | +1.64 | +1.13 |

Ours outperforms strong CBM baselines across 13 benchmarks and even exceeds black-box models like ResNet50/ViT, mitigating the long-standing "performance vs. interpretability" trade-off.

### Ablation Study (Selected from Table 3)

| IVQ | Magnet | ISIC ACC | IDRID ACC | IDRID BMAC |
|-----|--------|----------|-----------|------------|
| ✗ | ✔ | 80.88 | 57.14 | 45.27 |
| ✔ | ✗ | 89.42 | 65.38 | 61.25 |
| ✔ | ✔ | **90.11** | **67.35** | **73.06** |

IVQ contributes the most, with an 11.81 point gain in BMAC on the imbalanced IDRID dataset. Removing Magnet (reverting to [CLS] features) leads to significant drops.

**Key Findings**
- Representation collapse is a common pathology in modern CBMs (observed in MVP-CBM and DOT-CBM); it worsens as datasets become more complex.
- High rank is not the goal itself; "structured diversity aligned with concepts" is—which explains why IVQ outperforms indiscriminate de-correlation.
- The codebook can be interpreted as a "visual dictionary," providing additional interpretability.

## Highlights & Insights
- **Diagnosis is more valuable than the solution**: Identifying "representation collapse" as the fundamental barrier to many-to-many alignment in CBMs is a major contribution.
- **"Downgrading VQ" is a clever modification**: It retains the semantic anchoring of VQ while discarding the information bottleneck of hard quantization.
- **Critical Thinking on Regularization**: The paper distinguishes between "indiscriminate diversity" and "concept-aligned structured diversity," supported by counter-examples where high-rank led to worse performance.

## Limitations & Future Work
- Concept alignment relies on predefined text labels $c_i$, making it difficult to transfer to domains without concept annotations.
- The one-to-one binding of $M=K$ is simple but may not be optimal for large or unevenly distributed concept sets.
- While significant in medical/fine-grained scenarios, gains on general large-scale data (ImageNet) are relatively modest (+1.13).

## Related Work & Insights
- **CBM Lineage**: From models relying on single global features (LaBo/PCBM) to patch-level many-to-many alignment (ExplicD/MVP-CBM), this work solves the collapse side-effect of the latter.
- **Representation Regularization**: Compared to Barlow Twins or Spectral Regularization, IVQ better fits the cross-modal structured requirements of CBMs.
- **Inspiration**: Demoting a hard-constrained objective into a soft regularizer—retaining the gradient signal while discarding the forward product—is a generalizable design pattern for representation learning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The diagnosis of "representation collapse" and the "implicit quantization" perspective are clever and original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 13 datasets, 8 baselines, rank dynamics analysis, and codebook interpretability.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly organized with strong diagnostic narratives and logical RQ structure.
- **Value**: ⭐⭐⭐⭐ Effectively improves both accuracy and interpretability for CBMs, particularly useful for high-stakes fields like medicine.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition](towards_understanding_the_nature_of_attention_with_low-rank_sparse_decomposition.md)
- [\[ICLR 2026\] Sequences of Logits Reveal the Low Rank Structure of Language Models](sequences_of_logits_reveal_the_low_rank_structure_of_language_models.md)
- [\[ICML 2025\] SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression](../../ICML2025/interpretability/slim_one-shot_quantization_and_sparsity_with_low-rank_approximation_for_llm_weig.md)
- [\[ICLR 2026\] Decomposition of Concept-Level Rules in Visual Scenes](decomposition_of_concept-level_rules_in_visual_scenes.md)
- [\[ICLR 2026\] Sparling: End-to-End Spatial Concept Learning via Extremely Sparse Activations](sparling_end-to-end_spatial_concept_learning_via_extremely_sparse_activations.md)

</div>

<!-- RELATED:END -->
