---
title: >-
  [Paper Note] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations
description: >-
  [AAAI 2026][Interpretability][Concept Bottleneck Models] This paper proposes PCBM-ReD, a post-hoc concept bottleneck model that automatically extracts concepts from pretrained visual encoders via sparse autoencoders…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Concept Bottleneck Models"
  - "CLIP"
  - "Sparse Decomposition"
  - "Representation Learning"
date: 2026-05-08
content_hash: 9e327d79a52a924f
---

# Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations

**Conference**: AAAI 2026
**arXiv**: [2601.12303](https://arxiv.org/abs/2601.12303)
**Code**: [GitHub](https://github.com/peterant330/PCBM_ReD)
**Area**: Interpretability
**Keywords**: Concept Bottleneck Models, Interpretability, CLIP, Sparse Decomposition, Representation Learning

## TL;DR

This paper proposes PCBM-ReD, a post-hoc concept bottleneck model that automatically extracts concepts from pretrained visual encoders via sparse autoencoders, annotates and filters them using MLLMs, and selects a representative subset through reconstruction-guided search. Image representations are then sparsely decomposed into linear combinations of concept embeddings via CLIP's vision-language alignment. The method achieves state-of-the-art accuracy on 11 classification benchmarks while maintaining interpretability.

## Background & Motivation

The "black-box" nature of deep learning models hinders their deployment in safety-critical domains such as medical imaging and autonomous driving. Concept Bottleneck Models (CBMs) address this by routing predictions through human-interpretable intermediate concepts; however, existing approaches suffer from several limitations:

**Post-hoc methods**: Extracted concepts do not necessarily reflect the model's actual reasoning process, and causal relationships between concepts and predictions are not guaranteed.

**Hand-crafted concepts** (original CBM): Time-consuming to design and often incomplete in coverage.

**LLM-generated concepts** (LaBo, Label-free CBM): May include non-visual attributes (e.g., food taste, bird behavior) and are decoupled from the data distribution and encoder capacity.

**Concept independence**: Existing methods do not enforce linear independence among concepts, which compromises intervention effectiveness.

**Accuracy–interpretability trade-off**: Existing CBMs incur notable accuracy degradation compared to end-to-end models.

**Core Insight**: Concepts should be extracted from the representations of pretrained encoders in a data-driven manner, rather than designed independently of the model and data, so as to maximally exploit the encoder's representational capacity.

## Method

### Overall Architecture

PCBM-ReD follows a three-stage pipeline:

**Stage 1: Data-Driven Concept Discovery**
- A sparse autoencoder (SAE) is applied to the latent space of the CLIP visual encoder for dictionary learning: $\mathbf{I}_i \approx \mathbf{V}\mathbf{u}_i$
- Each column of $\mathbf{V}$ represents a concept; the values in $\mathbf{u}_i$ reflect the importance of each concept for a given image.
- For each concept, the top-$K$ most activated images are selected and described by an MLLM (Llama-3.2-11B-Vision).
- An LLM (DeepSeek-V3) aggregates the descriptions to generate candidate concepts, which are then scored and filtered to retain only those that are visually identifiable, discriminative, and free of shortcut features.

**Stage 2: Reconstruction-Guided Concept Selection**
- A linearly independent subset is greedily selected from the candidate concepts such that their embeddings maximally reconstruct the image representation space.
- Algorithm 1 iteratively selects the concept that minimizes reconstruction error while enforcing linear independence with respect to already-selected concepts.

**Stage 3: Post-hoc Class–Concept Association**
- Leveraging CLIP's vision-language alignment, image embeddings are sparsely decomposed into a weighted sum of concept text embeddings.
- A linear layer is trained on the reconstructed embeddings to predict class labels.

### Key Designs

#### Reconstruction-Guided Concept Selection Algorithm

The optimization objective is:

$$\min_{\mathcal{C}} \sum_{i=1}^N \min_{\beta_i(\mathcal{C})} \|\mathbf{I}_i - \mathbf{R}(\mathcal{C})^T \beta_i(\mathcal{C})\|_F^2$$

A greedy strategy avoids the combinatorial explosion of discrete optimization. Key efficiency techniques include:
- Incremental updates via projection matrix $\mathbf{P}$ to avoid solving from scratch at each step.
- Linear dependency checking (skipping when $z=0$) to ensure concept independence.
- The algorithm is fully unsupervised, making it applicable to zero-shot and few-shot settings.

#### Sparse Decomposition and Concept Scoring

Orthogonal Matching Pursuit (OMP) is used for sparse coding:

$$\mathbf{I}_i = \hat{\mathbf{I}}_i + \epsilon_i = \sum_{j=1}^m w_j^i \mathbf{c}_j + \epsilon_i$$

where only $n < m$ of the $w_j^i$ are nonzero, ensuring high interpretability (each image is explained by a small number of salient concepts).

The residual $\epsilon_i$ is discarded; only the fitted representation $\hat{\mathbf{I}}_i$ is used for classification, satisfying the CBM abstraction.

#### Weight Matrix Initialization

The classifier weights $\mathbf{W}$ are initialized with the text embeddings of "This is a photo of [cls]", inheriting CLIP's zero-shot capability.

### Loss & Training

- A linear head is trained using the Adam optimizer with batch size 64 and learning rate $5 \times 10^{-5}$.
- The visual encoder is frozen; only the linear classification layer is trained.
- The default backbone is CLIP ViT-L/14; performance saturates at a bottleneck size of approximately 300 concepts.

## Key Experimental Results

### Main Results

**Table 1: Top-1 accuracy on 11 datasets (CLIP ViT-L/14), fully supervised setting**

| Method | Interpretable | ImageNet | CIFAR10 | CIFAR100 | Food | Aircraft | Flower | CUB | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Linear Probe | ✗ | 83.90 | 98.10 | 87.48 | 93.17 | 64.03 | 99.45 | 84.54 | 87.38 |
| LaBo | ✓ | 83.97 | 97.75 | 86.04 | 92.45 | 61.42 | 99.35 | 81.90 | 85.72 |
| Res-CBM | ✓ | 82.98 | 97.77 | 83.01 | 90.17 | 54.67 | 97.85 | 79.27 | 83.39 |
| **PCBM-ReD** | ✓ | **84.48** | 98.05 | **87.27** | **93.16** | **62.95** | 99.39 | **84.80** | **86.97** |

**Table 2: Comparison with other CBMs on CLIP RN50**

| Method | CIFAR10 | CIFAR100 | CUB | Avg |
|---|---|---|---|---|
| Linear Probe | 88.80 | 70.10 | 72.14 | 77.01 |
| PCBM | 84.50 | 56.00 | 63.63 | 68.04 |
| Label-free CBM | 86.40 | 65.13 | 62.40 | 71.31 |
| VLG-CBM | 88.63 | 66.48 | 66.03 | 73.71 |
| **PCBM-ReD** | **88.61** | **70.03** | **72.01** | **76.88** |

### Ablation Study

1. **Bottleneck size**: Reasonable accuracy is achieved with as few as 50 concepts; performance saturates around 300 concepts, independent of the number of classes.
2. **Concept creation strategy**: Data-driven concept discovery > LLM-generated concepts > WordNet core concepts.
3. **Concept selection method**: Reconstruction-guided selection > K-means > random sampling; the gap is more pronounced at smaller bottleneck sizes.
4. **Concept scoring and association**: Sparse decomposition >> CLIP similarity scoring, with a substantial accuracy gap.
5. **Concept source alignment**: Performance degrades when the concept extraction encoder mismatches the inference encoder, confirming that concepts must be aligned with the encoder.

### Key Findings

1. **The accuracy gap between PCBM-ReD and Linear Probe is only 0.41%** (averaged over 11 datasets), marking the first time an interpretable CBM approaches end-to-end model performance.
2. **Zero-shot capability is preserved**: The reconstructed embeddings $\hat{\mathbf{I}}_i$ inherit the zero-shot properties of the original embeddings, with near-identical zero-shot accuracy to CLIP.
3. **Consistent few-shot superiority over LaBo**: PCBM-ReD outperforms LaBo by an average of 5.01% in few-shot settings.
4. **Human evaluation** (39 volunteers): PCBM-ReD outperforms LLM-concept baselines across three dimensions—visual identifiability, descriptive faithfulness, and causal relevance.

## Highlights & Insights

- The core paradigm of **"extracting concepts from representations"** bridges the gap between end-to-end models and interpretable CBMs: rather than imposing external concepts, the method discovers concepts already encoded by the model.
- **The three-stage pipeline is elegantly designed**: SAE extraction → MLLM annotation → reconstruction-guided selection → sparse decomposition, with each step having a clear and well-motivated objective.
- **The unsupervised nature of the concept selection algorithm** makes it naturally applicable to zero-shot and few-shot scenarios, a significant extension over existing CBM methods.
- Decomposing visual embeddings into linear combinations of text concept embeddings via CLIP's multimodal alignment rests on a solid theoretical foundation, supported by prior work from Gandelsman et al.

## Limitations & Future Work

1. **Dependence on general-purpose MLLM description quality**: Descriptions of domain-specific images (e.g., dermatological lesions) may be imprecise, leading to performance degradation on datasets such as HAM.
2. **Information loss from residual discarding**: Although the residual term has limited empirical impact, it is theoretically nonzero.
3. **Concept coverage depends on sampling**: A limited number of probe images may result in incomplete concept coverage.
4. **Downstream quality depends on SAE training**: The influence of different dictionary learning methods warrants further investigation.
5. **Extension to video understanding or domain-specific MLLMs for medical imaging** is a promising direction.

## Related Work & Insights

- **Original CBM (Koh et al., 2020)**: Relies on manually designed concepts with manual annotations; PCBM-ReD is fully automated.
- **LaBo (Yang et al., 2023)**: Generates concepts via LLMs, but concepts are decoupled from both data and model.
- **Res-CBM (Shang et al., 2024)**: Approximates residual connections by incrementally adding concepts.
- **Gandelsman et al.**: Demonstrate that CLIP image embeddings can be decomposed as weighted sums of text embeddings, providing the theoretical foundation for this work.
- **Takeaway**: The SAE + MLLM concept discovery pipeline can be applied to interpretability analysis of any multimodal model.

## Rating

| Dimension | Score (1–5) |
|---|---|
| Novelty | 4.0 |
| Technical Depth | 4.5 |
| Experimental Thoroughness | 4.5 |
| Writing Quality | 4.0 |
| Value | 4.0 |
| **Overall** | **4.2** |

## Related Work & Insights

| Method | Concept Source | Data-Driven | Model-Aware | Zero-Shot | Concept Independence | Accuracy (ViT-L avg) |
|---|---|---|---|---|---|---|
| Original CBM (Koh 2020) | Manual design + annotation | ✗ | ✗ | ✗ | ✗ | - |
| CompDL (Yun 2022) | Manual design + CLIP | ✗ | ✗ | ✗ | ✗ | - |
| PCBM (Yuksekgonul 2022) | Manual + residual connection | ✗ | ✗ | ✗ | ✗ | - |
| LaBo (Yang 2023) | LLM-generated | ✗ | ✗ | ✗ | ✗ | 85.72 |
| Label-free CBM (Oikarinen 2023) | LLM-generated | ✗ | ✗ | ✗ | ✗ | - |
| Res-CBM (Shang 2024) | LLM + incremental residual | Partial | ✗ | ✗ | Partial | 83.39 |
| **PCBM-ReD (Ours)** | **SAE + MLLM** | **✓** | **✓** | **✓** | **✓** | **86.97** |

Key differentiators:
- **Concept source**: This work is the first to extract concepts directly from encoder representations via SAE and annotate them with an MLLM, achieving data-driven and model-aware concept discovery simultaneously.
- **Guaranteed concept independence**: The reconstruction-guided selection algorithm explicitly checks for linear independence; no prior method provides this guarantee.
- **Zero-shot and few-shot capability**: Enabled by the unsupervised concept selection and the preserved alignment properties of CLIP embeddings; existing CBM methods lack this capability.
- **Source of accuracy gains**: Concepts are mined from within the representation space, maximally leveraging the encoder's capacity rather than imposing external concepts.

## Inspiration & Connections

1. **SAE as a concept discovery tool**: Sparse autoencoders have been widely used in mechanistic interpretability to analyze internal representations of LLMs (Anthropic's work); this paper extends the approach to visual encoders, suggesting that SAE is a general cross-modal concept discovery paradigm.
2. **"From the model, back to the model" interpretability paradigm**: Rather than assuming concepts are external priors, this approach acknowledges that concepts are already encoded in model representations and merely need to be discovered and named—a principle generalizable to any foundation model.
3. **MLLM as a concept annotator**: Using multimodal LLMs to assign human-interpretable semantic labels to unsupervisedly discovered concepts represents a low-cost human–machine collaborative annotation strategy.
4. **Dual role of sparse decomposition**: It simultaneously provides interpretability (each image depends on only a small number of concepts) and preserves accuracy (reconstructed embeddings approximate the original embeddings), constituting an elegant design.
5. **Implications for downstream applications**: The framework can be directly applied to interpretability requirements in medical image diagnosis by substituting a domain-specific MLLM to improve concept quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](partially_shared_concept_bottleneck_models.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)
- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](../../CVPR2026/interpretability/towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[AAAI 2026\] Flexible Concept Bottleneck Model](flexible_concept_bottleneck_model.md)

</div>

<!-- RELATED:END -->
