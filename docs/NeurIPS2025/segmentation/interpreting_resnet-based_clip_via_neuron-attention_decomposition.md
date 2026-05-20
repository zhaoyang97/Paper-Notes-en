---
title: >-
  [Paper Note] Interpreting ResNet-based CLIP via Neuron-Attention Decomposition
description: >-
  [NeurIPS 2025][Segmentation][CLIP interpretability] This paper proposes a neuron-attention decomposition method to interpret CLIP-ResNet by decomposing model outputs into pairwise contribution paths of neurons and attent…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "CLIP interpretability"
  - "neuron-attention decomposition"
  - "semantic segmentation"
  - "ResNet"
  - "mechanistic interpretability"
date: 2026-05-08
content_hash: 7ea5a57ca5cc5db9
---

# Interpreting ResNet-based CLIP via Neuron-Attention Decomposition

**Conference**: NeurIPS 2025
**arXiv**: [2509.19943](https://arxiv.org/abs/2509.19943)  
**Code**: None  
**Area**: Segmentation
**Keywords**: CLIP interpretability, neuron-attention decomposition, semantic segmentation, ResNet, mechanistic interpretability

## TL;DR

This paper proposes a neuron-attention decomposition method to interpret CLIP-ResNet by decomposing model outputs into pairwise contribution paths of neurons and attention pooling heads. The resulting neuron-head pairs are shown to admit rank-1 approximations, exhibit sparsity, and capture sub-concepts. The method is applied to training-free semantic segmentation (mIoU 26.2% on PASCAL Context, surpassing MaskCLIP by 15%) and dataset distribution shift monitoring.

## Background & Motivation

**Gap in CLIP interpretability**: Existing CLIP interpretability methods (TextSpan, SPLICE, SecondOrderLens) primarily target CLIP-ViT, exploiting the linear additivity of residual streams to decompose outputs. However, CLIP-ResNet applies ReLU nonlinearities after each residual block, rendering these methods inapplicable.

**Architectural differences**: CLIP-ViT employs self-attention blocks and a class token, whereas CLIP-ResNet uses convolutions followed by a final attention pooling layer. Existing ViT-based interpretation methods cannot be directly transferred.

**Core insight**: Although the early layers of CLIP-ResNet are not linearly decomposable, **the segment from the last convolutional layer to the attention pooling layer admits a linear decomposition**. Refining this decomposition to the granularity of neuron-head pairs yields more interpretable semantic units than analyzing neurons or attention heads in isolation.

## Method

### Overall Architecture

The core of the method is a three-level mathematical decomposition of CLIP-ResNet outputs:
1. **Attention head + token decomposition**: $M_{\text{image}}(I) = \sum_h \sum_i a_i^h(I) \cdot z_i \cdot W_{VO}^h$
2. **Neuron-head pair decomposition**: $M_{\text{image}}(I) = \sum_n \sum_h \sum_i r_i^{n,h}(I)$, where $r_i^{n,h} = a_i^h(I) \cdot z_i \cdot W_{VO}^{n,h}$
3. Since all contributions reside in the joint image-text embedding space of CLIP, they can be directly compared and interpreted against text.

### Key Designs

**1. Mathematical Decomposition**

The image representation of CLIP-ResNet is $M_{\text{image}}(I) = \text{AttnPool}(Z'(I))$, where $Z(I) \in \mathbb{R}^{C \times H' \times W'}$ is the output of the last convolutional layer. The attention pooling is a standard multi-head attention that returns only the class token output.

Exploiting the mathematical structure of multi-head attention:
- Each attention head $h$ maps tokens to the output space via an OV matrix $W_{VO}^h \in \mathbb{R}^{C \times d}$
- Each row of the OV matrix corresponds to a neuron $n$, enabling further decomposition into neuron-head pairs

The contribution of each neuron-head pair $(n,h)$ is $r^{n,h}(I) = \sum_i a_i^h(I) \cdot z_i \cdot W_{VO}^{n,h}$, a $d$-dimensional vector residing in the joint image-text space.

**2. Three Key Properties of Neuron-Head Pairs**

**(a) Rank-1 approximation**: The contribution $r^{n,h}(I)$ of each neuron-head pair can be well approximated by its first principal component $\hat{r}^{n,h}$. Reconstructing with a single direction preserves ImageNet zero-shot classification accuracy at 70.7% (unchanged), whereas neuron-only decomposition with one principal component degrades accuracy to 66.9% (−3.8%).

**(b) Sparsity**: Retaining 20% of neuron-head pairs (ranked by contribution norm) and mean-ablating the remaining 80% results in only ~5% drop in ImageNet accuracy. By contrast, retaining 20% of neuron-only contributions leads to approximately 25% accuracy degradation.

**(c) Sub-concept specificity**: Certain neuron-head pairs capture sub-concepts of their corresponding neuron's concept. For example, neuron #624 represents the concept "butterfly," while neuron-head pair #(624,21) specifically represents the sub-concept "butterfly costume."

**3. Sparse Decomposition against Text**

Orthogonal Matching Pursuit (OMP) is used to decompose each $\hat{r}^{n,h}$ as a sparse linear combination of 30,000 common English words: $\hat{r}^{n,h} \approx \sum_{j=1}^m \gamma_j^{n,h} \cdot M_{\text{text}}(t_j)$. Using 64 text descriptors, neuron-head pairs achieve higher reconstruction accuracy than the neuron-only baseline.

**4. Application to Semantic Segmentation**

Neuron-head pairs are employed for training-free semantic segmentation:
- Collect the last-layer activation map $Z(I)$ ($C \times H' \times W'$) and per-token per-head contributions $r_i^h(I)$
- Compute text similarity heatmaps $L_{\text{sim}}(I)$, representing the cosine similarity between each spatial position $i$, head $h$, and class text $t_j$
- Select top-$k$ neuron-head pairs $(n_r, h_r)$ with highest cosine similarity to class text $t_j$
- Segmentation logits: $\hat{L}(I) = \sum_{r=1}^k Z^{n_r}(I) \circ L_{\text{sim}}^{h_r}(I)$

The key idea is to **element-wise multiply** the spatial activation map of neurons with the semantic heatmap of attention heads. These two sources are complementary: neurons provide precise spatial localization, while attention heads provide semantic matching.

### Loss & Training

The proposed method is **entirely training-free**, requiring no additional training or fine-tuning. All analyses are based on the pretrained OpenAI CLIP-RN50x16 model. Decomposition, sparse coding, and segmentation are all performed at inference time. Segmentation evaluation uses slide inference (shorter side resized to 512, with a 384×384 window and stride 192).

## Key Experimental Results

### Main Results: PASCAL Context Semantic Segmentation

| Method | mIoU (%) | Backbone |
|--------|----------|----------|
| Self-self attention | 22.2 | RN50x16 |
| MaskCLIP | 22.8 | RN50x16 |
| **Ours** | **26.2** | **RN50x16** |
| SC-CLIP (SOTA) | 40.1 | ViT-B/16 |

The proposed method achieves a 15% relative improvement over MaskCLIP on CLIP-ResNet (22.8→26.2 mIoU), though a gap remains compared to SC-CLIP using ViT.

### Decomposition Strategy Comparison

| Method | mIoU (%) |
|--------|----------|
| Neuron activation map only | 16.5 |
| Attention head heatmap only | 24.7 |
| **Element-wise product (Ours)** | **26.2** |

The multiplicative combination of neurons and attention heads outperforms either in isolation, validating their complementarity.

### Ablation Study

**Rank-1 approximation validation** (ImageNet zero-shot classification accuracy):

| Reconstruction Method | Accuracy (%) |
|-----------------------|--------------|
| Original (baseline) | 70.7 |
| Neuron-head, 1 PC | 70.7 |
| Neuron, 1 PC | 66.9 |
| Neuron, 2 PCs | 69.0 |
| Neuron, 4 PCs | 70.0 |

Neuron-head pairs require only one principal component for perfect reconstruction, whereas neurons require four PCs to approach the baseline.

**Distribution shift monitoring** (Stanford Cars dataset):
- Point-biserial correlation between neuron-head contributions for the "yellow" concept and ground-truth proportion: 0.85
- Correlation for the "convertible" concept: 0.71
- All concept contributions are obtained from a single forward pass, making the approach suitable for large-scale datasets

### Key Findings

1. **Neuron-head pairs constitute more appropriate interpretable units in CLIP-ResNet than neurons or attention heads individually.**
2. Polysemanticity is substantially reduced in neuron-head pairs—far more pairs exhibit low inertia compared to neurons alone.
3. The sparsity of neuron-head pairs enables 20% of pairs to account for the majority of model output.
4. Sub-concept discovery: a "butterfly" neuron can be decomposed into sub-concepts such as "butterfly costume" and "butterfly wings."

## Highlights & Insights

1. **Filling the interpretability gap for CLIP-ResNet**: This is the first work to systematically analyze the internal computation paths of CLIP-ResNet.
2. **Elegant mathematical decomposition**: Leveraging the linear structure of attention pooling, the output is exactly decomposed into a sum of neuron-head pair contributions.
3. **Training-free segmentation**: The method does not modify CLIP's internal computation (e.g., no self-self attention tricks) and directly utilizes the genuine output decomposition.
4. **Integration of theory and application**: A natural progression from rank-1 properties and sparsity analysis to sub-concept discovery, segmentation, and distribution shift monitoring.

## Limitations & Future Work

1. **Only the last layer is analyzed**: The method is applicable only to the final convolutional block of ResNet and cannot analyze earlier layers (ViT-based methods can exploit spatial consistency across intermediate layers).
2. **Residual polysemanticity in neuron-head pairs**: Although superior to neurons, certain pairs still exhibit polysemantic perturbations.
3. **Large performance gap with ViT-based methods**: mIoU 26.2% vs. SC-CLIP's 40.1%, partly attributable to inherent limitations of the ResNet architecture.
4. **Evaluation limited to PASCAL Context**: Evaluation on larger segmentation benchmarks (e.g., ADE20K, COCO-Stuff) is absent.
5. Future work may explore generalizing this decomposition framework to other architectures employing attention pooling.

## Related Work & Insights

- Complementary to TextSpan (CLIP-ViT): TextSpan analyzes token contributions in ViT, while this paper analyzes neuron-head contributions in ResNet.
- The sub-concept discovery mechanism could be used to construct automated concept taxonomies.
- The distribution shift monitoring application can be extended into an automated auditing tool for CLIP models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First interpretability method for CLIP-ResNet with elegant mathematical derivation
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive quantitative and qualitative analysis, though segmentation benchmarks are limited
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical structure with smooth progression from theory to application
- Value: ⭐⭐⭐⭐ Opens a new direction for CLIP-ResNet interpretability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CLIP Is Shortsighted: Paying Attention Beyond the First Sentence](../../CVPR2026/segmentation/clip_is_shortsighted_paying_attention_beyond_the_first_sentence.md)
- [\[NeurIPS 2025\] Attention (as Discrete-Time Markov) Chains](attention_as_discrete-time_markov_chains.md)
- [\[ICCV 2025\] CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation](../../ICCV2025/segmentation/corrclip_reconstructing_patch_correlations_in_clip_for_openv.md)
- [\[NeurIPS 2025\] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)
- [\[NeurIPS 2025\] Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective](towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)

</div>

<!-- RELATED:END -->
