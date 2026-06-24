---
title: >-
  [Paper Note] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment
description: >-
  [NeurIPS 2025][Multimodal VLM][Cross-modal retrieval] This paper proposes MAPLE, a framework that leverages the inherent modality alignment capabilities of off-the-shelf MLLMs to automatically construct preference data, and introduces a Relative Preference Alignment (RPA) loss to guide cross-modal representation learning, achieving significant improvements on fine-grained retrieval tasks.
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Cross-modal retrieval"
  - "preference alignment"
  - "DPO"
  - "modality gap"
  - "MLLM"
date: 2026-05-08
content_hash: f9184d788fb6b6fa
---

# Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment

**Conference**: NeurIPS 2025
**arXiv**: [2506.06970](https://arxiv.org/abs/2506.06970)  
**Code**: N/A  
**Area**: Multimodal VLM
**Keywords**: Cross-modal retrieval, preference alignment, DPO, modality gap, MLLM

## TL;DR

This paper proposes MAPLE, a framework that leverages the inherent modality alignment capabilities of off-the-shelf MLLMs to automatically construct preference data, and introduces a Relative Preference Alignment (RPA) loss to guide cross-modal representation learning, achieving significant improvements on fine-grained retrieval tasks.

## Background & Motivation

Contrastive learning models such as CLIP have demonstrated strong performance in cross-modal retrieval; however, their feature spaces exhibit a pronounced **modality gap**—a systematic separation between image and text embeddings in the shared space—which limits retrieval effectiveness.

A key observation motivates this work: **off-the-shelf MLLMs (e.g., Qwen2-VL) inherently possess strong modality alignment capabilities.** By proposing a unified metric based on the 1-Wasserstein distance (WD) that enables simultaneous comparison of logit-based models (MLLMs) and embedding-based models (CLIP), the authors show that the alignment quality of MLLMs substantially surpasses that of CLIP.

However, fine-tuning MLLMs into retrieval models degrades this inherent alignment capability. The central question therefore becomes: **how can the powerful cross-modal alignment ability of MLLMs be preserved while converting them into retrievers?**

Two core challenges are identified:
1. Standard contrastive learning performs coarse-grained alignment by uniformly pushing away all negatives, ignoring fine-grained semantic differences.
2. Directly applying contrastive fine-tuning to MLLMs destroys their pre-existing modality alignment priors.

## Method

### Overall Architecture

MAPLE (Modality-Aligned Preference Learning for Embeddings) consists of two main components:
1. **Preference data construction**: offline hard negative mining followed by online MLLM-based scoring.
2. **Preference alignment training**: aligning the embedding space with MLLM preferences via the RPA loss.

### Key Designs

#### 1. MLLM-Based Retriever Architecture

- The model is initialized from a pretrained MLLM.
- The causal attention mask is replaced with **bidirectional attention**.
- **Mean pooling** is applied over the final hidden states to obtain retrieval representations.
- The model is fine-tuned using LoRA.

#### 2. Preference Data Construction

**Offline stage — Candidate generation**:
- DINOv2 is used to extract image embeddings; SemDedup is applied for deduplication.
- For each image, top-$K$ nearest neighbors are retrieved as the hard negative candidate set $\mathcal{C}_i^{img}$.
- The MLLM's multi-image reasoning capability is used to generate discriminative descriptions, forming the text candidate set $\mathcal{C}_i^{txt}$.

**Online stage — Scoring and structuring**:
- Alignment scores are computed by prompting the MLLM to output "yes"/"no" for image–text pairs; softmax is applied to obtain alignment scores $\alpha_{ii}$.
- Candidates are ranked by alignment score, and two preference structures are constructed:
    - **Pairwise preference**: all candidate pairs $(x_{r_a}, x_{r_b})$ satisfying $a < b$.
    - **Listwise preference**: exploiting the suffix structure of the complete ranking list.

#### 3. Relative Preference Alignment (RPA) Loss

Starting from DPO, two key modifications are introduced:

**Eliminating the reference model**: A uniform prior $U$ is adopted as the reference model $\pi_w$, simplifying the objective to:
$$\mathcal{L}_{\text{DPO-simplified}} = -\mathbb{E}[\log \sigma(\beta \log \pi_\theta(y_w|x) - \beta \log \pi_\theta(y_l|x))]$$

**Adapting to embedding models**: Log-probabilities are replaced with scaled similarity scores $\beta(z^{anchor} \cdot z^{candidate})$.

**Pairwise RPA loss**:
$$\mathcal{L}_{\text{RPA-Pairwise}}^{txt2img} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{0 \le k < l \le K} (\alpha_{i,r_k} - \alpha_{i,r_l}) \log \sigma(s_{ik} - s_{il})$$

The preference weight $(\alpha_{r_k} - \alpha_{r_l})$ ensures that pairs with larger preference gaps receive greater attention.

**Listwise RPA loss**:
$$\mathcal{L}_{\text{RPA-Listwise}}^{txt2img} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=0}^{K-1} w_{ik} \log \frac{\exp(s_{ik})}{\sum_{j=k}^{K} \exp(s_{ij})}$$

where the weight $w_{ik}$ is the average difference in MLLM alignment scores between the $k$-th candidate and all subsequent candidates.

### Loss & Training

**Regularized joint optimization**:
$$\mathcal{L} = \lambda \mathcal{L}_{RPA} + (1 - \lambda) \mathcal{L}_{contrast}$$

- $\mathcal{L}_{contrast}$ is the standard contrastive loss, serving as regularization to prevent feature collapse from excessive alignment.
- $\lambda$ balances the preference alignment strength against general retrieval capability.
- Training data: a subset of OpenImages.
- An extended negative pool strategy is employed to implicitly increase the effective batch size without additional computational overhead.

## Key Experimental Results

### Main Results

| Model | COCO T/I R@1 | Flickr30k T/I R@1 | Winoground T/I | NaturalBench T/I |
|------|-------------|-------------------|----------------|-----------------|
| SigLIPv2 (2B) | 72.8/56.1 | 95.4/86.0 | 39.8/17.0 | 65.5/68.7 |
| VladVA (LLaVA-7B) | 72.9/59.0 | 94.3/83.3 | 40.5/17.5 | -/- |
| **MAPLE (Qwen2-VL-7B)** | **75.5/60.3** | **94.3/86.1** | **56.0/32.7** | **76.1/76.8** |

Improvements on fine-grained retrieval are particularly pronounced: Winoground Text +13.5 and Image +15.2 relative to VladVA.

### Ablation Study

| Method | COCO T/I | Winoground T/I | NaturalBench T/I |
|------|---------|----------------|-----------------|
| Baseline ($\mathcal{L}_{contrast}$) | 74.0/54.4 | 42.5/20.5 | 61.4/62.5 |
| + RPA-Pairwise only | 51.9/52.4 | 48.8/34.7 | 70.1/77.3 |
| $\mathcal{L}_{contrast}$ + RPA-Listwise | 71.9/58.6 | 51.0/28.2 | 69.2/71.2 |
| + Extended negatives + RPA | **75.5/60.3** | **56.0/32.7** | **76.1/76.8** |

### Key Findings

1. **Using RPA loss alone** substantially improves fine-grained retrieval at the cost of general retrieval performance; adding $\mathcal{L}_{contrast}$ regularization recovers both.
2. **Listwise RPA outperforms Pairwise**: the listwise formulation accounts for the full ranking structure rather than treating pairs independently.
3. **Extended negative pool** benefits both general and fine-grained retrieval by effectively increasing the batch size.
4. **Modality gap analysis**: MAPLE with RPA significantly reduces the distributional gap ($W_{dist\text{-}gap}$) while improving the discriminative gap ($W_{disc\text{-}gap}$).

## Highlights & Insights

1. **MLLMs inherently possess strong modality alignment ability**: this observation is independently valuable, suggesting that MLLMs can serve as "alignment teachers" for other models.
2. **Elegant transfer from DPO to embedding space**: replacing log-probabilities with similarity scores and eliminating the reference model adapts the framework naturally to the retrieval setting.
3. **Refined preference weighting**: using MLLM alignment score differences as preference strength weights focuses training on sample pairs with clear preference signals.
4. **Unified modality gap metric**: the Wasserstein distance-based metric enables comparisons across architectures with different output formats (logit-based vs. embedding-based).

## Limitations & Future Work

1. Cross-modal representations may inherit biases inherent to the MLLM backbone.
2. The approach has not been validated on more complex tasks such as composed retrieval.
3. Computing MLLM alignment scores during the online stage introduces additional training cost.
4. General retrieval performance shows only marginal improvement over the baseline in some configurations (e.g., COCO Text R@1 increases from 74.0 to 75.5, with slight degradation in certain settings).
5. Future work could explore additional MLLM backbones and larger-scale training data.

## Related Work & Insights

- **Connection to DPO/RLHF**: the work extends LLM alignment techniques to the domain of visual retrieval.
- **Alternative direction for improving CLIP**: rather than modifying the encoder architecture, the method leverages external MLLM knowledge to improve alignment.
- **Inspiration**: similar ideas could be applied to video retrieval and fine-grained matching in multimodal RAG systems.

## Rating

- Novelty: ⭐⭐⭐⭐ (DPO-to-embedding-space transfer and discovery of MLLM alignment priors)
- Experimental Thoroughness: ⭐⭐⭐⭐ (multiple general and fine-grained benchmarks with comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (technically rich but somewhat complex in structure)
- Value: ⭐⭐⭐⭐ (significant gains on fine-grained retrieval; the MLLM prior finding is broadly inspiring)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs](../../ICCV2025/multimodal_vlm/instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[ICML 2025\] Vision-Language Models Create Cross-Modal Task Representations](../../ICML2025/multimodal_vlm/vision-language_models_create_cross-modal_task_representations.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[NeurIPS 2025\] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning](on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)

</div>

<!-- RELATED:END -->
