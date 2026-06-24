---
title: >-
  [Paper Note] PLOT: Text-based Person Search with Part Slot Attention for Corresponding Part Discovery
description: >-
  [ECCV 2024][Interpretability][text-based person search] This paper proposes the PLOT framework, which utilizes a Part Discovery Module based on Slot Attention to automatically discover corresponding human body parts across modalities (image-text). Combined with Text-based Dynamic Part Attention (TDPA) to dynamically adjust the importance of each part, it thoroughly outperforms state-of-the-art (SOTA) methods on three benchmarks without requiring part-level annotations.
tags:
  - "ECCV 2024"
  - "Interpretability"
  - "text-based person search"
  - "slot attention"
  - "part discovery"
  - "cross-modal retrieval"
  - "contrastive learning"
date: 2026-05-08
content_hash: dd757bd7a1adf248
---

# PLOT: Text-based Person Search with Part Slot Attention for Corresponding Part Discovery

**Conference**: ECCV 2024  
**arXiv**: [2409.13475](https://arxiv.org/abs/2409.13475)  
**Code**: [https://cvlab.postech.ac.kr/research/PLOT](https://cvlab.postech.ac.kr/research/PLOT)  
**Area**: Cross-Modal Retrieval / Person Search  
**Keywords**: text-based person search, slot attention, part discovery, cross-modal retrieval, contrastive learning

## TL;DR
This paper proposes the PLOT framework, which utilizes a Part Discovery Module based on Slot Attention to automatically discover corresponding human body parts across modalities (image-text). Combined with Text-based Dynamic Part Attention (TDPA) to dynamically adjust the importance of each part, it thoroughly outperforms state-of-the-art (SOTA) methods on three benchmarks without requiring part-level annotations.

## Background & Motivation
Text-based person search requires retrieving a target person from a large-scale image gallery based on a natural language description. The key challenge of this task lies in establishing fine-grained correspondence between text and human body parts in images, as distinguishing different individuals often relies on part-level details such as clothing and accessories.

Prior methods study three categories of problems:

**Heuristic part extraction**: Equidistant horizontal partitioning of images is used to extract part features, which is highly sensitive to occlusions and pose variations, and contains substantial irrelevant background noise.

**Limitations of prior work**: Traditional cross-attention methods tend to generate redundant and non-discriminative part features, or rely on external tools (e.g., keypoint detection, attribute segmentation) that increase computational costs.

**Limitations of global representation**: CLIP-based methods (e.g., IRRA) mainly focus on global features without targeted designs for human body parts.

**Core Idea**: This paper utilizes the competitive aggregation of the Slot Attention mechanism to let learnable "part slots" automatically discover and bind to different human body parts. By sharing slots across modalities, part correspondences are naturally established without any part-level supervision.

## Method

### Overall Architecture
Based on a pre-trained CLIP backbone (ViT-B/16 image encoder + CLIP-Xformer text encoder), two representations are extracted for each modality:
- **Global embedding**: from [cls]/[EOS] tokens, capturing global information.
- **Part embeddings**: from the Part Discovery Module, capturing fine-grained information of $K$ parts.

During inference, similarity = global similarity + TDPA-weighted part similarity: $c(\mathbf{g}^{\mathcal{V}}, \mathbf{g}^{\mathcal{T}}) + c_{\text{agg}}(\mathbf{P}^{\mathcal{V}}, \mathbf{P}^{\mathcal{T}}; \mathbf{g}^{\mathcal{T}})$

### Key Designs

1. **Part Discovery Module**:

    - **Function**: Automatically discovers and extracts $K$ part embeddings from image patch tokens and text word tokens.
    - **Mechanism**: Based on the Slot Attention mechanism. Defines $K$ learnable part slots $\mathbf{S}^0 \in \mathbb{R}^{K \times D}$, which are refined into part embeddings through $T$ iterations of Part Slot Attention (PSA) Blocks.
    - PSA Block workflow:
        - Compute attention map: $A_{n,k} = \frac{e^{M_{n,k}}}{\sum_{i=1}^{K} e^{M_{n,i}}}$, where $M = \frac{k(\mathbf{x}^{\mathcal{V}}) q(\mathbf{S}^{t-1})^{\top}}{\sqrt{D_h}}$
        - **Key**: the softmax is normalized over the slot dimension ($K$) instead of the input dimension, forcing slots to compete with each other to bind different input tokens, which prevents overlapping parts.
        - Weighted aggregation: $\bar{A}_{n,k} = \frac{A_{n,k}}{\sum_{i=1}^{N} A_{i,k}}$, $\bar{\mathbf{S}}^t = \text{GRU}(\mathbf{S}^{t-1}, \bar{A}^{\top} v(\mathbf{x}^{\mathcal{V}}))$
        - MLP + Residual connection: $\mathbf{S}^t = \text{MLP}(\bar{\mathbf{S}}^{t-1}) + \bar{\mathbf{S}}^{t-1}$
    - **Design Motivation**: Slot attention was originally used for unsupervised object discovery in object-centric learning. This work introduces it for the first time to human part discovery. Compared to traditional cross-attention (e.g., PAT), the competition mechanism of slot attention naturally encourages part separation.
    - **Cross-modal correspondence**: The Part Discovery Modules of the two modalities share the same set of initial part slots $\mathbf{S}^0$, allowing visual/textual part embeddings originating from the same slot to correspond naturally.

2. **Text-based Dynamic Part Attention (TDPA)**:

    - **Function**: Dynamically adjusts the weights of different part similarities based on the text query.
    - **Mechanism**: Different queries focus on different parts (e.g., if a query only mentions a jacket and shoes, the weights of other parts should be reduced).
    - Key equation:
        - Weight prediction: $\mathbf{a} = \sigma(\text{MLP}(\mathbf{g}^{\mathcal{T}})) \in \mathbb{R}^K$
        - Weighted similarity: $c_{\text{agg}}(\mathbf{P}^{\mathcal{V}}, \mathbf{P}^{\mathcal{T}}; \mathbf{g}^{\mathcal{T}}) = \sum_{k=1}^{K} a_k \cdot c(\mathbf{p}_k^{\mathcal{V}}, \mathbf{p}_k^{\mathcal{T}})$
    - **Difference from prior work**: Prior methods aggregate all part similarities with equal weights, ignoring the focus variance across different queries.

3. **Cross-Modal Masked Language Modeling (CMLM)**:

    - **Function**: Auxiliary loss to promote cross-modal interactive learning.
    - **Mechanism**: Similar to BERT's MLM, 15% of the text tokens are randomly masked. The model recovers the masked words using a transformer after concatenating with the image tokens.
    - Equation: $\mathcal{L}_{\text{CMLM}} = -\frac{1}{L}\sum_{l=1}^{L} \mathbf{y}_l \log(\sigma(\mathbf{f}_l \mathbf{W}_{\text{CMLM}}))$

### Loss & Training
Total loss: $\mathcal{L} = \mathcal{L}_{\text{Global}} + \mathcal{L}_{\text{Part}} + \mathcal{L}_{\text{CMLM}}$

- **Global Alignment Loss**: $\mathcal{L}_{\text{Global}} = \mathcal{L}_{\text{NCE}} + \mathcal{L}_{\text{ID}}$
    - $\mathcal{L}_{\text{NCE}}$: bidirectional InfoNCE contrastive loss to align global embeddings.
    - $\mathcal{L}_{\text{ID}}$: identity classification loss (classifier shared across modalities) to keep same-identity embeddings close.

- **Part Alignment Loss**: $\mathcal{L}_{\text{Part}} = \mathcal{L}_{\text{PartNCE}} + \mathcal{L}_{\text{PartID}}$
    - $\mathcal{L}_{\text{PartNCE}}$: InfoNCE loss using TDPA-weighted similarity $c_{\text{agg}}$.
    - $\mathcal{L}_{\text{PartID}}$: identity classification after concatenating all part embeddings.

- Training configuration: Adam optimizer, 60 epochs, batch size 128, learning rate of $5 \times 10^{-6}$ for CLIP encoders, other parameters ×20, cosine schedule with a 5-epoch warm-up.

## Key Experimental Results

### Main Results
R@1 comparison on three benchmarks (using CLIP-ViT-B/16 backbone):

| Method | CUHK-PEDES R@1 | ICFG-PEDES R@1 | RSTPReid R@1 |
|------|----------------|----------------|--------------|
| TIPCB | 64.26 | - | - |
| IVT | 65.59 | 56.04 | 46.70 |
| CFine | 69.57 | 60.83 | 50.55 |
| IRRA (Prev. SOTA) | 73.38 | 63.46 | 60.20 |
| **PLOT (Ours)** | **75.28** | **65.76** | **61.80** |

Exceeds IRRA by 1.9%, 2.3%, and 1.6% in terms of R@1 on three datasets, comprehensively establishing a new SOTA.

### Ablation Study

**Ablation of Loss Function Combinations (CUHK-PEDES)**:

| Configuration | R@1 | R@5 | R@10 | Description |
|------|-----|-----|------|------|
| Global Only ($\mathcal{L}_{\text{NCE}}$) | 71.39 | 87.65 | 92.74 | Baseline |
| + $\mathcal{L}_{\text{ID}}$ | 71.83 | 88.06 | 92.58 | +0.44 |
| + $\mathcal{L}_{\text{CMLM}}$ | 72.65 | 88.58 | 92.93 | +1.26 |
| + Part Embeddings (w/o $\mathcal{L}_{\text{PartID}}$) | 74.85 | 90.29 | 94.10 | +3.46 |
| **Full Model** | **75.28** | **90.42** | **94.12** | **+3.89** |

Part embeddings make the most significant contribution (R@1 +3.46%), with Part ID loss further enhancing the performance by 0.43%.

**Comparison of Part Discovery Methods**:

| Method | R@1 | R@5 | R@10 | Description |
|------|-----|-----|------|------|
| TIPCB (Equidistant Partitioning) | 73.23 | 89.10 | 94.04 | Heuristic parts |
| PAT (cross-attention) | 72.76 | 89.23 | 93.42 | Traditional attention |
| **PLOT (slot attention)** | **75.28** | **90.42** | **94.12** | Competitive slots |

Slot attention outperforms TIPCB by 2.05% and PAT by 2.52% on R@1, verifying the advantages of competitive part discovery.

### Key Findings
- Part embeddings are the most critical source of improvement: adding them increases R@1 by 3.46% (from 72.65 to 74.85).
- The competitive normalization mechanism of Slot Attention is superior to traditional cross-attention (PAT), as PAT's parts tend to overlap and focus on salient regions only.
- TDPA allows adaptive part weight adjustment based on the query text: for example, when the query does not mention "hat," the weight of the head slot is automatically reduced.
- Visualizations show that each slot learns a semantically consistent part mapping: slot 1 $\rightarrow$ bottom, slot 4 $\rightarrow$ shoes, slot 5 $\rightarrow$ carrying objects, slot 7 $\rightarrow$ top, slot 8 $\rightarrow$ head.
- Cross-modal shared slots make the same slot focus on the same semantic part in both image and text, providing interpretable retrieval.

## Highlights & Insights
- **Novelty of Slot Attention in Part Discovery**: The paper creatively introduces the slot attention mechanism from object-centric learning to human body part discovery. The competition mechanism naturally guarantees non-overlapping parts without external supervision.
- **Slot Sharing for Cross-Modal Correspondence**: By sharing the initial part slots, part embeddings across images and text are aligned naturally, eliminating the need for explicit part-level annotations.
- **TDPA for Query Adaptivity**: The idea of dynamically adjusting part weights according to different queries is simple yet effective, achieved using a MLP with end-to-end learning.
- **Interpretability**: The visualized attention maps of each slot clearly present the model's focus on different human body regions, thereby enhancing the interpretability of the retrieval task.

## Limitations & Future Work
- The authors acknowledge that slots cover the entire image and text, meaning some slots might bind to irrelevant background regions. TDPA partially alleviates this, but more explicit solutions (e.g., foreground constraints) would be beneficial.
- The number of slots $K=8$ is fixed, and adaptive slot counts have not been explored.
- Validated only on CLIP-ViT-B/16; larger backbones (such as ViT-L/14) remain untested.
- Comparison with recent LLM-based methods (e.g., utilizing GPT-4V for person search) is lacking.

## Related Work & Insights
- **vs IRRA**: IRRA only aligns global features. PLOT adds part-level alignment on top of this, achieving an R@1 improvement of 1.9-2.3%.
- **vs TIPCB**: TIPCB's equidistant partition is a coarse approximation of parts. PLOT learns semantically consistent parts through slot attention.
- **vs PAT (cross-attention)**: PAT's traditional cross-attention lacks a competitive mechanism, leading to overlapping parts. PLOT's slot attention makes each slot compete to bind different regions, making the parts more discriminative.

## Rating
- Novelty: ⭐⭐⭐⭐ The first application of Slot Attention in person search. The unified framework for part discovery and cross-modal correspondence is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Achieves state-of-the-art results across three benchmarks with extensive ablation studies (loss combinations, part methods, TDPA) and rich visualization analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical formulations, well-justified motivations, and effective visualizations.
- Value: ⭐⭐⭐⭐ The concept of part-level cross-modal alignment can be widely transferred to other fine-grained retrieval tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Interpretable Image Classification via Non-parametric Part Prototype Learning](../../CVPR2025/interpretability/interpretable_image_classification_via_non-parametric_part_prototype_learning.md)
- [\[ICLR 2026\] Adaptive Concept Discovery for Interpretable Few-Shot Text Classification](../../ICLR2026/interpretability/adaptive_concept_discovery_for_interpretable_few-shot_text_classification.md)
- [\[ICLR 2026\] From Concepts to Components: Concept-Agnostic Attention Module Discovery in Transformers](../../ICLR2026/interpretability/from_concepts_to_components_concept-agnostic_attention_module_discovery_in_trans.md)
- [\[ICML 2026\] AI Engram: In Search of Memory Traces in Artificial Intelligence](../../ICML2026/interpretability/ai_engram_in_search_of_memory_traces_in_artificial_intelligence.md)
- [\[ECCV 2024\] Improving Intervention Efficacy via Concept Realignment in Concept Bottleneck Models](improving_intervention_efficacy_via_concept_realignment_in_concept_bottleneck_mo.md)

</div>

<!-- RELATED:END -->
