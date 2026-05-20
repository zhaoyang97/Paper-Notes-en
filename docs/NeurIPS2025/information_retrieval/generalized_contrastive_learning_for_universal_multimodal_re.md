---
title: >-
  [Paper Note] Generalized Contrastive Learning for Universal Multimodal Retrieval
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][multimodal retrieval] This paper proposes Generalized Contrastive Learning (GCL), which performs contrastive learning over all 6 modality-pair combinations within a mini-batch…
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "multimodal retrieval"
  - "contrastive learning"
  - "GCL"
  - "fused modality"
  - "CLIP"
date: 2026-05-08
content_hash: 9f17582f95ce8937
---

# Generalized Contrastive Learning for Universal Multimodal Retrieval

**Conference**: NeurIPS 2025
**arXiv**: [2509.25638](https://arxiv.org/abs/2509.25638)  
**Code**: Not mentioned  
**Area**: Information Retrieval
**Keywords**: multimodal retrieval, contrastive learning, GCL, fused modality, CLIP

## TL;DR
This paper proposes Generalized Contrastive Learning (GCL), which performs contrastive learning over all 6 modality-pair combinations within a mini-batch (image↔text, image↔image+text, text↔image+text). Without constructing new triplet datasets and using only existing image-text pairs, GCL improves VISTA's average retrieval precision on M-BEIR from 21.18 to 34.06 (+60.8%), and on the text→image+text task of MMEB from 10.1% to 31.1%.

## Background & Motivation
**Background**: Cross-modal retrieval models such as CLIP perform well on standard image-text retrieval, but suffer significant performance degradation when queries or keys involve fused modalities (fused image+text, e.g., Wikipedia pages containing both images and text).

**Limitations of Prior Work**: Prior methods (e.g., VISTA, UniIR) train with triplet datasets that include fused modalities, but (a) require costly additional data annotation or generation; (b) the constructed data covers only limited modality combinations and fails to generalize to unseen ones; (c) training on generated data may cause forgetting of cross-modal tasks.

**Key Challenge**: Standard contrastive learning (InfoNCE) only contrasts image↔text pairs, neglecting the fused modality (image+text)—resulting in only 2 out of the $3 \times 3 = 9$ possible retrieval combinations among three modalities being learned.

**Goal**: Design a loss function that enables retrieval models to handle arbitrary modality combinations without requiring new annotated data.

**Key Insight**: Leverage existing image-text pair data to automatically construct positive and negative samples for all modality combinations within a mini-batch, covering all 6 cross-modal contrastive directions through a unified GCL loss.

**Core Idea**: Extend InfoNCE from 2 cross-modal pairs to 6 (by incorporating the fused modality), obtaining multimodal retrieval capability for free from existing data.

## Method

### Overall Architecture
Built upon existing multimodal retrieval models (VISTA/CLIP) without architectural modifications—only the loss function is replaced. Fused embeddings are computed via simple addition: $e_{it} = e_i + e_t$ (following UniIR). The three modalities $M = \{i, t, it\}$ yield 6 cross-modal positive sample pairs.

### Key Designs

1. **GCL Loss**:

    - Function: Performs contrastive learning over all 6 cross-modal pairs within a mini-batch
    - Standard CL: $S = \{(i,t), (t,i)\}$, only 2 pairs
    - GCL: $P = \{(i,t), (i,it), (t,i), (t,it), (it,t), (it,i)\}$, 6 pairs
    - Core formula: $\mathcal{L}_{GCL} = -\frac{1}{6N}\sum_{j=1}^{N}\sum_{(a,b)\in P}\log\frac{\exp[(e_a^j \cdot e_b^j)/\tau]}{\sum_{m\in M}\sum_{k=1}^{N}\exp[(e_a^j \cdot e_m^k)/\tau]}$
    - Key: The denominator includes embeddings from all 3 modalities, enabling the model to learn a truly unified representation space
    - Design Motivation: Cover all possible retrieval directions so the model is effective for any query→candidate modality combination

2. **Intra-modality Sample Handling**:

    - Function: Same-modality pairs (e.g., image↔image) are masked out and not treated as positive samples
    - Design Motivation: Prevent intra-modality collapse; enforce cross-modal alignment only

3. **Plug-and-Play**:

    - Function: GCL loss directly replaces standard CL loss with no architectural changes
    - Applicability: Effective across three distinct architectures—VISTA (dual-encoder + fusion), CLIP-SF (CLIP + score fusion), and TinyCLIP

### Loss & Training
Training uses existing image-text pair data without additional data construction. Fused embeddings are formed by simple vector addition $e_{it} = e_i + e_t$.

## Key Experimental Results

### Main Results
M-BEIR global retrieval (Recall@50, average over 10 datasets):

| Method | Pretrained | +CL | +CL+Triplet | +GCL |
|--------|-----------|-----|-------------|------|
| VISTA | 21.18 | 25.28 | 24.65 | **34.06** |
| CLIP-SF | 14.92 | 17.52 | - | **21.89** |

MMEB dataset (text→fused image+text Recall@1): VISTA+GCL 31.1% vs. +CL 17.3% (+80% relative improvement).

CoVR video retrieval (Recall@1): 37.32 vs. CL 33.76 vs. Pretrained 31.22.

### Ablation Study

| GCL Component | M-BEIR Avg↑ |
|---------------|-------------|
| CL baseline | 25.28 |
| + Intra-modality separation | 27.13 |
| GCL w/o it-candidate terms | Partial drop |
| **GCL (Full)** | **34.06** |

### Key Findings
- **60.8% gain without new data**: Changing only the loss improves from 21.18→34.06, demonstrating that the 2 contrastive directions in standard CL constitute a massive information bottleneck.
- Training with generated triplet data (+CL+Triplet) underperforms plain CL on some tasks (24.65 vs. 25.28), as it induces cross-modal forgetting.
- GCL yields the most significant gains on tasks involving fused modalities—text→image+text improves by 80%.
- The method generalizes across architectures: VISTA, CLIP-SF, and TinyCLIP all benefit.
- Applicability extends to video retrieval (CoVR), indicating that the fused modality concept is transferable.

## Highlights & Insights
- **A free lunch at the loss level**: Achieving a 60% improvement solely by redefining positive sample pairs in the loss function—without changing the model, data, or training pipeline—reveals a major blind spot in prior loss function design. This "loss-centric" improvement strategy merits exploration in other tasks.
- **The importance of fused modality**: Many real-world retrieval documents are image+text composites (Wikipedia, e-commerce, papers). Standard CLIP falls short in these scenarios, and GCL fills this gap.
- **The double-edged sword of generated data**: LLM-generated triplet data may cause cross-modal forgetting; addressing the problem through loss design is more effective.
- **The simplicity and efficacy of $e_{it} = e_i + e_t$**: Such a minimal fusion operation combined with GCL yields substantial gains, suggesting the bottleneck lies in the training objective rather than the fusion mechanism.

## Supplementary Analysis
- Training with VISTA-generated triplet data (CL+Triplet) underperforms plain CL on certain tasks (24.65 vs. 25.28), as forcing learning on specific modality combinations causes forgetting of others.
- The CoVR video retrieval experiment demonstrates that the fused modality concept extends naturally to video frame + text compositional retrieval.
- Effectiveness on TinyCLIP confirms that GCL benefits small models equally and does not rely on large model capacity.

## Limitations & Future Work
- The fused embedding uses simple addition $e_{it} = e_i + e_t$; more sophisticated fusion strategies (e.g., cross-attention, gated fusion) may yield further improvements.
- Computing 6 pairs per mini-batch triples the computational cost of standard CL, potentially increasing overhead for large-scale training.
- Validation is limited to retrieval tasks; performance on generative, classification, and other downstream tasks remains unexplored.
- The current framework covers three modalities (image/text/image+text); extension to additional modalities (audio, video frames, 3D point clouds, etc.) requires further investigation.
- Negative sample quality remains critical for GCL—when a mini-batch contains many similar samples, false negatives may arise.
- The combination of hard negative mining with GCL is a promising direction for future exploration.

## Related Work & Insights
- **vs. UniIR / VISTA**: These methods train fused-modality retrieval by constructing task-specific triplet datasets. GCL requires no additional data yet achieves superior performance.
- **vs. AlignCLIP**: AlignCLIP improves through intra-modality separation but with limited gains (25.28→27.13). GCL provides more comprehensive coverage (25.28→34.06).
- **Relationship to Barlow Twins**: Both improve representation learning by redesigning contrastive objectives, but GCL extends this to the multimodal setting and explicitly covers fused modalities.

## Rating
- Novelty: ⭐⭐⭐⭐ The GCL loss design is elegant and effective, extending InfoNCE from 2 contrastive directions to 6 with full fused-modality coverage.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive cross-validation across three benchmarks (M-BEIR/MMEB/CoVR) and three model architectures (VISTA/CLIP/TinyCLIP).
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, loss derivation is concise, and Figure 1's modality coverage comparison is immediately intuitive.
- Value: ⭐⭐⭐⭐ A plug-and-play general-purpose improvement for multimodal retrieval with direct applicability to fused-modality retrieval scenarios (Wikipedia/e-commerce/papers).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](../../CVPR2026/information_retrieval/muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)
- [\[NeurIPS 2025\] Scaling Language-Centric Omnimodal Representation Learning](scaling_language-centric_omnimodal_representation_learning.md)
- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)
- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](benchmarking_retrievalaugmented_multimodal_generation_for_do.md)
- [\[NeurIPS 2025\] SuperCLIP: CLIP with Simple Classification Supervision](superclip_clip_with_simple_classification_supervision.md)

</div>

<!-- RELATED:END -->
