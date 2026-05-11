---
title: >-
  [Paper Note] CVA: Context-aware Video-text Alignment for Video Temporal Grounding
description: >-
  [CVPR 2026][Video Understanding][Video Temporal Grounding] This paper proposes CVA (Context-aware Video-text Alignment), a framework comprising three synergistic components—Query-aware Context Diversification (QCD)…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Video Temporal Grounding"
  - "Data Augmentation"
  - "Contrastive Learning"
  - "Context Invariance"
  - "Video-Text Alignment"
date: 2026-05-08
content_hash: 2b45fecbbc57f0b1
---

# CVA: Context-aware Video-text Alignment for Video Temporal Grounding

**Conference**: CVPR 2026
**arXiv**: [2603.24934](https://arxiv.org/abs/2603.24934)
**Code**: [https://byeol3325.github.io/projects/CVA/](https://byeol3325.github.io/projects/CVA/)
**Area**: Video Understanding / Temporal Grounding
**Keywords**: Video Temporal Grounding, Data Augmentation, Contrastive Learning, Context Invariance, Video-Text Alignment

## TL;DR
This paper proposes CVA (Context-aware Video-text Alignment), a framework comprising three synergistic components—Query-aware Context Diversification (QCD), Context-invariant Boundary Discrimination (CBD) loss, and Context-enhanced Transformer Encoder (CTE)—to address false negatives and background association issues in video temporal grounding, achieving approximately 5-point improvement in R1@0.7 on QVHighlights.

## Background & Motivation

**Background**: Video Temporal Grounding (VTG) aims to localize target moments in untrimmed videos given text queries, encompassing two subtasks: Video Moment Retrieval (VMR) and Highlight Detection (HD). DETR-based end-to-end approaches have become the dominant paradigm in recent years.

**Limitations of Prior Work**: (1) Models tend to learn spurious correlations—over-associating text queries with static backgrounds rather than focusing on target actions or events. (2) TD-DETR introduced content-mixing augmentation to break such correlations, but the replacement segments are selected independently of the text query, potentially introducing false negatives (i.e., semantically relevant segments are replaced yet labeled as negatives).

**Key Challenge**: The effectiveness of content-mixing augmentation depends on the semantics of the replacement segments—query-agnostic mixing cannot guarantee that replaced segments are truly unrelated to the query.

**Goal**: How can context be diversified while avoiding false negatives? How can models learn context-invariant representations at temporal boundaries?

**Key Insight**: (1) Precompute text-video similarity statistics using CLIP to construct a query-aware valid replacement pool at the dataset level. (2) Apply contrastive learning to reinforce context-invariant representations at temporal boundaries. (3) Employ a hierarchical Transformer to capture multi-scale temporal context.

**Core Idea**: Query-aware data augmentation + boundary-focused contrastive learning + hierarchical temporal modeling = state-of-the-art temporal grounding.

## Method

### Overall Architecture
CVA augments a DETR-based VTG framework with three components: (1) QCD generates semantically consistent augmented samples during training; (2) CTE replaces the standard Transformer encoder to capture multi-scale temporal context; (3) CBD imposes boundary contrastive constraints between two augmented views.

### Key Designs

1. **Query-aware Context Diversification (QCD)**:

    - **Function**: Precomputes cosine similarity matrices between all video segments and all queries using CLIP. Based on the distributional statistics of GT and non-GT pairs, a valid sampling interval $[\theta_{\min}, \theta_{\max}]$ is determined:
    $\theta_{\min} = \text{Percentile}_\alpha(\mathcal{S}_{\text{non}}), \quad \theta_{\max} = \text{Percentile}_\beta(\mathcal{S}_{\text{gt}})$
    - Replacement segments are sampled exclusively from other videos within this similarity range.
    - GT segments and their $p$ neighboring segments are preserved (context preservation strategy).
    - **Design Motivation**: The lower bound $\theta_{\min}$ filters out trivially irrelevant negatives that provide no meaningful learning signal, while the upper bound $\theta_{\max}$ excludes highly similar segments that may constitute false negatives. Percentile-based thresholds are more robust than fixed values.

2. **Context-enhanced Transformer Encoder (CTE)**:

    - **Function**: Consists of $N_b$ stacked blocks, each containing: (a) window self-attention over video features to model local temporal patterns; (b) global self-attention over learnable queries; (c) bidirectional cross-attention to exchange information between video and queries.
    - **Core Formula**: The final output is obtained via hierarchical aggregation with learnable weight fusion:
    $\mathbf{F}_{\text{CTE}} = \omega \cdot \mathbf{F}_v + (1-\omega) \cdot \text{Norm}(\text{MLP}(\text{Concat}_{l=1}^{N_b}(\mathbf{F}^{(l)})))$
    - **Design Motivation**: Standard Transformers apply global attention directly, lacking explicit modeling of local temporal patterns. Window attention captures local dependencies, learnable queries provide global semantic anchors, and bidirectional cross-attention facilitates local-global information exchange.

3. **Context-invariant Boundary Discrimination (CBD)**:

    - **Function**: Given two augmented views $\mathbf{V}'_{\text{mix}}$ and $\mathbf{V}''_{\text{mix}}$ generated by QCD, boundary features at GT segment start and end frames are extracted as anchors and positives. Negatives are drawn from two sources: (a) spatially adjacent background frames (hard boundary negatives); (b) semantically most similar distant background frames (hard semantic negatives).
    - **Core Formula**:
    $\mathcal{L}_{CBD} = -\frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} \log \frac{\exp(s_{p,b})}{\exp(s_{p,b}) + \sum_{\mathbf{z}_n \in \mathcal{Z}^-} \exp(s_{n,b})}$
    - **Design Motivation**: Boundaries are the most critical and error-prone regions for localization. Enforcing consistent boundary representations across different context augmentations encourages the model to learn context-invariant discriminative features. Using both proximity-based and semantics-based hard negatives ensures discriminability along both temporal and semantic dimensions.

### Loss & Training
- $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MR}} + \mathcal{L}_{\text{HD}} + \lambda_{\text{CBD}} \mathcal{L}_{\text{CBD}}$
- MR loss: $\lambda_{\text{L1}} \mathcal{L}_{\text{L1}} + \lambda_{\text{gIoU}} \mathcal{L}_{\text{gIoU}}$
- HD loss: $\lambda_{\text{HD}}(\mathcal{L}_{\text{margin}} + \mathcal{L}_{\text{rank}})$
- $\lambda_{\text{L1}}=10$, $\lambda_{\text{gIoU}}=1$, $\lambda_{\text{HD}}=1$, $\lambda_{\text{CBD}}=0.005$
- QCD parameters: $\alpha=10$, $\beta=60$, replacement ratio 0.3, context preservation window $p=1$
- AdamW optimizer, cosine annealing, batch size 32

## Key Experimental Results

### Main Results — QVHighlights Test Split

| Method | R1@0.5↑ | R1@0.7↑ | mAP Avg↑ | HD mAP↑ |
|--------|---------|---------|----------|---------|
| Moment-DETR | 52.89 | 33.02 | 30.73 | 35.69 |
| QD-DETR | 62.40 | 44.98 | 39.86 | 38.94 |
| CG-DETR | 65.43 | 48.38 | 42.86 | 40.33 |
| TD-DETR | 64.53 | 50.37 | 46.69 | - |
| CDTR | 65.79 | 49.60 | 44.37 | - |
| **CVA (Ours)** | **70.05** | **55.32** | **47.49** | **44.43** |

Gains: R1@0.5 +4.26 (vs. CDTR), R1@0.7 +4.95 (vs. TD-DETR), HD mAP +4.1 (vs. CG-DETR)

### Charades-STA and TACoS

| Dataset | Method | R1@0.5↑ | R1@0.7↑ | mIoU↑ |
|---------|--------|---------|---------|-------|
| Charades | BAM-DETR (prev best) | 59.95 | 39.38 | 52.33 |
| Charades | **CVA** | **62.61** | **40.78** | **53.35** |
| TACoS | BAM-DETR (prev best) | 41.45 | 26.77 | 39.31 |
| TACoS | **CVA** | **43.21** | **27.73** | **41.07** |

### Ablation Study

| Configuration | R1@0.5↑ | R1@0.7↑ | HD mAP↑ | Notes |
|---------------|---------|---------|---------|-------|
| Baseline (QCD basic) | ~63 | ~48 | ~39 | Basic augmentation |
| + QCD (query-aware) | ~68 | ~52 | ~41 | Large R1 improvement |
| + QCD + CTE | ~68.5 | ~53.5 | ~43 | Architectural enhancement |
| + QCD + CTE + CBD | **70.05** | **55.32** | **44.43** | Full CVA |

### Key Findings
- The substantial R1 improvement (~5 points) directly validates the effectiveness of QCD in reducing false negatives.
- CBD contributes most to precise localization, particularly at R1@0.7, where boundary discrimination is more critical under stricter IoU thresholds.
- The three components exhibit clear synergy: QCD provides diverse, high-quality training samples; CTE enables improved temporal modeling; CBD ensures boundary discriminability.
- The method demonstrates consistent effectiveness across three benchmarks with distinct characteristics (QVHighlights, Charades-STA, and TACoS).

## Highlights & Insights
- **Data-centric perspective**: Rather than solely improving model architecture, the work prioritizes training data quality—QCD's approach to resolving false negatives from a data augmentation standpoint constitutes the key innovation.
- **Boundary focus**: CBD precisely directs contrastive learning toward the most critical boundary regions, yielding higher efficiency than frame-level contrastive approaches.
- **Statistics-driven thresholds**: Replacing manually set thresholds with dataset-level similarity distribution statistics (percentile-based) provides greater adaptability.
- **Dual-source hard negatives**: Incorporating both temporally adjacent and semantically similar hard negatives more comprehensively enhances discriminability.

## Limitations & Future Work
- QCD requires precomputing a CLIP similarity matrix (a one-time cost), which may be slow for very large datasets.
- Hyperparameter choices such as window size and query count are not thoroughly discussed.
- Only SlowFast + CLIP video features are used; stronger video encoders may yield further improvements.
- No comparison with recent VLM-based methods is provided.

## Related Work & Insights
- TD-DETR identified the background association problem and addressed it via content mixing; CVA's QCD corrects its false negative deficiency—a representative "problem → partial solution → refinement" research trajectory.
- The combination of window attention (inspired by Swin Transformer) and learnable queries (inspired by DETR) proves effective for temporal tasks.
- The concept of boundary-focused contrastive learning is generalizable to tasks requiring precise temporal boundaries, such as action detection and video segmentation.

## Rating
- Novelty: ⭐⭐⭐⭐ — QCD's query-aware augmentation and CBD's boundary-focused contrastive learning are innovative; CTE is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three benchmarks, comprehensive ablation, and component-level analysis.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear; method description is detailed.
- Value: ⭐⭐⭐⭐⭐ — An approximately 5-point R1 improvement is substantial and represents a meaningful contribution to the VTG field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling](hieramamba_video_temporal_grounding_via_hierarchical_anchor-mamba_pooling.md)
- [\[CVPR 2026\] SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding](slotvtg_object-centric_adapter_for_generalizable_video_temporal_grounding.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](how_should_video_llms_output_time.md)
- [\[AAAI 2026\] Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction](../../AAAI2026/video_understanding/explicit_temporal-semantic_modeling_for_dense_video_captioning_via_context-aware.md)
- [\[CVPR 2026\] Hear What Matters! Text-conditioned Selective Video-to-Audio Generation](hear_what_matters_text-conditioned_selective_video-to-audio_generation.md)

</div>

<!-- RELATED:END -->
