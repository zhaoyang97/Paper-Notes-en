---
title: >-
  [Paper Note] Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning
description: >-
  [CVPR 2026][Segmentation][Dense Video Captioning] This paper proposes STaRC, a framework that leverages supervised frame-level saliency learning to jointly drive retrieval (saliency-guided segmentation and retrieval) and…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Dense Video Captioning"
  - "Saliency Learning"
  - "Retrieval Augmentation"
  - "Temporal Segmentation"
  - "Optimal Transport"
date: 2026-05-08
content_hash: ffe38dc66bd36646
---

# Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning

**Conference**: CVPR 2026
**arXiv**: [2603.11460](https://arxiv.org/abs/2603.11460)
**Code**: [GitHub](https://github.com/ermitaju1/STaRC)
**Area**: Segmentation
**Keywords**: Dense Video Captioning, Saliency Learning, Retrieval Augmentation, Temporal Segmentation, Optimal Transport

## TL;DR

This paper proposes STaRC, a framework that leverages supervised frame-level saliency learning to jointly drive retrieval (saliency-guided segmentation and retrieval) and caption generation (saliency prompt injection into the decoder), achieving substantial improvements in temporal alignment and caption quality for dense video captioning (DVC).

## Background & Motivation

Dense video captioning (DVC) requires detecting multiple events in long videos and generating natural language descriptions for each, fundamentally differing from single-sentence video captioning. Recent retrieval-augmented approaches have shown promise by retrieving relevant captions from external databases to enhance event understanding in the decoder.

However, retrieval relies on video segmentation—clustering frames into segments—where segmentation quality directly impacts caption generation quality. Existing methods exhibit notable deficiencies in temporal segmentation:
- **HiCM2**: Employs uniform sampling to construct fixed-length segments, failing to accommodate variable-length events.
- **Sali4Vid**: Derives boundaries from inter-frame similarity changes, but saliency is inferred via timestamp heuristics without supervised learning.

Through correlation analysis, the authors verify a key finding: **improvements in segment quality metrics (Recall@0.5, Mean IoU, Matched Segments) are strongly positively correlated with downstream DVC metrics (CIDEr, METEOR)**. When segment boundaries more closely match ground-truth event boundaries, retrieved captions are more relevant and the decoder receives more accurate contextual information. This finding motivates the need for a framework that improves segment-to-event alignment.

## Method

### Overall Architecture

STaRC consists of three core components:
1. **Sliding Window Self-Attention (SWSA)** for refining frame features.
2. A **highlight detection module** for predicting supervised frame-level saliency scores.
3. A **unified saliency design**: Saliency-Guided Segmentation and Retrieval (SGSR) + Saliency Prompting (SaliP).

Input videos are processed through a frozen CLIP ViT-L/14 to extract spatial embeddings, which are encoded by a temporal Transformer before being passed to subsequent modules. Labels are derived directly from DVC event boundary annotations at no additional annotation cost.

### Key Designs

1. **Sliding Window Self-Attention (SWSA)**: Provides local context enhancement of frame features prior to saliency prediction. Multi-scale sliding windows $\{w_1, w_2, w_3\}$ of sizes 8, 32, and 64 aggregate neighborhood information without introducing learnable parameters. Overlapping outputs are averaged by coverage count, and refined features $X'$ are obtained via residual connection. **Design Motivation**: Local attention at multiple scales captures temporal dependencies at different granularities, while the parameter-free design avoids overfitting.

2. **Supervised Saliency Learning**: A highlight detection module combines local frame features $X'$ and a global video feature $X'_g$ (obtained via attention pooling) to compute frame-level saliency scores: $P_s(x'_n) = \frac{(x'_n \mathbf{W}_1^\top)(x'_{n_g} \mathbf{W}_2^\top)^\top}{\sqrt{D}}$. Training employs a listwise softmax loss: frames within event boundaries are labeled 1 and the rest 0, encouraging annotated frames to receive higher probabilities under softmax competition. This constitutes the paper's core innovation—converting DVC annotations into saliency supervision signals "for free."

3. **Saliency-Guided Segmentation and Retrieval (SGSR)**: Replaces heuristic segmentation with Optimal Transport (OT) clustering. $K$ learnable anchors are defined as semantic prototypes, with saliency injected in two ways: (a) Unbalanced OT is applied on the frame side, using $p_s$ as a soft constraint on the frame marginal distribution (via a KL divergence term $\gamma D_{\text{KL}}(\mathbf{T}^\top \mathbf{1}_K \| p_s)$), granting salient frames higher transport mass; (b) $p_s$ is incorporated as a bias term in the KOT cost matrix $C^k_{nj} = (1 - \text{cos}(x^s_n, a_j)) - \mu p_{s_n}$, prioritizing the assignment of salient frames. Segments are ranked by the product of an alignment score $\mathcal{S}_{\text{OT}}$ and a length regularization term $\mathcal{S}_{\text{len}}$, with top-$k$ segments selected for retrieval. Segment representations are constructed via saliency-weighted average pooling.

4. **Saliency Prompting (SaliP)**: Frame-level saliency scores are projected into prompt vectors $S$ via a learnable linear layer and concatenated with frame features $X'$, retrieval embeddings $R$, and transcript text $Y$ into a unified sequence: $T_{in} = [X'; S; R; Y]$. This enables the decoder to directly attend to semantically important frames during caption generation, rather than implicitly multiplying saliency by video features as in Sali4Vid.

### Loss & Training

- Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{saliency}}$
- During training, the decoder uses original frame features $X$ (for fine-grained text alignment); during inference, refined features $X'$ are used (for richer temporal context).
- Built on the Vid2Seq pretrained model (1.8M video-text pairs): first pretrained following the original configuration, then fine-tuned for 10 epochs.
- Learning rate 1e-5 with linear warm-up and cosine decay; single A6000 GPU, batch size 4.
- YouCook2: $\lambda=6.0$; ViTT: $\lambda=2.0$.

## Key Experimental Results

### Main Results

| Dataset | Metric | STaRC | Prev. SOTA (Sali4Vid) | Gain |
|--------|------|-------|---------------------|------|
| YouCook2 | CIDEr | 80.53 | 75.80 | +4.73 |
| YouCook2 | METEOR | 13.86 | 13.54 | +0.32 |
| YouCook2 | SODA_c | 10.73 | 10.28 | +0.45 |
| YouCook2 | BLEU_4 | 6.75 | 6.35 | +0.40 |
| YouCook2 | F1 (localization) | 34.34 | 33.61 | +0.73 |
| ViTT | CIDEr | 56.04 | 53.87 | +2.17 |
| ViTT | METEOR | 10.49 | 10.05 | +0.44 |

STaRC achieves state-of-the-art performance on most metrics on both YouCook2 and ViTT.

### Ablation Study

| Configuration | CIDEr | METEOR | Notes |
|------|-------|--------|------|
| Baseline (Vid2Seq) | 66.29 | 12.41 | No saliency components |
| + SGSR | 76.94 | 13.60 | Segmentation only, +10.65 |
| + SaliP | 78.74 | 13.75 | Prompt injection only, +12.45 |
| SGSR + SaliP (full) | 80.53 | 13.86 | Complementary, +14.24 |
| w/o SWSA | 75.82 | 13.23 | Feature refinement is beneficial |
| k-means segmentation | 75.63 | 13.34 | OT substantially outperforms k-means |
| Adaptive clustering | 78.19 | 13.69 | OT outperforms adaptive clustering |

### Key Findings

- SGSR and SaliP are each individually effective; their combination yields further improvement, validating the complementarity of the unified saliency design.
- The three-window configuration with sizes 8, 32, and 64 is optimal; excessively large windows degrade performance.
- Retrieval count $p=10$ is optimal; too few provides insufficient information, while too many introduces noise.
- Saliency prompt quality is critical: replacing prompts with Gaussian noise significantly degrades performance, and zero vectors also underperform compared to true saliency scores.

## Highlights & Insights

1. **"Free" supervision signals**: Existing event boundary annotations in DVC datasets are directly converted into frame-level saliency labels at no additional annotation cost—a highly elegant and practical idea.
2. **Unified signal design**: A single saliency score simultaneously serves retrieval (SGSR) and generation (SaliP), ensuring temporal consistency between segmentation and caption generation.
3. **Optimal Transport with saliency bias**: Saliency is doubly injected into the OT framework via frame marginal constraints and cost matrix biasing, providing a theoretically grounded formulation.
4. **Training-inference feature asymmetry**: Original features are used during training to ensure text alignment precision, while refined features are used during inference to provide richer contextual information—a thoughtful design choice.

## Limitations & Future Work

- The method relies on Vid2Seq's 1.8M-scale pretraining, making direct fair comparison with non-pretrained methods difficult (though results are reported separately by group).
- On ViTT, the F1 localization metric (44.34) falls below Sali4Vid (46.58) and HiCM2 (45.98), suggesting that segmentation may not be superior on short-label data.
- SWSA has no learnable parameters; ablation comparisons against learnable local attention alternatives are absent.
- Saliency labels are hard binary values and do not account for gradual transitions near event boundaries, which may introduce boundary noise.

## Related Work & Insights

- Sali4Vid first demonstrated that temporally salient frames benefit retrieval and caption generation, but its saliency is heuristic—STaRC's contribution lies in making this signal learnable and supervised.
- The OT clustering component is adapted from ASOT; STaRC's novelty lies in introducing saliency bias and unbalanced frame-side constraints.
- The highlight detection module draws on approaches from QD-DETR and EASeg, adapted to the DVC setting.
- The unified signal paradigm can generalize to other multimodal tasks requiring joint segmentation and generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of unifying saliency learning across both retrieval and generation channels is clear and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Component ablations, hyperparameter analyses, and qualitative comparisons are comprehensive; cross-dataset generalization testing is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated (the correlation analysis figure is persuasive), and the structure is compact.
- **Value**: ⭐⭐⭐⭐ A substantive advance in DVC research; the unified saliency paradigm has broader applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Weakly-Supervised Referring Video Object Segmentation through Text Supervision](wsrvos_weakly_supervised_rvos.md)
- [\[ICCV 2025\] Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection](../../ICCV2025/segmentation/beyond_single_images_retrieval_self-augmented_unsupervised_camouflaged_object_de.md)
- [\[NeurIPS 2025\] Exploring Structural Degradation in Dense Representations for Self-supervised Learning](../../NeurIPS2025/segmentation/exploring_structural_degradation_in_dense_representations_for_self-supervised_le.md)
- [\[NeurIPS 2025\] TabRAG: Improving Tabular Document Question Answering for Retrieval Augmented Generation via Structured Representations](../../NeurIPS2025/segmentation/tabrag_improving_tabular_document_question_answering_for_retrieval_augmented_gen.md)
- [\[ICCV 2025\] Joint Self-Supervised Video Alignment and Action Segmentation](../../ICCV2025/segmentation/joint_self-supervised_video_alignment_and_action_segmentation.md)

</div>

<!-- RELATED:END -->
