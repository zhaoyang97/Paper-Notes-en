---
title: >-
  [Paper Note] Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection
description: >-
  [CVPR 2026][Object Detection][Cross-domain few-shot detection] This paper proposes LMP, a dual-branch framework built upon GroundingDINO that introduces a visual prototype branch (comprising positive class prototypes and…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "Cross-domain few-shot detection"
  - "visual prototypes"
  - "multi-modal"
  - "GroundingDINO"
  - "hard negatives"
date: 2026-05-08
content_hash: 1e03078e92f8101c
---

# Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection

**Conference**: CVPR 2026
**arXiv**: [2602.18811](https://arxiv.org/abs/2602.18811)  
**Code**: N/A  
**Area**: Object Detection
**Keywords**: Cross-domain few-shot detection, visual prototypes, multi-modal, GroundingDINO, hard negatives

## TL;DR
This paper proposes LMP, a dual-branch framework built upon GroundingDINO that introduces a visual prototype branch (comprising positive class prototypes and hard negative prototypes) jointly trained and integrated with the text branch at inference, achieving state-of-the-art performance on cross-domain few-shot object detection.

## Background & Motivation
Cross-domain few-shot object detection (CD-FSOD) requires detecting novel categories in a target domain with only a handful of annotated examples. Although VLM-based open-vocabulary detectors (e.g., GroundingDINO) exhibit strong transfer capability, they rely exclusively on text prompts and are subject to two systematic failure modes:

**Semantic–appearance mismatch**: Text prototypes capture category semantics but overlook target-domain cues (e.g., style, texture, illumination), resulting in weak localization.

**Confusable context**: In few-shot scenarios, visually similar background regions or nearby objects dominate training, producing a large number of false positives.

Simply using support images as visual prompts offers little benefit—unstructured features conflate class evidence with incidental context, and hard negatives are not explicitly modeled. Structured visual prototypes are therefore necessary to provide domain-adaptive information.

## Method

### Overall Architecture
LMP builds a dual-branch architecture on top of GroundingDINO:
- **Text-guided branch**: Retains the original GroundingDINO text branch to preserve open-vocabulary semantic understanding.
- **Visual-guided branch**: Injects domain-specific visual prototypes, comprising visual prototype construction, prototype refinement (a feature enhancer), visual-guided query selection, and a visual decoder.

Both branches are jointly trained, and at inference their predictions are fused via ensemble to combine semantic abstraction with domain-adaptive detail.

### Key Designs

1. **Class-level Visual Prototypes**: Given $C$-way $K$-shot support images, features are extracted from each annotated instance via RoIAlign followed by global average pooling (GAP). Instance features of the same class are averaged to obtain a class prototype $\mathbf{p}_c \in \mathbb{R}^{D_I}$, which are then stacked into $\mathbf{P}_{\mathrm{cls}} \in \mathbb{R}^{C \times D_I}$. These prototypes encode representative visual characteristics of each category in the target domain.

2. **Hard Negative Prototypes**: For each ground-truth box $b_j$ in the query image, $N$ perturbed boxes are sampled via random jittering. Boxes with $\mathrm{IoU} \in [0.1, 0.5]$ are retained, and negative prototypes $\mathbf{p}_{\mathrm{neg},j}^{(n)}$ are extracted via RoIAlign + GAP. These prototypes capture domain-specific distractors and visually confusable background regions, which are the primary source of false positives in CD-FSOD. Positive and negative prototypes are concatenated as $\mathbf{V} = [\mathbf{P}_{\mathrm{cls}}; \mathbf{P}_{\mathrm{neg}}] \in \mathbb{R}^{N_V \times D_I}$.

3. **Visual Prototype Refinement and Decoding**:

    - **Visual Feature Enhancer**: Six layers of self-attention and cross-attention enable bidirectional interaction between image tokens $\mathbf{X}_I$ and visual prototypes $\mathbf{V}$, yielding refined prototypes $\mathbf{V}'$ and updated image tokens $\mathbf{X}'_I$.
    - **Visual-Guided Query Selection**: A cosine similarity matrix between image tokens and prototypes is computed, and the Top-900 entries are selected to initialize queries.
    - **Visual Decoder**: Mirrors the cross-modal decoder of the text branch; iterative refinement produces category logits and bounding box regression. Classification is based on prototype alignment scores (cosine similarity).

### Loss & Training
Both branches employ focal classification loss, $L_1$ box regression loss, and GIoU loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{text}} + \alpha \mathcal{L}_{\text{visual}}$$

where $\alpha = 1.0$. A two-stage training procedure is adopted: the visual branch is trained alone in the first stage, followed by joint training in the second stage. One-to-one supervision is applied via Hungarian matching. Hard negative prototypes are naturally incorporated through the attention mechanism without requiring additional contrastive losses.

## Key Experimental Results

### Main Results

| Dataset | Shot | LMP (Ours) | Domain-RAG (Prev. SOTA) | Gain |
|--------|------|-----------|----------------------|------|
| Average (6 domains) | 1-shot | **34.3** | 33.6 | +0.7 |
| Average (6 domains) | 5-shot | **44.0** | 42.7 | +1.3 |
| Average (6 domains) | 10-shot | **46.6** | 45.4 | +1.2 |
| ArTaxOr | 1-shot | **58.5** | 57.2 | +1.3 |
| ArTaxOr | 5-shot | **75.0** | 70.0 | +5.0 |

Compared to the GroundingDINO baseline, LMP improves mAP by 8.0 / 3.6 / 2.1 under the 1 / 5 / 10-shot settings, respectively.

### Ablation Study

| Configuration | Avg. mAP (5-shot) | Note |
|------|-------------------|------|
| Text prototypes only (GD baseline) | 40.4 | Baseline |
| + Class-level visual prototypes | 42.8 | Consistent gains across all domains |
| + Hard negative prototypes | **44.0** | Best performance across all domains |

Hyperparameter analysis: $N=3$ hard negative prototypes yields the best results ($N=5$ causes a slight drop); $\alpha=1.0$ is optimal across all shot settings.

### Key Findings
- The largest gains are observed on datasets with coarse-grained labels (ArTaxOr, with taxonomic names such as "Coleoptera"), where text prompts provide minimal visual guidance.
- Improvement is greatest at 1-shot (+8.0 mAP), confirming that multi-modal prototypes are most effective under extreme data scarcity.
- t-SNE visualizations show that hard negatives cluster near category decision boundaries, validating the role of negative prototypes.
- Qualitative analysis demonstrates reduced background false positives on Clipart, more accurate industrial texture discrimination on NEU-DET, and improved recall for small fish in DeepFish.

## Highlights & Insights
1. **Elegant dual-branch design**: The text branch preserves open-vocabulary capability while the visual branch provides domain adaptation; the two are complementary rather than redundant.
2. **Intuitive hard negative prototype design**: Negative samples are generated by jittering GT boxes, directly modeling the most common sources of false positives in detection (background interference and partially overlapping objects).
3. **Visual branch initialized from text branch weights**: Leveraging existing knowledge accelerates convergence and avoids the instability of training from scratch.
4. **No additional contrastive loss required**: Hard negative prototypes are naturally integrated through the attention mechanism, making the approach simple and efficient.
5. **Trainable on a single RTX 3090**: The computational footprint is reasonable, facilitating reproducibility.

## Limitations & Future Work
- Dual-branch inference doubles computational overhead; distillation into a single branch is a natural direction for deployment (also noted by the authors).
- The approach is sensitive to atypical support samples; extreme outliers in support images degrade prototype quality.
- Hard negative prototypes only consider regions near GT boxes; extension to ring/context regions and proposal-level distractors is a promising direction.
- Adaptive prototype construction (dynamically selecting prototype count and weights) could replace the current fixed strategy.
- The ensemble weighting between text and visual branches is currently fixed at 1:1; learning adaptive fusion weights warrants exploration.

## Related Work & Insights
- Unlike MQ-Det (which fuses visual exemplars into the text encoder via cross-attention) and VisTex-OVLM (which projects visual exemplars into textual tokens), LMP maintains a structurally independent visual branch for clarity.
- In the CD-FSOD literature, CD-ViTO established the first formal benchmark, ETS employs grid-search sub-domains, and Domain-RAG adopts retrieval-augmented generation. LMP is the first dual-branch multi-modal prototype approach in this setting.
- The hard negative mining strategy is generalizable to other few-shot tasks such as segmentation, instance retrieval, and Re-ID.
- The dual-branch ensemble paradigm suggests a broader principle: analogous approaches could inject additional modalities (e.g., depth maps, heat maps) into detection pipelines.

## Rating
- Novelty: ⭐⭐⭐⭐ Dual-branch design with hard negative prototypes is novel; the problem formulation is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ Six cross-domain datasets × three shot settings with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Logically structured with well-presented figures.
- Value: ⭐⭐⭐⭐ Represents a meaningful advance for the CD-FSOD community.

<!-- END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection](remedying_target-domain_astigmatism_for_cross-domain_few-shot_object_detection.md)
- [\[CVPR 2026\] A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps](a_closer_look_at_cross-domain_few-shot_object_detection_fine-tuning_matters_and_.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[CVPR 2026\] Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](evaluating_few-shot_pill_recognition_under_visual_domain_shift.md)
- [\[CVPR 2026\] DA-Mamba: Learning Domain-Aware State Space Model for Global-Local Alignment in Domain Adaptive Object Detection](da-mamba_learning_domain-aware_state_space_model_for_global-local_alignment_in_d.md)

</div>

<!-- RELATED:END -->
