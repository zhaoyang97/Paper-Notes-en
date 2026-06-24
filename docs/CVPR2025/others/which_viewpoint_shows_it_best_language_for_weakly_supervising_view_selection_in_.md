---
title: >-
  [Paper Note] Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos
description: >-
  [CVPR 2025][View selection] This paper proposes LangView, which utilizes viewpoint-agnostic textual narrations as weak supervision signals. By comparing the alignment between the predicted captions of each viewpoint and the ground-truth narration, it generates pseudo-labels for the best viewpoint, enabling automatic view selection in multi-view instructional videos without manual annotation.
tags:
  - "CVPR 2025"
  - "View selection"
  - "Weak supervision"
  - "Multi-view instructional videos"
  - "Language guidance"
  - "Pseudo-labels"
date: 2026-05-08
content_hash: c0d19d22d127f8a6
---

# Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos

**Conference**: CVPR 2025  
**arXiv**: [2411.08753](https://arxiv.org/abs/2411.08753)  
**Code**: [https://vision.cs.utexas.edu/projects/which-view-shows-it-best](https://vision.cs.utexas.edu/projects/which-view-shows-it-best)  
**Area**: Others  
**Keywords**: View selection, Weak supervision, Multi-view instructional videos, Language guidance, Pseudo-labels

## TL;DR
This paper proposes LangView, which utilizes viewpoint-agnostic textual narrations as weak supervision signals. By comparing the alignment between the predicted captions of each viewpoint and the ground-truth narration, it generates pseudo-labels for the best viewpoint, enabling automatic view selection in multi-view instructional videos without manual annotation.

## Background & Motivation

**Background**: Multi-view instructional videos (e.g., tutorial videos recorded from multiple camera positions) require selecting the most informative viewpoint at each timestamp to display to the audience. This has important applications in automated cinematography and instructional video editing.

**Limitations of Prior Work**: Existing automatic view selection methods either rely on heuristic rules (which lack generalization) or require expensive "best-view" human annotations for training (making them hard to scale). The lack of annotations limits the application of learning-based methods to this task.

**Key Challenge**: A large amount of best-view annotations is required to train the selection model. However, obtaining annotations requires humans to evaluate the informativeness of each viewpoint frame-by-frame, which is highly subjective and expensive.

**Goal**: Train a view selection model without any manual best-view annotations.

**Key Insight**: Multi-view instructional videos are often accompanied by viewpoint-agnostic narrations (e.g., "remove the rear wheel with both hands"). Key assumption: If a certain viewpoint can more accurately predict this narration, it indicates that it showcases the activity better, and is therefore a superior viewpoint.

**Core Idea**: Use the caption prediction accuracy of each viewpoint as a proxy metric for viewpoint quality, automatically generating best-view pseudo-labels to train the view selection model without any manual viewpoint annotation.

## Method

### Overall Architecture
Two stages: (1) **Pseudo-label Generation**: For each viewpoint, a pre-trained video captioner is used to generate a caption, which is compared with the ground-truth narration to calculate similarity. The best-matching viewpoint is labeled as the pseudo-best viewpoint; (2) **View Selector Training**: The view selection model is trained using the pseudo-labels, and an auxiliary task (camera pose prediction) is introduced to enhance viewpoint sensitivity. During inference, only the multi-view video input is required, without text or camera poses.

### Key Designs

1. **Pseudo-label Generation based on Caption Accuracy**:

    - **Function**: Automatically generate best-view annotations for multi-view videos.
    - **Mechanism**: For $N$ viewpoints, a pre-trained captioner is used to predict the narration. The similarity (e.g., ROUGE/BERTScore) between each prediction and the viewpoint-agnostic ground-truth narration is computed. The viewpoint with the highest score is labeled as the positive sample.
    - **Design Motivation**: Viewpoint-agnostic narrations describe the complete information of the activity (e.g., all involved objects and actions). The viewpoint that best predicts the entire information is naturally the most informative viewpoint.

2. **Auxiliary Camera Pose Prediction Task**:

    - **Function**: Enhance the model's sensitivity to differences between viewpoints.
    - **Mechanism**: Train the selector to simultaneously predict the relative camera poses between different viewpoints (e.g., spatial relations between first-person and third-person perspectives).
    - **Design Motivation**: Pure view selection might degenerate into focusing solely on the content while ignoring viewpoint differences; pose prediction forces the model to understand the geometric implications of the viewpoints.

3. **Training-Inference Decoupling**:

    - **Function**: Eliminate the dependency on text and poses during inference.
    - **Mechanism**: During training, textual narrations are used to generate pseudo-labels (computed offline), and camera poses are used for the auxiliary task. During inference, the selector only takes multi-view videos as input and directly outputs the best viewpoint for each moment.
    - **Design Motivation**: In practical applications, the end-user only provides the video and should not be required to provide extra inputs.

### Loss & Training
Cross-entropy loss (view selection classification) + auxiliary pose prediction loss.

## Key Experimental Results

### Main Results
On the Ego-Exo4D and LEMMA multi-view instructional video datasets:

| Method | Automatic Metric↑ | Human Evaluation↑ | Supervision Type |
|------|----------|----------|---------|
| Heuristic methods | Baseline | Baseline | Rules |
| Fully-supervised SOTA | Better | Better | Manual annotations |
| LangView (Ours) | Best | Best | Weak supervision (narrations only) |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| Random caption matching | Poor | Verifies that captions indeed carry viewpoint information |
| Removing auxiliary pose prediction | Degradation | Pose awareness is crucial for selection |
| Different captioner models | All effective | The method does not overly depend on a specific model |

### Key Findings
- The weakly supervised method outperforms the fully supervised baseline — demonstrating that language signals are richer than a small amount of human annotations.
- The auxiliary pose prediction significantly improves performance, confirming the importance of understanding viewpoint geometry.
- Human evaluation aligns with automatic metrics, proving the method indeed selects the viewpoints preferred by humans.

## Highlights & Insights
- **Language as a Visual Supervision Signal**: Using "textual description capability" as a proxy for viewpoint informativeness is a highly clever weak supervision concept, which can be transferred to other tasks requiring measurement of visual informativeness.
- **No Annotation Beats Annotation**: The weakly supervised method outperforms the fully supervised baseline, showing that large-scale weak signals can be more effective than a small amount of strong signals.

## Limitations & Future Work
- It relies on the quality of the pre-trained captioner; bias in the captioner regarding specific activities may propagate to the selector.
- It currently assumes there is a single best viewpoint, but in some scenarios, multi-view fusion might be needed.
- It is only validated on instructional videos; other domains like sports and meetings are unexplored.

## Related Work & Insights
- **vs Fully Supervised Methods**: These require expensive manual annotations, whereas LangView does not.
- **vs Temporal Summarization Methods**: Video summarization selects keyframes along the temporal axis, whereas this work selects the best camera along the viewpoint axis.
- The strategy of using language as a proxy for visual signal quality has broad application prospects.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The core idea of using caption accuracy as a proxy for viewpoint quality is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on two datasets + human evaluation + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent motivation and clear presentation of ideas.
- Value: ⭐⭐⭐⭐ Highly practical for multi-view video editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](../../ICCV2025/others/switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[CVPR 2025\] Three-View Focal Length Recovery From Homographies](three-view_focal_length_recovery_from_homographies.md)
- [\[ICCV 2025\] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](../../ICCV2025/others/intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [\[CVPR 2025\] Feature Selection for Latent Factor Models](feature_selection_for_latent_factor_models.md)
- [\[ECCV 2024\] Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization](../../ECCV2024/others/mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)

</div>

<!-- RELATED:END -->
