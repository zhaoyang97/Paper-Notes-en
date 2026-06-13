---
title: >-
  [Paper Note] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos
description: >-
  [ICCV 2025][View selection] This paper proposes Switch-a-view, a model that learns view-switching patterns (ego/exo) from large-scale unlabeled in-the-wild instructional videos to enable automatic view selection in multi…
tags:
  - "ICCV 2025"
  - "View selection"
  - "instructional videos"
  - "multi-view video"
  - "weakly supervised learning"
  - "automatic cinematography"
date: 2026-05-08
content_hash: 21096fb256e80cc1
---

# Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos

**Conference**: ICCV 2025  
**arXiv**: [2412.18386](https://arxiv.org/abs/2412.18386)  
**Code**: [Project Page](https://vision.cs.utexas.edu/projects/switch_a_view/)  
**Area**: Other  
**Keywords**: View selection, instructional videos, multi-view video, weakly supervised learning, automatic cinematography

## TL;DR

This paper proposes Switch-a-view, a model that learns view-switching patterns (ego/exo) from large-scale unlabeled in-the-wild instructional videos to enable automatic view selection in multi-view instructional videos, without requiring explicit best-view annotations.

## Background & Motivation

Instructional videos (how-to videos) attract massive audiences on platforms such as YouTube and TikTok. A high-quality instructional video requires intelligently switching among multiple camera angles—for example, when demonstrating dog grooming, a wide shot first captures the overall posture before cutting to a close-up showing scissor manipulation details. Such view orchestration currently relies entirely on manual human editing, which is prohibitively costly.

The **core challenge** lies in obtaining labeled data: the internet hosts a large volume of human-edited instructional videos, but these videos only retain the final selected view sequence, while the discarded views ("footage left on the cutting room floor") are not preserved, making it impossible to construct direct supervision signals for the "best view."

**Key insight**: Although these videos lack explicit view annotations, human editors have implicitly encoded view preferences during production—selecting the corresponding ego or exo view given specific narration and visual content. It is therefore possible to learn view-switching patterns from these videos via pseudo-labels, and then transfer the learned knowledge to a limited-annotation multi-view setting.

## Method

### Overall Architecture

Switch-a-view consists of two stages:
1. **Pre-training stage**: Learns a view-switch detection task (pretext task) on large-scale unlabeled HowTo100M instructional videos.
2. **Fine-tuning stage**: Uses a small amount of labeled data to transfer the view-switch detector into a view selector.

### Key Designs

1. **View Pseudo-Labeler**:

    - Function: Automatically generates ego/exo view labels for unlabeled instructional videos.
    - Mechanism: A scene detector (PySceneDetect) first segments each video into continuous shots; a pre-trained ego-exo classifier then classifies frames within each shot; shot-level pseudo-labels are obtained by averaging frame-level probabilities.
    - Design Motivation: Direct frame-level classification is noisy, especially near scene boundaries; shot-level aggregation combined with the scene detector effectively reduces noise. The classifier is a view classification model trained on Charades-Ego.

2. **View-Switch Detector $D$**:

    - Function: Predicts the view type to be used in the next $\Delta$ seconds at time $t$.
    - Mechanism: Takes past video frames $F_{[:t]}$, narration text $N_{[:t]}$, view history $V_{[:t]}$, and the next narration segment $N'_{(t,t+\Delta]}$ as input, and predicts the next view through multimodal fusion:
    $D(F_{[:t]}, N_{[:t]}, V_{[:t]}, N'_{(t,t+\Delta]}) = V_{(t,t+\Delta]}$
    - Encoding:
        - **Frame encoding**: DINOv2 extracts visual features, augmented with view embeddings and temporal positional encodings: $f_i = \mathcal{E}^F(F_i) + \mathcal{E}^V(V_i^F) + \mathcal{E}^{\mathcal{T}}(\mathcal{T}_i^F)$
        - **Narration encoding**: Llama 2 encodes text, similarly augmented with view embeddings and temporal encodings: $n_i = \mathcal{E}^N(N_i) + \mathcal{E}^V(V_i^N) + \mathcal{E}^{\mathcal{T}}(\mathcal{T}_i^N)$
        - **Feature aggregation**: Modality embeddings are added to distinguish visual/textual tokens; all features are processed via an 8-layer Transformer encoder with self-attention; the [CLS] token output is fed into a 2-layer MLP classification head.
    - Design Motivation: Past frames provide fine-grained visual context; past narrations supply high-level semantic information about activity steps; the next narration directly implies the required view (e.g., "Now let's take a closer look at…" implies ego). Multimodal fusion outperforms any single signal.

3. **View Selector $S$**:

    - Function: Selects the best view for the current time segment in a multi-view setting.
    - Mechanism: Extends detector $D$ by additionally receiving synchronized ego and exo frame streams within the prediction interval, encoding them and appending them to the Transformer input sequence:
    $\ddot{V}_{(t,t+\Delta]} = \mathcal{H}(\mathcal{A}(f, n, n', f^G, f^X, c)[j_{\text{CLS}}])$
    - Design Motivation: View selection requires more information than view-switch detection—it must not only predict "whether to switch" but also compare the content quality of two candidate views. Observing both candidate frame streams enables more precise decisions.

### Loss & Training

- **Pre-training stage**: The detector is trained with cross-entropy loss using pseudo-labels: $\mathcal{L}^D = \mathcal{L}_{CE}(\hat{V}, \tilde{V})$
- **Fine-tuning stage**: The selector is initialized with detector parameters and fine-tuned on a small amount of Ego-Exo4D labeled data: $\mathcal{L}^S = \mathcal{L}_{CE}(\ddot{V}, V)$
- Training data: 3,416 hours of HowTo100M video for pre-training; approximately 3.5 hours of Ego-Exo4D video (6,634 samples) for fine-tuning.
- Configuration: 8 seconds of past frames, 32 seconds of past narration, prediction window $\Delta = 2$ seconds.

## Key Experimental Results

### Main Results

**View-switch detection** (Table 1):

| Model | HT100M Acc | HT100M AUC | EgoExo4D AUC | Notes |
|------|-----------|------------|-------------|------|
| Random | 52.0 | 52.0 | 49.3 | Random baseline |
| Retrieval-F | 53.4 | 53.4 | 52.6 | InternVideo2 retrieval |
| **Switch-a-view** | **59.4** | **63.8** | **56.4** | Proposed method |

**View selection** (Table 2):

| Model | Accuracy | AUC | AP | Notes |
|------|----------|-----|-----|------|
| LangView-bigData | 53.3 | 54.8 | 54.5 | Prev. SOTA, uses 98× more data |
| Ours w/o pretrain | 50.1 | 51.6 | 51.3 | Ablation without pre-training |
| **Switch-a-view** | **54.0** | **57.3** | **56.0** | Proposed method |

### Ablation Study

| Configuration | HT100M AUC | EgoExo4D AUC | Notes |
|------|-----------|-------------|------|
| Narration only $N$ | 54.4 | 48.7 | Past narration, single modality |
| Next narration only $N'$ | 57.8 | - | Next narration more informative than past |
| Frames only $F$ | - | - | Frames provide fine-grained information |
| **Full model** | **63.8** | **56.4** | Multimodal fusion is optimal |

### Key Findings

1. The next narration segment ($N'$) is more important for prediction than past narrations, as it directly aligns with the prediction time window.
2. Zero-shot transfer to Ego-Exo4D is already effective, indicating that view preferences learned from in-the-wild videos generalize well.
3. Training directly on limited annotations without pre-training performs poorly (51.6 vs. 57.3 AUC), confirming that view-switch pre-training is the core contribution.
4. Human annotators show substantial agreement on the best view (Cohen's kappa 0.65–0.70).

## Highlights & Insights

- **Clever signal source**: Exploits large volumes of already-edited instructional videos from the internet as weak supervision, avoiding expensive annotation costs.
- **Principled task decomposition**: Learning when to switch (pretext task) before learning which view to switch to reduces the difficulty of directly learning view selection.
- **Comprehensive multimodal design**: Visual frames + text narrations + view history + temporal encodings, each contributing distinct information.
- The annotator study confirms that a consistent notion of "best view" exists, validating the task formulation.

## Limitations & Future Work

- The current formulation considers only binary ego/exo classification; practical scenarios may involve multiple exo cameras requiring further selection.
- Pseudo-label quality depends directly on the pre-trained ego-exo classifier used for labeling.
- The prediction window is fixed at 2 seconds, which does not adapt to the natural rhythm of different activities.
- Absolute performance on Ego-Exo4D remains limited (AUC 57.3%), indicating that the task itself is highly challenging.

## Related Work & Insights

- The complementarity with LangView (CVPR 2024) is noteworthy: LangView uses narration text for weakly supervised pre-training, while this paper leverages video editing patterns; combining the two could yield further improvements.
- The view selection paradigm is extensible to other domains, such as sports broadcasting, online education, and VR content production.
- Whether the view preference patterns learned from HowTo100M transfer to other domains (e.g., cooking → crafts) warrants further exploration.

## Rating

- Novelty: ⭐⭐⭐⭐ View-switch detection as a pre-training task is novel; the pseudo-labeling strategy is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablations and rigorous human annotation study, though absolute performance is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is clear and motivation is thoroughly articulated.
- Value: ⭐⭐⭐⭐ Practically meaningful for automatic video editing, though absolute performance still requires improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images (MV-AR)](autoregressively_generating_multiview_consistent_images.md)
- [\[ICCV 2025\] Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior](recover_biological_structure_from_sparse-view_diffraction_images_with_neural_vol.md)
- [\[NeurIPS 2025\] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](../../NeurIPS2025/others/incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)
- [\[NeurIPS 2025\] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?](../../NeurIPS2025/others/are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)

</div>

<!-- RELATED:END -->
