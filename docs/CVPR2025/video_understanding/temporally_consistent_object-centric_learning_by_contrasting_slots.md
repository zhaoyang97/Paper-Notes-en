---
title: >-
  [Paper Note] Temporally Consistent Object-Centric Learning by Contrasting Slots
description: >-
  [CVPR 2025][Video Understanding][Object-Centric Representation] Slot Contrast proposes a novel object-level temporal contrastive loss that contrasts slot representations across videos within a batch. This significantly improves the temporal consistency of video object-centric models, outperforming even weakly supervised methods using motion masks on object discovery tasks across synthetic and real-world datasets, while effectively supporting downstream unsupervised object dyn…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Object-Centric Representation"
  - "Temporal Consistency"
  - "Contrastive Learning"
  - "Slot Attention"
  - "Unsupervised Object Discovery"
date: 2026-05-08
content_hash: 5565a0e983d019f2
---

# Temporally Consistent Object-Centric Learning by Contrasting Slots

**Conference**: CVPR 2025  
**arXiv**: [2412.14295](https://arxiv.org/abs/2412.14295)  
**Code**: [https://slotcontrast.github.io/](https://slotcontrast.github.io/)  
**Area**: Video Understanding / Object-Centric Learning  
**Keywords**: Object-Centric Representation, Temporal Consistency, Contrastive Learning, Slot Attention, Unsupervised Object Discovery

## TL;DR

Slot Contrast proposes a novel object-level temporal contrastive loss that contrasts slot representations across videos within a batch. This significantly improves the temporal consistency of video object-centric models, outperforming even weakly supervised methods using motion masks on object discovery tasks across synthetic and real-world datasets, while effectively supporting downstream unsupervised object dynamics prediction.

## Background & Motivation

**Background**: Object-Centric Learning (OCL) is an unsupervised learning paradigm that decomposes high-dimensional visual data into independent object representations (commonly referred to as slots). Video-based OCL methods (e.g., SAVi, STEVE, VideoSAUR) establish object correspondences across frames by initializing the current frame's slots with those from the previous frame. Recent methods leverage self-supervised pre-trained features (e.g., DINOv2) and diverse training datasets (e.g., YouTube-VIS) to scale to real-world videos.

**Limitations of Prior Work**: Although existing methods can decompose scenes in short videos, maintaining consistent object representations over long temporal spans remains challenging. The root issue is that the training objective (typically feature reconstruction) does not explicitly encourage temporal consistency. When faced with object occlusion, reappearance, or complex interactions, slots tend to "jump" across different objects across frames (i.e., the same slot represents different objects in different frames).

**Key Challenge**: The feature reconstruction loss only requires slots to "cover" all content in the current frame, regardless of which slot covers which object. Consequently, it cannot guarantee slot-object correspondences remain stable over time. This struggle is further exacerbated by the random slot initialization strategy (sampling from a shared Gaussian distribution).

**Goal**: To design an explicit temporal consistency constraint that enables each slot to stably track the same object throughout the video sequence, without sacrificing and potentially even improving object discovery performance.

**Key Insight**: The authors observe that contrastive learning is naturally suited for defining "what should be similar and what should be different". If representations of the same slot in adjacent frames are defined as positive pairs and the remaining slots as negative pairs, the InfoNCE loss can drive the slots to learn temporally consistent representations.

**Core Idea**: A slot-slot contrastive loss is proposed to expand the negative contrastive set from a single video to all videos across the entire batch. Combined with learnable slot initialization, this forms the Slot Contrast framework, achieving strong temporal consistency and excellent object discovery performance.

## Method

### Overall Architecture

Slot Contrast is based on an encoder-decoder object-centric architecture. The encoder uses a frozen DINOv2 ViT to extract patch features, which are adapted by a learnable MLP and fed into a recurrent Slot Attention module for grouping. When processing each frame of the video, the model initializes the current frame's slots using the forecasted slots from the previous frame. Training is jointly optimized with two losses: a feature reconstruction loss (ensuring sufficient information in the slots) and a slot-slot contrastive loss (ensuring temporal consistency).

### Key Designs

1. **Batch Video Slot-Slot Contrastive Loss**:

    - **Function**: Explicitly forces each slot to remain consistent over time while maintaining distinctiveness from other slots (both within the same video and across different videos).
    - **Mechanism**: Given slot sets $S_{t-1}$ and $S_t$ of adjacent frames, the representation of the $i$-th slot in the $j$-th video at frame $t-1$ ($s_{t-1}^{i,j}$) and the slot at the same position in frame $t$ ($s_t^{i,j}$) are treated as a positive pair. All other slots $s_t^{k,b}$ ($k \neq i$ or $b \neq j$) in the batch serve as negatives. The InfoNCE loss is adopted: $\ell_{i,j}^{\text{ssc}} = -\log \frac{\exp(\text{sim}(s_{t-1}^{i,j}, s_t^{i,j}) / \tau)}{\sum_{b,k} \mathbb{1}_{[k,b \neq i,j]} \exp(\text{sim}(s_{t-1}^{i,j}, s_t^{k,b}) / \tau)}$.
    - **Design Motivation**: The issue with conducting contrastive learning solely within a single video (intra-video) is that the model can "cheat" by amplifying the differences between slot initializations rather than truly learning discriminative object features. Expanding the comparison to the entire batch scales up the size and diversity of the contrastive set. Since all videos in the same batch share the same initial slots $S_0$, the model cannot rely on initialization discrepancies to differentiate slots and is forced to learn distinctions based on visual content. Empirical validation shows that batch-level contrast provides an improvement of +14.5 FG-ARI on MOVi-C compared to intra-video contrast.

2. **Learned Initialization**:

    - **Function**: Provides a well-structured initial representation for slots, facilitating contrastive learning and object discovery.
    - **Mechanism**: Replaces the random slot initializations (sampling from a shared Gaussian distribution in vanilla Slot Attention) by learning a fixed set of initial slot vectors $S_0$ for the entire dataset. These slots naturally learn distinct initial queries during training, stably attending to different types of objects.
    - **Design Motivation**: Random initialization is detrimental to slot contrastive learning because slots sampled from the same distribution start with similar values, making it difficult to establish a structured slot space. Learnable initialization gives each slot a unique "preference", forming a synergistic effect when combined with the contrastive loss. In experiments, learnable initialization brings a +7.6 FG-ARI improvement on MOVi-E (from 75.3 to 82.9).

3. **Semantic Recurrent Slot Attention**:

    - **Function**: Performs temporal object grouping in the semantic feature space of DINOv2.
    - **Mechanism**: A frozen DINOv2 ViT extracts patch features $g_t$, which are adapted through a learnable MLP $g_\psi$ into $h_t = g_\psi(g_t)$, and then fed into a recurrent Slot Attention. The module consists of a grouping component $C_\theta$ (standard Slot Attention for updating slots) and a predictor $P_\omega$ (capturing spatio-temporal slot interactions) to output $S_t^c$ and $S_t^p$ respectively, where the grouping outputs are used for decoding and the prediction outputs are propagated to the next frame.
    - **Design Motivation**: Although DINOv2 features are semantically rich, they are mainly trained at the image level. Adapting them through a learnable MLP makes the features more suitable for temporal object grouping. The recurrent architecture allows slots to propagate information across frames.

### Loss & Training

The total loss is a weighted sum of the feature reconstruction loss and the contrastive loss: $\mathcal{L} = \sum_{t=1}^{T-1} \mathcal{L}_{\text{rec}}(h_t, \hat{h}_t) + \alpha \mathcal{L}_{\text{ssc}}(S_{t-1}, S_t)$. The decoder uses an MLP to reconstruct DINOv2 features from the slots. The DINOv2 encoder is frozen and not trained. The temperature parameter $\tau$ controls the sharpness of the contrastive loss. The MOVi datasets use a resolution of 336×336, and YouTube-VIS uses 518×518.

## Key Experimental Results

### Main Results

Temporally consistent object discovery (Video FG-ARI / mBO, calculated over the full video sequence):

| Method | MOVi-C FG-ARI↑ | MOVi-C mBO↑ | MOVi-E FG-ARI↑ | MOVi-E mBO↑ | YTVIS FG-ARI↑ | YTVIS mBO↑ |
|------|----------------|-------------|----------------|-------------|---------------|------------|
| SAVi | 22.2 | 13.6 | 42.8 | 16.0 | - | - |
| STEVE | 36.1 | 26.5 | 50.6 | 26.6 | 15.0 | 19.1 |
| VideoSAUR | 64.8 | 38.9 | 73.9 | 35.6 | 28.9 | 26.3 |
| VideoSAURv2 | - | - | 77.1 | 34.4 | 31.2 | 29.7 |
| **Slot Contrast** | **69.3** | 32.7 | **82.9** | 29.2 | **38.0** | **33.7** |

Single-frame object discovery (Image FG-ARI, MOVi-E):

| Method | Supervision Type | Image FG-ARI↑ |
|------|---------|---------------|
| DINOSAUR | Image only | 65.1 |
| DIOD | + Motion mask | 82.2 |
| SOLV | Video only | 80.8 |
| VideoSAUR | Video only | 78.4 |
| **Slot Contrast** | **Video only** | **84.8** |

### Ablation Study

Ablation of loss components (MOVi-C / MOVi-E / YouTube-VIS):

| Feature Recon. | Intra Contrast | Batch Contrast | MOVi-C FG-ARI | MOVi-E FG-ARI | YTVIS FG-ARI |
|---------|----------|----------|--------------|--------------|-------------|
| ✓ | | | 49.7 | 79.8 | 35.3 |
| ✓ | ✓ | | 54.8 | 78.7 | 35.7 |
| ✓ | | ✓ | **69.3** | **82.9** | **38.0** |

Ablation of initialization strategies:

| Configuration | MOVi-C FG-ARI | MOVi-E FG-ARI | YTVIS FG-ARI |
|------|--------------|--------------|-------------|
| Feature Recon. + Random Init. | 45.3 | 71.1 | 35.2 |
| Feature Recon. + Learned Init. | 49.4 | 79.8 | 35.3 |
| Slot Contrast + Random Init. | 62.9 | 75.3 | 36.1 |
| **Slot Contrast + Learned Init.** | **69.3** | **82.9** | **38.0** |

### Key Findings

- Batch-level contrast vastly outperforms intra-video contrast: FG-ARI on MOVi-C jumps from 54.8 to 69.3 because the larger negative set prevents trivial solutions that rely on initialization differences.
- The contrastive loss not only enhances temporal consistency but also significantly improves single-frame object discovery (84.8 Image FG-ARI, outperforming the weakly supervised method DIOD with motion masks at 82.2). This indicates that temporal constraints force the network to learn more discriminative object representations.
- In scenarios featuring complete occlusion, the mBO of Slot Contrast increases from a baseline of 16% to 21%, showing that the contrastive loss helps slots recover correspondences after objects reappear.
- There is a notable synergy between learnable initialization and the contrastive loss: while each yields improvements individually, their combined enhancement is substantially greater than the sum of their individual gains.
- The performance margin is most pronounced on the real-world YouTube-VIS dataset (+6.8 FG-ARI, +4.0 mBO vs. VideoSAURv2), demonstrating the value of the method in complex real-world scenes.

## Highlights & Insights

- **The 'Accidental' Discovery of Contrastive Learning Aiding Object Discovery**: The temporal consistency loss, as a byproduct, yields better single-frame object segmentation capabilities, surpassing methods utilizing additional motion supervision. This reveals that temporal signals provide a more effective inductive bias than motion segmentation masks.
- **Insights on Batch Contrast Preventing Degradation**: Since the entire batch shares the same slot initialization $S_0$, the model cannot cheat by differentiating the initializations, forcing it to truly learn object content. This serves as an excellent case study in contrastive learning design to avoid shortcut solutions.
- **Synergy of Slot Initialization and Contrastive Loss**: Learnable initialization provides a strong structural prior, while the contrastive loss further reinforces distinctiveness on top of this structure. The combination yields a system where the overall performance exceeds the sum of its parts.

## Limitations & Future Work

- The mBO metrics on synthetic datasets are lower than VideoSAUR (e.g., 32.7 vs. 38.9 on MOVi-C), indicating that contrastive constraints might prevent slot spatial masks from being sufficiently sharp, sometimes causing a single slot to partially cover multiple objects.
- The number of slots $K$ must be predefined and remains constant across all videos, failing to adaptively match the actual number of objects in the scene.
- YouTube-VIS videos are relatively short (up to 76 frames); the behavior of the method on genuinely long videos (hundreds of frames) has not yet been validated.
- The fully unsupervised setting means there is no guarantee of alignment between slots and semantic object categories; the same slot may represent different object classes across different videos.
- No significant improvement is observed in the downstream object dynamics prediction task on MOVi-E, suggesting that for scenes with camera motion, pure appearance consistency is insufficient to capture the complete dynamics.

## Related Work & Insights

- **vs. VideoSAUR**: VideoSAUR implicitly models temporal dynamics by predicting the temporal similarity of DINOv2 features; Slot Contrast more directly enforces temporal consistency via an explicit slot-level contrastive loss, offering stronger constraints by scaling to the batch level.
- **vs. SOLV**: SOLV achieves temporal consistency through agglomerative clustering and predicting intermediate frame features; Slot Contrast's contrastive framework is simpler and performs better (84.8 vs. 80.8 Image FG-ARI).
- **vs. SAVi/SAVi++**: The SAVi series utilizes recurrent processing but lacks explicit consistency constraints, while SAVi++ introduces depth as an extra supervision signal. Slot Contrast outperforms these methods without using any annotations.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of a slot-level contrastive loss is intuitive yet highly effective, and extending the negative set to the batch level elegantly addresses the representation degradation issue.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on three datasets, multiple downstream tasks, thorough ablation studies, and comparisons with weakly supervised methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, step-by-step mathematical derivation of the loss functions, and in-depth experimental analysis.
- Value: ⭐⭐⭐⭐ Provides a simple yet effective temporal consistency solution for video object-centric learning, while revealing important insights on how temporal contrastive loss promotes object discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning](../../CVPR2026/video_understanding/reconstruction-guided_slot_curriculum_addressing_object_over-fragmentation_in_vi.md)
- [\[ICLR 2026\] From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning](../../ICLR2026/video_understanding/from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv.md)
- [\[CVPR 2025\] H-MoRe: Learning Human-centric Motion Representation for Action Analysis](h-more_learning_human-centric_motion_representation_for_action_analysis.md)
- [\[ICCV 2025\] Factorized Learning for Temporally Grounded Video-Language Models](../../ICCV2025/video_understanding/factorized_learning_for_temporally_grounded_video-language_models.md)
- [\[CVPR 2025\] EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](edcflow_exploring_temporally_dense_difference_maps_for_event-based_optical_flow_.md)

</div>

<!-- RELATED:END -->
