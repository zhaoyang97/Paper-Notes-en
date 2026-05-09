---
title: >-
  [Paper Note] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking
description: >-
  [ICCV 2025][Video Understanding][Open-Vocabulary MOT] This paper proposes TRACT, a method that leverages trajectory-level information to enhance open-vocabulary multi-object tracking (OV-MOT). It improves association via Trajectory Consistency Reinforcement (TCR) and improves classification via Trajectory Feature Aggregation (TFA) and Trajectory Semantic Enrichment (TSE). TRACT achieves significant performance gains on the OV-TAO benchmark, particularly in classification accuracy.
tags:
  - ICCV 2025
  - Video Understanding
  - Open-Vocabulary MOT
  - Trajectory Information
  - CLIP
  - Association
  - Classification
date: 2026-05-08
content_hash: 3a47b5b08fd48dc7
---

# Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking

**Conference**: ICCV 2025
**arXiv**: [2503.08145](https://arxiv.org/abs/2503.08145)
**Code**: Unavailable (paper states code will be released)
**Area**: Video Understanding / Multi-Object Tracking
**Keywords**: Open-Vocabulary MOT, Trajectory Information, CLIP, Association, Classification

## TL;DR
This paper proposes TRACT, a method that leverages trajectory-level information to enhance open-vocabulary multi-object tracking (OV-MOT). It improves association via Trajectory Consistency Reinforcement (TCR) and improves classification via Trajectory Feature Aggregation (TFA) and Trajectory Semantic Enrichment (TSE). TRACT achieves significant performance gains on the OV-TAO benchmark, particularly in classification accuracy.

## Background & Motivation
Open-vocabulary multi-object tracking (OV-MOT) aims to track arbitrary object categories unseen during training. Existing OV-MOT methods (e.g., OVTrack, MASA) focus primarily on instance-level detection and association, neglecting the exploitation of trajectory-level information. However, trajectory information plays a uniquely critical role in video tracking:

**Association perspective**: Instability of open-vocabulary detectors leads to inaccurate or missed detections in certain frames; trajectory information can help recover broken matches and reduce identity switches.

**Classification perspective**: Frequent blur and occlusion in videos cause misclassification; trajectory information enables category inference by aggregating evidence across multiple frames and viewpoints.

The root cause lies in the fact that existing methods operate solely at the instance level, while trajectories—as video-specific contextual information—are severely underutilized. Starting from the dual value of trajectories, this paper simultaneously enhances both the association and classification stages.

## Method

### Overall Architecture
TRACT adopts a two-stage tracker architecture following the tracking-by-detection paradigm. The first stage employs a replaceable open-vocabulary detector to generate detections $\mathcal{R} = \{\mathbf{b}_i, \mathbf{c}_i, \mathbf{f}_i\}_{i=1}^N$. The second stage consists of two steps:
- **Trajectory-enhanced association**: Employs the TCR strategy, maintaining a feature bank and a category bank to enforce trajectory consistency.
- **Trajectory-assisted classification**: Employs the TraCLIP module, exploiting trajectory information from both visual and linguistic perspectives via TFA and TSE.

### Key Designs

1. **Trajectory Consistency Reinforcement (TCR)**:

    - Function: Leverages trajectory history during the association stage to reinforce identity and category consistency.
    - Mechanism:
        - **Identity consistency**: For each active trajectory, a trajectory memory $\mathbf{f}$ and a feature bank $\bar{\mathbf{f}} = \{f_{i-j}\}_{j=1}^{n_{\text{bank}}}$ are maintained. The trajectory memory is updated via exponential moving average: $\mathbf{f}_i = \alpha \times f_i + (1-\alpha) \times \mathbf{f}_{i-1}$. Similarity computation jointly considers trajectory memory and the feature bank: $\mathtt{S}(\mathbf{t}, r) = \alpha \cdot \Psi(f_i, \mathbf{f}) + (1-\alpha) \cdot \frac{1}{n_{\text{bank}}} \sum_{j=1}^{n_{\text{bank}}} \Psi(f_i, f_{i-j})$
        - **Category consistency**: A category bank $\bar{\mathbf{c}} = \{c_{i-j}\}_{j=1}^{n_{\text{clip}}}$ stores historical classification predictions. Detections are processed in a confidence-stratified manner: high-confidence detections adopt the current classification directly; medium-confidence detections fuse current and historical predictions via voting; low-confidence detections rely solely on historical voting.
    - Design Motivation: Open-vocabulary detection is inherently unstable, and single-frame detection is prone to errors. Maintaining cross-frame feature and category memory smooths out noise.

2. **Trajectory Feature Aggregation (TFA)**:

    - Function: Aggregates visual features from multiple frames within a trajectory into a unified trajectory-level representation.
    - Mechanism: $n_{\text{clip}}$ frames are sampled from the trajectory based on detection confidence. The CLIP visual encoder extracts per-frame 2D features $\dot{\mathbf{f}} \in \mathbb{R}^{n \times d}$, which are then enhanced via self-attention and MLP: $\ddot{\mathbf{f}} = \dot{\mathbf{f}} + \mathtt{SA}(\mathtt{LN}(\dot{\mathbf{f}}))$, $\tilde{\mathbf{f}} = \ddot{\mathbf{f}} + \mathtt{MLP}(\mathtt{LN}(\ddot{\mathbf{f}}))$. Global average pooling finally produces the trajectory feature $\mathbf{f}^{traj}$.
    - Design Motivation: Targets may be in different occlusion or blur states across frames; aggregating multi-frame features yields a more complete visual representation.

3. **Trajectory Semantic Enrichment (TSE)**:

    - Function: Leverages an LLM to expand simple category names into rich descriptions containing visual attributes.
    - Mechanism: ChatGPT is used to generate attribute descriptions per category (e.g., "Provide a brief description of the {category} focusing on two to three visual attributes"), concatenated as $\mathcal{A} = \mathtt{Concat}(\mathcal{V}, \Phi(\mathcal{V}))$. The CLIP text encoder then separately extracts vanilla category features $\mathcal{F}^{cate}$ and attribute-augmented features $\mathcal{F}^{attr}$.
    - Design Motivation: Trajectories provide target information from diverse viewpoints and lighting conditions; relying solely on category names fails to exploit these rich cues.

### Ternary Classification Selection
For each trajectory, three classification results are computed: classification based on vanilla category names $\mathbf{v}_{cate}$, based on attribute descriptions $\mathbf{v}_{attr}$, and based on detector prediction voting $\mathbf{v}_{det}$. The result with the highest similarity score is selected as the final category.

### Loss & Training
The TCR module requires no training. TraCLIP is initialized from CLIP ViT-L/14 with the visual and language encoders frozen; only the self-attention and MLP modules are trained. Training data include the LVIS, YouTube-VIS, and TAO training sets (known categories only); pseudo-trajectories are generated from LVIS images via data augmentation.

## Key Experimental Results

### Main Results (OV-TAO Validation Set, YOLO-World Detector)

| Method | Base TETA↑ | Base ClsA↑ | Novel TETA↑ | Novel ClsA↑ |
|--------|-----------|-----------|------------|------------|
| MASA | 38.2 | 18.6 | 32.2 | 4.4 |
| **TRACT** | **39.4** | **22.6** | **33.7** | **5.3** |
| DeepSORT | 27.3 | 17.9 | 21.5 | 3.8 |
| OC-SORT | 31.2 | 16.9 | 24.4 | 3.7 |

### Ablation Study

| Configuration | TETA | LocA | AssA | ClsA | Note |
|---------------|------|------|------|------|------|
| Baseline (no strategy) | 37.5 | 55.1 | 40.1 | 16.9 | MASA baseline |
| +TCR | 37.6 | 55.0 | 40.6 | 17.3 | Association consistency improved |
| +TCR+TFA | 38.5 | 54.9 | 40.5 | 19.9 | Trajectory features significantly boost classification |
| +TCR+TFA+TSE | 38.6 | 54.9 | 40.6 | 20.3 | Attribute descriptions bring further gains |

### Key Findings
- ClsA shows the most substantial improvement (+3.4%), validating the central value of trajectory information for classification.
- TSE yields larger classification gains on Novel categories (10.9→13.3), indicating that attribute descriptions are especially beneficial for unseen classes.
- A feature bank length of $n_{\text{bank}}=15$ is optimal, with negligible impact on speed (1.52→1.59 s/seq).
- A trajectory sampling length of $n_{\text{clip}}=5$ is sufficient; longer sequences show no meaningful gain but significantly reduce speed.

## Highlights & Insights
- Elevates trajectory information utilization from an auxiliary tool to a core design element in OV-MOT.
- TraCLIP serves as a plug-and-play trajectory classification module compatible with different detectors and association methods.
- The ternary classification selection mechanism elegantly combines detector predictions and trajectory-level CLIP matching.
- The paper provides an in-depth discussion of current challenges in OV-MOT data and evaluation protocols (excessive detection density, incomplete annotations, etc.).

## Limitations & Future Work
- Trajectory information is not leveraged to improve localization (LocA remains unchanged); the authors discuss preliminary attempts but report limited effectiveness.
- TSE depends on ChatGPT for attribute description generation, incurring additional offline preparation cost.
- TraCLIP training data covers only known categories, which may limit generalization to entirely unseen classes.
- The OV-TAO dataset itself suffers from incomplete annotations and excessive detection density, and the evaluation protocol may not fully reflect real-world scenarios.

## Related Work & Insights
- **vs. OVTrack**: OVTrack is a pioneering OV-MOT work but operates only at the instance level; TRACT substantially surpasses it on ClsA through trajectory-level information.
- **vs. MASA**: MASA employs SAM for instance matching but ignores trajectory context; TRACT consistently improves upon it via trajectory strategies.
- **vs. SLAck**: SLAck unifies early-stage association but does not exploit the classification potential of trajectories; TRACT's TraCLIP provides a complementary classification perspective.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of trajectory-level information utilization is clear and effective; the plug-and-play design of TraCLIP is practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-detector, multi-dataset grouped comparisons with ablations comprehensively covering all hyperparameters.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clear problem formulation and methodological exposition.
- Value: ⭐⭐⭐⭐ Introduces a valuable trajectory perspective to OV-MOT and identifies fundamental issues in data and evaluation.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition](trokens_semantic-aware_relational_trajectory_tokens_for_few-shot_action_recognit.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)
- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)
- [\[CVPR 2026\] TrajTok: Learning Trajectory Tokens Enhances Video Understanding](../../CVPR2026/video_understanding/trajtok_trajectory_token_video_understanding.md)

<!-- RELATED:END -->
