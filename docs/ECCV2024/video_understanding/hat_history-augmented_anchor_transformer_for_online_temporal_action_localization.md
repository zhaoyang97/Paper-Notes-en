---
title: >-
  [Paper Note] HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization
description: >-
  [ECCV 2024][Video Understanding][Online Temporal Action Localization] Proposes HAT—the first anchor-based Transformer framework that introduces long-term historical context in Online Temporal Action Localization (OnTAL). Through action anticipation-guided history compression and future-driven history refinement, it significantly outperforms OAT on procedural egocentric datasets (EGTEA/EK100) and achieves comparable or superior performance on standard datasets (THUMOS/MUSES).
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Online Temporal Action Localization"
  - "Transformer"
  - "Historical Context"
  - "Egocentric Vision"
  - "Adaptive Focal Loss"
date: 2026-05-08
content_hash: cd54347424d13e40
---

# HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization

**Conference**: ECCV 2024  
**arXiv**: [2408.06437](https://arxiv.org/abs/2408.06437)  
**Code**: [https://github.com/sakibreza/ECCV24-HAT/](https://github.com/sakibreza/ECCV24-HAT/)  
**Area**: Video Understanding  
**Keywords**: Online Temporal Action Localization, Transformer, Historical Context, Egocentric Vision, Adaptive Focal Loss

## TL;DR

Proposes HAT—the first anchor-based Transformer framework that introduces long-term historical context in Online Temporal Action Localization (OnTAL). Through action anticipation-guided history compression and future-driven history refinement, it significantly outperforms OAT on procedural egocentric datasets (EGTEA/EK100) and achieves comparable or superior performance on standard datasets (THUMOS/MUSES).

## Background & Motivation

Online Temporal Action Localization (OnTAL) is a relatively new video understanding task that requires detecting and classifying discrete action instances in real-time streaming videos (rather than frame-by-frame classification), where the generated proposals cannot be modified post-hoc.

Existing OnTAL methods (such as OAT) only utilize **short-term sliding window features** to generate anchor-level action proposals, **ignoring the value of long-term historical context**. Although methods like LSTR and GateHUB in the Online Action Detection (OAD) field have explored historical information, there are key differences:

**OAD and OnTAL operate at different task levels**: OAD is frame-level prediction, while OnTAL is instance-level prediction. Directly applying OAD's history processing methods is sub-optimal.

**Special needs of procedural action scenarios**: In egocentric videos such as kitchen operations, the current action strongly depends on the preceding sequence of operations (e.g., "cutting tomato" is typically preceded by "take tomato $\rightarrow$ take knife $\rightarrow$ place plate"). Long-term context is crucial for understanding and predicting actions.

**Challenges of the egocentric viewpoint**: Actions often occur out of field-of-view or are occluded by hands, requiring inference from temporal relations.

**Core Idea**: Design a history processing module specifically for OnTAL, using weak supervision from "action anticipation" to guide the history compressor to focus on relevant frames, and then refining the historical features by aligning them with the current short-term context, ultimately enhancing the quality of anchor features.

## Method

### Overall Architecture

The pipeline of HAT is divided into four modules:
1. **Initial Feature Extraction**: Pre-trained I3D/SlowFast encodes video frames, which are linearly projected into a D-dimensional space and segmented into long-term history $H \in \mathbb{R}^{L_h \times D}$ and short-term window $S \in \mathbb{R}^{L_s \times D}$.
2. **Future-Supervised History Module**: Compresses long-term history $\rightarrow$ Action anticipation head provides weak supervision $\rightarrow$ Future-driven refinement.
3. **History-Augmented Anchor Module**: Transformer Encoder processes the short-term window $\rightarrow$ Decoder generates anchor features $\rightarrow$ Enhances anchors using refined history.
4. **Prediction Module**: Classifier + regressor generate proposals $\rightarrow$ Online NMS + OSN post-processing.

### Key Designs

1. **History Compressor**: The long-term history $L_h$ can be very long (e.g., 256 frames), making direct self-attention complexity of $O(L_h^2)$ unacceptable. Borrowing from LSTR, a set of learnable historical tokens $Q_{hist} \in \mathbb{R}^{L_{comp} \times D}$ ($L_{comp} \ll L_h$) is used as the query of a Transformer Decoder, compressing the long-term history through cross-attention into $H_{comp} = d_{N_c} \circ \ldots \circ d_1(Q_{hist}, H)$. This reduces complexity from $O(L_h^2)$ to $O(L_{comp} \cdot L_h)$.

2. **Action Anticipation Head**: The attention of the history compressor should focus on "historical frames that are helpful for predicting the current action". Thus, a lightweight prediction head is introduced: $H_{comp}$ goes through a linear layer ($D \to D/4$) $\rightarrow$ flatten $\rightarrow$ two fully-connected layers (ReLU + Sigmoid) $\rightarrow$ predicts the action distribution $a \in [0,1]^{1 \times C}$ within the current window $S$. This is **window-level weak supervision** (rather than frame-level or anchor-level), exerting pressure on the compressor to "anticipate the future", forcing it to automatically focus on historical frames related to the current context while suppressing irrelevant background frames. This head is discarded during inference.

3. **Future-Driven History Refinement**: The compressed historical features $H_{comp}$ may contain information less relevant to the current moment. Through $N_r=2$ layers of Transformer Decoder, cross-attention refinement is performed with $H_{comp}$ as the query and $S_{enc}$ (encoded short-term features) as the key/value: $H'_{comp} = d_{N_r} \circ \ldots \circ d_1(H_{comp}, S_{enc})$. A residual connection is added to preserve the original history: $H_{ref} = \text{Norm}(H'_{comp} + H_{comp})$.

4. **History-Driven Anchor Refinement**: After generating initial anchor features $A \in \mathbb{R}^{M \times D}$, cross-attention enhancement is performed through $N_a=5$ layers of Transformer Decoder, with $A$ as query and $H_{ref}$ as key/value. A residual connection yields the final anchor features $A_{final}$, preserving both short-term and long-term information.

### Loss & Training

**Adaptive Focal Loss (AFL)**:

Video action datasets suffer from severe class imbalance (background vastly outnumbers actions, and imbalances also exist among different action classes). AFL dynamically adjusts the focal factor for each foreground class $j$:

$$AFL(p_i, y_i) = \sum_{j=0}^{C} -y_i^j (1-p_i^j)^{\lambda^j} \log(p_i^j)$$

where the background class has $\lambda^j = \lambda_b$, and the foreground classes have $\lambda^j = \lambda_b + \lambda_f^j$, with $\lambda_f^j = s(1 - r^j)$. Here, $r^j$ is the ratio of cumulative gradients of positive to negative samples for the $j$-th class. Classes with more severe imbalance have smaller $r^j$ and larger $\lambda_f^j$, thereby receiving more attention.

**Total Loss**: $\mathcal{L} = \alpha \mathcal{L}_c + \beta(\mathcal{L}_o + \mathcal{L}_l) + \gamma \mathcal{L}_a$
- $\mathcal{L}_c$: Classification loss (AFL), with $\alpha=1$
- $\mathcal{L}_o, \mathcal{L}_l$: Offset and length regression losses (L1), with $\beta=1$
- $\mathcal{L}_a$: Action anticipation loss (AFL), with $\gamma=0.2$

AFL parameters: $\lambda_b = 0.025$, $s = 0.05$.

## Key Experimental Results

### Main Results — Procedural Egocentric Datasets (PREGO)

| Dataset | Model | mAP@0.1 | mAP@0.3| mAP@0.5 | Avg mAP | Gain |
|--------|------|---------|---------|---------|---------|------|
| EGTEA | OAT | 24.9 | 20.6 | 12.2 | 19.6 | - |
| EGTEA | **HAT** | **27.5** | **22.6** | **13.5** | **21.5** | **+1.9** |
| EK-100 | OAT | 17.8 | 14.3 | 10.1 | 14.2 | - |
| EK-100 | **HAT** | **18.3** | **15.8** | **11.5** | **15.3** | **+1.1** |

### Main Results — Standard OnTAL Datasets (Non-PREGO)

| Dataset | Model | mAP@0.3 | mAP@0.5 | mAP@0.7 | Avg mAP |
|--------|------|---------|---------|---------|---------|
| THUMOS'14 | OAT | 63.0 | 47.1 | 20.0 | 44.6 |
| THUMOS'14 | **HAT** | 62.0 | **48.0** | **20.7** | **44.8** |
| MUSES | OAT | 16.7 | 10.0 | 3.2 | 9.8 |
| MUSES | **HAT** | **19.1** | **10.1** | **3.7** | **10.8** |

### Ablation Study

**Impact of historical module components (EGTEA Split-1)**:

| Configuration | mAP@0.1 | mAP@0.5 | Avg mAP | Description |
|------|---------|---------|---------|------|
| Full FSHM | **27.3** | **12.8** | **20.8** | Best |
| W/o action anticipation head | 25.8 | 11.5 | 19.8 | Anticipation-guided calibration is most critical |
| W/o future-driven refinement | 26.1 | 12.0 | 20.0 | Alignment with current context is helpful |
| W/o AH+FDHR (compression only) | 24.7 | 11.7 | 19.2 | Naive compression still outperforms no history |
| W/o historical module entirely | 24.2 | 10.9 | 18.9 | Drops by 1.9%, history is crucial |

**Comparison with other SOTA historical modules (EGTEA Split-1)**:

| Method | mAP@0.1 | mAP@0.5 | Avg mAP | Description |
|------|---------|---------|---------|------|
| No history | 24.2 | 10.9 | 18.9 | Baseline |
| LSTR (two-stage compression) | 25.3 | 12.0 | 19.5 | Designed for OAD |
| GateHUB (gated compression) | 26.2 | 11.7 | 19.9 | Gated mechanism underperforms anticipation guidance |
| **FSHM (Ours)** | **27.3** | **12.8** | **20.8** | Anticipation guidance + refinement is optimal |

**Comparison of loss functions (EGTEA Split-1)**:

| Loss Function | mAP@0.1 | mAP@0.5 | Avg mAP |
|----------|---------|---------|---------|
| Cross-Entropy | 25.1 | 12.1 | 19.4 |
| Regular Focal Loss | 26.6 | 11.8 | 19.9 |
| BG-Suppression Focal | 27.2 | 11.5 | 20.2 |
| **Adaptive Focal Loss** | **27.3** | **12.8** | **20.8** |

### Key Findings

- Historical context brings the largest improvement on **procedural egocentric datasets** (a 10% relative performance gain on EGTEA), because strong dependency relations exist between actions.
- Improvements are limited on standard datasets like THUMOS (+0.2 Avg mAP) because these videos contain very few action classes per video (averaging only 1.2 classes).
- Qualitative analysis shows that the attention of the history compressor indeed focuses on relevant frames (e.g., focusing on preceding actions like fetching tomatoes and taking a knife when cutting tomatoes), while irrelevant egocentric movements and background frames are automatically suppressed.
- AFL dynamically focuses on hard classes through gradient feedback, which is more flexible than having a fixed focal factor.

## Highlights & Insights

- **First to introduce long-term history to OnTAL**: Fills the gap in historical context utilization within OnTAL tasks.
- **Design concept of "anticipating the future to calibrate the past"**: The action anticipation head provides weak supervision to guide the compressor, avoiding the difficulties of explicit attention design.
- Adaptive Focal Loss addresses the imbalance among foreground classes, a dimension neglected by existing focal loss variants.
- Inference speed of 147 FPS (RTX 4090), satisfying online real-time requirements.

## Limitations & Future Work

- **Fixed history length**: $L_h$ requires manual hyperparameter tuning; different actions may require different lengths of historical context.
- Dynamic history span adjustment (similar to adaptive attention span in language models) has not been explored.
- The advantage is less pronounced on non-procedural datasets, indicating a certain dependency on data characteristics.
- The dynamic history length mentioned by the authors is a valuable direction for future work.

## Related Work & Insights

- Relationship with OAT: HAT adds the historical module and AFL based on OAT, which is an incremental improvement.
- Comparison with LSTR/GateHUB: Demonstrates that tailoring history processing specifically for OnTAL is more effective than directly reusing OAD methods.
- Insights for egocentric action understanding: Long-term context is crucial for procedural action localization, and future research should design more methods utilizing action dependencies.
- The gradient-guided class balancing idea of AFL can be generalized to other class-imbalance detection tasks.

## Rating

- Novelty: ⭐⭐⭐ The components of the historical module (compression + anticipation + refinement) are reasonably designed but represent incremental innovation. The AFL concept originates from the object detection domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets + comprehensive ablation studies + comparison with multiple historical methods + qualitative attention analysis.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the introduction of the PREGO scenario effectively highlights the value of historical context.
- Value: ⭐⭐⭐⭐ Pioneers the research direction of OnTAL combined with historical context, providing particular inspiration for the egocentric vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Online Temporal Action Localization with Memory-Augmented Transformer](online_temporal_action_localization_with_memory-augmented_transformer.md)
- [\[ECCV 2024\] Spherical World-Locking for Audio-Visual Localization in Egocentric Videos](spherical_world-locking_for_audio-visual_localization_in_egocentric_videos.md)
- [\[ECCV 2024\] Exploring the Feature Extraction and Relation Modeling For Light-Weight Transformer Tracking](exploring_the_feature_extraction_and_relation_modeling_for_light-weight_transfor.md)
- [\[ECCV 2024\] Bayesian Evidential Deep Learning for Online Action Detection](bayesian_evidential_deep_learning_for_online_action_detection.md)
- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)

</div>

<!-- RELATED:END -->
