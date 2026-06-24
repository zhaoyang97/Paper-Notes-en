---
title: >-
  [Paper Note] FIction: 4D Future Interaction Prediction from Video
description: >-
  [CVPR 2025 (Highlight)][4D interaction prediction] This paper proposes FIction, the first model for 4D future interaction prediction from video. Given an input video, it predicts which objects in the environment a person will interact with, at what 3D locations the interaction will occur, and how the interaction will be executed (3D human pose), achieving over 30%+ relative gain compared to prior methods on the EgoExo4D dataset.
tags:
  - "CVPR 2025 (Highlight)"
  - "4D interaction prediction"
  - "future prediction"
  - "human pose"
  - "egocentric video"
  - "EgoExo4D"
date: 2026-05-08
content_hash: d8417ac947decfd7
---

# FIction: 4D Future Interaction Prediction from Video

**Conference**: CVPR 2025 (Highlight)  
**arXiv**: [2412.00932](https://arxiv.org/abs/2412.00932)  
**Code**: None  
**Area**: Other  
**Keywords**: 4D interaction prediction, future prediction, human pose, egocentric video, EgoExo4D

## TL;DR
This paper proposes FIction, the first model for 4D future interaction prediction from video. Given an input video, it predicts which objects in the environment a person will interact with, at what 3D locations the interaction will occur, and how the interaction will be executed (3D human pose), achieving over 30%+ relative gain compared to prior methods on the EgoExo4D dataset.

## Background & Motivation

**Background**: Predicting how humans interact with objects in their environment is a core problem in activity understanding. Existing methods (such as video prediction, action anticipation, etc.) primarily perform future prediction in the 2D video frame space—predicting "what" action but are limited to physically ungrounded 2D predictions.

**Limitations of Prior Work**: Current methods suffer from three core deficiencies. First, predicting solely in the 2D space fails to answer the "where in 3D" question—such as which physical location in the room a person will walk to in order to fetch an object. Second, they overlook "how" the interaction occurs—for instance, the pose of bending over, reaching out, or pulling. Third, the "post-processing" approach of lifting 2D predictions to 3D accumulates errors because 2D-to-3D lifting is inherently highly ambiguous.

**Key Challenge**: 4D interaction prediction requires simultaneously answering what (which object to interact with), where (which location in the 3D space), and how (to execute with what pose). These three aspects are highly coupled, but existing methods treat them in isolation.

**Goal**: To design an end-to-end model that directly predicts the complete 4D information of future interactions from video—including the category of the interacted object, the 3D interaction location, and the 3D human pose.

**Key Insight**: Utilizing the rich 3D annotations provided by the first-person (ego) and third-person (exo) 4D dataset EgoExo4D to directly model future interactions in the 3D space.

**Core Idea**: Fusing observations of human actions from past video frames with environmental 3D information to simultaneously predict both the where (3D location) and the how (human pose sequence) of future interactions through a unified model.

## Method

### Overall Architecture
The input to FIction is an observed video segment (either ego or exo perspective), and the goal is to predict within the next time interval: (1) which objects the person will interact with (semantic information), (2) where the interaction will occur in 3D space (3D heatmap), and (3) what pose the person will use to perform the interaction (3D human pose sequence). The model consists of several core components: a video encoder to extract spatiotemporal features, an environment representation encoder to extract scene 3D structural information, and an interaction prediction head to predict the "where" and "how".

### Key Designs

1. **Video-Environment Feature Fusion**:

    - **Function**: Integrating human action patterns observed in past videos and the 3D structural information of the environment.
    - **Mechanism**: Utilizing a pre-trained video encoder (e.g., Video Transformer) to extract spatiotemporal video features, capturing past human action patterns and motion trajectories. Meanwhile, a 3D representation of the scene (point clouds or BEV features) is used to encode the spatial layout of the environment—such as the locations of doors, tables, and traversable areas. These two types of features are fused via a cross-attention mechanism, enabling the model to understand "what is available to interact with in the environment" alongside "what the person is doing".
    - **Design Motivation**: Future human behavior is jointly determined by action intent and environmental affordance—prior knowledge of where the refrigerator is in a kitchen is essential to predict where a person will walk to open it.

2. **3D Interaction Location Prediction (Where Prediction)**:

    - **Function**: Predicting the heatmap distribution of future interactions in 3D space.
    - **Mechanism**: Voxelizing the 3D space to predict the interaction probability distribution of each voxel. The fused features pass through a decoder head to generate a 3D heatmap, where high-value regions represent high-probability interaction locations. It can simultaneously predict multiple interaction hotspots—e.g., the person might visit a cabinet before going to the refrigerator. Compared to predicting in 2D and then lifting to 3D, direct prediction in 3D space avoids depth ambiguity.
    - **Design Motivation**: 3D location prediction is physically more meaningful than 2D—robots need to know "which coordinate to approach in the real world," rather than just "which pixel in an image."

3. **3D Pose Sequence Prediction (How Prediction)**:

    - **Function**: Predicting the 3D human pose sequence when executing the interaction.
    - **Mechanism**: Generating a sequence of future 3D human pose keypoints conditioned on the predicted interaction position. The pose prediction accounts for physical constraints of the interaction—e.g., bending down to pick up an object from the floor requires a completely different pose compared to reaching for something on a shelf. The pose sequence is generated via autoregressive or parallel decoding, outputting 3D joint positions at each timestep.
    - **Design Motivation**: Simply predicting "where" is insufficient—embodied AI needs to know "with what pose" to perform an action for motion planning.

### Loss & Training
Training is conducted with the EgoExo4D dataset, which contains rich egocentric and exocentric videos as well as 3D annotations. The loss function includes a heatmap regression loss for the interaction locations and an L2 regression loss for the pose joints. The model is trained across diverse daily activity scenarios, such as cooking, health/fitness, and crafting.

## Key Experimental Results

### Main Results (EgoExo4D Dataset)

| Method | Type | Where Metric | How Metric | Overall Relative Gain |
|------|------|----------|---------|------------|
| 2D Baseline (ATC) | 2D Prediction + Lifting | Baseline | Baseline | - |
| Autoregressive Video Model | 2D Autoregressive | Below Baseline | Below Baseline | - |
| Lifted 2D Method | 2D + 3D Lifting | Moderate | Moderate | - |
| **FIction (Ours)** | Direct 4D | **Best** | **Best** | **>30%** |

### Ablation Study

| Configuration | Where Performance | How Performance | Description |
|------|-----------|---------|------|
| Full FIction | Best | Best | Full Model |
| w/o Environmental Info | Significant Decrease | Decrease | No encoding of 3D scene structure |
| w/o Video History | Decrease | Decrease | Uses only current frame without past history |
| 2D→3D Lifting | Far Below Direct 3D | Far Below | Predicting in 2D then lifting to 3D |
| w/o Pose Prediction | Comparable | N/A | Only predicts location |

### Key Findings
- **Direct prediction in 3D space is significantly superior to "2D prediction + 3D lifting"**—the depth ambiguity in the latter acts as an insurmountable bottleneck, and the 30%+ gain emphasizes the importance of end-to-end 3D modeling.
- **Environmental 3D information is key**—performance dramatically drops when eliminating the environmental representations, validating the crucial role of "understanding environment layout" for future interaction prediction.
- **Video history provides action intent cues**—past behavioral patterns (e.g., cooking activities) effectively help predict the target of the next interaction.
- Performance is consistent across different activity types and environments, indicating that the model learns a generalized capacity for interaction prediction rather than scene memorization.

## Highlights & Insights
- The problem definition itself is the largest contribution—lifting "future interaction prediction" from 2D "what" to 4D "what + where + how" provides more practical predictive signals for embodied AI. This problem definition can inspire further research.
- The video-environment fusion concept can be transferred to many embodied tasks—such as navigation (predicting where a person is going to walk), robot manipulation (predicting what assistance a human needs), or AR applications (predicting where a user's attention translates).
- As a CVPR Highlight, the core contribution of this paper lies in its "pioneering problem definition" rather than pure technical complexity.

## Limitations & Future Work
- Limitations acknowledged by the authors: Reliance on the 3D annotations of EgoExo4D for training, which incurs a high data-acquisition cost.
- Self-identified limitations: (1) The prediction time-window is relatively fixed, and long-term prediction capability remains unverified; (2) Multi-person interaction scenarios are not handled; (3) Physical interaction constraints between humans and the environment (e.g., collision detection) are not considered.
- Scene diversity in EgoExo4D is limited (mainly indoor daily activities); the generalization ability to outdoor or sports scenarios is unknown.
- Future extensions could include: longer-term prediction horizons, multi-agent interaction prediction, and integration with large language models for interpretable intent reasoning.

## Related Work & Insights
- **vs. Traditional Action Anticipation**: Traditional methods predict only action labels (e.g., "pick up a cup"), whereas FIction also predicts 3D location and pose, containing significantly richer information.
- **vs. Human Motion Prediction**: Motion prediction typically assumes the scene layout and interaction intent are known, whereas FIction must autonomously infer "what object will be interacted with".
- **vs. EgoBody / PROX**: These works reconstruct or analyze current human-object interactions, whereas FIction predicts future interactions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 4D future interaction prediction is a brand-new problem definition, which is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation and ablation studies on EgoExo4D, though limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐⭐ The problem motivation is clearly articulated and the methodology description is intuitive, fully deserving its status as a Highlight.
- Value: ⭐⭐⭐⭐⭐ It plays a significant role in advancing the fields of embodied AI and human behavior understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PIPHEN: Physical Interaction Prediction with Hamiltonian Energy Networks](../../AAAI2026/others/piphen_physical_interaction_prediction_with_hamiltonian_energy_networks.md)
- [\[ICCV 2025\] C4D: 4D Made from 3D through Dual Correspondences](../../ICCV2025/others/c4d_4d_made_from_3d_through_dual_correspondences.md)
- [\[CVPR 2025\] Event Ellipsometer: Event-based Mueller-Matrix Video Imaging](event_ellipsometer_event-based_mueller-matrix_video_imaging.md)
- [\[CVPR 2025\] CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction](care_transformer_linear_attention.md)
- [\[NeurIPS 2025\] Evolutionary Prediction Games](../../NeurIPS2025/others/evolutionary_prediction_games.md)

</div>

<!-- RELATED:END -->
