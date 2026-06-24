---
title: >-
  [Paper Note] Generating Multimodal Driving Scenes via Next-Scene Prediction
description: >-
  [CVPR 2025][Autonomous Driving][Multimodal Scene Generation] This paper proposes UMGen, a unified multimodal driving scene generation framework. It tokenizes four modalities—ego-vehicle action, map, traffic participants (agents), and images—and generates scenes step-by-step using a two-stage strategy: temporal autoregression (TAR) across frames and ordered autoregression (OAR) within each frame. Additionally, it introduces an Action-aware Map Alignment (AMA) module to maintai…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Multimodal Scene Generation"
  - "Autoregressive Model"
  - "Driving Simulation"
  - "Next-Scene Prediction"
  - "Temporal Consistency"
date: 2026-05-08
content_hash: 0415df4577d0dca6
---

# Generating Multimodal Driving Scenes via Next-Scene Prediction

**Conference**: CVPR 2025  
**arXiv**: [2503.14945](https://arxiv.org/abs/2503.14945)  
**Code**: [https://yanhaowu.github.io/UMGen](https://yanhaowu.github.io/UMGen) (Project Page)  
**Area**: Autonomous Driving / Scene Generation  
**Keywords**: Multimodal Scene Generation, Autoregressive Model, Driving Simulation, Next-Scene Prediction, Temporal Consistency

## TL;DR

This paper proposes UMGen, a unified multimodal driving scene generation framework. It tokenizes four modalities—ego-vehicle action, map, traffic participants (agents), and images—and generates scenes step-by-step using a two-stage strategy: temporal autoregression (TAR) across frames and ordered autoregression (OAR) within each frame. Additionally, it introduces an Action-aware Map Alignment (AMA) module to maintain consistency between ego-motion and the map, enabling the autonomous generation of coherent driving sequences up to 60 seconds long.

## Background & Motivation

**Background**: Generative models are utilized in autonomous driving to create diverse driving scenes, especially rare or unmapped scenarios in datasets, and to build closed-loop simulation systems for safely testing autonomous driving pipelines.

**Limitations of Prior Work**: Existing approaches generally generate only a limited combination of modalities. GUMP and TrafficGen generate only ego-vehicle actions and agent trajectories without map evolution (making the map static), which limits realism. DriveDreamer and GAIA-1 can generate images but cannot predict the motion of traffic participants, lacking fine-grained control over agent behaviors. No existing method simultaneously generates and maintains consistency across all critical modalities.

**Key Challenge**: Multimodal scene generation faces two challenges: (1) flattening all modality tokens into a single long sequence for vanilla autoregressive (AR) modeling leads to a catastrophic computational explosion; (2) the lack of cross-modal consistency constraints within the same frame easily leads to conflicts.

**Goal**: How can we simultaneously generate all four key modalities (ego action, map, agents, images) within a unified framework, ensuring multimodal consistency and temporal coherence while controlling computational overhead?

**Key Insight**: Decompose the scene generation problem into two sub-problems, inter-frame prediction and intra-frame prediction, which are handled by TAR and OAR respectively. This avoids global attention over ultra-long token sequences. Simultaneously, use the ego-vehicle action to perform affine transformations on the map to maintain consistency between them.

**Core Idea**: Replace vanilla all-token autoregression with a two-level autoregressive strategy of "temporal parallel + intra-modality sequential", which substantially reduces the computational complexity of multimodal scene generation while explicitly enforcing ego-map consistency via the AMA module.

## Method

### Overall Architecture

The pipeline of UMGen is as follows: given a sequence of past $T$ frames of multimodal scenes, (1) each modality (ego action, map, agent, image) is converted into tokens via discretization or VQ-GAN; (2) the ego-action prediction module predicts the ego action for the next frame; (3) the AMA module applies an affine transformation to align map features according to the predicted ego action; (4) the TAR module parallelly aggregates temporal information at each token position via causal attention; (5) the OAR module autoregressively generates intra-frame tokens in a GPT-style with a fixed modality order (ego $\to$ map $\to$ agent $\to$ image); (6) the tokens are decoded to produce the next frame of the scene.

### Key Designs

1. **Temporal Autoregression (TAR)**:

    - **Function**: Capture the evolution patterns of each token position along the temporal dimension.
    - **Mechanism**: For the $T$-frame token embeddings $\bar{\mathbf{e}}_{1:T}$ aligned by AMA, causal self-attention is applied along the time dimension for each token position $i$: $\bar{\mathbf{e}}_{T+1}^i = \text{CSA}(\bar{\mathbf{e}}_1^i, ..., \bar{\mathbf{e}}_T^i)$. Then, bidirectional self-attention is used within the frame for preliminary cross-modal information exchange. Processing is performed in parallel across token positions, reducing the computational complexity to $O(T \times N)$ compared to $O((T \times N)^2)$ in vanilla AR.
    - **Design Motivation**: Each token position in adjacent frames typically corresponds to the same physical location/object. Thus, applying temporal attention by position efficiently captures motion and migration trends while avoiding the huge overhead of global attention over sequences of length $T \times N$.

2. **Intra-frame Ordered Autoregression (OAR)**:

    - **Function**: Generate tokens within a single frame according to a causal order of modalities, ensuring cross-modality consistency.
    - **Mechanism**: Using the output of TAR ($\mathbf{h}_{T+1}$) as a temporal prior, causal self-attention is performed with the already-generated preceding tokens $\mathbf{o}_{T+1}^{1:i-1}$ to predict the current token $\mathbf{o}_{T+1}^i$. The generation sequence follows the order of ego $\to$ map $\to$ agent $\to$ image, which reflects the physical causal chain: ego action changes the observable map, affects surrounding agent behavior, and is ultimately reflected in the camera images.
    - **Design Motivation**: Causal dependencies exist among modalities (e.g., ego turns $\to$ map rotates $\to$ agents yield $\to$ image changes). Generating them autoregressively in this order explicitly models these dependencies and prevents cross-modal conflicts.

3. **Action-aware Map Alignment (AMA)**:

    - **Function**: Geometrically transform map features based on ego actions to provide a strong prior for map prediction in the next frame.
    - **Mechanism**: Map token embeddings are reshaped into $H \times W$ spatial features. An affine transformation matrix is constructed using the predicted ego action ($\theta, dx, dy$) to generate a sampling grid. Rotative and translative transformations are applied to the map via bilinear interpolation, which are then added to the original map features to produce the transformed map embeddings.
    - **Design Motivation**: Changes in the map under the ego-vehicle coordinate system across adjacent frames are mainly caused by ego-motion. Explicit affine transformations allow low-cost propagation of map information, drastically lowering the difficulty of map generation.

### Loss & Training

The total loss is the sum of the cross-entropy losses of OAR and TAR: $\mathcal{L}_{total} = CE(\mathbf{p}^{OAR}_{T+1}, \mathbf{z}_{T+1}) + CE(\mathbf{p}^{TAR}_{T+1}, \mathbf{z}_{T+1})$. During training, a sequence of 21 frames is randomly sampled at each step. Training is performed on 32 RTX 4090 GPUs for 300 epochs (approx. 2 days). During inference, a Top-k sampling strategy is used to generate tokens.

## Key Experimental Results

### Main Results

Comparison of initial scene generation MMD metrics on nuPlan and WOMD datasets:

| Method | Position↓ | Heading↓ | Speed↓ | Dataset |
|------|-----------|----------|--------|--------|
| TrafficGen | 0.83 | 0.82 | 0.90 | WOMD |
| SceneDM | 0.39 | 0.37 | 0.62 | WOMD |
| **UMGen** | **0.17** | **0.22** | **0.35** | **WOMD** |
| TrafficGen | 3.29 | 1.04 | 4.34 | nuPlan |
| **UMGen** | **0.42** | **0.35** | **0.73** | **nuPlan** |

UMGen outperforms baseline methods by a large margin across all MMD metrics, demonstrating that its generated scenes are more aligned with the real data distribution.

### Ablation Study

| Configuration | Agent MMD↓ | Agent CR↓ | Note |
|------|-----------|-----------|------|
| Full model | 0.31 | 0.018 | Full model |
| w/o TAR | 0.45 | 0.032 | Temporal modeling removed, scene coherence drops |
| w/o OAR | 0.38 | 0.041 | Intra-frame sequential generation removed, collision rate rises significantly |
| w/o AMA | 0.34 | 0.025 | Map alignment removed, spatial consistency decreases |

### Key Findings

- OAR is critical to reducing conflicts between modalities (the collision rate doubles without it), verifying the necessity of intra-frame modality sequence modeling.
- Compared to vanilla AR, TAR shows a significant advantage in inference efficiency: per-token inference time is reduced by approximately 60%, and peak GPU memory is reduced by approximately 40%.
- UMGen can generate coherent multimodal driving sequences lasting up to 60 seconds, displaying strong temporal stability.
- By controlling the ego-action input, user-specified scenarios (e.g., turning, going straight) can be generated, providing flexibility for simulation testing.

## Highlights & Insights

- **Divide-and-Conquer Strategy of Two-Level AR**: TAR handles temporal modeling in parallel, while OAR processes modalities sequentially. This reduces the attention complexity from $O((TN)^2)$ to $O(T \times N)$, serving as a general acceleration scheme for long-sequence multimodal generation that can be transferred to tasks like joint video-audio generation.
- **Physical Causality of Modality Order**: The generation sequence of ego $\to$ map $\to$ agent $\to$ image is not arbitrary but mirrors the causal chain of the real world. Injecting this domain knowledge significantly improves generation quality.
- **Simplicity and Effectiveness of AMA**: Using a single affine transformation successfully maintains ego-map consistency with minimal computational cost yet yields significant improvements.

## Limitations & Future Work

- Image generation relies on VQ-GAN, which has limited resolution and quality; diffusion models could be considered as alternatives in the future.
- The current number of agents is fixed via padding, which lacks flexibility in modeling dynamic agent appearances or disappearances.
- Only front-view generation was demonstrated; whether this can be extended to multi-view consistent scene generation is worth exploring.
- The actual utility of the generated scenes for autonomous driving policy learning was not validated in closed-loop simulations.

## Related Work & Insights

- **vs GAIA-1**: GAIA-1 also performs AR driving video generation but only includes the image and ego-action modalities. UMGen expands this to four modalities and operates more efficiently.
- **vs GUMP**: GUMP generates agent trajectories but uses a static map. UMGen introduces the map modality and the AMA module to make scenes more realistic.
- **vs DriveDreamer**: DriveDreamer uses two independent networks to generate maps and videos, which lacks cross-modal consistency. UMGen's OAR module guarantees consistency under a unified framework.

## Rating

- Novelty: ⭐⭐⭐⭐ Unified generation of four modalities + TAR/OAR two-layer AR is a novel design scheme.
- Experimental Thoroughness: ⭐⭐⭐ Rich qualitative results are provided, but quantitative evaluations are mostly limited to MMD, lacking validation on downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ Clarity in the architecture diagram and complete methodological descriptions.
- Value: ⭐⭐⭐⭐ Provides a promising multimodal generation solution for autonomous driving simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments](scenario_dreamer_vectorized_latent_diffusion_for_generating_driving_simulation_e.md)
- [\[CVPR 2025\] FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes](freesim_toward_free-viewpoint_camera_simulation_in_driving_scenes.md)
- [\[CVPR 2025\] ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling](modeseq_taming_sparse_multimodal_motion_prediction_with_sequential_mode_modeling.md)
- [\[CVPR 2025\] SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction](sdgocc_semantic_and_depth-guided_birds-eye_view_transformation_for_3d_multimodal.md)
- [\[CVPR 2025\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)

</div>

<!-- RELATED:END -->
