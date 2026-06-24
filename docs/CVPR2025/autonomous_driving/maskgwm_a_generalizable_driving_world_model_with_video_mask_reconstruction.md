---
title: >-
  [Paper Note] MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction
description: >-
  [CVPR 2025][Autonomous Driving][World Models] This paper combines the MAE-style mask reconstruction task with the diffusion generation process to propose the MaskGWM driving world model. Through three innovative designs—diffusion-related mask tokens, row-wise mask attention, and a row-wise cross-view module—it significantly outperforms current SOTA methods in both long-term prediction and multi-view generation scenarios.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "World Models"
  - "Diffusion Transformer"
  - "Mask Reconstruction"
  - "Multi-view Generation"
  - "Long-term Prediction"
date: 2026-05-08
content_hash: 559545661919b71d
---

# MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2502.11663](https://arxiv.org/abs/2502.11663)  
**Code**: [https://github.com/SenseTime-FVG/OpenDWM](https://github.com/SenseTime-FVG/OpenDWM)  
**Area**: Autonomous Driving  
**Keywords**: World Models, Diffusion Transformer, Mask Reconstruction, Multi-view Generation, Long-term Prediction

## TL;DR
This paper combines the MAE-style mask reconstruction task with the diffusion generation process to propose the MaskGWM driving world model. Through three innovative designs—diffusion-related mask tokens, row-wise mask attention, and a row-wise cross-view module—it significantly outperforms current SOTA methods in both long-term prediction and multi-view generation scenarios.

## Background & Motivation

**Background**: Driving world models aim to predict future environmental changes based on actions, which is key to achieving strong generalization capabilities in autonomous driving. Current prevailing solutions are built upon video prediction models, utilizing diffusion models (such as SVD, Vista, etc.) to generate high-fidelity video sequences.

**Limitations of Prior Work**: Although diffusion-based generators can produce realistic video frames, these models face two major bottlenecks: (1) limited prediction horizons, making it difficult to support long-term rollouts; (2) insufficient generalization capability, with performance dropping significantly on unseen scenes (e.g., zero-shot transfer from nuScenes to Waymo). The fundamental reason is that pure pixel-level generation loss lacks explicit modeling of high-level semantic context, making the model prone to overfitting to visual details of the training distribution rather than the scene structure.

**Key Challenge**: Pixel-level generation focuses on local textures and details, whereas generalization requires the model to understand global spatio-temporal structures. MAE-style feature-level context learning excels at capturing such high-level structural information, but integrating it effectively with the diffusion generation process is non-trivial, as they differ fundamentally in information processing levels and training objectives.

**Goal**: To design a training paradigm that integrates pixel generation with feature mask reconstruction within the Diffusion Transformer (DiT) framework, allowing the model to maintain generation quality while gaining stronger scene understanding capabilities, thereby improving long-term prediction and cross-dataset generalization performance.

**Key Insight**: The authors observe that masked self-attention in MAE suffers from information leakage and computational inefficiency when scaled to spat-temporal domains. This work takes a different approach by adopting shifted self-attention with row-wise masking, which naturally fits the DiT architecture. It also designs diffusion-related mask tokens to bridge the semantic gap between mask reconstruction and diffusion denoising processes.

**Core Idea**: To introduce an auxiliary mask reconstruction task into the DiT driving world model. By utilizing diffusion-noise-aware mask tokens and a spatio-temporal row-wise masking strategy, the model learns more generalizable scene representations while generating high-quality videos.

## Method

### Overall Architecture
MaskGWM is based on the Diffusion Transformer (DiT) architecture, taking previous frame sequences and action conditions (e.g., 3D bounding box layouts) as inputs, and outputting future video frame sequences. Alongside standard diffusion generation training, the model incorporates a parallel mask reconstruction branch: a portion of the input tokens is randomly masked, and the model is required to reconstruct the masked content at the feature level. The final training objective is a weighted sum of the diffusion loss and the mask reconstruction loss. The model has two variants: MaskGWM-long, which focuses on long-term prediction (autoregressive multi-step rollout), and MaskGWM-mview, which focuses on multi-view consistent generation.

### Key Designs

1. **Diffusion-related Mask Tokens (Diffusion-related Mask Tokens)**:

    - **Function**: Insert learnable placeholder tokens at masked positions for subsequent reconstruction.
    - **Mechanism**: Unlike MAE, which uses fixed mask tokens, MaskGWM's mask tokens are related to the diffusion timestep $t$. Specifically, the mask tokens are injected with information matching the current diffusion noise level to align them with the surrounding noisy tokens in the semantic space. This is implemented by modulating the diffusion timestep embedding with the mask tokens.
    - **Design Motivation**: In diffusion models, features at different timesteps contain different levels of information (global structure at high noise levels, local details at low noise levels). If the mask tokens are unaware of the diffusion state, a semantic mismatch will occur between the reconstruction target and the generation target, causing the two tasks to interfere with each other rather than mutually benefit.

2. **Row-wise Mask with Shifted Self-Attention (Row-wise Mask with Shifted Self-Attention)**:

    - **Function**: Extend mask reconstruction from the spatial domain to the spatio-temporal domain while ensuring computational efficiency.
    - **Mechanism**: Instead of the random token-wise masking and masked self-attention used in MAE, MaskGWM adopts a row-wise masking strategy—masking tokens in units of spatial rows. During attention calculation, shifted self-attention is used instead of masked self-attention: masked rows and visible rows compute attention separately, and information is interacted with through shift operations. This design avoids the attention sparsity issue caused by a large number of masks in masked attention.
    - **Design Motivation**: The spatial structure of driving scenes exhibits prominent row-wise patterns (such as ground, vehicles, and sky distributed from bottom to top). Row-wise masking is more suitable for capturing this structural prior than random masking. Meanwhile, the computational complexity of shifted attention is significantly lower than full masked attention.

3. **Row-wise Cross-View Module (Row-wise Cross-View Module)**:

    - **Function**: Maintain geometric consistency across different camera views in multi-view generation scenarios.
    - **Mechanism**: The same physical regions from different perspectives often appear in similar spatial rows. The module establishes cross-view attention between corresponding rows of features from various views, constraining the representation of the same object across different cameras. This module naturally aligns with the row-wise masking design: masked rows can obtain complementary information from visible rows of other views during cross-view interaction.
    - **Design Motivation**: Traditional multi-view consistency in driving world models often relies on global attention or ray projection, which is computationally expensive and sensitive to calibration. Row-wise cross-view attention capitalizes on the structural patterns of driving scenes, achieving cross-view alignment at a lower cost.

### Loss & Training
The total loss is a weighted sum of the diffusion denoising loss $\mathcal{L}_{diff}$ and the mask reconstruction loss $\mathcal{L}_{mask}$: $\mathcal{L} = \mathcal{L}_{diff} + \lambda \mathcal{L}_{mask}$. Mask reconstruction is performed in the feature space (rather than the pixel space), aiming to restore the intermediate layer features of the masked tokens in the DiT. The masking ratio during training is approximately 50%. MaskGWM-long uses autoregressive rollout training, while MaskGWM-mview processes 6 surround-view camera inputs simultaneously.

## Key Experimental Results

### Main Results

| Dataset | Metric | MaskGWM | Prev. SOTA (Vista) | Gain |
|--------|------|---------|-----------------|------|
| nuScenes | FVD↓ | 89.5 | 122.7 | -27.1% |
| nuScenes | FID↓ | 15.3 | 24.1 | -36.5% |
| OpenDV-2K (15s long-term rollout) | FVD↓ | 178.3 | 265.4 | -32.8% |
| Waymo (Zero-shot) | FVD↓ | 134.2 | 198.6 | -32.4% |
| nuScenes Multi-view | FVD↓ | 95.7 | 142.3 | -32.7% |

### Ablation Study

| Configuration | nuScenes FVD↓ | Waymo FVD↓ (Zero-shot) | Description |
|------|--------------|-------------------|------|
| Full MaskGWM | 89.5 | 134.2 | Full model |
| w/o mask reconstruction task | 112.8 | 185.3 | Degenerates to pure DiT, generalization drops significantly |
| w/o diffusion-related mask token | 98.6 | 156.7 | Fixed mask tokens reduce generation-reconstruction synergy |
| Random masking instead of row-wise masking | 103.2 | 162.1 | Row-wise masking shows clear advantages |
| Masked attention instead of shifted attention | 105.1 | 168.4 | Shifted attention is more efficient and effective |

### Key Findings
- The mask reconstruction task brings the most significant improvement to generalization: removing this task degrades the Waymo zero-shot FVD by 38%, indicating that feature-level context learning is the key to cross-dataset generalization.
- The design of diffusion-related mask tokens contributes significantly to generation quality; using fixed mask tokens increases the nuScenes FVD by approximately 10%.
- In long-term rollout scenarios (15-second, 7-step autoregressive), MaskGWM's advantage is even more prominent, indicating that the structural understanding provided by mask reconstruction is more robust against error accumulation.
- Row-wise masking consistently outperforms random masking in driving scenes, validating the effectiveness of the row-wise spatial structure prior of driving environments.

## Highlights & Insights
- **The fusion idea of diffusion and MAE** is ingenious: instead of simply stacking two losses, it deeply integrates the two learning paradigms through diffusion-aware mask tokens and row-wise masking designs, allowing generation capability and representation learning to mutually benefit.
- **The utilization of domain priors in row-wise masking** is noteworthy: the structural characteristics of driving scenes (vertical stratification of ground-vehicles-sky) are cleverly encoded into the masking strategy. This approach of integrating domain knowledge into pre-training task design can be transferred to other scenarios with clear spatial structures.
- **Demonstrating the value of mask reconstruction from a generalization perspective**: While most MAE works focus on downstream fine-tuning performance, this paper proves the unique value of feature-level reconstruction from the angles of generalization and long-term reasoning in world models.

## Limitations & Future Work
- Limited scene complexity: Experiments are mainly conducted in structured road scenes, and performance in complex scenarios such as dense urban traffic and extreme weather is not fully validated.
- Error accumulation still exists in autoregressive long-term generation: Although much better than the baseline, the quality of videos over 15 seconds still decreases significantly.
- The row-wise masking strategy relies on a strong assumption about driving scenes (row-wise spatial structure); migrating to other video generation tasks requires redesigning the masking strategy.
- The evaluation of multi-view geometric consistency is relatively limited, lacking validation on downstream tasks such as depth estimation.

## Related Work & Insights
- **vs Vista**: Vista is a prior SOTA driving world model based on an SVD fine-tuning scheme. MaskGWM switches to the DiT architecture and introduces mask reconstruction, significantly outperforming Vista across all metrics. This demonstrates that the DiT with auxiliary tasks route is superior to the pure diffusion generation route.
- **vs DriveDreamer / GenAD**: These methods focus on control precision under conditions, while MaskGWM emphasizes generalization and long-term sequencing, making their focus areas complementary.
- **vs MAE / VideoMAE**: Classical masked autoencoding methods are used for representation learning. MaskGWM innovatively introduces this to generative model training, resolving the compatibility issues between MAE and diffusion models.

## Rating
- Novelty: ⭐⭐⭐⭐ The integration of mask reconstruction and diffusion generation is innovative, though individual components are somewhat engineering-oriented.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three datasets, two task scenarios, and detailed ablations, but lacks evaluation of downstream planning.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-explained motivation.
- Value: ⭐⭐⭐⭐ Provides an effective generalization enhancement scheme for driving world models, with open-source code adding practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration](recondreamer_crafting_world_models_for_driving_scene_reconstruction_via_online_r.md)
- [\[ICCV 2025\] DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving](../../ICCV2025/autonomous_driving/drivex_omni_scene_modeling_for_learning_generalizable_world_knowledge_in_autonom.md)
- [\[CVPR 2026\] GenieDrive: Towards Physics-Aware Driving World Model with 4D Occupancy Guided Video Generation](../../CVPR2026/autonomous_driving/geniedrive_towards_physics-aware_driving_world_model_with_4d_occupancy_guided_vi.md)
- [\[CVPR 2025\] GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](gaussianworld_gaussian_world_model_for_streaming_3d_occupancy_prediction.md)
- [\[CVPR 2025\] SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model](scenediffuser_city-scale_traffic_simulation_via_a_generative_world_model.md)

</div>

<!-- RELATED:END -->
