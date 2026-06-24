---
title: >-
  [Paper Note] AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale
description: >-
  [ECCV 2024][Reinforcement Learning][Active Visual Exploration] This paper proposes AdaGlimpse, which utilizes Soft Actor-Critic (SAC) reinforcement learning to select glimpses of arbitrary positions and scales from a continuous action space. Combined with a ViT encoder equipped with elastic positional encoding, it achieves multi-task active visual exploration (reconstruction, classification, and segmentation), outperforming state-of-the-art methods that use 18% of pixels…
tags:
  - "ECCV 2024"
  - "Reinforcement Learning"
  - "Active Visual Exploration"
  - "Soft Actor-Critic"
  - "variable-scale glimpse"
  - "Vision Transformer"
  - "reinforcement-learning"
date: 2026-05-08
content_hash: a73417075504b9ad
---

# AdaGlimpse: Active Visual Exploration with Arbitrary Glimpse Position and Scale

**Conference**: ECCV 2024  
**arXiv**: [2404.03482](https://arxiv.org/abs/2404.03482)  
**Code**: [https://github.com/apardyl/AdaGlimpse](https://github.com/apardyl/AdaGlimpse)  
**Area**: Active Visual Exploration / Efficient Visual Reasoning / Reinforcement Learning  
**Keywords**: Active Visual Exploration, Soft Actor-Critic, variable-scale glimpse, Vision Transformer, reinforcement-learning

## TL;DR
This paper proposes AdaGlimpse, which utilizes Soft Actor-Critic (SAC) reinforcement learning to select glimpses of arbitrary positions and scales from a continuous action space. Combined with a ViT encoder equipped with elastic positional encoding, it achieves multi-task active visual exploration (reconstruction, classification, and segmentation), outperforming state-of-the-art methods that use 18% of pixels, while requiring only 6% of pixels.

## Background & Motivation
Active Visual Exploration (AVE) investigates how agents dynamically select observation regions from the environment to perform visual tasks—which is crucial for resource-constrained platforms such as robots and UAVs. However, prior AVE methods have a critical limitation: they can only select glimpses of fixed scales from a fixed grid (either equal-sized patches or foveated sampling), failing to exploit the capabilities of modern hardware. For instance, PTZ cameras can freely adjust focal length and field of view, and UAVs can alter their altitude to adjust the observation range. Bridging this gap between software and hardware capabilities is the main goal of this paper.

## Core Problem
How to enable an AVE agent to freely select observation positions and zoom scales in a continuous space, mimicking the human visual system's ability to "first scan globally, then inspect locally," thereby understanding scenes with minimal observations? The core challenges lie in: (1) The continuous position and scale action space makes the sampling operation non-differentiable, preventing direct gradient optimization; (2) Patches of different scales need to be encoded and processed uniformly; (3) The agent must balance exploration (observing new regions) and exploitation (observing important regions).

## Method
The core idea of AdaGlimpse is to allow the agent to simultaneously determine the position $(x,y)$ and zoom scale $z$ of the next observation in a continuous space, forming a triplet $(x,y,z) \in [0,1]^3$. The overall pipeline is: at each step, the RL agent predicts the position and scale of the next glimpse based on prior observations $\rightarrow$ the camera captures this region $\rightarrow$ the ViT encoder processes all collected patches $\rightarrow$ the downstream task predictions and a new state representation are generated $\rightarrow$ loop.

### Overall Architecture
The system consists of two parts: (1) A ViT encoder based on ElasticViT combined with a task head to process variable-scale patches and complete downstream tasks; (2) A Soft Actor-Critic RL agent that decides the coordinates and scale of the next glimpse based on the state representation output by the encoder. The two modules are trained alternately—only the RL agent is trained during the first 30 epochs, followed by alternating optimization of the backbone and the RL agent.

### Key Designs
1. **Adaptive Glimpse Sampling in Continuous Action Space**: Unlike prior approaches restricted to discrete grids, AdaGlimpse defines glimpse coordinates as continuous values $(x,y,z)$, where $z$ controls the zoom level ($z=0$ representing maximum zoom-in, $z=1$ representing the widest field of view). Each glimpse is sampled at a fixed resolution $d_{cam} \times d_{cam}$, but can cover scene regions of varying sizes. This allows the model to first inspect the global view at a lower resolution, and then focus on details at a higher resolution.

2. **ElasticViT (Elastic Positional Encoding)**: Standard ViTs assume that patches originate from a regular grid, whereas the patches in AdaGlimpse vary in both position and scale. This paper adopts the positional encoding scheme of ElasticViT, which calculates positional embeddings based on the actual coordinates of each patch in the original scene, breaking the grid limitation. Meanwhile, attention rollout is utilized to estimate the importance of each patch $\hat{I}_t$, providing informativity signals to the RL agent.

3. **Task-Adaptive Decoder Design**: For classification tasks, the class token is directly passed through a linear layer to produce the output. For dense prediction tasks (reconstruction and segmentation), an MAE-style Transformer decoder is used. It takes all encoder tokens along with mask tokens for the entire grid as input—unlike MAE, it reconstructs the complete image rather than just the missing patches, since the encoder patches do not align with the decoder grid under variable-scale sampling.

4. **SAC Reinforcement Learning Agent**: Soft Actor-Critic (SAC) is chosen over other RL algorithms because its maximum entropy objective, $V^{\pi_\theta}(s) = \mathbb{E}_\pi[\sum \gamma^t r_t + \alpha \mathcal{H}(\pi(s_t))]$, naturally encourages the exploration of diverse action distributions, making it highly suitable for AVE tasks requiring environmental exploration. The Actor and Critic networks each consist of: a small CNN to process the patch image $\hat{G}_t$, and three sets of MLPs to process coordinates $\hat{C}_t$, importance $\hat{I}_t$, and latent representation $\hat{H}_t$, respectively. These features are then aggregated via attention pooling and passed through an MLP to generate the output. The two networks do not share parameters to ensure training stability.

### Loss & Training
- **Task Loss**: Reconstruction uses RMSE; classification and segmentation use soft labels from a teacher model combined with KL divergence distillation (the classification teacher is DeiT-III, and the segmentation teacher is DeepLabV3-ResNet101).
- **RL Reward**: Defined as the difference in task loss between adjacent steps $r_t = L_{t-1} - L_t$, representing the reduction in loss contributed by each glimpse.
- **Training Strategy**: The model is first pre-trained for 600 epochs with 196 randomly sampled glimpses to learn irregular positional encodings. Then, formal training is conducted for 100 epochs, where only the RL agent is trained in the first 30 epochs, followed by alternating optimization. 3-Augment data augmentation is applied.

## Key Experimental Results

| Dataset | Metric | Ours (AdaGlimpse) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| ImageNet-1k (Reconstruction) | RMSE↓ | **14.5** | 30.3 (AME) | 52.1% |
| SUN360 (Reconstruction) | RMSE↓ | **11.1** | 23.6 (AME) | 53.0% |
| ADE20K (Reconstruction) | RMSE↓ | **14.0** | 23.8 (AME) | 41.2% |
| MS COCO (Reconstruction) | RMSE↓ | **14.5** | 25.2 (AME) | 42.5% |
| ImageNet-1k (Classification) | Acc↑ | **77.54%** | 76.13% (STAM) | +1.41% |
| ImageNet-1k (Classification + Early Stopping) | Acc↑ | 76.30% | 76.13% (STAM) | 40%+ fewer pixels |
| ADE20K (Segmentation, 18% pixels) | PA↑ | **67.4%** | 52.4% (GlAtEx) | +15.0% |
| ADE20K (Segmentation, ~37% pixels) | IoU↑ | **25.7%** | 24.4% (AME, 56% pixels) | 35% fewer pixels |

### Ablation Study
- **Importance of RL State Components** (ImageNet classification): Removing the latent representation $\hat{H}_t$ leads to the largest drop in performance (77.54% $\rightarrow$ 61.82%, -15.7%), indicating that the latent representation of the ViT contains the most critical information. Removing coordinates $\hat{C}_t$ also severely hurts performance (77.54% $\rightarrow$ 68.25%, -9.3%). Removing importance $\hat{I}_t$ has the minimal impact (77.54% $\rightarrow$ 77.36%).
- **Glimpse Policy Analysis**: In reconstruction tasks, the model tends to observe the global view first and then cover the four quadrants evenly. In classification tasks, the model concentrates its attention on the image center after a quick global scan (as objects in ImageNet are mostly centered).
- The model outperforms other methods using 18% of pixels in reconstruction while utilizing only 6% of pixels.

## Highlights & Insights
- **Continuous Action Space as the Core Innovation**: This breaks the paradigm where all methods in the AVE field only select patches from discrete grids. It truly achieves the freedom of "zoom in/zoom out," which is closer to the hardware capability.
- **Well-Suited Selection of SAC**: The maximum entropy property of SAC aligns perfectly with the "exploration" requirement in AVE. The match between this RL algorithm and the problem is highly natural.
- **Task-Agnostic Architecture**: The same framework seamlessly adapts to reconstruction, classification, and segmentation tasks, demonstrating that the ability to "actively select observations" is a task-generic skill.
- **Emergent Coarse-to-Fine Behavior**: The model naturally learns a policy of "inspecting the global low-resolution view first, and then zooming in on details," which is highly consistent with the human visual exploration process.
- An early stopping mechanism is demonstrated in experiments (stopping once threshold confidence is reached), which further reduces computation.

## Limitations & Future Work
- **Quadratic Complexity of ViT**: As the number of glimpses increases, the attention computation overhead for all patches scales quadratically. The authors mention that Mamba could be used as an alternative.
- **Static Scene Assumption**: All experiments are simulated on cropped static images, without considering dynamic real-world environments.
- **High Pre-training Overhead**: The training cost is relatively high, requiring 600 epochs of random glimpse pre-training + 100 epochs of formal joint training.
- **Teacher Model Dependency**: Classification and segmentation tasks rely on distillation from teacher models, which limits the performance upper bound.
- **Extensible Directions**: Integrating Mamba to reduce sequence computation complexity; extending active exploration to videos or 3D scenes; incorporating VLMs for semantic-guided exploration strategies.

## Related Work & Insights
- **vs STAM** (CVPR 2022): STAM also uses actor-critic for glimpse selection in classification tasks, but only supports discrete grids + fixed scales and utilizes one-step AC. AdaGlimpse employs continuous SAC + multi-step exploration + variable scale, achieving a 1.41% accuracy gain on classification.
- **vs AME** (IJCAI 2023): AME performs region selection based on MAE attention maps, which is also restricted to fixed grids. For reconstruction, AdaGlimpse achieves a massive leap in RMSE from 30.3 to 14.5 while using fewer pixels (~24% vs 25%).
- **vs AutoGaze** (CVPR 2026): AutoGaze similarly focuses on "coarse-to-fine" autoregressive patch selection, but is tailored for token compression in video MLLMs. AdaGlimpse focuses more on RL-based decision-making in real active exploration scenarios.

## Inspiration & Connections
- Inspiring new ideas: Transferring AdaGlimpse's continuous-scale RL exploration strategy to multi-sensor attention allocation in autonomous driving—dynamically deciding which regions require high-resolution perception based on scene complexity.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of continuous action space and SAC is a first in AVE, although the overall concept (RL-based region selection + ViT encoding) is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three task categories (reconstruction, classification, and segmentation) across multiple datasets, with detailed ablation and visualization analyses.
- Writing Quality: ⭐⭐⭐⭐ The logic is clear and the MDP formulation is complete, though some notations are somewhat tedious.
- Value: ⭐⭐⭐⭐ Represents a substantial advancement in active visual exploration, but actual deployment remains distant due to ViT complexity limits and static scene assumptions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Deployed Reinforcement Learning should be Continual](../../ICML2026/reinforcement_learning/position_deployed_reinforcement_learning_should_be_continual.md)
- [\[ECCV 2024\] Visual Grounding for Object-Level Generalization in Reinforcement Learning](visual_grounding_for_object-level_generalization_in_reinforcement_learning.md)
- [\[ICML 2025\] LineFlow: A Framework to Learn Active Control of Production Lines](../../ICML2025/reinforcement_learning/lineflow_a_framework_to_learn_active_control_of_production_lines.md)
- [\[ICML 2025\] Position: Lifetime Tuning is Incompatible with Continual Reinforcement Learning](../../ICML2025/reinforcement_learning/position_lifetime_tuning_is_incompatible_with_continual_reinforcement_learning.md)
- [\[ICML 2025\] Stochastic Encodings for Active Feature Acquisition](../../ICML2025/reinforcement_learning/stochastic_encodings_for_active_feature_acquisition.md)

</div>

<!-- RELATED:END -->
