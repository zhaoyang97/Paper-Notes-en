---
title: >-
  [Paper Note] Progressive Pretext Task Learning for Human Trajectory Prediction
description: >-
  [ECCV 2024][Autonomous Driving][Pedestrian Trajectory Prediction] Proposes a progressive pretext task learning framework, PPT, which progressively enhances the model's ability to capture short-term dynamics and long-term dependencies through three-stage training (step-by-step next-position prediction → destination prediction → complete trajectory prediction). Together with an efficient two-step non-autoregressive Transformer predictor, it achieves SOTA on multiple pedestrian…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Pedestrian Trajectory Prediction"
  - "Progressive Learning"
  - "Pretext Task"
  - "Transformer"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: 3c6d995ebb1bf542
---

# Progressive Pretext Task Learning for Human Trajectory Prediction

**Conference**: ECCV 2024  
**arXiv**: [2407.11588](https://arxiv.org/abs/2407.11588)  
**Code**: Yes ([https://github.com/iSEE-Laboratory/PPT](https://github.com/iSEE-Laboratory/PPT))  
**Area**: Autonomous Driving  
**Keywords**: Pedestrian Trajectory Prediction, Progressive Learning, Pretext Task, Transformer, Knowledge Distillation

## TL;DR

Proposes a progressive pretext task learning framework, PPT, which progressively enhances the model's ability to capture short-term dynamics and long-term dependencies through three-stage training (step-by-step next-position prediction → destination prediction → complete trajectory prediction). Together with an efficient two-step non-autoregressive Transformer predictor, it achieves SOTA on multiple pedestrian trajectory prediction benchmarks.

## Background & Motivation

Pedestrian trajectory prediction requires predicting all future positions from the short term to the long term. However, short-term and long-term predictions rely on vastly different understanding capabilities:
- **Short-term prediction**: Requires identifying fine-grained local dynamic patterns between adjacent time steps.
- **Long-term prediction**: Requires inferring global motion trends and capturing long-range dependencies of the trajectory.

**Limitations of Prior Work**:
1. Most methods (Social-GAN, MID, LED, etc.) handle predictions across all time horizons using a single unified training paradigm, often making sub-optimal trade-offs between short-term and long-term performance.
2. Although destination-driven methods (MemoNet, PECNet, etc.) first predict the destination and then interpolate intermediate positions, there is a lack of knowledge transfer between the destination predictor and the trajectory predictor, leading to a disconnection between the two.
3. Existing Transformer methods mostly employ autoregressive generation, which has low inference efficiency; non-autoregressive methods like MID rely on diffusion models (slow), or TUTR ignores temporal dynamics (limiting performance).

Core Idea of this paper: Since short-term and long-term predictions require different capabilities, why not train these capabilities progressively in stages?

## Method

### Overall Architecture

The PPT framework consists of three progressive training stages and a Transformer backbone model:

1. **Stage I - Step-by-step next-position prediction**: Learn short-term dynamics
2. **Stage II - Jump-step destination prediction**: Learn long-range dependencies
3. **Stage III - Complete trajectory prediction**: Leverage knowledge from the first two stages to complete the final task

Each stage uses the same architecture but progressively enhances its capabilities, using cross-task knowledge distillation to prevent forgetting.

### Key Designs

**Task-I: Step-by-step next-position prediction**
- Randomly sample a subsequence $\mathcal{S}^{T_1:T_{t-1}}$ from the complete trajectory $\mathcal{S}^{T_1:T_e}$, and predict the next position $\mathcal{S}^{T_t}$.
- Utilize causal self-attention masks to achieve parallel processing of multiple random subsequences in a single forward pass, improving training efficiency.
- Inputs of arbitrary lengths enable the model to comprehensively understand local motion patterns in the trajectory.

**Task-II: Jump-step destination prediction**
- Input the observed trajectory $\mathcal{S}^{T_1:T_h}$, and predict the destination of the entire trajectory $\mathcal{S}^{T_e}$.
- Since there is no position input at time $T_{e-1}$, a **learnable prompt embedding** is appended to the observed sequence, assigned with the positional encoding of $T_{e-1}$, to achieve "jump-step" prediction.
- Predict K=20 candidate destinations, using a precision loss + diversity loss:

$$L_{Des} = \min_k L_2(\hat{\mathbf{E}}_k, \mathbf{E}) + \lambda_d \cdot \frac{1}{K(K-1)} \sum_i \sum_{j \neq i} e^{-L_2^2(\hat{\mathbf{E}}_i, \hat{\mathbf{E}}_j) / \sigma_s}$$

**Task-III: Complete trajectory prediction**
- Copy the trained Task-II model $\theta_{II}$ as both the destination predictor and the trajectory predictor.
- The destination predictor generates K candidate destinations; the destination closest to the ground truth (GT) is input into the trajectory predictor.
- The input to the trajectory predictor consists of three parts: observed trajectory + learnable prompt embeddings (representing future intermediate positions) + pseudo-destination.
- Outputs all future positions in a non-autoregressive, parallel fashion.

**Backbone: Transformer Encoder**
- 3-layer Transformer encoder, hidden dimension of 128, with 8 attention heads.
- The input 2D positions are mapped via an embedding layer and then added to temporal positional encodings.
- Outputs the next-frame prediction for each position, obtaining 2D coordinates through a LayerNorm and a linear projector.

**Cross-Task Knowledge Distillation**:
- $L_{kd}^t$: Trajectory features of the Task-I model guide the Task-III trajectory predictor
- $L_{kd}^d$: Destination features of the Task-II model guide the Task-III destination predictor
- Calculate the L2 distance after aligning feature dimensions using linear projections

### Loss & Training

- **Task-I**: Next-position prediction loss based on L2 distance
- **Task-II**: $L_{Des} = L_{Precision} + \lambda_d L_{Diversity}$, $\lambda_d = 100$
- **Task-III**: $L_{Traj} = L_{Recon} + \lambda_{kd}^t L_{kd}^t + \lambda_{kd}^d L_{kd}^d$, $\lambda_{kd}^t = 5$, $\lambda_{kd}^d = 0.5$

The learning rates for the three stages are 0.001, 0.0001, and 0.0015 respectively. Before Task-II training, the MLP is first warmed up, followed by joint training of the whole model.

## Key Experimental Results

### Main Results

minADE20/minFDE20 on SDD dataset (pixels):

| Method | ADE↓ | FDE↓ |
|------|------|------|
| Social-GAN | 27.23 | 41.44 |
| PECNet | 9.96 | 15.88 |
| MemoNet | 8.56 | 12.66 |
| Social-VAE | 8.10 | 11.72 |
| MID | 7.61 | 14.30 |
| LED | 8.48 | 11.66 |
| TUTR | 7.76 | 12.69 |
| **PPT (Ours)** | **7.03** | **10.65** |

minADE20/minFDE20 on ETH/UCY dataset (meters):

| Method | ETH | HOTEL | UNIV | ZARA1 | ZARA2 | AVG |
|------|-----|-------|------|-------|-------|-----|
| Social-GAN | 0.87/1.62 | 0.67/1.37 | 0.76/1.52 | 0.35/0.68 | 0.42/0.84 | 0.61/1.21 |
| MemoNet | 0.40/0.61 | 0.11/0.17 | 0.24/0.43 | 0.18/0.32 | 0.14/0.24 | 0.21/0.35 |
| SocialVAE | 0.41/0.58 | 0.13/0.19 | 0.21/0.36 | 0.17/0.29 | 0.13/0.22 | 0.21/0.33 |
| **PPT** | **0.36/0.51** | **0.11/0.15** | **0.22/0.40** | **0.17/0.30** | **0.12/0.21** | **0.20/0.31** |

GCS dataset (pixels):

| Method | ADE↓ | FDE↓ |
|------|------|------|
| EigenTrajectory | 7.42 | 12.49 |
| **PPT (Ours)** | **6.20** | **9.34** |

PPT overwhelmingly outperforms SOTA on GCS, with ADE reduced by 16.4% and FDE reduced by 25.2%.

### Ablation Study

Ablation of pretext tasks (SDD dataset):

| Task-I | Task-II | Task-III | ADE↓ | FDE↓ |
|--------|---------|----------|------|------|
| ✗ | ✗ | ✓ | 10.40 | 18.64 |
| ✗ | ✓ | ✓ | 7.71 | 11.42 |
| **✓** | **✓** | **✓** | **7.03** | **10.65** |

Other ablation findings:
- Task-I reduces the destination prediction FDE of Task-II from 11.58 to 10.70
- Cross-task knowledge distillation reduces prediction variance and improves training stability
- The diversity loss weight $\lambda_d = 100$ is optimal; too small leads to mode collapse, while too large sacrifices precision

### Key Findings

1. **Progressive training significantly outperforms direct training**: Direct training without pretext tasks yields an ADE/FDE of 10.40/18.64, which drops to 7.03/10.65 after incorporating two pretext tasks—a massive improvement (32%/43%).
2. **Both pretext tasks contribute and complement each other**: Task-I improves short-term precision, while Task-II enhances long-term accuracy and destination diversity; neither can be omitted.
3. **Efficient inference**: The two-step inference (first destination, then parallel generation of all intermediate points) takes only 5.28ms/sample, which is significantly faster than autoregressive methods (STAR 35.8ms, AgentFormer 99.3ms, MID 736.8ms), comparable to TUTR (4.06ms), but with far superior performance.
4. **Efficient training**: Pre-training in previous stages accelerates convergence in subsequent stages. The total training time on SDD is only 4.7 hours (on a single RTX 3090).

## Highlights & Insights

- **"Easy-to-hard" training philosophy**: Similar to curriculum learning, the model is first trained to walk (next-step prediction), then to look far ahead (destination prediction), and finally to complete the whole journey (complete trajectory). This progressive strategy prevents the model from making sub-optimal trade-offs when simultaneously learning short-term and long-term patterns.
- **Clever design of learnable prompt embeddings**: Prompts are used to represent unknown future positions. Combined with positional encodings, they enable non-autoregressive parallel generation. This maintains the sequence modeling advantages of Transformers while avoiding the efficiency bottleneck of step-by-step decoding.
- **Cross-task knowledge distillation** to prevent catastrophic forgetting: The models from the first two stages act as teachers to continuously supervise the Task-III model, ensuring that short-term and long-term capabilities are not forgotten.
- **Convincing visual validation**: Models with Task-I are more accurate in proximal trajectory segments, while models with Task-II are more accurate in distal trajectory segments; combining both achieves overall optimality.

## Limitations & Future Work

- It only models individual pedestrian trajectories. It does not explicitly model interactions between pedestrians or scene constraints (e.g., obstacles, road boundaries), which may be insufficient in highly crowded scenarios.
- Although training in each of the three sequential stages is relatively fast, it increases the complexity of the training pipeline, requiring careful design of hyperparameters for each stage.
- Employs the Best-of-20 evaluation strategy—which is standard practice but can mask the true quality of the generated distribution.
- The diversity loss for destination prediction is based on a Gaussian RBF kernel; more flexible distribution modeling approaches (e.g., normalizing flows) could be explored.
- The current Transformer is a 3-layer encoder-only model; an encoder-decoder architecture or deeper models could be explored.

## Related Work & Insights

- **Connection to curriculum learning / progressive training**: Similar to ProGAN progressively increasing resolution to train GANs, and PGBIG progressively refining motion prediction, PPT is the first to introduce progressive pretext tasks to the field of trajectory prediction.
- **Comparison with TUTR**: Like PPT, TUTR is a non-autoregressive Transformer, but it does not utilize prompt embeddings or progressive pre-training, resulting in significantly lower performance than PPT.
- **Advantages of the destination-driven strategy**: Explicitly decomposing long-range dependencies into destination prediction + intermediate point generation is more effective than end-to-end prediction of all positions. Furthermore, the shared model architecture between both stages allows knowledge transfer to occur naturally.
- The idea of progressive pretext tasks can be extended to related tasks such as vehicle trajectory prediction and robot path planning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Progressive pretext task training is a first in trajectory prediction
- **Technical Quality**: ⭐⭐⭐⭐ — The three-stage design is well-justified, with knowledge distillation preventing forgetting
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four datasets, detailed ablations, and visual analysis
- **Practicality**: ⭐⭐⭐⭐ — Efficient inference (5.28ms), suitable for real-time applications
- **Overall Recommendation**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Adaptive Human Trajectory Prediction via Latent Corridors](adaptive_human_trajectory_prediction_via_latent_corridors.md)
- [\[CVPR 2026\] Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction](../../CVPR2026/autonomous_driving/recover_to_predict_progressive_retrospective_learning_for_variable-length_trajec.md)
- [\[CVPR 2025\] Certified Human Trajectory Prediction](../../CVPR2025/autonomous_driving/certified_human_trajectory_prediction.md)
- [\[ECCV 2024\] UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction](unitraj_a_unified_framework_for_scalable_vehicle_trajectory_prediction.md)
- [\[ECCV 2024\] VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual Descriptions](visiontrap_vision-augmented_trajectory_prediction_guided_by_textual_descriptions.md)

</div>

<!-- RELATED:END -->
