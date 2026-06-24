---
title: >-
  [Paper Note] Multi-modal Knowledge Distillation-based Human Trajectory Forecasting
description: >-
  [CVPR 2025][Model Compression][Pedestrian Trajectory Prediction] This paper proposes the first multi-modal knowledge distillation framework for pedestrian trajectory prediction. A full-modal teacher model is trained using trajectories, human poses, and text descriptions, and its knowledge is distilled into a student model using only trajectories or trajectories and poses. This achieves up to approximately a 13% improvement in forecasting accuracy across three datasets: JRDB…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Pedestrian Trajectory Prediction"
  - "Multi-modal Knowledge Distillation"
  - "Human Pose"
  - "Text Descriptions"
  - "Instantaneous Forecasting"
date: 2026-05-08
content_hash: 89e1933404fcc23c
---

# Multi-modal Knowledge Distillation-based Human Trajectory Forecasting

**Conference**: CVPR 2025  
**arXiv**: [2503.22201](https://arxiv.org/abs/2503.22201)  
**Code**: [https://github.com/Jaewoo97/KDTF](https://github.com/Jaewoo97/KDTF)  
**Area**: Autonomous Driving / Trajectory Prediction  
**Keywords**: Pedestrian Trajectory Prediction, Multi-modal Knowledge Distillation, Human Pose, Text Descriptions, Instantaneous Forecasting

## TL;DR
This paper proposes the first multi-modal knowledge distillation framework for pedestrian trajectory prediction. A full-modal teacher model is trained using trajectories, human poses, and text descriptions, and its knowledge is distilled into a student model using only trajectories or trajectories and poses. This achieves up to approximately a 13% improvement in forecasting accuracy across three datasets: JRDB, SIT, and ETH-UCY.

## Background & Motivation
1. **Background**: Pedestrian trajectory forecasting predicts future motion based on historical 2D trajectory sequences and is widely applied in autonomous driving, mobile robot navigation, and surveillance systems. Mainstream methods such as HiVT (graph attention) and MART (Transformer) model pure trajectory sequences. Recently, some studies have attempted to use visual cues (e.g., human pose, bounding boxes) to enhance prediction.
2. **Limitations of Prior Work**: (a) It is difficult to accurately infer pedestrian motion intent solely from 2D coordinate sequences because pedestrians convey intent through visual signals (e.g., turning, lifting arms); (b) text descriptions are highly effective for modal fusion but expensive to acquire, requiring online VLM generation, which is infeasible for resource-constrained systems; (c) in scenarios with frequent occlusions, instantaneous forecasting with few observation frames is more challenging.
3. **Key Challenge**: Multi-modality (especially text) significantly improves prediction accuracy, but the computational cost of obtaining additional modalities during inference is excessively high. How can the performance gains from expensive modalities be preserved without actually using them during inference?
4. **Goal**: (1) How to transfer high-level semantic motion understanding contained in text to lightweight models? (2) How to separately distill the two layers of knowledge: intra-agent multi-modal fusion and inter-agent interaction? (3) How to effectively forecast under the extreme case of instantaneous observation (only 1-2 frames)?
5. **Key Insight**: The authors find that text descriptions play a crucial "bridging" role in integrating different modalities. Text bridges the domain gap between trajectories and poses, allowing the model to make full use of pose information through semantic context even when the pose is highly noisy. Based on this, a KD framework is designed to let the student model implicitly acquire this language-driven understanding by aligning with the teacher's embedding space during training.
6. **Core Idea**: During training, a full-modal teacher guides a limited-modality student. Multi-modal understanding of motion intent is transferred by separately aligning the intra-agent and inter-agent embedding spaces.

## Method

### Overall Architecture
Two-stage training. First stage: train a full-modal teacher model (trajectory $\mathcal{X}$ + 3D pose $\mathcal{P}$ + text $\mathcal{S}$) using regression loss jointly trained across three observation settings (full/2-frame/1-frame). Second stage: freeze the teacher model and train the student model (using only $\mathcal{X}$ or $\mathcal{X}+\mathcal{P}$) from scratch. In addition to the regression loss, a KD loss is added to align the distributions of the local encoder output $Q$ (intra-agent) and the global encoder output $H$ (inter-agent). The teacher and student share the same network architecture, differing only in the number of input modalities.

### Key Designs

1. **Modality Embedding and Local Encoder (Intra-agent Fusion)**:

    - **Function**: Encode each modality into a unified embedding and fuse them into a single-agent motion intent representation $q_n$.
    - **Mechanism**: Use MLPs to encode trajectory $z_x$ and SMPL pose parameters $z_p$ separately, and use a pre-trained TinyBERT to encode text $z_s$. For HiVT: process frame-by-frame through a graph network, and for each agent, fuse its own modality with neighbor modalities (neighbors' trajectories and poses transformed via rotation-invariant operations), then encode temporal information using a Transformer: $q_n^t = \psi_\mathcal{M}([(z_x,z_p,z_s)_i, (z_x,z_p,z_s)_j, (v_{ji})_e])$. For MART: use a Transformer to apply global attention across both modal and temporal dimensions, and aggregate using a class token: $q_n = \phi_{\mathcal{M},T_p}(\bar{q_n}, z_x, z_p, z_s)$.
    - **Design Motivation**: HiVT's graph structure naturally supports incorporating pose information (rotation-invariant) for each agent-neighbor pair individually, capturing subtle interaction cues more granularly. The SMPL representation generalizes better across datasets than keypoints.

2. **Global Encoder (Inter-agent Interaction Modeling)**:

    - **Function**: Model inter-agent interaction relationships on top of $q_n$ to obtain the complete motion intent representation $H$.
    - **Mechanism**: MART models global interaction using standard Transformer attention: $H = \phi_N(Q)$. HiVT utilizes a graph network and encodes description of relations between agents (e.g., "they are chatting together") in the text as edge attributes: $H = \psi_N([(q_n)_i, (q_n)_j, (v_{ji}, s_{R,ji})])$, where $s_{R,ji}$ is the text embedding describing the relationship between two agents.
    - **Design Motivation**: The JRDB dataset contains manual annotations of relation text between agents. HiVT's graph structure can naturally introduce relationship text into each edge, allowing the global encoder to benefit from textual information as well.

3. **Two-level Knowledge Distillation (KD Loss Design)**:

    - **Function**: Transfer intra-agent multi-modal fusion knowledge and inter-agent interaction knowledge separately.
    - **Mechanism**: Align the distributions of $Q$ and $H$ between the teacher and student using KL divergence. For MART: $\mathcal{L}_{KD} = \mathcal{L}_{KL}(Q_\mathcal{T}\|Q_\mathcal{S}) + \mathcal{L}_{KL}(H_\mathcal{T}\|H_\mathcal{S})$. For HiVT, to ensure stability, use cosine similarity with regularization: $\mathcal{L}_{KD}^L = \lambda_{cos}\mathcal{L}_{cos}(Q_\mathcal{T}, Q_\mathcal{S}) + \mathcal{L}_{KL}(\mathcal{N}\|Q_\mathcal{S})$. The KD losses are calculated independently for each of the three observation settings (full/2-frame/1-frame).
    - **Design Motivation**: Two-level distillation aligns student models with teachers at both individual motion understanding ($Q$) and social interaction understanding ($H$) layers, which is more effective than aligning only the final predictions.

### Loss & Training
Total student loss: $\mathcal{L} = \lambda_{reg}L_{reg}^F + L_{reg}^2 + L_{reg}^1 + \mathcal{L}_{KD}^F + \mathcal{L}_{KD}^2 + \mathcal{L}_{KD}^1$, where $\lambda_{cos}=0.5$ and $\lambda_{reg}=3$. HiVT uses NLL regression loss, while MART uses L2 loss. Joint training on three observation settings enables the model to excel in both full observation and instantaneous prediction. JRDB uses human-annotated text, SIT uses text generated by PLLaVa, and ETH/UCY uses rule-generated map description texts.

## Key Experimental Results

### Main Results

| Dataset | Model | Student Modalities | KD | ADE | ADE₁ | FDE | FDE₁ | Ave. Gain (%) |
|--------|------|---------|-----|-----|------|-----|------|--------|
| JRDB | HiVT | 𝒳 | ✗ | 0.221 | 0.342 | 0.432 | 0.632 | - |
| JRDB | HiVT | 𝒳 | ✓ | **0.220** | **0.326** | **0.438** | **0.604** | +2.38 |
| JRDB | HiVT | 𝒳+𝒫 | ✗ | 0.229 | 0.364 | 0.441 | 0.659 | - |
| JRDB | HiVT | 𝒳+𝒫 | ✓ | **0.232** | **0.308** | **0.445** | **0.560** | +4.98 |
| SIT | HiVT | 𝒳+𝒫 | ✗ | 0.518 | 0.531 | 0.979 | 1.006 | - |
| SIT | HiVT | 𝒳+𝒫 | ✓ | **0.414** | **0.500** | **0.789** | **0.951** | +13.03 |
| JRDB | MART | 𝒳 | ✗ | 0.286 | 0.395 | 0.545 | 0.753 | - |
| JRDB | MART | 𝒳 | ✓ | **0.259** | **0.366** | **0.495** | **0.684** | +7.61 |

Multi-modal teacher model performance (JRDB+MART):

| Modalities | ADE | ADE₁ | Ave. Gain (%) |
|------|-----|------|--------|
| 𝒳 | 0.286 | 0.395 | - |
| 𝒳+𝒫 | 0.287 | 0.366 | +2.02 |
| 𝒳+𝒮 | 0.261 | 0.301 | +12.41 |
| 𝒳+𝒫+𝒮 | **0.258** | **0.289** | **+14.98** |

### Ablation Study

| KD-Local | KD-Global | ADE₁ | FDE₁ | Ave. Gain (%) | Description |
|----------|-----------|------|------|--------|------|
| ✗ | ✗ | 0.364 | 0.659 | - | Baseline w/o KD |
| ✓ | ✗ | 0.352 | 0.647 | +1.5 | Intra-agent distillation only |
| ✗ | ✓ | 0.345 | 0.637 | +2.7 | Inter-agent distillation only |
| ✓ | ✓ | **0.308** | **0.560** | **+4.98** | Two-level distillation |

### Key Findings
- **Text is a crucial bridge for multi-modal fusion**: In HiVT, adding pose only even has a negative impact (-2.84%), but adding text yields a significant improvement (+6.53%), and text+pose further improves it to +8.38%. Text bridges the semantic gap between trajectories and noisy poses.
- **The effect of KD is most significant in instantaneous prediction scenarios**: The improvement in ADE₁ is generally much greater than that in ADE, because additional modalities provide key semantic complements when observation is insufficient.
- **KD achieves the strongest boost (+13%) on the SIT dataset (which is relatively small)**: On small-scale datasets, base models struggle to build multi-modal associations, and KD helps transfer the understanding obtained from large-scale pre-training.
- **Even when the student model uses only trajectory (the simplest configuration), KD still brings stable performance gains**: This indicates that the potential of numerical trajectories can be unlocked by semantic contextual knowledge.

## Highlights & Insights
- **The insight of text as a modality bridge** is the most valuable finding of this paper: text not only provides semantic information itself but, more importantly, allows the model to correctly understand and utilize noisy pose signals. This insight is transferable to any multi-modal fusion problem—adding a semantic modality can be more effective than adding more similar modalities.
- **The two-level distillation design** (intra+inter) outperforms single-level: decomposing trajectory prediction into "understanding individual motion intent" and "modeling social interaction" to align them separately is more precise than directly aligning the final outputs.
- **Joint training of the three observation settings** (full/2-frame/1-frame) enables a single model to excel in both standard and instantaneous predictions simultaneously without requiring separate models.

## Limitations & Future Work
- 3D pose extraction relies on the quality of external models; heavy noise can lead to negative impacts (e.g., in HiVT, $\mathcal{X}+\mathcal{P}$ performs 2.84% worse than pure $\mathcal{X}$).
- It is impossible to extract effective poses and text under the BEV perspective of ETH/UCY (VLMs perform poorly on BEV perspectives). Instead, CLIP image features and rule-based texts are used, yielding limited effectiveness (+1.55~3.80%).
- The quality of texts generated by VLMs directly impacts the upper bound of the teacher model. Future work can explore stronger VLMs or multi-turn dialogues to extract more accurate behavior descriptions.
- The teacher and student share the same architecture (differing only in input modalities). Exploring smaller student networks for further acceleration is a potential direction.
- Semantic scenes (map information, obstacles) are not considered. This work can be extended to support richer multi-modal scenarios with trajectory+map+pose+text.

## Related Work & Insights
- **vs SocialTransmotion**: ST also utilizes human pose to enhance trajectory forecasting but directly concatenates the inputs; this paper enables the student to bypass poses during inference via a KD framework, providing greater deployment flexibility.
- **vs LLM-based trajectory forecasting (e.g., LCF/DriveGPT)**: These methods require LLM inference online, incurring extreme computational costs. This paper completely removes the dependency on text after training through KD.
- The proposed framework can be adapted to any regression-focused trajectory forecasting model (HiVT/MART are only examples), demonstrating high generalizability.

## Rating
- Novelty: ⭐⭐⭐⭐ The first to introduce multi-modal KD into trajectory forecasting; the insight on text bridging modal fusion is deep and profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets $\times$ 2 models $\times$ multiple observation settings $\times$ multiple modality combinations, covering both egocentric and BEV views, extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clearly described method and a complete logical chain of motivation.
- Value: ⭐⭐⭐⭐ High generalizability of the KD framework; the discovery of text-modality bridging is inspiring for multi-modal learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting](../../ICCV2025/model_compression/frequency-aligned_knowledge_distillation_for_lightweight_spatiotemporal_forecast.md)
- [\[ACL 2026\] MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation](../../ACL2026/model_compression/mta_multi-granular_trajectory_alignment_for_large_language_model_distillation.md)
- [\[CVPR 2025\] Sketch Down the FLOPs: Towards Efficient Networks for Human Sketch](sketch_down_the_flops_towards_efficient_networks_for_human_sketch.md)
- [\[CVPR 2025\] What Makes a Good Dataset for Knowledge Distillation?](what_makes_a_good_dataset_for_knowledge_distillation.md)
- [\[ICML 2025\] A Cross Modal Knowledge Distillation & Data Augmentation Recipe for Improving Transcriptomics Representations through Morphological Features](../../ICML2025/model_compression/a_cross_modal_knowledge_distillation_data_augmentation_recipe_for_improving_tran.md)

</div>

<!-- RELATED:END -->
