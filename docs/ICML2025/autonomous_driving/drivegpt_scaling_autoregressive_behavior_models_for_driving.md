---
title: >-
  [Paper Note] DriveGPT: Scaling Autoregressive Behavior Models for Driving
description: >-
  [ICML 2025][Autonomous Driving][Autoregressive Models] Proposes DriveGPT, a 1.4B-parameter autoregressive Transformer driving behavior model trained on 120 million real driving clips (50x larger than the largest existing dataset). It systematically establishes the data/model/compute scaling laws for driving behavior modeling for the first time, demonstrates that data is the primary performance bottleneck, and outperforms the state-of-the-art on planning and WOMD prediction ta…
tags:
  - "ICML 2025"
  - "Autonomous Driving"
  - "Autoregressive Models"
  - "Scaling Laws"
  - "Transformer"
  - "Trajectory Prediction"
date: 2026-05-08
content_hash: 594acb7a6d9ac9bd
---

# DriveGPT: Scaling Autoregressive Behavior Models for Driving

**Conference**: ICML 2025  
**arXiv**: [2412.14415](https://arxiv.org/abs/2412.14415)  
**Code**: None (not open-sourced)  
**Area**: Autonomous Driving / Behavior Modeling  
**Keywords**: Autonomous Driving, Autoregressive Models, Scaling Laws, Transformer, Trajectory Prediction

## TL;DR

Proposes DriveGPT, a 1.4B-parameter autoregressive Transformer driving behavior model trained on 120 million real driving clips (50x larger than the largest existing dataset). It systematically establishes the data/model/compute scaling laws for driving behavior modeling for the first time, demonstrates that data is the primary performance bottleneck, and outperforms the state-of-the-art on planning and WOMD prediction tasks.

## Background & Motivation

**Success of Transformer Scaling**: In domains like NLP (GPT series), speech, and time-series forecasting, continuously increasing model parameters and training data consistently scales performance, a trend precisely characterized by the scaling laws of Kaplan et al. (2020) and Hoffmann et al. (2022).

**Unique Challenges of Driving Behavior Modeling**: Transferring scaling laws to the driving domain faces three primary challenges: (1) inputs involve multi-modality (agent trajectories + map information), unlike plain text; (2) it requires spatial reasoning and physical kinematics understanding; (3) acquiring large-scale driving data is extremely costly. Existing behavior models are constrained by data scale (the largest, GUMP, only contains 2.6M clips / 523M parameters), leaving the scaling potential largely unexplored.

**Core Problem**: Can similar continuous performance gains as in NLP be observed in driving behavior modeling by scale expansion of data (50x) and model size (3x)? Which is the more critical bottleneck, data or model? Core Idea: **Model driving trajectories with an LLM-style autoregressive decoder (treating each action step as a token) to validate scaling laws on an industry-scale dataset**.

## Method

### Overall Architecture

DriveGPT adopts a standard encoder-decoder architecture: a **Transformer Encoder** fuses multimodal scene context (target agent history, surrounding agent history, and map vectors) into a scene embedding $\mathbf{c} \in \mathbb{R}^{n \times d}$; an **LLM-style Transformer Decoder** progressively predicts the discrete action distribution of future positions in an autoregressive manner, conditioned on the encoder embeddings and previously predicted states. During inference, multi-modal predictions are generated via sampling multiple trajectories and applying K-Means sub-sampling.

### Key Designs

1. **Verlet Action Discretization**:
    - **Function**: Converts the continuous trajectory space into a sequence of discrete action tokens.
    - **Mechanism**: Defines the Verlet action $a_t$ as the second-order difference of positions, i.e., $s_{t+1} = s_t + (s_t - s_{t-1}) + a_t$, where the $(s_t - s_{t-1})$ term implies a constant velocity assumption. It discretizes the continuous action space into a finite set, converting trajectory prediction into a classification problem.
    - **Design Motivation**: The Verlet representation naturally encodes acceleration information, resulting in physically smooth trajectories. Discretization enables training with standard cross-entropy loss, aligning perfectly with the LLM paradigm.

2. **Multimodal Scene Encoder**:
    - **Function**: Encodes heterogeneous inputs (agent trajectories + map polylines) into a unified scene embedding.
    - **Mechanism**: All inputs are normalized to a target-agent-centric coordinate system. Each vector is mapped to a token embedding using a PointNet-like encoder, and all context is fused using a self-attention Transformer.
    - **Design Motivation**: Vectorized representations (VectorNet-style) are efficient and naturally integrate with Transformer architectures. An agent-centric view eliminates the impact of absolute coordinates.

3. **Large-Scale Dataset Construction and Scaling Experiment Design**:
    - **Function**: Selects 120 million high-quality clips from millions of miles of real-world driving data, covering multiple cities in the US, Japan, and UAE.
    - **Mechanism**: The dataset is balanced across day/night and geographical regions, covering scenarios such as lane changes, intersections, double-parks, construction zones, and pedestrian/cyclist interactions. Model sizes scale from 1.5M to 1.4B parameters (across 3 orders of magnitude), searching for the optimal learning rate at each scale.
    - **Design Motivation**: Prior works were constrained by small-scale datasets, preventing statistically significant scaling conclusions. This work aims to validate scaling trends across an unprecedented scope.

### Loss & Training

- **Teacher forcing** training: Ground-truth future positions are used as decoder inputs, allowing parallel prediction for all steps.
- **Single cross-entropy loss**: The target action is selected as the discrete action nearest to the ground truth.
- Each model scale is trained for a single epoch (consistent with LLM scaling literature).
- The optimal learning rate decreases as model size increases: 1.5M $\rightarrow$ 0.005, 1.4B $\rightarrow$ 0.0001 (with cosine decay).

## Key Experimental Results

### Main Results: Data Scaling (26M Parameter Model)

| Training Data Size | mADE ↓ | mFDE ↓ | Miss Rate ↓ | Offroad ↓ | Collision ↓ |
|-----------|--------|--------|-------------|-----------|-------------|
| 2.2M (WOMD-level) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 21M | 0.561 | 0.496 | 0.420 | 0.326 | 0.269 |
| 85M | 0.496 | 0.441 | 0.332 | 0.238 | 0.217 |
| 120M | **0.489** | **0.433** | **0.317** | **0.198** | **0.196** |

Data scaling from 2.2M $\rightarrow$ 120M: mFDE decreased by 56.7%, Offroad rate decreased by 80.2%, and Collision rate decreased by 80.4%.

### Model Scaling (120M Data)

| Model Parameters | mADE ↓ | mFDE ↓ | Miss Rate ↓ | Offroad ↓ | Collision ↓ |
|---------|--------|--------|-------------|-----------|-------------|
| 8M | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 26M | 0.954 | 0.950 | 0.902 | 0.858 | 0.915 |
| 94M | 0.937 | 0.925 | 0.866 | 0.815 | 0.890 |
| 163M | 0.943 | 0.925 | 0.875 | 0.815 | **0.817** |

Model scaling yields weaker benefits than data scaling—under 120M data points, performance tends to saturate once the model size reaches ~94M parameters.

### Scaling Laws

| Scaling Dimension | Fitting Formula | $R^2$ |
|---------|---------|-------|
| Data Scaling | $\log(L) = -0.102 \log(D) + 2.663$ | 0.986 |

Prediction: Improving performance by another 10% requires an additional 350M samples, while a 20% improvement requires an additional 1.4B samples.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Autoregressive vs. One-shot Decoder | AR is superior at >8M parameters | minFDE continuously improves vs. saturates |
| Fixed Compute Budget | Small model + more data > Large model + less data | Data is the bottleneck |
| Attention Heads / Hidden Dimension variance | No significant difference | Primarily determined by total parameter count |

### Key Findings

- Data scaling is the primary bottleneck in driving behavior modeling; model scaling yields limited benefits when data is insufficient.
- Model scaling only becomes effective when the data volume exceeds 21M (below 21M, variance in performance across different model sizes is negligible).
- Autoregressive decoders exhibit better scalability than one-shot decoders, allowing larger models to continue benefiting.
- In closed-loop evaluations, DriveGPT trained on massive data handles edge cases like pedestrian crossings and double-parked vehicles effectively.

## Highlights & Insights

- **First industry-level scaling study in driving behavior modeling**: 1.4B parameters / 120M clips, which is 1-2 orders of magnitude larger than existing works.
- Clearly answers the key question of "Data vs. Model": data is the primary bottleneck, consistent with NLP scaling literature.
- High-quality fit for scaling laws ($R^2 = 0.986$) provides quantitative guidance for future resource allocation.
- Closed-loop deployment validates the translation of scaling gains into real-world driving value (e.g., safe lane-changing, complex interactions).

## Limitations & Future Work

- Utilizes only vectorized trajectory and map inputs, lacking integration of raw visual perception data (camera/LiDAR).
- The model is not open-sourced, making scaling experiments hard to reproduce.
- Model scaling tends to saturate after ~94M parameters, indicating that even larger datasets might be required to unlock the potential of larger models.
- Verlet action discretization may introduce quantization errors, limiting fine-grained motion prediction.
- Closed-loop evaluation is conducted in simulation, leaving safety verification on real roads insufficient.
- The generalization of scaling laws is limited in range (only covering up to 120M data points and 1.4B parameters).

## Related Work & Insights

- **Kaplan et al. (2020)**: Pioneering work on NLP scaling laws; DriveGPT directly benchmarks against its methodology.
- **Hoffmann et al. (2022) Chinchilla**: Compute-optimal scaling; DriveGPT employs a similar compute-budget analysis.
- **Seff et al. (2023) MotionLM**: Models motion prediction as language modeling; DriveGPT scales it up significantly.
- **GUMP (523M)**: The previous largest behavior model; DriveGPT is 2.7x larger in parameters and 46x larger in data volume.
- **Insights**: The path for scaling data in driving foundation models has just begun, and scaling multimodal fusion remains a promising avenue.

## Rating

- Novelty: ⭐⭐⭐ Limited architectural novelty (standard encoder-decoder); the core contribution lies in scale and empirical validation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, with scaling experiments across three dimensions, ablation studies, qualitative analysis, and closed-loop evaluation.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with systematic scaling analysis and intuitive graphics.
- Value: ⭐⭐⭐⭐ Provides a valuable reference for scaling research in driving behavior modeling, though industry barriers limit its community impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PAR: Poly-Autoregressive Prediction for Modeling Interactions](../../CVPR2025/autonomous_driving/poly-autoregressive_prediction_for_modeling_interactions.md)
- [\[ICLR 2026\] DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving](../../ICLR2026/autonomous_driving/drivevla-w0_world_models_amplify_data_scaling_law_in_autonomous_driving.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](../../ICCV2025/autonomous_driving/epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[CVPR 2025\] Cubify Anything: Scaling Indoor 3D Object Detection](../../CVPR2025/autonomous_driving/cubify_anything_scaling_indoor_3d_object_detection.md)
- [\[NeurIPS 2025\] Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling](../../NeurIPS2025/autonomous_driving/flow_matching-based_autonomous_driving_planning_with_advanced_interactive_behavi.md)

</div>

<!-- RELATED:END -->
