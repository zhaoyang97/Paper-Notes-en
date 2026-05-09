---
title: >-
  [Paper Note] Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction
description: >-
  [CVPR2026][Autonomous Driving][trajectory prediction] This paper proposes the Progressive Retrospective Framework (PRF), which employs cascaded retrospective units to progressively align features from incomplete observations to those of complete observations, substantially improving variable-length trajectory prediction performance in a plug-and-play manner compatible with existing methods.
tags:
  - CVPR2026
  - Autonomous Driving
  - trajectory prediction
  - variable-length observation
  - progressive retrospection
  - knowledge distillation
date: 2026-05-08
content_hash: 79428385997371c2
---

# Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction

**Conference**: CVPR2026
**arXiv**: [2603.10597](https://arxiv.org/abs/2603.10597)
**Code**: [zhouhao94/PRF](https://github.com/zhouhao94/PRF)
**Area**: Autonomous Driving
**Keywords**: trajectory prediction, variable-length observation, progressive retrospection, knowledge distillation, autonomous driving

## TL;DR

This paper proposes the Progressive Retrospective Framework (PRF), which employs cascaded retrospective units to progressively align features from incomplete observations to those of complete observations, substantially improving variable-length trajectory prediction performance in a plug-and-play manner compatible with existing methods.

## Background & Motivation

1. **Trajectory prediction is a core task in autonomous driving**: Accurately forecasting the future motion of dynamic traffic participants is critical for safe planning and collision avoidance.
2. **Existing methods rely on fixed-length observations**: The vast majority of methods are optimized for standard-length inputs (e.g., 50 or 20 steps) and are highly sensitive to variation in observation length.
3. **Incomplete observations are pervasive in real-world scenarios**: Vehicles newly entering the perception range, re-detected after occlusion, or recovered from tracking loss all produce variable-length or incomplete trajectories.
4. **Performance degrades sharply as observations shorten**: The SOTA method DeMo sees mADE6 deteriorate from 0.658 to 0.861 on Argoverse 2 when observation length drops to 10 steps—a substantial regression.
5. **One-step mapping strategies struggle with short trajectories**: Existing approaches (DTO, FLN, LaKD, CLLS) directly map incomplete features to complete features, performing poorly when the information gap is large.
6. **Independent training (IT) offers diminishing returns at high cost**: Training a separate model for each observation length yields marginal improvements but incurs enormous computational and storage overhead.

## Method

### Overall Architecture: Progressive Retrospective Framework (PRF)

PRF inserts $\tau$ cascaded retrospective units between the encoder and decoder. Each unit $\Phi^v$ is responsible for retrospecting features of an incomplete observation of length $T_v$ toward features corresponding to length $T_{v-1}$ (recovering an additional $\Delta T$ steps). At inference, an input of length $T_v$ passes sequentially through $\Phi^v, \Phi^{v-1}, \dots, \Phi^1$, progressively recovering to the standard length $T_0$, before being fed into the shared decoder for prediction.

- **Plug-and-play**: PRF operates between the encoder and decoder and is directly compatible with existing prediction methods (QCNet, DeMo).
- **Shared encoder**: A single encoder extracts features for all variable-length observations, avoiding the need to maintain multiple models.
- **Progressive alignment**: Each unit need only bridge a small temporal gap $\Delta T$, reducing learning difficulty.

### Retrospective Distillation Module (RDM)

RDM adopts a **residual distillation strategy** that models missing time-step features as learnable residuals, avoiding feature conflicts induced by the shared encoder:

1. **Scene context injection**: Agent features are fused with HD Map features $\mathbf{F}_m$ via cross-attention.
2. **Dual-branch structure**:
   - **Logit branch**: self-attention → MLP → Sigmoid, producing an element-wise gating vector $\mathbf{g}^v$.
   - **Residual branch**: self-attention → MLP → ReLU, learning a residual feature $\mathbf{F}_r^v$.
3. **Gated fusion**: $\tilde{\mathbf{F}}^{v-1} = \mathbf{g}^v \odot \mathbf{F}^v + \mathbf{F}_r^v$, retaining reliable components while supplementing missing information.

### Retrospective Prediction Module (RPM)

RPM recovers the missing $\Delta T$ historical time steps from the distilled features, employing a **decoupled query strategy** for coarse-to-fine retrospection:

1. **Anchor-Free Mode Queries**: $K$ mode queries are initialized via MLP → cross-attention extracts scene features → self-attention models inter-mode interactions → multimodal coarse trajectory proposals are predicted.
2. **Anchor-Based State Queries**: $\Delta T$ state queries are initialized via MLP → cross-attention + **Mamba** models temporal dynamics → fine-grained refinement anchored on the coarse proposals.
3. **Cross-unit sharing**: All retrospective units share a single RPM (since each recovers a fixed $\Delta T$ steps); batched processing accelerates training.
4. **Training-only module**: RPM provides implicit supervision for RDM and is disabled at inference, **adding no inference overhead**.

### Rolling-Start Training Strategy (RSTS)

Exploiting PRF's natural support for short-trajectory training, RSTS generates multiple training samples from a single sequence:

- In addition to the standard sample $([1,50], [51,110])$, samples $([1,40],[41,100])$, $([1,30],[31,90])$, and $([1,20],[21,80])$ are also generated.
- Each retrospective unit receives a number of training samples inversely proportional to its input length—shorter observations are harder to retrospect and thus receive more training data.
- On Argoverse 2, a single sequence yields 4 decoder training samples and $\{4,3,2,1\}$ samples for each retrospective unit.

### Loss & Training

End-to-end training comprises three components:

- **Decoder loss**: Smooth-L1 (trajectory regression) + cross-entropy (mode probability classification), following the QCNet/DeMo setup.
- **RPM loss**: $\mathcal{L}_{rpm} = \frac{1}{\tau}\sum_{v=1}^{\tau}(\mathcal{L}_{mq}^v + \mathcal{L}_{sq}^v)$, supervising mode queries and state queries respectively.
- **RDM loss**: $\mathcal{L}_{rdm} = \frac{1}{\tau}\sum_{v=1}^{\tau}\text{SmoothL1}(\tilde{\mathbf{F}}^{v-1}, \mathbf{F}^{v-1})$

## Key Experimental Results

### Variable-Length Trajectory Prediction (Argoverse 2 Validation Set, mADE6/mFDE6)

| Method | Obs=10 | Obs=20 | Obs=30 | Obs=40 | Obs=50 | Avg-Δ50 |
|--------|--------|--------|--------|--------|--------|---------|
| DeMo-Ori | 0.861/1.533 | 0.700/1.358 | 0.671/1.306 | 0.662/1.288 | 0.658/1.278 | 0.066/0.093 |
| DeMo-CLLS | 0.641/1.258 | 0.630/1.249 | 0.623/1.234 | 0.614/1.225 | 0.615/1.223 | 0.012/0.019 |
| **DeMo-PRF** | **0.617/1.183** | **0.603/1.155** | **0.598/1.143** | **0.599/1.145** | **0.596/1.142** | **0.008/0.015** |
| QCNet-CLLS | 0.735/1.247 | 0.727/1.232 | 0.725/1.227 | 0.719/1.222 | 0.714/1.215 | 0.013/0.017 |
| **QCNet-PRF** | **0.727/1.213** | **0.711/1.181** | **0.706/1.169** | **0.702/1.164** | **0.702/1.166** | **0.010/0.016** |

### Standard Prediction — Argoverse 2 Leaderboard (b-mFDE6)

| Method | b-mFDE6 | mADE6 | mFDE6 | MR6 |
|--------|---------|-------|-------|-----|
| DeMo+ReMo | 1.84 | 0.61 | 1.17 | 0.13 |
| **DeMo-PRF** | **1.81** | **0.60** | **1.14** | **0.13** |

### Ablation Study (Argoverse 2 Validation Set, DeMo backbone)

| RDM | RPM | RSTS | Obs=10 | Obs=50 |
|-----|-----|------|--------|--------|
| ✗ | ✗ | ✗ | 0.876/1.455 | 0.725/1.256 |
| ✓ | ✗ | ✗ | 0.655/1.257 | 0.639/1.231 |
| ✓ | ✓ | ✗ | 0.652/1.241 | 0.635/1.208 |
| ✓ | ✓ | ✓ | **0.617/1.183** | **0.596/1.142** |

- RDM contributes the most: at Obs=10, mADE6 drops from 0.876 to 0.655 (↓25.2%).
- RPM further reduces mFDE6 by approximately 1.3% on top of RDM.
- RSTS improves performance across all observation lengths; at Obs=10, mADE6 decreases by an additional 5.3%.
- Progressive distillation vs. direct distillation: mADE6 of 0.652 vs. 0.663 at Obs=10, with the advantage more pronounced for shorter sequences.
- Mamba vs. GRU vs. Attention (RPM temporal modeling): Mamba achieves the best mFDE6 across all settings.
- Inference overhead: each additional retrospective stage adds only approximately 0.07G FLOPs and 0.03s latency.

## Highlights & Insights

1. **Progressive retrospection is both simple and effective**: Decomposing "long-range feature alignment" into multiple "short-range alignment" steps substantially reduces learning difficulty, as clearly verified by t-SNE visualization.
2. **Plug-and-play design**: Inserted between encoder and decoder, PRF successfully adapts to two SOTA methods, QCNet and DeMo.
3. **RPM is training-only**: It provides implicit supervision without adding inference overhead, making it engineering-friendly.
4. **RSTS data augmentation is elegantly designed**: Leveraging the variable-length property to generate multiple samples from a single sequence, with shorter trajectories receiving more training data.
5. **SOTA on both standard and variable-length tracks**: PRF leads comprehensively on variable-length prediction and simultaneously sets a new record on the standard Argoverse 2 leaderboard.

## Limitations & Future Work

1. **Discrete observation lengths**: Only observation lengths that are integer multiples of $\Delta T$ are supported; intermediate lengths must be truncated to the nearest valid value (e.g., 32→30), potentially wasting information.
2. **Inference latency scales linearly with missing steps**: The shortest observation must pass through all $\tau$ retrospective units; at 10 steps, inference time is 1.9× that of the standard setting (0.268s vs. 0.140s).
3. **Validation limited to two backbones**: Although plug-and-play compatibility is claimed, the method is only validated on QCNet and DeMo; diffusion-based and GPT-based predictors remain untested.
4. **Extremely short observations unexplored**: The behavior under only 1–5 observation steps is not investigated.
5. **Training cost not thoroughly compared**: RSTS generates several times more samples, yet total training time (8×RTX4090, 60 epochs) is not compared against baselines.
6. **No real-world deployment validation**: All experiments are conducted on offline datasets; online or on-vehicle deployment results are not presented.

## Related Work & Insights

| Method | Strategy | Short-Trajectory Performance | Inference Overhead | Compatibility |
|--------|----------|-----------------------------|--------------------|---------------|
| DTO | Teacher–student distillation | Moderate | None | Medium |
| FLN | Temporal-invariant representation | Moderate | None | Medium |
| LaKD | Length-agnostic distillation | Good | None | Medium |
| CLLS | Contrastive learning | Good | None | Medium |
| **PRF** | **Progressive retrospective distillation** | **Best** | **Marginal increase** | **High (plug-and-play)** |

Core distinction: the above methods all perform one-step mapping from short features to long features, whereas PRF progressively aligns features through cascaded units—yielding greater advantages as the information gap widens.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of progressive retrospection, residual distillation, and decoupled queries is original; the approach is conceptually clean and elegant.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two datasets, two backbones, six baselines, detailed ablations, t-SNE visualization, and efficiency analysis.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, well-formatted equations, and rich figures and tables.)
- Value: ⭐⭐⭐⭐ (Variable-length observation is a key pain point in real-world driving; the method is practical and achieves SOTA.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Den-TP: A Density-Balanced Data Curation and Evaluation Framework for Trajectory Prediction](den_tp_a_density_balanced_data_curation_and_evaluation_framework_for_trajectory.md)
- [\[CVPR 2026\] MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating](metadat_generalizable_trajectory_prediction_via_me.md)
- [\[ICCV 2025\] DONUT: A Decoder-Only Model for Trajectory Prediction](../../ICCV2025/autonomous_driving/donut_a_decoder-only_model_for_trajectory_prediction.md)
- [\[CVPR 2026\] FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier–State Space Integration](foss_modeling_long_range_dependencies_and_multimodal_uncertainty_in_trajectory_p.md)
- [\[CVPR 2026\] MindDriver: Introducing Progressive Multimodal Reasoning for Autonomous Driving](minddriver_introducing_progressive_multimodal_reasoning_for_autonomous_driving.md)

</div>

<!-- RELATED:END -->
