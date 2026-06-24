---
title: >-
  [Paper Note] MEGA: Masked Generative Autoencoder for Human Mesh Recovery
description: >-
  [CVPR 2025][3D Vision][Human Mesh Recovery] MEGA proposes a human mesh recovery method based on masked generative modeling. By discretizing the human mesh into a token sequence, the model performs image-conditioned generation after self-supervised pre-training. It supports both deterministic single-shot prediction and stochastic multi-output generation modes, achieving SOTA performance in both.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Human Mesh Recovery"
  - "Masked Generative Modeling"
  - "Multi-output Prediction"
  - "Self-supervised Pre-training"
  - "VQ-VAE"
date: 2026-05-08
content_hash: 6a6c7236687319b1
---

# MEGA: Masked Generative Autoencoder for Human Mesh Recovery

**Conference**: CVPR 2025  
**arXiv**: [2405.18839](https://arxiv.org/abs/2405.18839)  
**Code**: [https://g-fiche.github.io/research-pages/mega/](https://g-fiche.github.io/research-pages/mega/) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Human Mesh Recovery, Masked Generative Modeling, Multi-output Prediction, Self-supervised Pre-training, VQ-VAE

## TL;DR
MEGA proposes a human mesh recovery method based on masked generative modeling. By discretizing the human mesh into a token sequence, the model performs image-conditioned generation after self-supervised pre-training. It supports both deterministic single-shot prediction and stochastic multi-output generation modes, achieving SOTA performance in both.

## Background & Motivation

**Background**: 3D Human Mesh Recovery (HMR) from a single RGB image is a classic computer vision task. Current methods are mainly categorized into two groups: (1) single-output regression methods (such as HMR, CLIFF, VQ-HPS, etc.) that directly predict the single most likely mesh; (2) multi-output probabilistic methods (such as ProHMR, Diff-HMR, etc.) that generate multiple candidate meshes to address depth ambiguity.

**Limitations of Prior Work**: HMR is inherently an ill-posed problem, as infinitely many 3D interpretations can map to the same 2D observation, which is particularly severe in occluded scenarios. Single-output methods overlook this ambiguity and tend to predict the mean or most common pose. While multi-output probabilistic methods can produce diverse predictions, they face an accuracy-diversity trade-off—none of the existing multi-output methods can compete with state-of-the-art single-output methods in terms of single-prediction accuracy.

**Key Challenge**: The trade-off between diversity and accuracy: increasing prediction diversity typically comes at the expense of single-prediction accuracy.

**Goal**: Can a unified framework be designed to achieve SOTA single-shot accuracy in deterministic mode, while generating diverse and high-quality multi-output predictions in stochastic mode?

**Key Insight**: Drawing inspiration from the success of masked generative modeling in NLP and image generation, HMR is reformulated as a conditional generation problem of discrete token sequences. The human mesh is discretized via Mesh-VQ-VAE, and then trained using a BERT/MAE-like mask-and-predict strategy.

**Core Idea**: HMR is modeled as an image-conditioned masked token generation task, where 3D human priors are learned via self-supervised pre-training to achieve both high accuracy and diversity under two unified inference modes.

## Method

### Overall Architecture
MEGA is based on an encoder-decoder Transformer architecture. First, a pre-trained Mesh-VQ-VAE is used to encode the human mesh into N=54 discrete tokens (each corresponding to a specific body part, with a codebook size of S=512). Training consists of two stages: (1) Self-supervised pre-training—learning to reconstruct complete mesh tokens from partially visible tokens using motion capture (mocap) data, without image data; (2) Supervised training—adding image embeddings as conditions to train the model to predict the full mesh under randomly masked tokens. During inference, it supports both a deterministic mode (a single forward pass to predict all tokens) and a stochastic mode (iterative sampling).

### Key Designs

1. **Mesh Tokenization and Self-Supervised Pre-Training**:

    - **Function**: Convert continuous 3D mesh representation into discrete tokens, and leverage large-scale mocap data to learn 3D human priors.
    - **Mechanism**: Mesh-VQ-VAE is used to encode the 6890-vertex SMPL mesh into 54 tokens, with each token selected from a codebook of size 512. Following the VQ-MAE approach, the pre-training phase utilizes a variable mask ratio $M = \lfloor N \cos(\pi\tau/2) \rfloor$ (where $\tau \sim U[0,1)$). The encoder processes visible tokens, while the decoder predicts the masked tokens, supervised solely by cross-entropy loss. The model is trained on AMASS mocap data for 500 epochs.
    - **Design Motivation**: (1) Discrete token representations naturally constrain predictions within the valid human pose space, avoiding non-humanoid meshes. (2) Pre-training leverages massive mocap data without image pairs to learn human kinematic priors; ablation studies show this contributes to a 2.5–6.0mm improvement in PVE. (3) A variable mask ratio is crucial for the stochastic mode, as the number of visible tokens varies across each iterative generation step.

2. **Image-Conditioned Masked Generative Training**:

    - **Function**: Learn to predict randomly masked mesh tokens conditioned on image features.
    - **Mechanism**: Building upon the pre-trained model, image features (extracted via HRNet or ViT) are linearly mapped into a D=1024 dimensional embedding sequence. These are concatenated with mesh token embeddings and fed into the decoder. During training, the mesh tokens adopt the same cosine mask ratio schedule as in pre-training, while the image embeddings remain fully visible. Supervision relies solely on cross-entropy loss—which significantly simplifies training compared to traditional methods that rely on multiple losses such as 3D joints, 2D reprojection, and SMPL parameters. Additionally, an MLP predicts global 6D rotations and perspective camera parameters from image features.
    - **Design Motivation**: It is critical to maintain a mask ratio schedule consistent with pre-training (ablation shows that 100% mask training yields slightly degraded performance) because it allows the self-supervised and supervised stages to share the same training distribution.

3. **Deterministic and Stochastic Dual-Mode Inference**:

    - **Function**: Flexibly support high-accuracy single-output or diverse multi-output scenarios.
    - **Mechanism**: **Deterministic mode**—Starting from a fully masked sequence, a single forward pass predicts all 54 tokens (taking the argmax). This mode does not require the encoder and uses only the decoder, significantly reducing the model size ($B_e=12 > B_d=4$). **Stochastic mode**—Iterative generation over T steps. At step t, $n_t - n_{t-1}$ new tokens are predicted, where $n_t = \lfloor N(1-\cos(\pi t / 2T)) \rfloor$. Gumbel-max sampling is used to sample candidate tokens from the predicted distribution, and a fixed number of these candidates are then set to visible. This is repeated Q times to obtain Q diverse predictions.
    - **Design Motivation**: The deterministic mode achieves "the first attempt to discard the encoder and use only the decoder in MAE," whereas previous MAE-based works generally discard the decoder to use the encoder for downstream tasks. The stochastic mode introduces randomness through Gumbel sampling, generating different meshes across runs and naturally modeling the multi-solution nature of HMR.

### Loss & Training
Pre-training phase: Solely cross-entropy loss, AMASS dataset, 500 epochs. HMR training: First on MSCOCO for 100 epochs, and then on a mixed dataset (MSCOCO + Human3.6M + MPI-INF-3DHP + MPII) for 10 epochs. Rotation and camera parameters are supervised using rotation matrix Euclidean distance + 2D joint reprojection L1 loss. Total training takes about 2.5 days on 4 A100 GPUs.

## Key Experimental Results

### Main Results

**Deterministic Mode (3DPW Dataset):**

| Method | Backbone | PVE↓ | MPJPE↓ | PA-MPJPE↓ |
|------|----------|------|--------|-----------|
| CLIFF | HRNet-w48 | 87.6 | 73.9 | 46.4 |
| VQ-HPS | HRNet-w48 | 84.8 | 71.1 | 45.2 |
| **MEGA** | HRNet-w48 | **81.6** | **68.5** | **44.1** |
| HMR2.0 | ViT-H | 84.1 | 70.0 | 44.5 |
| **MEGA** | ViT-H | **80.0** | **67.5** | **41.0** |

**Stochastic Mode (3DPW Dataset, ResNet-50 backbone):**

| Method | PVE (Q=1) | PVE (Q=25) | Gain |
|------|-----------|------------|--------|
| Diff-HMR | 114.6 | 109.8 | 4.2% |
| ProHMR | - | - (84.0 MPJPE) | 13.4% |
| **MEGA** | **101.6** | **87.5** | **13.9%** |
| **MEGA det** | **90.6** | - | - |

### Ablation Study

| Configuration | PVE (3DPW)↓ | PVE (EMDB)↓ | Notes |
|------|-------------|-------------|------|
| MEGA (Full) | 81.6 | 107.9 | Cosine mask ratio scheduling |
| Linear masking | 86.5 | 118.7 | Linear mask ratio, performance degrades |
| Full mask | 81.8 | 110.3 | 100% mask training, slightly degraded |
| w/o pre-training + full mask | 84.1 | 113.9 | Without pre-training, PVE increases by 2.5/6.0mm |

### Key Findings
- Self-supervised pre-training is a key component of MEGA; removing it increases PVE by 2.5mm on 3DPW and 6.0mm on EMDB, demonstrating that human priors from mocap data are crucial for HMR.
- Cosine mask ratio scheduling outperforms linear scheduling, aligning with the MAE finding that high mask ratios favor learning.
- On the occluded dataset 3DPW-OCC, MEGA (HRNet) achieves a PVE of 93.8mm, outperforming all methods specifically designed for occlusion (such as SEFD at 97.1mm). This demonstrates the advantage of self-attention among mesh tokens, where visible parts are used to infer occluded parts.
- In stochastic mode with Q=25, the PVE drops to 87.5mm (vs. 90.6mm in deterministic mode), showing that multi-output sampling can discover better solutions than deterministic predictions.

## Highlights & Insights
- **First time discarding the encoder in MAE**: Traditional MAE-based models discard the decoder and use the encoder for downstream tasks. Differently, MEGA discards the encoder in deterministic mode and uses only the decoder, as fully masked inputs do not require encoder processing. This represents an insightful architectural design choice.
- **Ultra-simple training loss**: Relying solely on cross-entropy loss, the model outperforms traditional HMR methods that use 5–6 different losses, demonstrating that a superior representation space (discrete tokens) can significantly simplify the training objective.
- **Robustness to occlusion**: Token-level mask-and-predict training implicitly equips the model with the ability to "infer the whole from parts," which can be transferred to other tasks handling occlusions.

## Limitations & Future Work
- The performance depends on the reconstruction quality of the pre-trained Mesh-VQ-VAE—quantization errors in the codebook directly propagate as noise to the training objective.
- Stochastic mode requires multiple forward passes (T steps × Q times), which limits real-time application.
- The method is only validated in single-person scenarios; scaling to multi-person scenarios remains an open question.
- Global rotation and camera parameters are still predicted using deterministic regression and have not been integrated into the probabilistic modeling.

## Related Work & Insights
- **vs. VQ-HPS**: Both employ Mesh-VQ-VAE tokenization, but VQ-HPS relies on a deterministic classification mapping. MEGA introduces masked generative modeling, supporting multiple outputs with higher accuracy (PVE 81.6 vs. 84.8).
- **vs. Diff-HMR**: Diff-HMR utilizes diffusion models to generate diverse meshes but struggles with single-shot accuracy (PVE=114.6). MEGA's single-shot prediction (PVE=101.6) already far outperforms Diff-HMR's best results with 25 samples (109.8).
- **vs. HMR2.0/TokenHMR**: These are recent state-of-the-art ViT-backbone methods, and MEGA outperforms them under comparable backbone architectures.

## Rating
- Novelty: ⭐⭐⭐⭐ Masked generative modeling is applied to HMR for the first time, and the dual-mode inference design is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation covering both deterministic and stochastic modes across multiple benchmarks, detailed ablations, and occlusion benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear reasoning with thoroughly explained methodology.
- Value: ⭐⭐⭐⭐ Unifies the single-output and multi-output HMR paradigms, establishing fresh directions for subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PromptHMR: Promptable Human Mesh Recovery](prompthmr_promptable_human_mesh_recovery.md)
- [\[CVPR 2025\] HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery](heatformer_a_neural_optimizer_for_multiview_human_mesh_recovery.md)
- [\[ICCV 2025\] MaskHand: Generative Masked Modeling for Robust Hand Mesh Reconstruction in the Wild](../../ICCV2025/3d_vision/maskhand_generative_masked_modeling_for_robust_hand_mesh_reconstruction_in_the_w.md)
- [\[ICCV 2025\] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](../../ICCV2025/3d_vision/fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)
- [\[ICCV 2025\] AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](../../ICCV2025/3d_vision/ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)

</div>

<!-- RELATED:END -->
