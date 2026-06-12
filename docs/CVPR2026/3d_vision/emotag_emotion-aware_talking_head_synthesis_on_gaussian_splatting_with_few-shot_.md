---
title: >-
  [Paper Note] EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] This paper proposes EmoTaG, an emotion-aware 3D talking head synthesis framework built upon FLAME-Gaussian structural priors and a Gated Residual Motion Network (GRMN). It ac…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Talking Head"
  - "Emotion-Aware"
  - "Few-Shot"
  - "FLAME"
date: 2026-05-08
content_hash: d41efa7f5d3e4d0b
---

# EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization

**Conference**: CVPR 2026
**arXiv**: [2603.21332](https://arxiv.org/abs/2603.21332)  
**Code**: Available (Project page)  
**Area**: 3D Vision / Talking Head Synthesis
**Keywords**: 3D Gaussian Splatting, Talking Head, Emotion-Aware, Few-Shot, FLAME

## TL;DR

This paper proposes EmoTaG, an emotion-aware 3D talking head synthesis framework built upon FLAME-Gaussian structural priors and a Gated Residual Motion Network (GRMN). It achieves few-shot personalization from as little as 5 seconds of video while jointly addressing emotion expressiveness, lip-audio synchronization, and geometric stability.

## Background & Motivation

Audio-driven 3D talking head synthesis has advanced significantly with the development of NeRF and 3D Gaussian Splatting (3DGS). The prevailing Pretrain-and-Adapt (PAA) paradigm enables identity adaptation from a few seconds of video, yet two fundamental issues remain:

**Lack of explicit emotion modeling**: Existing few-shot methods (e.g., InsTaG, FIAG) target neutral speech and fail to capture emotion-driven facial motion. The authors demonstrate through visualization experiments (Fig. 2) that lip motion complexity under emotional audio is substantially higher than under neutral audio (standard deviations: 7.88 vs. 3.11).

**Geometric instability**: Unconstrained deformation directly on 3DGS leads to geometric distortion under intense expressions, a problem particularly pronounced in emotional synthesis scenarios.

Core Problem: **Can few-shot 3D talking head synthesis go beyond neutral speech to support emotion-aware facial animation?**

## Method

### Overall Architecture

EmoTaG consists of two core components (Fig. 3):
- **FLAME-Gaussian Model**: Provides robust structured 3D priors
- **Gated Residual Motion Network (GRMN)**: Comprises an Identity-Conditioned Encoder and an Expert Motion Decoder for predicting dynamic facial motion

Training proceeds in two stages: pretraining (learning general motion priors from multi-identity data) and adaptation (freezing GRMN and fine-tuning only AdaIN parameters for new identity personalization).

### Key Designs

1. **Motion Prediction in FLAME Parameter Space**:

    - Function: Transforms motion prediction from direct 3D Gaussian deformation to predicting FLAME expression parameters $\Psi$ and jaw pose $\Theta_\text{jaw}$
    - Mechanism: Drives 3D Gaussians bound to the FLAME mesh via mesh deformation, preserving geometric consistency
    - Design Motivation: FLAME serves as an explicit geometric prior that enforces facial topology constraints, resolving the geometric instability caused by directly deforming 3DGS

2. **Identity-Conditioned Encoder**:

    - Function: Fuses audio features $A$, expression features $E$, and identity features $s$
    - Mechanism: Extracts speech embeddings via Wav2Vec 2.0, enhanced with temporal and prosodic information through 1D CNN + Transformer; extracts AU parameters via OpenFace as upper-face supplements; extracts identity descriptors from neutral frames via AdaFace
    - Design Motivation: Identity features are injected into audio and expression streams via AdaIN modulation, enabling personalized motion prediction while requiring only AdaIN parameter fine-tuning during adaptation

3. **Expert Motion Decoder**:

    - **Base Branch**: Models identity-agnostic audio-to-motion mapping (neutral speech articulation), outputting base deformation parameters
    - **Residual Branch**: Captures emotion-driven deviations, generating motion residuals via an EMO Encoder-Decoder
    - **Gate Branch**: Predicts a scalar gate $g \in [0,1]$ for adaptive fusion of neutral and emotional motion: $\delta = \delta_b + g \cdot \delta_r$
    - Design Motivation: Emotional intensity varies frame by frame; the gating mechanism prevents over-exaggeration while maintaining stability on emotion-free frames

4. **Semantic Emotion Guidance (SEG)**:

    - Function: Distills emotion information from a pretrained DeepFace emotion recognizer
    - Mechanism: Provides two supervision signals — a seven-class emotion distribution $p_\text{emo}$ (supervising the residual branch) and an emotion intensity scalar $e = 1 - p(\text{neutral})$ (supervising the gate branch)
    - Design Motivation: Avoids the coarseness and subjectivity of manual emotion annotation; achieves fine-grained emotion-aware motion learning through knowledge distillation

5. **Intra-Oral Gaussian Refinement**:

    - Selects a subset of Gaussians $G_\text{mouth}$ in the oral region via lip landmarks
    - Applies network-predicted residual offsets $(\Delta\mu, \Delta r, \Delta s)$ to enhance fine-grained motions such as teeth and tongue articulation, compensating for FLAME's limited intra-oral modeling capacity

### Loss & Training

The total loss comprises four terms: $\mathcal{L} = \mathcal{L}_\text{Render} + \mathcal{L}_\text{KL} + \mathcal{L}_\text{Score} + \mathcal{L}_\text{Geo}$

| Loss | Formula / Description | Role |
|------|----------------------|------|
| $\mathcal{L}_\text{Render}$ | L1 + $\lambda \cdot (1 - \text{SSIM})$ | Pixel-level and perceptual structural fidelity |
| $\mathcal{L}_\text{KL}$ | $\text{KL}(p_\text{emo} \| \text{Softmax}(z_e))$ | Aligns residual branch with teacher emotion distribution |
| $\mathcal{L}_\text{Score}$ | $\|g_\text{pred} - e\|$ | Regresses gate branch to emotion intensity scalar |
| $\mathcal{L}_\text{Geo}$ | $\mathcal{L}_D(D, D_\text{GT}) + \mathcal{L}_N(N, N_\text{GT})$ | Depth/normal geometric constraints (adaptation stage only) |

- Pretraining: 250K iterations, lr = 5e-3
- Adaptation: 20K iterations, lr = 5e-4, fine-tuning AdaIN parameters only
- Inference: audio encoding ~25ms, GRMN ~6ms/frame, 3DGS rendering ~7ms/frame

## Key Experimental Results

### Main Results (Self-Reconstruction + 5s Training Data)

| Method | PSNR↑ | LPIPS↓ | LMD↓ | AUE↓ | Sync-C↑ | Training Time | FPS |
|--------|-------|--------|------|------|---------|--------------|-----|
| ER-NeRF | 28.21 | 0.038 | 3.549 | 1.314/0.466 | 3.142 | 2h | 33.2 |
| TalkingGaussian | 28.43 | 0.034 | 3.582 | 1.167/0.401 | 3.631 | 27min | 118.4 |
| InsTaG | 28.92 | 0.029 | 3.145 | 0.921/0.407 | 5.329 | 13min | 82.5 |
| MimicTalk | 25.26 | 0.071 | 3.478 | 0.964/0.781 | 6.341 | 17min | 8.6 |
| **EmoTaG** | **30.02** | **0.019** | **2.221** | **0.685/0.210** | 6.212 | **11min** | 76.4 |

### Ablation Study (Emotion Test Set)

| Variant | PSNR↑ | LPIPS↓ | LMD↓ | Sync-C↑ |
|---------|-------|--------|------|---------|
| Full Model | 29.95 | 0.022 | 2.456 | 6.147 |
| w/o Score Distill | 29.52 | 0.026 | 2.731 | 5.874 |
| w/o KL Distill | 29.36 | 0.031 | 2.985 | 5.712 |
| w/o SEG | 29.01 | 0.034 | 3.067 | 5.541 |
| w/o Gate Branch | 28.77 | 0.036 | 3.358 | 5.004 |
| w/o Residual Branch | 28.52 | 0.038 | 3.572 | 4.896 |
| w/o AdaIN | 28.38 | 0.040 | 4.021 | 4.621 |

### Key Findings

1. **AdaIN identity modulation is the most critical component**: Its removal causes LMD to surge from 2.456 to 4.021, underscoring the importance of identity disentanglement in multi-identity learning
2. **Strong generalization across emotion intensity levels**: Adapting at Level-2 and testing at Level-1/Level-3 yields top performance, with larger advantages at higher intensity levels
3. **Out-of-distribution generalization across identities and languages**: EmoTaG leads on both cross-identity (Sync-E: 9.133) and cross-language (Sync-E: 9.662) evaluations
4. **User study**: EmoTaG achieves the highest scores across emotion expressiveness (4.50), lip-audio synchronization (4.70), and visual realism (4.60)

## Highlights & Insights

- The combination of **structured representation and emotion disentanglement** is particularly elegant: motion prediction in FLAME parameter space ensures geometric stability, while the Base/Residual/Gate three-branch design disentangles neutral articulation from emotional variation
- **Knowledge distillation as a substitute for manual annotation**: Dual-level supervision from DeepFace — distribution-level (KL) and scalar-level (gate score) — avoids the coarseness inherent in discrete emotion labels
- **Minimal adaptation strategy**: Freezing the main network and fine-tuning only AdaIN parameters balances efficiency with personalization quality
- Isolating the intra-oral region for dedicated refinement compensates for FLAME's inherent limitation in modeling oral interior geometry

## Limitations & Future Work

1. Relies on external pose and expression frames as auxiliary inputs at inference time, precluding purely audio-driven generation
2. Emotion distillation depends on DeepFace as a specific teacher model, whose recognition accuracy directly affects training quality
3. Training data covers only 70 identities; generalization to diverse ethnicities and extreme expressions remains to be validated
4. Smooth control over multi-emotion blending or emotion transitions has not been explored

## Related Work & Insights

- **InsTaG** (CVPR 2025) pioneers the few-shot PAA paradigm but lacks emotion modeling
- **EMOTE / EmoVOCA** train FLAME with manually annotated emotion labels, constrained by label granularity
- **EmoTalk3D** achieves emotion synthesis on 3DGS but requires person-specific optimization
- Insight: The gated residual mechanism generalizes naturally to other generative tasks requiring "base + variation" disentanglement

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of FLAME structural priors, gated residual emotion disentanglement, and teacher distillation is elegant and well-motivated
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five evaluation settings (neutral, emotion, cross-intensity, cross-identity, cross-language), comprehensive ablations, and user study
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with coherent motivation–method–experiment flow and rich visualizations
- **Value**: ⭐⭐⭐⭐ — Fills a notable gap in the emotion dimension of few-shot 3D talking head synthesis with strong engineering deployment potential

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FastGHA: Generalized Few-Shot 3D Gaussian Head Avatars with Real-Time Animation](../../ICLR2026/3d_vision/fastgha_generalized_few-shot_3d_gaussian_head_avatars_with_real-time_animation.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](../../ICCV2025/3d_vision/self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[CVPR 2026\] PhysHead: Simulation-Ready Gaussian Head Avatars](physhead_simulation-ready_gaussian_head_avatars.md)
- [\[CVPR 2026\] SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](scope_scenecontextualized_incremental_fewshot_3d_s.md)
- [\[CVPR 2026\] MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second](movies_motion-aware_4d_dynamic_view_synthesis_in_one_second.md)

</div>

<!-- RELATED:END -->
