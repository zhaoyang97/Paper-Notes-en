---
title: >-
  [Paper Note] Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion
description: >-
  [ECCV 2024][Image Generation] MVSD is proposed, a mutual learning framework based on diffusion models that jointly trains visual acoustic matching (VAM) and dereverberation as symmetric mutual-inverse tasks. This framework leverages their reciprocal relationship to overcome paired data scarcity, marking the first application of diffusion models to visually-guided reverberation style transfer.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 1f11f4d8dc1e1e0c
---

# Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion

**Conference**: ECCV 2024  
**arXiv**: [2407.10373](https://arxiv.org/abs/2407.10373)  
**Area**: Image Generation  

## TL;DR

MVSD is proposed, a mutual learning framework based on diffusion models that jointly trains visual acoustic matching (VAM) and dereverberation as symmetric mutual-inverse tasks. This framework leverages their reciprocal relationship to overcome paired data scarcity, marking the first application of diffusion models to visually-guided reverberation style transfer.

## Background & Motivation

- **Importance of Visual Acoustic Matching (VAM)**: In augmented/virtual reality, realistic reverberant audio is crucial for immersive experiences, requiring the transformation of clean audio to its corresponding reverberant version based on visual scenes.
- **Value of Dereverberation**: Reverberation degrades speech intelligibility and ASR system accuracy. Dereverberation is highly valuable for applications such as teleconferencing, hearing aids, and voice assistants.
- **Limitations of Prior Work**:
    - VAM and dereverberation are studied independently, ignoring the inherent reciprocal relationship between them.
    - Training requires large amounts of paired data (clean and reverberant audio), which is difficult to acquire in practice.
    - GAN-based conditional generation suffers from training instability and over-smoothing issues.
- **Key Insight**: Dereverberation is the inverse task of VAM. The two can serve as reciprocal evaluators to provide feedback signals, thereby reducing dependency on paired data.

## Method

### Overall Architecture

MVSD consists of two converters based on visual scene-driven diffusion models:
- **Reverberator** $f_\theta$: Transforms clean audio $\mathbf{a}_c$ into reverberant audio $\hat{\mathbf{a}}_r$ conditioned on the visual scene $\mathbf{v}$.
- **Dereverberator** $g_\phi$: Restores reverberant audio $\mathbf{a}_r$ to clean audio $\hat{\mathbf{a}}_c$ conditioned on the visual scene.

The two form a closed loop, where the output of one serves as the input to the other, providing mutual feedback signals.

### Key Designs

**Mutual Learning Mechanism**:
- Forward Loop (VAM): $\mathbf{a}_c \xrightarrow{f_\theta} \hat{\mathbf{a}}_r \xrightarrow{g_\phi} \tilde{\mathbf{a}}_c$, computing the error between $\tilde{\mathbf{a}}_c$ and $\mathbf{a}_c$ to provide feedback to $f_\theta$.
- Backward Loop (Dereverberation): $\mathbf{a}_r \xrightarrow{g_\phi} \hat{\mathbf{a}}_c \xrightarrow{f_\theta} \tilde{\mathbf{a}}_r$, computing the error between $\tilde{\mathbf{a}}_r$ and $\mathbf{a}_r$ to provide feedback to $g_\phi$.
- **Key Advantage**: This cycle-consistency error does not require alignment between $\mathbf{a}_c$ and $\mathbf{a}_r$, enabling the utilization of unpaired data.

**Unpaired Data Utilization**:
- Unpaired natural reverberant audio $\mathcal{U} = \{(\mathbf{v}', \mathbf{a}_r')\}$: Dereverbed first and then reconstructed to compute the cycle-consistency error.
- Unpaired clean audio $\mathcal{C} = \{\mathbf{a}_c''\}$: Reverberated under a randomly sampled visual scene and then dereverbed to compute the cycle-consistency error.

**Visual Scene-driven Diffusion (VSD)**:
- **Visual Scene Encoder**: ResNet-18 extracts 256-dimensional visual embeddings as conditional control.
- **Controllable UNet**: Symmetric encoder-decoder design featuring 3 attention blocks each.
    - Self-attention blocks utilize dilated convolutions with stride=4 to rapidly reduce feature map dimensions.
    - **Selective Cross-modal Attention**: Cross-modal attention (visual $\leftrightarrow$ audio) is used only in the 3rd encoder block and the 1st decoder block to reduce computational overhead.
    - Source spectrograms are concatenated with noise as content input to preserve linguistic information.

**Training Strategy**:
- Supervised training is first conducted on paired data to establish a baseline.
- Unpaired data is gradually introduced after training exceeds 100 epochs.
- Predictions of both converters are calculated in each mini-batch, and parameters are updated based on feedback from the symmetric model.

### Loss & Training

The total loss consists of three components:

$$\mathcal{L}_{total} = \mathcal{L}_d + \mathcal{L}_m + \mathcal{L}_{sty}$$

- $\mathcal{L}_d$: Noise prediction loss for the diffusion model.
- $\mathcal{L}_m$: Mutual learning cycle-consistency loss (covering both paired and unpaired data), using L1 distance.
- $\mathcal{L}_{sty}$: Style loss, directly constraining the consistency between the predicted spectrogram and the target spectrogram using L1 distance.

## Key Experimental Results

### Main Results

**VAM Task Comparison (SoundSpaces-Speech)**:

| Method | STFT↓ (Seen) | RTE(s)↓ (Seen) | MOSE↓ (Seen) | STFT↓ (Unseen) | RTE(s)↓ (Unseen) |
|------|-------------|----------------|-------------|----------------|-----------------|
| AV U-Net | 0.638 | 0.095 | 0.353 | 0.658 | 0.118 |
| AViTAR | 0.665 | 0.034 | 0.161 | 0.822 | 0.062 |
| MVSD w/o unpaired | 0.573 | 0.033 | 0.148 | 0.736 | 0.055 |
| **MVSD** | **0.508** | **0.030** | **0.142** | **0.637** | **0.051** |

**Dereverberation Task Comparison (SoundSpaces-Speech)**:

| Method | PESQ↑ | WER(%)↓ | EER(%)↓ |
|------|-------|---------|---------|
| Reverberant (No Processing) | 1.54 | 8.86 | 5.23 |
| MetricGAN+ | 2.33 | 7.49 | 5.16 |
| VIDA | 2.37 | 4.44 | 4.58 |
| **MVSD** | **2.53** | **4.27** | **4.46** |

### Ablation Study

**User Study (VAM Preference Rate)**:

| Compared Method | SoundSpaces (Opponent/MVSD) | AVSpeech (Opponent/MVSD) |
|----------|------------------------|---------------------|
| Input Speech | 39.3% / 60.7% | 38.2% / 61.8% |
| Image2Reverb | 20.8% / 79.2% | - |
| AV U-Net | 23.4% / 76.6% | 21.9% / 78.1% |
| AViTAR | 34.7% / 65.3% | 44.1% / 55.9% |

**Ablation of Unpaired Data and Components (extracted from Table 1 of the paper)**:

| Variant | STFT↓ (Seen) | RTE(s)↓ | MOSE↓ |
|------|-------------|---------|-------|
| w/o visual scene | 0.691 | 0.188 | 0.156 |
| w/o unpaired data | 0.573 | 0.033 | 0.148 |
| **MVSD (Full)** | **0.508** | **0.030** | **0.142** |

### Key Findings

1. MVSD achieves a **23.6% relative improvement in STFT** (0.665 $\rightarrow$ 0.508) and an **11.8% relative improvement in RTE** on the SoundSpaces-Speech Seen set.
2. **Unpaired data (accounting for only 17.3% of training data) further boosts RTE performance by 9.1%**, demonstrating the effective utilization of unpaired data by the mutual learning framework.
3. Removing visual scene information increases RTE from 0.030 to 0.188 (a 6.3-fold degradation), proving the key dynamic of visual conditioning in reverberation style control.
4. MVSD consistently outperforms all baselines in the user study, including the SOTA method AViTAR (65.3% vs 34.7%).

## Highlights & Insights

- **Dual Perspective of Mutual Learning**: The mutual relationship between VAM and dereverberation is revealed for the first time. The design where both tasks act as reciprocal evaluators is elegant and highly efficient.
- **Unpaired Data Utilization**: Easily obtained unpaired data is cleverly leveraged through cycle-consistency loss, which essentially adopts the CycleGAN philosophy within the audio domain.
- **Diffusion Models Replacing GANs**: Employing a diffusion model as a conditional converter resolves the issues of training instability and over-smoothing that occur in GANs for acoustic matching.
- **Selective Cross-modal Attention**: Utilizing cross-modal attention only in key layers strikes a balance between performance and computational efficiency.

## Limitations & Future Work

- The inference time is approximately 1.09 seconds, which cannot yet meet the requirements of real-time applications.
- The introduction of unpaired data relies on the training converging to a certain degree first (>100 epochs); otherwise, it may lead to domain shift.
- Spectrogram-level L1 style loss may not fully capture perceptual room impulse response or reverberation characteristics.

## Rating

⭐⭐⭐⭐ (4/5) — The mutual learning framework is elegantly designed, and the unpaired data utilization strategy is highly effective. It achieves significant improvements in both VAM and dereverberation tasks, though there remains room for improvement in real-time execution and perceptual evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality](learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)
- [\[ECCV 2024\] WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation](webrpg_automatic_web_rendering_parameters_generation_for_visual_presentation.md)
- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)

</div>

<!-- RELATED:END -->
