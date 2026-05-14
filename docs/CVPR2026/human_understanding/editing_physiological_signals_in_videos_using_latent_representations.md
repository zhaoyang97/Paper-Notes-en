---
title: >-
  [Paper Note] Editing Physiological Signals in Videos Using Latent Representations
description: >-
  [CVPR 2026][Human Understanding][Heart rate editing] This paper proposes PhysioLatent, a framework that encodes input facial videos into the latent space of a 3D VAE…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Heart rate editing"
  - "rPPG privacy"
  - "video physiological signals"
  - "3D VAE"
  - "biometric anonymization"
  - "FiLM"
  - "AdaLN"
date: 2026-05-08
content_hash: 2a8a95e8b8fd170d
---

# Editing Physiological Signals in Videos Using Latent Representations

**Conference**: CVPR 2026
**arXiv**: [2509.25348](https://arxiv.org/abs/2509.25348)
**Code**: Available (promised by authors)
**Area**: Human Understanding
**Keywords**: Heart rate editing, rPPG privacy, video physiological signals, 3D VAE, biometric anonymization, FiLM, AdaLN

## TL;DR

This paper proposes PhysioLatent, a framework that encodes input facial videos into the latent space of a 3D VAE, fuses the resulting representation with target heart rate CLIP text embeddings, captures rPPG temporal coherence via AdaLN-enhanced spatiotemporal fusion layers, and employs a FiLM-modulated decoder with a fine-tuned output layer to achieve precise heart rate modification. The method attains a heart rate modulation MAE of 10 bpm while preserving visual quality at PSNR 38.96 dB / SSIM 0.98.

## Background & Motivation

**Background**: Remote photoplethysmography (rPPG) enables contactless heart rate extraction from facial videos and is a key technology for remote health monitoring. However, this also implies that sensitive physiological information is invisibly embedded in facial videos, where it can be extracted by algorithms for covert health inference, emotion surveillance, and biometric profiling.

**Limitations of Prior Work**:

1. Existing visual privacy methods (face blurring/replacement) may inadvertently destroy or preserve PPG signals and do not specifically address the physiological privacy dimension.
2. Face replacement simultaneously introduces new identity cues, which is undesirable in privacy-preserving scenarios.
3. Pixel-space methods such as PulseEdit suffer from low temporal coherency.
4. No existing method can precisely modify heart rate to a specified target value (e.g., fixing HR to a constant for anonymization).

**Key Challenge**: The rPPG signal is embedded in subtle skin color variations (<1% change in pixel values), invisible to the human eye yet extractable by algorithms—precise modification of this "invisible" signal is required without affecting visual appearance.

**Goal**: To construct a controllable video physiological signal editing framework capable of precisely modifying heart rate to an arbitrary target value while maintaining high visual fidelity.

**Key Insight**: Performing heart rate editing in the 3D VAE latent space—leveraging the compactness of the latent space for precise modulation, compatibility with video generation pipelines, and spatiotemporal fusion layers for capturing the periodic characteristics of rPPG.

**Core Idea**: 3D VAE latent space editing + AdaLN temporal attention conditioning + FiLM decoder fine-tuning = precise and controllable heart rate modification.

## Method

### Overall Architecture

Input facial video (128 frames, 72×72) → frozen 3D Causal VAE encoder produces latent representation $z$ → frozen CLIP ViT-B/32 encodes target HR prompt (e.g., "Heart rate 80 bpm") to condition $c$ → linear projection of $c$ to match $z$ dimensions, concatenated channel-wise → **trainable spatiotemporal fusion layers** (spatial convolution + temporal self-attention + AdaLN HR conditioning) → 3D Causal VAE decoder (FiLM conditioning + fine-tuned output layer) → Haar Cascade face detector generates facial mask → facial region of decoder output replaces corresponding region in original video → output video (visually unchanged, HR modified).

### Key Designs

1. **3D VAE Latent Space Selection**

    - A 3D Causal VAE (4× spatial downsampling, 8× temporal downsampling) is chosen over a 2D VAE: rPPG signals are inherently cross-frame temporal sequences requiring 3D representations to capture spatiotemporal coherence.
    - Compatible with Latent Diffusion / Video Diffusion latent spaces, enabling direct use as a post-processing module within generative video pipelines.
    - CLIP text embeddings (512-dim) are linearly projected and concatenated channel-wise with the video latent representation, providing a unified conditioning interface.

2. **AdaLN Spatiotemporal Fusion Layers**

    - Incorporates factorized spatial–temporal self-attention modules to explicitly model the strong temporal coherence of rPPG (heart rate is a periodic signal requiring consistent cross-frame modulation).
    - Adaptive Layer Normalization (AdaLN) injects HR conditioning into the temporal attention stream, encoding target frequency information in the normalization parameters.
    - Compared to naïve alternatives: (2+1)D convolutions capture only local patterns without modeling long-range temporal dependencies, leading to inaccurate HR modulation.
    - **Design Motivation**: rPPG variations are extremely subtle (sub-pixel-level color fluctuations, <1% pixel values), necessitating fine-grained long-range temporal conditioning.

3. **FiLM Decoder Conditioning + Fine-tuned Output Layer**

    - Feature-wise Linear Modulation (FiLM) injects HR conditioning into intermediate layers of the 3D VAE decoder—CLIP-projected HR embeddings generate scale and shift parameters that modulate decoder activations.
    - Only the decoder output layer is fine-tuned (rather than the entire decoder), preserving the pretrained visual reconstruction capability.
    - **Design Motivation**: The standard 3D VAE decoder is optimized for general video reconstruction and tends to erase subtle rPPG modulations during decoding; FiLM explicitly preserves physiological signal modifications, while output layer fine-tuning adapts the model to these extremely fine-grained changes.

4. **Facial Region Replacement Strategy**

    - A Haar Cascade face detector generates a facial mask $M$.
    - Only the facial region of the decoder output is composited back into the original video; all remaining pixels are kept unchanged, further guaranteeing perfect fidelity in non-facial regions.

### Loss & Training

- **Visual Fidelity Loss**: $\mathcal{L}_F = \text{MSE}(v, \hat{v}) + \text{LPIPS}(v, \hat{v})$
- **Waveform Loss** (morphology guidance): $\mathcal{L}_{\text{wave}} = 1 - \text{Pearson}(rPPG(\hat{v}), \sin(2\pi f t))$, where $f = \text{HR}_d / 60$; guides the rPPG toward a smooth periodic waveform at the target frequency.
- **Frequency Loss** (precise alignment): $\mathcal{L}_{\text{freq}} = |f - f_{\text{pred}}|$, where $f_{\text{pred}}$ is obtained via FFT.
- Curriculum learning strategy: only visual and waveform losses are used for the first 10 epochs; starting from epoch 10, the frequency loss weight is linearly ramped as $\beta(t) = 0.005(t - 10)$.
- **Total Loss**: $\mathcal{L} = 0.2\mathcal{L}_{\text{wave}} + \beta(t)\mathcal{L}_{\text{freq}} + 1.0\mathcal{L}_F$
- Training: 4× RTX 4090, batch size 4, 30 epochs, AdamW optimizer, OneCycle schedule with lr = 5e-4, input 128 frames × 72×72.

## Key Experimental Results

### Main Results (Cross-dataset, POS estimator)

| Dataset | Target HR | PSNR↑ (dB) | SSIM↑ | Input MAE (bpm) | Output MAE↓ (bpm) | Output MAPE↓ (%) |
|--------|--------|-----------|-------|-------------|--------------|-------------|
| PURE | 60 bpm | 39.04 | 0.98 | 38.95 | 9.22 | 8.20 |
| PURE | 80 bpm | 39.02 | 0.98 | 41.80 | 9.98 | 10.55 |
| PURE | 100 bpm | 38.85 | 0.98 | 44.67 | 10.41 | 10.29 |
| PURE | 120 bpm | 38.94 | 0.98 | 50.63 | 10.36 | 11.34 |
| **PURE Avg** | - | **38.96** | **0.98** | 44.01 | **10.00** | **10.09** |
| UBFC Avg | - | 40.09 | 0.98 | 26.77 | 11.08 | 10.57 |
| MMPD Avg | - | 37.50 | 0.95 | 44.58 | 9.84 | 8.09 |

### Benchmark Comparison (Target HR = 120 bpm, POS estimator)

| Dataset | Method | PSNR↑ (dB) | SSIM↑ | Output MAE↓ (bpm) | Output MAPE↓ (%) |
|--------|------|-----------|-------|--------------|-------------|
| PURE | **PhysioLatent** | 38.94 | **0.9761** | **10.36** | **11.34** |
| PURE | PulseEdit | **42.68** | 0.9720 | 16.71 | 12.26 |
| UBFC | **PhysioLatent** | 40.04 | 0.9803 | **11.18** | **10.15** |
| UBFC | PulseEdit | **43.08** | **0.9867** | 15.07 | 15.56 |
| MMPD | **PhysioLatent** | 37.87 | 0.9542 | **10.75** | **7.96** |
| MMPD | PulseEdit | **41.72** | **0.9664** | 20.36 | 18.30 |

### Key Findings

- PSNR > 38 dB + SSIM ≥ 0.95 indicates modifications imperceptible to the human eye, validating successful "invisible privacy protection."
- HR modulation MAE of approximately 10 bpm is 6–10 bpm lower than PulseEdit (16–20 bpm), demonstrating the advantage of temporal coherence modeling.
- PulseEdit achieves higher PSNR (42–43 vs. 38–40 dB) but inferior HR accuracy (MAE 15–20 vs. 10–11 bpm), reflecting a different trade-off.
- All seven rPPG estimators (PCA / POS / CHROM / TSCAN / DeepPhys / PhysNet / PhysFormer++) are successfully misled to the target HR, confirming the robustness of the proposed method.
- On MMPD (multiple skin tones and lighting conditions), PSNR is slightly lower (37.5 dB) but HR accuracy is actually the best (MAE 9.84 bpm), indicating that HR editing does not depend on specific appearance conditions.

## Highlights & Insights

- **Precise problem formulation for "invisible privacy"**: The first work specifically targeting controllable editing of invisible physiological signals in videos—a privacy dimension unaddressed by face blurring or replacement.
- **3D VAE compatibility with video generation pipelines**: Operating in the latent space of Latent Diffusion / Video Diffusion models enables direct use as a post-processing module within existing generative video frameworks.
- **Dual value for anonymization and synthesis**: Anonymization (fixing HR = 60 bpm) protects privacy; synthesis (specifying arbitrary HR) generates annotated training data for rPPG research.
- **Engineering solution for subtle signal editing**: rPPG variations are <1% of pixel values; a three-tier guarantee is achieved via AdaLN long-range temporal conditioning + FiLM decoder for explicit signal preservation + fine-tuned output layer for precision adaptation.

## Limitations & Future Work

- An HR MAE of approximately 10 bpm remains too large for precise health monitoring applications; further improvements in frequency alignment accuracy are needed.
- The current work addresses only heart rate; editing of other physiological signals (respiratory rate, blood pressure, SpO₂) remains unexplored.
- The 3D VAE encode–decode process itself introduces subtle visual changes (PSNR ≈ 39 dB rather than lossless), with localized distortions visible in high-frequency regions (edges/textures).
- Motion robustness has not been validated (marked as "No" in the motion robustness column of Table 1).
- PulseEdit achieves higher PSNR (+3–4 dB), indicating that VAE reconstruction quality is the bottleneck for visual fidelity.

## Related Work & Insights

- **vs. PulseEdit**: Operates directly in pixel space, achieving higher PSNR but suffering from low temporal coherency and poor HR accuracy (MAE 15–20 bpm); PhysioLatent operates in the latent space, achieving twice the HR accuracy (MAE 10 bpm) at the cost of 3–4 dB lower PSNR.
- **vs. Privacy-Phys**: Uses a 3D CNN to directly modify rPPG but exhibits low temporal coherency and cannot precisely control the target HR.
- **vs. Wang et al.**: GAN-based conditional rPPG video generation, but cannot handle signal removal.
- **Insights**: The latent-space editing paradigm is generalizable to controlling other invisible signals in video (e.g., audio watermarks, motion patterns); the 3D VAE + CLIP conditioning interface is natively compatible with existing generative pipelines.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Controllable editing of invisible physiological signals in video is an entirely new problem; the latent-space editing approach is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on 3 datasets with 7 rPPG estimators, systematic comparison against PulseEdit, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The problem motivation ("invisible privacy") is clear and compelling; illustrations of the two key rPPG properties (temporal coherence and visual invisibility) are intuitive.
- Value: ⭐⭐⭐⭐ Contributes to both AI privacy protection and rPPG training data generation; 3D VAE compatibility offers practical engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos](unidex_a_robot_foundation_suite_for_universal_dexterous_hand_control_from_egocen.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](../../ICLR2026/human_understanding/visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[ICLR 2026\] BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis](../../ICLR2026/human_understanding/bah_dataset_for_ambivalencehesitancy_recognition_in_videos_for_digital_behaviour.md)
- [\[NeurIPS 2025\] CPEP: Contrastive Pose-EMG Pre-training Enhances Gesture Generalization on EMG Signals](../../NeurIPS2025/human_understanding/cpep_contrastive_pose-emg_pre-training_enhances_gesture_generalization_on_emg_si.md)
- [\[CVPR 2026\] Mobile-VTON: High-Fidelity On-Device Virtual Try-On](mobile_vton_ondevice_virtual_tryon.md)

</div>

<!-- RELATED:END -->
