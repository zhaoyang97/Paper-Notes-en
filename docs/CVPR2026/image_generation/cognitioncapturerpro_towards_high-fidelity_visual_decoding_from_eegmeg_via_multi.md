---
title: >-
  [Paper Note] CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment
description: >-
  [CVPR 2026][Image Generation][EEG decoding] This paper proposes CognitionCapturerPro, which integrates EEG signals with four modalities (image, text, depth, and edge) via Uncertainty-Weighted Masking (UM), a multi-modal fusion encoder, and Shared-Trunk Multi-Head Alignment (STH-Align). On THINGS-EEG, the method achieves a Top-1 retrieval accuracy of 61.2% and Top-5 of 90.8%, improving over the predecessor CognitionCapturer by 25.9% and 10.6%, respectively.
tags:
  - CVPR 2026
  - Image Generation
  - EEG decoding
  - multi-modal fusion
  - visual reconstruction
  - contrastive learning
  - diffusion models
date: 2026-05-08
content_hash: 112bea13d33d62d0
---

# CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment

**Conference**: CVPR 2026
**arXiv**: [2603.12722](https://arxiv.org/abs/2603.12722)
**Code**: Not yet available (to be released upon publication)
**Area**: Brain Signal Decoding / Image Generation
**Keywords**: EEG decoding, multi-modal fusion, visual reconstruction, contrastive learning, diffusion models

## TL;DR

This paper proposes CognitionCapturerPro, which integrates EEG signals with four modalities (image, text, depth, and edge) via Uncertainty-Weighted Masking (UM), a multi-modal fusion encoder, and Shared-Trunk Multi-Head Alignment (STH-Align). On THINGS-EEG, the method achieves a Top-1 retrieval accuracy of 61.2% and Top-5 of 90.8%, improving over the predecessor CognitionCapturer by 25.9% and 10.6%, respectively.

## Background & Motivation

Decoding visual stimuli from EEG/MEG signals faces two fundamental bottlenecks:

**Fidelity Loss**: The visual system inevitably loses information when converting stimuli into neural signals — due to attentional constraints, human perception of images is inherently partial and selective.

**Representational Shift**: The brain's associative mechanisms activate semantic networks beyond the visual content itself during visual processing (e.g., viewing a penguin activates associations with ice and Antarctica), causing neural signals to deviate from the original image features.

Existing methods either focus solely on semantic alignment while ignoring fidelity loss, or address only perceptual uncertainty while overlooking representational shift. CognitionCapturerPro is the first work to jointly address both challenges.

## Method

### Overall Architecture

The system comprises five core components organized into three stages:

- **Encoding Stage**: Uncertainty-Weighted Masking → Modality Expert Encoders → Fusion Encoder
- **Alignment Stage**: STH-Align (shared trunk + multi-modal projection heads)
- **Generation Stage**: SDXL-Turbo + multi-branch IP-Adapter reconstruction

Each EEG sample is paired with four modalities (image, text, depth map, edge map) during training, forming a one-to-many multi-modal supervision scheme.

### Key Designs

1. **Uncertainty-Weighted Masking (UM)**: Simulates the human foveal vision mechanism by applying spatially varying blur to images. The core formulation is:

    $\mathbf{M}_{\text{fovea}}(i,j) = r_{\text{edge}} + (r_{\text{centre}} - r_{\text{edge}}) \cdot \exp\left(-\lambda \frac{d_{ij}}{d_{\max}}\right)$

   An EMA memory bank tracks the alignment score $\hat{s}_i$ of each sample to dynamically adjust blur intensity $\sigma$: "easy" samples receive greater blur to prevent overfitting, while "hard" samples receive less blur to focus on core features. The design motivation is to apply curriculum learning principles to address fidelity variance in EEG signals.

2. **Similarity-Category Masking Loss (SCM-Loss)**: Addresses the InfoNCE training paradox arising from one-to-many mappings in EEG datasets, where samples from the same semantic category are simultaneously attracted and repelled. SCM-Loss defines a masking probability matrix:

    $M_{ij} = \frac{\exp(S_{ij} \cdot m_{ij})}{\sum_{l=1}^{B} \exp(S_{il} \cdot m_{il})}$

   where $m_{ij} = 1$ if and only if $y_i = y_j$ and $j \in \text{top-}k(S_{i,\cdot})$, ensuring that only semantically identical and highly similar samples are treated as positive pairs. Top-$k$ is set to 10. This is the single most impactful module, improving Top-1 accuracy by 6.0% when introduced alone.

3. **Shared Trunk with Multi-Head Alignment (STH-Align)**: Replaces the computationally expensive Diffusion Prior with a lightweight 4-layer MLP shared trunk that processes the concatenated four-modality embeddings $\mathbf{x}_{\text{cat}} = [\mathbf{e}^{\text{img}}; \mathbf{e}^{\text{txt}}; \mathbf{e}^{\text{depth}}; \mathbf{e}^{\text{edge}}] \in \mathbb{R}^{4d}$, with modality-specific projection heads outputting L2-normalized features. The loss function is:

    $\mathcal{L}_{\text{STH}} = \sum_m \left[\lambda_{\text{mse}}\|\hat{\mathbf{e}}^m - \mathbf{v}^m\|_2^2 + \lambda_{\text{cos}}(1 - \cos(\hat{\mathbf{e}}^m, \mathbf{v}^m)) + \lambda_{\text{reg}}\|\hat{\mathbf{e}}^m\|_2^2\right]$

   Random modality dropout during training enhances robustness; only EEG input is required at inference.

4. **Fusion Encoder**: A 2-layer Transformer that takes embeddings from four modality expert encoders with learnable modality position encodings as input. After cross-modal interaction via self-attention, global average pooling and a residual MLP produce the unified representation $\mathbf{z}_{\text{fus}} \in \mathbb{R}^{1024}$.

### Loss & Training

- Encoder training uses SCM-Loss; each modality expert encoder has an independent optimizer to prevent cross-modal information leakage.
- STH-Align is trained separately (MSE + Cosine + L2 regularization) with weights $\lambda_{\text{mse}}=1.0$, $\lambda_{\text{cos}}=0.5$, $\lambda_{\text{reg}}=10^{-4}$.
- The generation stage uses frozen SDXL-Turbo with 3 IP-Adapter branches (image/depth/edge); the text modality is excluded to reduce uncertainty.
- Training runs for 80 epochs with batch size 1024 on 8 × RTX 3090 GPUs.

## Key Experimental Results

### Main Results

| Dataset | Metric | CogCapPro (Fusion) | Prev. SOTA (ATS) | Gain |
|--------|------|-------------------|-----------------|------|
| THINGS-EEG | Top-1 ↑ | **61.2%** | 60.2% | +1.0% |
| THINGS-EEG | Top-5 ↑ | **90.8%** | 86.7% | +4.1% |
| THINGS-MEG | Top-1 ↑ | **31.8%** | 32.3% (ATS) | -0.5% |
| THINGS-MEG | Top-5 ↑ | **64.6%** | 62.4% (ATS) | +2.2% |
| Reconstruction | CLIP ↑ | **0.830** | 0.786 (ATM) | +0.044 |
| Reconstruction | SSIM ↑ | **0.398** | 0.347 (CogCap) | +0.051 |

### Ablation Study

| Configuration | Top-1 ↑ | Top-5 ↑ | Notes |
|------|---------|---------|------|
| Baseline | 51.8 | 84.8 | No UM / SCM-Loss / Mask |
| + UM | 54.7 | 87.1 | Uncertainty-Weighted Masking, +2.9 |
| + UM + SCM-Loss | 60.7 | 90.4 | SCM contributes most, +6.0 |
| + UM + SCM-Loss + Modality Mask | **61.2** | **90.8** | Full model |

### Key Findings

- Multi-modal fusion significantly outperforms any single modality: image modality Top-1 52.7%, edge 29.9%, depth 17.5%, text 14.2%, vs. 61.2% with fusion.
- RN50 outperforms ViT-H-14 as the image encoder (61.2% vs. 56.0%), possibly because the limited information density of EEG signals better matches the feature distribution of RN50.
- The modality masking training strategy effectively improves robustness under missing-modality conditions.

## Highlights & Insights

- This work is the first to systematically decompose the EEG decoding problem into the two dimensions of "fidelity loss" and "representational shift," with dedicated modules designed for each.
- The UM mechanism elegantly draws on the foveal characteristics of the human visual system, adaptively modulating training difficulty via curriculum learning principles.
- STH-Align replaces the Diffusion Prior with a simple MLP, avoiding overfitting on limited data while enabling efficient inference.
- The multi-modal extension strategy (image → text/depth/edge) provides rich complementary supervision signals for EEG decoding.

## Limitations & Future Work

1. The limited training data (THINGS-EEG contains only ~16K images) constrains the potential of more complex models.
2. Top-1 accuracy on MEG data is slightly below ATS, indicating that generalization across different brain signal modalities requires further improvement.
3. Reconstruction quality still lags far behind fMRI-based methods (MindEye PixCorr 0.322 vs. CogCapPro 0.163); the EEG signal-to-noise ratio bottleneck remains unresolved.
4. The text modality contributes least to retrieval (14.2%); more effective utilization of semantic information warrants further exploration.

## Related Work & Insights

- Compared to the conference version CognitionCapturer, this paper introduces two key modules — UM and SCM-Loss — improving Top-1 from 35.6% to 61.2%, a substantial gain.
- Compared to ATM (attention-based alignment), CogCapPro improves high-level semantic metrics (CLIP) by 4.4% through multi-modal fusion.
- Insight: The critical bottleneck in EEG decoding lies not in the generative model but in the alignment strategy — specifically, how to establish robust cross-modal mappings under conditions of sparse signals and high noise.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The binary analytical framework of fidelity loss and representational shift is novel; the UM and SCM-Loss designs are elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Dual-dataset evaluation on EEG and MEG, 10+ baseline comparisons, and multi-dimensional ablations across modules, encoders, and modalities.
- **Writing Quality**: ⭐⭐⭐ Content is thorough but the structure is somewhat complex, with notation definitions scattered across sections.
- **Value**: ⭐⭐⭐⭐ Achieves new state of the art in EEG visual decoding; the multi-modal fusion approach has strong transfer value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [\[CVPR 2026\] Image Generation as a Visual Planner for Robotic Manipulation](image_generation_as_a_visual_planner_for_robotic_manipulation.md)
- [\[CVPR 2026\] Low-Resolution Editing is All You Need for High-Resolution Editing](low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)
- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](vihoi_human-object_interaction_synthesis_with_visual_priors.md)
- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](language-free_generative_editing_from_one_visual_example.md)

</div>

<!-- RELATED:END -->
