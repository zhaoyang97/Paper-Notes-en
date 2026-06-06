---
title: >-
  [Paper Note] Can Natural Image Autoencoders Compactly Tokenize fMRI Volumes for Long-Range Dynamics Modeling?
description: >-
  [CVPR 2026][Medical Imaging][fMRI analysis] This paper proposes TABLeT, which leverages a pretrained 2D natural image autoencoder (DCAE) to compress 3D fMRI volumes into as few as 27 continuous tokens per frame. Paired w…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "fMRI analysis"
  - "autoencoder transfer"
  - "long-sequence modeling"
  - "Transformer"
  - "masked token modeling"
date: 2026-05-08
content_hash: ca037e1a3b753cd0
---

# Can Natural Image Autoencoders Compactly Tokenize fMRI Volumes for Long-Range Dynamics Modeling?

**Conference**: CVPR 2026
**arXiv**: [2604.03619](https://arxiv.org/abs/2604.03619)  
**Code**: [GitHub](https://github.com/beotborry/TABLeT)  
**Area**: 3D Vision
**Keywords**: fMRI analysis, autoencoder transfer, long-sequence modeling, Transformer, masked token modeling

## TL;DR
This paper proposes TABLeT, which leverages a pretrained 2D natural image autoencoder (DCAE) to compress 3D fMRI volumes into as few as 27 continuous tokens per frame. Paired with a standard Transformer encoder, this enables unprecedented long-range temporal modeling (256 frames), surpassing SOTA voxel-based methods on multiple tasks across UKB, HCP, and ADHD-200, while significantly improving computational efficiency.

## Background & Motivation
**Background**: fMRI analysis methods fall into two categories: ROI-based methods (e.g., BrainNetCNN, BNT), which are efficient but lose spatial information, and voxel-based methods (e.g., TFF, SwiFT), which preserve complete information but incur prohibitive memory costs, limiting processing to ~20 temporal frames.

**Limitations of Prior Work**: fMRI is a 4D signal (3D spatial + temporal). Due to GPU memory constraints, voxel-based methods can only process very short temporal windows (20 frames), making it impossible to capture important long-range temporal dynamics such as ultra-slow BOLD-LFP coupling and whole-brain arousal waves.

**Key Challenge**: Modeling long temporal sequences requires compressing the spatial dimension, yet aggressive compression discards critical spatial information—the longstanding limitation of ROI-based approaches. The central challenge is achieving a high compression ratio while retaining sufficient information.

**Goal**: Design a compact tokenization scheme for fMRI volumes that enables Transformers to process significantly longer time series within limited GPU memory.

**Key Insight**: A counterintuitive finding—a 2D autoencoder pretrained on natural images, without any fine-tuning on medical data, can effectively tokenize fMRI volumes.

**Core Idea**: Slice 3D fMRI volumes into 2D images along three axes, encode them with a pretrained DCAE (32× spatial compression), and reassemble the results into 27 tokens per frame, enabling long-sequence inputs of 256 frames.

## Method

### Overall Architecture
4D fMRI volume → per-frame slicing along three axes into 2D images → pretrained 2D DCAE encoding → aggregation of tri-axial tokens into 27 tokens/frame → Transformer encoder processing 256 frames × 27 tokens → [CLS] token for downstream task prediction.

### Key Designs

1. **2D Slicing + DCAE Tokenization**:

    - Single-channel slices are replicated into RGB → sliced along the D/H/W axes → each 2D slice is encoded by DCAE into a latent representation of shape $C' \times \frac{H}{32} \times \frac{W}{32}$
    - An fMRI volume of size (96, 96, 96) yields $3 \times 3 \times 3 = 27$ spatial positions per axis
    - Tri-axial tokens are concatenated by spatial position: each token has dimension $96 \times C' = 96 \times 32 = 3072$
    - Result: **only 27 tokens per frame** (vs. ~12K voxel tokens in SwiFT)
    - **Design Motivation**: DCAE's 32× compression maintains excellent reconstruction quality on natural images; tri-axial slicing ensures every spatial position is covered from three orientations

2. **Why Can Natural Image AEs Tokenize fMRI?**:

    - Empirically validated: the reconstruction quality of 2D DCAE on fMRI is comparable to that of a 3D DCAE trained specifically on fMRI data
    - Information preservation: coarse-grained spatial details and global functional patterns are both retained
    - **Design Motivation**: Training an AE directly on fMRI is computationally expensive, data-hungry (limited medical data), and generalizes poorly across scanners; AEs pretrained on natural images have learned general-purpose spatial feature extraction capabilities

3. **TABLeT Transformer Architecture**:

    - Standard Transformer encoder with modern LLM components
    - 12 layers, 14 attention heads, 2 KV heads (grouped-query attention)
    - Rotary positional encoding (RoPE) + `F.scaled_dot_product_attention`
    - Input tokens are linearly projected to lower dimensions, prefixed with a [CLS] token, and normalized
    - Processes $T=256$ frames (vs. $T=20$ in SwiFT)
    - **Design Motivation**: The tokenization stage handles the compression burden, allowing a standard Transformer to suffice downstream; GQA efficiently handles long sequences

4. **Self-Supervised Pretraining (Masked Token Modeling)**:

    - Inspired by SimMIM: tokens are randomly masked and reconstructed via a Transformer with a linear head
    - Masking ratio of 0.5 with a tubular masking strategy (consistent masking across frames at the same spatial position to prevent temporal "cheating")
    - Loss: $L = \frac{1}{\Omega(\mathbf{Z}_M)} \|\mathbf{y}_M - \mathbf{Z}_M\|_1$
    - Pretrained on the large-scale UKB dataset → fine-tuned on HCP (only 10 epochs)
    - **Design Motivation**: Performing MIM directly in token space eliminates the need to load encoder/decoder modules, simplifying the training pipeline

### Loss & Training
- Classification tasks: cross-entropy loss
- Regression tasks: MSE loss
- Pretraining: $\ell_1$ masked reconstruction loss
- During training, 256 frames are randomly sampled; during validation, the full sequence is processed in segments and predictions are averaged

## Key Experimental Results

### Main Results

| Method | UKB Sex ACC | UKB Age MAE↓ | HCP Sex ACC | HCP Age ρ↑ | ADHD ACC |
|------|------------|-------------|------------|-----------|---------|
| BNT (ROI) | 92.4 | 0.588 | 86.3 | 0.444 | 63.6 |
| SwiFT (T=20) | 97.4 | 0.480 | 93.1 | 0.450 | 63.3 |
| SwiFT (T=50) | 98.1 | 0.477 | 92.2 | 0.460 | 63.9 |
| **TABLeT (T=256)** | 97.7 | **0.466** | **93.8** | **0.473** | **65.8** |

### Ablation Study

| Configuration | Sex ACC | Age ρ↑ | Intelligence ρ↑ |
|------|---------|--------|-----------------|
| TABLeT trained from scratch | 93.8 | 0.473 | 0.392 |
| TABLeT pretrained + fine-tuned | **95.3** | **0.552** | **0.435** |

| Comparison | HCP Sex | ADHD Diagnosis | Notes |
|------|---------|---------------|------|
| 2D DCAE tokenizer | **93.8** | **65.8** | Natural image AE |
| 3D DCAE tokenizer | 93.5 | 65.6 | fMRI-specific AE |

### Key Findings
- TABLeT matches or outperforms all baselines (both ROI-based and voxel-based) on most tasks
- Long-range temporal modeling (256 vs. 20 frames) yields the largest gains on intelligence prediction and ADHD diagnosis, suggesting these tasks require longer temporal dependencies
- 2D natural image DCAE performs nearly identically to 3D fMRI-specific DCAE, validating the central hypothesis
- Pretraining substantially improves downstream performance, particularly on age regression (ρ: 0.473 → 0.552)
- Computational efficiency: at equivalent input scale, TABLeT requires far less memory and computation than SwiFT

## Highlights & Insights
- The finding that natural image AEs can tokenize fMRI is itself highly instructive, suggesting that low-level visual feature extraction generalizes across domains
- The extreme compression to 27 tokens/frame makes long-sequence modeling practical (256 vs. 20 frames)
- The tubular masking strategy prevents information leakage along the temporal dimension
- Experiments span three large-scale datasets (UKB 8K+, HCP 1K+, ADHD-200 533) across multiple task types

## Limitations & Future Work
- Overall performance gains are modest (as acknowledged by the authors)—likely because resting-state fMRI signals are inherently weak
- Validation is limited to resting-state fMRI; task-based fMRI (with stronger temporal dynamics) may yield greater benefits
- The DCAE is frozen without fine-tuning, potentially discarding fMRI-specific low-level features
- Spatial information is substantially compressed during tokenization, limiting fine-grained spatial analysis
- Applying TABLeT to functional connectome analysis is a promising future direction

## Related Work & Insights
- SwiFT is the strongest voxel-based baseline but is bottlenecked by the memory requirements of 4D window attention
- DCAE's 32× compression was originally designed to accelerate diffusion models in image generation; its application to fMRI is novel
- The masked pretraining paradigm from MAE/VideoMAE is successfully transferred to the fMRI token space
- Key insight: the transferability of pretrained models may be broader than expected—even bridging the gap between natural image perception and brain functional imaging

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The finding and validation that natural image AEs can tokenize fMRI is highly original
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets, multiple tasks, pretraining ablations, and AE comparisons, though gains are modest
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated and experimental design is rigorous
- Value: ⭐⭐⭐⭐ — Opens a new pathway for long-sequence fMRI modeling; core findings have broad implications

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Modeling Spatiotemporal Neural Frames for High Resolution Brain Dynamics](modeling_spatiotemporal_neural_frames_for_high_resolution_brain_dynamic.md)
- [\[ICLR 2026\] Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer](../../ICLR2026/medical_imaging/brain-it_image_reconstruction_from_fmri_via_brain-interaction_transformer.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)
- [\[ICLR 2026\] Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?](../../ICLR2026/medical_imaging/can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)

</div>

<!-- RELATED:END -->
