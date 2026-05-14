---
title: >-
  [Paper Note] CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment
description: >-
  [CVPR 2026][Image Generation][EEG Visual Decoding] CognitionCapturerPro addresses Fidelity Loss via uncertainty-weighted masking and resolves Representational Shift by integrating image, text, depth…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "EEG Visual Decoding"
  - "Multi-modal Fusion"
  - "Uncertainty Modeling"
  - "Brain-Computer Interface"
  - "Diffusion Reconstruction"
date: 2026-05-08
content_hash: 4b454aa0996cab80
---

# CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment

**Conference**: CVPR 2026  
**arXiv**: [2603.12722](https://arxiv.org/abs/2603.12722)  
**Code**: To be released  
**Area**: Image Generation / Brain Signal Decoding  
**Keywords**: EEG Visual Decoding, Multi-modal Fusion, Uncertainty Modeling, Brain-Computer Interface, Diffusion Reconstruction

## TL;DR

CognitionCapturerPro addresses Fidelity Loss via uncertainty-weighted masking and resolves Representational Shift by integrating image, text, depth, and edge information through a multi-modal fusion encoder. Combined with a lightweight shared backbone alignment replacing diffusion priors, it improves Top-1/Top-5 retrieval accuracy on the THINGS-EEG dataset by 25.9% and 10.6%, respectively.

## Background & Motivation

**Background**: Decoding visual stimuli from EEG signals is a critical direction in Brain-Computer Interfaces (BCI). Prevailing methods align EEG embeddings with CLIP space to achieve retrieval and reconstruction. EEG has become the most promising practical modality due to its portability and millisecond-level temporal resolution.

**Limitations of Prior Work**: Neuroscience research reveals two core bottlenecks—(1) **Fidelity Loss**: As the visual system converts stimuli into neural signals, attention mechanisms lead to incomplete information (e.g., focusing on bicycle wheels rather than the whole bike); (2) **Representational Shift**: Brain association mechanisms introduce non-visual semantics (e.g., seeing a penguin and thinking of Antarctica), causing neural signals to deviate from visual features. Existing methods either handle semantic alignment while ignoring fidelity loss, or model uncertainty only within a visual framework while ignoring semantic associations.

**Key Challenge**: A systematic mismatch exists between EEG signals and visual stimuli, arising from two independent mechanisms (information loss + subjective bias), which must be addressed simultaneously for high-fidelity decoding.

**Goal**: To realize accurate brain-to-image retrieval and reconstruction under limited neural data conditions by simultaneously overcoming fidelity loss and representational shift.

**Key Insight**: The authors build upon the multi-modal expansion strategy of the conference version (CognitionCapturer), adding an uncertainty-weighted mechanism to solve fidelity loss and replacing the diffusion prior with a lightweight MLP alignment to reduce overfitting risks.

**Core Idea**: Use uncertainty-driven dynamic masking to simulate human foveal vision for solving fidelity loss, and use multi-modal fusion + shared backbone alignment to solve representational shift.

## Method

### Overall Architecture

The framework consists of five core components: (1) Uncertainty-Weighted Masking (UM) to simulate foveal vision; (2) Four modality-specific encoders mapping EEG to image/text/depth/edge embedding spaces; (3) A fusion encoder integrating all embeddings via cross-modal Transformers; (4) Shared Backbone and Task Head Alignment (STH-Align) mapping embeddings to a unified image space; (5) SDXL-Turbo + IP-Adapter for generating high-fidelity reconstructed images.

### Key Designs

1. **Uncertainty-Weighted Masking (UM)**:

    - **Function**: Dynamically adjusts image blur intensity for training samples to simulate human foveal vision mechanisms.
    - **Mechanism**: First, a fovea-like spatial gradient blur is applied—clear at the center and blurred at edges—where the blur kernel $\mathbf{M}_{\text{fovea}}(i,j)$ decays exponentially from center to periphery. Then, blur intensity is dynamically adjusted based on the model's current alignment level: utilizing EMA-smoothed historical similarity scores to establish confidence intervals, "easy" samples (high similarity) receive increased blur to prevent overfitting, while "hard" samples (low similarity) receive decreased blur to facilitate learning of key features.
    - **Design Motivation**: Direct alignment assumes neural signals fully represent the stimulus, ignoring the selectivity and locality of human attention. UM narrows the systematic gap between modalities by simulating the information loss process.

2. **Fusion Encoder**:

    - **Function**: Integrates four modality-specific EEG embeddings into a unified representation.
    - **Mechanism**: The four embeddings are aligned to a shared dimension $d=1024$ via linear projection, combined with learnable modality positional encodings, and fed into a two-layer Transformer encoder for cross-modal interaction via multi-head self-attention. Global average pooling + residual MLP outputs the fused embedding $\mathbf{z}_{\text{fus}}$. During training, one modality is randomly zeroed (Modality Masking) to enhance robustness.
    - **Design Motivation**: Different modalities provide complementary information—images capture semantics, text encodes associations, depth reflects 3D structure, and edges preserve contours. Fusion, rather than simple concatenation, allows for mutual enhancement between modalities.

3. **Shared Backbone and Task Head Alignment (STH-Align)**:

    - **Function**: Replaces diffusion priors with lightweight MLPs to align multi-modal embeddings into the image embedding space.
    - **Mechanism**: The four EEG embeddings are concatenated and fed into a 4-layer SiLU-MLP shared backbone to obtain a common representation $\mathbf{f}$, which is then passed through four 2-layer MLP modality heads to output $\hat{\mathbf{e}}^m$. The optimization targeted is a tripartite loss: $\mathcal{L}_{\text{STH}} = \sum_m [\lambda_{\text{mse}}\|\hat{\mathbf{e}}^m - \mathbf{v}^m\|_2^2 + \lambda_{\text{cos}}(1-\cos(\hat{\mathbf{e}}^m, \mathbf{v}^m)) + \lambda_{\text{reg}}\|\hat{\mathbf{e}}^m\|_2^2]$.
    - **Design Motivation**: Diffusion priors require large-scale data and are prone to overfitting and expensive inference when only tens of thousands of EEG-image pairs are available. Lightweight MLP alignment is more stable on small data.

### Loss & Training

The encoder stage uses SCM-Loss (Similarity-Category Masking Contrastive Loss): after constructing a similarity matrix, only samples that are of the same category and within the top-k similarity are treated as positive pairs, resolving a fundamental one-to-many mapping conflict inherent in EEG datasets. STH-Align is trained separately using MSE + Cosine + Regularization losses. The reconstruction stage uses three IP-Adapter branches (Image/Depth/Edge) injected into SDXL-Turbo.

## Key Experimental Results

### Main Results

THINGS-EEG Zero-shot Retrieval (Average across 10 subjects):

| Method | Top-1 | Top-5 |
|------|-------|-------|
| BraVL | 5.8 | 17.5 |
| NICE | 14.1 | 43.6 |
| MB2C | 28.4 | 60.3 |
| CognitionCapturer | 35.6 | 80.2 |
| ATS | 60.2 | 86.7 |
| **CogCapPro(F)** | **61.2** | **90.8** |

CogCapPro(F) in fusion mode achieves the best performance, showing an improvement of 25.6%/10.6% over the conference version CognitionCapturer.

### Ablation Study

| Modality Config | Top-1 | Top-5 | Note |
|---------|-------|-------|------|
| CogCapPro(I) | 52.7 | 83.5 | Image modality only |
| CogCapPro(T) | 14.2 | 38.6 | Text modality only |
| CogCapPro(D) | 17.5 | 44.3 | Depth modality only |
| CogCapPro(E) | 29.9 | 64.4 | Edge modality only |
| CogCapPro(F) | 61.2 | 90.8 | Full fusion |

### Key Findings

- Multi-modal fusion outperforms the best single modality (Image) by 8.5%/7.3%, proving the value of complementary multi-modal information.
- The image modality contributes most, followed by edges, with text being the weakest—aligning with cognitive science expectations that EEG signals primarily encode visual rather than linguistic information.
- Uncertainty-Weighted Masking shows the most significant improvement for "hard" subjects (e.g., Subject 5's Top-1 was only 45.2%).
- Effectiveness is also validated on the THINGS-MEG dataset, demonstrating generalization across brain signal modalities.

## Highlights & Insights

- **Uncertainty-driven Curriculum Learning**: Dynamically adjusting training difficulty based on the model's current alignment level is essentially adaptive curriculum learning. This "alignment feedback to data augmentation" loop design is transferable to any cross-modal alignment task.
- **SCM-Loss for One-to-Many Mapping**: In EEG datasets, different samples of the same category create conflicting gradients in InfoNCE; SCM solves this fundamental training issue via dual-filtering of positive pairs using semantic labels and similarity.
- **Lightweight Alignment vs. Diffusion Priors**: MLP alignment is superior to diffusion priors in small-data scenarios, providing a practical insight for the brain decoding community—complex architectures are not always necessary.

## Limitations & Future Work

- The THINGS-EEG dataset has few subjects (10), with large individual differences (28% gap between Subject 5 and Subject 8).
- Low spatial resolution of EEG results in noticeable blur in high-frequency details of reconstructed images.
- Multi-modal annotations (Text/Depth/Edge) require extra computation during deployment; latency in real-time BCI scenarios was not discussed.
- End-to-end training (currently encoders and alignment are trained separately) has not been explored.

## Related Work & Insights

- **vs CognitionCapturer (Conference version)**: The conference version did not explicitly handle fidelity loss and used diffusion priors for alignment. The Pro version adds UM to solve fidelity issues and replaces diffusion priors with simpler STH-Align.
- **vs ATS**: ATS focuses more on single-modality uncertainty modeling, while Pro addresses the problem from a more comprehensive perspective through the combination of multi-modal fusion and UM.
- **vs UBP**: UBP introduces uncertainty modeling but is restricted to a pure visual framework, ignoring representational shift; Pro's multi-modal extension is more complete.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Formalizing foveal vision mechanisms as uncertainty-weighted masking has cognitive depth; the combination of multi-modal fusion and lightweight alignment is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Dual-modality validation on EEG and MEG with individual reporting for 10 subjects, though lacking comparison with more recent methods.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation from a cognitive science perspective is deep, but the paper is long with some redundant descriptions.
- **Value**: ⭐⭐⭐⭐ Provides a practical multi-modal fusion framework for the BCI field; the lightweight alignment approach is highly generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FontCrafter: High-Fidelity Element-Driven Artistic Font Creation with Visual In-Context Generation](fontcrafter_high-fidelity_element-driven_artistic_font_creation_with_visual_in-c.md)
- [\[CVPR 2026\] Garments2Look: A Multi-Reference Dataset for High-Fidelity Outfit-Level Virtual Try-On with Clothing and Accessories](garments2look_a_multi-reference_dataset_for_high-fidelity_outfit-level_virtual_t.md)
- [\[CVPR 2026\] PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On](promo_promptable_virtual_tryon_efficient.md)
- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [\[CVPR 2026\] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)

</div>

<!-- RELATED:END -->
