---
title: >-
  [Paper Note] Identity-Preserving Text-to-Video Generation by Frequency Decomposition
description: >-
  [CVPR 2025][Video Generation][Identity-preserving video generation] ConsisID proposes a frequency-decomposition-based DiT control scheme. It decouples facial features into low-frequency global information and high-frequency intrinsic identity information, injecting them into different positions of the DiT. This achieves tuning-free, identity-preserving text-to-video generation, significantly outperforming existing methods in identity preservation, text correlation…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Identity-preserving video generation"
  - "frequency decomposition"
  - "DiT"
  - "facial consistency"
  - "tuning-free"
date: 2026-05-08
content_hash: 90d41ddfcb8fa5ee
---

# Identity-Preserving Text-to-Video Generation by Frequency Decomposition

**Conference**: CVPR 2025  
**arXiv**: [2411.17440](https://arxiv.org/abs/2411.17440)  
**Code**: [https://github.com/PKU-YuanGroup/ConsisID](https://github.com/PKU-YuanGroup/ConsisID)  
**Area**: Diffusion Models / Video Generation  
**Keywords**: Identity-preserving video generation, frequency decomposition, DiT, facial consistency, tuning-free

## TL;DR

ConsisID proposes a frequency-decomposition-based DiT control scheme. It decouples facial features into low-frequency global information and high-frequency intrinsic identity information, injecting them into different positions of the DiT. This achieves tuning-free, identity-preserving text-to-video generation, significantly outperforming existing methods in identity preservation, text correlation, and visual quality.

## Background & Motivation

**Background**: Identity-Preserving Text-to-Video generation (IPT2V) is an important task in video generation. Existing methods are mainly based on the U-Net architecture, and most require case-by-case fine-tuning for each new identity (e.g., DreamBooth, LoRA), which is highly inefficient. Within the open-source community, only ID-Animator supports tuning-free IPT2V, but it can only generate talking-head-like videos and suffers from poor identity preservation.

**Limitations of Prior Work**: Although the emerging DiT architecture shows great potential in video generation, transferring identity control signals to DiT faces two core challenges: (1) DiT lacks the long skip connections of U-Net, making it difficult to aggregate low-level features, which leads to slow training convergence; (2) Transformers have weak perception of high-frequency information, which is crucial for preserving facial details.

**Key Challenge**: U-Net naturally possesses multi-scale features and high-frequency perception through its encoder-decoder architecture, whereas DiT lacks these structural advantages. Directly applying control schemes designed for U-Net to DiT is ineffective.

**Goal**: (1) How to achieve tuning-free IPT2V on the DiT architecture? (2) How to design a frequency-aware control scheme to compensate for the architectural limitations of DiT?

**Key Insight**: Inspired by research on frequency analysis in vision/diffusion Transformers, the authors find that shallow features corresponding to low-frequency information facilitate training convergence, while Transformers lack perception of high-frequency information. Facial features can naturally be decomposed into low-frequency (contour, proportion) and high-frequency (identity markers) components, which perfectly complement the deficiencies of DiTs.

**Core Idea**: Decompose facial identity features into high- and low-frequency components based on frequency, and inject them into the shallow inputs and the internal attention blocks of DiT, respectively, achieving frequency-aware identity-preserving video generation.

## Method

### Overall Architecture

ConsisID is built upon the pretrained CogVideoX-5B (DiT architecture). Given a reference face image, the system extracts low- and high-frequency facial information via two complementary feature extractors: the Global Face Extractor (GFE) concatenates the reference image and facial landmarks with the noisy latent as the low-frequency signal input; the Local Face Extractor (LFE) utilizes high-frequency features fused from ArcFace and CLIP encoders, injecting them into each Transformer block via cross-attention. Coupled with a hierarchical training strategy, this generates identity-consistent videos.

### Key Designs

1. **Global Face Extractor (Low-frequency signal injection)**:

    - **Function**: Provide low-frequency global facial information (contours, proportions) to facilitate model convergence.
    - **Mechanism**: Extract facial landmarks from the reference image, convert them into RGB images, and concatenate them along with the reference image into the noisy latent after VAE encoding. The landmark image filters out irrelevant noise such as lighting and shadows, allowing the model to focus on low-frequency facial structural information. The objective function becomes $\mathcal{L}_b = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_0, t, \tau_\theta(y), \psi_\theta(f))\|^2]$
    - **Design Motivation**: DiT lacks the long skip connections of U-Net, making direct training difficult to converge. Injecting shallow, low-frequency signals simulates the function of U-Net skip connections, serving as an indispensable prerequisite for model training.

2. **Local Face Extractor (High-frequency signal injection)**:

    - **Function**: Complement high-frequency facial identity details (eye textures, lip details, and other intrinsic identity markers).
    - **Mechanism**: Employ a dual-tower feature extraction: ArcFace extracts expression/pose-invariant intrinsic identity features, and the CLIP encoder extracts semantic-rich editable features. These features are fused using a Q-Former. The fused features interact with visual tokens via cross-attention in each attention block: $Z_i' = Z_i + \text{Attention}(Q_i^v, K_i^f, V_i^f)$. Meanwhile, Dropout is applied to mitigate the impact of irrelevant features from CLIP.
    - **Design Motivation**: Transformers have weak perception of high-frequency information, and relying solely on low-frequency global features cannot preserve fine-grained identity details. Injecting high-frequency signals inside the attention blocks guides the attention mechanism to focus on intrinsic facial characteristics.

3. **Consistent Training Strategy (Coarse-to-Fine + Dynamic Loss)**:

    - **Function**: Multi-stage training plus dynamic loss design to enhance training efficiency and generalization capability.
    - **Mechanism**: (a) Coarse-to-fine training: The model first learns low-frequency features using the global extractor, followed by the introduction of the local extractor to learn high-frequency details. (b) Dynamic Masked Loss: Computes loss only on the facial region with a probability of $\alpha$, i.e., $\mathcal{L}_d = M \odot \mathcal{L}_c$, to avoid background noise interference. (c) Dynamic Cross-Face Loss: Uses a face outside the training frame as the reference image with a probability of $\beta$, adding Gaussian noise to prevent the model from learning "copy-paste" shortcuts.
    - **Design Motivation**: Video generation requires maintaining spatial-temporal consistency, making direct end-to-end training overwhelmingly complex. The multi-stage strategy reduces learning difficulty, while dynamic losses resolve background interference and overfitting issues, respectively.

### Loss & Training

The final loss function $\mathcal{L}_f$ integrates the dynamic masked loss and the dynamic cross-face loss. Training settings: resolution $480 \times 720$, 49 frames, batch size 80, learning rate $3 \times 10^{-6}$, 1.8k total steps, and $\alpha = \beta = 0.5$. During inference, the DPM sampler is used with 50 steps and CFG = 6.0.

## Key Experimental Results

### Main Results

| Method | FaceSim-Arc ↑ | FaceSim-Cur ↑ | CLIPScore ↑ | FID ↓ |
|------|---------------|---------------|-------------|-------|
| ID-Animator | 0.32 | 0.33 | 24.97 | 117.46 |
| **ConsisID** | **0.58** | **0.60** | **27.93** | 151.82 |

ConsisID significantly outperforms ID-Animator on identity preservation metrics (FaceSim-Arc +81%), while also displaying superiority in text correlation. A user study with 103 valid questionnaires indicates that ConsisID is preferred across all dimensions.

### Ablation Study

| Configuration | FaceSim-Arc ↑ | FaceSim-Cur ↑ | CLIPScore ↑ | FID ↓ |
|------|---------------|---------------|-------------|-------|
| Full model (plan c) | 0.73 | 0.75 | 36.77 | 127.42 |
| w/o GFE (plan b) | 0.05 | 0.05 | 34.86 | 269.88 |
| w/o LFE (plan a) | 0.66 | 0.68 | 34.48 | 104.34 |
| w/o CFT | 0.54 | 0.58 | 34.47 | 144.62 |
| w/o DML | 0.62 | 0.67 | 34.23 | 187.78 |
| w/o DCL | 0.65 | 0.69 | 32.21 | 117.80 |

### Key Findings

- Removing the Global Face Extractor (GFE) leads to a failure in model convergence, with FaceSim-Arc plummeting from 0.73 to 0.05, proving that low-frequency signal injection is a prerequisite for training.
- The injection location of high-frequency signals is critical: injecting inside the attention blocks (plan c) is far superior to injecting at the block outputs (plan e) or block inputs (plan f/g, which causes gradient explosion).
- Fourier spectrum analysis visually validates the effectiveness of frequency decomposition: injecting high/low-frequency signals indeed enhances information in corresponding frequency bands.

## Highlights & Insights

- **Ingenious idea of frequency decomposition control**: It turns the structural deficiencies of DiT into design advantages—the high/low-frequency decomposition of faces aligns perfectly with the shallow and high-frequency information that DiT needs to compensate for, forming a naturally complementary relationship.
- **High practical value of tuning-free generation**: Based on the pretrained CogVideoX-5B, the model achieves tuning-free IPT2V capability with only 1.8k steps of training, significantly lowering the barrier of usage.
- **Transferable methodology of frequency analysis**: Analyzing model limitations from the frequency domain and designing targeted compensations is a generalizable paradigm that can be extended to other DiT-controllable generation tasks (e.g., pose control, style transfer).

## Limitations & Future Work

- The FID metric is inferior to ID-Animator (151.82 vs 117.46), indicating room for improvement in visual quality and diversity of the generated content.
- The model is only evaluated on single-person scenarios, leaving multi-person identity preservation unresolved.
- Built upon the fixed CogVideoX-5B architecture, its adaptability to other DiT architectures (e.g., HunyuanVideo) remains unverified.
- The training dataset is an internal human dataset; the limited data scale and diversity might restrict the generalization capability.

## Related Work & Insights

- **vs ID-Animator**: ID-Animator employs an image-like model approach for IPT2V, which only generates facial regions and struggles to control motion or backgrounds. In contrast, ConsisID designs a control scheme specifically for DiT using frequency decomposition, enabling full-body generation and abundant editing features.
- **vs InstantID**: InstantID performs identity-preserving generation in the image domain using ArcFace and pose networks. ConsisID scales this to the video domain and introduces frequency decomposition to address temporal consistency issues.
- This paper provides valuable empirical evidence for understanding the internal frequency characteristics of DiT, offering insights for future research on controlled DiT generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The concept of frequency-decomposition-based control is novel and theoretically supported, though its core components (ArcFace + CLIP + Q-Former) heavily borrow from existing works.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies are detailed, incorporating frequency-domain visualization, but the comparison is limited to only one open-source approach, ID-Animator.
- **Writing Quality**: ⭐⭐⭐⭐ The deduction of motivation is clear, presenting a coherent logical chain from discoveries to structural design.
- **Value**: ⭐⭐⭐⭐ It represents the first open-source, tuning-free IPT2V model built on DiT, offering outstanding practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EvoID: Reinforced Evolution for Identity-Preserving Video Generation](../../CVPR2026/video_generation/evoid_reinforced_evolution_for_identity-preserving_video_generation.md)
- [\[CVPR 2025\] HyperNVD: Accelerating Neural Video Decomposition via Hypernetworks](hypernvd_accelerating_neural_video_decomposition_via_hypernetworks.md)
- [\[CVPR 2025\] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing](fade_frequency-aware_diffusion_model_factorization_for_video_editing.md)
- [\[CVPR 2026\] AnyID: Ultra-Fidelity Universal Identity-Preserving Video Generation from Any Visual References](../../CVPR2026/video_generation/anyid_ultra-fidelity_universal_identity-preserving_video_generation_from_any_vis.md)
- [\[CVPR 2025\] VIRES: Video Instance Repainting via Sketch and Text Guided Generation](vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)

</div>

<!-- RELATED:END -->
