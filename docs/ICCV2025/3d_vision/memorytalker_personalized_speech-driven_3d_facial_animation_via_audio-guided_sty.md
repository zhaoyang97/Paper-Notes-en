---
title: >-
  [Paper Note] MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization
description: >-
  [3D Vision] This paper proposes MemoryTalker, a two-stage training framework (Memorizing + Animating) that employs a key-value memory network to store generic facial motions and generates personalized 3D facial animation…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: a5706a39d705891a
---

# MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization

- **Conference**: ICCV 2025
- **arXiv**: [2507.20562](https://arxiv.org/abs/2507.20562)
- **Area**: 3D Vision
- **Keywords**: Speech-Driven 3D Facial Animation, Memory Network, Speaking Style, Personalization, Key-Value Memory

## TL;DR

This paper proposes MemoryTalker, a two-stage training framework (Memorizing + Animating) that employs a key-value memory network to store generic facial motions and generates personalized 3D facial animations driven solely by audio via audio-guided stylized memory retrieval, requiring no additional prior information at inference time.

## Background & Motivation

Speech-driven 3D facial animation aims to synthesize facial motion sequences from speech signals that match the speaker's individual style, and is a critical technology for immersive applications such as VR telepresence and character animation. The core challenge lies not only in achieving precise speech-motion synchronization, but also in capturing the personal speaking styles of different individuals (e.g., mouth opening amplitude, lip protrusion degree).

Existing methods exhibit two major limitations:

**One-hot encoding methods** (FaceFormer, CodeTalker, etc.): Speaker identity vectors from the training set are used to represent style, making it impossible to handle unseen speakers at inference time. Moreover, applying different IDs to the same audio yields different results, lacking generalizability.

**Reference 3D mesh methods** (Imitator, Mimic, Yang et al.): These methods require an additional 3D facial motion sequence from the target speaker to encode style at inference time, which is highly impractical to obtain in real-world applications.

The core goal of this paper is: **to generate 3D facial animations reflecting a speaker's personal style using only audio input, without any additional priors (ID labels or reference 3D meshes)**. This is the first personalized speech-driven 3D facial animation method that requires no additional prior information at inference time.

## Method

### Overall Architecture

MemoryTalker adopts a two-stage training strategy:

- **Stage 1 (Memorizing)**: A facial motion memory network is constructed to store and retrieve generic facial motions corresponding to speech.
- **Stage 2 (Animating)**: The model learns to extract speaking style features from audio, stylizing the generic motion memory into personalized memory to generate individualized facial animations.

### Key Designs

**1. Motion Memory Network**

A key-value memory $\mathbf{M}_m \in \mathbb{R}^{n \times c}$ is designed with $n$ slots and $c$-dimensional channels:

- **Writing**: A motion encoder $E_m$ encodes facial motion $v^t$ into feature $f_m^t$. Attention-based similarity scores with each slot are computed as value address vectors $\mathbf{V}_m^t$, and a weighted sum yields the recalled feature $\hat{f}_{m,val}^t$.
- **Reading**: Text representations $f_{txt}^t$ extracted by a pretrained ASR model (HuBERT) serve as query keys. After softmax normalization, key address vectors $\mathbf{K}_{txt}^t$ are obtained to retrieve generic motion features $\hat{f}_{m,key}^t$ from memory.

The key motivation for using text representations rather than raw audio features as queries is that audio styles vary significantly across speakers for the same phoneme, whereas text representations can neutralize such variation and map to consistent facial motions (e.g., all speakers saying "who" first close and then round their lips).

**2. Stylized Motion Memory**

Speaking style features $f_s$ are extracted from the mel-spectrogram of the audio via a style encoder $E_s$. Style weights $\tilde{w}_s$ are designed to re-weight each memory slot:

$$\tilde{w}_s = \text{sigmoid}(\psi'_{\rightarrow n}(f_s)) \cdot \psi_{\rightarrow 1}(f_s)$$

where the sigmoid component scores each slot and a scalar scaling factor controls the overall magnitude. The generic memory $\mathbf{M}_m$ is transformed into stylized memory $\tilde{\mathbf{M}}_m$ via these style weights, enabling different speakers to produce facial motions of varying amplitude for the same phoneme.

**3. Motion Decoder**

Based on a Transformer decoder architecture, the model takes the concatenation of text representations and retrieved motion features as input to generate the final 3D facial motion:

$$\hat{v}^t = D_m([f_{txt}^t; \hat{f}_{m,key}^t], f_{txt}^t)$$

### Loss & Training

**Stage 1**:
$$\mathcal{L}_{1\text{-stage}} = \mathcal{L}_{mse} + \mathcal{L}_{vel} + \lambda_1(\mathcal{L}_{mem} + \mathcal{L}_{align})$$

- $\mathcal{L}_{mse}$: Motion reconstruction loss (L2 distance between prediction and ground truth)
- $\mathcal{L}_{vel}$: Velocity loss (mitigates inter-frame jitter)
- $\mathcal{L}_{mem}$: Memory reconstruction loss (ensures motion information is written into memory)
- $\mathcal{L}_{align}$: KL divergence alignment loss (aligns text key addresses with motion value addresses)
- $\lambda_1 = 0.01$

**Stage 2** (all Stage 1 parameters are frozen; only the style encoder is trained):
$$\mathcal{L}_{2\text{-stage}} = \mathcal{L}_{mse} + \mathcal{L}_{vel} + \lambda_2(\mathcal{L}_{lip} + \mathcal{L}_{style})$$

- $\mathcal{L}_{lip}$: Lip vertex loss (focuses on fine-grained motion in the lower facial region)
- $\mathcal{L}_{style}$: Triplet loss (pulls together style features from the same speaker and pushes apart those from different speakers)
- $\lambda_2 = 0.01$

## Key Experimental Results

### Main Results (VOCASET Dataset)

| Method | FVE↓(×10⁻⁶) | LVE↓(×10⁻⁵) | FID↓(×10⁻¹) | LDTW↓(×10⁻⁵) | Lip-max↓(×10⁻⁴) |
|--------|-------------|-------------|-------------|--------------|-----------------|
| FaceFormer | 0.639 | 0.413 | 3.583 | 0.507 | 0.452 |
| CodeTalker | 0.721 | 0.498 | 3.713 | 0.554 | 0.484 |
| SelfTalk | 0.593 | 0.382 | 3.279 | 0.475 | 0.416 |
| UniTalker | 0.570 | 0.382 | 3.256 | 0.507 | 0.407 |
| **MemoryTalker** | **0.506** | **0.293** | **3.045** | **0.418** | **0.331** |

MemoryTalker achieves state-of-the-art performance on all metrics on VOCASET, reducing FVE by 11.2% and LVE by 23.3%.

### BIWI Dataset

| Method | FVE↓(×10⁻⁴) | LVE↓(×10⁻⁴) | FID↓(×10⁻¹) |
|--------|-------------|-------------|-------------|
| UniTalker | 0.919 | 0.196 | 7.234 |
| **MemoryTalker** | **0.901** | **0.187** | **7.202** |

MemoryTalker also achieves the best results on the cross-dataset evaluation.

### Efficiency Comparison

| Method | Inference Time | Parameters |
|--------|---------------|------------|
| CodeTalker | 297.6 ms | 315M |
| SelfTalk | 10.1 ms | 450M |
| UniTalker | 9.7 ms | 313M |
| **MemoryTalker** | **7.8 ms** | **94M** |

MemoryTalker achieves approximately 120 fps inference speed with only 94M parameters, combining both efficiency and performance.

### Ablation Study

| Configuration | FVE↓ | LVE↓ |
|--------------|------|------|
| w/o Memory Network (baseline) | 0.638 | 0.460 |
| + Stage 1 (Memorizing) | 0.531 | 0.313 |
| + Stage 2 (Stylizing) | **0.506** | **0.293** |

Both stages contribute meaningfully. Removing $\mathcal{L}_{style}$ (triplet loss) leads to unstable training and significant performance degradation, underscoring the importance of style discrimination.

### User Study

In a comparison with five state-of-the-art methods (33 participants), MemoryTalker achieves preference rates exceeding 79% on all three dimensions: lip synchronization, realism, and speaking style.

### Key Findings

- **t-SNE Visualization**: Retrieved motion features from Stage 1 are mixed across speakers (generic motion), while Stage 2 features cluster clearly by speaker (personalized style is successfully captured).
- **Inherent limitations of one-hot methods**: Applying different training-set IDs to the same audio yields FVE standard deviations of 0.036 for FaceFormer and 0.056 for CodeTalker, demonstrating that one-hot encoding cannot stably represent speaking style.

## Highlights & Insights

1. **First method to achieve personalization using audio alone**: No ID labels or reference 3D meshes are required, greatly improving practical applicability.
2. **Key-value memory bridges the modality gap**: Text representations are cleverly used as keys to eliminate style variation in audio, with personalization injected via stylization weights.
3. **Two-stage decoupled design**: Learning generic motion before personalization avoids conflicts between the two objectives.
4. **Extreme efficiency**: 94M parameters and 7.8 ms inference make the method suitable for real-time VR and metaverse deployment.
5. **Style weight design**: Sigmoid scoring × scalar scaling concisely and effectively controls the degree of personalization for each memory slot.

## Limitations & Future Work

1. Only mouth and facial motion are modeled; upper-face regions such as eye movements and eyebrow expressions are not covered.
2. The method depends on the quality of the pretrained ASR model (HuBERT), and generalization to non-English speech has not been validated.
3. The experimental datasets are relatively small in scale (VOCASET contains only 12 speakers and 480 sequences), raising questions about large-scale generalizability.
4. Emotional speech is not addressed; the method targets only neutral speaking styles.
5. The selection of the number of memory slots $n$ lacks systematic investigation.

## Related Work & Insights

- **One-hot methods**: FaceFormer and CodeTalker encode style using training-set speaker IDs, which cannot generalize to new speakers.
- **Reference motion methods**: Imitator (reference 2D video), Mimic (reference 3D motion for style-content disentanglement), and Yang et al. (progressive style injection) all require additional inputs at inference time.
- **Memory networks**: Widely applied in object tracking, few-shot learning, and anomaly detection; this paper is the first to apply key-value memory networks to cross-modal speech-to-3D-motion alignment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first method to achieve personalized 3D facial animation from audio alone; the cross-modal bridging design via memory networks is novel.
- **Practicality**: ⭐⭐⭐⭐⭐ — No additional priors, extremely low latency, and a small parameter footprint make it directly deployable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive quantitative, qualitative, and user study evaluation with thorough ablations, though dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear, figures are intuitive, and motivation is well-argued.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Identity Preserving 3D Head Stylization with Multiview Score Distillation](identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)
- [\[ICCV 2025\] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)
- [\[ICCV 2025\] DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness](dso_aligning_3d_generators_with_simulation_feedback_for_physical_soundness.md)

</div>

<!-- RELATED:END -->
