---
title: >-
  [Paper Note] EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis
description: >-
  [ECCV2024][Human Understanding][talking head generation] Proposes EDTalk, an efficient disentanglement framework based on learnable orthogonal basis vectors, which decomposes facial dynamics into three independent latent spaces: mouth shape, head pose, and emotional expression. It simultaneously supports both video-driven and audio-driven emotional talking head generation.
tags:
  - "ECCV2024"
  - "Human Understanding"
  - "talking head generation"
  - "facial disentanglement"
  - "emotional expression"
  - "orthogonal bases"
  - "audio-driven"
date: 2026-05-08
content_hash: 4c9867ddff1f78f6
---

# EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis

**Conference**: ECCV2024  
**arXiv**: [2404.01647](https://arxiv.org/abs/2404.01647)  
**Code**: [tanshuai0219/EDTalk](https://github.com/tanshuai0219/EDTalk)  
**Area**: Audio and Speech  
**Keywords**: talking head generation, facial disentanglement, emotional expression, orthogonal bases, audio-driven

## TL;DR

Proposes EDTalk, an efficient disentanglement framework based on learnable orthogonal basis vectors, which decomposes facial dynamics into three independent latent spaces: mouth shape, head pose, and emotional expression. It simultaneously supports both video-driven and audio-driven emotional talking head generation.

## Background & Motivation

Talking head generation has wide applications in education, film/television, virtual digital humans, and other fields. Existing methods primarily suffer from three limitations:

1. **Holistic Generation, Lacking Fine-Grained Control**: Most methods generate talking head videos in a holistic manner, making it impossible to independently control mouth shape, head pose, and emotional expression.
2. **Single Driving Modality**: Current works typically support only audio-driven or video-driven generation, limiting multi-modal application scenarios.
3. **Inefficient Disentanglement**: Existing facial disentanglement methods either over-rely on external priors (such as 3D reconstruction coefficients or additional audio data for contrastive learning), lack intrinsic constraints between spaces leading to incomplete disentanglement, or require training the entire heavyweight network from scratch when disentangling a new subspace.

The authors argue that an ideal disentangled space should satisfy two conditions: (a) the spaces should be mutually disjoint, with each space capturing only the motion of its corresponding component without interference from others; (b) the spaces disentangled from video data should be storable and shareable with audio inputs.

## Core Problem

How to efficiently decompose facial motion into three mutually non-interfering independent latent spaces (mouth shape, head pose, and emotional expression) without relying on external prior information, and achieve dual-modality driving (video/audio)?

## Method

### Overall Architecture

EDTalk is based on an autoencoder architecture, consisting of an encoder $E$, three Component-aware Latent Navigation (CLN) modules, and a generator $G$. Given an identity image $I^i$ and different driving sources $I^m, I^p, I^e$ (controlling mouth shape, pose, and expression, respectively), the encoder maps the images to latent features, while the CLN modules convert them into corresponding motion features.

**Core Design — Learnable Orthogonal Basis Vectors**: Each CLN module maintains a bank of learnable basis vectors $B^* = \{b_1^*, \dots, b_n^*\}$. Motion is represented as a linear combination of these basis vectors:

$$f^{r \to *} = \sum_{i=1}^{n} w_i^* b_i^*$$

where the weight $W^* = \text{MLP}^*(f^{* \to r})$ is predicted from the latent features by a lightweight MLP. The key constraints are:
- **Intra-space Orthogonality**: Different basis vectors within the same bank are orthogonal to each other $\langle b_i^*, b_j^* \rangle = 0 \ (i \neq j)$.
- **Inter-space Orthogonality**: The basis vectors among the three banks ($B^m, B^p, B^e$) are also mutually orthogonal.

This guarantees independent control over different facial components, and features from the three spaces can be directly summed to obtain the complete driving feature.

### Efficient Disentanglement Strategy

**Stage 1 — Mouth and Pose Disentanglement**: Assumes a cross-reconstruction strategy. For two frames $I^a$ and $I^b$, the mouth regions are swapped to generate synthetic images. Subsequently, PLN and MLN extract pose and mouth shape features respectively, which are cross-combined to reconstruct the original images. Training utilizes reconstruction loss $\mathcal{L}_{rec}$, perceptual loss $\mathcal{L}_{per}$, adversarial loss $\mathcal{L}_{adv}$, and a feature-level cosine similarity constraint $\mathcal{L}_{fea}$. Upon convergence, the parameters are frozen and no longer updated.

**Stage 2 — Expression Disentanglement**: Adopts self-reconstruction complementary learning. Utilizing the frozen $E$, MLN, PLN, and $G$ from the first stage, mouth and pose information are extracted from the driving image to generate intermediate results (which lack expression). The newly introduced ELN module is forced to learn the complementary expression information to complete reconstruction. Meanwhile, a lightweight Emotion Enhancement Module (EEM) is introduced to inject expression features into identity features via AdaIN operations. This stage only trains ELN and EEM, making it extremely efficient.

### Audio-to-Motion Module

Once disentanglement is complete, the basis vectors stored in the three banks serve as visual priors for audio-driven tasks:

1. **Audio-driven Lip Generation**: The audio encoder $E_a$ extracts audio features, and an MLP predicts the mouth bank weight $\hat{W}^m$. Training utilizes feature loss + reconstruction loss + SyncNet synchronization loss.
2. **Normalizing-Flow-based Probabilistic Pose Generation**: Normalizing Flow is used to model the one-to-many mapping from audio to head pose, sampling from a Gaussian distribution to generate diverse head movements that match the audio rhythm.
3. **Semantic-aware Expression Generation**: Fuses speech emotion features extracted by HuBERT and text emotion features extracted by EmoBERTa to predict the expression weight $\hat{W}^e$. Random masking of a single modality is applied during training to support inference when only audio or text is available.

## Key Experimental Results

Evaluation is performed on the MEAD and HDTF datasets, comparing against 13 SOTA methods:

| Metric | EDTalk-A (Audio-driven) | EDTalk-V (Video-driven) | Best Compared Method |
|------|---------------------|---------------------|-------------|
| PSNR (MEAD) | 21.628 | **22.771** | PD-FGC: 21.520 |
| SSIM (MEAD) | **0.722** | **0.769** | StyleTalk: 0.714 |
| M-LMD (MEAD) | **1.537** | **1.102** | PD-FGC: 1.571 |
| FID (MEAD) | **17.698** | **15.548** | EAT: 21.465 |
| Acc_emo (MEAD) | 67.32% | **68.85%** | EAT: 64.40% |
| PSNR (HDTF) | **25.156** | **26.504** | PD-FGC: 23.142 |

**Training Efficiency Comparison** (Key Highlight):

- EDTalk mouth-pose disentanglement: 15.8h of data, 2×3090 GPUs, 4K iterations, approx. **1 hour**
- DPE: 351h of data, 8×V100 GPUs, 150K iterations, more than **2 days**
- PD-FGC: Lip disentanglement 2 days + pose disentanglement 2 days + expression disentanglement 2 weeks (4×V100 GPUs)

**User Study** (Scored by 20 participants, scale of 1-5): EDTalk achieves the best performance in lip sync (4.13), realism (4.92), and emotional accuracy (64.5%).

**Ablation Study Verification**:
- Removing bank $\to$ disentanglement fails and image quality drops significantly (PSNR: 20.302 vs 21.628).
- Removing orthogonal constraints $\to$ inter-space interference occurs, with emotional accuracy dropping to 38.71% (vs 67.32%).
- Removing EEM $\to$ expression expressiveness declines (Acc_emo: 49.37% vs 67.32%).

## Highlights & Insights

1. **Latent Space Design with Orthogonal Basis Vectors**: Uses learnable basis vectors to represent facial motion directions. The orthogonal constraint elegantly ensures spatial independence, which is simpler and more effective than external constraints like contrastive learning.
2. **Extremely High Training Efficiency**: The progressive training strategy trains only lightweight modules at each stage. Mouth-pose disentanglement requires only 1 hour, which is two orders of magnitude faster than PD-FGC.
3. **Unified Support for Dual-Modality Driving (Audio & Video)**: The basis vectors in the banks serve as shared visual priors, naturally bridging the transfer from video-driven to audio-driven generation.
4. **First to Automatically Infer Expression from Audio Semantics**: Generates content-consistent emotional expressions directly from speech intonation and textual content without requiring external expression reference images/videos.

## Limitations & Future Work

1. Disentanglement is still based on three fixed components (mouth, pose, expression) as the level of granularity, without considering finer-grained facial Action Units (AUs) like eye movements or frowning.
2. The mouth region swap relies on image-level operations (overlaying the mouth region), which may not be robust under extreme poses or occlusions.
3. Semantic emotional perception relies heavily on pre-trained models (HuBERT, EmoBERTa), and its generalization capability to emotional categories not covered in the training data remains to be validated.
4. Normalizing Flow is used for head pose generation, which may introduce motion drift or unnatural periodicity in extremely long videos.
5. While training is efficient, it still requires multiple stages; an end-to-end joint training scheme is worth exploring.

## Related Work & Insights

| Method | Disentanglement Granularity | External Prior Dependence | Training Efficiency | Expression Control | Dual-modality Support |
|------|---------|------------|---------|---------|----------|
| PC-AVS | Mouth + Pose | Contrastive learning + 6D pose | Medium | ✗ | ✗ |
| PD-FGC | Mouth + Pose + Expression | Contrastive learning + 3DMM | Extremely Low (2 weeks+) | ✓ | ✗ |
| DPE | Pose + Expression | Bidirectional cyclic training | Low (2 days+) | ✓ | ✗ |
| EAT | N/A | Discrete emotion labels | Medium | Coarse-grained | ✗ |
| **EDTalk** | **Mouth + Pose + Expression** | **No external priors** | **High (1h+6h)** | **Fine-grained** | **✓** |

The critical advantage of EDTalk is that it does not rely on external prior information (e.g., 3DMM, extra audio data), achieving complete disentanglement purely through orthogonal constraints and progressive training.

## Related Work & Insights

- The idea of using orthogonal basis vectors to represent motion spaces can be transferred to other generation tasks requiring disentangled control (e.g., body motion generation, gesture synthesis).
- The progressive disentanglement training strategy (disentangling major components first, then using complementary learning to disentangle residual components) has general applicability.
- The concept of using a bank as a discretized visual prior is similar to the codebook in VQ-VAE, and can be further integrated with quantization methods.
- The Normalizing Flow scheme for probabilistic pose generation can inspire other one-to-many mapping tasks (such as audio-driven gesture generation).

## Rating
- Novelty: ⭐⭐⭐⭐ — Orthogonal basis vector disentanglement and the efficient training strategy are the main innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Compares against 13 methods, including quantitative, qualitative, ablation studies, user studies, and efficiency analyses.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, complete method description, and rich figures and tables.
- Value: ⭐⭐⭐⭐ — A unified framework that achieves efficient disentanglement + dual-modality driving + emotional generation for the first time.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos](avatar_fingerprinting_for_authorized_use_of_synthetic_talking-head_videos.md)
- [\[ICCV 2025\] GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation](../../ICCV2025/human_understanding/ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)
- [\[ECCV 2024\] Audio-Driven Talking Face Generation with Stabilized Synchronization Loss](audio-driven_talking_face_generation_with_stabilized_synchronization_loss.md)
- [\[CVPR 2026\] TriLite: Efficient WSOL with Universal Visual Features and Tri-Region Disentanglement](../../CVPR2026/human_understanding/trilite_efficient_weakly_supervised_object_localization_with_universal_visual_fe.md)
- [\[CVPR 2026\] SyncDreamer: Controllable and Expressive Avatar Generation Beyond the Talking Head](../../CVPR2026/human_understanding/syncdreamer_controllable_and_expressive_avatar_generation_beyond_the_talking_hea.md)

</div>

<!-- RELATED:END -->
