---
title: >-
  [Paper Note] MagicMirror: ID-Preserved Video Generation in Video Diffusion Transformers
description: >-
  [ICCV 2025][Video Generation][Identity-preserving video generation] MagicMirror is the first framework to achieve zero-shot identity-preserving video generation on a Video Diffusion Transformer (CogVideoX). It employs dual-branch facial feature extraction, Conditioned Adaptive Normalization (CAN), and a two-stage training strategy (image pre-training followed by video fine-tuning) to generate high-quality dynamic videos while maintaining consistent facial identity.
tags:
  - ICCV 2025
  - Video Generation
  - Identity-preserving video generation
  - Diffusion Transformer
  - facial feature extraction
  - adaptive normalization
  - two-stage training
date: 2026-05-08
content_hash: f9dac52104b3136e
---

# MagicMirror: ID-Preserved Video Generation in Video Diffusion Transformers

**Conference**: ICCV 2025
**arXiv**: [2501.03931](https://arxiv.org/abs/2501.03931)
**Code**: [https://github.com/dvlab-research/MagicMirror/](https://github.com/dvlab-research/MagicMirror/)
**Area**: Diffusion Models
**Keywords**: Identity-preserving video generation, Diffusion Transformer, facial feature extraction, adaptive normalization, two-stage training

## TL;DR

MagicMirror is the first framework to achieve zero-shot identity-preserving video generation on a Video Diffusion Transformer (CogVideoX). It employs dual-branch facial feature extraction, Conditioned Adaptive Normalization (CAN), and a two-stage training strategy (image pre-training followed by video fine-tuning) to generate high-quality dynamic videos while maintaining consistent facial identity.

## Background & Motivation

**Background**: Diffusion models have achieved remarkable success in text-to-image generation. Identity-preserving (ID-preserving) image generation methods such as PhotoMaker, InstantID, and PuLID have enabled the preservation of specific subject identities without fine-tuning. However, this capability remains underdeveloped in the video generation domain.

**Limitations of Prior Work**: Existing ID-preserving video generation methods exhibit two categories of limitations: (1) fine-tuning-based methods such as MagicMe require per-identity optimization, leading to low efficiency and poor generalizability; (2) inflated-UNet-based methods such as ID-Animator are constrained by the capacity of the backbone model, producing videos with limited motion dynamics where facial expressions are nearly static—essentially a "copy-paste" of the reference face with no natural facial movement. Another class of two-stage methods first generates a personalized reference image and then performs image-to-video (I2V) generation, resulting in poor identity stability over long sequences.

**Key Challenge**: State-of-the-art video generation models (e.g., CogVideoX) are built on full-attention DiT architectures, which are incompatible with the conventional cross-attention-based conditioning paradigm. DiTs employ layer-wise modulation rather than standalone cross-attention layers, making the integration of identity conditions non-trivial. Furthermore, high-quality identity–video paired training data is extremely scarce.

**Goal**: To achieve tuning-free ID-preserving video generation on a Video DiT architecture while producing dynamic and natural facial motion.

**Key Insight**: Leverage the existing layer-wise modulation mechanism in CogVideoX to design a lightweight identity-conditioning adapter; address data scarcity through synthetic data construction and progressive training.

**Core Idea**: Design a Conditioned Adaptive Normalization (CAN) module to predict identity-aware distribution shifts, combined with dual-branch facial feature extraction (high-level identity features + structural detail features), to enable efficient fusion of identity information within the DiT.

## Method

### Overall Architecture

MagicMirror is built upon CogVideoX-5B. The inputs consist of one or more reference face images and a text prompt. A dual-branch feature extractor on the left side extracts identity embeddings and facial structure embeddings separately. These embeddings are injected into alternating DiT layers via cross-modal adapters comprising CAN and decoupled cross-attention. Training follows two stages: identity-preserving capability is first pre-trained on image data, followed by fine-tuning on video data for temporal consistency.

### Key Designs

1. **Decoupled Facial Feature Extraction**:

    - **Function**: Simultaneously captures high-level identity semantics and fine-grained facial structural details.
    - **Mechanism**: Dense feature maps $\mathbf{f}$ are extracted from a pre-trained CLIP ViT. The ID branch applies ArcFace to extract high-level identity features $\mathbf{q}_{id}$, performs cross-attention over $\mathbf{f}$ via a Q-Former architecture to obtain $x_{id}$, and maps the result to the text embedding space via a fusion MLP, replacing text embeddings at identity-relevant token positions. The Face branch employs 32 learnable query tokens $\mathbf{q}_{face}$ and a separate Q-Former to extract facial structure features $x_{face}$ from $\mathbf{f}$, which are subsequently used in full-attention and cross-attention layers.
    - **Design Motivation**: A single identity embedding is insufficient to simultaneously preserve structural attributes such as hairstyle and face shape alongside identity characteristics. The decoupled design allows ID features to guide semantics through the text channel while facial structure features provide fine-grained reference through the attention channel.

2. **Conditioned Adaptive Normalization (CAN)**:

    - **Function**: Efficiently injects identity conditioning into the distribution modulation of the DiT.
    - **Mechanism**: CogVideoX already contains layer-wise modulation modules $\varphi_{txt}$ and $\varphi_{vid}$ for the text and video modalities, respectively, each predicting the corresponding scale/shift/gate parameters. MagicMirror introduces an additional facial modality modulation module $\varphi_{face}$ to process facial features. The key innovation is the CAN module $\varphi_{cond}$, which takes the timestep embedding $\mathbf{t}$, layer index $l$, video modulation factor $\mu_{vid}^1$, and ID embedding $x_{id}$ as conditions, and predicts distribution shifts $\hat{m}_{vid}$ and $\hat{m}_{txt}$ applied to the video and text modalities. The final modulation factors are obtained via residual addition: $m_{vid} = \hat{m}_{vid} + \varphi_{vid}(\mathbf{t}, l)$.
    - **Design Motivation**: Directly appending cross-attention layers has limited effect in full-attention DiTs, where conditioning is primarily exercised through distribution modulation. CAN allows identity information to directly influence the distributions of text and video features, accelerating convergence and improving identity fidelity. Ablation results show that without CAN, the model fails to learn even basic hairstyle attributes.

3. **Two-Stage Progressive Training Strategy**:

    - **Function**: Addresses the scarcity of identity–video paired training data.
    - **Mechanism**: Stage 1 pre-trains on diverse image data (LAION-Face 50K + SFHQ + FFHQ synthetic pairs) to learn robust identity-preserving capability, training for 30K steps with a batch size of 64. Stage 2 fine-tunes on high-quality video data (Pexels + Mixkit + a small amount of self-collected video) to enhance temporal consistency, training for 5K steps with a batch size of 8. Synthetic data is generated using PhotoMakerV2 to produce diverse-pose images of the same identity, filtered by ArcFace similarity > 0.65.
    - **Design Motivation**: Direct training on video data suffers from insufficient data volume and limited identity diversity. Image pre-training first establishes strong identity embedding capability, which is then transferred to the temporal domain during video fine-tuning. Training on image data alone causes color-shift artifacts during video inference; the two-stage strategy resolves modulation factor inconsistencies across training stages.

### Loss & Training

The loss function comprises a denoising loss and an identity-aware loss: $\mathcal{L} = \mathcal{L}_{noise} + \lambda (1 - \cos(q_{face}, D(x_0)))$, where $D(\cdot)$ denotes decoding the denoised latent. For 50% of training samples, the denoising loss is computed exclusively within the facial region.

## Key Experimental Results

### Main Results

Quantitative comparison with I2V and ID-preserving methods (using VBench and custom metrics):

| Method | Face Similarity↑ | Dynamics↑ | Prompt Consistency↑ | IS↑ | Face Motion (FM_ref)↑ | User Preference↑ |
|--------|-----------------|-----------|---------------------|------|----------------------|-----------------|
| DynamiCrafter | 0.455 | 0.168 | 8.20 | 0.896 | 0.237 | 5.87 |
| CogVideoX-I2V | 0.660 | 0.213 | 9.85 | 0.901 | 0.413 | 6.22 |
| ID-Animator | 0.140 | 0.211 | 7.57 | 0.923 | 0.652 | 5.63 |
| **MagicMirror** | **0.705** | **0.240** | **10.59** | 0.911 | **0.704** | **6.97** |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| w/o Face branch | Loss of structural detail guidance; significant degradation in identity fidelity |
| w/o CAN | Convergence difficulty; fails to learn even hairstyle during pre-training |
| Image training only | Color-shift artifacts during video inference |
| Video training only | Weak identity-preserving capability |
| Full two-stage | Optimal; high-fidelity identity + dynamic facial motion |

Computational overhead comparison:

| Model | GPU Memory | Parameters | Inference Time (49 frames, 480P) |
|-------|-----------|------------|----------------------------------|
| CogVideoX-5B | 24.9 GiB | 10.5B | 204s |
| MagicMirror | 28.6 GiB | 12.8B | 209s |

### Key Findings

- **CAN is critical for convergence**: Without CAN, the model fails to fit basic appearance attributes during image pre-training; adding CAN substantially improves both convergence speed and quality.
- **Significant advantage in facial motion**: MagicMirror achieves an FM_ref of 0.704, far exceeding CogVideoX-I2V's 0.413, demonstrating that the generated videos contain genuinely dynamic facial expressions.
- **Minimal computational overhead**: Only 2.3B additional parameters (mostly in the feature extractor, requiring a single forward pass) and 5 seconds of additional inference time.
- **User study superiority across all dimensions**: MagicMirror achieves the highest ratings in motion dynamics, text alignment, video quality, and identity consistency.

## Highlights & Insights

- **Design philosophy of CAN**: Rather than naively appending a cross-attention branch, CAN leverages the DiT's existing distribution modulation mechanism to inject identity information by predicting identity-conditioned distribution shifts. This "architecture-conforming" adaptation strategy is more elegant and efficient than forceful grafting, offering inspiration for other DiT adaptation tasks.
- **Synthetic data pipeline**: Diverse reference images of the same identity are generated using PhotoMakerV2 to construct training pairs, filtered by strict ArcFace similarity thresholds—a practical and scalable data production strategy.
- **Average Similarity metric**: The paper proposes evaluating ID preservation using average similarity to the reference image set rather than to a single image, avoiding inflated scores from "copy-paste" behavior.

## Limitations & Future Work

- Multi-subject identity customization is not supported; the current framework handles single-person scenarios only.
- The method focuses primarily on facial identity features; preservation of fine-grained attributes such as clothing and accessories is limited.
- Built on CogVideoX-5B, the approach is constrained by the backbone model's generation quality ceiling and maximum video duration.
- The method poses deepfake risks; societal implications and portrait rights protection warrant attention.

## Related Work & Insights

- **vs. ID-Animator**: Uses an inflated UNet with a face adapter, producing videos with nearly static facial expressions. MagicMirror, built on a more advanced DiT architecture, achieves more than 3× greater facial motion magnitude.
- **vs. MagicMe**: Requires per-identity fine-tuning, whereas MagicMirror operates in a zero-shot manner with substantially higher efficiency.
- **vs. IP-Adapter / PhotoMaker (image-domain methods)**: These methods are well-established in the image domain. MagicMirror successfully extends their paradigm to the Video DiT domain for the first time; the key breakthrough lies in the CAN-based conditioning injection mechanism.

## Rating

- Novelty: ⭐⭐⭐⭐ First zero-shot ID-preserving video generation on Video DiT; CAN design is elegant and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage including quantitative metrics, user study, ablation study, distribution visualization, and computational overhead analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear paper structure; the Appendix provides rich supplementary details.
- Value: ⭐⭐⭐⭐⭐ A milestone in personalized video generation; highly practical methodology with publicly available code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)
- [\[ICCV 2025\] Versatile Transition Generation with Image-to-Video Diffusion](versatile_transition_generation_with_image-to-video_diffusion.md)
- [\[CVPR 2026\] DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](../../CVPR2026/video_generation/disca_accelerating_video_diffusion_transformers_wi.md)
- [\[CVPR 2026\] I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers](../../CVPR2026/video_generation/interpretable_motion-attentive_maps_spatio-temporally_localizing_concepts_in_vid.md)
- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)

</div>

<!-- RELATED:END -->
