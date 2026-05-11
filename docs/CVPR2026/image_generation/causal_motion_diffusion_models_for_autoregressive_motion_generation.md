---
title: >-
  [Paper Note] Causal Motion Diffusion Models for Autoregressive Motion Generation
description: >-
  [CVPR 2026][Image Generation][Causal Diffusion] This paper proposes CMDM, a framework that unifies diffusion denoising and autoregressive generation within a motion-language-aligned causal latent space. By employing fram…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Causal Diffusion"
  - "Autoregressive Motion Generation"
  - "Text-to-Motion"
  - "Streaming Generation"
  - "Frame-wise Sampling Schedule"
date: 2026-05-08
content_hash: 893c1413977dcfcb
---

# Causal Motion Diffusion Models for Autoregressive Motion Generation

**Conference**: CVPR 2026
**arXiv**: [2602.22594](https://arxiv.org/abs/2602.22594)
**Code**: N/A
**Area**: Human Motion Generation / Diffusion Models
**Keywords**: Causal Diffusion, Autoregressive Motion Generation, Text-to-Motion, Streaming Generation, Frame-wise Sampling Schedule

## TL;DR

This paper proposes CMDM, a framework that unifies diffusion denoising and autoregressive generation within a motion-language-aligned causal latent space. By employing frame-wise independent noise and a causal uncertainty-based sampling schedule, CMDM achieves high-quality, low-latency text-to-motion generation and long-sequence streaming synthesis.

## Background & Motivation

Text-driven human motion generation must simultaneously guarantee spatial accuracy and temporal coherence. Existing approaches fall into two camps, each with its own limitations:

- **Full-sequence diffusion models** (MDM, MLD, MotionLCM, *etc.*): apply bidirectional denoising over entire sequences, yielding high quality but breaking temporal causality, thus precluding online/streaming generation.
- **Autoregressive models** (T2M-GPT, MotionStreamer, *etc.*): preserve causality but suffer from error accumulation, leading to instability on long sequences; they also rely on teacher forcing during training, causing severe exposure bias at inference time.

The root cause lies in the tension between achieving the generative fidelity of diffusion models and the causal structure of autoregressive models. CMDM resolves this by performing frame-level diffusion denoising within a semantically aligned causal latent space.

## Method

### Overall Architecture

CMDM consists of three core components: (1) a MAC-VAE encoder-decoder that maps motion sequences into a causal latent space; (2) a Causal-DiT that performs diffusion denoising in the latent space via causal attention; and (3) a Frame-wise Sampling Schedule (FSS) that accelerates inference using causal uncertainty.

### Key Designs

1. **MAC-VAE (Motion-Language-Aligned Causal VAE)**: The encoder and decoder are built with 1D causal convolutions and causal ResNet blocks, ensuring each frame depends only on preceding frames, with a temporal downsampling factor of 4. The key innovation is the introduction of a motion-language alignment loss: frame-level semantic features are extracted via a pretrained Part-TMR model, and alignment is enforced through a marginal cosine similarity loss $\mathcal{L}_{mcos}$ and a marginal distance matrix similarity loss $\mathcal{L}_{mdms}$:

    $\mathcal{L}_{\text{MAC-VAE}} = \mathcal{L}_{\text{rec}} + \beta D_{\text{KL}} + \lambda \mathcal{L}_{\text{align}}$

   where $\mathcal{L}_{\text{align}} = \mathcal{L}_{\text{mcos}} + \mathcal{L}_{\text{mdms}}$. The former minimizes feature-level cosine discrepancy, while the latter preserves the relative structural consistency of the feature space.

2. **Causal Diffusion Forcing**: Unlike conventional diffusion models that apply a uniform noise level to all frames, CMDM independently samples a noise level $k_t$ for each frame $t$ and performs denoising via causal self-attention (lower-triangular mask):

    $\mathcal{L}_{\text{DF}} = \mathbb{E}_{k_t, \epsilon_t^{k_t}} \left[ \| \epsilon_t^{k_t} - \epsilon_\theta(\tilde{\mathbf{z}}_{\leq t}, k_t, \mathbf{c}) \|_2^2 \right]$

   Each frame can only attend to past frames, thereby enforcing temporal causality within the diffusion framework. Frame-wise stochastic noise also acts as a regularizer that encourages smooth temporal transitions.

3. **Causal-DiT (Causal Diffusion Transformer)**: An 8-layer Transformer with 4-head attention and 512-dimensional hidden states, integrating three mechanisms: causal self-attention (lower-triangular mask preventing access to future frames), cross-attention (interaction with DistilBERT text embeddings), and AdaLN + RoPE (frame-level diffusion timestep embedding combined with rotary positional encoding to stabilize long-sequence denoising).

4. **Frame-wise Sampling Schedule (FSS)**: During inference, low noise is assigned to past frames and high noise to future frames. Each new frame is predicted from partially denoised preceding frames rather than waiting for the previous frame to be fully denoised. An uncertainty scale $L$ controls when the next frame begins denoising — specifically, at the point when the current frame has been denoised to step $K-L$. This substantially reduces the number of inference steps and alleviates exposure bias.

### Loss & Training

- MAC-VAE: reconstruction loss + KL divergence + motion-language alignment loss (weight $\lambda$ adaptively adjusted via gradient norm).
- Causal-DiT: causal diffusion forcing loss with Flow Matching as the ODE solver.
- Text conditioning is randomly dropped with probability 0.1 during training; classifier-free guidance (scale = 3.0) is applied at inference.

## Key Experimental Results

### Main Results

| Dataset | Metric | CMDM (FSS) | Prev. SOTA (SALAD) | Gain |
|--------|------|-----------|------------------|------|
| HumanML3D | R-Top1 | **0.588** | 0.581 | +0.007 |
| HumanML3D | FID | **0.068** | 0.076 | −0.008 |
| HumanML3D | CLIP-Score | **0.685** | 0.671 | +0.014 |
| SnapMoGen | R-Top1 | **0.831** | 0.802 (MoMask++) | +0.029 |
| SnapMoGen | FID | **14.451** | 15.061 (MoMask++) | −0.610 |

**Long-sequence generation** (compared with FlowMDM and MARDM):

| Dataset | Subsequence FID↓ | Transition AUJ↓ | Note |
|--------|-----------------|-----------------|------|
| HumanML3D CMDM | **0.12** | **0.42** | Substantially outperforms FlowMDM (0.29/0.51) |
| SnapMoGen CMDM | **32.49** | 70.35 | Subsequence quality far exceeds MARDM (40.80) |

### Ablation Study

| Configuration | R-Top1 | FID | Transition AUJ |
|------|--------|-----|----------------|
| Full CMDM | 0.588 | 0.068 | 0.42 |
| Standard VAE (w/o language alignment) | 0.561 | 0.107 | 0.52 |
| C-VAE (w/o language alignment) | 0.575 | 0.070 | 0.44 |
| Full-sequence diffusion (w/o causality) | 0.591 | 0.071 | **0.72** |
| w/o AdaLN | 0.583 | 0.076 | 0.47 |
| w/o RoPE | 0.581 | 0.087 | 0.51 |
| FSS K=50, L=5 | 0.583 | 0.077 | **0.38** |

### Key Findings

- **Remarkable efficiency**: CMDM has only 114M parameters and achieves 125 fps in FSS mode (vs. MARDM at 310M/20 fps and MotionStreamer at 318M/11 fps), representing an order-of-magnitude speedup.
- Full-sequence diffusion yields marginally higher R-Top1 on single-step T2M, but its transition AUJ nearly doubles (0.72 vs. 0.42), confirming that causal diffusion is critical for long-sequence coherence.
- The language alignment in MAC-VAE primarily improves semantic consistency rather than low-level motion quality.
- $L=5$ in FSS produces the smoothest transitions (AUJ = 0.38) at a slight cost to semantic accuracy.

## Highlights & Insights

- The paper successfully transfers Diffusion Forcing from next-token prediction to motion generation, unifying the diffusion and autoregressive paradigms through frame-wise independent noise.
- FSS is a highly practical inference acceleration strategy: by controlling the propagation of "uncertainty cascades" along the causal chain, it offers a flexible trade-off between speed and quality.
- The semantic alignment supervision in MAC-VAE endows the latent space with both causal structure and semantic meaningfulness; this dual-constraint design is worth emulating.
- With 2–3× fewer parameters than competing methods yet superior performance, the results suggest that architectural design matters more than model scale.

## Limitations & Future Work

- For highly abstract or ambiguous text descriptions, performance is bounded by the quality of the pretrained motion-language model (Part-TMR).
- Very long sequences (on the order of minutes) may still accumulate subtle temporal artifacts, necessitating motion-aware feedback or adaptive anchoring mechanisms.
- The current framework supports only single-person motion and has not been extended to multi-person interaction scenarios.
- The optimal combination of $K$ and $L$ in FSS requires task-specific hyperparameter tuning.

## Related Work & Insights

- **Diffusion Forcing** [Chen et al., 2024] is the inspiration for the frame-wise independent noise design; however, the original formulation targets next-token prediction, whereas CMDM extends it to continuous motion spaces.
- **MLD/MotionLCM** demonstrate strong results by performing full-sequence diffusion in latent space, but CMDM shows that causal constraints can yield streaming capability without sacrificing quality.
- **MARDM/MotionStreamer** adopt masked autoregressive architectures with diffusion heads but incur large parameter counts and slow inference.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First framework to unify diffusion and autoregressive generation in a motion-language-aligned causal latent space.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two benchmarks + long-sequence evaluation + comprehensive ablations across VAE, diffusion, and sampling dimensions + efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, complete mathematical derivations, and sufficient experimental detail.
- **Value**: ⭐⭐⭐⭐⭐ Strong practical utility for real-time motion generation; 125 fps inference speed has direct application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](../../ICCV2025/image_generation/motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[CVPR 2026\] Pixel Motion Diffusion Is What We Need for Robot Control](pixel_motion_diffusion_is_what_we_need_for_robot_control.md)
- [\[CVPR 2026\] EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation](egoflow_gradient-guided_flow_matching_for_egocentric_6dof_object_motion_generati.md)
- [\[CVPR 2026\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_textguided_multihuman_3d_moti.md)
- [\[CVPR 2026\] DynaVid: Learning to Generate Highly Dynamic Videos using Synthetic Motion Data](dynavid_learning_to_generate_highly_dynamic_videos_using_synthetic_motion_data.md)

</div>

<!-- RELATED:END -->
