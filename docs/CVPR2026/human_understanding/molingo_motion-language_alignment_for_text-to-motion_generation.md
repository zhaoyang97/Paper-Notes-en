---
title: >-
  [Paper Note] MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation
description: >-
  [CVPR 2026][Human Understanding][text-driven motion generation] MoLingo achieves comprehensive state-of-the-art performance on text-to-human motion generation—across FID, R-Precision, and user studies—by combining a Semantic Alignment Encoder (SAE) with multi-token cross-attention text conditioning, performing masked autoregressive rectified flow in a continuous latent space.
tags:
  - CVPR 2026
  - Human Understanding
  - text-driven motion generation
  - semantically aligned latent space
  - cross-attention conditioning
  - autoregressive diffusion
  - continuous latent space
date: 2026-05-08
content_hash: f596b7c921693721
---

# MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation

**Conference**: CVPR 2026
**arXiv**: [2512.13840](https://arxiv.org/abs/2512.13840)
**Code**: [https://hynann.github.io/molingo/MoLingo.html](https://hynann.github.io/molingo/MoLingo.html)
**Area**: Human Understanding
**Keywords**: text-driven motion generation, semantically aligned latent space, cross-attention conditioning, autoregressive diffusion, continuous latent space

## TL;DR

MoLingo achieves comprehensive state-of-the-art performance on text-to-human motion generation—across FID, R-Precision, and user studies—by combining a Semantic Alignment Encoder (SAE) with multi-token cross-attention text conditioning, performing masked autoregressive rectified flow in a continuous latent space.

## Background & Motivation

**Background**: Text-driven human motion generation is a key technology for computer animation, AR/VR, and human-computer interaction. Current mainstream approaches fall into two categories: (1) diffusion directly in pose space (e.g., MDM), and (2) encoding into a latent space prior to diffusion (e.g., MLD, MARDM). The latter further splits into compressing entire sequences into a single latent vector versus autoregressively generating multiple latent vectors.

**Limitations of Prior Work**: Diffusion in pose space struggles with complex joint distributions and is prone to preserving mocap noise as artifacts. Single-vector latent diffusion discards important temporal details. VQ-based methods introduce quantization error by mapping continuous motion to a finite codebook, reducing the realism of fine-grained motion. Furthermore, existing text conditioning strategies—such as single-token conditioning or AdaLN modulation—have limited expressive capacity, constraining text–motion alignment quality.

**Key Challenge**: How can one construct a diffusion-friendly latent space in which semantically similar motions are geometrically proximate? How can text conditioning be injected more effectively so that generated motions are more faithful to textual descriptions?

**Goal**: (1) What kind of latent space is best suited for motion diffusion? (2) How should text conditioning be injected most effectively?

**Key Insight**: Inspired by works such as REPA in image generation, the paper trains a semantically aligned motion encoder using frame-level text labels, so that semantically similar latent vectors are closer in the latent space. It also finds that multi-token cross-attention substantially outperforms single-token conditioning.

**Core Idea**: Align the semantic structure of the motion latent space using frame-level text labels, combined with multi-token cross-attention conditioning, and perform masked autoregressive rectified flow over the continuous latent space for motion generation.

## Method

### Overall Architecture

MoLingo comprises two core components: (1) a Semantic Alignment Encoder (SAE) that encodes an $N$-frame motion sequence into $l = N/h$ continuous latent vectors $m_{1:l} \in \mathbb{R}^{l \times d}$; and (2) a masked autoregressive Transformer with a rectified flow MLP, conditioned on multi-token T5-encoded text representations, which iteratively denoises latent vectors and decodes them back into motion sequences. Training proceeds in two stages: the SAE is trained first, followed by the autoregressive generation model.

### Key Designs

1. **Semantic Alignment Encoder (SAE)**:

    - **Function**: Encodes motion sequences into a semantically rich, diffusion-friendly continuous latent space.
    - **Mechanism**: Extends a standard VAE by introducing frame-level text semantic alignment. Using frame-level text labels from the BABEL dataset, the method collects temporally aligned text labels for each motion latent vector $m_j$, encodes them with a frozen text encoder, and averages the projections to obtain a class token $\kappa_j$. A cosine similarity loss $\mathcal{L}_\text{sem} = \frac{1}{|\mathcal{I}|}\sum_{i \in \mathcal{I}}\left(1 - \frac{m_i \cdot \kappa_i}{\|m_i\| \|\kappa_i\|}\right)$ pulls semantically similar motion latent vectors closer together. To prevent over-alignment caused by abundant repeated labels in BABEL, the cosine similarity $\Delta_i$ between adjacent class tokens is computed and samples exceeding a threshold $\tau$ are filtered out.
    - **Design Motivation**: Semantic alignment simplifies the diffusion process—semantically similar motions are closer in the latent space, reducing the mapping complexity the diffusion model must learn. A soft cosine loss is preferred over InfoNCE because human motion is continuous and ambiguous; the hard contrastive constraints of InfoNCE are overly rigid in this setting.

2. **Multi-Token Cross-Attention Text Conditioning**:

    - **Function**: More effectively injects textual descriptions into the motion generation process.
    - **Mechanism**: Text is encoded by T5-Large and further enhanced through an $l_\text{adapter}=6$-layer Transformer encoder adapter for cross-modal interaction, yielding a multi-token text representation $\mathbf{w} = \{w_1, \ldots, w_{l_\text{text}}\}$. In the decoder-only Transformer, self-attention and MLP layers operate solely on motion latent vectors, while cross-attention uses motion latent vectors as queries and text tokens as keys and values. Ablation studies show this scheme substantially outperforms single-token conditioning with AdaLN modulation.
    - **Design Motivation**: Single-token conditioning compresses all textual information into a single vector, which is insufficient in expressiveness. Experiments demonstrate that multi-token cross-attention yields significant improvements in both FID and R-Precision (e.g., FID decreases from 0.114 to 0.049).

3. **Masked Autoregressive Rectified Flow Generation**:

    - **Function**: High-quality autoregressive motion generation in a continuous latent space.
    - **Mechanism**: The joint distribution of latent vectors is factored via the chain rule: $p(m_1,\ldots,m_l) = \prod_i p(m_i \mid c, m_1,\ldots,m_{i-1})$. During training, a random subset of latent vectors is masked; a decoder-only Transformer produces conditioning vectors $z_i$, which are fed into an MLP $v_\theta$ that approximates the reverse distribution using a rectified flow objective: $\mathcal{L} = \mathbb{E}\left[\|v_\theta(m_i^t, t, z_i) - (\epsilon - m_i)\|^2\right]$. At inference, generation begins from fully masked tokens and proceeds iteratively, with classifier-free guidance (scale = 5.5) applied to enhance conditional fidelity.
    - **Design Motivation**: The autoregressive formulation preserves richer temporal detail than single-vector diffusion; the continuous latent space avoids the quantization error of VQ-based discrete methods; and rectified flow offers higher sampling efficiency than standard diffusion.

### Loss & Training

The total SAE loss is $\mathcal{L}_\text{SAE} = \mathcal{L}_\text{recon} + \lambda_\text{sem}\mathcal{L}_\text{sem} + \lambda_\text{KL}\mathcal{L}_\text{KL}$, where $\mathcal{L}_\text{recon}$ comprises three terms: feature reconstruction, joint position, and joint velocity. The SAE is trained for 5,000 epochs with a batch size of 256. The autoregressive model is trained for approximately 800 epochs with EMA for training stability; 10% of text inputs are replaced with null prompts for CFG. Total training time is approximately 10 hours on 4 H100 GPUs.

## Key Experimental Results

### Main Results

| Method | FID ↓ | R-Precision Top-1 ↑ | CLIP-Score ↑ | MModality ↑ |
|---|---|---|---|---|
| MDM | 0.518 | 0.440 | 0.578 | **3.604** |
| MLD | 0.431 | 0.461 | 0.610 | 3.506 |
| MoMask | 0.116 | 0.490 | 0.637 | 1.309 |
| ACMDM-XL | 0.058 | 0.522 | 0.652 | 2.077 |
| DisCoRD | 0.053 | 0.506 | 0.645 | 1.303 |
| MoLingo (VAE) | **0.049** | 0.528 | 0.672 | 1.414 |
| MoLingo (SAE) | 0.066 | **0.544** | **0.686** | 1.226 |

Under the MARDM-67 evaluation protocol, MoLingo (VAE) achieves the best FID, while MoLingo (SAE) achieves the best text–motion alignment in terms of R-Precision and CLIP-Score.

### Ablation Study

| Conditioning | Text Encoder | Autoencoder | FID ↓ | R-Precision Top-1 ↑ |
|---|---|---|---|---|
| AdaLN | CLIP | AE | 0.114 | 0.500 |
| AdaLN | T5 | AE | 0.077 | 0.508 |
| CrossAttn | T5 | VAE | **0.049** | 0.528 |
| CrossAttn | T5 | AE | 0.051 | 0.533 |
| CrossAttn | T5 | SAE | 0.066 | **0.544** |

### Key Findings

- **CrossAttn vs. AdaLN**: Multi-token cross-attention reduces FID from 0.077 to 0.049 and improves R-Precision Top-1 from 0.508 to 0.528–0.544, constituting a substantial gain.
- **Text Alignment Effect of SAE**: SAE outperforms both AE and VAE across R-Precision and CLIP-Score, but yields a slightly higher FID (0.066 vs. 0.049), indicating that semantic alignment trades a modest distribution-matching cost for stronger text faithfulness.
- **Cosine Loss vs. InfoNCE**: InfoNCE yields an FID of 0.129, far worse than the cosine loss result of 0.066, because the continuity of human motion makes hard contrastive constraints overly rigid.
- **User Study**: Users prefer MoLingo in 83.75% of comparisons against DisCoRD, 77.70% against MoMask, and 84.70% against MotionStreamer.
- A 4× temporal downsampling factor combined with a 16-dimensional latent space constitutes the optimal configuration; increasing latent dimensionality is detrimental.

## Highlights & Insights

- **Semantically Aligned Latent Space**: Using frame-level text labels to guide the structure of the motion latent space—so that semantically similar motions are geometrically proximate—is a transferable idea applicable to other sequence generation tasks (e.g., audio, trajectory generation) wherever aligned semantic labels are available.
- **Soft Cosine vs. Hard Contrastive Constraints**: For continuous, ambiguous signals such as motion, soft constraints outperform hard contrastive objectives. This insight has broader implications for representation learning on continuous signals.
- **Substantial Advantage of Multi-Token Cross-Attention**: Single-token conditioning discards too much information; multi-token representations preserve the structured semantics of text. While this has been validated in T2I generation (e.g., DALL-E 3), this paper successfully transfers the insight to T2M.

## Limitations & Future Work

- The method generates only body-level motion and does not include fine-grained hand motion, which is a notable limitation for real-world applications.
- The SAE relies on frame-level annotations from the BABEL dataset, which have limited coverage and high label redundancy; scaling to larger datasets may require automatic annotation pipelines.
- A trade-off exists between FID and R-Precision (SAE achieves the best R-Precision but not the best FID); whether both can be simultaneously optimized remains an open question.
- The evaluation metrics themselves are debatable, as different protocols yield inconsistent results, motivating the need for more robust evaluation frameworks.

## Related Work & Insights

- **vs. MARDM**: MARDM uses CLIP single-token conditioning with AdaLN; MoLingo uses T5 multi-token cross-attention with SAE. These three improvements each contribute distinct performance gains.
- **vs. MotionStreamer**: MotionStreamer employs a 272D representation to avoid IK artifacts; MoLingo operates on both 263D and 272D representations and achieves superior performance in both cases.
- **vs. VQ-based Methods (MoMask, DisCoRD)**: VQ-based methods perform better on diversity (MModality), while continuous latent space methods demonstrate clear advantages in realism (FID) and text alignment.
- The work is inspired by REPA (representation alignment in image generation), transplanting the semantic alignment idea into the motion generation domain.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The frame-level semantic alignment in SAE is the core innovation; multi-token conditioning has precedent in image generation but is effectively transferred here.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four evaluation protocols (MARDM-67, TMR-263, MS-272, user study) with extensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem-driven exposition is clear, with two core questions guiding a coherent progression.
- **Value**: ⭐⭐⭐⭐ State-of-the-art results combined with a transferable latent space design paradigm that meaningfully advances the motion generation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference](lamogen_language_to_motion_generation_through_llm-guided_symbolic_inference.md)
- [\[CVPR 2026\] ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis](party_part-guidance_for_expressive_text-to-motion_synthesis.md)
- [\[CVPR 2026\] HandX: Scaling Bimanual Motion and Interaction Generation](handx_scaling_bimanual_motion_and_interaction_generation.md)
- [\[ICCV 2025\] GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](../../ICCV2025/human_understanding/genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)
- [\[CVPR 2026\] RAM: Recover Any 3D Human Motion in-the-Wild](ram_recover_any_3d_human_motion_in-the-wild.md)

</div>

<!-- RELATED:END -->
