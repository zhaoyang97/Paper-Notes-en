---
title: >-
  [Paper Note] MiVE: Multiscale Vision-language features for reference-guided video Editing
description: >-
  [ICML 2026][Video Generation][Reference Image Guidance] MiVE simultaneously extracts the **first and last layer** hidden states from Qwen3-VL as multiscale condition tokens…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Reference Image Guidance"
  - "Video Editing"
  - "Multiscale VLM Features"
  - "Unified Self-Attention"
  - "DiT"
date: 2026-05-08
content_hash: 64d099f924e9566e
---

# MiVE: Multiscale Vision-language features for reference-guided video Editing

**Conference**: ICML 2026  
**arXiv**: [2605.14664](https://arxiv.org/abs/2605.14664)  
**Code**: https://mivepaper.github.io (project page, code not explicitly open-sourced)  
**Area**: Video Editing / Multimodal VLM / Diffusion Models  
**Keywords**: Reference Image Guidance, Video Editing, Multiscale VLM Features, Unified Self-Attention, DiT

## TL;DR
MiVE simultaneously extracts the **first and last layer** hidden states from Qwen3-VL as multiscale condition tokens, concatenates them with VAE visual latents into a long sequence, and performs reference-guided video editing using unified self-attention in DiT. On a 60-clip 720P benchmark, it achieves top human preference and six VLM-based automatic scores, outperforming open-source Wan-Animate and commercial Kling O1.

## Background & Motivation

**Background**: The reference-guided video editing task is defined as: given a source video $x_{src}$ and an editing instruction $x_{text}$, use an external image editor (e.g., FLUX.1 Kontext) to modify the first frame and obtain a reference image $x_{ref}$, then require the model to **faithfully propagate this modification throughout the video** while **preserving** the original motion and unedited regions. Current mainstream approaches follow two paths: (1) decoupled encoders like T5 + SigLIP, where text and vision are encoded separately and fused via cross-attention in DiT; (2) directly using VLMs such as Qwen3-VL / MiniCPM-V as **unified encoders** (as in Kling O1).

**Limitations of Prior Work**: Decoupled encoders inherently suffer from a "modality gap"—text and visual features reside in different semantic spaces, and the final cross-attention layer struggles to bridge them. This leads to "instruction misunderstanding" and "reference misalignment" errors in fine-grained cross-modal video editing. Unified VLM encoders address the modality gap but **only use the last layer's hidden state**, discarding the rich local spatial details in earlier layers, resulting in blurry details (e.g., hair strands, lighting, textures) in edited videos.

**Key Challenge**: VLMs possess an overlooked hierarchical structure—shallow layers tend to encode local spatial details (pixel-level alignment), while deep layers encode global semantics (instruction understanding). Existing methods either do not use VLMs at all (losing unified semantic space) or only use the last layer (losing spatial details), with none leveraging both ends. Moreover, cross-attention is inherently asymmetric—visual tokens query text, but text tokens are agnostic to vision, lacking fine-grained bidirectional correspondence.

**Goal**: (1) Validate the hypothesis that VLM shallow layers encode spatial details and deep layers encode semantics; (2) Design a video editing framework that leverages both shallow and deep VLM features; (3) Replace cross-attention with a truly symmetric unified attention mechanism.

**Key Insight**: The authors use a simple "cross-modal diagnostic matrix" $A_{txt \to vis}^{(l)} = E B^{\top}$ to quantify the attention concentration of each layer's text tokens on visual tokens, combined with SAM2-generated person masks to compute the Attention Mask Ratio. Results show that for Qwen3-VL, the 0th layer has $R_{mask} \approx 0.37$, dropping to $0.23$ at the last layer; shallow layers precisely localize person contours, while deep layers exhibit diffuse global attention. This directly supports the subsequent multi-scale design.

**Core Idea**: **Extract the first and last VLM layers → project into condition tokens → concatenate with visual latents into a long sequence → unified self-attention throughout**, using a shared attention manifold to simultaneously achieve "local detail propagation" and "global semantic understanding".

## Method

### Overall Architecture

MiVE takes as input the source video $x_{src}$, text instruction $x_{text}$, and reference image $x_{ref}$ (edited first frame from an external image editor); the output is the edited video $\hat{x}_{tgt}$. The pipeline consists of three stages:

1. **Multi-Level Context Extraction**: Feed $\{x_{text}, x_{ref}, x_{src}\}$ into Qwen3-VL-8B (frozen), simultaneously extract **1st layer** and **$L$-th layer** hidden states $\phi_1, \phi_L \in \mathbb{R}^{S \times D_{VLM}}$, each projected via RMSNorm + Linear to $\mathbb{R}^{S \times D/2}$, concatenated along the feature dimension and passed through a fusion linear layer to obtain the condition token $c \in \mathbb{R}^{N_c \times D}$.
2. **Reference-Aware Latent Encoding**: Encode $x_{src}, x_{tgt}, x_{ref}$ with a frozen VAE to obtain latents. During training, **prepend** the reference latent $z_{ref}$ along the temporal dimension to both the noisy target $\tilde z_t$ and control $z_{src}$ branches, then concatenate the two branches along the channel dimension, resulting in shape $(T'+1) \times 2C \times H' \times W'$. This ensures the model can "see" the reference image as an appearance anchor from the very first frame.
3. **Unified Self-Attention Backbone**: Concatenate the condition token $c$ and patchified visual token $v$ into $u^{(0)} = [c; v] \in \mathbb{R}^{(N_c + N_v) \times D}$, and process the entire sequence through DiT blocks with **unified self-attention** (no cross-attention). The key trick is **per-token AdaLN**: clean tokens (condition + reference frame patches) use a fixed time embedding $t=0$, while noisy tokens (target video patches) use the current diffusion timestep $t$. The model is initialized from Wan2.1-T2V-14B's self-attention block and trained with flow matching.

### Key Designs

1. **Multi-Level VLM Feature Extraction (Multi-Level Context Extraction)**:

    - **Function**: Enables the condition signal to carry both VLM shallow spatial details and deep global semantics.
    - **Mechanism**: For a single forward pass of Qwen3-VL, extract hidden states from the 1st and $L$-th layers, project each via independent adapters, **concatenate along the channel dimension to $D$**: $c_{raw} = \text{Concat}_D(\tilde\phi_1, \tilde\phi_L)$, then pass through $\text{Linear}_{fuse}$ to obtain the final condition token. The choice of endpoints (rather than uniform sampling) is justified by diagnostic experiments showing $R_{mask}$ extremums at layer 0 and layer $L$, with monotonic transition and high redundancy in intermediate layers.
    - **Design Motivation**: Addresses the issue that "unified VLM using only the last layer loses spatial details." The condition token $c$ does not depend on the diffusion timestep $t$ and serves as a fixed "semantic anchor" throughout denoising, unaffected by noise modulation.

2. **Reference-Aware Latent Encoding (Reference Frame Temporal Prepend + Dual-Branch Channel Concatenation)**:

    - **Function**: Provides the model with both an appearance anchor (from the reference image) and a motion anchor (from the source video) without introducing masks.
    - **Mechanism**: During training, construct $z_t = \text{Concat}_C([z_{ref}; \tilde z_t], [z_{ref}; z_{src}])$, i.e., both branches are headed by $z_{ref}$ in the temporal dimension. During inference, the noisy target branch is initialized with $\tilde z_T \sim \mathcal{N}(0, I)$, and the control branch uses the actual source video; the rest remains the same. The entire latent is patch-embedded into $N_v$ spatiotemporal patches.
    - **Design Motivation**: $z_{ref}$ serves as both appearance and structural anchor—temporal prepend ensures every attention layer can access it, and dual-branch channel concatenation aligns control (source video) and target (to be generated) signals at the latent level, avoiding the mask-guided method's pitfalls with inaccurate masks in fast motion/complex backgrounds.

3. **Unified Self-Attention + Per-Token AdaLN**:

    - **Function**: Replaces asymmetric cross-attention with symmetric, long-sequence self-attention, allowing condition and visual tokens to mutually query/key/value in the same space.
    - **Mechanism**: $u^{(0)} = [c; v]$ is input to $P$ DiT blocks. Each token independently determines its AdaLN modulation—condition tokens and reference frame patches use $t=0$ embedding (always "clean"), while other target patches use the current denoising $t$ embedding. At output, only $u^{(P)}[N_c:]$ is unpatchified, discarding condition tokens, and then the reference frame is removed to decode the final video $\hat x_{tgt} = \mathcal{D}(\hat z_0[1:])$.
    - **Design Motivation**: In cross-attention, vision queries text but text cannot query vision, preventing fine-grained bidirectional correspondence; unified self-attention is naturally symmetric, enabling the model to learn modality alignment rather than being artificially separated. Per-token AdaLN addresses the engineering issue where "clean and noisy signals are polluted by the same time embedding," ensuring the reference frame remains stable throughout denoising.

### Loss & Training

Flow matching objective (Lipman et al., 2023), initialized from Wan2.1-T2V-14B self-attention block, trained on 8 H100s at 720P / 81 frames for 8000 steps (about 2 epochs, ~65 hours). Optimizer: AdamW, lr $3 \times 10^{-5}$, $\beta = (0.9, 0.999)$, 200-step warmup, gradient clipping at 1.0. Inference: single H100 generates 81-frame 720P video in ~6.5 minutes (Qwen3-VL ~3s, DiT denoising ~328s, VAE decoding ~35s), peak memory 50 GB. Training data: 30K pairs—24K filtered from OpenVE-3M using Qwen3-VL score ≥9.3 (six editing categories, max 4000 per category), plus 6K portrait data constructed by segmenting foreground + compositing background videos (three types: removal/addition/background replacement).

## Key Experimental Results

### Main Results

Benchmark: 60 720P videos, split into a simple subset (30 clips from RoseBench + VPBench, with approximate masks) and a complex subset (30 portrait videos involving style transfer/lighting redistribution/background replacement, no masks). Evaluation uses Gemini-3-Flash scoring 0-10 on six dimensions (IA / CC / TS / PR / VA / SC), plus a 30-person user study with holistic 1-5 scores.

| Subset | Method | IA | CC | TS | VA | SC | User |
|--------|--------|----|----|----|----|----|------|
| Simple | VACE | 7.06 | 7.12 | 6.45 | 6.39 | 7.02 | 2.67 |
| Simple | LucyEdit | 6.14 | 7.56 | 7.55 | 5.96 | 7.13 | 1.58 |
| Simple | VideoCof | 7.53 | 8.04 | 8.62 | 6.41 | 8.28 | 1.46 |
| Simple | Kling O1 | 8.48 | 9.03 | 8.91 | 8.51 | 9.31 | 3.69 |
| Simple | **MiVE** | **9.30** | 8.65 | 8.81 | **8.83** | **9.46** | **4.18** |
| Complex | LucyEdit | 7.22 | 7.02 | 6.36 | 5.57 | 7.05 | 1.78 |
| Complex | Wan-Animate | 8.87 | 7.78 | 7.83 | 7.73 | 8.98 | 3.03 |
| Complex | Kling O1 | 8.68 | 7.71 | 8.11 | 7.74 | 9.14 | 3.61 |
| Complex | **MiVE** | **9.23** | **8.05** | **8.27** | **8.09** | **9.22** | **3.75** |

On the simple set, MiVE ranks first in IA / VA / SC, and second in CC / TS / PR (behind commercial Kling O1), but user study shows a clear gap (4.18 vs 3.69); on the complex set, MiVE leads in all six metrics plus user score.

### Ablation Study

| Configuration | IA | CC | TS | VA | SC | Notes |
|---------------|----|----|----|----|----|-------|
| Decoupled Enc. + Dual Cross-Attn | 6.76 | 6.10 | 5.88 | 5.87 | 7.45 | Legacy decoupled baseline |
| Unified Enc. (only last layer) + Dual Cross-Attn | 8.51 | 8.24 | 7.68 | 7.42 | 8.03 | Last layer only + cross-attn |
| Unified Enc. + Fused Cross-Attn | 8.53 | 8.22 | 7.87 | 8.08 | 9.00 | Single-branch cross-attn |

Switching from decoupled to unified VLM encoder boosts nearly all metrics by over 1.5 points—thanks to the unified semantic space; further replacing cross-attn with unified self-attention (i.e., full MiVE) increases IA from 8.53 to the 9.23 range.

### Key Findings

- VLM **first layer** Attention Mask Ratio (Qwen3-VL: 0.366, GLM-4.6V: 0.333) is significantly higher than the last layer (0.228, 0.270), confirming the "shallow spatial / deep semantic" hypothesis, which underpins the multi-scale design.
- Qwen3-VL's shallow layer localization outperforms GLM-4.6V (0.37 vs 0.33), motivating its selection as backbone.
- In complex scenarios (fast motion / strong lighting changes / hair color changes), MiVE maintains identity better than Wan-Animate and Kling O1, reflecting that reference latent temporal prepend keeps the appearance anchor visible throughout, especially effective on hard samples.
- The authors explicitly **do not report SSIM / LPIPS**, arguing that in editing tasks, the generated and input videos are inherently different, making these structural similarity metrics inapplicable.

## Highlights & Insights

- **Diagnostic Motivation Derivation**: Using a simple $E B^{\top}$ matrix + SAM2 mask to quantify "which VLM layer focuses most on the foreground," turning intuition into numbers via $R_{mask}$; this "quantify first, then design" approach is more convincing than merely stating "we observed...," and is worth emulating.
- **Unified Self-Attention Replacing Cross-Attention**: Conceptually aligned with Z-Image / FLUX, but MiVE explicitly highlights "clean / noisy per-token AdaLN" as a key design, encoding the prior that "reference frames are always clean" into the time embedding, avoiding noise schedule contamination of anchors.
- **Multiscale Using Only Endpoints**: Engineering-wise, selecting only the first and last layers saves over half the projection parameters compared to sampling every few layers, and the monotonicity of $R_{mask}$ shows intermediate layers are linearly interpolable, so endpoints suffice.
- **Courage to Omit SSIM / LPIPS**: Clearly stating that traditional structural similarity metrics are unsuitable for editing tasks, and instead using VLM judge + user study; this critical stance on evaluation methods is valuable for future video editing research.

## Limitations & Future Work

- **30K training pairs is relatively small and all from OpenVE-3M / synthetic data**; complex real-world physics (fluids / reflections / transparent objects) are not separately evaluated, raising concerns about generalization.
- **6.5 minutes to generate 81 frames / 50 GB memory**, still requires H100s, far from consumer deployment; the paper does not discuss acceleration (distillation / few-step flow / token pruning).
- **Evaluation loop may have bias**: backbone is Qwen3-VL, main evaluator is Gemini-3-Flash; although the appendix includes InternVL3.5 for cross-validation, VLM judges may resonate with the training backbone's instruction understanding style.
- **Is the first layer truly optimal?** The paper only tests Qwen3-VL and GLM-4.6V; different VLMs may have different layerwise distributions, so "always use endpoints" may not generalize—missing a layer-selection ablation.

## Related Work & Insights

- **vs VACE / VideoPainter (mask-guided)**: These rely on precise masks for spatial control, but fail in fast motion and complex backgrounds due to inaccurate masks; MiVE is completely mask-free, letting the model infer editing regions from instructions + reference image.
- **vs Lucy Edit / Wan-Animate (mask-free but unified encoder uses only last layer)**: They lose shallow spatial details, resulting in blurry fine textures/local objects; MiVE recovers this via multiscale condition tokens.
- **vs Kling O1 (commercial unified VLM)**: The commercial system's architecture is undisclosed, but experiments show MiVE achieves a significant gap in IA (instruction adherence), suggesting "last layer only + cross-attn" is Kling O1's bottleneck.
- **vs Ditto / ICVE (implicit prior token + DiT)**: Similar in spirit, but both use single-scale priors; MiVE's multiscale approach is better suited for complex scenarios, analogous to FPN vs single-scale detectors.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of multiscale VLM + unified self-attention is the first systematic application in video editing, though multiscale and unified self-attention have precedents in image generation.
- Experimental Thoroughness: ⭐⭐⭐⭐ 60-clip benchmark + dual evaluators + user study + 4 architecture ablations, quite solid; but training data is small and lacks in-the-wild evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Diagnostic motivation derivation + well-integrated formulas/figures, with a complete and elegant argument for "why use endpoints."
- Value: ⭐⭐⭐⭐ Advances the "VLM as unified encoder" paradigm to multiscale, establishing a strong baseline for future video editing/generation condition design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Language-guided Open-world Video Anomaly Detection under Weak Supervision](../../ICLR2026/video_generation/language-guided_open-world_video_anomaly_detection_under_weak_supervision.md)
- [\[ICLR 2026\] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](../../ICLR2026/video_generation/lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)
- [\[ICML 2026\] Lightning Unified Video Editing via In-Context Sparse Attention](lightning_unified_video_editing_via_in-context_sparse_attention.md)
- [\[CVPR 2026\] LAMP: Language-Assisted Motion Planning for Controllable Video Generation](../../CVPR2026/video_generation/lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [\[CVPR 2026\] VideoCoF: Unified Video Editing with Temporal Reasoner](../../CVPR2026/video_generation/videocof_unified_video_editing_with_temporal_reasoner.md)

</div>

<!-- RELATED:END -->
