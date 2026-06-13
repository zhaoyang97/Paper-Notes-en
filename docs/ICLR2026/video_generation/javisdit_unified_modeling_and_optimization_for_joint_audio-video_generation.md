---
title: >-
  [Paper Note] JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation
description: >-
  [ICLR 2026][Video Generation][Joint Audio-Video Generation] This paper proposes JavisDiT++, a clean and unified framework for joint audio-video generation (JAVG). It improves generation quality via modality-specific MoE…
tags:
  - "ICLR 2026"
  - "Video Generation"
  - "Joint Audio-Video Generation"
  - "DiT"
  - "Mixture-of-Experts"
  - "RoPE"
  - "DPO"
date: 2026-05-08
content_hash: b9854a042f6fd587
---

# JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation

**Conference**: ICLR 2026
**arXiv**: [2602.19163](https://arxiv.org/abs/2602.19163)  
**Code**: [GitHub](https://JavisVerse.github.io/JavisDiT2-page)  
**Area**: Video Generation
**Keywords**: Joint Audio-Video Generation, DiT, Mixture-of-Experts, RoPE, DPO

## TL;DR

This paper proposes JavisDiT++, a clean and unified framework for joint audio-video generation (JAVG). It improves generation quality via modality-specific MoE, achieves frame-level synchronization through temporally aligned RoPE, and aligns outputs with human preferences via audio-video DPO. Built on Wan2.1-1.3B with only ~1M public data, it achieves state-of-the-art performance.

## Background & Motivation

Joint audio-video generation (JAVG) requires a model to simultaneously produce temporally synchronized and semantically aligned video and audio from text descriptions. Existing open-source methods lag behind commercial systems (e.g., Veo3) in three respects:

**Generation Quality**: Existing methods either apply a shared FFN to both modalities (UniForm), causing modality information loss, or adopt dual-stream DiTs (JavisDiT, UniVerse-1), which are architecturally complex and scale poorly.

**Temporal Synchronization**: JavisDiT relies on ST-Prior and UniVerse-1 on a Stitching strategy—both are implicit synchronization mechanisms that are imprecise and introduce additional inference overhead.

**Human Preference Alignment**: No existing JAVG method incorporates preference optimization, leaving a gap in aesthetics and audio-visual harmony relative to human expectations. JavisDiT++ is the first work to introduce preference alignment into JAVG.

## Method

### Overall Architecture

The framework builds on Wan2.1-1.3B-T2V as the video backbone and adopts a three-stage training pipeline: audio pretraining → audio-video SFT → audio-video DPO. Rectified Flow is used as the noise scheduler; the video VAE is from Wan2.1 and the audio VAE from AudioLDM2, both frozen.

### Key Designs

1. **Modality-Specific MoE (MS-MoE)**: Audio and video tokens interact through a shared multi-head self-attention layer for cross-modal interaction, then pass through their respective FFN layers for intra-modal aggregation. The design is conceptually similar to BAGEL but routes tokens by modality rather than task. Although total parameters increase from 1.3B to 2.1B, each token activates only 1.3B parameters, incurring no additional inference cost. This outperforms the following two alternatives:

    - Shared-DiT + LoRA: Audio quality is bottlenecked by insufficient trainable capacity.
    - Shared-DiT + Full-FT: Excessive parameter drift during audio pretraining severely degrades video quality.

2. **Temporally Aligned RoPE (TA-RoPE)**: Absolute temporal alignment between audio and video tokens is enforced along the first (temporal) dimension of the 3D position IDs. Video token position IDs are $(t, h, w)$; audio token position IDs are defined as:

$$R_a(t, m) = \left(\left[t \cdot \frac{T_v}{T_a}\right], t + H, m + W\right)$$

where $[\cdot]$ denotes the floor operation and the $H$, $W$ offsets ensure non-overlapping position IDs between modalities. This design achieves temporal alignment within a full-attention framework purely through position ID manipulation, requiring no physical reordering of token sequences and incurring zero additional inference cost.

3. **Audio-Video DPO (AV-DPO)**: This work is the first to introduce preference alignment into JAVG. Key contributions include:

    - **Reward Models**: Evaluation along three dimensions—audio quality (AudioBox + ImageBind), video quality (VideoAlign + ImageBind), and audio-video alignment (ImageBind + Syncformer).
    - **Preference Data Construction**: 30K prompts × 3 generated samples + ground truth; scores are normalized per modality and ranked separately to select winner–loser pairs, ensuring winners outperform losers across all modality dimensions (~25K pairs obtained).
    - **Modality-Aware Loss**: DPO losses for audio and video are computed separately and combined with weighting:

$$\mathcal{L}_{\mathrm{DPO}}^{av} = -\mathbb{E}\left[\log\sigma\left(-\beta_v(\mathrm{Diff}_{\mathrm{policy}}^v - \mathrm{Diff}_{\mathrm{ref}}^v) - \beta_a(\mathrm{Diff}_{\mathrm{policy}}^a - \mathrm{Diff}_{\mathrm{ref}}^a)\right)\right]$$

### Loss & Training

- Audio pretraining: 780K audio-text pairs.
- Audio-video SFT: 330K audio-video-text triplets with a Flow Matching objective.
- Audio-video DPO: 25K preference pairs with Flow Matching regularization to prevent overfitting.
- Supports variable duration (2–5 s) and resolution (240p–480p) with multiple aspect ratios.

## Key Experimental Results

### Main Results (JavisBench, 240p4s)

| Model | Params | FVD↓ | FAD↓ | AV-IB↑ | JavisScore↑ | DeSync↓ | Inference Time |
|-------|--------|------|------|--------|-------------|---------|----------------|
| JavisDiT | 3.1B | 204.1 | 7.2 | 0.197 | 0.154 | 1.039 | 30s |
| UniVerse-1 | 6.4B | 194.2 | 8.7 | 0.104 | 0.077 | 0.929 | 13s |
| **JavisDiT++** | **2.1B** | **141.5** | **5.5** | **0.198** | **0.159** | **0.832** | **10s** |

### Ablation Study (JavisBench-mini)

| Configuration | FVD↓ | FAD↓ | JavisScore↑ | DeSync↓ | Note |
|---------------|------|------|-------------|---------|------|
| Shared-DiT + LoRA | 227.6 | 6.51 | 0.098 | 0.934 | Insufficient LoRA capacity |
| Shared-DiT + Full-FT | 269.3 | 5.66 | 0.137 | 0.945 | Degraded video quality |
| **MS-MoE** | **221.3** | **5.51** | **0.153** | **0.807** | Best architecture |
| No sync mechanism | — | — | 0.142 | 0.942 | Baseline |
| ST-Prior | — | — | 0.145 | 0.863 | +6s latency |
| **TA-RoPE** | — | — | **0.153** | **0.807** | Zero extra cost |
| No DPO | 221.3 | 5.51 | 0.153 | 0.807 | SFT baseline |
| Modality-Micro DPO | **198.5** | 5.32 | **0.156** | **0.776** | Best DPO strategy |

### Key Findings

- MS-MoE substantially improves audio quality while preserving video quality, validating the necessity of modality-specific FFNs.
- TA-RoPE achieves better synchronization at zero inference cost compared to ST-Prior and FrameAttn, which require additional computation.
- AV-DPO yields moderate gains on objective metrics but over 25% preference improvement in human evaluation, capturing aesthetic preferences that metrics cannot quantify.
- Modality-aware preference pair construction is critical—selecting winners with inconsistent modality rankings causes DPO to degrade.

## Highlights & Insights

- JavisDiT++ surpasses dual-stream architectures with fewer parameters (2.1B vs. 6.4B) and less data (~1M vs. large-scale), demonstrating that a clean unified architecture with carefully designed modules outperforms brute-force scaling.
- The position ID manipulation in TA-RoPE is an elegant solution—it exploits the symmetry of the full-attention framework to achieve temporal alignment without physically reordering token sequences.
- This is the first work to introduce DPO into multi-modal joint generation, with a modality-aware preference data construction pipeline.
- Inference overhead is only 1.6% higher than pure video generation, making the approach highly practical.

## Limitations & Future Work

- Current resolution and duration are limited (240–480p, 2–5 s), falling short of practical commercial deployment.
- Objective metric gains from AV-DPO are modest; the evaluation capability of the reward models may be a bottleneck.
- The audio VAE (AudioLDM2) was not designed for joint generation, potentially limiting audio diversity.
- Validation is limited to Wan2.1-1.3B; scalability to larger or different model families remains unexplored.
- A noticeable gap remains relative to commercial systems such as Veo3, particularly in semantic alignment for complex scenes.

## Related Work & Insights

- The dual-stream DiT approaches of JavisDiT and UniVerse-1 are superseded by MS-MoE, suggesting that shared attention with modality-specific FFNs is a more efficient paradigm.
- The modality-aware preference data strategy of AV-DPO generalizes to other multi-modal alignment scenarios (e.g., audio+3D, video+haptics).
- The temporal alignment idea behind TA-RoPE can be extended to other tasks requiring cross-modal synchronization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ TA-RoPE and AV-DPO are genuinely novel; MS-MoE is relatively incremental.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive comparisons of architectures, synchronization mechanisms, DPO strategies, and subjective evaluation; ablations are very thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich figures and tables, though some descriptions are slightly verbose.
- **Value**: ⭐⭐⭐⭐ Establishes a new SOTA and benchmark for open-source JAVG; the AV-DPO methodology offers valuable insights for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization](javisdit_joint_audio-video_diffusion_transformer_with_hierarchical_spatio-tempor.md)
- [\[CVPR 2026\] UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation](../../CVPR2026/video_generation/unitalking_a_unified_audio-video_framework_for_talking_portrait_generation.md)
- [\[ICLR 2026\] Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation](dual-ipo_dual-iterative_preference_optimization_for_text-to-video_generation.md)
- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [\[CVPR 2026\] UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions](../../CVPR2026/video_generation/uniavgen_unified_audio_and_video_generation_with_asymmetric_cross-modal_interact.md)

</div>

<!-- RELATED:END -->
