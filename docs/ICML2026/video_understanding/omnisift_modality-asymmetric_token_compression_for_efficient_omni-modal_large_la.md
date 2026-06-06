---
title: >-
  [Paper Note] OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models
description: >-
  [ICML 2026][Video Understanding][Omni-LLM] Ours highlights that existing Omni-LLM token compression methods are sub-optimal due to "symmetric" treatment of audio and video. Ours proposes OmniSIFT—a two-stage asymmetric c…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Omni-LLM"
  - "Token Compression"
  - "Video-Audio Understanding"
  - "Spatio-Temporal Pruning"
  - "Vision Guidance"
date: 2026-05-08
content_hash: 318fa33bcedcad0b
---

# OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.04804](https://arxiv.org/abs/2602.04804)  
**Code**: https://github.com/dingyue772/OmniSIFT  
**Area**: Multi-modal VLM / Video Understanding / Model Compression  
**Keywords**: Omni-LLM, Token Compression, Video-Audio Understanding, Spatio-Temporal Pruning, Vision Guidance

## TL;DR
Ours highlights that existing Omni-LLM token compression methods are sub-optimal due to "symmetric" treatment of audio and video. Ours proposes OmniSIFT—a two-stage asymmetric compression framework that first prunes video redundancy via spatio-temporal saliency to obtain "visual anchors," then uses these anchors to guide audio selection. Introducing only 4.85M additional parameters, it consistently outperforms existing compression baselines and even the original model on Qwen2.5-Omni-7B while retaining 25% of tokens.

## Background & Motivation

**Background**: Omni-LLMs (Qwen2.5-Omni, GPT-4o, Gemini) unify video, audio, and text into autoregressive LLMs for joint reasoning. However, high-density continuous video frames and high temporal resolution audio encoding generate over 20K tokens for a 20-second multi-modal clip, causing an explosion in inference costs due to long token sequences.

**Limitations of Prior Work**: Vision-centric MLLMs have extensive token compression research (FastV, VidCom2, TimeChat-Online, etc.), but direct migration to Omni-LLMs is infeasible. Existing Omni compression methods fall into two categories: (1) modality-decoupled—independent compression of audio and video, ignoring cross-modal semantic dependencies; (2) modality-symmetric—OmniZip uses audio attention scores to guide video pruning (dependency on attention scores makes it incompatible with FlashAttention), and EchoingPixels adds 4 LLM decoding layers for global cross-modal contextualization (high cost, late compression). Both treat audio and video as information sources of equal magnitude.

**Key Challenge**: Human perception of audio and video is inherently asymmetric—video redundancy can be estimated from within the visual stream (intra-frame spatial redundancy + inter-frame temporal redundancy), but audio significance relies more on context, often requiring visual scenes as semantic anchors (visible speakers, visually supported events). Treating the two modalities symmetrically collapses compression into "selecting temporal positions" while ignoring modality-specific semantic cues.

**Goal**: (1) Adhere to a vision-guided asymmetric paradigm for compression; (2) Maintain light weight (additional parameters $\ll$ backbone); (3) Ensure compatibility with efficient operators like FlashAttention (independent of attention scores).

**Key Insight**: Prune video redundancy using structural signals (cosine distance) to obtain a compact set of "visual anchors," then use these anchors to guide audio token selection. This divides tasks: video uses intra-modal signals, while audio uses cross-modal conditions.

**Core Idea**: Modality-asymmetric, vision-guided two-stage compression—STVP (Spatio-Temporal Video Pruning) performs dual-axis pruning for intra-frame spatial saliency and inter-frame temporal saliency; VGAS (Vision-Guided Audio Selector) uses pruned visual anchors as conditions to select audio tokens.

## Method

### Overall Architecture
Inputs: Video $\mathcal{V}$ and synchronized audio $\mathcal{A}$, mapped to token sequences $\mathbf{Z}_v \in \mathbb{R}^{N_v \times D}$ and $\mathbf{Z}_a \in \mathbb{R}^{N_a \times D}$ via the Qwen2.5-Omni encoder-projector. To maintain temporal alignment, audio-visual tokens are partitioned into 2-second chunks $\mathcal{C}_t = [\mathbf{Z}_v^{(t)}; \mathbf{Z}_a^{(t)}]$, each containing 2 frames of vision and corresponding audio. OmniSIFT executes two stages serially at the chunk level: (1) STVP prunes video redundancy in each chunk to obtain the compressed visual sequence $\hat{\mathbf{Z}}_v^{(t)}$; (2) VGAS uses $\hat{\mathbf{Z}}_v^{(t)}$ as a condition to select audio tokens from $\mathbf{Z}_a^{(t)}$. The framework is end-to-end differentiable (using a straight-through estimator for top-k selection), optimizing token selection to preserve downstream task performance.

### Key Designs

1. **STVP: Dual-axis Saliency Pruning (Spatial + Temporal)**:

    - **Function**: Prunes two types of video token redundancy within each 2-second chunk—patches similar to the global background in the same frame (spatial redundancy) and patches unchanged relative to the previous frame in adjacent frames (temporal redundancy).
    - **Mechanism**: Processes the two frames within a chunk separately. The first frame $\mathbf{F}_1^{(t)}$ undergoes **spatial saliency**—first mean-pooling to get the frame representation $\bar{\mathbf{v}}_1^{(t)} = \frac{1}{n_p}\sum_i \mathbf{v}_{1,i}^{(t)}$, where each token's spatial score is its cosine distance from the mean $s_{1,i}^{(t)} = 1 - \frac{\mathbf{v}_{1,i}^{(t)} \cdot \bar{\mathbf{v}}_1^{(t)}}{\|\mathbf{v}_{1,i}^{(t)}\|\|\bar{\mathbf{v}}_1^{(t)}\|}$. High scores represent patches most distinct from the background. The second frame $\mathbf{F}_2^{(t)}$ undergoes **temporal saliency**—utilizing positional encoding for one-to-one correspondence, where the score is the cosine distance from the token at the same position in the first frame $s_{2,i}^{(t)} = 1 - \frac{\mathbf{v}_{2,i}^{(t)} \cdot \mathbf{v}_{1,i}^{(t)}}{\|\mathbf{v}_{2,i}^{(t)}\|\|\mathbf{v}_{1,i}^{(t)}\|}$. High scores represent "moving" regions. For both frames, top-$\hat{n}_p = \alpha_v n_p$ are selected based on visual retention ratio $\alpha_v = 1 - \rho_v$, concatenated to form $\hat{\mathbf{Z}}_v^{(t)} = [\hat{\mathbf{F}}_1^{(t)}; \hat{\mathbf{F}}_2^{(t)}]$.
    - **Design Motivation**: Using cosine distance for saliency avoids dependence on attention scores, ensuring FlashAttention compatibility; separate frames for spatial/temporal criteria avoid interference between dual axes—the first frame focuses on unique content, while the second focuses on changes within the second.

2. **VGAS: Vision-Guided Audio Selector**:

    - **Function**: Uses visual tokens retained after STVP pruning as query conditions to select a subset of original audio tokens most relevant to the current visual scene.
    - **Mechanism**: Treats $\hat{\mathbf{Z}}_v^{(t)}$ as a "visual anchor pool," calculating the correlation score between each audio token $\mathbf{Z}_a^{(t)}$ and all visual anchors, then selecting top-k based on ratio $\alpha_a$. This stage is the core of asymmetric design: audio saliency relies on visual scenes as conditions rather than internal audio signals (like audio attention used in OmniZip).
    - **Design Motivation**: Ours cites evidence from psychology/perception science (Koppen 2008, Zhao 2018) proving humans process audio-visual information asymmetrically—video redundancy is estimable from within, while audio significance depends on visual anchors (visible speakers, visually supported events). This implies effective Omni-LLM token compression should be "vision-guided" rather than treating both modalities symmetrically.

3. **STE End-to-End Fine-Tuning and Lightweight Parameter Budget**:

    - **Function**: Makes top-k selection in STVP and VGAS differentiable, allowing end-to-end optimization of the compression pipeline without retraining the backbone LLM.
    - **Mechanism**: Since top-k is discrete and non-differentiable in backpropagation, a straight-through estimator is used—forward pass handles hard selection, while the backward pass uses soft scores for gradient flow. The entire module introduces only 4.85M parameters (compared to the 7B parameters of Qwen2.5-Omni-7B). The backbone is frozen while only the compression module is trained. Latency is lower than the training-free baseline OmniZip as attention scores are unnecessary.
    - **Design Motivation**: Compared to EchoingPixels' 4 LLM decoding layers, OmniSIFT's 4.85M parameters constitute a true "lightweight plugin" that does not extend the inference path. STE is a mature solution for discrete selection and is fully compatible with FlashAttention.

### Loss & Training
Retains downstream task loss (standard next-token prediction), freezes the Qwen2.5-Omni backbone, and trains only the learnable parameters of the OmniSIFT module. Compression ratios $\rho_v, \rho_a$ are hyperparameters; the paper primarily tests 35% and 25% retention ratios.

## Key Experimental Results

### Main Results
Performance comparison on 5 audio-visual benchmarks (WorldSense, OmniVideoBench, three subsets of VideoMME, video-SALMONN-2 testset, DailyOmni) against OmniZip, Random, and DyCoke compression baselines and the full token model. Backbone models: Qwen2.5-Omni-7B / Qwen2.5-Omni-3B.

Results for Qwen2.5-Omni-7B at 25% retention ratio:

| Method | Retention | WorldSense ↑ | OmniVideoBench ↑ | VideoMME Avg ↑ | video-SALMONN-2 Total ↓ |
|------|-------|--------------|-------------------|----------------|-------------------------|
| Full Tokens | 100% | 49.7 | 35.6 | 67.6 | 48.1 |
| OmniZip | 25% | 48.1 | 34.1 | 66.0 | 57.2 |
| Random | 25% | 47.1 | 32.6 | 66.1 | 56.9 |
| DyCoke | 25% | 48.1 | 34.1 | 65.9 | 56.3 |
| **OmniSIFT** | **25%** | **49.9** | **35.4** | **68.2** | **51.2** |

At 35% retention for Qwen2.5-Omni-7B, OmniSIFT's WorldSense (50.0), OmniVideoBench (35.6), and VideoMME Avg (68.3) meet or exceed the full token baseline (49.7 / 35.6 / 67.6).

### Ablation Study
Comparison for Qwen2.5-Omni-3B (OmniSIFT maintains its advantage on smaller models):

| Method | Retention | WorldSense ↑ | OmniVideoBench ↑ | video-SALMONN-2 Total ↓ |
|------|-------|--------------|-------------------|-------------------------|
| Full Tokens | 100% | 45.8 | 33.5 | 53.6 |
| OmniZip | 25% | 43.8 | 32.4 | 62.1 |
| **OmniSIFT** | **25%** | **45.8** | **33.1** | **58.3** |

Additional parameters and latency: OmniSIFT introduces only 4.85M parameters (far lower than EchoingPixels' 4 decoding layers), and inference latency is lower than training-free OmniZip due to skipping attention score calculation.

### Key Findings
- **Outperforming full token models at 25% retention**: On WorldSense and VideoMME Avg, it exceeds the Full Tokens baseline (49.9 vs 49.7, 68.2 vs 67.6), indicating that many tokens are redundant or even harmful; removing them improves the signal-to-noise ratio.
- **Asymmetric > Symmetric**: The gap between Ours and OmniZip (Prev. SOTA in symmetric mode) is consistent across all benchmarks (+1.8 on WorldSense and -6.0 on video-SALMONN-2 Total at 25% retention), confirming vision-guided audio as the superior paradigm.
- **Scale-invariance**: Compression Gains are maintained across 7B and 3B backbones, suggesting the method is insensitive to model scale.
- **Hallucination improvement on video-SALMONN-2**: Total (Miss + Hal) decreased from 57.2 (OmniZip) to 51.2 (OmniSIFT), showing that retaining correct vision-audio aligned tokens reduces model hallucinations.

## Highlights & Insights
- **Deriving compression paradigms from perception science**: The authors designed asymmetric compression based on the asymmetry of human audiovisual processing. This "understand human mechanisms first, then engineer" approach is highly valuable for emerging Omni-LLM directions.
- **Avoiding attention score dependency**: Using cosine distance for saliency makes the method compatible with FlashAttention—an engineering design choice of high practical value, contrasted with OmniZip's dependency on attention scores.
- **Lightweight 4.85M parameters + low latency**: While other methods either add decoding layers (EchoingPixels) or incur attention overhead (OmniZip), OmniSIFT provides a true "plugin-level" solution.
- **"Less is more" outperforms "More is redundant"**: The counter-intuitive result of 25% tokens outperforming 100% tokens suggests a significant portion of Omni input sequences are noise; future work could explore even more aggressive compression ratios.
- **Separated spatial/temporal saliency frames**: Handling spatial and temporal criteria on separate frames within a chunk avoids interference between the two axes—a simple yet effective engineering trick.

## Limitations & Future Work
- **Fixed 2-second chunk granularity**: Hard-coded to the alignment granularity of Qwen2.5-Omni; porting to other Omni-LLMs with different chunk sizes would require parameter tuning.
- **2-frame assumption per chunk**: In fast-motion long videos, 2 frames may be insufficient to capture complete dynamics; the paper does not discuss variable frame rates or adaptive chunking.
- **VGAS cross-modal correlation details**: The abstract description of correlation computation is somewhat abstract; checking the code is necessary to confirm if it uses cosine similarity or more complex attention, leaving room for better interpretability.
- **Audio-guided vision scenarios**: In "audio-dominant, vision-auxiliary" scenarios (e.g., listening to music while viewing an album cover), whether unidirectional vision guidance remains optimal warrants study.
- **Training data and generalization**: The specific data used to train the OmniSIFT module is not explicitly stated; stability across new tasks or datasets requires further experimentation.

## Related Work & Insights
- **vs OmniZip (modality-symmetric)**: OmniZip uses audio attention for symmetric compression, while OmniSIFT uses cosine saliency for asymmetric compression—Ours is superior across all 5 benchmarks and compatible with FlashAttention.
- **vs EchoingPixels (modality-symmetric)**: EP adds 4 LLM decoding layers for global contextualization, which is costly and results in late compression; OmniSIFT uses 4.85M parameters for early compression, making it much more engineering-friendly.
- **vs FASTAV / DyCoke**: These methods primarily perform audio-visual pruning during the LLM inference stage; OmniSIFT compresses before the LLM input, allowing for independent deployment.
- **vs Vision-centric methods (VidCom2 / TimeChat-Online)**: These only process the visual stream; OmniSIFT implements insights from vision methods (spatial + temporal redundancy) and extends them to guide audio.
- **vs General visual token compression (FastV, PruMerge, etc.)**: These works established the "pruning via structural signals" paradigm; OmniSIFT is a natural extension of this line of work for Omni-models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of asymmetric compression explicitly opposes the previous symmetric paradigm, and the combination of cosine saliency + vision-guided audio is a fresh design for Omni-LLMs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 5 benchmarks + 2 model sizes + multiple compression ratios, with clear parameter and latency comparisons; however, it lacks direct comparison with EchoingPixels.
- **Writing Quality**: ⭐⭐⭐⭐ The three-stage design principles → two-stage architecture → experimental chain is very clear, with clean mathematical notation.
- **Value**: ⭐⭐⭐⭐ Highly practical plugin for Omni-LLM deployment—4.85M parameters + FlashAttention compatibility + no performance drop at 25% tokens offers high industrial value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models](../../CVPR2026/video_understanding/token_reduction_via_local_and_global_contexts_optimization_for_efficient_video_l.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](../../CVPR2026/video_understanding/streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[ICLR 2026\] FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](../../ICLR2026/video_understanding/flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)
- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](../../ICLR2026/video_understanding/floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](../../ICCV2025/video_understanding/4dbench_benchmarking_multimodal_large_language_models_for_4d.md)

</div>

<!-- RELATED:END -->
