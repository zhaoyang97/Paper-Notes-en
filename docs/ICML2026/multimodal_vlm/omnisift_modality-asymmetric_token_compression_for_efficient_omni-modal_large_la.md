---
title: >-
  [Paper Note] OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models
description: >-
  [ICML 2026][Multimodal VLM][Omni-LLM] This paper identifies that existing Omni-LLM token compression methods treat audio and video "symmetrically," which is suboptimal. It proposes OmniSIFT—a two-stage…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Omni-LLM"
  - "Token Compression"
  - "Video-Audio Understanding"
  - "Spatio-Temporal Pruning"
  - "Vision Guidance"
date: 2026-05-08
content_hash: 6f6052e42b036a8c
---

# OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.04804](https://arxiv.org/abs/2602.04804)  
**Code**: https://github.com/dingyue772/OmniSIFT  
**Area**: Multimodal VLM / Video Understanding / Model Compression  
**Keywords**: Omni-LLM, Token Compression, Video-Audio Understanding, Spatio-Temporal Pruning, Vision Guidance

## TL;DR
This paper identifies that existing Omni-LLM token compression methods treat audio and video "symmetrically," which is suboptimal. It proposes OmniSIFT—a two-stage, modality-asymmetric compression framework: first, spatio-temporal saliency prunes video redundancy to obtain "visual anchors," then these anchors guide audio token selection. With only 4.85M extra parameters, OmniSIFT consistently outperforms existing compression baselines and even the original model on Qwen2.5-Omni-7B when retaining 25% of tokens.

## Background & Motivation

**Background**: Omni-LLMs (Qwen2.5-Omni, GPT-4o, Gemini) unify video, audio, and text into an autoregressive LLM for joint reasoning. However, video consists of dense continuous frames, and audio requires high temporal resolution encoding—a 20-second multimodal clip can generate over 20K tokens, making inference computationally expensive due to long token sequences.

**Limitations of Prior Work**: Vision-centric MLLMs have extensive research on token compression (FastV, VidCom2, TimeChat-Online, etc.), but direct transfer to Omni-LLMs is infeasible. Existing Omni compression methods fall into two categories: (1) modality-decoupled—audio and video are compressed independently, ignoring cross-modal semantic dependencies; (2) modality-symmetric—OmniZip uses audio attention scores to guide video pruning (reliance on attention scores makes it incompatible with FlashAttention), EchoingPixels adds 4 LLM decoder layers for global cross-modal contextualization (costly, delayed compression). Both treat audio and video as equally informative sources.

**Key Challenge**: Human perception of audio and video is inherently asymmetric—video redundancy can be estimated within the visual stream (spatial redundancy within frames + temporal redundancy across frames), but audio saliency depends more on context, often requiring visual scenes as semantic anchors (visible speakers, visually supported events). Symmetric treatment collapses the compression task into "selecting temporal positions," ignoring modality-specific semantic cues.

**Goal**: (1) Enable compression to follow a vision-guided, asymmetric paradigm; (2) Remain lightweight (extra parameters ≪ backbone); (3) Be compatible with efficient operators like FlashAttention (no reliance on attention scores).

**Key Insight**: First, prune video redundancy using pure structural signals (cosine distance) to obtain compact "visual anchors," then use these anchors to guide audio token selection. Thus, video compression uses intra-modal signals, while audio compression is conditioned cross-modally, with clear division of labor.

**Core Idea**: Modality-asymmetric, vision-guided two-stage compression—STVP (Spatio-Temporal Video Pruning) performs spatial saliency within frames and temporal saliency across frames, while VGAS (Vision-Guided Audio Selector) uses the pruned visual anchors as conditions to select audio tokens.

## Method

### Overall Architecture
Input: Video $\mathcal{V}$ and synchronized audio $\mathcal{A}$ are mapped by Qwen2.5-Omni's encoder-projector into token sequences $\mathbf{Z}_v \in \mathbb{R}^{N_v \times D}$ and $\mathbf{Z}_a \in \mathbb{R}^{N_a \times D}$. To maintain temporal alignment, tokens are chunked into $\mathcal{C}_t = [\mathbf{Z}_v^{(t)}; \mathbf{Z}_a^{(t)}]$ per 2-second segment, each containing 2 video frames and corresponding audio. OmniSIFT serially executes two stages at the chunk level: (1) STVP prunes visual redundancy in each chunk to yield compressed visual sequence $\hat{\mathbf{Z}}_v^{(t)}$; (2) VGAS uses $\hat{\mathbf{Z}}_v^{(t)}$ as a condition to select audio tokens from $\mathbf{Z}_a^{(t)}$. The entire framework is end-to-end differentiable (using straight-through estimator for top-k selection), optimizing token selection to preserve downstream task performance during training.

### Key Designs

1. **STVP: Dual-Axis Spatial and Temporal Saliency Pruning**:

    - **Function**: Within each 2-second chunk, prunes two types of video token redundancy—patches similar to the global background within the same frame (spatial redundancy), and patches with little change compared to the previous frame (temporal redundancy).
    - **Mechanism**: Processes the two frames in each chunk separately. The first frame $\mathbf{F}_1^{(t)}$ undergoes **spatial saliency**—mean-pool to obtain frame representation $\bar{\mathbf{v}}_1^{(t)} = \frac{1}{n_p}\sum_i \mathbf{v}_{1,i}^{(t)}$, and each token's spatial score is its cosine distance from the mean $s_{1,i}^{(t)} = 1 - \frac{\mathbf{v}_{1,i}^{(t)} \cdot \bar{\mathbf{v}}_1^{(t)}}{\|\mathbf{v}_{1,i}^{(t)}\|\|\bar{\mathbf{v}}_1^{(t)}\|}$—higher scores indicate patches most distinct from the background. The second frame $\mathbf{F}_2^{(t)}$ undergoes **temporal saliency**—using positional encoding for correspondence, the score is the cosine distance to the same-position token in the first frame $s_{2,i}^{(t)} = 1 - \frac{\mathbf{v}_{2,i}^{(t)} \cdot \mathbf{v}_{1,i}^{(t)}}{\|\mathbf{v}_{2,i}^{(t)}\|\|\mathbf{v}_{1,i}^{(t)}\|}$—higher scores indicate "moving" regions. Each frame selects top-$\hat{n}_p = \alpha_v n_p$ tokens by visual retention ratio $\alpha_v = 1 - \rho_v$, concatenated as $\hat{\mathbf{Z}}_v^{(t)} = [\hat{\mathbf{F}}_1^{(t)}; \hat{\mathbf{F}}_2^{(t)}]$.
    - **Design Motivation**: Using cosine distance for saliency avoids attention score dependency, ensuring compatibility with FlashAttention; separating spatial/temporal criteria for the two frames prevents interference—first frame focuses on "unique content," second on "what changed this second."

2. **VGAS: Vision-Anchored Audio Token Selection**:

    - **Function**: Uses visual tokens retained by STVP as query anchors to select the subset of original audio tokens most relevant to the current visual scene.
    - **Mechanism**: Treats $\hat{\mathbf{Z}}_v^{(t)}$ as a "visual anchor pool," computes the relevance score between each audio token $\mathbf{Z}_a^{(t)}$ and all visual anchors, and selects top-k by ratio $\alpha_a$. This stage embodies the core of asymmetric design—audio saliency is not based on internal audio signals (as in OmniZip's audio attention), but is entirely conditioned on the visual scene.
    - **Design Motivation**: The authors cite psychological/perceptual science evidence (Koppen 2008, Zhao 2018) that human audiovisual processing is asymmetric—video redundancy is internally estimable, while audio saliency depends on visual anchors (e.g., visible speakers, visually supported events). Thus, effective token compression for Omni-LLMs should be "vision-guided" rather than treating both modalities symmetrically.

3. **STE End-to-End Fine-Tuning and Lightweight Parameter Budget**:

    - **Function**: Enables STVP and VGAS top-k selection to be differentiable, allowing end-to-end optimization of the compression pipeline without retraining the backbone LLM.
    - **Mechanism**: Top-k selection is discrete and non-differentiable in the backward pass, so a straight-through estimator is used—hard selection in the forward pass, soft scores for gradient flow in the backward pass. The entire module introduces only 4.85M extra parameters (compared to Qwen2.5-Omni-7B's 7B), with the backbone frozen and only the compression module trained. Inference latency is even lower than the training-free baseline OmniZip, as attention scores need not be computed.
    - **Design Motivation**: Compared to EchoingPixels, which adds 4 LLM decoder layers for global contextualization, OmniSIFT's 4.85M parameters make it a true "lightweight plugin" without lengthening the inference path. STE is a mature engineering solution for discrete selection and is fully compatible with FlashAttention.

### Loss & Training
Downstream task loss (standard next-token prediction) is preserved, with the Qwen2.5-Omni backbone frozen and only the learnable parameters of the OmniSIFT module trained. Compression rates $\rho_v, \rho_a$ are hyperparameters; the paper mainly evaluates 35% and 25% retention ratios.

## Key Experimental Results

### Main Results
On five audio-video benchmarks (WorldSense, OmniVideoBench, three VideoMME subsets, video-SALMONN-2 testset, DailyOmni), OmniSIFT is compared with compression baselines OmniZip, Random, DyCoke, and the full-token model. Backbone models: Qwen2.5-Omni-7B / Qwen2.5-Omni-3B.

Qwen2.5-Omni-7B at 25% retention:

| Method | Retention | WorldSense ↑ | OmniVideoBench ↑ | VideoMME Avg ↑ | video-SALMONN-2 Total ↓ |
|--------|-----------|--------------|------------------|----------------|-------------------------|
| Full Tokens | 100% | 49.7 | 35.6 | 67.6 | 48.1 |
| OmniZip | 25% | 48.1 | 34.1 | 66.0 | 57.2 |
| Random | 25% | 47.1 | 32.6 | 66.1 | 56.9 |
| DyCoke | 25% | 48.1 | 34.1 | 65.9 | 56.3 |
| **OmniSIFT** | **25%** | **49.9** | **35.4** | **68.2** | **51.2** |

At 35% retention, OmniSIFT on Qwen2.5-Omni-7B achieves WorldSense (50.0), OmniVideoBench (35.6), VideoMME Avg (68.3), all matching or exceeding the full-token baseline (49.7 / 35.6 / 67.6).

### Ablation Study
The paper reports results on Qwen2.5-Omni-3B (the advantage of OmniSIFT holds for smaller models):

| Method | Retention | WorldSense ↑ | OmniVideoBench ↑ | video-SALMONN-2 Total ↓ |
|--------|-----------|--------------|------------------|-------------------------|
| Full Tokens | 100% | 45.8 | 33.5 | 53.6 |
| OmniZip | 25% | 43.8 | 32.4 | 62.1 |
| **OmniSIFT** | **25%** | **45.8** | **33.1** | **58.3** |

Extra parameters and latency: OmniSIFT introduces only 4.85M parameters (far less than EchoingPixels with 4 decoder layers), and inference latency is lower than the training-free OmniZip, as attention scores are not computed.

### Key Findings
- **Outperforms full-token model at 25% retention**: On WorldSense and VideoMME Avg, OmniSIFT even surpasses the full-token baseline (49.9 vs 49.7, 68.2 vs 67.6), indicating that most tokens are redundant or even harmful, and removing them improves signal-to-noise ratio.
- **Asymmetric > Symmetric**: The gap with OmniZip (symmetric SOTA) is consistent across all benchmarks (at 25% retention, WorldSense +1.8, video-SALMONN-2 Total -6.0), confirming vision-guided audio as a superior paradigm.
- **Consistent across model sizes**: Compression benefits hold for both 7B and 3B backbones, indicating robustness to model scale.
- **Significant improvement in video-SALMONN-2 hallucination metric**: Total (Miss + Hal) drops from OmniZip's 57.2 to OmniSIFT's 51.2, showing that retaining correct visual-audio aligned tokens reduces model hallucination.

## Highlights & Insights
- **Compression paradigm inspired by perceptual science**: The design of asymmetric compression based on human audiovisual processing is a valuable approach for emerging Omni-LLM research.
- **Avoids attention score dependency**: Using only cosine distance for saliency ensures compatibility with FlashAttention—a valuable engineering choice, as OmniZip is locked by attention score dependency.
- **Lightweight 4.85M parameters + low latency**: While other compression methods either add decoder layers (EchoingPixels) or incur attention computation overhead (OmniZip), OmniSIFT offers a true "plugin-level" solution.
- **"Less is more"**: The counterintuitive result that 25% tokens outperform 100% tokens suggests that a significant portion of Omni input tokens are noise; future work can explore even higher compression rates.
- **Separate spatial/temporal saliency for two frames**: Avoiding interference by not mixing both axes in a single frame is a simple yet effective engineering trick.

## Limitations & Future Work
- **Fixed 2-second chunk granularity**: Hard-coded to Qwen2.5-Omni's alignment granularity; for other Omni-LLMs (with different chunk sizes), hyperparameters need retuning, limiting portability.
- **Assumption of only 2 frames per chunk**: For long videos or fast motion, 2 frames may not capture full dynamics; the paper does not discuss variable frame rates or adaptive chunking.
- **Details of VGAS cross-modal relevance computation**: The paper's description is abstract; code inspection is needed to confirm whether cosine similarity or a more complex attention mechanism is used, leaving room for improved interpretability.
- **Reverse scenario of audio-guided vision**: In audio-dominant, vision-auxiliary scenarios (e.g., listening to music with album cover), whether unidirectional vision guidance remains optimal is worth investigating.
- **Training data and generalization**: The paper does not specify which data OmniSIFT was trained on; more experiments are needed to assess cross-domain generalization (new tasks, new datasets).

## Related Work & Insights
- **vs OmniZip (modality-symmetric)**: OmniZip uses audio attention for symmetric compression, while OmniSIFT uses cosine saliency for asymmetric compression—outperforming on all five benchmarks and compatible with FlashAttention.
- **vs EchoingPixels (modality-symmetric)**: EP adds 4 LLM decoder layers for global contextualization, which is costly and delays compression; OmniSIFT uses 4.85M parameters for early-stage compression, making it much more engineering-friendly.
- **vs FASTAV / DyCoke**: These methods mainly prune audio-video during LLM inference; OmniSIFT compresses before LLM input, allowing independent deployment.
- **vs Vision-centric methods (VidCom2 / TimeChat-Online)**: These only process the visual stream; OmniSIFT concretely implements insights from vision methods (spatial + temporal redundancy) and extends them to audio guidance.
- **vs General visual token compression research (FastV, PruMerge, etc.)**: These works established the "structural signal-based token pruning" paradigm; OmniSIFT is a natural extension of this line to Omni models.

## Rating
- Novelty: ⭐⭐⭐⭐ The asymmetric compression approach explicitly challenges prior symmetric paradigms; the combination of cosine saliency and vision-guided audio is a new design for Omni-LLMs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five benchmarks, two model sizes, multiple compression rates, and clear parameter/latency comparisons; unfortunately, no direct comparison with EchoingPixels.
- Writing Quality: ⭐⭐⭐⭐ Three-stage design principles → two-stage architecture → clear experimental chain, with well-organized formulas and notation.
- Value: ⭐⭐⭐⭐ A practical plugin for Omni-LLM deployment—4.85M parameters, FlashAttention compatibility, and 25% tokens with no performance drop, offering high industrial value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression](on_the_adversarial_robustness_of_large_vision-language_models_under_visual_token.md)
- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[CVPR 2026\] ApET: Approximation-Error Guided Token Compression for Efficient VLMs](../../CVPR2026/multimodal_vlm/apet_approximation-error_guided_token_compression_for_efficient_vlms.md)
- [\[ICML 2026\] Jailbreaking Vision-Language Models Through the Visual Modality](jailbreaking_vision-language_models_through_the_visual_modality.md)
- [\[ICML 2026\] CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models](clip_tricks_you_training-free_token_pruning_for_efficient_pixel_grounding_in_lar.md)

</div>

<!-- RELATED:END -->
