---
title: >-
  [Paper Note] ARGenSeg: Image Segmentation with Autoregressive Image Generation Model
description: >-
  [NeurIPS 2025][Segmentation][Autoregressive image generation] This paper proposes ARGenSeg — the first unified MLLM framework that leverages the autoregressive image generation paradigm for image segmentation. The model directly outputs visual tokens decoded by a VQ-VAE into segmentation masks, requiring no additional segmentation head. A next-scale prediction parallel generation strategy enables 4× inference speedup, and the method surpasses state-of-the-art on RefCOCO/+/g with significantly less training data.
tags:
  - NeurIPS 2025
  - Segmentation
  - Autoregressive image generation
  - VQ-VAE segmentation
  - MLLM unified framework
  - Next-Scale Prediction
  - unified understanding and generation
date: 2026-05-08
content_hash: d70bd7be8e04e481
---

# ARGenSeg: Image Segmentation with Autoregressive Image Generation Model

**Conference**: NeurIPS 2025
**arXiv**: [2510.20803](https://arxiv.org/abs/2510.20803)
**Code**: None
**Area**: Image Segmentation / Multimodal Large Language Models
**Keywords**: Autoregressive image generation, VQ-VAE segmentation, MLLM unified framework, Next-Scale Prediction, unified understanding and generation

## TL;DR
This paper proposes ARGenSeg — the first unified MLLM framework that leverages the autoregressive image generation paradigm for image segmentation. The model directly outputs visual tokens decoded by a VQ-VAE into segmentation masks, requiring no additional segmentation head. A next-scale prediction parallel generation strategy enables 4× inference speedup, and the method surpasses state-of-the-art on RefCOCO/+/g with significantly less training data.

## Background & Motivation

**Background**: Integrating image segmentation into MLLMs is a current research hotspot. Two dominant paradigms exist: (a) boundary point sequence representations (e.g., PolyFormer), which discretize masks into polygon point sequences but fail on complex shapes; and (b) dedicated segmentation decoders (e.g., LISA, PSALM), which use special tokens or hidden states to drive SAM/Mask2Former decoders, resulting in complex architectures where the LLM itself does not learn pixel-level understanding.

**Limitations of Prior Work**: (a) Point sequence representations lead to incomplete segmentation and unnatural boundaries; (b) dedicated decoders make LLMs dependent on external modules rather than learning fine-grained visual understanding internally; (c) inference is slow in methods such as HiMTok.

**Key Challenge**: Segmentation requires dense pixel-level output, whereas LLMs natively perform token-level prediction — the fundamental challenge is enabling an LLM to "generate" segmentation masks without relying on an external decoder.

**Goal**: Enable MLLMs to directly produce segmentation masks via autoregressive image generation, without any additional segmentation head.

**Key Insight**: Treat segmentation as a special case of image generation, where the generated "image" is the target object's mask.

**Core Idea**: The MLLM outputs VQ-VAE visual tokens → the VQ-VAE decoder reconstructs them into a mask image → no external segmentation decoder is needed, and segmentation capability derives entirely from the MLLM's pixel-level understanding.

## Method

### Overall Architecture
Built upon InternVL 2.5. Input: an image (continuous features via a vision encoder) and a text instruction (tokenizer). Output: when segmentation is required, the MLLM outputs visual tokens, which are decoded by VQ-VAE into a mask image. Understanding and generation tasks share a unified prediction head.

### Key Designs

1. **Unified Visual Token Prediction**

    - **Function**: Enable the MLLM to directly predict visual token IDs from the VQ-VAE codebook.
    - **Mechanism**: Tokens from the VQ-VAE codebook (size = 4096) are added to the LLM vocabulary as new "words." When generating a segmentation mask, the model begins predicting visual tokens upon encountering the `<gen_start>` marker. A unified classification head handles both text and visual token prediction, supervised with cross-entropy loss (GT visual tokens are obtained from the VQ-VAE encoder during training).
    - **Design Motivation**: By avoiding the special-token + external-decoder paradigm, the LLM must learn pixel-level information internally in order to predict correct visual tokens. Ablations confirm this is critical for achieving high accuracy.

2. **Next-Scale Prediction for Acceleration**

    - **Function**: Adopt the VAR multi-scale generation strategy, generating all tokens at each scale in parallel.
    - **Mechanism**: A VAR tokenizer quantizes features into $K=10$ scale-wise token maps $(r_1, \ldots, r_{10})$. At each step, all $h_k \times w_k$ tokens of the current scale are generated in parallel, with the upsampled token map from the previous step serving as the current query. A 256×256 image is represented by 680 visual tokens requiring only 10 autoregressive steps.
    - **Design Motivation**: (a) Coarse-to-fine multi-scale generation aligns with the intuition of "localize then refine" in segmentation; (b) over 4× faster than sequential per-token generation.

3. **Training Strategy: Single-Stage Joint Training**

    - **Function**: Joint SFT on segmentation data (402K) and understanding data (1.25M).
    - **Mechanism**: The vision encoder and VQ-VAE are frozen throughout; only the LLM and projector are trained. Pretrained multimodal understanding capabilities enable rapid convergence. The segmentation data (402K) is far fewer than the 2.91M used by HiMTok.
    - **Design Motivation**: Freezing the tokenizer ensures the LLM must learn pixel-level information on its own rather than relying on a learnable decoder.

### Loss & Training
- Unified cross-entropy loss applied to both text token and visual token prediction.

## Key Experimental Results

### Main Results (Referring Segmentation — RefCOCO/+/g cIoU)

| Method | Paradigm | RefCOCO val | RefCOCO+ val | RefCOCOg val | Training Data |
|--------|----------|-------------|--------------|--------------|---------------|
| LISA-7B (ft) | Dedicated head | 74.9 | 65.1 | 67.9 | — |
| PSALM | Dedicated head | 83.6 | 72.9 | 73.8 | — |
| HiMTok-8B | Generative (dedicated tokenizer) | 81.1 | 77.1 | 75.8 | 2.91M |
| HiMTok-8B (ft) | Same | 85.0 | 79.7 | 80.0 | 2.91M |
| **ARGenSeg** | **Generative (general VQ-VAE)** | **82.2** | **77.9** | **78.4** | **402K** |
| **ARGenSeg (ft)** | Same | **86.3** | **82.3** | **81.7** | **402K** |

### Inference Speed Comparison

| Method | Inference Time/Image | Speedup |
|--------|----------------------|---------|
| HiMTok | ~4× baseline | 1× |
| UniGS (diffusion) | ~10× | 0.4× |
| **ARGenSeg** | **~1×** | **4×+** |

### Key Findings
- **SOTA without any segmentation head**: ARGenSeg is the first unified framework to surpass all dedicated-head methods without requiring any segmentation head.
- **High data efficiency**: ARGenSeg exceeds HiMTok (RefCOCO val: 86.3 vs. 85.0) using only 402K segmentation samples versus 2.91M.
- **Direct visual token output is critical**: Ablations show that replacing direct visual token prediction with a LISA-style hidden-state + decoder scheme results in a notable performance drop.
- **Multi-scale generation improves robustness**: The coarse-to-fine process not only accelerates inference but also improves segmentation quality.
- **Extensible to image generation**: A small amount of additional training data unlocks text-to-image generation capability, validating the generality of the framework.

## Highlights & Insights
- **Segmentation = Image Generation**: Reframing segmentation as conditional image generation (where the generated "image" is a mask) is both conceptually elegant and practically effective. This entirely eliminates the need for dedicated segmentation heads and enables end-to-end pixel-level learning within the MLLM.
- **General VQ-VAE vs. Dedicated Tokenizer**: HiMTok requires training a specialized mask tokenizer, whereas ARGenSeg employs a general-purpose VQ-VAE — a more universal choice that naturally extends to other generation tasks.
- **Importance of Freezing the Tokenizer**: Freezing the VQ-VAE ensures that segmentation quality depends entirely on the MLLM's understanding capacity, a design choice aligned with the principle of "understanding-driven segmentation."

## Limitations & Future Work
- Output resolution is fixed at 256×256; high-resolution segmentation may require additional scales.
- VQ-VAE reconstruction quality constitutes a performance ceiling — a superior tokenizer could yield further gains.
- Evaluation on instance segmentation and panoptic segmentation is less thorough than on referring segmentation.
- End-to-end joint training of the tokenizer and LLM remains unexplored.

## Related Work & Insights
- **vs. LISA/PSALM**: These methods use special token embeddings to drive SAM/Mask2Former, with the LLM providing only semantic information without processing pixels. ARGenSeg enables the LLM to directly predict pixel-level tokens.
- **vs. HiMTok**: Both adopt a generative paradigm, but HiMTok relies on a dedicated mask tokenizer and 2.91M training samples, whereas ARGenSeg achieves superior results with a general VQ-VAE and only 402K samples.
- **vs. Janus/Emu3**: These are unified understanding-and-generation frameworks that do not perform segmentation; ARGenSeg demonstrates that such unified frameworks can be extended to pixel-level perception.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "segmentation = image generation" paradigm shift and achieving SOTA without any segmentation head represent a breakthrough contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on referring, generalized, and reasoning segmentation benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear with well-motivated design choices.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for pixel-level perception in unified MLLM frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Seg-VAR: Image Segmentation with Visual Autoregressive Modeling](seg-var_image_segmentation_with_visual_autoregressive_modeling.md)
- [\[NeurIPS 2025\] MultiHuman-Testbench: Benchmarking Image Generation for Multiple Humans](multihuman-testbench_benchmarking_image_generation_for_multiple_humans.md)
- [\[ICCV 2025\] Latent Expression Generation for Referring Image Segmentation and Grounding](../../ICCV2025/segmentation/latent_expression_generation_for_referring_image_segmentation_and_grounding.md)
- [\[NeurIPS 2025\] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)
- [\[NeurIPS 2025\] SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation](safire_saccade-fixation_reiteration_with_mamba_for_referring_image_segmentation.md)

</div>

<!-- RELATED:END -->
