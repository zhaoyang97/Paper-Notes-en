---
title: >-
  [Paper Note] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding
description: >-
  [ICCV 2025][Multimodal VLM][unified VLM] This paper proposes a Semantic Discrete Encoding (SDE) visual tokenizer that augments VQGAN with SigLIP semantic feature constraints…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "unified VLM"
  - "visual tokenizer"
  - "semantic discrete encoding"
  - "autoregressive"
  - "image generation"
date: 2026-05-08
content_hash: 3fad1b402a799387
---

# MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding

**Conference**: ICCV 2025
**arXiv**: [2411.17762](https://arxiv.org/abs/2411.17762)  
**Code**: None  
**Area**: Multimodal VLM / Unified Understanding and Generation
**Keywords**: unified VLM, visual tokenizer, semantic discrete encoding, autoregressive, image generation

## TL;DR
This paper proposes a Semantic Discrete Encoding (SDE) visual tokenizer that augments VQGAN with SigLIP semantic feature constraints, enabling discrete visual tokens to align semantically with language tokens. Built upon SDE, a unified autoregressive VLM (MUSE-VL) is constructed that, using only 24M training samples, outperforms Emu3 by 4.8% on understanding benchmarks, surpasses the specialist model LLaVA-NeXT 34B by 3.7%, and simultaneously supports image generation.

## Background & Motivation

**Background**: Unified multimodal understanding and generation is a critical research direction for VLMs. Existing unified models (e.g., Chameleon, Show-o, Emu3) employ visual tokenizers such as VQGAN to convert images into discrete tokens, which are then jointly processed with text tokens via next-token prediction. However, VQGAN focuses exclusively on low-level pixel information (reconstruction loss), yielding discrete tokens that lack semantic content.

**Limitations of Prior Work**: (1) A large semantic gap exists between VQGAN tokens and language tokens, causing unified models to underperform specialist models significantly on understanding tasks. (2) Models like Chameleon require training the LLM from scratch, imposing prohibitive data and compute requirements. (3) VILA-U attempts to align semantics by jointly training contrastive and reconstruction losses, but suffers from severe loss conflicts that hinder convergence.

**Key Challenge**: Visual tokens must simultaneously satisfy two conflicting requirements—preserving low-level pixel information for image reconstruction/generation while carrying high-level semantic information for understanding. VQGAN addresses only the former, CLIP only the latter, and naively combining them (as in VILA-U) leads to conflicts.

**Goal**: Design a visual tokenizer whose discrete codes retain image reconstruction capability while encoding rich semantic information, thereby reducing the training difficulty of unified VLMs.

**Key Insight**: Rather than relying on contrastive learning (which causes VILA-U's problems), the proposed approach extracts semantic features using a pretrained SigLIP image encoder and fuses them into the encoding process, with two decoder branches—a semantic decoder (reconstructing SigLIP features) and an image decoder (reconstructing pixels)—jointly supervising the tokenizer.

**Core Idea**: Fuse SigLIP semantic features into the visual quantization process and train the visual tokenizer with dual branches (semantic reconstruction + image reconstruction), so that discrete codes inherently encode semantic information.

## Method

### Overall Architecture
The SDE Tokenizer encodes images into discrete token sequences of size $16 \times 16$ or $27 \times 27$, which are fed alongside text tokens into an autoregressive Transformer (based on Qwen2.5/Yi-1.5 LLMs) and trained with standard cross-entropy next-token prediction. For understanding tasks, visual tokens appear in the prompt and the model outputs textual responses. For generation tasks, textual descriptions serve as prompts, and the model outputs visual tokens that are subsequently decoded by the image decoder.

### Key Designs

1. **Semantic Discrete Encoding (SDE) Tokenizer**:

    - *Function*: Discretize images into visual tokens that jointly encode pixel-level and semantic information.
    - *Mechanism*: An encoder initialized from SigLIP first extracts features $z$ from the input image, while a frozen SigLIP model extracts semantic features $T$. The two are summed and then vector-quantized: $z_q = \text{Quant}(T + z)$. The quantized features are then passed to two decoder branches: (1) a semantic decoder (Transformer) that reconstructs SigLIP features, with loss $L_{\text{sem}} = 1 - \cos(Dec_s(z_q), T)$; and (2) an image decoder (ConvNet) that reconstructs the original pixels, with loss $L_{\text{img}} = \ell_2 + L_P + \lambda_G L_G$ (pixel + perceptual + adversarial). The total loss is $L = L_{\text{sem}} + L_{\text{img}} + L_{\text{vq}}$.
    - *Design Motivation*: Injecting semantics via feature fusion (rather than contrastive learning) avoids the loss conflicts of VILA-U. The semantic decoder ensures that the quantized codes retain semantic information, while the image decoder ensures that pixel-level information is preserved. The codebook has size 32,768 and dimension 8.
    - *Key Difference from VILA-U*: VILA-U extracts semantics via a text encoder and contrastive learning, leading to conflicts; SDE directly fuses and reconstructs features from the SigLIP image encoder, resulting in more stable training.

2. **Unified Autoregressive Modeling**:

    - *Function*: Process both discrete visual tokens and text tokens uniformly within a single autoregressive model.
    - *Mechanism*: The embedding layer of an existing LLM (Qwen2.5/Yi-1.5) is extended by 32,768 dimensions (matching the codebook size), with `<soi>` and `<eoi>` tokens marking the boundaries of visual token sequences. The training objective is purely next-token prediction, requiring no architectural modifications to the LLM.
    - *Design Motivation*: Because SDE tokens are already semantically aligned with language, no additional adapters or architectural changes are needed, allowing the model to directly leverage the linguistic capabilities of the pretrained LLM and substantially reducing training complexity.

3. **Two-Stage Training**:

    - **Pretraining**: Loss is computed over all tokens on image–text pair data to learn visual token embeddings and achieve alignment.
    - **Instruction Tuning**: For understanding tasks, SFT data is used with loss computed only on response tokens; for generation tasks, inverted image–text pairs (text → image) are used with loss computed only on visual tokens.

### Data Efficiency
MUSE-VL uses only 24M image–text pairs, far fewer than Show-o (35M) and VILA-U (720M), yet achieves superior understanding performance.

## Key Experimental Results

### Main Results (Multimodal Understanding)

| Model | LLM | Token Type | MMBench | MMStar | SEED | MMMU | SQA-I | AI2D | MathVista | AVG |
|------|-----|---------|---------|--------|------|------|-------|------|-----------|-----|
| Chameleon-7B | 7B from scratch | Discrete | 31.1 | 31.1 | 30.6 | 25.4 | 46.8 | 46.0 | 22.3 | 33.3 |
| Emu3-8B | 8B from scratch | Discrete | 58.5 | 46.6 | 68.2 | 31.6 | 89.2 | 70.0 | 47.6 | 58.8 |
| LLaVA-NeXT-7B | Vicuna-7B | Continuous | 67.4 | 37.6 | 70.2 | 35.8 | 70.1 | 66.6 | 34.6 | 54.6 |
| LLaVA-NeXT-34B | Yi-34B | Continuous | 79.3 | 51.6 | 75.9 | 51.1 | 81.8 | 78.9 | 46.5 | 66.4 |
| **MUSE-VL-7B** | Qwen2.5-7B | Discrete | **72.1** | **49.6** | **69.1** | **39.7** | **93.5** | **69.8** | **51.3** | **63.6** |
| **MUSE-VL-32B** | Qwen2.5-32B | Discrete | **81.8** | **56.7** | 71.0 | 50.1 | **95.0** | **79.9** | **55.9** | **70.1** |

Tokenizer comparison (same LLM: Yi-1.5-9B; same data):

| Tokenizer | MMBench | SEED | MMStar | AVG |
|-----------|---------|------|--------|-----|
| VQGAN | 32.0 | 42.7 | 29.1 | 34.6 |
| SEED | 63.1 | 57.8 | 39.1 | 53.3 |
| LaVIT | 63.3 | 59.5 | 40.3 | 54.4 |
| **SDE (Ours)** | **70.6** | **68.1** | **43.8** | **60.8** |

### Ablation Study

| Image Branch | Semantic Branch | rFID↓ | MMBench | SEED | MMStar | AVG |
|---------|---------|-------|---------|------|--------|-----|
| ✓ | ✗ | 2.63 | 42.8 | 48.5 | 38.1 | 43.1 |
| ✗ | ✓ | - | 72.5 | 67.5 | 48.1 | 62.7 |
| ✓ | ✓ | 2.26 | 72.1 | 69.1 | 49.6 | 63.6 |

### Key Findings
- **Semantics is the critical gap**: A VLM using a pure VQGAN tokenizer achieves only 34.6% average understanding score; adding semantic constraints raises it to 60.8% (+26.2%), directly demonstrating that the lack of semantic information in discrete tokens is the primary cause of poor understanding in unified VLMs.
- **SDE improves understanding by 20.5% over VQGAN while maintaining comparable image reconstruction quality** (rFID 2.26 vs. 2.63), indicating that semantic constraints do not compromise generation capability and may in fact improve it.
- **Exceptional data efficiency**: 24M samples outperforms VILA-U trained on 720M samples, as semantic alignment substantially reduces the difficulty for the LLM to learn from visual tokens.
- **Strong scaling behavior**: Performance improves consistently from 7B to 32B (AVG: 63.6 → 70.1).
- MUSE-VL-7B surpasses LLaVA-NeXT-34B on SQA-I (93.5) and MathVista (51.3), suggesting that discrete token approaches have advantages on reasoning-intensive tasks.
- Visualization of visual codes reveals that semantically similar concepts (e.g., cat ears, strawberries) are consistently assigned the same code IDs.

## Highlights & Insights
- **SDE is an elegant design**: By replacing contrastive learning with "semantic feature fusion + dual-branch decoding," it avoids the loss conflict problem of VILA-U. The underlying insight is that SigLIP's image encoder features already implicitly encode text-aligned semantic information; explicit contrastive learning is unnecessary—semantics need only be "fused into" and "preserved through" the quantization process.
- **Paradigmatic simplicity of the unified model**: No LLM architectural modifications, no additional adapters, no diffusion models—purely next-token prediction. This demonstrates that, given a sufficiently capable tokenizer, a unified model can be remarkably simple.
- **Implications for unified VLM research**: The bottleneck lies not in the LLM architecture but in the visual tokenizer. A well-designed tokenizer enables simple methods to outperform complex systems.

## Limitations & Future Work
- Image generation quality, while surpassing prior unified models, still falls short of dedicated diffusion models such as SD-XL.
- Resolution is limited (256 or 384); higher resolutions may require larger codebooks or multi-scale schemes.
- The semantic encoder (SigLIP) in SDE is frozen; joint training could potentially yield further improvements at the cost of increased complexity.
- The image reconstruction relies solely on LlamaGEN's convolutional decoder; a stronger decoder may improve generation quality.

## Related Work & Insights
- **vs. Emu3**: Emu3 requires training an 8B LLM from scratch; MUSE-VL directly employs a pretrained LLM with 24M data and still surpasses Emu3 (+4.8%). The key differentiator is the SDE tokenizer, which ensures natural alignment between visual and text tokens.
- **vs. Chameleon**: Chameleon uses a standard VQGAN and achieves only 33.3% average understanding; MUSE-VL's 63.6% is nearly twice as high.
- **vs. VILA-U**: VILA-U aligns semantics via dual losses (contrastive + reconstruction), requiring 720M data and suffering from convergence difficulties. SDE uses feature fusion and semantic reconstruction, requiring only 24M data.
- **vs. Janus**: Janus employs dual encoders to decouple understanding and generation, increasing model complexity. MUSE-VL achieves both with a single SDE encoder.

## Rating
- Novelty: ⭐⭐⭐⭐ — SDE tokenizer design is novel and directly addresses the core semantic deficiency of VQ-based tokenizers.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 7 understanding benchmarks, 2 generation benchmarks, tokenizer comparisons, ablations, and scaling experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear presentation with precise problem formulation.
- Value: ⭐⭐⭐⭐⭐ — Provides a simple and efficient pathway for unified VLMs; the SDE tokenizer is reusable as a general-purpose component.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unified Vision-Language Modeling via Concept Space Alignment](../../ICLR2026/multimodal_vlm/unified_vision-language_modeling_via_concept_space_alignment.md)
- [\[ICCV 2025\] MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling](matvlm_hybrid_mamba-transformer_for_efficient_vision-language_modeling.md)
- [\[ICCV 2025\] Background Invariance Testing According to Semantic Proximity](background_invariance_testing_according_to_semantic_proximity.md)
- [\[CVPR 2026\] Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens](../../CVPR2026/multimodal_vlm/cubic_discrete_diffusion_discrete_visual_generation_on_high-dimensional_represen.md)
- [\[ICCV 2025\] Harmonizing Visual Representations for Unified Multimodal Understanding and Generation](harmonizing_visual_representations_for_unified_multimodal_un.md)

</div>

<!-- RELATED:END -->
