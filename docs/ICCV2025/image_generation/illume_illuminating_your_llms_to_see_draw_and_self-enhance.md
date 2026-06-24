---
title: >-
  [Paper Note] ILLUME: Illuminating Your LLMs to See, Draw, and Self-Enhance
description: >-
  [ICCV 2025][Image Generation][Unified multimodal model] This paper proposes ILLUME, a unified MLLM that integrates multimodal understanding and generation capabilities into a single LLM via a unified next-token prediction paradigm. Through a **semantic visual tokenizer** (reducing pretraining data by 4× to 15M) and a **self-enhancement multimodal alignment scheme** (enabling the model to self-evaluate the consistency between its generated images and text)…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Unified multimodal model"
  - "visual understanding and generation"
  - "semantic visual tokenizer"
  - "self-enhancement alignment"
  - "next-token prediction"
date: 2026-05-08
content_hash: 1a9504b6ac0b919a
---

# ILLUME: Illuminating Your LLMs to See, Draw, and Self-Enhance

**Conference**: ICCV 2025
**arXiv**: [2412.06673](https://arxiv.org/abs/2412.06673)  
**Code**: Not publicly available  
**Area**: Image Generation
**Keywords**: Unified multimodal model, visual understanding and generation, semantic visual tokenizer, self-enhancement alignment, next-token prediction

## TL;DR

This paper proposes ILLUME, a unified MLLM that integrates multimodal understanding and generation capabilities into a single LLM via a unified next-token prediction paradigm. Through a **semantic visual tokenizer** (reducing pretraining data by 4× to 15M) and a **self-enhancement multimodal alignment scheme** (enabling the model to self-evaluate the consistency between its generated images and text), ILLUME achieves competitive or superior performance compared to state-of-the-art unified models across diverse understanding, generation, and editing tasks.

## Background & Motivation

### Core Problem

How to construct an **efficient** unified MLLM that supports visual understanding, image generation, and image editing within a single framework?

### Limitations of Prior Work

**Tool-calling approaches** (e.g., LLaVA + DALL-E): Decoupled architectures limit model potential.

**Regression-based unified models** (e.g., Emu, Emu2): Require joint training of LLMs and diffusion models, leading to high engineering cost and instability.

**VQ tokenizer approaches** (e.g., Chameleon, AnyGPT): Unify next-token prediction but require **massive data** for image-text alignment — Chameleon requires 1.4B image-text pairs, Janus requires 65M.

### Key Insight

Existing VQ tokenizers (e.g., VQGAN) are trained with image reconstruction losses, so their quantized representations focus on low-level textures and lack semantic information, making image-text alignment within LLMs **extremely slow**. Performing quantization in **semantic feature space** can substantially accelerate the alignment process.

### Secondary Problem

Can the understanding and generation capabilities of a unified model **mutually reinforce** each other? The authors find experimentally (Table 2) that naive joint training yields **no clear mutual benefit**, motivating a more refined approach.

## Method

### Overall Architecture

ILLUME is built on Vicuna-7B, extending the visual vocabulary to support discrete visual token generation:

- **Understanding branch**: UNIT visual encoder → visual adapter → LLM text space (retaining continuous features to avoid VQ information loss)
- **Generation branch**: LLM predicts discrete visual tokens → semantic visual tokenizer decodes → Stable Diffusion reconstructs high-resolution images
- **Unified optimization objective**: $\mathcal{L} = -\sum_{i=1} \log P_\theta(y_i | y_{\leq i})$, where $y_i$ denotes text or visual tokens

### Key Design 1: Semantic Visual Tokenizer

Unlike conventional VQGAN (trained with image reconstruction loss):

- Leverages a pretrained UNIT visual encoder to extract **semantic features**
- Supervises the quantization process and codebook learning via **feature reconstruction loss**
- Codebook size: 16,384; each image is represented by 256 discrete tokens
- Uses **Stable Diffusion** to reconstruct images from semantic features (high compression ratio 32×), compensating for low-level details lost during quantization
- Enables high-resolution (512×512) image generation from a fixed number of tokens

### Key Design 2: Three-Stage Progressive Training

| Stage | Objective | Trainable Components | Data Volume | Training Steps |
|-------|-----------|----------------------|-------------|----------------|
| Stage-1: Visual Embedding Initialization | Initialize visual representations | Visual adapter + visual embedding/classification head | 558K (LLaVA-Pretrain) + image reconstruction task | 5,000 |
| Stage-2: Unified Image-Text Alignment | Learn understanding + generation | LLM + visual adapter | **15M** multimodal data | 15,000 |
| Stage-3: Supervised Fine-Tuning | Task-specific capabilities | Full model | Instruction tuning + high-quality image-text pairs + mixed-modal data | 8,000 |

Stage-1 innovation: An **image reconstruction task** is introduced — having the LLM generate the original image — to rapidly initialize the newly added visual embedding weights.

Stage-3 supports high-resolution input via an image patchify strategy (up to 9 slices, base resolution 448), with each slice downsampled to 256 tokens.

### Key Design 3: Self-Enhancement Multimodal Alignment Scheme

Core idea: Train the MLLM to self-evaluate the quality of its own generated images, forming a positive feedback loop between understanding and generation.

**Step 1: Self-generated corpus** — Generate images from a text subset of the training set using the model itself.

**Step 2: Assessment data generation** — Use GPT-4o to evaluate the consistency between self-generated images and their corresponding texts (evaluation dimensions: object accuracy, quantity, color, spatial relationships), producing scores and rationales.

**Step 3: SFT alignment training** — Format assessment data as dialogues:
- High-quality generations → single-turn assessment dialogue
- Low-quality generations → two-turn dialogue (assessment + correction)

A total of **50K** assessment samples are generated and incorporated into Stage-3 training.

**Mutual benefit mechanism**:
- **Generation aids understanding**: Analyzing self-generated negative samples improves understanding of failure modes, enhancing image interpretation accuracy.
- **Understanding aids generation**: Discriminative capabilities are leveraged to evaluate whether generated images align with text, preventing erroneous generation.

## Key Experimental Results

### Main Results: Multimodal Understanding (Table 3)

| Model | Type | POPE | MMBench | SEED | MME-P | MM-Vet | MMMU | AI2D |
|-------|------|------|---------|------|-------|--------|------|------|
| LLaVA-1.5 (7B) | Understanding-only | 85.9 | 64.3 | 58.6 | 1510.7 | 31.1 | 35.4 | 54.8 |
| LLaVA-NeXT (7B) | Understanding-only | 86.5 | 67.4 | 64.7 | - | 43.9 | 35.1 | 66.6 |
| Emu3-Chat (8B) | Understanding-only | 85.2 | 58.5 | 68.2 | - | 37.2 | 31.6 | 70.0 |
| Janus (1.3B) | Unified | 87.0 | 69.4 | 63.7 | 1338.0 | 34.3 | 30.5 | - |
| **ILLUME (7B)** | **Unified** | **88.5** | **75.1** | **72.9** | **1445.3** | 37.0 | **38.2** | **71.4** |

ILLUME achieves first or second place on 10 out of 12 benchmarks. Compared to Janus, MMMU improves by 25% and SEED by 14%.

### Main Results: Image Generation (Table 4)

| Model | Type | MJHQ FID↓ | GenAI Overall | GenEval Overall |
|-------|------|-----------|---------------|-----------------|
| SDXL (2.6B) | Diffusion | 9.55 | 0.55 | 0.55 |
| Janus (1.3B) | Unified | 10.10 | - | 0.61 |
| Show-o (1.5B) | Unified | 15.18 | 0.53 | 0.53 |
| **ILLUME (7B)** | **Unified** | **7.76** | 0.61 | **0.61** |

### Ablation Study: Effect of Self-Enhancement Alignment (Table 7)

| Setting | POPE | MME-P | MMBench | SEED | MM-Vet | MMMU | GenEval Overall |
|---------|------|-------|---------|------|--------|------|----------------|
| Baseline | 86.4 | 1358.6 | 61.7 | 65.0 | 27.4 | 31.2 | 0.56 |
| **+ assessment** | 86.1 | **1446.7** | **63.1** | **66.0** | **29.0** | **32.0** | **0.59** |

Adding only 50K assessment samples improves both understanding and generation. MME-P increases by nearly 90 points; GenEval improves by 0.03.

### Key Findings

1. Semantic tokenizer vs. reconstruction tokenizer: Training loss convergence speed differs substantially at 20M data; the reconstruction tokenizer yields inferior generation quality under the same data volume.
2. Joint training alone provides no significant mutual benefit (Table 2), whereas the self-enhancement scheme effectively promotes synergy between understanding and generation.
3. ILLUME requires only 15M data to achieve competitive performance — 1/4 of Janus and 1/93 of Chameleon.

## Highlights & Insights

1. **Root cause of data efficiency**: Semantic information is the key to accelerating image-text alignment in LLMs — VQ tokenizer design should target LLM compatibility rather than image reconstruction.
2. **Elegance of the self-enhancement scheme**: The model's own imperfect outputs are used as learning signals, requiring no additional human annotation.
3. **Architectural symmetry**: The understanding branch uses continuous features (preserving precision); the generation branch uses discrete tokens (unified paradigm). Both branches share the encoder but exploit it differently.
4. **Inference flexibility**: Classifier-free guidance (CFG) is employed for image generation inference, supporting any-to-any tasks over interleaved image-text data.

## Limitations & Future Work

1. **Base model scale**: Validated only on Vicuna-7B; performance on larger or more recent LLMs remains unexplored.
2. **Generation resolution**: Fixed at 512×512, still lagging behind SDXL (1024×1024).
3. **Diffusion model dependency**: The generation branch still relies on a Stable Diffusion model for image reconstruction, rather than end-to-end pure autoregressive generation.
4. **Training cost**: 32 nodes × 8 NPUs × 3 days remains expensive for academic labs.
5. **Self-enhancement at 50K only**: The effects of larger-scale assessment data or additional evaluation dimensions remain unexplored.

## Related Work & Insights

- **vs. Janus**: Janus uses separate encoders to decouple understanding and generation representations; ILLUME shares the encoder but routes continuous and discrete features separately.
- **vs. Emu3**: Emu3 adopts a pure AR architecture; ILLUME still relies on a diffusion decoder on the generation side but achieves higher data efficiency.
- **vs. Show-o**: Show-o uses the smaller Phi-1.5B backbone; ILLUME employs a larger LLM paired with a more efficient tokenizer.
- **Inspiration**: The self-enhancement scheme can be extended to additional modalities (video, audio, 3D); the semantic tokenizer can serve as a general design principle.

## Rating ⭐⭐⭐⭐

**Novelty**: ⭐⭐⭐⭐ — Dual contributions of semantic tokenizer and self-enhancement alignment scheme
**Practicality**: ⭐⭐⭐⭐ — Substantially reduces data requirements within a unified multi-task framework
**Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers understanding, generation, and editing; ablations are comprehensive
**Writing Quality**: ⭐⭐⭐⭐ — Architecture diagrams are clear; motivation is well-articulated; experiments are well-organized

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning to See in the Extremely Dark](learning_to_see_in_the_extremely_dark.md)
- [\[ICCV 2025\] PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs](panollama_generating_endless_and_coherent_panoramas_with_next-token-prediction_l.md)
- [\[ICCV 2025\] Your Text Encoder Can Be An Object-Level Watermarking Controller](your_text_encoder_can_be_an_object-level_watermarking_controller.md)
- [\[CVPR 2025\] See Further When Clear: Curriculum Consistency Model](../../CVPR2025/image_generation/see_further_when_clear_curriculum_consistency_model.md)
- [\[ICCV 2025\] Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!](attention_to_neural_plagiarism_diffusion_models_can_plagiarize_your_copyrighted_.md)

</div>

<!-- RELATED:END -->
