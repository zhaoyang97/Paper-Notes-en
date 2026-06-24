---
title: >-
  [Paper Note] JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation
description: >-
  [CVPR 2025][Image Generation][unified model] This paper proposes JanusFlow, which directly integrates rectified flow into the autoregressive LLM framework. By decoupling understanding/generation encoders and utilizing representation alignment regularization, it achieves state-of-the-art performance in both multimodal understanding and image generation with only 1.3B parameters.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "unified model"
  - "rectified flow"
  - "autoregressive LLM"
  - "multimodal understanding"
  - "decoupled encoder"
  - "REPA"
date: 2026-05-08
content_hash: 85a2019b4b6c142f
---

# JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation

**Conference**: CVPR 2025  
**arXiv**: [2411.07975](https://arxiv.org/abs/2411.07975)  
**Code**: [deepseek-ai/Janus](https://github.com/deepseek-ai/Janus)  
**Area**: Image Generation  
**Keywords**: unified model, rectified flow, autoregressive LLM, multimodal understanding, decoupled encoder, REPA

## TL;DR

This paper proposes JanusFlow, which directly integrates rectified flow into the autoregressive LLM framework. By decoupling understanding/generation encoders and utilizing representation alignment regularization, it achieves state-of-the-art performance in both multimodal understanding and image generation with only 1.3B parameters.

## Background & Motivation

**Background**: Large language models have made significant progress in both multimodal understanding (such as LLaVA and Qwen-VL) and text-to-image generation (such as SD and DALL-E). However, these two capabilities are typically realized using separate, independent models.

**Limitations of Prior Work**: Existing unified approaches suffer from two main limitations: (1) using pretrained diffusion models as external tools (e.g., SEED-X), which results in complex architectures and constrains generation capabilities to conditional generation; (2) relying on VQ discrete tokens for autoregressive generation (e.g., Chameleon), where quality is limited by the precision of the tokenizer.

**Key Challenge**: The understanding task requires high-level semantic features, whereas the generation task demands low-level detailed features, leading to task interference at the encoder level. Furthermore, there is a lack of a simple and effective scheme for integrating autoregression and continuous diffusion paradigms.

**Goal**: To integrate rectified flow (a continuous generation paradigm) with an autoregressive LLM in a minimalist architecture, enabling high-quality multimodal understanding and image generation simultaneously within a single model.

**Key Insight**: A minimalist design—requiring only lightweight encoders/decoders to train rectified flow inside the LLM, coupled with decoupled encoders to eliminate task interference, and representation alignment to enhance the semantic coherence of generation.

## Method

### Overall Architecture

JanusFlow is based on the 1.3B DeepSeek-LLM. The core idea is to run two modes within the identical LLM:
- **Understanding Mode**: Image $\rightarrow$ SigLIP encoding $\rightarrow$ linear projection $\rightarrow$ concatenation with text tokens $\rightarrow$ autoregressive prediction of text tokens.
- **Generation Mode**: Text conditions $\rightarrow$ LLM processing $\rightarrow$ starting from Gaussian noise, the LLM iteratively predicts velocity vectors $\rightarrow$ Euler solver updates the latent state $\rightarrow$ VAE decodes the final image.

### Key Designs

**1. Decoupled Understanding and Generation Encoders**
- **Function**: Multimodal understanding utilizes a pretrained SigLIP-Large-Patch/16 (~300M) to extract semantic features. Image generation utilizes ConvNeXt blocks (~70M) trained from scratch as the encoder $g_{enc}$ and decoder $g_{dec}$.
- **Mechanism**: Semantic encoders excel at high-level abstraction but are not suited for pixel-level reconstruction, whereas generation encoders need to operate in the VAE latent space. Decoupling them allows each to specialize in its respective task.
- **Design Motivation**: Previous works (e.g., Show-o, Transfusion) share the encoder, which leads to task interference between understanding and generation. Ablation studies confirm that decoupling significantly improves performance.

**2. Representation Alignment Regularization (REPA)**
- **Function**: During generation training, the intermediate features of the 6th layer of the LLM are projected and aligned with the real image features from the SigLIP encoder using cosine similarity.
- **Mechanism**: $\mathcal{L}_{REPA} = -\text{sim}(\text{stop\_grad}(f_{enc}(x^{res})), h_\varphi(q_\theta(z_t)))$, where $h_\varphi$ is a 3-layer MLP and gradients are not backpropagated to the understanding encoder.
- **Design Motivation**: This alignment ensures that the internal representation space of the LLM during generation is consistent with the semantic space of the understanding encoder, thereby improving the semantic quality of images generated from random noise and text prompts. This is a unique advantage brought by the decoupled design, as shared-encoder architectures cannot facilitate such cross-module alignment.

**3. Rectified Flow Integration and CFG**
- **Function**: Rectified flow is trained within the SDXL-VAE latent space. The output of the LLM is transformed into velocity vectors via $g_{dec}$ and iteratively updated using an Euler solver.
- **Mechanism**: During training, time $t$ is sampled from a logit-normal distribution, and text prompts are randomly dropped with a 10% probability for CFG inference. During inference, classifier-free guidance is applied via $v = w \cdot v_{cond} + (1-w) \cdot v_{uncond}$.
- **Design Motivation**: It relies solely on standard causal attention without requiring complex attention masking, which greatly simplifies the architecture.

### Loss & Training

Three-stage training:
1. **Stage 1 (Adaptation Phase)**: 10K steps, training only the randomly initialized linear layers, $g_{enc}$, and $g_{dec}$, with $\text{LR}=1\times10^{-4}$.
2. **Stage 2 (Unified Pre-training)**: 390K steps, training all modules except SigLIP, with an understanding:generation:text ratio of 14:80:6.
3. **Stage 3 (SFT)**: 26K steps, unfreezing SigLIP for instruction tuning, with $\text{LR}=2\times10^{-5}$.

Total loss: $\mathcal{L}_{AR}$ (autoregressive cross-entropy) for understanding, and $\mathcal{L}_{RF}$ (flow matching L2) + $\mathcal{L}_{REPA}$ (representation alignment) for generation. EMA is employed with a ratio of 0.99. The total training resource consumption is approximately 1600 A100 GPU days.

## Key Experimental Results

### Main Results — Image Generation

| Method | Type | Parameters | GenEval↑ | DPG-Bench↑ | MJHQ FID↓ |
|---|---|---|---|---|---|
| SDv1.5 | Generation-specific | 0.9B | 0.43 | 63.18 | - |
| SDXL | Generation-specific | 2.6B | 0.55 | 74.65 | - |
| DALL-E 3 | Generation-specific | - | **0.67** | - | - |
| Show-o | Unified | 1.3B | 0.53 | - | 15.18 |
| Janus | Unified | 1.3B | 0.61 | - | 10.10 |
| **JanusFlow** | **Unified** | **1.3B** | **0.63** | **80.09** | **9.51** |

### Main Results — Multimodal Understanding

| Method | Type | LLM | MMBench | SEED | GQA | VQAv2 |
|---|---|---|---|---|---|---|
| LLaVA-v1.5 | Understanding-specific | 7B | 64.3 | 58.6 | 62.0 | 78.5 |
| Qwen-VL-Chat | Understanding-specific | 7B | 60.6 | 58.2 | - | 78.2 |
| **JanusFlow** | **Unified** | **1.3B** | **74.9** | **70.5** | **60.3** | **79.8** |

The 1.3B-parameter unified model outperforms several 7B-parameter understanding-specific models.

### Ablation Study

| Settings | GenEval↑ | DPG-Bench↑ |
|---|---|---|
| Shared Encoder (baseline) | 0.56 | 73.02 |
| + Decoupled Encoder | 0.60 | 76.98 |
| + REPA | **0.63** | **80.09** |
| w/o REPA | 0.60 | 76.98 |

### Key Findings

1. **Rectified flow can be trained directly within the LLM framework**: No modification to the LLM architecture is needed; only lightweight ConvNeXt encoders/decoders and standard causal attention are sufficient.
2. **Decoupled encoders are critical to performance gains**: The GenEval score improves from 0.56 to 0.60, resolving the interference between understanding and generation tasks.
3. **REPA provides further significant improvements**: Adding REPA on top of the decoupled design yields an additional 0.03 GenEval improvement and a 3-percentage-point gain on DPG-Bench.
4. **A 1.3B unified model can outperform multiple specialized 7B models**: This demonstrates the massive potential of the unified autoregression + flow paradigm.

## Highlights & Insights

- **Minimalist architectural design**: Rectified flow is executed within the LLM utilizing only lightweight encoder/decoder components, bypassing complex attention masking.
- **Elegant decoupled-and-aligned strategy**: It physically decouples the encoders for both tasks and then aligns them in the semantic space via regularization.
- **Continuous generation (flow) outperforms discrete generation (VQ)**: With equivalent parameters, FID decreases from 10.10 to 9.51.
- **Substantial boost in generation quality** is realized while simultaneously improving (rather than degrading) the model's understanding capabilities.

## Limitations & Future Work

- **Generation resolution is limited to 384×384**, which is far lower than that of SDXL (1024×1024).
- **The 1.3B model scale is relatively small**; scaling to larger LLMs remains to be validated.
- **Supports only image generation** and has not yet been extended to video generation.
- **CFG inference requires double forward passes**, leaving room for efficiency optimizations.
- **High training resource consumption** (~1600 A100 GPU days); more efficient training strategies need to be explored.

## Related Work & Insights

- **Janus**: The predecessor of JanusFlow, which uses VQ tokens for generation, whereas this work shifts to rectified flow for continuous generation.
- **Show-o / Transfusion**: Unified models utilizing shared encoders; JanusFlow demonstrates that decoupling is superior.
- **REPA (Yu et al.)**: The concept of cross-model representation alignment is integrated into the unified framework, serving as a key innovation.
- **Stable Diffusion 3**: The source of the logit-normal time distribution and flow matching training strategies.

## Rating

⭐⭐⭐⭐ — The powerful trifecta of a minimalist architecture, decoupled encoders, and REPA yields excellent results, with the 1.3B unified model outperforming 7B-parameter specialized models. However, resolution and model scale limitations constrain its practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](../../NeurIPS2025/image_generation/coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2025\] WeGen: A Unified Model for Interactive Multimodal Generation as We Chat](wegen_a_unified_model_for_interactive_multimodal_generation_as_we_chat.md)
- [\[CVPR 2025\] OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)

</div>

<!-- RELATED:END -->
