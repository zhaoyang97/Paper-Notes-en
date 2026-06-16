---
title: >-
  [Paper Note] TUNA: Taming Unified Visual Representations for Native Unified Multimodal Models
description: >-
  [CVPR 2026][Multimodal VLM][Flow Matching] TUNA **cascades** a VAE encoder and a semantic representation encoder to obtain a set of continuous unified visual representations compatible with both "understanding" and "generation." Combined with an autoregressive text head and a flow-matching generation head, a single native model at 1.5B/7B scale achieves SOTA re
tags:
  - CVPR 2026
  - Multimodal VLM
  - Flow Matching
date: 2026-05-08
content_hash: 61ba81e3777842d4
---
# TUNA: Taming Unified Visual Representations for Native Unified Multimodal Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_TUNA_Taming_Unified_Visual_Representations_for_Native_Unified_Multimodal_Models_CVPR_2026_paper.html)  
**Area**: Multimodal VLM  
**Keywords**: Unified Multimodal Models, Unified Visual Representations, Flow Matching, VAE Latent Space, Image-Text Generation  

## TL;DR
TUNA **cascades** a VAE encoder and a semantic representation encoder to obtain a set of continuous unified visual representations compatible with both "understanding" and "generation." Combined with an autoregressive text head and a flow-matching generation head, a single native model at 1.5B/7B scale achieves SOTA results in image/video understanding, image/video generation, and image editing (MMStar 61.2, GenEval 0.90).

## Background & Motivation

**Background**: Unified Multimodal Models (UMM) aim to perform both multimodal understanding and generation within a single model. Current mainstream approaches are divided into two categories: one uses **decoupled representations** (understanding uses semantic encoders like SigLIP, while generation uses VAE latent spaces), represented by BAGEL and Mogao; the other pursues **unified representations** where all tasks share a single visual encoding, represented by Chameleon, Transfusion, and Harmon.

**Limitations of Prior Work**: The decoupled approach requires stuffing two completely incompatible visual encoders into one model. For the same image, SigLIP features and Wan VAE latents mismatch in spatial compression (16× vs 8×), temporal compression (None vs 4×), and channel dimensions (1152 vs 16), forcing the use of MoE-style architectures which increase parameters and training/inference costs while introducing representation conflicts. The unified approach should be more efficient and elegant, **but its actual performance often lags behind decoupled approaches**: a single encoder (VQ-VAE or MAR) is naturally biased—favoring understanding leads to weak generation, and vice versa. Show-o2 attempts to mitigate this via "dual-path late-fusion" of SigLIP and VAE features, but the analysis in §3.4 reveals that the fused representation is **heavily biased toward semantic features**, limiting generation quality.

**Key Challenge**: Understanding tasks require high-level semantic features, while generation tasks require high-fidelity reconstructible latent spaces. A single encoder cannot satisfy both, and forced concatenation of two encoders introduces format conflicts. The fundamental problem is the **lack of a unified visual representation that is both "semantic" and "reconstructible."**

**Key Insight**: The authors observe several overlooked facts: ① Continuous representations (KL-VAE latent space) are superior to discrete ones for generation, and understanding models also prefer continuous semantic features; thus, unified representations should be built on **continuous VAE latent spaces**. ② Semantic features can assist generation (as proven by REPA and RAE). ③ VAE latents can already support semantic understanding (as seen in UniTok, TokLIP, etc.). Since the VAE latent space is viable for both directions, it should not be discarded; instead, **another semantic encoder layer should be stacked on top of it** to extract high-level features.

**Core Idea**: Directly cascade a representation encoder (SigLIP2) after the VAE encoder. The VAE preserves the reconstructible continuous latent space, while the semantic encoder extracts high-level semantics from it. This yields a unified visual representation "expressive enough" for both understanding and generation. Understanding follows autoregressive text generation, and generation follows flow matching, with end-to-end joint training throughout.

## Method

### Overall Architecture

TUNA is a **native** UMM (jointly pre-trained on both understanding and generation from scratch, rather than stitching pre-trained models with connectors). The data flow for a single forward pass is: Input image/video → 3D Causal VAE encoder compressed into continuous latents → Noised latents via flow-matching → Modified SigLIP2 encoder for semantic extraction → MLP connector projected into unified visual representation $z$ → Concatenated with text tokens and fed into the LLM decoder → Two task-specific exits: Understanding uses a language modeling head for next-token prediction; Generation uses a flow-matching head to predict the velocity field for denoising.

The switch between understanding and generation is solely the **noise timestep $t$**: for understanding, $t=1$ is fixed to degrade noised latents to clean ones; for generation, $t$ is randomly sampled from $[0,1]$ to train the model to recover images from noise. The same representation and decoder backbone are used, toggling between tasks via $t$ and attention masks.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input Image / Video X"] --> B["3D Causal VAE Encoder<br/>Space 16× Time 4× Compression"]
    B --> C["Cascaded Unified Visual Representation<br/>VAE Latent Noising → SigLIP2 → MLP"]
    C -->|Video| D["Video Window Attention<br/>4-frame Independent Window Encoding"]
    C --> E["Concatenate Text Tokens + Timestep Token"]
    D --> E
    E --> F["LLM Decoder with Dual Output Heads<br/>Text Causal Mask / Visual Bidirectional Mask"]
    F -->|Understanding t=1| G["Language Modeling Head<br/>Autoregressive Text"]
    F -->|Generation t∈[0,1]| H["Flow Matching Head<br/>Predict Velocity Field for Denoising"]
```

### Key Designs

**1. Cascaded Unified Visual Representation: Stacking a semantic encoder directly after the VAE encoder for a dual-task representation**

This is the core of TUNA, addressing the "single encoder bias and dual encoder format conflict" pain point. The approach is counter-intuitively simple: given input $X$, it is first compressed into continuous latent $x_1$ by the 3D Causal VAE of Wan 2.2 (16× spatial, 4× temporal downsampling); then, semantic features are extracted by a SigLIP2 visual encoder $\Phi$ on this latent; finally, a two-layer MLP connector outputs the unified representation $z=\mathrm{MLP}(\Phi'(x_t))$. A key engineering change involves replacing SigLIP2's original 16×16 patch embedding with a **randomly initialized 1×1 patch embedding** layer (denoted as $\Phi'$) because the VAE has already compressed the image by 16×, and a further 16×16 patch sequence would be too short.

Why is this effective? Contrasting with Show-o2's late-fusion (where VAE and semantic encoders run independently and fuse at the final layer), CKNNA alignment analysis in §3.4 shows that late-fusion causes the final representation to inherit almost exclusively from the semantic branch. TUNA allows the semantic encoder to **perform deep fusion layer-by-layer on the VAE latents**, resulting in a representation balanced between SigLIP2 (semantic reference, CKNNA > 0.5) and SD3-Medium (generation reference).

**2. Shared Latent Noising + Timestep Condition: Integrating generation into the same representation via Flow Matching**

Addressing the tension between "clean features for understanding" and "noisy features for generation," TUNA does not create a separate input for generation. Instead, it interpolates noise into the same VAE latent following flow-matching rules:

$$x_t = t\,x_1 + (1-t)\,x_0,\quad t\in[0,1],\ x_0\sim\mathcal{N}(0,1)$$

During training, $t$ is sampled randomly for generation and fixed at $t=1$ for understanding (where $x_t=x_1$, the clean latent). Since the semantic encoder $\Phi'$ consumes $x_t$, both tasks share the same representation path, differing only in noise level. For generation, a timestep token representing $t$ is prepended to $z$ to inform the decoder of the noise intensity. This design allows gradients from generation to backpropagate to the representation encoder, ensuring a truly unified representation shaped by both objectives.

**3. Video Window Attention: Folding frame dimensions into batch dimensions to avoid sequence explosion**

To address efficiency issues with long video sequences, the video latent $x_t\in\mathbb{R}^{b\times c\times f\times h\times w}$ is rearranged by folding the frame dimension $f$ into the batch dimension:

$$\bar{x}_t=\mathrm{rearrange}(x_t,\ b\,c\,f\,h\,w\to (b f)\,c\,h\,w)$$

$$\bar{z}_v=\mathrm{MLP}(\Phi'(\bar{x}_t)),\quad z_v=\mathrm{rearrange}(\bar{z}_v,\ (b f)\,d\to b\,(f d))$$

This enables the semantic encoder to **perform attention independently on each 4-frame window** (the temporal compression unit of the VAE). While windows do not interact during encoding, efficiency improves significantly, allowing the 1.5B decoder to handle video generation.

**4. Dual-Head LLM Decoder: Causal mask for text and bidirectional mask for vision**

To allow a single decoder to handle both autoregressive and diffusion-like tasks, unified representation $z$ and text tokens are concatenated and fed into a Qwen2.5 decoder. The attention mask shifts based on modality: text tokens use a causal mask, while visual tokens use a bidirectional mask. The output splits into two paths: understanding passes through a language modeling head to predict text tokens; generation/editing feeds the complete token sequence into a randomly initialized flow-matching head to predict the denoising velocity field. This head reuses the decoder architecture and injects timestep conditions via AdaLN-Zero.

### Loss & Training

A three-stage progressive training strategy adapts components to dual tasks:

- **Stage 1 (Unified Representation + Flow Matching Pre-training)**: Freezes the LLM decoder and trains only the representation encoder and flow-matching head using image captioning and text-to-image (T2I) objectives. Captioning aligns the encoder with strong semantic targets, while T2I initializes the flow-matching head.
- **Stage 2 (Full Model Continued Pre-training)**: Unfreezes the LLM decoder for end-to-end training. Image instruction following, image editing, and video captioning data are added to expand capabilities.
- **Stage 3 (Supervised Fine-Tuning - SFT)**: Refines the model with image editing, image/video instruction following, and high-quality generation data at a lower learning rate ($2\times10^{-5}$).

Data scale: 177M image-text pairs + 13M FineVision dialogue samples + 2M OmniEdit samples + 10M high-quality SFT pairs for images; 10M video-description pairs + 1.6M LLaVA-Video instruction samples for videos. The 7B variant did not use video data due to cost.

## Key Experimental Results

### Main Results

On the understanding side, TUNA achieves SOTA across 9 benchmarks at both 1.5B and 7B scales, outperforming larger composite UMMs:

| Scale | Model | MMStar | RealWorldQA | ChartQA | OCRBench |
|------|------|--------|-------------|---------|----------|
| 1.5B | Show-o2 | 43.4 | 56.5 | 40.0 | 24.5 |
| 1.5B | **TUNA** | **54.6** | **62.5** | **82.1** | **71.9** |
| 7B | Show-o2 | 56.6 | 64.7 | 52.3 | 32.4 |
| 7B | **TUNA** | **61.2** | **66.1** | **85.8** | **74.3** |

On the generation side, it leads concurrent methods (including decoupled models like BAGEL and Mogao):

| Benchmark | Task | TUNA-1.5B | TUNA-7B | Comparison SOTA |
|-----------|------|-----------|---------|-----------|
| GenEval | Image Gen | 0.88 | **0.90** | Mogao-7B 0.89 |
| DPG-Bench | Image Gen | 86.03 | **86.76** | Show-o2-7B 86.14 |
| VBench | Video Gen (1.5B Dec) | **84.06** | — | Show-o2-1.5B 81.34 |
| ImgEdit | Image Editing | — | **4.31** | BAGEL-14B 3.20 |

### Ablation Study

Core ablation (Table 6, 1.5B lightweight version, 2-stage training) comparing visual representation designs:

| ID | Rep Design | Data | MMMU | SEED | GenEval | DPG |
|----|---------|---------|------|------|---------|-----|
| 6 | Decoupled (SigLIP2 + VAE) | Und & Gen | 37.2 | 61.4 | 78.3 | 83.50 |
| 7 | TUNA (SigLIP) | Und & Gen | 36.3 | 64.6 | 76.9 | 83.10 |
| 8 | TUNA (SigLIP2) | Und & Gen | 38.1 | 66.5 | **79.4** | **84.20** |
| 9 | TUNA (DINOv3) | Und & Gen | 37.3 | 65.6 | 78.9 | 84.08 |
| 2 | TUNA (SigLIP2) | Und Only | 37.6 | 62.9 | — | — |
| 4 | TUNA (SigLIP2) | Gen Only | — | — | 77.8 | 83.33 |

### Key Findings

- **Unified > Decoupled**: Model 8 (Unified TUNA) outperforms Model 6 (Decoupled) across all metrics, proving that multiple visual representations in one model cause conflict.
- **Stronger Encoders Help**: SigLIP2 and DINOv3 outperform SigLIP. SigLIP2 was chosen for its balance of performance and efficiency.
- **Understanding-Generation Synergy**: Model 8 (dual-task) exceeds Model 2 (understanding only) and Model 4 (generation only), showing that joint training on a unified representation leads to mutual task enhancement.
- **Why it Beats Show-o2**: CKNNA analysis shows TUNA's representation aligns more balancedly with both semantic and generation references, whereas Show-o2 is heavily biased toward the semantic branch.

## Highlights & Insights
- **The cascade of VAE + Semantic Encoder is clever**: It uses existing modules but cascades them with a 1×1 patch embedding to solve format conflicts—a "simple yet effective" solution.
- **Timestep $t$ as a switch**: Using $t=1$ for understanding and $t \in [0,1]$ for generation seamlessly embeds "diffusion" into an "autoregressive" framework without architectural redundancy.
- **CKNNA Analysis**: Provides a quantitative way to prove why deep fusion beats late-fusion, offering a methodological tool for representation alignment.
- **Video Window Attention**: By folding dimensions, it allows small decoders to handle video tasks, which is highly practical for resource-constrained scenarios.

## Limitations & Future Work
- **Inconsistent Video Scaling**: The 7B variant lacks video training; hence unified representation scalability for massive video data remains untested.
- **Inter-window Dependencies**: Video window attention might weaken long-term temporal dependencies since windows are encoded independently.
- **Data Dependency**: The reliance on vast in-house data (177M images) makes reproduction difficult and obscures the exact architectural contribution versus data scaling.
- **Cold-start Flow Matching Head**: The generation head is trained from scratch; exploring pre-trained diffusion weights for initialization could improve efficiency.

## Related Work & Insights
- **vs Show-o2**: Show-o2 uses late-fusion to merge VAE and semantic features at the end, while TUNA uses deep fusion layer-by-layer. TUNA's balanced representation leads to superior performance across all benchmarks.
- **vs BAGEL / Mogao**: These use MoE-style architectures with higher parameter counts and potential conflicts; TUNA proves that a single representation space is more efficient and effective.
- **vs REPA / RAE**: While REPA/RAE show that semantic alignment helps generation, TUNA generalizes this "semantic-assisting-generation" insight to a full unified multimodal framework.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DuetSVG: Unified Multimodal SVG Generation with Internal Visual Guidance](duetsvg_unified_multimodal_svg_generation_with_internal_visual_guidance.md)
- [\[CVPR 2026\] GGBench: A Geometric Generative Reasoning Benchmark for Unified Multimodal Models](ggbench_a_geometric_generative_reasoning_benchmark_for_unified_multimodal_models.md)
- [\[CVPR 2026\] Customized Visual Storytelling with Unified Multimodal LLMs](customized_visual_storytelling_with_unified_multimodal_llms.md)
- [\[ICCV 2025\] Harmonizing Visual Representations for Unified Multimodal Understanding and Generation](../../ICCV2025/multimodal_vlm/harmonizing_visual_representations_for_unified_multimodal_un.md)
- [\[CVPR 2026\] Unified Personalized Understanding, Generating and Editing](unified_personalized_understanding_generating_and_editing.md)

</div>

<!-- RELATED:END -->
