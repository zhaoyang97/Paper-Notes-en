---
title: >-
  [Paper Note] OmniZip: Learning a Unified and Lightweight Lossless Compressor for Multi-Modal Data
description: >-
  [CVPR 2026][Multimodal Efficiency][Lossless compression] OmniZip utilizes a lightweight RWKV backbone (ranging from several MB to 152M parameters) combined with "unified modality tokenization + modality-routed MoE." It achieves lossless compression for seven modalities (image, text, speech, tactile, gene, and database) within a single model. It improves compression rates by 42%–62% over gzip and achieves near real-time speeds of approximately 1MB/s on MacBook CPUs and iPhone…
tags:
  - "CVPR 2026"
  - "Multimodal Efficiency"
  - "Lossless compression"
  - "Multimodal"
  - "RWKV"
  - "Mixture of Experts (MoE)"
  - "Lightweight"
date: 2026-05-08
content_hash: d35131440a0d9691
---

# OmniZip: Learning a Unified and Lightweight Lossless Compressor for Multi-Modal Data

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_OmniZip_Learning_a_Unified_and_Lightweight_Lossless_Compressor_for_Multi-Modal_CVPR_2026_paper.html)  
**Code**: https://github.com/adminasmi/OmniZip-CVPR2026  
**Area**: Multimodal VLM / Lossless Compression  
**Keywords**: Lossless compression, Multimodal, RWKV, Mixture of Experts (MoE), Lightweight

## TL;DR
OmniZip utilizes a lightweight RWKV backbone (ranging from several MB to 152M parameters) combined with "unified modality tokenization + modality-routed MoE." It achieves lossless compression for seven modalities (image, text, speech, tactile, gene, and database) within a single model. It improves compression rates by 42%–62% over gzip and achieves near real-time speeds of approximately 1MB/s on MacBook CPUs and iPhone NPUs.

## Background & Motivation
**Background**: Modern lossless compression typically follows the pipeline of "likelihood estimation via a probability model + entropy coding (arithmetic coding) to approach the Shannon lower bound." The optimal code length for a symbol is its negative log-probability. Recent approaches using neural networks or Large Language Models (LLMs) to estimate $p(x_i\mid x_{<i})$ have significantly outperformed classical compressors like gzip/bzip2.

**Limitations of Prior Work**: Two specific issues are highlighted. First, LLM-based methods (e.g., using LLaMA3-8B for compression) involve billions of parameters, often exceeding the size of the data being compressed. Since compression speed is governed by model inference, compressing a 1080p image can take over 30 minutes, making practical deployment impossible. Second, most learned compressors are designed for a single modality, necessitating the deployment of multiple compressors in multimodal systems, which increases software complexity and hardware costs.

**Key Challenge**: The difficulty of multimodal lossless compression stems from modality heterogeneity. Text consists of discrete sequences, images have 2D spatial structures, speech features continuous smooth spectra, databases contain structured categorical fields, and genes are symbolic sequences with specific motifs. Prior attempts at unified compression (e.g., converting everything to ASCII text for pre-trained LLMs) perform poorly on non-textual modalities.

**Goal**: Develop a lossless compressor that is both "unified" (a single model for multiple modalities) and "lightweight" (capable of near real-time execution on edge devices).

**Key Insight**: Instead of scaling up model size, an efficient autoregressive backbone (RWKV was selected over Transformer and Mamba) is used. Modality heterogeneity is absorbed by incorporating "modality-routed" sparse experts within the architecture, where each token activates only a small subset of experts to maintain low parameter overhead and high speed.

**Core Idea**: Reversibly map different modalities into a single token space and replace the standard V-projections and MLPs in a lightweight RWKV backbone with modality-routed MoE modules. This allows a small model to adaptively model context and non-linear transformations based on the modality.

## Method

### Overall Architecture
OmniZip decomposes compression into a standard three-step pipeline: (1) Any modality input is transformed via **unified modality tokenization** into a unified, fully reversible token sequence $\{x_1,\dots,x_n\}$; (2) A lightweight RWKV-7 prediction model estimates the context probability $p(x_i\mid x_{<i})$ token-by-token; (3) **Arithmetic coding** compresses the data into a bitstream approaching the entropy lower bound $H(p)=\mathbb{E}\big[\sum_i -\log_2 p(x_i\mid x_{<i})\big]$. Decompression reverses this process for bit-accurate restoration.

The innovation lies in the prediction model, which stacks $N$ RWKV blocks. Each block’s Time Mixing and MLP modules are replaced with **modality-routed context learning** (MoE in V-projection) and **modality-routed feed-forward** (MoE in MLP). This allows sparse expert selection based on the token's modality. Reparameterization branches are used during training to increase capacity and are merged back during inference.

```mermaid
mermaid
flowchart TD
    A["Multi-modal Input<br/>Image/Text/Speech/Tactile/Gene/DB"] --> B["Unified Modality Tokenization<br/>Reversible Mapping + Modality Prefix"]
    B --> C["Modality-Routed Context Learning<br/>V-layer in Time Mixing as MoE"]
    C --> D["Modality-Routed Feed-Forward<br/>MLP as MoE"]
    D -->|"Stack N RWKV Blocks"| E["Per-token Probability Prediction<br/>p(xi | x<i) + Modality Mask"]
    E --> F["Arithmetic Coding → Bitstream"]
```

### Key Designs

**1. Unified Modality Tokenization: Reversibly mapping seven heterogeneous data types into a single token space**

The challenge is the vast difference in formats and the requirement for reversibility. OmniZip categorizes data into three types: **Textual** (Natural language/Gene/Database) uses SentencePiece BPE with a 16K vocabulary, explicitly adding domain symbols (e.g., A/T/G/C for genes, `select`/`and` for databases) to improve efficiency. **Images** (Natural/Medical/Tactile) are split into $16\times16\times3$ patches. Within each patch, pixel RGB sub-pixels $(R_1,G_1,B_1,R_2,\dots)$ are treated as individual tokens in raster order (vocabulary size 256). For grayscale medical images, each intensity value is a token, while tactile force data is mapped to RGB images before processing. **Speech** is read as a raw byte stream where each byte is a token (vocabulary size 256). After merging vocabularies, a modality prefix like `<image>` or `<text>` is prepended. A **modality mask** is applied before softmax to zero out probabilities of non-target tokens, reducing estimation error.

**2. Modality-Routed Context Learning: MoE applied only to the V-layer of RWKV**

Contextual structures vary across modalities. The Mixture of Experts (MoE) is applied strictly to the **V-projection layer** in Time Mixing, while K and R layers are shared. This is motivated by the RWKV mechanism where the V-layer carries the actual memory content, which is most sensitive to modality diversity. A learnable router assigns a score $g_{i,e}$ for token $x_i$ to expert $e$:

$$g_{i,e}=\mathrm{softmax}(x_i W_g)_e=\frac{\exp(x_i W_{g,e})}{\sum_{e'=1}^{E}\exp(x_i W_{g,e'})},$$

The top-$k$ experts are used for weighted output: $V(x_i)=\sum_{e\in\text{top-}k}\hat{g}_{i,e}\cdot e(x_i)$. Implementation uses 4 experts with top-$k{=}2$, adding negligible overhead while providing modality adaptation.

**3. Modality-Routed Feed-forward: Replacing universal MLPs with MoE**

Unified MLPs fail to capture modality-specific non-linearities. OmniZip replaces them with MoE feed-forward layers. Each MLP expert uses a $2\times$ hidden factor (half of the original), with 4 experts and top-$k{=}2$. This maintains active parameter counts per token while increasing flexibility.

### Loss & Training
**Reparameterization Training**: Following L3TC, additional high-rank branches are added to Time Mixing layers during training to increase capacity and merged during inference to maintain low complexity. **Three-stage training** stabilizes MoE optimization: (1) Freeze feed-forward routers for 2 epochs; (2) Freeze context routers for 2 epochs; (3) Full fine-tuning for 20 epochs with cosine annealing ($1\times10^{-4}\!\to\!1\times10^{-5}$). The **Loss** includes cross-entropy, a router Z-loss to prevent instability, and a load balancing loss (squared coefficient of variation $\mathrm{CV}^2$) with weights $\lambda{=}0.001$ and $\mu{=}0.01$.

## Key Experimental Results
The primary metric is **bits/Byte** (lower is better; percentages in parentheses indicate **Gain** over gzip). Efficiency is measured via **MACs** and throughput (**KB/s**). Testing across 16 datasets and 7 modalities using OmniZip-S/M/L (4.8M / 38M / 152M parameters).

### Main Results
Image-based modalities (Kodak, TouchandGo, Coronal):

| Compressor | Params | Kodak | TouchandGo | Coronal | Multi-modal? |
|------------|--------|-------|------------|---------|--------------|
| gzip | – | 4.349 | 2.298 | 4.563 | ✓ |
| JPEG-XL | – | 2.902 | 0.739 | 3.891 | ✗ |
| P2LLM | 8B | 2.830 | – | – | ✗ |
| Llama3 (LLM) | 8B | 4.862 | 2.455 | 4.832 | ✓ |
| **Ours-S** | 4.8M | 3.307 (-24%) | 1.338 (-42%) | 4.179 (-8%) | ✓ |
| **Ours-L** | 152M | 2.925 (-33%) | 0.987 (-57%) | 3.837 (-16%) | ✓ |

Text/Speech modalities (enwik9, WikiSQL, LibriSpeech):

| Compressor | Params | enwik9 | WikiSQL | LibriSpeech | Multi-modal? |
|------------|--------|--------|---------|-------------|--------------|
| gzip | – | 2.590 | 1.672 | 6.511 | ✓ |
| FLAC | – | – | – | 4.961 | ✗ |
| Llama3 (LLM) | 8B | 0.722 | 0.645 | 3.616 | ✓ |
| **Ours-S** | 4.8M | 1.370 (-47%) | 1.170 (-30%) | 4.155 (-36%) | ✓ |
| **Ours-L** | 152M | 0.980 (-62%) | 0.787 (-53%) | 3.810 (-42%) | ✓ |

**Key Finding**: OmniZip matches or exceeds specialized learned compressors using orders of magnitude fewer parameters. Compared to 8B LLM-based multimodal compressors, it provides better compression rates and significantly higher throughput (Ours-S at 1223KB/s vs Llama3 at 20KB/s on A100).

### Ablation Study
Evaluation on OmniZip-S (bits/Byte, MacBook CPU, batch 128):

| Configuration | enwik9 | Kodak | TouchandGo | Speed (KB/s) |
|---------------|--------|-------|------------|--------------|
| Backbone only | 1.658 | – | – | 856 |
| + Reparam | 1.590 | – | – | 856 |
| + Unified Tokenization | 1.660 | 3.383 | 1.453 | 856 |
| + FF-MoE | 1.424 | 3.352 | 1.431 | 633 |
| + Context-MoE | 1.419 | 3.339 | – | 780 |

### Key Findings
- Both MoE modules contribute to multimodal gains; FF-MoE is particularly effective for text.
- Reparameterization provides "free" capacity during training without inference cost.
- RWKV-7 is more efficient than Transformer for compression, especially on CPUs.
- Edge performance (MacBook/iPhone) remains near 1MB/s for the small model.

## Highlights & Insights
- **Efficiency of "Modality-routed MoE"**: Sparse experts allow a small backbone to handle heterogeneous data, making it viable for NPU deployment.
- **Strategic MoE Placement**: Focusing MoE on the V-layer (content) while sharing K/R (indexing/gating) provides the best balance of flexibility and parameter efficiency.
- **Modality Masking**: Simplifies multimodal probability estimation by treating the joint vocabulary as a dynamic, modality-specific one before coding.
- **Model Decoupling**: Reparameterization allows for a high-capacity training model that collapses into a fast, compact inference model.

## Limitations & Future Work
- Pure text compression still lags behind dedicated compressors like NNCP/CMIX.
- Evaluation lacks detailed energy and memory profiling for end-to-end edge deployment.
- Optimal expert counts and top-$k$ values for more diverse distributions are not yet fully explored.
- Speech compression (via byte-tokens) does not fully exploit spectral structures.

## Related Work & Insights
- **vs. Classical (gzip/zstd)**: OmniZip significantly improves compression rates (24%–62% gain) while maintaining MB/s throughput on GPUs.
- **vs. LLM-based Compression**: Avoids the slow ASCII-conversion bottleneck and high parameter counts of Llama3/RWKV-7B routes.
- **vs. Specialized Compressors**: Replaces the need for per-modality models by achieving competitive performance in a unified framework.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data and Training Recipes](minicpm-v_45_cooking_efficient_mllms_via_architecture_data_and_training_recipe.md)
- [\[CVPR 2026\] MM-SeR: Multimodal Self-Refinement for Lightweight Image Captioning](mm-ser_multimodal_self-refinement_for_lightweight_image_captioning.md)
- [\[CVPR 2026\] Adapting Lightweight Image-based Counting Models for Video Crowd Counting](adapting_lightweight_image-based_counting_models_for_video_crowd_counting.md)
- [\[ICCV 2025\] FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance](../../ICCV2025/vlm_efficiency/folder_accelerating_multi-modal_large_language_models_with_enhanced_performance.md)
- [\[CVPR 2026\] OmniZip: Audio-Guided Dynamic Token Compression for Fast Omnimodal Large Language Models](omnizip_audio-guided_dynamic_token_compression_for_fast_omnimodal_large_language.md)

</div>

<!-- RELATED:END -->
