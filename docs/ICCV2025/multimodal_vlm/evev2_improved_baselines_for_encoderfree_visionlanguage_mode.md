---
title: >-
  [Paper Note] EVEv2: Improved Baselines for Encoder-Free Vision-Language Models
description: >-
  [ICCV 2025 (Highlight)][Multimodal VLM][encoder-free VLM] This work systematically investigates the optimal architecture and training strategy for encoder-free VLMs, proposing a Divide-and-Conquer architecture that fully decomposes a transformer into modality-specific components (independent attention/FFN/LayerNorm per modality). Using only 100M publicly available data, EVEv2 surpasses all encoder-free counterparts and approaches the performance of encoder-based VLMs.
tags:
  - "ICCV 2025 (Highlight)"
  - "Multimodal VLM"
  - "encoder-free VLM"
  - "Divide-and-Conquer"
  - "modality sparsity"
  - "decoder-only"
  - "visual perception learned from scratch"
date: 2026-05-08
content_hash: 5dd7552956a8b3ae
---

# EVEv2: Improved Baselines for Encoder-Free Vision-Language Models

**Conference**: ICCV 2025 (Highlight)  
**arXiv**: [2502.06788](https://arxiv.org/abs/2502.06788)  
**Code**: [https://github.com/baaivision/EVE](https://github.com/baaivision/EVE)  
**Area**: Multimodal VLM / Encoder-Free Architecture  
**Keywords**: encoder-free VLM, Divide-and-Conquer, modality sparsity, decoder-only, visual perception learned from scratch  

## TL;DR
This work systematically investigates the optimal architecture and training strategy for encoder-free VLMs, proposing a Divide-and-Conquer architecture that fully decomposes a transformer into modality-specific components (independent attention/FFN/LayerNorm per modality). Using only 100M publicly available data, EVEv2 surpasses all encoder-free counterparts and approaches the performance of encoder-based VLMs.

## Background & Motivation

### Limitations of Prior Work

**Background**: Mainstream VLMs (e.g., LLaVA/InternVL) rely on pretrained visual encoders (e.g., CLIP-ViT), which introduce resolution/aspect-ratio biases, complex multi-component coordination, and difficulties in independent scaling. Encoder-free VLMs (e.g., Fuyu/EVEv1) let a unified decoder-only model learn visual perception from scratch, yielding a simpler architecture. However, two key challenges remain: (1) learning visual perception from scratch requires substantial data and computation; (2) visual and linguistic representations within the same model can interfere with each other — naive weight sharing or MoE-based decoupling proves insufficient.

### Mechanism

**Goal**: How can encoder-free VLMs efficiently learn visual perception from scratch while minimizing representational interference between visual and linguistic modalities?

## Method

### Overall Architecture
EVEv2.0 is built upon Qwen2.5-7B. A two-layer convolutional patch embedding (stride 16+2) maps image patches into tokens, which are then processed together with text tokens by a fully modality-sparse decoder-only transformer. Training proceeds in four stages: (1) patch embedding pre-alignment → (2) visual layer training with frozen LLM → (3) full-model QA fine-tuning → (4) instruction tuning.

### Key Designs
1. **Divide-and-Conquer (Full Modality Decoupling) Architecture**: Unlike EVEv1 (dense model), EVEv1.2 (re-parameterization), and EVEv1.5 (MoE-decoupled FFN), EVEv2.0 introduces modality-specific grouping across **all components** of every layer: Query/Key/Value matrices, output projections, LayerNorms, and FFNs each maintain independent visual and textual parameters. The total parameter count is 2×7B, while active FLOPs per token remain equivalent to a 7B dense model. A key finding is that LayerNorm suffers the most severe modality interference (largest weight shift from LLM to VLM) and must be fully decoupled.

2. **DenseFusion++ Annotation Engine**: LLaVA-1.6 (7B) is used to fuse multiple vision experts (tagging, detection, OCR, etc.), learning GPT-4V's fusion strategy to generate highly detailed image captions. This outperforms the prior LLaVA-1.5 (13B) + Emu2 (17B) combination and can annotate 700K images per day on a single node with 8×A100 GPUs, enabling EVEv2.0 to achieve strong results with only 100M public data.

3. **Progressive Four-Stage Training**: Stage 1 trains only the patch embedding (alignment initialization) → Stage 2.1 freezes the LLM and trains visual layers with progressively increasing resolution (low → high) → Stage 2.2 fine-tunes all parameters for multimodal alignment → Stage 3 performs instruction tuning. Visual layers are initialized from LLM weights to preserve language capability at the start of training.

### Loss & Training
- Standard cross-entropy autoregressive loss
- Data scale: 44M Datacomp + 15M LAION + 11M SA-1B + 7M OpenImages (pretraining); 15M multi-task (QA); 7.3M instruction tuning
- Training on 16 nodes with 128×A100 GPUs
- Resolution progressively increases from 800×800 to 1600×1600, up to 2500 patch tokens

## Key Experimental Results

| Model | Type | Params | MMMU | MMBench | TextVQA | ChartQA | AI2D | OCRBench |
|-------|------|--------|------|---------|---------|---------|------|----------|
| LLaVA-1.5 | encoder | 7B | 35.3 | 64.3 | 46.1 | 18.2 | 54.8 | 318 |
| LLaVA-1.6 | encoder | 7B | 35.1 | 67.4 | 64.9 | 54.8 | 66.6 | 532 |
| Cambrian | encoder | 7B | 42.7 | 75.9 | 71.7 | 73.3 | 73.0 | 614 |
| Fuyu | enc-free | 8B | 27.9 | 10.7 | - | - | 64.5 | 366 |
| EVEv1 | enc-free | 7B | 32.6 | 52.3 | 56.8 | 59.1 | 61.0 | 398 |
| Mono-InternVL | enc-free | 1.8B | 33.7 | 65.5 | 72.6 | 73.7 | 68.6 | 767 |
| **EVEv2.0** | **enc-free** | **7B** | **39.3** | **66.3** | **71.1** | **73.9** | **74.8** | **702** |

- Surpasses all encoder-free methods (except Mono-InternVL on certain metrics, which uses 13× more data)
- Approaches encoder-based methods such as LLaVA-1.6 and Cambrian
- Achieves 96.2% on ScienceQA, outperforming most encoder-based methods
- Data efficiency: 100M data vs. Mono-InternVL's 1.3B data

### Ablation Study
- DaC > MoE > ReP > Dense: full decoupling outperforms MoE by 1.4% at 24M data scale, with the gap widening as data increases
- DenseFusion++ > LLaVA-1.5+Emu2 annotation > raw web captions
- Multi-source data mixture (Datacomp+LAION+SA1B+OpenImages) significantly outperforms single-source data
- LayerNorm is the most critical module to decouple (decoupling LN alone yields a notable improvement)
- Inference efficiency: EVEv2.0 TTFT is only 13% higher than EVEv1.0, with identical TPS (35 tok/s)

## Highlights & Insights
- **ICCV Highlight and a model of systematic research**: Rather than chasing SOTA, this work systematically answers "what is the optimal path for encoder-free VLMs?"
- **Divide-and-Conquer architecture**: Full modality decoupling represents a paradigmatic breakthrough for encoder-free VLMs — simple yet effective, with the necessity of decoupling LayerNorm rigorously motivated through quantitative weight-shift analysis
- **Efficiency of DenseFusion++**: A 7B annotation model surpasses a 13B+17B combination in caption quality while remaining scalable
- **Strong data efficiency**: 100M public data achieves results comparable to Mono-InternVL, which requires 1.3B data
- **Transparency and reproducibility**: All data are publicly available, code is open-sourced, and training details are thoroughly documented

## Limitations & Future Work
- Scaling to larger models (>7B) and more data is not fully explored due to computational constraints
- Performance on knowledge-intensive tasks (MMMU) still lags behind encoder-based methods
- A gap remains on document understanding tasks (DocVQA, etc.)
- The 2×7B parameter storage requirement exceeds that of a standard 7B model
- Extension to audio and video modalities has not yet been explored

## Related Work & Insights
- **vs. EVEv1**: EVEv1 uses a single dense model with visual supervision; EVEv2 adopts full decoupling and DenseFusion++ without visual supervision, achieving substantially higher performance
- **vs. Mono-InternVL**: Mono-InternVL only decouples FFNs (via MoE), whereas EVEv2 fully decouples all components; however, Mono-InternVL uses 13× more data
- **vs. Scaling Language-Free Visual Representations**: Web-SSL demonstrates that SSL can match CLIP; EVEv2 demonstrates that a fully from-scratch visual encoder can match a pretrained one — the two directions are complementary
- **vs. FALCON**: FALCON compresses high-resolution tokens inside the encoder using registers; EVEv2 fundamentally eliminates the encoder

## Related Work & Insights
- **Idea potential**: The concept of full modality decoupling can be extended to tri-modal native multimodal models (vision/text/audio)
- The Divide-and-Conquer approach and Dynamic-DINO's MoE method could be combined — using modality-aware fine-grained experts within a decoder-only architecture
- The DenseFusion++ annotation engine provides an important reference for data engineering pipelines

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Full modality decoupling represents a paradigmatic advance for encoder-free VLMs; the finding that LayerNorm must be decoupled is particularly insightful
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 13 benchmarks, 4 architectural variants (v1.0/1.2/1.5/2.0), with systematic ablations over data sources, annotation engines, and training strategies
- Writing Quality: ⭐⭐⭐⭐⭐ A benchmark for systematic research writing; the weight-shift quantitative analysis in Figure 2 and the scaling comparison in Figure 5 are highly convincing
- Value: ⭐⭐⭐⭐⭐ The Highlight designation is well-deserved; this work establishes a clear technical roadmap for the encoder-free VLM direction

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion](../../CVPR2025/multimodal_vlm/florence-vl_enhancing_vision-language_models_with_generative_vision_encoder_and_.md)
- [\[ICCV 2025\] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension](scaling_inferencetime_search_with_vision_value_model_for_imp.md)
- [\[CVPR 2026\] Efficient Encoder-Free Fourier-based 3D Large Multimodal Model](../../CVPR2026/multimodal_vlm/efficient_encoder-free_fourier-based_3d_large_multimodal_model.md)
- [\[ICML 2026\] Layer-Specific Fine-Tuning for Improved Negation Handling in Medical Vision-Language Models](../../ICML2026/multimodal_vlm/layer-specific_fine-tuning_for_improved_negation_handling_in_medical_vision-lang.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)

</div>

<!-- RELATED:END -->
