---
title: >-
  [Paper Note] Harmonizing Visual Representations for Unified Multimodal Understanding and Generation
description: >-
  [ICCV 2025][Multimodal VLM][MAR encoder] This work identifies that the encoder of masked autoregressive (MAR) models inherently possesses both the fine-grained image features required for generation and the high-level se…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "MAR encoder"
  - "unified visual representation"
  - "masked autoregression"
  - "image generation and understanding"
  - "three-stage training"
date: 2026-05-08
content_hash: 917fd05f2e9e8c04
---

# Harmonizing Visual Representations for Unified Multimodal Understanding and Generation

**Conference**: ICCV 2025
**arXiv**: [2503.21979](https://arxiv.org/abs/2503.21979)
**Code**: [GitHub](https://github.com/wusize/Harmon)
**Area**: Multimodal / Unified Generation & Understanding
**Keywords**: MAR encoder, unified visual representation, masked autoregression, image generation and understanding, three-stage training

## TL;DR

This work identifies that the encoder of masked autoregressive (MAR) models inherently possesses both the fine-grained image features required for generation and the high-level semantic representations required for understanding. Based on this observation, Harmon is proposed — an autoregressive framework that unifies image generation and understanding via a shared MAR encoder. Through three-stage progressive training, Harmon achieves an Overall score of 0.76 on GenEval, surpassing all unified models, while matching the understanding performance of the Janus series that employs a dedicated SigLIP encoder.

## Background & Motivation

**Background**: Unifying image generation and understanding has become a key direction for next-generation multimodal intelligence. Existing approaches either loosely couple diffusion models with MLLMs (weak interaction), or unify visual representations through VQ discretization or VAE encoding (tight coupling).

**Limitations of Prior Work**: (1) VQGAN and VAE encoders are pretrained primarily for pixel-level reconstruction and lack high-level semantics, performing significantly worse than CLIP/SigLIP encoders on understanding tasks; (2) Methods such as Janus employ separate encoders for generation and understanding — effective, but at the cost of abandoning the cross-task synergy potential of a unified representation; (3) ViLA-U attempts joint training of contrastive alignment and reconstruction on VQ tokens, but struggles to balance semantic alignment with pixel fidelity.

**Key Challenge**: Understanding requires coarse-grained, high-level semantics, while generation requires fine-grained pixel features — how can a single encoder satisfy both heterogeneous demands simultaneously?

**Goal**: To identify a visual representation that naturally accommodates both generation and understanding, and to construct a truly unified framework with a shared encoder.

**Key Insight**: Masked image modeling (MIM) learns rich semantics through mask-and-reconstruct pretraining; MAR extends MIM to autoregressive generation — its encoder may inherently possess dual capabilities.

**Core Idea**: The "learning to understand through generation" property of the MAR encoder makes it an ideal candidate for a unified encoder — understanding capability emerges as a byproduct of generative pretraining.

## Method

### Overall Architecture

Harmon consists of three components: a MAR encoder $f_{\text{enc}}$, an LLM $f_{\text{LLM}}$ (Qwen2.5), and a MAR decoder $f_{\text{dec}}$. **Generation path**: text prompt → LLM → interaction with MAR encoder outputs → MAR decoder predicts masked patches (masked autoregression, $K=64$ steps). **Understanding path**: all image patches are fed into the MAR encoder → encoder outputs + text embeddings → LLM performs next-token prediction to answer questions. Both paths share the same MAR encoder. A three-stage progressive training scheme is employed to unlock capabilities incrementally.

### Key Designs

1. **Shared MAR Encoder for Dual Tasks**:

    - Function: A single MIM-pretrained MAR encoder serves both image generation and understanding.
    - Mechanism: During generation, the encoder receives visible patches $\mathbf{X}_{\text{seen}}$ and buffer embeddings $\mathbf{X}_{\text{buffer}}$, producing $\mathbf{Z}_{\text{enc}} = f_{\text{enc}}(\mathbf{X}_{\text{seen}}, \mathbf{X}_{\text{buffer}})$, which is passed through the LLM and then to the decoder to predict masked patches. During understanding, the encoder receives all patches and outputs visual representations for LLM-based text generation.
    - Design Motivation: Linear probing experiments show that MAR encoder features achieve substantially higher accuracy on ImageNet than VQGAN/VAE features, and GradCAM++ visualizations demonstrate precise activation responses to visual concepts — generative pretraining has implicitly learned semantic representations.

2. **Three-Stage Progressive Training**:

    - Function: Capabilities are unlocked stage by stage to avoid task conflicts.
    - Mechanism: Stage I (visual-language alignment): 22M image-text pairs are used to train the MAR encoder and decoder with the LLM frozen, at 256 resolution. Stage II (comprehensive multimodal training): the LLM is unfrozen and jointly trained on 25M QA data and 50M image-text data at 256 resolution. Stage III (high-quality fine-tuning): high-quality QA data and 10M curated images at 512 resolution.
    - Design Motivation: Direct end-to-end training leads to interference between generation and understanding tasks. The staged approach — alignment → capability building → quality refinement — allows the encoder to be fully optimized for both tasks at each stage.

3. **Masked Autoregressive Generation with Diffusion Decoding**:

    - Function: Images are generated progressively with a cosine-scheduled decrease in masking ratio.
    - Mechanism: Starting from full masking $m_0 = hw$, the number of masked tokens decreases over $K$ steps following a cosine schedule $m_k = hw \cdot \cos(\frac{k}{2K}\pi)$. At each step, the decoder uses a lightweight MLP as a denoiser to predict masked patches, with the loss $\mathcal{L} = \mathbb{E}_{\varepsilon,t}[\|\varepsilon - \varepsilon_\theta(x_t|t, x_{\text{mask}})\|^2]$. Classifier-free guidance (CFG, weight 3.0) is applied at inference to enhance text control.
    - Design Motivation: The masked autoregressive paradigm of MAR naturally aligns with the causal attention of LLMs, and $K$-step inference is substantially more efficient than pure token-by-token autoregression.

### Loss & Training

Generation: diffusion loss (MSE noise prediction) + CFG training with 10% empty captions. Understanding: cross-entropy loss (computed only on answer tokens). The two losses are mixed according to data ratios (Stage III ratio 1:3:16). Total training cost: Harmon-1.5B trained on 32×A100 for 8 days.

## Key Experimental Results

### Main Results

Image understanding (multimodal QA benchmarks):

| Model | Encoder | LLM Size | POPE↑ | MME-P↑ | MMB↑ | SEED↑ | MMMU↑ |
|-------|---------|:--------:|:-----:|:------:|:----:|:-----:|:-----:|
| Janus-Pro† | SigLIP | 1.5B | 86.2 | 1444 | 75.5 | 68.3 | 36.3 |
| Show-o | MAGVIT-v2 | 1.3B | 80.0 | 1097 | 51.6 | 54.4 | 26.7 |
| **Harmon-1.5B** | MAR-H | 1.5B | **87.6** | 1155 | 65.5 | 67.1 | **38.9** |

Image generation (GenEval benchmark):

| Model | Single | Two | Count | Colors | Position | ColorAttr | Overall↑ |
|-------|:------:|:---:|:-----:|:------:|:--------:|:---------:|:--------:|
| Janus-Pro-1.5B | 0.98 | 0.82 | 0.51 | 0.89 | 0.65 | 0.56 | 0.73 |
| SDXL | 0.98 | 0.74 | 0.39 | 0.85 | 0.15 | 0.23 | 0.55 |
| **Harmon-1.5B** | **0.99** | **0.86** | **0.66** | 0.85 | **0.74** | 0.48 | **0.76** |

Visual quality (MJHQ-30K FID↓):

| Model | MJHQ FID↓ |
|-------|:---------:|
| Janus-Pro-1.5B | 9.53 |
| Show-o | 15.18 |
| **Harmon-1.5B** | **5.15** |

### Ablation Study

Synergistic effects of the shared encoder:

| Analysis | Finding |
|----------|---------|
| Understanding vs. dedicated encoder | POPE: Harmon 87.6 ≈ Janus-Pro 86.2; MMMU: Harmon 38.9 > Janus-Pro 36.3 |
| Generation vs. VQ-based methods | GenEval: Harmon 0.76 >> Show-o 0.53, LWM 0.47 |
| Shared vs. separate pathways | Joint training with understanding data improves generation performance (cross-task synergy observed in paper figures) |

### Key Findings

1. **The MAR encoder genuinely possesses dual capabilities**: The shared encoder matches dedicated SigLIP encoders on understanding benchmarks.
2. **Substantial improvements over VQ/VAE unified methods**: Harmon significantly outperforms Show-o, LWM, and Chameleon on all understanding benchmarks.
3. **State-of-the-art generation quality**: MJHQ FID of 5.15 substantially outperforms all unified models; GenEval Overall of 0.76 surpasses Janus-Pro.
4. **Cross-task synergy is real**: Joint training with understanding data improves generation performance, validating the value of unification over separation.

## Highlights & Insights

- **"Learning to understand through generation" insight**: MAR's MIM pretraining leads the encoder to jointly learn pixel fidelity and semantic representations — this finding has independent value beyond the proposed system.
- **Rigorous empirical validation**: Linear probing and GradCAM++ analyses provide compelling preliminary evidence that the MAR encoder is well-suited for unified modeling.
- **Elegant design**: Shared encoder + LLM + decoder, with no additional branches or adapters.
- **MJHQ FID of 5.15**: A commanding lead among unified models, demonstrating that generation quality need not be sacrificed.

## Limitations & Future Work

- High training cost (32×A100, 8 days), making reproduction difficult for small research groups.
- Image resolution is limited to 512×512; high-resolution generation has not been validated.
- Understanding performance still lags behind dedicated MLLMs (e.g., InternVL2, Qwen2-VL).
- Unified video understanding and generation remains unexplored.
- The Stage III data ratio (1:3:16) is skewed toward generation, potentially leaving the understanding side under-optimized.

## Related Work & Insights

- **vs. Janus/Janus-Pro**: These methods use a dedicated SigLIP encoder for understanding — Harmon demonstrates that a shared encoder can achieve comparable performance.
- **vs. Show-o/D-DiT**: These methods rely on VQGAN/VAE encoders — substantially weaker on understanding tasks; the MAR encoder is the key differentiator.
- **vs. ViLA-U**: Joint training of semantic alignment and reconstruction on VQ tokens proves difficult to balance; the MAR encoder naturally accommodates both objectives.
- **Broader insight**: "Generation is a sufficient condition for understanding" — Feynman's dictum, "What I cannot create, I do not understand," finds empirical support in this work.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The insight of using the MAR encoder for a unified framework is original and thoroughly validated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparisons across both understanding and generation dimensions, with complete ablations and preliminary analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is compellingly argued; experimental results are presented clearly and systematically.
- Value: ⭐⭐⭐⭐⭐ Makes a paradigmatic contribution to unified multimodal architecture design; the discovery of MAR as a unified encoder is likely to influence subsequent work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] UniTok: A Unified Tokenizer for Visual Generation and Understanding](../../NeurIPS2025/multimodal_vlm/unitok_a_unified_tokenizer_for_visual_generation_and_understanding.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ICCV 2025\] DWIM: Towards Tool-aware Visual Reasoning via Discrepancy-aware Workflow Generation & Instruct-Masking Tuning](dwim_towards_tool-aware_visual_reasoning_via_discrepancy-aware_workflow_generati.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[NeurIPS 2025\] HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation](../../NeurIPS2025/multimodal_vlm/hermesflow_seamlessly_closing_the_gap_in_multimodal_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
