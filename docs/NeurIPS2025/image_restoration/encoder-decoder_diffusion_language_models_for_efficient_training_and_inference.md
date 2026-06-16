---
title: >-
  [Paper Note] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference
description: >-
  [NeurIPS 2025][Image Restoration][discrete diffusion] This paper proposes E2D2, an encoder-decoder architecture for discrete diffusion language models that performs iterative denoising via a lightweight decoder while per…
tags:
  - "NeurIPS 2025"
  - "Image Restoration"
  - "discrete diffusion"
  - "encoder-decoder"
  - "block diffusion"
  - "language model"
  - "KV caching"
date: 2026-05-08
content_hash: afa4fc4488c7a1b3
---

# Encoder-Decoder Diffusion Language Models for Efficient Training and Inference

**Conference**: NeurIPS 2025
**arXiv**: [2510.22852](https://arxiv.org/abs/2510.22852)  
**Code**: [kuleshov-group/e2d2](https://github.com/kuleshov-group/e2d2)  
**Area**: Image Generation
**Keywords**: discrete diffusion, encoder-decoder, block diffusion, language model, KV caching

## TL;DR

This paper proposes E2D2, an encoder-decoder architecture for discrete diffusion language models that performs iterative denoising via a lightweight decoder while periodically updating representations through a large encoder, achieving faster inference (~3× vs. MDLM) and more efficient block diffusion training (halving FLOPs).

## Background & Motivation

Discrete diffusion models enable parallel token sampling in language modeling, offering inference speed advantages over autoregressive methods. However, existing approaches suffer from critical efficiency bottlenecks:

1. **Full-network invocation**: Decoder-only architectures require a complete forward pass at every denoising step, incurring substantial computational overhead.
2. **Block diffusion training cost**: BD3LM must simultaneously process a full clean sequence and a noised sequence ($2L$ tokens), making training forward FLOPs $2\times$ that of standard diffusion models.
3. **Lack of KV cache support**: Full-sequence diffusion models (e.g., MDLM, LLaDA) employ bidirectional attention, which precludes KV caching.

The authors' core insight is that discrete diffusion denoising fundamentally performs two types of computation—(1) representing clean tokens and (2) denoising masked tokens—which can be handled by separate specialized modules.

## Core Problem

How can an encoder-decoder decoupled design accelerate both training and inference of discrete diffusion models without sacrificing generation quality?

## Method

### 1. Architecture Design

**Encoder**: An $N_{\text{Enc}}$-layer Transformer that receives clean tokens (prompt + previously decoded tokens) and produces feature representations $\mathbf{h}_t = \text{Encoder}(\mathbf{x}_{t,\text{Enc}})$.

**Decoder**: A lightweight $N_{\text{Dec}}$-layer Transformer ($N_{\text{Dec}} \ll N_{\text{Enc}}$) that receives the current noised sequence being denoised and interacts with encoder outputs via cross-attention:

$$\mathbf{x}_{\text{logit}} = \text{Decoder}(\mathbf{z}_{t,\text{Dec}}, \mathbf{h}_t)$$

**Key design**: The decoder is invoked **multiple times** for iterative denoising, while the encoder is called **periodically**—only once per newly generated block—to update representations.

### 2. Two Connection Variants

- **Last Hidden State**: The encoder's final-layer output is prepended to each decoder layer's input, analogous to T5.
- **Shared KV Cache**: Layer $i$ of the decoder reuses the KV cache from layer $j$ of the encoder—suitable for fine-tuning from pretrained decoder-only models.

Both variants employ a **fused attention kernel** that merges self-attention and cross-attention into a single call, reducing kernel launch overhead.

### 3. Sampling Algorithm

```
1. Encode prompt → obtain encoder features h and KV cache
2. for each block b:
     Initialize block as all [MASK]
     for each denoising step:
       z_t ← Sample(Decoder(z_t, h))  // invoke only the lightweight decoder
     Feed generated block to encoder to update KV cache
```

Since the $T$ denoising steps per block invoke only the lightweight decoder, the primary computational cost is reduced from $BT \cdot O(\theta_{\text{full}})$ to $B \cdot O(\phi) + BT \cdot O(\theta_{\text{small}})$, where $O(\theta_{\text{small}}) \ll O(\phi)$.

### 4. Training Algorithm

A customized attention mask enables **single-pass forward training** over all blocks:
- The encoder uses a block-causal mask, where clean tokens attend only to the current and preceding blocks.
- The decoder uses a mask spanning $2L$ KV positions, where noised tokens attend to their own block plus the encoder representations of preceding clean blocks.

**FLOPs comparison** ($N$: total layers, $L$: sequence length, $S$: block size):

| Model | Attention FLOPs | MLP FLOPs |
|-------|----------------|-----------|
| MDLM | $4NDL^2$ | $24NLD^2$ |
| BD3LM | $4ND(L^2 + LS)$ | $48NLD^2$ |
| **E2D2** | $4(N_E+N_D)D\frac{L^2+LS}{2}$ | $24(N_E+N_D)LD^2$ |

Under the same total layer count, E2D2 reduces training FLOPs by **2× compared to BD3LM**.

## Key Experimental Results

### Text Summarization (CNN/DailyMail)

| Model | Layers | Throughput (tok/s) | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|--------|--------------------|---------|---------|---------|
| AR | 28 | 89.1 | 31.7 | 11.7 | 22.1 |
| MDLM | 28 | 49.3 | 30.6 | 12.5 | 22.7 |
| BD3LM | 12 | 135.1 | 35.8 | 13.7 | 23.7 |
| **E2D2** | 20/8 | **155.8** | **36.0** | **14.1** | **23.9** |

E2D2 achieves ~75% higher throughput than AR with superior ROUGE-L, and is ~3× faster than MDLM.

### Machine Translation (WMT14 de-en)

| Model | Layers | Throughput (tok/s) | BLEU |
|-------|--------|--------------------|------|
| AR | 32 | 77.6 | 25.2 |
| MDLM | 32 | 60.4 | 18.4 |
| BD3LM | 16 | 102.4 | 24.7 |
| **E2D2** | 24/8 | **124.3** | **25.1** |

### Mathematical Reasoning (GSM8K)

By fine-tuning from pretrained Qwen3 1.7B, E2D2 achieves competitive pass@1 accuracy, demonstrating the framework's effectiveness on reasoning tasks.

## Highlights & Insights

1. The insight of **decoupling clean representation from denoising computation** is both elegant and effective, representing an important architectural innovation for discrete diffusion modeling.
2. **Halving block diffusion training FLOPs** has direct practical value for scaling to larger models and longer sequences.
3. Support for **KV caching** addresses a core bottleneck in diffusion language model inference efficiency.
4. By adjusting the encoder-to-decoder layer ratio, E2D2 flexibly **traces the quality–throughput Pareto frontier**.
5. The **Shared KV Cache variant** enables direct fine-tuning from pretrained AR models, lowering the barrier to practical adoption.

## Limitations & Future Work

1. Experiments are conducted at a relatively small scale (<2B parameters); scalability of E2D2 at the 7–8B level remains unvalidated.
2. The encoder must periodically re-encode previously generated tokens, which may become a bottleneck for very long sequences.
3. Evaluation is currently limited to task-specific models; effectiveness for general language modeling warrants further investigation.
4. The fused attention kernel forces the decoder to distribute attention between encoder representations and its own hidden states, potentially limiting expressiveness in certain settings.

## Related Work & Insights

| Dimension | MDLM | BD3LM | LLaDA | E2D2 |
|-----------|------|-------|-------|------|
| Architecture | decoder-only | decoder-only | decoder-only | **enc-dec** |
| KV Cache | ✗ | ✓ | approximate | **✓** |
| Training FLOPs (vs. BD3LM) | 1× | 2× | 1× | **1×** |
| Inference Throughput (relative) | low | medium | low | **high** |
| Block Decoding | optional | native | inference-time | **native** |

**Broader insights and connections:**

1. **Separation of "representation" and "generation"** is a general principle: analogous encoder-decoder decoupling may yield efficiency gains in multi-step iterative generation settings such as VLMs and speech synthesis.
2. **Block diffusion as a bridge between AR and diffusion**: E2D2 further reinforces this paradigm, suggesting that block-level autoregression combined with intra-block diffusion may represent optimal practice for diffusion language models.
3. Initializing diffusion training from pretrained AR models has been validated across multiple works, suggesting that AR pretraining may provide favorable weight initialization for diffusion models.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of applying an encoder-decoder architecture to discrete diffusion is natural yet previously unexplored.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-task coverage, Pareto frontier analysis, and rigorous theoretical FLOPs derivation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Excellent paper structure, clear algorithmic pseudocode, and rigorous FLOPs derivation.
- Value: ⭐⭐⭐⭐ — Provides an effective architectural solution toward practical deployment of diffusion language models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates](rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates.md)
- [\[ICML 2026\] Consistent Diffusion Language Models](../../ICML2026/image_restoration/consistent_diffusion_language_models.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)
- [\[NeurIPS 2025\] Elucidated Rolling Diffusion Models for Probabilistic Forecasting of Complex Dynamics](elucidated_rolling_diffusion_models_for_probabilistic_forecasting_of_complex_dyn.md)
- [\[ICML 2026\] DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention](../../ICML2026/image_restoration/dyllm_efficient_diffusion_llm_inference_via_saliency-based_token_selection_and_p.md)

</div>

<!-- RELATED:END -->
