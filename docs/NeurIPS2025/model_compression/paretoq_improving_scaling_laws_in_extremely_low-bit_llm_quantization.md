---
title: >-
  [Paper Note] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization
description: >-
  [NeurIPS 2025][Model Compression][Extremely low-bit quantization] This paper proposes ParetoQ — the first unified framework supporting 1/1.58/2/3/4-bit quantization — which systematically studies training strategies (ful…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Extremely low-bit quantization"
  - "scaling laws"
  - "quantization-aware training"
  - "2-bit quantization"
  - "Pareto optimality"
date: 2026-05-08
content_hash: e042d4d6dc4487fd
---

# ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization

**Conference**: NeurIPS 2025
**arXiv**: [2502.02631](https://arxiv.org/abs/2502.02631)  
**Code**: Not available  
**Area**: Model Compression
**Keywords**: Extremely low-bit quantization, scaling laws, quantization-aware training, 2-bit quantization, Pareto optimality

## TL;DR

This paper proposes ParetoQ — the first unified framework supporting 1/1.58/2/3/4-bit quantization — which systematically studies training strategies (full-precision pretraining vs. QAT budget allocation) and quantization function design (introducing the SEQ quantizer). The work demonstrates that 2-bit and 1.58-bit quantization outperform conventional 4-bit in the accuracy–model-size trade-off, and achieves state-of-the-art results across all bit-widths.

## Background & Motivation

A central debate in LLM quantization concerns: **what is the optimal bit-width?**

- One camp (Dettmers & Zettlemoyer 2023) argues that 4-bit or 6-bit is Pareto optimal.
- Another camp (Ma et al. 2024; Kaushal et al. 2024) claims that 1.58-bit suffices to match full-precision performance.

**Why do conclusions conflict?** The absence of a unified framework means different works use different training recipes, quantization functions, and baselines, rendering their conclusions incomparable.

**Key observation**: Prior scaling-law studies reduce the search space to $\mathcal{L}(\mathcal{N}, \mathcal{D}, \mathcal{P})$ (model size, data volume, precision), neglecting two critical factors: training strategy $\mathcal{S}_{\text{train}}$ and bit-specific quantization function $\mathcal{F}$. The correct search space is $\mathcal{L}(\mathcal{N}, \mathcal{D}, \mathcal{P}, \mathcal{S}_{\text{train}}, \mathcal{F})$ — a five-dimensional space.

**Core finding**: A significant behavioral shift exists between 2-bit and 3-bit quantization — at 3-bit and above, QAT acts as *compensation* (weight changes of 10–20%), whereas at 2-bit and below, it performs *reconstruction* (weight changes of ~40%).

## Method

### Overall Architecture

ParetoQ's methodology proceeds in three steps:

1. **Fix the quantization function, find the optimal training strategy**: $\mathcal{L}(\mathcal{N}, \mathcal{D}, \mathcal{S}_{\text{train}} | \mathcal{P}, \mathcal{F})$
2. **Fix the optimal training strategy, find the optimal quantization function**: $\mathcal{L}(\mathcal{N}, \mathcal{F} | \mathcal{P}, \mathcal{D}^*, \mathcal{S}_{\text{train}}^*)$
3. **Fix optimal training and quantization, compare across bit-widths**: $\mathcal{L}(\mathcal{N}, \mathcal{P} | \mathcal{F}^*, \mathcal{D}^*, \mathcal{S}_{\text{train}}^*)$

### Key Designs

1. **Training budget allocation strategy**: Given a fixed total training budget $\mathcal{B}_{\text{train}} = \mathcal{B}_{\text{FP}} + \mathcal{B}_{\text{QAT}}$, the paper investigates the optimal allocation between full-precision pretraining and QAT fine-tuning. Experiments on MobileLLM-125M reveal:

    - **~90% for full-precision pretraining + ~10% for QAT** is the optimal allocation, holding consistently across nearly all bit-widths.
    - Training from scratch with QAT (allocating the entire budget to quantized training) consistently underperforms pretrain-then-finetune.
    - QAT for 3-bit/4-bit saturates at ~10B tokens; 1-bit/1.58-bit/2-bit saturates at ~30B tokens.

   **Design Motivation**: Low-bit quantization (≤2-bit) requires more training tokens because weight *reconstruction* demands a larger search space than *compensation*.

2. **Stretched Elastic Quant (SEQ) quantizer**: A key innovation for 1.58-bit and 2-bit quantization. The problem: 2-bit quantization has 4 quantization levels — including zero (e.g., $\{-2,-1,0,1\}$) yields only one positive level (imbalanced); excluding zero (e.g., $\{-1.5,-0.5,0.5,1.5\}$) yields balance but cannot represent zero. SEQ resolves this via:

    $\mathbf{W}_Q^i = \alpha \left(\lfloor \text{Clip}\left(\frac{\mathbf{W}_R^i}{\alpha}, -1, 1\right) \times \frac{k}{2} - 0.5 \rceil + 0.5 \right) / k \times 2$

   This simultaneously achieves balanced quantization levels and uniform coverage of the full-precision weight range. For 3-bit/4-bit, LSQ (which benefits from including zero) is retained.

3. **Unified ParetoQ quantization formula**:

    $\mathbf{W}_Q^i = \begin{cases} \alpha \cdot \text{Sign}(\mathbf{W}_R^i), & N_{\text{bit}} = 1 \\ \alpha(\lfloor \text{Clip}(\frac{\mathbf{W}_R^i}{\alpha}, -1, 1) \times k/2 - 0.5 \rceil + 0.5)/k \times 2, & N_{\text{bit}} = 1.58, 2 \\ \alpha \lfloor \text{Clip}(\frac{\mathbf{W}_R^i}{\alpha}, n, p) \rceil, & N_{\text{bit}} = 3, 4 \end{cases}$

   Backpropagation uses the Straight-Through Estimator (STE), with separate gradient definitions for weights and scaling factor $\alpha$. Initialization of $\alpha$: $\ell_1$ mean for 1-bit, maximum absolute value for all other bit-widths.

### Loss & Training

- AdamW optimizer, zero weight decay, 16 GPUs, batch size 8 per GPU.
- 1/1.58/2-bit: 120K steps, learning rate $2 \times 10^{-5}$, cosine decay.
- 3/4-bit: 40K steps, learning rate $1 \times 10^{-5}$, cosine decay.
- All weights quantized except embeddings and the output layer.

## Key Experimental Results

### Main Results: LLaMA-3 8B Across Bit-Widths

| Method | Bits | ARC-e | ARC-c | PIQA | HellaS | WinoG | Avg. | Wiki2 |
|---|---|---|---|---|---|---|---|---|
| Full Precision | 16 | 81.0 | 57.7 | 81.0 | 79.5 | 73.9 | 74.6 | 6.15 |
| EfficientQAT | 2 | 69.3 | 46.8 | 76.4 | 69.0 | 66.3 | 65.5 | 9.6 |
| **ParetoQ** | **2** | **78.5** | **54.5** | **79.2** | **73.8** | **70.0** | **71.2** | **8.0** |
| 1-bit Era | 1.58 | 72.8 | 45.4 | 81.0 | 70.6 | 58.0 | 65.6 | 11.7 |
| **ParetoQ** | **1.58** | **76.3** | **51.4** | **77.7** | **71.9** | **67.7** | **69.0** | **8.6** |
| BiLLM | 1 | 33.2 | 25.6 | 54.6 | 32.7 | 50.5 | 39.3 | 38.5 |
| **ParetoQ** | **1** | **75.5** | **51.9** | **76.6** | **69.4** | **65.6** | **67.8** | **9.5** |

### Ablation Study: Impact of Quantization Function Choice

| Quantizer | 1.58-bit Acc. | 2-bit Acc. | 3-bit Acc. | 4-bit Acc. |
|---|---|---|---|---|
| Min-Max (stats) | Poor | Collapse | Usable | Good |
| Range clipping (stats) | Good | Good | Poor | Poor |
| LSQ (learnable) | Medium | Medium | **Best** | **Best** |
| **SEQ (learnable)** | **Best** | **Best** | Slightly lower | Slightly lower |

### Key Findings

- **Pareto curves challenge conventional wisdom**: 1.58-bit, 2-bit, and 3-bit all outperform 4-bit in the accuracy–model-size trade-off.
- ParetoQ 1.58-bit 8B closes the gap to full precision by 37.8% relative to 1-bit Era, using only 30% of the training tokens.
- **A ParetoQ 600M ternary model surpasses the previous SOTA 3B ternary model** — achieving equivalent performance with 1/5 the parameters.
- A behavioral transition exists between 2-bit and 3-bit: ≥3-bit is *compensation* (small weight adjustments); ≤2-bit is *reconstruction* (large weight changes).
- 2-bit benefits from CPU kernel acceleration, yielding a better accuracy–speed trade-off than 4-bit.
- 1.58-bit and 3-bit are less hardware-friendly than 2-bit (1.58-bit has complex storage requirements; 3-bit suffers from alignment difficulties).

## Highlights & Insights

- The paper's greatest methodological contribution is formalizing the fragmented low-bit quantization landscape as a five-dimensional search problem, enabling rigorous apples-to-apples comparisons for the first time.
- The SEQ quantizer embodies the insight that, in low-bit settings, balanced quantization levels are more important than including zero.
- The *compensation vs. reconstruction* dichotomy provides an intuitive explanation for QAT behavior across different bit-widths.
- 2-bit quantization holds practical deployment value as a potential successor to 4-bit — native INT2 hardware support represents an important direction for the community to pursue.

## Limitations & Future Work

- Experiments cover only the MobileLLM and LLaMA-3 families (up to 8B); larger models (70B+) remain unvalidated.
- The 2-bit kernel implementation is CPU-only; GPU-side native INT2 support is lacking.
- Only weight quantization is addressed; activation quantization is not considered.
- Training costs remain substantial (120K steps × 16 GPUs), posing challenges for resource-constrained researchers.
- Comparisons with mixed-precision quantization methods are absent.

## Related Work & Insights

- The comparison with 1-bit Era (Ma et al. 2024) is the most compelling: under the unified framework, ParetoQ substantially outperforms it with fewer training tokens and simpler optimization.
- Relative to the 4-bit conclusion of Dettmers & Zettlemoyer (2023), ParetoQ demonstrates that better quantization function design can shift the Pareto frontier toward lower bit-widths.
- Takeaway: quantization exhibits a "no free lunch" property — each bit-width requires a tailored quantization function, and the value of a unified framework lies in providing a fair benchmark.

## Rating

- Novelty: ⭐⭐⭐⭐ The unified framework and SEQ quantizer are substantive contributions, though the work leans more toward systematic engineering research.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Eight models × five bit-widths, covering PTQ/QAT/VQ baselines — exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and rich figures; the unfolding of the five-dimensional search space is occasionally verbose.
- Value: ⭐⭐⭐⭐⭐ Establishes an authoritative benchmark for the low-bit quantization field; the discovery of 2-bit's potential carries significant practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LittleBit: Ultra Low-Bit Quantization via Latent Factorization](littlebit_ultra_low-bit_quantization_via_latent_factorization.md)
- [\[NeurIPS 2025\] Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression](learning_grouped_lattice_vector_quantizers_for_low-bit_llm_compression.md)
- [\[NeurIPS 2025\] Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment](q-palette_fractional-bit_quantizers_toward_optimal_bit_allocation_for_efficient_.md)
- [\[NeurIPS 2025\] FALQON: Accelerating LoRA Fine-tuning with Low-Bit Floating-Point Arithmetic](falqon_accelerating_lora_fine-tuning_with_low-bit_floating-point_arithmetic.md)
- [\[ICCV 2025\] MSQ: Memory-Efficient Bit Sparsification Quantization](../../ICCV2025/model_compression/msq_memory-efficient_bit_sparsification_quantization.md)

</div>

<!-- RELATED:END -->
