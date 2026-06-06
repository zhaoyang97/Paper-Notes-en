---
title: >-
  [Paper Note] DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression
description: >-
  [AAAI 2026][Model Compression][Image compression quantization] To address the deployment inefficiency of learned image compression (LIC) models, this paper proposes DynaQuant…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Image compression quantization"
  - "mixed-precision"
  - "dynamic bit-width allocation"
  - "quantization-aware training"
  - "learned image compression"
date: 2026-05-08
content_hash: 27aa7884e7ca2fc6
---

# DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression

**Conference**: AAAI 2026
**arXiv**: [2511.07903](https://arxiv.org/abs/2511.07903)  
**Code**: [https://github.com/baoyu2020/DynaQuant](https://github.com/baoyu2020/DynaQuant)  
**Area**: Model Compression
**Keywords**: Image compression quantization, mixed-precision, dynamic bit-width allocation, quantization-aware training, learned image compression

## TL;DR
To address the deployment inefficiency of learned image compression (LIC) models, this paper proposes DynaQuant, a framework that achieves content-adaptive quantization at the parameter level via learnable scale/zero-point combined with a Distance-Aware Gradient Modulator, and dynamically assigns optimal bit-widths per layer at the architecture level via a lightweight Bit-Width Selector. Across three baselines (Cheng2020, ELIC, Ballé), the framework achieves near-FP32 R-D performance while delivering up to 5.17× speedup and reducing model size to approximately 1/4 of the original.

## Background & Motivation
LIC models such as ELIC and Cheng2020 have surpassed traditional codecs like VVC in R-D performance, but their computational complexity and memory demands make deployment on edge devices (e.g., mobile phones, drones) highly challenging. Existing quantization methods exhibit two critical limitations: (1) they apply a **globally uniform bit-width** (e.g., full INT8), ignoring the large variation in sensitivity to quantization noise across different layers of LIC models; and (2) quantization parameters (scale, zero-point) are **statically fixed**, unable to adapt to the highly input-dependent latent feature distributions in LIC. This results in either over-conservative treatment of robust layers (wasting compute) or over-aggressive treatment of sensitive layers (degrading R-D performance).

## Core Problem
How to design a **two-level dynamic** quantization strategy for LIC models: (1) at the parameter level — quantization parameters that adapt to input content; and (2) at the architecture level — bit-widths that are dynamically allocated per layer according to layer sensitivity and data characteristics? Additionally, how to address the training difficulty caused by the non-differentiable rounding operation in quantization?

## Method

### Overall Architecture
DynaQuant comprises two complementary modules: **Dynamic Parameter Adaptation (DPA)** for parameter-level adaptation, and **Dynamic Bit-Width Selector (DBWS)** for layer-level bit-width allocation. Both are embedded within a standard QAT pipeline and jointly optimized end-to-end. The hyperencoder is fixed at 8-bit quantization, as it contains few parameters and is sensitive to quantization, making dynamic allocation both marginally beneficial and costly.

### Key Designs
1. **Content-Aware Quantization Mapping**: The static scale $s$ and zero-point $z$ in conventional QAT are replaced by **learnable per-channel parameters**, optimized end-to-end via backpropagation through the R-D loss. This allows the quantization mapping to adapt to the latent feature distribution of each input image.

2. **Distance-Aware Gradient Modulator (DGM)**: To address the limitation of the Straight-Through Estimator (STE), which crudely approximates the rounding gradient as a constant 1, a new gradient surrogate function is proposed: $g(x) = \frac{1}{2} \cdot \frac{\tanh(\beta(x - \lfloor x \rfloor) - 0.5)}{\tanh(0.5)} + 0.5$. Its gradient varies with the distance of the input to the nearest quantization boundary (half-integer, e.g., 0.5): **values near the boundary receive larger gradients** (emphasizing the need for further optimization), while **values near quantization centers receive smaller gradients** (indicating stability), providing a more precise optimization signal than STE.

3. **Dynamic Bit-Width Selector (DBWS)**: A lightweight network module that takes an input activation tensor $A \in \mathbb{R}^{C \times H \times W}$, processes it through AdaptivePool (output 5×5) → Flatten → two-layer MLP (with Dropout $p=0.2$) → Reshape → **Gumbel-Softmax** (soft sampling during training, argmax hard selection during inference), and outputs a probability distribution over the candidate bit-width set $\mathcal{B} = \{b_1, b_2, ..., b_M\}$ for each layer. The encoder and decoder each have independent DBWS modules with symmetric architectures, ensuring consistent bit-width strategies between encoder and decoder — **without requiring transmission of additional bit-width configuration metadata**.

### Loss & Training
The joint optimization loss is: $\mathcal{L} = R + \lambda D + \gamma \mathcal{L}_{\text{bits}}$

- $R$: entropy model estimated bitrate of the quantized latent
- $D$: reconstruction distortion (MSE or MS-SSIM)
- $\mathcal{L}_{\text{bits}} = \frac{1}{L} \sum_{l=1}^{L} \sum_{k=1}^{M} (p_l)_k \cdot b_k$: expected average bit-width across all dynamically quantized layers; $\gamma$ controls the trade-off between R-D performance and computational efficiency

DBWS input strategy: the first module of each encoder/decoder is fixed at 8-bit quantization, and its output serves as the input to the corresponding DBWS; all subsequent modules (2nd through $BL$-th) use the adaptive bit-widths output by DBWS.

## Key Experimental Results

**Table 1 Main Results** (BD-Rate loss % / Speedup / Model size):

| Model | Method | Kodak BD-Rate | Avg. BD-Rate | Speedup | Model Size |
|------|------|:---:|:---:|:---:|:---:|
| Cheng | FP32 Baseline | 0.00% | 0.00% | 1.00× | 45.08 MB |
| Cheng | FMPQ | 0.89% | 1.30% | 4.00× | ~11.27 MB |
| Cheng | RDO-PTQ | 4.88% | 4.88% | 4.00× | ~11.27 MB |
| Cheng | **Q-Cheng (DPA)** | **1.02%** | **1.60%** | 4.00× | 11.27 MB |
| Cheng | **DQ-Cheng (DPA+DBWS)** | 7.15% | 12.18% | **5.17×** | **8.72 MB** |
| ELIC | FP32 Baseline | 0.00% | 0.00% | 1.00× | 137.11 MB |
| ELIC | **Q-ELIC** | 5.97% | 4.92% | 4.00× | 34.28 MB |
| ELIC | **DQ-ELIC** | 7.62% | 6.39% | **4.61×** | **29.78 MB** |
| Ballé | FP32 Baseline | 0.00% | 0.00% | 1.00× | 19.37 MB |
| Ballé | FMPQ | 6.48% | 7.50% | ~3.98× | ~4.87 MB |
| Ballé | **Q-Ballé** | 5.85% | **5.01%** | 4.00× | 4.84 MB |
| Ballé | **DQ-Ballé** | 7.63% | 6.84% | **4.55×** | **4.26 MB** |

Key observations: Q- (fixed 8-bit DPA) achieves only 1.60% BD-Rate loss on Cheng, outperforming RDO-PTQ (4.88%) and approaching FMPQ (1.30%); DQ- variants trade a modest increase in BD-Rate for an additional ~1.2× speedup.

### Ablation Study

**Table 2 General Ablation** (Cheng2020 q=6, Kodak):
- DPA INT8: bpp=0.828, PSNR=36.649, R-D loss=1.56 (outperforms PAMS: 36.185/1.64)
- DPA-DQ: avg. 6.42-bit, PSNR=36.636, R-D loss=1.57 (reduces average bit-width by 25% with negligible quality loss)
- PAMS-DQ: avg. 6.85-bit, PSNR=30.262, R-D loss=4.28 → DPA and DBWS exhibit **synergistic effects** (not simply additive)

**Table 3 DPA Component Ablation** (removing any component degrades performance):
- Remove learnable $s$: PSNR 36.649 → 36.185 (−0.464 dB)
- Remove learnable $z$: PSNR → 36.323 (−0.326 dB)
- Remove DGM gradient modulation $g(x)$: PSNR → 36.288 (−0.361 dB)
- All three components are important; scale $s$ has the largest impact

**Table 4 DBWS Candidate Bit-Width Set Ablation**:
- {4,6,8}: avg. 5.47-bit, PSNR=36.432
- {6,8,10}: avg. 6.42-bit, PSNR=36.636 (better efficiency–fidelity trade-off)

**Bit-Width Allocation Visualization** (Fig. 6): The texture-rich Kodim14 is assigned 10-bit in the gs-1 layer (exceeding the 8-bit assigned to other images); boundary layers (ga-0, ga-6, gs-1) tend toward higher precision while intermediate layers use lower precision — confirming that bottleneck layers genuinely require more bits.

## Highlights & Insights
- **Clear two-level dynamic design**: The parameter-level and architecture-level components are both independent and complementary, forming a complete adaptive quantization pipeline
- **Theoretically motivated DGM**: Rather than simply replacing STE, DGM is designed based on the intuition of "distance to decision boundary," enabling more targeted optimization of quantization parameters
- **Symmetric DBWS for encoder/decoder**: Eliminates the need to transmit additional bit-width configuration metadata, improving practical deployability
- **Convincing synergy experiment**: Table 2 clearly demonstrates that DPA + DBWS yields greater benefit than the sum of each component individually
- **Cross-architecture generalization**: Effective across three structurally distinct LIC models — Cheng2020, ELIC, and Ballé

## Limitations & Future Work
- **Significant BD-Rate degradation in DQ mode**: DQ-Cheng reaches a BD-Rate loss as high as 16.52% on JPEG-AI, indicating that dynamic bit-width allocation still incurs noticeable quality degradation on certain datasets/content types, with speedup coming at a non-trivial quality cost
- **Validation limited to three relatively dated LIC baselines**: Cheng2020 (2020), Ballé (2018), and ELIC (2022) are not the latest state-of-the-art; architectures such as MambaIC are not evaluated
- **Insufficient justification for fixing the hyperencoder at 8-bit**: The decision is attributed to empirical observation without quantitative sensitivity analysis
- **Candidate bit-width sets must be manually specified**: Presets of {4,6,8} or {6,8,10} are used without exploring adaptive determination of candidate sets
- **No comparison with recent general PTQ/QAT methods** (e.g., LIC-adapted variants of GPTQ, AWQ, etc.)
- **Latency overhead introduced by DBWS is not explicitly reported**: Although claimed to be lightweight, the concrete overhead is not clearly quantified

## Related Work & Insights
- vs. **FMPQ**: DynaQuant's fixed-precision mode (Q-Cheng) is competitive with FMPQ (1.60% vs. 1.30%), while additionally providing dynamic bit-width capability for greater flexibility
- vs. **RDO-PTQ**: RDO-PTQ requires no retraining but incurs larger BD-Rate loss (4.88% vs. 1.60%); DynaQuant's QAT approach is clearly superior
- vs. **RAQ**: RAQ achieves a BD-Rate loss of 27.84% on Kodak, rendering it essentially unusable
- vs. **General mixed-precision quantization (HAQ/HAWQ, etc.)**: These methods search for bit-width allocation via reinforcement learning or Hessian information; DynaQuant instead learns the allocation end-to-end with a lightweight MLP + Gumbel-Softmax, avoiding costly search procedures
- vs. **Instance-aware quantization (InstAQ, etc.)**: DynaQuant is similarly content-adaptive, but adds the additional dimension of per-layer dynamic bit-width allocation

The DGM gradient modulation idea is transferable to other scenarios requiring differentiable rounding (e.g., codebook learning in VQ-VAE, quantization modules in neural codecs). The "differentiable discrete selection via Gumbel-Softmax" paradigm in DBWS is common in NAS and dynamic networks, but its application to bit-width allocation in LIC represents a novel combination. The approach is conceptually analogous to **token pruning in model compression** — both allocate varying computational resources to different components, here along the precision dimension rather than the quantity dimension. Incorporating semantic region information (e.g., ROI) could potentially enable finer-grained spatially adaptive quantization.

## Rating
- **Novelty**: ⭐⭐⭐☆☆ — DGM and DBWS are not individually novel concepts (drawing from QuantSR and Gumbel-Softmax in NAS, respectively), but their integration into a unified framework targeting LIC with demonstrated synergistic effects constitutes a meaningful contribution
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ — Ablation study design is comprehensive (separate ablations for general, DPA, and DBWS components) with generalization validated across three baselines; however, the baselines are somewhat dated and comparisons with recent quantization methods are lacking
- **Writing Quality**: ⭐⭐⭐⭐☆ — Structure is clear, method description is complete, and figures are high quality (particularly the bit-width visualization in Fig. 6); the future work discussion in the Conclusion is overly brief
- **Value**: ⭐⭐⭐☆☆ — Practically valuable for LIC deployment (5× speedup is significant), but the substantial R-D loss in DQ mode limits practical applicability; the transferability of core technical components is moderate

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache](kvmix_gradient-based_layer_importance-aware_mixed-precision_.md)
- [\[ICML 2026\] Efficient Learned Image Compression without Entropy Coding](../../ICML2026/model_compression/efficient_learned_image_compression_without_entropy_coding.md)
- [\[ICML 2026\] GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs](../../ICML2026/model_compression/gemq_global_expert-level_mixed-precision_quantization_for_moe_llms.md)
- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](../../ICCV2025/model_compression/learned_image_compression_with_hierarchical_progressive_context_modeling.md)
- [\[ICCV 2025\] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](../../ICCV2025/model_compression/mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)

</div>

<!-- RELATED:END -->
