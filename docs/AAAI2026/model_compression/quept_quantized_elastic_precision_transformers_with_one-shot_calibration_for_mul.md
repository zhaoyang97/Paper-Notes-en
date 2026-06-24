---
title: >-
  [Paper Note] QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching
description: >-
  [AAAI 2026][Model Compression][Elastic Precision Quantization] QuEPT is an elastic precision quantization framework that enables real-time switching among arbitrary predefined bit-widths on ViT/LLM/MLLM after a single calibration pass, via two core modules—Multi-Bit Token Merging and Multi-Bit Cascaded LoRA—achieving performance on par with or exceeding single-bit-width SOTA PTQ methods.
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Elastic Precision Quantization"
  - "Post-Training Quantization"
  - "Multi-Bit Switching"
  - "Cascaded LoRA"
  - "Token Merging"
date: 2026-05-08
content_hash: 2164e21144e8abbb
---

# QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching

**Conference**: AAAI 2026
**arXiv**: [2602.12609](https://arxiv.org/abs/2602.12609)  
**Code**: [https://github.com/xuke225/QuEPT](https://github.com/xuke225/QuEPT)  
**Area**: Model Compression
**Keywords**: Elastic Precision Quantization, Post-Training Quantization, Multi-Bit Switching, Cascaded LoRA, Token Merging

## TL;DR
QuEPT is an elastic precision quantization framework that enables real-time switching among arbitrary predefined bit-widths on ViT/LLM/MLLM after a single calibration pass, via two core modules—Multi-Bit Token Merging and Multi-Bit Cascaded LoRA—achieving performance on par with or exceeding single-bit-width SOTA PTQ methods.

## Background & Motivation

Model quantization reduces computation and memory overhead by representing weights and activations with low-bit fixed-point numbers. However, most existing quantization methods target a single predefined bit-width, and changing the bit-width requires re-optimization. **Elastic Precision Quantization** addresses this by handling multiple predefined bit-widths in a single optimization pass, enabling quantized models to dynamically adapt to varying bit-width requirements.

**Challenges in Transitioning from CNNs to Transformers**: Multi-bit-width quantization has seen reasonable success on CNNs (RobustQuant, Any-Precision, EQ-Net, etc.), but Transformers introduce unique difficulties:

- **Dense inter-token dependencies**: Dynamic sparsity induced by the attention mechanism amplifies quantization noise
- **Heterogeneous layer-wise bit-width sensitivity**: Different layers exhibit vastly different sensitivity to different bit-widths
- **Large activation ranges**: Activation distributions in Transformers are far wider than those in CNNs

**Limitations of Prior Work on Multi-Bit-Width Quantization**:
- **Any-Precision LLM**: Covers weight-only quantization without activation quantization, resulting in poor low-precision performance
- **MatQuant**: Employs co-training and co-distillation regularization but does not account for inter-bit-width competition
- **QAT-based methods** (RobustQuant, EQ-Net, etc.): Training overhead is prohibitive for large-scale Transformers

**Key Challenge**: In joint multi-bit-width optimization, aggressive quantization at low bit-widths degrades the representational quality of medium- and high-bit-width settings, causing overall performance to be bottlenecked by the lowest precision configuration—a phenomenon of **competitive conflict** among bit-widths.

**Key Insight**: Design a PTQ framework that maintains balanced cross-bit-width performance via a token merging strategy, and establishes hierarchical parameter sharing across bit-widths via a cascaded LoRA structure, converting conflict into synergy.

## Method

### Overall Architecture

QuEPT proceeds via block-wise reconstruction:
1. Initialize weight clipping parameters and cascaded LoRA adapters (weights and quantization scales are frozen)
2. Partition the target bit-width set $\mathcal{B}$ into low-bit group $\mathcal{B}_L$, mid-bit group $\mathcal{B}_M$, and high-bit group $\mathcal{B}_H$
3. At each step, sample one bit-width from each group; fuse multi-bit-width features from the previous block via MB-ToMe
4. Forward propagate the current block under the three sampled bit-widths; backpropagate the reconstruction loss to update cascaded LoRA and clipping thresholds
5. At deployment, simply select the corresponding LoRA slice to switch to any target bit-width

### Key Designs

#### 1. **Multi-Bit Token Merging (MB-ToMe)**
- **Function**: Fuses token features across different bit-widths during multi-bit-width optimization to maintain balanced performance across precisions
- **Mechanism**: Three strategies were explored:
    - Case 1 (Random Selection): Each token randomly adopts features from one bit-width → unstable quality
    - Case 2 (Uniform Fusion): Equal 1:1:1 weighted fusion of the three bit-width groups → loss of high-bit-width detail
    - Case 3 (Selective Merging, final design): Selectively retain and fuse based on quantization robustness
- **Key Formula**:
$$X_k' = \begin{cases} X_k^H, & \text{if } k \in \Phi \\ \lambda_1 X_k^H + \lambda_2 X_k^M + \lambda_3 X_k^L, & \text{else} \end{cases}$$
  where $\Phi$ is the index set of the top-p% stable tokens selected by ranking cosine similarity between 8-bit and 4-bit features
- **Design Motivation**: High-precision features of robust tokens should be preserved as structural anchors, while unstable tokens maintain feature continuity through fusion. Merging dissimilar tokens produces more uniform numerical distributions, making them more robust across bit-widths

#### 2. **Multi-Bit Cascaded LoRA (MB-CLoRA)**
- **Function**: Establishes synergistic relationships among different bit-widths through a hierarchical LoRA parameter-sharing structure
- **Mechanism**: All bit-widths share a unified LoRA parameter pair $A \in \mathbb{R}^{r \times q}$ and $B \in \mathbb{R}^{p \times r}$; different ranks are extracted via cascaded truncation:
$$R^{(b)} = B_{[:,:r_b]} A_{[:r_b,:]}$$
  Rank allocation follows a cascaded pattern where lower bit-widths are assigned larger ranks:
$$r_b = \begin{cases} r_h & b \in \mathcal{B}_H \\ r_h + r_m & b \in \mathcal{B}_M \\ r_h + r_m + r_l & b \in \mathcal{B}_L \end{cases}$$
- **Design Motivation**: Lower bit-widths incur larger quantization errors and require greater compensatory capacity. The cascaded structure naturally makes the compensation matrix of lower bit-widths include the higher bit-width parameters as a leading submatrix, establishing an inheritance relationship. Gradients from high-precision settings naturally propagate to their low-precision counterparts.

#### 3. **Joint Optimization of Weight Clipping Parameters**
- **Function**: Jointly optimizes weight clipping thresholds alongside LoRA parameters
- **Key Formula**:
$$s_w^b = \frac{\alpha_b \times \max(W + B_b A_b) - \beta_b \times \min(W + B_b A_b)}{2^{b-1}}$$
- **Design Motivation**: As LoRA parameters are updated, the quantized weights continuously change, rendering fixed clipping thresholds suboptimal. The clipping mechanism handles large errors caused by weight outliers, providing LoRA with smoother error signals.

### Loss & Training

- MAE loss is used for block-wise reconstruction (experimentally shown to outperform MSE):
$$\min_{A,B,\alpha,\beta} \sum_{b \in \{b_L, b_M, b_H\}} \|WX - \widehat{(W+R^{(b)})} X'\|_1$$
- ViT calibration data: 1,024 unlabeled ImageNet images
- LLM calibration data: 128 C4 samples
- MLLM calibration data: 128 image-text pairs from ShareGPT4V
- LoRA is placed inside the clipping operator, incurring no additional inference overhead
- Mixed-precision can be achieved by evaluating per-layer sensitivity via KL divergence combined with dynamic programming

## Key Experimental Results

### Main Results (ViT ImageNet Top-1 Accuracy)

| Model | Method | Type | W4A4 | W5A5 | W6A6 | W8A8 | Time |
|-------|--------|------|------|------|------|------|------|
| ViT-S | ERQ | Single-bit | 68.9 | 78.8 | 80.5 | 81.2 | 9×N |
| ViT-S | PTMQ | Multi-bit | - | - | 76.1 | 78.2 | 430min |
| ViT-S | **QuEPT** | Multi-bit | **75.1** | **79.7** | **80.6** | **81.2** | **17min** |
| ViT-B | PTMQ | Multi-bit | - | - | 77.7 | 79.1 | 950min |
| ViT-B | **QuEPT** | Multi-bit | **80.7** | **83.3** | **83.8** | **84.3** | **36min** |

QuEPT achieves +6.2% over ERQ at W4A4 on ViT-S, with a calibration time only 1/26 that of PTMQ.

### LLM Results (LLaMA Series)

| Model | Bits | Method | WikiText2 PPL↓ | C4 PPL↓ | 5-Task Avg↑ |
|-------|------|--------|---------------|---------|------------|
| L2-7B | W4A4 | DuQuant | 6.28 | 7.90 | 59.11 |
| L2-7B | W4A4 | **QuEPT** | **6.33** | **7.86** | **61.62** |
| L3-8B | W4A4 | DuQuant | 8.56 | 11.98 | 65.05 |
| L3-8B | W4A4 | **QuEPT** | **8.25** | **11.67** | **67.04** |

### MLLM Results (LLaVA-OneVision-7B)

| Bits | Method | MMMU | OCRBench | TextVQA | VizWiz | SEED |
|------|--------|------|----------|---------|--------|------|
| W3A16 | MBQ | 42.0 | 61.1 | 73.3 | 60.7 | 66.4 |
| W3A16 | **QuEPT** | **44.6** | 60.6 | **74.1** | 60.3 | **71.6** |
| W4A8 | MBQ | 42.6 | 52.3 | 68.3 | 58.9 | 64.4 |
| W4A8 | **QuEPT** | **43.4** | **61.2** | **71.5** | **61.3** | **70.7** |

### Ablation Study

**MB-CLoRA Strategy Comparison (LLaMA2-7B Average Accuracy):**

| Strategy | W4A4 | W5A5 | W8A8 |
|----------|------|------|------|
| Full Sharing (Case 1) | 60.9 | 64.2 | 65.5 |
| Independent LoRA (Case 2) | 59.2 | 64.4 | 65.7 |
| **MB-CLoRA (Case 3)** | **61.6** | **64.5** | 65.6 |

**MB-ToMe Strategy Comparison (LLaMA2-7B):**

| Strategy | W4A4 | W6A6 | W8A8 |
|----------|------|------|------|
| Random Selection (Case 1) | 55.7 | 64.7 | 65.4 |
| Uniform Fusion (Case 2) | 60.2 | 65.2 | 65.5 |
| **Selective Merging (Case 3)** | **61.6** | **65.5** | **65.6** |

### Key Findings

1. MB-ToMe yields the largest gain at low bit-widths (W4A4), outperforming Case 1 by 5.9%, as token features are most unstable at low precision and selective retention of high-precision features is critical
2. The cascaded sharing in MB-CLoRA outperforms both fully independent and fully shared strategies at low bit-widths, because it allows lower bit-widths to "inherit" the optimization benefits of higher bit-widths
3. MAE loss outperforms MSE loss
4. In mixed-precision experiments, WikiText2 PPL is only 8.97 / 5.93 / 5.54 at average bit-widths of 2.25 / 3.00 / 4.00, respectively

## Highlights & Insights

1. **Unified Multimodal Framework**: The first multi-bit-width PTQ method validated uniformly across ViT, LLM, and MLLM
2. **Converting Conflict into Synergy**: The combination of MB-ToMe and MB-CLoRA addresses inter-bit-width competition from complementary perspectives—the former mitigates it in feature space, the latter coordinates it in parameter space
3. **Clever LoRA Placement**: Positioning LoRA inside the clipping operator ensures zero inference overhead, making deployment straightforward
4. **Seamless Transition to Mixed Precision**: Elastic quantization can be converted to mixed precision via per-layer sensitivity analysis combined with dynamic programming, without retraining
5. **High Training Efficiency**: Calibration takes only 17 minutes for ViT-S, 1/26 the time of PTMQ

## Limitations & Future Work

1. Outliers in LLMs are not explicitly handled; combining with outlier mitigation techniques such as SpinQuant could yield further improvements
2. Performance at extremely low bit-widths (e.g., W2) remains limited
3. The calibration set is small (128 samples); larger calibration sets may improve performance at the cost of increased calibration time
4. Hyperparameters $\lambda_1, \lambda_2, \lambda_3$ and top-p% in MB-ToMe require tuning

## Related Work & Insights

- **Any-Precision LLM (Park et al., 2024)**: Based on truncated bit-widths and incremental upsampling, but does not support activation quantization
- **MatQuant (Nair et al., 2025)**: Co-distillation regularization without consideration of inter-bit-width competition
- **SVDQuant (Li et al., 2025b)**: Uses LoRA to learn outlier branches, making the residual weights easier to quantize
- **QuaRot (Ashkboos et al., 2024)**: Rotation matrices for outlier elimination enabling 4-bit inference
- **PTMQ (Xu et al., 2024a)**: Multi-bit-width PTQ for CNNs/ViTs with high computational overhead

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](../../CVPR2026/model_compression/binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)
- [\[AAAI 2026\] EfficientFSL: Enhancing Few-Shot Classification via Query-Only Tuning in Vision Transformers](efficientfsl_enhancing_few-shot_classification_via_query-only_tuning_in_vision_t.md)
- [\[CVPR 2026\] Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](../../CVPR2026/model_compression/frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)
- [\[AAAI 2026\] First-Order Error Matters: Accurate Compensation for Quantized Large Language Models](first-order_error_matters_accurate_compensation_for_quantized_large_language_mod.md)
- [\[ICML 2026\] Multi-Adapter Representation Interventions via Energy Calibration](../../ICML2026/model_compression/multi-adapter_representation_interventions_via_energy_calibration.md)

</div>

<!-- RELATED:END -->
