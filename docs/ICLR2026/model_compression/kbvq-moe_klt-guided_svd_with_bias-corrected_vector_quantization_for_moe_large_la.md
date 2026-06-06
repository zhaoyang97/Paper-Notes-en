---
title: >-
  [Paper Note] KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models
description: >-
  [ICLR 2026][Model Compression][MoE quantization] This paper proposes KBVQ-MoE, the first vector quantization framework specifically designed for MoE architectures. It eliminates inter-expert redundancy sharing (IDRE) via…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "MoE quantization"
  - "vector quantization"
  - "KLT transform"
  - "SVD redundancy elimination"
  - "bias correction"
date: 2026-05-08
content_hash: 4528be3a9a7a4134
---

# KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

**Conference**: ICLR 2026  
**arXiv**: [2602.11184](https://arxiv.org/abs/2602.11184)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: MoE quantization, vector quantization, KLT transform, SVD redundancy elimination, bias correction

## TL;DR
This paper proposes KBVQ-MoE, the first vector quantization framework specifically designed for MoE architectures. It eliminates inter-expert redundancy sharing (IDRE) via KLT-guided SVD and stabilizes outputs through bias-corrected output stabilization (BCOS), achieving 10%+ accuracy improvement over existing methods at 2-bit quantization.

## Background & Motivation
MoE models (e.g., Qwen3-30B-A3B, Mixtral-8x7B) achieve a performance–efficiency balance through sparse expert activation, yet their massive parameter counts make deployment challenging (Qwen3-80B-A3B requires 160GB+ GPU memory).

Vector quantization (VQ) has demonstrated strong potential for ultra-low-bit compression of dense LLMs—mapping weight vectors to the nearest codeword in a discrete codebook. However, directly applying VQ to MoE encounters two critical obstacles:

**Inter-expert redundant representations**: MoE experts frequently capture similar feature patterns; VQ repeatedly quantizes similar representations for each expert, leading to inefficient utilization of limited codebook capacity.

**Accumulated output bias amplified by expert aggregation**: Quantization errors accumulate across layers to produce bias; the weighted aggregation of multiple experts in MoE further amplifies this bias, causing more severe distributional drift than in dense LLMs.

**Core Idea**: KLT and SVD are first applied to extract shared weight structures across experts (retained in full precision), VQ is applied only to the expert-specific residuals, and channel-wise affine correction is then used to repair distributional drift.

## Method

### Overall Architecture
$W \xrightarrow[\text{KLT+SVD}]{\text{IDRE}} \underbrace{W_{\text{share}}}_{\text{shared component}} + \underbrace{W_{\text{quant}}}_{\text{specific component}} \xrightarrow[\text{Bias Correction}]{\text{BCOS}} W_{\text{share}} + W_{\text{quant}}^{\text{VQ}} + (s, b)$

### Key Designs

1. **Input-Driven Redundancy Elimination (IDRE)**:

    - **Function**: Extracts shared weight structures across experts and retains them in full precision.
    - **Mechanism** (3 steps):
        - *Step 1: KLT decomposition of input activations*: Compute the input covariance matrix $C_X = \frac{1}{B-1}X^TX$ and perform eigendecomposition to obtain energy-sorted orthogonal bases $U_X = U_{\text{KLT}} \Lambda_{\text{KLT}}^{1/2}$.
        - *Step 2: Map weights into the input-coherent space*: $\hat{W} = WU_X$, orienting weight analysis along the principal directions of the input.
        - *Step 3: Extract shared structure*: Concatenate the transformed weights of all $n$ experts into a unified representation $\bar{W} \in \mathbb{R}^{(n \cdot oc) \times ic}$, then apply SVD to $\bar{W}$ to extract the shared structure corresponding to the top-$k$ singular values. The shared directions are mapped back to the original space: $U_{\text{share}} = U^T \cdot U_X^{-1}$.
    - **Design Motivation**: KLT ensures that redundancy extraction is guided by input statistics rather than pure weight-space decomposition; SVD operating on the unified representation handles all experts simultaneously. The truncation rank $k$ is set to $1/128$ of the full rank, incurring only a 0.12% parameter overhead.

2. **Bias-Corrected Output Stabilization (BCOS)**:

    - **Function**: Repairs output distributional drift introduced by VQ quantization.
    - **Mechanism**: After applying VQ to the expert-specific weights $W_{\text{quant}}$, a channel-wise affine transform corrects the output:
      $\mathbf{y}_{\text{corr}} = (s+1) \odot (W_{\text{VQ}}x) + b$
      where $s_j \approx \frac{\sigma_{y_j}}{\sigma_{\hat{y}_j}} - 1$ and $b_j = \mu_{y_j} - (1+s_j)\mu_{\hat{y}_j}$, aligning the mean and variance of the quantized output with those of the full-precision output.
    - **Design Motivation**: Only $2 \cdot oc$ additional parameters per layer are required, making computational and storage overhead negligible. The optimal parameters under the MMSE criterion admit a closed-form solution.

3. **Vector Quantization Details**:

    - Vector length is set to 4; k-means++ initialization with 100 iterations is used.
    - Calibration data: 256 samples from Red Pajama with sequence length 4096.

### Loss & Training
Pure post-training quantization (PTQ) — no retraining required. All experiments are conducted on an NVIDIA RTX A6000.

## Key Experimental Results

### Main Results (Multiple Models and Bit-widths)

| Model | Bits | Method | PPL (↓) | Avg. Accuracy (↑) |
|------|------|------|--------|-------------|
| Qwen1.5-MoE-A2.7B | FP16 | — | 7.22 | 68.07 |
| Qwen1.5-MoE-A2.7B | 3-bit | VQ | 11.47 | 55.94 |
| Qwen1.5-MoE-A2.7B | 3-bit | GPTQ | 7.58 | 66.36 |
| Qwen1.5-MoE-A2.7B | 3-bit | **KBVQ-MoE** | **7.74** | **67.99** |
| Qwen3-30B-A3B | 2-bit | VQ | 115.30 | 30.61 |
| Qwen3-30B-A3B | 2-bit | **KBVQ-MoE** | **11.87** | **63.37** |
| Mixtral-8x7B | 3-bit | GPTQ | 4.17 | 77.43 |
| Mixtral-8x7B | 3-bit | **KBVQ-MoE** | **4.07** | **78.35** |

### Ablation Study (Qwen3-30B-A3B, 3-bit)

| IDRE | BCOS | PPL | ARC-E | ARC-C | HellaSwag |
|------|------|-----|-------|-------|-----------|
| ✗ | ✗ | 18.72 | 57.83 | 40.87 | 63.23 |
| ✓ | ✗ | 11.67 | 71.35 | 50.55 | 73.51 |
| ✗ | ✓ | 14.32 | 65.49 | 47.33 | 68.37 |
| ✓ | ✓ | **9.26** | **—** | **—** | **—** |

### Key Findings
- Qwen1.5-MoE at 3-bit achieves 67.99% accuracy, nearly matching the FP16 baseline of 68.07%—a loss of only 0.08%.
- For Qwen3-30B-A3B at 2-bit, KBVQ-MoE reduces PPL by 6 points and improves accuracy by 10%+ compared to vanilla VQ.
- After IDRE eliminates redundancy, inter-expert output similarity decreases substantially (verified via comparison figures).
- BCOS effectively repairs distributional drift—corrected channel means and variances align precisely with full-precision values.
- IDRE contributes more than BCOS (PPL reduction of 7 vs. 4.4), yet the two components together achieve the best results.

## Highlights & Insights
- This work is the first to systematically address VQ-specific challenges in MoE architectures—redundancy waste and bias amplification.
- The KLT-guided SVD design is elegant: aligning the weight space with input statistics makes redundancy extraction more precise.
- The closed-form solution of BCOS is simple and practical, requiring only calibration data statistics without additional training.
- Usable performance is maintained even at extreme bit-widths (2-bit), indicating that the method's compression ceiling is high.

## Limitations & Future Work
- Retaining the shared structure in full precision increases storage, and the choice of truncation rank $k$ may require per-layer tuning.
- KLT assumes a stationary input distribution; dynamic inputs may yield suboptimal KLT bases.
- Evaluation is limited to the inference (PTQ) setting; combining with QAT could further improve performance.
- Scalability to larger MoE models (e.g., Qwen3-80B-A3B) has not been verified.

## Related Work & Insights
- **vs. GPTQ/MoEQuant**: Scalar quantization methods perform poorly at ≤3 bits; KBVQ-MoE exploits the structural advantages of VQ.
- **vs. VPTQ/AQLM**: General-purpose VQ methods do not account for inter-expert redundancy in MoE, resulting in poor performance when applied directly.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of KLT + SVD + VQ + affine correction is novel, though each component has precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 4 models, 2/3-bit settings, 7 datasets, and a complete ablation study.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is detailed and mathematical derivations are clear.
- **Value**: ⭐⭐⭐⭐⭐ The first MoE-specific VQ framework; near-lossless 3-bit quantization offers high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](../../ICML2026/model_compression/rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](steering_moe_llms_via_expert_deactivation.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[ICLR 2026\] MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE](mone_replacing_redundant_experts_with_lightweight_novices_for_structured_pruning.md)
- [\[ICLR 2026\] Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)

</div>

<!-- RELATED:END -->
