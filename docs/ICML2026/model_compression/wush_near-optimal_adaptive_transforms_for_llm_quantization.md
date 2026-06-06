---
title: >-
  [Paper Note] WUSH: Near-Optimal Adaptive Transforms for LLM Quantization
description: >-
  [ICML 2026][Model Compression][W4A4 Quantization] WUSH derives closed-form, data-adaptive blockwise linear transforms for LLM weight-activation low-bit quantization…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "W4A4 Quantization"
  - "Adaptive Transforms"
  - "Hadamard"
  - "MXFP4"
  - "GPTQ"
date: 2026-05-08
content_hash: 8b0f69aa5fb61f3e
---

# WUSH: Near-Optimal Adaptive Transforms for LLM Quantization

**Conference**: ICML 2026  
**arXiv**: [2512.00956](https://arxiv.org/abs/2512.00956)  
**Code**: https://github.com/IST-DASLab/WUSH  
**Area**: Model Compression / LLM Quantization  
**Keywords**: W4A4 Quantization, Adaptive Transforms, Hadamard, MXFP4, GPTQ  

## TL;DR
WUSH derives closed-form, data-adaptive blockwise linear transforms for LLM weight-activation low-bit quantization, combining the uniform diffusion capability of Hadamard with second-order statistics of weights/activations. It significantly improves accuracy in W4A4—especially MXFP4—with minimal sacrifice to FP4 kernel throughput.

## Background & Motivation
**Background**: In LLM deployment, weight and activation quantization are standard methods to reduce memory footprint and increase throughput. For W4A4 schemes, where both weights and activations are compressed to 4 bits, mainstream approaches involve not only choosing quantizers like RTN or GPTQ but also applying channel scaling or rotations before quantization to reduce the dominance of outliers on the AbsMax scale.

**Limitations of Prior Work**: Transforms like Hadamard rotation, QuaRot, and MR-GPTQ are effective in practice but are typically fixed and data-independent. While they spread outlier energy, they do not address what transform is optimal for given weight and activation statistics. Methods like SpinQuant and FlatQuant attempt to learn transforms through iterative optimization, which incurs higher calibration and engineering costs and may not be suitable for fast per-token activation quantization.

**Key Challenge**: Quantization error depends on the output error determined jointly by weights and activations, rather than simply making weights or activations more uniform in isolation. The ideal transform must adapt to the second-order statistics of each block and be efficiently fused into activation transforms and quantization kernels during inference; if the transform is too complex, accuracy gains are offset by runtime overhead.

**Goal**: The authors aim to derive a closed-form, near-optimal transform for blockwise RTN AbsMax quantizers covering both FP and INT low-bit formats, which can naturally integrate with GPTQ. The method needs to improve W4A4 accuracy on real-world LLMs while preserving the throughput advantages of FP4 MatMul.

**Key Insight**: Starting from the output error of a single quantization block, weight columns and activation columns are treated as samples from a distribution. Their typical shapes are described using second moments, and a linear transform is sought to minimize the quantization error of the transformed weight and activation. Hadamard is no longer the sole focus but acts as a uniform diffusion backbone within an optimal construction.

**Core Idea**: First, construct a data-adaptive non-orthogonal transform using blockwise second moments of weights and activations, then wrap it with a Hadamard backbone. This ensures the transform possesses both statistical optimality and energy diffusion properties friendly to AbsMax quantization.

## Method

### Overall Architecture
WUSH segments each linear layer's input channels into blocks according to quantization groups. During offline calibration, the method collects weight and activation second moments for each block and solves for a pair of reciprocal transpose transforms in closed form: the weight side uses $T_{xvsh}$ for pre-transformation and quantization, while the activation side uses $T_{wush}$ during inference before quantization. Since they satisfy $T_{xvsh}=T_{wush}^{-\top}$, the inner product remains consistent before quantization; after quantization, the goal is to minimize output error.

During inference, the weight-side transform is already absorbed into the pre-quantized weights. Online, only the WUSH transform and quantization are applied to the activation blocks. The authors implement a fused WUSH + Quant kernel and store the $G\times G$ matrix for each block in a layout suitable for CUTLASS GEMM, enabling multiple small matrix transforms to be fused as efficiently as Hadamard + quantization.

### Key Designs
1. **Defining the blockwise joint quantization objective from output error**:

	- **Function**: Avoids optimizing weight or activation error in isolation by using the $W^\top X$ error—which directly affects model output—as the objective.
	- **Mechanism**: For each input channel block, let the weight block be $W_{(i)}$ and the calibration activation block be $X_{(i)}$. A transform $T_W, T_X$ is selected to minimize $\|q(T_WW)^\top q(T_XX)-W^\top X\|_F^2$. The paper further approximates the full-layer error as a sum of blockwise losses, allowing each block to be solved independently.
	- **Design Motivation**: The error in AbsMax group quantization is determined by the block maximum, distribution shape, and weight/activation interaction. Modeling the output error explains why fixed Hadamard transforms are sometimes effective and sometimes insufficient.

2. **WUSH Closed-form Construction: Hadamard + Second Moments + SVD**:

	- **Function**: Generates data-adaptive, non-orthogonal, near-optimal transform matrices for each block.
	- **Mechanism**: Perform Cholesky decomposition on $d_{out}^{-1}WW^\top$ and $d_{batch}^{-1}XX^\top$ to obtain $W'$ and $X'$, then apply SVD to $W'^\top X'$ to get $U, S, V$. The activation-side transform is $T_{wush}=HS^{-1/2}U^\top W'^\top$, and the weight-side is $T_{xvsh}=HS^{-1/2}V^\top X'^\top$, which are reciprocal transposes.
	- **Design Motivation**: The $S^{-1/2}$ and second-moment terms adjust the coordinate system based on real statistics, while Hadamard uniformly diffuses energy within the group to prevent non-orthogonal transforms from amplifying single coordinates and worsening the scale in INT/AbsMax scenarios.

3. **Engineering Integration with RTN/GPTQ and Fused GPU Kernels**:

	- **Function**: Transitions the theoretical transform to LLM W4A4 inference rather than just offline error analysis.
	- **Mechanism**: In RTN, WUSH is computed in parallel for each block to pre-quantize weights. In GPTQ, WUSH uses the same activation second-order information as the GPTQ Hessian, interleaving transformed weight calculations with GPTQ block updates and error propagation. On the online stage, only the activation-side transform is kept; WUSH + Quant is mapped to CUTLASS-style small GEMMs followed by FP4 MatMul.
	- **Design Motivation**: If every block has an independent matrix, a naive implementation would be much slower than Hadamard. The authors use storage layouts and fused kernels to minimize overhead, allowing WUSH to succeed in both accuracy and throughput.

### Loss & Training
WUSH is a post-training quantization method and does not train model parameters. During the offline phase, calibration data is used to compute weight/activation second moments, and each linear layer is calibrated sequentially; after quantizing one layer, the calibration activations are forwarded to the next. The RTN version uses direct round-to-nearest; the GPTQ version follows GPTQ's Hessian and error propagation but applies the WUSH transform to the current block first. The additional computational cost mainly involves blockwise second moments, Cholesky, and SVD; as the block size is much smaller than the channel count, overall calibration cost is close to standard GPTQ.

## Key Experimental Results

### Main Results
Main results focus on Llama-3.1-8B-Instruct using the W4A4 LM Evaluation Harness. WUSH shows slight improvements on NVFP4 and more significant gains on the more challenging MXFP4 format, clearly outperforming Hadamard/MR-GPTQ.

| Format | Method | MMLU-CoT | GSM8K | HellaSwag | WinoGrande | Average | Recovery |
|--------|--------|----------|-------|-----------|------------|---------|----------|
| BF16 | Original | 72.76 | 85.06 | 80.01 | 77.90 | 78.93 | 100.0 |
| NVFP4 | RTN-I | 68.26 | 78.39 | 78.15 | 74.11 | 74.73 | 94.67 |
| NVFP4 | GPTQ-H / MR-GPTQ | 69.12 | 80.80 | 78.17 | 75.24 | 75.84 | 96.08 |
| NVFP4 | GPTQ-WUSH | 69.69 | 80.11 | 78.52 | 76.09 | 76.10 | 96.40 |
| MXFP4 | RTN-I | 62.21 | 67.85 | 73.99 | 73.24 | 69.32 | 87.83 |
| MXFP4 | RTN-H | 62.38 | 72.48 | 75.29 | 71.67 | 70.45 | 89.26 |
| MXFP4 | RTN-WUSH | 66.85 | 75.16 | 77.28 | 73.56 | 73.21 | 92.75 |
| MXFP4 | GPTQ-H / MR-GPTQ | 67.19 | 75.70 | 76.91 | 74.80 | 73.65 | 93.31 |
| MXFP4 | GPTQ-WUSH | 67.79 | 77.41 | 77.44 | 74.78 | 74.35 | 94.20 |

### Ablation Study
Layerwise quantization loss directly validates WUSH components. The following table showcases average trends for the 18th block of Qwen3-8B with FineWeb-Edu calibration and RTN loss: WUSH significantly outperforms identity (I), random rotation, Hadamard (H), and WUS (without Hadamard) in MXFP4 and INT4.

| Format | Transform | Q | K | V | O | G | U | D | Conclusion |
|--------|-----------|---|---|---|---|---|---|---|------------|
| MXFP4 | I | 11.1 | 12.0 | 10.7 | 4.35 | 7.10 | 6.56 | 5.47 | Outliers increase error |
| MXFP4 | H | 7.24 | 7.20 | 8.60 | 3.79 | 5.45 | 5.61 | 3.90 | Fixed Hadamard helps |
| MXFP4 | WUS | 6.27 | 7.22 | 4.05 | 3.57 | 5.76 | 4.75 | 4.46 | Adaptive but lacks diffusion |
| MXFP4 | WUSH | 3.34 | 3.34 | 3.30 | 2.76 | 4.49 | 4.39 | 3.39 | Lowest error |
| INT4 | H | 5.57 | 5.55 | 6.80 | 2.86 | 4.09 | 4.25 | 3.03 | Hadamard stabilizes AbsMax scale |
| INT4 | WUS | 213.0 | 142.0 | 10.7 | 4.54 | 50.2 | 7.42 | 13.1 | Non-orthogonal terms amplify coords |
| INT4 | WUSH | 2.39 | 2.43 | 2.54 | 2.10 | 3.43 | 3.43 | 2.55 | Hadamard component essential |

| System/Robustness Analysis | Value | Description |
|----------------------------|-------|-------------|
| Max per-layer speedup (WUSH+Quant+FP4 MatMul) | 5.8x vs BF16 | Near hardware gains of FP4 MatMul |
| Avg throughput diff vs H+Quant+FP4 MatMul | ~1.3% | Block-independent matrices don't lag kernel |
| Llama-3.1-8B RTN Preprocessing cost | 19 mins / 19 GB H100 | Scalability comparable to GPTQ |
| Qwen3-32B RTN Preprocessing cost | 38 mins / 40 GB B200 | Scalable to large models |
| Llama-3.1-8B WUSH transform storage overhead | MXFP4 1.4%, NVFP4 0.7% | Minimal compared to full checkpoint |
| Qwen3-8B MXFP4 calibration set sensitivity | FineWeb 74.91 / C4 75.57 | Not dependent on a single calibration set |

### Key Findings
- WUSH's primary gains are concentrated in harder FP4 formats like MXFP4. On Llama-3.1-8B, MXFP4 RTN-WUSH scored 2.76 points higher than RTN-H, and GPTQ-WUSH was 0.70 higher than MR-GPTQ on average.
- WUS alone matches WUSH on NVFP4 but causes catastrophic outlier amplification on INT4, proving the Hadamard backbone is an essential stabilizer for controlling AbsMax scale, not just a decorative element.
- Fused kernel results are critical: WUSH uses different matrices per block, which is theoretically harder to implement efficiently, yet the measured throughput difference against Hadamard fused kernels is only 1.3%, ensuring accuracy gains are not lost to engineering overhead.
- Calibration set stability and KL divergence results support that the method does not merely overfit to specific benchmarks. WUSH's KL divergence on Qwen3-8B is lower than Hadamard's, and average accuracies for FineWeb/C4 calibration remain close.

## Highlights & Insights
- The paper clarifies why Hadamard is useful: it is not just random rotation but serves the role of distributing energy uniformly across group dimensions within the WUSH construction. This insight is more transferable than simply reporting benchmarks.
- WUSH's non-orthogonal adaptive component stems from weight and activation second-order statistics, allowing it to address joint W4A4 errors rather than just weight-only quantization. For activation quantization, this modeling granularity is closer to actual deployment issues.
- The method is compatible with both RTN and GPTQ, covering both "fast direct quantization" and "second-order corrected quantization" paths. This makes it more attractive to engineering practitioners.
- The GPU kernel component significantly completes the paper. While block-independent matrices typically raise throughput concerns, the authors demonstrate speeds near Hadamard by using specific layouts and CUTLASS GEMM mapping.

## Limitations & Future Work
- WUSH still relies on calibration data statistics. Although sensitivity experiments on FineWeb/C4 are promising, further validation of activation second moment stability is needed under strong domain shifts, long contexts, or specialized tool-calling distributions.
- The paper focuses on W4A4 inference for dense linear layers and has not fully discussed adaptations for complex modules like MoE routing, KV-cache quantization, or attention score quantization.
- The transform matrices are block-specific; while storage overhead is low, it increases implementation complexity. Integration into broader inference frameworks will require mature kernels, format support, and quantization export toolchains.
- Theoretical derivations rely on moderate assumptions and approximations, such as block loss independence, stochastic quantization surrogates, and second moments representing typical distributions, which may not hold in extreme heavy-tail or strongly correlated blocks.

## Related Work & Insights
- **vs SmoothQuant / AWQ**: These methods primarily balance weight and activation dynamic ranges via channel scaling. WUSH uses full blockwise linear transforms, handling correlations between dimensions rather than just per-channel scaling.
- **vs QuaRot / Hadamard-based methods**: QuaRot and MR-GPTQ rely on fixed rotations or Hadamard, which is simple and efficient but data-independent. WUSH retains Hadamard's hardware friendliness while introducing second-order statistical adaptation.
- **vs SpinQuant / FlatQuant**: Learning-based transforms adapt to data but require iterative optimization. WUSH provides a closed-form solution with a calibration process similar to GPTQ, offering better cost and controllability for mass deployment.
- **vs GPTQ**: GPTQ optimizes error propagation for weight quantization. WUSH can be embedded into GPTQ, performing GPTQ on transformed blocks and utilizing the same Hessian information for activation-side transforms.
- **Insight**: For low-bit LLMs, the key forward path may not be inventing new quantizers but co-designing data statistics, format characteristics, and kernel shapes. WUSH serves as a strong exemplar of this math-to-hardware closed-loop design.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Closed-form derivation of adaptive block transforms with a clear explanation of Hadamard's role.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple models, formats, RTN/GPTQ, kernels, and calibration stability; more real-world end-to-end latency data would be beneficial.
- Writing Quality: ⭐⭐⭐⭐☆ Clear connections between theory, algorithm, kernel, and experiments, though the high derivation density requires prior quantization background.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for W4A4 LLM deployment, especially for the actual landing of new FP4 formats like MXFP/NVFP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs](neuqi_near-optimal_uniform_quantization_parameter_initialization_for_low-bit_llm.md)
- [\[ICML 2026\] OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization](osaq_outlier_self-absorption_for_accurate_low-bit_llm_quantization.md)
- [\[ICML 2026\] ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation](respinquant_efficient_layer-wise_llm_quantization_via_subspace_residual_rotation.md)
- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](../../ICLR2026/model_compression/dataset_distillation_as_pushforward_optimal_quantization.md)
- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](../../ICLR2026/model_compression/compute-optimal_quantization-aware_training.md)

</div>

<!-- RELATED:END -->
