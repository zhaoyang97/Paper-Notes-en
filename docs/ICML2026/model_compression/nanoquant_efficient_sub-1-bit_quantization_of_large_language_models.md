---
title: >-
  [Paper Note] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models
description: >-
  [ICML 2026][Model Compression][Post-Training Quantization] NanoQuant reformulates weight quantization as a "low-rank binary decomposition" problem. It employs Hessian-aware ADMM for the precise initialization of $\pm 1$…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Post-Training Quantization"
  - "Sub-1-Bit"
  - "Low-Rank Binary Decomposition"
  - "ADMM"
  - "LLM Deployment"
date: 2026-05-08
content_hash: 01e0b0b3914527cf
---

# NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.06694](https://arxiv.org/abs/2602.06694)  
**Code**: Not yet public  
**Area**: Model Compression / LLM Quantization  
**Keywords**: Post-Training Quantization, Sub-1-Bit, Low-Rank Binary Decomposition, ADMM, LLM Deployment  

## TL;DR
NanoQuant reformulates weight quantization as a "low-rank binary decomposition" problem. It employs Hessian-aware ADMM for the precise initialization of $\pm 1$ factors and floating-point scales, followed by block-level STE reconstruction and global scale KL calibration. Using only 0.26M calibration tokens on a single H100 GPU, it enables PTQ to compress LLMs to 1-bit or even sub-1-bit for the first time. Notably, it reduces Llama2-70B from 138 GB to 5.35 GB, allowing it to run on an 8 GB consumer-grade GPU.

## Background & Motivation

**Background**: Weight quantization has become a standard for LLM deployment. Post-Training Quantization (PTQ) methods such as GPTQ, AWQ, and QuIP can stably achieve 2-bit quantization. Recent binary PTQ methods (BiLLM, ARB-LLM, STBLLM, HBLLM) attempt 1-bit, while binary Quantization-Aware Training (QAT) like OneBit, LittleBit, and DBF have achieved sub-1-bit levels.

**Limitations of Prior Work**: Binary PTQ typically utilizes an "in-place binarization + full-precision scale" structure $\mathbf{W}\approx\alpha\mathbf{B}_{\pm 1}$. This imposes a structural lower bound of at least 1 bit per parameter. When accounting for group masks and scale metadata, the effective bits-per-weight (BPW) often requires 2.5–4 bits to maintain acceptable PPL. Conversely, sub-1-bit binary QAT requires hundreds of millions of tokens and days of multi-GPU training, making it impractical for 70B models.

**Key Challenge**: PTQ is computationally efficient but limited by its representation structure, whereas QAT is flexible but too costly to scale to 70B models. The fundamental question is whether a more compact representation than direct binarization can be discovered within a PTQ budget.

**Goal**: This work addresses three sub-problems: (1) Finding a binary representation structurally capable of sub-1-bit compression; (2) Precisely initializing this representation using a small calibration set; (3) Enabling the quantization of 70B models on a single GPU.

**Key Insight**: The authors adopt the "low-rank binary decomposition" representation from LittleBit/DBF, where weights are decomposed into two $\pm 1$ low-rank matrices and two floating-point scales. Storage complexity is controlled by $r/d$, which can be lower than 1 bit. While QAT learns this decomposition end-to-end, the authors propose a two-stage "precise initialization + block-level reconstruction" method to approximate QAT accuracy within PTQ constraints.

**Core Idea**: Sub-1-bit PTQ is reformulated as "Hessian-weighted low-rank binary matrix decomposition + block-level STE fine-tuning + global scale KL calibration." ADMM is used to decouple combinatorial optimization from continuous relaxation, bypassing the NP-hard nature of binary optimization.

## Method

### Overall Architecture
NanoQuant decomposes each linear layer weight $\mathbf{W}\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$ into $\widehat{\mathbf{W}}=\mathbf{s}_1\odot(\mathbf{U}_{\pm 1}\mathbf{V}_{\pm 1}^\top)\odot\mathbf{s}_2^\top$, where $\mathbf{U}_{\pm 1}\in\{-1,+1\}^{d_\text{out}\times r}$, $\mathbf{V}_{\pm 1}\in\{-1,+1\}^{d_\text{in}\times r}$, and $\mathbf{s}_1,\mathbf{s}_2$ are full-precision channel-wise scale vectors. The pipeline consists of three stages: **(1) Global Calibration**: 128 samples are fed into a FP teacher to calculate K-FAC-style input/output diagonal pre-conditioners $\widetilde{\mathbf{D}}_\text{in},\widetilde{\mathbf{D}}_\text{out}$ for each layer. **(2) Block-level Reconstruction**: Proceeding per Transformer block, FP weights are adjusted to compensate for prior quantization errors, followed by LB-ADMM initialization of $\mathbf{U},\mathbf{V},\mathbf{s}_1,\mathbf{s}_2$. STE is then used for joint fine-tuning of latent variables and scales before freezing signs. **(3) Model Reconstruction**: All binary matrices are frozen, and only the global floating-point scale set $\mathbf{S}_\text{global}$ is optimized using KL divergence to align the quantized model's logits with the FP teacher.

### Key Designs

1.  **Low-Rank Binary Decomposition + Hessian-aware Pre-conditioning**:
    - **Function**: Uses a representation with higher expressivity than "in-place binarization" for sub-1-bit targets.
    - **Mechanism**: The objective is reformulated as $\mathcal{L}(\widehat{\mathbf{W}})\approx\|\widetilde{\mathbf{D}}_\text{out}(\mathbf{W}-\widehat{\mathbf{W}})\widetilde{\mathbf{D}}_\text{in}\|_F^2$, which is equivalent to a low-rank binary approximation under an elliptical norm defined by activation/gradient statistics. Pre-conditioners are stabilized using shrinkage estimation $[\widetilde{\mathbf{D}}]_{ii}\leftarrow(1-\gamma)[\mathbf{D}]_{ii}+\gamma\,\mathrm{mean}(\mathbf{D})$.
    - **Design Motivation**: Direct minimization of Euclidean error is sensitive to small calibration sets; Hessian weighting prioritizes directions that significantly impact downstream loss.

2.  **Latent-Binary ADMM (LB-ADMM) Initialization**:
    - **Function**: Solves the NP-hard low-rank $\pm 1$ decomposition within a PTQ budget to provide a high-quality starting point for STE.
    - **Mechanism**: The problem is formulated as $\min_{\mathbf{U},\mathbf{V},\mathbf{Z}_U,\mathbf{Z}_V}\tfrac{1}{2}\|\widetilde{\mathbf{W}}_\text{target}-\mathbf{U}\mathbf{V}^\top\|_F^2+\tfrac{\lambda}{2}(\|\mathbf{U}\|_F^2+\|\mathbf{V}\|_F^2)$ s.t. $\mathbf{U}=\mathbf{Z}_U,\mathbf{V}=\mathbf{Z}_V$. It alternates between a linear system for continuous factors (using Cholesky decomposition), Sign-Value Independent Decomposition (SVID) for auxiliary variables $\mathbf{Z}$, and dual variable updates.
    - **Design Motivation**: Decoupling continuous reconstruction from binary constraints allows the continuous solution to be pulled toward the binary manifold. LB-ADMM achieves a PPL of 20.06 at 0.8 bit, compared to 30.27 for DBF-ADMM and 167.73 for Dual-SVID.

3.  **Block-level STE Fine-tuning + Scale-only Model-level KL Calibration**:
    - **Function**: Transforms local initialization into a globally aligned model while restricting backpropagation costs.
    - **Mechanism**: At the block level, the Straight-Through Estimator (STE) jointly optimizes latent variables and scales to minimize $\|\mathcal{B}(\mathbf{X}_\text{in})-\widehat{\mathcal{B}}(\mathbf{X}_\text{in};\mathrm{sign}(\mathcal{U}),\mathrm{sign}(\mathcal{V}),\mathbf{s}_1,\mathbf{s}_2)\|_F^2$. At the model level, only global scales $\mathbf{S}_\text{global}$ are optimized using KL divergence.
    - **Design Motivation**: Unlike schemes that require full-weight gradients (which are impractical for 70B models), this approach keeps gradients localized or limited to scale vectors, enabling 70B quantization on a single H100.

### Loss & Training
Block-level objectives use MSE, while model-level objectives use KL divergence. Optimization steps $(T_\text{pre},T_\text{post},T_\text{glob})$ are set independently. Four core hyperparameters—ADMM iterations $K$, penalty $\rho$, ridge regularization $\lambda$, and convergence threshold $\epsilon$—are utilized. The calibration set contains 128 WikiText-2 samples (approx. 0.26M tokens) with a sequence length of 2048.

## Key Experimental Results

### Main Results
Evaluations cover Llama-2/3, Gemma-3, Qwen-3, and Rnj-1 (0.6B–70B).

| Model / Bitrate | Method | Effective BPW | WikiText-2 PPL ↓ | Notes |
|---|---|---|---|---|
| Llama-2-7B / 1 bit | NanoQuant | 1.00 | 10.34 | Single H100, 0.26M tokens |
| Llama-2-7B / 1 bit | HBLLM_R | 3.25 | 7.60 | 3.25× more storage |
| Llama-2-7B / 1 bit | BiLLM | 2.88 | 19.87 | Worse than NanoQuant |
| Llama-2-70B / 1 bit | NanoQuant | 1.00 | 6.52 | 138 GB → 5.35 GB |
| Llama-3-8B / 0.8 bit | NanoQuant | 0.80 | 18.16 | First sub-1 bit PTQ |
| Llama-3-8B / 0.55 bit | NanoQuant | 0.55 | 25.69 | Extreme compression |

### Ablation Study
| Configuration | PPL ↓ | Zero-shot ↑ | Description |
|---|---|---|---|
| LB-ADMM Initialization only | 206.03 | 36.89 | Failure without reconstruction |
| + Error Mitigation | 15.07 | 46.40 | Offsets accumulated error |
| + Factorized Refinement | 13.58 | 46.75 | STE fine-tuning |
| Full (+ Model-level KL) | 12.47 | 48.94 | Qwen3-8B 0.8 bit |

### Key Findings
- Initialization is the critical factor for sub-1-bit PTQ. Replacing the initialization alone reduced PPL from 167 to 20, suggesting that solving combinatorial issues during initialization is more vital than the subsequent STE.
- All four pipeline modules are essential. Error mitigation is particularly crucial to address the severe error accumulation inherent in binarization.
- At equivalent bitrates, NanoQuant outperforms methods like HBLLM/STBLLM/BiLLM. For Llama-2-7B, NanoQuant at 1.00 bit (PPL 10.34) beats BiLLM at 2.88 bit (PPL 19.87).
- In deployment, Llama-3.2-3B achieves 3.7× throughput and 5.4× memory efficiency on an RTX 3050 8GB compared to BF16. A 70B model runs at 20.11 tok/s on the same 8GB consumer card.

## Highlights & Insights
- **Structural Novelty**: Shifting from "scale × binary matrix" to "scale × product of binary factors" breaks the 1-bit structural floor without requiring codebooks or sparsity.
- **Effective ADMM Usage**: Decoupling combinatorial constraints via dual variables and continuous constraints via linear systems provides a template for handling non-convex discrete constraints in other areas like pruning or codebook quantization.
- **Budget Distribution**: Confining expensive STE to the block level and optimizing only scale vectors globally is a key engineering philosophy for scaling quantization to 70B models.
- **Pre-conditioner Shrinkage**: Using convex combinations of the K-FAC diagonal and the mean improves stability for models with sharp distributions, such as Gemma.

## Limitations & Future Work
- PPL remains significantly higher than the BF16 baseline (e.g., 5.47 vs 10.34 for Llama-2-7B). Further evaluation on long-context and complex reasoning tasks (GSM8K, MMLU) is needed.
- "QAT baselines" were reproduced locally; the accuracy boundaries of original LittleBit/DBF with larger token budgets were not fully aligned.
- The 128 WikiText-2 samples may be biased for multilingual or code-heavy models. Shrinkage coefficients $\gamma$ currently require manual tuning per model family.
- Theoretical convergence of ADMM in sub-1-bit scenarios is not guaranteed; the importance of Hessian-aware initialization remains an empirical observation.

## Related Work & Insights
- **vs. Binary PTQ (BiLLM, etc.)**: These methods use in-place binarization and group masks, stalling at 2.5–4 bits. NanoQuant's low-rank decomposition bypasses these structural limits.
- **vs. Binary QAT (OneBit, DBF, etc.)**: These rely on massive end-to-end training. NanoQuant reduces data/compute requirements by 100–1000× while maintaining competitive accuracy and enabling 70B scale quantization.
- **vs. Integer PTQ (GPTQ, QuIP)**: Integer methods are limited by discrete bit-widths. NanoQuant's BPW is continuously adjustable via rank $r$, allowing for a true Pareto frontier.
- **vs. QMoE / BTC-LLM**: These are architecture-specific or require codebooks. NanoQuant is a general-purpose sub-1-bit PTQ.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs](neuqi_near-optimal_uniform_quantization_parameter_initialization_for_low-bit_llm.md)
- [\[ICML 2026\] Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models](bounded_hyperbolic_tangent_a_stable_and_efficient_alternative_to_pre-layer_norma.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] LFQ: Logit-aware Final-block Quantization for Boosting the Generation Quality of Low-Bit Quantized LLMs](lfq_logit-aware_final-block_quantization_for_boosting_the_generation_quality_of_.md)
- [\[ICML 2026\] The Shape of Addition: Geometric Structures of Arithmetic in Large Language Models](the_shape_of_addition_geometric_structures_of_arithmetic_in_large_language_model.md)

</div>

<!-- RELATED:END -->
