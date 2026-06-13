---
title: >-
  [Paper Note] RT-Lynx: Putting the GEMM Sparsity In a Right Way for Diffusion Models
description: >-
  [ICML 2026][Image Generation][Activation Sparsity] The authors discover that the **activations** of DiT are naturally more sparse than the **weights** (with only 5–10% of channels activated per token). Consequently…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Activation Sparsity"
  - "2:4 Semi-structured Sparsity"
  - "DiT Inference Acceleration"
  - "LoRA Error Compensation"
  - "Sparse Tensor Core"
date: 2026-05-08
content_hash: 18a92cfdddf76e8c
---

# RT-Lynx: Putting the GEMM Sparsity In a Right Way for Diffusion Models

**Conference**: ICML 2026  
**arXiv**: [2605.26632](https://arxiv.org/abs/2605.26632)  
**Code**: To be confirmed  
**Area**: Model Compression / Diffusion Model Acceleration / N:M Sparsity / CUDA Kernel  
**Keywords**: Activation Sparsity, 2:4 Semi-structured Sparsity, DiT Inference Acceleration, LoRA Error Compensation, Sparse Tensor Core

## TL;DR
The authors discover that the **activations** of DiT are naturally more sparse than the **weights** (with only 5–10% of channels activated per token). Consequently, they shift 2:4 semi-structured sparsity from the weight side to the activation side, employing norm scaling, LoRA residual compensation, and selective layer skipping to recover quality loss. A fused CUDA pipeline integrating "Online Top-K Selection + Sparse GEMM" into a single kernel is developed, achieving an average 1.55× speedup for linear layers in Qwen-Image, FLUX, and Z-Image without degradation in FID/IR.

## Background & Motivation

**Background**: Diffusion Transformers (DiT) have become the mainstream backbone for high-resolution image generation. However, each inference step is dominated by dense GEMM operations with large matrices, and when combined with dozens of denoising iterations, latency and energy consumption remain difficult to reduce. The LLM domain has proven that N:M semi-structured sparsity (especially the 2:4 mode natively supported by NVIDIA) is an acceleration path that balances accuracy and hardware friendliness, with representative methods including SparseGPT, Wanda, RIA, BaWA, and Slim. However, almost all of these focus on **weight pruning**.

**Limitations of Prior Work**: Directly applying the "weight pruning" paradigm to DiT severely degrades generation quality. Control experiments on Qwen-Image show that naive 2:4 weight sparsity causes FID to surge from 21.98 to 51.63 and Image Reward to drop from 1.219 to −0.16. Even state-of-the-art methods like Wanda/RIA/BaWA only recover FID to around 40. While Slim performs better, it utilizes LoRA compensation with rank $R=0.1d$, leading to significant inference overhead. Furthermore, even with activation sparsity, the total overhead of **online** Top-K selection, format restructuring, and Sparse GEMM calls can consume 40–59% of execution time, negating the theoretical 2× speedup.

**Key Challenge**: DiT weight distributions are quasi-Gaussian and "universally dispersed," lacking the local sparse structure required for 2:4 patterns; forcing this structure prunes critical parameters. In contrast, the sparsity effectively resides in the **activations**, where a single token activates only a small subset of neurons in the FFN. Additionally, even if shifted to the activation side, naive pruning leads to a drop in $\ell_2$ norm and loss of high-frequency details. Online sparsification is also hindered by kernel scheduling overhead.

**Goal**: (1) Demonstrate that activation sparsity is naturally more suitable for DiT than weight sparsity; (2) Design a sparsification pipeline capable of compensating for quality loss to achieve "lossless" results; (3) Fuse online Top-K and Sparse GEMM into a single CUDA kernel to realize over 1.5× end-to-end speedup for linear layers.

**Key Insight**: The authors observe statistical features of weights and activations at the distribution level. Weight elements are quasi-Gaussian with broad frequency coverage and no local structure. Activations, due to the superposition phenomenon in Transformers, are concentrated near zero, with only ~5–10% of channels significantly activated. Based on this, the relative error introduced by imposing a 2:4 hard constraint on activations is much smaller than on weights.

**Core Idea**: Shift from "weight pruning" to "activation pruning," using norm scaling and low-rank LoRA residuals to recover lost energy and high-frequency details. A fused Sparse GEMM kernel is utilized to compress online overhead to under 10%, translating theoretical gains into approximately 1.2× end-to-end acceleration.

## Method

### Overall Architecture
For a linear layer $\mathbf{Y}=\mathbf{X}\cdot\mathbf{W}^{\top}$, the authors replace traditional $\mathbf{Y}=\mathbf{X}\cdot\mathbf{W}_s^{\top}$ (weight sparsity) with $\mathbf{Y}=S(\mathbf{X})\cdot\mathbf{W}^{\top}+\mathbf{X}\cdot(\mathbf{L}_A\mathbf{L}_B)^{\top}$. Here, $S(\cdot)$ represents 2:4 Top-K selection on the token dimension combined with norm scaling to restore the activation to its original norm. The term $\mathbf{X}(\mathbf{L}_A\mathbf{L}_B)^{\top}$ is a low-rank LoRA branch ($R=64$) dedicated to fitting the sparsification residual. The layers targeted for sparsification are primarily QKV projections and MLP Up/Down projections within DiT blocks. Layers in single-stream paths where LoRA cannot compensate for the loss are selectively skipped. Finally, "Online Top-K + Format Restructuring + Sparse GEMM + LoRA addition" are fused into a single CUDA execution path to avoid intermediate tensor materialization and repeated synchronization. Training focuses only on fine-tuning LoRA matrices while the backbone weights remain frozen, aiming to minimize $\|\mathbf{X}\mathbf{W}^{\top}-\mathbf{Y}\|^2$, with convergence in approximately 2k steps.

### Key Designs

1.  **Norm-Compensated Activation Sparsification**:
    - **Function**: Performs 2:4 Top-K selection on the token dimension and scales the pruned vector back to its original $\ell_2$ norm to eliminate the systematic bias caused by magnitude loss.
    - **Mechanism**: Selects the two largest absolute values within a 4-element group to obtain $\tilde{\mathbf{X}}$, calculates a scaling factor $s=\sqrt{\|\mathbf{X}\|_2^2/(\|\tilde{\mathbf{X}}\|_2^2+\epsilon)}$, and sets $S(\mathbf{X})=s\cdot\tilde{\mathbf{X}}$ with $\epsilon=10^{-8}$. This step involves only a reduction and division, which can be completed within the same kernel as Top-K with negligible overhead.
    - **Design Motivation**: Activation distributions are sharp, but the energy proportion of the two retained elements is not fixed. Naive Top-K makes the output of each token systematically smaller, biasing the statistics of downstream RMSNorm/Attention. Norm compensation aligns the magnitude to the dense path without changing the direction, ensuring that only "near-zero information loss" remains.

2.  **LoRA Residual Compensation (High-frequency Detail Recovery)**:
    - **Function**: Uses a low-rank branch $\mathbf{X}(\mathbf{L}_A\mathbf{L}_B)^{\top}$ with rank $R=64$ to fit the output residuals corresponding to the "pruned near-zero activations," specifically restoring high-frequency details like hair, edges, and textures blurred by sparsification.
    - **Mechanism**: The training objective is to minimize $\|\mathbf{X}\mathbf{W}^{\top}-(S(\mathbf{X})\mathbf{W}^{\top}+\mathbf{X}(\mathbf{L}_A\mathbf{L}_B)^{\top})\|^2$, ensuring the combined sparse and LoRA paths reconstruct the dense output. The backbone $\mathbf{W}$ is frozen while LoRA is updated. During inference, LoRA calculates $\mathbf{Y}_r$ on dense Tensor Cores, which is accumulated on-chip with the $\mathbf{Y}_s$ from Sparse GEMM.
    - **Design Motivation**: Compared to Slim's "heavy compensation" ($R=0.1d$, e.g., 307), the authors argue that residual information is inherently low-rank. Most energy is in the Top-K channels, and discarded parts are fine-grained high-frequency perturbations. $R=64$ is sufficient for fitting, and the extra LoRA GEMM overhead is significantly lower than the savings from dense computation. "Hard-to-prune" layers in single-stream DiTs (e.g., `attn.o_proj`/`mlp.up` in Z-Image) are kept dense to prevent quality collapse.

3.  **Fused Online Sparse GEMM Kernel**:
    - **Function**: Integrates pattern determination, Top-K, compression to SpTC layout, Sparse GEMM, and on-chip accumulation with LoRA output into a single CUDA execution path, reducing online sparsification overhead from 40–59% to under 10%.
    - **Mechanism**: Generates the 2:4 structured activation and its 2-bit index directly at the register level to avoid global memory write-backs. The Sparse GEMM utilizes a streamK-style block-parallel pipeline, tiling the K-dimension through SpTC to overlap bandwidth and latency across memory layers. The LoRA branch runs asynchronously with Sparse GEMM, and $\mathbf{Y}_s$ is added directly to $\mathbf{Y}_r$ registers once calculated.
    - **Design Motivation**: Existing frameworks like PyTorch-SpMM or cuSPARSElt split pruning, reformatting, and calculation into multiple kernels, consuming gains through launch overhead and intermediate I/O. By folding the entire pipeline into one kernel, the "algorithmic sparsification" and "hardware SpTC" are bound to the same register file, making the 2× theoretical limit attainable.

### Loss & Training
Only the LoRA matrices $\mathbf{L}_A, \mathbf{L}_B$ are trainable; backbone weights and optimizer states are not updated. Training data consists of prompt-image pairs generated by Qwen-Image on 20k user prompts. The loss is MSE-based: $\|\mathbf{X}\mathbf{W}^{\top}-\mathbf{Y}\|^2$. Convergence takes approximately 2k steps on NVIDIA H20 using CUDA 13.0. Inference is orthogonally compatible with FP8 quantization, step distillation, TeaCache, and SpargeAttn.

## Key Experimental Results

### Main Results (Sparsity Strategy Comparison on Qwen-Image, MJHQ / sDCI)

| Method | MJHQ FID↓ | MJHQ IR↑ | sDCI FID↓ | sDCI IR↑ |
|:---|:---:|:---:|:---:|:---:|
| Full (FP16) | 21.98 | 1.219 | 31.15 | 1.172 |
| Sparse Weight (naive 2:4) | 51.63 | −0.16 | 66.91 | −0.22 |
| Sparse Activation (naive 2:4) | 35.85 | 0.599 | 48.59 | 0.472 |
| Wanda (ICLR'24) | 40.81 | 0.536 | 55.61 | 0.325 |
| BaWA (ICML'25) | 39.68 | 0.589 | 54.54 | 0.376 |
| Slim (ICML'25, $R\approx 0.1d$) | 22.25 | 1.278 | 29.26 | 1.217 |
| **RT-Lynx (Ours, $R=64$)** | **21.25** | **1.304** | **25.78** | **1.226** |

RT-Lynx is the only sparse method that **outperforms the FP16 dense baseline** in both FID and IR across datasets, while using less than 1/5 of the LoRA rank required by Slim.

### Kernel and End-to-End Acceleration (H20, select Qwen-Image matrix sizes)

| $M=N$ | $K$ | PyTorch GEMM | cuSPARSElt | RT-Lynx Kernel | Online Overhead |
|:---|:---|:---:|:---:|:---:|:---:|
| 4096 | 3072 | 0.781 ms | 0.709 (1.10×) | 0.465 (**1.68×**) | 4.60% |
| 4096 | 12288 | 3.099 ms | 2.669 (1.16×) | 1.652 (**1.88×**) | 4.83% |
| 8192 | 12288 | 11.95 ms | 8.202 (1.45×) | 6.754 (**1.77×**) | 2.37% |

End-to-End: Qwen-Image per image 0.75s → 0.62s (1.21×); combined with 8-step Turbo distillation, Z-Image reaches 11.86× total speedup (vs 9.91× for Turbo alone).

### Ablation Study (Qwen-Image / FLUX / Z-Image, Selected)

| Model | Configuration | MJHQ FID↓ | MJHQ IR↑ | Description |
|:---|:---|:---:|:---:|:---|
| Qwen-Image | SA-Native | 35.85 | 0.599 | Activation sparse only, no compensation |
| Qwen-Image | + Norm Comp. | 25.28 | 0.939 | Norm compensation alone contributes ~10 FID |
| Qwen-Image | + LoRA ($R=64$) | **21.25** | **1.304** | LoRA further closes the gap, outperforming Full |
| FLUX.1-dev | SA-NC-LoRA | 22.61 | 0.978 | Double-stream compensation is sufficient |
| FLUX.1-dev | + Skip Layers | **21.17** | **1.011** | Skip layers required for single-stream paths |
| Z-Image | SA-NC-LoRA | 27.39 | 0.929 | LoRA not entirely sufficient |
| Z-Image | + Skip Layers | **26.17** | **0.967** | Skipping `o_proj`/`up` matches Full (25.70) |

### Key Findings
- The contributions of the three compensations rank as: LoRA > Norm Comp. > Skip Layer. However, **for single-stream DiTs (FLUX, Z-Image), skip layers are indispensable**; otherwise, even LoRA cannot bridge the gap.
- Reducing online sparsification overhead from 40–59% (PyTorch/cuSPARSElt) to **<10%** is the decisive factor for end-to-end acceleration.
- RT-Lynx is orthogonally stackable with W8A8 quantization, step distillation, TeaCache, and SpargeAttn.
- Specifically for GEMM, the RT-Lynx Kernel achieves a 1.88× speedup at $4096\times 4096\times 12288$, nearly reaching the 2× theoretical limit of SpTC.

## Highlights & Insights
- "Putting the sparsity in the right place": The authors use Figure 2 to demonstrate weight vs. activation distributions, arguing against forcing 2:4 on weights. This is the first systematic transfer of the superposition observation from the LLM circle to DiT acceleration.
- Norm compensation is a nearly zero-cost trick with significant impact—adding it alone pulls Qwen-Image FID from 35.85 to 25.28, outperforming many complex weight pruning methods. This "rescale back to original norm after pruning" approach is broadly applicable to token-wise sparse scenarios.
- The true value of LoRA residual compensation lies in **proving the residual is low-rank**: $R=64$ suffices for Qwen-Image, implying that dropped information consists of sparse, low-rank, near-zero perturbations.
- The core innovation of the kernel is the **fusion of Top-K, format restructuring, Sparse GEMM, and LoRA accumulation into one register-level pipeline**. This algorithm-system co-design is what enables the theoretical 2× gain to become a 1.2× end-to-end reality.
- Orthogonality is rigorously verified, meaning RT-Lynx can serve as an independent building block in the DiT inference stack.

## Limitations & Future Work
- The paper acknowledges that for single-stream DiTs, **manual selection of skip layers** is required, and these vary by model. A principled criterion for which layers "cannot be sparse" is currently missing.
- Evaluation is concentrated on SpTC on H20. Performance data for FP8/FP4 + 2:4 on H100/B200 or consumer cards (without SpTC) is not provided.
- Datasets like MJHQ-30K and sDCI may have limited sensitivity to "blurred details." Systemic failure mode analysis for faces, text, and complex scenes is missing.
- Training data is generated by the model itself, and cross-model transferability of the LoRA recipes has not been verified.
- Only 2:4 (50%) sparsity is explored. More aggressive patterns like 4:8 or 1:4 are natural next steps for hardware that supports them.

## Related Work & Insights
- **vs SparseGPT / Wanda / RIA / BaWA / Slim**: These prune **weights** based on calibration statistics or sensitivity. RT-Lynx proves weight distributions in DiT lack 2:4 structures and outperforms Slim's $R\approx 0.1d$ with a much smaller $R=64$ LoRA.
- **vs LLM Activation Sparsity (PowerInfer / CATS / RoSA)**: These focus on LLM **decoding** (tokens ≤ 4) and are not directly applicable to DiT image generation with tokens > 1000. RT-Lynx is the first to implement activation sparsity for DiT inference acceleration.
- **vs Amber / Haziza et al.**: Amber uses 8:16 (not natively supported); Haziza et al. apply activation sparsity only to FFN during **pre-training**. RT-Lynx is the first to complete the full 2:4 + DiT + end-to-end kernel suite.
- **vs Other DiT Acceleration (SVDQuant, DMD2, TeaCache, SpargeAttn)**: These focus on quantization, distillation, caching, or sparse attention. RT-Lynx targets linear layer acceleration and is complementary to all.

## Rating
- Novelty: ⭐⭐⭐⭐ Transparent explanation of activation as the true source of DiT sparsity with full kernel implementation, though the underlying components (activation sparsity + LoRA) have precedents in LLMs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three major DiTs, four quality metrics, itemized ablations, and stacking with four types of orthogonal acceleration across kernel and end-to-end benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Excellent motivational and algorithmic figures (Fig 1/2/4/5), though some text reads like an engineering report.
- Value: ⭐⭐⭐⭐⭐ One of the few schemes achieving lossless ~1.2× end-to-end acceleration via SpTC; directly applicable for industrial deployment with a clear path.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Right Looks, Wrong Reasons: Compositional Fidelity in Text-to-Image Generation](../../AAAI2026/image_generation/right_looks_wrong_reasons_compositional_fidelity_in_text-to-image_generation.md)
- [\[CVPR 2026\] SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration](../../CVPR2026/image_generation/sparvar_exploring_sparsity_in_visual_autoregressive_modeling_for_training-free_a.md)
- [\[ICLR 2026\] ToProVAR: Efficient Visual Autoregressive Modeling via Tri-Dimensional Entropy-Aware Semantic Analysis and Sparsity Optimization](../../ICLR2026/image_generation/toprovar_efficient_visual_autoregressive_modeling_via_tri-dimensional_entropy-aw.md)
- [\[ICML 2026\] Orthogonal Concept Erasure for Diffusion Models](orthogonal_concept_erasure_for_diffusion_models.md)
- [\[ICML 2026\] Image Restoration via Diffusion Models with Dynamic Resolution](image_restoration_via_diffusion_models_with_dynamic_resolution.md)

</div>

<!-- RELATED:END -->
