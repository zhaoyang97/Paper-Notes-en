---
title: >-
  [Paper Note] RT-Lynx: Putting the GEMM Sparsity In a Right Way for Diffusion Models
description: >-
  [ICML 2026][Image Generation][Activation Sparsity] Authors find that DiT **activations** are more naturally sparse than **weights** (only 5–10% of channels per token are activated). They migrate 2:4 semi-structured sparsity from the weight side to the activation side, utilizing norm scaling, LoRA residual compensation, and selective layer skipping to recover quality. A fused CUDA pipeline for "Online Top-K + Sparse GEMM" is implemented, achieving an average 1.55× speedup for…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Activation Sparsity"
  - "2:4 Semi-structured Sparsity"
  - "DiT Inference Acceleration"
  - "LoRA Error Compensation"
  - "Sparse Tensor Core"
date: 2026-05-08
content_hash: dd55eeada1e46c5d
---

# RT-Lynx: Putting the GEMM Sparsity In a Right Way for Diffusion Models

**Conference**: ICML 2026  
**arXiv**: [2605.26632](https://arxiv.org/abs/2605.26632)  
**Code**: To be confirmed  
**Area**: Model Compression / Diffusion Model Acceleration / N:M Sparsity / CUDA Kernel  
**Keywords**: Activation Sparsity, 2:4 Semi-structured Sparsity, DiT Inference Acceleration, LoRA Error Compensation, Sparse Tensor Core

## TL;DR
Authors find that DiT **activations** are more naturally sparse than **weights** (only 5–10% of channels per token are activated). They migrate 2:4 semi-structured sparsity from the weight side to the activation side, utilizing norm scaling, LoRA residual compensation, and selective layer skipping to recover quality. A fused CUDA pipeline for "Online Top-K + Sparse GEMM" is implemented, achieving an average 1.55× speedup for linear layers on Qwen-Image / FLUX / Z-Image without FID/IR degradation.

## Background & Motivation

**Background**: Diffusion Transformer (DiT) has become the mainstream backbone for high-resolution image generation. However, each inference step is dominated by dense large matrix multiplications (GEMM), making latency and energy consumption difficult to reduce across dozens of denoising iterations. In the LLM field, N:M semi-structured sparsity (especially the 2:4 mode natively supported by NVIDIA) has proven to be a hardware-friendly acceleration route balancing accuracy, represented by methods like SparseGPT, Wanda, RIA, BaWA, and Slim, yet they almost exclusively focus on **weight pruning**.

**Limitations of Prior Work**: Directly applying "weight pruning" to DiT severely degrades generation quality. Control experiments on Qwen-Image showed that naive 2:4 weight sparsity caused FID to surge from 21.98 to 51.63 and Image Reward to drop from 1.219 to −0.16. Recent methods like Wanda/RIA/BaWA only recover FID to around 40. While Slim performs better, it utilizes LoRA compensation with rank $R=0.1d$, leading to high inference overhead. Furthermore, even with activation sparsity, the overhead of **online** Top-K selection, format restructuring, and calling Sparse GEMM eats up 40–59% of execution time, neutralizing the theoretical 2× speedup.

**Key Challenge**: DiT weight distributions are approximately Gaussian and "widely dispersed," lacking the local sparse structure required for 2:4 patterns; forcing this structure removes critical parameters. Conversely, what determines "what to prune" is the fact that **individual tokens in FFNs only activate a small subset of neurons**—sparsity naturally exists in activations rather than weights. However, naive activation pruning causes $\ell_2$ norm drops and loss of high-frequency details, while online sparsification is hampered by kernel scheduling overhead.

**Goal**: (1) Demonstrate that activation sparsity is more naturally suited for DiT than weight sparsity; (2) Design a sparsification pipeline capable of compensating for quality loss to "lossless" levels; (3) Fuse Online Top-K and Sparse GEMM into a single CUDA kernel to achieve over 1.5× end-to-end acceleration for linear layers.

**Key Insight**: Distributional analysis shows weights are quasi-Gaussian with no local structure, while activations are concentrated near zero due to Transformer superposition, with only ~5–10% of channels significantly activated. Consequently, the relative error introduced by 2:4 hard constraints on activations is far lower than on weights.

**Core Idea**: Shift from "weight pruning" to "activation pruning." Use norm scaling and low-rank LoRA residuals to recover lost energy and high-frequency details. Implement a fused Sparse GEMM kernel to keep online overhead below 10%, translating theoretical gains into ~1.2× end-to-end acceleration.

## Method

### Overall Architecture
RT-Lynx aims to enable 2:4 sparse acceleration via SpTC for DiT linear layers $\mathbf{Y}=\mathbf{X}\cdot\mathbf{W}^{\top}$ without quality loss. Traditional weight sparsity $\mathbf{Y}=\mathbf{X}\cdot\mathbf{W}_s^{\top}$ is rewritten as $\mathbf{Y}=S(\mathbf{X})\cdot\mathbf{W}^{\top}+\mathbf{X}\cdot(\mathbf{L}_A\mathbf{L}_B)^{\top}$: the first term applies sparsity to **activations** ($S(\cdot)$ denotes token-wise 2:4 Top-K plus norm scaling), and the second term is a LoRA branch with rank $R=64$ to compensate for the residual. Sparsified layers include QKV projections and MLP Up/Down projections. Layers in single-stream paths that LoRA cannot fully recover are skipped. The entire "Online Top-K + Format Restructuring + Sparse GEMM + LoRA Accumulation" is folded into a single CUDA execution path. During training, backbone weights are frozen, and only LoRA is fine-tuned.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    X["DiT Linear Layer Activation X<br/>(QKV / MLP Up·Down)"]
    subgraph K["Fused Online Sparse GEMM Kernel (Single CUDA Path)"]
        direction TB
        SP["Norm-Compensated Activation Sparsity<br/>2:4 Top-K → ℓ2 Scaling → S(X)"]
        SG["Sparse GEMM on SpTC<br/>Y_s = S(X)·W^T"]
        LR["LoRA Residual Compensation<br/>Y_r = X(L_A·L_B)^T, dense TC"]
        ACC["On-chip Accumulation Y = Y_s + Y_r"]
        SP --> SG --> ACC
        LR --> ACC
    end
    X --> SP
    X --> LR
    X -->|Hard Single-stream Layers: o_proj / up·down| SKIP["Skip Sparsification<br/>Retain Dense Computation"]
    ACC --> Y["Layer Output Y"]
    SKIP --> Y
```

### Key Designs

**1. Norm-Compensated Activation Sparsification: Eliminating Systematic Magnitude Bias**
Naive 2:4 Top-K yields $\tilde{\mathbf{X}}$ by keeping only the two largest absolute values in every 4-element group, but the energy ratio of these two elements varies, leading to systematically smaller token outputs. This skews downstream RMSNorm/Attention statistics, collapsing FID. The authors scale the pruned vector back to its original $\ell_2$ norm using $s=\sqrt{\|\mathbf{X}\|_2^2/(\|\tilde{\mathbf{X}}\|_2^2+\epsilon)}$ ($\epsilon=10^{-8}$), setting $S(\mathbf{X})=s\cdot\tilde{\mathbf{X}}$. This aligns the magnitude with the dense path without changing the direction. The cost is negligible (one reduction and one division fused into the Top-K kernel), but it recovers Qwen-Image FID from 35.85 to 25.28.

**2. LoRA Residual Compensation: Recovering High-Frequency Details**
While norm compensation aligns magnitude, discarded near-zero activations still carry high-frequency details (edges, textures). A low-rank branch $\mathbf{X}(\mathbf{L}_A\mathbf{L}_B)^{\top}$ fits this residual. The training objective minimizes $\|\mathbf{X}\mathbf{W}^{\top}-(S(\mathbf{X})\mathbf{W}^{\top}+\mathbf{X}(\mathbf{L}_A\mathbf{L}_B)^{\top})\|^2$. Backbone $\mathbf{W}$ is frozen while LoRA is updated. During inference, LoRA computes $\mathbf{Y}_r$ on dense Tensor Cores and accumulates with sparse GEMM result $\mathbf{Y}_s$ on-chip. Authors argue the residual is low-rank as most energy remains in Top-K channels; thus, $R=64$ suffices. This is much more efficient than Slim’s $R\approx 0.1d$ (e.g., 307). Layers in single-stream DiTs that remain challenging (e.g., `attn.o_proj`, `mlp.up` in Z-Image) are kept dense to preserve stability.

**3. Fused Online Sparse GEMM Kernel: Realizing Theoretical 2× Speedup**
Existing tools (PyTorch-SpMM, cuSPARSElt, CUTLASS) split "pruning, restructuring, and calculation" into multiple kernels, where launch overhead and intermediate memory I/O consume 40–59% of runtime. RT-Lynx folds pattern determination → Top-K → SpTC layout compression → Sparse GEMM → LoRA accumulation into one CUDA path. 2:4 structured activations and 2-bit indices are generated in registers and never written back to global memory. Sparse GEMM uses a streamK-style block-parallel pipeline to stream K-dimensions through SpTC, overlapping bandwidth and latency. The LoRA branch runs asynchronously, and $\mathbf{Y}_s$ is added directly to $\mathbf{Y}_r$ registers. This binds algorithmic sparsification to hardware SpTC on the same register file, keeping online overhead below 10%. Verified speeds: Sparse GEMM 1.88×, linear layers average 1.55×, end-to-end ~1.2×.

### Loss & Training
Only LoRA matrices $\mathbf{L}_A, \mathbf{L}_B$ are trainable; backbone weights and optimizer states are fixed. Training data consists of 20k prompt-image pairs generated by Qwen-Image. Loss is MSE-based $\|\mathbf{X}\mathbf{W}^{\top}-\mathbf{Y}\|^2$, converging in ~2k steps. All training was performed on NVIDIA H20 using CUDA 13.0. Inference is orthogonal to FP8, step distillation, TeaCache, and SpargeAttn.

## Key Experimental Results

### Main Results (Sparsity Comparison on Qwen-Image, MJHQ / sDCI)

| Method | MJHQ FID↓ | MJHQ IR↑ | sDCI FID↓ | sDCI IR↑ |
|------|-----------|----------|-----------|----------|
| Full (FP16) | 21.98 | 1.219 | 31.15 | 1.172 |
| Sparse Weight (naive 2:4) | 51.63 | −0.16 | 66.91 | −0.22 |
| Sparse Activation (naive 2:4) | 35.85 | 0.599 | 48.59 | 0.472 |
| Wanda (ICLR'24) | 40.81 | 0.536 | 55.61 | 0.325 |
| BaWA (ICML'25) | 39.68 | 0.589 | 54.54 | 0.376 |
| Slim (ICML'25, $R\approx 0.1d$) | 22.25 | 1.278 | 29.26 | 1.217 |
| **RT-Lynx (Ours, $R=64$)** | **21.25** | **1.304** | **25.78** | **1.226** |

RT-Lynx is the only sparse method that **outperforms the FP16 dense baseline** in both FID and IR on both datasets, despite using less than 1/5 of the LoRA rank compared to Slim.

### Kernel and End-to-End Speedup (H20, Qwen-Image matrix sizes)

| $M{=}N$ | $K$ | PyTorch GEMM | cuSPARSElt | RT-Lynx Kernel | Online Overhead |
|---------|-----|--------------|-----------|----------------|--------------|
| 4096 | 3072 | 0.781 ms | 0.709 (1.10×) | 0.465 (**1.68×**) | 4.60% |
| 4096 | 12288 | 3.099 ms | 2.669 (1.16×) | 1.652 (**1.88×**) | 4.83% |
| 8192 | 12288 | 11.95 ms | 8.202 (1.45×) | 6.754 (**1.77×**) | 2.37% |

End-to-end: Qwen-Image per image 0.75s → 0.62s (1.21×). Combined with 8-step Turbo distillation, Z-Image reached 11.86× total speedup (vs. 9.91× for Turbo alone). It maintains ~1.3× additional speedup when combined with W8A8, TeaCache, or SpargeAttn.

### Ablation Study (Selected)

| Model | Configuration | MJHQ FID↓ | MJHQ IR↑ | Description |
|------|------|-----------|----------|------|
| Qwen-Image | SA-Native | 35.85 | 0.599 | Activation sparse only, no compensation |
| Qwen-Image | + Norm Comp. | 25.28 | 0.939 | Norm compensation alone contributes ~10 FID |
| Qwen-Image | + LoRA ($R=64$) | **21.25** | **1.304** | LoRA eliminates remaining gap, beating Full |
| FLUX.1-dev | SA-NC-LoRA | 22.61 | 0.978 | Dual-stream compensation is sufficient |
| FLUX.1-dev | + Skip Layers | **21.17** | **1.011** | Skip layers required for single-stream paths |
| Z-Image | SA-NC-LoRA | 27.39 | 0.929 | LoRA not entirely sufficient |
| Z-Image | + Skip Layers | **26.17** | **0.967** | Skip `o_proj`/`up` single-stream layers reaches near Full (25.70) |

### Key Findings
- Contribution of the three compensations: LoRA > Norm Comp. > Skip Layer. However, **for single-stream DiTs (FLUX, Z-Image), skipping layers is essential** to avoid quality collapse.
- Online sparsification overhead was reduced from 40–59% (PyTorch/cuSPARSElt) to **<10%**, which is the decisive factor for end-to-end speedup.
- RT-Lynx is orthogonal to W8A8, step distillation, TeaCache, and SpargeAttn. On Z-Image, 8-step Turbo + RT-Lynx achieved 11.86× speedup. Weight sparsity on 8-step models resulted in FID of 360.2, while RT-Lynx remained lossless.
- Purely for GEMM, the RT-Lynx kernel achieved 1.88× speedup on $4096\times 4096\times 12288$ matrices, approaching the 2× theoretical limit of SpTC.

## Highlights & Insights
- "Putting sparsity in the right place": Authors use Figure 2 to compare weight vs. activation distributions, arguing 2:4 doesn't fit weights. This is the first systematic migration of LLM superposition observations (few active FFN neurons) to DiT acceleration.
- Norm compensation is a near zero-cost trick with high impact—alone, it recovers Qwen-Image FID from 35.85 to 25.28. This "rescale back to original norm" approach is applicable to any token-wise sparsity scenario (LLM decoding, video DiT).
- The value of LoRA residual compensation lies in **proving the residual is low-rank**: $R=64$ being sufficient implies discarded information is a low-rank, near-zero perturbation, not dense energy. This opens a path for "pruning + tiny LoRA repair."
- The core kernel innovation is **fusing Online Top-K + Format Restructuring + Sparse GEMM + LoRA Accumulation into a single register pipeline**. This enables the 2× theoretical gain to manifest as ~1.2× end-to-end acceleration.
- "Orthogonal and stackable": Verified compatibility with quantization/distillation/caching means RT-Lynx can serve as an independent building block in the DiT inference stack.

## Limitations & Future Work
- Authors admit that for single-stream DiTs, they must **manually select layers to skip** (`attn.o_proj`, `mlp.up`, or `mlp.down`), and the list varies by model. A principled criterion for "non-sparsifiable layers" is missing.
- Evaluation is focused on H20 SpTC. Performance of FP8/FP4 + 2:4 on H100/B200 or performance on consumer cards (without SpTC, like RTX 4090) using PyTorch-SpMM is not documented.
- Datasets like MJHQ-30K and sDCI may lack sensitivity to "blurred details." Systematic failure mode analysis for faces, text, and complex scenes is missing.
- Training data is generated by the **model itself** to fit LoRA (self-distillation). Cross-model transferability of LoRA recipes is unverified.
- Currently limited to 2:4 (50% sparsity). Designing for more aggressive sparsity (4:8 or 1:4) if hardware supports it is the next step.

## Related Work & Insights
- **vs. SparseGPT / Wanda / RIA / BaWA / Slim**: These methods prune **weights**. RT-Lynx proves weights lack 2:4 structure in DiT, moves to the activation side, and surpasses Slim ($R\approx 0.1d$) using only $R=64$ LoRA—improving both accuracy and efficiency.
- **vs. LLM Activation Sparsity (PowerInfer / CATS / RoSA)**: These typically serve LLM **decoding** (tokens ≤ 4) and cannot be directly applied to DiT generation (tokens > 1000). RT-Lynx is the first to implement activation sparsity for DiT.
- **vs. Amber / Haziza et al.**: Amber uses 8:16 mode (not natively supported by current GPUs); Haziza et al. focus on LLM **pre-training** activation sparsity for FFNs only. RT-Lynx is the first to solve native 2:4 + DiT + end-to-end kernels.
- **vs. Other DiT Acceleration (SVDQuant, DMD2, TeaCache, SpargeAttn)**: Those focus on quantization, distillation, feature caching, or sparse attention. RT-Lynx specifically targets GEMM-dominant linear layers and is orthogonal to them.
- **Insights**: (1) Distribution analysis before applying sparsity constraints is a valuable paradigm; (2) "Norm compensation + Low-rank residual" is applicable to all hard-mask lossy compression; (3) Algorithm-kernel co-design is critical for realizing theoretical gains.

## Rating
- Novelty: ⭐⭐⭐⭐ Solid demonstration of "activation as the true source of DiT sparsity" with full kernel implementation. Single ideas (activation sparsity + LoRA) have precedents in LLM.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three DiT models, four quality metrics, detailed ablations, and stacking with four orthogonal methods. Both kernel and end-to-end benchmarks provided.
- Writing Quality: ⭐⭐⭐⭐ Excellent motivation and algorithm diagrams (Figs 1/2/4/5), though some text reflects engineering report style.
- Value: ⭐⭐⭐⭐⭐ Breakthrough for realizing ~1.2× lossless end-to-end acceleration via SpTC. Highly practical for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DiffSparse: Accelerating Diffusion Transformers with Learned Token Sparsity](../../ICLR2026/image_generation/diffsparse_accelerating_diffusion_transformers_with_learned_token_sparsity.md)
- [\[ICLR 2026\] Forget Many, Forget Right: Scalable and Precise Concept Unlearning in Diffusion Models](../../ICLR2026/image_generation/forget_many_forget_right_scalable_and_precise_concept_unlearning_in_diffusion_mo.md)
- [\[CVPR 2026\] Semantics Lead the Way: Harmonizing Semantic and Texture Modeling with Asynchronous Latent Diffusion](../../CVPR2026/image_generation/semantics_lead_the_way_harmonizing_semantic_and_texture_modeling_with_asynchrono.md)
- [\[CVPR 2026\] SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration](../../CVPR2026/image_generation/sparvar_exploring_sparsity_in_visual_autoregressive_modeling_for_training-free_a.md)
- [\[ECCV 2024\] Getting it Right: Improving Spatial Consistency in Text-to-Image Models](../../ECCV2024/image_generation/getting_it_right_improving_spatial_consistency_in_text-to-image_models.md)

</div>

<!-- RELATED:END -->
