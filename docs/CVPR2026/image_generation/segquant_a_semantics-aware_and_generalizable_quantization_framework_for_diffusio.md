---
title: >-
  [Paper Note] SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models
description: >-
  [CVPR2026][Image Generation][diffusion model quantization] This paper proposes SegQuant, a framework that achieves high-fidelity post-training quantization of diffusion models through two novel components: SegLinear…
tags:
  - "CVPR2026"
  - "Image Generation"
  - "diffusion model quantization"
  - "post-training quantization"
  - "semantics-aware segmentation"
  - "polarity preservation"
  - "deployment-friendly"
date: 2026-05-08
content_hash: 513e3df9d0ad6fd1
---

# SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models

**Conference**: CVPR2026
**arXiv**: [2507.14811](https://arxiv.org/abs/2507.14811)
**Code**: None
**Area**: Image Generation
**Keywords**: diffusion model quantization, post-training quantization, semantics-aware segmentation, polarity preservation, deployment-friendly

## TL;DR

This paper proposes SegQuant, a framework that achieves high-fidelity post-training quantization of diffusion models through two novel components: SegLinear, a semantics-aware segmented quantization scheme based on static computational graph analysis, and DualScale, a hardware-native dual-scale polarity-preserving quantization scheme. The approach is cross-architecture generalizable and compatible with deployment pipelines, requiring neither handcrafted rules nor runtime dynamic information.

## Background & Motivation

**Deployment bottleneck of diffusion models**: Diffusion models (e.g., SD3.5, FLUX) achieve excellent image generation quality, but their multi-step denoising inference (typically 50 steps) imposes substantial computational overhead. Quantization is a key technique for reducing model size and inference latency, and post-training quantization (PTQ) has become the preferred industrial deployment approach due to its applicability to pre-trained models without retraining.

**Existing methods exhibit a "Compiler Gap"**: This is the central insight of the paper. Existing diffusion model PTQ methods fall into two categories, both of which are incompatible with modern AI compilers:
   - **Architecture-specific methods** (e.g., Q-Diffusion): Use manually hardcoded rules to handle the bimodal distributions in UNet skip-connections; these do not generalize to newer architectures such as DiT.
   - **Data-dependent methods** (e.g., PTQ4DiT): Rely on runtime dynamic information (timestep-varying activations, salient channels), which is fundamentally incompatible with compilers such as TensorRT that are based on static graph analysis, precluding automated deployment.

**Semantic heterogeneity in linear layers is overlooked**: In DiT architectures, linear layers in modules such as AdaNorm and TimeEmbedding operate on inputs composed of multiple semantically distinct segments produced by chunk/split/concat operations. Different segments exhibit markedly different data distributions (e.g., AdaNorm weights display clear segmented patterns as shown in Figure 4). Applying uniform quantization across the entire layer causes "quantization interference"—the numerical characteristics of one segment degrade the accuracy of another.

**Quantization challenges for polarity-asymmetric activations**: Modern activation functions such as SiLU and GELU (widely used in DiT, SD3, and FLUX), unlike ReLU, preserve dense low-magnitude negative values. Their outputs are highly skewed: positive values can reach 3.5, while negative values are confined to $[-0.3, 0]$. Standard quantization uniformly distributes the finite quantization bins across the entire range, severely compressing the semantically critical negative region. Visualization experiments (Figure 7) clearly demonstrate that negative activations carry high-frequency details and texture consistency, and quantization loss in this region directly degrades image quality.

**Existing polarity-handling solutions break GPU acceleration paths**: Logarithmic quantizers and custom bit-width schemes from the ViT quantization literature redefine data representations, breaking Tensor Core fixed-width PTX instructions and CUDA epilogue fusion mechanisms, rendering them unusable for high-throughput GPU inference.

## Method

### Overall Architecture

SegQuant adopts a top-down modular design (Figure 1) with four pluggable components:

| Component | Role | Available Implementations |
|-----------|------|--------------------------|
| **Optimizer** | Activation distribution preprocessing to smooth quantization difficulty | SmoothQuant, SVDQuant, DMQ, SpinQuant |
| **Calibrator** | Quantization parameter calibration (scale/zero-point) | GPTQ (Hessian reconstruction), AMax (maximum absolute value) |
| **SegLinear** ★ | Semantics-aware segmented quantization via computational graph analysis | Automatic graph analysis, no manual configuration |
| **DualScale** ★ | Hardware-native polarity-preserving quantization | BatchedGEMM implementation, no custom operators |

The default combination is SmoothQuant + GPTQ + SegLinear + DualScale. Users may freely substitute the Optimizer and Calibrator, making the framework a general-purpose quantization platform.

### SegLinear: Semantics-Aware Segmented Quantization

**Core principle**: Linear layers in complex neural networks frequently operate on semantically heterogeneous inputs—different segments of the input vector encode semantically distinct information. SegLinear automatically identifies semantic boundaries by analyzing chunk/split/concat/reshape operation patterns in the static computational graph (torch.fx DAG), independently quantizes each segment, and eliminates quantization interference.

**Pattern 1: Output-Segmented Quantization**
When the output of a linear layer is followed by a chunk or split operation, the different portions of the output vector flow to semantically distinct downstream branches. The weight matrix $\mathbf{W} \in \mathbb{R}^{k \times n}$ is partitioned column-wise into $[\mathbf{W}_1, \ldots, \mathbf{W}_N]$, where $\mathbf{W}_i \in \mathbb{R}^{k \times d_i}$; each segment is quantized independently and then concatenated:

$$\hat{\mathbf{Y}} = [\hat{\mathbf{X}}\hat{\mathbf{W}}_1, \hat{\mathbf{X}}\hat{\mathbf{W}}_2, \cdots, \hat{\mathbf{X}}\hat{\mathbf{W}}_N]$$

Typical use case: The AdaNorm layer output in DiT is split via chunk into shift and scale parameters, whose distributional characteristics are entirely different.

**Pattern 2: Input-Segmented Quantization**
When the input to a linear layer originates from a concat or reshape operation (e.g., multi-head merging in MHA), different segments of the input vector come from semantically distinct upstream paths. The weight matrix is partitioned row-wise into $[\mathbf{W}_1^T, \ldots, \mathbf{W}_N^T]^T$; each segment is quantized independently and the results are summed:

$$\hat{\mathbf{Y}} = \sum_{i=1}^{N} \hat{\mathbf{X}}_i \hat{\mathbf{W}}_i$$

Typical use case: UNet skip-connections concatenate features before passing them to a linear layer, where the two sources exhibit large distributional discrepancies.

**Fundamental distinction from Q-Diffusion**: Q-Diffusion uses handcrafted rules specifically targeting the bimodal distribution in UNet skip-connections—a non-generalizable special case. SegLinear is a fully automated graph analysis algorithm that requires no manual specification of which layers need segmentation and is applicable to arbitrary structural patterns including AdaNorm, MHA, and TimeEmbedding.

**Complementarity with channel-wise quantization**: Channel-wise quantization processes each output channel independently, whereas SegLinear captures higher-level inter-channel semantic relationships defined by the computational graph structure. Within semantically consistent channel groups, SegLinear jointly optimizes shared hyperparameters (e.g., the migration strength $\alpha$ in SmoothQuant), providing more stable optimization and better low-bit scalability.

### DualScale: Dual-Scale Polarity-Preserving Quantization

**Problem quantification**: The following table shows activation polarity statistics on the COCO dataset for SD3.5-ControlNet (averaged over 30 timesteps), revealing that a large proportion of channels are consistently dominated by negative values:

| Layer (Module) | Activation | Channels | Neg./Pos. Ratio |
|----------------|------------|----------|-----------------|
| AdaNorm (DiT) | SiLU | 1536 | 0.955 / 0.021 |
| AdaNorm (Ctrl.) | SiLU | 1536 | 0.645 / 0.338 |
| FFN (DiT) | GELU | 6144 | 0.744 / 0.256 |
| FFN (Ctrl.) | GELU | 6144 | 0.589 / 0.400 |

In the AdaNorm layer of DiT, 95.5% of channels are dominated by negative values, indicating that the negative region carries substantial semantic information.

**Quantization scheme**: The activation matrix $\mathbf{X}$ is decomposed by polarity into positive and negative components, each quantized with an independent scale:

$$\mathbf{X}_+ = \max(\mathbf{X}, 0), \quad \mathbf{X}_- = \min(\mathbf{X}, 0)$$

$$s_- = \frac{|\min(x)|}{q_{\min}}, \quad s_+ = \frac{\max(x)}{q_{\max}}$$

The final output is reconstructed via linear combination:

$$\mathbf{Y} \approx s_+ s_w \cdot (\hat{\mathbf{X}}_+ \hat{\mathbf{W}}) + s_- s_w \cdot (\hat{\mathbf{X}}_- \hat{\mathbf{W}})$$

**Hardware-native implementation**: While DualScale nominally requires two matrix multiplications, the key design insight is that $\hat{\mathbf{X}}_+ \hat{\mathbf{W}}$ and $\hat{\mathbf{X}}_- \hat{\mathbf{W}}$ are executed in parallel within a single kernel launch via CUTLASS BatchedGEMM, with the two scaled results merged in a fused epilogue. This fully preserves the standard integer GEMM path, leveraging Tensor Core parallelism and CUDA epilogue fusion **without any custom operators or additional kernel launches**. DualScale further avoids reverse zero-point correction, requiring only fixed positive/negative scales to reconstruct the output.

### Loss & Training

SegQuant is a pure PTQ framework that introduces no additional training losses. Quantization quality is assessed via layer-wise Frobenius norm error $\|\Delta \epsilon_t\|_F$ (Figure 3). The calibration stage supports two options:
- **GPTQ**: Hessian-based layer-wise reconstruction optimization; higher accuracy but requires calibration data (256 images for SD3/SDXL, 64 for FLUX 8-bit, 32 for 4-bit).
- **AMax**: Maximum absolute value calibration; simpler and faster.

All experiments use 50-step sampling with default schedulers, executed on Ada Lovelace architecture GPUs (24GB/48GB VRAM).

## Key Experimental Results

### Main Results: Cross-Model Cross-Precision Evaluation on MJHQ-30K (Table 2)

| Model | Params | W/A | Method | FID↓ | IR↑ | LPIPS↓ | PSNR↑ | SSIM↑ |
|-------|--------|-----|--------|------|-----|--------|-------|-------|
| SD3.5-DiT | 2B | FP16 | Baseline | 23.70 | 0.952 | - | - | - |
| SD3.5-DiT | 2B | W8A8 | PTQD | 36.84 | 0.309 | 0.520 | 10.20 | 0.417 |
| SD3.5-DiT | 2B | W8A8 | PTQ4DiT | 25.66 | 0.752 | 0.426 | 12.18 | 0.532 |
| SD3.5-DiT | 2B | W8A8 | Smooth+ | 24.10 | 0.851 | 0.404 | 12.16 | 0.552 |
| SD3.5-DiT | 2B | W8A8 | **SegQuant-A** | 24.33 | **0.924** | 0.384 | 12.78 | 0.563 |
| SD3.5-DiT | 2B | W8A8 | **SegQuant-G** | **23.94** | 0.859 | **0.383** | **12.83** | **0.564** |
| SD3.5-DiT | 2B | W4A8 | PTQ4DiT | 60.47 | -0.190 | 0.577 | 10.06 | 0.429 |
| SD3.5-DiT | 2B | W4A8 | SVDQuant | 27.95 | 0.725 | 0.456 | 11.76 | 0.523 |
| SD3.5-DiT | 2B | W4A8 | **SegQuant-G** | **27.30** | **0.762** | **0.453** | 11.69 | 0.521 |
| FLUX-DiT | 12B | BF16 | Baseline | 23.21 | 0.837 | - | - | - |
| FLUX-DiT | 12B | W8A8 | Q-Diffusion | 23.99 | 0.732 | 0.299 | 15.87 | 0.633 |
| FLUX-DiT | 12B | W8A8 | PTQ4DiT | 27.34 | 0.630 | 0.325 | 15.36 | 0.611 |
| FLUX-DiT | 12B | W8A8 | **SegQuant-G** | **23.07** | **0.822** | **0.138** | **20.32** | **0.782** |
| FLUX-DiT | 12B | W4A8 | SVDQuant | 23.61 | 0.783 | 0.232 | 17.29 | 0.697 |
| FLUX-DiT | 12B | W4A8 | **SegQuant-G** | **23.45** | **0.789** | **0.225** | **17.48** | **0.702** |
| SDXL-UNet | - | FP16 | Baseline | 17.10 | 0.910 | - | - | - |
| SDXL-UNet | - | W8A8(fp) | Q-Diffusion | 17.04 | 0.897 | 0.093 | 24.31 | 0.827 |
| SDXL-UNet | - | W8A8(fp) | **SegQuant-G** | **17.03** | **0.903** | **0.082** | **24.84** | **0.838** |

### Ablation Study (SD3.5 W8A8, MJHQ-30K, SmoothQuant+AMax, Table 4)

| Configuration | FID↓ | IR↑ | LPIPS↓ | PSNR↑ | SSIM↑ |
|---------------|------|-----|--------|-------|-------|
| Baseline (no Seg./Dual) | 23.35 | 0.877 | 0.419 | 11.93 | 0.536 |
| +SegLinear | 23.36 | 0.899 | 0.395 | 12.03 | 0.554 |
| +DualScale | 22.61 | 0.909 | 0.401 | 12.14 | 0.551 |
| +Seg.+Dual. (full SegQuant) | **22.54** | **0.952** | **0.377** | **12.50** | **0.567** |

### SegLinear Layer-Wise Error Reduction (SD3.5, Table 3)

| Layer | Calibration | F-norm w/o Seg. | F-norm w/ Seg. | Reduction |
|-------|-------------|-----------------|----------------|-----------|
| DiT.0.norm1 | SmoothQuant | 0.7041 | 0.5381 | -23.6% |
| DiT.0.norm1 | GPTQ | 0.8350 | 0.4441 | -46.8% |
| DiT.0.norm1_context | GPTQ | 1.5166 | 0.7441 | -50.9% |
| DiT.11.norm1_context | GPTQ | 3.0176 | 1.7637 | -41.6% |
| DiT.11.attn.out | SmoothQuant | 2273.3 | 1879.3 | -17.3% |
| DiT.11.attn.out | SVDQuant | 2031.6 | 1810.7 | -10.9% |

### Key Findings

- **Most dramatic improvement on FLUX**: Under W8A8, LPIPS drops substantially from 0.299 (Q-Diffusion) to 0.138 (a 54% reduction), and PSNR improves from 15.87 to 20.32 (+4.45 dB), demonstrating that SegLinear is particularly effective at addressing semantic heterogeneity in large models (12B).
- **High complementarity between SegLinear and DualScale**: In the ablation study, each component individually improves Image Reward from 0.877 to 0.899/0.909; their combination yields 0.952 (exceeding the FP16 baseline), reflecting the orthogonal and complementary nature of structural segmentation and polarity preservation.
- **SegLinear is most effective for normalization layers**: Under GPTQ calibration, the Frobenius error of DiT.0.norm1_context is reduced by half (−50.9%), confirming that semantic heterogeneity introduced by chunk operations in AdaNorm is indeed a critical source of quantization degradation.
- **Cross-architecture generalization**: The same SegQuant achieves optimal or near-optimal performance on both DiT (SD3.5, FLUX) and UNet (SDXL) architectures without any architecture-specific modifications.
- **Efficiency-quality trade-off**: INT8 models are approximately half the size of FP16 models (Figure 10); the additional inference overhead introduced by DualScale is modest, while the quality gains substantially outweigh the cost.

## Highlights & Insights

- **Precise framing of the "Compiler Gap"**: Reframing the core challenge of diffusion model quantization from "accuracy" to "deployment compatibility" represents a pragmatic and important perspective shift. Existing methods may perform well in experiments but cannot be automatically integrated into deployment pipelines; SegQuant is the first to systematically address this industrial pain point.
- **Purely static graph-driven**: SegLinear is based entirely on structural analysis of the torch.fx computational graph, with no dependence on any runtime data (activation statistics, timestep information), making it naturally compatible with static-graph-based compilers such as TensorRT and TVM.
- **Elegant hardware-native design of DualScale**: Polarity decomposition with dual-scale quantization is mapped to BatchedGEMM plus epilogue fusion, executing what appear to be two GEMMs in parallel within a single kernel launch at zero custom operator overhead. This approach of maximally exploiting existing hardware primitives is broadly instructive.
- **Modular architecture**: The pluggable Optimizer and Calibrator components make SegQuant not merely a method but an extensible quantization platform into which new PTQ techniques can be directly integrated.

## Limitations & Future Work

- **DualScale theoretically doubles FLOPs**: Although latency overhead is eliminated through BatchedGEMM parallelization, the computational volume remains twice that of standard quantization, which may still be impactful in latency-critical inference scenarios. An adaptive strategy—enabling DualScale only for layers with severe polarity asymmetry (e.g., AdaNorm)—could be explored.
- **Limited gains at low bit-widths**: The advantage over SVDQuant under W4A8 is less pronounced than under W8A8, and extreme low-bit (W4A4) scenarios remain challenging. This is likely because weight quantization error itself becomes the bottleneck at 4-bit, and improvements on the activation side alone are insufficient to compensate.
- **Limited to image generation**: Validation on video generation (e.g., ViDiT-Q, Q-VDiT with temporal token scenarios) and 3D generation tasks has not been conducted. SegLinear's graph analysis is theoretically generalizable, but experimental evidence is needed.
- **Calibration data requirement**: The GPTQ variant still requires 32–256 calibration images; fully zero-shot PTQ is not feasible.
- **SegLinear search space**: The current implementation only matches known graph operation patterns (chunk/split/concat/reshape); for more complex custom operator graph structures, segmentation boundaries may not be automatically discovered.

## Related Work & Insights

- **Q-Diffusion**: First identified the bimodal distribution problem caused by UNet skip-connections and addressed it with handcrafted segmentation—a non-generalizable special case and a direct inspiration for SegLinear. SegQuant generalizes this from manual rules to automatic graph analysis.
- **PTQ4DiT**: Achieves strong performance on DiT by exploiting timestep-dynamic activation information, but is incompatible with static graph compilers—a canonical example of the "Compiler Gap" as defined by SegQuant.
- **SmoothQuant / SVDQuant**: Activation distribution smoothing and low-rank decomposition methods, integrated into SegQuant as pluggable Optimizers, validating the framework's compatibility.
- **GPTQ**: Hessian-based layer-wise reconstruction calibration, used as the default Calibrator in SegQuant.
- **ViDiT-Q / Q-VDiT**: Video diffusion quantization methods leveraging temporal redundancy and token-level adaptation. Complementary to SegQuant's general computational graph analysis approach and theoretically integrable as Calibrator components.
- **TFMQ-DM / TAC-Diffusion**: Temporal feature maintenance and timestep-aware calibration methods; orthogonal techniques that could be integrated as Calibrator components within the SegQuant framework.

## Rating

- Novelty: ⭐⭐⭐⭐ — The "Compiler Gap" framing is novel and pragmatic; purely static graph-driven semantic quantization is distinctive; the hardware-native design of DualScale is particularly inventive.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three architectures (SD3.5-DiT / FLUX-DiT / SDXL-UNet), three precision levels (W8A8 / W4A8 / W8A8fp), three datasets, five evaluation metrics, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear framework hierarchy, precise and compelling problem definition (Compiler Gap), concise mathematical derivations, and information-rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ — Directly applicable to industrial deployment; the modular design enables use as a unified quantization platform.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models](seacache_spectral-evolution-aware_cache_for_accelerating_diffusion_models.md)
- [\[CVPR 2026\] RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models](razor_ratio-aware_layer_editing_for_targeted_unlearning_in_vision_transformers_a.md)
- [\[ICCV 2025\] DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization](../../ICCV2025/image_generation/dmq_dissecting_outliers_of_diffusion_models_for_post-training_quantization.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[CVPR 2026\] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)

</div>

<!-- RELATED:END -->
