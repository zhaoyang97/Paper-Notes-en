---
title: >-
  [Paper Note] Efficient Learned Image Compression without Entropy Coding
description: >-
  [ICML 2026][Model Compression][Learned Image Compression] EF-LIC replaces the slow and serial entropy coding module in the learned image compression pipeline with a two-step process: "maximizing index entropy via unconst…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Learned Image Compression"
  - "Entropy-less Coding"
  - "Vector Quantization"
  - "Contextual Reparameterization"
  - "GPU Parallelization"
date: 2026-05-08
content_hash: 829429e229bda6be
---

# Efficient Learned Image Compression without Entropy Coding

**Conference**: ICML 2026  
**arXiv**: [2605.23323](https://arxiv.org/abs/2605.23323)  
**Code**: To be confirmed  
**Area**: Model Compression / Learned Image Coding / Generative Compression  
**Keywords**: Learned Image Compression, Entropy-less Coding, Vector Quantization, Contextual Reparameterization, GPU Parallelization  

## TL;DR
EF-LIC replaces the slow and serial entropy coding module in the learned image compression pipeline with a two-step process: "maximizing index entropy via unconstrained vector quantization + eliminating inter-latent correlation via representation-domain contextual reparameterization." It theoretically proves that its R–D performance can approach entropy coding schemes, while practically saving 67.86% in bit-rate compared to MS-ILLM on Kodak/LPIPS with 10x faster decoding.

## Background & Motivation

**Background**: Modern learned image compression (LIC) follows the three-stage paradigm from Ballé 2018: VAE encoder + quantization + entropy coding. Performance-wise, it has surpassed JPEG/VVC, and the strongest models significantly outperform traditional codecs on perceptual metrics. Entropy coding (rANS), combined with context models to eliminate both statistical and correlation redundancy, represents the "last mile" of performance.

**Limitations of Prior Work**: Entropy coding (especially rANS) has complex control flows and is **inherently serial**, necessitating execution on the CPU. Entropy coding can take over 100 ms in a single forward pass, more than all other GPU modules combined. Simplifying or removing entropy coding immediately degrades performance—COIN uses INR to bypass entropy coding but only reaches JPEG levels, while OSCAR uses diffusion but incurs astronomical inference costs.

**Key Challenge**: From an information theory perspective, the end-to-end code length $R \ge H(X)$. Entropy coding exists to make the actual code length close to the entropy lower bound. Once removed, indices can only be encoded with fixed lengths, forcing the code length to be $\log K^n$. To ensure this upper bound is not wasted, the **index distribution must approach a uniform distribution** (maximum entropy), and there must be **no predictable correlation** between adjacent latents—two tasks that have not been systematically addressed in historical research.

**Goal**: Construct a completely GPU-friendly LIC framework that **does not call any entropy coder**, while maintaining R–D performance comparable to entropy coding schemes.

**Key Insight**: Address LIC redundancy by treating "statistical redundancy" and "correlation redundancy" separately. The former is managed by unconstrained VQ to push indices toward maximum entropy, and the latter is handled via representation-domain contextual affine reparameterization to "wash away" correlations. Both are tensor operators and are naturally GPU-parallelizable.

**Core Idea**: Instead of predicting conditional distributions and sending logits to an entropy coder, this method **directly in the representation domain** uses context-driven $(\bm\mu_i, \bm\sigma_i)$ to affine transform the current latent group $\bm y_i$ into a decorrelated space before quantization. Using a sufficiently large VQ codebook theoretically ensures $\Delta H \to 0$.

## Method

### Overall Architecture

The EF-LIC pipeline is as follows:

1.  **Main Encoder** $g_a$: Image $\bm x \in \mathbb{R}^{3 \times H \times W}$ → Latent variable $\bm y$, with a downsampling factor $f_y=16$.
2.  **Hyper-prior Branch**: $\bm z = h_a(\bm y)$, with a downsampling factor $f_z=64$. RVQ quantization yields $\hat{\bm z}$, and context features $\bm\phi = h_s(\hat{\bm z})$ are decoded.
3.  **Latent Grouping**: $\bm y$ is split into $N=4$ groups $(\bm y_1, \bm y_2, \bm y_3, \bm y_4)$ using a quadtree structure.
4.  **Representation-domain Decorrelation (RD)**: For each group $\bm y_i$, reference context $\bm \psi_i$ is calculated from decoded groups $\hat{\bm y}_{<i}$ and $\bm\phi$. Then affine parameters $(\bm\mu_i, \bm\sigma_i) = f_i^{\text{RD}}(\bm\psi_i)$ are derived to affine transform $\bm y_i$ into $\bm y_i' = \bm\sigma_i^{-1} \odot (\bm y_i - \bm\mu_i)$.
5.  **VQ Quantization**: $\bm y_i'$ is quantized using RVQ to obtain $\hat{\bm y}_i'$, then affine-transformed back to $\hat{\bm y}_i = \bm\sigma_i \odot \hat{\bm y}_i' + \bm\mu_i$.
6.  **Main Decoder** $g_s$: $\hat{\bm y} \to \hat{\bm x}$.
7.  **Multi-rate Support**: All RVQs share the same set of codebook counts $\mathcal{M} = \{1, 2, 3, 4, 5\}$. During inference, selecting $m \in \mathcal{M}$ provides different BPP: $\text{BPP} = \frac{m}{f_y^2} \left( \frac{f_y^2}{f_z^2} \log K_{\bm z} + \frac{1}{N} \sum_i \log K_i \right)$.

The entire pipeline **contains no entropy encoder/decoder**. All indices are sent as fixed-length codes, and all modules can be executed in a single batch on the GPU.

### Key Designs

1.  **Unconstrained VQ as Maximum-Entropy Probabilistic Shaping**:
    - **Function**: Ensures the entropy of the fixed-length VQ index sequence $J$ is close to the upper bound $n \log K$, such that the statistical redundancy $\Delta H = \frac{n \log K - H(J)}{n \log K} \to 0$.
    - **Mechanism**: During training, **no rate constraint is applied**; only codebook commitment, update loss, and reconstruction loss ($L1$ + LPIPS + PatchGAN) are used. Only quantization error is constrained without forcing index distributions. Proposition 3.1 proves via contradiction that for a fixed-length budget $R = \log K$, any distortion-optimal $Q^*$ must satisfy $\Delta H = 0$. Gersho's 1979 high-rate formula provides a weak version $p_J(j) \propto p_Y(\bm c_j)^{2/(C+2)}$, where $\Delta H \le 5\%$ for $C=8$.
    - **Design Motivation**: Empirically, index distributions of VQ-VAE / DAC are close to uniform after convergence. This paper **elevates this phenomenon to a theorem**, indicating that with a sufficiently large codebook and end-to-end reconstruction loss, VQ index sequences **theoretically no longer require entropy coding**—this is the legal foundation for removing entropy coding.

2.  **Representation-Domain Decorrelation instead of Probability-Domain Context Modeling**:
    - **Function**: Removes correlation between latent groups without predicting conditional probability distributions or calling entropy coding.
    - **Mechanism**: Traditional LIC uses a context model $f_i^{\text{CM}}$ to output conditional distribution parameters $(\bm\mu_i, \bm\sigma_i)$, which entropy coding use to compress $P_{\hat Y_i \mid \hat Y_{<i}}(\cdot; \bm\mu_i, \bm\sigma_i)$. EF-LIC directly performs affine transformation $\bm y_i' = \bm\sigma_i^{-1} \odot (\bm y_i - \bm\mu_i)$ in the representation domain using the same $(\bm\mu_i, \bm\sigma_i)$, replacing "predicting probability using context" with "whitening latents using context." Theorem 3.5 proves that when the codebook is sufficiently large and for any $\varepsilon \in (0,1)$, there exists an implementation such that $D_X^{\text{RD}}(R') \le D_X^{\text{CM}}(R)$ under a fixed-length budget $R' = R/(1-\varepsilon)$, meaning a slightly larger bit-rate yields the same R–D upper bound.
    - **Design Motivation**: By moving "context modeling" from the probability domain to the representation domain, the **entire pipeline becomes pure tensor operations**, completed in one forward batch pass without repeated CPU-GPU transfers of logits/probabilities. This is the true source of EF-LIC's speed jump.

3.  **Residual VQ + Shared Codebook Configuration for Single-Model Multi-rate**:
    - **Function**: A single model supports five bit-rate points by selecting $m \in \mathcal{M}$ at runtime.
    - **Mechanism**: All quantizers $Q_{\bm z}, \{Q_i^{\text{RD}}\}$ are implemented as Residual VQ (RVQ), each containing $m$ codebooks. During training, reconstruction loss is calculated and averaged for each $m \in \mathcal{M} = \{1, \dots, 5\}$ (Eq. 8). Codebook sizes decrease across layers ($K_1=1024, K_2=512, K_3=256, K_4=128, K_{\bm z}=1024$), naturally forming a coarse-to-fine multi-rate gradient.
    - **Design Motivation**: Multi-rate deployment is vital for practical codecs. The "stackable codebook" structure of RVQ allows multi-rate training with almost zero additional parameters, and switching bit-rates does not require changing checkpoints.

### Loss & Training
$\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{m \in \mathcal{M}} (\|\bm x - \hat{\bm x}_m\|_1 + \lambda_{\text{per}} \mathcal{L}_{\text{per}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{cb}} \mathcal{L}_{\text{cb}}^m)$, where $\mathcal{L}_{\text{per}}$ uses VGG-LPIPS, $\mathcal{L}_{\text{adv}}$ uses adaptive PatchGAN, and $\mathcal{L}_{\text{cb}}$ is the VQ-VAE commitment and codebook update loss. A 1% ImageNet subset was resampled each epoch with 256x256 random crops. Training used Adam $(\beta_1, \beta_2) = (0.5, 0.9)$, batch size 16, 2M iterations, learning rate $10^{-4} \to 10^{-5}$ at 1.5M, on a single A100 GPU with peak VRAM ~10.5 GB.

## Key Experimental Results

### Main Results (BD-rate vs. MS-ILLM, LPIPS, the more negative the better)

| Method | Enc. (ms) | Dec. (ms) | Params (M) | Kodak | DIV2K |
| :--- | :--- | :--- | :--- | :--- | :--- |
| VVC (VTM-23.10) | >9999 | 150.30 | — | +313.84% | +285.10% |
| HiFiC | 526.51 | 1408.60 | 181.6 | +45.82% | +46.36% |
| MS-ILLM | 165.38 | 147.79 | 181.4 | 0.00% | 0.00% |
| DiffEIC | 210.18 | 4661.74 | 1379.5 | −37.71% | −15.76% |
| OSCAR (diffusion, no EC) | 53.04 | 167.56 | 1009.3 | −37.31% | −14.51% |
| RDEIC | 157.25 | 426.68 | 1380.3 | −52.08% | −35.70% |
| **EF-LIC-s** | **9.94** | **6.26** | **11.51** | −55.38% | −47.36% |
| **EF-LIC** | **17.62** | **13.72** | **35.74** | **−67.86%** | **−62.33%** |

EF-LIC achieves the best BD-rate across Kodak, Tecnick, DIV2K, and CLIC2020. EF-LIC-s outperforms RDEIC even with 10x fewer parameters.

### Ablation Study (Kodak / LPIPS / 1M iter)

| Configuration | BD-rate | ΔFLOPs | Enc. (ms) | Dec. (ms) |
| :--- | :--- | :--- | :--- | :--- |
| VQ baseline (no decorr) | 0.00% | 0.00% | 5.51 | 7.06 |
| VQ + EC | −14.73% | +4.30% | 362.07 | 300.83 |
| UQ + EC (Typical LIC) | −20.73% | +7.53% | 63.12 | 71.72 |
| **EF-LIC** | **−22.20%** | +7.54% | 17.62 | **13.72** |
| EF-LIC-s | −10.76% | −56.30% | 9.94 | 6.26 |

Per-module runtime breakdown: In UQ+EC, entropy coding alone takes 108.60 ms (the bulk of the 71.72 ms decoding), while VQ+EC is even more extreme at 507.89 ms. EF-LIC completely skips this segment.

### Key Findings
- **The R–D performance of EF-LIC is slightly better than its entropy-coded variant UQ+EC** (−22.20% vs −20.73%), while encoding is 3.6x faster and decoding is 5.2x faster, verifying that the "no EC, no drop" theory in Theorem 3.5 is indeed achievable.
- **Entropy coding is the latency bottleneck**: The EC module accounts for 96.7% (507.89/525.09) of total decoding time in VQ+EC; removing it drops decoding time from 525 ms to 12.5 ms.
- **Representation-domain reparameterization contributes 22.2% BD-rate**: Simply adding the RD module to the VQ baseline (without entropy coding) achieves bit-rate savings equivalent to UQ+EC, illustrating that affine whitening and probability-domain context modeling exhibit **equivalent efficacy**.
- **EF-LIC-s demonstrates gains from decorrelation rather than computation**: Shrinking EF-LIC-s to have the same decoding latency as the VQ baseline (6.26 vs 7.06 ms) still yields a 10.76% improvement, ruling out the suspicion of "exchanging computation for performance."

## Highlights & Insights
- **The first theoretical removal of the "essential" entropy coding module**: Previously, the industry assumed "no entropy coding = poor performance." This paper uses Proposition 3.1 and Theorem 3.5 to provide clean proof: with a large enough codebook and sufficient training, the bit-rate upper bound of fixed-length VQ indices can losslessly approach the entropy coding lower bound, redefining the boundary of possibilities for LIC.
- **The duality between probability and representation domains is elegant**: Traditional context models use $(\bm\mu, \bm\sigma)$ as likelihood parameters for entropy coding. EF-LIC uses the same $(\bm\mu, \bm\sigma)$ as affine whitening parameters for VQ, **achieving equivalent decorrelation effects with the same context network**, but transforming the pipeline from serial to parallel. This "converting probability modeling to representation transformation" idea can be applied to audio/video codecs.
- **Small models can outperform diffusion models**: With 35.7M parameters, EF-LIC crushes the 1380M RDEIC and 1009M OSCAR. This indicates that after removing the entropy coding bottleneck, **asymmetric coding-decoding design budgets can be redistributed** to main codecs or contexts rather than stacking generators.
- **Highly deployable code**: All modules are conv / attention / vector-quantize, with no CPU bridge or rANS library dependencies. Complete 768x512 encoding/decoding is finished in 17.6/13.7 ms on a single A100, making it a strong candidate for real-time video and low-latency streaming scenarios.

## Limitations & Future Work
- Evaluation is biased towards **perceptual metrics (LPIPS/DISTS)**. Performance on PSNR/MS-SSIM is not presented in main tables; for applications requiring pixel-level precision like medical/scientific imaging, the advantages of this scheme might diminish.
- Theoretical support (Theorem 3.5) depends on "$K$ being sufficiently large and $e^{\text{RD}}, d^{\text{RD}}, Q^{\text{RD}}$ being sufficiently expressive," lacking quantification of actual gaps under small codebooks or limited-layer transformers.
- VQ failure cases (index collapse, dead codebooks) are not discussed in depth and represent known engineering risks, especially in high-residual levels of RVQ.
- The paper only validates on **images**. Whether higher-dimensional latents with temporal dimensions like **video/audio** satisfy the maximum entropy hypothesis "unconstrained VQ → $\Delta H \to 0$" requires further verification.
- Bit-rate error robustness is not discussed. While rANS provides an explicit byte stream, the degradation curve of fixed-length VQ index sequences under packet loss or bit errors is missing.

## Related Work & Insights
- **vs MS-ILLM / HiFiC**: Traditional generative LIC models use GANs to improve visual quality but depend on entropy coding. EF-LIC saves the EC while maintaining perceptual advantages, improving BD-rate by over 50%.
- **vs OSCAR / DiffEIC / RDEIC**: Although diffusion-based methods can "bypass entropy coding," they rely on INR or multi-step diffusion, making them 100x–1000x slower than EF-LIC with parameter counts an order of magnitude larger.
- **vs Control-GIC / Mao 2024 (VQ-GAN variants)**: These use VQ but ignore inter-latent correlation. Proposition 3.3 provides a theoretical guarantee that "adding RD never degrades performance," and experiments confirm RD brings a 22.2% BD-rate improvement.
- **vs UQ + Context + EC (LIC-HPCM / DCVC-RT)**: This paper replicates the efficacy of EC in the representation domain using affine reparameterization, skipping CPU-GPU switching. The insight is that any codec bottlenecked by CPU entropy coding during GPU inference could consider a similar approach.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First clean information-theoretic proof and SOTA implementation for "no entropy coding LIC without performance loss," making a paradigm-level contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 standard benchmarks + dual LPIPS/DISTS indicators + comprehensive ablation (VQ / VQ+EC / UQ+EC / EF-LIC / EF-LIC-s) + per-module timing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear alignment between theory (Prop 3.1, 3.3, Theorem 3.5) and experiments, with every core claim verified.
- Value: ⭐⭐⭐⭐⭐ Industrially viable; 35M parameters, 17 ms encode, 13 ms decode, 67% bit-rate gain, directly threatening existing LIC deployment pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Float8@2bits: Entropy Coding Enables Data-Free Model Compression](float82bits_entropy_coding_enables_data-free_model_compression.md)
- [\[AAAI 2026\] DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression](../../AAAI2026/model_compression/dynaquant_dynamic_mixed-precision_quantization_for_learned_i.md)
- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](../../ICCV2025/model_compression/learned_image_compression_with_hierarchical_progressive_context_modeling.md)
- [\[ACL 2026\] Efficient Learned Data Compression via Dual-Stream Feature Decoupling](../../ACL2026/model_compression/efficient_learned_data_compression_via_dual-stream_feature_decoupling.md)
- [\[ICML 2026\] Hierarchical Image Tokenization for Multi-Scale Image Super Resolution](hierarchical_image_tokenization_for_multi-scale_image_super_resolution.md)

</div>

<!-- RELATED:END -->
