---
title: >-
  [Paper Note] Degradation-Aware Metric Prompting for Hyperspectral Image Restoration
description: >-
  [ICML 2026][Image Restoration][Hyperspectral Image Restoration] DAMP utilizes six interpretable spatial-spectral physical metrics (e.g., high-frequency energy ratio, texture uniformity…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "Hyperspectral Image Restoration"
  - "Degradation-Aware Prompting"
  - "Interpretable Metrics"
  - "Mixture-of-Experts"
  - "Zero-shot Generalization"
date: 2026-05-08
content_hash: 8c1ed7245cfb1b06
---

# Degradation-Aware Metric Prompting for Hyperspectral Image Restoration

**Conference**: ICML 2026  
**arXiv**: [2512.20251](https://arxiv.org/abs/2512.20251)  
**Code**: https://github.com/MiliLab/DAMP (Available)  
**Area**: Image Restoration / Hyperspectral Imaging / Unified Restoration  
**Keywords**: Hyperspectral Image Restoration, Degradation-Aware Prompting, Interpretable Metrics, Mixture-of-Experts, Zero-shot Generalization

## TL;DR
DAMP utilizes six interpretable spatial-spectral physical metrics (e.g., high-frequency energy ratio, texture uniformity, spectral curvature) as "Degradation Prompts" (DP) instead of black-box embeddings or explicit labels. These DPs serve as gating signals to drive a Spatial-Spectral Adaptive MoE, selecting specific "spatial/spectral experts." The method achieves SOTA performance across five HSI restoration tasks and demonstrates zero-shot generalization to unseen degradations (motion blur, Poisson noise).

## Background & Motivation
**Background**: Hyperspectral images (HSI) record the spectral response of materials across hundreds of contiguous bands, but are prone to various degradations such as low SNR, motion blur, stripe artifacts, band loss, and compression. Early methods trained specialized networks for each degradation; later, inspired by "Unified Image Restoration (UIR)" frameworks like PromptIR and InstructIR, models like PromptHSI and MP-HSIR began adopting the "one model for multiple degradations" paradigm.

**Limitations of Prior Work**: Current unified HSI restoration methods follow two problematic paths:
- **Explicit Prior-based** (PromptHSI/MP-HSIR): Require externally provided degradation labels or text descriptions. In real-world scenarios, the specific combination and severity of "blur + stripes + band loss" are often unknown.
- **Implicit Black-box-based** (PromptIR/DFPIR): Encode a latent prompt directly from the input. These force unseen degradations onto the manifold of the training distribution, leading to poor generalization and lack of explicit mechanisms for modeling spectral correlation, which results in low spectral fidelity.

**Key Challenge**: HSI degradation is physically "continuous, mixed, and cross-dimensional" (texture destruction in the spatial dimension and spectral curve distortion in the spectral dimension). However, existing prompts are either discrete classes (discontinuous) or uninterpretable latents (dimension-agnostic). This geometric mismatch between the prompt space and the physical structure of degradation leads to failures in both generalization and interpretability.

**Goal**: Construct a degradation representation that is **independent of external labels, interpretable, cross-dimensional, and naturally continuous for unseen degradations**, enabling the network to allocate computational resources "on-demand" (e.g., when to reconstruct spatial textures vs. when to restore spectral continuity).

**Key Insight**: The authors conducted a pilot experiment on 1,000 degraded HSIs using three simple physical metrics: High-Frequency Energy Ratio (HFER), Spatial Texture Uniformity (STU), and Spectral Curvature Mean (SCM). A Random Forest classifier could clearly distinguish five types of degradation using these metrics. Simultaneously, different degradations showed overlapping distributions in specific metrics (e.g., mild blur and low noise have similar SCM). This indicates that **a few interpretable metrics can identify degradation identity while naturally reflecting commonalities between degradations**—the former ensures interpretability, while the latter enables generalization.

**Core Idea**: Replace "degradation prompts" from black-box embeddings or category labels with a **multi-dimensional physical metric vector** (Degradation Prompt, DP). This DP serves as the gating signal for a Mixture-of-Experts (MoE) module, forcing the routing logic to anchor explicitly on physical rules (e.g., "higher high-frequency energy $\Rightarrow$ bias towards spectral filtering experts"), thereby addressing interpretability, mixed degradation, and zero-shot generalization simultaneously.

## Method

### Overall Architecture
DAMP is a hierarchical U-Net-style unified HSI restoration network that takes a degraded HSI $\mathcal{Y}$ and outputs a clean HSI $\hat{\mathcal{X}} = \mathcal{R}_\theta(\mathcal{Y})$. The pipeline consists of two parallel flows:

- **DP Extraction Flow**: Calculates six spatial-spectral physical metrics directly from the input $\rightarrow$ projects them into an embedding space to obtain a DP vector $\mathbf{e} \in \mathbb{R}^d$. This vector is utilized throughout all decoder layers.
- **Feature Restoration Flow**: Extracts shallow features via $3\times 3$ convolutions $\rightarrow$ 4-level hierarchical encoder (standard attention blocks) $\rightarrow$ 4-level decoder, where each decoder stage replaces standard blocks with DAMoE, dynamically adjusted by the DP as a global condition. Residual fusion of input and decoded features yields the output $\hat{\mathcal{X}}$.

The non-trivial design lies in the selection of DPs, how DAMoE uses DPs for routing, and the division of labor between spatial and spectral components within each expert (SSAM).

### Key Designs

1.  **Degradation Prompt (DP): Interpretable Metric-based Representation**:
    - **Function**: Encodes any degraded HSI into a 6-dimensional physically interpretable vector as a global condition.
    - **Mechanism**: Starting from 25 candidate metrics (entropy, gradients, frequency statistics), a three-stage screening is performed: (i) **Interpretability screening**: Remove abstract statistics without clear physical correlates; (ii) **Spatial-spectral coverage screening**: Ensure representation of both spatial structure and spectral fidelity; (iii) **Separability screening**: Select top metrics via Random Forest feature importance. The final six are: High-Frequency Energy Ratio HFER $=\frac{1}{C}\sum_c \frac{\sum_{(u,v)\in\Omega_H}|\mathcal{F}[x_c]|^2}{\sum_{(u,v)}|\mathcal{F}[x_c]|^2}$, Spatial Texture Uniformity (STU), Spectral Curvature Mean SCM $=\frac{1}{C-2}\sum_i|\nabla^2 s_i|$, Spectral Curvature Std, Gradient Std, and Spatial Correlation Coefficient. These are projected to obtain the DP embedding $\mathbf{e}$.
    - **Design Motivation**: HFER reflects the "degree of high-frequency detail destruction," sensitive to noise, blur, and downsampling. SCM reflects whether the "spectral curve is smooth," identifying band loss or distortion. Since these are **objective physical indicators** unconstrained by the training distribution, the DP for unseen degradations (like Poisson noise) remains in a reasonable numerical range, preventing incorrect classification and enabling zero-shot generalization.

2.  **Degradation-Adaptive MoE (DAMoE): DP-Driven Physical Gating**:
    - **Function**: Dynamically selects a top-$k$ combination of experts in each decoder stage, driven explicitly by the DP.
    - **Mechanism**: For input features $\mathbf{x}$, the spatial dimensions are squeezed via GAP into a global vector, concatenated with DP embedding $\mathbf{e}$, and passed through two projection layers + softmax + top-$k$ sparsification to get gating scores $\mathbf{g} = \mathcal{T}_k(\text{softmax}(\mathbf{W}_g \cdot \sigma(\mathbf{W}_{proj}[\text{GAP}(\mathbf{x}), \mathbf{e}]) + \epsilon))$. Noise $\epsilon \sim \mathcal{N}(0,1)$ is added during training for load balancing. Final features are $\mathbf{f}_{deg} = \sum_{i \in \mathcal{K}} g_i \cdot \mathbf{f}_i$, fused with "degradation-agnostic features" from a shared expert.
    - **Design Motivation**: Unlike MoE modules using visual feature routing (e.g., MoCE-IR), DAMoE routing is anchored by "physical interpretability + input conditions." If HFER is high (heavy noise), the gate explicitly favors experts skilled in spectral filtering, maintaining stable routing even when visual features are severely blurred.

3.  **SSAM (Spatial-Spectral Adaptive Module): Specialized Experts via Learned Coefficients**:
    - **Function**: Acts as the expert operator in DAMoE, where each expert uses learnable coefficients to determine its specialization in spatial texture or spectral fidelity.
    - **Mechanism**: Each expert has two parallel branches: $\mathcal{E}_s$ uses Window-based Multi-head Self-Attention for spatial structure, and $\mathcal{E}_c$ uses 1D convolution for inter-band correlation. The $i$-th expert output is $\mathbf{F}_{expert}^{(i)} = \lambda_s^{(i)} \mathcal{E}_s(\mathbf{F}) + \lambda_c^{(i)} \mathcal{E}_c(\mathbf{F})$, where $\lambda_s^{(i)} + \lambda_c^{(i)} = 1$. **Key constraint**: $\lambda_s^{(i)}, \lambda_c^{(i)}$ are **expert-specific learnable parameters**, not instance-specific predictions. Each expert "takes a side" during training, naturally evolving into "spatial experts" (large $\lambda_s$) or "spectral experts" (large $\lambda_c$).
    - **Design Motivation**: Spatial and spectral HSI degradations are often asynchronous (blur destroys space but leaves curves relatively intact; noise destroys both). By making weights expert-wise, experts are forced to diverge, allowing the router to select the optimal spatial/spectral ratio based on the DP.

### Loss & Training
The model uses L1 loss: $\mathcal{L} = \|\hat{\mathcal{X}} - \mathcal{X}\|_1$. Gating noise is the primary load-balancing mechanism. Training uses AdamW ($\beta_1=0.9, \beta_2=0.999$), lr $=1\times 10^{-4}$, batch size 4; 3000 epochs for natural scenes and 1500 for remote sensing HSIs on an RTX 4090.

## Key Experimental Results

### Main Results
Comprehensive comparison of PSNR/SSIM/SAM for 5 unified restoration tasks (Table 2, units: dB / – / °):

| Task (Dataset) | MP-HSIR | PromptIR | MoCE-IR | **DAMP** | Gain |
|---|---|---|---|---|---|
| Gaussian Deblur (ARAD) | 44.58 / 0.984 / 0.900 | 49.18 / 0.996 / 0.822 | 50.52 / 0.996 / 0.673 | **52.84 / 0.998 / 0.508** | +2.32 dB |
| Super-Resolution (ARAD) | 41.77 / 0.972 / 1.142 | 40.57 / 0.966 / 1.168 | 40.62 / 0.967 / 1.110 | **44.01 / 0.981 / 0.866** | +2.24 dB |
| Inpainting (Xiong'an) | 33.42 / 0.697 / 11.13 | 31.36 / 0.579 / 13.60 | 29.04 / 0.518 / 15.79 | **33.62 / 0.711 / 10.98** | +0.20 dB |
| Gaussian Denoise (ICVL) | 42.16 / 0.968 / 3.030 | 42.35 / 0.970 / 2.659 | 42.66 / 0.973 / 2.434 | **42.86 / 0.974 / 2.229** | +0.20 dB |
| Avg. on ARAD (5 tasks) | 47.85 / 0.984 / 1.608 | 47.20 / 0.984 / 1.510 | 48.72 / 0.985 / 1.203 | **51.43 / 0.989 / 0.936** | +2.71 dB |
| Avg. on RS Data | 38.33 / 0.839 / 12.73 | 38.19 / 0.812 / 13.25 | 36.78 / 0.774 / 15.09 | **39.42 / 0.851 / 10.11** | +1.09 dB |

Zero-shot results (on CAVE, unseen degradations, Table 3):

| Method | Motion Blur PSNR/SSIM | Poisson Denoise PSNR/SSIM |
|---|---|---|
| PromptIR | 30.53 / 0.881 | 21.98 / 0.442 |
| MoCE-IR | 30.34 / 0.878 | 19.51 / 0.401 |
| MP-HSIR | 23.63 / 0.688 | 16.96 / 0.240 |
| **DAMP** | **31.05 / 0.899** | **24.08 / 0.538** |

The +2.10 dB zero-shot gain in Poisson denoising demonstrates that the DP physical metrics are not "locked" to the training distribution.

### Ablation Study
Component ablation (Table 4, average over 5 tasks on ARAD):

| Config | PSNR (dB) | SSIM | Insight |
|---|---|---|---|
| Baseline (No DP, No SSAM) | 45.82 | 0.976 | Reverts to standard U-Net |
| + DP | 50.02 | 0.986 | **+4.20 dB**, DP is the primary contributor |
| + DP + SSAM (Full) | **51.43** | **0.989** | +1.41 dB, SSAM adds further gain |

Routing strategy ablation (Table 5):

| Routing Signal | PSNR (dB) | Gap with DP |
|---|---|---|
| Frequency-based (as in MoCE-IR) | 47.72 | −3.71 |
| Degradation Type (Category Label) | 46.27 | −5.16 |
| Implicit Prompt (as in PromptIR) | 46.81 | −4.62 |
| **DP (Ours)** | **51.43** | – |

### Key Findings
- Adding DP alone increases performance by 4.20 dB, far exceeding SSAM's 1.41 dB—**the true innovation is the degradation representation, not the MoE architecture itself.**
- Category label routing performs 0.5 dB worse than implicit prompt routing, suggesting "hard classification" loses continuity information. DP captures both continuity and interpretability.
- Spectral error analysis across bands (Fig. 6) shows SSAM achieves the lowest errors in all tasks, proving that expert-wise learned $\lambda_s/\lambda_c$ allows spectral experts to function effectively.

## Highlights & Insights
- **Pulling "prompts" back from semantics to physics**: While the UIR field focuses on text/visual/implicit prompts, DAMP uses closed-form frequency and curvature statistics. This "physicalization of prompts" is applicable to any inverse problem with a physical model (medical imaging, low-dose CT, seismic signals).
- **Expert-wise rather than instance-wise coefficients**: Forcing expert combination coefficients to be fixed learnable parameters sacrifices individual expert flexibility for better differentiation, providing the router with truly diverse choices.
- **Routing signals determine the MoE ceiling**: Changing routing signals causes a 3-5 dB performance swing, more significant than modifying the expert architecture.

## Limitations & Future Work
- **Hand-picked metrics**: The 6-D DP is determined via Random Forest and three-stage screening, which may carry human bias. A natural extension would be making the metric pool a learnable dictionary.
- **Lack of coupling with explicit physical models**: DP describes degradation severity but does not invert the operator $\mathcal{D}(\cdot)$. Adding a light inversion head (e.g., estimating blur kernels) could provide diagnostic information.
- **Separate training for Natural vs. Remote Sensing scenes**: Universal restoration should ideally be cross-domain; future work could use DP as a cross-domain bridge since metrics are sensor-independent.

## Related Work & Insights
- **vs. PromptIR / InstructIR**: Both use prompts, but PromptIR uses implicit embeddings and InstructIR uses text. DAMP's low-dimensional physical prompts are objective and lead to 2+ dB gains in zero-shot scenarios.
- **vs. MP-HSIR / PromptHSI**: These rely on external labels unavailable in real scenarios. DAMP is self-contained.
- **vs. MoCE-IR**: Similar MoE structure, but MoCE-IR uses only spatial frequency for routing. DAMP uses spatial-spectral physical quantities and forces expert specialization via expert-wise weights, leading to a 2.71 dB average gain in HSI tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Replacing black-box prompts with physical metrics is a clear conceptual shift in HSI UIR.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 5 tasks, 8 datasets, zero-shot tests, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and rich visualizations.
- Value: ⭐⭐⭐⭐ Direct design utility for HSI/multi-spectral/medical imaging researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](../../CVPR2026/image_restoration/drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[ICCV 2025\] MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration](../../ICCV2025/image_restoration/mp-hsir_a_multi-prompt_framework_for_universal_hyperspectral_image_restoration.md)
- [\[ICML 2026\] Image Restoration via Diffusion Models with Dynamic Resolution](image_restoration_via_diffusion_models_with_dynamic_resolution.md)
- [\[ICCV 2025\] Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](../../ICCV2025/image_restoration/towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)
- [\[ICML 2026\] DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs](dapd_dependency-aware_parallel_decoding_via_attention_for_diffusion_llms.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](../../CVPR2026/image_restoration/drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[CVPR 2025\] Degradation-Aware Feature Perturbation for All-in-One Image Restoration](../../CVPR2025/image_restoration/degradation-aware_feature_perturbation_for_all-in-one_image_restoration.md)
- [\[CVPR 2025\] DPIR: Dual Prompting Image Restoration with Diffusion Transformers](../../CVPR2025/image_restoration/dpir_dual_prompting_restoration_dit.md)
- [\[ICCV 2025\] MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration](../../ICCV2025/image_restoration/mp-hsir_a_multi-prompt_framework_for_universal_hyperspectral_image_restoration.md)
- [\[ICML 2026\] DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs](dapd_dependency-aware_parallel_decoding_via_attention_for_diffusion_llms.md)

</div>

<!-- RELATED:END -->
