---
title: >-
  [Paper Note] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution
description: >-
  [AAAI 2026][Image Generation][Image Super-Resolution] This paper introduces the sparse Mixture-of-Experts (MoE) paradigm into real-world image super-resolution…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Image Super-Resolution"
  - "Mixture of Experts"
  - "LoRA"
  - "Degradation-Aware"
  - "One-Step Diffusion"
date: 2026-05-08
content_hash: d0643fd887e9d2bd
---

# Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution

**Conference**: AAAI 2026
**arXiv**: [2511.16024](https://arxiv.org/abs/2511.16024)  
**Code**: None  
**Area**: Image Generation / Image Super-Resolution
**Keywords**: Image Super-Resolution, Mixture of Experts, LoRA, Degradation-Aware, One-Step Diffusion

## TL;DR

This paper introduces the sparse Mixture-of-Experts (MoE) paradigm into real-world image super-resolution, proposing a Mixture-of-Ranks (MoR) architecture that treats each LoRA rank as an independent expert. Combined with a CLIP-driven degradation estimation module and a degradation-aware load balancing loss, the method achieves one-step high-fidelity super-resolution reconstruction.

## Background & Motivation

### State of the Field

Real-world image super-resolution (Real-ISR) is a fundamental task in computer vision, requiring the recovery of high-resolution details from low-resolution images corrupted by complex degradations such as blur and noise. Diffusion models have become the dominant approach for Real-ISR owing to their powerful generative priors; however, their iterative multi-step sampling incurs substantial computational latency. Recent one-step methods (e.g., OSEDiff, S3Diff) accelerate inference via distillation or LoRA fine-tuning of pretrained diffusion models, yet still face fundamental limitations.

### Limitations of Prior Work

**Dense models struggle with heterogeneous degradations**: Degradation types (blur, noise) and severity vary greatly across Real-ISR samples, but existing dense LoRA fine-tuning methods apply identical parameters to all inputs, failing to adaptively capture heterogeneous degradation characteristics.

**Inflexible computational resource allocation**: Simple and complex degradation samples consume the same computational budget, wasting resources on simple cases while limiting capacity for complex ones.

**Insufficient knowledge sharing**: Common knowledge exists across different degradation types (e.g., basic texture restoration), but dense models cannot achieve knowledge sharing within an equivalent computational budget.

### Root Cause

How can a model, under a fixed computational budget, **dynamically allocate computation** according to the severity and type of input degradation—allocating less for simple samples and more for complex ones?

### Core Idea

Inspired by sparse MoE architectures in LLMs such as DeepSeek, this paper introduces the MoE concept into LoRA fine-tuning. Rather than treating an entire LoRA module as a single expert, **each individual rank is treated as an independent expert** (fine-grained expert segmentation). A CLIP-driven degradation estimation module provides degradation-aware signals for routing, while a zero-expert mechanism and a degradation-aware load balancing loss enable dynamic computation allocation.

## Method

### Overall Architecture

MoR-DASR is built upon Stable Diffusion 2.1 and alternates between two training stages:
1. Optimizing the variational score network $\epsilon_\psi$ to fit the generated sample distribution (diffusion loss $\mathcal{L}_{diff}$)
2. Fine-tuning the diffusion model $\epsilon_\theta$ with MoR modules and encoder $Enc_\theta$ (via reconstruction loss $\mathcal{L}_{rec}$, VSD loss $\mathcal{L}_{VSD}$, and GAN loss $\mathcal{L}_{GAN}$)

At inference, the LR image is directly fed into the encoder and passes through a single forward pass of the diffusion network to produce the HR output.

### Key Designs

#### 1. **Mixture-of-Ranks (MoR) Architecture**

**Function**: Treats each rank in LoRA as an independent expert, enabling fine-grained expert segmentation and knowledge recombination.

**Mechanism**: Conventional LoRA MoE treats a complete LoRA module ($A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$) as a single expert unit—a coarse-grained strategy that limits flexible decomposition and recombination of knowledge. MoR designates each rank ($A_i, B_i$, rank=1) as an independent expert, divided into two categories:

- **Shared Experts**: Fixed-position ranks that are always activated, capturing universal knowledge (e.g., basic texture restoration, fundamental denoising operations)
- **Routed Experts**: Ranks dynamically activated via a gating mechanism and top-$k$ selection strategy

Forward pass formula:
$$MoR(z_t) = W_0 x + \sum_{i=1}^{k} g_i(s) B_i A_i x + \sum_{j=1}^{m} B_j A_j x$$

where $g_i(s)$ denotes routing weights, $k$ is the number of activated routed experts, and $m$ is the number of shared experts.

**Design Motivation**: Analogous to the two key innovations in DeepSeek-MoE—fine-grained expert segmentation and shared expert isolation. Fine granularity enables more flexible knowledge combination, while shared experts capture universal features across degradation types and reduce routing redundancy.

**Configuration**: 40 total ranks—8 shared experts + 32 routed experts; top-8 routing strategy during training.

#### 2. **Degradation Estimation Module**

**Function**: Leverages CLIP's cross-modal alignment capability to compute a degradation severity score for the input image, which guides expert selection in the router.

**Mechanism**: Positive/negative text prompt pairs are defined across multiple dimensions (overall quality, blurriness, noise, resolution, edge sharpness, detail, etc.; 7 dimensions in total). The CLIP image and text encoders compute cosine similarities between the LR image and each prompt pair, which are normalized to produce a degradation score:

$$s_i = \frac{\exp(d^{(i,n)})}{\exp(d^{(i,p)}) + \exp(d^{(i,n)})}$$

where $d^{(i,p)}$ and $d^{(i,n)}$ are the cosine distances between the image and the positive/negative prompts of dimension $i$, respectively. A higher $s_i$ indicates more severe degradation.

**Design Motivation**: Semantic-feature-driven data partitioning is unsuitable for Real-ISR (prior work has shown that degradation characteristics are more critical). Severely degraded images require more computation for reconstruction, while mildly degraded images do not; degradation severity is therefore the most appropriate basis for dynamic computation allocation.

#### 3. **Zero-Expert and Degradation-Aware Load Balancing Loss**

**Function**: Introduces "zero experts" (virtual experts with zero output) and an improved load balancing loss to dynamically adjust the number of activated experts based on degradation severity.

**Mechanism**: Several zero-expert slots are added among the 32 routed experts. The conventional load balancing loss treats zero experts and real experts identically, assigning zero experts a fixed activation probability of approximately $k/N$. The improved degradation-aware load balancing loss is:

$$\mathcal{L}_{balance} = N \sum_{i=1}^{N} \alpha_i f_i p_i$$

where:
$$\alpha_i = \begin{cases} \alpha & \text{if } i \leq n \text{ (real experts)} \\ s \cdot \alpha & \text{if } i > n \text{ (zero experts)} \end{cases}$$

When the degradation score $s$ is large (severe degradation), the penalty weight on zero experts increases, encouraging the model to activate more real experts; when $s$ is small, the penalty on zero experts decreases, allowing the model to activate more zero experts and conserve computation.

**Design Motivation**: Addresses two issues: (1) samples with different degradation severities require different computational budgets; (2) the optimal LoRA rank may differ across network layers. Zero experts enable adaptive computation allocation.

### Loss & Training

Total loss function:
$$\mathcal{L}_\theta = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{VSD} + \lambda_2 \mathcal{L}_{GAN} + \mathcal{L}_{balance}$$

- $\mathcal{L}_{rec}$: L2 + LPIPS reconstruction loss
- $\mathcal{L}_{VSD}$: Variational score distillation loss (aligning with the high-quality prior of the pretrained diffusion model)
- $\mathcal{L}_{GAN}$: Multi-scale discriminator loss utilizing features from the pretrained diffusion model
- $\mathcal{L}_{balance}$: Degradation-aware load balancing loss

Training details: LSDIR + 10k FFHQ face images, Real-ESRGAN degradation pipeline, batch size 16, learning rate 5e-5, 25,000 iterations.

## Key Experimental Results

### Main Results

Quantitative comparison with one-step Real-ISR methods across three datasets:

| Dataset | Method | PSNR↑ | LPIPS↓ | CLIPIQA↑ | MUSIQ↑ | MANIQA↑ | TOPIQ↑ | TRES↑ |
|--------|------|-------|--------|----------|--------|---------|--------|-------|
| DIV2K-Val | OSEDiff | 23.92 | 0.310 | 0.659 | 67.71 | 0.435 | 0.606 | 78.40 |
| DIV2K-Val | S3Diff | 23.53 | **0.258** | 0.699 | 67.92 | 0.452 | 0.633 | 80.72 |
| DIV2K-Val | **MoR-DASR** | **24.01** | 0.289 | 0.681 | **68.09** | **0.475** | **0.663** | **84.14** |
| RealSR | OSEDiff | 25.26 | 0.301 | 0.651 | 68.41 | 0.468 | 0.614 | 80.18 |
| RealSR | **MoR-DASR** | 25.32 | 0.291 | **0.691** | **69.78** | **0.512** | **0.662** | **84.97** |
| DRealSR | OSEDiff | **28.29** | 0.302 | 0.673 | 64.47 | 0.469 | 0.616 | 76.76 |
| DRealSR | **MoR-DASR** | 28.37 | 0.307 | **0.717** | **65.94** | **0.509** | **0.652** | **81.78** |

Comparison with multi-step Real-ISR methods (key results):

| Dataset | Method | Steps | CLIPIQA↑ | MANIQA↑ | TRES↑ |
|--------|------|------|----------|---------|-------|
| DIV2K-Val | SeeSR | 50 | 0.693 | **0.504** | **85.80** |
| DIV2K-Val | MoR-DASR | **1** | 0.681 | 0.475 | 84.14 |
| RealSR | SeeSR | 50 | 0.669 | **0.540** | **88.60** |
| RealSR | MoR-DASR | **1** | **0.691** | 0.512 | 84.97 |

MoR-DASR achieves comparable quality to SeeSR's 50-step inference with a single step, representing approximately **40×** speedup.

### Ablation Study

Ablation of MoR architecture (on DRealSR):

| Configuration | CLIPIQA↑ | MANIQA↑ | TRES↑ | Description |
|------|----------|---------|-------|------|
| LoRA (baseline) | 0.670 | 0.481 | 78.81 | Standard LoRA fine-tuning |
| LoRA+MoE | 0.689 | 0.484 | 79.36 | Conventional MoE, entire LoRA as expert |
| MoR-v1 | 0.704 | 0.491 | 80.32 | MoR without zero experts |
| MoR-v2 | 0.699 | 0.479 | 79.41 | MoR with zero experts but conventional balancing loss |
| **MoR-full** | **0.717** | **0.509** | **81.78** | Full proposed method |

Key observations:
- LoRA → LoRA+MoE: Validates the effectiveness of MoE for Real-ISR
- LoRA+MoE → MoR-v1: Fine-grained rank-level experts outperform module-level experts
- MoR-v1 → MoR-v2: Introducing zero experts with the conventional balancing loss alone **degrades performance**
- MoR-v2 → MoR-full: The degradation-aware balancing loss is the key to making zero experts effective; final gains of CLIPIQA +7%, MANIQA +5.8%, TRES +3.8%

### Key Findings

1. **Degradation-adaptive activation**: Images with more severe degradation activate fewer zero experts (i.e., more real experts), consistent with intuition.
2. **Layer-wise computational variation**: Different network layers require different numbers of ranks; certain layers activate only zero experts (LoRA not needed), while others activate none, validating the necessity of fine-grained MoR.
3. **Perceptual metric advantage**: MoR-DASR achieves the best performance on perceptual metrics such as MANIQA, TOPIQ, and TRES, which are highly correlated with human visual perception.

## Highlights & Insights

1. **Successful transfer of LLM MoE to visual tasks**: The fine-grained expert segmentation and shared expert isolation concepts from DeepSeek-MoE prove equally effective in Real-ISR.
2. **Degradation-aware routing is the core innovation**: Routing based on degradation severity rather than semantic features is better suited to Real-ISR than image content-based routing.
3. **Elegant zero-expert mechanism**: By introducing virtual experts with zero output, the method achieves a "dynamic rank" effect, allowing each layer and each sample to adaptively determine its computational load.
4. **One-step inference efficiency**: Compared to multi-step methods (SeeSR at 50 steps), the method achieves a 40× speedup at comparable quality.

## Limitations & Future Work

1. **Manual design of degradation prompts**: The positive/negative prompt pairs for CLIP-based degradation estimation are hand-crafted across 7 dimensions and may not cover all degradation types.
2. **Validation limited to SD 2.1**: The effectiveness of MoR on more recent backbones such as SDXL or SD 3.0 has not been verified.
3. **Training overhead**: Compared to standard LoRA fine-tuning, the routing mechanism and degradation estimation module increase training complexity.
4. **Expert count and top-$k$ configuration**: The configuration of 40 ranks and top-8 was determined empirically, lacking systematic hyperparameter analysis.

## Related Work & Insights

- **DeepSeek-MoE**: Directly inspired MoR's fine-grained expert segmentation and shared expert isolation design.
- **OSEDiff**: Provides the foundational one-step LR-to-HR direct mapping framework.
- **CLIP for IQA**: The degradation estimation module draws on the CLIP-IQA approach of leveraging CLIP's cross-modal capability for image quality assessment.
- **Implications for other tasks**: The MoR concept is generalizable to other vision tasks involving heterogeneous inputs, such as image restoration tasks including deblurring, denoising, and deraining.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of MoE with LoRA ranks and the degradation-aware routing design are novel
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, multiple comparison methods, detailed ablations and visualizations
- **Writing Quality**: ⭐⭐⭐⭐ — Method motivation is clear and figures are intuitive
- **Value**: ⭐⭐⭐⭐ — Achieves significant progress in one-step super-resolution; the MoR idea is broadly transferable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [\[AAAI 2026\] Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution](continuous_degradation_modeling_via_latent_flow_matching_for_real-world_super-re.md)
- [\[AAAI 2026\] QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution](quantvsr_low-bit_post-training_quantization_for_real-world_video_super-resolutio.md)
- [\[NeurIPS 2025\] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](../../NeurIPS2025/image_generation/dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)
- [\[CVPR 2026\] OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](../../CVPR2026/image_generation/oars_process-aware_online_alignment_for_generative_real-world_image_super-resolu.md)

</div>

<!-- RELATED:END -->
