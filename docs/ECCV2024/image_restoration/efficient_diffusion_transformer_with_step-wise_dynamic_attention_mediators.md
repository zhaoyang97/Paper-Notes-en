---
title: >-
  [Paper Note] Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators
description: >-
  [ECCV 2024][Image Restoration][Diffusion Transformer] Discovering that query-key interactions in Diffusion Transformers exhibit significant redundancy (especially in the early stages of denoising), this work proposes the Attention Mediator mechanism to reduce attention complexity to linear. It further designs a step-wise dynamic adjustment strategy, achieving a state-of-the-art FID of 2.01 on SiT-XL/2 while reducing computational overhead.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Diffusion Transformer"
  - "Attention Mediators"
  - "Dynamic Networks"
  - "Linear Attention"
  - "Denoising Redundancy"
date: 2026-05-08
content_hash: 23b2e1fe1fccdd03
---

# Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators

**Conference**: ECCV 2024  
**arXiv**: [2408.05710](https://arxiv.org/abs/2408.05710)  
**Code**: Yes ([https://github.com/LeapLabTHU/Attention-Mediators](https://github.com/LeapLabTHU/Attention-Mediators))  
**Area**: Image Restoration  
**Keywords**: Diffusion Transformer, Attention Mediators, Dynamic Networks, Linear Attention, Denoising Redundancy

## TL;DR

Discovering that query-key interactions in Diffusion Transformers exhibit significant redundancy (especially in the early stages of denoising), this work proposes the Attention Mediator mechanism to reduce attention complexity to linear. It further designs a step-wise dynamic adjustment strategy, achieving a state-of-the-art FID of 2.01 on SiT-XL/2 while reducing computational overhead.

## Background & Motivation

### Efficiency Dilemma of Diffusion Transformers

Diffusion Transformers (DiTs) are replacing U-Net as the mainstream backbone for diffusion models due to their simplicity, effectiveness, and scalability, driving applications such as Stable Diffusion V3, PixArt-$\alpha/\Sigma/\delta$, Huawei DiT, and Sora. However, a major criticism of DiTs is the **high computational consumption of the global attention mechanism**. The $O(N^2C)$ complexity of self-attention acts as an inference bottleneck, seriously hindering practical deployment for high-resolution images and long videos.

Although various attention acceleration approaches (such as window attention and linear attention) exist in the field of visual recognition, **attention efficiency optimization in the domain of diffusion generation remains virtually unexplored**.

### Key Observation: Attention Redundancy in the Denoising Process

Through quantitative analysis, this work discovers two key phenomena in DiTs:

**Observation 1: A high volume of query-key redundancy is prevalent**. Across all self-attention layers, the attention distributions of different queries over keys are highly similar. For instance, in the 10th layer of DiT-S/2, the internal distances between all queries are close to zero during the initial steps, indicating they are completely homogenized.

**Observation 2: Redundancy decreases as the denoising process progresses**. Attention redundancy is most severe in the early stages of denoising (pure noise phase). As denoising proceeds, queries become increasingly diverse. This implies that the **full one-to-one attention interactions in the early stages are unnecessary**.

### Metric for Quantifying Redundancy

This work designs a redundancy metric using Jensen-Shannon Divergence (JSD). Treating each row of the attention matrix $\mathbf{A}^{(m)}$ as a probability distribution (the weight distribution of a query over all keys), the redundancy score of the $l$-th layer is calculated as:

$$S_l = \frac{2}{MN(N-1)} \sum_{m=1}^{M} \sum_{i_1=1}^{N-1} \sum_{i_2=i_1+1}^{N} \mathcal{D}_{\text{JS}}(\mathbf{A}_{i_1}^{(m)}, \mathbf{A}_{i_2}^{(m)})$$

Lower $S_l$ indicates highly similar attention distributions, representing severe redundancy. Experiments measure $S_l$ across all layers and timesteps on DiT-S/2 and SiT-S/2, validating the two observations mentioned above.

## Method

### Overall Architecture

A set of auxiliary **mediator tokens (Attention Mediators)**, whose count is far smaller than the original tokens (e.g., <10%), is introduced into the standard self-attention layer to interact separately with queries and keys. At the same time, the number of mediators is horizontally and dynamically adjusted based on the degree of redundancy across denoising timesteps—fewer in the early stages and more in the later stages.

### Key Designs

#### 1. **Attention Mediators Mechanism**

**Function**: Compress the redundant query-key interactions using a small set of mediator tokens $\mathbf{t}^{(m)} \in \mathbb{R}^{n \times d}$ ($n \ll N$).

**Mechanism**: Split the standard attention's single-step Q-K-V interaction into two steps:

**Step 1**: Mediators aggregate key information ($n \times N$ interaction):
$$\mathbf{v}_{\text{med}}^{(m)} = \text{Softmax}\left(\frac{\mathbf{t}^{(m)} \mathbf{k}^{(m)\top}}{\sqrt{d}}\right) \mathbf{v}^{(m)}$$

**Step 2**: Queries extract information from the mediators ($N \times n$ interaction):
$$\mathbf{h}^{(m)} = \text{Softmax}\left(\frac{\mathbf{q}^{(m)} \mathbf{t}^{(m)\top}}{\sqrt{d}}\right) \mathbf{v}_{\text{med}}^{(m)}$$

Generation of mediator tokens: Adaptive pooling is performed on the query tokens. Specifically, they are reshaped to the latent image shape $\mathbb{R}^{H \times W \times d}$, pooled in the spatial dimension to $\mathbb{R}^{h \times w \times d}$, and then flattened to obtain $n = h \times w$ mediators.

**Design Motivation**: (1) The mediators act as an information "bottleneck" that compresses the redundant one-to-one Q-K interactions; (2) Since Q and K are decoupled by the mediators, the computation order can be swapped—computing $\mathbf{A}_{\text{tk}}^{(m)} \cdot \mathbf{v}^{(m)}$ ($n \times N \cdot N \times d$) first, and then interacting with Q, thereby avoiding the costly $N \times N$ matrix; (3) Complementing with DWConv to compensate for the loss of feature diversity in linear attention.

#### 2. **Complexity Analysis**

The complexity of standard self-attention is $O(N^2 C)$. Each step of the mediator attention is $O(Nnd)$, yielding a total complexity of $O(nNC)$. Since $n \ll N$, the computational cost is **reduced from quadratic to linear**.

| Operation | Standard Attention | Mediator Attention |
|------|-----------|-------------|
| Complexity | $O(N^2C)$ | $O(nNC)$ |
| 256×256 image ($N=256$) | $\propto 65536$ | $\propto 256n$ (approx. 1/4 when $n=64$) |
| Resolution Growth | Quadratic Growth | Linear Growth |

**High-Resolution Advantage**: The higher the image resolution, the more prominent the advantage of linear complexity.

#### 3. **Step-wise Dynamic Mediator Tuning**

**Function**: Dynamically increase the number of mediator tokens based on the variation of redundancy during the denoising process.

**Mechanism**: Quantify the variation of redundancy using the latent distance between consecutive denoising steps $\Delta_t = \|x_t - x_{t+1}\|$. When the distance falls below a certain threshold of the initial distance, the model switches to use more mediators:

$$n_t = \begin{cases} n_1, & \Delta_t > \rho_0 \cdot \Delta_0 \quad \text{(早期，高冗余，少中介者)} \\ n_2, & \Delta_t \leq \rho_1 \cdot \Delta_0 \quad \text{(中期)} \\ \vdots \\ n_k, & \Delta_t \leq \rho_{k-1} \cdot \Delta_0 \quad \text{(后期，低冗余，多中介者)} \end{cases}$$

**Sample-wise Independent Scheduling**: The threshold switching is sample-adaptive, as different images have distinct denoising processes and varying latent change rates.

**Design Motivation**: (1) Redundancy is high in the early stages, where a few mediators are sufficient for representation, leading to substantial computational savings; (2) Late stages feature rich details, necessitating more mediators to preserve diversity; (3) L1 distance performs better than L2 (validated by ablation).

### Loss & Training

- Training utilizes ImageNet-1k, class-conditional diffusion models.
- AdamW optimizer, no weight decay, learning rate $1 \times 10^{-4}$.
- Global batch size 256, trained for 400K iterations.
- EMA decay 0.9999.
- Only the first 4 self-attention layers are replaced by mediator attention (XL model).
- High-resolution (512/1024) models are obtained by fine-tuning from the 256 model.

## Key Experimental Results

### Main Results: ImageNet 256×256 Class-Conditional Generation

| Model | FID↓ | sFID↓ | IS↑ | Precision↑ | Recall↑ |
|------|------|-------|-----|-----------|---------|
| ADM | 10.94 | 6.02 | 100.98 | 0.69 | 0.63 |
| StyleGAN-XL | 2.30 | 4.02 | 265.12 | 0.78 | 0.53 |
| VDM++ | 2.12 | - | 267.7 | - | - |
| DiT-XL (cfg=1.5) | 2.27 | 4.60 | 278.24 | 0.83 | 0.57 |
| SiT-XL (cfg=1.5) | 2.06 | 4.50 | 270.27 | 0.82 | 0.59 |
| **Ours (cfg=1.5)** | **2.01** | **4.49** | **271.04** | **0.82** | **0.60** |

Based on SiT-XL/2, the method achieves **SOTA results with an FID of 2.01**, while reducing the computational cost.

### Ablation Study: Comparison of Static Mediator Counts (SiT-S/2, 256×256)

| Configuration | FLOPs(G) | FID↓ | sFID↓ | IS↑ | Precision↑ | Recall↑ |
|------|----------|------|-------|-----|-----------|---------|
| SiT-S/2 baseline | 6.06 | 58.61 | 9.25 | 24.31 | 0.41 | 0.59 |
| + Ours (n=4) | 5.49 (-9.4%) | 57.67 | 10.01 | 26.66 | 0.42 | 0.56 |
| + Ours (n=16) | 5.55 (-8.4%) | 54.55 | 9.28 | 26.55 | 0.43 | 0.59 |
| + Ours (n=64) | 5.78 (-4.6%) | **53.57** | **9.01** | **27.26** | **0.43** | **0.61** |

Even with the minimum of 4 mediators, the FID outperforms the baseline; at n=64, the FID is reduced by 5.04 while FLOPs are still reduced by 4.6%.

### Ablation Study: Comparison with Simple Q-K Dimension Compression

| Method | FLOPs(G) | FID↓ | Precision↑ | Recall↑ |
|------|----------|------|-----------|---------|
| SiT-S/2 baseline | 6.06 | 58.61 | 0.41 | 0.59 |
| Q-K dimension compression r=0.875 | 5.91 | 58.98 (+) | 0.40 | 0.60 |
| Q-K dimension compression r=0.750 | 5.76 | 59.18 (+) | 0.39 | 0.59 |
| Q-K dimension compression r=0.500 | 5.46 | 60.02 (+) | 0.40 | 0.57 |
| **Ours (n=64)** | **5.78** | **53.57** | **0.43** | **0.61** |

Although directly reducing the Q-K hidden dimension saves computation, the FID **consistently deteriorates**; in contrast, our method **significantly improves quality** while saving computation.

### Key Findings

1. **Mediators not only reduce computational cost but also improve generation quality**: This is because compressing redundant interactions acts as an implicit regularization, reducing the homogenization of attention outputs.
2. **Acceleration is more significant at high resolutions**: SiT-B/2 achieves a speedup of 15.7% at 512² resolution and 45.4% at 1024² resolution, demonstrating that the advantage of linear complexity scales with resolution.
3. **Dynamic strategies outperform static ones**: Adjusting the number of mediators step-wise and adaptively consistently yields a better FID under the same FLOPs budget.
4. **L1 distance outperforms L2**: L1 distance serves as a better threshold criterion for measuring latent variation.

## Highlights & Insights

1. **Self-contained logic from redundancy analysis to method design**: First quantifying redundancy using JSD $\rightarrow$ discovering the variation pattern across timesteps $\rightarrow$ proposing the mediator mechanism to resolve redundancy $\rightarrow$ dynamically adjusting to fit the variation pattern. The entire technical pipeline flows seamlessly.
2. **A rare win-win of "quality improvement and efficiency gain"**: Acceleration methods typically compromise quality, but here, compressing redundancy improves feature diversity instead.
3. **Semantic understanding of mediators**: Mediator tokens are not merely a means of computational optimization, but can also be understood as a semantic compression of latent information—guiding the generation process with a small number of representative tokens.
4. **Sample-adaptive training-free scheduling**: The dynamic thresholding is based on latent variation, eliminating the need to train an auxiliary network to decide when to switch.

## Limitations & Future Work

1. **Replacing only a subset of layers**: The XL model only replaces the first four layers with mediator attention, without fully exploring optimal replacement strategies.
2. **Threshold searching**: The thresholds $\rho_i$ for dynamic adjustment are obtained via grid search, which has a limited search space.
3. **Validated only on class-conditional generation**: The method has not been validated in more practical scenarios such as text-to-image generation (e.g., Stable Diffusion).
4. **Training costs not significantly reduced**: The approach mainly accelerates inference, offering limited efficiency gains during training.
5. **Not combined with other acceleration methods**: Complementary methods such as distillation or step reduction have not yet been integrated.

## Related Work & Insights

- **Agent Attention [ECCV 2024]**: Also uses additional tokens as a Q-K bridge in visual recognition tasks; this work extends this concept to diffusion generation.
- **SiT [ICML 2024]**: The primary baseline model of this work, introducing an interpolant framework from discrete to continuous time.
- **DiT [ICCV 2023]**: Demonstrated the scalability of ViTs in diffusion models; this work optimizes attention efficiency on top of its architecture.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Mediator design driven by redundancy analysis along with a step-wise dynamic strategy; the approach is clean and unique.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-scale models, multiple resolutions, detailed ablations, comparisons with various methods, and visual validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Smooth narrative flow from observation to methodology, with a very clear complexity analysis.
- **Value**: ⭐⭐⭐⭐ — SOTA FID with reduced computational costs, offering direct practical value for the optimization of DiT inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] One-Step Diffusion Transformer for Controllable Real-World Image Super-Resolution](../../CVPR2026/image_restoration/one-step_diffusion_transformer_for_controllable_real-world_image_super-resolutio.md)
- [\[ICLR 2026\] LinearSR: Unlocking Linear Attention for Stable and Efficient Image Super-Resolution](../../ICLR2026/image_restoration/linearsr_unlocking_linear_attention_for_stable_and_efficient_image_super-resolut.md)
- [\[ICML 2026\] DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention](../../ICML2026/image_restoration/dyllm_efficient_diffusion_llm_inference_via_saliency-based_token_selection_and_p.md)
- [\[ECCV 2024\] Restoring Images in Adverse Weather Conditions via Histogram Transformer](restoring_images_in_adverse_weather_conditions_via_histogram_transformer.md)
- [\[ECCV 2024\] Learning to Robustly Reconstruct Dynamic Scenes from Low-Light Spike Streams](learning_to_robustly_reconstruct_dynamic_scenes_from_low-light_spike_streams.md)

</div>

<!-- RELATED:END -->
