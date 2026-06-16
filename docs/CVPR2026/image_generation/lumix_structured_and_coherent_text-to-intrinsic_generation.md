---
title: >-
  [Paper Note] LumiX: Structured and Coherent Text-to-Intrinsic Generation
description: >-
  [CVPR 2026][Image Generation][Diffusion Model] LumiX proposes the new task of "text-to-intrinsic" generation based on the FLUX diffusion model: generating a set of pixel-aligned intrinsic maps (color, albedo, irradiance, depth, normals) from a single text prompt. It achieves this through two key designs: **Query-Broadcast Attention**, which broadcasts the color bra
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion Model
date: 2026-05-08
content_hash: 5a7a005959f81754
---
# LumiX: Structured and Coherent Text-to-Intrinsic Generation

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Han_LumiX_Structured_and_Coherent_Text-to-Intrinsic_Generation_CVPR_2026_paper.html)  
**Code**: https://github.com/xhanxu/LumiX  
**Area**: Image Generation / Diffusion Models / Intrinsic Decomposition  
**Keywords**: Text-to-Intrinsic, Diffusion Models, Query-Broadcast Attention, Tensor LoRA, Cross-map Consistency

## TL;DR
LumiX proposes the new task of "text-to-intrinsic" generation based on the FLUX diffusion model: generating a set of pixel-aligned intrinsic maps (color, albedo, irradiance, depth, normals) from a single text prompt. It achieves this through two key designs: **Query-Broadcast Attention**, which broadcasts the color branch query to all intrinsic maps to ensure structural consistency, and **Tensor LoRA**, which efficiently models cross-map relationships via tensor decomposition. LumiX achieves a 23% higher alignment score than the SOTA and improves preference scores from -0.41 to 0.19, while the same framework can be reversed for image-conditioned intrinsic decomposition.

## Background & Motivation

**Background**: Text-to-image diffusion models (e.g., FLUX, Stable Diffusion) can generate realistic images from text, but they only output a single RGB image without revealing the underlying physical structures like geometry, lighting, and materials. Many vision and graphics tasks (relighting, material editing, synthetic rendering) require a set of **separated intrinsic factors**: albedo, irradiance, normals, depth, and the final color.

**Limitations of Prior Work**: Existing research almost exclusively focuses on "intrinsic image decomposition"—predicting components like albedo and shading given a rendered or captured image. This path is inherently limited: **it depends on a given image and cannot generate new scenes directly from language**. "Generating a full set of intrinsic maps from text" is nearly unexplored. The Key Challenge is: **how to maintain cross-map structural consistency under text-only conditions**. In image-conditioned settings (e.g., RGB-to-X translation), all outputs share the same input image, making spatial alignment natural. However, in text-to-intrinsic generation, each map starts from its own noise sample and is only weakly coupled through shared text embeddings, leading to structural drift: objects may appear in the color map but vanish in the normal map, or depth geometry may not match irradiance shading.

**Key Challenge**: Existing approaches to consistency have critical flaws. One approach (e.g., independent LoRA branches for each map) offers high quality but **fails on cross-map semantic alignment**. Another (IntrinsiX, which concatenates K and V of all maps for cross-intrinsic attention) improves consistency but suffers from **unstable training and quadratic computational growth** relative to the number of maps. This reveals a clear trade-off between consistency and efficiency.

**Goal**: To build a unified model that can jointly generate all intrinsic maps under text-only conditions (consistency) while preserving the physical characteristics of each map (quality), remaining parameter-efficient and scalable.

**Key Insight**: The authors leverage a prior regarding content/style separation in attention mechanisms—research indicates that in self-attention, **queries primarily encode scene "content," while keys/values encode "style/modal characteristics"** (the appearance differences between albedo, lighting, and normals). By having all intrinsic maps share the same query, they can be forced to align with the same scene content without image supervision, while allowing their respective K/V to retain attribute-specific properties.

**Core Idea**: Achieve cross-map structural consistency by "broadcasting the color branch query to all maps" and capture cross-map relationships efficiently with "Tensor LoRA," simultaneously addressing consistency and efficiency.

## Method

### Overall Architecture
Given a text prompt $C$, LumiX aims to jointly generate 5 pixel-aligned intrinsic maps of the same scene: color $x^{(c)}$, albedo $x^{(a)}$, irradiance $x^{(i)}$, depth $x^{(d)}$, and normals $x^{(n)}$, corresponding to latent variables $z^{(c)}, z^{(a)}, z^{(i)}, z^{(d)}, z^{(n)}$. The task has two objectives: **consistency** (cross-attribute structural alignment and shared content) and **quality** (realism for each attribute).

The method is based on fine-tuning a pretrained text-to-image diffusion model (FLUX.1-dev). A naive approach would train an independent model for each map ($M=5$), which provides high single-map quality but no cross-map consistency. LumiX introduces two core components into the FLUX blocks, affecting the **forward process** and the **fine-tuning process**:

- **Query-Broadcast Attention (Forward)**: In each self-attention block, the query $Q^{(c)}$ from the color model is broadcast to all intrinsic maps, forcing them to share the same "content" for pixel alignment.
- **Tensor LoRA (Fine-tuning)**: A low-rank update using tensor decomposition uniformly models the K/V projection adaptation for all intrinsic maps, capturing cross-map relationships while keeping parameter counts near-linear.

During training, multiple intrinsic maps are encoded into latent space via VAE, concatenated along the batch dimension, and fed into FLUX blocks equipped with these components, optimized using flow matching loss. Different attributes are assigned different diffusion timesteps. At inference, given text or an image, the model jointly outputs all intrinsic maps in a single forward pass, supporting both text-to-intrinsic generation and image-conditioned intrinsic decomposition.

### Key Designs

**1. Query-Broadcast Attention: Broadcasting Queries for Structural Alignment**

This is the core for ensuring consistency. Vanilla Attention calculates self-attention independently for each map $m\in\{c,a,i,d,n\}$ as $H^{(m)}\leftarrow \mathrm{softmax}(Q^{(m)}K^{(m)\top}/\sqrt{d})V^{(m)}$, ignoring inter-map interaction. IntrinsiX concatenates K and V from all maps ($K^{(\mathcal{M})}=\mathrm{Concat}([K^{(c)},K^{(a)},\dots])$) for cross-intrinsic attention, which helps consistency but increases computation by $M$ times. LumiX leverages the insight that "queries encode content while K/V encode style," **broadcasting only the color branch query $Q^{(c)}$ to all maps**:

$$H^{(m)}\leftarrow \mathrm{softmax}\!\left(Q^{(c)}K^{(m)\top}/\sqrt{d}\right)V^{(m)}$$

This ensures all intrinsic maps use the same scene content (from the color map query) to retrieve their attribute-specific K/V, leading to natural structural alignment while maintaining independent appearances. It is significantly more efficient than the concatenation approach of IntrinsiX (FLOPs 145.1 vs 724.7 per attention block) because the sequence length is not multiplied by $M$, yet it achieves higher alignment.

**2. Tensor LoRA: Efficient Modeling of Cross-map LoRA Updates**

Since $Q^{(c)}$ is broadcast, the query projection $W_Q$ is no longer fine-tuned, leaving the K/V projections $\{(W^{(m)}_K,W^{(m)}_V)\}$ as trainable parameters. The authors analyze several LoRA designs: **Separate LoRA** (independent $\Delta^{(m)}$ per map) is efficient but ignores cross-map interaction; **Fused LoRA** (concatenating activations into $h^{(\mathcal{M})}\in\mathbb{R}^{Md}$ with a dense matrix $\Delta^{(\mathcal{M})}\in\mathbb{R}^{Md\times Md}$) models all interactions but risks oversmoothing; **Hybrid LoRA** (high-rank $R_1$ for diagonals, low-rank $R_2$ for off-diagonals) improves consistency but doubles parameters.

Tensor LoRA reshapes the entire update $\Delta^{(\mathcal{M})}\in\mathbb{R}^{Md\times Md}$ into a 4th-order tensor $\Delta^{(\mathcal{M})}\in\mathbb{R}^{N\times d_{out}\times M\times d_{in}}$ (where $M, N$ are the number of input/output maps) and applies a tensor-train-like decomposition:

$$\Delta^{(\mathcal{M})}[i,j,k,l]=\sum_{\alpha_1=1}^{R_1}\sum_{\alpha_2=1}^{R_2}A[i,j,\alpha_1]\,B[i,k,\alpha_2]\,C[i,l,\alpha_1,\alpha_2]$$

where $A\in\mathbb{R}^{N\times d_{out}\times R_1}$, $B\in\mathbb{R}^{N\times M\times R_2}$, and $C\in\mathbb{R}^{N\times d_{in}\times R_1\times R_2}$. Computation is performed via three `einsum` contractions. It models all LoRA updates with a structured tensor decomposed into "shared cores + per-map components," **capturing cross-map relations while maintaining near-linear parameter costs**. In experiments, Tensor LoRA achieves the best balance of quality, consistency, and efficiency (2.34M parameters and 12.1G FLOPs per block with a high alignment score of 8.30).

**3. Disentangled Timestep Sampling: Unlocking Image-conditioned Decomposition**

LumiX assigns independent diffusion timesteps to each intrinsic attribute, allowing different noise levels for different attributes. This acts as a "soft mask" encouraging flexible conditioning. Although trained solely on text, this design **naturally supports image-conditioned generation**: during inference, one attribute can be kept clean (no noise) while others are denoised. Combined with Query-Broadcast Attention, this ensures generated maps align with the clean condition map. This turns a training trick into a key mechanism for unified generation and understanding.

### Loss & Training
The model is trained using flow matching loss $\min_\theta \mathbb{E}_{t,\epsilon,z}\|v_\theta(z_t,t,C)-(\epsilon-z)\|$ by fine-tuning FLUX.1-dev. Data consists of a ~3K image subset of Hypersim (using BLIP-2 for captions). Tensor LoRA rank is set to 8 (~133.1M trainable parameters). The Prodigy optimizer is used with a 1.0 learning rate and batch size 16, training for 10K steps on 4×A100 (80GB) for approximately 40 hours. Images are aspect-ratio preserved and randomly cropped to 512×512.

## Key Experimental Results

### Main Results
Since text-to-intrinsic quality lacks ground truth, evaluation uses Human Preference models ImageReward (IR) and PickScore (PS), and Qwen3-VL for alignment (Align.). The table below compares Attention × LoRA designs (selected, higher is better; FLOPs per attention block):

| Attention | LoRA | #P(M)↓ | Attn FLOPs(G)↓ | Align.↑ | Avg IR↑ | Avg PS↑ |
|-----------|------|--------|----------------|---------|---------|---------|
| Vanilla (FLUX) | Separate | 2.95 | 145.1 | 2.40 | 0.06 | 20.37 |
| Cross-Intrinsic (IntrinsiX‡) | Separate | — | 724.7 | 6.73 | -0.41 | 19.78 |
| Cross-Intrinsic (IntrinsiX) | Tensor | 2.46 | 724.7 | 7.98 | -0.28 | 19.71 |
| **Query-Broadcast (Ours)** | Hybrid | 4.03 | 145.1 | 8.21 | 0.18 | 20.12 |
| **Query-Broadcast (LumiX)** | **Tensor** | **2.34** | **145.1** | **8.30** | **0.19** | **20.52** |

LumiX (Query-Broadcast + Tensor LoRA) achieves the highest alignment (8.30) and IR/PS scores. Compared to the official IntrinsiX (Align 6.73, IR -0.41), consistency improves by ~23% and the preference score rises from -0.41 to 0.19, while attention FLOPs (145.1) are only ~1/5 of IntrinsiX (724.7).

### Ablation Study
Ablation on Hypersim (Avg IR / PS; #P and #F denote parameters and FLOPs per LoRA block):

| Configuration | #P(M)↓ | #F(G)↓ | Align.↑ | Avg IR↑ | Avg PS↑ | Note |
|---------------|--------|--------|---------|---------|---------|------|
| LumiX (R=8) | 2.34 | 12.1 | 8.30 | 0.19 | 20.52 | Full Model |
| + Tune $W_Q$ | 2.46 | 14.1 | 7.14 | -0.09 | 20.04 | Tuning $W_Q$ hurts performance |
| R=4 | 0.68 | 4.7 | 7.86 | -0.18 | 19.79 | Very efficient but competitive |
| R=12 | 4.98 | 22.5 | 8.10 | 0.14 | 20.29 | Increasing rank yields no gain |

In zero-shot intrinsic decomposition (ARAP dataset, albedo quality): LumiX, based on FLUX and trained on only 3K images, achieves RMSE 0.165 / SSIM 0.753, comparable to or better than diffusion baselines trained on 900K images (e.g., RGB↔X with RMSE 0.238), while being the only model supporting text-to-intrinsic generation. In-the-wild preference scores for LumiX (IR 0.14 / PS 20.16) also exceed RGB↔X and Colorful Shading.

### Key Findings
- **Avoiding $W_Q$ fine-tuning is critical**: Tuning $W_Q$ (+ Tune $W_Q$) causes alignment to drop from 8.30 to 7.14 and IR from 0.19 to -0.09. This is because it disrupts the diffusion prior and the "query encodes content" division of labor, justifying the Query-Broadcast design.
- **Rank 8 is the sweet spot**: $R=4$ remains competitive with only 0.68M parameters; $R=8$ provides the best quality/efficiency; $R=12$ yields no extra gain and even slight drops, suggesting cross-map relationships do not require high rank.
- **Structured sharing outperforms supervision scale**: LumiX trained on 3K images exceeds decomposition baselines trained on 900K images. The authors argue that "stable multi-map generation stems from structured parameter sharing rather than simple' data scaling."
- **IntrinsiX crashes without Stage-1**: Cross-intrinsic attention collapses if not initialized via stage-1 due to global K/V dependencies. Tensor LoRA helps, but distinguishing modalities remains difficult—demonstrating that LumiX's query broadcasting is more stable than concatenation-based sharing.

## Highlights & Insights
- **Content/Style Separation as a Consistency Tool**: The prior that query=content and K/V=style is cleverly utilized. Shared queries achieve cross-map alignment in one step without sacrificing attribute specificity, proving more efficient and accurate than K/V concatenation. This "query sharing only" trick is transferable to any multi-image/multi-view generation task requiring structural alignment.
- **Tensor LoRA Linearizes Quadratic Parameters**: Using a tensor-train-like decomposition to represent cross-map LoRA updates solves the dilemma between independent LoRA (ignoring interaction) and fused LoRA (quadratic expansion). It serves as a beautiful paradigm for parameter-efficient fine-tuning (PEFT) in multi-task/multi-output scenarios.
- **One Training Trick Unlocks Dual Capabilities**: Disentangled timestep sampling allows a text-only model to perform image-conditioned decomposition without modification. Unifying "generation" and "understanding" under one framework—training without images but inferring with them—is highly insightful.

## Limitations & Future Work
- **Limited Attribute Set and Data Scale**: Currently covers only 5 intrinsic attributes and fine-tuned on a 3K subset. The authors plan to expand to broader attributes and larger datasets.
- **Reliance on Preference Models**: Text-to-intrinsic quality lacks ground truth and relies on IR/PS proxies and Qwen3-VL for alignment. There is a lack of objective physical consistency metrics.
- **Heavily Dependent on Base Model Priors**: The method relies on FLUX priors (tuning queries destroys alignment). Whether "shared query = alignment" holds for weaker base models or those with different biases is unverified.
- **Decomposition is not the Core focus**: While image-conditioned decomposition is comparable to baselines, the model is not end-to-end optimized for it, and accuracy in extreme material/lighting conditions may still lag behind specialized models.

## Related Work & Insights
- **vs IntrinsiX (cross-intrinsic attention)**: IntrinsiX relies on global interaction via K/V concatenation for consistency, but computation scales quadratically and requires stage-1 initialization to avoid collapse. LumiX broadcasts only queries, reducing FLOPs to ~1/5 while improving alignment and requiring no special initialization.
- **vs Separate LoRA / Independent Models**: These offer high single-map quality but fail on semantic alignment. LumiX uses Tensor LoRA to explicitly model cross-map relations, leading to significantly higher alignment scores.
- **vs RGB↔X / Colorful Shading (Decomposition Baselines)**: These are image-conditioned, require large-scale supervision, and cannot generate new scenes from text. LumiX achieves comparable or better albedo quality on ARAP/wild data with only 3K training images and is the only model supporting text-to-intrinsic generation.
- **Insight**: Tensor decomposition (tensor-train/tensor-ring) was previously used for compressing large weights. Its adaptation here for multi-output LoRA proves that "structured parameter sharing" in multi-task generation can both capture relationships and save parameters, offering a promising direction for future research.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Proposes the new text-to-intrinsic task and solves consistency and efficiency through original query-broadcast and Tensor LoRA designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive design matrix (Attention × LoRA), ablations, zero-shot decomposition, and wild generalization. However, quality evaluation relies on preference proxies without objective physical metrics.
- Writing Quality: ⭐⭐⭐⭐ Motivations are logically progressive with clear design comparisons, though Tensor LoRA math may have a learning curve.
- Value: ⭐⭐⭐⭐ Provides a unified framework for controllable generation and inverse rendering. Both Tensor LoRA and query-broadcast mechanisms have high general transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](solace_self_confidence_rewards_t2i.md)
- [\[CVPR 2026\] Intrinsic Concept Extraction Based on Compositional Interpretability](intrinsic_concept_extraction_based_on_compositional_interpretability.md)
- [\[CVPR 2026\] Anchoring and Rescaling Attention for Semantically Coherent Inbetweening](anchoring_and_rescaling_attention_for_semantically_coherent_inbetweening.md)
- [\[CVPR 2026\] Re-Align: Structured Reasoning-guided Alignment for In-Context Image Generation and Editing](re-align_structured_reasoning-guided_alignment_for_in-context_image_generation_a.md)
- [\[AAAI 2026\] LongT2IBench: A Benchmark for Evaluating Long Text-to-Image Generation with Graph-structured Annotations](../../AAAI2026/image_generation/longt2ibench_a_benchmark_for_evaluating_long_text-to-image_generation_with_graph.md)

</div>

<!-- RELATED:END -->
