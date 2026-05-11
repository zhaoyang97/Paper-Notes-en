---
title: >-
  [Paper Note] Anchor Token Matching: Implicit Structure Locking for Training-free AR Image Editing
description: >-
  [ICCV 2025][Image Generation][Autoregressive models] This paper proposes ISLock, the first training-free image editing framework for autoregressive (AR) visual generation models. Through Anchor Token Matching (ATM)…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Autoregressive models"
  - "image editing"
  - "training-free"
  - "attention mechanism"
  - "structural consistency"
date: 2026-05-08
content_hash: 3fecb79cab2cec3e
---

# Anchor Token Matching: Implicit Structure Locking for Training-free AR Image Editing

**Conference**: ICCV 2025
**arXiv**: [2504.10434](https://arxiv.org/abs/2504.10434)
**Code**: [https://github.com/hutaiHang/ATM](https://github.com/hutaiHang/ATM)
**Area**: Image Editing / Autoregressive Generation
**Keywords**: Autoregressive models, image editing, training-free, attention mechanism, structural consistency

## TL;DR

This paper proposes ISLock, the first training-free image editing framework for autoregressive (AR) visual generation models. Through Anchor Token Matching (ATM), it implicitly aligns self-attention patterns in the latent space, enabling structure-consistent text-guided image editing.

## Background & Motivation

Diffusion models have achieved remarkable success in text-guided image editing via cross-attention manipulation for precise spatial control. However, autoregressive (AR) models have re-emerged as powerful alternatives (e.g., LlamaGen, Emu3), and their next-token prediction paradigm renders diffusion-based editing techniques non-transferable.

AR models face two core challenges:

**Spatial impoverishment of attention maps**: Text-to-image attention maps in AR models lack precise spatial correspondences — each token tends to attend heavily to its immediate predecessor, making them unreliable as editing anchors.

**Sequential accumulation of structural errors**: Naively modifying target tokens (e.g., replacing "cat" with "dog") induces local shifts in hidden states that propagate through the autoregressive dependency chain, ultimately distorting global structure.

Existing methods either require large-scale paired data fine-tuning or incur high computational costs while sacrificing zero-shot flexibility. The key question is therefore: how can structure-consistent editing be achieved in AR models without any training?

## Method

### Overall Architecture

ISLock (Implicit Structure Locking) preserves the structural blueprint by dynamically aligning self-attention patterns with a reference image, without relying on explicit attention manipulation or fine-tuning. The core mechanism selectively matches tokens during autoregressive decoding via the ATM protocol, allowing attention consistency to emerge naturally as a byproduct.

### Key Designs

1. **Structural Information Analysis**

   - PCA decomposition and visualization of the self-attention matrix $A \in \mathbb{R}^{(h \times w) \times (h \times w)}$ reveals that semantically similar tokens exhibit consistent attention patterns.
   - Cross-attention maps lack spatial structural information, whereas self-attention maps encode rich structural content.
   - Perturbation experiments confirm the critical role of early tokens: perturbing the first 20% of tokens causes an SSIM drop of $0.56 \pm 0.02$, while perturbing the last 20% yields only $0.08 \pm 0.05$.

2. **Anchor Token Matching (ATM) and Dynamic Window**

   - Given the original prompt $\mathcal{P}_{org}$ and the editing prompt $\mathcal{P}_{edit}$, at each step $i$, $K$ candidate tokens $\mathcal{C}_i = \{z_i^{(1)}, ..., z_i^{(K)}\}$ are sampled.
   - The Euclidean distance between each candidate and the reference anchor is computed: $s^{(k)} = \|z_i^{(k)} - z_i^{org}\|_2^2$
   - A dynamic window mechanism is introduced, with window size linearly shrinking over decoding progress: $|\mathcal{W}_i| = \lfloor K \cdot (1 - \alpha \cdot \frac{i}{N}) \rfloor$, where $\alpha = 0.6$
   - The full candidate set is retained in early steps to enforce strict structural alignment, while constraints are progressively tightened in later steps.
   - The final token is selected as the closest candidate within the window: $z_i^{edit} = \arg\min_{k \in \mathcal{W}_i} s^{(k)}$

3. **Adaptive Constraint Relaxation (AdaCR)**

   - A similarity threshold $\tau$ is introduced to balance structural preservation and generative freedom:
     $$z_i^{edit} = \begin{cases} \arg\min s^{(k)} & \text{if } \min s^{(k)} \leq \tau \\ \arg\max p(z_i|z_{<i}^{edit}, c_{edit}) & \text{otherwise} \end{cases}$$
   - A larger $\tau$ enforces higher fidelity to the original image; a smaller $\tau$ permits greater editing diversity.
   - Two-level protection is provided: candidate window pre-filtering and dynamic thresholding.

### Loss & Training

The method is entirely training-free, requiring no parameter updates or fine-tuning. Default hyperparameters: $K=150$, $\tau=1.0$, $\alpha=0.6$.

## Key Experimental Results

### Main Results

Quantitative comparison on the PIE-Bench dataset:

| Method | Base Model | Structure Dist.↓ | PSNR↑ | SSIM↑ | CLIP Whole↑ | CLIP Edited↑ |
|--------|------------|------------------|-------|-------|-------------|-------------|
| NPM | LlamaGen | 113.95 | 12.14 | 53.67 | 24.71 | 21.28 |
| PnP-AR | LlamaGen | 103.94 | 13.20 | 58.25 | 23.56 | 20.65 |
| **ISLock (Ours)** | **LlamaGen** | **31.79** | **19.75** | **76.71** | **24.19** | **21.33** |
| P2P | SD1.4 | 88.46 | 16.80 | 69.93 | 26.70 | 21.43 |
| Null-text Inv. | SD1.4 | 18.42 | 25.68 | 85.71 | 24.55 | 20.73 |

### Ablation Study

| Window Configuration | Struc. Dist.↓ | CLIP Sim.↑ | S/C Ratio↓ |
|----------------------|--------------|------------|------------|
| $|\mathcal{W}|=50$ | 60.83 | 24.79 | 2.45 |
| $|\mathcal{W}|=100$ | 38.03 | 24.33 | 1.56 |
| $|\mathcal{W}|=150$ | 30.39 | 22.18 | 1.37 |
| Dynamic (Ours) | 31.79 | 24.19 | **1.31** |

### Key Findings

- ISLock achieves the best structural consistency among AR-based methods (Struc. Dist. 31.79, far superior to NPM's 113.95).
- The dynamic window strategy achieves the best trade-off between structural distance and CLIP similarity (S/C ratio 1.31).
- Five editing types are supported: attribute replacement, object addition/removal, and style transfer, among others.
- The method generalizes across different AR base models (LlamaGen and Lumina-mGPT).

## Highlights & Insights

- **Strong originality**: This work presents the first training-free editing framework for AR visual models, filling a notable gap in the AR image editing literature.
- **Implicit vs. explicit control**: The key insight is to avoid directly injecting attention maps (which introduces artifacts), instead allowing attention consistency to emerge naturally through token matching.
- **Progressive structure locking**: A systematic analysis of structural control mechanisms in AR generation reveals the decisive role of early tokens in determining global structure.

## Limitations & Future Work

- Overall performance still lags behind editing methods optimized for diffusion frameworks (e.g., Null-text Inversion).
- The method relies on AR-generated images and cannot directly edit real photographs.
- The candidate size $K=150$ introduces additional sampling overhead.
- Only 5 of the 10 editing types in PIE-Bench are evaluated.

## Related Work & Insights

- Contrasted with the Prompt-to-Prompt paradigm for diffusion models, this work reveals fundamental differences in structural control across generative paradigms.
- It offers a new technical pathway for controllable image editing in the era of AR-based multimodal large models.
- The candidate matching concept underlying ATM is generalizable to other controllability problems in AR generation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First training-free editing method for AR models; core idea is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Quantitative comparisons and ablations are provided, but dataset coverage is limited.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated with rich experimental visualizations.
- Value: ⭐⭐⭐⭐ — Opens a new direction for AR model editing, though practical utility warrants further improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering](larender_training-free_occlusion_control_in_image_generation_via_latent_renderin.md)
- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](ale_attribute_leakage_free_editing.md)
- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [\[ICCV 2025\] IntroStyle: Training-Free Introspective Style Attribution using Diffusion Features](introstyle_training-free_introspective_style_attribution_using_diffusion_feature.md)
- [\[ICCV 2025\] MotionDiff: Training-Free Zero-Shot Interactive Motion Editing via Flow-Assisted Multi-View Diffusion](motiondiff_training-free_zero-shot_interactive_motion_editing_via_flow-assisted_.md)

</div>

<!-- RELATED:END -->
