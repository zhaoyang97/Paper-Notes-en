---
title: >-
  [Paper Note] Universal Multi-Domain Translation via Diffusion Routers
description: >-
  [ICLR 2026][Segmentation][Multi-Domain Translation] This paper proposes the Diffusion Router (DR), which employs a single noise prediction network conditioned on source/target domain labels to handle all cross-domain mappings. It supports indirect translation via a center domain and direct non-center-domain translation based on a variational upper-bound objective combined with Tweedie refinement, achieving state-of-the-art performance on three large-scale UMDT benchmarks.
tags:
  - ICLR 2026
  - Segmentation
  - Multi-Domain Translation
  - Diffusion Models
  - Diffusion Router
  - Tweedie Refinement
  - Universal Translation
date: 2026-05-08
content_hash: 89a880af00e2297c
---

# Universal Multi-Domain Translation via Diffusion Routers

**Conference**: ICLR 2026
**arXiv**: [2510.03252](https://arxiv.org/abs/2510.03252)
**Code**: N/A
**Area**: Image Segmentation
**Keywords**: Multi-Domain Translation, Diffusion Models, Diffusion Router, Tweedie Refinement, Universal Translation

## TL;DR

This paper proposes the Diffusion Router (DR), which employs a single noise prediction network conditioned on source/target domain labels to handle all cross-domain mappings. It supports indirect translation via a center domain and direct non-center-domain translation based on a variational upper-bound objective combined with Tweedie refinement, achieving state-of-the-art performance on three large-scale UMDT benchmarks.

## Background & Motivation

**Background**: Multi-Domain Translation (MDT) aims to learn mappings among multiple domains and has broad applications in image-to-image translation, image captioning, text-to-speech synthesis, and related areas. Existing MDT methods fall into two paradigms: (1) training on fully aligned multi-tuples, which does not scale with the number of domains; and (2) training on paired data sharing a center domain, which supports only translations between the center domain and non-center domains.

**Limitations of Prior Work**:
1. **Fully aligned tuple paradigm**: $K$ domains require $K$-tuple aligned data, and collection costs grow exponentially with the number of domains.
2. **Center-domain paradigm**: Only center-domain $\leftrightarrow$ non-center-domain translations are supported; direct translations between non-center domains (e.g., sketch $\leftrightarrow$ segmentation) cannot be achieved.
3. **Model scalability**: Training independent models for each domain pair requires $2(K-1)$ models, which becomes infeasible as the number of domains grows.
4. **Quality degradation in indirect translation**: Two-stage sampling via the center domain is computationally expensive and sensitive to the quality of intermediate samples.

**Key Challenge**: Fully aligned multi-domain data is scarce in practice, whereas paired data with a center domain is relatively abundant (e.g., large-scale image–text and text–audio paired datasets each exist independently). The central challenge is how to achieve translation between arbitrary domain pairs given only $K-1$ paired datasets with the center domain.

**Goal**: This paper formalizes the Universal Multi-Domain Translation (UMDT) problem—achieving arbitrary translation among $K$ domains using only $K-1$ datasets paired with a center domain. It proposes the Diffusion Router (DR), inspired by the source/destination addressing concept in network routers, employing a single noise prediction network $\epsilon_\theta(x_t^{tgt}, t, x^{src}, tgt, src)$ to handle all translation directions.

## Method

### Overall Architecture

UMDT setting: $K$ domains $X^1, X^2, \ldots, X^K$ share a center domain $X^c$, and training data consists of $K-1$ paired datasets $\mathcal{D}_{k,c} = \{(x^k, x^c)\}$. The Diffusion Router achieves arbitrary cross-domain translation in two stages:

**Stage 1: Indirect Diffusion Router (iDR)**
- Learns all bidirectional mappings between the center domain and non-center domains.
- A single noise prediction network is conditioned on source/target domain labels:

$$\mathcal{L}_{paired}(\theta) = \mathbb{E}_{(x^k, x^c) \sim \mathcal{D}_{k,c}} \left[ \zeta \|\epsilon_\theta(x_t^k, t, x^c, k, c) - \epsilon\|_2^2 + (1-\zeta)\|\epsilon_\theta(x_t^c, t, x^k, c, k) - \epsilon\|_2^2 \right]$$

- Translations between non-center domains are routed through the center domain: $X^i \to X^c \to X^j$.

**Stage 2: Direct Diffusion Router (dDR)**
- Fine-tunes iDR to support direct cross-domain translation $X^i \to X^j$.
- Minimizes a variational upper-bound objective while preserving previously learned mappings.

### Key Design 1: Variational Upper-Bound Learning Objective

The core difficulty in directly learning $p_\theta(x^j | x^i)$ is twofold: (1) sampling from $p(x'^c | x^i)$ is computationally expensive; and (2) evaluating $p(x^j | x'^c)$ has no closed-form solution. This paper decomposes the original KL divergence into a variational upper bound expressed as a sum of transition kernel KL divergences:

$$\mathbb{E}_{\mathcal{D}_{i,c}} \left[ D_{KL}(p_{ref}(x^j | x^c) \| p_\theta(x^j | x^i)) \right] \leq \sum_{t=1}^T \mathbb{E}_{p_{ref}(x_t^j | x^c)} \left[ D_{KL}(p_{ref}(x_{t-1}^j | x_t^j, x^c) \| p_\theta(x_{t-1}^j | x_t^j, x^i)) \right]$$

This is converted into a noise prediction loss via standard reparameterization:

$$\mathcal{L}_{unpaired}(\theta) = \mathbb{E} \left[ \|\epsilon_\theta(x_t^j, t, x^i, j, i) - \epsilon_{ref}(x_t^j, t, x^c, j, c)\|_2^2 \right]$$

where $\epsilon_{ref}$ is the frozen pretrained DR noise prediction network. The final loss $\mathcal{L}_{final} = \lambda_1 \mathcal{L}_{unpaired} + \lambda_2 \mathcal{L}_{paired}$ balances learning new mappings against preserving existing ones.

### Key Design 2: Tweedie Refinement Sampling

Sampling from the conditional distribution $p_{ref}(x_t^j | x^c)$ in $\mathcal{L}_{unpaired}$ traditionally requires full denoising from time $T$ to $t$, which is computationally expensive. This paper proposes Tweedie Refinement—a lightweight iterative sampling method:

$$x_{t,(n+1)}^j = x_{t,(n)}^j + \sigma_t (\epsilon - \epsilon_\theta(x_{t,(n)}^j, t, x^c, j, c))$$

initialized with $x_{t,(0)}^j \sim p_{ref}(x_t^j)$ (unconditional sampling). A small number of refinement steps (empirically $n \leq 7$) suffices to transform unconditional samples into conditional ones. Compared with existing refinement techniques, Tweedie Refinement: (1) converts unconditional samples into conditional samples rather than correcting off-manifold samples back to the marginal distribution; (2) is applied during training rather than inference; and (3) possesses a distinct mathematical formulation.

### Key Design 3: Unified Conditioning and Scalability

Inspired by the source/destination addressing paradigm in network routers, DR directly encodes source and target domain labels into the noise prediction network. Rather than training $2(K-1)$ independent models for each translation direction, DR uses a single network to cover all translation paths. This design naturally generalizes to spanning-tree topologies with multiple center domains, and is not restricted to a star-shaped structure.

## Key Experimental Results

### Main Results

Evaluation is conducted on three newly constructed UMDT benchmarks, comparing against StarGAN (GAN), Rectified Flow (flow), and UniDiffuser (diffusion) baselines.

**Shoes-UMDT (FID↓, format: A←B / A→B)**:

| Method | Edge↔Shoe | Gray↔Shoe | Edge↔Gray |
|------|:---:|:---:|:---:|
| StarGAN | 9.92/20.18 | 19.73/42.61 | 18.64/27.41 |
| Rectified Flow | 2.88/30.92 | 3.75/43.38 | 20.14/18.83 |
| UniDiffuser | 2.98/11.94 | 2.72/4.40 | 4.81/12.26 |
| iDR | **1.66/5.15** | **0.53/1.60** | 1.85/5.48 |
| dDR | 2.01/5.76 | 0.57/1.69 | **2.74/6.51** |

**COCO-UMDT-Star (FID↓)**:

| Method | Ske↔Color | Seg↔Color | Depth↔Color | Ske↔Seg | Ske↔Depth | Seg↔Depth |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Rectified Flow | 23.18/80.80 | 54.00/142.15 | 17.32/112.64 | 64.47/75.58 | 78.41/28.69 | 79.20/35.53 |
| UniDiffuser | 15.39/40.93 | 35.81/89.58 | 12.64/59.72 | 39.62/38.44 | 28.12/15.72 | 38.39/23.41 |
| iDR | **10.72/21.73** | **21.64/29.28** | **7.25/24.19** | 22.77/22.96 | **17.88/8.63** | **23.19/12.00** |
| dDR | 10.12/20.94 | 21.23/28.32 | 7.00/23.20 | **26.73/23.64** | 20.75/9.42 | 24.91/14.87 |

DR consistently outperforms all baselines on center-domain translation and also demonstrates competitive direct translation capability on non-center-domain pairs (highlighted) where no paired data is available. In most cases, iDR's indirect translation outperforms dDR's direct translation, indicating high quality in the intermediate representations.

### Ablation Study

**Ablation on Tweedie Refinement Steps (Faces-UMDT-Latent)**:

| Refinement Steps $n$ | Ske→Face FID↓ | Face→Ske FID↓ | Seg→Face FID↓ |
|:---:|:---:|:---:|:---:|
| 0 | Baseline (no refinement) | — | — |
| 1 | Significant improvement | Significant improvement | Significant improvement |
| 3 | Further improvement | Further improvement | Further improvement |
| 5 | Near optimal | Near optimal | Near optimal |
| 7 | **Optimal** | **Optimal** | **Optimal** |

Tweedie Refinement progressively converts unconditional samples into conditional ones starting from $n=0$; as few as 3–5 steps yield substantial gains, substantially reducing the sampling cost during training.

**Training from Scratch vs. Fine-Tuning (dDR Learning Strategy)**: Fine-tuning the pretrained iDR outperforms training from scratch, validating the effectiveness of the two-stage strategy. Training from scratch is also feasible by treating $\epsilon_{ref}$ as an online frozen network, but converges more slowly.

## Highlights & Insights

### Strengths

1. **Practically motivated problem formulation**: UMDT captures the realistic setting of "hub domain + sparse pairing" in multimodal translation, which is far more practical than the fully aligned assumption.
2. **Elegant architectural design**: The router paradigm compresses $O(K^2)$ models into a single network, offering strong scalability.
3. **Rigorous theoretical derivation**: The variational upper-bound objective and conditional independence assumption are supported by complete mathematical derivations.
4. **Novel Tweedie Refinement**: Addresses the efficiency bottleneck of conditional sampling during training.
5. **Three newly constructed UMDT benchmarks**: Provides standardized evaluation platforms for the newly defined problem.

### Limitations & Future Work

1. The conditional independence assumption $X^i \perp X^j | X^c$ may not hold exactly in practice, potentially limiting the quality of indirect translation.
2. Experiments are primarily conducted within the image domain; truly cross-modal scenarios (e.g., image↔text↔audio) have not been validated.
3. dDR's direct translation does not consistently outperform iDR's indirect translation, and the advantages of direct mapping warrant further analysis.

## Rating

⭐⭐⭐⭐ — The problem formulation is novel and practically motivated; the method design is clear and theoretically well-grounded; Tweedie Refinement is a notable technical contribution. However, cross-modal validation and discussion of the conditional independence assumption could be more thorough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] VIRTUE: Visual-Interactive Text-Image Universal Embedder](virtue_visual-interactive_text-image_universal_embedder.md)
- [\[ACL 2026\] Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech](../../ACL2026/segmentation/hierarchical_policy_optimization_for_simultaneous_translation_of_unbounded_speec.md)
- [\[ICLR 2026\] TRACE: Your Diffusion Model is Secretly an Instance Edge Detector](trace_your_diffusion_model_is_secretly_an_instance_edge_detector.md)
- [\[ICLR 2026\] RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](regionreasoner_region-grounded_multi-round_visual_reasoning.md)
- [\[CVPR 2026\] 3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion](../../CVPR2026/segmentation/3m-ti_high-quality_mobile_thermal_imaging_via_calibration-free_multi-camera_cros.md)

</div>

<!-- RELATED:END -->
