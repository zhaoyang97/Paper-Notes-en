---
title: >-
  [Paper Note] UnZipLoRA: Separating Content and Style from a Single Image
description: >-
  [ICCV 2025][Robotics][LoRA] This paper proposes UnZipLoRA, a method that simultaneously trains two decoupled and compatible LoRAs (a content LoRA and a style LoRA) from a single image. Through three strategies—prompt separation, column separation, and block separation—the method achieves effective disentanglement of content and style, enabling independent manipulation and free recombination. UnZipLoRA surpasses DreamBooth-LoRA, Inspiration Tree, and B-LoRA across all user preference metrics.
tags:
  - ICCV 2025
  - Robotics
  - LoRA
  - content-style disentanglement
  - diffusion models
  - image generation
  - concept decoupling
date: 2026-05-08
content_hash: 32895193db1c1629
---

# UnZipLoRA: Separating Content and Style from a Single Image

**Conference**: ICCV 2025
**arXiv**: [2412.04465](https://arxiv.org/abs/2412.04465)
**Code**: [https://unziplora.github.io](https://unziplora.github.io)
**Area**: Robotics
**Keywords**: LoRA, content-style disentanglement, diffusion models, image generation, concept decoupling

## TL;DR

This paper proposes UnZipLoRA, a method that simultaneously trains two decoupled and compatible LoRAs (a content LoRA and a style LoRA) from a single image. Through three strategies—prompt separation, column separation, and block separation—the method achieves effective disentanglement of content and style, enabling independent manipulation and free recombination. UnZipLoRA surpasses DreamBooth-LoRA, Inspiration Tree, and B-LoRA across all user preference metrics.

## Background & Motivation

**State of the Field**: Diffusion models (e.g., SDXL) can learn specific subject or style concepts via LoRA fine-tuning. DreamBooth and StyleDrop focus on capturing content or style respectively, but neither can extract both simultaneously from a single image. ZipLoRA can merge independently trained subject and style LoRAs, but requires separate training images for each. B-LoRA exploits block-level specialization in the SDXL U-Net to separate content and style, but the granularity of its partition is too coarse.

**Limitations of Prior Work**: Disentangling content and style from a single image is an ill-posed problem. With naive joint training (e.g., training two LoRAs simultaneously under a prompt such as "A <cc> in <ss> style"), the two LoRAs cross-contaminate: the content LoRA absorbs style information and the style LoRA absorbs content information.

**Root Cause**: A single image must supervise both LoRAs (since it jointly encodes content and style), yet each LoRA must learn only its corresponding concept, and their combination must faithfully reconstruct the original image.

**Paper Goals**: Learn two decoupled and compatible LoRAs from a single image, enabling independent use (generating style or content variants) as well as joint use (reconstructing the original image or creating novel combinations).

**Starting Point**: Cross-attention layers in diffusion models bind text conditioning to visual generation. If the content LoRA can be made to attend only to the content trigger token <cc> and the style LoRA only to the style trigger token <ss>, cross-contamination can be avoided.

**Core Idea**: Three orthogonal separation strategies—prompt separation (to prevent cross-contamination), column separation (to ensure weight orthogonality and compatibility), and block separation (to assign dedicated U-Net blocks to style and content)—are combined to train two LoRAs that are disentangled in both concept space and weight space from a single image.

## Method

### Overall Architecture

The method is built on an SDXL LoRA fine-tuning framework. Given a single stylized image, a content LoRA $L_c = \{\Delta W_c^i\}$ and a style LoRA $L_s = \{\Delta W_s^i\}$ are trained simultaneously. Three separate prompts drive the base model and the two LoRAs during training. At inference, either LoRA can be used independently, or both can be merged via direct addition.

### Key Designs

1. **Prompt Separation**:

    - **Function**: Prevents cross-contamination between the two LoRAs during joint training.
    - **Mechanism**: In cross-attention layers, instead of using a single prompt to drive all weights, three independent prompts compute K/V separately: $K \text{ or } V(x, x_s, x_c) = W_0^T x + \Delta W_s^T x_s + \Delta W_c^T x_c$, where $x$ is the embedding of the full prompt "A <cc> in <ss> style" (used only for the base model $W_0$), $x_c$ is the embedding of the content description <cc> (used only for $\Delta W_c$), and $x_s$ is the embedding of the style description <ss> (used only for $\Delta W_s$).
    - **Design Motivation**: The naive formulation $(W_0 + \Delta W_c + \Delta W_s)^T x$ allows the content LoRA to attend to the style token <ss>, causing cross-contamination. With prompt separation, each LoRA attends exclusively to its own concept token.

2. **Column Separation**:

    - **Function**: Ensures weight orthogonality between the two LoRAs, enabling effective merging via direct addition.
    - **Mechanism**: Dynamic column masks $m_s$ and $m_c$ control the per-column contribution of each LoRA: $K \text{ or } V = W_0^T x + m_s \Delta W_s^T x_s + m_c \Delta W_c^T x_c$. During training, only the top $N\% = 30\%$ most important columns of each weight matrix are activated (importance assessed via the Cone method). Column masks are recalibrated every $t=200$ steps. An orthogonality loss $\mathcal{L}_\perp = \sum_i |m_c^i \cdot m_s^i|$ penalizes overlap between the two LoRAs' active columns.
    - **Design Motivation**: Prompt separation addresses concept binding but does not guarantee weight compatibility. Column masks encourage the two LoRAs to operate on disjoint columns, reducing interference at merge time. Training on only 30% of columns also regularizes learning and prevents overfitting to the single input image.

3. **Block Separation**:

    - **Function**: Assigns dedicated SDXL U-Net blocks to style and content respectively.
    - **Mechanism**: Building on the findings of B-LoRA (different U-Net blocks exhibit different sensitivities to content vs. style), block separation extends to more blocks with finer-grained assignment. The column sparsity constraint is relaxed (all columns used) for the style LoRA in style-sensitive blocks and for the content LoRA in content-sensitive blocks. All upsampling blocks of SDXL are involved, rather than the two blocks used in B-LoRA.
    - **Design Motivation**: 30% of columns may be insufficient for style learning, as style is a global concept requiring greater parameter capacity. Block separation provides adequate representational space for each concept within its dedicated blocks.

### Loss & Training

The training objective combines the DreamBooth reconstruction loss $\mathcal{L}_{DB}$ and the orthogonality loss $\lambda_\perp \mathcal{L}_\perp$ ($\lambda_\perp = 0.5$). The method is based on SDXL v1.0 with LoRA rank 64, Adam optimizer (lr=5e-5), 600 steps, batch size 1, with the base model and text encoders frozen. The content trigger token is "sks [class name]" and the style trigger token is a 2–3 word high-level description (e.g., "watercolor painting").

## Key Experimental Results

### Main Results

User preference study (204 questionnaires, 34 participants):

| Comparison | Decomposition Preference | Recombination Preference |
|---|---|---|
| UnZipLoRA vs DreamBooth-LoRA | 91.17% | 98.10% |
| UnZipLoRA vs Inspiration Tree | 81.53% | 79.17% |
| UnZipLoRA vs B-LoRA | 62.74% | 77.14% |

Automatic alignment scores:

| Method | Style Align. (CLIP-I)↑ | Content Align. (DINO)↑ | Style Align. (CSD)↑ | Content Align. (CSD)↑ |
|---|---|---|---|---|
| DB-LoRA | 0.417 | 0.339 | 0.245 | 0.338 |
| Inspiration Tree | 0.404 | 0.291 | 0.229 | 0.334 |
| B-LoRA | 0.418 | 0.337 | 0.244 | 0.342 |
| **UnZipLoRA** | **0.427** | **0.349** | **0.265** | **0.358** |

### Ablation Study

| Configuration | Content Decomp. Pref.↑ | Style Decomp. Pref.↑ | Recomb. Pref.↑ | Notes |
|---|---|---|---|---|
| M1 vs Baseline | 91.67% | 12.35% | 92.80% | Prompt separation greatly improves content de-stylization but style learning is insufficient |
| M2 vs M1 | 55.74% | 39.51% | 93.64% | Column separation reduces interference and improves compatibility |
| M3 vs M2 | 55.36% | **86.42%** | 61.90% | Block separation significantly improves style capture |

### Key Findings

- **All three strategies are complementary and indispensable**: Prompt separation resolves de-stylization (content preference 91.67%) but yields poor style learning (only 12.35%); column separation improves compatibility (recombination preference 93.64%); block separation dramatically improves style fidelity (86.42%).
- **CSD is the most discriminative metric**: CLIP-I and DINO are insufficiently sensitive to style alignment; the dedicated CSD model more clearly separates performance differences across methods.
- **Training on 30% of columns is sufficient**: Column sparsity not only preserves quality but improves concept separation through its regularization effect.
- The method generalizes to other architectures such as KOALA, though quality is slightly below SDXL.
- Cross-image content/style LoRA combinations also yield coherent results.

## Highlights & Insights

- **The problem framing as a reverse operation is highly innovative**: ZipLoRA merges two independent LoRAs (zip); UnZipLoRA decomposes a single image into two LoRAs (unzip). The inverse problem is harder but more practical.
- **Prompt separation is the key technical contribution**: Using three distinct prompts to drive the base model and the two LoRAs separately within cross-attention layers is an elegant solution that addresses the core challenge of cross-contamination in joint training.
- **Dynamic column importance recalibration**: Rather than statically assigning columns, the column masks are recalibrated every 200 steps based on current weight importance, making the allocation adaptive to the training state.
- **Practical application potential**: Designers can extract style and content from any reference image, edit them independently, or cross-combine them, substantially enhancing creative flexibility.

## Limitations & Future Work

- For highly abstract styles (e.g., artistic styles involving significant geometric distortion), accurate content de-stylization may not be achievable.
- Only single-image training is supported; multiple images could provide richer concept information.
- Manual selection of trigger tokens (content class label + brief style description) is required; automation could be improved.
- 600 training steps may be insufficient for complex concepts.
- Validation on DiT architectures has not been performed; the transferability of the block separation strategy remains to be confirmed.
- Evaluation relies solely on user preference studies; quantitative evaluation on downstream tasks (e.g., FID for style transfer quality) is absent.

## Related Work & Insights

- **vs ZipLoRA**: ZipLoRA merges two existing LoRAs (forward problem); UnZipLoRA decomposes a single image (inverse problem). ZipLoRA's weight optimization ideas are instructive but cannot be directly inverted.
- **vs B-LoRA**: B-LoRA trains content and style independently using two U-Net blocks, resulting in too coarse a partition—residual style leaks into content and background leaks into style. UnZipLoRA's multi-block and fine-grained column separation is more effective.
- **vs Inspiration Tree**: Relies on Textual Inversion, learning only text embeddings without fine-tuning weights, which limits expressiveness and fails to capture fine-grained details.
- **vs CusConcept**: Requires LLM-assisted data augmentation, incurring high computational cost and not producing LoRAs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Disentangling content and style LoRAs from a single image is a genuinely novel problem formulation; the three separation strategies are elegant and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ User studies, automatic metrics, ablations, and cross-architecture validation are comprehensive, though large-scale quantitative evaluation is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is clearly developed, the method is presented with progressive logic, and figures are rich and intuitive.
- Value: ⭐⭐⭐⭐ Clear value for creative generation and design workflows, though the connection to robotics is tenuous.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents](../../NeurIPS2025/robotics/mip_against_agent_malicious_image_patches_hijacking_multimod.md)
- [\[ICLR 2026\] One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration](../../ICLR2026/robotics/one_demo_is_all_it_takes_planning_domain_derivation_with_llms_from_a_single_demo.md)
- [\[ICLR 2026\] TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models](../../ICLR2026/robotics/twinvla_data-efficient_bimanual_manipulation_with_twin_single-arm_vision-languag.md)
- [\[AAAI 2026\] From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations](../../AAAI2026/robotics/from_passive_perception_to_active_memory_a_weakly_supervised_image_manipulation_.md)
- [\[ICCV 2025\] EvolvingGrasp: Evolutionary Grasp Generation via Efficient Preference Alignment](evolvinggrasp_evolutionary_grasp_generation_via_efficient_preference_alignment.md)

<!-- RELATED:END -->
