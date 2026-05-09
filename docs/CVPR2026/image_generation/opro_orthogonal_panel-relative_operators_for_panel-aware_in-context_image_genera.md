---
title: >-
  [Paper Note] OPRO: Orthogonal Panel-Relative Operators for Panel-Aware In-Context Image Generation
description: >-
  [CVPR 2026][Image Generation][In-context image generation] This paper proposes OPRO, a parameter-efficient adaptation method based on orthogonal matrices. By applying learnable panel-specific orthogonal operators to position-aware queries and keys of a frozen backbone, OPRO explicitly modulates cross-panel attention interactions while preserving the pre-trained same-panel synthesis behavior. With only 0.93M additional parameters, it significantly improves the editing quality of multiple state-of-the-art methods on MagicBrush.
tags:
  - CVPR 2026
  - Image Generation
  - In-context image generation
  - orthogonal operators
  - parameter-efficient fine-tuning
  - panel-aware attention
  - diffusion Transformer
date: 2026-05-08
content_hash: 91e8113985bd9e5c
---

# OPRO: Orthogonal Panel-Relative Operators for Panel-Aware In-Context Image Generation

**Conference**: CVPR 2026
**arXiv**: [2603.27637](https://arxiv.org/abs/2603.27637)
**Code**: None
**Area**: Diffusion Models / Image Generation
**Keywords**: In-context image generation, orthogonal operators, parameter-efficient fine-tuning, panel-aware attention, diffusion Transformer

## TL;DR

This paper proposes OPRO, a parameter-efficient adaptation method based on orthogonal matrices. By applying learnable panel-specific orthogonal operators to position-aware queries and keys of a frozen backbone, OPRO explicitly modulates cross-panel attention interactions while preserving the pre-trained same-panel synthesis behavior. With only 0.93M additional parameters, it significantly improves the editing quality of multiple state-of-the-art methods on MagicBrush.

## Background & Motivation

1. **State of the Field**: In-context image generation (ICG) is an important application of diffusion models, achieving example-based generation by arranging reference and target images in a tiled-panel layout. Existing approaches fall into two paradigms: inpainting-based global canvas encoding (e.g., FluxFill) and T2I-based per-panel encoding (e.g., UNO).

2. **Limitations of Prior Work**: In both paradigms, the attention mechanism is **panel-agnostic**. In global canvas encoding, tokens from different panels are treated as different regions of the same canvas; in per-panel encoding, different panels may share the same positional indices. The attention layers are entirely unaware of whether a pair of tokens originates from the same panel or different panels.

3. **Root Cause**: Standard PEFT methods (e.g., LoRA) must simultaneously learn two objectives: (1) transferring cross-panel relationships, and (2) preserving pre-trained same-panel synthesis behavior. This dual burden leads to low adaptation efficiency.

4. **Paper Goals**: Design an adaptation mechanism that explicitly distinguishes between inter-panel and intra-panel attention interactions.

5. **Starting Point**: Orthogonal transformations preserve inner products; thus, if the same orthogonal operator is applied to tokens within the same panel, same-panel attention scores remain unchanged, while the combination of different orthogonal operators across panels introduces learnable cross-panel modulation.

6. **Core Idea**: "Rotate" Q/K with panel-specific orthogonal operators so that cross-panel attention becomes learnable while same-panel attention remains invariant.

## Method

### Overall Architecture

Given a tiled-panel layout with $P$ panels, all panel tokens are concatenated into a sequence of length $L$. Each token $i$ is associated with a panel index $p(i)$ and spatial coordinate $x_i$. At each attention layer of the backbone, OPRO applies a panel-specific orthogonal operator $U_{p(i)}$ to the frozen position-aware Q/K, so that same-panel attention is invariant while cross-panel attention is modulated by the relative orthogonal operator $U_{p(i)}^\top U_{p(j)}$.

### Key Designs

1. **Orthogonal Panel-Relative Operator**:

    - **Function**: Learn an orthogonal matrix $U_p \in SO(d_h)$ per panel and apply it to Q/K.
    - **Mechanism**: The transformed attention score is $s'_{ij} = \tilde{q}_i^\top (U_{p(i)}^\top U_{p(j)}) \tilde{k}_j / \sqrt{d_h}$. When $p(i) = p(j)$, $U_p^\top U_p = I$ and the score is unchanged (Same-Panel Invariance); when $p(i) \neq p(j)$, $U_{p(i)}^\top U_{p(j)}$ provides learnable cross-panel modulation. Orthogonality further guarantees $\|\hat{q}_i\| = \|\tilde{q}_i\|$ (Isometry), preventing unintended scaling of attention logits.
    - **Design Motivation**: Decouple the dual burden of cross-panel learning and same-panel preservation, allowing the adapter to focus solely on cross-panel relationship learning.

2. **Low-Rank Lie Exponential Parameterization**:

    - **Function**: Efficiently optimize over the orthogonal matrix manifold.
    - **Mechanism**: For each panel, two low-rank matrices $L_p, R_p \in \mathbb{R}^{d_h \times \rho}$ are learned to construct a skew-symmetric generator $A_p = L_p R_p^\top - R_p L_p^\top$, from which the orthogonal matrix is obtained via matrix exponential $U_p = \exp(A_p)$. Skew-symmetry guarantees orthogonality.
    - **Design Motivation**: Avoid costly constrained optimization such as Riemannian gradient descent; standard optimizers can operate in unconstrained Euclidean space.

3. **Zero-Interference Initialization**:

    - **Function**: Ensure that OPRO acts as an identity mapping at the start of training, without disturbing pre-trained weights.
    - **Mechanism**: Set $L_p = 0$, $R_p \sim \mathcal{N}(0, \sigma^2)$, so that $A_p = 0$ and $U_p = \exp(0) = I$. The gradient with respect to $L_p$ is nonzero, so optimization can begin immediately.
    - **Design Motivation**: Inspired by ControlNet's zero-initialization strategy, this ensures the adapter does not corrupt pre-trained representations.

### Loss & Training

OPRO is inserted as a module into the attention layers of existing methods and trained end-to-end with the original training objective. Training is conducted on MagicBrush for 5,000 steps using the Adam optimizer with a learning rate of $1 \times 10^{-4}$ and batch size 8.

## Key Experimental Results

### Main Results

Instruction-based editing results on the MagicBrush test set ($\rho=32$, only +0.93M additional parameters):

| Method | Trainable Params | L1 ↓ | CLIP-I ↑ | DINO ↑ |
|--------|-----------------|------|----------|--------|
| ACE++ | 76.6M | 0.1215 | 0.8658 | 0.7394 |
| ACE++ + OPRO | +0.93M | 0.1114 | 0.8749 | 0.7767 |
| ICEdit | 22.4M | 0.1189 | 0.8703 | 0.7706 |
| ICEdit + OPRO | +0.93M | **0.0781** | **0.9002** | **0.8531** |
| UNO | 478.2M | 0.0575 | 0.9236 | 0.8961 |
| UNO + OPRO | +0.93M | **0.0387** | **0.9281** | **0.8980** |

### Ablation Study

Ablation on a two-stage compositional reasoning task (ViT-B, 3×3):

| Method | Isometry | SP-Inv | Accuracy |
|--------|----------|--------|----------|
| LoRA (Baseline) | - | - | 36.20 |
| + APB (additive bias) | No | No | 35.70 |
| + Asym-OPRO | Yes | No | 39.70 |
| + OPRO (w/o Zero Init) | Yes | Yes | 38.60 |
| + OPRO (Ours) | Yes | Yes | **42.00** |

### Key Findings

- **Most significant gains on ICEdit**: L1 decreases by 34.31% (0.1189→0.0781) and DINO improves from 0.7706 to 0.8531, suggesting OPRO is most effective on medium-scale models.
- **Exceptional parameter efficiency**: At $\rho=8$, only 0.111M parameters are added, accounting for 8.4% of LoRA parameters and 0.13% of the backbone.
- **Generalization across positional encodings**: Consistent improvements are observed on APE, RoPE, LieRE, and ComRoPE; on 4×4 grids with ComRoPE, the gain reaches +18.0%.
- **Both properties are necessary**: Removing Isometry (APB) yields performance below the baseline; removing SP-Inv (Asym-OPRO) provides some improvement but is inferior to the full OPRO.

## Highlights & Insights

- **Theoretical elegance**: The mathematical properties of the orthogonal group rigorously guarantee two propositions (Isometry and Same-Panel Invariance), making this one of the few PEFT works with clear theoretical guarantees.
- **Clever Lie algebra parameterization**: The constrained optimization problem is converted into an unconstrained one while maintaining parameter efficiency via low-rank structure; this technique is transferable to other settings requiring orthogonal constraints.
- **Strong transferability**: OPRO makes no assumptions about the underlying positional encoding and can be seamlessly inserted into both the inpainting (global canvas) and T2I (per-panel) paradigms.

## Limitations & Future Work

- The panel layout (e.g., 2-panel) is fixed during training; while shared operators can handle multi-reference configurations at inference time, entirely new layouts require additional training.
- The matrix exponential operation introduces additional training time and inference latency.
- The combination of OPRO with other PEFT methods (e.g., AdaLoRA, QLoRA) has not been explored.

## Related Work & Insights

- **vs. LoRA**: LoRA adds low-rank updates to Q/K/V but is panel-agnostic; OPRO focuses on panel-aware modulation at the positional encoding level. The two approaches are complementary.
- **vs. RoPE/LieRE/ComRoPE**: These are positional encoding methods; OPRO superimposes panel-level orthogonal modulation on top of them, functioning as a form of "meta positional encoding."
- **vs. ControlNet zero-init**: The zero-initialization idea is borrowed from ControlNet, but its realization on the orthogonal manifold is more sophisticated.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of orthogonal operators and panel-awareness is highly innovative, with clear theoretical guarantees.
- Experimental Thoroughness: ⭐⭐⭐⭐ Both synthetic and real-world editing evaluations are conducted with well-designed ablations, though the number of quantitative evaluation scenarios for real-world applications is relatively limited.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from motivation to theory to experiments is complete, and the proposition proofs are rigorous.
- Value: ⭐⭐⭐⭐ Highly valuable for panel-based ICG, though the application scope is relatively narrow.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation](edgedit_hardware-aware_diffusion_transformers_for_efficient_on-device_image_gene.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation](adapt_attention_driven_adaptive_prompt_scheduling_and_interpolating_orthogonal_c.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](freqflow_frequency_aware_flow_matching.md)
- [\[CVPR 2026\] FontCrafter: High-Fidelity Element-Driven Artistic Font Creation with Visual In-Context Generation](fontcrafter_high-fidelity_element-driven_artistic_font_creation_with_visual_in-c.md)

<!-- RELATED:END -->
