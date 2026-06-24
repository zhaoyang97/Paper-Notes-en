---
title: >-
  [Paper Note] OPRO: Orthogonal Panel-Relative Operators for Panel-Aware In-Context Image Generation
description: >-
  [CVPR 2026][Image Generation][In-context image generation] This paper proposes OPRO, a parameter-efficient adaptation method based on orthogonal matrices. By imposing learnable panel-specific orthogonal operators on the position-aware query/key of a frozen backbone, it explicitly modulates inter-panel attention interactions while preserving pre-trained intra-panel synthesis behavior. With only 0.93M additional parameters, OPRO significantly enhances the editing quality of sev…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "In-context image generation"
  - "Orthogonal operators"
  - "Parameter-efficient fine-tuning"
  - "Panel-aware attention"
  - "Diffusion Transformer"
date: 2026-05-08
content_hash: d33394d4187c3fad
---

# OPRO: Orthogonal Panel-Relative Operators for Panel-Aware In-Context Image Generation

**Conference**: CVPR 2026  
**arXiv**: [2603.27637](https://arxiv.org/abs/2603.27637)  
**Code**: None  
**Area**: Diffusion Models / Image Generation  
**Keywords**: In-context image generation, Orthogonal operators, Parameter-efficient fine-tuning, Panel-aware attention, Diffusion Transformer

## TL;DR

This paper proposes OPRO, a parameter-efficient adaptation method based on orthogonal matrices. By imposing learnable panel-specific orthogonal operators on the position-aware query/key of a frozen backbone, it explicitly modulates inter-panel attention interactions while preserving pre-trained intra-panel synthesis behavior. With only 0.93M additional parameters, OPRO significantly enhances the editing quality of several SOTA methods on MagicBrush.

## Background & Motivation

1. **Background**: In-context image generation (ICG) is a significant application for diffusion models, where reference and target images are arranged in a tiled-panel layout for exemplar-based generation. Current methods follow two paradigms: global canvas encoding (e.g., FluxFill) based on inpainting and per-panel encoding (e.g., UNO) based on T2I.

2. **Limitations of Prior Work**: Regardless of global or per-panel encoding, attention mechanisms remain **panel-unaware**. In a global canvas, tokens from different panels are treated as regions on the same canvas; in per-panel encoding, different panels might share identical position indices. The attention layers cannot distinguish whether a pair of tokens originates from the same panel or different panels.

3. **Key Challenge**: Standard PEFT (such as LoRA) must simultaneously learn two tasks: (1) inter-panel relationship transfer and (2) preservation of pre-trained intra-panel synthesis. This dual burden leads to inefficient adaptation.

4. **Goal**: Design an adaptation mechanism that can explicitly distinguish between inter-panel and intra-panel attention interactions.

5. **Key Insight**: Orthogonal transformations preserve inner products. Consequently, applying the same orthogonal operator to tokens within the same panel keeps intra-panel attention scores invariant. Combinations of orthogonal operators from different panels introduce learnable cross-panel modulation.

6. **Core Idea**: "Rotate" Q/K using panel-specific orthogonal operators, allowing cross-panel attention to be learnable while keeping same-panel attention invariant.

## Method

### Overall Architecture

OPRO addresses the "panel-unawareness" in attention within in-context image generation. When multiple panels (reference and target images) are concatenated into a token sequence of length $L$, the attention layer lacks knowledge regarding whether a token pair is from the same or different panels, causing inter-panel relationship transfer and intra-panel synthesis to be confounded. Instead of modifying the backbone weights, OPRO assigns a learnable orthogonal matrix $U_p$ to each panel in every attention layer. Before computing attention, the frozen position-aware Q/K are rotated. After rotation, tokens within the same panel cancel out the transformation because they share the same $U_p$, leaving the scores unchanged. Tokens across different panels interact through a relative operator $U_{p(i)}^\top U_{p(j)}$, which is the primary learnable component. The pipeline follows: "concatenate sequence $\to$ rotate Q/K by panel per layer $\to$ freeze intra-panel and adjust inter-panel."

### Key Designs

**1. Orthogonal Panel Operators: Freezing Intra-panel Attention while Enabling Inter-panel Interaction**

The most critical property of orthogonal transformations is the preservation of inner products. OPRO learns an orthogonal matrix $U_p \in SO(d_h)$ for the $p$-th panel. After rotation, the attention score between two tokens becomes:

$$s'_{ij} = \tilde{q}_i^\top \, (U_{p(i)}^\top U_{p(j)}) \, \tilde{k}_j \big/ \sqrt{d_h}.$$

When $p(i)=p(j)$ (same panel), $U_p^\top U_p = I$, making the score invariant. This ensures Same-Panel Invariance, preserving pre-trained synthesis behaviors. When $p(i)\neq p(j)$ (cross-panel), the relative operator $U_{p(i)}^\top U_{p(j)}$ provides a learnable modulation for inter-panel relations. Furthermore, orthogonality ensures Isometry ($\|\hat q_i\| = \|\tilde q_i\|$), preventing unintended scaling of attention logits. This decouples the dual burden found in standard PEFT.

**2. Low-rank Lie Exponential Parameterization: Converting Orthogonal Constraints into Unconstrained Optimization**

Optimizing directly on the manifold of orthogonal matrices requires constrained optimization like Riemannian gradient descent, which is computationally expensive. OPRO parameterizes each panel using two low-rank matrices $L_p, R_p \in \mathbb{R}^{d_h \times \rho}$ to construct a skew-symmetric generator:

$$A_p = L_p R_p^\top - R_p L_p^\top,$$

which is then mapped to the orthogonal group via the matrix exponential $U_p = \exp(A_p)$. The skew-symmetry ($A_p^\top = -A_p$) ensures that $\exp(A_p)$ is orthogonal, satisfying the constraint structurally. $L_p$ and $R_p$ can be optimized in unconstrained Euclidean space using Adam. With $\rho=8$, the adapter accounts for only 0.13% of the backbone parameters.

**3. Zero Interference Initialization: Identity Mapping at the Start of Training**

To avoid disrupting pre-trained attention distributions initially, OPRO adopts a zero-initialization strategy. By setting $L_p = 0$ and $R_p \sim \mathcal N(0,\sigma^2)$, then $A_p = 0$ and $U_p = \exp(0) = I$. Training begins as an identity transformation, identical to the original model. Crucially, the gradient at $L_p$ is non-zero, allowing the optimization to proceed immediately without being stuck.

### Loss & Training

OPRO is integrated into the attention layers of existing methods and does not introduce extra losses. It is trained end-to-end following the host method's objective. On MagicBrush, it is trained for 5000 steps using the Adam optimizer with a learning rate of $1 \times 10^{-4}$ and a batch size of 8.

## Key Experimental Results

### Main Results

Instruction editing results on the MagicBrush test set ($\rho=32$, with +0.93M parameters):

| Method | Trainable Params | L1 ↓ | CLIP-I ↑ | DINO ↑ |
|------|---------|------|----------|--------|
| ACE++ | 76.6M | 0.1215 | 0.8658 | 0.7394 |
| ACE++ + OPRO | +0.93M | 0.1114 | 0.8749 | 0.7767 |
| ICEdit | 22.4M | 0.1189 | 0.8703 | 0.7706 |
| ICEdit + OPRO | +0.93M | **0.0781** | **0.9002** | **0.8531** |
| UNO | 478.2M | 0.0575 | 0.9236 | 0.8961 |
| UNO + OPRO | +0.93M | **0.0387** | **0.9281** | **0.8980** |

### Ablation Study

Ablation on a two-stage combinatorial reasoning task (ViT-B, 3×3):

| Method | Isometry | SP-Inv | Accuracy |
|------|----------|--------|----------|
| LoRA (Baseline) | - | - | 36.20 |
| + APB (Additive Bias) | No | No | 35.70 |
| + Asym-OPRO | Yes | No | 39.70 |
| + OPRO (w/o Zero Init) | Yes | Yes | 38.60 |
| + OPRO (Ours) | Yes | Yes | **42.00** |

### Key Findings

- **Significant Improvement for ICEdit**: L1 decreased by 34.31% and DINO increased from 0.7706 to 0.8531, indicating that OPRO is highly effective for medium-scale models.
- **High Parameter Efficiency**: At $\rho=8$, it adds only 0.111M parameters, which is 8.4% of LoRA parameters and 0.13% of the backbone.
- **Generalization across Position Encodings**: Improvements are observed across APE, RoPE, LieRE, and ComRoPE.
- **Importance of Properties**: Removing Isometry (APB) results in performance lower than the baseline. Removing SP-Inv (Asym-OPRO) yields improvements but remains inferior to full OPRO.

## Highlights & Insights

- **Theoretical Elegance**: By leveraging mathematical properties of the orthogonal group, it strictly guarantees Isometry and Same-Panel Invariance, providing clear theoretical grounding.
- **Clever Lie Algebra Parameterization**: Translates constrained optimization into an unconstrained problem while maintaining low-rank efficiency, applicable to other scenarios requiring orthogonal constraints.
- **Strong Transferability**: OPRO makes no assumptions about the underlying position encoding, allowing it to be seamlessly inserted into both inpainting and T2I paradigms.

## Limitations & Future Work

- Training assumes fixed panel layouts; while shared operators can handle multi-reference configs at inference, entirely new layouts may require additional training.
- Matrix exponential operations introduce some overhead in training time and inference latency.
- The combination with other PEFT methods like AdaLoRA or QLoRA remains unexplored.

## Related Work & Insights

- **vs LoRA**: LoRA adds low-rank updates to Q/K/V but is panel-unaware. OPRO focuses on panel-aware modulation at the position encoding level; the two are complementary.
- **vs RoPE/LieRE/ComRoPE**: These are position encoding methods. OPRO acts as a "meta-position encoding" by adding panel-level orthogonal modulation on top.
- **vs ControlNet zero-init**: It adopts the zero-initialization philosophy but implements it more elegantly on the orthogonal manifold.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of orthogonal operators and panel-awareness is highly innovative with solid theoretical backing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes both synthetic tasks and real editing with precise ablations, though quantitative evaluation scenarios for real applications could be broader.
- Writing Quality: ⭐⭐⭐⭐⭐ Complete logic chain from motivation to theory and experiments; proofs are rigorous.
- Value: ⭐⭐⭐⭐ highly valuable for panel-based ICG, though the application scenario is relatively specific.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Panel-by-Panel Souls: A Performative Workflow for Expressive Faces in AI-Assisted Manga Creation](../../NeurIPS2025/image_generation/panel-by-panel_souls_a_performative_workflow_for_expressive_faces_in_ai-assisted.md)
- [\[CVPR 2026\] CAST: Context-Aware Dynamic Latent Space Transformation for Interactive Text-to-Image Retrieval](cast_context-aware_dynamic_latent_space_transformation_for_interactive_text-to-i.md)
- [\[CVPR 2026\] Re-Align: Structured Reasoning-guided Alignment for In-Context Image Generation and Editing](re-align_structured_reasoning-guided_alignment_for_in-context_image_generation_a.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](freqflow_frequency_aware_flow_matching.md)

</div>

<!-- RELATED:END -->
