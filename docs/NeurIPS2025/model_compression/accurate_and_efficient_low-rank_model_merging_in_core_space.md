---
title: >-
  [Paper Note] Accurate and Efficient Low-Rank Model Merging in Core Space
description: >-
  [NeurIPS 2025][Model Compression][Model Merging] This paper proposes the Core Space Merging framework, which performs model merging within a common reference basis space constructed from low-rank LoRA matrices. This approach **losslessly** reduces the merging operation from the full $m \times n$ space to a compact $Tr \times Tr$ space (where $T$ is the number of tasks and $r$ is the LoRA rank), achieving state-of-the-art merging accuracy on Llama 3 8B while reducing computational cost by several orders of magnitude.
tags:
  - NeurIPS 2025
  - Model Compression
  - Model Merging
  - LoRA
  - Low-Rank Projection
  - Parameter-Efficient Fine-Tuning
  - Core Space
date: 2026-05-08
content_hash: 34893fb0b3017ab5
---

# Accurate and Efficient Low-Rank Model Merging in Core Space

**Conference**: NeurIPS 2025
**arXiv**: [2509.17786](https://arxiv.org/abs/2509.17786)
**Code**: [GitHub](https://github.com/apanariello4/core-space-merging)
**Area**: Model Compression / LLM Efficiency
**Keywords**: Model Merging, LoRA, Low-Rank Projection, Parameter-Efficient Fine-Tuning, Core Space

## TL;DR
This paper proposes the Core Space Merging framework, which performs model merging within a common reference basis space constructed from low-rank LoRA matrices. This approach **losslessly** reduces the merging operation from the full $m \times n$ space to a compact $Tr \times Tr$ space (where $T$ is the number of tasks and $r$ is the LoRA rank), achieving state-of-the-art merging accuracy on Llama 3 8B while reducing computational cost by several orders of magnitude.

## Background & Motivation

**Background**: As model scale grows, parameter-efficient fine-tuning (PEFT) methods such as LoRA have become mainstream. Model merging aims to combine multiple task-specific LoRA adapters into a single multi-task model without additional training.

**Limitations of Prior Work**: (a) Directly applying Task Arithmetic to LoRA matrices yields poor results due to misaligned bases across tasks; (b) KnOTS performs merging in an aligned space but requires SVD over full-size matrices, incurring $O(n^3 T^2)$ complexity that is prohibitive for large models; (c) Advanced merging methods (TSV, Iso-C) operating in the full space are equally expensive.

**Key Challenge**: Efficient merging (direct full-space Task Arithmetic) sacrifices accuracy, while accurate merging (KnOTS + TSV) forfeits the computational efficiency advantage of low-rank LoRA representations.

**Goal**: Achieve high-accuracy LoRA model merging while preserving the computational efficiency of the low-rank structure.

**Key Insight**: The observation that LoRA updates across all tasks share a common subspace — identifying a reference basis for this subspace and performing merging within it.

**Core Idea**: Merge within the Core Space — a common reference basis space spanned by the SVDs of the LoRA low-rank matrices — whose dimensionality is only $Tr \times Tr$, with strictly zero information loss.

## Method

### Overall Architecture
Given $T$ task-specific LoRA updates $\{(A^{(t)}, B^{(t)})\}_{t=1}^T$ → apply SVD to stacked $A^{(t)}$ and $B^{(t)}$ matrices to obtain reference bases $(U_B^{ref}, V_A^{ref})$ → project each task update into the Core Space to obtain core matrices $M^{(t)} \in \mathbb{R}^{Tr \times Tr}$ → apply any merging method $\mathcal{M}$ within the Core Space → project back to the original parameter space.

### Key Designs

1. **Reference Bases Construction**:

    - Function: Identify a set of orthonormal bases that jointly represent all task-specific LoRA directions without information loss.
    - Mechanism: Horizontally concatenate all $B^{(t)}$ matrices and vertically concatenate all $A^{(t)}$ matrices, apply SVD to each, and extract $U_B^{ref} \in \mathbb{R}^{m \times Tr}$ and $V_A^{ref} \in \mathbb{R}^{n \times Tr}$.
    - Design Motivation: SVD over the concatenated matrices naturally spans the union of all task subspaces.
    - Key Advantage: SVD operates on $\mathbb{R}^{Tr \times n}$ rather than $\mathbb{R}^{Tn \times n}$ as in KnOTS, substantially reducing computation.

2. **Core Matrix Projection**:

    - Function: Project each task update into the compact $Tr \times Tr$ Core Space.
    - Mechanism: $M^{(t)} = (U_B^{ref\top} B^{(t)})(A^{(t)} V_A^{ref}) \in \mathbb{R}^{Tr \times Tr}$
    - Theoretical Guarantee: The projection error is proven to be **strictly zero** via least-squares analysis — $\|U_B^{ref} R_B^{(t)} - U_B^{(t)}\|_F^2 = 0$ — since the column space of the reference basis subsumes the column space of each individual task basis.
    - Reconstruction: $\Delta W^{(t)} = U_B^{ref} M^{(t)} V_A^{ref\top}$, with exact lossless recovery.

3. **Merging in Core Space**:

    - Function: Execute any merging method within the compact $Tr \times Tr$ space.
    - Mechanism: $M_{merged} = \mathcal{M}(\{M^{(t)}\}_{t=1}^T)$, followed by $\Delta W = U_B^{ref} M_{merged} V_A^{ref\top}$.
    - For linear merging methods (TA): Core Space merging is **mathematically equivalent** to full-space merging.
    - For nonlinear methods (TIES, TSV, Iso-C): Operating in Core Space yields **superior performance**, as the aligned representations reduce cross-task directional interference.

### Computational Complexity

| Method | TA | Iso-C | TSV |
|--------|----|-------|-----|
| Full Space | $O(n^2 Tr)$ | $O(n^3)$ | $O(n^3 T)$ |
| KnOTS | $O(n^3 T^2)$ | $O(n^3 T^2)$ | $O(n^3 T^2)$ |
| **Core Space** | $O(n^2 Tr)$ | $O(n^2 Tr + T^3 r^3)$ | $O(n^2 Tr + T^4 r^3)$ |

## Key Experimental Results

### Main Results (Llama 3 8B, 6 NLI Tasks)

| Method | Space | Avg. Normalized Accuracy | Time (s) |
|--------|-------|--------------------------|----------|
| TA | Full/Core | ~90% | Baseline |
| TIES | Full | ~91% | Moderate |
| TSV | Full | ~93% | Very high |
| TIES | KnOTS | ~92% | Very high |
| TIES | **Core** | **~94%** | **Low** |
| TSV+Iso-C | **Core** | **SOTA** | **Low** |

### Ablation Study (ViT-B/32, 8 Vision Tasks)

| Merging Space | TIES Norm. Accuracy | TSV Norm. Accuracy |
|---------------|--------------------|--------------------|
| Full Space | Baseline | Baseline |
| KnOTS | +Gain but slow | +Gain but very slow |
| **Core Space** | **Highest** | **Highest** |

### Key Findings
- **Nonlinear merging is more effective in Core Space**: For nonlinear methods such as TIES and TSV, Core Space merging is not only faster but also yields higher accuracy.
- **Efficiency gains of several orders of magnitude**: On Llama 3 8B, TSV completes in seconds in Core Space versus hours in Full Space.
- **Extremely compact Core Space dimensionality**: For $T=6, r=16$, the core matrix is only $96 \times 96$ (vs. $4096 \times 4096$ in full space).
- **Rigorous proof of zero information loss**: The zero projection error is not an approximation but a mathematically exact result.

## Highlights & Insights
- **Theoretical elegance of "compact space, superior performance"**: From the perspective of joint subspace analysis, the $Tr \times Tr$ space is shown to be sufficient for lossless representation of all task information, and merging within it yields better results due to improved alignment.
- **Fundamental distinction from KnOTS**: KnOTS performs SVD on full-size matrices ($O(n^3 T^2)$), whereas Core Space performs SVD on small matrices ($O(n^2 Tr)$) — an efficiency gap of a factor of $n/r$.
- **Benefits for nonlinear methods**: Operating in the aligned space reduces cross-task directional interference, thereby mitigating sign conflicts in TIES.
- **Composability**: Orthogonally compatible with any existing merging method as a plug-and-play module.

## Limitations & Future Work
- **Assumption $Tr \leq \min(m,n)$**: The product of task count and rank must not exceed the model dimension; truncation may be required for a very large number of tasks.
- **Restricted to LoRA**: Application to fully fine-tuned models requires a prior low-rank approximation step.
- **No benefit for linear methods**: For linear merging methods such as Task Arithmetic, Core Space and Full Space are mathematically equivalent, and the advantage is exclusive to nonlinear methods.
- **Future Directions**: (1) Adaptive rank selection; (2) Extension to other PEFT methods (e.g., Adapter, Prefix Tuning); (3) Investigation of optimal merging strategies within Core Space.

## Related Work & Insights
- **vs. KnOTS**: Both merge in an aligned space, but KnOTS constructs the alignment via SVD on concatenated full-size matrices, whereas Core Space exploits the low-rank structure of LoRA for a fundamentally more efficient construction.
- **vs. Task Arithmetic**: TA is a special case of linear merging in Core Space, but nonlinear methods gain additional benefits from the Core Space alignment.
- **vs. TSV / Iso-C**: These advanced merging methods can be executed at negligible cost within Core Space, making them scalable to large models for the first time.

## Rating
- Novelty: ⭐⭐⭐⭐ The Core Space concept is elegant, and the zero information loss proof is a theoretical highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers vision (ViT-B/32, ViT-L/14) and language (Llama 3 8B) benchmarks with comparisons across multiple merging methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are clear and rigorous, with a complete proof of zero projection error.
- Value: ⭐⭐⭐⭐⭐ Enables advanced model merging methods to scale to large models for the first time, offering both theoretical and practical contributions.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging](mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](gora_gradient-driven_adaptive_low_rank_adaptation.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)

<!-- RELATED:END -->
