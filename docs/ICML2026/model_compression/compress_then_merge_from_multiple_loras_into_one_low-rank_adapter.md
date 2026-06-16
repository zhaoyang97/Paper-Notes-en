---
title: >-
  [Paper Note] Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter
description: >-
  [ICML 2026][Model Compression][Paper Note] The Compress-then-Merge (CtM) pipeline is proposed to learn a shared $r$-dimensional subspace and project each adapter into $r \times r$ coordinate matrices before merging multiple LoRAs. Merging is then executed in the low-dimensional space, architecturally ensuring the output is a rank-$r$ LoRA and avoiding the perfo
tags:
  - ICML 2026
  - Model Compression
date: 2026-05-08
content_hash: caa387a3d1227e42
---
# Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter

**Conference**: ICML2026  
**arXiv**: [2606.03723](https://arxiv.org/abs/2606.03723)  
**Code**: https://github.com/ZhengbaoHe/compress-then-merge  
**Area**: Model Compression  
**Keywords**: LoRA Merging, Low-rank Constraint, Shared Subspace, Tucker Decomposition, Parameter-Efficient Fine-Tuning  

## TL;DR
The Compress-then-Merge (CtM) pipeline is proposed to learn a shared $r$-dimensional subspace and project each adapter into $r \times r$ coordinate matrices before merging multiple LoRAs. Merging is then executed in the low-dimensional space, architecturally ensuring the output is a rank-$r$ LoRA and avoiding the performance loss associated with truncated SVD in traditional Merge-then-Compress methods.

## Background & Motivation

**Background**: LoRA has become the standard choice for parameter-efficient fine-tuning of large models, with a vast number of LoRA adapters for different tasks accumulated on platforms like HuggingFace. Merging multiple LoRAs into a single multi-task adapter is an increasingly important practical requirement.

**Limitations of Prior Work**: The current mainstream approach is Merge-then-Compress (MtC)—merging multiple LoRAs in the full parameter space to obtain a high-rank update $\Delta W_{\text{merged}}$, followed by compression to a target rank $r$ via truncated SVD. This "merge-then-compress" strategy treats the rank constraint as an afterthought, leading to two problems: (1) Huge norm discrepancies between LoRAs of different tasks cause truncated SVD (which is Frobenius norm optimal) to favor tasks with large norms, systematically suppressing tasks with smaller norms. (2) "Invasive dimensions"—high-energy directions weakly correlated with task accuracy generated during LoRA training—may monopolize the limited rank budget.

**Key Challenge**: The merged rank-$r$ result must reside within some $r$-dimensional subspace. MtC treats subspace selection as a byproduct of the merged result's spectrum rather than an active design choice. Once a subspace is determined, all components orthogonal to it are irrecoverable; therefore, subspace selection should be the core design consideration.

**Key Insight**: Recent research indicates that weight changes in multi-task models tend to concentrate in low-dimensional spectral subspaces. Based on this, the authors propose **learning the shared subspace before merging**, transforming subspace selection from a passive byproduct into an actively controllable mechanism.

**Core Idea**: Reverse the MtC pipeline—first compress and project each LoRA into a learned shared $r$-dimensional subspace, then execute merging rules in the low-dimensional coordinate space. The output naturally satisfies the rank-$r$ constraint.

## Method

### Overall Architecture
Given $T$ isomorphic LoRA adapters $\{(A^{(t)}, B^{(t)})\}_{t=1}^{T}$ (same base model, same injection layer, same input rank $r_{\text{in}}$), CtM reverses the traditional MtC "merge-then-compress" into "compress-then-merge": it first learns a pair of shared $r$-dimensional orthogonal bases $(U, V)$, projects each adapter into a compact $r \times r$ coordinate matrix, applies standard merging rules in this low-dimensional coordinate space, and finally lifts it back to the original parameter space. Since the subspace is fixed before merging, the output naturally resides in the rank-$r$ subspace, eliminating the need for post-hoc truncated SVD.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Input: T isomorphic LoRA adapters<br/>Task updates ΔW⁽ᵗ⁾ = B⁽ᵗ⁾A⁽ᵗ⁾"] --> S1
    subgraph S1["Rescaling-aware Shared Subspace Learning (Design 1)"]
        direction TB
        B["Rescaling Proxy Target ΔW_target<br/>Normalize by norm to eliminate bias, then re-inject scale with λ"]
        C["Tucker-2 Decomposition for Shared Orthogonal Bases (U, V)<br/>HOSVD Initialization + HOOI Iteration"]
        D["Lossless Core-Space Acceleration (Design 3)<br/>Thin SVD projection to (Tr)×(Tr) core space for solving"]
        B --> C
        D -.Accelerate Tucker Solving.-> C
    end
    S1 -->|Output Shared Bases (U, V)| S2
    subgraph S2["Low-dimensional Coordinate Space Merging (Design 2)"]
        direction TB
        E["Project Original LoRAs to Coordinates<br/>O⁽ᵗ⁾ = Uᵀ ΔW⁽ᵗ⁾ V (r×r)"]
        F["Apply Existing Merging Rules in r×r Space<br/>TIES / DARE, etc."]
        G["Lift back to Parameter Space<br/>ΔW_LoRA = U · O_merged · Vᵀ"]
        E --> F --> G
    end
    S2 --> H["Output: Single rank-r LoRA<br/>Rank constraint guaranteed by construction, no truncated SVD"]
```

### Key Designs

**1. Rescaling-aware Shared Subspace Learning: Selecting a subspace that does not favor large-norm tasks**

The crux of CtM lies in the orthogonal bases $U \in \mathbb{R}^{n \times r}$ and $V \in \mathbb{R}^{m \times r}$. Once determined, all orthogonal components are irrecoverable, necessitating a balanced reconstruction of all task updates. The difficulty lies in the vast norm discrepancies between LoRAs: learning a subspace directly on the original $\Delta W^{(t)}$ would be dominated by large-norm tasks. However, complete normalization loses scale information. The solution is to construct a rescaling proxy target $\Delta W_{\text{target}}^{(t)} = \lambda^{(t)} \cdot \Delta W^{(t)} / \|\Delta W^{(t)}\|_F$. This first eliminates norm bias through normalization and then softly re-injects the scale signal using $\lambda^{(t)} = \beta \|\Delta W^{(t)}\|_F + (1-\beta) \|\Delta W\|_{F,\text{Avg}}$, where $\beta \in [0,1]$ acts as a knob between "complete normalization" and "retaining original scale". By stacking these proxy targets into a third-order tensor $\mathcal{X} \in \mathbb{R}^{n \times m \times T}$, subspace learning becomes a standard Tucker-2 decomposition problem, solved via HOSVD initialization and HOOI iteration. The resulting $(U, V)$ reflect common structures across tasks rather than the specifics of a single task.

**2. Low-dimensional Coordinate Space Merging: Running merging rules in a compact $r \times r$ space**

Once the subspace is defined, merging becomes straightforward. Notably, **original** (non-rescaled) LoRAs are used here: coordinates are calculated for each task as $O^{(t)} = U^\top \Delta W^{(t)} V$, projecting the true updates into the shared bases. Any existing rules (TIES, DARE, etc.) are then applied to merge these coordinates $O_{\text{merged}} = \mathcal{M}(\{O^{(t)}\})$. Finally, the result is lifted back to the parameter space $\Delta W_{\text{LoRA}} = U \cdot O_{\text{merged}} \cdot V^\top$. Output rank is guaranteed by construction to be $\leq r$, which is the key to bypassing truncated SVD. An additional benefit is that cross-task conflicts are more concentrated in the compact $r \times r$ space, making sign conflict resolution methods like TIES more effective.

**3. Lossless Core-Space Acceleration: Moving tensor decomposition to a small core space**

Directly performing HOSVD/HOOI in the full parameter space is prohibitively expensive for large models like LLaMA3-8B, roughly $\mathcal{O}(n^3 T)$. The authors observed that all LoRA updates exist within a low-dimensional core: by performing thin SVD on concatenated factors $B_{\text{cat}}$ and $A_{\text{cat}}$ to obtain $U_B$ and $V_A$, each LoRA is projected to a core representation $M^{(t)} = U_B^\top \Delta W^{(t)} V_A$. Solving the Tucker problem in the $(Tr_{\text{in}}) \times (Tr_{\text{in}})$ core space and lifting back via $U = U_B U_{\text{core}}$ allows for massive acceleration. The paper proves this projection is lossless for all LoRA updates (Theorem 4.1) and that the optimal solution in the core space is equivalent to the original space (Theorem 4.3). This achieves approximately $n^2 / (Tr)^2$ speedup, reducing subspace learning time on LLaMA3-8B from 691s to 21s.

## Key Experimental Results

### Main Results—Vision Tasks (CLIP ViT-B/32, 8 Datasets)

| Merge Rule | Method | Cars | DTD | EuroSAT | GTSRB | MNIST | RESISC | SUN397 | SVHN | Avg |
|---------|------|------|-----|---------|-------|-------|--------|--------|------|-----|
| TIES | CoreSpace (MtC) | 83.54 | 73.81 | 53.09 | 39.97 | 65.18 | 68.05 | 95.25 | 43.73 | 65.33 |
| TIES | **CtM (Ours)** | 82.87 | 75.27 | **64.46** | 38.50 | **78.22** | 70.67 | 97.07 | **50.93** | **69.75** |
| DARE-TIES | CoreSpace (MtC) | 84.01 | 73.45 | 53.95 | 40.90 | 64.55 | 69.13 | 96.17 | 45.10 | 65.91 |
| DARE-TIES | **CtM (Ours)** | 83.04 | 75.00 | **64.95** | 38.28 | **79.98** | 69.90 | 96.53 | **57.73** | **70.68** |

### Main Results—Language Tasks (LLaMA3-8B, 6 NLI Datasets)

| Merge Rule | Method | SNLI | MNLI | SICK | QNLI | RTE | SCITAIL | Avg |
|---------|------|------|------|------|------|-----|---------|-----|
| TIES | CoreSpace (MtC) | 91.65 | 93.15 | 93.46 | 83.46 | 99.19 | 97.71 | 93.10 |
| TIES | **CtM (Ours)** | 91.08 | 90.26 | **96.11** | **89.71** | 100.81 | 96.93 | **94.15** |
| DARE-TIES | CoreSpace (MtC) | 91.90 | 91.82 | 95.39 | 80.79 | 100.00 | 97.37 | 92.88 |
| DARE-TIES | **CtM (Ours)** | 92.96 | **94.12** | **97.44** | **94.24** | 97.58 | **97.42** | **95.63** |

### Ablation Study

| Configuration | TIES Avg | DARE-TIES Avg | Description |
|------|---------|--------------|------|
| Best MtC baseline | 65.33 | 65.91 | CoreSpace + Truncated SVD |
| CtM + SVD Subspace | 67.42 | 69.04 | Simplified SVD instead of learned subspace |
| CtM + Learned Subspace | **69.75** | **70.68** | Full method |
| $\beta=1$ (No Rescaling) | ~67 | ~68 | Consistent performance drop |

### Key Findings
- **Truncation loss is non-negligible**: MtC experiences a drop from Avg$_\text{full}$ = 75.87 to 65.71 after truncation on vision tasks (over 10 points loss); CtM avoids this by ensuring rank constraints by construction.
- **Subspace balance advantage**: The subspace learned by CtM leads MtC significantly in energy retention (mean 87.94 vs 69.70) and functional preservation (mean 96.18 vs 86.54) with extremely low variance (std 2.71 vs 21.52), indicating CtM distributes the rank budget more evenly across tasks.
- **Core-Space Acceleration**: On LLaMA3-8B, time was reduced from 691s to 21s ($33\times$ speedup) with identical precision (Chordal distance only 0.031).

## Highlights & Insights
- **Inverted pipeline design philosophy**: Cleverly elevates "subspace selection" from a passive byproduct to an active design object. This mindset is valuable for scenarios with constrained outputs—whenever "manipulate then constrain" is problematic, "constrain then manipulate" is worth considering.
- **Rescaling mechanism**: Uses $\beta$ interpolation to balance complete normalization and original scale preservation. This is simple, effective, and insensitive to hyperparameters ($\beta \in [0, 0.75]$ performs stably), providing a practical tip for handling heterogeneous scales.
- **Tucker decomposition perspective**: Framing subspace learning as a standard tensor decomposition problem provides both mature algorithms (HOSVD + HOOI) and compatibility with the Core-Space framework for lossless acceleration, making it elegant in both theory and engineering.

## Limitations & Future Work
- When LoRA updates from multiple tasks are highly orthogonal, any fixed-rank method inevitably loses information, and CtM cannot bypass this fundamental bottleneck.
- Hyperparameters (target rank $r$, rescaling coefficient $\beta$, base merging rules) still require tuning; developing adaptive strategies is a future direction.
- Current focus is on isomorphic LoRAs (same rank and injection modules); while extensions to heterogeneous scenarios are discussed, verification is insufficient.
- Custom merging rules designed for the CtM coordinate space (rather than directly reusing TIES/DARE) could be explored to further exploit low-dimensional advantages.

## Related Work & Insights
- **Model Merging**: Task Arithmetic, TIES (sign conflict resolution), DARE (sparsification), Iso-C (full-parameter subspace alignment)—CtM is orthogonal to these and acts as an enhancement framework for "base" merging rules.
- **LoRA Merging**: KnOTS and CoreSpace build lossless bases for coordinate alignment but still rely on truncated SVD; LoRA-LEGO and RobustMerge output low-rank directly but do not explicitly learn a shared subspace.
- **Insights**: The "constrain then manipulate" paradigm of CtM can be transferred to other merging scenarios requiring structured output, such as merging different PEFT methods (Adapter, Prefix) or multimodal adapters.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs](../../ACL2026/model_compression/rethinking_parameter_sharing_for_llm_fine-tuning_with_multiple_loras.md)
- [\[ICML 2026\] Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs](preserve-then-quantize_balancing_rank_budgets_for_quantization_error_reconstruct.md)
- [\[ICML 2026\] Finer Parameter Steps for Low-Rank PEFT: A Controlled Study with CP Tensor Adapters](finer_parameter_steps_for_low-rank_peft_a_controlled_study_with_cp_tensor_adapte.md)
- [\[ICML 2026\] ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](scalora_optimally_scaled_low-rank_adaptation_for_efficient_high-rank_fine-tuning.md)
- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](energy-structured_low-rank_adaptation_for_continual_learning.md)

</div>

<!-- RELATED:END -->
