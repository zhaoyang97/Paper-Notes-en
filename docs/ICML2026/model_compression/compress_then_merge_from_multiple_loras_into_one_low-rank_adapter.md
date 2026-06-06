---
title: >-
  [Paper Note] Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter
description: >-
  [ICML2026][Model Compression][LoRA merging] The Compress-then-Merge (CtM) pipeline is proposed, which learns a shared $r$-dimensional subspace and projects each adapter into $r \times r$ coordinate matrices before mergin…
tags:
  - "ICML2026"
  - "Model Compression"
  - "LoRA merging"
  - "low-rank constraint"
  - "shared subspace"
  - "Tucker decomposition"
  - "parameter-efficient fine-tuning"
date: 2026-05-08
content_hash: 2c667b8c4cb478d0
---

# Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter

**Conference**: ICML2026  
**arXiv**: [2606.03723](https://arxiv.org/abs/2606.03723)  
**Code**: https://github.com/ZhengbaoHe/compress-then-merge  
**Area**: Model Compression  
**Keywords**: LoRA merging, low-rank constraint, shared subspace, Tucker decomposition, parameter-efficient fine-tuning  

## TL;DR
The Compress-then-Merge (CtM) pipeline is proposed, which learns a shared $r$-dimensional subspace and projects each adapter into $r \times r$ coordinate matrices before merging. By performing merging within this low-dimensional space, the method guarantees a rank-$r$ LoRA output by construction, avoiding the performance degradation caused by truncated SVD in traditional Merge-then-Compress approaches.

## Background & Motivation

**Background**: LoRA has become the standard choice for parameter-efficient fine-tuning (PEFT) of large models. Platforms like HuggingFace have accumulated a vast number of LoRA adapters for different tasks. Merging multiple LoRAs into a single multi-task adapter is an increasingly important practical requirement.

**Limitations of Prior Work**: The current mainstream solution is Merge-then-Compress (MtC)—first merging multiple LoRAs in the full parameter space to obtain a high-rank update $\Delta W_{\text{merged}}$, then compressing it to the target rank $r$ via truncated SVD. This "merge-then-compress" strategy treats the rank constraint as an afterthought, leading to two issues: (1) Large differences in norm across different task LoRAs cause truncated SVD (which is Frobenius-norm optimal) to bias toward high-norm tasks, systematically suppressing low-norm tasks. (2) LoRA training may produce "intrusive dimensions"—directions with high energy but weak correlation to task accuracy—which occupy the limited rank budget.

**Key Challenge**: The merged rank-$r$ result must reside in some $r$-dimensional subspace. MtC treats subspace selection as a byproduct of the merged result's spectrum rather than an active design choice. Once the subspace is determined, all components orthogonal to it are unrecoverable; thus, subspace selection should be a core design consideration.

**Key Insight**: Recent research indicates that weight changes in multi-task models tend to concentrate in a low-dimensional spectral subspace. The authors propose the **Compress-then-Merge** strategy: learn a shared subspace before merging, transforming subspace selection from a passive byproduct into an active, controllable mechanism.

**Core Idea**: Invert the MtC pipeline—first compress and project each LoRA into a learned shared $r$-dimensional subspace, then execute merging rules in the low-dimensional coordinate space. The output naturally satisfies the rank-$r$ constraint.

## Method

### Overall Architecture
Given $T$ isomorphic LoRA adapters $\{(A^{(t)}, B^{(t)})\}_{t=1}^{T}$ (same base model, same injection layers, same input rank $r_{\text{in}}$), the goal is to merge them layer-wise into a rank-$r$ LoRA update. CtM consists of three steps: (1) Learning shared orthogonal bases $(U, V)$; (2) Projecting each adapter into $r \times r$ coordinates; (3) Executing standard merging rules in the coordinate space and lifting back to the original parameter space.

### Key Designs

1.  **Rescaling-aware Shared Subspace Learning**:
    - **Function**: Learns a pair of $r$-dimensional orthogonal bases $U \in \mathbb{R}^{n \times r}$ and $V \in \mathbb{R}^{m \times r}$ that can reconstruct all task LoRA updates in a balanced manner.
    - **Mechanism**: Constructs a rescaled proxy target $\Delta W_{\text{target}}^{(t)} = \lambda^{(t)} \cdot \Delta W^{(t)} / \|\Delta W^{(t)}\|_F$, where $\lambda^{(t)} = \beta \|\Delta W^{(t)}\|_F + (1-\beta) \|\Delta W\|_{F,\text{Avg}}$. Each LoRA is first normalized to eliminate norm bias, then scale signals are softly reintroduced via $\beta \in [0,1]$. These proxy targets are stacked into a third-order tensor $\mathcal{X} \in \mathbb{R}^{n \times m \times T}$, transformed into a Tucker-2 decomposition problem, and solved via HOSVD initialization plus HOOI iteration.
    - **Design Motivation**: Directly learning a subspace on original LoRAs would be dominated by high-norm tasks; complete normalization loses scale information. $\beta$-interpolation provides a knob to balance both, ensuring the subspace reflects common structures across tasks.

2.  **Low-dimensional Coordinate Space Merging**:
    - **Function**: Executes any standard merging rule within compact $r \times r$ coordinates.
    - **Mechanism**: Original (non-rescaled) LoRAs are used to compute coordinates $O^{(t)} = U^\top \Delta W^{(t)} V$. Standard rules (e.g., TIES, DARE) are then applied to obtain $O_{\text{merged}} = \mathcal{M}(\{O^{(t)}\})$. Finally, this is lifted back to $\Delta W_{\text{LoRA}} = U \cdot O_{\text{merged}} \cdot V^\top$. Output rank is guaranteed to be $\leq r$ by construction.
    - **Design Motivation**: Proxy targets are only for learning the subspace basis; coordinate calculation and merging use the actual LoRA updates to preserve true task contributions. Merging in a compact space concentrates cross-task conflicts, making them easier to resolve with methods like TIES.

3.  **Core-Space Lossless Acceleration**:
    - **Function**: Compresses the computation of subspace learning from $n \times m$ space to a $(Tr_{\text{in}}) \times (Tr_{\text{in}})$ core space.
    - **Mechanism**: Thin SVD is performed on concatenated LoRA factors $B_{\text{cat}}$ and $A_{\text{cat}}$ to obtain $U_B$ and $V_A$. Each LoRA is projected to a core representation $M^{(t)} = U_B^\top \Delta W^{(t)} V_A$. After solving the Tucker problem in the core space, the result is lifted back via $U = U_B U_{\text{core}}$. The paper proves this projection is lossless for all LoRA updates (Theorem 4.1) and that the optimal solution is equivalent (Theorem 4.3).
    - **Design Motivation**: For large models like LLaMA3-8B, direct HOSVD/HOOI in the full parameter space costs $\mathcal{O}(n^3 T)$. Core-Space reduction lowers the dimension from $n$ to $Tr$, yielding a speedup of approximately $n^2 / (Tr)^2$, reducing time from 691s to 21s in practice.

## Key Experimental Results

### Main Results — Vision Tasks (CLIP ViT-B/32, 8 Datasets)

| Merge Rule | Method | Cars | DTD | EuroSAT | GTSRB | MNIST | RESISC | SUN397 | SVHN | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TIES | CoreSpace (MtC) | 83.54 | 73.81 | 53.09 | 39.97 | 65.18 | 68.05 | 95.25 | 43.73 | 65.33 |
| TIES | **CtM (Ours)** | 82.87 | 75.27 | **64.46** | 38.50 | **78.22** | 70.67 | 97.07 | **50.93** | **69.75** |
| DARE-TIES | CoreSpace (MtC) | 84.01 | 73.45 | 53.95 | 40.90 | 64.55 | 69.13 | 96.17 | 45.10 | 65.91 |
| DARE-TIES | **CtM (Ours)** | 83.04 | 75.00 | **64.95** | 38.28 | **79.98** | 69.90 | 96.53 | **57.73** | **70.68** |

### Main Results — Language Tasks (LLaMA3-8B, 6 NLI Datasets)

| Merge Rule | Method | SNLI | MNLI | SICK | QNLI | RTE | SCITAIL | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TIES | CoreSpace (MtC) | 91.65 | 93.15 | 93.46 | 83.46 | 99.19 | 97.71 | 93.10 |
| TIES | **CtM (Ours)** | 91.08 | 90.26 | **96.11** | **89.71** | 100.81 | 96.93 | **94.15** |
| DARE-TIES | CoreSpace (MtC) | 91.90 | 91.82 | 95.39 | 80.79 | 100.00 | 97.37 | 92.88 |
| DARE-TIES | **CtM (Ours)** | 92.96 | **94.12** | **97.44** | **94.24** | 97.58 | **97.42** | **95.63** |

### Ablation Study

| Configuration | TIES Avg | DARE-TIES Avg | Note |
| :--- | :--- | :--- | :--- |
| Best MtC baseline | 65.33 | 65.91 | CoreSpace + Truncated SVD |
| CtM + SVD Subspace | 67.42 | 69.04 | Replacing learned subspace with simple SVD |
| CtM + Learned Subspace | **69.75** | **70.68** | Complete method |
| $\beta=1$ (No rescaling) | ~67 | ~68 | Consistent drop in performance |

### Key Findings
- **Non-negligible Truncation Loss**: In vision tasks, MtC before truncation achieved Avg$_\text{full}$ = 75.87 (Iso-C CoreSpace), but dropped sharply to 65.71 after truncation—a loss of over 10 points. CtM avoids this loss by guaranteeing rank constraints by construction.
- **Subspace Balance Advantage**: The subspace learned by CtM significantly leads MtC in energy preservation (mean 87.94 vs 69.70) and functional preservation (mean 96.18 vs 86.54), with extremely low variance (std 2.71 vs 21.52). This indicates CtM distributes the rank budget more evenly across tasks.
- **Core-Space Acceleration**: Execution time on LLaMA3-8B dropped from 691s to 21s (33× speedup) with identical precision; the Chordal distance was only 0.031.

## Highlights & Insights
- **Inverted Pipeline Philosophy**: Cleverly upgrades "subspace selection" from a passive byproduct to an active design object. This mode of thinking is valuable in scenarios with constrained outputs—whenever "manipulate then constrain" might be inferior to "constrain then manipulate."
- **Rescaling Mechanism**: Balancing between full normalization and original scale via $\beta$-interpolation is simple and effective, showing robustness to hyperparameters ($\beta \in [0, 0.75]$ performs stably). It is a practical trick for handling heterogeneous scales.
- **Tucker Decomposition Perspective**: Transforming subspace learning into a standard tensor decomposition problem allows the use of mature algorithms (HOSVD + HOOI) and integrates with the Core-Space framework for lossless acceleration, making it elegant both theoretically and computationally.

## Limitations & Future Work
- When LoRA updates for multiple tasks are highly orthogonal, any fixed-rank method inevitably loses information; CtM cannot bypass this fundamental bottleneck.
- Hyperparameters (target rank $r$, rescaling coefficient $\beta$, base merging rules) still require tuning; developing adaptive strategies is a future direction.
- Currently focused on isomorphic LoRAs (same rank and injection modules); while extensions to heterogeneous scenarios are discussed, they lack sufficient verification.
- Tailored merging rules for the CtM coordinate space (instead of directly reusing TIES/DARE) could be explored to further exploit low-dimensional advantages.

## Related Work & Insights
- **Model Merging**: Task Arithmetic, TIES (sign conflict resolution), DARE (sparsification), Iso-C (full-parameter subspace alignment)—CtM is orthogonal to these methods and can serve as an enhancement framework for "base" merging rules.
- **LoRA Merging**: KnOTS and CoreSpace build lossless bases for coordinate alignment but still rely on truncated SVD. LoRA-LEGO and RobustMerge directly output low-rank results but do not explicitly learn a shared subspace.
- **Insights**: The "constrain then manipulate" paradigm of CtM can be transferred to other merging scenarios requiring structured outputs, such as merging different PEFT methods (Adapters, Prefixes) or multi-modal adapters.

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
