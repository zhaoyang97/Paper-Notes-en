---
title: >-
  [Paper Note] Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging
description: >-
  [ACL 2025][Model Compression][LoRA merging] OSRM identifies that the failure of LoRA model merging stems from interaction interference between parameters and data distributions (rather than merely parameter conflicts). It proposes initializing the LoRA A matrix prior to fine-tuning via eigenvalue decomposition of the data covariance matrix, making its subspace orthogonal to the data distributions of other tasks. This minimizes cross-task interference during merging…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "LoRA merging"
  - "model merging"
  - "orthogonal subspace"
  - "parameter-data interaction"
  - "task interference"
date: 2026-05-08
content_hash: 9bef81cd05e42824
---

# Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging

**Conference**: ACL 2025  
**arXiv**: [2505.22934](https://arxiv.org/abs/2505.22934)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: LoRA merging, model merging, orthogonal subspace, parameter-data interaction, task interference

## TL;DR
OSRM identifies that the failure of LoRA model merging stems from interaction interference between parameters and data distributions (rather than merely parameter conflicts). It proposes initializing the LoRA A matrix prior to fine-tuning via eigenvalue decomposition of the data covariance matrix, making its subspace orthogonal to the data distributions of other tasks. This minimizes cross-task interference during merging, significantly improving merging performance across 8 datasets and 5 models.

## Background & Motivation
**Background**: Model merging (such as Task Arithmetic, TIES, etc.) can combine multiple task models into one without retraining, but performs poorly on LoRA fine-tuned models—resulting in severe performance degradation.

**Limitations of Prior Work**: Existing methods (such as KnOTS and parameter orthogonalization) only focus on downstream alignment or decoupling of parameters while ignoring how parameters interact with input data. After merging, data from Task 1 passing through Task 2's LoRA yields unexpected output shift $B_2 A_2 h_1$.

**Key Challenge**: Even if the task vectors of two LoRAs are orthogonal, $A_2 h_1$ can still be non-zero—parameter orthogonality does not equal functional orthogonality.

**Goal**: To ensure that the LoRA updates of task $i$ do not interfere with the data of task $j$.

**Key Insight**: Constraining the LoRA subspace before fine-tuning—ensuring the row space of $A_i$ is orthogonal to the principal components of the data covariance matrices of other tasks.

**Core Idea**: Utilizing the eigenvalue decomposition of other tasks' data covariance matrices to find orthogonal subspaces for initializing the LoRA A matrix, thereby mitigating merging interference at its root.

## Method

### Overall Architecture
Before fine-tuning, OSRM performs the following operations for each task: (1) collects data samples from other tasks and computes the covariance matrices of hidden features at each layer; (2) performs eigenvalue decomposition on the covariance matrices and selects the eigenvectors corresponding to the smallest eigenvalues as the "orthogonal subspace"; (3) initializes the LoRA A matrix with these eigenvectors (instead of random initialization). After fine-tuning, the models can be directly merged using existing methods such as Task Arithmetic or TIES.

### Key Designs

1. **Data-Parameter Interference Analysis**:

    - Key Finding: After merging, $W_m h_1 = W_1 h_1 + B_2 A_2 h_1$, where $B_2 A_2 h_1$ is the interference term. To minimize this interference, $\|A_2 h_1\|_F \approx 0$ is required.
    - Under the orthogonal basis assumption, $\|A_2 h_1\|_F$ can be minimized by making the row space of $A_2$ orthogonal to the principal directions of $h_1$.

2. **Orthogonal Subspace Initialization**:

    - Function: Initializes the LoRA A matrix using the eigenvectors associated with the smallest eigenvalues of the other tasks' data covariance matrices.
    - Mechanism: Computes the hidden feature covariance $\Sigma = \mathbb{E}[h h^\top]$ of all other tasks' data at the target layer, performs eigenvalue decomposition $\Sigma = U \Lambda U^\top$, and selects the eigenvectors corresponding to the smallest $r$ eigenvalues as the initial row vectors of $A$.
    - Design Motivation: The directions of the smallest eigenvalues represent the least active directions of other tasks' data—yielding the smallest projections on these directions, thereby maximizing the reduction of merging interference.

3. **Plug-and-play Compatibility**:

    - OSRM only modifies the initialization phase of LoRA, leaving the fine-tuning and merging processes unchanged.
    - It can be seamlessly combined with any merging method, such as Task Arithmetic, TIES, Fisher Merging, RegMean, or EMR.

## Key Experimental Results

### Main Results

**RoBERTa-base, 4-Task Merging (Task Arithmetic)**:

| Method | Multi-Task Average Performance | Single-Task Preservation |
|------|:----------:|:--------:|
| Task Arithmetic (Standard LoRA init) | ~65% | ~85% |
| + OSRM init | **~75%** | **~87%** |

OSRM improves multi-task performance by approximately 10%, while maintaining or even enhancing single-task performance.

### Ablation Study

| Configuration | Effect | Description |
|------|:----:|------|
| OSRM + TA | Optimal | Full method |
| OSRM + TIES | Significant Improvement | Also compatible with TIES |
| Random init (baseline) | Baseline | Standard LoRA |
| Parameter Orthogonalization Only | Small Gain | Ignores data interaction, limited effectiveness |

### Key Findings
- OSRM is more robust to merging hyperparameters (scaling coefficient $\lambda$)—whereas standard methods are highly sensitive to $\lambda$, OSRM remains stable across a wider range.
- Low sample requirements—requiring only dozens of samples per other task to compute the covariance matrices.
- Equally effective on large language models (LLaMA-7B).

## Highlights & Insights
- **Identified the true cause of merging failure**: Parameter-data interaction interference rather than parameter conflict—an insight that shifts the direction of merging research.
- **Addressing merging issues prior to fine-tuning**: Unlike post-processing methods (which optimize merging post-fine-tuning), OSRM eliminates the sources of interference before fine-tuning, which is more fundamental.
- **Extremely low overhead**: Requires only a one-time covariance computation and eigenvalue decomposition during initialization, with zero extra overhead during the fine-tuning process.

## Limitations & Future Work
- **Requires access to other tasks' data**: Needs prior knowledge of which tasks will be merged and access to a small amount of data, making it less suitable for merging completely unknown tasks.
- **Orthogonal basis assumption**: The analysis relies on the assumption that matrix $A$ is an orthogonal basis, which may deviate after actual fine-tuning.
- **Evaluated only on LoRA**: Parallel analysis of merging interference in full fine-tuning models remains unexplored.

## Related Work & Insights
- **vs KnOTS**: KnOTS aligns LoRA in a shared space in a data-independent manner; OSRM utilizes data-driven initialization, providing higher targeted effectiveness.
- **vs TIES/Task Arithmetic**: These are post-processing methods, whereas OSRM is a pre-processing method—making them orthogonal and stackable.
- Future work could explore replacing orthogonal initialization with regularization (continuously constraining orthogonality during the fine-tuning process).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The data-parameter interference analysis and pre-fine-tuning orthogonal initialization provide a completely fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across 8 datasets and 5 models, integrated with multiple merging methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Thorough problem analysis with rigorous mathematical derivations.
- Value: ⭐⭐⭐⭐ Provides a simple and highly effective plug-and-play solution for LoRA model merging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Task Singular Vectors: Reducing Task Interference in Model Merging](../../CVPR2025/model_compression/task_singular_vectors_reducing_task_interference_in_model_merging.md)
- [\[CVPR 2026\] DuetMerging: Synergizing Dynamic and Static Strategies for Mitigating Task Interference in Model Merging](../../CVPR2026/model_compression/duetmerging_synergizing_dynamic_and_static_strategies_for_mitigating_task_interf.md)
- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/model_compression/free-merging_fourier_transform_for_efficient_model_merging.md)
- [\[ICLR 2026\] MergOPT: A Merge-Aware Optimizer for Robust Model Merging](../../ICLR2026/model_compression/mergopt_a_merge-aware_optimizer_for_robust_model_merging.md)
- [\[ACL 2025\] Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](bone_soups_multi_objective_gen.md)

</div>

<!-- RELATED:END -->
