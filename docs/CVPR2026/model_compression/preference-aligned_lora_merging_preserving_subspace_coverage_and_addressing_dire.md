---
title: >-
  [Paper Note] Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy
description: >-
  [CVPR 2026][Model Compression][LoRA merging] This paper revisits the LoRA merging problem through two complementary lenses—subspace coverage and directional anisotropy—and proposes the TARA-Merging framework. By retainin…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "LoRA merging"
  - "subspace coverage"
  - "anisotropy"
  - "multi-objective optimization"
  - "model merging"
date: 2026-05-08
content_hash: 560a3032152f257d
---

# Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy

**Conference**: CVPR 2026
**arXiv**: [2603.26299](https://arxiv.org/abs/2603.26299)  
**Code**: [https://github.com/wooseong97/TARA-Merge](https://github.com/wooseong97/TARA-Merge)  
**Area**: Model Compression / Model Merging
**Keywords**: LoRA merging, subspace coverage, anisotropy, multi-objective optimization, model merging

## TL;DR

This paper revisits the LoRA merging problem through two complementary lenses—subspace coverage and directional anisotropy—and proposes the TARA-Merging framework. By retaining LoRA directions to preserve subspace coverage and applying preference-weighted cross-entropy pseudo-loss for direction-level reweighting, TARA consistently outperforms existing merging methods across 8 vision and 6 NLI benchmarks.

## Background & Motivation

1. **Background**: LoRA has become the standard paradigm for fine-tuning large models. Merging multiple task-specific LoRA adapters into a single unified model (model merging) offers an effective alternative to expensive multi-task joint training.
2. **Limitations of Prior Work**: Existing merging methods suffer from two categories of issues: (1) general-purpose methods (Task Arithmetic, TIES, DARE) disregard the low-rank structure of LoRA and operate directly in the full parameter space, leading to severe cross-task interference; (2) LoRA-aware methods (KnOTS, LoRA-LEGO) exploit the LoRA structure but typically address only one of the two problems—coverage or anisotropy.
3. **Key Challenge**: The update directions of LoRA adapters span different subspaces and contribute unevenly. Naive merging attenuates directions most critical to certain task losses while over-emphasizing relatively unimportant ones.
4. **Goal**: (a) Subspace coverage—whether the diversity of per-task LoRA directions is preserved after merging; (b) Directional anisotropy—different LoRA directions exhibit unequal sensitivity to task losses, requiring fine-grained direction-level control.
5. **Key Insight**: Effective rank (erank) analysis reveals that LoRA-aware rank-1 stacking retains approximately 70% of the per-task independent dimensions, whereas interpolation-based merging (e.g., Task Arithmetic) causes severe subspace collapse. Directional sensitivity analysis via Jacobians further shows that sensitivity distributions are highly inconsistent across different preferences.
6. **Core Idea**: While preserving LoRA rank-1 directions to maintain subspace coverage, TARA addresses anisotropy by optimizing direction-level weights through preference-weighted smooth Tchebycheff scalarization.

## Method

### Overall Architecture

The TARA framework takes as input $N$ task-specific LoRA adapters $\{\Delta W_i = B_i A_i^\top\}$ and a user-specified task preference vector $\boldsymbol{\rho}$. Each LoRA is decomposed into rank-1 directions ($\mathbf{b}_{ij}\mathbf{a}_{ij}^\top$), and a learnable scalar weight $\phi_{ij}$ is assigned to each direction. These weights are optimized by minimizing a preference-weighted entropy pseudo-loss. The final merged model weight is $W = W_0 + \sum_i\sum_j \phi_{ij} \mathbf{b}_{ij}\mathbf{a}_{ij}^\top$.

### Key Designs

1. **Subspace Coverage Analysis and Preservation**:

    - **Function**: Quantify and preserve representational capacity in LoRA merging.
    - **Mechanism**: An entropy-based effective rank (erank) is used to measure the effective dimensionality of LoRA directions. The rank-1 components of per-task LoRAs are vectorized and stacked, and the erank of three stacking strategies is compared: (1) per-task independent summation, (2) LoRA-agnostic $\Delta W$ stacking, and (3) LoRA-aware rank-1 stacking. Rank-1 stacking retains approximately 70% of the independent dimensions, whereas $\Delta W$ stacking suffers from severe collapse due to interpolation interference. TARA maintains subspace coverage by treating rank-1 directions as the fundamental optimization units.
    - **Design Motivation**: Merging directly in the full parameter space destroys the internal low-rank structure of LoRA, resulting in a loss of representational capacity.

2. **Directional Anisotropy Alignment**:

    - **Function**: Address the unequal sensitivity of different LoRA directions to task losses.
    - **Mechanism**: A task-loss Jacobian is constructed as $J_{i,k} = \langle \nabla f_i(W), S_k \rangle_F$, where $S_k$ denotes the rank-1 LoRA direction. The condition number $\kappa(J)$ reflects the degree of anisotropy. A directional sensitivity misalignment metric is further defined as $\xi(\boldsymbol{\rho}_1, \boldsymbol{\rho}_2) = 1 - |\cos(\mathbf{h}(\boldsymbol{\rho}_1), \mathbf{h}(\boldsymbol{\rho}_2))|$, where $h_k(\boldsymbol{\rho}) = \langle g(\boldsymbol{\rho}; W), S_k \rangle_F$ denotes the direction-level sensitivity. Experiments show that sensitivity distributions are highly inconsistent across different preferences (large $\xi$), validating the necessity of direction-level reweighting.
    - **Design Motivation**: Equal-norm LoRA direction updates do not produce proportional changes in task loss; sensitivity differences must be accounted for at the direction level.

3. **Two Merging Variants (Variant A/B) and Optimization**:

    - **Function**: Provide different efficiency–accuracy trade-offs.
    - **Mechanism**: Variant A directly assigns weights to the rank-1 factors of each task: $W_A = W_0 + \sum_i\sum_j \phi_{ij} \mathbf{b}_{ij}\mathbf{a}_{ij}^\top$, offering simplicity and efficiency. Variant B horizontally concatenates all adapters and applies SVD to obtain a shared orthonormal basis $\{u_k\}$, then assigns weights to each task's projections onto the shared basis: $W_B = W_0 + \sum_i\sum_k \phi_{ik}\sigma_k u_k v_{ki}^\top$. Optimization employs smooth Tchebycheff scalarization $\Psi(\phi, \boldsymbol{\rho}) = \alpha \log(\sum_i \exp(\rho_i |f_i - z_i| / \alpha))$, using predictive entropy as a proxy for the true loss to avoid label dependency.
    - **Design Motivation**: Variant A avoids SVD computation, making it suitable for resource-constrained settings; Variant B better decorrelates and reduces interference through the shared orthonormal basis, yielding higher accuracy.

### Loss & Training

An AdaMerging-style predictive entropy is used as an unsupervised proxy loss, combined with smooth Tchebycheff scalarization and user preferences to optimize direction-level weights. The AdamW optimizer is used with a learning rate of 0.001, weight initialization of 0.4, 500 iterations, and a batch size of 16. The anchor $z_i$ is set to the entropy loss obtained when using only the single adapter for task $i$.

## Key Experimental Results

### Main Results

| Method | Cars | DTD | EuroSAT | GTSRB | MNIST | RESISC45 | SUN397 | SVHN | Avg (normalized%) |
|--------|------|-----|---------|-------|-------|----------|--------|------|-------------------|
| TA | 82.1 | 74.3 | 48.7 | 41.8 | 53.4 | 71.5 | 96.6 | 42.0 | 63.8 |
| TIES | 81.0 | 72.5 | 53.8 | 37.4 | 69.0 | 65.3 | 94.8 | 45.3 | 64.9 |
| AdaMerging | 79.5 | 73.5 | 70.9 | 39.7 | 63.0 | 69.0 | 97.8 | 66.6 | 70.0 |
| KnOTS-TIES | 82.7 | 73.7 | 49.3 | 48.9 | 68.9 | 70.9 | 95.5 | 53.8 | 68.0 |
| LoRA-LEGO | 81.1 | 73.0 | 54.4 | 40.3 | 48.6 | 71.5 | 97.3 | 37.1 | 62.9 |
| **TARA-A** | 82.2 | 76.0 | 74.9 | 43.5 | 76.3 | 70.2 | 98.0 | 70.8 | **74.0** |
| **TARA-B** | 86.2 | 78.4 | 76.8 | 42.9 | 82.7 | 75.4 | 98.6 | 69.7 | **76.3** |

### Ablation Study (NLI Tasks, LLaMA-3 8B)

| Method | MNLI | QNLI | SNLI | RTE | SICK | SCITAIL | Avg (normalized%) |
|--------|------|------|------|-----|------|---------|-------------------|
| TA | 67.3 | 87.3 | 41.8 | 95.7 | 77.9 | 76.9 | 74.6 |
| AdaMerging | 47.5 | 92.9 | 41.3 | 102.6 | 93.8 | 94.2 | 78.7 |
| KnOTS-TIES | 41.1 | 83.4 | 56.6 | 87.2 | 87.9 | 94.8 | 75.2 |
| **TARA-A** | 51.7 | 92.6 | 41.4 | 102.6 | 95.3 | 94.4 | **79.7** |
| **TARA-B** | 46.8 | 94.1 | 41.4 | 103.4 | 98.1 | 97.8 | **80.3** |

### Key Findings

- **TARA-B achieves the best results on both vision and NLI tasks**: 76.3% average on 8 vision tasks (vs. AdaMerging 70.0%) and 80.3% average on 6 NLI tasks (vs. AdaMerging 78.7%), demonstrating the importance of jointly addressing coverage and anisotropy.
- **LoRA-LEGO underperforms vanilla baselines**: Retaining rank-1 directions without sensitivity weighting is insufficient (62.9% vs. TA's 63.8%), confirming that both problems must be addressed simultaneously.
- **Generalization to unseen tasks**: After merging on 6 known tasks, TARA-B achieves 52.2% average accuracy on 2 unseen tasks, substantially outperforming TA (42.9%) and KnOTS-TIES (41.8%).
- **Joint task evaluation**: TARA-B achieves Hits@1 of 49.3% (vs. TA 43.5%, AdaMerging 48.1%).

## Highlights & Insights

- **Unification of two orthogonal perspectives**: Decomposing the LoRA merging problem into the independent yet complementary dimensions of "coverage" and "anisotropy" yields a theoretically clear and elegant framework. Prior methods (KnOTS addressing coverage, AdaMerging addressing global weighting) each solve only half of the problem.
- **Discovery of directional sensitivity misalignment**: Jacobian analysis quantifies the inconsistency of LoRA direction sensitivity distributions across different preferences, providing a rigorous theoretical motivation for direction-level weighting.
- **Efficiency advantage of LoRA-level operations**: Compared to full-parameter merging, LoRA-level operations substantially reduce memory and computational requirements, making gradient-based merging methods scalable to foundation model sizes.

## Limitations & Future Work

- Variant B requires joint SVD over all adapters, which may incur significant computational overhead when the number of tasks $N$ is large or the LoRA rank is high.
- Experiments are primarily conducted on ViT-B/32 and LLaMA-3 8B; performance on larger models (e.g., 70B-scale) remains unknown.
- The preference vector $\boldsymbol{\rho}$ must be manually specified by the user; automatic preference discovery would be more practical.
- The framework does not address conflicting gradients across tasks (e.g., PCGrad-style gradient projection).

## Related Work & Insights

- **vs. AdaMerging**: AdaMerging learns layer-level weights but ignores sensitivity differences within LoRA directions. TARA operates at finer direction-level granularity. Both methods share the same entropy minimization proxy.
- **vs. KnOTS**: KnOTS aligns subspaces via SVD to address coverage but neglects anisotropic weighting. TARA-B builds upon the SVD basis and additionally introduces direction-level weight optimization.
- **vs. LoRA-LEGO**: LoRA-LEGO preserves modularity through clustering, but the clustering process may discard critical directional information. TARA retains all original directions.

## Rating

- Novelty: ⭐⭐⭐⭐ The identification of two analytical perspectives and the unified framework design are original, though the specific implementation is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The dual-track evaluation across vision and NLI, joint evaluation, generalization testing, and preference sensitivity analysis are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, though the dense notation imposes a non-trivial reading burden.
- Value: ⭐⭐⭐⭐ The work makes a substantive contribution to the LoRA merging field; the open-sourced code enables direct practical use.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evolutionary Negative Module Pruning for Better LoRA Merging](../../ACL2026/model_compression/evolutionary_negative_module_pruning_for_better_lora_merging.md)
- [\[ACL 2026\] LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging](../../ACL2026/model_compression/lora_on_the_go_instance-level_dynamic_lora_selection_and_merging.md)
- [\[ICML 2026\] FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](../../ICML2026/model_compression/frism_fine-grained_reasoning_injection_via_subspace-level_model_merging_for_visi.md)
- [\[ICLR 2026\] Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](../../ICLR2026/model_compression/null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)
- [\[ICML 2026\] Saliency-Aware Model Merging](../../ICML2026/model_compression/saliency-aware_model_merging.md)

</div>

<!-- RELATED:END -->
