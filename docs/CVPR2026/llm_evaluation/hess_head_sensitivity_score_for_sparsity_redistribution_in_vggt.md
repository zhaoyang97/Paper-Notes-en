---
title: >-
  [Paper Note] HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT
description: >-
  [CVPR 2026][LLM Evaluation][Attention Sparsification] HeSS proposes a Head Sensitivity Score to quantify the sensitivity of each attention head in VGGT's global attention layers to sparsification…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "Attention Sparsification"
  - "VGGT"
  - "Head Sensitivity"
  - "Fisher Information Matrix"
  - "3D Reconstruction Acceleration"
date: 2026-05-08
content_hash: 93cefbf47fa06837
---

# HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT

**Conference**: CVPR 2026
**arXiv**: [2603.25336](https://arxiv.org/abs/2603.25336)  
**Code**: [https://github.com/libary753/HeSS](https://github.com/libary753/HeSS)  
**Area**: LLM Evaluation
**Keywords**: Attention Sparsification, VGGT, Head Sensitivity, Fisher Information Matrix, 3D Reconstruction Acceleration

## TL;DR

HeSS proposes a Head Sensitivity Score to quantify the sensitivity of each attention head in VGGT's global attention layers to sparsification, and redistributes the attention budget from insensitive heads to sensitive ones accordingly. This approach significantly outperforms the uniform sparsification method SparseVGGT at high sparsity ratios with virtually no additional runtime overhead.

## Background & Motivation

1. **Background**: VGGT (Visual Geometry Grounded Transformer) is a powerful foundation model for multi-view 3D reconstruction that unifies traditional SfM and MVS tasks through alternating stacks of Global Attention (GA) layers and Frame Attention (FA) layers. GA layers enable all frame tokens to interact with one another and are critical for understanding global scene structure.

2. **Limitations of Prior Work**: The self-attention computation in GA layers scales quadratically with the number of input views, $O(S^2)$, causing GPU memory and compute costs to surge dramatically for large-scale inputs, which limits VGGT's applicability in real-time and large-scale scenarios.

3. **Key Challenge**: Existing acceleration methods (e.g., SparseVGGT) apply a uniform sparsification pattern—identical CDF threshold $\tau$ and sparsity ratio $\rho$—to all attention heads. In practice, however, different attention heads exhibit substantially different sensitivities to sparsification: some heads suffer dramatic performance degradation when over-sparsified, while others remain largely unaffected even under aggressive sparsification.

4. **Goal**: (1) How can the sensitivity of each attention head to sparsification be quantified? (2) How can the attention budget be adaptively allocated based on sensitivity differences? (3) How can budget redistribution significantly reduce performance degradation at high sparsity without changing the total computational cost?

5. **Key Insight**: The authors hypothesize that the root cause of performance degradation is the heterogeneity of head sensitivity—uniform sparsification inevitably over-compresses critical heads. They approximate the Hessian via the Fisher Information Matrix to measure head sensitivity, combining two task-specific error signals (camera pose and point cloud) unique to 3D reconstruction.

6. **Core Idea**: The Fisher Information Matrix is used as a Hessian approximation with respect to camera pose error and point cloud error to quantify the sparsification sensitivity of each attention head. The total attention budget is then redistributed proportionally to sensitivity scores, preserving more attention for sensitive heads while applying stronger sparsification to robust ones.

## Method

### Overall Architecture

HeSS is a two-stage pipeline: (1) **Calibration stage**: HeSS scores are computed for all GA-layer attention heads on a small calibration set and fixed thereafter. (2) **Inference stage**: The precomputed HeSS scores guide the reallocation of each head's attention budget (number of blocks), with more budget assigned to sensitive heads and less to robust ones. The total budget remains identical to SparseVGGT; only the inter-head allocation changes.

### Key Designs

1. **Camera Pose Error $e_{\text{cam}}$**:

    - **Function**: Serves as the first error signal for HeSS, evaluating the model's understanding of global scene geometry.
    - **Mechanism**: The Umeyama + ICP algorithm first aligns predicted and ground-truth point clouds to obtain a transformation matrix $\mathbf{H}$ (with stop-gradient applied to $\mathbf{H}$); the MSE between the transformed predicted camera positions $\hat{\mathbf{t}}_i$ and ground-truth positions $\mathbf{t}_i$ is then computed: $e_{\text{cam}} = \frac{1}{2N}\sum_{i=1}^N |\text{sg}(\mathbf{H})\hat{\mathbf{t}}_i - \mathbf{t}_i|_2^2$
    - **Design Motivation**: Camera pose is the geometric scaffold for all downstream predictions in 3D vision. Stop-gradient on $\mathbf{H}$ prevents gradients from flowing through the auxiliary alignment step.

2. **Point Cloud Error $e_{\text{pc}}$**:

    - **Function**: Complements $e_{\text{cam}}$ by evaluating fine-grained local geometry.
    - **Mechanism**: Inlier set $\mathcal{I}$ is selected using confidence threshold $\epsilon = 0.05$ (predicted points whose nearest distance to the ground-truth point cloud is $< \epsilon$); the average nearest-point distance over inliers is then computed: $e_{\text{pc}} = \frac{1}{2|\mathcal{I}|}\sum_{j \in \mathcal{I}} \min_{\mathbf{p} \in P}\|\text{sg}(\mathbf{H})\hat{\mathbf{p}}_j - \mathbf{p}\|_2^2$
    - **Design Motivation**: $e_{\text{cam}}$ reflects only global geometric consistency and is insensitive to per-pixel fine-grained structure. The point cloud error requires the model to precisely regress each pixel to its 3D position, capturing local geometric detail.

3. **HeSS Computation**:

    - **Function**: Merges the sensitivity from both error signals into a unified per-head score.
    - **Mechanism**: For each head $h$, the Fisher Information Matrices $\mathbf{F}_{\text{cam}}^h$ and $\mathbf{F}_{\text{pc}}^h$ are computed with respect to the Query projection weight $W_Q^h$; their traces are taken and normalized across all heads within the same layer: $\text{HeSS}_{\text{cam}}(h) = \frac{\text{tr}(\mathbf{F}_{\text{cam}}^h)}{\sum_h \text{tr}(\mathbf{F}_{\text{cam}}^h)}$. The final score is $\text{HeSS}(h) = \lambda \cdot \text{HeSS}_{\text{cam}}(h) + (1-\lambda) \cdot \text{HeSS}_{\text{pc}}(h)$, with default $\lambda = 0.5$.
    - **Design Motivation**: The FIM is a tractable approximation of the Hessian, estimating second-order information via the outer product of first-order gradients. $W_Q^h$ is chosen over $W_K^h$ or $W_V^h$ because experiments show that the Hessian of $W_Q$ yields more reliable sensitivity estimates. The two error signals are complementary—$e_{\text{cam}}$ is more important at low sparsity while $e_{\text{pc}}$ is more critical at high sparsity.

4. **HeSS-Guided Budget Redistribution**:

    - **Function**: Redistributes the total attention budget across heads according to HeSS scores.
    - **Mechanism**: A three-step procedure — (a) compute the total budget $C_{\text{total}} = \sum_n c_{h_n}$ (sum of baseline budgets across all heads); (b) compute the ideal budget proportional to HeSS: $c_h' = C_{\text{total}} \cdot w_h$, where $w_h = \text{HeSS}(h) / \sum_n \text{HeSS}(h_n)$; (c) apply an iterative water-filling algorithm to handle overflow — if any head's ideal budget exceeds its maximum capacity $C_{\max}$, it is clamped to $C_{\max}$ and the excess is redistributed proportionally (by HeSS weight) to remaining uncapped heads, repeating until no overflow remains.
    - **Design Motivation**: Naive proportional allocation may assign some highly sensitive heads more budget than their maximum available block count. Iterative capping ensures structural feasibility of the budget allocation.

### Loss & Training

- The calibration set uses the CO3Dv2 dev split with 20 views sampled per scene.
- HeSS scores are computed once and fixed; they do not change with inference data.
- No training is required — the method is purely an inference-time budget redistribution strategy.

## Key Experimental Results

### Main Results

Comparison on CO3Dv2 (camera pose estimation) and DTU (MVS reconstruction):

| Method | Sparsity | CO3Dv2 AUC@30↑ | DTU Chamfer↓ | Runtime (s) |
|--------|----------|----------------|-------------|-------------|
| VGGT (original) | 0% | baseline | baseline | 10.35 |
| SparseVGGT | 43% | notable drop | notable drop | 8.42 |
| **HeSS (Ours)** | **43%** | **near original** | **near original** | **8.37** |
| SparseVGGT | 73% | severe degradation | severe degradation | 6.59 |
| **HeSS (Ours)** | **73%** | **significantly better than SparseVGGT** | **significantly better than SparseVGGT** | **6.58** |

### Ablation Study

| Configuration | DTU Chamfer↓ | Acc.↓ | Comp.↓ | Notes |
|---------------|-------------|-------|--------|-------|
| Ours ($W_Q^h$) | **1.603** | **2.839** | **0.367** | Hessian w.r.t. Query projection |
| $W_K^h$ | 1.917 | 3.450 | 0.384 | Hessian w.r.t. Key projection, degraded |
| $W_V^h$ | 1.966 | 3.540 | 0.392 | Hessian w.r.t. Value projection, worse |
| No normalization (Linear) | 1.840 | 3.272 | 0.408 | Without sum-normalization |
| Inverted HeSS | catastrophic failure | — | — | Pruning sensitive heads first, sanity check |

### Key Findings

- HeSS distribution visualization (Figure 4) confirms high heterogeneity in head sensitivity — most heads exhibit low sensitivity on both error signals, while only a few are critically important (e.g., H5 in GA19).
- Certain heads exhibit clear task preferences — H13 in GA13 is more sensitive to camera pose, while H5 in GA19 is more sensitive to point cloud error.
- Ablation of $\lambda$ (Figure 11) demonstrates the complementarity of the two error signals: using $e_{\text{cam}}$ alone causes performance degradation at high sparsity; using $e_{\text{pc}}$ alone causes degradation at low sparsity; combining both maintains stable performance across all sparsity levels.
- Removing iterative capping leads to noticeable performance degradation (Figure 10), as unused budget is wasted.
- The method generalizes to the $\pi^3$ model but requires a different $\lambda$ ($\lambda = 0$ is optimal there).

## Highlights & Insights

- **Performance gains with zero additional inference overhead**: HeSS only changes the budget allocation scheme while keeping total computation constant (or even slightly reducing it through more efficient scheduling after redistribution), yet achieves significant performance gains at high sparsity. This constitutes a "free lunch" improvement.
- **Task-specific sensitivity metric for 3D reconstruction**: Unlike general ViT pruning that uses classification loss Hessians, HeSS defines two complementary error terms — camera pose and point cloud — tailored to 3D reconstruction. This domain-specific design yields more accurate sensitivity estimates.
- **Sanity check via inverted HeSS**: Reversing the ranking causes catastrophic performance collapse, providing a simple yet compelling validation that HeSS correctly identifies the critical heads.

## Limitations & Future Work

- The authors acknowledge that HeSS currently treats all GA layers uniformly, whereas different layers also differ in their sensitivity to sparsification, and FIM scales are not directly comparable across layers — a layer-comparable sensitivity metric is an important direction for future work.
- The method addresses only inference-time sparsification and does not consider training-time adaptation — training the model to be inherently robust to sparsification could further improve the compression ratio.
- The $\pi^3$ generalization experiment reveals that different models require different $\lambda$ values, indicating the need for an automatic search mechanism.
- The choice and size of the calibration set may affect the stability of HeSS scores, a point not thoroughly analyzed in the paper.

## Related Work & Insights

- **vs. SparseVGGT**: SparseVGGT introduces a block-sparse attention mechanism but applies it uniformly across all heads. HeSS builds upon it by adding head-level budget redistribution, achieving smarter allocation with the same total budget.
- **vs. Generic ViT Pruning (Michel et al.)**: General head pruning degrades sharply above 50% sparsity due to the absence of 3D geometric inductive bias; HeSS preserves the structural integrity of spatial tokens and maintains high fidelity even at 75% sparsity.
- **Transferability of HeSS's idea**: The paradigm of "using task-specific error Hessians to guide head-level resource allocation in multi-head attention" is generalizable to compression of other multi-head attention models, such as KV-cache compression in LLMs.

## Rating

- Novelty: ⭐⭐⭐⭐ The ideas of head sensitivity quantification and adaptive budget redistribution are not entirely novel, but their application to 3D reconstruction in VGGT is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-sparsity, extensive ablations, and generalization experiments are all present, with a convincing sanity check.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, mathematical derivations are detailed, and figures are intuitive.
- Value: ⭐⭐⭐⭐ Directly beneficial for practical deployment of VGGT, with transferable methodological ideas.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](../../ACL2026/llm_evaluation/contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[ICCV 2025\] Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating](../../ICCV2025/llm_evaluation/spectral_sensitivity_estimation_with_an_uncalibrated_diffraction_grating.md)
- [\[ACL 2026\] RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity](../../ACL2026/llm_evaluation/roleconflictbench_a_benchmark_of_role_conflict_scenarios_for_evaluating_llms39_c.md)
- [\[CVPR 2026\] Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline](pioneering_perceptual_video_fluency_assessment_a_novel_task_with_benchmark_datas.md)

</div>

<!-- RELATED:END -->
