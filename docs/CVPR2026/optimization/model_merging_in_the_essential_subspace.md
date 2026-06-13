---
title: >-
  [Paper Note] Model Merging in the Essential Subspace
description: >-
  [CVPR 2026][Optimization][Model Merging] ESM constructs an "essential subspace" via PCA on activation shifts induced by parameter updates (rather than SVD on parameter matrices)…
tags:
  - "CVPR 2026"
  - "Optimization"
  - "Model Merging"
  - "PCA"
  - "Essential Subspace"
  - "Polarized Scaling"
  - "Low-Rank Decomposition"
date: 2026-05-08
content_hash: e48f91151b150c41
---

# Model Merging in the Essential Subspace

**Conference**: CVPR 2026  
**arXiv**: [2602.20208](https://arxiv.org/abs/2602.20208)  
**Code**: N/A  
**Area**: Pretraining  
**Keywords**: Model Merging, PCA, Essential Subspace, Polarized Scaling, Low-Rank Decomposition

## TL;DR
ESM constructs an "essential subspace" via PCA on activation shifts induced by parameter updates (rather than SVD on parameter matrices), and applies three-level polarized scaling to amplify critical parameters while suppressing noise. On 20-task ViT-B/32 merging, it improves over Iso-CTS by 3.2% absolute accuracy.

## Background & Motivation

**Background**: Model merging fuses multiple expert models fine-tuned from the same pretrained checkpoint into a single multi-task model without retraining. Recent SVD-based methods (TSV-M, Iso-CTS) achieve good results by truncating task matrices via SVD to reduce interference.

**Limitations of Prior Work**: SVD decomposition minimizes the Frobenius norm reconstruction error of parameter matrix $\Delta W$, but ignores the input feature distribution. The truncation error is $\sum_{i=k+1}^r \sigma_i^2 \cdot \mathbb{E}[(v_i^\top x)^2]$—even if $\sigma_i$ is small, truncation causes severe functional loss when inputs have large projections along $v_i$.

**Key Challenge**: The SVD subspace is misaligned with the task's feature space, causing low-rank merging to discard functionally important directions. Moreover, when merging many tasks, noise parameters may overwhelm critical knowledge.

**Goal**: (1) Construct a more essential, feature-distribution-aligned low-rank subspace; (2) amplify high-SNR parameters and suppress noise during merging.

**Key Insight**: Instead of decomposing parameter matrices directly, perform PCA on the **activation shifts** $\Delta O = X_{\text{proxy}} \Delta W^\top$ induced by parameter updates, obtaining principal directions directly relevant to task functionality. Additionally, parameter norms correlate highly with directional importance.

**Core Idea**: Perform low-rank decomposition and merging in the principal component space of activation shifts (rather than the singular value space of parameters), and use polarized scaling to amplify consensus signals.

## Method

### Overall Architecture
Input: pretrained weights $\theta_0$ + $T$ fine-tuned expert weights. Output: merged model $\theta_M$. The method has two modules: (1) Essential Subspace Decomposition (ESD)—low-rank decomposition of each task's parameter update matrix along activation shift principal directions; (2) Polarized Scaling (PS)—three-level scaling (inter-task, inter-dimension, inter-layer) to amplify high-signal parameters. The final merging formula: $\theta_M^{(\ell)} = \theta_0^{(\ell)} + \alpha \cdot \beta_\ell \cdot \Delta W_{\text{merged}}^{(\ell)}$.

### Key Designs

1. **Essential Subspace Decomposition (ESD)**:

    - Function: Construct a low-rank subspace aligned with the task's feature distribution
    - Mechanism: Forward-propagate 32 unlabeled proxy samples, compute activation shifts $\Delta O = X_{\text{proxy}} \Delta W^\top \in \mathbb{R}^{n \times d_{\text{out}}}$, and perform PCA on $\Delta O$ to obtain eigenvectors $P = [p_1, ..., p_{d_{\text{out}}}]$. Project $\Delta W$ onto $P$ to get coordinate matrix $A = P^\top \Delta W$, then truncate to top-$k$: $\widehat{\Delta W} = \hat{P} \hat{A}$. The truncation error is $\sum_{i=k+1}^{d_{\text{out}}} \lambda_i$, depending only on discarded eigenvalues and decoupled from the input distribution
    - Design Motivation: SVD truncation error also contains an $\mathbb{E}[(v_i^\top x)^2]$ weight, providing no guarantee of functional optimality. ESD decomposes directly in the output activation space, ensuring that only the functionally least important directions are discarded. Experiments show ESD achieves far higher CKA similarity than SVD when retaining 5% of components

2. **Essential Subspace Merging (ESM) Three-Step Process**:

    - Function: Orthogonally fuse ESD low-rank factors from multiple tasks
    - Mechanism: (a) **Decompose and truncate**: assign each task a rank budget $k = \lfloor d_{\text{out}} / T \rfloor$; (b) **Concatenate**: horizontally concatenate basis matrices $P_{\text{cat}} = [\hat{P}_1 | ... | \hat{P}_T]$, vertically concatenate coordinates $A_{\text{cat}}$; (c) **Orthogonalize**: perform SVD whitening on $P_{\text{cat}}$ and $A_{\text{cat}}$ separately: $\tilde{P} = U_P V_P^\top$, $\tilde{A} = U_A V_A^\top$, eliminating cross-task subspace correlations
    - Design Motivation: ESD bases from different tasks are not necessarily orthogonal; direct concatenation introduces interference. Whitening maximally decorrelates the basis vectors

3. **Three-Level Polarized Scaling (PS)**:

    - Function: Amplify high-confidence parameters and suppress noisy ones
    - Mechanism: All scaling factors take the form of (relative norm)²:
        - **Inter-task scaling** $s_t^{(\ell)} = (|\hat{A}_t^{(\ell)}|_F / \mathbb{E}_i[|\hat{A}_i^{(\ell)}|_F])^2$: prevents important task signals from being overwhelmed by numerous weak-task noise
        - **Inter-dimension scaling** $c_j^{(\ell)} = (|\mathbf{a}_j^{(\ell)}|_2 / \mathbb{E}_i[|\mathbf{a}_i^{(\ell)}|_2])^2$: amplifies input dimensions with strong cross-task consensus
        - **Inter-layer scaling** $\beta_\ell = (|\Delta W_{\text{merged}}^{(\ell)}|_F / \mathbb{E}_{i \in \mathcal{L}_{\text{type}}}[|\Delta W_{\text{merged}}^{(i)}|_F])^2$: compares only within same-type layers (e.g., all QKV layers), avoiding cross-layer competition caused by residual connections
    - Design Motivation: Experiments (Figures 3-4) verify that high-norm parameters correspond to high-confidence directions—loading parameters in high-norm-first order consistently performs best, even after norm normalization, indicating that directional quality rather than magnitude matters

### Loss & Training
ESM involves no training; it requires only a single forward pass of 32 unlabeled proxy samples. The merging coefficient $\alpha$ is selected on a validation set. Total additional overhead is minimal: 1.39s/task for PCA + 13.89s for orthogonalization (one-time) on ViT-B/16.

## Key Experimental Results

### Main Results

**ViT-B/16, Average Absolute Accuracy (%)**

| Method | 8 tasks | 14 tasks | 20 tasks |
|--------|---------|----------|----------|
| Task Arithmetic | 75.4 | 70.5 | 65.8 |
| TSV-M | 89.0 | 84.6 | 80.6 |
| Iso-CTS | 91.1 | 86.4 | 82.4 |
| **ESM (Ours)** | **91.8** | **87.4** | **84.9** |

**ViT-L/14, Average Absolute Accuracy (%)**

| Method | 8 tasks | 14 tasks | 20 tasks |
|--------|---------|----------|----------|
| TSV-M | 93.0 | 89.2 | 87.7 |
| Iso-CTS | 94.7 | 91.0 | 90.1 |
| **ESM (Ours)** | **94.8** | **91.3** | **90.4** |

### Ablation Study

| Decomposition | PS | ViT-B/16 8tasks | ViT-B/16 20tasks | Note |
|--------------|-----|-----------------|-----------------|------|
| SVD | None | 89.0 | 80.6 | Baseline (TSV-M) |
| SVD | All | 89.6 | 82.1 | PS also helps SVD |
| ESD | None | 90.9 | 82.8 | ESD alone: +1.9/+2.2 |
| ESD | Layer only | 91.4 | 83.7 | Inter-layer scaling contributes most |
| ESD | All | **91.8** | **84.9** | Three-level scaling further improves |

### Key Findings
- ESD's energy concentration is far higher than SVD's: fewer components capture equivalent energy, with significantly higher CKA similarity
- Polarized scaling is a universal module: it improves even SVD-based decomposition by 1.5%, demonstrating the generality of the norm-based scaling approach
- Proxy dataset composition has minimal impact: OOD data (ImageNet-1k) performs almost identically to ID data (difference <0.1%), and even single-class sampling does not affect results—indicating that activation shift principal directions are input-invariant
- As few as 4 proxy samples consistently outperform the SVD baseline
- Reciprocal scaling causes severe performance degradation (−5%+), validating the "high norm = high importance" hypothesis

## Highlights & Insights
- **The SVD vs ESD comparison is highly convincing**: From theory (truncation error formulas) to experiments (energy concentration, CKA), the paper comprehensively demonstrates that "decomposing in activation shift space is superior to parameter space." The core insight is that model merging should focus on functional impact rather than parameter energy
- **The robustness to proxy datasets is surprising**: Even PCA with completely OOD data yields nearly identical results. This suggests that fine-tuned models' activation shift principal directions are an intrinsic property independent of input data—itself an interesting finding
- **The polarized scaling design is simple yet effective**: The simple (relative norm)² formula achieves "amplify signal, compress noise" with three independently operating levels. More efficient and validation-free (for inter-layer scaling) than learned scaling coefficients (e.g., AdaMerging)

## Limitations & Future Work
- Requires 32 proxy samples for forward propagation—although the data requirement is minimal, it is not strictly data-free (compared to ACE-Merging)
- Validated only on vision tasks (ViT/CLIP); language model experiments are missing
- Global $\alpha$ coefficient still requires validation set selection; not fully automated
- Rank budget $k = \lfloor d_{\text{out}} / T \rfloor$ is uniformly allocated across all tasks without considering task complexity differences—adaptive rank allocation could be explored
- Whitening operation discards singular value information, potentially over-compressing useful scale differences

## Related Work & Insights
- **vs TSV-M**: TSV-M merges in parameter SVD space; ESM merges in activation PCA space. ESM consistently outperforms TSV-M across all settings (+2~3%)
- **vs Iso-CTS**: Iso-CTS constructs an isotropic common subspace via singular value normalization; ESM constructs an essential subspace from a functional perspective. ESM's advantage is more pronounced in the 20-task scenario (+2.5%)
- **vs ACE-Merging**: Both are CVPR'26 model merging works with entirely different approaches—ACE starts from covariance estimation with closed-form solutions; ESM starts from activation shift PCA with low-rank decomposition + orthogonalization. ACE is completely data-free; ESM needs 32 samples. The two may be complementary

## Rating
- Novelty: ⭐⭐⭐⭐ — ESD decomposition approach is novel, though PCA on activations is not entirely new in other domains
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Detailed ablations, proxy data robustness analysis, scaling coefficient visualization, and computational overhead reporting
- Writing Quality: ⭐⭐⭐⭐ — Clear logic and rich figures/tables, though the polarized scaling experimental motivation is slightly verbose
- Value: ⭐⭐⭐⭐ — Vision model merging SOTA, but lack of language model validation limits impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation](ace-merging_data-free_model_merging_with_adaptive_covariance_estimation.md)
- [\[CVPR 2026\] BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning](bd-merging_bias-aware_dynamic_model_merging_with_evidence-guided_contrastive_lea.md)
- [\[CVPR 2026\] DC-Merge: Improving Model Merging with Directional Consistency](dc-merge_improving_model_merging_with_directional_consistency.md)
- [\[CVPR 2026\] Defending Unauthorized Model Merging via Dual-Stage Weight Protection](defending_unauthorized_model_merging_via_dual-stage_weight_protection.md)
- [\[CVPR 2026\] Label-Free Cross-Task LoRA Merging with Null-Space Compression](label-free_cross-task_lora_merging_with_null-space_compression.md)

</div>

<!-- RELATED:END -->
