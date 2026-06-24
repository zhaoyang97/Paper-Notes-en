---
title: >-
  [Paper Note] Revisiting Sparsity Constraint Under High-Rank Property in Partial Multi-Label Learning
description: >-
  [CVPR 2026][Partial Multi-Label Learning] This paper points out that the long-standing joint assumptions of "sparse noise labels + low-rank true labels" in Partial Multi-Label Learning (PML) are actually contradictory. It proves that sparse perturbations instead **preserve** the high-rank property of predicted label matrices. Accordingly, the authors propose Schirn—which applies a sparsity constraint to the noise matrix and a high-rank (nuclear norm) constraint to the predict…
tags:
  - "CVPR 2026"
  - "Partial Multi-Label Learning"
  - "Sparsity Constraint"
  - "High-Rank Property"
  - "Nuclear Norm"
  - "Label Disambiguation"
date: 2026-05-08
content_hash: 512936ffae359595
---

# Revisiting Sparsity Constraint Under High-Rank Property in Partial Multi-Label Learning

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Si_Revisiting_Sparsity_Constraint_Under_High-Rank_Property_in_Partial_Multi-Label_Learning_CVPR_2026_paper.html)  
**Code**: None  
**Area**: Partial Multi-Label Learning / Weakly Supervised Learning  
**Keywords**: Partial Multi-Label Learning, Sparsity Constraint, High-Rank Property, Nuclear Norm, Label Disambiguation

## TL;DR
This paper points out that the long-standing joint assumptions of "sparse noise labels + low-rank true labels" in Partial Multi-Label Learning (PML) are actually contradictory. It proves that sparse perturbations instead **preserve** the high-rank property of predicted label matrices. Accordingly, the authors propose Schirn—which applies a sparsity constraint to the noise matrix and a high-rank (nuclear norm) constraint to the prediction matrix—consistently outperforming 9 SOTAs across 11 datasets.

## Background & Motivation

**Background**: Partial Multi-Label Learning (PML) addresses a weakly supervised setting where each sample is provided with a **candidate label set** $C_i$, containing both ground-truth and noise labels. The model must identify the true labels from this set. Recent mainstream approaches are almost entirely built on two "golden assumptions": the noise label matrix $N$ is **sparse** (noise is a minority), and the ground-truth label matrix is **low-rank** (labels are highly correlated, making the matrix compressible). Jointly optimizing these as a unified loss is believed to achieve both disambiguation and label correlation exploitation.

**Limitations of Prior Work**: The authors question the validity of this paradigm from two perspectives. First, **sparsity and low-rankness are naturally in conflict**: The observed matrix $Y$ equals the ground-truth matrix $Y_g$ plus sparse noise $N$. According to Wedin's theorem, sparse perturbations have minimal impact on matrix singular values; thus, the rank of the predicted matrix should align closely with the rank of the observed matrix $Y$. However, Table 1 shows that observed matrices $Y$ for almost all real PML datasets are **full-rank**. Forcing the prediction matrix to be low-rank contradicts this fact. Second, **ground-truth label matrices are themselves nearly full-rank**: Label correlation does not imply universal co-occurrence; correlations do not eliminate the independence of labels. The rank of $Y_g$ in Table 1 (e.g., YeastBP 200/217, YeastCC 45/50) is indeed close to full rank.

**Key Challenge**: The sparsity assumption has been misused. Previously, "noise sparsity" was treated as a means to suppress the rank of the true matrix, but this runs counter to the full-rank structure of real-world data.

**Goal**: To clarify the relationship between sparsity and rank, and provide a non-contradictory PML modeling framework.

**Key Insight**: Since sparse perturbations only slightly modify singular values, the true function of sparsity constraints is not "rank reduction" but **preserving the high-rank structure** of the predicted matrix—which coincides with the fact that ground-truth matrices are full-rank.

**Core Idea**: Replace the contradictory "sparse noise + low-rank ground-truth" assumptions with the **compatible** pair of "sparse noise + high-rank prediction."

## Method

### Overall Architecture
Schirn (Sparsity constraint under high-rank property) is essentially a **matrix-decomposition-based linear classifier + dual-constraint objective + alternating optimization** scheme. It avoids complex multi-module pipelines, focusing entirely on "how the objective is formulated, why, and how to solve it."

The input consists of an instance matrix $X \in \mathbb{R}^{n\times d}$ and an observed (candidate) label matrix $Y \in \{0,1\}^{n\times l}$. The goal is to learn a weight matrix $W \in \mathbb{R}^{d\times l}$ and a noise label matrix $N \in \mathbb{R}^{n\times l}$. The model assumes the linear mapping $XW$ approximates the **denoised** true labels $Y-N$. Building on a basic least-squares classifier (Eq. 1), Schirn adds two elements: a sparsity constraint on $N$ (noise is rare) and a high-rank constraint on the prediction matrix $XW$ (retaining the richness of label structures). Since both the rank function and the $\ell_0$ norm are non-convex and difficult to solve, the authors use the nuclear norm and $\ell_1$ norm as convex relaxations. They introduce an auxiliary variable $C=XW$ and utilize the Augmented Lagrangian Method (ALM) to decompose the problem into three sub-problems for $W$, $N$, and $C$, solved iteratively via closed-form or proximal methods.

The overall objective function (Eq. 5) is:

$$\min_{W,N}\ \|XW-(Y-N)\|_F^2 + \alpha\|N\|_1 - \beta\|XW\|_* + \lambda\|W\|_F^2$$

Subject to $N\in\{0,1\}^{n\times l}$ and $\forall i,j,\ N_{ij}\le Y_{ij}$ (noise can only exist within the candidate set).

### Key Designs

**1. Elevating "Sparsity $\Rightarrow$ High-Rank" from Intuition to Theory: Theorem 1**

This is the foundation of the paper. While previous works assumed "noise sparsity" serves "ground-truth low-rankness," this paper proves that sparse perturbations actually **maintain** high rank. Given a full-rank $Y\in\mathbb{R}^{n\times l}$ ($\text{rank}(Y)=\min(n,l)$) and a sparse binary matrix $N$ where $\|N\|_0\le\epsilon$ ($\epsilon$ being a small integer), the rank of $Y_g=Y-N$ satisfies:

$$\text{rank}(Y_g)\ \ge\ \min(n,l)-\Delta,$$

where $\Delta$ is a very small positive integer depending only on the sparsity $\epsilon$. Intuitively, since singular values are robust to sparse perturbations (Wedin’s theorem), changing a few elements at most reduces the rank by $\Delta$, remaining near full-rank. This settles the debate on rank: since true matrices are full-rank and sparse noise preserves rank, high-rank modeling is the correct choice.

**2. Dual-Constraint Objective and Convex Relaxation**

The ideal objective would be $\min_{W,N}\ \alpha\|N\|_0 - \beta\,\text{rank}(XW)$ (Eq. 3), where the second term **maximizes** (note the negative sign) the rank of the prediction matrix. Due to non-convexity, Schirn applies two relaxations: replacing the rank function with its convex surrogate, the **nuclear norm** $\|XW\|_*$ (sum of singular values), and the $\ell_0$ norm with the $\ell_1$ norm $\|N\|_1$. This results in $\min_{W,N}\ \alpha\|N\|_1 - \beta\|XW\|_*$ (Eq. 4). The term $-\beta\|XW\|_*$ is the key innovation: a negative sign before the nuclear norm **encourages larger singular values**, actively pushing the prediction matrix toward high rank. This is the exact opposite of traditional low-rank regularization.

**3. ALM Alternating Optimization**

To decouple $XW$, the auxiliary variable $C=XW$ is introduced. The ALM approach uses a dual variable $\Lambda$ and penalty coefficient $\mu$, splitting the problem into:

- **$W$ sub-problem**: Has a closed-form solution $W=(\mu X^TX+2\lambda I)^{-1}(\mu X^TC-X^T\Lambda)$.
- **$N$ sub-problem**: Box-constrained $\ell_1$ minimization solved via ISTA. The solution involves soft-thresholding $S_{\alpha/L_f}$ followed by a projection $T_Y$ to ensure $N$ remains binary and $N_{ij}\le Y_{ij}$.
- **$C$ sub-problem**: Solved via the Singular Value Shrinkage theorem on $G=\frac{2Y-2N+\Lambda+\mu XW}{2+\mu}$. The solution is $C=U\max(0,\Sigma+\frac{2\beta}{2+\mu}I)V^T$. Crucially, this **adds** a positive value to singular values, reflecting the high-rank constraint.

### Loss & Training
The objective is Eq. 5. Three hyperparameters are used: $\alpha$ controls sparsity (searched in $[0.1, 2]$), $\beta$ controls high-rank intensity (searched in $[0.01, 0.1]$), and $\lambda$ controls model complexity. The method requires no deep networks and converges quickly within a few iterations.

## Key Experimental Results

Experiments used 5 real PML datasets + 6 synthetic datasets (11 total) against 9 SOTAs (NLR, FPML, PML-LRS, PML-NI, P-MAP, P-VLS, PAKS, GLC, PARD) using five metrics.

### Main Results (Average Precision ↑, %)

| Dataset (r) | Schirn | NLR | PML-NI | PAKS | PARD |
|---|---|---|---|---|---|
| Music emotion | **62.6** | 58.6 | 60.8 | 61.3 | 60.8 |
| Music style | **75.0** | 71.4 | 73.8 | 72.8 | 73.2 |
| YeastCC | **66.5** | 64.4 | 45.5 | 62.0 | 33.3 |
| YeastBP | **43.8** | 40.9 | 25.5 | 39.9 | 30.8 |
| Birds (r=3) | **61.8** | 55.8 | 54.0 | 46.3 | 37.8 |
| Medical (r=3) | **90.8** | 87.4 | 87.6 | 61.4 | 85.2 |
| Enron (r=3) | **70.6** | 64.2 | 60.6 | 67.6 | 66.5 |

Schirn achieved the best average precision across all listed datasets and noise rates.

### Ablation Study (Selected Results, Average Precision %)

| High-Rank | Sparsity | Low-Rank | Scene | Birds | Medical | Enron | Chess |
|:---:|:---:|:---:|---|---|---|---|---|
| ✕ | ✓ | ✕ | 83.1 | 53.3 | 88.4 | 68.6 | 43.3 |
| ✓ | ✕ | ✕ | 58.2 | 44.2 | 84.5 | 47.0 | — |
| ✕ | ✓ | ✓ (Replace) | 83.7 | 53.7 | 88.4 | 68.0 | 43.1 |
| ✓ | ✓ | ✕ (Ours) | **86.2** | **61.8** | **90.8** | **70.6** | **47.5** |

### Key Findings
- **High-rank constraint provides gains**: Performance improved across all datasets when the high-rank term was included (e.g., Birds precision rose from 0.533 to 0.618), validating that maintaining rank is vital for label structure richness.
- **Sparsity is indispensable**: Performance collapsed without sparsity constraints (e.g., Enron precision dropped from 0.706 to 0.470), proving it is the primary driver for suppressing noise labels.
- **High-rank outperforms low-rank**: Replacing high-rank with low-rank constraints resulted in inferior performance, debunking the years-old low-rank assumption.
- **Rank is preserved**: Schirn's predicted rank $r(P)$ aligns closely with $r(Y_g)$, whereas removing the high-rank term ($\beta=0$) caused the rank to collapse.

## Highlights & Insights
- The most compelling aspect is **refuting a multi-year consensus**: Instead of inventing a new network, the author uses Wedin's theorem and statistical evidence to show "sparsity + low-rank" is contradictory, then rebuilds the framework based on "sparsity $\Rightarrow$ high-rank."
- The **negative sign before the nuclear norm** is the most elegant technical detail: while low-rank methods use $+\|\cdot\|_*$ to penalize singular values, this paper uses $-\beta\|\cdot\|_*$ to encourage them, representing a fundamental shift in perspective.
- The method relies on **closed-form/proximal iterations without deep training**, making it efficient for small-to-mid-scale datasets and potentially applicable to other "full-rank + sparse noise" scenarios.

## Limitations & Future Work
- **Linear Mapping Assumption**: Using $XW$ might be insufficient for datasets with strong non-linear features or extreme label counts.
- **Full-Rank Boundary**: The motivation relies on data being nearly full-rank. In domains where labels are truly highly co-occurrent and inherently low-rank, this method's advantages might diminish.
- **Scalability**: Sub-problems involve $d \times d$ matrix inversions and SVDs, which may become computationally expensive for very large feature dimensions or sample sizes.

## Related Work & Insights
- **vs PML-LRS / PML-NI**: These apply **low-rank** regularization on $W$. This paper proves this conflicts with sparsity and full-rank facts, demonstrating that high-rank constraints are consistently superior.
- **vs NLR**: NLR only uses sparsity to suppress irrelevant labels. Schirn adds the high-rank term, significantly outperforming NLR on datasets like Birds (61.8 vs 55.8).
- **vs Two-stage Methods**: Unlike methods that perform disambiguation before training, Schirn integrates disambiguation and classification into a single dual-constraint objective, avoiding error accumulation between stages.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Directly refutes the "low-rank" assumption and theoretically rebuilds it as "high-rank."
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive datasets, metrics, and ablation studies, though lacking extreme-scale MLL.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic from questioning consensus to theoretical proof and formulaic implementation.
- Value: ⭐⭐⭐⭐ Corrects a widely accepted erroneous assumption and provides a simple, reproducible method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Revisiting F-measure Optimization in Multi-Label Classification: A Sampling-based Approach](revisiting_f-measure_optimization_in_multi-label_classification_a_sampling-based.md)
- [\[CVPR 2026\] Evidential Deep Partial Label Learning to Quantify Disambiguation Uncertainty](evidential_deep_partial_label_learning_to_quantify_disambiguation_uncertainty.md)
- [\[CVPR 2026\] Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)
- [\[CVPR 2026\] NexusFlow: Unifying Disparate Tasks under Partial Supervision via Invertible Flow Networks](nexusflow_unifying_disparate_tasks_under_partial_supervision_via_invertible_flow.md)
- [\[CVPR 2026\] Prototype-based Causal Intervention for Multi-Label Image Classification](prototype-based_causal_intervention_for_multi-label_image_classification.md)

</div>

<!-- RELATED:END -->
