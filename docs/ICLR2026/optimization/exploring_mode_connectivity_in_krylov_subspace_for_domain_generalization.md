---
title: >-
  [Paper Note] Exploring Mode Connectivity in Krylov Subspace for Domain Generalization
description: >-
  [ICLR 2026][Optimization][mode connectivity] This paper moves beyond the mainstream approach of "finding flat minima" and instead exploits the **global geometric property of the loss surface—mode connectivity**. It proposes the Billiard Optimization Algorithm (BOA), which simulates billiard dynamics within a low-dimensional Krylov subspace to "walk" from a standard minimum along low-loss tunnels to a solution with stronger generalization, outperforming sharpness-aware methods…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "mode connectivity"
  - "Krylov subspace"
  - "domain generalization"
  - "loss landscape"
  - "sharpness-aware minimization"
  - "billiard dynamics"
date: 2026-05-08
content_hash: 8d51b7ead37cee93
---

# Exploring Mode Connectivity in Krylov Subspace for Domain Generalization

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=fpH2GYXJwD](https://openreview.net/forum?id=fpH2GYXJwD)  
**Code**: To be confirmed  
**Area**: Optimization / Domain Generalization  
**Keywords**: mode connectivity, Krylov subspace, domain generalization, loss landscape, sharpness-aware minimization, billiard dynamics  

## TL;DR
This paper moves beyond the mainstream approach of "finding flat minima" and instead exploits the **global geometric property of the loss surface—mode connectivity**. It proposes the Billiard Optimization Algorithm (BOA), which simulates billiard dynamics within a low-dimensional Krylov subspace to "walk" from a standard minimum along low-loss tunnels to a solution with stronger generalization, outperforming sharpness-aware methods like SAM on DomainBed.

## Background & Motivation
**Background**: Understanding the geometric structure of deep network loss surfaces has become a powerful tool for explaining generalization. The mainstream consensus suggests that "flat minima generalize better," leading to a wide range of sharpness-aware methods such as SAM, GSAM, SAGM, and DISAM, which have been extensively applied to Domain Generalization (DG) tasks.

**Limitations of Prior Work**: Recent theoretical and empirical studies (Dinh et al. 2017; Wen et al. 2023) indicate that **flatness is not universally equivalent to better generalization**. Sharp minima can generalize well, and the generalization benefits of flatness in standard neural networks (even two-layer ReLU) cannot be directly generalized from linear model conclusions. In DG scenarios with severe distribution shifts, parameter space flatness may fail to characterize model vulnerability to domain shifts in the feature space. In other words, focusing solely on **local sharpness** has reached a bottleneck.

**Key Challenge**: Loss surfaces do not consist of isolated basins; different minima are often connected by continuous low-loss paths (mode connectivity). This implies that a "suboptimal model" with poor DG accuracy and an "ideal model" with high accuracy might **reside at two ends of the same low-loss tunnel**. However, existing optimizers (SGD, Langevin dynamics, etc.) tend to get trapped locally and cannot actively traverse these tunnels. Furthermore, the "curse of dimensionality" in high-dimensional parameter spaces renders random direction searches almost inevitably ineffective.

**Goal**: This work aims to apply mode connectivity to DG algorithm design for the first time, providing an optimizer capable of efficient "navigation" within high-dimensional loss tunnels to reach strong generalization solutions from any starting point.

**Core Idea**: **[Geometric Insight]** Optimization trajectories are conceptualized as billiard balls moving within a "table" (the sub-level set of training loss below a threshold)—rolling in straight lines toward boundaries and undergoing specular reflection upon impact, thereby traversing low-loss regions while keeping training loss nearly constant. **[Dimensionality Reduction]** It is further discovered that "test gradients are highly aligned with the Krylov subspace spanned by training gradients." Based on this, the search is constrained to a subspace of only 5–20 dimensions, providing a near-optimal initial direction while overcoming the curse of dimensionality.

## Method

### Overall Architecture
BOA models DG optimization as a billiard game: it starts with a warm-up phase (ERM/SAM) to obtain an initial point $\theta_0$, then defines the "billiard table" $\mathcal{T}$ as the parameter region where training loss is below a threshold $\ell_{th}=\ell_{\text{train}}(\theta_0)+\Delta\ell$. Inside the table, two actions are performed iteratively: **line search** (the ball rolls toward the edge to locate the loss contour boundary) and **reflection** (changing direction according to physical laws upon collision). This generates a parameter trajectory, from which the best model is selected using a validation set. The key acceleration involves constraining all direction searches and reflections within a low-dimensional **Krylov subspace** generated by training gradients.

```mermaid
flowchart LR
    A[Warm-up ERM/SAM<br/>Start θ₀] --> B[Define Table T<br/>ℓ_train ≤ ℓ_th]
    B --> C[Construct Krylov Subspace<br/>K_K = span g, Hg, …, H^{K-1}g]
    C --> D[Determine Initial Dir p₀<br/>≈ Krylov approx of -∇ℓ_test]
    D --> E[Line Search<br/>Roll to contour boundary θ_i]
    E --> F[Reflection<br/>p_i = I-2ñ_iñ_iᵀ · p_{i-1}]
    F --> E
    F --> G[Select Best Model<br/>via Validation Set]
```

### Key Designs

**1. Table Definition: Turning "Constant Training Loss" into a Hard Constraint**
The billiard table is mathematically defined as the sub-level set of the training loss $\mathcal{T}:=\{\theta\in\mathbb{R}^d \mid \ell_{\text{train}}(\theta)\le \ell_{th}\}$, with the threshold $\ell_{th}=\ell_{\text{train}}(\theta_0)+\Delta\ell$, where $\Delta\ell>0$ is a small loss increment. This construction ensures the pre-trained model $\theta_0$ starts inside the table and, more importantly, restricts the optimization to a bounded space where training performance is nearly constant. Therefore, all solutions found by the algorithm share similar training losses, and their differences lie solely in their generalization to unseen domains. This is the prerequisite for using mode connectivity in DG: moving along an iso-loss surface essentially involves traversing different minima along low-loss tunnels.

**2. Line Search: Locating Contour Boundaries Like a Rolling Ball**
Given the current parameter $\theta_{i-1}$ and direction $p_{i-1}$, the line search solves the non-linear equation $\ell(\theta_{i-1}+\alpha p_{i-1})=\ell_{th}$ to find the step size where the "ball" hits the "table edge" (loss contour). BOA uses an adaptive bracketing strategy for coarse localization: if $\ell(x_k)<\ell_{th}$ (inside the table), the step size is expanded exponentially $h_{k+1}=(2k-1)h$ until the threshold is crossed; if $\ell(x_k)>\ell_{th}$ (outside the table), the step size is shrunk geometrically $h_k=h_{k-1}/2$. Once the interval $[h_L,h_R]$ is locked, a golden section search is used for refinement with $\psi=(1+\sqrt5)/2$, finally placing $\theta_i=\theta_{i-1}+\alpha^\star p_{i-1}$ precisely on the contour. This process only requires forward loss evaluations without second-order derivatives.

**3. Reflection: Momentum-Conserving Specular Bounce with O(d) Curvature Information**
Upon reaching the boundary point $\theta_i$, BOA uses the local loss geometry to determine a new direction. The unit normal vector is taken as the steepest descent direction $n_i=-\nabla\ell(\theta_i)/\|\nabla\ell(\theta_i)\|_2$ (acting as the normal to the table edge). The incident direction $p_{i-1}$ is updated via specular reflection:

$$p_i=p_{i-1}-2(p_{i-1}^\top n_i)n_i=(I-2n_in_i^\top)p_{i-1}.$$

This transformation subtracts twice the projection of $p_{i-1}$ onto the normal, **strictly conserving momentum** $\|p_i\|_2=\|p_{i-1}\|_2$ and satisfying the law of reflection (angle of incidence = angle of reflection). The reflection operator $R[n_i]=I-2n_in_i^\top$ is an improper rotation with $\det=-1$. Since $n_i\propto\nabla\ell(\theta_i)$, it implicitly injects curvature information **without recalculating the Hessian**, allowing efficient exploration along the contour with $O(d)$ complexity per step.

**4. Krylov Subspace Alignment: A "Free Lunch" for the Curse of Dimensionality**
The curse of dimensionality manifests in two ways: random vectors are almost orthogonal to the oracle optimal direction (making random initialization ineffective), and trajectories are sparse. The solution is to constrain everything within the Krylov subspace $\mathcal{K}_K(H_{\text{train}},g_{\text{train}})=\text{span}\{g_0,Hg_0,\dots,H^{K-1}g_0\}$, where $g_0=\nabla\ell_{\text{train}}(\theta_0)$ and $H=\nabla^2\ell_{\text{train}}(\theta_0)$. A core empirical finding of this paper is that **the test gradient $\nabla\ell_{\text{test}}(\theta_0)$ is highly aligned with this Krylov subspace**. Thus, the initial direction can be approximated as $p_0=-\sum_{k=0}^{K-1}\beta_k H^k g_0\approx-\nabla\ell_{\text{test}}(\theta_0)$. Experiments show that simply setting $\beta_k=1$ yields a $p_0$ with a sufficiently small angle to the oracle negative test gradient. Hessian-vector products are approximated using finite differences $Hg_0\approx[\nabla\ell(\theta_0+\epsilon g_0)-\nabla\ell(\theta_0)]/\epsilon$, maintaining $O(d)$ complexity. Reflections are also projected back: $p_i=(I-2\tilde n_i\tilde n_i^\top)p_{i-1}$ with $\tilde n_i=\text{proj}_{\mathcal{K}_K}n_i$. This allows obtaining a near-optimal initial direction without accessing test data while compressing the search space to 5–20 dimensions.

## Key Experimental Results

### Main Results
Out-of-domain accuracy (%) on five DomainBed datasets using ViT-B/16 (CLIP pre-trained + VPT), with all baselines fairly reproduced using a test-domain validation set:

| Method | VLCS | PACS | OfficeHome | TerraInc | DomainNet | Avg. |
|------|------|------|-----------|----------|-----------|------|
| ERM (VPT) | 81.9 | 95.9 | 84.1 | 56.1 | 59.5 | 75.5 |
| CORAL | 82.6 | 96.4 | 83.8 | 57.5 | 59.8 | 76.0 |
| SAM | 82.9 | 96.6 | 85.4 | 56.2 | 59.8 | 76.2 |
| GSAM | 82.9 | 96.6 | 85.6 | 55.4 | 59.8 | 76.1 |
| SAGM | 82.8 | 96.8 | 85.2 | 58.0 | 59.1 | 76.4 |
| DISAM | 82.7 | 97.1 | 85.4 | 57.3 | 59.8 | 76.5 |
| **BOA (Ours)** | **86.5** | **97.4** | **86.0** | **60.3** | **60.2** | **78.1** |

BOA achieves an average of 78.1%, a 1.6% improvement over the strongest baseline DISAM (76.5%), and a significant **3.6%** jump over SAM on VLCS.

### Ablation Study
Validation of billiard dynamics using the oracle negative test gradient $-\nabla\ell_{\text{test}}(\theta_0)$ as the initial direction (Summary of Table 4):

| Configuration | Avg. Improvement (5 Datasets) |
|------|------|
| ERM + BOA (Oracle Dir) | +4.9% |
| SAM + BOA (Oracle Dir) | +4.8% |

This indicates that given the correct initial direction, walking along low-loss tunnels consistently finds models with stronger generalization. The Krylov approximation ($\beta_k=1$) effectively approximates this oracle direction without test data.

### Key Findings
- **Krylov Alignment is Universal**: On Caltech101/LabelMe/SUN09/VOC2007, the cosine similarity of the test gradient projection in the Krylov subspace converges to 1 as dimensions increase, whereas random subspaces stay near 0. This alignment is robust across datasets and architectures.
- **Low Dimensions Suffice**: A Krylov subspace of only 5–20 dimensions is sufficient to contain high-generalization solutions, validating the efficacy of dimensionality reduction.
- **Validation Selection Trap**: The authors found that models with similar validation accuracy can differ by over 10% in DG performance along the trajectory. Thus, a test-domain validation set was used for fair comparison across all baselines.
- **Step Size Sensitivity**: There is a clear optimal range for step size $h$ (e.g., $h\approx0.5$ for Caltech101 to reach 99.2%); excessively large values lead to performance drops.

## Highlights & Insights
- **New Geometric Perspective**: Shifting focus from local flatness to global mode connectivity provides a strong counter-argument to the "flat minima as a universal solution" narrative and implements connectivity as a practical DG optimizer.
- **Enlightening Billiard Metaphor**: The combination of line search and reflection is intuitive yet mathematically rigorous, mapping physical dynamics to traversal on an iso-loss surface within bounded sub-level sets.
- **The "Free Lunch" of Krylov Alignment**: The empirical discovery that test gradients align with training Krylov subspaces allows the algorithm to find near-optimal directions without test data, which is perhaps the most valuable insight of the paper.
- **Computationally Friendly**: By using finite-difference HVPs, the process remains $O(d)$ and avoids explicit Hessian computation, making it feasible for real-world engineering.

## Limitations & Future Work
- **Dependency on Warm-up and Validation Selection**: BOA runs after SAM warm-up and requires a validation set to pick the best model along the trajectory. The paper admits that standard validation sets can be misleading, necessitating a test-domain validation set, which might be scrutinized under strict DG evaluation protocols.
- **Hyperparameter Overhead**: Parameters such as step size $h$, number of reflections, Krylov dimension $K$, and loss increment $\Delta\ell$ require grid searching; the cost of tuning for new tasks is not fully discussed.
- **Theoretical Gap**: Krylov alignment is currently a strong empirical observation. Why test gradients fall into training Krylov subspaces and when this alignment fails lack formal theoretical characterization.
- **Scope Limitation**: Experiments are concentrated on vision DG using CLIP-ViT + VPT. Generalization to other tasks like NLP, detection/segmentation, or full fine-tuning scenarios remains to be verified.

## Related Work & Insights
- **Local Structure / Sharpness-Awareness**: SAM, GSAM, SAGM, DISAM, and SWAD aim for flat minima; this work moves in the opposite direction, utilizing connectivity rather than sharpness for selection.
- **Mode Connectivity**: Initially explored by Garipov et al. (2018) and Draxler et al. (2018), low-loss paths between minima have been used for model merging, machine unlearning, and continual learning. This is the first application to OOD/DG optimization.
- **Krylov Subspace Methods**: Repurposing classical numerical linear algebra tools for high-dimensional optimization reduction suggests that the bridge between training-side second-order information and test-side gradients might be broadly applicable to transfer or adaptation problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First implementation of mode connectivity as a DG optimizer and the discovery of non-trivial Krylov alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid results on five DomainBed datasets with ViT backbones and extensive ablations; however, limited to CLIP-ViT+VPT.
- Writing Quality: ⭐⭐⭐⭐ Clear billiard metaphor, well-integrated formulas and figures, and an engaging narrative.
- Value: ⭐⭐⭐⭐ Provides a new tool beyond "finding flat minima," with the Krylov "free lunch" offering significant inspiration for the transfer learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [\[ICML 2025\] Understanding Mode Connectivity via Parameter Space Symmetry](../../ICML2025/optimization/understanding_mode_connectivity_via_parameter_space_symmetry.md)
- [\[CVPR 2026\] HFedATM: Hierarchical Federated Domain Generalization via Optimal Transport and Regularized Mean Aggregation](../../CVPR2026/optimization/hfedatm_hierarchical_federated_domain_generalization_via_optimal_transport_and_r.md)
- [\[ICLR 2026\] Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering](exploring_diverse_generation_paths_via_inference-time_stiefel_activation_steerin.md)
- [\[ICLR 2026\] Trion: FFT-based Dynamic Subspace Selection for Low-Rank Adaptive Optimization of LLMs](trion_fft-based_dynamic_subspace_selection_for_low-rank_adaptive_optimization_of.md)

</div>

<!-- RELATED:END -->
