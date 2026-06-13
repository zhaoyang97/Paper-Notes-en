---
title: >-
  [Paper Note] Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation
description: >-
  [ICML 2026][Interpretability][Self-Supervised Learning] KREPES utilizes eNTK to approximate arbitrary SSL models as kernel models and leverages the Representer Theorem to express representations as kernel-weighted combin…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Self-Supervised Learning"
  - "Representer Theorem"
  - "eNTK"
  - "Nyström Approximation"
date: 2026-05-08
content_hash: 0d8187aea492e660
---

# Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation

**Conference**: ICML 2026  
**arXiv**: [2509.24467](https://arxiv.org/abs/2509.24467)  
**Code**: To be confirmed  
**Area**: Interpretability / Self-Supervised Learning  
**Keywords**: Self-Supervised Learning, Interpretability, Representer Theorem, eNTK, Nyström Approximation

## TL;DR
KREPES utilizes eNTK to approximate arbitrary SSL models as kernel models and leverages the Representer Theorem to express representations as kernel-weighted combinations of "landmark samples." By employing Nyström approximation and one-step GGN-Newton, it analytically solves the influence coefficients for non-convex objectives such as SimCLR, BYOL, VICReg, and Barlow Twins, enabling the unsupervised auditing of SSL latent spaces and scaling to datasets with over 1M samples.

## Background & Motivation

**Background**: SSL methods like SimCLR, BYOL, VICReg, and Barlow Twins are currently the mainstream for learning representations from massive unlabeled data, but the trained networks remain black boxes. The community primarily relies on post-hoc methods like saliency maps and linear probes, or domain-specific interpretable architectures (e.g., geometric bottlenecks for video poses, prototype decoding for single-cell transcriptomics).

**Limitations of Prior Work**: Post-hoc methods cannot explain "what was actually learned inside" the SSL representations. Domain-specific solutions are tied to specific tasks and lack portability. Meanwhile, "intrinsically interpretable" approaches based on the Representer Theorem (Yeh 2018, Tsai 2023, Engel 2023) all depend on supervised signals—they derive representative point coefficients $\alpha_i \propto \partial L/\partial f(x_i)$ using label gradients, which are undefined without labels.

**Key Challenge**: (i) SSL lacks labels and specific prediction tasks, causing the feature-attribution paradigm to naturally fail. (ii) Using kernel methods for sample-level explanation incurs $O(n^2)$ memory and $O(n^3)$ time complexity for 1M+ samples, while existing Nyström/RFF accelerators (Rudi 2017, Della Vecchia 2024) only target convex losses and cannot handle non-convex objectives like SimCLR/BYOL.

**Goal**: Construct a unified framework to provide "intrinsic interpretability" for networks trained with any SSL objective—allowing sample-level tracing of "why $x_t$ is mapped to its current position" and concept-level auditing of "which concepts drive this embedding," while scaling to million-scale data like ImageNet-1K and Adult-1M.

**Key Insight**: The authors observe that eNTK can approximate a deep network as a linear kernel model. Once linearized, the Representer Theorem allows the learned representation to be written as $f(x_t) = \sum_l k(x_l, x_t) A_{l,:}$. The remaining problem is analytically obtaining the coefficient $A$ under non-convex SSL losses. The authors use Generalized Gauss–Newton (GGN) approximation to locally convexify the loss and Nyström approximation to project the RKHS into a finite-dimensional subspace spanned by $m \ll n$ landmarks.

**Core Idea**: Compress the SSL network into an eNTK + Representer Theorem form, then use "PC Initialization + One-step GGN-Newton + CG for Hessian-Vector Products" to analytically obtain dual coefficients in the Nyström subspace, allowing the entire interpretability pipeline to run in $O(n\sqrt{n})$ time.

## Method

### Overall Architecture
The KREPES pipeline consists of three stages: (a) **eNTK-ization**—freeze the pre-trained SSL network, compute the Jacobian of the forward output with respect to parameters, and construct the eNTK $k(x, x')$; (b) **Nyström + GGN Analytical Inference** (Sec.4)—select $m \ll n$ landmarks $\{\tilde{x}_l\}$, restrict representations to $f(x) = \tilde{A}^\top k_x + \gamma$, and compute $\Delta\tilde{A}$ via a one-step GGN-Newton starting from PC initialization $\tilde{A}_0 = U_h \Lambda_h^{-1/2}$; (c) **Interpretability Metrics** (Sec.3)—use $\Delta\tilde{A}$ to calculate Sample-Specific / Concept-Conditioned Influence Scores and Feature Alignment Gaps for unsupervised auditing of the latent space. Inputs are a pre-trained SSL backbone and unlabeled data $\{x_i\}$ (optionally with concept sets $\mathcal{P}_c, \mathcal{N}_c$), while outputs are top-K influence landmarks and concept scores for each test sample $x_t$.

### Key Designs

1. **SSL Influence Functions based on Representer Theorem + GGN (Sample-Specific & Concept-Conditioned Influence Score)**:
    - **Function**: Decomposes the SSL representation $f(x_t)$ into additive contributions from a set of training landmarks $\{\tilde{x}_l\}$ and further projects these contributions onto arbitrary concept vectors.
    - **Mechanism**: Based on the Representer Theorem, $f(x_t) = \sum_l k(\tilde{x}_l, x_t) \tilde{A}_{l,:}$. The influence of a small perturbation of landmark $\tilde{x}_l$ on parameters $\tilde{A}_{l,:}$ regarding $f(x_t)$ is $\nabla_{\tilde{A}_{l,:}} f(x_t) = k(\tilde{x}_l, x_t) I_h$. By performing a Taylor expansion of the SSL loss around $\tilde{A}_0$ and replacing the Hessian with a GGN proxy $\bar{H}_{GN} = J^\top Q J + \lambda I$, the one-step Newton solution is $\mathrm{vec}(\Delta\tilde{A}) = -\bar{H}_{GN}^{-1} \mathrm{vec}(\nabla_{\tilde{A}} L(\tilde{A}_0))$. The Sample-Specific Influence Score is defined as $\mathrm{IS}(\tilde{x}_l, x_t) = \|\nabla_{\tilde{A}_{l,:}} f(x_t) \Delta\tilde{A}_{l,:}^\top\|_2$. Concept explanations utilize CAV $v_c$ (Kim 2018), defining Concept-Conditioned Influence as $\mathrm{IS}(\tilde{x}_l, x_t; v_c) = \langle \nabla_{\tilde{A}_{l,:}} f(x_t) \Delta\tilde{A}_{l,:}^\top, v_c\rangle$.
    - **Design Motivation**: Supervised Representer coefficients $\alpha_i \propto \partial L/\partial f(x_i)$ are meaningless for SSL. GGN makes non-convex SSL losses locally PSD, allowing an analytical Newton step and decoupling the "causal impact of the training objective on geometry" from the "geometric covariance of the data itself"—the latter being captured by the PC initialization $\tilde{A}_0$.

2. **Scalable Analytical Inference via Nyström + PC Initialization + GGN-HVP**:
    - **Function**: Computes the Newton step for 1M+ samples while ensuring the landmark subspace carries geometric information of the data manifold.
    - **Mechanism**: Restricts the function class to $f(x) = \sum_{i \in [m], j \in [p]} \tilde{\alpha}_i^j k(\tilde{x}_i^j, x) + \gamma$. Uses Nyström approximation $K_{nn} \approx K_{nm} K_{mm}^\dagger K_{mn}$ and takes the truncated eigen-decomposition $K_{mm} \approx U_h \Lambda_h U_h^\top$. Setting $\tilde{A}_0 = U_h \Lambda_h^{-1/2}$ makes $f(X) = K_{nm} \tilde{A}_0$ exactly the Nyström feature map, equivalent to using PCA components as a prior. Solving for $\Delta\tilde{A}$ avoids explicitly forming the $O(m^3)$ Hessian by using Conjugate Gradient (CG) to solve the linear system $\bar{H}_{GN} \Delta\tilde{A} = -\nabla_{\tilde{A}} L(\tilde{A}_0)$, requiring only Hessian-Vector Products (HVP). GGN HVPs are derived for various SSL targets: Barlow Twins is treated as non-linear least squares of residuals $r(\theta)$, while SimCLR uses row-softmax cross-entropy.
    - **Design Motivation**: (i) Searching parameters directly in RKHS causes $O(n^2)$ memory explosion. (ii) PC initialization ensures $\Delta\tilde{A}$ only reflects the causal bias of the SSL objective. (iii) CG + HVP + GGN enables an analytical one-step process without forming dense matrices, reducing complexity to $O(n\sqrt{n})$.

3. **Feature Alignment Gap and Efficient Landmark Sampling**:
    - **Function**: Performs implicit bias auditing in tabular domains and ensures the Nyström subspace covers the most informative directions.
    - **Mechanism**: For a feature $\xi$, sample feature consistency is defined as $v_\xi(x_t, x_l) = 1 - \min(|x_{t,\xi} - x_{l,\xi}|/\Delta\xi, 1)$. The Feature-Conditioned Influence is $\mathrm{IS}(\tilde{x}_l, x_t; v_\xi) = \|\nabla_{\tilde{A}_{l,:}} f(x_t)\|_2 \cdot v_\xi(x_t, x_l)$. The Feature Alignment Gap is defined as $\mathrm{AG}_\xi = \mathbb{E}_{x_t}[\Psi(x_t; v_\xi) - \Psi_{\mathcal{R}_{\mathrm{rand}}}(x_t; v_\xi)]$. Two landmark sampling strategies are provided: k-means++ and approximate leverage score sampling.
    - **Design Motivation**: In tabular scenarios, features are semantic concepts themselves. The Alignment Gap quantifies whether a feature is systematically amplified by the SSL geometry, enabling the detection of algorithmic biases (e.g., preference for gender over education) without labels.

### Loss & Training
KREPES does not retrain the SSL model; it applies eNTK linearization and Nyström projection on a **frozen** pre-trained backbone and solves for a GGN-Newton step once. This is a post-training audit process with no additional training loss. Key hyperparameters include the number of landmarks $m = O(\sqrt{n})$, Tikhonov regularization $\lambda$, and projection dimension $h$.

## Key Experimental Results

### Main Results

| Dataset (Size) | SSL Objective | KREPES Acc Gap $\Delta$ | Kendall-$\tau$ (NN vs KREPES) | Confidence Drop (random / KREPES) |
|--------|------|------|----------|------|
| Adult (1M) | BT / SimCLR / VICReg | +0.06 / +0.12 / +0.12 | 0.845 / 0.842 / 0.840 | .0002 / .0572 etc. |
| Higgs (1M) | BT / SimCLR / VICReg | +0.03 / -0.10 / +0.25 | 0.781 / 0.778 / 0.783 | .0003 / .0461 etc. |
| ImageNet (1.2M) | BT / SimCLR / VICReg | -0.24 / -0.39 / -0.31 | 0.801 / 0.797 / 0.790 | .0001 / .0583 etc. |
| CoverType (1M) | BT / SimCLR / VICReg | -0.41 / +0.87 / +0.47 | 0.872 / 0.861 / 0.863 | .0003 / .0810 etc. |
| CIFAR-10 (60k) | BT / SimCLR / VICReg | -0.92 / -0.38 / -1.10 | 0.878 / 0.881 / 0.880 | .0011 / .0667 etc. |

eNTK + KREPES accuracy is nearly identical to the original NN ($|\Delta| < 1\%$), and $\tau \geq 0.78$ indicates nearly overlapping decision boundaries. Removing the top-10 landmarks identified by KREPES reduces k-NN confidence hundreds of times more than random removal, verifying that landmarks are "causal pillars."

### Ablation Study

| Configuration / Metric | Value | Description |
|------|---------|------|
| CIFAR-10 Class Coverage $\kappa$ — Barlow Twins | 12 (Acc 91.18%) | 12 top-norm landmarks cover 10 classes, high semantic alignment |
| CIFAR-10 $\kappa$ — VICReg / BYOL / SimCLR | 18 / 26 / 27 | Lower $\kappa$ correlates with higher downstream accuracy |
| Adult Precision@1 — KREPES vs cosine baseline | 0.872 vs 0.809 | KREPES top-1 landmark probability of being the same class is higher |
| Cover Precision@1 — KREPES vs baseline | 0.772 vs 0.550 | Gap widens to 22 percentage points on complex tabular data |

### Key Findings
- **Landmark Ranking as a Downstream Ability Proxy**: On CIFAR-10, Barlow Twins' $\kappa=12$ corresponds to the highest linear probe accuracy, suggesting landmark coverage is an unsupervised signal for SSL model quality.
- **Spectral Entropy for Hyperparameter Selection**: On MNIST + Barlow Twins, the peak of normalized spectral entropy for $\tilde{A}^\top \tilde{A}$ aligns with the peak linear probe accuracy, providing a zero-label tuning solution.
- **Unsupervised Auditing of Implicit Bias**: On Adult-1M, Alignment Gap shows SSL models amplify sensitive attributes like gender/relationship over education. On FairFace, KREPES reveals cross-group confusion for Southeast Asian and Indian faces.
- **Visualization of Repulsive Forces**: KREPES models both positive and negative influences; red "repulsive landmarks" show SSL explicitly pushing apart visually similar but semantically different samples (e.g., bird vs. plane).

## Highlights & Insights
- The **three-stage combination of Representer + eNTK + GGN** is ingenious: eNTK linearizes deep nets, Representer provides the landmark decomposition form, and GGN reduces non-convex SSL targets into solvable quadratic problems.
- **Geometric Meaning of PC Initialization**: Fixing the Taylor expansion point on Nyström principal components ensures $\Delta\tilde{A}$ only captures the SSL objective's causal bias, decoupling "data prior" from "loss contribution."
- **HVP-only Inference**: Using CG + jvp/vjp to compute HVPs without forming the Hessian is the key engineering trick for scaling kernel methods to 1M+ samples.

## Limitations & Future Work
- The framework is based on **eNTK linearization**; for very deep or highly non-linear networks (e.g., global attention), the fidelity of eNTK approximation may decrease.
- **One-step GGN-Newton** assumes the loss curvature is well-characterized by the PSD proxy, which might be unreliable for SSL losses near plateaus or collapses.
- **Concept sets require manual provision**, necessitating integration with automatic concept discovery in open domains.

## Related Work & Insights
- **vs. Yeh et al. 2018 / Tsai et al. 2023**: They explain **supervised** DNNs using label gradients; this work is the first to bring the Representer framework to SSL using GGN.
- **vs. Rudi et al. 2017**: Their Nyström acceleration only supports convex losses; this work extends Nyström to non-convex SSL objectives via GGN-based local convexification.
- **vs. Koh & Liang 2017 (Influence Function)**: Classic IF requires any optimal point to be PSD and relies on labels; KREPES replaces the Hessian with eNTK and labels with the Representer Theorem.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to extend Representer Theorem to SSL; complete integration of eNTK + GGN + Nyström.
- Experimental Thoroughness: ⭐⭐⭐⭐ 1M+ scale across images/tables and 4 SSL objectives, though lacks Transformer-scale vision models.
- Writing Quality: ⭐⭐⭐⭐ Rigorous notation and clear architecture diagrams.
- Value: ⭐⭐⭐⭐⭐ Provides a unified, scalable path for SSL interpretability and bias auditing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension](idest_assessing_self-supervised_learning_representations_via_intrinsic_dimension.md)
- [\[NeurIPS 2025\] Dataset Distillation for Pre-Trained Self-Supervised Vision Models](../../NeurIPS2025/interpretability/dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [\[ICML 2026\] MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification](minimax_learning_of_interpretable_factored_stochastic_policies_from_conjoint_dat.md)
- [\[CVPR 2026\] RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation](../../CVPR2026/interpretability/riskprop_collision-anchored_self-supervised_risk_propagation_for_early_accident_.md)
- [\[ICCV 2025\] AIM: Amending Inherent Interpretability via Self-Supervised Masking](../../ICCV2025/interpretability/aim_amending_inherent_interpretability_via_self-supervised_masking.md)

</div>

<!-- RELATED:END -->
