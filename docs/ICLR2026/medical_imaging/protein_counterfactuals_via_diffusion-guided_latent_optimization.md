---
title: >-
  [Paper Note] Protein Counterfactuals via Diffusion-Guided Latent Optimization
description: >-
  [ICLR 2026][Medical Imaging][Counterfactual Explanation] This paper proposes MCCOP, a framework that performs gradient-guided counterfactual optimization in a continuous joint sequence–structure latent space…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Counterfactual Explanation"
  - "Protein Engineering"
  - "Diffusion Models"
  - "Manifold Constraints"
  - "Sequence–Structure Embedding"
date: 2026-05-08
content_hash: f63346b2067eb47d
---

# Protein Counterfactuals via Diffusion-Guided Latent Optimization

**Conference**: ICLR 2026
**arXiv**: [2603.10811](https://arxiv.org/abs/2603.10811)
**Code**: [GitHub](https://github.com/weroks/mccop)
**Area**: Protein Engineering / Explainable AI
**Keywords**: Counterfactual Explanation, Protein Engineering, Diffusion Models, Manifold Constraints, Sequence–Structure Embedding

## TL;DR
This paper proposes MCCOP, a framework that performs gradient-guided counterfactual optimization in a continuous joint sequence–structure latent space, using a pretrained diffusion model as a manifold prior. With as few as 2–3 mutations, MCCOP generates biologically plausible protein variants that flip predictor outputs, simultaneously enabling model interpretation and protein design hypothesis generation.

## Background & Motivation

**Background**: Deep learning models can predict protein properties (stability, fluorescence, etc.) with high accuracy, yet they function only as "black-box oracles"—when a model predicts that an antibody is unstable, engineers receive no guidance on which mutations might rescue it.

**Limitations of Prior Work**: (1) Naive gradient optimization on proteins produces adversarial examples—sequences that satisfy the predictor but cannot fold; (2) Discrete search methods (genetic algorithms, hill climbing) require large numbers of mutations (8–11), are inefficient, and cannot guarantee structural plausibility; (3) Existing explainability methods (attention visualization, feature attribution) answer "why" but provide no actionable guidance on "how to fix."

**Key Challenge**: A fundamental tension exists between the discrete sequence and the continuous 3D structure of proteins—gradient-based methods require continuous relaxation, yet operating directly in sequence space ignores spatial relationships; meanwhile, optimization trajectories must remain on the biologically feasible manifold.

**Goal**: Given a protein predicted to lack a target property, how can one identify the minimal, biologically plausible sequence edits that flip the prediction? How can sparsity, validity, and biological plausibility be jointly achieved?

**Key Insight**: Operations are performed in the continuous joint sequence–structure embedding space provided by the CHEAP encoder. A pretrained diffusion model, DiMA, serves as a manifold prior—its denoising steps act as manifold projections, interleaved with classifier-guided gradient optimization.

**Core Idea**: In the joint protein embedding space, sparse gradient descent and diffusion-model manifold projection are applied alternately to generate counterfactual protein sequences with minimal mutations and structural plausibility.

## Method

### Overall Architecture
Input protein sequence → CHEAP encoder maps to joint sequence–structure latent space $z \in \mathbb{R}^{L' \times D}$ → alternating steps in latent space: (1) sparse gradient step (updates only top-$k$ sensitive positions) + (2) DiMA diffusion-model manifold projection → repeat until predictor flips → CHEAP decoder maps back to sequence and structure.

### Key Designs

1. **Predictor Smoothing**:

    - Function: Smooths the classifier's gradient field to prevent high-frequency gradients from steering optimization toward adversarial examples.
    - Mechanism: Four complementary mechanisms—(a) spectral normalization constrains the Lipschitz constant of all linear layers; (b) Jacobian regularization penalizes $\|\nabla_z f_\theta(z)\|_F^2$; (c) Softplus activation ($\beta=1$) replaces ReLU; (d) embedding-space FGSM adversarial augmentation—perturbations decoded back to the original sequence are added to training with the original label, teaching invariance to semantically irrelevant perturbations. After smoothing, gradient norms decrease by up to 4×, while AUROC is maintained or improved (stability: 0.94→0.98).
    - Design Motivation: High-frequency gradients from non-smooth classifiers produce "adversarial" perturbations—unconstrained gradient descent generates adversarial examples 100% of the time (when decoded back to sequences). Smoothing is a prerequisite for the framework's feasibility.

2. **Gradient-Sensitivity Sparse Mask**:

    - Function: Restricts gradient updates to the most sensitive positions, achieving sequence-level sparse mutations.
    - Mechanism: Per-position sensitivity is computed as $s_i = \|\nabla_{z_i} \mathcal{L}_{\text{CF}}\|_2$; a binary mask selects the top-$k$ positions $M_i = \mathbf{1}[s_i \geq s_{(k)}]$. Gradients are applied only at masked positions; non-masked positions are hard-reset to the original embedding. Because the CHEAP decoder is a position-wise MLP ($\hat{S}_i$ depends only on $z_i$), row-level masking in latent space directly corresponds to sparsity in sequence space.
    - Design Motivation: Minimal mutation is a core constraint in protein engineering—each additional mutation increases experimental cost and failure risk. Gradient sensitivity naturally identifies which positions are most critical for flipping the prediction.

3. **Diffusion-Model Manifold Projection**:

    - Function: Uses the pretrained DiMA diffusion model to pull the optimization trajectory back onto the feasible manifold of protein embeddings.
    - Mechanism: After each optimization step, the current embedding is partially diffused to noise level $t_{\text{diff}}$, then denoised to obtain the manifold projection $\Pi_\phi(z'_t)$, which is blended with mixing coefficient $\alpha=0.3$: $z_{t+1} = (1-\alpha)z'_t + \alpha \Pi_\phi(z'_t)$. This inverts the role of classifier guidance—diffusion is used not for generation but as a regularizer within the optimization loop.
    - Design Motivation: Unconstrained gradient optimization drifts away from the feasible protein space, producing sequences that cannot fold. The denoising steps of a diffusion model inherently pull samples back toward the data manifold.

### Loss & Training
Counterfactual optimization objective: $\mathcal{L}_{\text{CF}}(z_t) = \log(1+\exp(m - \tilde{y} \cdot f_\theta(z_t))) + \lambda_{\text{dist}} \|z_t - z_{\text{orig}}\|_2^2$, where $m$ is the confidence margin and $\lambda_{\text{dist}}$ controls the proximity–validity trade-off. Early stopping condition: $\sigma(\tilde{y} \cdot f_\theta(z_{t+1})) \geq \tau$ and the decoded sequence differs from the original. The predictor is trained as a shallow MLP with the four smoothing techniques described above.

## Key Experimental Results

### Main Results

| Dataset | Method | Success Rate↑ | Adversarial Rate↓ | Edit Distance↓ |
|--------|------|------|----------|------|
| Stability | MCCOP | **1.00** | 0.03 | **2.32** |
| Stability | Genetic Algorithm | 0.55 | 0.00 | 7.76 |
| Stability | Hill Climbing | 0.23 | 0.00 | 9.46 |
| Stability | Gradient Descent (unconstrained) | 1.00 | **1.00** | — |
| Fluorescence | MCCOP | 0.19 | 0.01 | **1.37** |
| Activity | MCCOP | **1.00** | 0.02 | **2.46** |
| Activity | Genetic Algorithm | 0.17 | 0.00 | 6.24 |

### Ablation Study

| Configuration | Description |
|------|------|
| No smoothing | 100% adversarial rate; gradient descent fails completely |
| Gradient norm after smoothing | Reduced by 2–4×; AUROC does not decrease (Stability: +0.04) |
| $\alpha=0$ (no manifold projection) | Degenerates to adversarial optimization |
| $\alpha=1$ (full projection) | Optimization becomes unstable |
| $\alpha=0.3$ (default) | Optimal balance |

### Key Findings
- Unconstrained gradient descent achieves a 100% adversarial rate, confirming the necessity of manifold constraints and smoothing.
- MCCOP achieves near-perfect success rates with only 2–3 mutations, whereas discrete methods require 8–11.
- Mutations are concentrated in functionally relevant regions: near the chromophore of GFP fluorescence (residues 63–69) and at the E2-binding interface of Ube4b activity (residues 66–71).
- MCCOP precisely recovers known counterfactual sequences in the datasets (16 fluorescence, 18 activity, 4 stability cases), some from held-out test sets.

## Highlights & Insights
- This work is the first to extend counterfactual explanation from image/tabular domains to proteins, elegantly leveraging CHEAP's position-wise decoding to directly map latent-space sparsity to sequence-space sparsity. The identified mutations align with known biophysical mechanisms (chromophore stacking, hydrophobic core consolidation), indicating that the predictor has learned meaningful sequence–function relationships.
- The use of a diffusion model as a "regularizer" rather than a "generator" is novel and provides a general manifold-constrained optimization paradigm.

## Limitations & Future Work
- Biological plausibility assessment relies on computational proxies (ESM3 pLDDT, radius of gyration, etc.); wet-lab validation is absent.
- Reconstruction errors introduced by the CHEAP encoder–decoder may produce artifacts for proteins far from the ESMFold training distribution.
- Only binary classification tasks are evaluated; continuous regression objectives (e.g., precise $T_m$ prediction) would require replacing the loss function.

## Related Work & Insights
- **vs. DVCE/DIME (image counterfactuals)**: These methods generate counterfactual images via guided denoising; MCCOP inverts this paradigm—using diffusion as a regularizing projection within the optimization loop rather than as a generator, which is better suited to the discrete–continuous hybrid nature of proteins.
- **vs. latent-space fitness optimization (ReLSO, etc.)**: These methods seek a global optimum rather than a minimal edit, and require task-specific generative model training; MCCOP requires no retraining and focuses on minimal modifications.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First application of diffusion-guided counterfactual explanation to proteins; framework design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, multiple baselines, and comprehensive physicochemical plausibility evaluation; wet-lab validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, algorithmic description is rigorous, and the discussion is candid.
- Value: ⭐⭐⭐⭐ Serves both model interpretability and protein engineering with practical application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](../../CVPR2026/medical_imaging/multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)
- [\[ICLR 2026\] Scalable Spatio-Temporal SE(3) Diffusion for Long-Horizon Protein Dynamics](scalable_spatio-temporal_se3_diffusion_for_long-horizon_protein_dynamics.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/medical_imaging/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](../../NeurIPS2025/medical_imaging/generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[ICLR 2026\] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)

</div>

<!-- RELATED:END -->
