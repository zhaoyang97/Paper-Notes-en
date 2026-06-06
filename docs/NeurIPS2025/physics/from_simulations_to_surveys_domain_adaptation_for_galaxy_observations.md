---
title: >-
  [Paper Note] From Simulations to Surveys: Domain Adaptation for Galaxy Observations
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][domain adaptation] This work constructs a domain adaptation pipeline from simulated galaxies (TNG50) to real survey observations (SDSS) via feature-level alignment using Euc…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "domain adaptation"
  - "galaxy morphology"
  - "optimal transport"
  - "simulation-to-survey"
  - "GeomLoss"
date: 2026-05-08
content_hash: 216dd2abc7681080
---

# From Simulations to Surveys: Domain Adaptation for Galaxy Observations

**Conference**: NeurIPS 2025
**arXiv**: [2511.18590](https://arxiv.org/abs/2511.18590)  
**Code**: [GitHub](https://github.com/ahmedsalim3/galaxy-da)  
**Area**: Astrophysics / Domain Adaptation
**Keywords**: domain adaptation, galaxy morphology, optimal transport, simulation-to-survey, GeomLoss

## TL;DR
This work constructs a domain adaptation pipeline from simulated galaxies (TNG50) to real survey observations (SDSS) via feature-level alignment using Euclidean distance, optimal transport, and a top-$k$ soft-matching loss with trainable weight scheduling, improving target-domain morphology classification accuracy from 46.8% (no adaptation) to 87.3%, and Macro F1 from 0.298 to 0.626.

## Background & Motivation
**Background**: Large-scale survey projects (Rubin, Roman, Euclid) will image billions of galaxies, requiring automated inference of morphology, stellar mass, star formation rate, and other physical properties. Simulation data (e.g., Illustris TNG50) provide galaxy images with ground-truth physical labels.

**Limitations of Prior Work**: Significant domain shift exists between simulations and real observations — differences in PSF, noise patterns, background, selection functions, and class priors. Direct model transfer severely biases physical inference (e.g., distorting galaxy type mixing ratios and the mass–SFR relation).

**Key Challenge**: Real survey data lack reliable physical labels (citizen science projects such as Galaxy Zoo provide morphological labels but at high cost and limited scale), while simulated data carry ground-truth labels but follow a different distribution.

**Core Idea**: Align domains in feature space using multiple distance metrics — combining Sinkhorn OT, energy distance, and Gaussian MMD — and introducing a top-$k$ soft-matching loss to focus on the hardest-to-align samples.

## Method

### Overall Architecture
Source domain (TNG50 simulated galaxies + morphology labels) → Feature extractor (CNN / E(2)-Steerable CNN / ResNet-18) → Feature embeddings $z_s, z_t$ → Classification head (Focal Loss + class weights) + domain alignment loss ($\mathcal{L}_D + \mathcal{L}_{OT}$) → Evaluation on the target domain (SDSS real galaxies).

### Key Designs

1. **Supervised Loss and Class Imbalance Handling**

    - Function: Addresses severe imbalance among three morphological classes (elliptical / spiral / irregular).
    - Mechanism: Focal Loss ($\gamma=2$) + Effective Number class weights + learnable per-class logit scaling initialized from data statistics.
    - Design Motivation: Irregular galaxies are extremely rare; standard cross-entropy fails to learn them effectively.

2. **Domain Alignment Loss $\mathcal{L}_D$**

    - Function: Minimizes the source–target distributional discrepancy in the L2-normalized feature space.
    - Mechanism: Multiple distance metrics implemented via the GeomLoss library — (i) Sinkhorn divergence (entropic OT), (ii) energy distance, (iii) Gaussian MMD. The study further extends the comparison to 46 distance/similarity metrics across 8 families ($L_p$ Minkowski, $L_1$, Intersection, Inner Product, etc.), with systematic evaluation on 12 representative metrics.
    - Design Motivation: The topological structure of the embedding space fundamentally constrains cross-domain alignment quality, making metric selection critical.

3. **OT + Top-$k$ Soft-Matching Loss $\mathcal{L}_{OT}$**

    - Function: Penalizes the $k$ hardest-to-align sample pairs on top of global OT alignment.
    - Formula: $\mathcal{L}_{OT} = \lambda_{OT} d_\lambda(p_s, p_t) + \lambda_{match} \text{MSE}(z_s, P^\lambda z_t) + \lambda_{topk} \frac{1}{k}\sum_{\ell=1}^k d_{(\ell)}$
    - The three terms correspond to: global OT distance, soft barycentric matching between source and target, and a penalty on the $k$ largest nearest-neighbor distances.
    - Design Motivation: Global OT may overlook a minority of difficult samples; the top-$k$ loss focuses optimization on the hardest instances.

4. **Weight Scheduling Strategies**

    - Several strategies are explored: fixed weights, linear scheduling, and trainable weights $(\eta_1, \eta_2)$.
    - Optimal configuration: trainable weights with a 20-epoch warmup (classifier trained first, domain alignment introduced progressively).

### Loss & Training
- Three backbone architectures compared: CNN, E(2)-Steerable CNN, and ResNet-18 (ImageNet-pretrained, lower layers frozen, upper layers fine-tuned).
- Training for 200 epochs, batch size 128, AdamW optimizer.
- Source: 3,232 galaxies augmented to 25,856; target: 6,416 SDSS galaxies.

## Key Experimental Results

### Main Results

| Method | Target Accuracy | Macro F1 | Domain AUC |
|--------|----------------|----------|------------|
| Baseline (no adaptation) | 46.8% | 0.298 | 1.00 (fully separated) |
| DANN (adversarial) | 86.5% | — | ~0.5 |
| Euclidean (fixed weights) | ~85% | — | ~0.51 |
| **Euclidean (trainable weights)** | **87.3%** | **0.626** | **~0.51** |

Domain AUC ≈ 0.5 indicates that source and target features are indistinguishable in the latent space (ideal alignment).

### Key Findings
- Domain adaptation yields substantial gains: accuracy improves from 46.8% to 87.3% (+40.5 percentage points), confirming that the simulation-to-observation domain shift is a severe problem.
- Euclidean distance achieves the best alignment performance among the 12 evaluated metrics.
- Trainable weight scheduling outperforms both fixed and linear schedules, allowing the network to learn its own alignment pace.
- The 20-epoch warmup is critical — establishing good classification features before domain alignment is applied.
- The top-$k$ loss combined with ResNet effectively drives class-level cross-domain alignment.
- Irregular galaxies (the rare class) remain the primary challenge, as evidenced by Macro F1 substantially lagging behind accuracy.

## Highlights & Insights
- **Practical Pipeline**: Directly addresses the needs of forthcoming Rubin/Roman surveys, with a clear path from simulation-based training and real-data evaluation to future multi-task extension.
- **Metric Engineering**: Systematic comparison of 46 distance metrics is a distinctive contribution; the extended GeomLoss library offers practical value to the domain adaptation community.
- **Physical Significance**: The goal of domain alignment is not merely to improve classification accuracy, but more importantly to ensure the calibration of physical inference — a distinction that sets this work apart from general domain adaptation research.

## Limitations & Future Work
- Only three morphological classes (elliptical / spiral / irregular) are considered; extension to continuous physical quantities (stellar mass, SFR) has not yet been attempted.
- Performance on the irregular galaxy class remains poor, with Macro F1 of only 0.626.
- The paper is described as a "preliminary pipeline" and has not yet been validated on large-scale multi-redshift data.
- The method assumes a stable conditional label distribution $p_S(y|x) \approx p_T(y|x)$, whereas systematic differences in label definitions between simulations and observations may exist.

## Related Work & Insights
- **vs. DeepAstroUDA (Ciprijanović 2023)**: Prior work pioneered astronomical domain adaptation; this paper advances it through metric selection and the top-$k$ loss formulation.
- **vs. DANN**: The classical adversarial domain adaptation method serves as a baseline; the proposed distance-based approach achieves comparable performance with greater stability.

## Rating
- Novelty: ⭐⭐⭐ The domain adaptation methodology itself is not novel, but the top-$k$ OT matching and the systematic comparison of 46 metrics are valuable contributions.
- Experimental Thoroughness: ⭐⭐⭐ A preliminary pipeline with sufficient comparison across three backbones and multiple alignment strategies, though limited to three-class classification.
- Writing Quality: ⭐⭐⭐ Method descriptions are clear, but the paper is relatively short for a NeurIPS submission.
- Value: ⭐⭐⭐⭐ Directly addresses practical needs of the astronomical AI community and is well-positioned for next-generation surveys.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Primacy of Magnitude in Low-Rank Adaptation](the_primacy_of_magnitude_in_low-rank_adaptation.md)
- [\[NeurIPS 2025\] Neural Deprojection of Galaxy Stellar Mass Profiles](neural_deprojection_of_galaxy_stellar_mass_profiles.md)
- [\[NeurIPS 2025\] GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations](gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)
- [\[NeurIPS 2025\] EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale](eddyformer_accelerated_neural_simulations_of_three-dimensional_turbulence_at_sca.md)
- [\[NeurIPS 2025\] From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)

</div>

<!-- RELATED:END -->
