---
title: >-
  [Paper Note] Amortized Active Generation of Pareto Sets
description: >-
  [NeurIPS 2025][Computational Biology][Multi-objective optimization] This paper proposes the A-GPS framework, which learns a conditional generative model over the Pareto set to perform online discrete black-box multi-obje…
tags:
  - "NeurIPS 2025"
  - "Computational Biology"
  - "Multi-objective optimization"
  - "Pareto set"
  - "generative models"
  - "active learning"
  - "preference conditioning"
date: 2026-05-08
content_hash: 0a7fda3cca46b9ba
---

# Amortized Active Generation of Pareto Sets

**Conference**: NeurIPS 2025
**arXiv**: [2510.21052](https://arxiv.org/abs/2510.21052)  
**Code**: None  
**Area**: Medical Imaging
**Keywords**: Multi-objective optimization, Pareto set, generative models, active learning, preference conditioning

## TL;DR
This paper proposes the A-GPS framework, which learns a conditional generative model over the Pareto set to perform online discrete black-box multi-objective optimization. It employs a non-dominance class probability estimator (CPE) as an implicit substitute for explicit hypervolume computation in PHVI, and achieves amortized posterior preference conditioning via preference direction vectors (without retraining). The approach demonstrates superior sample efficiency on synthetic benchmarks and protein design tasks.

## Background & Motivation

**Background**: Multi-objective black-box optimization (MOO) arises broadly in protein engineering, drug design, and related domains, where multiple conflicting objectives (e.g., stability vs. activity) must be simultaneously optimized. Conventional multi-objective Bayesian optimization (MOBO) relies on acquisition functions such as expected hypervolume improvement (EHVI), which are computationally expensive and scale poorly with the number of objectives. Random scalarization is simpler but fails to capture complex Pareto front geometries.

**Limitations of Prior Work**: (a) EHVI requires complex numerical integration whose cost grows exponentially with the number of objectives; (b) scalarization-based methods require retraining for each new preference weight vector; (c) existing methods do not support posterior preference conditioning—users must specify preferences prior to optimization.

**Key Challenge**: There is a need for a method that efficiently approximates the Pareto set without explicit hypervolume computation while supporting flexible, post-hoc preference conditioning.

**Goal**
- How can a generative model directly model the Pareto set?
- How can posterior preference specification be supported without retraining?

**Key Insight**: MOO is reformulated as learning a conditional generative model over the Pareto set—a non-dominance label $z$ guides the generative model toward high-performing regions, while a preference direction vector $\mathbf{u}$ enables amortized conditioning.

**Core Idea**: A CPE predicts non-dominance probability (implicitly estimating PHVI), and a conditional generative model $q_\phi(\mathbf{x}|\mathbf{u})$ conditioned on preference direction vectors enables Pareto set generation across multiple preferences after a single training run.

## Method

### Overall Architecture
At each iteration, A-GPS: (1) constructs non-dominance labels $z_n$ and preference directions $\mathbf{u}_n$ from observed data; (2) trains a CPE $\pi_\theta^z(\mathbf{x}) \approx p(z=1|\mathbf{x})$ to predict non-dominance probability; (3) trains a conditional generative model $q_\phi(\mathbf{x}|\mathbf{u})$ by maximizing an ELBO to approximate $p(\mathbf{x}|\mathbf{u}, z=1, a=1)$; (4) samples new candidates from the generative model, evaluates them, and updates the dataset. At inference time, the user specifies a preference $\mathbf{u}_\star$ and samples directly from $q_\phi(\mathbf{x}|\mathbf{u}_\star)$.

### Key Designs

1. **Non-dominance CPE as Implicit PHVI Estimator**

    - **Function**: Trains a classifier to predict whether a design belongs to the Pareto set.
    - **Mechanism**: Theorem 1 establishes that the hypervolume improvement indicator is equivalent to the non-dominance indicator: $\mathbb{1}[\text{HVI}(\mathbf{x}) > 0] = z(\mathbf{x})$. Consequently, a CPE trained with a proper loss automatically estimates PHVI: $\pi_\theta^z(\mathbf{x}) \approx \mathbb{P}(\text{HVI}(\mathbf{x}) > 0 | \mathbf{x})$.
    - **Design Motivation**: Explicit hypervolume computation scales exponentially with the number of objectives; replacing it with a simple classifier dramatically reduces computational complexity. The CPE guides the generative model to focus on non-dominated regions.

2. **Preference Direction Vectors and Alignment Indicator**

    - **Function**: Supports posterior preference conditioning without retraining.
    - **Mechanism**: The preference direction is defined as $\mathbf{u}_n = \frac{\mathbf{y}_n - \mathbf{r}}{\|\mathbf{y}_n - \mathbf{r}\|}$ (a unit vector, where $\mathbf{r}$ is a reference point), capturing the relative weighting among objectives. An alignment indicator $a$ is defined such that $a=1$ when an $(\mathbf{x}, \mathbf{u})$ pair is "aligned"; it is trained by contrasting true pairings against randomly permuted pairings. The learned conditional generative model $q_\phi(\mathbf{x}|\mathbf{u}) \approx p(\mathbf{x}|\mathbf{u}, z=1, a=1)$ can accept any user-specified preference $\mathbf{u}_\star$ at inference time.
    - **Design Motivation**: Preference direction vectors are more flexible than scalarization weights $\boldsymbol{\lambda}$—each new $\boldsymbol{\lambda}$ requires retraining, whereas the proposed method **trains once and serves all preferences** (amortization).

3. **Amortized ELBO Optimization**

    - **Function**: Jointly optimizes the CPE and the conditional generative model.
    - **Mechanism**: Minimizes $\mathbb{E}_{p(\mathbf{u}|z)}[D_{\text{KL}}[q_\phi(\mathbf{x}|\mathbf{u}) \| p(\mathbf{x}|\mathbf{u},z,a)]]$, decomposed via the ELBO into: a non-dominance CPE term (focusing on the Pareto set), an alignment CPE term (respecting user preferences), and a KL prior term.
    - **Design Motivation**: Amortized variational inference allows a single model to capture the full diversity of the Pareto front, enabling on-demand sampling conditioned on $\mathbf{u}$.

### Loss & Training
- CPE: proper scoring loss (log loss)
- Generative model: ELBO $= \mathbb{E}_{q_\phi}[\log \pi_\theta^z + \log \pi_\psi^a] - D_{\text{KL}}[q_\phi \| p_0]$
- Online iteration: each round updates the dataset, retrains the CPE, updates the generative model, and samples new candidates

## Key Experimental Results

### Main Results: Synthetic MOO Benchmarks

| Method | Hypervolume | Sample Efficiency |
|--------|-------------|------------------|
| MOBO (EHVI) | High | Low |
| Random Scalarization | Medium | Medium |
| **A-GPS** | **High** | **High** |

### Protein Design Task

| Task | A-GPS vs. Baselines |
|------|---------------------|
| Multi-objective protein optimization | Better Pareto front approximation + greater preference flexibility |

### Ablation Study

| Component | Contribution |
|-----------|-------------|
| Non-dominance CPE | Core—guides the generative model toward the Pareto set |
| Preference conditioning | Flexibility—enables preference specification without retraining |
| Alignment indicator | Precision—ensures generated samples are consistent with user preferences |

### Key Findings
- **Non-dominance CPE = implicit PHVI**: The theoretical equivalence is empirically confirmed—a simple classifier suffices to replace complex hypervolume computation.
- **Amortized preference conditioning is effective**: Training over multiple preference directions and accepting novel preferences at inference time eliminates the retraining overhead of scalarization-based methods.
- **Alignment indicator improves quality**: Training the alignment CPE via true vs. randomly permuted pairings ensures the generative model samples not only from the Pareto set but specifically from the user-specified region.
- **Particularly effective for discrete design spaces**: Discrete spaces such as protein sequences preclude gradient-based optimization; generative models provide a natural alternative.

## Highlights & Insights
- The **theoretical result that non-dominance CPE ≡ PHVI** is the central contribution: it reduces hypervolume computation to a simple classification problem, achieving an exponential reduction in computational complexity.
- The **preference direction vector** design is more natural than scalarization: unit vectors point toward the user's desired direction in objective space, offering clear geometric intuition and supporting amortization—after a single training run, users can "dial" across different preferences.
- The **Active Generation paradigm** (extended from VSD) reformulates optimization as generative modeling—rather than "searching for the optimum," the goal becomes "learning the optimal distribution," which is particularly advantageous in high-dimensional discrete spaces.

## Limitations & Future Work
- **Discrete space assumption**: The method is primarily designed for discrete design spaces; its effectiveness in continuous spaces has not been thoroughly validated.
- **CPE quality depends on data volume**: Early in the online process, when data are scarce, CPE estimates may be unreliable.
- **Non-convex Pareto fronts**: Preference direction vectors are best suited for convex Pareto fronts; coverage of non-convex regions may be incomplete.
- **Black-box evaluation cost**: Despite good sample efficiency, each round still requires expensive black-box evaluations.

## Related Work & Insights
- **vs. MOBO (EHVI)**: EHVI requires explicit hypervolume computation ($O(2^L)$); A-GPS uses CPE for implicit estimation ($O(1)$).
- **vs. ParetoFlow / ProUD (MOG)**: These methods perform offline Pareto set generation; A-GPS combines online active learning with generation.
- **vs. VSD**: VSD addresses single-objective active generation; A-GPS extends it to multi-objective settings with preference conditioning.
- **Transferable insight**: The idea of using CPEs to replace complex acquisition functions may generalize to other Bayesian optimization settings.

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical result that non-dominance CPE ≡ PHVI is novel; the amortized preference design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic benchmarks and protein design, though large-scale experiments are lacking.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous, though the notation system is heavy.
- Value: ⭐⭐⭐⭐ Practically significant for multi-objective black-box optimization, especially in protein and drug design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PROSPERO: Active Learning for Robust Protein Design Beyond Wild-Type Neighborhood](prospero_active_learning_for_robust_protein_design_beyond_wild-type_neighborhood.md)
- [\[ICML 2026\] From Feasible to Practical: Pareto-Optimal Synthesis Planning](../../ICML2026/computational_biology/from_feasible_to_practical_pareto-optimal_synthesis_planning.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)
- [\[NeurIPS 2025\] SpecMER: Fast Protein Generation with K-mer Guided Speculative Decoding](specmer_fast_protein_generation_with_k-mer_guided_speculative_decoding.md)

</div>

<!-- RELATED:END -->
