---
title: >-
  [Paper Note] The Pareto Frontier of Resilient Jet Tagging
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][jet tagging] This work systematically evaluates the AUC–resilience trade-off across multiple architectures (DNN/PFN/EFN/ParT) for LHC jet tagging tasks…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "jet tagging"
  - "Pareto frontier"
  - "resilience"
  - "model dependence"
  - "quark/gluon discrimination"
date: 2026-05-08
content_hash: 01ebb8bbdbb8f9bf
---

# The Pareto Frontier of Resilient Jet Tagging

**Conference**: NeurIPS 2025
**arXiv**: [2509.19431](https://arxiv.org/abs/2509.19431)  
**Code**: [Zenodo](https://zenodo.org/) (dataset publicly available)  
**Area**: Physics
**Keywords**: jet tagging, Pareto frontier, resilience, model dependence, quark/gluon discrimination

## TL;DR
This work systematically evaluates the AUC–resilience trade-off across multiple architectures (DNN/PFN/EFN/ParT) for LHC jet tagging tasks, revealing that more complex models achieve higher AUC but exhibit stronger Monte Carlo model dependence. A Pareto frontier is constructed, and a case study demonstrates that low-resilience classifiers introduce bias in downstream parameter estimation even after calibration.

## Background & Motivation

**Background**: Jet tagging is a core task in LHC data analysis, and state-of-the-art architectures based on Transformers and GNNs (e.g., ParT) substantially outperform traditional methods in AUC.

**Limitations of Prior Work**: Optimizing solely for AUC leads to selecting architectures with strong model dependence—these models may learn simulator-specific artifacts rather than genuine physical structure. ATLAS has found that classifiers are more sensitive to physics modeling uncertainties than to detector effects.

**Key Challenge**: The community treats AUC as the sole optimization objective ("when a measure becomes a target, it ceases to be a good measure"), neglecting classifier robustness across different Monte Carlo generators.

**Goal**: Quantify the trade-off between AUC and resilience, and demonstrate that low-resilience models introduce bias in real physics analyses.

**Key Insight**: Resilience is measured as the percentage difference in AUC between training on Pythia-8 and testing on Herwig-7.

**Core Idea**: Construct a Pareto frontier of AUC vs. resilience, showing that complex models occupy the high-AUC/low-resilience corner while simpler models such as EFN and expert features occupy the low-AUC/high-resilience corner. Knowledge distillation is shown to be incapable of pushing beyond this frontier.

## Method

### Overall Architecture
Pythia-8 generates training data → Multiple architectures are trained on q/g and top tagging tasks → AUC is evaluated on both Pythia and Herwig test sets → Resilience (percentage AUC difference) is computed → Pareto frontier is constructed → Case study validates downstream impact.

### Key Designs

1. **Systematic Multi-Architecture Evaluation**:

    - Coverage: Expert Features (angularities/multiplicities), DNN (2–10 layers, 1–300 nodes), PFN/EFN (latent space dimension 1–1024), Particle Transformer (attention heads 2/4/8)
    - Unified particle-level kinematic inputs ($p_T$, $\eta$, $\phi$), $p_T$ in 500–550 GeV range, no detector simulation

2. **Resilience Metric**:

    - Pythia and Herwig employ different parton shower and hadronization models, representing systematic uncertainties in physics modeling

3. **Knowledge Distillation Experiments**:

    - Teacher model: PFN; student models: various DNNs and EFNs, trained by minimizing KL divergence
    - Objective: Attempt to push beyond the Pareto frontier

4. **Case Study: q/g Mixture Fraction Estimation**:

    - Two PFNs on the Pareto frontier are selected (small = high resilience, large = high AUC)
    - The quark jet fraction $\kappa$ is estimated from a mixed sample via likelihood ratio, calibrated by Pythia–Herwig reweighting

## Key Experimental Results

### Main Results — Pareto Frontier Observations

| Architecture | AUC Range (q/g) | Resilience Range | Trend |
|---|---|---|---|
| Expert Features | Lower | Lowest (most robust) | Physics-motivated features are most stable |
| EFN | Moderate | Low–moderate | IRC-safe architecture is relatively stable |
| PFN | Moderate–high | Moderate–high | Latent space dimension drives the trade-off |
| ParT | Highest | Highest (least robust) | Complex attention mechanisms are most model-dependent |

### Case Study — q/g Mixture Fraction $\kappa$ Estimation

| Classifier | True $\kappa$ | Pythia Inference | Calibrated Herwig | Conclusion |
|---|---|---|---|---|
| Large PFN | 0.50 | 0.490±0.005 | 0.529±0.006 | Biased ✗ |
| Large PFN | 0.25 | 0.253±0.005 | 0.305±0.006 | Biased ✗ |
| Small PFN | 0.50 | 0.504±0.013 | 0.478±0.017 | Unbiased ✓ |
| Small PFN | 0.25 | 0.258±0.013 | 0.268±0.013 | Unbiased ✓ |

### Key Findings
- **The Pareto frontier is clearly observable**: Model complexity is the primary driver of movement along the frontier; different architectures form vertically clustered distributions in the top tagging task.
- **Knowledge distillation cannot break the frontier**: Distilled student models outperform linear combinations but cannot surpass the Pareto frontier.
- **Downstream analyses are affected by bias**: The high-AUC large PFN exhibits significant bias after calibration, whereas the low-AUC small PFN remains unbiased (within $2\sigma$).

## Highlights & Insights
- **A concrete manifestation of "when a measure becomes a target" in physics ML**: Chasing AUC leaderboards can introduce systematic bias into analysis results.
- **The Pareto frontier as a model selection tool**: Provides a more comprehensive architectural evaluation framework than a single AUC metric.
- **The value of simple models**: IRC-safe EFN and physics-motivated Expert Features substantially outperform ParT in terms of robustness.

## Limitations & Future Work
- Resilience is measured solely via Pythia vs. Herwig; real systematic uncertainties are more complex.
- No detector simulation is included—detector effects also influence resilience in realistic analyses.
- Knowledge distillation experiments are relatively preliminary; more advanced strategies remain unexplored.

## Related Work & Insights
- **vs. ATLAS accuracy-vs-precision study**: ATLAS identified physics modeling uncertainties as dominant; this work quantifies them and proposes the Pareto framework.
- **vs. Butter et al. (2023)**: That work also investigates q/g resilience; this paper extends the analysis to top tagging and adds a downstream bias case study.

## Rating
- Novelty: ⭐⭐⭐⭐ The Pareto frontier perspective is original, and the downstream bias case study is convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic multi-architecture sweep + knowledge distillation + case study.
- Writing Quality: ⭐⭐⭐⭐⭐ Concise and compelling, with high information density.
- Value: ⭐⭐⭐⭐⭐ Directly informs model selection practices for the broader HEP-ML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Transfer Learning Beyond the Standard Model](transfer_learning_beyond_the_standard_model.md)
- [\[NeurIPS 2025\] Knowledge is Overrated: A Zero-Knowledge ML and Cryptographic Hashing-Based Framework for Verifiable, Low Latency Inference at the LHC](knowledge_is_overrated_a_zero-knowledge_machine_learning_and_cryptographic_hashi.md)
- [\[NeurIPS 2025\] From Simulations to Surveys: Domain Adaptation for Galaxy Observations](from_simulations_to_surveys_domain_adaptation_for_galaxy_observations.md)
- [\[NeurIPS 2025\] Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning](simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)
- [\[NeurIPS 2025\] One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs](oneshot_transfer_learning_nonlinear_pdes_perturbative_pinns.md)

</div>

<!-- RELATED:END -->
