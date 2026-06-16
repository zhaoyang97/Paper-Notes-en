---
title: >-
  [Paper Note] Dynamic Causal Discovery in Alzheimer's Disease through Latent Pseudotime Modelling
description: >-
  [NEURIPS2025][Medical Imaging][Causal discovery] This paper applies BN-LTE (Bayesian Network with Latent Time Embedding) to real-world ADNI data from AD patients to infer dynamic causal graphs that evolve along a disease…
tags:
  - "NEURIPS2025"
  - "Medical Imaging"
  - "Causal discovery"
  - "Alzheimer's disease"
  - "Bayesian networks"
  - "pseudotime"
  - "time-varying causal graphs"
  - "biomarkers"
date: 2026-05-08
content_hash: fabb66e6e6333608
---

# Dynamic Causal Discovery in Alzheimer's Disease through Latent Pseudotime Modelling

**Conference**: NEURIPS2025
**arXiv**: [2511.04619](https://arxiv.org/abs/2511.04619)  
**Code**: To be confirmed  
**Area**: Medical Imaging
**Keywords**: Causal discovery, Alzheimer's disease, Bayesian networks, pseudotime, time-varying causal graphs, biomarkers

## TL;DR
This paper applies BN-LTE (Bayesian Network with Latent Time Embedding) to real-world ADNI data from AD patients to infer dynamic causal graphs that evolve along a disease pseudotime axis. The learned pseudotime achieves a diagnostic AUC of 0.82, substantially outperforming chronological age (AUC 0.59), and reveals dynamic causal relationships between emerging biomarkers NfL/GFAP and established AD markers.

## Background & Motivation

**Background**: Approximately $380 billion is invested annually in Alzheimer's disease (AD) research, yet clinical trials continue to fail. A fundamental reason is that the causal relationships among the thousands of pathways involved in AD remain poorly understood. Causal inference provides a powerful framework for elucidating these relationships.

**Limitations of Prior Work**:
- Most causal discovery methods assume a static causal graph, whereas the pathophysiological processes of AD are dynamically evolving—causal relationships differ across disease stages.
- Core assumptions (acyclicity, no unobserved confounders) are frequently violated or untestable in medical data.
- Individual rates of disease progression vary due to latent factors such as cognitive reserve, meaning chronological age does not equal disease stage.
- The causal relationships between emerging plasma biomarkers NfL and GFAP and established AD markers (Aβ, pTau) remain unclear.

**Key Challenge**: Inter-individual variability in disease progression rates complicates time-series analysis—patients of the same age may be at entirely different disease stages. Cross-sectional data cannot directly capture disease dynamics.

**Goal**:
- Infer a data-driven "pseudotime" to order patients along their disease progression trajectory.
- Learn how causal relationships evolve as a function of pseudotime.
- Integrate the dynamic causal interactions between novel and established biomarkers.

**Key Insight**: The paper leverages the BN-LTE model (Zhou et al. 2023), treating pseudotime as a latent variable that modulates causal mechanisms, and infers dynamic causal graphs from cross-sectional ADNI data.

**Core Idea**: Order patients using a latent pseudotime and learn a causal graph that evolves with disease progression, thereby revealing dynamic causal relationships among AD biomarkers.

## Method

### Overall Architecture

- **Input**: Cross-sectional ADNI data from 380 patients (48 AD, 117 MCI, 215 CN), comprising 16 variables (demographics, regional brain volumes, plasma biomarkers, cognitive scores).
- **Model**: BN-LTE (Bayesian Network with Latent Time Embedding), with posterior inference via MCMC sampling.
- **Output**: (1) A disease pseudotime $Z$ for each patient; (2) A pseudotime-varying causal graph $G(Z)$.

### Key Designs

1. **Pseudotime Model**:

    - **Function**: Replace chronological age with a data-driven latent variable $Z$ to order patients.
    - **Mechanism**: The conditional distribution of each variable is modelled as $X_j = a_j(Z) + \sum_l b_{jl}(Z) X_l + \epsilon_j$, where $a_j(Z)$ is a baseline trajectory function (the natural progression of a marker along pseudotime) and $b_{jl}(Z)$ is a pseudotime-dependent causal effect coefficient; both are parameterised using cubic B-splines.
    - **Design Motivation**: Age $\neq$ disease stage—factors such as cognitive reserve cause considerable variation in progression among age-matched individuals. The identifiability of pseudotime $Z$ is theoretically guaranteed under the condition that causal relationships vary along this axis.

2. **Background Knowledge Constraints**:

    - **Function**: Incorporate minimal, disease-agnostic prior knowledge.
    - **Mechanism**: (1) *Root nodes*: immutable variables (sex, APOE genotype) cannot have incoming edges. (2) *Sink nodes*: cognitive scores cannot have outgoing edges (in the elderly ADNI cohort, reverse effects of cognition on other variables are negligible).
    - **Design Motivation**: In real-world data where model assumptions may be violated, disease-agnostic background knowledge substantially improves graph recovery (Table 2: directional precision increases from 62% to 96%), while avoiding the introduction of subjective biases regarding disease mechanisms.

3. **MCMC Posterior Inference**:

    - **Function**: Estimate the posterior distributions of pseudotime and the causal graph.
    - **Mechanism**: Four chains × 5,000 iterations (1,000 burn-in). Posterior inclusion probability (PIP) is used as a confidence measure for causal edges; the final causal graph is constructed by thresholding at PIP ≥ 0.5.
    - **Design Motivation**: The Bayesian approach naturally provides uncertainty quantification, and PIP avoids hard binary decisions about edge presence.

### Loss & Training

- Gaussian likelihood model: $\epsilon_j \sim \mathcal{N}(0, \sigma_j^2)$
- Cubic B-spline parameterisation with 5 knots
- The Coulomb prior used in the original BN-LTE is removed, as AD patients are not uniformly distributed across disease stages

## Key Experimental Results

### Main Results — Diagnostic Predictive Power: Pseudotime vs. Age

| Predictor | AUC | p-value | Note |
|-----------|-----|---------|------|
| Pseudotime $Z$ | **0.82** (95% CI: 0.81, 0.82) | <0.001 | Strong predictive power |
| Age | 0.59 | <0.01 | Weak predictive power |

### Ablation Study — Effect of Background Knowledge on Graph Recovery

| Configuration | Edge Presence Precision | Edge Presence Recall | Direction Precision | Direction Recall | SHD |
|---------------|------------------------|---------------------|--------------------|-----------------:|-----|
| No background knowledge | 0.80 | 0.16 | 0.62 | 0.50 | 67 |
| + Root node constraints | 0.72 | 0.35 | 0.89 | 0.84 | 53 |
| + Root + Sink nodes | **0.88** | **0.45** | **0.96** | **0.88** | **41** |

### Key Causal Findings

| Causal Edge | PIP (with background knowledge) | Literature Consistency |
|-------------|--------------------------------|----------------------|
| pTau217 → GFAP | 0.80 | Possible/Unknown |
| Aβ42 → Aβ40 | 0.75 | Confirmed |
| pTau217 → NfL | 0.57 | Possible |
| NfL → Hippocampus | 0.53 | Possible |
| Aβ42 → NfL | 0.46 | Possible |

### Key Findings

- **Pseudotime ordering is consistent with disease severity**: Figure 1 shows that CN patients cluster at early pseudotime, MCI patients at intermediate values, and AD patients at late pseudotime; biomarker trajectories including declining hippocampal volume and rising NfL and GFAP are consistent with known AD pathology.
- **Causal relationships change dynamically**: The influence of pTau on NfL emerges at early pseudotime—consistent with the consensus that pTau effects precede neurodegeneration—whereas the influence of age on GFAP remains constant throughout the disease course.
- **Background knowledge yields substantial gains**: Imposing only two disease-agnostic constraints—that sex/APOE are not influenced by other variables and that cognitive scores do not influence other variables—improves directional precision from 62% to 96%.
- **Inconsistencies are also identified**: The inferred edges pTau → GFAP and NfL → Aβ40 conflict with the literature, which holds that amyloid pathology precedes tau pathology, highlighting remaining limitations of the model and data.

## Highlights & Insights

- **Transfer of the pseudotime concept from single-cell biology to clinical disease modelling**: Pseudotime is widely used in single-cell RNA-seq for cell trajectory inference; this paper transfers the concept to patient-level disease progression modelling, elegantly addressing the inability of cross-sectional data to directly capture dynamics.
- **Outsized impact of disease-agnostic background knowledge**: Without any expert knowledge of AD mechanisms, simply encoding "immutable variables are root nodes" and "cognitive scores are sink nodes" raises directional precision from 62% to 96%—a finding with important implications for the practical application of causal discovery.
- **Clinical value of dynamic causal graphs**: The fact that causal relationships vary across disease stages implies that the timing of combination therapies may need to be tailored to a patient's disease stage—an insight with direct relevance to clinical trial design.
- **Causal positioning of novel biomarkers**: This paper provides the first causal-framework analysis of the dynamic interactions between NfL and GFAP and traditional AD biomarkers; the early emergence of the pTau → NfL edge offers a causal rationale for the clinical interpretation of these emerging markers.

## Limitations & Future Work

- **Strong assumptions**: The model assumes causal sufficiency (no unobserved confounders) and faithfulness, both of which are likely violated in medical data.
- **Limited sample size**: With only 380 patients, certain subgroups (e.g., 48 AD patients) have insufficient statistical power.
- **Uni-dimensional pseudotime**: Compressing disease progression into a one-dimensional scalar may be inadequate for the true heterogeneity of AD, which may require a multi-dimensional representation.
- **Consensus graph as ground truth**: The literature-derived consensus graph may itself be incomplete or contested, with the directionality of some edges unknown.
- **Longitudinal data not utilised**: ADNI contains longitudinal follow-up data that were not used; longitudinal analysis could validate the predictive validity of the pseudotime model.
- **Future directions**:
    - Relax the causal sufficiency assumption and model unobserved confounders (e.g., via FCI-based methods).
    - Extend to multi-dimensional pseudotime (multiple latent progression factors).
    - Cross-cohort validation (multi-dataset causal discovery).
    - Use longitudinal data to validate dynamic causal relationships.

## Related Work & Insights

- **vs. static causal graph methods**: Classical methods such as PC and GES produce a single fixed graph and cannot capture changes in causal relationships during disease progression. The dynamic graph of BN-LTE represents a qualitative advancement.
- **vs. Zhou et al. (2023)**: This paper constitutes the first application of BN-LTE to real AD data, contributing the discovery of the substantial value of disease-agnostic background knowledge and providing a causal analysis of NfL and GFAP.
- **vs. time-series causal discovery**: Methods such as Granger causality require longitudinal data; this paper infers dynamic relationships from cross-sectional data, making it applicable to a broader range of clinical settings.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic application of pseudotime combined with dynamic causal discovery in AD; the disease-agnostic background knowledge strategy has methodological value.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablations across multiple configurations, quantitative comparison against a consensus graph, and MCMC convergence diagnostics.
- Writing Quality: ⭐⭐⭐⭐ — Clinical motivation and methodological descriptions are clear; both findings and inconsistencies are discussed candidly.
- Value: ⭐⭐⭐⭐⭐ — Substantive contributions to both AD research and causal discovery methodology, with strong translational potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback](doctor_approved_generating_medically_accurate_skin_disease_images_through_ai-exp.md)
- [\[CVPR 2026\] EMAD: Evidence-Centric Grounded Multimodal Diagnosis for Alzheimer's Disease](../../CVPR2026/medical_imaging/emad_evidence-centric_grounded_multimodal_diagnosis_for_alzheimers_disease.md)
- [\[NeurIPS 2025\] Active Target Discovery under Uninformative Prior: The Power of Permanent and Transient Memory](active_target_discovery_under_uninformative_prior_the_power_of_permanent_and_tra.md)
- [\[NeurIPS 2025\] The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective](the_boundaries_of_fair_ai_in_medical_image_prognosis_a_causal_perspective.md)
- [\[NeurIPS 2025\] Online Feedback Efficient Active Target Discovery in Partially Observable Environments](online_feedback_efficient_active_target_discovery_in_partially_observable_enviro.md)

</div>

<!-- RELATED:END -->
