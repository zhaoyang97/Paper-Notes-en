---
title: >-
  [Paper Note] Is Sequence Information All You Need for Bayesian Optimization of Antibodies?
description: >-
  [NeurIPS 2025 (AI for Science Workshop)][Computational Biology][Bayesian Optimization] This paper systematically compares the roles of sequence and structural information in antibody Bayesian optimization…
tags:
  - "NeurIPS 2025 (AI for Science Workshop)"
  - "Computational Biology"
  - "Bayesian Optimization"
  - "Antibody Design"
  - "Protein Language Models"
  - "Structural Information"
  - "Gaussian Processes"
date: 2026-05-08
content_hash: 970521e238546923
---

# Is Sequence Information All You Need for Bayesian Optimization of Antibodies?

**Conference**: NeurIPS 2025 (AI for Science Workshop)  
**arXiv**: [2509.24933](https://arxiv.org/abs/2509.24933)  
**Code**: None  
**Area**: Bioinformatics / Antibody Engineering  
**Keywords**: Bayesian Optimization, Antibody Design, Protein Language Models, Structural Information, Gaussian Processes

## TL;DR

This paper systematically compares the roles of sequence and structural information in antibody Bayesian optimization, finding that sequence-only methods augmented with protein language model (pLM) soft constraints can match the performance of structure-based methods, thereby questioning the necessity of structural information in antibody Bayesian optimization.

## Background & Motivation

**Background**: Therapeutic antibodies constitute an important class of drugs whose development requires multi-round iterative optimization (e.g., binding affinity, thermostability). Bayesian optimization (BO) is well-suited to such high-cost, low-data settings due to its uncertainty-guided exploration.

**Limitations of Prior Work**: The success of BO depends critically on the choice of surrogate model and acquisition function, yet the role of structural information in antibody BO has not been systematically investigated. While structural diffusion models have proven useful in non-iterative design, the generated antibodies still require further optimization.

**Key Challenge**: For which antibody properties does structural information help? Is it intrinsic antibody properties (e.g., thermostability) or antigen-specific properties (e.g., binding affinity), particularly in the common scenario where the binding pose is unknown?

**Goal**: To systematically evaluate the effect of incorporating structural information in various ways, benchmarked against sequence-only approaches.

**Key Insight**: A pLM-based "soft constraint" mechanism is proposed to guide optimization toward promising regions of sequence space.

**Core Idea**: With appropriate sequence prior information (pLM soft constraints), sequence-only methods can match or eliminate the advantage of structure-based methods.

## Method

### Overall Architecture

The framework is based on Pareto-aware batch Bayesian optimization, using the qHSRI acquisition function combined with the NSGA-II genetic algorithm for discrete-space optimization. Approximately 80 candidate molecules are acquired per round, with a Gaussian process (GP) surrogate model.

### Key Designs

1. **Sequence-Based Methods**:

    - **OneHot-T**: GP with one-hot encoding and Tanimoto kernel, serving as a baseline.
    - **BLO-T**: GP with BLOSUM-62 matrix encoding and Tanimoto kernel; achieves the best performance in affinity optimization.
    - **ESM-M**: GP with mean-pooled embeddings from ESM-2 650M and Matérn-5/2 kernel.

2. **Structure-Based Methods**:

    - **IgFold-M**: Uses α-carbon coordinates from IgFold-predicted structures as input, capturing pure 3D geometric information.
    - **IgFold-ESM-M / IgFold-BLO-T**: Combines structural and sequence features via concatenation or weighted kernel summation.
    - **Kermut-T**: A composite kernel model integrating ProteinMPNN structural information with a sequence kernel:
    $k(\mathbf{x}, \mathbf{x}') = \pi k_{\text{struct}}(\mathbf{x}, \mathbf{x}') + (1-\pi) k_{\text{seq}}(\mathbf{x}, \mathbf{x}')$

3. **Antibody-Specific Refinement (AbMPNN-Kermut-T)**: Replaces the general-purpose ProteinMPNN with antibody-specific AbMPNN, improving early-iteration performance in affinity optimization.

4. **pLM Soft Constraints (Core Contribution)**: Inspired by constrained BO, pLM pseudo-likelihood is used as a soft constraint on the acquisition function:
    $a_{\text{pLM}}(\mathbf{x}) = p_{\text{pLM}}(\mathbf{x}) \cdot a(\mathbf{x})$
   where $p_{\text{pLM}}(\cdot)$ is derived from Sapiens, a lightweight antibody-specific language model. This prevents BO from exploring "unnatural" mutations that lead to expression failure.

### Loss & Training

- An initial dataset of 50 samples is taken from early stages of real optimization campaigns.
- 80 molecules are acquired per round, with 30 randomly discarded to simulate experimental failure.
- A total of 9 acquisition rounds are performed, with each experiment repeated 3 times.

## Key Experimental Results

### Main Results

Experiments are conducted using an internal oracle trained on real optimization campaign data, optimizing dissociation constant $K_D$ (binding affinity) and melting temperature $T_m$ (thermostability).

| Method | Affinity ($K_D$) | Thermostability ($T_m$) |
|--------|------------------|------------------------|
| OneHot-T | Excellent | Moderate, eventually catches up |
| BLO-T | **Best** (among sequence methods) | Comparable to OneHot-T |
| ESM-M | Underperforms Tanimoto-kernel methods | Strong early, eventually on par |
| IgFold-M | Strong in early iterations | Moderate |
| Kermut-T | Worst affinity | **Best** (among structure methods) |
| AbMPNN-Kermut-T | Notable gain over Kermut-T | Comparable to Kermut-T |

### Effect of Soft Constraints

| Method | Affinity Change | Thermostability Change |
|--------|-----------------|----------------------|
| C-OneHot-T vs. OneHot-T | Marginal | **Matches structure-based methods** |
| C-BLO-T vs. BLO-T | Occasionally negative | Improved |
| C-Kermut-T vs. Kermut-T | No significant change | No significant change |

### Ablation Study

| Ablation | Affinity | Thermostability |
|----------|----------|-----------------|
| Kermut (original) → Kermut-M (numerical precision improvement) | Improved | Improved |
| Kermut-M → Kermut-T (Tanimoto kernel replacing ESM kernel) | On par or slightly better | On par |
| ProteinMPNN → AbMPNN | Improved | No change |
| ESM-2 prior mean → constant mean | No change | Decreased |

### Key Findings

1. **Affinity optimization**: Sequence-only methods (BLO-T) achieve the best asymptotic performance; structural information only provides data efficiency advantages in early iterations.
2. **Thermostability optimization**: Kermut-T (structure-based) performs best, but with pLM soft constraints, C-OneHot-T (sequence-only) can match its performance.
3. **No single method achieves simultaneous optimality on both properties**, suggesting that different properties require different feature representations.
4. The early-iteration advantage of IgFold-M stems from its tendency to remain close to the parent structure, as confirmed by RMSD analysis.

## Highlights & Insights

- The **pLM soft constraint** is an elegant and concise design that steers BO away from infeasible regions with a single formula modification.
- The paper provides a thorough analysis of the fundamental distinction between "pure structural" information (IgFold coordinates) and "statistical structural" information (ProteinMPNN probabilities).
- The conclusions have direct practical value: in the common scenario where the binding pose is unknown, sequence-only methods with pLM constraints are sufficient.
- BLOSUM encoding unexpectedly outperforms ESM-2 embeddings, demonstrating that domain-specific simple features can sometimes surpass large general-purpose models.

## Limitations & Future Work

1. Antibody–antigen complex structure (binding pose) is not considered, which may improve structure-based methods for affinity optimization.
2. The integration of structural information is relatively simple; more sophisticated fusion strategies warrant investigation.
3. Experiments rely on in silico oracles and lack in vitro validation.
4. Only affinity and thermostability are evaluated; other developability properties remain to be assessed.

## Related Work & Insights

- **LaMBO / LaMBO-2**: Sequence-based BO methods operating in VAE latent space; found to underperform with small initial datasets.
- **Kermut**: A GP method combining ProteinMPNN structural information with sequence kernels; this paper introduces several improvements.
- **GAUCHE**: A BO toolkit for proteins and chemical molecules, providing foundational methods such as the Tanimoto kernel.
- Insight: In protein optimization, simple but domain-adapted methods (BLOSUM encoding) may be more effective than complex general-purpose approaches.

## Rating

- Novelty: ⭐⭐⭐⭐ The pLM soft constraint idea is concise and effective; the systematic comparison is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive baselines, though limited to in silico oracles.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, rigorous logic, and well-defined research questions.
- Value: ⭐⭐⭐⭐ Directly applicable to antibody engineering practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[ICML 2026\] Neural Estimation of Pairwise Mutual Information in Masked Discrete Sequence Models](../../ICML2026/computational_biology/neural_estimation_of_pairwise_mutual_information_in_masked_discrete_sequence_mod.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[NeurIPS 2025\] CrossNovo: Bidirectional Representations Augmented Autoregressive Biological Sequence Generation](bidirectional_representations_augmented_autoregressive_biological_sequence_gener.md)
- [\[NeurIPS 2025\] DesignX: Human-Competitive Algorithm Designer for Black-Box Optimization](designx_human-competitive_algorithm_designer_for_black-box_optimization.md)

</div>

<!-- RELATED:END -->
