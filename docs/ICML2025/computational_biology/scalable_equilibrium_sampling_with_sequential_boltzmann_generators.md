---
title: >-
  [Paper Note] Scalable Equilibrium Sampling with Sequential Boltzmann Generators
description: >-
  [ICML2025][Computational Biology][Boltzmann Generators] SBG achieves efficient equilibrium sampling of hexapeptide (66 atoms) systems in Cartesian coordinates for the first time by utilizing a Transformer-based normalizing flow (TarFlow) and sequential Monte Carlo with annealed Langevin dynamics.
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Boltzmann Generators"
  - "Normalizing Flows"
  - "Molecular Sampling"
  - "Annealed Langevin Dynamics"
  - "Protein Peptides"
date: 2026-05-08
content_hash: ab4af2383e99c291
---

# Scalable Equilibrium Sampling with Sequential Boltzmann Generators

**Conference**: ICML2025  
**arXiv**: [2502.18462](https://arxiv.org/abs/2502.18462)  
**Code**: [GitHub](https://github.com/transferable-samplers/transferable-samplers)  
**Area**: Computational Biology  
**Keywords**: Boltzmann Generators, Normalizing Flows, Molecular Sampling, Annealed Langevin Dynamics, Protein Peptides

## TL;DR
SBG achieves efficient equilibrium sampling of hexapeptide (66 atoms) systems in Cartesian coordinates for the first time by utilizing a Transformer-based normalizing flow (TarFlow) and sequential Monte Carlo with annealed Langevin dynamics.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: Molecular systems possess multiple metastable states, and energy barriers make transitions between states extremely slow. Traditional MCMC/MD requires extremely long simulations with femtosecond-scale timesteps.

### Bottlenecks of Existing Boltzmann Generators

1. Insufficient architectural expressiveness: Equivariant continuous flows are not efficient enough.
2. Poor overlap between proposal and target distributions: SNIS suffers from extremely large variance and very small ESS.
3. Prior state-of-the-art BG methods can only handle dipeptides (2 amino acids, 22 atoms).

### Biaxial Scaling

Pre-training improvement: A scalable non-equivariant architecture (TarFlow) replaces equivariant flows. Inference-time improvement: Annealed Langevin dynamics progressively transport samples.

## Method

### Key Designs

#### Key Design 1: Softly Equivariant Normalizing Flows
Rigid equivariance is discarded in favor of TarFlow (Vision Transformer-based patch-masked autoregressive flow).

Soft equivariance implementation:
- Rotational equivariance: Random rotation data augmentation during training.
- Translational equivariance: Center-of-mass noise + compensation via Proposition 1 during inference.

#### Key Design 2: Annealed Langevin Dynamics
Energy interpolation from proposal flow energy to target Boltzmann energy, leveraging the Jarzynski equality to track importance weights. This computation is more informative than starting from the prior.

### Theoretical Guarantees
Proposition 1 proves that ESS strictly improves after center-of-mass adjustment.

### Loss & Training
The model is trained end-to-end, with the optimization objective considering both task loss and regularization terms.


## Key Experimental Results

### Peptide System Sampling Capability


### Main Results

| System | Number of Atoms | SBG-SNIS | SBG-AIS | Continuous BG |
|------|-------|---------|---------|--------|
| Dipeptide | 22 | Excellent | Excellent | Feasible |
| Tripeptide | 33 | Good | Excellent | Failed |
| Tetrapeptide | 44 | Feasible | Good | Failed |
| Hexapeptide | 66 | - | Feasible | Failed |

### ESS Improvement


### Ablation Study

| System | SNIS ESS | AIS ESS | Gain |
|------|---------|---------|------|
| Tripeptide | ~0.3 | >0.8 | 2.7x |
| Tetrapeptide | ~0.1 | >0.5 | 5x |
| Hexapeptide | ~0 | Statistically significant | From infeasible to feasible |

### Key Findings
1. Annealing is key: A significant performance leap occurs after incorporating annealing.
2. Soft equivariance is effective: Flexible parameterization outperforms strict geometric constraints.
3. Transitioning from dipeptide to hexapeptide represents a qualitative leap.

## Highlights & Insights

1. Discarding rigid equivariance to use Transformers for scaling breakthroughs reflects the major trend in ML.
2. Inference-time annealing yields exponential improvements in sampling quality.
3. A perfect fusion of physics and ML: Boltzmann distribution, Jarzynski equality + TarFlow, SMC.
4. Center-of-mass adjustment is rigorously proven theoretically (Proposition 1).
5. Open-source code.

## Limitations & Future Work

1. Hexapeptide (66 atoms) is the limit; real proteins (thousands of atoms) remain far off.
2. Computational cost of annealing is significantly higher than SNIS.
3. Reliance on exact energy gradients makes black-box potential energies inapplicable.
4. Selection of center-of-mass noise parameters is not fully discussed.
5. No comparison was made against diffusion model sampling methods.

### Future Vision
- Combine TarFlow architecture with more efficient equivariant operations, leveraging the strengths of both.
- Explore multi-resolution annealing strategies to dynamically adjust step sizes.
- Validate on all-atom (non-coarse-grained) proteins.
- Integrate with structure prediction models like AlphaFold to leverage their priors.
- Complete validation on decapeptides and larger systems is the next milestone that must be conquered.
- Attempt to apply SBG to drug-protein binding free energy estimation.

## Related Work & Insights

- Noe et al. 2019: Original BG framework. NETS: flow matching + non-equilibrium sampling.
- TarFlow: This paper represents its first application to molecules.
- Insight: The philosophy that soft constraints outperform rigid constraints can be generalized to scientific ML.

## Rating
- Novelty: 4.5/5
- Experimental Thoroughness: 4.0/5
- Writing Quality: 4.0/5
- Value: 4.0/5

## Supplementary Analysis

### Method Comparison Summary Table

| Method | Uses Energy | Exact Likelihood | Uses Data | Annealing |
|------|---------|---------|---------|------|
| DEM | Yes | No | No | No |
| NETS | Yes | Yes | No | Yes |
| BG | Yes | Yes | Yes | No |
| SBG (Ours) | Yes | Yes | Yes | Yes |

SBG is the only method that simultaneously possesses all four key characteristics.

### Alanine dipeptide validation
In a 22-atom system, OM optimization generates two possible transition pathways. Using these pathways as collective variables for umbrella sampling, the free energy barrier is accurately estimated to be approximately 6 kcal/mol.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [\[ICML 2025\] Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching](scalable_generation_of_spatial_transcriptomics_from_histology_images_via_whole-s.md)
- [\[NeurIPS 2025\] Amortized Sampling with Transferable Normalizing Flows](../../NeurIPS2025/computational_biology/amortized_sampling_with_transferable_normalizing_flows.md)
- [\[ICML 2026\] RETROSPECT: RETROsynthesis via Sequential Prediction, and Chemically Transformed-ranking](../../ICML2026/computational_biology/retrospect_retrosynthesis_via_sequential_prediction_and_chemically_transformed-r.md)
- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/computational_biology/thompson_sampling_via_fine-tuning_of_llms.md)

</div>

<!-- RELATED:END -->
