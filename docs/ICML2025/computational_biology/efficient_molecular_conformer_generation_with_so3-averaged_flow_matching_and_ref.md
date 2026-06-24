---
title: >-
  [Paper Note] Efficient Molecular Conformer Generation with SO(3)-Averaged Flow Matching and Reflow
description: >-
  [ICML2025][Computational Biology][Molecular Conformer Generation] Proposes an SO(3)-Averaged Flow training objective to eliminate the need for rotation alignment between the prior and data distributions by analytically averaging over all rotations in the rotation group SO(3). Combined with Reflow and distillation, it achieves high-quality few-step or even single-step molecular conformer generation.
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Molecular Conformer Generation"
  - "Flow Matching"
  - "SO(3) Symmetry"
  - "Reflow"
  - "Distillation"
  - "Drug Discovery"
date: 2026-05-08
content_hash: f6936086732d9cfe
---

# Efficient Molecular Conformer Generation with SO(3)-Averaged Flow Matching and Reflow

**Conference**: ICML2025  
**arXiv**: [2507.09785](https://arxiv.org/abs/2507.09785)  
**Code**: Not open-sourced  
**Area**: Molecular Conformer Generation / Computational Chemistry  
**Keywords**: Molecular Conformer Generation, Flow Matching, SO(3) Symmetry, Reflow, Distillation, Drug Discovery

## TL;DR

Proposes an SO(3)-Averaged Flow training objective to eliminate the need for rotation alignment between the prior and data distributions by analytically averaging over all rotations in the rotation group SO(3). Combined with Reflow and distillation, it achieves high-quality few-step or even single-step molecular conformer generation.

## Background & Motivation

Molecular Conformer Generation is the task of predicting a set of 3D conformers given a 2D molecular graph, which is fundamental to computational chemistry and drug discovery. Existing methods face a **trade-off between generation quality and speed**:

- **Semi-empirical quantum chemistry methods** (e.g., CREST): High quality but extremely slow, requiring extensive energy function evaluations.
- **Cheminformatics tools** (e.g., RDKit, OMEGA): Fast but limited in diversity and quality.
- **Diffusion/Flow Matching models** (e.g., MCF, ET-Flow): Good quality but require hundreds of ODE/SDE solver steps, making them difficult to scale to billion-scale virtual screening.

The core pain points are: (1) During flow matching training, there exists a **rotational freedom** between the prior distribution (Gaussian noise) and the data distribution. Existing methods either use random rotation (Conditional OT) or perform Kabsch alignment, neither of which is optimal. (2) Sampling requires a large number of ODE steps, leading to high computational overhead.

## Method

### 3.1 SO(3)-Averaged Flow

Core idea: Molecular conformers possess SO(3) rotational symmetry, meaning $q(x) = q(Rx)$ holds for any rotation matrix $R$. Instead of choosing a specific rotation for alignment, this paper **analytically averages over all rotations** to compute the expected probability flow path.

Given atomic coordinates $x \in \mathbb{R}^{N \times 3}$, the conditional probability path is a Gaussian distribution:

$$p_t(x | x_1) \propto \exp\left(-\frac{1}{2} \frac{\|x - tx_1\|_\Sigma^2}{(1-t)^2}\right)$$

After integrating over the SO(3) group, the average vector field is:

$$u_t(x) = \frac{1}{Z_t(x,0)} \sum_{\hat{x} \in \mathcal{X}} \hat{q}(\hat{x}) \int_{SO(3)} dR \frac{\hat{x}R^T - x}{1-t} \exp\left(-\frac{1}{2}\frac{\|x - t\hat{x}R^T\|_\Sigma^2}{(1-t)^2}\right)$$

The key lies in using the closed-form solution of Mohlin et al. (2020) to compute the integral over SO(3), avoiding Monte Carlo sampling. The final training loss is:

$$\mathcal{L}_{\text{AvgFlow}}(\theta) = \mathbb{E}\left[\|v_t^\theta(x_t) - u_t(x_t)\|^2\right], \quad t \in [0,1]$$

**Interpolation schemes**: For equivariant networks, linear interpolation $x_t = t \cdot x_0 + (1-t) \cdot x_1$ is used. For non-equivariant networks, the integration interpolant must be computed via ODE integration.

### 3.2 Reflow + Distillation

To accelerate sampling, a three-stage training strategy is adopted:

1. **Base Training**: Train the model using the Averaged Flow objective.
2. **Reflow**: Generate pairs $(x_0', x_1')$ from noise $x_0'$ and fine-tune with the rectified flow loss to straighten the trajectories:
   $$\mathcal{L}_{\text{Reflow}}(\theta) = \mathbb{E}\left[\|v_t^\theta(x_t', t) - (x_1' - x_0')\|^2\right]$$
   where the time $t$ is sampled from an exponential distribution $p(t) \propto \exp(\lambda t), \lambda = -1.2$, focusing on high-curvature regions ($t < 0.5$).
3. **Distillation**: Fix $t=0$ to train the model to learn the single-step mapping:
   $$\mathcal{L}_{\text{Distill}}(\theta) = \mathbb{E}\left[\|v_t^\theta(x_0', 0) - (x_1' - x_0')\|^2\right]$$

### 3.3 Model Architecture

The method is architecture-agnostic. The paper evaluates two architectures:

- **NequIP** (~4.7M parameters): An SE(3)-equivariant Graph Neural Network with 6 interaction blocks.
- **DiT** (~52M parameters): A non-equivariant Diffusion Transformer that injects pairwise distance and bond features as attention biases (inspired by AlphaFold3).
- **DiT-L** (~64M parameters): A scaled-up version of DiT, matching the parameter size of MCF-B.

## Key Experimental Results

Datasets: GEOM-QM9 (small molecules) and GEOM-Drugs (drug-like molecules), each containing 1,000 test molecules.

### GEOM-QM9 Benchmark ($\delta=0.5$Å)

| Model | Steps | COV-R↑ | AMR-R↓ | COV-P↑ | AMR-P↓ |
|------|------|--------|--------|--------|--------|
| RDKit | - | 85.1 | 0.235 | 86.8 | 0.232 |
| Tor. Diff. | 20 | 92.8 | 0.178 | 92.7 | 0.221 |
| MCF-B (64M) | 1000 | 95.0 | 0.103 | 93.7 | 0.119 |
| ET-Flow-SS (8.3M) | 50 | 95.0 | 0.083 | 91.0 | 0.116 |
| **AvgFlow-DiT (52M)** | **100** | **96.0** | **0.082** | **95.0** | **0.088** |
| AvgFlow-NequIP-R | 2 | 95.9 | 0.151 | 87.7 | 0.236 |
| AvgFlow-NequIP-D | 1 | 95.1 | 0.220 | 84.8 | 0.304 |

### GEOM-Drugs Benchmark ($\delta=0.75$Å)

| Model | Steps | COV-R↑ | AMR-R↓ | COV-P↑ | AMR-P↓ |
|------|------|--------|--------|--------|--------|
| RDKit | - | 38.4 | 1.058 | 40.9 | 0.995 |
| Tor. Diff. | 20 | 72.7 | 0.582 | 55.2 | 0.778 |
| MCF-B (64M) | 1000 | 84.0 | 0.427 | 64.0 | 0.667 |
| MCF-L (242M) | 1000 | 84.7 | 0.390 | 66.8 | 0.618 |
| ET-Flow-SS (8.3M) | 50 | 79.6 | 0.439 | 75.2 | 0.517 |
| **AvgFlow-DiT (52M)** | **100** | **82.0** | **0.428** | **72.9** | **0.566** |
| **AvgFlow-DiT-L (64M)** | **100** | **82.0** | **0.409** | **75.7** | **0.516** |
| AvgFlow-DiT-R (52M) | 2 | 75.7 | 0.545 | 57.2 | 0.748 |
| **AvgFlow-DiT-D (52M)** | **1** | **76.8** | **0.548** | **61.0** | **0.720** |
| MCF-L (242M) | 1 | 27.2 | 0.932 | 8.9 | 1.511 |
| ET-Flow (8.3M) | 1 | 27.6 | 0.996 | 25.7 | 0.939 |

**Key Findings**:

- AvgFlow-DiT achieves comprehensive state-of-the-art (SOTA) results across all four metrics on QM9.
- **Single-step AvgFlow-DiT-D (COV-R 76.8%) substantially outperforms single-step MCF-L (27.2%) and ET-Flow (27.6%)**.
- Single-step generation results even exceed those of 20-step Torsional Diffusion, outperforming full simulation accuracy metrics of MCF-S (1000 steps).
- Averaged Flow allows DiT to surpass Kabsch-OT trained for 100 epochs within only 12 epochs.
- NequIP-R (2 steps) is 21-50x faster in sampling than MCF (3 steps) and 48x faster than Tor. Diff. (5 steps).
- AvgFlow-DiT-L (64M) outperforms all MCF variants in precision metrics while being more parameter-efficient.

## Highlights & Insights

1. **Theoretical elegance**: Uses closed-form solutions for integrals over the SO(3) group to avoid Monte Carlo rotation sampling or heuristic alignments, providing an optimal solution for handling rotational symmetries.
2. **Architecture agnostic**: Averaged Flow can be directly applied to both equivariant and non-equivariant architectures, demonstrating broad applicability.
3. **Significant training acceleration**: Especially for non-equivariant DiT, convergence is speeded up by approximately 8x (12 epochs vs. 100 epochs).
4. **Single-step generation breakthrough**: Through Reflow + distillation, high-quality single-step molecular conformer generation is realized for the first time, offering practical value for large-scale virtual screening.
5. **Three-stage training pipeline**: The design (AvgFlow $\rightarrow$ Reflow $\rightarrow$ Distill) is highly structured; each stage is decoupled and can yield benefits independently.

## Limitations & Future Work

1. **Limited dataset scale**: Validated only on GEOM-QM9/Drugs without testing on larger scale or more complex molecules (proteins, macrocycles, etc.).
2. **Reflow data generation overhead**: Requires generating a large amount of paired data beforehand using the base model, increasing the overall training cost.
3. **Integration interpolant required for non-equivariant architecture**: Adds an extra computation of 20 Euler integration steps during DiT training.
4. **Slight quality degradation post-Reflow/distillation**: Single-step generation compared to full simulation on Drugs drops in COV-R from 82.0% to 76.8% (AMR-P increases from 0.566 to 0.720).
5. **No comparison with latest Consistency Models**, nor exploration of more advanced distillation strategies.
6. **Lack of energy evaluation**: The energy distribution of generated conformers is not reported, which is highly important for drug discovery applications.

## Related Work & Insights

- **Torsional Diffusion** (Jing et al., 2022): Restricts degrees of freedom to torsion angles; lightweight but relies on RDKit initial conformers.
- **MCF** (Wang et al., 2024): Large-scale Transformer + DDPM on Cartesian coordinates; SOTA but slow in inference.
- **ET-Flow** (Hassan et al., 2024): Equivariant Transformer + Flow Matching + Kabsch alignment.
- **Rectified Flow** (Liu et al., 2022): Theoretical foundation for the Reflow + distillation framework.
- Shares the pairwise bias attention design philosophy with AlphaFold3.

## Rating
- Novelty: ⭐⭐⭐⭐ (The idea of analytical averaging over the SO(3) group is novel and elegant)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Two standard datasets, two architectures, comprehensive ablation, but lacks energy evaluation)
- Writing Quality: ⭐⭐⭐⭐ (Mathematical derivations are clear and figures/tables are intuitive)
- Value: ⭐⭐⭐⭐ (Single-step generation is highly practical for industrial-scale virtual screening)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching](scalable_generation_of_spatial_transcriptomics_from_histology_images_via_whole-s.md)
- [\[ICML 2025\] Improving Flow Matching by Aligning Flow Divergence](improving_flow_matching_by_aligning_flow_divergence.md)
- [\[ICLR 2026\] Hierarchical Multi-Scale Molecular Conformer Generation](../../ICLR2026/computational_biology/hierarchical_multi-scale_molecular_conformer_generation.md)
- [\[ICML 2025\] Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning](neural_graph_matching_improves_retrieval_augmented_generation_in_molecular_machi.md)
- [\[ICML 2025\] Flexibility-conditioned Protein Structure Design with Flow Matching](flexibility-conditioned_protein_structure_design_with_flow_matching.md)

</div>

<!-- RELATED:END -->
