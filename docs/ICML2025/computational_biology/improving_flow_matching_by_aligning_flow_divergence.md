---
title: >-
  [Paper Note] Improving Flow Matching by Aligning Flow Divergence
description: >-
  [ICML 2025][Computational Biology][Flow Matching] Analyses the error between the learned probability path and the true probability path in Flow Matching (FM) from a PDE perspective. It proves that this error is controlled by the divergence gap of the vector fields, and proposes a joint flow and divergence matching (FDM) training objective, which significantly improves FM performance on density estimation, DNA sequence generation, and video prediction tasks.
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Flow Matching"
  - "divergence matching"
  - "probability path error"
  - "Total Variation"
  - "conditional divergence loss"
date: 2026-05-08
content_hash: ab31eb6bc9367b08
---

# Improving Flow Matching by Aligning Flow Divergence

**Conference**: ICML 2025  
**arXiv**: [2602.00869](https://arxiv.org/abs/2602.00869)  
**Code**: [https://github.com/Utah-Math-Data-Science](https://github.com/Utah-Math-Data-Science)  
**Area**: Computational Biology  
**Keywords**: Flow Matching, divergence matching, probability path error, Total Variation, conditional divergence loss

## TL;DR
Analyses the error between the learned probability path and the true probability path in Flow Matching (FM) from a PDE perspective. It proves that this error is controlled by the divergence gap of the vector fields, and proposes a joint flow and divergence matching (FDM) training objective, which significantly improves FM performance on density estimation, DNA sequence generation, and video prediction tasks.

## Background & Motivation
**Background**: Conditional Flow Matching (CFM) is an efficient method for training flow-based generative models by regressing conditional vector fields to learn the mapping from noise to data without the need for simulation.

**Limitations of Prior Work**: CFM only ensures that the learned vector field $\boldsymbol{v}_t$ is close to the true vector field $\boldsymbol{u}_t$. However, the divergence gap between them, $|\nabla \cdot \boldsymbol{v}_t - \nabla \cdot \boldsymbol{u}_t|$, can remain large, leading to significant deviations between the learned and true probability paths.

**Key Challenge**: The CFM loss is equivalent to the FM loss plus a constant. Minimizing it effectively trains the vector field itself but cannot guarantee the accuracy of the probability path (density function) because the vector field's divergence determines the dynamics of the density.

**Goal**: How to simultaneously control the accuracy of the vector field and its divergence during FM training to obtain more accurate probability paths?

**Key Insight**: Starting from the continuity equation, a PDE governing the error between the exact and learned probability paths is derived. Solving this PDE using Duhamel's principle yields an upper bound on the TV distance of the error.

**Core Idea**: The probability path error in FM is jointly determined by both the vector field difference and the divergence difference. Consequently, FDM (CFM loss + conditional divergence loss) is proposed to optimize both simultaneously.

## Method

### Overall Architecture
FDM adds a conditional divergence matching loss $\mathcal{L}_{\text{CDM}}$ on top of the standard CFM loss to construct a weighted objective:
$$\mathcal{L}_{\text{FDM}} = \lambda_1 \mathcal{L}_{\text{CFM}} + \lambda_2 \mathcal{L}_{\text{CDM}}$$

### Key Designs

1. **Probability Path Error PDE (Proposition 3.1)**:

    - Function: Characterises the error $\epsilon_t = p_t - \hat{p}_t$ between the true path $p_t$ and the learned path $\hat{p}_t$.
    - Mechanism: The error satisfies $\partial_t \epsilon_t + \nabla \cdot (\epsilon_t \boldsymbol{v}_t) = L_t$, where the forcing term is $L_t = -p_t[\nabla \cdot (\boldsymbol{u}_t - \boldsymbol{v}_t) + (\boldsymbol{u}_t - \boldsymbol{v}_t) \cdot \nabla \log p_t]$.
    - Design Motivation: The forcing term includes both the vector field difference and the divergence difference, showing that matching only the vector field is insufficient.

2. **TV Distance Upper Bound (Theorem 3.3)**:

    - Function: Quantifies the probability path error into an optimizable target.
    - Mechanism: $\text{TV}(p_t, \hat{p}_t) \leq \frac{1}{2}\mathcal{L}_{\text{DM}}$, where $\mathcal{L}_{\text{DM}} = \mathbb{E}_{t, p_t}[|\nabla \cdot (\boldsymbol{u}_t - \boldsymbol{v}_t) + (\boldsymbol{u}_t - \boldsymbol{v}_t) \cdot \nabla \log p_t|]$.
    - Design Motivation: Establishes a theoretical bridge between the optimizable loss and distribution accuracy.

3. **Conditional Divergence Matching (Theorem 4.1 → FDM)**:

    - Function: Because $\mathcal{L}_{\text{DM}}$ is not directly computable (depending on the marginal vector field), its conditional version $\mathcal{L}_{\text{CDM}}$ is derived as an upper bound.
    - Mechanism: Employing a conditioning trick similar to CFM, the unconditional divergence difference is structurally replaced with the conditional divergence difference. This yields an efficiently computable $\mathcal{L}_{\text{CDM}}$, which is further optimized using the Hutchinson trace estimator.
    - Design Motivation: Minimizing $\mathcal{L}_{\text{CDM}}$ alone cannot guarantee good outcomes due to cancellations of positive and negative terms, necessitating joint optimization with $\mathcal{L}_{\text{CFM}}$.

### Loss & Training
- $\mathcal{L}_{\text{FDM}} = \lambda_1 \mathcal{L}_{\text{CFM}} + \lambda_2 \mathcal{L}_{\text{CDM}}$
- The efficient version $\mathcal{L}_{\text{CDM-2}}^{\text{eff}}$ uses stop-gradient + Hutchinson trace estimation, requiring only one additional backpropagation pass.
- Hyperparameters $\lambda_1, \lambda_2$ are selected via search.

## Key Experimental Results

### Main Results

| Task | Model | FM Metric | FDM Metric | Gain |
|------|------|---------|----------|------|
| Checkerboard Density Estimation (OT) | Likelihood ↑ | 2.38×10⁻² | **2.53×10⁻²** | +6.3% |
| CIFAR-10 (OT) | NLL ↓ | 2.99 | **2.85** | -4.7% |
| CIFAR-10 (OT) | FID ↓ | 6.35 | **5.62** | -11.5% |
| KTH Video Prediction | FVD ↓ | 180 | **155.5** | -13.6% |
| BAIR Video Prediction | FVD ↓ | 146 | **123** | -15.8% |

### Ablation Study

| Dataset | Metric | FM (OT) | FDM (OT) | FM (VP) | FDM (VP) |
|--------|------|---------|----------|---------|----------|
| Lorenz Trajectories p(x₁) | TV ↓ | 0.0348 | **0.0306** | - | - |
| FitzHugh p(x₁) | TV ↓ | 0.0314 | **0.0266** | - | - |
| DNA Sequence | MSE ↓ | 2.82E-2 | **2.78E-2** | - | - |
| DNA Dirichlet | MSE ↓ | 2.68E-2 | **2.59E-2** | - | - |

### Key Findings
- FDM outperforms FM across all path types (OT, VP, VE, Dirichlet).
- The impact of the divergence gap is most significant in exact likelihood estimation tasks (e.g., yielding pronounced improvements in NLL).
- The additional computational overhead is only around 50% (one extra backpropagation pass), offering high cost-effectiveness.

## Highlights & Insights
- **Theory-Driven Method Design**: The loss function is derived from first-principles PDE error analysis, rather than being heuristically designed.
- **Elegant Conditioning Trick**: Converts the intractable marginal divergence difference into a trainable loss using conditioning and Jensen's inequality.
- **Broad Applicability**: Applicable to various probability paths such as OT, VP, VE, and Dirichlet, without being restricted to image generation.

## Limitations & Future Work
- Bounded TV distance is not equivalent to bounded KL divergence; the authors acknowledge that controlling KL divergence remains an open challenge.
- The choice of hyperparameters $\lambda_1, \lambda_2$ lacks a principled approach and requires manual search.
- Large-scale image generation (e.g., ImageNet 256) experiments are missing, with validation restricted to CIFAR-10 and smaller datasets.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Establishes a theoretical framework for FM probability path errors from a PDE perspective for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Wide coverage of synthetic and real-world tasks, but lacks large-scale visual generation experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theory and fluent presentation.
- Value: ⭐⭐⭐⭐ Provides significant advancements to the foundational theory of FM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Flexibility-conditioned Protein Structure Design with Flow Matching](flexibility-conditioned_protein_structure_design_with_flow_matching.md)
- [\[ICML 2025\] Efficient Molecular Conformer Generation with SO(3)-Averaged Flow Matching and Reflow](efficient_molecular_conformer_generation_with_so3-averaged_flow_matching_and_ref.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](../../NeurIPS2025/computational_biology/energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[ICML 2025\] Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching](scalable_generation_of_spatial_transcriptomics_from_histology_images_via_whole-s.md)
- [\[NeurIPS 2025\] Curly Flow Matching for Learning Non-gradient Field Dynamics](../../NeurIPS2025/computational_biology/curly_flow_matching_for_learning_non-gradient_field_dynamics.md)

</div>

<!-- RELATED:END -->
