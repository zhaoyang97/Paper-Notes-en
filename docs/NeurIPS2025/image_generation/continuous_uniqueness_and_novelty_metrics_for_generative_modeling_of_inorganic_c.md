---
title: >-
  [Paper Note] Continuous Uniqueness and Novelty Metrics for Generative Modeling of Inorganic Crystals
description: >-
  [NeurIPS 2025 (AI4Mat Workshop)][Image Generation][crystal generation] This paper identifies four critical flaws in the widely adopted discrete distance function (StructureMatcher) used to evaluate inorganic crystal gene…
tags:
  - "NeurIPS 2025 (AI4Mat Workshop)"
  - "Image Generation"
  - "crystal generation"
  - "uniqueness"
  - "novelty"
  - "distance function"
  - "AMD"
  - "Magpie"
  - "StructureMatcher"
date: 2026-05-08
content_hash: 0271c37391b93234
---

# Continuous Uniqueness and Novelty Metrics for Generative Modeling of Inorganic Crystals

**Conference**: NeurIPS 2025 (AI4Mat Workshop)
**arXiv**: [2510.12405](https://arxiv.org/abs/2510.12405)  
**Code**: [GitHub](https://github.com/WMD-group/xtalmet)  
**Area**: Generative Models / Materials Science / Evaluation Metrics
**Keywords**: crystal generation, uniqueness, novelty, distance function, AMD, Magpie, StructureMatcher

## TL;DR
This paper identifies four critical flaws in the widely adopted discrete distance function (StructureMatcher) used to evaluate inorganic crystal generative models, and proposes continuous distance functions based on Magpie fingerprints (composition) and AMD vectors (structure) to achieve more reliable uniqueness and novelty metrics.

## Background & Motivation

**Background**: Machine learning generative models (CDVAE, DiffCSP, MatterGen, etc.) enable rapid sampling of novel crystal structures from chemical space. The core evaluation metrics for these models are uniqueness (diversity among generated samples) and novelty (deviation from training data), both of which rely on a distance function defined between crystal structures.

**Limitations of Prior Work**: The most commonly used distance function $d_{\text{smat}}$ (pymatgen's StructureMatcher) suffers from four key problems:
   - **(a) Discreteness**: Returns only 0/1, precluding quantification of structural similarity—whereas physical properties vary continuously with crystal structure.
   - **(b) Conflation of composition and structure**: A nonzero distance does not distinguish whether the discrepancy arises from differing compositions or differing structures.
   - **(c) Lack of Lipschitz continuity**: Small perturbations to atomic coordinates can cause discontinuous changes in the primitive cell, leading to abrupt jumps in the distance value.
   - **(d) Non-permutation-invariance**: The uniqueness score depends on the generation order—the same set of samples may yield different scores under different orderings.

**Key Challenge**: The unreliability of current evaluation metrics may lead to erroneous assessments of generative model capability. For example, a model that generates large numbers of physically unreasonable yet compositionally diverse structures may receive high scores under discrete metrics.

## Method

### Two Proposed Continuous Distance Functions

**1. Composition distance $d_{\text{magpie}}$**: Euclidean distance based on Magpie fingerprints
- The Magpie fingerprint comprises 145 attributes (stoichiometric attributes and statistical summaries of elemental properties).
- For two crystals $x, x'$: $d_{\text{magpie}}(x, x') = \|\text{Magpie}(x) - \text{Magpie}(x')\|_2$

**2. Structure distance $d_{\text{amd}}$**: $L_\infty$ distance based on Average Minimum Distance (AMD) vectors
- The AMD vector is a structural fingerprint: $\text{AMD}[k]$ denotes the average distance from each atom to its $k$-th nearest neighbor, averaged over all atoms in the unit cell.
- $d_{\text{amd}}(x, x') = \|\text{AMD}(x) - \text{AMD}(x')\|_\infty$

### Continuous Uniqueness and Novelty Definitions

$$\text{continuous uniqueness} = \frac{1}{\binom{n}{2}} \sum_{i=1}^{n} \sum_{j=1}^{i-1} d_{\text{continuous}}(x_i, x_j)$$

$$\text{continuous novelty} = \frac{1}{n} \sum_{i=1}^{n} \min_{j=1 \sim m} d_{\text{continuous}}(x_i, y_j)$$

Unlike their discrete counterparts (which aggregate indicator functions), the continuous versions yield genuine distance-based measures of inter-sample diversity and training-set deviation.

### Theoretical Advantages

Both distance functions satisfy two key properties for robust evaluation:
- **Isometry invariance**: For isometric crystals $x \cong x'$, $d(x, x') = 0$.
- **Lipschitz continuity**: If $x'$ is obtained from $x$ by displacing each atom by at most $\varepsilon$, then $d(x, x') \leq C\varepsilon$.

$d_{\text{smat}}$ does not satisfy Lipschitz continuity (due to its reliance on primitive cell comparison, which changes discontinuously with atomic coordinates); $d_{\text{wyckoff}}$ (another discrete structural distance) satisfies neither property.

### Permutation Invariance

The uniqueness score of $d_{\text{smat}}$ is not permutation-invariant because it violates the triangle inequality. For example, given three structures $x, x', x''$ where $d_{\text{smat}}(x,x')=d_{\text{smat}}(x,x'')=0$ but $d_{\text{smat}}(x',x'')=1$, different generation orderings yield uniqueness scores of either 1/3 or 2/3. Continuous distance functions are inherently permutation-invariant.

## Key Experimental Results

### Main Results: Evaluation of 6 Generative Models on the MP20 Dataset (10k Samples)

| Metric | CDVAE | DiffCSP | DiffCSP++ | MatterGen | Chemeleon | ADiT |
|--------|-------|---------|-----------|-----------|-----------|------|
| U ($d_{\text{smat}}$) | 0.995 | 0.977 | 0.981 | 0.984 | 0.979 | 0.884 |
| U ($d_{\text{comp}}$) | **0.972** | 0.946 | 0.952 | 0.952 | 0.937 | 0.774 |
| U ($d_{\text{magpie}}$, ×10⁻³) | 1.795 | 1.982 | 2.070 | 2.089 | 2.084 | 2.074 |
| U ($d_{\text{amd}}$) | 1.207 | 1.591 | 1.377 | 1.415 | **2.679** | 1.273 |

**After thermodynamic stability filtering** ($E_{\text{hull}} \leq 0.1$ eV/atom):

| Metric | CDVAE | DiffCSP | DiffCSP++ | MatterGen | Chemeleon | ADiT |
|--------|-------|---------|-----------|-----------|-----------|------|
| U ($d_{\text{smat}}$) | 0.035 | 0.289 | 0.272 | 0.352 | **0.375** | 0.316 |
| U ($d_{\text{magpie}}$, ×10⁻³) | 0.002 | 0.177 | 0.160 | 0.253 | **0.298** | - |

### Key Findings

1. **$d_{\text{smat}}$ primarily reflects compositional differences**: $d_{\text{smat}}$ and $d_{\text{comp}}$ are highly correlated—nonzero distances predominantly arise from differing compositions rather than differing structures.
2. **Continuous metrics expose weaknesses hidden by discrete metrics**: CDVAE achieves the highest $d_{\text{comp}}$ uniqueness but the lowest $d_{\text{magpie}}$ uniqueness, indicating that while exact composition repeats are rare, the overall distribution is highly concentrated. DiffCSP++ ranks highest under $d_{\text{wyckoff}}$ but performs poorly under $d_{\text{amd}}$.
3. **Stability filtering is critical**: CDVAE's performance drops sharply after filtering (~3% of samples pass), revealing that its high "diversity" scores derive from large numbers of physically unreasonable structures.
4. **After stability filtering, Chemeleon-DNG is best across all uniqueness metrics; MatterGen leads on most novelty metrics.**

## Highlights & Insights

- **Precise problem identification**: The paper accurately diagnoses systematic flaws in evaluation standards for crystal generation, and the proposed solutions are theoretically rigorous.
- **Orthogonal decomposition of composition vs. structure**: Decomposing the monolithic $d_{\text{smat}}$ into separate composition and structure distances yields more informative evaluations.
- **Discovery of the permutation invariance issue**: The previously overlooked dependency of uniqueness scores on generation order constitutes an important bug report for the field.
- **Open-source tooling**: A reusable Python package, xtalmet, is publicly released.

## Limitations & Future Work

- **Workshop paper with limited scope**: Several analyses are relatively shallow; no systematic ablation over the choice of distance functions is provided.
- **Sensitivity of AMD to disorder and defects is not discussed**: AMD vectors may be inaccurate for crystals containing point defects.
- **Electronic structure differences are not captured**: Both Magpie and AMD operate at the geometric level and do not reflect differences in electronic properties.
- **Threshold and parameter sensitivity is not analyzed**: The $k$ cutoff for AMD and the weights of the 145 Magpie attributes are all used at default settings.
- **Absolute values of continuous uniqueness/novelty lack interpretability**: Values can exceed 1, and normalization is required for meaningful cross-model comparison.

## Related Work & Insights

- **vs $d_{\text{smat}}$ (pymatgen StructureMatcher)**: Discrete → continuous; gains Lipschitz continuity and permutation invariance.
- **vs $d_{\text{wyckoff}}$**: A discrete structural distance that does not satisfy isometry invariance (depends on the origin/choice of the conventional cell); $d_{\text{amd}}$ is invariant to unit cell choice.
- **vs CrystalNN fingerprint distance**: Does not satisfy Lipschitz continuity.
- **vs SOAP-based regularized entropy matching distance**: Also does not satisfy Lipschitz continuity.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective on problem identification is novel and practically significant, though the proposed solutions (Magpie/AMD) are existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 6 models and multiple distance functions, with complete comparisons with and without stability filtering.
- Writing Quality: ⭐⭐⭐⭐ Problem statement is clear, experimental tables are information-dense, and theoretical analysis is concise.
- Value: ⭐⭐⭐⭐ Provides important corrections to evaluation standards in the crystal generation field; open-source tooling enhances practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](continuous_diffusion_model_for_language_modeling.md)
- [\[NeurIPS 2025\] Diffusion Generative Modeling on Lie Group Representations](diffusion_generative_modeling_on_lie_group_representations.md)
- [\[NeurIPS 2025\] Coupling Generative Modeling and an Autoencoder with the Causal Bridge](coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)
- [\[NeurIPS 2025\] Evolve to Inspire: Novelty Search for Diverse Image Generation](evolve_to_inspire_novelty_search_for_diverse_image_generation.md)
- [\[NeurIPS 2025\] Boosting Generative Image Modeling via Joint Image-Feature Synthesis](boosting_generative_image_modeling_via_joint_imagefeature_sy.md)

</div>

<!-- RELATED:END -->
