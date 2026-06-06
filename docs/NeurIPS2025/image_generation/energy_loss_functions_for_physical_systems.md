---
title: >-
  [Paper Note] Energy Loss Functions for Physical Systems
description: >-
  [NeurIPS 2025][Image Generation][Energy loss functions] This paper proposes a physics-based energy loss function framework. By deriving an energy-difference loss grounded in pairwise distances via reverse KL divergence a…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Energy loss functions"
  - "physical priors"
  - "diffusion models"
  - "symmetry invariance"
  - "molecular generation"
date: 2026-05-08
content_hash: 0d33d991f632efb1
---

# Energy Loss Functions for Physical Systems

**Conference**: NeurIPS 2025
**arXiv**: [2511.02087](https://arxiv.org/abs/2511.02087)  
**Code**: [Available](https://github.com/kushasareen/energy_loss)  
**Area**: Image Generation
**Keywords**: Energy loss functions, physical priors, diffusion models, symmetry invariance, molecular generation

## TL;DR

This paper proposes a physics-based energy loss function framework. By deriving an energy-difference loss grounded in pairwise distances via reverse KL divergence and the Boltzmann distribution, the framework naturally satisfies SE(d) invariance and substantially outperforms MSE and cross-entropy losses on molecular generation and spin ground-state prediction tasks.

## Background & Motivation

When applying machine learning to physical systems, data is typically scarce and expensive to obtain. The incorporation of physical priors has largely focused on architectural choices (e.g., equivariant networks), while the complementary direction of loss function design has been severely neglected.

Core Problem: **Why are generic loss functions (MSE, cross-entropy) ill-suited for physical systems?**

**MSE corresponds to non-physical energy**: MSE is equivalent to a data-centric isotropic harmonic potential $E = \|\hat{y} - y\|^2$—this describes an external force pulling a particle toward a fixed reference point, rather than inter-particle interactions.

**Symmetry breaking**: Physical configurations are typically equivalent under rigid-body transformations SE(d), yet MSE penalizes correct predictions that are related to the target by a rotation or translation.

**Bias of the MSE minimizer**: When multiple symmetry-equivalent targets exist, the MSE minimizer is their mean—which is not a valid physical configuration.

## Method

### Overall Architecture

The core mechanism of the loss function design:

1. Model the uncertainty associated with each data sample as a **Boltzmann distribution** centered at that sample.
2. Use **reverse KL divergence** (rather than forward KL / MLE) as the optimization objective.
3. The resulting loss is equivalent to the **energy difference** between the data and the prediction.

Two key advantages of reverse KL: (a) the partition function $Z(y, T)$ does not depend on the model parameters $\theta$, avoiding the need to compute an intractable normalization constant; (b) only the potential near the data point needs to be defined (defining energy at a potentially poor prediction location is physically meaningless).

### Key Designs

#### 1. From Reverse KL to Energy Loss

For a Boltzmann distribution $p(\hat{y}|y) = \exp(-E(\hat{y},y)/T) / Z(y,T)$, the reverse KL loss is:

$$\mathcal{J}(\theta) = \sum_i \frac{E(\hat{y}_\theta^{(i)}, y^{(i)})}{T} + \log Z(y^{(i)}, T)$$

The $\log Z$ term is independent of $\theta$ and can be discarded. The model is penalized by the incremental energy of its prediction relative to the data.

MSE is a special case of the energy loss: setting $E(\hat{y},y) = \|\hat{y}-y\|^2$ and $T=2\sigma^2$ recovers a Gaussian conditional distribution. However, this corresponds to a non-physical external harmonic potential.

#### 2. Pairwise Quadratic Potential for Atomic Systems

For a system of $n$ atoms in $d$-dimensional space, a pairwise quadratic potential based on interatomic distances is defined as:

$$E(\hat{y}, y) = \sum_{i,j}^n \frac{1}{2}k_{ij}(y)(\|y_i - y_j\| - \|\hat{y}_i - \hat{y}_j\|)^2$$

Physical interpretation: this is the second-order Taylor approximation of a general pairwise interaction potential (spring model). The choice of spring constant $k_{ij}(y)$:

| Coefficient Type | Origin | Use Case |
|---------|------|---------|
| Constant $k$ | Taylor approximation of Morse potential | Theoretical analysis |
| Exponential decay | Empirically optimal | EDM/GDM models |
| Inverse square distance | Taylor approximation of Lennard-Jones potential | Physical correspondence |
| Inverse distance | Empirical | JODO model |

#### 3. Symmetry Invariance—Obtained for Free

The energy loss depends only on invariant pairwise distances, and thus naturally satisfies $E(d)$ invariance. Furthermore, the loss is also invariant under the automorphism group of the distance matrix:

$$G = E(d) \times (\text{Aut}(k(y)) \cap \text{Aut}(\Delta y))$$

**Corollary**: The set of global minimizers is $\{g \cdot y \mid g \in G\}$—the model may regress to any configuration symmetrically equivalent to the data, without requiring Kabsch alignment (which costs $O(n^3)$).

#### 4. Application to Diffusion Models

The energy loss directly replaces MSE in diffusion model training. Key theoretical result:

**Proposition 4.4**: For SE(d)-invariant densities, at small $\sigma_t$, the minimum-norm minimizer satisfies:

$$\hat{\epsilon}^*_{\text{dist}} \approx -\sigma_t \nabla_{x_t}\log p(x_t)$$

thereby correctly recovering the score function.

**Proposition 4.5**: The variance of the score estimator under the energy loss is no greater than under MSE:

$$\text{Var}[\hat{\epsilon}^*_{\text{dist}}] \lesssim \text{Var}[\hat{\epsilon}^*_{\text{MSE}}]$$

#### 5. Linear Scaling via Rigidity Theory

The pairwise distance sum contains $O(n^2)$ terms. By Laman's theorem, only $O(n)$ edges forming a rigid graph are needed to recover coordinates from distances. Sparse rigid graphs reduce computation to linear complexity without affecting the global optimal solution.

#### 6. Discrete Systems: Spin Energy Loss

For spin systems $\hat{y}, y \in \{1,-1\}^\Lambda$, the local field energy is defined as:

$$E(\hat{y}, y) = \sum_i h_i^{LF}(y)\hat{y}_i, \quad h_i^{LF}(y) = \sum_j (J_{ij} + h^0)y_j$$

The local field captures the energy change upon flipping a spin. The objective becomes variational free energy minimization. The local field loss is convex and has the data as its unique minimizer (when $h^0 > 4$).

### Loss & Training

The energy loss directly replaces MSE/cross-entropy without modifying any other part of the training pipeline. Key hyperparameters to tune: the functional form of $k_{ij}$ and the weighting ratio between position and atom-type losses. All comparative experiments include thorough sweeps over learning rates and weights.

## Key Experimental Results

### Main Results

**Table 1: GEOM-Drugs Molecular Generation (GDM-aug)**

| Loss | Mol. Stability (%) | Atom Stability (%) | Validity (%) | Uniqueness (%) |
|------|-------------|-------------|----------|----------|
| MSE | 0.8 | 85.6 | 94.8 | 100 |
| **Energy** | **24.6** | **96.0** | **89.7** | 100 |

Molecular stability improves from 0.8% to **24.6%**, a roughly **30×** improvement.

**Table 2: QM9 Molecular Generation (GDM-aug)**

| Loss | Mol. Stability (%) | Atom Stability (%) | Validity (%) |
|------|-------------|-------------|----------|
| MSE | 83.7±2.3 | 98.3 | 93.6 |
| Kabsch align | 82.3±0.5 | 97.8 | 90.8 |
| **Energy** | **89.8±2.8** | **99.3** | **97.7** |
| Energy (sparse) | 89.1±0.9 | 99.0 | 97.4 |

**Table 3: JODO 3D and Alignment Metrics**

| Model | Mol. Stability | Validity | Bond↓ | Angle↓ |
|------|----------|--------|-------|--------|
| JODO (original) | 93.4 | — | 0.1475 | 0.0121 |
| JODO+Energy(Inv.) | **94.3** | **97.1** | **0.1125** | **0.0046** |

The energy loss yields substantial gains even on the near-SOTA model JODO, with markedly improved alignment metrics.

### Ablation Study

**Ablation on Spring Coefficient $k_{ij}$ (QM9, GDM-aug)**

| Coefficient Type | Mol. Stability (%) | Validity (%) |
|---------|-------------|----------|
| Exponential decay | **89.8** | **97.7** |
| Inverse square distance | 84.6 | 96.6 |
| Inverse distance | 84.5 | 95.0 |
| Constant | 83.6 | 93.6 |

**Spin Ground-State Prediction**

| Loss | Test Energy |
|------|---------|
| Cross-entropy | 58.8±0.8 |
| Margin | 49.87±1.5 |
| **Local field energy** | **45.6±1.6** |
| True energy | 14.6±0.3 |

### Key Findings

1. Energy loss + non-equivariant architecture outperforms equivariant architecture + MSE, with negligible computational overhead.
2. High data efficiency: with 50% of the training data, the energy loss still produces >75% stable molecules, whereas MSE degrades substantially.
3. Sparse rigid graphs reduce complexity from $O(n^2)$ to $O(n)$ with almost no performance loss.
4. The optimal $k_{ij}$ differs across models (exponential decay for GDM, inverse distance for JODO), requiring task-specific tuning.
5. Regular-shape experiments provide intuitive evidence: the energy loss is entirely immune to rotational augmentation, while MSE collapses at $\theta_{aug}=\pi$.

## Highlights & Insights

- **Reverse KL + Boltzmann = energy difference loss**: The derivation is elegant and the physical intuition is transparent.
- **Invariance obtained for free**: The distance-based loss is naturally SE(d)-invariant, eliminating the need for $O(n^3)$ Kabsch alignment.
- **Architecture-agnostic and plug-and-play**: The energy loss can directly replace MSE in any training objective.
- **Physical pathology of MSE formally characterized**: The three-panel Figure 1 provides highly intuitive evidence—rotationally correct configurations are incorrectly penalized by MSE.
- **Elegant application of rigidity theory**: Graph rigidity enables $O(n^2) \to O(n)$ scaling.

## Limitations & Future Work

1. Score recovery theory is exact only in the low-noise regime; explicit corrections are required at high noise levels.
2. The choice of spring coefficient still relies on heuristic tuning; an automatic selection mechanism is lacking.
3. Torsion angle information is not exploited; richer surrogate energies incorporating angular terms warrant exploration.
4. Validation is limited to small molecules and 2D spin systems; generalization to proteins, crystals, and other large systems requires further investigation.
5. Non-physical coefficients (exponential decay) outperform theoretically better-motivated constant coefficients—this tension requires deeper understanding.

## Related Work & Insights

- **AlphaFold 3 (Abramson 2024)**: Has heuristically employed a distance-based loss as a regularizer; this work provides a first-principles theoretical foundation for that choice.
- **Kabsch alignment loss (Klein 2023)**: An alternative SE(3)-invariant loss, but requires $O(n^3)$ alignment.
- **JODO (Huang 2023)**: A near-SOTA joint 2D/3D molecular diffusion model; the energy loss further improves its metrics.
- **Insight**: The loss function as a channel for injecting physical priors is severely underexplored—architectural equivariance and loss invariance are complementary, not substitutes.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Loss functions derived from first principles; a uniquely physical perspective.
- Technical Depth: ⭐⭐⭐⭐⭐ — Unified framework for continuous and discrete systems; proof of score recovery in diffusion models; application of rigidity theory.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three experimental domains (molecules, spins, regular shapes); limited to small molecular scales.
- Value: ⭐⭐⭐⭐⭐ — Plug-and-play, computationally efficient, and theoretically grounded.

Choosing the isotropic harmonic potential $E(\hat{\mathbf{y}}, \mathbf{y}) = \|\hat{\mathbf{y}} - \mathbf{y}\|^2$ recovers the Gaussian distribution and MSE loss as a special case. However, this "energy" describes an external force pulling particles toward a reference position, which is physically unreasonable.

### 3. Pairwise Quadratic Potential for Atomic Systems

For $n$ atoms with positions $\mathbf{y} \in \mathbb{R}^{n \times d}$, the paper proposes a pairwise quadratic potential based on interparticle distances:

$$E(\hat{\mathbf{y}}, \mathbf{y}) = \sum_{i,j}^{n} \frac{1}{2} k_{ij}(\mathbf{y}) (\|\mathbf{y}_i - \mathbf{y}_j\| - \|\hat{\mathbf{y}}_i - \hat{\mathbf{y}}_j\|)^2$$

This is the second-order Taylor approximation of a general pairwise interaction potential in terms of pairwise distances. The coefficient $k_{ij}(\mathbf{y})$ admits several choices: constant (Morse potential approximation), inverse distance, inverse square distance (Lennard-Jones approximation), or exponential decay.

### 4. Symmetry Invariance

**Proposition**: When $k_{ij}(\mathbf{y})$ is invariant, the energy loss is invariant under the group $G = E(d) \times (\text{Aut}(k(\mathbf{y})) \cap \text{Aut}(\Delta y))$—naturally respecting Euclidean transformations and particle permutation symmetries.

**Corollary**: The family of global minimizers of the loss is exactly $\{g \cdot \mathbf{y} | g \in G\}$; the model may regress to any configuration symmetrically equivalent to the data.

Compared to computing MSE after Kabsch alignment, the energy loss **requires no alignment step**, resulting in greater computational efficiency.

### 5. Application to Diffusion Models

The energy loss directly replaces MSE during diffusion model training. Theoretical analysis shows that, at low noise levels, the optimal predictor under the constant-coefficient energy loss approximates the correct score function up to a degree of freedom in the rigid-body motion direction:

$$\hat{\epsilon}^* \approx -\sigma_t \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \mathbf{v}, \quad \mathbf{v} \in \ker(J(\mathbf{x}_t))$$

The minimum-norm minimizer is exactly the correct score. It is further proven that the variance of the energy loss estimator does not exceed that of the MSE estimator.

### 6. Discrete Energy Loss for Spin Systems

For an Ising-type Hamiltonian $E(\mathbf{y}) = -\frac{1}{2}\sum_{ij} J_{ij} \mathbf{y}_i \mathbf{y}_j$, the local field energy is defined as:

$$E(\hat{\mathbf{y}}, \mathbf{y}) = \sum_{i} h_i^{\text{LF}}(\mathbf{y}) \hat{\mathbf{y}}_i, \quad h_i^{\text{LF}}(\mathbf{y}) = \sum_j (J_{ij} + h^0) \mathbf{y}_j$$

The loss reduces to the variational free energy: $\mathcal{J}(\theta) = \frac{1}{T}\left[\mathbb{E}_q[E] - T S[q] + T \log Z\right]$.

## Key Experimental Results

### Molecular Generation — QM9 (GDM-aug)

| Loss | Mol. Stability (%) | Atom Stability (%) | Validity (%) | Uniqueness (%) |
|------|-------------|-------------|----------|----------|
| MSE | 83.7 ± 2.3 | 98.3 | 93.6 | 100.0 |
| Kabsch alignment | 82.3 ± 0.5 | 97.8 | 90.8 | 100.0 |
| **Energy loss** | **89.8 ± 2.8** | **99.3** | **97.7** | 99.9 |
| Energy loss (sparse) | 89.1 ± 0.9 | 99.0 | 97.4 | 100.0 |

Molecular stability improves by **+6.1%** and validity by **+4.1%**.

### GEOM-Drugs Large Molecules

| Loss | Mol. Stability (%) | Atom Stability (%) | Validity (%) |
|------|-------------|-------------|----------|
| MSE | 0.8 | 85.6 | 94.8 |
| **Energy loss** | **24.6** | **96.0** | 89.7 |

Molecular stability increases from 0.8% to **24.6%**—an extremely significant improvement in the large-molecule regime.

### JODO Model (Near SOTA)

The energy loss yields consistent improvements in atom stability (+0.4%), mol stability (+3.8%), and validity (+2.8%); bond length MMD decreases from 0.1218 to **0.0928**, indicating more accurate bond lengths in generated molecules.

### Spin Ground-State Prediction (16×16 Lattice)

| Loss | Test Energy |
|------|---------|
| Cross-entropy | 58.8 ± 0.8 |
| Margin Loss | 49.87 ± 1.5 |
| **Local field energy loss** | **45.6 ± 1.6** |
| True energy (non-classification) | 14.6 ± 0.3 |

The energy loss substantially reduces the energy of predicted configurations compared to cross-entropy.

### Low-Data Regime

With only 50% of the training data (50K samples), the energy loss still generates over 75% stable molecules; MSE degrades markedly under the same data budget—demonstrating a **significant improvement in data efficiency**.

## Highlights & Insights

1. **Elegant theoretical framework**: Starting from reverse KL and the Boltzmann distribution, the paper unifies MSE, cross-entropy, and energy losses, providing a physical interpretation of loss functions.
2. **Natural symmetry invariance**: SE(d) symmetry is respected without Kabsch alignment, with negligible computational overhead.
3. **Orthogonal to architecture**: The energy loss is an architecture-agnostic plug-and-play module whose benefits are complementary to equivariant architectures (e.g., EGNN).
4. **Theoretical guarantees**: It is proven that the energy loss correctly estimates the score function in diffusion models with lower variance.
5. The improvement from 0.8% to 24.6% on the large-molecule GEOM-Drugs benchmark is quantitatively impressive.

## Limitations & Future Work

1. In diffusion models, exact score recovery holds only in the **low-noise regime**; explicit correction is needed at high noise levels.
2. The choice of energy function retains some degree of ad hoc character; the coefficient $k_{ij}$ requires ablation for new tasks.
3. The standard formulation has $O(N^2)$ complexity (over particle pairs); although sparse rigid graphs can reduce this to $O(N)$, constructing the sparse graph requires additional handling.
4. Richer geometric descriptors such as torsion angles remain unexplored, and validation on proteins or crystalline materials has not been conducted.

## Related Work & Insights

| Dimension | MSE | Kabsch + MSE | Energy Loss |
|------|-----|-------------|---------|
| Symmetry invariant | ✗ | ✓ (requires alignment) | **✓ (natural)** |
| Physical prior | None | None | **Inter-particle interaction** |
| Computational cost | O(N) | O(N³) alignment | O(N²) or O(N) |
| Score recovery | Exact | Exact | **Approximately exact + lower variance** |
| Complementarity with equivariant architectures | Partial | Partial | **Strong** |

## Related Work & Insights

1. **"Loss functions are also priors"**: The loss function implicitly encodes a distributional assumption about errors; choosing a physically appropriate distribution (Boltzmann) is more principled than a simple Gaussian.
2. **Addressing symmetry at the loss level is more lightweight than at the architecture level**: Equivariant architecture design is complex and increases parameter count, whereas an invariant loss function only changes the objective computation.
3. **The gradient of the energy loss points toward the nearest valid configuration** (rather than the data itself), which is particularly valuable for prediction tasks with symmetry breaking.
4. The framework generalizes directly to any physical system in thermal equilibrium—crystalline materials, proteins, soft matter, etc.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Loss functions derived from first principles; outstanding theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two system types (molecular + spin), multi-model validation, thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous and elegant theoretical derivations, well-placed intuitive explanations, illuminating figures.
- Value: ⭐⭐⭐⭐⭐ — Provides a general and practical framework that can directly improve existing molecular generation pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[ICML 2026\] A Diffusive Classification Loss for Learning Energy-based Generative Models](../../ICML2026/image_generation/a_diffusive_classification_loss_for_learning_energy-based_generative_models.md)
- [\[NeurIPS 2025\] Aligning Compound AI Systems via System-level DPO](aligning_compound_ai_systems_via_system-level_dpo.md)
- [\[NeurIPS 2025\] LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss](lineas_end-to-end_learning_of_activation_steering_with_a_distributional_loss.md)
- [\[NeurIPS 2025\] Hephaestus: Mixture Generative Modeling with Energy Guidance for Large-scale QoS Degradation](hephaestus_mixture_generative_modeling_with_energy_guidance_for_large-scale_qos_.md)

</div>

<!-- RELATED:END -->
