---
title: >-
  [Paper Note] There Was Never a Bottleneck in Concept Bottleneck Models
description: >-
  [ICLR 2026][Interpretability][Paper Note] The paper points out that Concept Bottleneck Models (CBMs) do not actually possess a true "bottleneck"—the fact that a representation variable $z_j$ can predict concept $c_j$ does not mean it encodes *only* information about $c_j$. It proposes the Minimal Concept Bottleneck Model (MCBM), which uses information bottlene
tags:
  - ICLR 2026
  - Interpretability
date: 2026-05-08
content_hash: 578bbdf6df9c4439
---
# There Was Never a Bottleneck in Concept Bottleneck Models

**Conference**: ICLR 2026  
**arXiv**: [2506.04877](https://arxiv.org/abs/2506.04877)  
**Code**: None (according to the paper)  
**Area**: Interpretability / Concept Bottleneck Models  
**Keywords**: Concept Bottleneck Models, Information Bottleneck, Information Leakage, Intervenability, Representation Learning

## TL;DR

The paper points out that Concept Bottleneck Models (CBMs) do not actually possess a true "bottleneck"—the fact that a representation variable $z_j$ can predict concept $c_j$ does not mean it encodes *only* information about $c_j$. It proposes the Minimal Concept Bottleneck Model (MCBM), which uses information bottleneck regularization to constrain each $z_j$ to retain only information from its corresponding concept, achieving true decoupled representations and reliable concept interventions.

## Background & Motivation

- **The Promise of CBMs**: Providing interpretability and intervenability by ensuring each component $z_j$ of the representation predicts an understandable concept $c_j$.
- **Information Leakage**: $z_j$ predicting $c_j$ is not equivalent to $z_j$ only encoding $c_j$. In extreme cases, $z_j$ might encode the entire input $\mathbf{x}$ while still satisfying CBM constraints.
- **Two Consequences**:
  1. **Compromised Interpretability**: $z_j$ cannot be fully explained by $c_j$.
  2. **Ineffective Intervention**: Modifying $z_j$ not only changes $c_j$ but also affects other information encoded within it.
- **Theoretical Flaws in CBM Interventions**: There is no directed path from $c_j$ to $z_j$ in CBMs; $p(z_j|c_j)$ is undefined in the graphical model. Existing interventions rely on ad-hoc empirical quantile approximations of the inverse sigmoid function.

### Core Difference: CBM vs. MCBM

| | VM | CBM | MCBM |
|--|-----|-----|------|
| $z_j$ encodes all of $c_j$? | ✗ | ✓ | ✓ |
| $z_j$ encodes *only* $c_j$? | ✗ | ✗ | **✓** |

## Method

### Overall Architecture

MCBM supplements standard CBMs with an omitted constraint: not only must each representation component $z_j$ predict the corresponding concept $c_j$, but $z_j$ must also carry no residual information about the input $\mathbf{x}$ given $c_j$. This constraint is added to the training objective in the form of information bottleneck regularization, compressing $z_j$ into a minimal sufficient statistic of $c_j$. This establishes a strict correspondence between "modifying $z_j$" and "intervening on $c_j$."

### Key Designs

**1. Data Generation Assumption: Explicitly Modeling the Source of Leakage**

The paper assumes a generative process $p(\mathbf{x}, \mathbf{y}, \mathbf{c}, \mathbf{n}) = p(\mathbf{x}|\mathbf{c}, \mathbf{n})\, p(\mathbf{y}|\mathbf{x})\, p(\mathbf{c}, \mathbf{n})$. The input $\mathbf{x}$ is determined by both labeled concepts $\mathbf{c}$ and unlabeled nuisances $\mathbf{n}$. Nuisances are further divided into task-relevant $\mathbf{n}_y$ and task-irrelevant $\mathbf{n}_{\bar{y}}$. This breakdown is necessary because "leakage" in CBMs essentially means $z_j$ encodes these nuisances that should have been blocked; explicitly defining nuisances allows for precise characterization of what to retain and what to discard.

**2. Three Information-Theoretic Objectives: Progressing from Sufficient to Minimal**

MCBM optimization consists of three complementary information constraints to tighten the representation. The first is task sufficiency (common to all variants), maximizing $I(Z;Y)$, which corresponds to $\max_{\theta, \phi} \mathbb{E}_{p(\mathbf{x}, \mathbf{y})}[\mathbb{E}_{p_\theta(\mathbf{z}|\mathbf{x})}[\log q_\phi(\hat{\mathbf{y}}|\mathbf{z})]]$, ensuring the representation can predict labels. The second is concept sufficiency (also in CBM), maximizing $I(Z_j; C_j)$, written as $\max_{\theta, \phi} \mathbb{E}_{p(\mathbf{x}, c_j)}[\mathbb{E}_{p_\theta(z_j|\mathbf{x})}[\log q_\phi(\hat{c}_j|z_j)]]$, requiring $z_j$ to be sufficient to decode $c_j$. What distinguishes MCBM is the third: minimizing conditional mutual information $I(Z_j; X | C_j)$, approximated via KL divergence as $\min_{\theta, \phi} \mathbb{E}_{p(\mathbf{x}, c_j)}[D_{KL}(p_\theta(z_j|\mathbf{x}) \| q_\phi(\hat{z}_j|c_j))]$. This forces $z_j$ to contain no extra information about $\mathbf{x}$ once $c_j$ is known, making the Markov chain $X \leftrightarrow C_j \leftrightarrow Z_j$ hold. While the first two only ensure $z_j$ is "sufficient" to encode $c_j$, the third ensures it "only" encodes $c_j$, which is the bottleneck missing in CBMs.

**3. Theoretically Grounded Intervention: Equating $z_j$ Modification to $c_j$ Intervention**

In standard CBMs, no directed path exists from $c_j$ to $z_j$, making $p(z_j|c_j)$ undefined. Existing interventions are thus ad-hoc approximations. After optimizing the third objective in MCBM, $z_j$ only contains information about $c_j$, allowing interventions to directly follow $p(z_j|c_j) = q_\phi(z_j|c_j)$. Replacing $z_j$ strictly corresponds to setting the concept to a target value without inadvertently modifying other secretly encoded information, upgrading intervenability from "approximate" to "exact."

**4. Concept-Conditioned Representation Heads and Stochastic Encoders: Trainable KL Terms**

To provide a concrete form for $q_\phi(\hat{z}_j|c_j)$, the paper designs prototypical representation heads: for binary concepts, $g_\phi^z(c_j) = \lambda$ ($c_j=1$) or $-\lambda$ ($c_j=0$); for multi-class, $g_\phi^z(c_j) = \lambda \cdot \text{one\_hot}(c_j)$; for continuous concepts, $g_\phi^z(c_j) = \lambda \cdot c_j$. This essentially learns a prototype anchor for each concept value. The encoder uses a stochastic version $p_\theta(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; f_\theta(\mathbf{x}), \sigma_x^2 I)$ trained with the reparameterization trick. Under this Gaussian assumption, the KL term simplifies to a straightforward MSE between the representation and the prototype.

### Loss & Training

By merging the three objectives, the total MCBM training goal is:

$$\max_{\theta, \phi} \sum_{k=1}^N \sum_i \log q_\phi(\hat{\mathbf{y}}|f'_\theta(x^{(k)}, \epsilon^{(i)})) + \beta \sum_{j=1}^n \log q_\phi(\hat{c}_j|f'_{\theta,j}(\mathbf{x}^{(k)}, \epsilon^{(i)})) - \gamma \sum_{j=1}^n D_{KL}(p_\theta(z_j|\mathbf{x}^{(k)}) \| q_\phi(\hat{z}_j|c_j^{(k)}))$$

The first term is task prediction loss, the second is concept prediction loss weighted by $\beta$, and the third is the information bottleneck regularization unique to MCBM, weighted by $\gamma$. $\gamma$ controls the strength of decoupling: increasing it pushes nuisance leakage toward zero but may discard task-useful information in $\mathbf{n}_y$, making it a knob for the trade-off between "concept purity" and "task accuracy."

## Key Experimental Results

### Information Leakage Metric: URR (Uncertainty Reduction Ratio)

Measures how much nuisance information beyond the concept set is encoded in $z$ (lower is better).

#### Task-Relevant Nuisance Leakage

| Method | MPI3D | Shapes3D | CIFAR-10 | CUB | AwA2 |
|------|-------|----------|----------|-----|------|
| Vanilla | 35.0 | 45.5 | 19.8 | 3.8 | 1.5 |
| CBM | 28.1 | 18.1 | 18.5 | 3.8 | 1.4 |
| CEM | 43.2 | 15.8 | **27.2** | 3.9 | 1.1 |
| ECBM | 25.2 | 47.1 | 18.1 | 4.5 | 1.1 |
| **MCBM (high γ)** | **0.0** | **0.0** | **17.6** | **2.4** | **0.7** |

#### Task-Irrelevant Nuisance Leakage

| Method | MPI3D | Shapes3D |
|------|-------|----------|
| Vanilla | 11.3 | 42.7 |
| CBM | 7.4 | 20.6 |
| CEM | 15.5 | 40.9 |
| **MCBM (Any γ)** | **0.0** | **0.0** |

### Key Findings

1. **CEM and ECBM Exacerbate Leakage**: On some datasets, leakage is higher than in Vanilla models.
2. **MCBM Eliminates Leakage**: Under high γ, nuisance information drops to 0 across all datasets.
3. **No Systematic Advantage for ARCBM and HCBM**: They do not control leakage better than standard CBMs.
4. **The Cost**: MCBM's task accuracy drops slightly because it excludes task-useful information in $\mathbf{n}_y$.

## Highlights & Insights

1. **Fundamental Concept Critique**: Points out that CBMs are misnamed—there was never a true "bottleneck."
2. **Natural Intro of Information Bottleneck**: Precisely formalizes "encoding concepts only" using $I(Z_j; X | C_j) = 0$.
3. **Analysis of Theoretical Flaws in CBM Intervention** (Section 5): Proves that CBM intervention assumptions are probabilistically invalid.
4. **Practical KL Regularization**: Reduces to simple MSE loss under Gaussian assumptions.
5. **Visualization of Decoupled Representations**: Samples with the same concept values cluster tightly in the MCBM representation space.

## Limitations & Future Work

- Inherent trade-off between task accuracy and concept purity: if the concept set is incomplete, excluding nuisances inevitably degrades performance.
- Requires concept labeling—a limitation shared with all CBM methods.
- Treatment of continuous concepts relies on Gaussian assumptions.
- Hyperparameter $\gamma$ needs tuning to balance decoupling and performance.
- Not yet validated on larger-scale models or more complex tasks.

## Related Work & Insights

- **CBM Variants**: CEM (Concept Embedding), HCBM (Hard Bottleneck), ARCBM (Autoregressive), SCBM (Stochastic).
- **Information Leakage Analysis**: Margeloiu et al. 2021, Parisini et al. 2025.
- **Information Bottleneck**: Tishby et al. 2000, Alemi et al. 2016 (Variational Information Bottleneck).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — A fundamental re-examination of the CBM field.
- **Technical Depth**: ⭐⭐⭐⭐ — Rigorous information-theoretic formalization and clear variational derivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 5 datasets, 8+ methods compared, multi-angle analysis.
- **Value**: ⭐⭐⭐⭐ — Provides a principled solution for truly interpretable concept models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Concepts' Information Bottleneck Models](concepts_information_bottleneck_models.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)
- [\[ICLR 2026\] Debugging Concept Bottleneck Models through Removal and Retraining](debugging_concept_bottleneck_models_through_removal_and_retraining.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](../../CVPR2026/interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)
- [\[AAAI 2026\] Flexible Concept Bottleneck Model](../../AAAI2026/interpretability/flexible_concept_bottleneck_model.md)

</div>

<!-- RELATED:END -->
