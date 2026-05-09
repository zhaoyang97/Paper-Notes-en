---
title: >-
  [Paper Note] There Was Never a Bottleneck in Concept Bottleneck Models
description: >-
  [ICLR 2026][Concept Bottleneck Models] This paper identifies that Concept Bottleneck Models (CBMs) do not enforce a true "bottleneck" — the fact that a representation variable $z_j$ can predict concept $c_j$ does not imply it encodes *only* the information of $c_j$. The paper proposes MCBM (Minimal Concept Bottleneck Model), which applies information bottleneck regularization to constrain each $z_j$ to retain only the information of its corresponding concept, thereby achieving genuinely disentangled representations and reliable concept interventions.
tags:
  - ICLR 2026
  - Concept Bottleneck Models
  - Information Bottleneck
  - Information Leakage
  - Interventionability
  - Representation Learning
date: 2026-05-08
content_hash: f7db0c79f2f56629
---

# There Was Never a Bottleneck in Concept Bottleneck Models

**Conference**: ICLR 2026
**arXiv**: [2506.04877](https://arxiv.org/abs/2506.04877)
**Code**: None (per paper description)
**Area**: Interpretability / Concept Bottleneck Models
**Keywords**: Concept Bottleneck Models, Information Bottleneck, Information Leakage, Interventionability, Representation Learning

## TL;DR

This paper identifies that Concept Bottleneck Models (CBMs) do not enforce a true "bottleneck" — the fact that a representation variable $z_j$ can predict concept $c_j$ does not imply it encodes *only* the information of $c_j$. The paper proposes MCBM (Minimal Concept Bottleneck Model), which applies information bottleneck regularization to constrain each $z_j$ to retain only the information of its corresponding concept, thereby achieving genuinely disentangled representations and reliable concept interventions.

## Background & Motivation

- **The Promise of CBMs**: By training each component $z_j$ of the representation to predict an interpretable concept $c_j$, CBMs aim to provide both interpretability and interventionability.
- **Information Leakage**: The ability of $z_j$ to predict $c_j$ does not imply that $z_j$ encodes only $c_j$. In the extreme case, $z_j$ may encode the entire input $\mathbf{x}$ while still satisfying the CBM constraint.
- **Two Consequences**:
  1. **Impaired Interpretability**: $z_j$ cannot be fully explained by $c_j$.
  2. **Ineffective Interventions**: Modifying $z_j$ alters not only $c_j$ but also any other information encoded therein.
- **Theoretical Flaw in CBM Interventions**: There is no directed path from $c_j$ to $z_j$ in CBMs, leaving $p(z_j|c_j)$ undefined in the graphical model. Existing interventions rely on an ad-hoc approximation via empirical quantiles of the sigmoid inverse.

### Core Distinction: VM vs. CBM vs. MCBM

| | VM | CBM | MCBM |
|--|-----|-----|------|
| $z_j$ encodes all of $c_j$? | ✗ | ✓ | ✓ |
| $z_j$ encodes only $c_j$? | ✗ | ✗ | **✓** |

## Method

### 1. Data Generating Process

- Input $\mathbf{x}$ is determined by concepts $\mathbf{c}$ and nuisance $\mathbf{n}$: $p(\mathbf{x}, \mathbf{y}, \mathbf{c}, \mathbf{n}) = p(\mathbf{x}|\mathbf{c}, \mathbf{n}) p(\mathbf{y}|\mathbf{x}) p(\mathbf{c}, \mathbf{n})$
- Nuisance is decomposed into task-relevant $\mathbf{n}_y$ and task-irrelevant $\mathbf{n}_{\bar{y}}$ components.

### 2. Three Information-Theoretic Objectives

**Objective 1** (shared by VM/CBM/MCBM): Maximize $I(Z; Y)$ — representation predicts the target.

$$\max_{\theta, \phi} \mathbb{E}_{p(\mathbf{x}, \mathbf{y})} \left[\mathbb{E}_{p_\theta(\mathbf{z}|\mathbf{x})} \left[\log q_\phi(\hat{\mathbf{y}}|\mathbf{z})\right]\right]$$

**Objective 2** (CBM/MCBM): Maximize $I(Z_j; C_j)$ — $z_j$ is a sufficient statistic for $c_j$.

$$\max_{\theta, \phi} \mathbb{E}_{p(\mathbf{x}, c_j)} \left[\mathbb{E}_{p_\theta(z_j|\mathbf{x})} \left[\log q_\phi(\hat{c}_j|z_j)\right]\right]$$

**Objective 3** (MCBM only): Minimize $I(Z_j; X | C_j)$ — information bottleneck.

$$\min_{\theta, \phi} \mathbb{E}_{p(\mathbf{x}, c_j)} \left[D_{KL}\left(p_\theta(z_j|\mathbf{x}) \| q_\phi(\hat{z}_j|c_j)\right)\right]$$

This ensures that $z_j$ retains no additional information about $\mathbf{x}$ beyond what is captured by $c_j$, enforcing the Markov chain $X \leftrightarrow C_j \leftrightarrow Z_j$.

### 3. MCBM Training Objective

$$\max_{\theta, \phi} \sum_{k=1}^N \sum_i \log q_\phi(\hat{\mathbf{y}}|f'_\theta(x^{(k)}, \epsilon^{(i)})) + \beta \sum_{j=1}^n \log q_\phi(\hat{c}_j|f'_{\theta,j}(\mathbf{x}^{(k)}, \epsilon^{(i)})) - \gamma \sum_{j=1}^n D_{KL}(p_\theta(z_j|\mathbf{x}^{(k)}) \| q_\phi(\hat{z}_j|c_j^{(k)}))$$

- First term: task prediction loss.
- Second term ($\beta$-weighted): concept prediction loss.
- Third term ($\gamma$-weighted): **information bottleneck regularization** (exclusive to MCBM).

### 4. Intervention Mechanism

Interventions in MCBM are theoretically grounded:
$$p(z_j|c_j) = q_\phi(z_j|c_j)$$

Since optimizing Objective 3 ensures $z_j$ encodes only the information of $c_j$, modifying $z_j$ strictly corresponds to an intervention on $c_j$.

### 5. Representation Head Design

- Binary concepts: $g_\phi^z(c_j) = \lambda$ if $c_j=1$, else $-\lambda$
- Multi-class concepts: $g_\phi^z(c_j) = \lambda \cdot \text{one\_hot}(c_j)$ (prototype learning)
- Continuous concepts: $g_\phi^z(c_j) = \lambda \cdot c_j$

The encoder uses a stochastic formulation $p_\theta(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; f_\theta(\mathbf{x}), \sigma_x^2 I)$, trained via the reparameterization trick.

## Key Experimental Results

### Information Leakage Metric: URR (Uncertainty Reduction Ratio)

Measures how much nuisance information beyond the concept set is encoded in $z$ (lower is better).

#### Task-Relevant Nuisance Leakage

| Method | MPI3D | Shapes3D | CIFAR-10 | CUB | AwA2 |
|--------|-------|----------|----------|-----|------|
| Vanilla | 35.0 | 45.5 | 19.8 | 3.8 | 1.5 |
| CBM | 28.1 | 18.1 | 18.5 | 3.8 | 1.4 |
| CEM | 43.2 | 15.8 | **27.2** | 3.9 | 1.1 |
| ECBM | 25.2 | 47.1 | 18.1 | 4.5 | 1.1 |
| **MCBM (high γ)** | **0.0** | **0.0** | **17.6** | **2.4** | **0.7** |

#### Task-Irrelevant Nuisance Leakage

| Method | MPI3D | Shapes3D |
|--------|-------|----------|
| Vanilla | 11.3 | 42.7 |
| CBM | 7.4 | 20.6 |
| CEM | 15.5 | 40.9 |
| **MCBM (any γ)** | **0.0** | **0.0** |

### Key Findings

1. **CEM and ECBM exacerbate leakage**: On certain datasets, leakage exceeds that of the Vanilla model.
2. **MCBM eliminates leakage entirely**: Under high $\gamma$, nuisance information is reduced to 0 across all datasets.
3. **ARCBM and HCBM offer no systematic advantage**: Neither consistently outperforms standard CBM in controlling leakage.
4. **Trade-off**: MCBM incurs a slight decrease in task accuracy, as task-relevant nuisance $\mathbf{n}_y$ is also excluded.

## Highlights & Insights

1. **Fundamental critique of CBMs**: The paper demonstrates that CBMs are, in a principled sense, misnamed — a true bottleneck has never existed in these models.
2. **Natural introduction of the information bottleneck**: The condition $I(Z_j; X | C_j) = 0$ precisely formalizes the requirement that $z_j$ encodes only its corresponding concept.
3. **Theoretical analysis of CBM intervention failures** (Section 5): The paper proves that the intervention assumptions underlying CBMs are probabilistically ill-founded.
4. **Practical KL divergence regularization**: Under Gaussian assumptions, the regularization reduces to a simple MSE loss.
5. **Visualization of disentangled representations**: In MCBM's representation space, samples sharing the same concept value cluster tightly together.

## Limitations & Future Work

- There is an inherent trade-off between task accuracy and concept purity: when the concept set is incomplete, excluding nuisance necessarily reduces performance.
- Concept annotations are required — a limitation shared with all CBM-based methods.
- Handling of continuous concepts relies on Gaussian assumptions.
- The hyperparameter $\gamma$ requires tuning to balance disentanglement and task performance.
- Validation on larger-scale models and more complex tasks remains to be conducted.

## Related Work & Insights

- **CBM Variants**: CEM (concept embeddings), HCBM (hard bottleneck), ARCBM (autoregressive), SCBM (stochastic)
- **Information Leakage Analysis**: Margeloiu et al. 2021; Parisini et al. 2025
- **Information Bottleneck**: Tishby et al. 2000; Alemi et al. 2016 (Variational Information Bottleneck)

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — A fundamental reexamination of the CBM paradigm.
- **Technical Depth**: ⭐⭐⭐⭐ — Rigorous information-theoretic formalization with clear variational derivations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Five datasets, 8+ competing methods, and multi-faceted analysis.
- **Value**: ⭐⭐⭐⭐ — Provides a principled solution toward genuinely interpretable concept-based models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Concepts' Information Bottleneck Models](concepts_information_bottleneck_models.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)
- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](../../CVPR2026/interpretability/towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](../../CVPR2026/interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)

</div>

<!-- RELATED:END -->
