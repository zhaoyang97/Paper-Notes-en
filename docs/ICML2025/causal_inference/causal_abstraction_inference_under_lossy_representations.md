---
title: >-
  [Paper Note] Causal Abstraction Inference under Lossy Representations
description: >-
  [ICML 2025][Causal Inference][Causal Abstraction] This paper proposes the **Projected Abstraction** framework, breaking the reliance of existing causal abstraction theory on the "Abstract Invariance Condition (AIC)." This enables mathematically consistent causal inference under lossy/dimension-reduced representations and provides identifiability criteria at the graphical model level.
tags:
  - "ICML 2025"
  - "Causal Inference"
  - "Causal Abstraction"
  - "Lossy Representation"
  - "Structural Causal Models"
  - "Representation Learning"
date: 2026-05-08
content_hash: 3fbf669cbb57bd4d
---

# Causal Abstraction Inference under Lossy Representations

**Conference**: ICML 2025  
**arXiv**: [2509.21607](https://arxiv.org/abs/2509.21607)  
**Code**: [CausalAILab/ProjectedCausalAbstractions](https://github.com/CausalAILab/ProjectedCausalAbstractions)  
**Area**: Causal Inference  
**Keywords**: Causal Abstraction, Lossy Representation, Structural Causal Models, Causal Inference, Representation Learning

## TL;DR

This paper proposes the **Projected Abstraction** framework, breaking the reliance of existing causal abstraction theory on the "Abstract Invariance Condition (AIC)." This enables mathematically consistent causal inference under lossy/dimension-reduced representations and provides identifiability criteria at the graphical model level.

## Background & Motivation

Causal inference and abstract reasoning are two core capabilities of human intelligence. Causal inference is typically studied under the semantics of Structural Causal Models (SCMs), which induce a three-layer distribution of the "Pearl Causal Hierarchy" (PCH): the observational layer $\mathcal{L}_1$, the interventional layer $\mathcal{L}_2$, and the counterfactual layer $\mathcal{L}_3$.

Causal abstraction aims to establish a correspondence between a complex low-level causal model $\mathcal{M}_L$ and a simpler high-level model $\mathcal{M}_H$. Significant progress has been made by existing works (Rubenstein et al., 2017; Beckers & Halpern, 2019; Xia & Bareinboim, 2024, etc.), but **almost all definitions rely on the Abstract Invariance Condition (AIC)**:

> AIC requires: if two low-level values map to the same high-level value, their impact on downstream variables must be identical.

This restriction is highly demanding in practice—representation learning and dimensionality reduction are inherently lossy transformations, making AIC almost impossible to satisfy. For instance, when HDL and LDL cholesterol are summed into "total cholesterol," their opposing effects on heart disease immediately violate AIC. Similarly, when compressing high-dimensional images into low-dimensional representations, AIC is often unverifiable and unlikely to hold.

The core motivation of this paper is: **even if AIC is violated, can mathematically consistent causal abstractions still be defined and causal inference performed using finite data?**

## Method

### Overall Architecture

The technical pipeline proposed in this paper consists of three steps:

1. **Define Projected Abstraction**: Encode the information lost upon AIC violation into the exogenous variable space, maintaining mathematical consistency in the high-level model (Section 2).
2. **Construct Partially Projected C-DAG**: Provide a new graphical model representation to capture extra dependencies arising from AIC violations (Section 3).
3. **Establish Abstract Identifiability Theorems**: Reduce the cross-abstraction-level causal inference problem to the classic identifiability problem (Section 3).

### Key Designs

#### 1. Partial SCM Projection

Traditional SCM projections only allow entire variables to be included or excluded. This paper introduces the concept of **partial projection**: for each variable $W$, it is decomposed via a surjective function $\delta$ into:

- **Observed part** $W^o$: information retained in the high-level model.
- **Unobserved part** $W^u$: information lost due to lossy mapping, encoded as new exogenous variables.

The key equality (Prop. 1) guarantees that the output under any intervention remains consistent after partial projection:

$$\mathbf{w}^o_{\mathbf{x}} = \mathcal{M}'_{[\mathbf{x}^o]}(\mathbf{u}, \mathbf{x}^u, \mathbf{z}^u)$$

where $\delta(\mathbf{w}^o_{\mathbf{x}}, \mathbf{w}^u_{\mathbf{x}}) = \mathbf{W}_{\mathbf{x}}(\mathbf{u})$.

#### 2. Soft Intervention as the Low-Level Counterpart of High-Level Hard Intervention

An AIC violation causes high-level hard intervention $X_H \leftarrow x_H$ to be ambiguous at the low level (potentially corresponding to multiple low-level values $x_L$ with different effects). This paper interprets it as a **soft intervention** $\sigma_{X_L}$:

$$P(\sigma_{\mathbf{C}_i} = \mathbf{c}_i) = P(\mathbf{c}_i \mid \tau(\mathbf{c}_i) = v_{H,i}, \mathbf{pa}_{V_{H,i}}, \mathbf{u}^c_{V_{H,i}})$$

Intuitively, a high-level intervention is equivalent to performing a weighted average over the corresponding low-level intervention values based on prior probabilities, where the weights depend on the parent node values.

**Insurance Company Example**: When abstracting insurance plans $x_1$ (cheap, efficient) and $x_2$ (cheap, inefficient) into a "cheap plan" $x_C$, the intervention $X_H \leftarrow x_C$ manifests at the low level as:

$$\sigma_{X_L} = \begin{cases} x_1 & \text{with probability } P(x_1 \mid X_L \in \{x_1,x_2\}, z) \\ x_2 & \text{with probability } P(x_2 \mid X_L \in \{x_1,x_2\}, z) \end{cases}$$

#### 3. Algorithmic Construction of High-Level Models (Algorithm 1)

Given a low-level model $\mathcal{M}_L$ and a constructive abstraction function $\tau$, the algorithm systematically constructs a projected abstraction $\mathcal{M}_H$:

- For each low-level variable $W$, split it into $(W^o, W^u)$.
- Incorporate $W^u$ into the exogenous variable set $\mathbf{U}_H$.
- Set the distribution of $W^u$ according to the soft intervention formula.
- High-level function $f^H_i$ reconstructs low-level inputs via $\delta$, then maps outputs using $\tau$.

**Theorem 1** proves that the $\mathcal{M}_H$ constructed by this algorithm maintains $Q$-$\tau$ consistency with $\mathcal{M}_L$ for all $\mathcal{L}_3$ queries.

#### 4. Partially Projected C-DAG

When AIC is violated, the constraints of traditional C-DAGs may be incorrect (Prop. 2). This paper defines the projected C-DAG $\mathcal{G}^\dagger_\mathbb{C}$, adding new edges to the graph caused by AIC violation using three rules:

| Rule | Original Structure | Added Edge | Intuition |
|------|--------------------|------------|-----------|
| (1) | $Z \to X \to Y$, $X$ violates AIC | Add $Z \to Y$ | $X^u$ depends on $Z$, propagating causal effects |
| (2) | $Z \leftrightarrow X \to Y$, $X$ violates AIC | Add $Z \leftrightarrow Y$ and $X \leftrightarrow Y$ | Confounding propagation |
| (3) | $Z \leftarrow X \to Y$, $X$ violates AIC | Add $Z \leftrightarrow Y$ | Unobserved part of common cause introduces confounding |

**Theorem 2** proves that the projected C-DAG completely describes all constraints on high-level variables—both sufficient and necessary.

#### 5. Abstract Identification

**Theorem 3 (Dual Abstract Identifiability)** is the most practically valuable result of this paper:

> A low-level query $Q$ is $\tau$-identifiable under the projected C-DAG $\mathcal{G}^\dagger_\mathbb{C}$ and data $\mathbb{Z}$ if and only if the high-level query $\tau(Q)$ is classically identifiable under $\mathcal{G}^\dagger_\mathbb{C}$ and $\tau(\mathbb{Z})$.

This means researchers can directly reuse existing causal identification algorithms (backdoor criterion, frontdoor criterion, etc.), operating solely on the projected C-DAG.

### Loss & Training

The experiments leverage GAN-NCM (GAN implementation of Neural Causal Models) for training:

- **Generator**: Modeled according to the causal graph structure; each variable corresponds to a subnetwork.
- **Discriminator**: Distinguishes between real and generated data.
- **Projected Sampling**: Directly models and samples using the soft intervention definition, allowing reconstruction of low-level data even under extreme dimensionality reduction.

## Key Experimental Results

### Main Results

**Experiment 1: MNIST Estimation Task (Validating the Necessity of the Projected C-DAG)**

Setting: $Z$ is a digit (0-9), $X$ is a colored MNIST image, and $Y$ is the predicted color label. Target query is $P(y_x \mid z)$.

| Method | Graphical Model Used | Convergence Behavior | Large-Sample MAE |
|------|-----------|----------|-----------|
| Non-abstract NCM | Original graph | Convergent but slow | ~0.04 |
| C-DAG NCM | C-DAG (ignoring AIC violation) | **Diverged** | ~0.08 (biased) |
| Projected C-DAG NCM | Projected C-DAG | **Convergent and fast** | **~0.02** |

**Experiment 2: Causal Colored MNIST (Projected Sampling Quality)**

Setting: Digit $D$ and color $C$ jointly determine image $I$, but $D$ and $C$ are confounded (e.g., 0 tends to be red, 5 tends to be cyan).

| Method | Representation Dim | $\mathcal{L}_1$: $P(I \mid D=0)$ | $\mathcal{L}_2$: $P(I_{D=0})$ | $\mathcal{L}_3$: $P(I_{D=0} \mid D=5)$ |
|------|---------|------|------|------|
| Non-causal | Original | ✓ Correct (red 0) | ✗ Failed | ✗ Failed |
| RNCM (dim=16) | 16-dim | ✓ | ✓ Multi-color 0 | ✓ Cyan 0 |
| RNCM (dim=2) | 2-dim | ✗ Blurry | ✗ Blurry | ✗ Blurry |
| **Projected Sampling (dim=2)** | **2-dim** | **✓** | **✓** | **✓** |

### Ablation Study

| Configuration| Key Metric | Description |
|------|---------|------|
| No abstraction (original space) | Higher MAE, slow convergence | High dimensionality makes learning difficult |
| C-DAG Constraint | Biased MAE, diverged | Incorrect graph constraints; AIC violation not accounted for |
| Projected C-DAG Constraint | Lowest MAE, fast convergence | Correct graphical constraints |
| RNCM high-dim representation (16d) | Good image quality | Limited by AIC, cannot be further compressed |
| Projected Sampling low-dim representation (2d) | Equally good image quality | Breaks through AIC limit, extreme compression remains effective |

### Key Findings

1. **C-DAG is insufficient under AIC violation**: Using incorrect graphical constraints introduces systematic bias, which cannot be corrected even by increasing data volume.
2. **Projected C-DAG is both sufficient and necessary**: It correctly captures extra dependencies caused by AIC violation.
3. **Projected sampling overcomes the representation dimension bottleneck**: Even when compressing high-dimensional images to binary representations, it still correctly generates causal queries across all three levels.
4. **Abstraction brings efficiency gains**: In high-dimensional spaces, methods utilizing abstraction converge faster and with lower error than those without abstraction.

## Highlights & Insights

1. **The core insight is highly elegant**: Instead of trying to force compliance with AIC, the lost information from AIC violation is encoded into exogenous variables. This is a philosophy of "acknowledging and quantifying uncertainty."
2. **The dual identifiability theorem (Thm 3) bridges theory and practice**: It reduces the brand-new abstract identifiability problem directly to classic problems, allowing decades of causal inference algorithms to be reused.
3. **Construction rules for the Projected C-DAG are simple and intuitive**: Only three rules completely describe the impact of AIC violation on graph structure.
4. **Simultaneous coverage of observational, interventional, and counterfactual levels**: Theorem 1 guarantees $\mathcal{L}_3$-level consistency, which is the strongest assurance possible.
5. **Practical value of projected sampling**: Provides a viable solution for the combination of representation learning and causal inference.

## Limitations & Future Work

1. **Assumption of a known low-level causal graph**: Constructing the projected C-DAG requires knowledge of the low-level causal graph structure and the set of AIC-violating variables $\mathbf{V}^\dagger_H$. In practice, this information may not be completely available.
2. **Discrete variable assumption**: The theoretical framework is confined to endogenous variables in finite discrete domains; expansion to continuous variables remains undiscussed.
3. **Recursive SCM assumption**: Requires the causal graph to be acyclic, excluding feedback systems.
4. **Limited experimental scale**: Verified only on simple MNIST-level images, not yet tested on large-scale real-world datasets.
5. **Choice of soft intervention distribution**: Eq. 11 provides a "natural" choice, but multiple alternative options exist. The optimal choice may depend on specific scenarios.

## Related Work & Insights

- **Xia & Bareinboim (2024)**: Directly preceding this work, establishing constructive abstraction functions and abstract consistency theory at the PCH level, though still requiring AIC.
- **Beckers & Halpern (2019)**: The classic definition of $\tau$-abstraction; the projected abstraction in this work generalizes its partial projection.
- **Anand et al. (2023)**: Proposed C-DAGs for cluster causal inference; this paper demonstrates the deficiency of C-DAGs under AIC violation and provides a correction.
- **NCM (Xia et al., 2021, 2023)**: Neural Causal Models, the basic tool used in experiments in this work.
- **Causal Representation Learning** (Ahuja et al., 2023; Brehmer et al., 2022): Complementary to this study, this work provides theoretical guarantees of causal consistency in representation learning.

## Rating

| Dimension | Rating (1-5) | Description |
|------|-----------|------|
| Novelty | ⭐⭐⭐⭐⭐ | Systematically resolves the causal abstraction problem under AIC violations for the first time |
| Theoretical Depth | ⭐⭐⭐⭐⭐ | Three main theorems are closely linked, forming a complete theoretical framework |
| Experimental Thoroughness | ⭐⭐⭐ | Ingeniously designed but small-scale experiments |
| Writing Quality | ⭐⭐⭐⭐ | Dense mathematical notation but clear examples and systematic organization |
| Value | ⭐⭐⭐⭐ | Dual theorem is directly applicable; projected sampling offers practical value |
| **Overall** | **⭐⭐⭐⭐☆** | A significant advancement in causal abstraction theory, though experiments could be further strengthened |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Latent Variable Causal Discovery under Selection Bias](latent_variable_causal_discovery_under_selection_bias.md)
- [\[AAAI 2026\] Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects](../../AAAI2026/causal_inference/causal_inference_under_threshold_manipulation_bayesian_mixtu.md)
- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](../../NeurIPS2025/causal_inference/practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[ICLR 2026\] Learning Robust Intervention Representations with Delta Embeddings](../../ICLR2026/causal_inference/learning_robust_intervention_representations_with_delta_embeddings.md)
- [\[ICML 2026\] Toward Scalable and Valid Conditional Independence Testing with Spectral Representations](../../ICML2026/causal_inference/toward_scalable_and_valid_conditional_independence_testing_with_spectral_represe.md)

</div>

<!-- RELATED:END -->
