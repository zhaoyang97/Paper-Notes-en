---
title: >-
  [Paper Note] UnCLe: Towards Scalable Dynamic Causal Discovery in Non-Linear Temporal Systems
description: >-
  [NeurIPS 2025][Human Understanding][dynamic causal discovery] This paper proposes UnCLe, a scalable dynamic causal discovery method based on TCN autoencoder disentanglement and autoregressive dependency matrices. It infe…
tags:
  - "NeurIPS 2025"
  - "Human Understanding"
  - "dynamic causal discovery"
  - "time series"
  - "Granger causality"
  - "temporal perturbation"
  - "nonlinear systems"
date: 2026-05-08
content_hash: e15591bf5bb4ab06
---

# UnCLe: Towards Scalable Dynamic Causal Discovery in Non-Linear Temporal Systems

**Conference**: NeurIPS 2025
**arXiv**: [2511.03168](https://arxiv.org/abs/2511.03168)  
**Code**: [GitHub](https://github.com/etigerstudio/uncle-causal-discovery)  
**Area**: Causal Discovery / Time Series Analysis
**Keywords**: dynamic causal discovery, time series, Granger causality, temporal perturbation, nonlinear systems

## TL;DR

This paper proposes UnCLe, a scalable dynamic causal discovery method based on TCN autoencoder disentanglement and autoregressive dependency matrices. It infers time-varying causal relationships by measuring per-timestep prediction error increments following temporal perturbation, achieving state-of-the-art performance on both static and dynamic causal discovery benchmarks.

## Background & Motivation

Discovering causal relationships from observational time series is a fundamental problem in understanding complex systems. Real-world systems typically exhibit **dynamic causality**—causal relationships that evolve over time:

**Predator–prey relationships** shift with seasons

**Gene regulatory networks** change across developmental stages

**Human biomechanics** exhibit different inter-joint causal relationships across motion phases (takeoff → flight → landing)

Nevertheless, mainstream temporal causal discovery methods almost exclusively infer **static causal graphs** (time-averaged dependencies), disregarding the evolving nature of causal relationships.

Specific limitations of existing methods:
- **NeuralGC / GVAR / TCDF**: Component-wise design with $O(N^2)$ parameters, not scalable to large systems
- **GVAR**: Capable of generating dynamic causal graphs but with limited accuracy (e.g., direction reversals in TVSEM cannot be captured)
- **JRNGC / CUTS+**: Address scalability via parameter sharing but do not support dynamic causal discovery
- No existing method has been rigorously evaluated on dynamic causal datasets

## Method

### Overall Architecture

UnCLe operates in two stages:

1. **Training stage**: Learns a semantically disentangled representation of the time series along with autoregressive inter-variable dependency relationships
2. **Post-hoc analysis stage**: Infers dynamic causal graphs via temporal perturbation, or infers static causal graphs via dependency matrix aggregation

### Key Designs

1. **Uncoupler-Recoupler Network (Semantic Disentanglement Autoencoder)**

   **Core Idea**: Disentangles complex multivariate time series into multi-channel semantic representations, enabling inter-variable dependencies to be modeled linearly in the latent space.

    - **Uncoupler**: A parameter-shared TCN encoder mapping each univariate time series $x_i \in \mathbb{R}^T$ to a $C$-channel latent representation $z_i \in \mathbb{R}^{T \times C}$
    - **Recoupler**: A parameter-shared TCN decoder reconstructing the original series $\tilde{x}_i$ from latent representations
    - **Parameter sharing** is critical: all $N$ variables share a single set of TCN parameters, substantially improving learning efficiency and generalization (ablations show significant performance degradation without it)
    - Causal convolutions in the TCN ensure no information leaks from the future to the past

   **Theoretical Motivation**: An appropriate coordinate transformation (learned by the Uncoupler) can approximate complex nonlinear dynamics as a linear system—the central insight of Koopman operator theory.

2. **Auto-regressive Dependency Matrices**

   **Core Idea**: In the disentangled latent space, simple linear matrices model inter-variable dependencies.

    - Maintains $C$ dependency matrices $\Psi = \{\Psi^1, \ldots, \Psi^C\}$, where each $\Psi^c \in \mathbb{R}^{N \times N}$
    - Autoregressive prediction: $\hat{z}^c_{:,t+1} = \sigma(\Psi^c \cdot z^c_{:,t})$
    - The predicted latent representation is fed into the Recoupler to generate predictions in the original space
    - $L_1$ regularization encourages sparsity in the dependency matrices, suppressing spurious causal connections

3. **Dynamic Granger Causal Inference via Temporal Perturbation**

   **Core Idea**: If $x_j$ is a true cause of $x_i$, randomly permuting the temporal structure of $x_j$ will substantially increase the model's prediction error for $x_i$.

   Procedure:
    - Apply a random temporal permutation to variable $j$ (preserving marginal distribution while destroying temporal information)
    - Compute the per-timestep prediction error increment before and after perturbation: $\Delta\varepsilon^{\setminus j}_{i,t} = \max(0,\ \varepsilon^{\setminus j}_{i,t} - \varepsilon_{i,t})$
    - The error increment serves as the dynamic causal strength of $x_j \to x_i$ at time $t$
    - Repeating this for all variable pairs and timesteps yields the time-resolved causal graph $\hat{G}^{\text{Pert}}$

   **Why temporal permutation over other perturbations?**
    - Zero-value masking distorts the data distribution, destabilizing the model
    - Noise injection cannot fully eliminate the original signal
    - Temporal permutation both eliminates predictive temporal information and perfectly preserves the marginal distribution of each variable

4. **Dependency Aggregation for Static Causal Graphs**

    - Aggregates dependency matrices $\Psi$ along the channel dimension via $L_2$-norm: $A^{\text{Agg}}_{l,k} = \sqrt{\frac{1}{C} \sum_c (\Psi^c_{l,k})^2}$
    - Efficiently obtains a static global causal graph without post-hoc perturbation analysis

### Loss & Training

- **Two-stage training**:
  1. Pre-training: Optimize reconstruction loss $L_{\text{Recon}}$ (MSE) only, training Uncoupler + Recoupler
  2. Full training: Joint optimization $L_{\text{Total}} = L_{\text{Recon}} + \alpha \cdot L_{\text{Pred}} + L_{L_1}$
- **$L_{\text{Recon}}$**: Reconstruction MSE loss
- **$L_{\text{Pred}}$**: Prediction MSE loss (autoregressive next-step prediction)
- **$L_{L_1}$**: $L_1$ regularization on dependency matrices to promote sparsity
- **Dropout 0.2**: Additional regularization for TCN training
- $\alpha$ is a hyperparameter weighting the prediction task

## Key Experimental Results

### Main Results

**Static Causal Discovery (Synthetic Dataset Lorenz96)**:

| Dataset | Metric | UnCLe(P) | CUTS+ | JRNGC | NeuralGC |
|---------|--------|----------|-------|-------|----------|
| Lorenz#1 ($p$=20, $F$=10) | AUROC | **.999** | .986 | .963 | .891 |
| Lorenz#2 ($p$=20, $F$=40) | AUROC | **.969** | .852 | .786 | .657 |
| Lorenz#3 ($p$=100, $T$=500) | AUROC | **.964** | .946 | .884 | Not scalable |

**Dynamic Causal Discovery**:

| Dataset | Metric | UnCLe(P) | GVAR | Static Best |
|---------|--------|----------|------|-------------|
| TVSEM (bivariate alternating causality) | AUROC | **1.000** | 0.733 | 0.467 |
| TVSEM | AUPRC | **1.000** | 0.400 | 0.300 |
| ND8 (8-variable nonlinear) | AUROC | **.921** | .723 | .905 |
| ND8 | AUPRC | **.633** | .220 | .799 |

**Human Motion Capture (MoCap) — Skeletal Connection Missing Rate**:

| Method | Missing Rate ↓ |
|--------|---------------|
| UnCLe | **.200** |
| GVAR | .622 |
| JRNGC | .600 |

### Ablation Study

| Configuration | Lorenz#3 AUROC | Note |
|---------------|---------------|------|
| Full UnCLe(P) | ~0.96 | Baseline |
| w/o parameter sharing | Significant drop | Parameter sharing critical in high-dimensional settings |
| w/o dependency matrices | <0.5 (random) | Model completely fails to learn causal structure |
| w/o prediction task | Very low | Reconstruction alone is insufficient to model inter-variable dependencies |

**Perturbation Strategy Comparison (Lorenz#1)**:

| Perturbation Strategy | AUROC | AUPRC | ACC |
|-----------------------|-------|-------|-----|
| Temporal permutation (Ours) | **.999** | **.996** | **.994** |
| Noise injection | .981 | .946 | .978 |
| Zero-value masking | .974 | .932 | .969 |
| No perturbation | .500 | .575 | .850 |

### Key Findings

- **Perfect dynamic causal discovery on TVSEM** (AUROC=1.0), while GVAR achieves only 0.733 and fails to capture causal direction reversals
- In **human motion analysis**, UnCLe's dynamic causal graphs are highly consistent with biomechanical knowledge: upper limbs dominate during takeoff → lower limbs coordinate during flight → whole-body involvement at landing
- Skeletal connection missing rate of only 0.200 (vs. ~0.6 for GVAR/JRNGC), indicating that UnCLe preserves fundamental anatomical structure
- Dependency matrices are the core component—removing them reduces AUROC to random chance; parameter sharing is essential for high-dimensional settings

## Highlights & Insights

- **A rare contribution to dynamic causal discovery**: The vast majority of existing methods only infer static causal graphs; UnCLe is among the few capable of generating time-resolved causal graphs
- **Elegant "disentanglement + linearization" design**: The TCN Uncoupler transforms nonlinear dynamics into a latent space amenable to linear modeling, while the dependency matrices capture inter-variable relationships via simple linear transformations—analogous to a discrete approximation of Koopman operator theory
- **Principled choice of perturbation analysis**: Temporal permutation is the optimal perturbation strategy as it simultaneously satisfies two conditions: eliminating temporal predictive information and preserving marginal distributions
- **Compelling MoCap case study**: Anchoring the dynamic causal graphs to the human skeletal structure reveals causal patterns across motion phases that align fully with the biomechanics literature
- **Excellent scalability**: Parameter sharing enables the model to handle 100+ variables, where most baseline methods fail or suffer severe performance degradation

## Limitations & Future Work

- **Lack of identifiability guarantees**: The authors explicitly acknowledge this as the primary limitation—no theoretical proof that latent linearization ensures causal faithfulness
- Inherent assumptions of Granger causality: The framework assumes no latent confounders; spurious causal connections may arise in the presence of unobserved variables
- Temporal perturbation must be applied separately for each variable, incurring $O(N)$ times the computational cost of a single forward pass
- Only uniformly sampled time series are supported; irregular time series are not handled
- Evaluation metrics for dynamic causal graphs remain limited—quantitatively measuring "the quality of causal evolution capture" is an open challenge

## Related Work & Insights

- **NeuralGC (Tank et al.)**: A classical neural Granger causality method; component-wise design is not scalable
- **GVAR**: The only prior method capable of generating dynamic causal graphs, but with far lower accuracy than UnCLe
- **CUTS+**: A scalable method with parameter sharing, but limited to static causal discovery
- **Koopman theory**: The idea of linearizing nonlinear dynamics inspired the design of the Uncoupler
- **Takeaway**: Causal discovery can be fruitfully approached from the perspective of "disentangled representations + linear modeling"; parameter sharing and perturbation analysis constitute simple yet effective engineering choices

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Dynamic causal discovery is an underexplored yet important problem; the combination of Uncoupler and perturbation analysis is novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation covering synthetic and real data, static and dynamic settings, scalability, ablations, and perturbation strategy comparisons
- Writing Quality: ⭐⭐⭐⭐ The MoCap case study is vivid and illustrative; mathematical formulations are clear
- Value: ⭐⭐⭐⭐⭐ Fills the gap in dynamic causal discovery; the method is concise and practical, with open-source code and data

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] One-Shot Knowledge Transfer for Scalable Person Re-Identification](../../ICCV2025/human_understanding/one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](../../ICCV2025/human_understanding/cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)
- [\[CVPR 2026\] LaScA: Language-Conditioned Scalable Modelling of Affective Dynamics](../../CVPR2026/human_understanding/lasca_language-conditioned_scalable_modelling_of_affective_dynamics.md)
- [\[ICCV 2025\] Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation](../../ICCV2025/human_understanding/dynamic_reconstruction_of_hand-object_interaction_with_distributed_force-aware_c.md)
- [\[CVPR 2026\] 4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction](../../CVPR2026/human_understanding/textit4dsurf_high-fidelity_dynamic_scene_surface_reconstruction.md)

</div>

<!-- RELATED:END -->
