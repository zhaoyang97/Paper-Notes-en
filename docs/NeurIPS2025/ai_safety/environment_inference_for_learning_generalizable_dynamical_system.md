---
title: >-
  [Paper Note] Environment Inference for Learning Generalizable Dynamical System
description: >-
  [NeurIPS 2025][AI Safety][dynamical systems] This paper proposes DynaInfer, a framework that infers environment labels for unlabeled trajectories by analyzing the prediction errors of a fixed neural network…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "dynamical systems"
  - "environment inference"
  - "OOD generalization"
  - "multi-environment learning"
  - "K-means analogy"
date: 2026-05-08
content_hash: fa1eb2aabf69656f
---

# Environment Inference for Learning Generalizable Dynamical System

**Conference**: NeurIPS 2025
**arXiv**: [2510.19784](https://arxiv.org/abs/2510.19784)  
**Code**: [GitHub](https://github.com/shixuanliu-andy/DynaInfer)  
**Area**: AI Safety / Dynamical System Generalization
**Keywords**: dynamical systems, environment inference, OOD generalization, multi-environment learning, K-means analogy

## TL;DR
This paper proposes DynaInfer, a framework that infers environment labels for unlabeled trajectories by analyzing the prediction errors of a fixed neural network, enabling generalizable dynamical system learning without environment annotations. DynaInfer matches or surpasses Oracle (known-label) performance on ODE/PDE systems.

## Background & Motivation

**Background**: Data-driven methods such as Neural ODE and FNO have achieved success in modeling complex dynamical systems, but typically assume i.i.d. training data. Recent works (LEADS, CoDA) introduce multi-environment settings—decomposing dynamics into globally shared and environment-specific components—to improve generalization.

**Limitations of Prior Work**: These generalization methods **rely on environment labels**, i.e., they require knowledge of which environment each trajectory belongs to. In practice, such labels are often unavailable: data in scientific experiments may be collected under unknown conditions; labels are lost when aggregating data from multiple sources; and environment information is restricted in privacy-sensitive domains such as healthcare and finance.

**Key Challenge**: Multi-environment generalization methods require environment labels to disentangle shared and environment-specific dynamics, yet such labels are difficult to obtain during data collection. The key challenge is: how can meaningful environment partitions be inferred **without any labels**?

**Goal**: To infer environment assignments for mixed trajectories when environment labels are entirely absent during training—even when the number of environments $M$ is unknown—so that downstream generalization algorithms can function properly.

**Key Insight**: Trajectories from the same environment follow the same governing equation and should therefore produce similar prediction errors under the same neural network. The consistency of prediction errors can thus serve as a signal for inferring environment membership—analogous to minimizing the distance from data points to centroids in K-means clustering.

**Core Idea**: The neural network is treated as the analog of a K-means "centroid," and the trajectory prediction loss as the analog of "distance." Environment labels are inferred by alternately updating environment assignments and network parameters.

## Method

### Overall Architecture
DynaInfer is an iterative framework:
- **Input**: $N$ mixed trajectories (no environment labels), with a assumed number of environments $M$
- **Output**: Environment assignments $\hat{e}$, global parameters $\theta$, environment-specific parameters $\phi$
- **Pipeline**: Random initialization → Alternating steps (environment inference + parameter optimization) → Convergence

### Problem Formulation

A dynamical system is described by the differential equation $dx_t/dt = f(x_t)$. In the multi-environment setting, the dynamics of each trajectory $x^i$ are given by $dx^i_t/dt = h(x^i_t; \theta, \phi_{e_i})$, where $\theta$ denotes global parameters and $\phi_{e_i}$ denotes environment-specific parameters. The optimization objective is:

$$\hat{e}^*, \theta^*, \phi^* = \arg\min_{\hat{e}, \theta, \phi} R_{\hat{e}}(\theta, \phi) = \sum_{i=1}^N \int_{t \in I} \left\|\frac{dx^i_t}{dt} - h(x^i_t; \theta, \phi_{\hat{e}_i})\right\|_2^2 dt + \lambda \sum_{e=1}^M \Omega(\phi_e)$$

### Key Designs

1. **Bias-aware Environment Assignment**:

    - **Function**: At iteration $r$, infers the optimal environment label for each trajectory using the network parameters fixed from the previous round.
    - **Mechanism**: $\hat{e}_i^{(r)} = \arg\min_{e \in [M]} \int_{t \in I} \|dx^i_t/dt - h(x^i_t; \theta^{(r-1)}, \phi_e^{(r-1)})\|_2^2 dt$
    - Each trajectory is assigned to the environment that minimizes its prediction loss—analogous to assigning data points to their nearest centroid in K-means.
    - **Design Motivation**: Trajectories from the same environment share the same governing equation, yielding consistent error patterns under the same network.

2. **Assignment-driven Optimization**:

    - **Function**: Optimizes global and environment-specific parameters based on the current environment assignments.
    - **Mechanism**: $\theta^{(r)}, \phi^{(r)} = \arg\min_{\theta, \phi} R_{\hat{e}^{(r)}}(\theta, \phi)$
    - Analogous to updating centroids as the within-cluster mean in K-means.
    - **Design Motivation**: Parameters are updated in an unbiased manner, with each trajectory contributing equally.

3. **Theoretical Guarantee (Proposition 3.1)**:

    - The loss is monotonically non-increasing across iterations: $R_{\hat{e}^{(r+1)}}(\theta^{(r+1)}, \phi^{(r+1)}) \leq R_{\hat{e}^{(r)}}(\theta^{(r)}, \phi^{(r)})$
    - If the loss strictly decreases, each step reduces it by at least a constant $C > 0$, guaranteeing convergence within a finite number of iterations.
    - Under the assumptions that $h$ is linear in $\theta$ and $\phi_e$ and that $\Omega$ is strictly convex, the solution space of $\arg\min_{\theta,\phi} R_{\hat{e}}(\theta, \phi)$ is finite, satisfying the convergence conditions.

### Test-time Environment Inference
- For in-domain testing without labels: a short segment of less than $2\Delta t$ from each trajectory is evaluated across all environment-specific networks, and the environment with the lowest prediction error is selected.
- For domain adaptation: fine-tuning is performed using labeled samples from the target domain.

## Key Experimental Results

### In-domain Generalization (Test MSE / MAPE)

| Dataset | Assignment Strategy | LEADS MSE | CoDA-$l_1$ MSE | CoDA-$l_2$ MSE |
|---------|---------------------|-----------|----------------|----------------|
| LV | All in One | 7.41E-2 | 7.40E-2 | 7.41E-2 |
| LV | Random | 7.38E-2 | 7.39E-2 | 7.39E-2 |
| LV | One per Env | 4.91E-4 | 9.14E-4 | 8.43E-4 |
| LV | **DynaInfer** | **7.93E-5** | **1.83E-4** | **1.82E-4** |
| LV | Oracle | 7.02E-5 | 3.19E-5 | 2.72E-5 |
| GS | DynaInfer | **4.14E-5** | 1.23E-4 | 7.25E-5 |
| GS | Oracle | 1.34E-4 | 9.60E-5 | 7.04E-5 |
| NS | DynaInfer | **7.05E-3** | 1.62E-2 | 1.19E-2 |
| NS | Oracle | 6.55E-3 | 1.73E-2 | 9.46E-3 |

### Domain Adaptation Results

| Dataset | Assignment Strategy | LEADS MAPE | CoDA-$l_1$ MAPE | CoDA-$l_2$ MAPE |
|---------|---------------------|------------|-----------------|-----------------|
| LV | All in One | 9.92 | 26.90 | 27.80 |
| LV | DynaInfer | **2.84** | **10.16** | **10.30** |
| LV | Oracle | 3.16 | 8.27 | 10.61 |
| NS | DynaInfer | **77.29** | 108.17 | **96.57** |
| NS | Oracle | 67.58 | 124.22 | 91.06 |

### Key Findings
- DynaInfer **consistently and substantially outperforms** all other label-free strategies (All in One, Random, One per Env) across all datasets and backbone models.
- On GS and NS datasets, DynaInfer **surpasses Oracle** (LEADS backbone), suggesting that bias-aware environment inference can compensate for the limitations of manual annotation.
- "All in One" and "Random" strategies nearly completely fail (MSE is approximately 3 orders of magnitude higher), confirming that environment partitioning is critical for generalization.
- Environment labels converge rapidly to the ground-truth assignments, validating the effectiveness of the K-means analogy.

## Highlights & Insights
- The **K-means analogy** is elegant and powerful: neural network ≈ centroid, prediction loss ≈ distance, alternating optimization ≈ EM algorithm. This transfers clustering intuition from Euclidean space to function space.
- The **surpassing Oracle** phenomenon is surprising: it suggests that manually annotated environment boundaries may not be optimal for learning, and data-driven partitions may be superior.
- The framework is **model-agnostic**: it can be combined with any backbone generalization model such as LEADS or CoDA, functioning as a general upstream module.
- The theoretical guarantees (monotonic convergence + constant-step descent) are simple yet practically useful.

## Limitations & Future Work
- The number of environments $M$ must be specified in advance; although the paper claims robustness to the choice of $M$, no method for automatically determining $M$ is provided.
- The convergence analysis relies on the assumptions that $h$ is linear in the parameters and that $\Omega$ is strictly convex, which do not fully hold for deep nonlinear networks.
- When inter-environment differences are small (e.g., slight variations in physical parameters), prediction losses may lack sufficient discriminability.
- Test-time environment inference relies on early trajectory segments, which may be unstable for systems sensitive to initial conditions.

## Related Work & Insights
- **vs. LEADS**: LEADS first proposed multi-environment generalization for dynamical systems but assumes known environment labels; DynaInfer removes this assumption.
- **vs. CoDA**: CoDA offers functional and parametric decomposition modes, both requiring labels; DynaInfer serves as an upstream tool to supply those labels.
- **vs. Classical Clustering (K-means in Euclidean space)**: DynaInfer extends the clustering paradigm to function space, replacing Euclidean distance with prediction loss.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to address the missing environment label problem in dynamical system generalization; the K-means analogy is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage of ODE and PDE systems, both in-domain and adaptation settings, with 3 backbone models × 5 assignment strategies.
- **Writing Quality**: ⭐⭐⭐⭐ Framework description is clear, the algorithm is concise, and the correspondence between theory and experiments is well-established.
- **Value**: ⭐⭐⭐⭐ Addresses the pervasive label-scarcity problem in practical scenarios; strong practical utility as a plug-in module.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection](../../ICCV2025/ai_safety/vulnerability-aware_spatio-temporal_learning_for_generalizable_deepfake_video_de.md)
- [\[NeurIPS 2025\] Fair Representation Learning with Controllable High Confidence Guarantees via Adversarial Inference](fair_representation_learning_with_controllable_high_confidence_guarantees_via_ad.md)
- [\[NeurIPS 2025\] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)
- [\[NeurIPS 2025\] DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning](design_encrypted_gnn_inference_via_server-side_input_graph_pruning.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)

</div>

<!-- RELATED:END -->
