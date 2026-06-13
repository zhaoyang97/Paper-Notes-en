---
title: >-
  [Paper Note] Meta-learning Structure-Preserving Dynamics
description: >-
  [ICML 2026][Signal & Communication][Hamiltonian NN] This paper systematically introduces modulation-based meta-learning (where a hyper-network maps latent codes $\bm{z}^{(k)}$ to hierarchical modulation parameters) into…
tags:
  - "ICML 2026"
  - "Signal & Communication"
  - "Hamiltonian NN"
  - "GENERIC"
  - "Modulation Meta-learning"
  - "Low-rank Adaptation"
  - "SVD modulation"
date: 2026-05-08
content_hash: fa206afc6418342a
---

# Meta-learning Structure-Preserving Dynamics

**Conference**: ICML 2026  
**arXiv**: [2508.11205](https://arxiv.org/abs/2508.11205)  
**Code**: None  
**Area**: Scientific Machine Learning / Meta-learning / Structure-Preserving Neural Networks  
**Keywords**: Hamiltonian NN, GENERIC, Modulation Meta-learning, Low-rank Adaptation, SVD modulation

## TL;DR
This paper systematically introduces modulation-based meta-learning (where a hyper-network maps latent codes $\bm{z}^{(k)}$ to hierarchical modulation parameters) into Hamiltonian and GENERIC neural networks. It proposes two novel modulations—latent multi-rank (MR) and latent SVD-like modulation—enabling a shared network to adapt to entire families of new parameter instances with few shots without knowing the system parameters $\bm{\mu}$, while strictly maintaining energy conservation/dissipation structures.

## Background & Motivation

**Background**: Structure-preserving neural networks (HNN, LNN, port-Hamiltonian NN, GENERIC/metriplectic NN) hardcode conservation laws, symplectic structures, and dissipation laws into their architectures, allowing physically faithful predictions for dynamical systems with known parameters $\bm{\mu}$.

**Limitations of Prior Work**: Existing models are mostly "one model per parameter instance." If parameters change slightly, the models require retraining, making many-query scenarios (e.g., families of pendulums with different masses or oscillators with different stiffness) prohibitively expensive. Few meta-learning extensions (Lee 2021, Song 2024) follow MAML/ANIL paths, requiring unstable and inefficient high-dimensional inner-loop parameter updates.

**Key Challenge**: HNN-style models only need to learn a scalar potential $\mathcal{H}_\Theta(\bm{q}, \bm{p})$ to describe complete dynamics, and the dependency of weights on parameters $\bm{\mu}$ is naturally low-dimensional. Existing meta-learning methods waste this low-dimensional structure by updating all parameters $\Theta$ via full gradients.

**Goal**: (1) Systematically compare various modulation strategies within Hamiltonian/GENERIC frameworks; (2) Design more expressive yet parameter-efficient modulation methods; (3) Ensure that the conservation/dissipation structures are strictly preserved after modulation.

**Key Insight**: Borrowing from latent modulation in INRs/NeRFs (e.g., CODA by Dupont 2022), each system is compressed into a low-dimensional latent code $\bm{z}^{(k)}$. A hyper-network $\bm{f}_\text{hyper}(\bm{z}^{(k)}; \bm\phi)$ then generates small corrections for each layer, while the base weights are shared across all tasks.

**Core Idea**: The combination of "shared base + instance latent + hierarchical low-rank modulation" can capture the "low-dimensional manifold of parameters $\bm{\mu}$" with minimal trainable parameters. Using SVD-like decomposition further learns orthogonal bases during the base stage, reducing test-time adaptation to updating only a few singular value scalars.

## Method

### Overall Architecture
The input is a family of Hamiltonian/GENERIC systems $\{\mathcal{H}^{(k)}(\bm{q}, \bm{p}) = \mathcal{H}(\bm{q}, \bm{p}; \bm{\mu}^{(k)})\}_{k=1}^{n_\mu}$, with trajectories sampled for each. Model parameters are split into $\Theta^{(k)} = \Theta_\text{base} \cup \Theta_\text{indv}^{(k)}$. The base parameters are updated by meta-gradients in the outer loop, while individual parameters (latent codes $\bm{z}^{(k)}$ for each system) are updated in the inner loop. The hyper-network maps $\bm{z}^{(k)}$ to low-rank or bias correction parameters for each layer. The final $\tilde{\mathcal{H}}(\bm{q}, \bm{p}; \Theta^{(k)})$ serves as a latent-conditioned energy function, providing dynamics via $\dot{\bm q} = \partial \tilde{\mathcal H} / \partial \bm p,\ \dot{\bm p} = -\partial \tilde{\mathcal H} / \partial \bm q$. Thus, structure preservation is inherited from the base architecture.

### Key Designs

1.  **Latent Multi-Rank (MR) Modulation**:
    - **Function**: Adds an instance-specific correction of rank $r$ ($\bm{U}^{(\ell,k)} \bm{V}^{(\ell,k)\top}$) and a bias correction $\bm{s}^{(\ell,k)}$ to the MLP weights $\bm{W}^{(\ell)}$ at each layer, all generated from $\bm{z}^{(k)}$ via the hyper-network.
    - **Mechanism**: Each layer is updated as $\bm{h} \mapsto \sigma\left((\bm{W}^{(\ell)} + \bm{U}^{(\ell,k)} \bm{V}^{(\ell,k)\top}) \bm{h} + \bm{b}^{(\ell)} + \bm{s}^{(\ell,k)}\right)$, where $\bm{U}, \bm{V} \in \mathbb{R}^{w_\ell \times r}$. When $r=1$, it reduces to RO (rank-one), which is equivalent to a minimalist LoRA-like modulation. MR(5) uses $r=5$. $\bm{U}$ and $\bm{V}$ are instance-specific, meaning the hyper-network regenerates the rank-$r$ factors for each instance.
    - **Design Motivation**: Proposition 3.1 shows that if the local rank of $\partial_{\bm\mu} \bm{f} \le r$, an $r$-dimensional modulation is sufficient to capture all local parameter variations. MR leverages this by placing expressivity in "LoRA-style low-rank matrices."

2.  **Latent SVD-like Modulation (Best Solution)**:
    - **Function**: Further factorizes low-rank modulation into "shared bases + instance singular values," allowing the hyper-network to output only a few scalars.
    - **Mechanism**: Each layer is formulated as $\bm{h} \mapsto \sigma\left((\bm{W}^{(\ell)} + \sum_{i=1}^r d_i^{(\ell,k)} \bm{u}_i^{(\ell)} \bm{v}_i^{(\ell)\top}) \bm{h} + \bm{b}^{(\ell)} + \bm{s}^{(\ell,k)}\right)$. Here, $\bm{u}_i^{(\ell)}, \bm{v}_i^{(\ell)}$ are base parameters (updated via meta-gradients), while only the singular values $d_i^{(\ell,k)}$ and offsets $\bm{s}^{(\ell,k)}$ are generated from $\bm{z}^{(k)}$ by the hyper-network. Soft orthogonality penalties $\|\bm{U}^\top \bm{U} - \bm{I}\|_F$ and $\|\bm{V}^\top \bm{V} - \bm{I}\|_F$ plus ReLU activation in the hyper-network ensure non-negative singular values.
    - **Design Motivation**: The base stage learns "cross-system invariant modulation directions" into $\bm{u}_i, \bm{v}_i$. During testing, only a few singular values need to be fitted to adapt to new instances. This mirrors the successful INR pattern of "learning shared bases then fitting individual coefficients."

3.  **Locality Regularization + Evolving Latent Code Protocol**:
    - **Function**: (a) Constrains instance parameters from deviating too far from the base; (b) Keeps $\bm{z}^{(k)}$ evolving across training rather than resetting at each epoch.
    - **Mechanism**: Adding $\lambda_z \|\bm{z}\|_2 + \lambda_\phi \|\bm\phi\|_2$ to the loss keeps updates near the shared base. At test time, latents are initialized to the Euclidean mean of training latents, $\bm{z}_\text{avg} = \tfrac{1}{n_\mu^\text{train}} \sum_k \bm{z}_\text{train}^{(k)}$, followed by few-shot auto-decoding.
    - **Design Motivation**: Zero-initialization (as in CODA) pushes the base to accommodate arbitrary latents, losing training signals. Evolving latents allow the base and the mean latent to co-evolve, ensuring check-time initialization falls within a "learned parameter neighborhood."

### Loss & Training
Hamiltonian systems use a symplecticity loss $\mathcal{L}_\text{symp} = \|\dot{\bm q} - \partial_{\bm p} \tilde{\mathcal H}_\Theta\|_2^2 + \|\dot{\bm p} + \partial_{\bm q} \tilde{\mathcal H}_\Theta\|_2^2$, while GENERIC systems use the corresponding metriplectic loss. The outer loop performs $N_\text{out}$ updates on $\Theta_\text{base}$, and the inner loop performs $N_\text{in}$ updates on the batch's latents. Test-time adaptation utilizes Algorithm 2 for an $N_\text{test}$-shot latent fit with a frozen base.

## Key Experimental Results

### Main Results
Testing on three conservative systems (Duffing, mass-spring, pendulum) and one dissipative system (DNO). 80 parameter instances per system (70 train / 10 test), 10 trajectories each. Metrics: $\epsilon_\text{field}$ (relative $\ell^2$ error on uniform grid, OOD metric), $\epsilon_\text{traj}$ (relative error of test trajectories).

| System | Method | $\epsilon_\text{field}$ ($\times 10^{-2}$) | $\epsilon_\text{traj}$ ($\times 10^{-2}$) |
|------|------|--------------------------------------------|-------------------------------------------|
| Pendulum | Scratch | 83.35 | 79.84 |
| Pendulum | MAML | 99.13 | 52.37 |
| Pendulum | Reptile | 88.72 | 75.73 |
| Pendulum | FW (CODA) | 8.23 | 10.65 |
| Pendulum | Shift | 9.76 | 12.88 |
| Pendulum | RO (MR-1) | 6.47 | 8.27 |
| Pendulum | **SVD(5)** | **4.62** | **5.33** |
| Mass Spring | FW | 1.60 | 1.31 |
| Mass Spring | **SVD(5)** | **1.51** | **1.12** |
| Duffing | FW | 10.30 | 2.78 |
| Duffing | **SVD(5)** | **10.03** | **2.30** |

### Ablation Study

| Configuration | Pendulum $\epsilon_\text{field}$ | Notes |
|------|----------------------------------|------|
| Multi-domain training (Duffing + spring + pendulum) | SVD(5) still best | Modulation works across different dynamics families |
| Variable shot counts | SVD consistently best (1 to 300 shots) | Strong few-shot adaptation |
| Locality weight $\lambda_\phi, \lambda_z$ scans | SVD has lowest variance | Robust to regularization strength |
| Latent init (zero vs $\bm z_\text{avg}$) | $\bm z_\text{avg}$ always superior | Validates evolving-latent protocol |
| Dissipative DNO system | SVD(3) $\epsilon_\text{traj} = 0.142$ | Reptile/ANIL NaN or diverge; SVD remains stable |

### Key Findings
- Modulation-based methods (FW/Shift/MR/RO/SVD) overall reduce errors by ~65% compared to optimization-based methods (MAML/Reptile/ANIL). This suggests that for structure-preserving networks where weight dependency is low-dimensional, modulation is more efficient than inner gradients.
- SVD(5) is not only the most accurate but also has a significantly smaller hyper-network than FW (FW outputs the full matrix, while SVD outputs $r$ scalars), achieving the "accuracy/params" Pareto optimum.
- MAML-style methods diverge on the dissipative DNO system, whereas modulation methods remain stable because they do not modify the main weights via high-variance inner loops.
- Multi-domain training proves that the same base network can switch between Duffing, spring, and pendulum dynamics purely via latent code modulation.

## Highlights & Insights
- Decomposing parameters into "shared base + latent SVD" provides "interpretability" to the latent modulation found in INRs—individual singular values reveal the importance of specific principal modulation directions for a given instance.
- Proposition 3.1 provides a clean theoretical justification for low-rank modulation: the local rank of the parameter space determines the required modulation dimensions.
- Combining modulation with Hamiltonian/GENERIC frameworks is naturally robust; modulation only alters the scalar value of $\mathcal{H}_\Theta$ and does not break the symplectic or metriplectic structure.
- The evolving-latent protocol is a critical detail: keeping the base synchronized with used latents prevents distribution shift.

## Limitations & Future Work
- Experiments are restricted to low-dimensional "toy" systems ($\le 4$ dimensions). Scalability to PDEs or high-dimensional multi-body systems (e.g., molecular dynamics) remains unproven.
- Modulation was only applied to MLP layers; more general architectures like Transformers or GNNs were not explored.
- SVD orthogonality relies on soft penalties; convergence sensitivity regarding base orthogonality was not fully discussed.
- Interaction with long-horizon stability and specific symplectic integrators requires more systematic analysis.

## Related Work & Insights
- **vs MAML / Reptile / ANIL**: This work replaces high-dimensional inner-loop gradient updates with low-dimensional auto-decoding of latent codes, reducing error by 65%.
- **vs CODA / FW (Kirchmeyer 2022)**: FW modulates all $\bm{W}, \bm{b}$, resulting in a massive hyper-network. MR/SVD outperform FW with far fewer parameters.
- **vs Shift modulation (Dupont 2022)**: Shift only adjusts biases, which is too weak. SVD-like modulation retains biases while adding shared rank-$r$ matrices for higher expressivity.
- **vs LoRA**: While LoRA is a task-agnostic fine-tuning method, MR is essentially task-conditioned LoRA driven by a hyper-network, upgrading fine-tuning to meta-learning.

## Rating
- Novelty: ⭐⭐⭐ Applying LoRA/SVD-style modulation to structure-preserving meta-learning is a solid combination.
- Experimental Thoroughness: ⭐⭐⭐ Cover 4 systems and 6 baselines, but the system dimensionality is low.
- Writing Quality: ⭐⭐⭐⭐ Clear formulas and algorithm blocks; Prop 3.1 provides rigorous grounding.
- Value: ⭐⭐⭐⭐ Provides a simple, reusable meta-learning template for many-query SciML scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](../../ICCV2025/signal_comm/boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[CVPR 2026\] Dual-Imbalance Continual Learning for Real-World Food Recognition](../../CVPR2026/signal_comm/dual-imbalance_continual_learning_for_real-world_food_recognition.md)
- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](../../NeurIPS2025/signal_comm/feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[NeurIPS 2025\] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](../../NeurIPS2025/signal_comm/contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)

</div>

<!-- RELATED:END -->
