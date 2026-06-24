---
title: >-
  [Paper Note] From Embedding to Control: Representations for Stochastic Multi-Object Systems
description: >-
  [ICLR 2026][Robotics][Controllable embedding] This paper proposes Graph Controllable Embeddings (GCE), which embeds the conditional distributions of stochastic multi-body systems into a Reproducing Kernel Hilbert Space (RKHS) to linearize non-linear dynamics. Combined with Graph Neural Networks and mean-field approximations for adaptive modeling of non-uniform interactions, it enables efficient control and few-shot generalization of stochastic…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "Controllable embedding"
  - "Reproducing Kernel Hilbert Space"
  - "Mean-field approximation"
  - "Graph Neural Networks"
  - "Stochastic dynamics"
  - "Linear control"
date: 2026-05-08
content_hash: 9c860606e456bf8c
---

# From Embedding to Control: Representations for Stochastic Multi-Object Systems

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=SZzpGvBRv5](https://openreview.net/forum?id=SZzpGvBRv5)  
**Code**: Released with supplementary materials (no independent repository link provided in the paper)  
**Area**: Robotics Control / Multi-body System Modeling / Representation Learning  
**Keywords**: Controllable embedding, Reproducing Kernel Hilbert Space, Mean-field approximation, Graph Neural Networks, Stochastic dynamics, Linear control  

## TL;DR
This paper proposes Graph Controllable Embeddings (GCE), which embeds the conditional distributions of stochastic multi-body systems into a Reproducing Kernel Hilbert Space (RKHS) to linearize non-linear dynamics. Combined with Graph Neural Networks and mean-field approximations for adaptive modeling of non-uniform interactions, it enables efficient control and few-shot generalization of stochastic, variable-topology multi-body systems using simple linear LQR controllers.

## Background & Motivation
- **Background**: In robotics, power grids, and autonomous systems, the controlled objects are often multiple interacting entities with non-linear stochastic dynamics in continuous state/action spaces. A mainstream "controllable embedding" strategy involves lifting states into a latent space where dynamics become approximately linear, allowing the application of mature linear control methods (e.g., LQR). Global linearization is represented by Koopman theory, while local linearization often utilizes VAEs to learn low-dimensional manifolds.
- **Limitations of Prior Work**: (1) Koopman operators were originally designed for **deterministic** dynamics, and their extension to stochastic settings is non-trivial; (2) Most Koopman/VAE methods treat the system as a **single entity**, ignoring the relational topology between objects, which causes the parameter count to grow **quadratically** with the number of objects, leading to overfitting and poor generalization; (3) GNN-based methods naturally model interactions but target **prediction rather than control**. The learned embeddings lack linear or locally linear structures, requiring either subsequent local linearization or complex non-linear control, which is often inefficient.
- **Key Challenge**: Existing works that combine graph representations with controllable embeddings (e.g., Compositional Koopman) **lack theoretical guarantees** in stochastic settings, generally assume **uniform neighbor interactions** (which are mis-specified in a probabilistic sense), and **lack sufficient validation of scalability and generalization** on large-scale/random graphs. To achieve both accurate modeling and effective control in stochastic multi-body systems, one must simultaneously address "stochasticity modeling + non-uniform interactions + scalable generalization."
- **Goal**: Construct a theoretically guaranteed controllable embedding framework that linearizes stochastic multi-body dynamics in an embedding space, supports simple linear control, scales seamlessly with the number of objects and topologies, and generalizes to unseen topologies in a few-shot manner.
- **Core Idea**: **Represent the conditional distributions of controlled stochastic dynamics directly using Hilbert space embeddings.** By embedding probability distributions into an RKHS, conditional expectations can be expressed in closed form by a linear "conditional embedding operator." This preserves non-linear expressiveness while allowing dynamics to evolve linearly in the RKHS. Furthermore, **mean-field approximation + GNN adaptive weights** are used to compress multi-body interactions into a form that can be estimated with low complexity.

## Method

### Overall Architecture
GCE unifies "modeling" and "control" within RKHS embeddings: it first uses a message-passing GNN to map the history/observations of each object into RKHS features. Then, a conditional embedding operator linearly propagates "history features + action features" to the next-step observation features, where non-uniform influences between objects are aggregated via adaptive Boltzmann–Gibbs weights under a mean-field approximation. Finally, optimal action sequences are synthesized by directly solving a quadratic cost in the linearized feature space using LQR.

```mermaid
flowchart LR
    A[Step t Multi-body Obs o_t<br/>Graph G=V,E] --> B[GNN Encoder<br/>Map to RKHS Feature ψ]
    B --> C[Mean-field Approx<br/>Adaptive Weights α Aggregate Neighbor history]
    C --> D[Cond. Embedding Operator<br/>History Block C_O|H + Action Block C_O|A]
    D --> E[Predict t+1 Obs Feature]
    E --> F[GNN Decoder pullback<br/>Back to Obs Space]
    D --> G[LQR Control in RKHS<br/>Solve Quadratic Cost]
    G -->|Optimal Action Sequence a_t| A
```

### Key Designs

**1. Hilbert Space Embedding of Conditional Distributions: The foundation for linearizing stochastic dynamics.** GCE does not explicitly estimate probability densities. Instead, it represents the conditional distribution of future observations $O_t$ given actions $a_t$ and history $h_t$ via the conditional expectation of feature maps: $\mathbb{E}[\psi^O_t \mid a_t, h_t] = \mathcal{C}_{O|AH}[\psi^h_t \otimes \psi^a_t]$. Here, $\psi$ is a feature map into the RKHS, and $\mathcal{C}_{O|AH}$ is a **linear** conditional embedding operator. Since the kernel is characteristic, this embedding uniquely determines the conditional distribution, avoiding density estimation. Crucially, non-linear stochastic evolution becomes a "linear operation on features" in the RKHS, enabling multi-step rollouts and control planning to be performed recursively in a linear space. Theorem 1 proves that the empirically estimated operator converges uniformly as the sample size approaches infinity, providing theoretical backing for existence and consistency.

**2. Decoupling History-Action Features: Making sequential action optimization tractable.** Directly using the tensor product $\psi^h_t \otimes \psi^a_t$ results in extremely high dimensionality and requires large sample sizes, and the entanglement of history and actions prevents step-by-step optimization of action sequences. Borrowing from the decomposition of joint distributions in exponential families, this work approximates the tensor product as a linear summation after concatenation: $\mathcal{C}_{O^i|A^jH^j}[\psi^{h,j}_t \otimes \psi^{a,j}_t] \approx \mathcal{C}_{O^i|H^j}\psi^{h,j}_t + \mathcal{C}_{O^i|A^j}\psi^{a,j}_t$. This step replaces the tensor product with concatenation, sacrificing high-order interaction terms for two benefits: computational complexity is significantly reduced, and the action representation is **decoupled** from history, making the search for an optimal action sequence a tractable linear optimization problem—a prerequisite for applying LQR.

**3. Adaptive Mean-field Approximation: Breaking uniform neighbor assumptions while maintaining linear complexity.** Even after decoupling, estimating all $\mathcal{C}_{O^i|A^jH^j}$ pairs remains $O(N^2)$. This paper uses a mean-field approximation to aggregate the "collective influence of all neighbors" into a weighted sum while allowing weights to vary. The interaction weight of neighbor $j$ on $i$ is given by a Boltzmann–Gibbs form: $\alpha^{i,j}_t = \frac{\exp(f(\psi^{h,i}_t,\psi^{h,j}_t))}{\sum_{k\in E(i)}\exp(f(\psi^{h,i}_t,\psi^{h,k}_t))}$, where $f$ is a pairwise negative potential function (parameterized by Gaussian/Laplace/vMF kernels or an MLP). Consequently, the history term is approximated by applying a **shared** operator to the aggregated features: $\sum_{j\in E(i)}\mathcal{C}_{O^i|H^j}\psi^{h,j}_t \approx \mathcal{C}_{O^i|H}\big(\sum_{j\in E(i)}\alpha^{i,j}_t\psi^{h,j}_t\big)$, reducing per-object history computation to constant time and overall complexity to $O(N)$. The action side $\mathcal{C}_{O^i|A^j}$ remains distinct for action optimization. The final observation feature expectation for each object is written as Eq.9. The paper compares four embedding forms (Tensor, Dense, Hom, Hom+Mean in Table 1) and proves that Hom+Mean achieves the best balance in sample complexity, computation time, and generalization.

**4. End-to-End Training and LQR Control in RKHS: Direct transformation of linear structures into controllers.** The encoder uses a message-passing GNN to map observations to RKHS features, while actions are linearly projected. Training involves two losses: a forward loss $L_{\mathrm{fwd}}$ in the feature space using the Hilbert-Schmidt norm to constrain predicted features to real ones, and a reconstruction loss $L_{\mathrm{rec}}$ using the same GNN as a decoder for pullback. In the control phase, an $M$-step quadratic cost $\min_{\{V^a_t\}}\mathbb{E}[\sum_t \|\hat\psi^O_t-\psi^o_*\|^2_{Q_1}+\|\psi^a_t\|^2_{Q_2}]$ is solved directly in the feature space. Because the embedding space is linear, this is a standard LQR problem requiring no additional non-linear controllers.

## Key Experimental Results
Four control environments: Rope (point-mass chain), Soft (soft robot composed of interconnected objects), Swim (soft robot swimming in fluid), and Power-Grid (random topology, 100-150 nodes). Metrics are control cost and control error $\|V^o_M-V^o_*\|/\|V^o_*\|$ averaged over 200 trials. Baselines include controllable embeddings without relational structure (VAE, PCC) and graph representation methods (KPM, CKO, GraphODE), with CKO being the current SOTA.

### Main Results (Swim: In-Distribution / Few-Shot Control Cost and Error)

| Method | ID Cost | ID Error | Few-Shot Cost | Few-Shot Error |
|---|---|---|---|---|
| VAE | 573.1 | 0.73 | 835.4 | 0.92 |
| PCC | 513.3 | 0.68 | 732.8 | 0.80 |
| GraphODE | 417.8 | 0.52 | 693.5 | 0.58 |
| KPM | 385.5 | 0.44 | 523.4 | 0.61 |
| CKO (SOTA) | 389.1 | 0.42 | 421.0 | 0.44 |
| Ours (vMF) | 392.7 | 0.45 | 452.3 | 0.43 |
| Ours (Laplace) | 403.1 | 0.46 | 435.7 | 0.45 |
| **Ours (Gaussian)** | **383.7** | **0.41** | **404.3** | **0.41** |

The Gaussian variant is optimal in both ID and Few-Shot settings. Compared to CKO, the advantage in few-shot generalization is more pronounced.

### Ablation Study
**Power-Grid under Different Noise (Random Graph 100-150 objects, Control Error, NaN=Instability)**

| Method | No Noise | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| GraphODE | 0.58 | 0.62 | NaN | NaN | NaN |
| KPM | 0.42 | 0.50 | NaN | NaN | NaN |
| CKO | 0.47 | 0.48 | 0.51 | 0.65 | 0.85 |
| **Ours (Gaussian)** | **0.21** | **0.27** | **0.39** | 0.63 | 0.83 |

**Sample Efficiency of Different Embedding Forms (Rope, Control Error vs. Training Trajectories)**

| Method | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Dense | 0.79 | 0.41 | 0.36 | 0.28 | 0.26 |
| Hom | 0.74 | 0.32 | 0.30 | 0.30 | 0.30 |
| **Hom + Mean** | **0.51** | **0.29** | **0.26** | **0.25** | **0.23** |

### Key Findings
- **Multi-body specific controllable embeddings are necessary**: VAE/PCC can fit single trajectories but fail to provide controllable structured features. GraphODE includes relational structure but lacks explicit controllable design, relying on auto-diff for local linearization, which yields sub-optimal results.
- **Theoretical predictions validated by experiments**: CKO is essentially the Hom subclass in GCE (where uniform weights are mis-specified), leading to faster error accumulation. KPM's polynomial features are not characteristic and cannot faithfully embed distributions, causing collapse under high noise (consistent with Theorem 1).
- **Non-uniform weighting in mean-field significantly boosts generalization**: Hom+Mean yields an error of 0.51 with only 1 trajectory, far outperforming Dense (0.79) and Hom (0.74), while consistently maintaining lower control costs.
- **Gaussian kernel is most robust**: Compared to direction-aligned vMF or slow-decaying Laplace, Gaussian provides smoother and more stable mean-field approximations.

## Highlights & Insights
- **Translating "Stochastic Multi-body Control" into "Linear Algebra in RKHS"**: The core insight is using RKHS embeddings for conditional distributions to bypass density estimation and transform stochastic non-linear dynamics into linear operators, returning control to mature tools like LQR.
- **Elegant Loop between Theory and Engineering**: From the consistency in Theorem 1 to the complexity proofs for tensor decoupling and mean-field approximation, every step has a probabilistic explanation and complexity analysis, rather than being purely empirical.
- **Unified Perspective Explaining Baseline Failures**: By showing CKO=Hom (mis-specified weights) and KPM features are non-characteristic, the paper integrates prior methods into its own framework and identifies their root failure causes, which is highly persuasive.
- **Scalability as a Major Highlight**: Reducing $O(N^2)$ to $O(N)$ allows the framework to operate on 100-150 node random topology power grids and generalize to unseen topologies with few-shot data.

## Limitations & Future Work
- **Pair-wise Interaction Limitation**: The current framework is restricted to pair-wise interactions. The authors note that extensions to hypergraphs or larger relational structures and using attention mechanisms for interaction weights are unexplored.
- **Energy Function Parameterization**: Neural network-based $f$ functions can be unstable in RKHS; currently, the method relies on analytical kernels (e.g., Gaussian), which might limit expressive capacity.
- **Homogeneity Assumption**: The shared operator $\mathcal{C}_{O|H}$ assumes all nodes follow the same interaction laws, which may need relaxation for highly heterogeneous real-world systems.
- **High Noise Robustness**: In Power-Grid scenarios with 10%/20% noise, the advantage over CKO narrows, indicating room for improvement in extreme noise robustness.

## Related Work & Insights
- **Koopman Lineage**: This work is a stochastic and theoretical upgrade to Compositional Koopman (Li et al. 2020), extending global linearization from deterministic to stochastic settings and identifying the limitations of classic Koopman in multi-body contexts.
- **Kernel Embedding Lineage**: RKHS embeddings of conditional distributions (Song, Fukumizu, Sriperumbudur, etc.) provide the methodological foundation. This paper moves from static inference to sequential control for controlled dynamics.
- **GNN Physical Simulation Lineage**: It builds on data-driven multi-body simulators (Battaglia, Sanchez-Gonzalez, etc.) but shifts focus from "prediction" to "control," which is the critical distinction.
- **Insights**: For researchers in embodied or multi-body control, the combination of "linearizing dynamics before control" + "mean-field aggregation for non-uniform interactions" is a valuable paradigm. The use of characteristic kernels to ensure distribution uniqueness also provides a theoretical standard for feature space selection.

## Rating
- Novelty: ⭐⭐⭐⭐ — Systematically combines RKHS embedding of conditional distributions + mean-field approximation + GNN for stochastic multi-body controllable embeddings with a unified theoretical framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers four diverse environments, ID/Few-Shot settings, multiple noise levels, sample efficiency, and bandwidth ablations. Power-Grid validates large-scale random topologies. Hardware verification is a missing piece.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic from theory to method to experiment. Using the proposed framework to explain baseline failures is very convincing, though operator notation is dense.
- Value: ⭐⭐⭐⭐ — Provides a scalable, guaranteed representation learning paradigm for stochastic multi-body control, applicable to robotics, power grids, and relational control problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Primer on SO(3) Action Representations in Deep Reinforcement Learning](a_primer_on_so3_action_representations_in_deep_reinforcement_learning.md)
- [\[ICLR 2026\] On Entropy Control in LLM-RL Algorithms](on_entropy_control_in_llm-rl_algorithms.md)
- [\[ICLR 2026\] Masked Generative Policy for Robotic Control](masked_generative_policy_for_robotic_control.md)
- [\[ICLR 2026\] Empowering Multi-Robot Cooperation via Sequential World Models](empowering_multi-robot_cooperation_via_sequential_world_models.md)
- [\[ICLR 2026\] MVR: Multi-view Video Reward Shaping for Reinforcement Learning](mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
