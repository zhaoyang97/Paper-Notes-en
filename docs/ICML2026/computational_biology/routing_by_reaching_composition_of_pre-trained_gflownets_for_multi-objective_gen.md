---
title: >-
  [Paper Note] Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation
description: >-
  [ICML 2026][Computational Biology][GFlowNets] This paper proposes a training-free GFlowNet composition framework. By using the "reaching probability" of each pre-trained model as a weight to mix their respective forward…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "GFlowNets"
  - "Multi-Objective Generation"
  - "Model Composition"
  - "Test-time Mixing"
  - "Molecular Design"
date: 2026-05-08
content_hash: 043c50efb9dc27ac
---

# Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation

**Conference**: ICML 2026  
**arXiv**: [2602.21565](https://arxiv.org/abs/2602.21565)  
**Code**: https://github.com/ml-postech/gflownet-composition (Available)  
**Area**: Scientific Computing / Generative Flow Networks / Molecule Generation  
**Keywords**: GFlowNets, Multi-Objective Generation, Model Composition, Test-time Mixing, Molecular Design  

## TL;DR
This paper proposes a training-free GFlowNet composition framework. By using the "reaching probability" of each pre-trained model as a weight to mix their respective forward policies, it enables direct sampling for arbitrary multi-objective compositions of linear scalarization or logic operators during the inference stage. It is theoretically proven to exactly recover the target distribution in linear cases.

## Background & Motivation

**Background**: GFlowNets are a class of generative models trained with the objective of "sampling diverse candidates proportional to rewards." They are particularly popular in scientific discovery tasks for discrete structures like molecules, graphs, and biological sequences because, unlike traditional optimization that converges to a single point, they cover the entire high-reward region. When a problem involves multiple rewards (e.g., binding affinity SEH, synthetic accessibility SA, drug-likeness QED), mainstream approaches follow two paths: (i) training a preference-conditioned model (MOGFN, HN-GFN) on a weighted sum of rewards, or (ii) composing pre-trained single-objective models into logic operators (e.g., harmonic mean for "conjunction," contrastive operator for "difference") via classifier guidance (compositional sculpting).

**Limitations of Prior Work**: Both paths share major drawbacks—any modification to the set of objectives requires retraining. In preference-conditioned methods, the reward set is frozen during training; adding a new objective requires retraining from scratch. While classifier guidance does not modify the GFlowNet, a new auxiliary classifier must be trained for every new combination, and the cost of training this classifier often exceeds that of training a standalone GFlowNet. Furthermore, linear scalarization and logic operators are treated as two incompatible sub-problems without a unified framework.

**Key Challenge**: A trade-off exists between the "universality" of composition and "training cost." Supporting arbitrary combinations typically requires training after knowing the operator form, while avoiding training requires sacrificing flexibility. Both approaches are fundamentally based on "distribution-level operations" and lack a tool to compose directly within the internal structure of GFlowNets—the forward policy.

**Goal**: (i) Treat several pre-trained single-objective GFlowNets as "building blocks" for zero-shot composition to form arbitrary distributions at inference time; (ii) Support both linear scalarization and logic operators with the same mechanism; (iii) Provide theoretical guarantees.

**Key Insight**: The authors observe that the marginal distribution of a GFlowNet on a terminal state $x$ can be decomposed into the product of the "reaching probability $u(s)$" and the "termination action probability $p_{F}(s_{f}\mid x)$," i.e., $p(x)=u(x)\cdot p_{F}(s_{f}\mid x)$. The reaching probability naturally characterizes "how much probability mass a model routes to state $s$"—representing how "relevant" that state is to that specific model.

**Core Idea**: Use the reaching probability of each pre-trained GFlowNet at the current state as a local weight to mix their forward transition probabilities according to the target composition function $\mathcal{G}$. It is theoretically proven that this rule exactly recovers the target distribution under linear scalarization, while for non-linear operators, it exhibits controllable distortion in high-density regions.

## Method

### Overall Architecture
Given $k$ pre-trained GFlowNets, where the $i$-th model targets reward $R_{i}(x)$ with forward policy $p_{i,F}$, reaching probability $u_{i}$, and termination distribution $p_{i}\propto R_{i}$. For a target composition $p_{M}^{*}(x)\propto\mathcal{G}(p_{1}(x),\dots,p_{k}(x))$, the framework constructs a mixed forward policy $p_{M,F}$ during inference. This ensures that the terminal distribution $p_{M}(x)$ sampled via the DAG sequence equals (or approximates) $p_{M}^{*}$ without introducing new parameters or additional training.

The overall pipeline is: at each non-sink state $s$, for each candidate child $s'$, the "reaching probability × forward transition probability" of each model is used as a weighted sample fed into $\mathcal{G}$, then divided by a local normalization constant to obtain the transition probability of the mixed policy. The next state is then sampled according to this new policy until a terminal state is reached.

### Key Designs

1.  **Reaching-Weighted Mixing**:
    - **Function**: Merges multiple single-objective forward policies into a unified multi-objective policy.
    - **Mechanism**: For each child $s'$ of state $s$, define $p_{M,F}(s'\mid s)=\mathcal{G}\bigl(u_{1}(s)p_{1,F}(s'\mid s),\dots,u_{k}(s)p_{k,F}(s'\mid s)\bigr)/N_{M}(s)$, where $N_{M}(s)$ is the normalization constant summing these quantities over all children. Intuitively, $u_{i}(s)$ measures "how much probability mass model $i$ sent to $s$." The model that sends more mass is more "familiar" with this region of the state space and should have a greater "voice" at $s$.
    - **Design Motivation**: Naive uniform mixing (setting all $u_{i}$ to $1$) is equivalent to an "ensemble baseline," which ignores the relative importance of each model for the current trajectory. Weighting by reaching probability ensures each model's "contribution" is proportional to its "presence" on the current trajectory, which is key to the theoretical closure.

2.  **Flow-Based $u_{i}$ Computation**:
    - **Function**: Efficiently retrieves $u_{i}(s)$ in large state spaces for immediate use during inference.
    - **Mechanism**: Computing $u(s)=\sum_{s_{*}}u(s_{*})p_{F}(s\mid s_{*})$ requires iterating over all parent states, which is infeasible. The authors utilize the identity $u_{i}(s)=F_{i}(s)/Z_{i}$, where $F_{i}(s)$ is the state flow and $Z_{i}=F_{i}(s_{0})$ is the total flow. If the training objective explicitly parametrizes flow (flow matching / detailed balance / sub-trajectory balance), the "Model $F$" path is used. If training used objectives like trajectory balance that do not explicitly model $F$, the detailed balance condition $F_{i}(s')p_{i,B}(s\mid s')=F_{i}(s)p_{i,F}(s'\mid s)$ allows online recovery of $F_{i}$ by cumulative products $\prod_{j} p_{i,F}/p_{i,B}$ along the current trajectory, referred to as the "DB $F$" path.
    - **Design Motivation**: This covers the two major categories of GFlowNet training objectives, making the framework agnostic to the training paradigm of pre-trained models and maximizing real-world "building block reuse."

3.  **Distortion-Factor Analysis**:
    - **Function**: Explains what distribution the mixed policy actually recovers and when it is exact versus approximate.
    - **Mechanism**: The marginal distribution resulting from a full trajectory of the mixed policy is expanded as $p_{M}(x)=\delta(x)\cdot\mathcal{G}(p_{1}(x),\dots,p_{k}(x))$, where the distortion factor $\delta(x)=u_{M}(x)/N_{M}(x)$ measures the deviation from the target. For linear scalarization $\mathcal{G}=\sum_{i}\omega_{i} Z_{i} p_{i}$, the paper proves $\delta(x)$ is a constant $1/Z_{M}$ for all $x$, so $p_{M}$ exactly equals $p_{M}^{*}\propto\sum_{i}\omega_{i} R_{i}$ (Proposition 4.1). For non-linear operators like harmonic mean, contrastive, or scalarization with temperature $\beta\neq 1$, $\delta(x)$ is no longer constant, but experiments show it remains close to $1/Z_{M}$ in high-density regions where $\mathcal{G}$ is large. Thus, $L_{1}$ error is mainly contributed by low-density regions and remains controllable.
    - **Design Motivation**: Characterizing "exactness vs. approximation" using the same mathematical quantity $\delta(x)$ provides provable guarantees for linear cases and interpretable degradation modes for non-linear cases, rather than treating non-linearity as a black box.

### Loss & Training
Entirely training-free. All calculations occur at inference time: starting from $s_{0}$, the next state is sampled step-by-step, retrieving or cumulatively calculating $u_{i}$ at each step, constructing the mixed distribution according to the weighting formula, and sampling after normalization. Inference overhead is roughly equivalent to one forward pass per each of the $k$ models plus the composition, which is much lower than the cost of training a new model.

## Key Experimental Results

### Main Results

Evaluated on a $32\times 32$ 2-D grid (where the ground truth is computationally analytical for $L_{1}$ evaluation) and molecule generation (fragment-based + atom-based QM9).

| Task / Setting | Metric | Ours | Prev. SOTA | Gain |
| :--- | :--- | :--- | :--- | :--- |
| 2D Grid 5-Objective Linear | $L_{1}$ ↓ | 0.003 | HN-GFN 0.035 / MOGFN 0.048 | ~10–16x |
| 2D Grid Multi-obj Scalability | $L_{1}$ (2→5 obj) | 0.003→0.003 (stable) | MOGFN 0.021→0.048 (degraded) | Linear Scalability |
| 2D Grid Harmonic Mean $p_{\text{Circle1}}\otimes p_{\text{Circle2}}$ | $L_{1}$ ↓ | 0.229 | Classifier 0.397 | -0.168 |
| Fragment SEH-QED Scalarization | Avg Top-10 Reward ↑ | 0.777 (DB $F$) | MOGFN 0.764 | +0.013 (Zero-shot) |
| Fragment 3-Objective ALL | Top-10 Reward ↑ | 0.742 (DB $F$) | MOGFN 0.723 | +0.019 |
| Fragment SEH⊗SA⊗QED Harmonic Mean | 3-High Hit Rate (%) | 65–66 | Classifier 40 | +25 pt |
| QM9 GAP-SA Scalarization | Top-10 Reward ↑ | 0.873 | MOGFN 0.799 | +0.074 |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| **Ours** (reaching-weighted) | 2D Grid 5-obj $L_{1}$ = 0.003 | Includes reaching probability weighting |
| Ensemble (uniform mixing, no $u_{i}$) | $L_{1}$ ≈ 0.10–0.12 | Confirms $u_{i}$ is key, not just policy averaging |
| Model $F$ path vs DB $F$ path | Fragment ALL reward 0.741 vs 0.742 | Paths perform similarly; framework is objective-agnostic |
| $\delta(x)$ distribution check | Constant for linear; near-constant for logic in high-density regions | Empirical support for theoretical analysis |

### Key Findings
- **Reaching probability weighting is essential**: Removing it degrades the method to an ensemble baseline, with $L_{1}$ error jumping by 1-2 orders of magnitude. This shows local info on "who is responsible for which state space" is more important than global weights.
- **Scalability**: As the number of objectives increases from 2 to 5, the $L_{1}$ of **Ours** remains almost unchanged, while preference-conditioned baselines degrade linearly. This transforms "objective set scalability" from a retraining problem to a zero-cost problem.
- **Logic Conjunction**: In three-objective conjunction for molecular generation (SEH⊗SA⊗QED), **Ours** increases the hit rate from 40% to 65%. Unlike classifier guidance, which requires retraining a classifier for every combination, **Ours** is zero-shot.
- **Inference Speed**: Logic operator composition is significantly faster than classifier guidance (which requires classifier forward passes), matching the claim of minimal inference overhead.

## Highlights & Insights
- Extracting the "reaching probability $u_{i}(s)$" as a mixing weight and treating the internal trajectory structure of GFlowNets as a composable resource is a clever shift in perspective. Previous compositions were done at the terminal distribution level; this work pushes composition down to the per-step forward policy on the DAG.
- The single formula $p_{M,F}\propto \mathcal{G}(u_{i} p_{i,F})$ covers linear scalarization, temperature scaling, harmonic mean, and contrastive operators. This provides an "algebraic" language for GFlowNet composition, representing a discrete counterpart to score composition in diffusion models.
- The introduction of the distortion factor $\delta(x)$ quantifies the "when is it exact" question as "whether $\delta$ varies with $x$." Explaining approximation accuracy via empirical near-constancy in high-density regions elegantly links theory and practice, a strategy transferable to other compositional distribution analyses.

## Limitations & Future Work
- The distortion factor is only near-constant in high-density regions; non-negligible bias remains in low-density tails. If downstream tasks are sensitive to rare structures (e.g., safety screening, rare side-effect prediction), this approximation might introduce systematic blind spots.
- The method relies on having a high-quality pre-trained GFlowNet for each reward. In practice, new rewards are often targets for rapid experimentation themselves; pre-training costs are not eliminated but rather front-loaded into the "building block library" phase.
- The DB $F$ path uses online products $\prod p_{F}/p_{B}$, which may face numerical instability (multiplication of small probabilities) on long trajectories. The paper lacks a detailed discussion on the critical point of numerical degradation.
- The current framework is "non-adaptive"—the mixing rules for a given $\mathcal{G}$ are fixed. There is no mechanism to adaptively learn mixing weights based on sampling goals. Allowing minimal fine-tuning might further reduce the variance of the distortion factor.

## Related Work & Insights
- **vs MOGFN / HN-GFN [Jain 2023; Zhu 2024]**: These use preference-conditioned GFlowNets to bundle all preferences during training. **Ours** is training-free and better suited for "pre-train a single-objective library, then compose on-the-fly" workflows, provided the library covers the desired rewards.
- **vs Compositional Sculpting [Garipov 2024]**: Classifier guidance requires retraining a classifier for every new combination of logic operators. **Ours** uses the same mixing policy for logic operators with comparable quality, higher speed, and zero training cost.
- **vs Products of Experts / Diffusion Composition [Hinton 2002; Liu 2022; Du 2023]**: Score composition tools are mature in continuous settings. **Ours** provides an equivalent for discrete DAGs and identifies reaching probability as the natural candidate for "local weights" in discrete domains—an idea that could inform research on state-dependent mixing coefficients in diffusion models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Mixing forward policies via reaching probabilities is a clean, new perspective that unifies previous paths.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers both synthetic grids (analytical $L_{1}$) and real molecular tasks, including multi-combinations of SEH/SA/QED/GAP across scalarization, harmonic mean, and contrastive operators.
- **Writing Quality**: ⭐⭐⭐⭐ Clear correspondence between theory and evidence; the distortion factor makes the analysis highly readable.
- **Value**: ⭐⭐⭐⭐ An engineering-friendly, "ready-to-use" work for multi-objective GFlowNets with direct significance for accelerating workflows in drug/molecule discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](../../CVPR2026/computational_biology/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)
- [\[ICML 2026\] Transformed Latent Variable Multi-Output Gaussian Processes](transformed_latent_variable_multi-output_gaussian_processes.md)
- [\[ICML 2026\] Protein Autoregressive Modeling via Multiscale Structure Generation](protein_autoregressive_modeling_via_multiscale_structure_generation.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](../../NeurIPS2025/computational_biology/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[ICML 2026\] TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation](td3b_transition-directed_discrete_diffusion_for_allosteric_binder_generation.md)

</div>

<!-- RELATED:END -->
