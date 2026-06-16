---
title: >-
  [Paper Note] Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation
description: >-
  [ICML 2026][Computational Biology][GFlowNets] This paper proposes a training-free framework for composing GFlowNets. By using the "reaching probability" of each pre-trained model as weights to mix their respective forward policies, the framework enables direct sampling for arbitrary linear scalarizations or logical operator combinations during the inference phase.
tags:
  - ICML 2026
  - Computational Biology
  - GFlowNets
date: 2026-05-08
content_hash: df8e38dd69e98b4d
---
# Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation

**Conference**: ICML 2026  
**arXiv**: [2602.21565](https://arxiv.org/abs/2602.21565)  
**Code**: https://github.com/ml-postech/gflownet-composition (Available)  
**Area**: Scientific Computing / GFlowNets / Molecular Generation  
**Keywords**: GFlowNets, Multi-Objective Generation, Model Composition, Inference-time Mixing, Molecular Design  

## TL;DR
This paper proposes a training-free framework for composing GFlowNets. By using the "reaching probability" of each pre-trained model as weights to mix their respective forward policies, the framework enables direct sampling for arbitrary linear scalarizations or logical operator combinations during the inference phase. It is theoretically proven to exactly recover the target distribution in the linear case.

## Background & Motivation

**Background**: GFlowNets are a class of generative models trained to sample diverse candidates proportional to rewards. They are particularly popular in scientific discovery tasks involving discrete structures like molecules, graphs, and biological sequences because, unlike traditional optimization, they cover the entire high-reward region rather than converging to a single point. When a problem involves multiple rewards (e.g., binding affinity SEH, synthetic accessibility SA, drug-likeness QED), mainstream approaches follow two paths: (1) training a preference-conditioned model (MOGFN, HN-GFN) on a weighted sum of rewards; or (2) combining pre-trained single-objective models into logical operators (e.g., harmonic mean for intersection, contrastive operators for difference) via classifier guiding (compositional sculpting).

**Limitations of Prior Work**: Both paths suffer from significant drawbacks—any modification to the target set requires re-training. In preference-conditioned methods, the set of reward functions is frozen during training; adding a new objective necessitates training from scratch. While classifier guiding leaves the GFlowNet unchanged, it requires training an auxiliary classifier for every new combination, the cost of which often exceeds training a standalone GFlowNet. Furthermore, linear scalarization and logical operators are treated as two incompatible sub-problems without a unified language.

**Key Challenge**: There exists a trade-off between the "universality" of composition and the "training cost." Supporting arbitrary combinations typically requires knowing the operator form before training, while avoiding training usually necessitates sacrificing flexibility. Both approaches are fundamentally built on "distribution-level operations" and lack a tool to perform composition directly on the internal structure of GFlowNets—the forward policy.

**Goal**: (i) Treat several pre-trained single-objective GFlowNets as "building blocks" for zero-shot composition to assemble arbitrary target distributions at inference time; (ii) support both linear scalarization and logical operators using the same mechanism; (iii) provide theoretical guarantees.

**Key Insight**: The authors observe that the marginal distribution of a GFlowNet over a terminal state $x$ can be decomposed into the product of the "reaching probability $u(s)$" and the "terminating action probability $p_F(s_f\mid x)$", i.e., $p(x)=u(x)\cdot p_F(s_f\mid x)$. Reaching probability naturally characterizes "how much probability mass a model routes to state $s$"—identifying how "relevant" that state is to the model.

**Core Idea**: Use the reaching probability of each pre-trained GFlowNet at the current state as a local weight to mix their forward transition probabilities according to a target composition function $\mathcal{G}$. Theoretically, this rule exactly recovers the target distribution under linear scalarization and maintains controllable distortion in high-density regions for non-linear operators.

## Method

### Overall Architecture
Given $k$ pre-trained single-objective GFlowNets (where the $i$-th model targets reward $R_i(x)$, with forward policy $p_{i,F}$, reaching probability $u_i$, and terminal distribution $p_i\propto R_i$), the paper aims to construct a sampler at inference time for any composed distribution $p_M^*(x)\propto\mathcal{G}(p_1,\dots,p_k)$ without any training. The approach pushes "composition" from the terminal distribution level down to every step of the DAG—calculating a mixed forward policy $p_{M,F}$ on the fly at each non-sink state. By sampling step-by-step according to this policy, the resulting marginal distribution (exactly or approximately) equals the target $p_M^*$.

### Key Designs

**1. Reaching-weighted Mixing Strategy: Leveraging Expertise in Familiar States**

Simply averaging $k$ forward policies degrades to an ensemble of the policies themselves, ignoring "which model is more reliable at the current state." The core operator of this paper uses the reaching probability of each model as a local weight to mix forward transitions: for each child $s'$ of state $s$,

$$p_{M,F}(s'\mid s)=\frac{\mathcal{G}\bigl(u_1(s)\,p_{1,F}(s'\mid s),\dots,u_k(s)\,p_{k,F}(s'\mid s)\bigr)}{N_M(s)},$$

where $N_M(s)$ is the normalization constant summing these quantities over all children of the state. Here, $u_i(s)$ measures "how much probability mass model $i$ routed to $s$." A higher value indicates the model is more familiar with this region of the state space and should have more influence at $s$. Weighting each model's contribution by its "presence" along the current trajectory is the key to the theoretical closure—naive uniform mixing (setting all $u_i$ to $1$) degrades to an ensemble baseline in experiments, with $L_1$ errors increasing by orders of magnitude.

**2. Computable Reaching Probability: Supporting Two Major Training Paradigms**

For the mixing strategy to be usable at inference time, $u_i(s)$ must be obtained efficiently. Solving the recursive formula $u(s)=\sum_{s_*}u(s_*)p_F(s\mid s_*)$ is infeasible as it requires enumerating all parent states. Instead, the authors use the identity $u_i(s)=F_i(s)/Z_i$ ($F_i$ is the state flow, $Z_i=F_i(s_0)$ is the total flow), leading to two complementary routes: if the pre-training objective explicitly parameterizes the flow (flow matching / detailed balance / sub-trajectory balance), it is read directly ("Model $F$"); if the objective does not explicitly model $F$ (like trajectory balance), $F_i$ is recovered online along the current trajectory using the detailed balance condition $F_i(s')p_{i,B}(s\mid s')=F_i(s)p_{i,F}(s'\mid s)$ by cumulative multiplication $\prod_j p_{i,F}/p_{i,B}$ ("DB $F$"). Together, these cover the two major classes of GFlowNet training objectives, ensuring the framework is agnostic to the training paradigm of the building blocks.

**3. Distortion Factor Analysis: A Unified Measure of Error**

To explain what distribution the mixing strategy actually recovers, the marginal distribution of a full trajectory is expanded as $p_M(x)=\delta(x)\cdot\mathcal{G}(p_1(x),\dots,p_k(x))$, where the distortion factor $\delta(x)=u_M(x)/N_M(x)$ characterizes the deviation. For linear scalarization $\mathcal{G}=\sum_i\omega_i Z_i p_i$, the paper proves $\delta(x)$ is a constant $1/Z_M$ for all $x$, so $p_M$ exactly equals $p_M^*\propto\sum_i\omega_i R_i$ (Proposition 4.1). For non-linear operators like harmonic mean, contrastive, or scalarization with temperature $\beta\neq 1$, $\delta(x)$ is no longer constant, but experiments show it remains close to $1/Z_M$ in high-density regions where $\mathcal{G}$ is large. $L_1$ error is mainly contributed by low-density regions and remains controllable overall.

### Loss & Training
Completely training-free. All calculations occur at inference time: starting from $s_0$, steps are sampled sequentially. At each step, $u_i$ is read or calculated online, the mixed distribution is constructed using the weighted formula, and the next state is sampled after normalization. The per-step inference overhead is roughly equivalent to one forward pass for each of the $k$ models plus a composition, which is far lower than the cost of training a new model.

## Key Experimental Results

### Main Results

Evaluated on a $32\times 32$ 2D grid (where target distributions are analytically computable for $L_1$ evaluation) and molecular generation (fragment-based and atom-based QM9).

| Task / Setting | Metric | Ours | Prev. SOTA | Gain |
| :--- | :--- | :--- | :--- | :--- |
| 2D Grid 5-Objective Linear | $L_1$ ↓ | 0.003 | HN-GFN 0.035 / MOGFN 0.048 | ~10–16x improvement |
| 2D Grid Scalability | $L_1$ (2→5 obj) | 0.003→0.003 (stable) | MOGFN 0.021→0.048 (degrades) | Linear Scalability |
| 2D Grid Harmonic Mean $p_{\text{C1}}\otimes p_{\text{C2}}$ | $L_1$ ↓ | 0.229 | Classifier 0.397 | -0.168 |
| Fragment SEH-QED Scalarization | Avg Top-10 Reward ↑ | 0.777 (DB $F$) | MOGFN 0.764 | +0.013 (Zero-shot) |
| Fragment 3-Objective ALL | Top-10 Reward ↑ | 0.742 (DB $F$) | MOGFN 0.723 | +0.019 |
| Fragment SEH⊗SA⊗QED Harmonic | High-Reward Hit Rate (%) | 65–66 | Classifier 40 | +25 pt |
| QM9 GAP-SA Scalarization | Top-10 Reward ↑ | 0.873 | MOGFN 0.799 | +0.074 |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full method (reaching-weighted) | 2D Grid 5-Obj $L_1$ = 0.003 | Includes reaching probability weighting |
| Ensemble (Uniform mixing) | $L_1$ ≈ 0.10–0.12 | Confirms $u_i$ is essential; policy averaging is insufficient |
| Model $F$ vs DB $F$ | Fragment ALL Reward 0.741 vs 0.742 | Both methods perform similarly; agnostic to training objective |
| Distortion Factor $\delta(x)$ Check | Constant $1/Z_M$ for linear; near-constant in high-density for logic | Empirically supports theoretical analysis |

### Key Findings
- Reaching probability weighting is mandatory: removing it leads to an ensemble baseline where $L_1$ error jumps by one to two orders of magnitude, proving that local information about "who is responsible for which part of the state space" is more important than global weights.
- When the number of objectives increases from 2 to 5, the $L_1$ error of **Ours** remains nearly constant, whereas preference-conditioned baselines degrade linearly. This transforms the "objective set scalability" problem from a re-training issue into a zero-cost issue.
- In 3-objective molecular intersection (SEH⊗SA⊗QED), **Ours** improves the hit rate from 40% to 65%. While classifier guiding requires re-training for every combination, **Ours** is entirely training-free.
- Inference speed for logical operator composition is significantly faster than classifier guiding (which requires extra classifier forward passes), matching the claim of "minimal inference overhead."

## Highlights & Insights
- Extracting "reaching probability $u_i(s)$" as a mixing weight and treating the internal trajectory structure of GFlowNets as a composable resource is a clever perspective shift. While previous compositions occurred at the terminal distribution level, this work pushes composition down to every step of the DAG.
- The single formula $p_{M,F}\propto \mathcal{G}(u_i p_{i,F})$ covers linear scalarization, temperature scaling, harmonic mean, and contrastive operators, providing an "algebraic" language for GFlowNet composition—a discrete counterpart to score composition in diffusion models.
- The introduction of the distortion factor $\delta(x)$ quantifies the "exactness" of the method as the variance of $\delta(x)$. Showing empirically that $\delta(x)$ is nearly constant in high-density regions provides a clear explanation for the approximation accuracy, effectively linking theory and empirical results.

## Limitations & Future Work
- The distortion factor is only near-constant in high-density regions; non-negligible bias may still exist in low-density tails. This might introduce systematic blind spots if downstream tasks are sensitive to rare structures (e.g., safety screening).
- The framework relies on having a high-quality pre-trained GFlowNet for every reward. In practice, the pre-training cost is not eliminated but shifted to the "library construction" phase.
- The "DB $F$" route involves cumulative multiplication $\prod p_F/p_B$ along trajectories, which might suffer from numerical instability in very long trajectories.
- The current framework is "non-adaptive"—the mixing rule is fixed for a given $\mathcal{G}$. There is no mechanism to adapt mixing weights based on sampling targets; allowing a small amount of fine-tuning might further reduce the variance of the distortion factor.

## Related Work & Insights
- **vs MOGFN / HN-GFN [Jain 2023; Zhu 2024]**: They use preference-conditioned GFlowNets to pack all preferences during training. **Ours** is training-free and better suited for workflows where one builds a single-objective library and composes on demand, provided the library covers the desired rewards.
- **vs Compositional Sculpting [Garipov 2024]**: Classifier guiding requires re-training a classifier for every new combination. **Ours** covers logical operators with the same mixing strategy, achieving comparable quality with zero training cost and faster speed.
- **vs Products of Experts / Diffusion Composition [Hinton 2002; Liu 2022; Du 2023]**: Score composition in continuous spaces is mature; this work provides a discrete equivalent on DAGs and identifies reaching probability as a natural candidate for "local weights" in the discrete domain—a concept that could inversely inform research on state-dependent mixing coefficients in diffusion models.

## Rating
- Novelty: ⭐⭐⭐⭐ Reaching-weighted policy mixing is a clean new perspective that unifies previous approaches.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic grids (analytical $L_1$) and real molecular tasks with various operator types (linear/harmonic/contrastive).
- Writing Quality: ⭐⭐⭐⭐ Clear correspondence between theory and empirical analysis; the distortion factor makes the analysis highly readable.
- Value: ⭐⭐⭐⭐ A "ready-to-use" engineering-friendly work in the GFlowNet multi-objective space, with direct implications for speeding up drug/molecule discovery workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Graph Generative Pre-trained Transformer (G2PT)](../../ICML2025/computational_biology/graph_generative_pre-trained_transformer.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](../../CVPR2026/computational_biology/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)
- [\[ICML 2025\] PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion](../../ICML2025/computational_biology/peptune_de_novo_generation_of_therapeutic_peptides_with_multi-objective-guided_d.md)
- [\[ICML 2025\] eccDNAMamba: A Pre-Trained Model for Ultra-Long eccDNA Sequence Analysis](../../ICML2025/computational_biology/eccdnamamba_a_pre-trained_model_for_ultra-long_eccdna_sequence_analysis.md)
- [\[ICML 2026\] Transformed Latent Variable Multi-Output Gaussian Processes](transformed_latent_variable_multi-output_gaussian_processes.md)

</div>

<!-- RELATED:END -->
