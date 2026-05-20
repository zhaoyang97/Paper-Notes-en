---
title: >-
  [Paper Note] Adaptive Estimation and Inference in Semi-parametric Heterogeneous Clustered Multitask Learning via Neyman Orthogonality
description: >-
  [ICML 2026][Optimization][Neyman orthogonality] This work bridges double machine learning and clustered multitask learning, proposing an adaptive framework that combines Neyman orthogonality with data-driven pairwise fus…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Neyman orthogonality"
  - "adaptive fusion"
  - "latent clustering"
  - "heterogeneous noise"
  - "asymptotic normality"
date: 2026-05-08
content_hash: a7401999cf6ff311
---

# Adaptive Estimation and Inference in Semi-parametric Heterogeneous Clustered Multitask Learning via Neyman Orthogonality

**Conference**: ICML 2026  
**arXiv**: [2605.01907](https://arxiv.org/abs/2605.01907)  
**Code**: None  
**Area**: Multitask Learning / Causal Inference / Semiparametric Statistics  
**Keywords**: Neyman orthogonality, adaptive fusion, latent clustering, heterogeneous noise, asymptotic normality

## TL;DR
This work bridges double machine learning and clustered multitask learning, proposing an adaptive framework that combines Neyman orthogonality with data-driven pairwise fusion penalties. In a semiparametric setting with heterogeneous (possibly infinite-dimensional) noise, it accurately recovers latent task clusters, achieves oracle rates at the aggregation level, and establishes asymptotic normality for valid statistical inference.

## Background & Motivation

**Background**  
Multitask learning (MTL) improves statistical efficiency via shared structure, but in practice, tasks are often only partially related: they may share target parameters, but auxiliary features, data distributions, and confounders can differ greatly. Clustered multitask learning seeks to discover latent groupings among tasks. Recent advances in double machine learning (DML) enable estimation of low-dimensional target parameters under high-dimensional/nonparametric noise.

**Limitations of Prior Work**  
1. **Overly strong MTL assumptions**: Most methods assume aligned feature spaces or isomorphic task structures, lacking robustness to heterogeneous features and distribution shifts.
2. **DML is single-task**: DML does not leverage cross-task similarity; variance can be high when individual task sample sizes are small.
3. **Clustering + infinite-dimensional noise challenge**: Existing clustered MTL methods (fusion penalties, centroid regularization) mostly assume parametric models and cannot handle task-specific complex high-dimensional noise.

**Key Challenge**  
There is a need to share information across tasks to reduce variance, while retaining localized, flexible noise estimation for valid inference—these goals appear to be in tension.

**Goal**  
Design a method that simultaneously: (i) discovers and leverages shared target parameter structure, (ii) remains robust to heterogeneous, possibly infinite-dimensional noise, and (iii) establishes precise inferential guarantees.

**Key Insight**  
Starting from task-level initial estimates (for similarity quantification), the second stage uses Neyman orthogonality to protect inference, with fusion penalties applied only to target parameters (across tasks), while noise parameters remain task-local (no cross-task contamination).

**Core Idea**  
Two-stage adaptive fusion: Stage 1 uses any (possibly non-orthogonal) initial loss to obtain rough consistent estimates and compute task-pair distances; Stage 2 applies adaptive pairwise penalties $\lambda_{jj'}=\min(c_w\|\hat\theta_j^{\text{init}}-\hat\theta_{j'}^{\text{init}}\|_2^{-\gamma},\text{const})$ to strengthen fusion among similar tasks, combining orthogonal loss and sample splitting, so that even after adaptive clustering, the estimator achieves $\sqrt{N_k}$ (aggregate sample size) CAN property.

## Method

### Overall Architecture
There are $m$ tasks; task $j$ has target parameter $\theta_j^*\in\Theta\subseteq\mathbb R^d$ and noise parameter $\eta_j^*\in\mathcal H_j$. Assume $\{\theta_j^*\}$ admit latent clusters $\{S_k\}_{k=1}^K$, with $\theta_j^*=\beta_k^*$ within each cluster, but $\eta_j^*$ may differ greatly in dimension, smoothness, etc.

**Two-stage estimator**:
- **Stage 1 (Structure Discovery)**: For each task $j$, use possibly non-orthogonal loss $\ell_j^{\text{init}}$ to obtain initial $\hat\theta_j^{\text{init}}$. These initial estimates are only for diagnosing task similarity, not required to be rate-optimal.
- **Stage 2 (Clustered Fusion)**: Split samples $\mathcal D_j=\mathcal D_{j,1}\cup\mathcal D_{j,2}$. Estimate noise $\hat\eta_j$ on $\mathcal D_{j,1}$, then solve multitask objective on $\mathcal D_{j,2}$: $\hat{\boldsymbol\theta}=\arg\min\sum_j f_j^\dagger(\theta_j,\hat\eta_j)+\sum_{j<j'}\lambda_{jj'}\|\theta_j-\theta_{j'}\|_2$, where $f_j^\dagger$ is the orthogonal loss. Penalty $\lambda_{jj'}$ takes minimum value $\epsilon_n$ (strong fusion) if initial distance $<\tau$, otherwise uses weight $c_w\|\cdot\|^{-\gamma}$.

### Key Designs

1. **Within-task Neyman Orthogonality + Sample Splitting**:

    - **Function**: Protects target parameter estimation from noise estimation error, even if noise is high-dimensional or nonparametric.
    - **Mechanism**: Design $\ell_j^\dagger$ so that the Gâteaux derivative $D_\eta\nabla_\theta\mathbb E[\ell_j^\dagger]|_{(\theta_j^*,\eta_j^*)}[h]=0$ holds for all $h$ in the noise realization set. First-order noise error $\|\hat\eta-\eta^*\|=O_p(1/\sqrt n)$ has no effect on $\theta$ estimation. Sample splitting (using different folds for noise and target) prevents overfitting.
    - **Design Motivation**: Multitask fusion is performed at the target level, but noise remains task-local, avoiding propagation of cross-task bias.

2. **Adaptive Pairwise Fusion Penalty**:

    - **Function**: Infers the probability that task pairs belong to the same cluster from initial estimate distances, dynamically adjusting fusion strength.
    - **Mechanism**: Weight $w_{jj'}=c_w\|\hat\theta_j^{\text{init}}-\hat\theta_{j'}^{\text{init}}\|_2^{-\gamma}$; large distance $\to$ small weight $\to$ weak fusion. Threshold $\tau$: pairs with distance $<\tau$ take minimum penalty $\epsilon_n$ (strong fusion), others use $w_{jj'}$ (moderate fusion). This two-level structure achieves exact cluster recovery under strong separation (Theorem 3.5).
    - **Design Motivation**: Compared to ARMUL's hard discrete clustering, adaptive weights provide smooth transitions and are more robust to hyperparameters and separation conditions; compared to fixed weights (MeTaG), adaptive weights automatically follow the task similarity landscape.

3. **Two-stage Separation Design**:

    - **Function**: Allows "cluster discovery" and "precise inference" to each use the most suitable tools.
    - **Mechanism**: The initial stage does not require optimal rates, only consistency—enabling selection of more stable (lower variance) estimators, even if slightly biased in noise. The resulting initial estimates are more stable in finite samples, reducing noise in $w_{jj'}$ calculation. The second stage focuses on fine estimation and inference.
    - **Design Motivation**: Separates the goals of "discovery" and "inference," allowing each to use the best tools, rather than forcing a single framework to do both.

### Loss & Training
The second-stage optimization over $\theta$ is convex and can use accelerated gradient or proximal methods. Orthogonality is naturally achieved by designing the loss on $\mathcal D_{j,2}$ via sample splitting. The paper proves that results hold for a wide range of $(c_w,\gamma,\tau,\epsilon_n)$, providing robust hyperparameter selection guidance.

## Key Experimental Results

### Main Results

| Model | Setting | RMSE | ARI | vs Personalized | vs ARMUL (correct K) | vs MeTaG |
|-------|---------|------|-----|-----------------|----------------------|----------|
| PLM | $\delta=1/3$ | **0.18** | **0.98** | -67% | +2% | -85% |
| PLM | $\delta=2/3$ | **0.12** | **0.99** | -72% | -1% | -88% |
| PLM | $\delta=1.0$ | **0.08** | **1.00** | -78% | -3% | -91% |
| ATE | $\delta=1/3$ | **0.22** | **0.97** | -63% | +5% | -80% |
| ATE | $\delta=2/3$ | **0.15** | **0.99** | -70% | 0% | -85% |
| DID | $\delta=2/3$ | **0.19** | **0.98** | -68% | +1% | -83% |

ARMUL is slightly better when $K$ is correct, but its performance drops sharply when $K$ is misspecified; the proposed method remains optimal regardless of $K$.

### Ablation Study

| Component | Change | RMSE Increase | ARI Drop | Note |
|-----------|--------|--------------|----------|------|
| Full Method | - | - | - | Baseline |
| Remove Orthogonality | Use non-orthogonal loss in stage 2 | +45% | Unchanged | Unbiased but higher variance |
| Fixed Penalty | All pairs $\lambda_{jj'}=0.01$ | +28% | +0.15 | No adaptation, under-fusion |
| No Two-level Threshold | Single-layer $\lambda=w_{jj'}$ | +18% | +0.08 | Improper fusion strength |
| No Sample Splitting | Noise and target share fold | +32% | Unchanged | Overfitting, unreliable inference |

### Key Findings
- **Accurate cluster recovery**: Even with weak separation ($\delta=1/3$), ARI ≈ 0.98, while ARMUL requires exact $K$ to achieve this.
- **Adaptive weights are critical**: Fixed weights increase RMSE by 28%, confirming the importance of personalized fusion strength for task pairs.
- **Orthogonality is essential**: Removing orthogonality increases RMSE by 45%; clustering is unaffected, but confidence interval coverage fails.
- **Sample splitting protects inference**: Point estimates are unaffected, but inference (CI coverage) fails without splitting.
- **Hyperparameter robustness**: Results across multiple $(\gamma,\tau)$ settings are insensitive to parameter ranges, supporting the "broad conditions" theory.

### Real Data Application
In the analysis of electricity price elasticity for 50 US states + DC, the method discovers 3 clusters:
- Cluster 0 (VA): High elasticity -1.138, cooling-intensive, highly adjustable.
- Cluster 1 (KY/AL/OK/TN): Medium elasticity -0.788, southern warm states.
- Cluster 2 (remaining 46 states): Low elasticity -0.221.

Clusters align with climate geography, validating the method's effectiveness in real heterogeneous multitask settings.

## Highlights & Insights
- **Role of Neyman orthogonality in multitask**: Combines DML with cluster fusion, ensuring valid inference even with cross-task fusion.
- **Subtlety of adaptive weights**: Compared to hard clustering, soft adaptive weights learn from data and are significantly more robust to hyperparameters.
- **Two-stage separation philosophy**: Separates "cluster discovery" and "precise inference," allowing each stage to use optimal tools and avoiding rigidity of a single framework.
- **Economic application integration**: Discovery of regional electricity elasticity both validates the method and provides policy-relevant insights.

## Limitations & Future Work
- **Limited to low-dimensional targets**: Extension to high-dimensional targets (dimension growing with sample size) is not considered.
- **Cluster separation assumption**: Still requires minimum cross-cluster separation $\delta$; not applicable to fully continuous task spaces.
- **Practical challenges in noise estimation**: Theoretical requirement of $O_p(n_j^{-1/4})$ rate is difficult to achieve for complex models.

## Related Work & Insights
- **vs ARMUL**: Both perform clustered MTL, but ARMUL requires known $K$; the proposed method recovers $K$ automatically and is more robust to hyperparameters.
- **vs DML single-task**: Extends the DML framework to clustered multitask, retaining the inferential validity advantage.
- **vs classical clustered learning (Jacob et al.)**: These methods are mostly limited to parametric models; this work handles heterogeneous semiparametric noise, representing a significant extension.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of Neyman orthogonality and adaptive cluster fusion is novel; the two-stage framework is also new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three types of semiparametric models, multiple separation levels, thorough ablation, real application.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous, clear theorem statements, intuitive main results.
- Value: ⭐⭐⭐⭐ Directly applicable in causal inference and economics; the theoretical framework has far-reaching impact on multitask inference.

## Related Papers

- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)
- [\[ICML 2026\] FAB: A First-Order AB-based Gradient Algorithm for Distributed Bilevel Optimization over Time-Varying Directed Graphs](fab_a_first-order_ab-based_gradient_algorithm_for_distributed_bilevel_optimizati.md)
- [\[ICML 2026\] Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization](neural_qaoa2_differentiable_joint_graph_partitioning_and_parameter_initializatio.md)
- [\[ICML 2026\] On the Expressive Power of GNNs to Solve Linear SDPs](on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)
- [\[ICML 2026\] AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction](agentxray_white-boxing_agentic_systems_via_workflow_reconstruction.md)

</div>

<!-- RELATED:END -->
