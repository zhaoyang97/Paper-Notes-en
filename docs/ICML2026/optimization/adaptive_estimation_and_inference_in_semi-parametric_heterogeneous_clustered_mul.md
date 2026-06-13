---
title: >-
  [Paper Note] Adaptive Estimation and Inference in Semi-parametric Heterogeneous Clustered Multitask Learning via Neyman Orthogonality
description: >-
  [ICML 2026][Optimization][Neyman Orthogonality] This paper bridges double machine learning with clustered multitask learning by proposing an adaptive framework that combines Neyman orthogonality with a data-driven pairwi…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Neyman Orthogonality"
  - "Adaptive Fusion"
  - "Latent Clustering"
  - "Heterogeneous Noise"
  - "Asymptotic Normality"
date: 2026-05-08
content_hash: ba605aaab1744f4f
---

# Adaptive Estimation and Inference in Semi-parametric Heterogeneous Clustered Multitask Learning via Neyman Orthogonality

**Conference**: ICML 2026  
**arXiv**: [2605.01907](https://arxiv.org/abs/2605.01907)  
**Code**: None  
**Area**: Multitask Learning / Causal Inference / Semi-parametric Statistics  
**Keywords**: Neyman Orthogonality, Adaptive Fusion, Latent Clustering, Heterogeneous Noise, Asymptotic Normality

## TL;DR
This paper bridges double machine learning with clustered multitask learning by proposing an adaptive framework that combines Neyman orthogonality with a data-driven pairwise fusion penalty. In semi-parametric settings with heterogeneous (potentially infinite-dimensional) noise, it accurately recovers latent task clusters, achieves oracle-level aggregate convergence rates, and establishes asymptotic normality to enable effective statistical inference.

## Background & Motivation

**Background**
Multitask learning (MTL) improves statistical efficiency through shared structures. However, in reality, tasks are often only partially related: they may share target parameters, but auxiliary features, data distributions, and confounding factors vary significantly. Clustered multitask learning attempts to discover latent groupings among tasks. Recent advances in double machine learning (DML) have made it possible to estimate low-dimensional target parameters under high-dimensional or non-parametric noise.

**Limitations of Prior Work**
1. **Existing MTL assumptions are too strong**: Most methods assume aligned feature spaces or isomorphic task structures, which are insufficient for handling heterogeneous features and distribution shifts.
2. **DML is a single-task procedure**: DML itself does not exploit cross-task similarities; variance can be high when the sample size for an individual task is limited.
3. **Challenge of clustered learning with infinite-dimensional noise**: Existing clustered MTL methods (fusion penalties, centroid regularization) mostly assume parametric models and cannot handle task-specific complex high-dimensional noise.

**Key Challenge**
There is a need to share information across tasks to reduce variance while maintaining the flexibility of localized noise estimation to preserve inference validity—two goals that seem to conflict.

**Goal**
Design a method that simultaneously: (i) discovers and utilizes the shared structure of target parameters, (ii) remains robust to heterogeneous and potentially infinite-dimensional noise, and (iii) establishes precise inference guarantees.

**Key Insight**
Starting from a first-stage task-level initial estimate (used for similarity quantification), the second stage uses Neyman orthogonality to protect the inference. The fusion penalty acts only on the target parameters (across tasks), while the noise parameters remain task-local (avoiding cross-task contamination).

**Core Idea**
Two-stage adaptive fusion: Stage 1 uses an arbitrary (potentially non-orthogonal) initial loss to obtain coarse consistent estimates and compute distances between task pairs. Stage 2 strengthens similar tasks through an adaptive pairwise penalty $\lambda_{jj'}=\min(c_w\|\hat\theta_j^{\text{init}}-\hat\theta_{j'}^{\text{init}}\|_2^{-\gamma},\text{const})$, combined with an orthogonal loss and sample splitting. This allows the estimator to achieve Continuous Asymptotic Normality (CAN) at the $\sqrt{N_k}$ (aggregated sample size) rate even after adaptive clustering.

## Method

### Overall Architecture
Given $m$ tasks, task $j$ has a target parameter $\theta_j^*\in\Theta\subseteq\mathbb R^d$ and a nuisance (noise) parameter $\eta_j^*\in\mathcal H_j$. It is assumed that $\{\theta_j^*\}$ admits a latent clustering $\{S_k\}_{k=1}^K$ where $\theta_j^*=\beta_k^*$ for tasks within the same cluster, though $\eta_j^*$ can differ significantly in terms of dimensionality or smoothness.

**Two-stage Estimator**:
- **Stage 1 (Structure Discovery)**: For each task $j$, an initial $\hat\theta_j^{\text{init}}$ is obtained using a potentially non-orthogonal loss $\ell_j^{\text{init}}$. These initial estimates are used only to diagnose task similarity and do not require optimal rates.
- **Stage 2 (Clustered Fusion)**: Sample splitting is applied $\mathcal D_j=\mathcal D_{j,1}\cup\mathcal D_{j,2}$. Nuisance parameters $\hat\eta_j$ are estimated on $\mathcal D_{j,1}$. On $\mathcal D_{j,2}$, the multitask objective is solved: $\hat{\boldsymbol\theta}=\arg\min\sum_j f_j^\dagger(\theta_j,\hat\eta_j)+\sum_{j<j'}\lambda_{jj'}\|\theta_j-\theta_{j'}\|_2$, where $f_j^\dagger$ is the orthogonal loss. The penalty $\lambda_{jj'}$ takes a minimum value $\epsilon_n$ (strong fusion) if the initial distance is $<\tau$, and otherwise takes the weight $c_w\|\cdot\|^{-\gamma}$.

### Key Designs

1. **Intra-task Neyman Orthogonality + Sample Splitting**:
    - **Function**: Protects target parameter estimation from noise estimation errors, even when noise is high-dimensional or non-parametric.
    - **Mechanism**: Designs $\ell_j^\dagger$ such that the Gâteaux derivative $D_\eta\nabla_\theta\mathbb E[\ell_j^\dagger]|_{(\theta_j^*,\eta_j^*)}[h]=0$ holds for all $h$ in the noise realization set. The impact of the first-order noise error $\|\hat\eta-\eta^*\|=O_p(1/\sqrt n)$ on the $\theta$ estimation is eliminated. Combined with sample splitting (using different folds for noise and target), this prevents overfitting.
    - **Design Motivation**: Multitask fusion performs cross-task mixing at the target level, but noise remains task-local to avoid propagating incorrect cross-task biases.

2. **Adaptive Pairwise Fusion Penalty**:
    - **Function**: Infers the probability that task pairs belong to the same cluster based on initial estimation distances and dynamically adjusts fusion strength accordingly.
    - **Mechanism**: The weight is $w_{jj'}=c_w\|\hat\theta_j^{\text{init}}-\hat\theta_{j'}^{\text{init}}\|_2^{-\gamma}$; a larger distance leads to a smaller weight and weaker fusion. For pairs with distance $<\tau$, a minimum penalty $\epsilon_n$ (strong fusion) is applied. This two-layer structure achieves exact cluster recovery (Theorem 3.5) under a strong separation assumption.
    - **Design Motivation**: Compared to the discreteness of ARMUL's hard clustering, adaptive weights provide a smooth transition and are more robust to hyperparameters and separation conditions; compared to fixed weights (MeTaG), adaptive weights automatically follow the task similarity landscape.

3. **Two-stage Separation Design**:
    - **Function**: Allows "cluster discovery" and "precise inference" to use the tools best suited for each goal.
    - **Mechanism**: The initial stage does not require optimal rates, only consistency—allowing for the selection of more stable (lower variance) estimators even if they have slight bias in the presence of noise. This results in initial estimates that are more stable under finite samples for computing $w_{jj'}$. The second stage focuses on refined estimation and inference.
    - **Design Motivation**: Decoupling the "discovery" and "inference" objectives allows each to use the most appropriate tool without forcing a single framework to accommodate both perfectly.

### Loss & Training
The second-stage optimization regarding $\theta$ is a convex problem, solvable via accelerated gradient or proximal methods. Orthogonality is naturally achieved through loss design on $\mathcal D_{j,2}$ using sample splitting. The paper proves that results hold across a wide range of $(c_w,\gamma,\tau,\epsilon_n)$, providing a robust guide for hyperparameter selection.

## Key Experimental Results

### Main Results

| Model | Setting | RMSE | ARI | vs Personalized | vs ARMUL (Correct K) | vs MeTaG |
|------|------|------|-----|-----------------|------------------|----------|
| PLM | $\delta=1/3$ | **0.18** | **0.98** | -67% | +2% | -85% |
| PLM | $\delta=2/3$ | **0.12** | **0.99** | -72% | -1% | -88% |
| PLM | $\delta=1.0$ | **0.08** | **1.00** | -78% | -3% | -91% |
| ATE | $\delta=1/3$ | **0.22** | **0.97** | -63% | +5% | -80% |
| ATE | $\delta=2/3$ | **0.15** | **0.99** | -70% | 0% | -85% |
| DID | $\delta=2/3$ | **0.19** | **0.98** | -68% | +1% | -83% |

ARMUL performs slightly better when K is correct, but its performance drops significantly when K is incorrect; Ours maintains optimality regardless of whether K is known.

### Ablation Study

| Component | Change | RMSE Gain | ARI Drop | Description |
|------|------|---------|---------|------|
| Full Method | - | - | - | Baseline |
| Remove Orthogonality | Non-orthogonal loss in Stage 2 | +45% | Unchanged | No bias but increased variance |
| Fixed Penalty | $\lambda_{jj'}=0.01$ for all pairs | +28% | +0.15 | No adaptation, under-fusion |
| No Threshold | Single-layer $\lambda=w_{jj'}$ | +18% | +0.08 | Improper fusion strength |
| No Sample Splitting | Shared fold for noise and target | +32% | Unchanged | Overfitting, unreliable inference |

### Key Findings
- **Accurate Cluster Recovery**: Even when cluster separation is weak ($\delta=1/3$), ARI≈0.98, whereas ARMUL requires knowing the exact K to achieve this.
- **Importance of Adaptive Weights**: Fixed weights lead to a +28% RMSE, confirming that personalized fusion strength for task pairs is crucial.
- **Necessity of Orthogonality**: Removing orthogonality increases RMSE by 45%; although it doesn't affect clustering, the confidence interval coverage fails.
- **Sample Splitting Protects Inference**: While it has less impact on point estimates, inference (CI coverage) fails significantly without splitting.
- **Hyperparameter Robustness**: Experiments across multiple sets of $(\gamma,\tau)$ show results are insensitive to the parameter range, supporting the "broad conditions" theory.

### Real-world Application
In an analysis of electricity price elasticity across 50 US states + DC, the method discovered 3 clusters:
- Cluster 0 (VA): High elasticity -1.138, cooling-intensive and highly adjustable.
- Cluster 1 (KY/AL/OK/TN): Moderate elasticity -0.788, warm southern states.
- Cluster 2 (Remaining 46 states): Low elasticity -0.221.

The clusters align with climate and geography, validating the method's effectiveness in real-world heterogeneous multitask settings.

## Highlights & Insights
- **Role of Neyman Orthogonality in MTL**: Combining DML with clustered fusion ensures that inference validity is maintained even with cross-task fusion.
- **Subtlety of Adaptive Weights**: Compared to hard clustering, soft adaptive weights learn from data and are significantly more robust to hyperparameters.
- **Design Philosophy of Two-stage Separation**: Separating "cluster discovery" from "precise inference" allows each stage to use optimal tools, avoiding the rigidity of a single framework.
- **Integration with Economic Applications**: The discovery of regional electricity elasticity both validates the method and provides policy-relevant insights.

## Limitations & Future Work
- **Limited to Low-dimensional Target Parameters**: Extensions for high-dimensional targets (where dimension grows with sample size) have not been considered.
- **Cluster Separation Assumption**: A minimum separation $\delta$ between clusters is still required; the method is not applicable to entirely continuous task spaces.
- **Practical Challenges of Noise Estimation**: The theory requires a $O_p(n_j^{-1/4})$ rate, which is not easily achieved for complex models.

## Related Work & Insights
- **vs ARMUL**: Both perform clustered MTL, but ARMUL requires prior knowledge of K; Ours recovers it automatically and is more robust to hyperparameters.
- **vs Single-task DML**: Extends the DML framework to multitask clustering while retaining the advantages of inference validity.
- **vs Classic Clustered Learning (Jacob et al.)**: Those methods are mostly limited to parametric models; this work handles heterogeneous semi-parametric noise, representing a significant extension.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of Neyman orthogonality and adaptive clustered fusion is novel, as is the two-stage framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three types of semi-parametric models, multiple separation levels, thorough ablation, and real-world application.
- Writing Quality: ⭐⭐⭐⭐ Mathematical rigor, clear theorem presentation, and intuitive main results.
- Value: ⭐⭐⭐⭐ Directly applicable in causal inference and economics; the theoretical framework has a profound impact on the field of multitask inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments](../../ICLR2026/optimization/feddag_clustered_federated_learning_via_global_data_and_gradient_integration_for.md)
- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](../../NeurIPS2025/optimization/robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[CVPR 2026\] ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation](../../CVPR2026/optimization/ace-merging_data-free_model_merging_with_adaptive_covariance_estimation.md)
- [\[ICLR 2026\] Incentives in Federated Learning with Heterogeneous Agents](../../ICLR2026/optimization/incentives_in_federated_learning_with_heterogeneous_agents.md)
- [\[ICML 2026\] Adaptive Preconditioners Trigger Loss Spikes in Adam](adaptive_preconditioners_trigger_loss_spikes_in_adam.md)

</div>

<!-- RELATED:END -->
