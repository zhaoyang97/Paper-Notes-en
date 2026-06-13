---
title: >-
  [Paper Note] Active Timepoint Selection for Learning Measure-Valued Trajectories
description: >-
  [ICML 2026][Computational Biology][Active Sampling] This paper investigates the optimal timing for sampling a distribution snapshot. It employs Linearized Optimal Transport (LOT) to linearize measure trajectories in Wass…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Active Sampling"
  - "Wasserstein Trajectory"
  - "Linearized Optimal Transport"
  - "Gaussian Process"
  - "Single-Cell Time Series"
date: 2026-05-08
content_hash: 72598c5d155a6041
---

# Active Timepoint Selection for Learning Measure-Valued Trajectories

**Conference**: ICML 2026  
**arXiv**: [2605.30625](https://arxiv.org/abs/2605.30625)  
**Code**: https://github.com/nicolashuynh/active_wass  
**Area**: Time Series / Measure-Valued Trajectory Learning  
**Keywords**: Active Sampling, Wasserstein Trajectory, Linearized Optimal Transport, Gaussian Process, Single-Cell Time Series  

## TL;DR
This paper investigates the optimal timing for sampling a distribution snapshot. It employs Linearized Optimal Transport (LOT) to linearize measure trajectories in Wasserstein space and utilizes multi-output Gaussian Processes (GPs) with time warping to provide epistemic uncertainty, thereby actively selecting timepoints that most effectively reduce trajectory reconstruction errors.

## Background & Motivation
**Background**: In scenarios such as single-cell transcriptomics, fluid dynamics, and macroeconomics, the research objects are often probability distribution paths over time rather than single vector time series. Physical observations typically consist of empirical measures at discrete timepoints, and the task involves recovering continuous measure-valued trajectories from these sparse snapshots.

**Limitations of Prior Work**: Generating high-quality snapshots is expensive, and experiments like single-cell sequencing often involve destructive sampling, preventing dense temporal observation. Traditional active learning assumes outputs reside in Euclidean space, allowing for direct use of GP posterior variance for sampling decisions. However, probability measures live in Wasserstein space, where linear averaging leads to "ghosting" effects (splitting mass), and standard GPs applied to density vectors violate transport geometry.

**Key Challenge**: Active sampling necessitates identifying areas of high uncertainty. Existing Wasserstein interpolation or flow methods generally yield a single deterministic trajectory, lacking available epistemic uncertainty. Furthermore, processes such as biological differentiation are highly non-stationary: they change slowly for long periods but bifurcate rapidly in narrow windows, which uniform sampling often fails to capture.

**Goal**: To select the most informative timepoints under a fixed observation budget to improve the accuracy of recovered probability paths in terms of Wasserstein error, specifically targeting regions of rapid change and transient branching.

**Key Insight**: The authors utilize Linearized Optimal Transport (LOT) to map each measure snapshot to the tangent space of a reference measure. PCA and GP are subsequently applied within this linear space. This approach preserves a first-order approximation of Wasserstein geometry while leveraging the GP posterior covariance to obtain the uncertainty required for active sampling.

**Core Idea**: The measure trajectory is first projected into the LOT tangent space, followed by constructing a warped GP on low-dimensional latent coefficients to select the next measurement time based on posterior variance.

## Method
The proposed method follows a cycle of "geometric linearization + probabilistic surrogate + active sampling." In each round, a Wasserstein barycenter of the existing snapshots is estimated as the reference measure to transform all snapshots into displacement fields relative to this reference. These high-dimensional displacements are compressed into low-dimensional coefficients, upon which a GP is fitted. The GP uncertainty is then used to select the next measurement time.

### Overall Architecture
The input consists of a set of existing snapshots $\mathcal{D}=\{(t_i,\hat{\mu}_{t_i})\}_{i=1}^N$, a candidate time pool $\mathcal{T}_{pool}$, and a remaining sampling budget $B$. In each iteration, the algorithm updates the reference measure $\sigma$ and maps each snapshot $\hat{\mu}_{t_i}$ to $T_\sigma\mathcal{P}_2(\mathcal{X})$ via OT coupling to obtain a displacement matrix $\mathbf{V}_i$. Weighted PCA is then applied to derive low-dimensional coefficients $\mathbf{c}_i$, forming the GP training set $\{(t_i,\mathbf{c}_i)\}$.

To account for non-stationary dynamics, the authors estimate an intrinsic time $\tau=\Phi(t)$ rather than using stationary kernels on physical time $t$. $\Phi(t)$ approximates the cumulative Wasserstein arc length (the sum of transport distances between adjacent snapshots), extended to candidate times via monotonic cubic splines. The GP kernel is defined as $\mathbf{K}(t,t')=\mathbf{K}_{base}(\Phi(t),\Phi(t'))$, which automatically results in shorter effective lengthscales in regions of rapid change.

For active sampling, the paper primarily uses point-wise uncertainty: $\alpha_{unc}(t;\mathcal{D})=\mathrm{Tr}(\mathbf{S}(\Phi(t)))$, though an expected integrated risk reduction version is also provided. After selecting $t^*$, a real measurement is performed, and the new snapshot is added to the dataset for the next round.

### Key Designs
1. **LOT Tangent Space Representation**:
    - **Function**: Transforms non-Euclidean probability measure outputs into regressible vector objects.
    - **Mechanism**: Using reference measure $\sigma$ as an anchor, the OT coupling from $\sigma$ to the target snapshot $\hat{\mu}_{t_i}$ is computed, and an approximate transport map is obtained via barycentric projection. The displacement field $\mathbf{V}_i=\hat{\mathbf{Z}_i}-\mathbf{Z}_\sigma$ serves as the representation of the snapshot in the tangent space.
    - **Design Motivation**: Applying GPs directly to densities "averages" mass into incorrect positions; LOT represents changes via transport displacement, which is more consistent with Wasserstein geometry.

2. **Low-dimensional GP Surrogate and Distribution Reconstruction**:
    - **Function**: Provides a posterior mean and epistemic uncertainty for the probability path across continuous time.
    - **Mechanism**: PCA is performed on the weighted flattened displacements to extract latent coefficients $\mathbf{c}_i$. A multi-output GP is built for the mapping $t \mapsto \mathbf{c}(t)$. Predictions are back-projected from the GP posterior mean or samples to displacement fields and added back to the reference landmarks.
    - **Design Motivation**: Performing GP directly on high-dimensional displacement fields is computationally expensive and data-scarce. PCA extracts the primary directions of variation, stabilizing uncertainty estimation.

3. **Intrinsic Time Warping and Acquisition**:
    - **Function**: Sensitizes the sampling strategy to non-stationary trajectories, prioritizing coverage of high-velocity windows.
    - **Mechanism**: The cumulative arc length $\Phi(t)$ is estimated using Wasserstein distances between adjacent snapshots, and the GP kernel computes correlations based on $\Phi(t)$. The acquisition function selects the timepoint with the maximum posterior covariance trace.
    - **Design Motivation**: Assuming stationarity in physical time causes the model to underestimate uncertainty during short bifurcations or sudden transitions. With intrinsic time, rapidly changing regions are "stretched" within the model, making them more likely to be captured by active sampling.

### Loss & Training
The method is not an end-to-end neural network but reconstructs a probabilistic surrogate in each round. Core optimizations include OT coupling, Wasserstein barycenter computation, PCA, and maximization of the GP marginal log-likelihood for hyperparameters. In default experiments, the multi-output GP is simplified to independent GPs for each latent dimension using a Matérn 5/2 kernel. Acquisition is calculated over a fixed candidate pool.

Computational complexity is dominated by OT. Given $N$ snapshots with $n$ samples each and $M$ landmarks in the reference measure, LOT embedding and time warping require approximately $O(N \cdot \mathcal{C}_{OT}(M,n))$. GP computation is typically not a bottleneck given the low-dimensional latent space and small number of snapshots.

## Key Experimental Results

### Main Results
The method was validated on synthetic branching trajectories, real-world single-cell fibroblast reprogramming data, and labor market data. Quantitative results indicate that active sampling outperforms uniform or random sampling, particularly as the branching window length decreases.

| Branching Window Length | vs Uniform: Rel. W2 ↑ | vs Uniform: Rel. w-W2 ↑ | vs Random: Rel. W2 ↑ | vs Random: Rel. w-W2 ↑ | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.05 | 0.231 | 0.357 | 0.342 | 0.454 | Short bifurcations benefit most from active sampling with velocity weighting. |
| 0.10 | 0.189 | 0.277 | 0.397 | 0.464 | Stable advantage maintained as the window widens. |
| 0.20 | -0.172 | 0.071 | 0.118 | 0.295 | Uniform sampling covers wide windows well, but active sampling still prioritizes high-velocity regions. |

In real single-cell reprogramming experiments, the active strategy achieved the lowest reconstruction error at low-to-medium budgets ($B \leq 12$). As the budget increases, the gap narrows as uniform/random sampling also covers most transient phases.

### Ablation Study
The impact of four design components in the surrogate/acquisition pipeline was analyzed:

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full method | Lowest reconstruction error | LOT barycenter + sufficient PCA dims + time warping + Matérn GP. |
| RBF kernel | Close to Full method | Suggests the method is not strictly dependent on a specific prior. |
| Fixed reference $\sigma$ | Significant error increase | Failure to update the reference measure amplifies LOT linearization errors. |
| PCA rank $K=2$ | Most significant decline | Low-dimensional latents cannot express complex transcriptomic variations. |
| No warp | Degradation at low budgets | Proves intrinsic time is essential for non-stationary dynamics. |

### Key Findings
- The advantage of the active strategy stems from "allocating budget to high-velocity regions" rather than simply acquiring more samples. Visualizations show acquisition points concentrated near branching windows.
- Localized events are most easily missed by uniform sampling. While uniform sampling's disadvantage diminishes for wider event windows, the velocity-weighted error still favors the active method.
- PCA dimensionality is critical. Significant degradation at $K=2$ shows that primary variations in measure trajectories are not always representable by a very small number of components.
- Dynamically updating the Wasserstein barycenter is crucial to minimize LOT approximation errors as the distribution drifts.

## Highlights & Insights
- The paper adapts "uncertainty sampling" for active learning to Wasserstein space by identifying a computable uncertainty surrogate rather than attempting a direct Bayesian prior in measure space.
- Intrinsic time warping is intuitive: biological physical time and geometric change velocity are often uncoupled. Re-scaling time using transport distance is more structurally sound than tuning kernel lengthscales.
- The method is highly relevant for expensive experimental designs (e.g., "where to sample next" rather than high-frequency real-time sampling).
- The LOT + GP combination is more interpretable than end-to-end deep models, as sampling decisions can be explained via posterior variance and trajectory velocity.

## Limitations & Future Work
- The tangent space approximation is a core assumption. If the trajectory deviates significantly from the reference or contains large jumps, a single tangent chart may be insufficient; multi-chart or atlas-like methods could be considered.
- OT subproblems represent the primary computational cost. This is acceptable for expensive snapshots with $\sim 10^5$ samples but may require scalable OT approximations for million-point clouds.
- The input is currently restricted to 1D time. Extension to multi-dimensional covariates (dosage, perturbation types, spatial location) is needed for broader scientific applications.
- While reconstruction error is minimized, further work could connect the method to downstream tasks like fate prediction or critical state discovery.

## Related Work & Insights
- **vs. Euclidean GP on densities**: Standard GP regression on density vectors leads to mass splitting; this work uses LOT displacement to preserve transport directionality.
- **vs. Deterministic distribution interpolation/flow matching**: These methods fit trajectories from fixed snapshots but lack epistemic uncertainty; the GP posterior in this work enables active acquisition.
- **vs. Single-cell timepoint selection**: Early methods focused on Euclidean gene-expression curves; this work directly processes empirical measures, better suited for population distribution shifts.
- **Insight**: For active learning in non-Euclidean output spaces, mapping to a local linearization or low-dimensional chart before building an interpretable uncertainty model is an effective strategy.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Combines LOT, GP uncertainty, and active timepoint selection for measure-valued trajectories; both the problem setting and approach are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes synthetic, real single-cell, and supplementary data, though some ablations are limited to graphical representations.
- **Writing Quality**: ⭐⭐⭐⭐ Geometric motivation is clear and algorithm details are complete, though requires background in OT/GP.
- **Value**: ⭐⭐⭐⭐⭐ Highly insightful for expensive experimental design and Wasserstein trajectory modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PROSPERO: Active Learning for Robust Protein Design Beyond Wild-Type Neighborhood](../../NeurIPS2025/computational_biology/prospero_active_learning_for_robust_protein_design_beyond_wild-type_neighborhood.md)
- [\[ICLR 2026\] Controllable Sequence Editing for Biological and Clinical Trajectories](../../ICLR2026/computational_biology/controllable_sequence_editing_for_biological_and_clinical_trajectories.md)
- [\[NeurIPS 2025\] Amortized Active Generation of Pareto Sets](../../NeurIPS2025/computational_biology/amortized_active_generation_of_pareto_sets.md)
- [\[ICML 2026\] Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes](flow_sampling_learning_to_sample_from_unnormalized_densities_via_denoising_condi.md)
- [\[NeurIPS 2025\] Modeling Microenvironment Trajectories on Spatial Transcriptomics with NicheFlow](../../NeurIPS2025/computational_biology/modeling_microenvironment_trajectories_on_spatial_transcriptomics_with_nicheflow.md)

</div>

<!-- RELATED:END -->
