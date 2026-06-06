---
title: >-
  [Paper Note] Nested Spatio-Temporal Time Series Forecasting
description: >-
  [ICML 2026][Time Series][Spatio-Temporal Forecasting] NeST utilizes "future macro-regional trends" as top-down guidance, combined with semantic regions constructed via spectral clustering and bidirectional cross-scale at…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Spatio-Temporal Forecasting"
  - "Spectral Clustering"
  - "Macro Guidance"
  - "Cross-Scale Attention"
  - "Autoregressive Rollout"
date: 2026-05-08
content_hash: ac7e57f08a0eeb20
---

# Nested Spatio-Temporal Time Series Forecasting

**Conference**: ICML 2026  
**arXiv**: [2605.16447](https://arxiv.org/abs/2605.16447)  
**Code**: Undisclosed  
**Area**: Time Series  
**Keywords**: Spatio-Temporal Forecasting, Spectral Clustering, Macro Guidance, Cross-Scale Attention, Autoregressive Rollout  

## TL;DR
NeST utilizes "future macro-regional trends" as top-down guidance, combined with semantic regions constructed via spectral clustering and bidirectional cross-scale attention. This allows node-level spatio-temporal forecasting on large-scale traffic networks to simultaneously achieve improvements in accuracy, long-range stability, and near-linear complexity.

## Background & Motivation
**Background**: Spatio-temporal forecasting (STF) is a branch of multivariate time series forecasting. Conventional approaches organize sensors into a graph, using GNNs/Attention for spatial correlations and RNNs/TCNs for temporal dynamics. Evolution has moved from fixed topologies (DCRNN/STGCN) to adaptive adjacency (GraphWaveNet/MTGNN/AGCRN), and recently to dynamic time-varying graphs and attention (DSTAGNN/STAEFormer/PatchSTG).

**Limitations of Prior Work**: As graph size increases (thousands of sensors), fine-grained whole-graph modeling easily learns spurious correlations and becomes highly sensitive to local noise, missing values, and short-term anomalies. Autoregressive long-range forecasting also suffers from accumulated error. Existing hierarchical methods (HGCN/HiSTGNN/HSDGNN, etc.) introduce regional abstractions but treat coarse-grained signals only as historical auxiliary inputs, failing to address how future uncertainty is constrained by macro structures.

**Key Challenge**: Microscopic historical modeling at a single scale is inefficient and unstable in high-dimensional noisy scenarios; existing hierarchical frameworks only use historical macro information and cannot provide structural anchors for future trajectories.

**Goal**: (i) Unsupervised construction of regional representations that are semantically consistent with the future; (ii) explicit top-down guidance of node-level predictions using regional-level "future trends"; (iii) maintaining stability and controllable complexity of this guidance during autoregressive rollout.

**Key Insight**: The authors observe that predicting the "future macro state" first and then using it to guide fine-grained node prediction acts as a top-down regularization, similar to "sketching the outline before filling in details." The key problem shifts to ensuring macro representations are both high-fidelity and aligned with micro-structures topologically and semantically.

**Core Idea**: Use spectral clustering to construct semantically consistent regions, predict future regional-level trajectories as macro guidance, and constrain node-level predictions via bidirectional cross-attention, forming a nested coarse-to-fine autoregressive framework.

## Method

### Overall Architecture
NeST handles patch-wise autoregressive prediction for $N$ sensors, historical window length $L$, target horizon $H$, and patch length $P$. The process consists of three steps:

1.  **Offline Preprocessing**: Construct a feature-driven affinity matrix $\mathbf{A}\in\mathbb{R}^{N\times N}$ from training sequences, perform spectral clustering to obtain $M$ regions ($M<N$, set as $M=0.2N$ in experiments) and an assignment matrix $\mathbf{S}\in\{0,1\}^{N\times M}$. Regional sequences $\mathbf{Z}_{t,m}$ are obtained via intra-region average pooling.
2.  **Training Phase**: Node history $\mathbf{X}_{t-L+1:t}$ and regional future $\mathbf{Z}_{t+1:t+P}$ are projected into $d$-dimensional tokens via decoupled Linear+TE+SE layers. Bidirectional cross-attention allows node tokens to query future regional tokens (top-down), and regional tokens to query updated node tokens (bottom-up). Two heads simultaneously output $\hat{\mathbf{X}}_{t+1:t+P}$ and $\hat{\mathbf{Z}}_{t+P+1:t+2P}$.
3.  **Inference Phase**: Since the future $\mathbf{Z}$ is invisible, a boundary decoder first uses all-zero mask tokens to reconstruct $\hat{\mathbf{Z}}_{t+1:t+P}$ as the initial guidance, followed by multi-step rollout.

### Key Designs

1.  **Semantic Region Extraction and SNR Theoretical Guarantee based on Spectral Clustering**:
    - **Function**: Compress $N$ noisy nodes into $M$ semantically consistent regions to serve as structural anchors for the system.
    - **Mechanism**: Training sequences are divided into $\tilde{T}$ non-overlapping chunks based on periodicity. Nodes within each chunk are averaged to compute the affinity matrix $\mathbf{A}_{ij}=\exp(-\frac{1}{2\sigma^2\tilde{T}}\sum_k \|\mathbf{X}_i^{(k)}-\mathbf{X}_j^{(k)}\|_2^2)$, emphasizing long-term evolutionary similarity over instantaneous fluctuations. Normalized Laplacian $\mathbf{L}_{\text{sym}}=\mathbf{I}-\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ is then used to extract low-frequency eigenvectors for K-Means to obtain $\mathbf{S}$. Regional representation is defined as $\mathbf{Z}_{t,m}=\sum_i S_{i,m}\mathbf{X}_{t,i}/\sum_i S_{i,m}$.
    - **Design Motivation**: The authors prove (Theorem 1) that if the correlation coefficient of true node signals within a cluster is $\rho_m$, then $\text{SNR}(\mathbf{Z}_m)\ge[1+(|\mathcal{C}_m|-1)\rho_m]\cdot\overline{\text{SNR}}_m$. Thus, co-directional aggregation naturally acts as a low-pass filter, suppressing local high-frequency noise and preserving regional trends—explaining why affinity is driven by raw features rather than physical distance or static topology.

2.  **Bidirectional Cross-Scale Attention (Top-down Guidance + Bottom-up Refinement)**:
    - **Function**: Couple historical fine-grained dynamics with future coarse-grained trends, allowing the macro future to regularize node predictions.
    - **Mechanism**: Top-down is performed first, where node tokens query future regional tokens, $\tilde{\mathbf{H}}_x=\text{Attn}(\mathbf{H}_x^{\text{past}},\mathbf{H}_z^{\text{fut}},\mathbf{H}_z^{\text{fut}})$, allowing node representations to absorb macro trends. Then bottom-up is performed, where updated nodes refine regional tokens, $\tilde{\mathbf{H}}_z=\text{Attn}(\mathbf{H}_z^{\text{fut}},\tilde{\mathbf{H}}_x,\tilde{\mathbf{H}}_x)$, anchoring macro guidance to the latest fine-grained context. Complexity is reduced from $\mathcal{O}(lN^2 d)$ in full self-attention to $\mathcal{O}(lNMd)$, which is near-linear since $M<N$.
    - **Design Motivation**: Unidirectional top-down guidance causes regional tokens to decouple from history, while unidirectional bottom-up guidance degrades into existing hierarchical methods. Bidirectional coupling ensures mutual calibration between "future macro trends" and "current fine-grained states." Use of cluster count $M$ as a bottleneck is the fundamental reason it scales to large graphs.

3.  **Boundary Reconstruction + Multi-step Rollout + Quantile Regression Uncertainty**:
    - **Function**: Address exposure bias between training (teacher forcing) and inference (invisible future $\mathbf{Z}$), and handle inaccuracies in macro guidance robustly.
    - **Mechanism**: During training, ground-truth regional tokens are used with probability $P_{\text{tf}}$, and the model's own rollout $\hat{\mathbf{Z}}$ with $1-P_{\text{tf}}$. Simultaneously, a boundary decoder $\hat{\mathbf{Z}}_{t+1:t+P}=\text{Proj}_{\text{bd}}(\text{Attn}(\mathbf{H}_z^{\text{zeros}},\tilde{\mathbf{H}}_x,\tilde{\mathbf{H}}_x))$ is explicitly trained to learn a "macro starting point" from node history when future observations are absent. The regional head uses quantile regression to estimate multiple conditional quantiles $\{\tau_q\}_{q=1}^Q$, with only the median $\tau=0.5$ used as guidance during inference. The total loss is $\mathcal{L}=\mathcal{L}_x+\lambda_1\mathcal{L}_z+\lambda_2\mathcal{L}_{\text{bd}}$.
    - **Design Motivation**: Pure teacher forcing leads to out-of-distribution inputs during inference, while pure rollout is unstable early on. The boundary decoder provides a "cold-start anchor," after which regional stability takes over. Quantile regression transforms macro guidance from point estimation to distribution estimation, making it more robust to guidance errors—crucial for 12-hour long horizons.

### Loss & Training
End-to-end multi-task training: (i) Node-level prediction loss $\mathcal{L}_x$; (ii) Regional-level multi-quantile loss $\mathcal{L}_z$ (pinball loss); (iii) Boundary reconstruction loss $\mathcal{L}_{\text{bd}}$. Lookback $L=12$, horizon $H=12$, generated autoregressively with patch length $P$. $\tilde{T}$ is aligned with intrinsic data periods, and $M=0.2N$ is optimal.

## Key Experimental Results

### Main Results
Evaluation on GBA, GLA, and CA large-scale traffic datasets from the LargeST benchmark (nodes ranging from thousands to over ten thousand), compared against 11 SOTA baselines.

| Dataset | Metric | NeST | PatchSTG (Prev. SOTA) | Gain |
| :--- | :--- | :--- | :--- | :--- |
| GBA (Avg. Horizon 12) | MAE | 18.73 | 19.50 | 3.95% |
| GBA | MAPE | 12.90% | 14.64% | 11.88% |
| GLA (Avg.) | MAE | 17.89 | 18.96 | 5.65% |
| GLA | MAPE | 10.74% | 11.44% | 6.14% |
| CA (Avg.) | MAE | 16.54 | 17.35 | 4.69% |
| CA | MAPE | 11.28% | 12.79% | 11.78% |

Average across three datasets: MAE +4.71%, RMSE +4.41%, MAPE +9.34%. For long-horizon (48 steps / 12 hours autoregressive rollout), the MAE gap between NeST and PatchSTG on GLA widens from 2.0 at step 16 to 2.4 at step 48, proving that macro guidance effectively enhances long-range stability. NeST also outperforms MAGE / iTransformer / Air-DualODE in non-traffic domains like KnowAir / UrbanEV / Electricity / Solar-Energy.

### Ablation Study

| Configuration | GBA MAE | GBA RMSE | GLA MAE | Description |
| :--- | :--- | :--- | :--- | :--- |
| NeST (Full) | **18.73** | **31.85** | **17.89** | Full Model |
| w/o CA | 19.76 | 34.11 | 19.00 | Remove cross-attention (largest drop: +1.04 MAE) |
| w/o FG | 19.64 | 32.89 | 18.85 | Replace future Z with historical Z (drop: +0.91 MAE) |
| w/ KM | 18.93 | 32.33 | 18.39 | Use K-Means on raw features instead of spectral clustering |
| w/ RP | 19.07 | 32.47 | 18.46 | Random partitioning (MAPE worsens by 13% on GBA) |
| w/ DA | 18.93 | 32.22 | 18.34 | Use static geographic distance for affinity |

### Key Findings
- **Cross-attention is core**: Removing it causes the model to degrade into a pure local predictor with the most significant performance drop, proving that top-down macro regularization is a vital pillar, not just an auxiliary.
- **"Future" is crucial**: w/o FG (using only historical regions) still shows a significant drop, indicating that upgrading macro signals from "historical auxiliary" to "future guidance" is the true delta of the paper, rather than just the hierarchical structure.
- **Semantic clustering > Geographic clustering**: w/ DA (physical proximity) performs slightly better than w/ KM, but neither matches spectral clustering—suggesting that functional similarity in large-scale traffic networks is indeed orthogonal to geographic location (Case study: Region 547 spans non-contiguous segments but shares evolution patterns).
- **$M=0.2N$ is the U-shaped optimum**: Too few ($0.1N$) loses local patterns through over-aggregation; too many ($0.3N$) drowns in structural noise.
- **High efficiency**: Training time on GBA reduced from 185s to 75s/epoch (-59.5%), inference from 32s to 20s (-37.5%); affinity matrix preprocessing (91s) is a one-time cost.

## Highlights & Insights
- **The paradigm of "predicting future macro to guide micro" is genuinely novel**: Previous hierarchical methods mostly treated coarse granularity as historical input. Placing it on the future side with a boundary reconstruction mechanism turns the macro signal into a usable top-down prior. This idea is transferable to any hierarchical sequence modeling task (video, trajectory, molecular dynamics).
- **Elegant coupling of SNR theory and structural design**: Theorem 1 explains "why cluster-based average pooling is a low-pass filter," elevating spectral clustering from an "empirical choice" to a "mathematically guaranteed" one. This "theory explaining experience" narrative is highly effective.
- **Reducing complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(NM)$ is key for large graphs**: Cross-attention caps the quadratic growth of self-attention, which, alongside patch-based methods like PatchSTG, represents a primary pathway for large-scale traffic prediction.
- **Boundary decoder + scheduled sampling is a practical exposure bias solution**: Reconstructing with masks during training and starting with masks during inference cleans up the train-inference distribution alignment. This can be directly applied to any teacher-forcing + autoregressive scenario.

## Limitations & Future Work
- **Limitations acknowledged by authors**: (i) $\mathcal{O}(N^2)$ preprocessing for affinity matrix construction becomes a bottleneck at the 100k node level; (ii) Global clustering assumes time-invariant spatial correlation, which adapts poorly to abrupt topological changes (accidents, temporary control); (iii) Exposure bias persists in teacher-forcing; (iv) Autoregressive rollout is slower than direct multi-step forecasting.
- **Personal observations**: (i) $M=0.2N$ seems robust but requires tuning per dataset; (ii) Lacks direct efficiency/accuracy Pareto comparison with channel-independent models like PatchTST/iTransformer; (iii) The quantile head only utilizes the median; uncertainty information is not feedbacked into node prediction loss or confidence estimation, wasting data.
- **Ideas for improvement**: Dynamic clustering (updating $\mathbf{S}$ per time window), replacing boundary decoder with a diffusion prior, and backpropagating quantile uncertainty to node heads for risk-aware decoding.

## Related Work & Insights
- **vs PatchSTG**: PatchSTG reduces complexity via patching and spatial data management. NeST uses clustering and macro guidance. Both achieve near-linear complexity but for different reasons—combining them is a natural next step.
- **vs HiSTGNN / HIEST / HSDGNN**: Traditional hierarchical methods lack the "predict future macro then guide back" loop; NeST closes this loop with boundary reconstruction and rollout.
- **vs Jiang et al. 2025 (neural operator with future info)**: That work proves "future information stabilizes long-range prediction" but requires regular grids; NeST adapts this to irregular, noisy, and incomplete graph data.
- **vs iTransformer / MAGE**: Channel-independent time series models are NeST's strongest rivals on non-traffic data; NeST’s advantage lies in explicit spatial structure. Fusing channel mixing with macro guidance could be the next breakthrough.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Using "future macro prediction" as a top-down guide is a true paradigm delta, not just module stacking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 3 LargeST datasets + 4 non-traffic domains + ablation + long horizon + runtime analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Smooth narrative across theory (SNR), intuition (low-pass filter), and engineering (boundary decoder). Figure 2 case study is very intuitive.
- **Value**: ⭐⭐⭐⭐ Achieve 4-6% MAE / 10%+ MAPE gains in a highly competitive track while being 2x faster, offering high industrial value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic.md)
- [\[NeurIPS 2025\] Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting](../../NeurIPS2025/time_series/learning_with_calibration_exploring_test-time_computing_of_spatio-temporal_forec.md)
- [\[ACL 2026\] STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning](../../ACL2026/time_series/streasoner_empowering_llms_for_spatio-temporal_reasoning_in_time_series_via_spat.md)
- [\[ICML 2026\] Ellipsoidal Time Series Forecasting](ellipsoidal_time_series_forecasting.md)
- [\[NeurIPS 2025\] StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](../../NeurIPS2025/time_series/strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)

</div>

<!-- RELATED:END -->
