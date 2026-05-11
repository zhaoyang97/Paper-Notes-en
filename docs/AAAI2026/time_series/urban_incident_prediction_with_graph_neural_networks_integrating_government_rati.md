---
title: >-
  [Paper Note] Urban Incident Prediction with Graph Neural Networks: Integrating Government Ratings and Crowdsourced Reports
description: >-
  [AAAI 2026][Time Series][Urban incident prediction] This paper proposes URBAN, a multi-view multi-output GNN model that jointly leverages sparse but unbiased government inspection rating data and dense but biased crowdso…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "Urban incident prediction"
  - "GNN"
  - "crowdsourced data bias"
  - "multi-view learning"
  - "latent state estimation"
date: 2026-05-08
content_hash: 99457cad8c3bb4a2
---

# Urban Incident Prediction with Graph Neural Networks: Integrating Government Ratings and Crowdsourced Reports

**Conference**: AAAI 2026
**arXiv**: [2506.08740](https://arxiv.org/abs/2506.08740)
**Code**: [GitHub](https://github.com/sidhikabalachandar/nyc_urban_incident_model)
**Area**: Time Series / Graph Neural Networks
**Keywords**: Urban incident prediction, GNN, crowdsourced data bias, multi-view learning, latent state estimation

## TL;DR
This paper proposes URBAN, a multi-view multi-output GNN model that jointly leverages sparse but unbiased government inspection rating data and dense but biased crowdsourced report data to predict the true latent state of urban incidents. Validated on 9.6M+ reports and 1M+ inspections in New York City, the model achieves a 5.3× higher prediction correlation than using report data alone.

## Background & Motivation
**Background**: Urban management requires knowledge of which neighborhoods suffer from potholes, rat infestations, and similar incidents in order to allocate resources effectively. GNNs have been widely adopted for urban spatiotemporal prediction.

**Limitations of Prior Work**: Both available data sources are individually flawed—(a) government inspection ratings are unbiased but extremely sparse (e.g., NYC streets are inspected only once per year); (b) crowdsourced reports are dense but exhibit systematic bias (higher-income communities report at higher rates, and reporting propensity varies across neighborhoods under identical conditions).

**Key Challenge**: Training solely on report data yields biased predictions, while training solely on ratings faces severe sparsity. A fundamental identifiability problem exists between the two sources: does a high report count reflect a genuine problem or a community's greater tendency to report?

**Goal**: (a) Jointly leverage both data sources to predict true incident states; (b) quantify demographic bias present in crowdsourced reports.

**Key Insight**: Ratings and reports are modeled as different observations of a shared latent ground-truth state. A GNN learns node embeddings to capture spatial correlations, while type embeddings capture inter-incident-type associations.

**Core Idea**: A multi-output loss function bridges the sparse unbiased rating view and the dense biased report view, enabling the GNN to simultaneously estimate the true latent state and the reporting bias.

## Method

### Overall Architecture
**Input**: Urban graph (nodes = Census tracts, edges = adjacency relationships) + demographic features $X_i$ + sparse ratings $r_{ikt}$ + dense reports $T_{ikt}$. **Output**: Predicted ratings $\hat{r}_{ikt}$ and report probabilities $\hat{P}(T_{ikt})$ for all nodes, types, and time steps, along with demographic bias coefficients $\theta_k$.

### Key Designs

1. **Dual-Embedding Rating Prediction**:

    - **Function**: Predict the true inspection rating for incident type $k$ at node $i$.
    - **Mechanism**: A GCN learns node embeddings $e_n[i]$ (capturing spatial correlations) alongside type embeddings $e_\tau[k]$ (capturing inter-type relationships); the predicted rating is their inner product $\hat{r}_{ikt} = e_n[i]^\top e_\tau[k]$.
    - **Design Motivation**: Analogous to matrix factorization, the inner product of node and type embeddings naturally encodes the severity of a given incident type in a given area.

2. **Case-Wise Report Probability Modeling (Three Cases)**:

    - **Function**: Model report probability using different strategies depending on rating data availability.
    - **Case 1** (rating available): $\hat{P}(T_{ikt}) = \sigma(\alpha_k r_{ikt} + \theta_k^\top X_i)$, using the observed rating directly.
    - **Case 2** (no rating for this node, but ratings exist for this type): Replace with predicted rating $\hat{r}_{ikt}$; reporting coefficients $[\alpha_k, \theta_k]$ are learned from rated nodes.
    - **Case 3** (no rating available for this type at all): Use the average reporting coefficients $[\bar\alpha, \bar\theta]$ across all rated types.
    - **Design Motivation**: Addresses the identifiability problem—rating values and reporting coefficients cannot be learned simultaneously; annotated data must anchor one set of parameters.

3. **Multi-Output Weighted Loss Function**:

    - **Function**: Jointly optimize rating prediction and report prediction.
    - $\mathcal{L} = \mathcal{L}_{\text{unobs}} + \gamma_1 \mathcal{L}_{\text{obs}} + \gamma_2 \mathcal{L}_{\text{rating}} + \gamma_3 \mathcal{L}_{\text{reg}}$
    - Components: BCE for report prediction at unrated nodes, BCE for report prediction at rated nodes, MSE for rating prediction, and regularization to prevent predicted ratings from growing unbounded.
    - **Design Motivation**: The multi-output loss connects the two data views, allowing the model to implicitly exploit rating information when predicting reports.

### Loss & Training
The loss consists of four weighted terms; weights $\gamma_1, \gamma_2, \gamma_3$ are selected via hyperparameter search to minimize prediction RMSE on a validation set. Temporal split: the first 18 months are used for training and the final 6 months for testing.

## Key Experimental Results

### Main Results (Real Data, New York City)

| Model | Rating Prediction $r$ | Report Prediction $r$ | Notes |
|---|---|---|---|
| URBAN (full) | **0.53** | **0.24** | Jointly uses both data sources |
| Reports-only | 0.10 | — | Uses report data only |
| Ratings-only | Varies with sparsity | — | Uses rating data only |
| Gaussian Process | 0.03 | — | Report data + GP |
| Test-set reports → ratings | 0.018 | — | Reports are naturally weakly correlated with true ratings |

### Ablation Study (Semi-Synthetic Data)

| Configuration | Rating Prediction $r$ | Notes |
|---|---|---|
| Full model | 0.62 | Complete model |
| Reports-only | 0.33 | −47%; fails to recover true reporting coefficients |
| Ratings-only (sparse regime) | → 0 | Approaches random under extreme sparsity |
| Reporting coefficient recovery | $r \approx 1.0$ | Accurately recovers true reporting parameters |

### Key Findings
- **The full model's rating prediction is 5.3× that of the reports-only baseline** (0.53 vs. 0.10), confirming the value of joint modeling.
- Under extreme rating sparsity, the full model continues to leverage report data to maintain predictive performance, whereas the ratings-only model degrades to near-random predictions.
- **Higher-income communities exhibit higher conditional reporting rates** ($\theta_{\text{income}} = 0.173$); communities with higher population density, higher educational attainment, and a larger proportion of white residents also tend to report more.
- Learned ratings are spatially highly correlated and cluster into geographically meaningful zones.

## Highlights & Insights
- **Elegant resolution of the identifiability problem**: By handling nodes with and without ratings through three distinct cases, the model elegantly sidesteps the difficulty of simultaneously learning ratings and reporting coefficients. This case-wise modeling strategy is transferable to other settings with sparse ground-truth labels and biased proxies.
- **Highly practical dataset**: 9.6M crowdsourced reports and 1M government inspections spanning 139 incident types and 2,292 Census tracts, with complete code and data provided.
- **Policy implications of bias quantification**: The work quantitatively demonstrates that resource allocation based solely on 311 reports systematically disadvantages low-income and minority communities.

## Limitations & Future Work
- **No temporal dynamics modeling**: The current model learns a fixed rating per node and type, without explicitly modeling temporal evolution (though the authors discuss extensions to spatiotemporal GNNs).
- The **logistic assumption for report probability** is overly simplistic; the real report generation process is considerably more complex.
- The **assumption that government inspections are unbiased** warrants careful scrutiny—although the authors filter out reactive inspections, selection bias may remain.
- Attention mechanisms could be introduced to allow different incident types to share more flexible reporting coefficients.

## Related Work & Insights
- **vs. Agostini et al. 2024 (AAAI)**: That work uses a Bayesian spatial model to correct for underreporting but handles only a small number of incident types under strong assumptions; the present paper uses a GNN to handle 100+ types in a unified framework.
- **vs. Casey et al. 2018**: Their findings show that 311 reports in Washington, D.C. cannot predict rat inspection outcomes; the present paper resolves this limitation through joint modeling.
- The general paradigm of "sparse ground truth + dense biased proxy" is extensible to domains such as epidemic forecasting (search trends + official case counts).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The multi-view GNN approach to handling biased data is novel and practically motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Both real and semi-synthetic experiments are conducted; the dataset is at the 9.6M-record scale; bias analysis is thorough.
- **Writing Quality**: ⭐⭐⭐⭐ The problem is clearly articulated, though the notation is somewhat dense.
- **Value**: ⭐⭐⭐⭐ Direct implications for urban governance; the methodology is generalizable to other domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WARP: Weight-Space Linear Recurrent Neural Networks](../../ICLR2026/time_series/weight-space_linear_recurrent_neural_networks.md)
- [\[AAAI 2026\] Transparent Networks for Multivariate Time Series](transparent_networks_for_multivariate_time_series.md)
- [\[NeurIPS 2025\] A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](../../NeurIPS2025/time_series/a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](../../NeurIPS2025/time_series/simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[AAAI 2026\] SELDON: Supernova Explosions Learned by Deep ODE Networks](seldon_supernova_explosions_learned_by_deep_ode_networks.md)

</div>

<!-- RELATED:END -->
