---
title: >-
  [Paper Note] Transparent Networks for Multivariate Time Series
description: >-
  [AAAI 2026][Time Series][Interpretable Models] This paper proposes GATSM (Generalized Additive Time Series Model), a transparent neural network for time series that employs weight-sharing feature networks to learn featur…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "Interpretable Models"
  - "Generalized Additive Models"
  - "Transparent Networks"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: a8c7f8d40d55e59c
---

# Transparent Networks for Multivariate Time Series

**Conference**: AAAI 2026
**arXiv**: [2410.10535](https://arxiv.org/abs/2410.10535)  
**Code**: [https://github.com/gim4855744/GATSM](https://github.com/gim4855744/GATSM)  
**Area**: Time Series
**Keywords**: Interpretable Models, Generalized Additive Models, Time Series, Transparent Networks, Attention Mechanism

## TL;DR

This paper proposes GATSM (Generalized Additive Time Series Model), a transparent neural network for time series that employs weight-sharing feature networks to learn feature representations and masked multi-head attention to capture temporal patterns. GATSM achieves performance comparable to black-box models such as Transformers while maintaining full interpretability.

## Background & Motivation

### State of the Field

In high-stakes domains such as healthcare and fraud detection, model **transparency (interpretability)** is critical. Existing interpretability approaches fall into two categories:
- **Post-hoc XAI**: Methods such as LIME and SHAP explain already-trained black-box models, but may produce incorrect or unfaithful explanations that do not reflect the true contribution of input features.
- **Inherently Interpretable Models**: Models whose structure is interpretable by design, such as Generalized Additive Models (GAMs).

### Limitations of Prior Work

Classic GAMs take the form $g(\mathbb{E}(y|\mathbf{x})) = \sum_{i=1}^M f_i(x_i)$, where each feature has an independent function that directly reveals its contribution. However, applying GAMs to time series faces three challenges:

**Inability to handle sequential data**: Conventional tabular GAMs assume static feature vectors and cannot process temporal sequences.

**Inability to capture temporal patterns**: NATM naively applies independent functions $f_{i,j}(x_{i,j})$ at each time step without cross-step interactions, preventing the model from learning temporal dependencies.

**Fixed-length constraint**: NATM requires fixed-length inputs and cannot handle variable-length time series (e.g., patient stays of unpredictable duration in clinical settings).

### Paper Goals

The authors define a novel temporal GAM formulation:

$$g(\mathbb{E}(y_t | \mathbf{X}_{:t})) = \sum_{i=1}^{t} \sum_{j=1}^{M} f_{i,j}(x_{i,j}, \mathbf{X}_{:t})$$

The key distinction is that each function $f_{i,j}$ can take the entire historical sequence $\mathbf{X}_{:t}$ as an additional input, enabling temporal pattern capture while preserving the additive structure. GATSM is the first transparent model to realize this formulation.

## Method

### Overall Architecture

GATSM consists of two modules:
1. **Time-Sharing NBM (Neural Basis Model)**: Learns nonlinear feature representations.
2. **Masked MHA (Masked Multi-Head Attention)**: Captures temporal patterns.

### Key Designs

#### 1. Time-Sharing NBM

**Problem**: Assigning independent functions to every time step and feature requires $T \times M$ functions, leading to parameter explosion.

**Solution**: Share basis function weights across all time steps.

$$\tilde{x}_{i,j} = f_j(x_{i,j}) = \sum_{k=1}^{B} h_k(x_{i,j}) w_{j,k}^{nbm}$$

- $B = 100$ basis functions $h_k(\cdot)$ implemented as MLPs.
- Basis functions are shared across all features and time steps, substantially reducing parameters.
- Each feature retains independent weights $w_{j,k}^{nbm}$ to preserve feature specificity.
- Parameter count is reduced from $T \times M$ to $B$.

**Design Motivation**: The basis strategy of NBM is particularly well-suited for time series — the same nonlinear transformation can be shared across time steps for a given feature, while feature-specific weights maintain discriminative capacity.

#### 2. Masked MHA

A two-layer attention mechanism (derived from GAT) is adopted instead of simple dot-product attention to achieve greater expressive power.

**Step 1**: Transform feature representations and add positional encodings.

$$\mathbf{v}_i = \tilde{\mathbf{x}}_i^\intercal \mathbf{Z} + \mathbf{pe}_i$$

- $\mathbf{Z} \in \mathbb{R}^{M \times D}$ is a learnable weight matrix.
- Sinusoidal positional encodings (rather than learnable ones) are used to support variable-length sequences.

**Step 2**: Compute attention scores.

$$e_{k,i,j} = \sigma([\mathbf{v}_i | \mathbf{v}_j]^\intercal \mathbf{w}_k^{attn}) m_{i,j}$$

$$a_{k,i,j} = \frac{\exp(e_{k,i,j})}{\sum_{t=1}^{T} \exp(e_{k,i,t})}$$

- A causal mask $m_{i,j}$ ensures that time step $i$ attends only to time steps $j \leq i$.
- The nonlinear activation $\sigma(\cdot)$ enhances expressive power.

#### 3. Inference and Interpretability

The final prediction is:

$$\hat{y}_t = \sum_{k=1}^{K} \mathbf{a}_{k,t}^\intercal \tilde{\mathbf{X}} \mathbf{w}_k^{out}$$

Expanding into scalar form:

$$= \sum_{u=1}^{t} \sum_{m=1}^{M} \underbrace{\sum_{k=1}^{K} \sum_{b=1}^{B} a_{k,t,u} h_b(x_{u,m}) w_{m,b}^{nbm} w_{k,m}^{out}}_{f_{u,m}(x_{u,m}, \mathbf{X}_{:t})}$$

This demonstrates that GATSM satisfies the temporal GAM definition (Definition 3.1) and supports **three levels of interpretability**:

1. **Time step importance**: $a_{k,t,u}$ quantifies the importance of time step $u$ in the prediction at time step $t$.
2. **Time-independent feature contribution**: $h_b(x_{u,m}) w_{m,b}^{nbm} w_{k,m}^{out}$ reflects the intrinsic contribution of feature $m$.
3. **Time-dependent feature contribution**: $a_{k,t,u} h_b(x_{u,m}) w_{m,b}^{nbm} w_{k,m}^{out}$ captures the contribution of feature $m$ at a specific time step.

### Loss & Training

- Regression: MSE
- Binary classification: Binary Cross-Entropy
- Multi-class classification: Cross-Entropy
- Optimizer: AdamW
- Early stopping: training halts if validation loss does not improve for 20 epochs
- Hyperparameters: automatically tuned via Optuna

## Key Experimental Results

### Main Results

Single-step prediction performance on 8 public time series datasets:

| Model Type | Model | Energy (R²↑) | Rainfall (R²↑) | AirQuality (R²↑) | Heartbeat (AUROC↑) | LSST (Acc↑) | NATOPS (Acc↑) | Avg. Rank |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Black-box temporal | GRU | 0.435 | 0.089 | 0.701 | 0.694 | 0.629 | 0.931 | 4.500 |
| Black-box temporal | Transformer | 0.263 | 0.098 | 0.711 | 0.690 | 0.679 | 0.967 | 4.125 |
| Transparent tabular | NAM | 0.363 | 0.006 | 0.300 | 0.645 | 0.400 | 0.242 | 9.375 |
| Transparent tabular | NBM | 0.330 | 0.007 | 0.301 | 0.716 | 0.388 | 0.189 | 9.250 |
| Transparent temporal | NATM | 0.304 | 0.038 | 0.548 | 0.724 | 0.452 | 0.878 | 6.833 |
| **Transparent temporal** | **GATSM** | **0.493** | **0.073** | **0.583** | **0.843** | **0.570** | **0.956** | **3.375** |

GATSM achieves the **best average rank** (3.375) across all models, outperforming the Transformer (4.125) while substantially surpassing all prior transparent models.

### Ablation Study

#### Feature Function Selection

| Feature Function | Energy | Rainfall | AirQuality | Heartbeat | LSST | NATOPS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Linear | 0.283 | 0.071 | 0.563 | 0.766 | — | — |
| NAM | — | — | — | — | — | — |
| **NBM** | **0.493** | **0.073** | **0.583** | **0.843** | **0.570** | **0.956** |

The basis strategy of NBM achieves the best performance on 6 of 8 datasets.

#### Temporal Module Design

| Configuration | Energy | Heartbeat | LSST | NATOPS | Description |
|:---|:---:|:---:|:---:|:---:|:---|
| Base | Poor | Poor | Poor | Poor | No temporal module |
| Base + PE | Similar to Base | Similar to Base | Similar to Base | Similar to Base | Positional encoding alone is insufficient |
| Base + MHA | Moderate | Moderate | Moderate | Moderate | Attention is beneficial |
| **Base + PE + MHA** | **Best** | **Best** | **Best** | **Best** | PE and MHA exhibit synergy |

Positional encodings and multi-head attention must be **used together** to effectively capture temporal patterns.

### Key Findings

1. **GATSM surpasses the Transformer in average rank**: This is the first time a transparent model has outperformed a classic black-box temporal model in overall ranking.
2. **Temporal patterns matter**: On datasets with strong temporal structure (Rainfall, AirQuality, LSST, NATOPS), temporal models substantially outperform tabular models.
3. **Variable-length sequence support**: GATSM handles variable-length datasets such as Mortality and Sepsis, whereas NATM cannot.
4. **Clinical data characteristics**: The performance gap between temporal and tabular models is small on Mortality and Sepsis, possibly because current patient state already encodes historical information.
5. **Multi-level interpretability**: GATSM simultaneously provides time step importance, global feature contributions, and local time-dependent feature contributions.

## Highlights & Insights

1. **Theoretical rigor**: A formal definition of temporal GAMs (Definition 3.1) is proposed, and GATSM is rigorously proven to satisfy it.
2. **Elegant weight-sharing design**: Sharing basis functions across time steps achieves an excellent balance between parameter efficiency and expressive power.
3. **Three-level interpretability**: Explanations at different granularities — time step importance, time-independent feature contributions, and time-dependent feature contributions — are highly valuable in practice.
4. **No performance compromise**: GATSM maintains full transparency while matching or exceeding Transformer performance, challenging the conventional trade-off between interpretability and accuracy.

## Limitations & Future Work

1. **First-order additive effects only**: The GAM structure inherently cannot capture feature interactions (e.g., second-order terms as in GA²M), leading to performance gaps on certain datasets (e.g., AirQuality) relative to black-box models.
2. **Primarily single-step prediction**: Although multi-step forecasting is discussed, experiments are conducted mainly on single-step tasks; multi-step performance remains to be verified.
3. **Limited attention capacity**: The expressiveness of multi-head attention is constrained by the number of heads and hidden dimensionality, which may be insufficient for highly complex temporal patterns.
4. **No native missing value handling**: The current approach relies on simple imputation strategies for missing values.
5. **Scalability**: Computational efficiency on very long time series or high-dimensional feature settings is not discussed.

## Related Work & Insights

- **NBM**: The feature network of GATSM directly extends NBM; the basis strategy proves highly effective for time series.
- **NAM / NodeGAM / EBM**: Other transparent tabular models, none of which can handle sequential data.
- **NATM**: The only prior transparent temporal model, but incapable of capturing temporal patterns and restricted to fixed-length inputs.
- **GAT**: The source of the two-layer attention mechanism, which is better suited to the GAM setting than dot-product attention.
- **Insight**: Combining the interpretability advantages of GAMs with modern deep learning components is a direction worthy of further exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First transparent GAM capable of capturing temporal patterns, with a rigorous formal definition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 8 datasets, 15 baselines, complete ablations, though mainly on single-step tasks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear logic, complete theoretical derivations, and well-presented interpretability analysis.
- **Value**: ⭐⭐⭐⭐⭐ — A breakthrough contribution to the important direction of interpretable temporal modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting](hn-mvts_hypernetwork-based_multivariate_time_series_forecasting.md)
- [\[AAAI 2026\] Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [\[AAAI 2026\] SELDON: Supernova Explosions Learned by Deep ODE Networks](seldon_supernova_explosions_learned_by_deep_ode_networks.md)
- [\[AAAI 2026\] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)
- [\[AAAI 2026\] Mask the Redundancy: Evolving Masking Representation Learning for Multivariate Time-Series Clustering](mask_the_redundancy_evolving_masking_representation_learning_for_multivariate_ti.md)

</div>

<!-- RELATED:END -->
