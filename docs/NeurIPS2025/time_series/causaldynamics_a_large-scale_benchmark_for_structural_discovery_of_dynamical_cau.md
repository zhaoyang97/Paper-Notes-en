---
title: >-
  [Paper Note] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models
description: >-
  [NeurIPS 2025][Time Series][causal discovery] This paper introduces CausalDynamics — the largest benchmark to date for causal discovery in dynamical systems (14,000+ graphs…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "causal discovery"
  - "dynamical systems"
  - "benchmark"
  - "ODE/SDE"
  - "causal graph"
date: 2026-05-08
content_hash: 686fadfe77ed0aa5
---

# CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.16620](https://arxiv.org/abs/2505.16620)  
**Code**: [kausable/CausalDynamics](https://github.com/kausable/CausalDynamics)  
**Area**: Time Series
**Keywords**: causal discovery, dynamical systems, benchmark, time series, ODE/SDE, causal graph

## TL;DR

This paper introduces CausalDynamics — the largest benchmark to date for causal discovery in dynamical systems (14,000+ graphs, 50M+ samples) — encompassing a three-tier progressively complex hierarchy ranging from 3-dimensional chaotic ODE/SDE systems and hierarchically coupled systems to realistic climate models. The benchmark comprehensively evaluates 10 state-of-the-art causal discovery algorithms, revealing the shortcomings of current deep learning methods on high-dimensional nonlinear dynamical systems.

## Background & Motivation

- **Broad demand for causal discovery**: In nonlinear dynamical systems arising in climate science, biology, and finance, direct interventions are often infeasible, necessitating causal inference from observational time series data.
- **Inadequacy of existing benchmarks**: Most causal discovery benchmarks are based on static causal graphs or autoregressive models, lacking characterization of continuous state-space evolution, complex feedback loops, stochasticity, and regime shifts.
- **Absence of ground truth in real data**: Existing real-world datasets (e.g., CausalRivers, MoCap) lack fully interpretable causal ground truth, making it impossible to disentangle algorithmic limitations from data characteristics.
- **Limitations of synthetic data**: Prior synthetic benchmarks typically contain only a small number of graphs (e.g., Netsim, DREAM3&4) and predominantly feature weakly nonlinear systems, which are insufficient for comprehensive evaluation on complex dynamical systems.
- **Unsystematic treatment of key challenges**: Practical issues including noise, unobserved confounders, time-lag, and varsortability require a unified framework for systematic evaluation.
- **Paradigm-shift analogy**: Just as CASP13 catalyzed AlphaFold and ImageNet enabled AlexNet, the authors argue that the causal discovery community similarly requires a large-scale standardized benchmark to drive methodological innovation.

## Method

### Overall Architecture: Three-Tier Progressive Complexity Hierarchy

CausalDynamics adopts a hierarchical design that increases in complexity across tiers:

| Tier | Data Source | Challenges | # Graphs |
|------|-------------|------------|----------|
| Tier 1 (Simple) | 59 three-dimensional chaotic ODE/SDE systems | Confounders, noise | 585 |
| Tier 2 (Coupled) | Hierarchically coupled ODE/SDE ($N=3,5,10$) | Confounders, time-lag, normalization, forcing | 14,096 |
| Tier 3 (Climate) | MAOOAM + ENSO climate models | High dimensionality | 12 |

### Key Design 1: Structural Dynamical Causal Model (SDCM)

The classical structural causal model (SCM) is combined with differential equations to define the SDCM:

$$\frac{d}{dt}x_{k,t} := f^k(\boldsymbol{x}_{\text{PA}_k, t}, \delta), \quad x_{k,0} = x_k(0)$$

where $\delta$ controls the noise magnitude: $\delta=0$ corresponds to an ODE and $\delta>0$ to an SDE. The causal graph of each system is represented by an adjacency matrix $\mathcal{A}$.

### Key Design 2: GNR-Based Hierarchical Coupled Graph Generation

Tier 2 employs the Growing Network with Redirection (GNR) model to generate scale-free DAGs:

- **Nodes (causal units)**: Each node represents a $d$-dimensional time series; root nodes can be driven by a dynamical system (Lorenz/Rössler), a periodic forcing ($A\sin(\omega t + \phi)$), or a linear driver.
- **Edges (coupling functions)**: Implemented via MLPs, with activation functions sampled from {identity, sin, sigmoid, tanh, ReLU} and weights sparsified by a dropout probability $p_{\text{zero}}$.
- **Information aggregation**: The value of a non-root node is obtained by summing MLP-transformed signals from its parent nodes: $x_{v_k}(t) = \sum_{k \in \text{pa}_i} f_{(k,i)}(x_k(t))$.

### Key Design 3: Systematic Injection of Causal Challenges

- **Confounders**: Two adjacency matrices are sampled; the off-diagonal entries of the second are rotated by 90° and merged with the first, ensuring the resulting confounded graph remains a scale-free DAG.
- **Time-lag**: With probability $p_t$, a fixed delay $\tau$ is introduced on an edge: $x_{v_k}(t) = f_{(k,i)}(x_k(t-\tau))$, forming a temporally cyclic graph.
- **Normalization**: Node values are standardized along the temporal dimension to eliminate varsortability artifacts.

### Key Design 4: Tier 3 Realistic Climate Models

- **ENSO model (XRO)**: Integrates the Hasselmann stochastic framework with recharge oscillator dynamics, initialized from observed SSTs, with adjustable cross-basin coupling strength.
- **MAOOAM model (qgs)**: A quasi-geostrophic two-layer model solving barotropic/baroclinic interactions to simulate high-dimensional atmosphere–ocean coupled dynamics.

## Loss & Training

As a benchmark paper, no new model training is involved. Evaluation metrics include:

- **AUROC** (higher is better): Measures the classification capability for graph reconstruction.
- **AUPRC** (higher is better): More sensitive to sparse graphs and penalizes false positives.
- **SHD** (lower is better): Structural Hamming Distance, measuring the edit distance between the predicted and ground-truth graphs.

## Key Experimental Results

### Table 1: AUROC / AUPRC Results (Selected, 10 Algorithms)

| Experiment | PCMCI+ | F-PCMCI | VARLiNGAM | DYNOTEARS | NGC | TSCI |
|------------|--------|---------|-----------|-----------|-----|------|
| Simple-Default | **.52/.71** | .51/.70 | .50/.69 | .43/.67 | .50/.69 | .46/.68 |
| Coupled-Default | .67/.25 | **.67/.27** | .60/.19 | .59/.21 | .50/.15 | .60/.23 |
| Coupled-Confounder | **.58/.20** | .55/.19 | .51/.17 | .49/.17 | .50/.16 | .51/.18 |
| Climate-MAOOAM | **.69/.88** | .50/.81 | .50/.81 | .64/.86 | .50/.81 | .58/.84 |
| Climate-ENSO | .57/.70 | **.57/.70** | .56/.69 | .55/.69 | .50/.67 | .50/.67 |

### Table 2: SHD Results (Selected, Lower is Better)

| Experiment | PCMCI+ | F-PCMCI | NGC | CUTS+ | RCD |
|------------|--------|---------|-----|-------|-----|
| Simple-Default | 41.04 | 35.30 | **28.91** | 48.11 | 61.85 |
| Coupled-Default | 224.80 | 192.90 | 840.95 | **152.00** | 157.05 |
| Coupled-Time-lag | 327.72 | 350.61 | 793.67 | 247.22 | **201.11** |
| Climate-MAOOAM | 80.00 | 130.00 | **31.00** | 130.00 | 130.00 |
| Climate-ENSO | 529.36 | 530.27 | **337.09** | 608.73 | 665.36 |

**Key Findings**:

- Non-DL methods (PCMCI+, F-PCMCI) outperform DL methods in most settings, particularly on high-dimensional coupled systems.
- The topology-based method TSCI outperforms pure DL approaches in the presence of confounders and in high-dimensional settings.
- All methods perform poorly on coupled dynamical systems: spurious autocorrelation inference and overly dense adjacency matrix predictions under non-stationary dynamics are widespread.
- Normalization (to eliminate varsortability) yields limited improvements, especially for methods relying on topological sorting such as VARLiNGAM.

## Highlights & Insights

- **Unprecedented scale**: 14,000+ graphs and 50M+ samples, far exceeding existing benchmarks of the same type.
- **Three-tier progressive design**: Systematically covers varying complexity from simple chaotic systems to realistic climate models.
- **Extensible framework**: A plug-and-play workflow that allows users to customize coupling structures, noise levels, time-lag parameters, and other settings to generate new data.
- **Comprehensive evaluation**: Simultaneously evaluates 10 causal discovery algorithms spanning 5 major categories (Granger, constraint-based, noise-based, score-based, and topology-based).
- **Important insight revealed**: DL methods perform worse than simple non-DL methods in high-dimensional nonlinear settings — precisely the regime they claim to handle well — pointing to clear directions for future research.

## Limitations & Future Work

- The current benchmark only handles fixed parameters (time-lag, noise level) and 3-dimensional base systems, without covering higher-dimensional or variable-parameter scenarios.
- Tier 3 contains only 12 climate graphs, limiting statistical significance.
- A gap remains between generated data and real-world observations; although the framework is extensible, it has not yet been connected to real climate reanalysis data.
- The SHD metric does not account for graph size or edge density, making cross-tier comparisons difficult.
- Recent Transformer- or diffusion-model-based causal discovery methods have not been included as baselines.

## Related Work & Insights

| Aspect | CausalDynamics | CausalTime | CauseMe | CausalBench |
|--------|---------------|------------|---------|-------------|
| Data Type | ODE/SDE + climate models | DL-generated | SAVAR + climate | Single-cell RNA |
| # Graphs | 14,000+ | Few | Few | Thousands (static graphs) |
| Dynamical Systems | ✓ (chaotic, stochastic) | Partial | Partial | ✗ |
| Ground Truth | ✓ (analytically derived) | ✗ (lacks reliable validation) | Partial | ✓ |
| Extensibility | ✓ (plug-and-play) | Limited | Limited | Limited |

## Rating

- Novelty: ⭐⭐⭐⭐ — The first large-scale benchmark for causal discovery in dynamical systems; the three-tier design and GNR-based coupled graph generation are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation of 10 SOTA methods, though the number of Tier 3 graphs is relatively small.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rigorous theoretical derivations and well-presented experiments.
- Value: ⭐⭐⭐⭐ — Has the potential to become a standard benchmark in the causal discovery community and to drive methodological advances.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](../../ICLR2026/time_series/omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[NeurIPS 2025\] Causal Masking on Spatial Data: An Information-Theoretic Case for Learning Spatial Datasets with Unimodal Language Models](causal_masking_on_spatial_data_an_information-theoretic_case_for_learning_spatia.md)
- [\[ICCV 2025\] VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models](../../ICCV2025/time_series/vlrmbench_a_comprehensive_and_challenging_benchmark_for_vision-language_reward_m.md)
- [\[ICML 2026\] FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models](../../ICML2026/time_series/factorynet_a_large-scale_dataset_toward_industrial_time-series_foundation_models.md)

</div>

<!-- RELATED:END -->
