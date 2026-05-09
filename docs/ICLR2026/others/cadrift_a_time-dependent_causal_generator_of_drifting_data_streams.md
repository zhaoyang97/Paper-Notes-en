---
title: >-
  [Paper Note] CaDrift: A Time-dependent Causal Generator of Drifting Data Streams
description: >-
  [ICLR2026][concept drift] This paper proposes CaDrift, a time-dependent synthetic data stream generation framework based on structural causal models (SCMs). It introduces temporal correlation via EWMA smoothing and autoregressive noise, and realizes controllable distributional drift, covariate drift, severe drift, and local drift by modifying causal mapping functions. CaDrift fills the gap left by existing data stream generators that lack both causal structure and temporal dependence.
tags:
  - ICLR2026
  - concept drift
  - structural causal model
  - synthetic data generation
  - data streams
  - time dependence
date: 2026-05-08
content_hash: 4111283611cda123
---

# CaDrift: A Time-dependent Causal Generator of Drifting Data Streams

**Conference**: ICLR2026
**arXiv**: [2602.20329](https://arxiv.org/abs/2602.20329)
**Code**: [https://github.com/eduardovlb/CaDrift](https://github.com/eduardovlb/CaDrift)
**Area**: Other
**Keywords**: concept drift, structural causal model, synthetic data generation, data streams, time dependence

## TL;DR
This paper proposes CaDrift, a time-dependent synthetic data stream generation framework based on structural causal models (SCMs). It introduces temporal correlation via EWMA smoothing and autoregressive noise, and realizes controllable distributional drift, covariate drift, severe drift, and local drift by modifying causal mapping functions. CaDrift fills the gap left by existing data stream generators that lack both causal structure and temporal dependence.

## Background & Motivation

**Background**: Concept drift is a core challenge in data stream mining—data distributions change over time, requiring models to continuously adapt. Evaluating adaptive learning algorithms demands synthetic benchmark data with controllable drift events.

**Limitations of Prior Work**: Existing synthetic generators (e.g., SEA, Sine, RandomRBF) rely on linear or probabilistic functions and produce essentially i.i.d. samples—even when drift events are present, there is no temporal correlation between consecutive samples. This is fundamentally inconsistent with real-world data streams such as IoT sensor data or financial time series.

**Key Challenge**: Real-world data streams exhibit two critical properties: (a) variables are related through causal mechanisms rather than mere correlations, and (b) consecutive samples exhibit serial correlation. Existing generators satisfy neither property.

**Goal**: To design a synthetic data stream generator that simultaneously incorporates causal structure and temporal dependence while supporting multiple controllable drift types.

**Key Insight**: The causal graph structure of SCMs is combined with EWMA smoothing and autoregressive noise, naturally introducing temporal dependence along the causal propagation chain; different types of drift are produced by modifying the SCM's mapping functions.

**Core Idea**: Temporal dynamics (EWMA + AR noise) are superimposed on the SCM causal graph, and controllable drift is achieved by modifying mapping functions—realizing, for the first time, a unified framework combining causal structure, temporal dependence, and controllable drift in data stream generation.

## Method

### Overall Architecture
CaDrift is built upon a structural causal model defined by a directed acyclic graph (DAG). Its inputs are the DAG structure (nodes = features + target, edges = causal relationships) and various hyperparameters ($\alpha$ for smoothing, $\rho$ for autoregression, drift type, and drift timing). Its output is a temporally dependent synthetic data stream in which consecutive samples exhibit serial correlation and controllable concept drift occurs at specified time points.

The generation pipeline proceeds as follows: root nodes are sampled from EWMA-smoothed distributions → internal nodes are computed from parent nodes via mapping functions (small neural networks) → autoregressive noise is added to all nodes → target nodes compute labels through the causal chain → mapping functions or distribution parameters are modified at drift time points.

### Key Designs

1. **Time-dependent SCM**:

    - **Function**: Enables an otherwise i.i.d. SCM to generate data streams with temporal correlation.
    - **Mechanism**: The standard SCM defines $E := f_E(C) + N_E$. CaDrift introduces two temporal components: (a) root nodes apply EWMA smoothing $Z_t = (1-\alpha)Z_{t-1} + \alpha X_t$, endowing root node values with historical memory; (b) the noise of all non-root nodes is replaced by an autoregressive form $N_E^{(t)} = \rho N_E^{(t-1)} + \epsilon^{(t)}$, where $\rho \in [0,1]$ controls the degree of temporal smoothness.
    - **Design Motivation**: EWMA ensures smooth transitions in root node values rather than abrupt changes, while autoregressive noise introduces continuity in the stochastic fluctuations of each node. Together, temporal correlation propagates naturally along the causal chain to all downstream nodes and the target.

2. **Causal Mapping Function Initialization**:

    - **Function**: Defines the causal relationship functions between nodes.
    - **Mechanism**: Rather than randomly initializing MLPs or decision trees as in TabPFN, small neural networks are first fitted to target values using the parent node distribution, ensuring that the mapping function captures causal relationships within the support of the parent node distribution.
    - **Design Motivation**: This avoids the risk that randomly initialized tree models may place split points outside the parent node distribution, leading to negligibly small causal variations or degenerate single-class outputs.

3. **Intervention Mechanism**:

    - **Function**: Simulates real-world environmental perturbations (e.g., device failures, environmental shocks).
    - **Mechanism**: All incoming edges of the intervened node are severed, and its value is forcibly assigned from a normal or uniform distribution, bypassing the causal mapping function. The effect is described using do-calculus: $P(y | \text{do}(x_3 \sim \mathcal{N}(\mu, \sigma^2)))$.
    - **Design Motivation**: Sporadic anomalous events in real data streams (e.g., sensor failures) do not follow normal causal relationships. The intervention mechanism naturally simulates such scenarios.

4. **Controllable Drift Generation**:

    - **Function**: Introduces different types of concept drift at specified time points.
    - **Mechanism**: Four drift types are implemented by modifying different SCM components:
        - **Distributional Drift**: Modifies the inter-node mapping function $f_E(C)$ or the target mapping $f_y(C)$, altering $P(y|X)$.
        - **Covariate Drift**: Modifies the distribution parameters of root nodes, changing $P(X)$ without altering causal relationships.
        - **Severe Drift**: Reverses the output class labels of the target mapping.
        - **Local Drift**: Modifies the distribution parameters of a single feature only.
    - Drift speed is controlled by parameter $\Delta$: $\Delta=1$ corresponds to abrupt drift, and $\Delta>1$ to gradual/incremental drift; recurring drift (reverting to an old concept) is also supported.

### Loss & Training
CaDrift is a generative framework rather than a trained model. Mapping functions are fitted once during initialization and subsequently used to propagate the data stream through the causal graph. Drift events trigger mapping function replacement at pre-specified time points.

## Key Experimental Results

### Main Results
Seven data stream classifiers are evaluated on 8 CaDrift-generated datasets and 3 traditional benchmarks (SEA, Sine, RandomRBF):

| Method | Avg. Accuracy on 8 CaDrift Datasets | Avg. Accuracy on 3 Traditional Datasets | Overall Avg. Rank |
|--------|-------------------------------------|------------------------------------------|-------------------|
| ARF | ~73% | ~85.5% | **2.2** |
| TabPFN$^{\text{Stream}}$ | ~69% | ~83.0% | 3.0 |
| IncA-DES | ~73% | ~85.1% | 3.4 |
| LevBag | ~73% | ~79.5% | 3.0 |
| LAST | ~71% | ~77.7% | 4.9 |
| OAUE | ~63% | ~74.6% | 5.3 |
| HT | ~60% | ~65.3% | 6.2 |

### Ablation Study (Stationarity Testing — Ljung-Box Test)

| Configuration | Proportion of Features Rejecting $H_0$ | Notes |
|---------------|----------------------------------------|-------|
| i.i.d. (no temporal mechanism) | 0/6 | All features and target fail to reject; no serial correlation |
| EWMA only ($\alpha$=0.05) | 4/6 | Most features exhibit serial correlation; some downstream nodes are not significant |
| AR only ($\rho$=0.1) | 6/6 | All features and target show significant serial correlation |
| EWMA + AR | **6/6** | All nodes pass; serial correlation propagates along the causal chain |

### Key Findings
- **Covariate drift does not affect classification performance**: In datasets 1 and 2, covariate drift at sample 500 does not cause performance degradation, as expected (causal relationships remain unchanged).
- **Adaptive methods (ARF, LevBag) are more drift-resistant**: They recover faster than non-adaptive methods such as HT and TabPFN.
- **TabPFN's context window issue**: When concept duration is shorter than the context window, TabPFN mixes data from two concepts, causing severe post-drift performance degradation—a classic manifestation of the stability-plasticity dilemma.
- **Autoregressive noise is the key driver of temporal dependence**: Even a small $\rho$ (0.1) produces significant serial correlation across all nodes.
- **CaDrift generates more challenging datasets than traditional generators**: Classifiers frequently achieve near-100% accuracy on SEA and Sine, whereas CaDrift datasets prove substantially harder.

## Highlights & Insights
- **Unified causal + temporal + drift framework**: CaDrift is the first to introduce temporal dependence and controllable drift into an SCM—a natural and elegant combination, since the causal graph inherently defines variable relationships, modifying mapping functions naturally induces drift, and EWMA/AR naturally introduce temporal structure.
- **Intervention mechanism for perturbation simulation**: Employing the do-calculus concept from causal inference to simulate environmental perturbations is conceptually clean—severing incoming edges and forcibly assigning values perfectly mirrors real-world scenarios such as sensor failures.
- **Clever ablation design**: The Ljung-Box statistical test is used to quantitatively verify temporal dependence, rather than relying solely on indirect metrics such as classification accuracy.
- **Potential for foundation model pretraining**: Data generated by CaDrift can serve as a pretraining data prior for temporal tabular foundation models, representing a valuable downstream application direction.

## Limitations & Future Work
- **Restricted to tabular data**: The current framework is limited to tabular data streams and is not applicable to drift simulation in unstructured data streams such as images or text.
- **Limited variety of mapping functions**: Small neural networks are the primary causal mapping mechanism; extending to a broader family of functions (e.g., piecewise linear, kernel-based) would increase diversity.
- **Lack of quantitative comparison with real data streams**: Although MMD analysis is conducted, there is no systematic assessment of how closely CaDrift-generated streams match real data streams in terms of statistical properties.
- **No drift detection evaluation**: Evaluating drift detection algorithms is a key intended use of such generators, yet no drift detection methods are tested in the experiments.
- **Computational cost not discussed**: Time and memory overhead for generating large-scale data streams (DAGs with 100–200 nodes) is not reported.

## Related Work & Insights
- **vs. TabPFN's SCM generator**: TabPFN also uses SCMs to generate synthetic data, but the samples are i.i.d. and drift is uncontrolled. CaDrift's advantages—temporal dependence and controllable drift—position it as a temporal extension of TabPFN's generation paradigm.
- **vs. RealDriftGenerator**: RealDriftGenerator requires a source dataset and introduces drift via Clip Swap. CaDrift is fully synthetic, requires no source data, and supports a richer variety of drift types.
- **vs. OWDSG**: OWDSG introduces drift by modifying clusters, but the underlying generator remains Madelon with no causal structure. CaDrift's causal chain renders drift more natural and interpretable.

## Rating
- Novelty: ⭐⭐⭐⭐ First to combine causal models, temporal dependence, and controllable drift; the conceptual innovation is clear and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Eight self-generated datasets and three traditional benchmarks; statistical tests used in ablation; drift detection evaluation is absent.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, comparison tables are comprehensive, and mathematical definitions are rigorous.
- Value: ⭐⭐⭐⭐ Offers practical tool value for the data stream mining community; code is publicly available.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Addressing Divergent Representations from Causal Interventions on Neural Networks](addressing_divergent_representations_causal.md)
- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](../../AAAI2026/others/how_to_marginalize_in_causal_structure_learning.md)
- [\[CVPR 2026\] Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](../../CVPR2026/others/mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)
- [\[ICLR 2026\] CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning](chlu_the_causal_hamiltonian_learning_unit_as_a_symplectic_primitive_for_deep_lea.md)
- [\[ICLR 2026\] When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency](when_to_retrain_after_drift_a_data-only_test_of_post-drift_data_size_sufficiency.md)

<!-- RELATED:END -->
