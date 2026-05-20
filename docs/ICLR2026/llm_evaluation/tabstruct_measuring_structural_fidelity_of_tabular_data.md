---
title: >-
  [Paper Note] TabStruct: Measuring Structural Fidelity of Tabular Data
description: >-
  [ICLR 2026][LLM Evaluation][Tabular data generation] This paper proposes the TabStruct evaluation framework and a global utility metric that measures the structural fidelity of tabular data generators with respect to cau…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "Tabular data generation"
  - "structural fidelity"
  - "causal structure"
  - "global utility"
  - "conditional independence"
date: 2026-05-08
content_hash: 8c73b33911cc4ed1
---

# TabStruct: Measuring Structural Fidelity of Tabular Data

**Conference**: ICLR 2026
**arXiv**: [2509.11950](https://arxiv.org/abs/2509.11950)  
**Code**: [https://github.com/SilenceX12138/TabStruct](https://github.com/SilenceX12138/TabStruct)  
**Area**: Data Generation / Tabular Data / Causal Structure
**Keywords**: Tabular data generation, structural fidelity, causal structure, global utility, conditional independence

## TL;DR

This paper proposes the TabStruct evaluation framework and a global utility metric that measures the structural fidelity of tabular data generators with respect to causal structure, without requiring ground-truth causal graphs. A systematic comparison of 13 generators across 29 datasets reveals that diffusion models significantly outperform other methods in preserving global structure.

## Background & Motivation

**Background**: Tabular data generation underlies tasks such as training data augmentation and missing value imputation. Existing evaluations focus on three dimensions: density estimation (distributional similarity), ML utility (downstream predictive performance), and privacy protection (distance between synthetic and real data).

**Limitations of Prior Work**: These three dimensions are adapted from homogeneous modalities (text/images) and do not account for the heterogeneous characteristics of tabular data. A representative failure case is SMOTE, which scores well on density estimation and ML utility yet produces data that severely violates inter-variable causal relationships—for instance, breaking conditional independencies implied by physical laws.

**Key Challenge**: The fundamental prior for tabular data is the structural causal model (SCM), in which variables exhibit causal dependencies. Conventional metrics evaluate only marginal distributions or predictive performance for a single target, failing to capture global causal interactions among features. The only existing benchmark addressing structural fidelity, CauTabBench, is restricted to toy SCM datasets, because quantifying structural fidelity requires ground-truth causal graphs that are virtually unavailable for real-world datasets.

**Goal**
- How can structural fidelity be evaluated without ground-truth causal graphs?
- How do existing generators perform in terms of structural fidelity?
- What is the relationship between structural fidelity and conventional evaluation dimensions?

**Key Insight**: The authors observe that if a generator truly learns the causal structure of the data, a model trained on synthetic data should be able to predict each variable from all others with performance close to that achieved on real data. This "all-variable predictability" property is closely related to the Markov blanket concept in SCMs.

**Core Idea**: By rotating each variable as the prediction target and aggregating the ratio of predictive performance across all variables, global utility is defined as a proxy for the global structural fidelity of tabular data generators—without requiring a causal graph.

## Method

### Overall Architecture

TabStruct is a unified evaluation framework that takes a reference dataset $\mathcal{D}_{\text{ref}}$ and a synthetic dataset $\mathcal{D}_{\text{syn}}$ as input and outputs scores along four dimensions: density estimation, privacy protection, ML utility, and structural fidelity. Structural fidelity is the core dimension newly introduced by this work.

The evaluation pipeline operates under two scenarios:
- **SCM datasets**: Conditional independence (CI) tests are used to directly quantify structural fidelity.
- **Real-world datasets without SCMs**: Global utility serves as an indirect proxy for structural fidelity.

### Key Designs

1. **Conditional Independence (CI) Score — Structural Fidelity with Known Causal Graphs**

    - **Function**: On datasets with known ground-truth SCMs, structural fidelity is quantified by comparing the consistency of CI statements between real and synthetic data.
    - **Mechanism**: All CI statements $\mathcal{C}_{\text{global}}$ are enumerated from the CPDAG of the ground-truth SCM, including both d-separation and d-connection statements. For each CI statement, a statistical test ($\alpha=0.01$) is performed on the synthetic data, and the pass rate is computed as $CI(\mathcal{C}, \mathcal{D}) = \frac{1}{|\mathcal{C}|}\sum \mathbb{1}[\hat{\mathcal{I}}_\alpha = 1]$.
    - **Design Motivation**: Evaluation is conducted at the CPDAG level rather than the DAG level, because existing causal discovery methods become unreliable when the number of features exceeds 10. Skeleton-level evaluation is also avoided, as it discards directional information.
    - **Local vs. Global**: Local CI considers only CI statements involving the prediction target $y$; global CI considers all variable pairs.

2. **Global Utility — A Proxy for Structural Fidelity Without Causal Graphs**

    - **Function**: Measures the degree of global structural preservation on datasets without ground-truth SCMs.
    - **Mechanism**: Each variable $x_j$ is rotated as the prediction target, with all remaining variables used as predictors. The per-variable utility is defined as the ratio of predictive performance relative to the reference data (balanced accuracy for classification; inverse RMSE for regression). Global utility is the average across all variables: $\text{Global Utility}(\mathcal{D}) = \frac{1}{D+1}\sum_{j=1}^{D+1}\text{Utility}_j(\mathcal{D})$.
    - **Design Motivation**: This design addresses two issues: (1) it avoids the target-specific bias of local utility (which only predicts $y$); (2) aggregating normalized performance ratios enables comparability across tasks of different types. AutoGluon with an ensemble of 9 predictors is employed to reduce single-model bias.
    - **Theoretical Basis**: A high-fidelity generator should preserve the conditional distribution $p(x_j | \mathcal{X} \setminus \{x_j\})$ for every variable, which is consistent with the Markov blanket concept.

3. **Local Utility vs. Global Utility**

    - **Function**: The comparison highlights the limitations of local utility as equivalent to ML efficacy.
    - **Mechanism**: Local utility focuses solely on predicting target $y$. Experiments show that local utility is strongly correlated with local CI ($r_s=0.78$) but nearly uncorrelated with global CI ($r_s=0.14$), whereas global utility is strongly correlated with global CI ($r_s=0.84$).
    - **Design Motivation**: This demonstrates that conventional ML efficacy reflects only local structure and is insufficient for comprehensive generator evaluation.

### Integration of Evaluation Dimensions

The framework jointly considers four dimensions: density estimation (Shape/Trend), privacy protection ($\alpha$-precision/$\beta$-recall/DCR/$\delta$-Presence), ML utility (local utility), and structural fidelity (CI score/global utility), providing a comprehensive assessment of generators.

## Key Experimental Results

### Main Results — Structural Fidelity of 13 Generators on SCM Datasets

| Generator | Global CI ↑ | Global Utility ↑ | Local Utility ↑ | Shape ↑ |
|--------|------------|-------------------|-----------------|---------|
| $\mathcal{D}_\text{ref}$ | 1.00 | 0.99 | 0.99 | 1.00 |
| TabSyn | **0.70** | 0.76 | 0.76 | 0.50 |
| TabDDPM | **0.69** | **0.80** | 0.29 | 0.62 |
| TabDiff | 0.57 | 0.75 | 0.80 | 0.69 |
| SMOTE | 0.30 | 0.39 | **0.92** | 0.82 |
| CTGAN | 0.08 | 0.26 | 0.80 | 0.46 |
| GReaT | 0.16 | 0.25 | 0.27 | 0.62 |
| NRGBoost | 0.11 | 0.16 | 0.75 | 0.65 |

**Key Findings**: SMOTE achieves the highest local utility (0.92) but a global CI of only 0.30, demonstrating that conventional ML utility metrics can be misleading. Diffusion models (TabDDPM/TabSyn/TabDiff) consistently achieve the best global structural fidelity.

### Global Utility Rankings on Real-World Datasets

| Generator | Global Utility ↑ | Local Utility ↑ |
|--------|-------------------|-----------------|
| $\mathcal{D}_\text{ref}$ | 0.99 | 0.96 |
| TabSyn | **0.73** | 0.76 |
| TabDiff | **0.73** | 0.78 |
| TabDDPM | **0.72** | 0.27 |
| ARF | 0.56 | 0.54 |
| TVAE | 0.53 | 0.70 |
| BN | 0.44 | 0.38 |
| SMOTE | 0.41 | **0.91** |
| CTGAN | 0.13 | 0.70 |
| GReaT | 0.20 | 0.23 |

On real-world datasets, the top-3 generators by global utility remain the diffusion models (TabSyn/TabDiff/TabDDPM), consistent with results on SCM datasets, validating the generalizability of global utility.

### Correlation Analysis

| Metric Pair | Spearman $r_s$ | p-value |
|--------|----------------|------|
| Global Utility ↔ Global CI | **0.84** | <0.001 |
| Local Utility ↔ Local CI | 0.78 | <0.001 |
| Local Utility ↔ Global CI | 0.14 | <0.001 |

The strong correlation between global utility and global CI (0.84) is the central empirical result, validating the effectiveness of global utility as a proxy metric in the absence of ground-truth SCMs.

## Highlights & Insights

- **Filling an Evaluation Gap**: This work is the first to systematically incorporate structural fidelity into the evaluation framework for tabular generators, and the proposed global utility requires no ground-truth causal graphs.
- **Large-Scale Benchmark**: 13 generators × 29 datasets × over 150,000 evaluations, far exceeding the coverage of existing benchmarks.
- **Explaining Why Diffusion Models Excel**: Diffusion models independently add noise to each feature and reconstruct all features jointly during denoising, naturally learning permutation-invariant conditional distributions that align with the structural prior of tabular data.
- **Exposing the "Illusion" of SMOTE**: Experiments clearly show that SMOTE performs well on conventional metrics while severely violating causal structure, exposing systematic evaluation bias.

## Limitations & Future Work

- The strong correlation between global utility and global CI is an empirical finding without theoretical proof.
- Evaluation at the CPDAG level is coarser than at the full DAG level and may miss certain directional causal relationships.
- The framework relies on AutoGluon ensemble predictors, whose computational cost grows with the number of features (though the Tiny-default variant is efficient at 0.64s/1000 samples).
- SCM datasets use expert-validated causal graphs, but such datasets are scarce (only 6 available), limiting generalizability.

## Related Work & Insights

- **Tabular Generation Benchmarks**: Synthcity (Qian et al., 2024) and SynMeter (Du & Li, 2024) cover density, privacy, and ML utility but not structural fidelity; CauTabBench (Tu et al., 2024) evaluates structural fidelity but is limited to toy SCMs.
- **Tabular Generators**: The comparison spans SMOTE, BN, TVAE, CTGAN, NFlow, ARF, diffusion models (TabDDPM/TabSyn/TabDiff/TabEBM), LLM-based GReaT, and tree-based NRGBoost.
- **Causal Discovery**: DAG learning methods become unreliable for more than 10 features (Zanga et al., 2022), motivating the CPDAG-level evaluation adopted in this work.
- **Tabular Foundation Models**: Hollmann et al. (2025) empirically demonstrate that SCMs are effective structural priors for tabular data.

## Rating

| Dimension | Score | Remarks |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | First global structural fidelity metric that requires no causal graph |
| Technical Depth | ⭐⭐⭐⭐ | Clear theoretical analysis; well-motivated CI framework and global utility design |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | 13 models × 29 datasets, 150,000+ evaluations; highly comprehensive |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured; motivating examples are intuitive |
| Value | ⭐⭐⭐⭐ | Open-source framework directly applicable to evaluating new generators |
| Overall | ⭐⭐⭐⭐ | An important contribution to tabular data generation evaluation; global utility has the potential to become a standard metric |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)
- [\[ICLR 2026\] Measuring Uncertainty Calibration](measuring_uncertainty_calibration.md)
- [\[ICLR 2026\] Human-LLM Collaborative Feature Engineering for Tabular Learning](human-llm_collaborative_feature_engineering_for_tabular_data.md)
- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](../../NeurIPS2025/llm_evaluation/climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](revisiting_the_past_data_unlearning_with_model_state_history.md)

</div>

<!-- RELATED:END -->
