---
title: >-
  [Paper Note] TabStruct: Measuring Structural Fidelity of Tabular Data
description: >-
  [ICLR 2026 Oral][Tabular data generation] This paper proposes the TabStruct evaluation framework and the global utility metric to measure the structural fidelity of tabular data generators without requiring ground-truth causal graphs. By systematically comparing 13 generators across 29 datasets, the study reveals that diffusion models significantly outperform other methods in maintaining global structural integrity.
tags:
  - "ICLR 2026 Oral"
  - "Tabular data generation"
  - "structural fidelity"
  - "causal structure"
  - "global utility"
  - "conditional independence"
date: 2026-05-08
content_hash: d176191f68901209
---

# TabStruct: Measuring Structural Fidelity of Tabular Data

**Conference**: ICLR 2026 Oral  
**arXiv**: [2509.11950](https://arxiv.org/abs/2509.11950)  
**Code**: [https://github.com/SilenceX12138/TabStruct](https://github.com/SilenceX12138/TabStruct)  
**Area**: Data Generation / Tabular Data / Causal Structure  
**Keywords**: Tabular data generation, structural fidelity, causal structure, global utility, conditional independence

## TL;DR

This paper proposes the TabStruct evaluation framework and the global utility metric to measure the structural fidelity of tabular data generators without requiring ground-truth causal graphs. By systematically comparing 13 generators across 29 datasets, the study reveals that diffusion models significantly outperform other methods in maintaining global structural integrity.

## Background & Motivation

**Background**: Tabular data generation is basic to tasks such as training augmentation and missing value imputation. Existing evaluations primarily focus on three dimensions: density estimation (distribution proximity), ML utility (downstream predictive performance), and privacy protection (distance between synthetic and real data).

**Limitations of Prior Work**: These three dimensions originate from homogeneous modalities (text/image) and lack specialized evaluation for the heterogeneous characteristics of tabular data. A typical problem is that SMOTE performs well in density estimation and ML utility, but the generated data severely violates causal relationships—for instance, breaking conditional independencies implied by physical laws.

**Key Challenge**: The core prior of tabular data is the Structural Causal Model (SCM), where causal dependencies exist between variables. Traditional metrics only evaluate marginal distributions or specific target predictions, failing to capture global causal interactions between features. Furthermore, CauTabBench, the only existing work considering structural fidelity, is limited to toy SCM datasets because quantifying structural fidelity requires ground-truth causal graphs, which are almost impossible to obtain for real-world datasets.

**Goal**
   - How to evaluate structural fidelity without ground-truth causal graphs?
   - How do existing generators perform regarding structural fidelity?
   - What is the relationship between structural fidelity and traditional evaluation dimensions?

**Key Insight**: The authors observe that if a generator truly learns the causal structure of the data, then a model trained on synthetic data should be able to predict each variable using others as features, with performance close to that of real data. This "all-variable predictability" property is closely related to the concept of the Markov blanket in SCMs.

**Core Idea**: By treating each variable as a prediction target in turn and aggregating the predictive performance ratios as "global utility," the global structural fidelity of tabular data can be evaluated without the need for causal graphs.

## Method

### Overall Architecture

TabStruct unifies the evaluation of tabular generators into a single framework. Given a reference dataset $\mathcal{D}_{\text{ref}}$ and a synthetic dataset $\mathcal{D}_{\text{syn}}$ produced by a generator, it scores along four dimensions: density estimation (Shape/Trend), privacy protection ($\alpha$-precision/$\beta$-recall/DCR/$\delta$-Presence), ML utility (local utility), and the core dimension introduced in this paper: structural fidelity. The first three dimensions follow existing metrics, while the framework's innovation lies entirely in the structural fidelity component.

The calculation of structural fidelity follows two paths depending on the availability of ground-truth causal graphs. For datasets with available SCMs, it uses Conditional Independence (CI) tests to verify if the synthetic data preserves causal relationships. For real-world datasets without causal graphs, it uses a proxy metric called "global utility" to indirectly characterize global structural preservation.

### Key Designs

**1. Conditional Independence (CI) Scoring: Direct Quantification with Causal Graphs**

When a dataset includes expert-verified ground-truth SCMs, "structural preservation" is translated into a set of testable propositions. Specifically, all CI statements $\mathcal{C}_{\text{global}}$ are enumerated from the CPDAG of the true SCM, including both d-separation (variables that should be independent) and d-connection (variables that should be dependent). Statistical tests are then performed on the synthetic data (significance level $\alpha=0.01$), with the success rate calculated as $CI(\mathcal{C}, \mathcal{D}) = \frac{1}{|\mathcal{C}|}\sum \mathbb{1}[\hat{\mathcal{I}}_\alpha = 1]$.

The evaluation is conducted at the CPDAG level rather than the full DAG level because existing causal discovery methods become unreliable when the number of features exceeds 10; forcing a full DAG would introduce noise. Conversely, skeleton-level analysis would lose edge directionality. CPDAG represents a balance between precision and feasibility. This scoring distinguishes between local and global: local CI only considers CI statements related to the prediction target $y$, while global CI covers all variable pairs.

**2. Global Utility: A Proxy Metric for Structural Fidelity without Causal Graphs**

Since causal graphs are rarely available for real-world datasets, an indirect metric independent of SCMs is required. The authors propose that if a generator learns the causal structure, then training a model on synthetic data to predict each variable $x_j$ using the remaining variables should yield performance similar to that on real data. Single-variable utility is defined as the ratio of this performance to the reference data (balanced accuracy for classification, normalized inverse RMSE for regression). Averaging across all variables yields $\text{Global Utility}(\mathcal{D}) = \frac{1}{D+1}\sum_{j=1}^{D+1}\text{Utility}_j(\mathcal{D})$.

This design accomplishes two things: rotating the prediction target avoids the bias of local utility (predicting only $y$), and the aggregated normalized ratios allow for comparison across different task types. To ensure results are not biased by a specific model, each sub-task uses AutoGluon to ensemble 9 predictors. The theoretical basis is that a high-fidelity generator should ensure the conditional distribution $p(x_j | \mathcal{X} \setminus \{x_j\})$ matches the real data, which corresponds to the Markov blanket concept in SCMs.

**3. Local Utility vs. Global Utility: Validating the Proxy through Comparison**

The authors rename traditional ML efficacy as "local utility"—it only treats $y$ as the target, making it a special case of global utility. A comparison reveals a critical finding: local utility correlates strongly with local CI ($r_s=0.78$) but shows almost no correlation with global CI ($r_s=0.14$). In contrast, global utility correlates strongly with global CI ($r_s=0.84$). This demonstrates that ML efficacy only reflects local structures near the target variable and cannot support judgments regarding a generator's global causal fidelity.

## Key Experimental Results

### Main Results — Structural Fidelity of 13 Generators on SCM Datasets

| Generator | Global CI ↑ | Global Utility ↑ | Local Utility ↑ | Shape ↑ |
|-----------|------------|-------------------|-----------------|---------|
| $\mathcal{D}_\text{ref}$ | 1.00 | 0.99 | 0.99 | 1.00 |
| TabSyn | **0.70** | 0.76 | 0.76 | 0.50 |
| TabDDPM | **0.69** | **0.80** | 0.29 | 0.62 |
| TabDiff | 0.57 | 0.75 | 0.80 | 0.69 |
| SMOTE | 0.30 | 0.39 | **0.92** | 0.82 |
| CTGAN | 0.08 | 0.26 | 0.80 | 0.46 |
| GReaT | 0.16 | 0.25 | 0.27 | 0.62 |
| NRGBoost | 0.11 | 0.16 | 0.75 | 0.65 |

**Key Findings**: SMOTE achieves the highest local utility (0.92), yet its global CI is only 0.30—demonstrating how traditional ML utility can mislead evaluations. Diffusion models (TabDDPM/TabSyn/TabDiff) are consistently optimal in global structural fidelity.

### Global Utility Rankings on Real-World Datasets

| Generator | Global Utility ↑ | Local Utility ↑ |
|-----------|-------------------|-----------------|
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

On real-world datasets, the Top-3 for Global Utility remain diffusion models, consistent with the SCM datasets, validating the generalizability of the global utility metric.

### Correlation Analysis

| Metric Pair | Spearman $r_s$ | p-value |
|-------------|----------------|---------|
| Global Utility ↔ Global CI | **0.84** | <0.001 |
| Local Utility ↔ Local CI | 0.78 | <0.001 |
| Local Utility ↔ Global CI | 0.14 | <0.001 |

The strong correlation (0.84) between Global Utility and Global CI is a core empirical result, proving its effectiveness as a proxy metric in the absence of SCMs.

## Highlights & Insights

- **Filling the Evaluation Gap**: This work is the first to systematically incorporate structural fidelity into tabular generator evaluation frameworks, and the proposed global utility metric does not require ground-truth causal graphs.
- **Large-scale Benchmark**: 13 generators × 29 datasets × over 150,000 evaluations, far exceeding the coverage of existing benchmarks.
- **Revealing the Essence of Diffusion Model Superiority**: Diffusion models naturally learn permutation-invariant conditional distributions because noise is added independently to each feature and all features are reconstructed simultaneously during denoising, aligning with tabular data structural priors.
- **Debunking the SMOTE "Illusion"**: Experiments clearly show that while SMOTE performs well on traditional metrics, it severely violates causal structures, revealing significant evaluation bias.

## Limitations & Future Work

- The strong correlation between global utility and global CI is an empirical finding lacking formal theoretical proof.
- Evaluation is conducted at the CPDAG level, which is coarser than the full DAG level and may miss certain directional causal relationships.
- Reliance on AutoGluon ensemble predictors increases computational overhead as the number of features grows (though the Tiny-default variant shows good efficiency at 0.64s/1000 samples).
- SCM datasets utilize expert-verified causal graphs, but high-quality datasets of this type are limited in number (only 6), and generalizability requires further extension.

## Related Work & Insights

- **Tabular Generation Benchmarks**: Synthcity (Qian et al., 2024) and SynMeter (Du & Li, 2024) cover density/privacy/ML utility but lack structural fidelity; CauTabBench (Tu et al., 2024) evaluates structural fidelity but is limited to toy SCMs.
- **Generators**: Progressing from SMOTE and BN to TVAE, CTGAN, NFlow, ARF, and then to diffusion models (TabDDPM/TabSyn/TabDiff/TabEBM), LLM-based GReaT, and tree-based NRGBoost.
- **Causal Discovery**: DAG learning methods encounter significant difficulties when the number of features exceeds 10 (Zanga et al., 2022), which is why this paper evaluates at the CPDAG level.
- **Tabular Foundation Models**: Hollmann et al. (2025) empirically demonstrated that SCMs are effective structural priors for tabular data.

## Rating

| Dimension | Rating | Description |
|-----------|--------|-------------|
| Novelty | ⭐⭐⭐⭐ | First to propose a global structural fidelity metric without requiring causal graphs. |
| Technical Depth | ⭐⭐⭐⭐ | Clear theoretical analysis; CI framework and global utility designs are sound. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | 13 models × 29 datasets and 150k evaluations make it extremely thorough. |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured with intuitive motivational examples. |
| Value | ⭐⭐⭐⭐ | Open-source framework that can be directly used to evaluate new generators. |
| Total | ⭐⭐⭐⭐ | Significant contribution to tabular evaluation; global utility may become a standard metric. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](../../ACL2025/others/generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [\[ICLR 2026\] Measuring Uncertainty Calibration](measuring_uncertainty_calibration.md)
- [\[ICML 2026\] Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features](../../ICML2026/others/cascaded_flow_matching_for_heterogeneous_tabular_data_with_mixed-type_features.md)
- [\[ICLR 2026\] Prior-Free Tabular Test-Time Adaptation](prior-free_tabular_test-time_adaptation.md)
- [\[ICLR 2026\] Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion](harpoon_generalised_manifold_guidance_for_conditional_tabular_diffusion.md)

</div>

<!-- RELATED:END -->
