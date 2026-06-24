---
title: >-
  [Paper Note] LCDB 1.1: A Database Illustrating Learning Curves Are More Ill-Behaved Than Previously Thought
description: >-
  [NeurIPS 2025][LLM Pretraining][learning curves] This paper constructs LCDB 1.1, a large-scale high-resolution learning curve database, demonstrating that ill-behaved sample learning curves (non-monotonic, non-convex) are approximately twice as prevalent as previously believed, with roughly 15% of curves exhibiting significant ill-behavior that feature scaling largely fails to remedy.
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "learning curves"
  - "scaling laws"
  - "model selection"
  - "ill-behavior"
  - "tabular data"
  - "benchmark"
date: 2026-05-08
content_hash: 47b5768e61de34ad
---

# LCDB 1.1: A Database Illustrating Learning Curves Are More Ill-Behaved Than Previously Thought

**Conference**: NeurIPS 2025
**arXiv**: [2505.15657](https://arxiv.org/abs/2505.15657)  
**Code**: [GitHub](https://github.com/learning-curve-research/LCDB-1.1)  
**Area**: LLM Evaluation
**Keywords**: learning curves, scaling laws, model selection, ill-behavior, tabular data, benchmark

## TL;DR

This paper constructs LCDB 1.1, a large-scale high-resolution learning curve database, demonstrating that ill-behaved sample learning curves (non-monotonic, non-convex) are approximately twice as prevalent as previously believed, with roughly 15% of curves exhibiting significant ill-behavior that feature scaling largely fails to remedy.

## Background & Motivation

Sample-wise learning curves characterize how model performance varies with training set size. They are critical for multi-fidelity hyperparameter search, scaling law estimation, and data requirement forecasting. The field has long assumed that learning curves are **monotonically decreasing** and **convex** (well-behaved)—i.e., more data always leads to better generalization with diminishing marginal returns.

The predecessor LCDB 1.0 claimed that "the vast majority of learning curves are well-behaved." The authors argue this conclusion is premature, for the following reasons:

**Insufficient resolution**: LCDB 1.0 uses anchor points with excessively large spacing, making it difficult to capture subtle ill-behavior.

**Lack of statistical testing**: No quantification of which ill-behaviors are statistically significant.

**Absence of feature scaling**: Feature scaling is standard practice; its omission may introduce spurious ill-behavior.

**Data quality issues**: Presence of data leakage and missing values.

These limitations motivated the construction of LCDB 1.1 to address four core questions: How prevalent is ill-behavior? Which learners are responsible? Can feature scaling remedy it? How much does ill-behavior affect downstream tasks (curve fitting, model selection)?

## Method

### Overall Architecture

LCDB 1.1 is a large-scale learning curve database covering 265 OpenML datasets (including 72 from the carefully curated CC-18 subset) and 32 learners, including modern methods CatBoost, TabNet, RealMLP, and TabPFN v2.

### Database Design Improvements

**Data splitting**: A $5 \times 5$ nested outer/inner seed split is adopted:

$$D \xrightarrow{\text{outer split}} (D_{\text{train-val}}^{(m_o)}, D_{\text{test}}^{(m_o)}) \xrightarrow{\text{inner split}} (D_{\text{train}}^{(m_o,m_i)}, D_{\text{val}}^{(m_o,m_i)})$$

**Fourfold increase in anchor resolution**: From $n_k = \lceil 16 \cdot 2^{k/2} \rceil$ to $n_k = \lceil 16 \cdot 2^{k/8} \rceil$.

**Dual versions**: Versions with and without data leakage are provided to accommodate different use cases (model selection vs. data requirement estimation).

**Three feature scaling variants**: No scaling (noFS), min-max scaling, and standardization.

### Rigorous Detection of Ill-Behavior

**Monotonicity violation**: The maximum violation error is defined as:

$$\epsilon_{\text{mono}} = \max\left(0, \max_{1 \leq i < j \leq N} (C(n_j) - C(n_i))\right)$$

After identifying the maximizing anchor pair $(i^*, j^*)$, a paired one-sided $t$-test with Bonferroni correction ($\alpha' = \frac{\alpha}{N(N-1)/2}$) is applied over 25 repetitions to assess significance.

**Convexity violation**: A violation error based on linear interpolation is defined as:

$$\epsilon_{\text{conv}} = \max\left(0, \max_{1 \leq h < i < j \leq N} (C(n_i) - C_{\text{interpolated}}(n_i; n_h, n_j))\right)$$

Triple comparisons apply Bonferroni correction $\alpha' = \frac{\alpha}{N(N-1)(N-2)/6}$. Unlike LCDB 1.0, this work correctly accounts for anchor point distribution on a logarithmic scale.

**Peaking**: Combines convexity and monotonicity violations—at the convexity-violating triple $(h^*, i^*, j^*)$, the method verifies a monotonicity violation from $h^*$ to $i^*$ and a statistically significant improvement from $i^*$ to $j^*$.

**Dipping**: A monotonicity violation where $j$ is fixed to the last anchor $N$, indicating an irrecoverable performance degradation.

### Downstream Impact Analysis

- **Curve fitting**: Parametric models such as POW4 ($\hat{C}(n) = a - b(d+n)^{-c}$) are used for interpolation, with MSE as the comparison metric.
- **Model selection**: Successive Halving (SH) is applied using training set size as the fidelity dimension for model selection.

## Key Experimental Results

### Main Results: Ill-Behavior Statistics (DA1)

| Metric | LCDB 1.1 CC-18 (noFS) | LCDB 1.1 FULL (noFS) | LCDB 1.0 |
|--------|----------------------|---------------------|----------|
| Non-monotonic (¬M) | **9.9%** | 9.6% | 5.1% |
| Non-convex (¬C) | **11.5%** | 12.3% | 5.7% |
| Ill-behaved (¬M ∪ ¬C) | **14.9%** | 15.4% | 8.1% |
| Peaking | **5.0%** | 5.7% | 2.5% |
| Dipping | **6.1%** | 6.9% | 4.6% |

Key finding: approximately 15% of learning curves are significantly ill-behaved, nearly **twice** the estimate reported by LCDB 1.0.

**Ill-behavior rates by learner** (selected extreme cases):

| Learner | Ill-behavior Rate |
|---------|------------------|
| CatBoost | 1.5% |
| TabPFN v2 | 1.5% |
| Decision Tree | 1.5% |
| KNN | 3.8% |
| MLP | 27.9% |
| LDA | 37.7% |
| QDA | 45.7% |
| Sigmoid SVM | 58.1% |
| TabNet | **74.3%** |

### Ablation Study: Feature Scaling (DA2)

| Scaling | Ill-behavior Rate (CC-18) |
|---------|--------------------------|
| No scaling | 14.9% |
| Min-max | 13.5% |
| Standardization | 11.2% |

Feature scaling only marginally reduces ill-behavior. Ridge and MLP are among the few learners that benefit noticeably, while Nearest Centroid performs worse after scaling.

### Curve Fitting Experiment (DA3)

The mean POW4 fitting MSE (on a logarithmic scale) for non-monotonic curves is **more than 10 times** larger than for monotonic curves. A clear positive correlation between violation error and MSE confirms that parametric models struggle to fit ill-behaved curves.

### Model Selection Experiment (DA4)

Groups of learners with frequently crossing learning curves exhibit significantly worse optimal selection rates and higher regret under Successive Halving compared to groups with rarely crossing curves, demonstrating that crossing curves substantially complicate multi-fidelity model selection.

### Key Findings

1. Ensemble methods (Random Forest, gradient boosting) and CatBoost exhibit the most well-behaved learning curves.
2. TabNet's ill-behavior rate reaches 74.3%, primarily manifesting as phase transitions, likely due to default hyperparameters being ill-suited for small datasets.
3. Sigmoid SVM is the most unstable among classical methods, predominantly exhibiting dipping.
4. Peaking in LDA and Ridge is consistent with Fisher classifier theory—the peak is most severe when training set size approaches the feature dimensionality.
5. Phase transition phenomena in MLPs are observed empirically on real datasets for the first time.

## Highlights & Insights

1. **Debunking the "learning curves are always well-behaved" myth**: Rigorous statistical methods demonstrate that approximately 15% of curves are significantly ill-behaved, serving as an important contribution to the NeurIPS community's discourse on scaling laws.
2. **High-quality benchmark**: Fourfold resolution, dual versions, three scaling variants, 32 learners, and publicly released data constitute a highly valuable resource for future learning curve modeling research.
3. **Statistical rigor**: Bonferroni correction combined with paired testing ensures that noise is not misclassified as ill-behavior, lending greater credibility to the conclusions.
4. **Inclusion of modern learners**: The strong performance of CatBoost and TabPFN contrasts sharply with TabNet, providing practical guidance for tabular data method selection.
5. **Quantification of practical impact**: Beyond identifying the existence of ill-behavior, the paper quantifies its actual harm to curve fitting and model selection.

## Limitations & Future Work

1. **Coverage limited to tabular classification**: Learning curves in regression, NLP, and CV domains are not addressed.
2. **Default hyperparameters**: All learners are evaluated with default hyperparameters; tuning may reduce ill-behavior, particularly for TabNet.
3. **Substantial computational cost**: The database already consumed 800,000 CPU hours; incorporating hyperparameter tuning would be considerably more expensive.
4. **Bonferroni correction is conservative**: True ill-behavior rates may be underestimated (the Holm method yields approximately 19%).
5. **QDA reproducibility issues**: Numerical non-determinism in SVD renders some results non-reproducible.

## Related Work & Insights

- Complements the scaling law literature (Kaplan et al. 2020), which focuses on large-scale deep learning, whereas this work addresses classical methods on tabular data.
- Provides a more realistic benchmark for multi-fidelity methods such as BOHB and Hyperband.
- Ill-behavior phenomena may inspire new parametric learning curve model designs, as existing power-law models assume well-behaved curves.
- Dipping and phase transition phenomena warrant further theoretical analysis.

## Rating

- **Novelty**: ⭐⭐⭐ — The core contribution is a database and analytical framework rather than a novel algorithm, though the statistical detection methodology is well-designed.
- **Practicality**: ⭐⭐⭐⭐⭐ — Directly applicable to hyperparameter search, data budget estimation, and learner selection.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 32 learners × 265 datasets × 3 scaling variants with rigorous statistical testing.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with rich figures and tables, though content density is high.
- **Recommended Reading**: ⭐⭐⭐⭐ — Essential reading for researchers in tabular data, AutoML, and scaling laws.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs](efficient_pre-training_of_llms_via_topology-aware_communication_alignment_on_mor.md)
- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[ICML 2025\] In-Context Adaptation to Concept Drift for Learned Database Operations](../../ICML2025/llm_pretraining/in-context_adaptation_to_concept_drift_for_learned_database_operations.md)
- [\[ACL 2025\] Byte Latent Transformer: Patches Scale Better Than Tokens](../../ACL2025/llm_pretraining/byte_latent_transformer.md)
- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](composition_and_alignment_of_diffusion_models_using_constrai.md)

</div>

<!-- RELATED:END -->
