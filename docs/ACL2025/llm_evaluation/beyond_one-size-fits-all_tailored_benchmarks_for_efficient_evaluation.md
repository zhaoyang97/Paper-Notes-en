---
title: >-
  [Paper Note] Beyond One-Size-Fits-All: Tailored Benchmarks for Efficient Evaluation
description: >-
  [ACL 2025][LLM Evaluation][Efficient Evaluation] This paper proposes TailoredBench, a method that **adaptively constructs a customized coreset** (Native-coreset) for each target model to be evaluated, instead of using a static subset shared across all models. By utilizing adaptive source model selection, scalable K-Medoids clustering, and a calibrated estimation strategy, it reduces the Mean Absolute Error (MAE) of accuracy estimation by **31.4%** on average under an inferenc…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Efficient Evaluation"
  - "Benchmark Compression"
  - "Coreset Selection"
  - "Prediction Consistency"
  - "K-Medoids Clustering"
  - "Model Ranking"
date: 2026-05-08
content_hash: bbb553c27a625630
---

# Beyond One-Size-Fits-All: Tailored Benchmarks for Efficient Evaluation

**Conference**: ACL 2025  
**arXiv**: [2502.13576](https://arxiv.org/abs/2502.13576)  
**Code**: [https://github.com/marvelcell/TailoredBench](https://github.com/marvelcell/TailoredBench)  
**Authors**: Peiwen Yuan, Yueqi Zhang, Shaoxiong Feng, Yiwei Li, Xinglin Wang, Jiayi Shi, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li  
**Institutions**: Beijing Institute of Technology, Xiaohongshu Inc  
**Area**: Efficient Evaluation / Benchmark Compression  
**Keywords**: Efficient Evaluation, Benchmark Compression, Coreset Selection, Prediction Consistency, K-Medoids Clustering, Model Ranking

## TL;DR

This paper proposes TailoredBench, a method that **adaptively constructs a customized coreset** (Native-coreset) for each target model to be evaluated, instead of using a static subset shared across all models. By utilizing adaptive source model selection, scalable K-Medoids clustering, and a calibrated estimation strategy, it reduces the Mean Absolute Error (MAE) of accuracy estimation by **31.4%** on average under an inference budget of only 20-40 samples.

## Background & Motivation

**Evaluation Cost Crisis**: Evaluating a 10B-parameter model on the HELM benchmark costs $1,700 (via API) or 1,200+ GPU hours. The cost scales linearly when comparing $X$ configurations.

**Existing Efficient Evaluation Paradigms**:
   - Construct sample embeddings using evaluation results from publicly available source models.
   - Perform clustering to select a small representative coreset (typically <100 samples).
   - Estimate the overall benchmark performance using the target model's performance on this coreset.

**Flaws in the Core Assumption**: Existing methods assume high **prediction consistency** between source models and the target model—if source models perform similarly on sample a and sample b, the target model should as well. However, the authors find that **this assumption does not hold in practice**:
   - t-SNE visualization (Hellaswag) shows that when embedding with the target model, the average distance from samples to cluster centroids increases from 10.09 to 12.48.
   - The coreset selected by source models fails to effectively represent the behavioral patterns of the target model.

**Core Motivation**: Abandon the "one-size-fits-all" static coreset and tailor a customized evaluation subset for each target model.

## Method

### Overall Architecture: Global-to-Native Evaluation Pipeline

TailoredBench consists of four tightly integrated steps:

### Step 1: Construct G-set (Global-coreset)

- For each sample $x_k$, construct an embedding vector $\dot{x}_k^{\mathcal{S}}$ using the correctness of all $|\mathcal{S}|$ source models.
- Perform **K-Medoids clustering** on this embedding space, choosing cluster centroids to form the G-set (default is 10 samples).
- Use the G-set as a **probe** to identify source models that are most consistent with the target model.
- **Distance Metric Selection**: Adopt **Manhattan distance** (element-wise) rather than correlation distance, as correlation distance assumes linear relationships and is unsuitable for discrete binary embeddings.

### Step 2: Adaptive Native Source Model Selection

- Run inference with the target model on the G-set.
- Encode the G-set predictions of all source models and the target model into embeddings.
- Calculate the average prediction consistency $\bar{d}$ among all models as a threshold.
- For each target model $t_m$, select source models with distance less than $\bar{d}$ to form the native source model set $\mathcal{S}_{t_m}$.
- **Unify the number of native source models** $\bar{n}$ across all target models to maintain consistent embedding dimensions.

### Step 3: Scalable K-Medoids Clustering to Construct N-set

- Reconstruct sample embeddings based on the predictions of native source models (aligning better with the target model's perspective).
- **Anchored Initialization**: Fix the samples in G-set as initial cluster centroids, and add $|N-set| - |G-set|$ random samples.
- **Dynamic Refinement**: Non-G-set cluster centroids can be updated (selecting the sample that minimizes intra-cluster distance), while G-set centroids remain fixed.
- Iterate until convergence $\rightarrow$ obtain the customized N-set $\mathcal{N}_{t_m}$.

### Step 4: Calibrated Performance Estimation

Instead of directly using the centroid's performance to represent the entire cluster, leverage the prediction consistency of source models for **calibration**:

- For a non-centroid sample $x'$ within a cluster, calculate the scaling factor: $\text{Scale}(x') = \frac{\bar{c}_{\mathcal{S}_{t_m}, x'} + 0.5}{\bar{c}_{\mathcal{S}_{t_m}, x} + 0.5}$
- Calibrate the target model's prediction on all samples using the scaling factor: $c_{t_m, x'} = (c_{t_m, x} + 0.5) \cdot \text{Scale}(x') - 0.5$
- Overall benchmark performance: $P_{t_m} = \frac{1}{|\mathcal{D}|} \sum_{x' \in \mathcal{D}} c_{t_m, x'}$

## Key Experimental Results

### Main Results (5 Benchmarks, 300+ Models)

| Benchmark | Inferences | Best Baseline MAE | TailoredBench MAE | MAE Reduction |
|------|--------|------------|-------------------|--------|
| ARC Challenge | 30 | 0.036 | **0.028** | 22.2% |
| Hellaswag | 30 | 0.043 | **0.018** | 58.1% |
| GSM8K | 30 | 0.041 | **0.033** | 19.5% |
| Winogrande | 30 | 0.038 | **0.024** | 36.8% |
| POPE | 30 | 0.034 | **0.031** | 8.8% |

- Average MAE is reduced by **31.4%**
- Kendall's τ is also consistently improved (higher ranking accuracy)
- Pairwise model ranking accuracy on Hellaswag reaches **96.0%**

### Distance Metric Ablation

| Distance Type | Kendall's τ | MAE |
|---------|------------|-----|
| Correlation | 0.720 | 0.032 |
| Cosine | 0.736 | 0.028 |
| **Manhattan** | **0.740** | **0.027** |

- Manhattan distance outperforms correlation distance on both continuous and discrete correctness formats.

### Calibration Strategy Ablation

| Strategy | Kendall's τ | MAE |
|------|------------|-----|
| Without Calibration | 0.724 | 0.030 |
| **With Calibration** | **0.740** | **0.027** |

### G-set Size Analysis

| G-set Size | Kendall's τ | MAE |
|-----------|------------|-----|
| 5 | 0.734 | 0.030 |
| **10** | **0.740** | **0.027** |
| 15 | 0.736 | 0.028 |
| 25 | 0.731 | 0.029 |

- **10 samples are sufficient as a probe**; an excessively large G-set restricts the flexibility of the N-set.

### Large Inference Budget Validation (Hellaswag)

| Inferences | Method | Kendall's τ | MAE |
|--------|------|------------|-----|
| 150 | Random | 0.935 | 0.030 |
| 150 | AnchorPoints | 0.940 | 0.040 |
| 150 | gp-IRT | 0.936 | **0.012** |
| 150 | **TailoredBench** | **0.943** | **0.012** |

- Retains advantages even under a larger inference budget.

### Key Findings

- A larger number of native source models and a higher prediction consistency with the target model lead to more accurate estimations.
- Target models tend to select **source models from the same family** (e.g., Llama selects Llama, Qwen selects Qwen), indicating shared prediction patterns within model families.
- All advantages are verified as statistically significant via Z-test (p < 0.05).

## Highlights & Insights

1. **Directly addressing flaws in the core assumption**: The failure of the "prediction consistency assumption" is intuitively demonstrated through t-SNE visualization, making the motivation clear and compelling.
2. **Elegant global-to-local design**: The four-step pipeline (G-set as probe $\rightarrow$ adaptive source model selection $\rightarrow$ customized N-set $\rightarrow$ calibrated estimation) is logically self-consistent.
3. **10-sample probe**: Requires inference of the target model on only 10 samples to identify the best-matching subset of source models, which is extremely efficient.
4. **Cross-modal generalization**: It is effective across both NLP (ARC, Hellaswag, GSM8K, Winogrande) and multimodal (POPE) benchmarks.
5. **Significant practical impact**: For model developers, it can reduce evaluation costs by several orders of magnitude during hyperparameter/configuration search.

## Limitations & Future Work

1. **Dependence on evaluation results of source models**: It requires complete evaluation results of a large number of source models across the entire benchmark; thus, the cold-start cost for new or private benchmarks still exists.
2. **Assumption of sufficient source model coverage**: If the target model's behavioral patterns differ drastically from all available source models, the method might fail.
3. **Validated only on multiple-choice/classification benchmarks**: The applicability to open-ended generation tasks (such as summarization or dialogue) remains unexplored.
4. **Static G-set**: The G-set is shared across all target models; if there is extreme divergence in the target model population, a single G-set may not be sufficient.
5. **Simplistic calibration strategy**: The linear scaling factor assumes that the relationship between source and target model behaviors is linear.

## Related Work & Insights

- **AnchorPoints (Vivek et al. 2024)**: Selects a static coreset using K-Medoids clustering based on source model predictions.
- **gp-IRT (Polo et al. 2024)**: Extracts latent representations for samples based on Item Response Theory (IRT).
- **Flash-HELM (Perlitz et al. 2023)**: Dynamically adjusts the size of a random subset.
- **Sort & Search (Prabhu et al. 2024)**: Utilizes question difficulty and dynamic programming.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ The concept of a customized coreset is clear and novel, and the critique of the prediction consistency assumption is well-supported by extensive experiments.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 5 benchmarks and 300+ models, with diverse ablation studies and analyses; extremely thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear, though the proliferation of symbols requires careful reading.
- **Value**: ⭐⭐⭐⭐ Direct application value for leaderboard maintainers and model developers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ChatBench: From Static Benchmarks to Human-AI Evaluation](chatbench_from_static_benchmarks_to_human-ai_evaluation.md)
- [\[ACL 2025\] ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities](onebench_to_test_them_all_sample-level_benchmarking_over_open-ended_capabilities.md)
- [\[ACL 2026\] Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation](../../ACL2026/llm_evaluation/beyond_static_benchmarks_synthesizing_harmful_content_via_persona-based_simulati.md)
- [\[ACL 2025\] Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks](somethings_fishy_in_the_data_lake_a_critical_re-evaluation_of_table_union_search.md)
- [\[ICLR 2026\] DISCO: Diversifying Sample Condensation for Efficient Model Evaluation](../../ICLR2026/llm_evaluation/disco_diversifying_sample_condensation_for_efficient_model_evaluation.md)

</div>

<!-- RELATED:END -->
