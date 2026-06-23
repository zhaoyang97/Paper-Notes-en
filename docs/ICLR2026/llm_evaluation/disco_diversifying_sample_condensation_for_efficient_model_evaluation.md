---
title: >-
  [Paper Note] DISCO: Diversifying Sample Condensation for Efficient Model Evaluation
description: >-
  [ICLR 2026][LLM Evaluation][Paper Note] DISCO proposes a minimalist criterion—"selecting samples where models disagree most"—to condense evaluation sets. Coupled with a "model signature + simple regression" approach to directly predict full-set performance, it reduces evaluation costs by 99% using only 100 samples on MMLU/HellaSwag/Winogrande/ARC with an err
tags:
  - ICLR 2026
  - LLM Evaluation
date: 2026-05-08
content_hash: 589b6e6adffc8cce
---
# DISCO: Diversifying Sample Condensation for Efficient Model Evaluation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=SoOgBHa3dZ](https://openreview.net/forum?id=SoOgBHa3dZ)  
**Code**: [DISCO Codebase (Provided on project page)](https://openreview.net/forum?id=SoOgBHa3dZ)  
**Area**: Efficient Model Evaluation / LLM Evaluation  
**Keywords**: Efficient Evaluation, Sample Condensation, Model Disagreement, Performance Prediction, Anchor Selection  

## TL;DR
DISCO proposes a minimalist criterion—"selecting samples where models disagree most"—to condense evaluation sets. Coupled with a "model signature + simple regression" approach to directly predict full-set performance, it reduces evaluation costs by 99% using only 100 samples on MMLU/HellaSwag/Winogrande/ARC with an error of approximately 1 percentage point, setting a new SOTA for efficient evaluation.

## Background & Motivation
**Background**: Modern Large Model evaluation has become prohibitively expensive—running LMMs-Eval on 8×A100 takes 30 to 1,400 hours per model, while HELM exceeds 4,000 GPU hours for a single model. To reduce costs, efficient evaluation methods have emerged. The mainstream framework consists of two steps: selecting a small set of "anchor" samples from the full test set, and then training a mapping from anchor accuracy to full results for extrapolation.

**Limitations of Prior Work**: Anchor selection generally relies on **clustering**—grouping samples by the "response similarity" they induce across a set of reference models, and picking representative points from each cluster (e.g., Anchor Points, tinyBenchmarks). Clustering is complex and sensitive to design choices (distance metrics, number of clusters, embedding methods). On the prediction side, performance estimation often involves estimating latent variables (like the ability parameter $\theta$ in IRT) before making predictions, adding psychometric complexity.

**Key Challenge**: The prevailing assumption is that "anchors must cover sample diversity and represent the data difficulty distribution." However, this paper asks: **What truly determines the ability to differentiate and rank models is not the diversity of the samples themselves, but whether the samples can induce diversity in responses between models.** A sample that all models answer correctly (or incorrectly) provides nearly zero information for differentiation; only samples where "some models are right and others are wrong" possess true discriminative power.

**Goal**: Simplify both ends of efficient evaluation—using per-sample statistics instead of global clustering for sample selection, and using raw output regression instead of latent variable modeling for performance prediction.

**Core Idea**: **[Information-Theoretic Optimal Greedy Selection]** It is proved that "inter-model disagreement," when the goal is to differentiate/rank models, is the information-theoretically optimal per-sample signal for estimating benchmark performance. Thus, one only needs to rank samples by disagreement scores and take the top-k, without any clustering. **[Direct Regression of Model Signatures]** The raw outputs of a model on the selected subset are concatenated into a "model signature," which is fed directly into a simple predictor (Random Forest/kNN), bypassing latent parameter estimation in IRT.

## Method

### Overall Architecture
DISCO follows the "select samples → predict performance" two-stage pipeline but replaces both components with simpler parts. Given a set of **Source Models** $F=\{f^1,\dots,f^M\}$ with known full-set performance and a full test set $D$, the first step sorts samples by per-sample disagreement scores and takes the top-k to obtain the condensed subset $D_{\text{DISCO}}$. The second step concatenates the outputs of any model on this subset into a "model signature," which, after PCA dimensionality reduction, is mapped to full performance via a regressor. During training, the predictor is fitted on source models; during testing, an unseen **Target Model** only needs to perform inference on the 100 anchors once to estimate its full performance.

```mermaid
flowchart LR
    A[Full Test Set D] --> B[Source Models F Inference on D]
    B --> C[Per-sample Disagreement Score PDS/JSD]
    C --> D[Rank & Pick Top-k<br/>Get Subset D_DISCO]
    D --> E[Model Output on D_DISCO]
    E --> F[Concatenate to Model Signature]
    F --> G[PCA Dimensionality Reduction]
    G --> H[Regressor RF/kNN]
    H --> I[Estimate Full Performance]
```

### Key Designs

**1. Information-Theoretically Optimal Disagreement Selection: Reducing "which samples to test" to a mutual information maximization problem.** The starting point for DISCO is Proposition 1: Let the model index $m$ be uniformly sampled from the source model set, and let $\bar{y}_i$ be the categorical random variable corresponding to the ensemble mean prediction. The mutual information between the "performance function $S(m)$ induced by model identity" and the "prediction $\hat{y}_i$ on sample $i$" is exactly equal to the generalized Jensen-Shannon Divergence (JSD) of the prediction distributions across models on that sample:

$$\mathrm{MI}_{m,\hat y_i}\big(S(m);\hat y_i\big)=H(\hat y_i)-\mathbb{E}_m\big[H(\hat y_i^m)\big]=\mathrm{JSD}\big(\hat y_i^1,\dots,\hat y_i^M\big).$$

Intuitively, the amount of information a sample carries "to distinguish models" is proportional to how dispersed the prediction distributions of various models are on it. Thus, the optimal greedy criterion is to select samples by descending JSD—providing a rigorous justification for "picking samples with the highest model disagreement" rather than approximating "representativeness" through clustering.

**2. PDS: An Interpretable Proxy for JSD, Grounding Disagreement in Computable Continuous Scores.** Directly calculating multi-distribution JSD is cumbersome. The authors use the Predictive Diversity Score (PDS, derived from OOD detection), which is a continuous generalization of the "number of unique argmax categories among $M$ source models":

$$\mathrm{PDS}\big(\hat y_i^1,\dots,\hat y_i^M\big):=\frac1C\sum_c \max_m f_c^m(x_i).$$

Higher PDS indicates that more models concentrate probability mass on different categories, reflecting stronger disagreement. Proposition 2 further provides a sandwich inequality between PDS and JSD: $\frac{2}{M^2\ln 2}(\mathrm{PDS}_i-1)^2\le \mathrm{JSD}_i\le \frac{M}{M-1}\log M\cdot(\mathrm{PDS}_i-1)$, showing that ranking by PDS is highly consistent with ranking by JSD. Both were tested in experiments, but PDS with Random Forest tended to be the most stable.

**3. Model Signature + Simple Regression: Using Raw Outputs Instead of Scalar Accuracies for Extrapolation.** Previous prediction methods used only scalar summaries like (weighted/calibrated) accuracy on anchors, causing information loss. DISCO directly concatenates the raw outputs of the target model on the subset into a high-dimensional "model signature" $f(D_{\text{DISCO}})=[f(x_1),\dots,f(x_L)]$, preserving much richer discriminative signals. Since the dimensionality can be high (number of categories × number of samples), PCA is first used to compress the signature to, for example, 256 dimensions $Q\circ f(D_{\text{DISCO}})$ to suppress overfitting. Then, two naive prediction paths are used: **kNN**—finding the $K$ nearest neighbors in the source model signatures and averaging their performance; and **Parametric Regression**—training a mapping $R$ (Linear/Random Forest/Neural Network) such that $R\circ Q\circ f^m(D_{\text{DISCO}})$ approximates the ground truth. The deliberate choice of simple components demonstrates that "simple is optimal."

**4. Chronological Split: Verifying Generalization Closer to Reality.** Meta-model methods are often criticized for relying on existing model distributions and failing for future models. Instead of artificial stress tests based on performance levels, DISCO uses a chronological split—source models are those released before 2024-01-13, and the test set consists of models released after (9:1), simulating the realistic scenario of "training a predictor with old models to estimate new ones." Experiments show that under this more realistic split, rank correlation is .987, nearly identical to the .986 of a uniform split, proving the robustness of the method.

## Key Experimental Results

### Main Results (Language Domain, each dataset compressed to 100 samples; lower MAE / higher Rank is better)

| Method | Selection | Prediction | MMLU MAE/Rank | HS MAE/Rank | WG MAE/Rank | ARC MAE/Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline | Random | Direct | 3.45 / .916 | 2.85 / .839 | 3.60 / .827 | 2.61 / .898 |
| tinyBenchmarks | Random | gp-IRT | 2.79 / .922 | 1.96 / .819 | 1.64 / .928 | 2.22 / .921 |
| Anchor-corr | — | gp-IRT | 2.08 / .927 | 1.27 / .937 | 1.95 / .918 | 2.18 / .948 |
| Metabench† | Best for val. | ability-IRT | 2.08 / .904 | 0.80 / .974 | 1.23 / .947 | 1.14 / .971 |
| Model signature | Random | Sig.+RF | 1.81 / .933 | 1.36 / .938 | 1.29 / .926 | 1.72 / .938 |
| **Ours (DISCO)** | High PDS | Sig.+kNN | 1.31 / .972 | 1.32 / .956 | 1.19 / .951 | 1.96 / .937 |
| **Ours (DISCO)** | **High PDS** | **Sig.+RF** | **1.07 / .987** | **1.01 / .984** | **1.00 / .967** | 1.47 / .971 |
| Ours (DISCO) | High JSD | Sig.+RF | 1.30 / .987 | 0.86 / .972 | 1.09 / .973 | 1.75 / .938 |

† Metabench requires more samples to converge (150 for MMLU/ARC, 450 for HS, 200 for WG), making it not entirely comparable.

### Ablation Study (MMLU, 100 samples, Rank correlation)

| Dimension | Setting | Rank |
| :--- | :--- | :--- |
| (a) Model Split | Chronological / Uniform | .987 / .986 (Robust) |
| (b) Stratified Sampling | Off / On | .987 / .978 (Stratification not helpful for PDS) |
| (c) # Source Models | 100 / 382 | .969 / .987 (100 models already outperform tinyBenchmarks' .927) |
| (d) Dim. Reduction | None (3100-d) / PCA-256 | .918 / .987 |
| (e) Predictor | Random Forest is best | .987 |

### Key Findings
- **Decomposition of Contributions**: "Model signature + RF" alone (with random sampling) achieves a SOTA level of 1.81%p / .933; adding PDS sampling pushes MMLU to 1.07%p / .987, proving that both innovations are effective and additive.
- **Extreme Compression**: When the sample count drops to 10, non-parametric kNN is more stable than Random Forest, suggesting non-parametric predictors are preferable for extreme compression.
- **Cross-Domain Generalization**: Moving to the vision domain on the ImageNet validation set (400 timm models) and compressing to 100 points (99.8% cost reduction), DISCO achieves 0.63%p / .969, entirely surpassing Lifelong Benchmark (.838/2.06) and SSEPY (.762/3.05).

## Highlights & Insights
- **Elevating "Sample Selection" from Engineering Intuition to an Information-Theoretic Proposition**: Proposition 1 uses a single mutual information equality to show that "model disagreement = information to distinguish models," giving theoretical legitimacy to "selecting the most disagreed-upon samples" rather than just another heuristic clustering trick.
- **Anti-Consensus Design**: The implicit consensus that "samples must be diverse and cover the difficulty spectrum" is directly challenged. The paper proves experimentally that stratified sampling is actually unhelpful under PDS—offering a clean shift in perspective.
- **Empirical Proof that Simple is Optimal**: Using the most basic Random Forest/kNN + PCA outperforms Metabench with IRT latent variables, using only 100 instead of 150–450 samples, making it extremely easy to replicate and deploy.

## Limitations & Future Work
- **Sensitivity to Model Distribution Shift**: The predictor is trained on source models. When models with entirely new architectures, training paradigms, or objectives emerge, performance may degrade due to unseen patterns—the authors admit this is the primary weakness and suggest adaptive sampling or periodic retraining as mitigation.
- **Applicable Only to Closed-Ended Tasks with "Predefined Class Probabilities"**: The method relies on prediction probabilities across several candidate categories for each question (Proposition 1's classes). It is thus not suited for open-ended generation tasks like translation or summarization without predefined sets of correct/incorrect outputs.
- **Dependency on Source Model Pool Size**: While 100 source models are strong enough, discriminative power in the signature space may drop if there are too few source models or if they are highly homogeneous.

## Related Work & Insights
- **Anchor/Efficient Evaluation Lineage**: Anchor Points (Vivek 2023), tinyBenchmarks (Polo 2024, IRT), Metabench (Kipnis 2024), and dynamic anchors (Hofmann 2025) all fall into the "select anchors + predict" framework. DISCO simplifies and surpasses them in both selection (disagreement vs. representativeness) and prediction (signature vs. latent variables).
- **Interdisciplinary Reuse of PDS**: The disagreement score PDS was originally an OOD detection tool (Rubinstein 2024). Repurposing it here to measure sample information content is an interesting transfer from "OOD uncertainty measurement → evaluation sample selection," potentially inspiring the use of ensemble disagreement signals in other "data selection" scenarios (active learning, data pruning, curriculum construction).
- **Distinction from Active Testing**: Active testing allocates labeling budgets to information-rich samples but requires inference on the full set first. DISCO focuses on the inference cost itself, selecting static anchors to perform only one inference on the target model—making the two complementary.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Information-theoretic characterization of "disagreement as information" + minimalist anti-consensus sampling. The logic is clear and theoretically supported, even if the components (PDS, signatures, RF) are largely existing building blocks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 4 language benchmarks + ImageNet vision domain, 424/400 real models, complete factor analysis, and cross-compression rate curves. Chronological split design is realistic; lacks verification on open-ended generation.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation-Proposition-Method-Experiment logic is smooth. Two propositions set the tone, tables are clear, and formulas are well-explained.
- **Value**: ⭐⭐⭐⭐ — 99% cost reduction with ~1%p error is highly practical for frequent monitoring during training, evaluation with limited compute, and spot-checking after deployment. Low barrier to engineering adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SparseEval: Efficient Evaluation of Large Language Models by Sparse Optimization](sparseeval_efficient_evaluation_of_large_language_models_by_sparse_optimization.md)
- [\[ICML 2025\] Sample Efficient Demonstration Selection for In-Context Learning](../../ICML2025/llm_evaluation/sample_efficient_demonstration_selection_for_in-context_learning.md)
- [\[ICLR 2026\] Fewer Battles, More Gain: An Information-Efficient Framework for Arena-based LLM Evaluation](fewer_battles_more_gain_an_information-efficient_framework_for_arena-based_llm_e.md)
- [\[ICLR 2026\] Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation](dont_passk_a_bayesian_framework_for_large_language_model_evaluation.md)
- [\[ICLR 2026\] NAIPv2: Debiased Pairwise Learning for Efficient Paper Quality Estimation](naipv2_debiased_pairwise_learning_for_efficient_paper_quality_estimation.md)

</div>

<!-- RELATED:END -->
