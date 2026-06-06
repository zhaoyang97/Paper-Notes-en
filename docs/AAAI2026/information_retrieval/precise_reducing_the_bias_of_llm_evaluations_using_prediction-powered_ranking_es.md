---
title: >-
  [Paper Note] PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation
description: >-
  [AAAI 2026][Information Retrieval & RAG][LLM Evaluation] This paper extends the Prediction-Powered Inference (PPI) framework to sub-instance-level ranking metrics (e.g., Precision@K)…
tags:
  - "AAAI 2026"
  - "Information Retrieval & RAG"
  - "LLM Evaluation"
  - "Prediction-Powered Inference"
  - "Ranking Metrics"
  - "Bias Correction"
  - "Human Annotation"
date: 2026-05-08
content_hash: b75e84ff19b42314
---

# PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation

**Conference**: AAAI 2026
**arXiv**: [2601.18777](https://arxiv.org/abs/2601.18777)  
**Code**: None  
**Area**: Information Retrieval
**Keywords**: LLM Evaluation, Prediction-Powered Inference, Ranking Metrics, Bias Correction, Human Annotation

## TL;DR
This paper extends the Prediction-Powered Inference (PPI) framework to sub-instance-level ranking metrics (e.g., Precision@K), achieving unbiased ranking metric estimation using only 30–100 human annotations combined with large-scale LLM judgments. The computational complexity is reduced from $O(2^{|C|})$ to $O(2^K)$, and the approach has been successfully deployed to guide an LLM-based query rewriting system in an Indian e-commerce search setting.

## Background & Motivation

**Background**: LLM-as-Judge has become a mainstream approach for evaluating search and recommendation systems, serving as a cost-effective alternative to human annotation. Ranking systems undergo frequent iteration cycles requiring repeated evaluation, while implicit signals such as clicks suffer from position bias.

**Limitations of Prior Work**: (a) LLM judgments exhibit systematic bias, and direct use tends to overestimate search relevance (e.g., Claude Sonnet's Precision@4 bias on ESCI exceeds 15%); (b) human annotation is costly and does not scale, particularly for applications lacking dedicated annotation teams; (c) the original PPI framework supports only instance-level metrics (e.g., accuracy) and cannot handle ranking metrics (Precision@K, NDCG, etc.) due to a mismatch between annotation granularity (query-document) and metric granularity (query-level).

**Key Challenge**: Estimating ranking metrics requires aggregating human and LLM annotations at the sub-instance (query-document) level, yet the PPI output space $Y = \{0,1\}^{|C|}$ is computationally intractable for corpora of millions of documents.

**Goal**: To achieve statistically guaranteed ranking metric estimation using a minimal number of human annotations (30–100 queries) to correct LLM judgment bias.

**Key Insight**: The observation that Precision@K computation involves only the relevance of the Top-K documents allows the output space to be reduced from $\{0,1\}^{|C|}$ to $\{0,1\}^K$, making PPI tractable for ranking metrics.

**Core Idea**: By exploiting the sparse structure of ranking metrics, the PPI computation space is reduced from corpus size to $K$, enabling unbiased ranking metric estimation from a small set of human annotations augmented with large-scale LLM judgments.

## Method

### Overall Architecture
The input consists of a small gold dataset $\mathcal{D}_g$ with human annotations ($n = 30$–$100$ queries, each with Top-K document relevance labels) and a large unlabeled dataset $\mathcal{D}_u$ ($N = 3{,}000$–$60{,}000$ queries) for which an LLM provides probabilistic relevance judgments. The output is an unbiased estimate of Precision@K and other ranking metrics along with confidence intervals.

### Key Designs

1. **Extension of PPI to Ranking Metrics (PRECISE-PPI)**:

    - **Function**: Enables the PPI framework to support ranking metric estimation with sub-instance-level (query-document) annotations.
    - **Mechanism**: Precision@K is expressed as a scaled dot product of predicted and annotated K-hot vectors: $\phi(\hat{y}, y) = \hat{y}^T y / K$. Since both vectors are sparse (at most $K$ ones), the output space can be compressed from $Y=\{0,1\}^{|C|}$ to $Y=\{0,1\}^K$. For each query's $K$ documents, the LLM probability $\tilde{p}'(d_k) = M(d_k|x)$ is used to construct a probability distribution over K-dimensional binary vectors (Eq. 3), after which the standard PPI++ estimator yields an unbiased estimate.
    - **Design Motivation**: The original PPI requires enumerating $2^{|C|}$ possibilities (where $|C|$ is on the order of millions), which is entirely infeasible. The sparsity observation reduces computation to $2^K$ (with $K \leq 10$ in typical settings), making the method practically applicable.

2. **Isotonic Calibration**:

    - **Function**: Calibrates LLM probability scores prior to applying PPI.
    - **Mechanism**: Isotonic regression fitted on the gold annotation set maps LLM uncertainty scores (e.g., "Probably" → 0.8, "Almost Certain" → 0.95) to more accurate probability values.
    - **Design Motivation**: Raw LLM probability scores are poorly calibrated; calibration further reduces the variance of PPI estimates.

3. **$\lambda$ Hyperparameter Optimization**:

    - **Function**: Controls the relative weight between LLM and human annotations, $0 \leq \lambda \leq 1$.
    - **Mechanism**: The PPI++ estimator $\hat{\mu}_{PPI++} = \lambda[\frac{1}{N}\sum \tilde{\mu}_u^{(i)}] + \frac{1}{n}\sum[\phi(f(x_g^{(i)}), y_g^{(i)}) - \lambda \cdot \tilde{\mu}_g^{(i)}]$ is unbiased for any $\lambda > 0$, but the variance varies with $\lambda$. In practice, $\lambda = 0.95$ yields strong performance.
    - **Design Motivation**: A high $\lambda$ fully leverages the low-variance advantage of LLM annotations, while human annotations are used solely to correct bias.

## Key Experimental Results

### Main Results (ESCI Dataset, Ground-Truth Precision@4 = 89.73%)

| Estimator | Unlabeled Data Size | Bias↓ | Std Error↓ |
|-----------|-------------------|-------|-----------|
| Gold Only ($n=30$) | — | 1.04 | 4.45 |
| PRECISE+Sonnet | 3,000 (100×) | **0.52** | **3.67** |
| PRECISE+Haiku | 3,000 (100×) | **0.42** | **4.10** |
| PRECISE+Sonnet | 60,000 (2,000×) | 0.82 | 4.45 |

### Application Scenario Validation (Indian E-Commerce Query Rewriting)

| Estimator | Production Search P@4 | Rewrite V1 P@4 | Rewrite V2 P@4 |
|-----------|----------------------|---------------|---------------|
| Gold ($n=100$) | 61.10% | 65.70% | 63.60% |
| Sonnet-LLM Only | 74.90% | 77.40% | 77.60% |
| **PRECISE-PPI** | **55.30%** | **60.30%** | **59.40%** |

- PRECISE correctly identifies V1 rewriting as the best option, which was subsequently validated and deployed via A/B testing.

### Key Findings
- 100× unlabeled data (3,000 queries) is sufficient to achieve near-optimal performance; further increases yield diminishing returns.
- PRECISE bias is only 0.52–0.70 points, an order of magnitude lower than pure LLM estimators (bias exceeding 15 points).
- Claude 3 Haiku achieves competitive performance at approximately 1/12 the cost of Sonnet.
- Variance is substantially lower than pure gold estimation — confidence intervals from 30 annotations + PPI are narrower than those from 100 gold annotations alone.

## Highlights & Insights
- **The combination of statistical inference and LLM evaluation** is highly practical: PPI provides theoretically guaranteed bias correction, going beyond merely "validating LLM outputs with a small labeled set" to mathematically proving unbiasedness. This framework generalizes to any LLM-as-Judge scenario.
- **Exploitation of sparsity** is both elegant and critical: the reduction from $2^{|C|}$ to $2^K$ brings the method from theoretical possibility to practical utility — a seemingly simple observation that resolves the core computability challenge.
- **A real-world deployment case** strengthens the paper's credibility: the method successfully guided production decisions in an Indian e-commerce search system, demonstrating its industrial applicability.

## Limitations & Future Work
- Validation is limited to Precision@K; extension to more complex ranking metrics such as NDCG remains to be verified (though the theoretical framework supports it).
- The method assumes that LLM judgment bias is consistent across the gold set and the unlabeled set; it may fail under severe distribution shift.
- $K$ is constrained to small values ($\leq 10$) since $2^K$ remains exponential.
- The sensitivity of results to prompt design in obtaining LLM annotation probabilities is not discussed.

## Related Work & Insights
- **vs. Pure LLM-as-Judge**: LLM judgments exhibit severe bias (overestimation exceeding 15%), whereas PRECISE fully corrects this with only 30 human annotations.
- **vs. Pure Human Annotation**: Variance is high with only 100 human annotations; PRECISE substantially reduces variance by incorporating LLM annotations.

## Rating
- Novelty: ⭐⭐⭐⭐ The extension of PPI to ranking metrics offers a theoretical contribution, and the sparsity observation is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both a public benchmark and a real deployment setting.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous; the application scenario is vividly described.
- Value: ⭐⭐⭐⭐⭐ Offers direct and significant practical value for reducing LLM evaluation costs.

## Additional Notes
- The methodology and experimental design of this work provide useful reference for related research areas.
- Future work could validate the generalizability and scalability of the approach across broader scenarios and larger scales.
- There is potential research value in combining this work with recent advances (e.g., intersections with RL/MCTS or multimodal methods).
- Deployment feasibility and computational efficiency should be assessed in light of specific application requirements.
- The choice of datasets and evaluation metrics may affect the generalizability of conclusions; cross-validation on additional benchmarks is recommended.

## Additional Notes
- The methodology and experimental design of this work provide useful reference for related research areas.
- Future work could validate the generalizability and scalability of the approach across broader scenarios and larger scales.
- There is potential research value in combining this work with recent advances (e.g., intersections with RL/MCTS or multimodal methods).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Positional Bias in Multimodal Embedding Models: Do They Favor the Beginning, the Middle, or the End?](positional_bias_in_multimodal_embedding_models_do_they_favor_the_beginning_the_m.md)
- [\[ICML 2026\] Ranking-Free RAG: Replacing Re-Ranking with Selection in RAG for Sensitive Domains](../../ICML2026/information_retrieval/ranking_free_rag_replacing_re-ranking_with_selection_in_rag_for_sensitive_domain.md)
- [\[AAAI 2026\] "As Eastern Powers, I Will Veto." : An Investigation of Nation-Level Bias of Large Language Models in International Relations](as_eastern_powers_i_will_veto_an_investigation_of_nation-level_bias_of_large_lan.md)
- [\[AAAI 2026\] Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-Based Machine Translation](exposing_the_cracks_vulnerabilities_of_retrieval-augmented_llm-based_machine_tra.md)
- [\[ICLR 2026\] Leveraging Data to Say No: Memory Augmented Plug-and-Play Selective Prediction](../../ICLR2026/information_retrieval/leveraging_data_to_say_no_memory_augmented_plug-and-play_selective_prediction.md)

</div>

<!-- RELATED:END -->
