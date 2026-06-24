---
title: >-
  [Paper Note] Reranking-based Generation for Unbiased Perspective Summarization
description: >-
  [ACL 2025][Information Retrieval & RAG][Perspective summarization] Constructs a controlled test suite for the political perspective summarization task to evaluate the reliability of existing evaluation metrics, finding that LLM-based metrics significantly outperform traditional ones. It also demonstrates that reranking-based methods and DPO training on reranked data can substantially improve both the coverage and faithfulness of perspective summaries.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Perspective summarization"
  - "reranking"
  - "DPO"
  - "LLM evaluation metrics"
  - "unbiased summarization"
date: 2026-05-08
content_hash: 939193aabb11ccf7
---

# Reranking-based Generation for Unbiased Perspective Summarization

**Conference**: ACL 2025  
**arXiv**: [2506.15925](https://arxiv.org/abs/2506.15925)  
**Code**: [narutatsuri/Unbiased-Perspective-Summarization](https://github.com/narutatsuri/Unbiased-Perspective-Summarization)  
**Area**: Information Retrieval  
**Keywords**: Perspective summarization, reranking, DPO, LLM evaluation metrics, unbiased summarization  

## TL;DR

Constructs a controlled test suite for the political perspective summarization task to evaluate the reliability of existing evaluation metrics, finding that LLM-based metrics significantly outperform traditional ones. It also demonstrates that reranking-based methods and DPO training on reranked data can substantially improve both the coverage and faithfulness of perspective summaries.

## Background & Motivation

- **Background**: LLMs have achieved breakthroughs in text summarization, but in summarizing opinionated articles (such as political news), models often fail to present diverse perspectives fairly due to position biases, uneven input coverage, and hallucination issues.
- **Limitations of Prior Work**: (1) Existing evaluation frameworks originate from the news summarization domain and have not been validated for their suitability in perspective summarization tasks; (2) Apart from zero-shot inference, other LLM methods (such as prompt engineering and preference training) have not been fully explored in perspective summarization.
- **Key Challenge**: A good perspective summary must simultaneously satisfy "coverage" (including all key points) and "faithfulness" (excluding opposing views/hallucinations), creating an inherent tension between these two objectives.
- **Goal**: (1) Identify reliable metrics for evaluating perspective summary quality; (2) Explore LLM methods beyond zero-shot inference to improve summary quality.
- **Key Insight**: Evaluate metrics by constructing a controlled test suite with ground-truth scores, and then use the validated, reliable metrics to guide reranking and DPO training.
- **Core Idea**: First validate metric reliability, then use reliable metrics for reranking to select the optimal summary, and finally perform DPO training using the reranked data.

## Method

### Overall Architecture

Divided into two phases: (1) **Metric Evaluation Phase** — constructing a controlled test suite to validate which metrics can reliably measure coverage and faithfulness; (2) **Method Evaluation Phase** — comparing four types of methods: Prompting, mechanical attention modification (PINE), Reranking, and DPO+Reranking.

### Key Designs

#### 1. Controlled Test Suite Construction (Evaluating Metric Reliability)
- **Function**: Create multiple summaries with varying coverage/faithfulness scores for each article.
- **Mechanism**: Annotators extract key excerpts $E_{t,\theta}$ from the article, which are rewritten by an LLM into key points $K_{t,\theta}$. Opposing key points $\bar{K}_{t,\theta}$ are generated (either from opposing perspectives or semantic inversion). Summaries are constructed by combining $k_g$ correct points selected from $K_{t,\theta}$ and $k_b$ incorrect points selected from $\bar{K}_{t,\theta}$.
    - Coverage = $k_g / |K_{t,\theta}|$
    - Faithfulness = $k_g / (k_g + k_b)$
- **Scale**: 50 documents, 5 annotators, 370 article-summary pairs.
- **Design Motivation**: Controlling the composition of selected key points allows precise knowledge of the true coverage and faithfulness scores for each summary.

#### 2. Reranking Method
- **Function**: Generate multiple candidate summaries and select the best one using surrogate metrics.
- **Mechanism**: Use the backbone model (Llama-3.1-8B-Instruct) to generate 9 candidate summaries, score them using LLM-Coverage and LLM-Faithfulness metrics, and select the highest-scoring summary.
- **Scoring Model**: Use Qwen2.5-14B-Instruct as the scorer (avoiding overlap with the model used for automatic evaluation metrics).
- **Design Motivation**: Reranking leverages the backbone model's existing capability to generate diverse, high-quality summaries without requiring additional training.

#### 3. DPO + Reranking (Preference Training)
- **Function**: Train the model using preference pairs labeled by Reranking.
- **Mechanism**: Iterate the process: model generates summaries -> scores them using surrogate metrics -> high-scoring summaries are designated as preferred, low-scoring as rejected -> train the backbone model using DPO. Run for 10 epochs on the PoliSum training set (1,716 articles).
- **Design Motivation**: Construct preference pairs using model-generated data and automatic scoring, enabling preference training without human annotation.

### Loss & Training

DPO Loss: Standard Direct Preference Optimization loss, performing preference training by using high-scoring summaries as preferred samples and low-scoring summaries as rejected samples.

## Key Experimental Results

### Main Results: Metric Reliability Evaluation

| Metric | Coverage Spearman $\rho$ | Coverage Winrate | Faithfulness Spearman $\rho$ | Faithfulness Winrate |
|------|-------------------|---------------|-------------------|---------------|
| ROUGE_L (R) | 0.473 | 0.780 | -0.038 | 0.393 |
| BERTScore (R) | 0.527 | 0.815 | -0.032 | 0.415 |
| **LLM-Coverage** | **0.707** | 0.739 | 0.393 | 0.431 |
| AlignScore | 0.261 | 0.503 | **0.650** | **0.773** |
| **LLM-Faithfulness** | 0.462 | 0.398 | **0.706** | 0.537 |

- LLM-Coverage is the best coverage metric ($\rho = 0.707$), and AlignScore is the best faithfulness metric ($\rho = 0.650$).
- Traditional metrics (ROUGE, BERTScore) are practically ineffective for measuring faithfulness.

### Ablation Study: Automatic and Human Evaluation of Different Methods

**Automatic Evaluation** (Coverage/Faithfulness):

| Method | Coverage Score | Faithfulness Score |
|------|-----------|-----------|
| Zero-Shot | Baseline | Baseline |
| Self-Refine | Minor Improvement | Slight Decrease |
| Debate | Minor Improvement | Slight Decrease |
| PINE | No Improvement | No Improvement |
| Reranking | **Significant Improvement** | **Significant Improvement** |
| DPO+RR | **Best** | **Best** |

**Human Evaluation**:

| Method | Coverage | Faithfulness |
|------|--------|--------|
| Zero-Shot | 0.347 | 0.642 |
| Reranking | 0.410 | 0.673 |
| **DPO+RR** | **0.437** | **0.724** |

- DPO+RR improves coverage by approximately **12%** and faithfulness by approximately **8%**.

### Key Findings

1. **Traditional metrics are unreliable**: ROUGE and BERTScore completely fail (or even negatively correlate) when measuring faithfulness; BLEURT and SummaC also perform poorly.
2. **Prompting methods have limited efficacy**: Multi-Agent Debate and Self-Refine only slightly improve coverage, while faithfulness actually decreases.
3. **Reranking is a strong baseline**: It substantially outperforms all inference-time methods without any training.
4. **DPO is effective on synthetic data**: Using only model-generated data consistently improves both metrics, with the most significant improvement in faithfulness.
5. **No loss in abstractiveness**: The novel 4-gram ratio (0.953) and extractive fragment density (1.415) of DPO+RR are both superior to most baselines.

## Highlights & Insights

- The paradigm of **"validating metrics before optimizing methods"** is highly valuable—optimizing directly with unvalidated metrics may lead to efforts in the wrong direction.
- Reranking is found to consistently outperform complex prompting methods (such as Self-Refine and Debate), challenging the assumption that "more complex reasoning equals better results."
- The effectiveness of DPO on self-generated reranking data indicates that the model's generation space already contains high-quality summaries; the key lies in the selection mechanism.
- The decoupled evaluation of coverage and faithfulness reveals blind spots in commonly used metrics, which also offers insights for other summarization tasks.

## Limitations & Future Work

- The size of the controlled test suite is relatively small (50 documents, 370 pairs), which may affect statistical significance.
- The findings are only validated on political perspective summarization (PoliSum); other opinion-dense domains (e.g., product reviews, medical controversies) remain untested.
- DPO training is iterated for 10 epochs on small-scale data, posing a potential risk of overfitting.
- Using a different model (Qwen2.5-14B) as the scorer might introduce cross-model bias.
- The sample size for human evaluation is small, and larger-scale validation is still required.

## Related Work & Insights

- **PoliSum**: The political perspective summarization dataset and task definition utilized in this work.
- **AlignScore / SummaC**: Factual consistency metrics, where AlignScore demonstrates outstanding performance in this task.
- **DPO (Rafailov et al., 2023)**: This work extends DPO to automatic preference learning scenarios without human annotations.
- **Self-Refine / Multi-Agent Debate**: Popular inference-time enhancement methods, which show limited effectiveness in this task.
- **Insight**: For generation quality optimization, selection (Reranking) can be more efficient than refining generation strategies (Prompting).

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: The paradigm combining metric validation and method optimization is novel; the self-training loop of DPO+Reranking is simple and effective.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Double validation via both automatic and human evaluation, with dedicated experiments for metric assessment.
- **Value** ⭐⭐⭐⭐: The reranking approach is simple and easy to use, enhancing summary quality without requiring additional training.
- **Writing Quality** ⭐⭐⭐⭐: Clear structure and precise problem definition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](gor_rag_long_context_summary.md)
- [\[ACL 2025\] Gumbel Reranking: Differentiable End-to-End Reranker Optimization](gumbel_reranking.md)
- [\[NeurIPS 2025\] AcuRank: Uncertainty-Aware Adaptive Computation for Reranking](../../NeurIPS2025/information_retrieval/acurank_uncertainty-aware_adaptive_computation_for_listwise_reranking.md)
- [\[AAAI 2026\] RRRA: Resampling and Reranking through a Retriever Adapter](../../AAAI2026/information_retrieval/rrra_resampling_and_reranking_through_a_retriever_adapter.md)
- [\[ACL 2026\] Enhancing Factuality through Consensus and Consistency in Summarization Using Minimum Bayes Risk Decoding](../../ACL2026/information_retrieval/enhancing_factuality_through_consensus_and_consistency_in_summarization_using_mi.md)

</div>

<!-- RELATED:END -->
