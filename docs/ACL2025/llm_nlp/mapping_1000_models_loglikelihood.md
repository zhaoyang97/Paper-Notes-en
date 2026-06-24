---
title: >-
  [Paper Note] Mapping 1,000+ Language Models via the Log-Likelihood Vector
description: >-
  [ACL 2025][LLM (Other)][model mapping] This paper proposes mapping 1,000+ language models into a unified space using the log-likelihood vector, proving that the Euclidean distance between vectors approximates the KL divergence. This approach enables model clustering visualization, benchmark performance prediction ($r=0.96$), and data leakage detection.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "model mapping"
  - "log-likelihood vector"
  - "KL divergence"
  - "model clustering"
  - "benchmark prediction"
date: 2026-05-08
content_hash: 63921a6b9d9463a4
---

# Mapping 1,000+ Language Models via the Log-Likelihood Vector

**Conference**: ACL 2025  
**arXiv**: [2502.16173](https://arxiv.org/abs/2502.16173)  
**Code**: [https://github.com/shimo-lab/modelmap](https://github.com/shimo-lab/modelmap)  
**Area**: LLM / NLP  
**Keywords**: model mapping, log-likelihood vector, KL divergence, model clustering, benchmark prediction

## TL;DR
This paper proposes mapping 1,000+ language models into a unified space using the log-likelihood vector, proving that the Euclidean distance between vectors approximates the KL divergence. This approach enables model clustering visualization, benchmark performance prediction ($r=0.96$), and data leakage detection.

## Background & Motivation
**Background**: The LLM ecosystem is experiencing explosive growth, with massive model variants (base, fine-tuned, merged, etc.) on HuggingFace, yet systematic methods to compare and understand the relationships among these models are lacking.

**Limitations of Prior Work**: Existing approaches have their own limitations: classification based on model names/attributes relies on metadata, comparison based on output text lacks theoretical foundations, and leaderboard rankings are discrete and only reflect specific task dimensions.

**Key Challenge**: Language models are essentially probability distributions and should be analyzed using the geometric structure of probability distributions rather than relying on arbitrary metrics.

**Goal**: How to systematically compare 1,000+ models at scale using a theory-driven approach, revealing their relationships and structures?

**Key Insight**: Leveraging concepts from information geometry, the log-likelihood values of models on a fixed text set are used as coordinates, proving that this representation approximates the KL divergence.

**Core Idea**: Each language model is represented by its log-likelihood vector across 10,000 text segments, where the Euclidean distance between vectors approximates the KL divergence, thereby constructing a "model map."

## Method

### Overall Architecture
Given $K$ models and $N$ text segments, a $K \times N$ log-likelihood matrix $\mathbf{L}$ is computed. After double centering, the coordinate vector $\mathbf{q}_i \in \mathbb{R}^N$ is obtained for each model. In this space, t-SNE visualization, clustering analysis, and regression for benchmark score prediction are performed.

### Key Designs

1. **Log-Likelihood Vector Representation**:

    - **Function**: For each model $p_i$ and each text segment $x_s$, calculate $\ell_i(x_s) = \sum_{t=1}^n \log p_i(y_t | y^{t-1})$ to construct the vector $\boldsymbol{\ell}_i \in \mathbb{R}^N$.
    - **Mechanism**: This is the negative of the cross-entropy loss and requires no extra computation—it is naturally yielded during model training or evaluation.
    - **Design Motivation**: Log-likelihood is the most fundamental quantity of probabilistic models, directly reflecting how well a model models each text segment.

2. **Double Centering**:

    - **Function**: First perform row centering (subtract the mean log-likelihood $\bar{\ell}_i$ of each model to eliminate differences in overall model capabilities), and then perform column centering (subtract the mean across models for each text segment to eliminate differences in text difficulty).
    - **Mechanism**: $\xi_{is} = \ell_i(x_s) - \bar{\ell}_i$, and then $\mathbf{q}_i = \boldsymbol{\xi}_i - \bar{\boldsymbol{\xi}}$.
    - **Design Motivation**: Row centering eliminates details of perplexity differences caused by model scale (otherwise large models would cluster together), while column centering removes the inherent difficulty differences of the texts.

3. **KL Divergence Approximation (Core Theory)**:

    - **Function**: Prove that under the assumption that the models approximate the true distribution, $2 \text{KL}(p_i, p_j) \approx \text{Var}_{x \sim p_0}(\ell_i(x) - \ell_j(x))$.
    - **Mechanism**: The data-driven estimation is $2 \text{KL}(p_i, p_j) \approx \|\mathbf{q}_i - \mathbf{q}_j\|^2 / N$.
    - **Design Motivation**: Convert distribution differences into Euclidean distances in vector space, making large-scale model comparison efficient and theoretically grounded.

### Application Scenarios
- **Visualization**: t-SNE dimensionality reduction to paint the "model map", where models from the same family naturally cluster.
- **Performance Prediction**: Use $\mathbf{q}_i$ for Ridge regression to predict benchmark scores.
- **Data Leakage Detection**: Compare normalized average log-likelihood with benchmark scores; anomalously high ones may indicate data leakage.

## Key Experimental Results

### Benchmark Performance Prediction (Ridge Regression)

| Benchmark | Pearson's r | Spearman's ρ |
|-----------|-------------|--------------|
| ARC | 0.946 | 0.948 |
| HellaSwag | 0.909 | 0.956 |
| MMLU | 0.932 | 0.934 |
| TruthfulQA | 0.901 | 0.884 |
| Winogrande | 0.941 | 0.948 |
| GSM8K | 0.884 | 0.857 |
| **6-TaskMean** | **0.953** | **0.960** |

### Comparison: Prediction Directly Using Average Log-Likelihood (Perplexity)

| Benchmark | Pearson's r | Spearman's ρ |
|-----------|-------------|--------------|
| ARC | 0.453 | 0.432 |
| MMLU | 0.346 | 0.422 |
| TruthfulQA | 0.072 | 0.048 |
| **6-TaskMean** | **0.395** | **0.400** |

### Key Findings
- Utilizing log-likelihood vectors of 10,000 texts can predict the average score across 6 benchmarks with an accuracy of $r=0.96$—vastly outperforming simple perplexity ($r=0.40$).
- Reaching a cumulative explained variance of 90% requires only 42 dimensions, and 95% requires only 82 dimensions, demonstrating that the effective dimensionality of differences between models is very low.
- Models of the same family (Llama-2, Mistral, Gemma, etc.) cluster tightly on the map.
- Code-specialized models possess unique features along the GitHub/StackExchange dimensions.
- Weight interpolation experiments confirm that linear interpolation in weight space maintains a linear structure in log-likelihood space as well.
- Theoretical validation: token-level KL approximation correlation is $r=0.893$, and text-level correlation is $r=0.904$.
- The computation across 1,018 models requires only ~10 minutes on a single GPU.

## Highlights & Insights
- **Probability Theory-Driven Model Analysis**: Unlike empirical leaderboard comparisons, this work approaches from the perspective of information geometry to provide theoretical guarantees for "model distance" ($\approx$ KL divergence), which is its most elegant aspect.
- **The Design of Double Centering is Crucial**: It eliminates two confounding factors—model scale and text difficulty—without which all analyses would be dominated by these two factors.
- **Data Leakage Detection is an Unexpected and Valuable Application**: Models pre-trained on the Pile display anomalously high log-likelihoods mismatching their benchmark scores, which can be leveraged for screening.
- **High Computational Efficiency**: It avoids generating text or running benchmarks; a single forward pass yields coordinates of a model in the entire space.

## Limitations & Future Work
- Reliance on the choice of the reference text set: different text sets might yield different model maps.
- The premise of the KL divergence approximation (that models approximate the true distribution) does not entirely hold in practice.
- Only models with 1-13B parameters were tested; the performance of larger models or novel architectures (such as MoEs) remains unknown.
- Unable to directly compare models using different tokenizers (requiring token-level approximations).

## Related Work & Insights
- **vs Open LLM Leaderboard**: Leaderboards only offer discrete rankings, whereas this work provides positions and distances in a continuous space, delivering substantially richer information.
- **vs Activation Space Comparison**: Prior model comparisons relied on internal activations and required white-box access; this method only requires log-likelihoods, enabling black-box applicability.
- This methodology can be extended to: selecting the most suitable model for a specific task (by locating the nearest neighbor on the map) and detecting whether a model is a fine-tuned version of a certain base model.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First large-scale theory-driven mapping of model space
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 1,018 models + theoretical validation + demonstration of multiple applications
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivation, but highly dense content
- Value: ⭐⭐⭐⭐⭐ Provides a foundational tool for understanding the LLM ecosystem

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding](enhancing_input-label_mapping_in_in-context_learning_with_contrastive_decoding.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](../../ACL2026/llm_nlp/decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[NeurIPS 2025\] Writing in Symbiosis: Mapping Human Creative Agency in the AI Era](../../NeurIPS2025/llm_nlp/writing_in_symbiosis_mapping_human_creative_agency_in_the_ai_era.md)
- [\[ACL 2025\] On Entity Identification in Language Models](on_entity_identification_in_language_models.md)
- [\[ACL 2025\] Quantifying Semantic Emergence in Language Models](quantifying_semantic_emergence_in_language_models.md)

</div>

<!-- RELATED:END -->
