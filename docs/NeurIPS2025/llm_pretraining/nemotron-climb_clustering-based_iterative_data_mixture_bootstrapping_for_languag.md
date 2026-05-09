---
title: >-
  [Paper Note] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training
description: >-
  [NeurIPS 2025][LLM Pretraining][data mixture] NVIDIA proposes the CLIMB framework, which automatically discovers optimal pre-training data mixture ratios through embedding-based clustering and iterative bootstrapped search. On a 1B-scale model, CLIMB outperforms Llama-3.2-1B by 2.0%, and releases the 1.2T-token ClimbLab corpus and the 400B-token ClimbMix high-quality dataset.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - data mixture
  - pre-training
  - clustering
  - iterative optimization
  - LLM
date: 2026-05-08
content_hash: 0ef07a1f2931f79f
---

# Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training

**Conference**: NeurIPS 2025
**arXiv**: [2504.13161](https://arxiv.org/abs/2504.13161)
**Code**: [Data available](https://huggingface.co/nvidia/Nemotron-ClimbMix)
**Area**: LLM Pre-training
**Keywords**: data mixture, pre-training, clustering, iterative optimization, LLM

## TL;DR

NVIDIA proposes the CLIMB framework, which automatically discovers optimal pre-training data mixture ratios through embedding-based clustering and iterative bootstrapped search. On a 1B-scale model, CLIMB outperforms Llama-3.2-1B by 2.0%, and releases the 1.2T-token ClimbLab corpus and the 400B-token ClimbMix high-quality dataset.

## Background & Motivation

Large-scale pre-training datasets (e.g., Common Crawl) lack explicit domain labels, while manual annotation (e.g., The Pile) is extremely time-consuming. Even when domain labels are available, the complex nonlinear relationship between data mixture ratios and model performance makes optimal mixing an open problem. Core challenges facing existing methods include:

**Lack of domain partitioning**: Web-scale data does not directly provide domain information, and traditional perplexity- or educational-value-based filtering is insufficiently precise.

**Complexity of mixture strategies**: Improving coding ability requires not only code data but also complementary knowledge in mathematics, reasoning, safety, and other areas.

**High search cost**: Exhaustively training full models under different mixture ratios is computationally infeasible.

Prior work such as DoReMi and RegMix has explored data mixture optimization, but the former relies on predefined domains while the latter performs only a single-pass search. The core innovation of CLIMB lies in **automatically discovering domains** and **iteratively optimizing mixture weights**.

## Method

### Overall Architecture

The CLIMB framework consists of two stages: **data preprocessing** (embedding + clustering) and **iterative bootstrapped search** (mixture weight optimization).

**Stage 1: Data Preprocessing**

1. **Text Embedding**: Documents $\hat{D} = \{D_1, D_2, \dots, D_n\}$ are mapped to an embedding space using the stella_en_400M_v5 model, yielding vector set $E = \{E_1, E_2, \dots, E_n\}$.
2. **Embedding Clustering**: The K-means algorithm in FAISS clusters embeddings into $K_{\text{init}} = 1000$ initial clusters.
3. **Cluster Merging**: A fasttext model first evaluates data quality (overall quality, educational value, informational value, and advertisement score); clusters are pruned to $K_{\text{pruned}} = 240$ using a threshold of 3.0, then merged into $K_{\text{enhanced}}$ super-clusters based on a Euclidean distance threshold of 1.5.

**Stage 2: Iterative Bootstrapped Search**

### Key Designs

The core idea formulates data mixture as a **bi-level optimization problem**:

$$\min_{\alpha \in A} \ell_{val}(\alpha, \omega^*(\alpha)) \quad \text{s.t.} \quad \omega^*(\alpha) = \arg\min_{\omega} \ell_{train}(\alpha, \omega), \quad \sum_{i=1}^{k} \alpha_i = 1, \alpha_i \geq 0$$

Direct solving requires training a full model for each $\alpha$, which is computationally prohibitive. CLIMB employs a **predictor** $f_\theta(\alpha)$ to approximate the objective $\ell(\alpha, \omega)$, reformulating the problem as:

$$\min_{\alpha \in A} f(\alpha | S) \quad \text{s.t.} \quad f = \arg\min_{S, f \in \tilde{\mathcal{F}}} \sum_{s \in S} \mathcal{L}(f(s), \ell(s, w^*))$$

Unlike the single-pass search in RegMix, CLIMB alternately optimizes the sampling strategy $S$ and predictor $f_\theta$ via **coordinate descent**:

- **Subroutine 1 (Configuration Sampling)**: At iteration $k+1$, the predictor $f_k$ ranks all untested configurations, and $M$ new configurations are randomly sampled from the top-$N$ candidates, balancing exploitation and exploration.
- **Subroutine 2 (Weak Predictor Fitting)**: A LightGBM regression model trains predictor $f_{\theta}^{k+1}$ on the accumulated sample set $S^{k+1}$.

The iterative search proceeds over three rounds, evaluating 64, 32, and 16 candidate configurations respectively (112 total searches), with Dirichlet distribution used for initialization sampling. LightGBM is regularized with L1/L2 penalties, maximum depth 4, minimum 5 leaf samples, and early stopping after 20 rounds to prevent overfitting.

### Loss & Training

- **Base Model Training**: Phase-1 pre-training is conducted on 10T tokens (DCLM + TxT360) using a WSD learning rate schedule.
- **Proxy Models**: 62M and 350M models are used for efficient search; the 350M proxy is used in the main experiments.
- **Target Model Evaluation**: PIQA, ARC_E, and HellaSwag validation sets serve as the optimization objective, with generalization assessed on test sets.

## Key Experimental Results

### Main Results

| Model Scale | Method | piqa | arc_c | arc_e | hellaswag | winogrande | siqa | Avg. |
|-------------|--------|------|-------|-------|-----------|------------|------|------|
| 350M | Random | 71.16 | 30.54 | 62.50 | 52.14 | 55.40 | 41.29 | 52.17 |
| 350M | DoReMi | 70.29 | 33.53 | 66.41 | 52.25 | 55.95 | 41.86 | 53.38 |
| 350M | RegMix | 71.92 | 33.42 | 66.12 | 53.69 | 55.27 | 42.23 | 53.78 |
| 350M | **CLIMB** | **72.21** | **34.87** | **67.25** | **55.32** | **56.79** | **42.54** | **54.83** |
| 1B | Random | 74.05 | 37.12 | 70.24 | 62.90 | 60.77 | 42.48 | 57.93 |
| 1B | DoReMi | 74.91 | 40.01 | 72.34 | 63.53 | 61.08 | 43.09 | 59.16 |
| 1B | RegMix | 75.22 | 40.42 | 71.32 | 64.73 | 62.33 | 42.22 | 59.37 |
| 1B | **CLIMB** | **75.78** | **40.98** | **72.97** | **66.01** | **63.32** | **43.37** | **60.41** |

Trained on 400B tokens, the 950M CLIMB model achieves an average score of 53.54%, surpassing Llama-3.2-1B (51.56%) by 2.0%.

### Ablation Study

| Ablation Dimension | Configuration | Avg. Accuracy |
|--------------------|---------------|---------------|
| Search compute 100% | 64:32:16 | 60.41 |
| Search compute 150% | — | 60.72 |
| Search compute 200% | — | 61.12 |
| Compute allocation 6:1 | 2 rounds | 60.05 |
| Compute allocation 4:2:1 | 3 rounds | **60.41** |
| Compute allocation 2:2:1:1 | 4 rounds | 60.14 |
| Proxy model 62M | — | 60.11 |
| Proxy model 132M | — | 60.19 |
| Proxy model 350M | — | **60.41** |
| Init: Random | — | 60.21 |
| Init: Dirichlet | — | **60.41** |

### Key Findings

1. **Iterative search substantially outperforms single-pass search**: Each CLIMB iteration yields consistent gains; Social Sciences accuracy improves from 40.18% to 41.79% across iter1→iter3.
2. **Domain specialization is highly effective**: Optimizing for Social Sciences yields up to 5% improvement over random sampling.
3. **Proxy model scale has limited impact**: The 62M proxy underperforms the 350M proxy by only 0.3%, demonstrating that small proxies suffice for efficient search.
4. **Compute allocation balance is critical**: Three-round iteration with a 4:2:1 ratio achieves the optimal depth–breadth trade-off.
5. **Training from scratch requires more balanced mixtures**: Unlike continual training, training from scratch demands greater diversity in cluster coverage.

## Highlights & Insights

- **End-to-end automation**: The framework requires no manual domain labels and is fully automated from embedding clustering to mixture search.
- **Weak predictor + iterative refinement**: Drawing inspiration from boosting, multiple rounds of weak predictors progressively focus the search on high-quality regions.
- **Open-source contribution**: The release of 1.2T-token ClimbLab (20 semantic clusters) and 400B-token ClimbMix provides a unified experimental platform for data mixture research.
- **Practical insights**: Analysis of cluster weight distributions and topic characteristics in optimal mixtures reveals that general reasoning tasks primarily depend on four clusters: C8 (math/logic), C9 (tech news), C18 (education), and C19 (encyclopedic content).

## Limitations & Future Work

1. **Search space constrained by clustering granularity**: 21 super-clusters may inadequately capture fine-grained domain distinctions.
2. **Proxy-to-target alignment assumption**: While transfer from 350M to 1B is empirically validated, effectiveness at larger scales (7B+) remains unverified.
3. **Static mixture vs. dynamic curriculum**: CLIMB identifies a globally fixed mixture ratio and does not explore the potential of dynamically adjusting ratios during training.
4. **Evaluation coverage**: Validation is primarily conducted on reasoning benchmarks; the impact on practical scenarios such as code generation and conversational ability is not thoroughly analyzed.

## Related Work & Insights

- **RegMix** (Liu et al.): The direct predecessor of CLIMB; CLIMB extends its single-pass search into an iterative framework.
- **DoReMi** (Xie et al.): Online mixture optimization based on DRO, relying on predefined domains.
- **WebOrganizer** (Wettig et al.): A concurrent work that annotates web data using classifiers, whereas CLIMB employs clustering more directly.
- Insights: The paradigm of data quality > data quantity continues to gain traction; automated data curation pipelines represent an important research direction for LLM pre-training.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of iterative bootstrapping and embedding clustering is novel, though individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Ablations are highly comprehensive (compute budget, allocation, proxy scale, clustering, initialization), with multi-scale validation from 350M to 1B.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with complete mathematical derivations and intuitive figures.
- Value: ⭐⭐⭐⭐⭐ The release of high-quality open-source datasets combined with a practical automated mixture framework is of great value to the LLM pre-training community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](lm_behavioral_phases.md)
- [\[NeurIPS 2025\] Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training](through_the_river_understanding_the_benefit_of_schedule-free_methods_for_languag.md)
- [\[NeurIPS 2025\] Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods](final-model-only_data_attribution_with_a_unifying_view_of_gradient-based_methods.md)
- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](enhancing_training_data_attribution_with_representational_optimization.md)
- [\[NeurIPS 2025\] Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization](broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)

<!-- RELATED:END -->
