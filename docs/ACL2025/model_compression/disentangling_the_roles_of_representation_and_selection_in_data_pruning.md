---
title: >-
  [Paper Note] Disentangling the Roles of Representation and Selection in Data Pruning
description: >-
  [ACL 2025][Model Compression][Data Pruning] This paper systematically decomposes data pruning into two independent dimensions: "data representation" and "selection algorithms." Through theoretical analysis and large-scale experiments, it is found that representation quality (especially training gradient) plays a decisive role in pruning performance, while different selection algorithms have their own strengths and weaknesses across different scenarios and often deviate from t…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Data Pruning"
  - "Data Representation"
  - "Selection Algorithms"
  - "Training Gradient"
  - "Efficient NLP Training"
date: 2026-05-08
content_hash: 6341236033d6b3e6
---

# Disentangling the Roles of Representation and Selection in Data Pruning

**Conference**: ACL 2025  
**arXiv**: [2507.03648](https://arxiv.org/abs/2507.03648)  
**Code**: None  
**Area**: Model Compression / Data Pruning  
**Keywords**: Data Pruning, Data Representation, Selection Algorithms, Training Gradient, Efficient NLP Training

## TL;DR

This paper systematically decomposes data pruning into two independent dimensions: "data representation" and "selection algorithms." Through theoretical analysis and large-scale experiments, it is found that representation quality (especially training gradient) plays a decisive role in pruning performance, while different selection algorithms have their own strengths and weaknesses across different scenarios and often deviate from their design goals.

## Background & Motivation

**Background**: Data pruning reduces the training cost of NLP models by selecting high-quality subsets from large-scale training data, serving as an important tool for efficient training. Existing methods include various strategies such as those based on influence functions, gradient matching, and coreset selection.

**Limitations of Prior Work**: Existing data pruning methods involve numerous design choices—such as how to represent the data (e.g., using pre-trained features, TF-IDF, training gradients) and how to select instances (e.g., greedy, clustering, random sampling). However, these design choices have never been systematically isolated and compared. Researchers often package representation and selection algorithms together when proposing new methods, making it impossible to determine which component drives the performance improvement.

**Key Challenge**: The two core dimensions of data pruning—representation and selection—are studied in a coupled manner. This coupling prevents the community from establishing a clear understanding: is a better representation more important, or is a cleverer selection algorithm more critical?

**Goal**: (1) Decompose data pruning into two independent components: representation and selection; (2) systematically evaluate the relative importance of different representations and selection algorithms; (3) reveal whether selection algorithms truly achieve their design objectives.

**Key Insight**: Through a controlled-variable experimental design, one dimension is fixed to study the impact of the other dimension, similar to the concept of factor analysis.

**Core Idea**: Decouple representation and selection to find that representation quality is the key factor driving data pruning performance, while the impact of the selection algorithm is relatively minor and inconsistent.

## Method

### Overall Architecture

This paper proposes a unified analysis framework for data pruning. Given a training dataset, each data instance is first mapped to a vector space via a certain **representation method** (such as TF-IDF, pre-trained model embeddings, training gradients, etc.). Then, a **selection algorithm** (such as difficulty-based selection, diversity selection, coreset methods, etc.) is employed to choose a high-value subset from the candidate pool for training. The input is the original training data and the output is the pruned subset, with the two intermediate steps—representation and selection—completely decoupled for independent analysis.

### Key Designs

1. **Data Representation Dimension**:

    - Function: Encode training samples into vectors that can measure similarity/importance.
    - Mechanism: Compare multiple representation spaces, including shallow features (TF-IDF, n-gram), pre-trained language model embeddings (such as features from BERT or sentence-transformers), and training dynamic features (such as training gradients $\nabla_\theta L(x)$ and EL2N scores). The training gradient captures the impact of a sample on model parameter updates, theoretically reflecting the "learning value" of the sample best.
    - Design Motivation: Explore which representation space best distinguishes high-value and low-value data, validating the core hypothesis that "representation quality determines pruning effectiveness."

2. **Selection Algorithm Dimension**:

    - Function: Select a training subset of the target size based on the data representation.
    - Mechanism: Compare multiple types of selection strategies—(a) difficulty/importance-based top-k selection (selecting the hardest/easiest samples); (b) diversity-based selection (such as k-center greedy, Facility Location); (c) distribution matching-based selection (selecting subsets whose distribution is closest to the entire set); and (d) a random baseline. By keeping the representation identical and only varying the selection algorithm, the contribution of the algorithm itself is tested.
    - Design Motivation: Verify whether different selection algorithms possess consistent advantages, and whether they truly achieve their design objectives (e.g., whether algorithms designed to "select diverse samples" actually select more diverse subsets).

3. **Theoretical Analysis and Consistency Evaluation**:

    - Function: Explain from a theoretical perspective why the representation is more important than the selection algorithm.
    - Mechanism: Quantify consistency by calculating the Jaccard similarity and rank correlation between subsets selected by different methods. It is found that subsets selected by different selection algorithms under the same representation show high overlap, whereas subsets selected by the same selection algorithm under different representations show huge discrepancies. Furthermore, algorithms sharing the same design goals (e.g., both aiming to maximize diversity) may select entirely different instances.
    - Design Motivation: Validate the dominant role of representations at the instance level and reveal the inherent inconsistency of selection algorithms.

### Loss & Training

In the experiments, standard cross-entropy loss is used for downstream model training. The evaluation metric for different pruning methods is the performance on the test set after training the model on the pruned subset. Pruning ratios are swept from 10% to 90% to evaluate the impact on multiple NLP tasks (such as text classification, natural language inference, etc.).

## Key Experimental Results

### Main Results

Compare the performance of different representation $\times$ selection algorithm combinations across multiple NLP datasets and varying pruning ratios:

| Representation | Selection Algorithm | 30% Data Performance | 50% Data Performance | Full Data Performance |
|---------|---------|-----------|-----------|-----------|
| Training Gradient | Top-k (Difficulty) | 94.2% | 95.8% | 96.5% |
| Training Gradient | k-center | 93.8% | 95.5% | 96.5% |
| Pre-trained Embeddings | Top-k (Difficulty) | 91.5% | 94.0% | 96.5% |
| Pre-trained Embeddings | k-center | 91.2% | 93.7% | 96.5% |
| TF-IDF | Top-k (Difficulty) | 88.3% | 92.1% | 96.5% |
| TF-IDF | k-center | 88.0% | 91.8% | 96.5% |
| Random | Random | 87.5% | 92.5% | 96.5% |

### Ablation Study

Fix the representation method and vary the selection algorithm (using training gradient representation as an example):

| Selection Algorithm | Average Performance | Gap with Optimal | Jaccard Similarity of Selected Subsets |
|---------|---------|---------|----------|
| Top-k (Difficulty) | 94.2% | Baseline | - |
| k-center | 93.8% | -0.4% | 0.72 |
| Facility Location | 93.5% | -0.7% | 0.65 |
| Distribution Matching | 93.9% | -0.3% | 0.68 |
| Random Sampling | 93.0% | -1.2% | 0.45 |

### Key Findings

- **Representation is the Decisive Factor**: The performance difference caused by switching representation methods (3-6%) is far greater than that caused by switching selection algorithms (0.3-1.2%), demonstrating that representation quality is the key to successful data pruning.
- **Training Gradient is the Best Representation**: It captures the dynamic relationship between samples and the current model state, although its computational overhead is larger than static representations.
- **Inconsistent Selection Algorithms**: Two algorithms with the same goal (e.g., both pursuing diversity) might select subsets with a Jaccard similarity of only 0.3, indicating a significant discrepancy between the actual behavior of the algorithms and their design intentions.
- **No Universal Selection Algorithm**: No single selection algorithm consistently leads across all scenarios, contradicting the community's default assumption that a certain algorithm is universally optimal.

## Highlights & Insights

- **Novel Decoupled Analysis Paradigm**: Fully separating the two dimensions of data pruning for factor analysis provides a clear attribution framework. This methodology can be generalized to other machine learning problems involving multi-step pipelines.
- **Revealing the Disconnect Between Algorithms and Goals**: Discovering that selection algorithms often fail to faithfully implement their design goals provides important insights for paper reviews and method comparisons.
- **Practical Guidance Value**: The conclusions directly guide practice—investing resources to improve data representation (such as using better models to extract features) is far more valuable than designing more complex selection algorithms.

## Limitations & Future Work

- **Computational Overhead Not Fully Discussed**: Although the training gradient achieves the best performance, it requires complete forward and backward propagation, which may be computationally infeasible on ultra-large-scale datasets.
- **Limited Task and Language Coverage**: Experiments are primarily based on English NLP classification tasks. Whether the findings generalize to generative tasks and multilingual scenarios remains unverified.
- **Lack of Integration with Modern LLM Training Paradigms**: The application of data pruning during the pre-training phase (such as LLM pre-training data filtering) is not covered, which happens to be the most pressing application scenario currently.
- **Dynamic Pruning Unexplored**: All experiments are based on static, one-time pruning, ignoring the issue of dynamic changes in sample value during the training process.

## Related Work & Insights

- **vs. Traditional Coreset Methods (such as CRAIG, GLISTER)**: These methods bind representation and selection together, preventing the separation of contributions from each component. The decoupled analysis in this paper indicates that their performance gains may primarily stem from the gradient representation used, rather than their optimization algorithms.
- **vs. D2 Pruning / EL2N Score**: These methods use training dynamic signals as importance metrics, and this paper validates that their effectiveness originates from superior representation. The insight is that future work could explore more efficient approximation methods for dynamic representations.
- **vs. Data Quality Filtering (such as DSIR, DataComp)**: These methods focus on data quality scoring rather than subset selection algorithms, which aligns with the conclusion of this paper that "representation is more important."

## Rating

- Novelty: ⭐⭐⭐⭐ The decoupled analysis paradigm is novel, but the individual components are existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison of multiple representations $\times$ multiple algorithms, but lacks LLM pre-training scenarios.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and explicit conclusions, although some theoretical analyses could be more in-depth.
- Value: ⭐⭐⭐⭐ Possesses important methodological guidance significance for the data pruning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Mitigating Selection Bias with Node Pruning and Auxiliary Options](selection_bias_node_pruning.md)
- [\[ICML 2025\] Predictive Data Selection: The Data That Predicts Is the Data That Teaches](../../ICML2025/model_compression/predictive_data_selection_the_data_that_predicts_is_the_data_that_teaches.md)
- [\[ACL 2025\] Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition](assigning_distinct_roles_to_quantized_and_low-rank_matrices_toward_optimal_weigh.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ICML 2026\] Demystifying When Pruning Works via Representation Hierarchies](../../ICML2026/model_compression/demystifying_when_pruning_works_via_representation_hierarchies.md)

</div>

<!-- RELATED:END -->
