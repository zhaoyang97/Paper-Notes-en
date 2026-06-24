---
title: >-
  [Paper Note] Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis
description: >-
  [ACL 2025][NLP Understanding][ABSA] This paper proposes the Dynamic Order Template (DOT) method, which decomposes ABSA sentiment quadruple generation into two stages: first predicting the template size (number of quadruples) and generating the initial template, and then generating specific sentiment quadruples based on the dynamic templates. This approach achieves SOTA performance across 9 ABSA datasets while reducing inference time by 7 times compared to MvP.
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "ABSA"
  - "Sentiment Quadruple"
  - "Dynamic Template"
  - "Multi-View Prompting"
  - "Generative Sentiment Analysis"
date: 2026-05-08
content_hash: 84be9cca10bc30c4
---

# Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis

**Conference**: ACL 2025  
**arXiv**: [2406.11130](https://arxiv.org/abs/2406.11130)  
**Code**: [GitHub](https://github.com/imsongpasimin/DOT)  
**Area**: NLP Understanding / Sentiment Analysis  
**Keywords**: ABSA, Sentiment Quadruple, Dynamic Template, Multi-View Prompting, Generative Sentiment Analysis

## TL;DR

This paper proposes the Dynamic Order Template (DOT) method, which decomposes ABSA sentiment quadruple generation into two stages: first predicting the template size (number of quadruples) and generating the initial template, and then generating specific sentiment quadruples based on the dynamic templates. This approach achieves SOTA performance across 9 ABSA datasets while reducing inference time by 7 times compared to MvP.

## Background & Motivation

**Background**: Aspect-Based Sentiment Analysis (ABSA) requires extracting (Aspect, Category, Opinion, Sentiment) quadruples from text. Mainstream methods utilize generative models (such as T5) to format quadruples into target sequences based on fixed templates for generation.

**Limitations of Prior Work**: Static, single-view templates (e.g., "C is S because A is O") fail to capture all directional dependencies among elements due to the autoregressive generation process—elements positioned later cannot influence the prediction of preceding elements. MvP (Multi-View Prompting) alleviates this issue by enumerating multiple element permutations and using ensemble voting, but it introduces two severe drawbacks: (1) low efficiency—even for simple samples that only require a single view to be predicted correctly, all 15 views must be generated, causing a surge in inference time; (2) poor transferability—the number of views $k$ acts as a hyperparameter whose optimal value varies across datasets, requiring manual adjustment during cross-domain transfer.

**Key Challenge**: Although multi-view ensemble improves accuracy, its fixed-view strategy leads to an unnecessary trade-off between efficiency and accuracy—a large number of simple samples waste computational resources on multi-view processing.

**Goal**: How can the model adaptively determine the required number of views based on the complexity of each instance, applying multi-view prompting only to complex samples while retaining the accuracy benefits of the ensemble?

**Key Insight**: The authors observe that the count of sentiment quadruples itself is a predictable signal—an instance with $N$ quadruples requires $N$ views (with each view responsible for generating one quadruple). They decouple the tasks of "predicting the number of quadruples" and "generating the quadruple content" into two sub-tasks, handled by dedicated models.

**Core Idea**: To implement dynamic view selection using a two-stage T5 model: the first stage predicts the quadruple count and generates initial templates, while the second stage generates sentiment quadruples based on the selected views.

## Method

### Overall Architecture

The input is a review text, and the output is all sentiment quadruples (A, C, S, O) contained in the text. The method consists of two stages: Stage 1 uses a T5 model, taking the raw text as input to predict the initial order template (spanning $K_i$ views, where $K_i$ corresponds to the number of quadruples for the instance). Stage 2 uses another T5 model, taking the raw text combined with the final order template as a prompt to generate the specific sentiment quadruples for each view.

### Key Designs

1. **Instance-level Entropy-based View Ranking and Sampling**:

    - **Function**: Determine the optimal permutation of elements for each instance.
    - **Mechanism**: For all (A, C, S, O) permutations, a vanilla T5 is used to compute the conditional generation entropy $\mathcal{E}_{i,v} = -\sum P(v|x_i) \log P(v|x_i)$ of each view $v$ on instance $x_i$. Lower entropy indicates a more natural permutation that is easier for the model to generate. Views are sorted in ascending order of entropy, and the top-$K_i$ views are selected.
    - **Design Motivation**: Compared to MvP's uniform dataset-level ranking, instance-level ranking customizes the generation order for each input. Ablation studies show that random view sampling drops F1 by 1.8%.

2. **Two-stage Decoupling: Separating Quadruple Count Prediction and Quadruple Generation**:

    - **Function**: Decompose the complex quadruple generation task into two simpler sub-tasks.
    - **Mechanism**: Stage 1 only predicts the view templates for three elements (A, C, S) (excluding O), with T5 generating $y_i^{(1)} = P_{i,1}^{(1)} [\text{SSEP}] P_{i,2}^{(1)} [\text{SSEP}] \ldots P_{i,K_i}^{(1)}$; during inference, the predicted count $\hat{K}$ is determined by the number of `[SSEP]` tokens. Stage 2 re-ranks the views using the complete four elements (A, C, S, O), appends the top-$\hat{K}$ views as a prompt to the end of the input sentence, and generates the corresponding quadruple content.
    - **Design Motivation**: Experiments reveal that excluding the O element in Stage 1 yields the best performance because opinion generation is the most challenging part. If the model attempts to learn O during Stage 1, its focus during Stage 2 becomes dispersed.

3. **Stopword Filtering and Constrained Decoding**:

    - **Function**: Clean up inconsistent stopword labels in the dataset to ensure correct target generation formats.
    - **Mechanism**: Inconsistent use of stopwords in dataset annotations (e.g., negation words are sometimes included and sometimes not) can cause model predictions to be penalized due to mismatching stopwords. An NLTK-based stopword list is used to filter both the generated outputs and target labels. During inference, a constrained decoding strategy is applied to ensure compliant output format.
    - **Design Motivation**: Ablation shows that stopword filtering gains 1.02 F1 points, serving as a high-reward, low-cost engineering heuristic.

### Loss & Training

Both stages utilize standard autoregressive cross-entropy loss, with Stage 1 trained for 30 epochs and Stage 2 for 40 epochs. The model trained in Stage 1 is directly used to initialize the Stage 2 model (similar to warm-starting), providing a regularization effect on the ABSA datasets. The AdamW optimizer is employed with a learning rate of 1e-4, a training batch size of 16, and an inference batch size of 24.

## Key Experimental Results

### Main Results: F1 Comparison Across 9 ABSA Datasets

| Method | R15 | R16 | Lap | Rest | M-Rest | Avg F1 | Inference Time (s) |
|------|-----|-----|-----|------|--------|--------|-----------|
| Paraphrase | 46.93 | 57.93 | 43.51 | 61.16 | 57.38 | 50.34 | 40.63 |
| MvP | 51.04 | 60.39 | 43.92 | 61.54 | 58.12 | 51.76 | **2161.81** |
| DOT (Ours) | **51.91** | **61.24** | **44.92** | 59.25 | **58.25** | **52.28** | 298.17 |
| GPT-4o | 40.45 | 47.29 | 24.77 | 46.53 | 35.11 | 34.48 | - |

### Ablation Study

| Configuration | Avg F1 | Description |
|------|--------|------|
| Full Model | 54.33 | Full model |
| w/o Multi-view | 52.31 (-2.02) | Use only the single view with the lowest entropy |
| w/o Entropy Ranking | 52.53 (-1.80) | Randomly sampled views |
| w/o Stage Separation | 52.73 (-1.60) | Single-stage direct generation |
| w/o Stopword Filtering | 53.31 (-1.02) | Do not filter stopwords |
| w/o All Three | 45.80 (-8.53) | Remove filtering + stage separation + entropy ranking |

### Key Findings

- **Multi-view is the single most contributing component** (removing it drops F1 by 2.02), followed by instance-level entropy ranking (drop of 1.80) and two-stage separation (drop of 1.60). Removing all three simultaneously causes F1 to plunge by 8.53, suggesting synergistic effects among these components.
- **Remarkable inference efficiency advantage**: DOT is approximately 7.2 times faster than MvP (298s vs 2162s), with the efficiency gap widening on larger datasets.
- **Superior cross-domain transfer performance**: In the SemEval $\rightarrow$ Yelp cross-domain evaluation, DOT's performance decline (17.59/11.28) is smaller than that of MvP-15 (21.45/16.18), demonstrating the robust cross-domain capability of dynamic view selection.
- **LLM baseline is significantly weaker than fine-tuned T5**: GPT-4o only achieves an F1 of 34.48, which is far below DOT's 52.28, showing that ABSA quadruple extraction still heavily relies on specialized fine-tuning.

## Highlights & Insights

- **Decoupling quadruple count prediction from content generation** is clever and intuitive. This essentially adopts a "plan first, then execute" strategy—first deciding how many items to generate and then generating them one by one, which prevents the autoregressive model from missing or duplicating quadruples in long sequences.
- **Instance-level dynamic view selection** strikes an elegant balance between efficiency and effectiveness. Simple samples only consume 1 view, while complex samples are automatically allocated multiple views, eliminating the need for manual tuning of the hyperparameter $k$ in MvP.
- **The discovery of excluding the O element in Stage 1** is intriguing: Opinion is the most challenging element to generate. If the model attempts to learn O during the count prediction phase, its focus gets distracted in the second stage. This suggests that in multi-stage systems, each stage should strictly focus on the most relevant information.

## Limitations & Future Work

- **Two-stage inference is non-end-to-end**, meaning errors in Stage 1 propagate to Stage 2 (incorrect counts lead to false positives/negatives), and running two models in series increases training time and memory overhead.
- **Relatively poor performance on small datasets with many implicit aspects/opinions** (such as ACOS Rest16), since predicting implicit elements is inherently difficult and limited data further exacerbates the problem.
- **Equating the number of required views to the number of quadruples is a simplifying assumption** — for instances with difficult quadruples, more views than the actual number of quadruples might be necessary to guarantee correct prediction.

## Related Work & Insights

- **vs MvP (Gou et al., 2023)**: MvP statically ranks views at the dataset level and keeps the view count $k$ fixed, whereas DOT determines this dynamically at the instance level. DOT achieves slightly higher accuracy while keeping inference 7 times faster.
- **vs DLO (Hu et al., 2022)**: DLO enhances robustness using multi-template data augmentation, but still utilizes a single template during inference. DOT dynamically selects templates during inference.
- **vs Seq2Path (Mao et al., 2022)**: Seq2Path generates quadruples as tree paths, automatically choosing valid paths. DOT's two-stage method provides more explicit control over the generation process.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of dynamic view choice is intuitive yet effective; the two-stage decoupled design is well-rationalized.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 datasets, multiple backbones, cross-domain experiments, LLM comparisons, and intensive ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear method descriptions, abundant figures/tables, and a solid appendix.
- Value: ⭐⭐⭐⭐ High practical value with clear efficiency and performance gains for ABSA tasks, and the underlying methodology can be generalized to other structured generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Analysis](syngraph_a_dynamic_graph-llm_synthesis_framework_for_sparse_streaming_user_senti.md)
- [\[ACL 2026\] DimABSA: Building Multilingual and Multidomain Datasets for Dimensional Aspect-Based Sentiment Analysis](../../ACL2026/nlp_understanding/dimabsa_building_multilingual_and_multidomain_datasets_for_dimensional_aspect-ba.md)
- [\[ACL 2026\] MSMO-ABSA: Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis](../../ACL2026/nlp_understanding/msmo-absa_multi-scale_and_multi-objective_optimization_for_cross-lingual_aspect-.md)
- [\[ACL 2025\] Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)
- [\[ACL 2025\] Sentiment Reasoning for Healthcare](sentiment_reasoning_for_healthcare.md)

</div>

<!-- RELATED:END -->
