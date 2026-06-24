---
title: >-
  [Paper Note] COLING-UniA at SciVQA 2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models
description: >-
  [ACL 2025][Multimodal VLM][Scientific Chart VQA] This paper proposes a scientific chart visual question answering system based on Multimodal Large Language Model (MLLM) ensembling. Utilizing a few-shot exemplar retrieval strategy and a confidence-aware model selection mechanism, the system achieved third place (average F1 = 85.12) in the SciVQA 2025 shared task.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Scientific Chart VQA"
  - "Multimodal Large Language Models"
  - "Few-Shot Retrieval"
  - "Confidence-Informed Ensemble"
  - "Model Calibration"
date: 2026-05-08
content_hash: 38ed729e72a73ffe
---

# COLING-UniA at SciVQA 2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2507.02357](https://arxiv.org/abs/2507.02357)  
**Code**: [Available](https://github.com/coling-unia/few-shot-scivqa2025)  
**Area**: Multimodal VLM  
**Keywords**: Scientific Chart VQA, Multimodal Large Language Models, Few-Shot Retrieval, Confidence-Informed Ensemble, Model Calibration

## TL;DR

This paper proposes a scientific chart visual question answering system based on Multimodal Large Language Model (MLLM) ensembling. Utilizing a few-shot exemplar retrieval strategy and a confidence-aware model selection mechanism, the system achieved third place (average F1 = 85.12) in the SciVQA 2025 shared task.

## Background & Motivation

Scientific Visual Question Answering (Scientific VQA) requires systems to answer natural language questions about scientific charts (e.g., line charts, bar charts, architecture diagrams). Unlike general VQA, scientific chart VQA faces the following challenges:

**High diversity of chart types**: The dataset contains various types such as line plots, scatter plots, pie charts, confusion matrices, and neural network architectures, where different types require distinct comprehension capabilities.

**Complex question types**: Includes seven question types, such as binary questions, multiple choice (four options), open-ended (visual/non-visual), and unanswerable questions.

**Limitations of Prior Work**: Most chart VQA methods rely on specialized models fine-tuned on the chart domain, which limits their generalizability.

The core motivation of this study is: **Without any fine-tuning, and solely utilizing zero/few-shot prompting and intelligent ensembling strategies**, can open-source MLLMs achieve competitive performance on scientific chart VQA?

## Method

### Overall Architecture

The system adopts a two-stage ensembling architecture (Confidence-Informed Ensemble):

1. **First Stage**: Processes all instances using a well-calibrated InternVL3-78B configuration (1s_q_img_f, BLIP2), retaining highly confident answers with confidence $\ge 90\%$ (covering approximately half of the instances).
2. **Second Stage**: For the remaining low-confidence instances, selects the optimal model and few-shot configuration combination based on the question type.

The models utilized are all open-source weight models:
- **InternVL3-78B**: Stronger performance but limited context window, supporting only 0-shot and 1-shot.
- **Pixtral-Large-Instruct-2411**: Larger context window, supporting up to 2-shot, where additional exemplars yield greater improvements.

All models utilize 16-bit quantization, with the temperature set to 0.

### Key Designs

#### 1. Few-Shot Exemplar Retrieval Strategy

This study evaluates combinations of multiple retrieval strategies:

| Dimension | Options |
|------|------|
| Similarity Source | Question similarity only (q) / Question + image similarity (q_img) |
| Embedding Model | SBERT / CLIP / BLIP-2 |
| Retrieval Scope | Filtered by chart type (f) / Search entire training set (nf) |
| Number of Exemplars | 0-shot / 1-shot / 2-shot |

- **Question similarity**: Ranked using the cosine similarity of SBERT embeddings.
- **Question-image similarity**: Computes question and image vectors via CLIP embeddings, normalizes and averages them, and ranks them by cosine similarity.
- **BLIP-2 variant**: Embeddings from BLIP-2 primarily reflect image content, leading to many tied ranks.
- **2-shot design**: Selects one answerable and one unanswerable exemplar, which effectively assists the model in distinguishing between the two categories.

#### 2. Confidence Estimation and Model Calibration

Confidence calculation method: exponentiates the average log-probability of all generated answer tokens:

$$\text{confidence} = \exp\left(\frac{1}{|T|}\sum_{t \in T} \log p(t)\right)$$

Key finding: InternVL3-78B (1s_q_img_f, BLIP2) exhibits strong calibration characteristics—predictions with high confidence ($\ge 0.9$) indeed correspond to high accuracy. This makes the model suitable as the "gatekeeper" in the first stage.

#### 3. Question/Chart Type Ensemble

To prevent overfitting, the data is partitioned into 16 groups (8 homogeneous chart types + 1 "other" + 7 line chart subsets breakdown by question type). The optimal configuration for each group is determined via 5-fold cross-validation. Key steps:
- Compute the ROUGE-1 F1 of all configurations on each fold.
- Subtract the highest score of that fold to calculate the average gap across folds.
- Repeat at least 10 times until the optimal configuration stabilizes.

### Loss & Training

This method requires no training or fine-tuning, relying entirely on inference-time strategies:
- Input construction: Image + Question + Metadata (chart title, type, whether it contains subplots).
- For questions with predefined options, prompt the model to choose from the options.
- For open-ended questions, instruct the model to answer freely.
- Instruct the model to judge whether the question can be answered based on the provided information.

## Key Experimental Results

### Main Results

**Official test set ranking (Table 1)**:

| Rank | Team | ROUGE-1 F1 | ROUGE-L F1 | BERTScore F1 | Average |
|------|------|-----------|-----------|-------------|------|
| 1 | ExpertNeurons | 80.49 | 80.43 | 98.49 | 86.47 |
| 2 | THAii_LAB | 78.99 | 78.92 | 98.39 | 85.43 |
| **3** | **Coling-UniA** | **78.62** | **78.56** | **98.17** | **85.12** |
| Median | - | 75.83 | 75.75 | 98.36 | 83.31 |

**Comparison of different methods on the test set (Table 2 Summary)**:

| Method | ROUGE-1 F1 | ROUGE-L F1 | BERTScore F1 |
|------|-----------|-----------|-------------|
| InternVL (1s_q_img_f, BLIP2) Single Model | 77.2 | 77.2 | 98.1 |
| Question/Chart Type Ensemble | 77.7 | 77.6 | 98.1 |
| **Confidence-Aware Ensemble** | **78.6** | **78.6** | **98.2** |

### Ablation Study

**Impact of few-shot on performance**:
- InternVL3-78B: 0-shot R1-F1 = 74.2 $\rightarrow$ optimal 1-shot@75.0 (+0.8).
- Pixtral-Large: 0-shot R1-F1 = 71.4 $\rightarrow$ optimal 2-shot@74.1 (+2.7).
- 2-shot is almost always beneficial for Pixtral; adding exemplars is particularly helpful for identifying unanswerable questions.

**Unanswerable question identification accuracy (Table 3)**:
- Pixtral 0-shot: 93.0% $\rightarrow$ 2-shot(q_img_f): 94.1%.
- The 2-shot strategy using one answerable and one unanswerable exemplar performs the best.

**Performance varies extremely across different question types**:
- Binary questions (visual): ~81% 
- Multiple choice (four options): ~76-79%
- Open-ended (non-visual): ~65-68%
- Open-ended (visual): ~49-54% (most difficult)
- Unanswerable: ~77-91%

### Key Findings

1. **Competitive performance without fine-tuning**: Solely relying on zero/few-shot strategies, open-source MLLMs can achieve 3rd place in the competition, outperforming the baseline by approximately 4 percentage points.
2. **Question/chart type significantly impacts the optimal strategy**: No single configuration is optimal across all subsets.
3. **Practical value of model calibration**: High-confidence predictions from InternVL3-78B are indeed reliable, enabling tiered processing.
4. **Minor differences between retrieval strategies**: Question similarity vs. question-image similarity shows limited performance differences overall.

## Highlights & Insights

1. **Novel tiered decision-making approach**: Filtering "easy" instances first with a high-confidence model before finely dispatching "hard" instances is more efficient than a single model or simple ensembling.
2. **Discovery in confidence calibration**: InternVL3 under the BLIP-2 retrieval configuration shows excellent calibration in the high-confidence range, which is a practical and reusable finding.
3. **Competitiveness without fine-tuning**: Demonstrates the strong zero-shot capabilities of MLLMs in scientific chart understanding, implying that extensive chart-specific fine-tuning might no longer be necessary.
4. **Systematic experimental design**: Detailed evaluation of all configuration combinations with analysis granular enough to cover every question type $\times$ chart type.

## Limitations & Future Work

1. **Potential data leakage**: Images in ACL-Fig and SciGraphQA may have already been seen by MLLMs during pre-training.
2. **Evaluation bias in unanswerable questions**: Unanswerable questions in the dataset follow fixed patterns (mostly referring to materials inaccessible to the model), which might not reflect real-world scenarios.
3. **High computational resource demand**: The experimental cost of approximately 3600 GPU hours limits larger-scale exploration.
4. **Unexplored fine-tuning**: Performance might be further enhanced if lightweight fine-tuning were applied to InternVL3 or Pixtral.
5. **Basic retrieval strategies**: More sophisticated exemplar selection methods could be considered (e.g., difficulty-based retrieval, chart structure-aware retrieval).

## Related Work & Insights

- Various types of VQA datasets such as **VQA v2, DocVQA, ChartQA, PlotQA**, with this study focusing on scientific charts.
- **InternVL3-78B, Pixtral-Large** as representatives of open-source MLLMs, demonstrating the capabilities of large models in scientific chart comprehension.
- **SBERT, CLIP, BLIP-2** utilized for text similarity, cross-modal similarity, and multimodal embedding retrieval, respectively.
- Insight: The value of model calibration in practical system deployment—it can be leveraged to build a cascaded inference pipeline that addresses "easy first, hard later".

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 3 |
| Practicality | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Overall Rating | 3.5 |

The method is relatively engineering-oriented, but the system design is reasonable and the experiments are solid. In particular, the comprehensive comparisons among different configurations and the confidence calibration analysis offer valuable references.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)
- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](../../ICCV2025/multimodal_vlm/enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)
- [\[CVPR 2026\] CICA: Coupling Confidence-Aware Pretraining with Confidence-Informed Attention for Robust Multimodal Sentiment Analysis](../../CVPR2026/multimodal_vlm/cica_coupling_confidence-aware_pretraining_with_confidence-informed_attention_fo.md)
- [\[ACL 2025\] Burn After Reading: Do Multimodal Large Language Models Truly Capture Order of Events in Image Sequences?](burn_after_reading_do_multimodal_large_language_models_truly_capture_order_of_ev.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](../../CVPR2025/multimodal_vlm/rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)

</div>

<!-- RELATED:END -->
