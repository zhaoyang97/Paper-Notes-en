---
title: >-
  [Paper Note] Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation
description: >-
  [ACL 2025][Model Compression][Data Laundering] This paper exposes a vulnerability where knowledge distillation can be abused to artificially inflate benchmark scores. Through "Data Laundering," knowledge learned by a teacher model on a test set is covertly transferred to a student model via seemingly legitimate intermediate training steps. This allows a 2-layer BERT to achieve 73.94% on GPQA (close to OpenAI o1's 77.30%) without actually learning how to reason.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Data Laundering"
  - "Knowledge Distillation"
  - "Benchmark Manipulation"
  - "Data Contamination"
  - "Evaluation Security"
date: 2026-05-08
content_hash: 9c0ff9178f525d14
---

# Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation

**Conference**: ACL 2025  
**arXiv**: [2412.15255](https://arxiv.org/abs/2412.15255)  
**Code**: [https://github.com/mbzuai-nlp/data_laundering](https://github.com/mbzuai-nlp/data_laundering)  
**Area**: Model Compression  
**Keywords**: Data Laundering, Knowledge Distillation, Benchmark Manipulation, Data Contamination, Evaluation Security

## TL;DR
This paper exposes a vulnerability where knowledge distillation can be abused to artificially inflate benchmark scores. Through "Data Laundering," knowledge learned by a teacher model on a test set is covertly transferred to a student model via seemingly legitimate intermediate training steps. This allows a 2-layer BERT to achieve 73.94% on GPQA (close to OpenAI o1's 77.30%) without actually learning how to reason.

## Background & Motivation
As benchmarks such as MMLU, GPQA, and BigBench have become standard metrics for evaluating and comparing LLM capabilities, benchmark scores have driven the direction of AI R&D. However, this over-reliance on benchmark scores introduces vulnerabilities to manipulation and gaming.

Prior studies have revealed issues where models like GPT-3/GPT-4 unintentionally learn from leaked benchmark data. Existing contamination detection methods (e.g., n-gram overlap, LM Contamination Index) may fail to identify more subtle benchmark gaming. This work discovers a more covert method of manipulation: **using knowledge distillation as an intermediary to legitimately "launder" benchmark-specific knowledge from a contaminated teacher model to a student model**.

Crucially, this manipulation can be intentional or unintentional—researchers distilling from teacher models of unknown origin might not know that the teacher was trained on the benchmark test set. Analogous to the three phases of financial money laundering (placement, layering, integration), this paper conceptualizes this process as "data laundering."

## Method

### Overall Architecture
Data Laundering consists of three phases, corresponding to the three steps in financial laundering:
1. **Placement**: The teacher model is trained directly on the benchmark test set, acquiring "illicit" knowledge.
2. **Layering**: Through knowledge distillation, the knowledge is transferred to the student model using a seemingly legitimate intermediate dataset.
3. **Integration**: The student model is evaluated on the original benchmark to verify whether the "laundered" knowledge was successfully transferred.

### Key Designs
1. **Placement Phase - Teacher Model Training**:

    - The teacher model is trained directly on the benchmark test set (e.g., GPQA, MMLU-Redux).
    - The contaminated teacher model achieves nearly 100% accuracy on the benchmark.
    - Various teacher models are evaluated: BERT-base, GPT-2, LLaMA3.2-3B, LLaMA3.1-8B.

2. **Layering Phase - Knowledge Distillation**:

    - Key point: The student model **never has contact with the test set** during the distillation process.
    - MedMCQA or RACE is used as the intermediate training dataset (completely different domains from the benchmark).
    - Distillation loss: $L_{student} = (1-\alpha)L_{hard} + \alpha L_{soft}$
    - $L_{hard}$: Cross-entropy loss on the intermediate dataset.
    - $L_{soft}$: MSE or KL divergence loss on the teacher model's logits.
    - The effects of different $\alpha$ values (from 0 to 1.0) are explored.

3. **Integration Phase - Benchmark Evaluation**:

    - The student model is tested on the original benchmark (GPQA, MMLU-Redux).
    - Despite never seeing the test set, the student model acquires benchmark-relevant knowledge through distillation.

4. **Iterative Distillation Experiments**:

    - After each round of distillation, the student becomes the new teacher for the next student.
    - The retention of knowledge across multiple rounds of transfer is tested.

### Loss & Training
Core distillation loss: $L_{student} = (1-\alpha)L_{hard} + \alpha L_{soft}$

$L_{soft}$ can be either MSE or KL divergence loss. Experiments indicate that MSE loss typically performs better (making knowledge leakage more pronounced). $\alpha=1.0$ is found to be the most stable in iterative distillation.

## Key Experimental Results

### Main Results

| Model | Training Data | GPQA(%) | MMLU-Redux(%) | Description |
|------|---------|---------|---------------|------|
| OpenAI o1 | - | 77.30 | - | SOTA Baseline |
| Claude 3.5 Sonnet | - | 59.40 | 81.00 | Strong Baseline |
| GPT-4o | - | 50.60 | 81.00 | Strong Baseline |
| LLaMA3-70B | - | 39.50 | 76.00 | Strong Baseline |
| BERT-2-layer (Normal) | MedMCQA | 25.76 | 25.33 | Random level |
| **BERT-2-layer + Laundered** | MedMCQA | **73.94** | **62.31** | Close to o1! |
| BERT-12-layer + Laundered | MedMCQA | 59.39 | 47.00 | Far exceeds normal |
| GPT-2-2-layer + Laundered | MedMCQA | 43.01 | 33.17 | Exceeds LLaMA3-70B |
| LLaMA3.2-3B + Laundered | MedMCQA | 39.39 | 47.48 | Significant effect |

### Impact of Training Data Selection

| Intermediate Dataset | GPQA(%) | MMLU-Redux(%) | Description |
|----------|---------|---------------|------|
| MedMCQA (2-layer BERT) | 73.94 | 62.31 | Better domain alignment |
| RACE (2-layer BERT) | 69.16 | 47.14 | Poorer domain alignment |

### Iterative Distillation

| Iteration Round | $\alpha=1.0$ (BERT) | $\alpha=0.6$ (BERT) | Description |
|---------|------------|------------|------|
| Round 1 | ~75% | ~73% | Start |
| Round 5 | ~72% | ~55% | $\alpha=1.0$ is more stable |

### Key Findings
- A 2-layer BERT achieves 73.94% on GPQA through data laundering, close to OpenAI o1's 77.30%, and far exceeding GPT-4o's 50.60%.
- Knowledge leakage persists across all $\alpha$ values and loss functions; even at $\alpha=0.1$, performance far exceeds the random level.
- Knowledge leakage is more severe with MSE loss compared to KL divergence loss (75% vs. 72% on BERT).
- Domain alignment of the intermediate dataset is crucial: MedMCQA performs better than RACE (owing to higher semantic similarity with GPQA).
- Smaller BERT models perform better than larger ones (shallow encoder architectures excel at distilling compact knowledge), whereas for GPT-2, larger models perform better.
- In iterative distillation, $\alpha=1.0$ remains stable (retaining 70-75% after 5 rounds), while $\alpha=0.6$ suffers from knowledge drift.
- Even with a training set of only 500 instances, knowledge leakage still occurs ($48.99\% \gg 25\%$ random).
- Diminishing returns occur beyond approximately 15,000 data points.

## Highlights & Insights
- The analogy to financial money laundering is vivid and highly accurate, making a complex technical problem easy to understand.
- The scenario of unintentional manipulation is particularly realistic: researchers distilling from opaque teacher models might not know that the teacher has been contaminated.
- The result of a 2-layer BERT outperforming GPT-4o is highly striking, demonstrating that benchmark scores may not reflect actual capabilities.
- The code is open-sourced, ensuring good reproducibility.
- It sounds a crucial warning regarding the security of AI evaluation frameworks.

## Limitations & Future Work
- The experiments focus solely on classification tasks, leaving the effects of data laundering in generative tasks unexplored.
- The experiments leverage relatively small datasets; the effect might diminish on large, diverse datasets.
- No effective defense methods are proposed (only suggesting the use of private benchmarks or teacher models of known origin).
- There is no analysis of what form the laundered knowledge takes: does the model memorize answer patterns, or has it learned a reasoning shortcut?
- Research idea: Future work could develop laundering detection methods based on model behavior analysis (e.g., checking model consistency across different prompt variations).
- Ethical consideration: While published to raise awareness, open-sourcing the manipulation method carries a risk of malicious exploitation.

## Related Work & Insights
- This work complements data contamination detection methods (such as n-gram overlap and LM Contamination Index) by revealing a more hidden pathway of contamination propagation.
- Unlike the "null model" attack by Zheng et al., the proposed method achieves manipulation through seemingly legitimate training processes.
- While distillation approaches like DistiLLM and SinKD focus on improving distillation, this work exposes the dark side of distillation.
- Crucial implications for benchmark designers: there is a pressing need to develop evaluation methods resilient to distillation-based leakage.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uncovers a completely new security vulnerability in knowledge distillation; the problem is well-defined and has far-reaching implications.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation studies cover diverse model architectures, loss functions, $\alpha$ values, and data sizes, though generative tasks are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ The financial analogy makes the paper highly engaging, with well-structured experiments and insightful discussions.
- Value: ⭐⭐⭐⭐⭐ Serves as a vital warning for AI evaluation safety, offering broad community impact and practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Knowledge Distillation as Decontamination? Revisiting the "Data Laundering" Concern in Classification Tasks](../../ICLR2026/model_compression/knowledge_distillation_as_decontamination_revisiting_the_data_laundering_concern.md)
- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](../../NeurIPS2025/model_compression/single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)
- [\[ACL 2025\] Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation](beyond_logits_aligning_feature_dynamics_for_effective_knowledge_distillation.md)
- [\[ACL 2025\] Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](flipping_kd_small_to_large.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)

</div>

<!-- RELATED:END -->
