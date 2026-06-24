---
title: >-
  [Paper Note] AD-LLM: Benchmarking Large Language Models for Anomaly Detection
description: >-
  [ACL 2025][LLM Evaluation][Anomaly Detection] This paper proposes the first LLM anomaly detection benchmark, AD-LLM, to systematically evaluate the capability of LLMs in three core tasks: zero-shot detection, data augmentation, and unsupervised model selection. It reveals that GPT-4o zero-shot detection outperforms traditional training-based methods on most datasets. Additionally, synthetic data benefits detectors utilizing flexible representation learning but harms models wi…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Anomaly Detection"
  - "LLM Benchmark"
  - "Zero-shot Detection"
  - "Data Augmentation"
  - "Model Selection"
date: 2026-05-08
content_hash: dae1e3a8a22a52a9
---

# AD-LLM: Benchmarking Large Language Models for Anomaly Detection

**Conference**: ACL 2025  
**arXiv**: [2412.11142](https://arxiv.org/abs/2412.11142)  
**Code**: [GitHub](https://github.com/USC-FORTIS/AD-LLM)  
**Area**: LLM Evaluation  
**Keywords**: Anomaly Detection, LLM Benchmark, Zero-shot Detection, Data Augmentation, Model Selection

## TL;DR

This paper proposes the first LLM anomaly detection benchmark, AD-LLM, to systematically evaluate the capability of LLMs in three core tasks: zero-shot detection, data augmentation, and unsupervised model selection. It reveals that GPT-4o zero-shot detection outperforms traditional training-based methods on most datasets. Additionally, synthetic data benefits detectors utilizing flexible representation learning but harms models with fixed geometric assumptions. Finally, reasoning LLMs achieve near-optimal model selection, though their explanations lack explicit dataset specificity.

## Background & Motivation

**Background**: Anomaly Detection (AD) is a critical task in machine learning, utilized in NLP to detect spam, misinformation, and anomalous user activity. Traditional approaches are divided into two categories: end-to-end algorithms (directly processing raw text) and two-step methods (first extracting embeddings with language models, then employing traditional AD algorithms like LOF or iForest). These methods generally require extensive training data, and model selection heavily relies on domain experts' experience.

**Limitations of Prior Work**: (1) Many AD tasks lack labeled data, making it costly to train supervised/unsupervised models; (2) AD data is typically highly imbalanced—such as extremely sparse insurance fraud samples; (3) Choosing an appropriate AD model requires extensive trial-and-error and domain knowledge, often leading to arbitrary selections in practice. Although LLMs exhibit stellar performance in text generation and summarization, their potential within AD remains systematically unexplored.

**Key Challenge**: LLMs possess extensive pre-trained knowledge and robust semantic understanding capabilities, indicating they should theoretically contribute to multiple stages of AD. However, a unified evaluation framework is lacking to validate this assumption—existing studies either focus exclusively on a single task (such as zero-shot detection) or cover only specific modalities (such as industrial image AD).

**Goal**: To establish the first unified LLM+AD benchmark spanning three core tasks: detection, augmentation, and selection, thereby systematically answering "where, how, and to what extent LLMs can contribute to NLP anomaly detection."

**Key Insight**: Decompose the role of LLMs in AD into three complementary tasks—serving as detectors (replacing traditional methods), data generators (alleviating data scarcity), and consultants (recommending models), designing dedicated experimental protocols and evaluation metrics for each role.

**Core Idea**: LLMs can comprehensively empower NLP anomaly detection across the three dimensions of detection, augmentation, and selection, though the effectiveness varies depending on the specific tasks and model characteristics.

## Method

### Overall Architecture

The AD-LLM benchmark is designed around three core tasks: **Task 1: Zero-shot Detection**—utilizing pre-trained LLM knowledge to directly determine if a sample is anomalous; **Task 2: Data Augmentation**—leveraging LLMs to generate synthetic samples or class descriptions to enhance traditional AD models; **Task 3: Model Selection**—employing LLMs to analyze dataset attributes and model descriptions to recommend the optimal AD model. Evaluation covers 5 NLP datasets (AG News, BBC News, IMDB Reviews, N24 News, SMS Spam) using AUROC and AUPRC, compared against 18 traditional unsupervised AD baselines.

### Key Designs

1. **Zero-shot Anomaly Detection (Task 1)**:

    - **Function**: Directly determine whether a text sample is anomalous based on pre-trained LLM knowledge, without requiring task-specific training data.
    - **Mechanism**: Designs two evaluation settings: "Normal Only" (providing only the normal class name $\mathcal{C}_{\text{normal}}$) and "Normal + Anomaly" (simultaneously providing the anomaly class name $\mathcal{C}_{\text{anomaly}}$). The input is constructed via a prompt template $\mathcal{P} = T(x_i, \mathcal{C}_{\text{normal}}, \mathcal{C}_{\text{anomaly}}^*)$, and the LLM outputs an anomaly score $s$ and reasoning explanation $r$, denoted as $(r, s) = f_{\text{LLM}}(\mathcal{P})$. Llama 3.1 8B, GPT-4o, and DeepSeek-V3 are evaluated.
    - **Design Motivation**: Simulates real-world scenarios where labeled data is scarce, verifying the direct value of LLM pre-training knowledge for AD. The two settings correspond to different levels of prior knowledge.

2. **LLM-Driven Data Augmentation (Task 2)**:

    - **Function**: Generate synthetic data and class descriptions via LLMs to mitigate AD data scarcity and imbalance issues.
    - **Mechanism**: Splits into two pathways: (a) **Synthetic Sample Generation**: Adopts a two-step strategy to avoid repetitiveness, first generating multi-granularity keyword sets (coarse/medium/fine-grained) and then generating samples $\tilde{x}_i$ based on these keywords. During multi-round generation, random seeds, temperature, and templates are adjusted to guarantee diversity, finally merging into $\mathcal{D}_{\text{DA}} = \mathcal{D}_{\text{small\_train}} \cup \mathcal{D}_{\text{synth}}$ to train AD models. (b) **Class Description Generation**: Generates textual descriptions $d_{\text{normal}}, d_{\text{anomaly}}$ for normal/anomalous classes to embed in detection prompts for enhanced semantic reasoning, updating the prompt to $\mathcal{P} = T(x_i, (\mathcal{C}_{\text{normal}}, d_{\text{normal}}), (\mathcal{C}_{\text{anomaly}}, d_{\text{anomaly}})^*)$.
    - **Design Motivation**: Direct prompting of LLMs to generate samples often leads to highly repetitive outputs; the keyword-sample two-step approach + multi-granularity control ensures the diversity and semantic consistency of synthetic data.

3. **LLM-Assisted Model Selection (Task 3)**:

    - **Function**: Utilize the reasoning capability of LLMs to recommend the optimal unsupervised AD model for a given dataset.
    - **Mechanism**: Provides structured inputs to the LLM—dataset descriptions (name, size, background, normal/anomalous classes, text length statistics, representative samples) and candidate model descriptions (paper abstracts), based on which the LLM outputs recommended models and justifications. Reasoning-enhanced models (o1-preview, o1, o3-mini, DeepSeek-R1) are selected, with each dataset queried 5 times to aggregate results, alongside context-free control experiments to test the intrinsic preferences of the LLM.
    - **Design Motivation**: Traditional model selection relies on historical performance data or domain experts, which fails for new datasets; the extensive knowledge of LLMs may enable zero-shot model recommendation.

## Key Experimental Results

### Main Results: Comparison of Zero-Shot Detection Performance

| Model | Setting | AG News AUROC | BBC News AUROC | IMDB AUROC | N24 News AUROC | SMS Spam AUROC |
|------|------|------|------|------|------|------|
| GPT-4o | Normal Only | 0.933 | 0.957 | 0.935 | 0.767 | 0.794 |
| GPT-4o | Normal+Anomaly | 0.929 | **0.992** | **0.967** | **0.990** | **0.986** |
| DeepSeek-V3 | Normal+Anomaly | 0.927 | 0.958 | 0.963 | 0.951 | 0.954 |
| Llama 3.1 8B | Normal+Anomaly | 0.875 | 0.861 | 0.863 | 0.878 | 0.949 |
| Best Baseline | — | 0.923 | 0.973 | 0.737 | 0.832 | 0.940 |

GPT-4o outperforms the best baseline on 4 out of 5 datasets under the Normal+Anomaly setting, with the AUROC on IMDB increasing from the baseline of 0.737 to 0.967 (+31%).

### Analysis of Data Augmentation Effects

| Detector Type | Representative Model | Augmented Effect | Reason Analysis |
|-----------|---------|-----------|---------|
| Flexible Representation Learning | AE, VAE, LUNAR, ECOD | ✅ Significant improvement, close to full-data performance | Reconstruction/empirical distribution/graph aggregation objectives can utilize rich embedding manifolds |
| Fixed Geometric Assumptions | DeepSVDD, iForest, LOF | ❌ Performance drop | Variance of synthetic data expands the hypersphere/disrupts isolation statistics/blurs local density |
| Adversarial Training | SO_GAAL | ❌ Performance drop | Variance expands the definition of normal data, making discriminator convergence difficult |

### Model Selection: LLM Recommendation vs. Baselines

| Reasoning LLM | Most Frequent Recommendation | Avg AUROC of Recommended Models | Best Baseline AUROC | Avg AUROC of Random Selection |
|---------|-----------|-----------------|-------------|-----------------|
| o1-preview | OpenAI+LUNAR (13/25) | Near-optimal | 0.923-0.992 | 0.65-0.80 |
| DeepSeek-R1 | OpenAI+ECOD (16/25) | Near-optimal | — | — |
| o1 | OpenAI+DeepSVDD (11/25) | Lower | — | — |
| o3-mini | BERT+DeepSVDD (10/25) | Lower | — | — |

### Key Findings

- **Zero-shot detection is the most promising application of LLMs in AD**: Without any training data, GPT-4o outperforms traditional methods requiring training on 4/5 datasets, with the most notable improvement on the IMDB dataset (+31% AUROC).
- **Richer contexts yield better detection**: Moving from the "Normal Only" to the "Normal+Anomaly" setting, the AUROC and AUPRC for all LLMs improve, indicating that prior knowledge of anomalous classes is crucial for detection.
- **Synthetic data is a double-edged sword**: It is effective for models learning flexible representations (AE/VAE/ECOD/LUNAR), but harmful to models relying on fixed geometric assumptions (iForest/LOF/DeepSVDD)—the variance introduced by synthetic data disrupts the core assumptions of these models.
- **LLMs exhibit intrinsic preferences in model selection**: Without context, LLMs display fixed default preferences (e.g., o1 prefers VAE); when context is provided, preferences shift and align closer to the optimal, demonstrating that LLMs are influenced by both pre-training bias and input information.
- **Specificity of explanations remains an open challenge**: The justifications for model selection provided by LLMs tend to be generalized (e.g., "suitable for high-dimensional data") and lack specific analytical depth tailored to the properties of the given dataset.

## Highlights & Insights

- **Systematic Design of the Three-Task Unified Framework**: Decomposing the role of LLMs in AD into three complementary dimensions—detection, augmentation, and selection—covers the complete spectrum from no training data to equipped training data, and from direct detection to indirect assistance. This decomposition provides a clear organizing framework for subsequent research.
- **Opposing Effects of Synthetic Data Reveal Deep-Seated Mechanisms**: Flexible models (AE/VAE) benefit because rich embedding manifolds facilitate reconstruction learning, whereas geometric models (iForest/LOF) suffer because heterogeneous synthetic data violates their isolation or density assumptions. This finding provides clear guidance on "when to use synthetic data."
- **Ingenious Context-Free Control Experiment**: Comparing the shifts in model recommendation preferences with and without context successfully decouples the effects of pre-training bias from input information, offering a fresh perspective on understanding LLM reasoning mechanisms.

## Limitations & Future Work

- **Limited Dataset Coverage**: The evaluation is restricted to only 5 NLP classification datasets with clear boundaries between normal and anomalous classes—scenarios with ambiguous anomaly definitions (e.g., financial fraud, medical anomalies) remain unverified.
- **Few-Shot and Fine-Tuning Unexplored**: Although the zero-shot setting represents extreme cases, labeled data is often partially available in practice, where few-shot and fine-tuning strategies might yield superior performance gains.
- **Unstable Category Description Enhancement**: As shown in the results, description enhancement degrades performance on certain datasets (e.g., Llama drops by 0.088 AUROC on N24 News), indicating a need for finer quality control over the descriptions.
- **Insufficient Interpretability of Model Selection**: The justifications provided by LLMs for model selection are overly generalized, failing to explain *why* a specific model suits a particular dataset, which limits practical trustworthiness.
- **Inadequate Discussion on Inference Overhead**: Zero-shot detection requires invoking the LLM API for each sample, which potentially incurs time and economic costs vastly exceeding those of traditional methods on large-scale datasets.

## Related Work & Insights

- **vs. Xu & Ding (2024)**: The latter proposes a taxonomy of LLMs in AD (detection vs. generation tools) but remains a conceptual framework without experimental validation; AD-LLM provides the first systematic empirical benchmark.
- **vs. MMAD (Jiang et al., 2024b)**: The latter focuses on multimodal scenarios for industrial image AD; AD-LLM covers NLP textual AD, filling the gap in the text modality.
- **vs. Liu et al. (2024b)**: The latter evaluates LLM embeddings for OOD detection (cosine distance + isotropic embeddings) but does not explore data augmentation and model selection; AD-LLM covers a more comprehensive AD lifecycle.

## Rating

- Novelty: ⭐⭐⭐⭐ The first LLM benchmark covering three major AD tasks, featuring highly systematic framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage with 5 datasets, 18 baselines, and 3-4 LLMs, though the dataset types remain relatively uniform.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, formalized problem definitions, and rich visualizations/tables.
- Value: ⭐⭐⭐⭐ Provides a unified evaluation platform and empirical insights for LLM+AD research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Ad-hoc Concept Forming in the Game Codenames as a Means for Evaluating Large Language Models](ad-hoc_concept_forming_in_the_game_codenames_as_a_means_for_evaluating_large_lan.md)
- [\[ACL 2025\] CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)
- [\[ACL 2025\] Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories](benchmarking_llms_and_llm-based_agents_in_practical_vulnerability_detection_for_.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](../../NeurIPS2025/llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [\[ACL 2025\] Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling](mis-prompt_benchmarking_large_language_models_for_proactive_error_handling.md)

</div>

<!-- RELATED:END -->
