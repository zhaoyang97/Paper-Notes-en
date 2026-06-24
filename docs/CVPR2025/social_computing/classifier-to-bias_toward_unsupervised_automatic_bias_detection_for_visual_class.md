---
title: >-
  [Paper Note] Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers
description: >-
  [CVPR 2025][Social Computing][Bias Detection] Proposes C2B (Classifier-to-Bias), the first framework to automatically detect biases in pre-trained visual classifiers using only the textual descriptions of the classification task (without any labeled data). By leveraging LLMs to generate class-specific bias candidates, creating retrieval queries to collect image datasets, and finally calculating bias scores, C2B outperforms supervised SOTA bias detection methods on CelebA and…
tags:
  - "CVPR 2025"
  - "Social Computing"
  - "Bias Detection"
  - "Unsupervised"
  - "LLM"
  - "Retrieval Augmentation"
  - "Model Fairness"
date: 2026-05-08
content_hash: 405501e02f09c729
---

# Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers

**Conference**: CVPR 2025  
**arXiv**: [2504.20902](https://arxiv.org/abs/2504.20902)  
**Code**: [https://github.com/mardgui/C2B](https://github.com/mardgui/C2B)  
**Area**: Social Computing  
**Keywords**: Bias Detection, Unsupervised, LLM, Retrieval Augmentation, Model Fairness

## TL;DR

Proposes C2B (Classifier-to-Bias), the first framework to automatically detect biases in pre-trained visual classifiers using only the textual descriptions of the classification task (without any labeled data). By leveraging LLMs to generate class-specific bias candidates, creating retrieval queries to collect image datasets, and finally calculating bias scores, C2B outperforms supervised SOTA bias detection methods on CelebA and ImageNet-X.

## Background & Motivation

**Background**: Pre-trained models are widely shared on platforms like HuggingFace, but users downloading and using them may be unaware of the models' biases and failure modes. Existing bias detection methods (such as B2T, UDIS) rely on annotated data to discover model biases.

**Limitations of Prior Work**: All existing methods require task-specific annotated data (at least classification labels), which significantly limits the applicability of bias detection. Average users may lack the resources to collect annotated data.

**Key Challenge**: Bias detection should be a mandatory step before model deployment, but its reliance on annotated data makes it prohibitively expensive, creating a contradiction where "detecting bias requires data, yet obtaining the data itself is the bottleneck."

**Goal**: Automatically detect model bias under zero-annotation conditions, using only the textual description of the classification task.

**Key Insight**: LLMs possess the reasoning capability to infer what biases might exist for different classification tasks, while large-scale image databases combined with text-based retrieval can provide unannotated test data.

**Core Idea**: Propose bias candidates using LLMs $\rightarrow$ Generate retrieval queries $\rightarrow$ Retrieve images from large-scale databases $\rightarrow$ Test performance differences of the model across different bias classes.

## Method

### Overall Architecture

The three-step workflow of C2B: (1) Use LLMs to generate a list of class-specific bias attributes and bias classes for each target class; (2) Use LLMs to generate queries describing the "target class + bias class" combinations, and collect corresponding images through CLIP retrieval or Bing search; (3) Evaluates the target model on the collected pseudo-labeled dataset, quantifying the bias by comparing accuracy differences across different bias classes.

### Key Designs

1. **Class-Specific Bias Candidate Generation**:

    - **Function**: Automatically generates possible bias attributes and their values for each target class.
    - **Mechanism**: Concatenates the task description $T$ and the class name $y$ as input to an LLM (Llama 3.1 8B), prompting the LLM to output a list of bias attributes $\mathcal{B}^y$ for that class. For example, for bird classification, it might output "background" (forest/sky/water surface), "pose" (flying/perched), etc. Different classes have different bias lists, as bias factors vary across different objects.
    - **Design Motivation**: Biases are task- and class-specific; a global, uniform bias list cannot cover all scenarios. The world knowledge of LLMs is naturally suited for reasoning about "what factors might affect classification."

2. **Retrieval-Augmented Data Collection**:

    - **Function**: Collects test images for each target class-bias class combination.
    - **Mechanism**: A two-step query generation process: first, prompt the LLM to generate task-adapted query templates, and then refine these queries for each specific target class and bias class. The generated queries are used to retrieve the top-k images from the CC12M large-scale database via CLIP, or to harvest them through Bing search engine. Each query retrieves 20 images, with the query itself providing the pseudo-labeled target and bias tags.
    - **Design Motivation**: Avoid collecting and annotating data from scratch by leveraging existing large-scale, unannotated image repositories. Carefully constructed queries ensure that the retrieved images match both the target and bias classes simultaneously.

3. **Bias Score Calculation**:

    - **Function**: Quantifies the performance disparities of the model across different bias classes.
    - **Mechanism**: For each target class $y$ and bias class $b_{i,j}$, the model's accuracy $A_y(f, b_{i,j})$ on this subset is calculated. The bias score is defined as the difference between this accuracy and the average accuracy of other bias classes: $\phi_{y,i,j} = A_y(f, b_{i,j}) - \frac{1}{n_i-1}\sum_{k \neq j} A_y(f, b_{i,k})$. A positive score indicates a model bias towards that class, while a negative score indicates a bias against it.
    - **Design Motivation**: Eliminate the influence of baseline accuracy through relative comparison, directly measuring the performance differences of the classifier across different bias classes.

### Loss & Training

C2B is completely training-free and serves as an inference-time framework. It only requires an LLM and a retrieval engine to run.

## Key Experimental Results

### Main Results

| Method | CelebA Hit↑ | CelebA FH↓ | CelebA Miss↓ |
|------|-------------|------------|--------------|
| B2T (Supervised) | Low | - | High |
| C2B-cc12m (Unsupervised) | **High** | Low | **Low** |
| C2B-Bing (Unsupervised) | **Highest** | Lowest | **Lowest** |

*C2B outperforms B2T (which requires annotated data) under fully unsupervised conditions*

### Ablation Study

| Configuration | Hit Rate | Description |
|------|----------|------|
| CC12M Retrieval | Base | CLIP embedding retrieval |
| Bing Search | + Gain | Web search provides wider coverage |
| Multi-prompt Variants | + Gain | Increases query diversity |

### Key Findings
- C2B can discover novel biases not covered by dataset annotations (e.g., the "age" bias in CelebA, which was not annotated in the original 40 attributes).
- Bing search performs better than CC12M retrieval due to its broader domain coverage.
- The accuracy of retrieved images (i.e., whether they match the query description) is a key bottleneck for C2B's performance.
- Class-specific bias lists are more effective than a globally unified list.

## Highlights & Insights
- **A New Paradigm for Zero-Annotation Bias Detection**: Shifts from "collecting data before bias detection" to "reasoning about potential biases and then collecting data on-demand," significantly lowering the barrier to bias detection.
- **LLM as a Bias Hypothesis Generator**: Leverages the world knowledge of LLMs to replace domain experts, automating the process of bias hypothesis formulation.
- **Modular Design**: The LLM, retrieval engine, and target model are fully decoupled, allowing seamless upgrades of each component as technologies advance.

## Limitations & Future Work
- Dependency on retrieved image quality, where incorrectly retrieved images introduce noise to the bias scores.
- LLMs may miss certain biases (such as culture-specific factors not considered by the LLM).
- Currently validated only on classification tasks; bias detection for other tasks like detection/segmentation needs further expansion.
- The commercial API of Bing search increases usage costs.

## Related Work & Insights
- **vs B2T**: B2T requires annotated data to identify failed samples and then extract bias keywords, whereas C2B is completely unsupervised.
- **vs OpenBias**: OpenBias detects biases in generative models, while C2B detects biases in discriminative models, making the methodologies complementary.
- The framework can be transferred to LLM bias detection—using LLMs to generate bias test cases.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Automatically detects bias with zero annotations for the first time; the problem definition is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two datasets, CelebA and ImageNet-X, containing multiple evaluation protocols.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, systematic description of the method.
- Value: ⭐⭐⭐⭐⭐ Highly practical, lowering the barrier for fairness auditing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification](classifier-guided_clip_distillation_for_unsupervised_multi-label_classification.md)
- [\[AAAI 2026\] Argumentative Debates for Transparent Bias Detection](../../AAAI2026/social_computing/argumentative_debates_for_transparent_bias_detection_technic.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](../../ACL2025/social_computing/biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)
- [\[NeurIPS 2025\] Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector](../../NeurIPS2025/social_computing/any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)
- [\[ACL 2025\] GG-BBQ: German Gender Bias Benchmark for Question Answering](../../ACL2025/social_computing/gg-bbq_german_gender_bias_benchmark_for_question_answering.md)

</div>

<!-- RELATED:END -->
