---
title: >-
  [Paper Note] Genixer: Empowering Multimodal Large Language Model as a Powerful Data Generator
description: >-
  [ECCV2024][Multimodal VLM][Multimodal Large Language Models (MLLMs)] The Genixer data generation pipeline is proposed to train an MLLM itself as a data generator, automatically generating high-quality visual instruction tuning data without relying on GPT-4V. The generated 915K VQA and 350K REC data respectively improve the performance of LLaVA1.5 and Shikra across multiple benchmarks.
tags:
  - "ECCV2024"
  - "Multimodal VLM"
  - "Multimodal Large Language Models (MLLMs)"
  - "Visual Instruction Tuning"
  - "Data Generation"
  - "Data Filtering"
  - "VQA"
  - "Visual Grounding"
date: 2026-05-08
content_hash: f0a4096055cf69cb
---

# Genixer: Empowering Multimodal Large Language Model as a Powerful Data Generator

**Conference**: ECCV2024  
**arXiv**: [2312.06731](https://arxiv.org/abs/2312.06731)  
**Code**: [zhaohengyuan1/Genixer](https://github.com/zhaohengyuan1/Genixer)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Large Language Models (MLLMs), Visual Instruction Tuning, Data Generation, Data Filtering, VQA, Visual Grounding

## TL;DR

The Genixer data generation pipeline is proposed to train an MLLM itself as a data generator, automatically generating high-quality visual instruction tuning data without relying on GPT-4V. The generated 915K VQA and 350K REC data respectively improve the performance of LLaVA1.5 and Shikra across multiple benchmarks.

## Background & Motivation

Currently, the training of Multimodal Large Language Models (MLLMs) heavily relies on visual instruction tuning data, which are mainly obtained through two paths:

**Conversion of Existing VL Datasets**: For example, InstructBLIP converts traditional VQA/Caption datasets into instruction formats. However, the images mostly originate from COCO, which limits diversity and leads to insufficient generalization capability.

**GPT-4 Assisted Generation**: Models like LLaVA, Shikra, and ShareGPT4V rely on GPT-4 to generate instruction data. This exhibits two main issues: (a) high financial costs for large-scale generation; (b) poor generation quality of GPT-4V in complex tasks like Referential Expression Comprehension (REC), where it fails to output correct bounding boxes.

**Core Motivation**: Given that existing MLLMs already possess strong multimodal comprehension capabilities, can an MLLM itself be trained as a data generator to break free from the reliance on GPT-4V? This would achieve zero additional cost and provide the flexibility to generate high-quality instruction data for arbitrary unlabeled images.

## Method

### Overall Architecture

The Genixer pipeline consists of four key steps:

1. **Instruction Data Collection**: Selecting 9 representative VL tasks, categorized into common tasks (Common VQA, Adv VQA, MC VQA, Multi-turn Dialogue) and grounding tasks (REC, REG, PointQA, Q→CBoxA, Referential Dialogue).
2. **Instruction Template Design**: Proposing two-level instruction templates that support two modes: Task-agnostic (allowing the model to freely generate any type of data) and Task-specific (controlling the generated task type through specific instructions).
3. **Empowering MLLMs**: Training GenixerL (common task generator) using LLaVA1.5 as the backbone, and GenixerS (grounding task generator) using Shikra as the backbone.
4. **Data Generation and Filtering**: Feeding 1.4M unlabeled images into the generators, and then filtering out high-quality samples through an automatic filtering framework.

### Key Designs

**Two-level Instruction Templates**:

- **Generic Instruction**: Randomly sampled from 58 human-written instructions, such as "Please provide a clear and direct question and answer after examining the image", allowing the model to freely decide which task type to generate.
- **Specific Instruction**: Such as "This is a Common VQA task", to precisely control the output task type.
- **Control Constant $\tau$**: Controls the proportion of samples that use only generic instructions during training (Common VQA $\tau=0.2$, MC VQA $\tau=0.5$, etc.) to balance tasks with different data scales.

**Two-stage Training (GenixerS)**: Since tasks like Referential Dialogue have extremely small datasets (only 1.8K), GenixerS adopts a two-stage training strategy. The first stage focuses on REC/REG data generation, while the second stage introduces PointQA, Q→CBoxA, and RD, while downsampling the REC/REG data volume to maintain balance.

**Data Filtering Framework**:

- **Fuyu-driven Filtering (Common Tasks)**: Feeding the generated QA pairs into Fuyu-8B to calculate the probability $P(Y_r)$ of the model predicting "Yes". A threshold of $\lambda=0.7$ is set to filter out low-quality samples. (1.4M $\rightarrow$ 915K).
- **CLIP-driven Filtering (Grounding Tasks)**: A three-step coarse-to-fine filtering process: (1) checking formatting via regular expressions; (2) removing boxes with width/height $< 50$; (3) using OpenCLIP-L to calculate the similarity between the textual description and the image region, with a threshold of 0.6. (1.4M $\rightarrow$ 350K).

### Loss & Training

Standard autoregressive language modeling loss, where the training objective is to generate question-answer pairs given an image and instructions:

$$\max \sum_{i=1}^{L} \log p_\theta(x_i | X_G, X_S, X_I, X_{o,<i})$$

Where $X_G$ and $X_S$ are generic and specific instructions respectively, $X_I$ is the image, $X_o$ is the complete output sequence (question + answer), and $\theta$ represents the trainable parameters. GenixerL is trained using AdamW with $\text{lr}=1\text{e-}5$, $\text{batch size}=128$, for 1 epoch (approx. 14 hours). The two stages of GenixerS use $\text{lr}=3\text{e-}5$ / $1\text{e-}5$ and $\text{batch size}=128$ / $64$, respectively.

## Key Experimental Results

### Main Results: Common Tasks (LLaVA1.5 + Genixer-915K)

| Method | VQAv2 | GQA | VizWiz | SQA-I | TextVQA | POPE | MME | MMB | MMB-CN | SEED-I | LLaVA-W | MM-Vet |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5 (7B) | 78.5 | 62.0 | 50.0 | 66.8 | 58.2 | 85.9 | 1465.0 | 64.3 | 58.3 | 66.2 | 65.4 | 31.1 |
| **+Genixer-915K** | **79.1** | **63.1** | **53.8** | **69.7** | **59.0** | **87.3** | **1502.7** | **65.3** | **59.4** | **66.6** | 64.0 | 30.1 |
| Δ | +0.6 | +1.1 | **+3.8** | **+2.9** | +0.8 | +1.4 | **+37.7** | +1.0 | +1.1 | +0.4 | -1.4 | -1.0 |

**Performance gains are achieved on 10 out of 12 benchmarks**, with notable improvements on VizWiz (+3.8%), ScienceQA (+2.9%), and MME (+37.7 points).

### Main Results: Grounding Tasks (Shikra + Genixer-350K)

| Method | RefCOCO val | RefCOCO test-A | RefCOCO test-B | RefCOCO+ val | RefCOCO+ test-A | RefCOCO+ test-B | RefCOCOg val | RefCOCOg test | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Shikra | 87.01 | 90.61 | 80.24 | 81.60 | 87.36 | 72.12 | 82.27 | 82.19 | 82.92 |
| **+Genixer-350K** | **87.48** | **91.05** | **81.77** | **81.89** | **87.43** | **73.14** | 81.99 | **83.15** | **83.49** |

**Performance gains are achieved on 7 out of 8 REC test sets**, with an average improvement of +0.57%.

### Ablation Study: Data Scale Effect

| Dataset | VQAv2 | GQA | VizWiz | SQA-I | POPE | SEED-I |
|---|---|---|---|---|---|---|
| Baseline | 78.5 | 62.0 | 50.0 | 66.8 | 85.9 | 66.2 |
| Genixer-300K | 79.0 | 62.9 | 52.7 | 68.5 | 87.1 | 65.8 |
| Genixer-610K | 79.0 | 63.1 | 53.7 | 69.2 | 87.2 | 66.2 |
| Genixer-915K | 79.1 | 63.1 | 53.8 | 69.7 | 87.3 | 66.6 |

Performance scales positively with the volume of data, showing steady improvements and demonstrating the reliable quality of the synthetic data.

### Ablation Study: Impact of Filtering Threshold $\lambda$

| Threshold $\lambda$ | Data Volume | VQAv2 | GQA | VizWiz | SQA-I | POPE | SEED-I |
|---|---|---|---|---|---|---|---|
| 0 (No filtering) | 1.4M | 79.0 | 62.9 | 53.5 | 69.6 | 87.1 | 66.2 |
| 0.5 | 1.1M | 79.1 | 63.1 | 53.2 | 69.1 | 86.9 | 66.4 |
| **0.7** | **0.9M** | **79.1** | **63.1** | **53.8** | **69.7** | **87.3** | **66.6** |

Higher threshold $\rightarrow$ fewer but higher-quality data $\rightarrow$ better performance. **Data quality is more important than quantity**.

### Key Findings

1. **High quality of data generated by GenixerL**: Evaluated by Fuyu-8B, the accuracy of Common VQA is 82.4%, MC VQA is 87.8%, and MD is 82.5%, with average probabilities all $> 0.8$.
2. **GenixerS outperforms GPT-4V**: For REC data generation, GPT-4V fails to output bounding boxes correctly, whereas GenixerS succeeds. In user preference studies, users preferred Genixer on REC tasks.
3. **Synthetic data alleviates hallucination**: The POPE benchmark performance increases from 85.9 to 87.3, suggesting that synthetic data helps mitigate model hallucinations.
4. **Comprehensive improvement after joint training with GenixerL**: Jointly training with LLaVA1.5 fine-tuning data leads to substantial and comprehensive improvements across 6 benchmarks (e.g., VizWiz +4.1%, POPE +1.6%).

## Highlights & Insights

1. **First to demonstrate that MLLMs can bootstrap data generation**: Without relying on commercial models like GPT-4, existing open-source MLLMs can become effective data generators with proper training, opening up possibilities for "self-evolution."
2. **Ingenious design of two-level instruction templates**: By controlling the constant $\tau$ and the two-level instructions, the system elegantly coordinates switching between task-agnostic and task-specific generation modes.
3. **Two complementary data filtering frameworks**: Fuyu-driven (probability threshold-based) and CLIP-driven (vision-language similarity-based) filtering are customized for common and grounding tasks respectively, which is automated and effective.
4. **Rigorous experimental design**: Covers six dimensions including statistical analysis, human evaluation, training evaluation, ablation studies, visualization, and user studies, making it highly persuasive.
5. **Succeeding where GPT-4V fails**: The generation quality of GenixerS on REC tasks is significantly superior to GPT-4V, demonstrating the advantages of task-specific fine-tuning.

## Limitations & Future Work

1. **Limited scale of LLMs**: Only validated on 7B models. Larger-scale models such as 13B or 34B have not been tested, which could potentially yield greater benefits.
2. **Limited image scale**: Only 1.4M image corpora were used, without extending to larger image libraries such as LAION-2B, whereas ablation studies have demonstrated that scaling up data translates to larger performance gains.
3. **Difficulty in filtering open-ended tasks**: For complex open-ended tasks like Referential Dialogue, automatic filtering and quality evaluation remain challenging.
4. **Insignificant drops in some metrics**: LLaVA-Bench and MM-Vet show slight decreases (approx. 1 point), which might be attributed to biases introduced by using GPT-4 as an evaluator.
5. **Genixer-350K dataset comes solely from REC tasks**, without covering large-scale generation across all 5 categories of grounding tasks.

## Related Work & Insights

- **Relationship with the LLaVA series**: Genixer is based on the LLaVA1.5 architecture, but the training objective shifts from "understanding" to "generating data." The generated data, in turn, boosts the performance of LLaVA1.5, forming a self-reinforcing closed loop.
- **Similarities and differences with Self-Instruct**: Self-Instruct in the LLM domain uses models to generate textual instruction data. Genixer extends this idea to the multimodal domain for the first time, introducing additional image-conditional generation and automated filtering.
- **Comparison with ShareGPT4V**: ShareGPT4V still relies on GPT-4V to generate caption data, whereas Genixer completely eliminates reliance on commercial models.
- **Insight on data filtering**: The Fuyu-driven probability threshold approach represents a general paradigm—using a stronger/different model to validate the output quality of the generative model, which is highly extensible to other scenarios.
- **Insight on data generation for grounding tasks**: GPT-4V fails to reliably generate bounding box coordinates, whereas task-specific fine-tuning addresses this limitation, indicating that general-purpose foundation models are not necessarily optimal for all downstream tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ (First systematic exploration of MLLMs as multimodal data generators, creative design of two-level instruction templates)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Very comprehensive evaluation on 12+8 benchmarks, including statistical/human/ablation/user studies)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures/tables, and complete pipeline description, though with slightly redundant symbols)
- Value: ⭐⭐⭐⭐ (Provides a viable pathway for data generation without relying on commercial models, offering high practical value for resource-constrained groups)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Merlin: Empowering Multimodal LLMs with Foresight Minds](merlin_empowering_multimodal_llms_with_foresight_minds.md)
- [\[ACL 2025\] Error-driven Data-efficient Large Multimodal Model Tuning](../../ACL2025/multimodal_vlm/error-driven_data-efficient_large_multimodal_model_tuning.md)
- [\[ECCV 2024\] CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios](cat_enhancing_multimodal_large_language_model_to_answer_questions_in_dynamic_aud.md)
- [\[CVPR 2026\] Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping](../../CVPR2026/multimodal_vlm/venus_benchmarking_and_empowering_multimodal_large_language_models_for_aesthetic.md)
- [\[ICML 2026\] Model-Dowser: Data-Free Importance Probing to Mitigate Catastrophic Forgetting in Multimodal Large Language Models](../../ICML2026/multimodal_vlm/model-dowser_data-free_importance_probing_to_mitigate_catastrophic_forgetting_in.md)

</div>

<!-- RELATED:END -->
