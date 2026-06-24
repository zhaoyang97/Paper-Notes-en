---
title: >-
  [Paper Note] Vision-Language Model Selection and Reuse for Downstream Adaptation
description: >-
  [ICML 2025][Multimodal VLM][Model Selection] Proposes the Model Label Learning (MLL) paradigm, which performs offline "labeling" of 49 pre-trained VLMs (describing each model's capability across different visual concepts) by constructing a semantic graph. For a new task, it selects and ensembles the most suitable models via semantic matching, achieving data-efficient, computationally efficient, and scalable VLM selection and reuse.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Model Selection"
  - "VLM"
  - "Model Hub"
  - "Semantic Graph"
  - "Ensemble"
  - "Model Label Learning"
date: 2026-05-08
content_hash: 886c3579bf36a35d
---

# Vision-Language Model Selection and Reuse for Downstream Adaptation

**Conference**: ICML 2025  
**arXiv**: [2501.18271](https://arxiv.org/abs/2501.18271)  
**Code**: Not released  
**Area**: VLM Model Selection, Model Reuse, Zero-Shot Vision Tasks  
**Keywords**: Model Selection, VLM, Model Hub, Semantic Graph, Ensemble, Model Label Learning

## TL;DR

Proposes the Model Label Learning (MLL) paradigm, which performs offline "labeling" of 49 pre-trained VLMs (describing each model's capability across different visual concepts) by constructing a semantic graph. For a new task, it selects and ensembles the most suitable models via semantic matching, achieving data-efficient, computationally efficient, and scalable VLM selection and reuse.

## Background & Motivation

### Background

**Background**: The number of open-source VLMs (such as CLIP and its variants) is rapidly growing, with the open-clip library containing over 100+ models. However:

### Limitations of Prior Work

**Limitations of Prior Work**: No single VLM is optimal across all tasks: different models exhibit significant performance variance across different tasks or even across different categories within the same task.

### Key Challenge

**Key Challenge**: Evaluating all candidate models is impractical due to time and data constraints.

**Core Idea**: Existing model selection methods (**NCE, LEEP, LogME**) target **single-modal models** and are inapplicable to VLMs.

The first VLM selection work, LOVM, proposes evaluating VLMs using text data, but relies on real performance on ImageNet, failing when downstream tasks have a domain shift from generic datasets.

## Method

### Overall Architecture — Three Modules of MLL

**Module 1: Model Labeling**

Constructing the **semantic graph $\mathcal{G}$**:
- Nodes: WordNet synsets (>9,000 visual concepts)
- Edges: Hypernymy-hyponymy relations
- Each node is associated with representative image samples $X_v$
- Caption: "{synset name} which is {synset definition}"

Each VLM $f_m$ is pre-evaluated on the semantic graph to generate model labels:

$$s_{m,x}^v = \text{sim}(\mathcal{I}_m(x), \mathcal{T}_m(d_v)), \quad S_m = \{s_m^v | v \in V_\mathcal{G}\}$$

The labels describe the model's capability distribution across various visual concepts. **This process is independent of the target task and is pre-computed only once.**

**Module 2: Model Selection**

Given the target task categories $Y_T$:
1. GPT-4 generates extended descriptions $D_T$ for each category.
2. A language model calculates the similarity between $D_T$ and $D_\mathcal{G}$, selecting the top-$k$ semantic nodes for each category.
3. Build the transition matrix $Z$.
4. Estimate the precision of each model on each target category using the model labels: $p_{m,y} = \sum_v p_{m,v} \cdot z_{vy}$
5. Combine category-level precision and global precision: $r_{m,y} = \alpha \cdot p_{m,y} + \frac{1-\alpha}{|Y_T|}\sum_{y'} p_{m,y'}$

**Module 3: Model Reuse**

An ensemble predictor is constructed by selecting the top-$k$ models for each category:

$$p_y^k(x) = \sum_{f_m \in \mathcal{F}_y^k} w_{m,y} \cdot \frac{\exp(\text{sim}(\mathcal{I}_m(x), \mathcal{T}_m(y)))}{\sum_{y'} \exp(\text{sim}(\mathcal{I}_m(x), \mathcal{T}_m(y')))}$$

The weight $w_{m,y}$ is based on the entropy of the prediction probability—reducing the weight for models with high confidence (which may be overconfident).

Final prediction: $\hat{y} = \arg\max_y p_y^k(x)$

## Key Experimental Results

### Benchmark

49 pre-trained VLMs + 17 downstream datasets.

### Single Model Selection (k=1)

### Main Results

| Method | CIFAR100 | Flowers102 | MNIST | FER2013 | StanfordCars | Average |
|------|---------|-----------|-------|---------|-------------|------|
| INB (ImageNet-best) | 0.860 | 0.876 | 0.796 | 0.286 | 0.949 | 0.643 |
| ModelGPT | 0.860 | 0.876 | 0.565 | 0.401 | 0.949 | 0.637 |
| **MLL** | **0.877**| **0.891** | **0.810** | **0.493** | **0.957** | **0.662** |

### 3-Model Ensemble (k=3)

Average accuracy across 17 datasets: MLL achieves the best performance, outperforming the INB and ModelGPT baselines.

### Key Findings

- The optimal model on ImageNet is not necessarily the best on specific downstream tasks (e.g., massive gap on FER2013).
- Selecting different models for each category (fine-grained selection) performs better than selecting a single global model.
- A larger model hub leads to better MLL performance (strong scalability).

## Highlights & Insights

1. **Target-task-independent labeling**: Model labels are calculated once during upload, eliminating the need to run candidate models during the selection process.
2. **Fine-grained category-level selection**: Different categories can select different models, making full use of the strengths of each model.
3. **Scalability**: The semantic graph can continuously expand its nodes; the model hub capability scales with its size.
4. **Comprehensive Benchmark**: Systematically evaluating 49 models × 17 datasets promotes research in the field of VLM selection.

## Limitations & Future Work

- The coverage of the semantic graph depends on WordNet, which might miss domain-specific concept definitions.
- Generating captions with GPT-4 introduces a dependency on closed-source models.
- Model labels may be inaccurate when the number of samples per category is small.
- Ensembling multiple models increases inference overhead.

## Related Work & Insights

- Model Selection (NCE, LEEP, LogME, Model Spider)
- LOVM (the first VLM selection work)
- Learnware paradigm (model specification)
- VLM model libraries (open-clip, HuggingFace)

## Rating

⭐⭐⭐⭐ — Novel concept of "model labels" that shifts model selection from online evaluation to offline pre-computation and semantic matching. The large-scale benchmark of 49 VLMs × 17 datasets holds independent value. Practical contributions outweigh theoretical depth.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Mordal: Automated Pretrained Model Selection for Vision Language Models](../../ICLR2026/multimodal_vlm/mordal_automated_pretrained_model_selection_for_vision_language_models.md)
- [\[NeurIPS 2025\] Metacognitive Sensitivity for Test-Time Dynamic Model Selection](../../NeurIPS2025/multimodal_vlm/metacognitive_sensitivity_for_test-time_dynamic_model_selection.md)
- [\[CVPR 2026\] Vision-Language Model Guided Source-Free Domain Adaptation via Optimal Transport](../../CVPR2026/multimodal_vlm/vision-language_model_guided_source-free_domain_adaptation_via_optimal_transport.md)
- [\[CVPR 2025\] Realistic Test-Time Adaptation of Vision-Language Models](../../CVPR2025/multimodal_vlm/realistic_test-time_adaptation_of_vision-language_models.md)
- [\[ICML 2025\] ELEMENTAL: Interactive Learning from Demonstrations and Vision-Language Models for Reward Design in Robotics](elemental_interactive_learning_from_demonstrations_and_vision-language_models_fo.md)

</div>

<!-- RELATED:END -->
