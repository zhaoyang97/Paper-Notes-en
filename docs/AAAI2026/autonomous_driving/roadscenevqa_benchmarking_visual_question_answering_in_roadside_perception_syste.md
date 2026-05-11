---
title: >-
  [Paper Note] RoadSceneVQA: Benchmarking Visual Question Answering in Roadside Perception Systems for Intelligent Transportation System
description: >-
  [AAAI 2026][Autonomous Driving][Visual Question Answering] This paper introduces RoadSceneVQA—the first large-scale visual question answering dataset for roadside perception scenarios (34…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "Visual Question Answering"
  - "Roadside Perception"
  - "Multimodal Large Language Models"
  - "Chain-of-Thought Reasoning"
  - "Vision-Language Fusion"
date: 2026-05-08
content_hash: 4b0a3be7398f6aec
---

# RoadSceneVQA: Benchmarking Visual Question Answering in Roadside Perception Systems for Intelligent Transportation System

**Conference**: AAAI 2026
**arXiv**: [2511.18286](https://arxiv.org/abs/2511.18286)
**Code**: [github.com/GuanRunwei/RS-VQA](https://github.com/GuanRunwei/RS-VQA)
**Area**: Autonomous Driving / Intelligent Transportation
**Keywords**: Visual Question Answering, Roadside Perception, Multimodal Large Language Models, Chain-of-Thought Reasoning, Vision-Language Fusion

## TL;DR

This paper introduces RoadSceneVQA—the first large-scale visual question answering dataset for roadside perception scenarios (34,736 QA pairs)—and proposes the RoadMind model, which significantly improves lightweight MLLM performance on traffic scene reasoning through CogniAnchor Fusion (CAF) and Assisted Decoupled Chain-of-Thought (AD-CoT), enabling a 0.9B-parameter model to surpass 8B-parameter counterparts.

## Background & Motivation

### Advantages and Current State of Roadside Perception

Roadside perception holds unique advantages over onboard perception: a top-down viewpoint enables clearer observation of traffic participant states and behaviors, as well as more comprehensive scene-level understanding. However, existing roadside perception systems predominantly focus on instance-level automated tasks (detection, tracking, trajectory prediction, and traffic flow forecasting), leading to the following critical issues:

**Lack of human-in-the-loop perception**: Emphasis is placed solely on instance-level recognition, with insufficient event-level or holistic scene-level understanding.

**Limited scalability and interpretability**: Systems lack the flexibility to recognize unforeseen objects and events in complex environments.

**Absence of semantic reasoning capability**: Existing benchmarks measure only perceptual accuracy and cannot evaluate whether models understand implicit traffic regulations.

### Limitations of Existing VQA Datasets

| Issue | Description |
|-------|-------------|
| Predominantly ego-vehicle perspectives | Talk2Car, NuScenes-QA, DriveLM, etc. are all driving-centric |
| Lack of reasoning questions | Most datasets focus only on explicit attribute recognition without traffic rule reasoning |
| Roadside VQA is nascent | TUM-VideoQA is roadside-based but contains no reasoning questions |

### Core Challenges

RoadSceneVQA is the first benchmark to require models to answer questions such as "Is the pedestrian violating traffic regulations?"—questions that demand the integration of signal states, spatial context, and behavioral dynamics. This requires three converging capabilities:
- Visual-semantic grounding (associating pedestrian location with signal state)
- Internalization of traffic regulation knowledge
- Counterfactual causal reasoning (inferring "would the cyclist still be violating the rule if the light were green?")

## Method

### Overall Architecture

The RoadMind model consists of three core components:

1. **Adaptive Visual Encoding**: Decomposes input images into patch sequences and a globally downsampled image; extracts high-level features via InternViT with Pixel Shuffle for enhanced feature density.
2. **CogniAnchor Fusion (CAF)**: A cognitive anchor fusion module enabling language-driven visual attention.
3. **AD-CoT**: Assisted Decoupled Chain-of-Thought, leveraging GPT-4o-generated reasoning processes to augment lightweight models.

Final output: $\mathbf{A} = \mathtt{Qwen2.5}(\mathtt{Concat}(\mathtt{CAF}(\mathbf{V}, \mathbf{T}^{OC}), \mathbf{T}^{OC}))$

### Key Designs

#### 1. Collaborative Human-Machine Annotation System (CH-MA)

**Function**: Constructs a high-quality, scalable VQA annotation framework.

**Three-stage pipeline**:
- **Stage A**: QwenVL-Max generates 4 candidate QA pairs per roadside image based on customized prompts; annotators select the highest-quality pair.
- **Stage B**: Annotators correct and refine the selected QA pairs to ensure factual accuracy, contextual alignment, and linguistic clarity.
- **Stage C**: A quality control panel of 7 annotators conducts majority-vote review; only samples passing the majority threshold are included in the dataset.

**Design Motivation**: Addresses four major challenges—subjective bias in human question formulation, isolated focus on individual participants that neglects contextual cues, question diversity depending on annotator expertise, and lack of naturalness in template-based generation.

#### 2. CogniAnchor Fusion (CAF)

**Function**: Resolves the visual-language interaction issues caused by naive token concatenation in MLLMs.

**Mechanism**: Inspired by human visual-language co-cognitive processing, CAF pre-anchors potential regions of interest in a text-driven manner, employing a linear attention mechanism for efficient language-guided image attention:

$$\mathtt{AW}(\mathbf{Q_i}^T, \mathbf{K}) = [\phi(\mathbf{Q_i}^T)^\top \phi(\mathbf{K_1}), \ldots, \phi(\mathbf{Q_i}^T)^\top \phi(\mathbf{K_N})]^\top - \frac{1}{N}\sum_{s=1}^N \phi(\mathbf{Q_i}^T)^\top \phi(\mathbf{K_s}) + \frac{1}{N}$$

By reordering computation (aggregating key-value pairs before interacting with queries), complexity is reduced from $O(N^2)$ in standard Softmax attention to $O(N)$.

**Design Motivation**: Standard token concatenation suffers from two deficiencies: (1) irrelevant visual tokens (e.g., background noise) interfere with target localization of key text tokens; (2) imbalanced information interaction, where dominant visual features suppress text signals. CAF adopts the InLine Attention technique to address the non-injectivity problem in linear attention.

#### 3. Assisted Decoupled Chain-of-Thought (AD-CoT)

**Function**: Enhances the reasoning capability of lightweight roadside MLLMs through knowledge distillation and cognitive transfer.

**Mechanism**:
1. Input images and CoT prompts are fed into GPT-4o to generate auxiliary reasoning context (including perceptual reasoning processes and conclusive answers).
2. The reasoning process is concatenated with the original question as an augmented input to RoadMind.
3. GPT-4o's conclusive answers are paired with human-annotated ground truth to construct a multi-task learning objective.

**Loss Function**:

$$L_{\text{MTL}} = \frac{1}{\sigma_{\text{hard}}^2} \sum_{l=1}^L \log p(\mathbf{y}_l^{\text{hard}} | \mathbf{y}_{<l}^{\text{hard}}, \mathbf{x}, \mathbf{q}) + \frac{1}{\sigma_{\text{soft}}^2} \sum_{l=1}^{\min(L,L')} D_{\text{KL}}(p_l^{\text{GPT}} || \hat{p}_l) + \log \sigma_{\text{hard}} + \log \sigma_{\text{soft}}$$

where $\sigma_{\text{hard}}$ and $\sigma_{\text{soft}}$ are two learnable uncertainty weights.

**Design Motivation**: Lightweight roadside MLLMs have limited reasoning capacity. GPT-4o serves as a soft supervisory prior to enable knowledge transfer, allowing smaller models to acquire the reasoning capability of larger ones.

### Loss & Training

- Pretrained models are fine-tuned for 1 epoch with an initial learning rate of 1e-5.
- The visual encoder is frozen; the LLM and MLP projector are unfrozen.
- Input images are resized to 448×448 with a maximum sequence length of 16,384.
- AdamW optimizer with weight decay 0.05, cosine scheduler with warm-up ratio 0.03.
- Training uses 4 × A100 GPUs with a per-GPU batch size of 1.

## Key Experimental Results

### Main Results

#### Overall Performance on RoadSceneVQA

| Model | LLM | Params | Exact Match | CIDEr | SPICE | GPT-Score |
|-------|-----|--------|-------------|-------|-------|-----------|
| MiniCPM-o 2.6 (no fine-tuning) | LLaMA3 | 8B | 0.021 | 0.661 | 0.124 | 0.428 |
| InternVL3 | Qwen 2.5 | 0.9B | 0.142 | 1.656 | 0.170 | 0.403 |
| **RoadMind** | **Qwen 2.5** | **0.9B** | **0.144** | **1.867** | **0.188** | **0.440** |
| InternVL3 | Qwen 2.5 | 2B | 0.151 | 1.834 | 0.201 | 0.465 |
| **RoadMind** | **Qwen 2.5** | **2B** | 0.142 | 1.705 | **0.219** | **0.489** |
| Qwen2.5-VL | Qwen2.5 | 7B | 0.152 | 1.689 | 0.213 | 0.497 |
| InternVL3 | Vicuna | 8B | 0.161 | 1.735 | 0.208 | 0.532 |
| **RoadMind** | **Qwen 2.5** | **8B** | 0.157 | 1.836 | **0.221** | **0.554** |

RoadMind-0.9B achieves a GPT-Score of 0.440, surpassing both the untuned MiniCPM-o 2.6 (8B, 0.428) and the fine-tuned MobileVLM v2 (1.7B, 0.417).

#### Generalization Performance on CODA-LM

| Model | GTS↑ | Overall↑ | Vehicle↑ | VRU↑ | Sign↑ | STS↑ |
|-------|------|----------|----------|------|-------|------|
| InternVL1.5-20B | 38.38 | 61.53 | 63.77 | 53.14 | 50.57 | 41.18 |
| **RoadMind-8B** | **48.50** | **70.65** | **74.25** | **59.78** | 47.43 | **54.28** |

RoadMind-8B surpasses InternVL1.5-20B on CODA-LM, demonstrating strong cross-scenario generalization.

### Ablation Study

#### Transfer Performance of CAF and AD-CoT (MiniCPM-o 2.6, GPT-Score)

| Configuration | GPT-Score | Note |
|---------------|-----------|------|
| No fine-tuning | 0.428 | Baseline |
| LoRA | 0.452 | +5.6% |
| SFT | 0.527 | +23.1% |
| SFT + CAF | 0.533 | CAF contributes +1.1% |
| **SFT + CAF + AD-CoT** | **0.549** | AD-CoT contributes +3.0% |

#### Comparison of Vision-Language Fusion Methods

| Fusion Method | Params | FLOPs | ROUGE-L | METEOR | SPICE |
|---------------|--------|-------|---------|--------|-------|
| Concat only | - | - | 0.366 | 0.397 | 0.187 |
| LCA+Concat | 1.049M | 371.86M | 0.397 | 0.386 | 0.201 |
| CA+Concat | 1.063M | 495.41M | 0.418 | 0.422 | 0.217 |
| **CAF+Concat** | **0.924K** | **61.08M** | **0.425** | **0.411** | **0.221** |

CAF achieves the best performance with the fewest parameters (only 924) and lowest computation (61M FLOPs).

#### Contribution of AD-CoT Components

| CoT Variant | Perception METEOR | Perception GPT-Score | Reasoning METEOR | Reasoning GPT-Score |
|-------------|-------------------|----------------------|------------------|----------------------|
| **AD-CoT (full)** | **0.420** | **0.568** | **0.339** | **0.445** |
| Original question input only | 0.392 | 0.523 | 0.301 | 0.401 |
| GT training only | 0.428 | 0.549 | 0.325 | 0.439 |
| GPT answer training only | 0.387 | 0.491 | 0.331 | 0.395 |
| MCoT | 0.422 | 0.536 | 0.319 | 0.431 |

### Key Findings

1. **Significant perception–reasoning gap**: RoadMind-8B achieves GPT-Scores of 0.53–0.60 on perception questions, but only 0.34–0.45 on reasoning questions, indicating that high-level reasoning remains a substantial challenge.
2. **CAF achieves optimal performance with minimal parameters**: Only 924 parameters and 61M FLOPs, far below conventional cross-attention at 1M+ parameters / 495M FLOPs.
3. **Multi-task learning is critical for AD-CoT**: Joint training with both GT answers and GPT-4o answers outperforms either used alone.
4. **Small models can surpass large models through deliberate design**: RoadMind-0.9B > MiniCPM-o 2.6 (8B).

## Highlights & Insights

- **First reasoning-oriented roadside VQA dataset**: Shifts evaluation from perceptual recognition toward regulation-aware cognitive reasoning.
- **Efficient human-machine collaborative annotation**: The CH-MA system achieves a balance between annotation quality and efficiency.
- **Elegant CAF design**: Linear attention combined with the InLine technique resolves non-injectivity, achieving $O(N)$ complexity.
- **Sophisticated knowledge distillation**: GPT-4o's reasoning chain serves as input augmentation while its conclusive answers function as soft labels, enabling dual-channel knowledge transfer.

## Limitations & Future Work

- The dataset covers only 26 scenes (based on Rope3D), limiting scene diversity.
- Performance on reasoning questions still has considerable room for improvement.
- AD-CoT relies on GPT-4o for chain-of-thought generation, incurring higher training costs.
- The current approach is limited to single-frame images, without leveraging video temporal information.
- Reasoning about traffic regulations from the roadside perspective requires richer injection of regulatory domain knowledge.

## Related Work & Insights

- **DriveLM/CODA-LM**: Representative works in onboard VQA; RoadSceneVQA extends the VQA paradigm to the roadside setting.
- **InLine Attention**: A key technique for efficient linear attention, adopted by CAF.
- **CoT distillation**: The paradigm of using GPT-4o as a teacher model is becoming increasingly prevalent in autonomous driving research.

## Rating

- Novelty: ⭐⭐⭐⭐ (First roadside reasoning VQA benchmark + novel CAF fusion design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multi-scale models, multiple datasets, comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures and tables)
- Value: ⭐⭐⭐⭐ (Dataset and method make significant contributions to roadside intelligent transportation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] STRIDE-QA: Visual Question Answering Dataset for Spatiotemporal Reasoning in Urban Driving Scenes](stride-qa_visual_question_answering_dataset_for_spatiotemporal_reasoning_in_urba.md)
- [\[AAAI 2026\] ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts](expertad_enhancing_autonomous_driving_systems_with_mixture_of_experts.md)
- [\[AAAI 2026\] LUCID: Learning-Enabled Uncertainty-Aware Certification of Stochastic Dynamical Systems](lucid_learning-enabled_uncertainty-aware_certification_of_stochastic_dynamical_s.md)
- [\[CVPR 2026\] Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule](../../CVPR2026/autonomous_driving/perception_characteristics_distance_measuring_stability_and_robustness_of_percep.md)
- [\[AAAI 2026\] SparseCoop: Cooperative Perception with Kinematic-Grounded Queries](sparsecoop_cooperative_perception_with_kinematic-grounded_queries.md)

</div>

<!-- RELATED:END -->
