---
title: >-
  [Paper Note] Vamos: Versatile Action Models for Video Understanding
description: >-
  [ECCV 2024][Video Understanding][Text Representation] Proposes the Vamos framework, which uses Large Language Models as reasoners to flexibly unify visual embeddings and general text descriptions as video representations. It discovers that text-only representations consistently achieve competitive or even superior performance across multiple video understanding benchmarks. Furthermore, it designs a Token Bottleneck Model to achieve interpretable evidence selection and a 5x in…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Text Representation"
  - "Large Language Models"
  - "Video Question Answering"
  - "Action Forecasting"
  - "Token Bottleneck"
date: 2026-05-08
content_hash: ac348a30aa53375b
---

# Vamos: Versatile Action Models for Video Understanding

**Conference**: ECCV 2024  
**arXiv**: [2311.13627](https://arxiv.org/abs/2311.13627)  
**Code**: [https://brown-palm.github.io/Vamos/](https://brown-palm.github.io/Vamos/)  
**Area**: Video Understanding  
**Keywords**: Text Representation, Large Language Models, Video Question Answering, Action Forecasting, Token Bottleneck

## TL;DR

Proposes the Vamos framework, which uses Large Language Models as reasoners to flexibly unify visual embeddings and general text descriptions as video representations. It discovers that text-only representations consistently achieve competitive or even superior performance across multiple video understanding benchmarks. Furthermore, it designs a Token Bottleneck Model to achieve interpretable evidence selection and a 5x inference speedup.

## Background & Motivation

**Background**: Core problems in video understanding include modeling the temporal dynamics of human activities from video observations, forecasting future behaviors, and answering video-related questions. End-to-end vision-language models (VLMs) represent the dominant direction.

**Limitations of Prior Work**: Visual representations in end-to-end models are uninterpretable, making it difficult to diagnose and correct erroneous predictions; visual embeddings are task-specific, requiring re-encoding for different tasks; and joint training of large-scale VLMs is highly expensive.

**Key Challenge**: Video understanding tasks may require complementary representations at different granularities, but existing methods typically rely only on a single visual embedding.

**Goal**: To explore whether general natural language descriptions can serve as effective video representations, and whether pretrained LLMs can function as action generation models.

**Key Insight**: Decoupling perception and reasoning, with text descriptions serving as intermediate representations to facilitate interpretability and efficiency.

**Core Idea**: Combining general video captions as text representations with LLMs as reasoners to construct an efficient, interpretable, and generalizable video understanding framework.

## Method

### Overall Architecture

Vamos is a unified framework that accepts three video representations as input: discrete action labels (from action recognition models), general text descriptions (from BLIP-2 or LLaVA), and distributed visual embeddings (from CLIP). The core idea is to unify these representations into the input space of LLMs, leverage the sequence completion ability of LLMs to accomplish different video understanding tasks.

The input sequence $\mathbf{x}_t = [\mathbf{x}_{\text{tvr}}, \mathbf{x}_{\text{task}}]$ contains the text-based video representation and task-related language inputs. A frozen word embedding layer generates text tokens $\mathbf{z}_t = \mathcal{F}_{\text{emb}}(\mathbf{x}_t) \in \mathbb{R}^{L_t \times D}$. For visual embeddings, they are aligned to the language space through a learnable linear projection:

$$\mathbf{z}_v = \mathcal{F}_{\text{proj}}(\mathcal{E}(v_1, \ldots, v_{N_v})) \in \mathbb{R}^{N_v \times D}$$

An early fusion strategy is adopted to concatenate visual and text tokens as LLM inputs.

### Key Designs

1. **Text-based Video Representation**: Text-based representations of videos are obtained in two ways. For general captions, image-level captions are generated on sampled frames using BLIP-2 or LLaVA-1.5 and concatenated into video-level captions; LLaVA generates more detailed descriptions (around 100 tokens per frame) and yields better performance. For action labels, a pretrained Transformer encoder is used to predict verb-noun pairs within the Ego4D predefined action space. A key insight is that text representations are general and reusable, allowing them to be extracted once and then serve various downstream tasks.

2. **LLM as a Temporal Reasoner**: Leverages the sequence completion capabilities of LLMs to unify multiple tasks. In video question answering (VQA), the task input consists of instructions, questions, and candidate answers, with the training target being the correct answer, and inference selecting the candidate answer that maximizes sequence likelihood. In long-term action forecasting (LTA), the task input consists of instructions and past action sequences, with the LLM autoregressively generating future actions. Parameter-efficient fine-tuning is conducted using LoRA or LLaMA-Adapter.

3. **Token Bottleneck Model (TBM)**: Inspired by the Concept Bottleneck Model, two key generalizations are made. First, generalizing from predefined concepts to free-form text tokens, operating directly on tokenized text. Second, generalizing from linear classifiers to a non-linear model combined with hard attention, using binary attention to select a subset of tokens as input for the LLM.

   Implementation: The input sequence is uniformly split into $k$ segments, each containing $n$ tokens. A lightweight token selector (a shallow Transformer encoder followed by a linear layer) selects 1 token from each segment. During training, Gumbel-Softmax is used to maintain differentiability:

    $\mathbf{g}^{(i)} = \text{TokenSelector}(z_1^{(i)}, \ldots, z_n^{(i)}) \in \mathbb{R}^n$

   From the $k$ segments, $k$ tokens are selected as the condensed representation (e.g., $k=40$ retains only about 6% of the original tokens). TBM delivers threefold value: interpretability (directly examining the selected tokens as evidence for decisions), intervenability (supporting manual correction at test time), and efficiency (adding only 0.7M parameters while accelerating inference by 5x).

4. **Modality Fusion and Modality Dropout**: When fine-tuning with both text and visual inputs simultaneously, the model tends to overfit. The solution is to randomly drop the entire visual embedding sequence during training, allowing stable training of the vis+text model.

### Loss & Training

- **Training Objective**: Standard language modeling objective (next token prediction)
- **Fine-tuning Method**: Parameter-efficient fine-tuning via LoRA or LLaMA-Adapter
- **LLM Selection**: LLaMA2-7B (Ego4D), LLaMA3-8B (NeXT-QA/IntentQA), GPT-4o (zero-shot)
- **Visual Encoder**: Frozen CLIP ViT-L/14
- **Caption Generation**: BLIP-2 or LLaVA-1.5, sampling 6-12 frames

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (Vamos) | Best Comparative Method | Gain |
|--------|------|------|------|------|
| NeXT-QA | Acc (All) | 77.3% | LLaMA-VQA-33B 75.5% | +1.8% (8B outperforms 33B) |
| IntentQA | Acc (All) | 74.16% | CaVIR 57.64% | +28.7% |
| Ego4D LTA | Edit Dist (Action) | 0.868 | AntGPT 0.877 | +0.009 |
| EgoSchema (zero-shot) | Acc (Full) | 53.55% | InternVideo 32.1% | +66.8% |
| Spacewalk-18 (zero-shot) | Acc | 18.6% | Prev. Best 13.6% | +5.0% |

### Comparison of Different Representations

| Input Type | Ego4D-LTA (Action, lower is better) | NeXT-QA (All) | IntentQA (All) |
|---------|---------------------|-----------------|-----------------|
| Visual Embeddings | 0.884 | 69.6% | 66.7% |
| Text Representations | 0.878 | 75.0% | 73.2% |
| Vision + Text | 0.868 | 74.5% | 71.7% |

### Ablation Study on Token Bottleneck Model

| Input | Selected Tokens / Total | NeXT-QA All | Inference Time |
|------|---------------------|------------|---------|
| Visual Embeddings | all | 71.0% | - |
| Text | all (644 avg) | 77.3% | 1.41s |
| Text + TBM | 20 / 644 | 67.4% | - |
| Text + TBM | 40 / 644 | 69.6% | 0.29s (5x) |

### Key Findings

- Text-based representations consistently outperform or match visual embeddings across all benchmarks, which is the most surprising finding of this work.
- Adding visual features yields almost no additional gain on VQA tasks, indicating that CLIP embeddings struggle to encode residual information not captured by captions.
- Vamos directly benefits from stronger LLMs: LLaMA3 improves over LLaMA1 by 2.3% on NeXT-QA.
- The immense advantage in zero-shot long-video QA stems from the decoupling of perception and reasoning, which allows the reasoning module to generalize much more easily.
- Oracle captions reach 81.8% on a subset of EgoSchema, demonstrating that better captioning models could further elevate the performance ceiling.

## Highlights & Insights

1. **Counter-intuitive Finding**: Text-only representations perform comparably to or even better than visual embeddings in video understanding tasks, challenging the assumption that visual features are irreplaceable.
2. **Power of Perception-Reasoning Decoupling**: Decoupled reasoning modules can generalize in a zero-shot manner and directly benefit from advancements in LLMs.
3. **Token Bottleneck Model**: Elegantly generalizes Concept Bottleneck Models (CBMs) to free-form text and non-linear models, balancing both interpretability and efficiency.
4. **Test-time Intervention**: The interpretability of text representations naturally supports manual correction of erroneous predictions without needing model retraining.
5. **Framework Versatility**: A single set of text representations can be reused across multiple tasks, allowing different tasks to share the reasoning module.

## Limitations & Future Work

1. Text-based representations are lossy compressions of visual inputs, which may lose fine-grained visual information (e.g., precise object locations, subtle motions).
2. The performance upper bound of the framework is constrained by the quality of the captioning models.
3. Generating captions itself requires large models, meaning the overall computational cost is not zero.
4. CLIP embeddings fail to effectively supplement caption information; alternative visual encoders need to be explored.
5. Better benchmarks are required to evaluate scenarios demanding fine-grained structured visual understanding.

## Related Work & Insights

- **Socratic Models**: Similarly utilize natural language as a shared interface across modalities; Vamos adds visual embedding fusion and TBM.
- **Concept Bottleneck Model**: Vamos generalizes CBM to TBM, moving from predefined concepts to free-form text tokens.
- **VidIL**: Leverages expert knowledge to design concepts for few-shot video understanding, whereas Vamos utilizes general captions which is more flexible.
- **Insight**: Against the backdrop of continuously improving LLM capabilities, the paradigm of textualized perception combined with LLM reasoning represents an efficient and scalable direction.

## Rating

- Novelty: ⭐⭐⭐⭐ The exploration of using text representations to replace visual embeddings is inspiring, and the TBM design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with evaluation across 5 complementary benchmarks, systematic representation comparisons, multi-LLM ablations, and in-depth TBM analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous experimental design, and intuitive visualization analysis.
- Value: ⭐⭐⭐⭐⭐ Unveils the powerful potential of "text-as-representation" and offers key insights for the paradigm of video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Optimizing Factorized Encoder Models: Time and Memory Reduction for Scalable and Efficient Action Recognition](optimizing_factorized_encoder_models_time_and_memory_reduction_for_scalable_and_.md)
- [\[ECCV 2024\] Referring Atomic Video Action Recognition](referring_atomic_video_action_recognition.md)
- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)
- [\[ICLR 2026\] LLaVAction: Evaluating and Training Multi-modal Large Language Models for Action Understanding](../../ICLR2026/video_understanding/llavaction_evaluating_and_training_multi-modal_large_language_models_for_action_.md)
- [\[CVPR 2026\] InternVideo-Next: Towards World-Understanding Video Models](../../CVPR2026/video_understanding/internvideo-next_towards_world-understanding_video_models.md)

</div>

<!-- RELATED:END -->
