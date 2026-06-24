---
title: >-
  [Paper Note] Period-LLM: Extending the Periodic Capability of Multimodal Large Language Model
description: >-
  [CVPR 2025][Multimodal VLM][Periodic tasks] This paper proposes Period-LLM—the first MLLM equipped with period-perception capabilities. It adopts an "easy-to-hard" progressive training paradigm (text repetition $\rightarrow$ macro-periodic video $\rightarrow$ micro-periodic signals) paired with a "Resisting Logical Oblivion" (RLO) gradient optimization strategy, significantly outperforming existing MLLMs on cross-modal periodic tasks such as repetitive action counting and rPP…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Periodic tasks"
  - "Multimodal Large Language Model"
  - "Progressive training"
  - "Gradient optimization"
  - "Repetition counting"
date: 2026-05-08
content_hash: bb19ebf57f5ff5f2
---

# Period-LLM: Extending the Periodic Capability of Multimodal Large Language Model

**Conference**: CVPR 2025  
**arXiv**: [2505.24476](https://arxiv.org/abs/2505.24476)  
**Code**: [https://github.com/keke-nice/Period-LLM](https://github.com/keke-nice/Period-LLM)  
**Area**: Multimodal VLM  
**Keywords**: Periodic tasks, Multimodal Large Language Model, Progressive training, Gradient optimization, Repetition counting

## TL;DR
This paper proposes Period-LLM—the first MLLM equipped with period-perception capabilities. It adopts an "easy-to-hard" progressive training paradigm (text repetition $\rightarrow$ macro-periodic video $\rightarrow$ micro-periodic signals) paired with a "Resisting Logical Oblivion" (RLO) gradient optimization strategy, significantly outperforming existing MLLMs on cross-modal periodic tasks such as repetitive action counting and rPPG heart rate estimation.

## Background & Motivation
Periodic/quasi-periodic phenomena are widely present in nature: human movement counting (rope skipping, pull-ups), weather cycles (weather forecasting), physiological signals (heart rate, respiration rate), traffic flow, etc. These tasks span multiple modalities, and in theory, MLLMs should be able to handle them. However, current MLLMs (such as GPT-4, Video-LLaMA) perform poorly on periodic tasks—failing to accurately count repetitive actions or detect periodic signals. Three core problems exist: (1) **Interference from spatial pseudo-temporal information**—numbers appearing in a video can mislead the model into taking shortcuts instead of learning real periodic information; (2) **Conflict between long-term periodic reasoning and short-term semantic understanding**—over-optimizing for semantic understanding leads to forgetting periodic reasoning capabilities; (3) **Lack of counting descriptions in training data**—almost no precise counting descriptions like "completed N pull-ups" exist in MLLM training corpora. The core idea of this paper is to first learn the concept of "periodicity" in a simple text repetition counting task, then progressively transfer it to more complex video periodic tasks, while using a special gradient optimization to prevent capability forgetting.

## Method

### Overall Architecture
Period-LLM is based on the LLaVA architecture. The input video features are extracted via a visual encoder and visual projector, concatenated with text features, and then fed into the LLM. The training consists of three stages: (1) text-only periodic pre-training (repetitive word counting); (2) macro-periodic video fine-tuning (Countix repetitive action counting); (3) micro-periodic signal fine-tuning (rPPG heart rate estimation). During the multimodal generalization stage, the RLO optimization strategy is applied to prevent the forgetting of periodic reasoning capabilities.

### Key Designs
1. **Easy-to-Hard Generalization**:
    - **Function**: Enables the LLM to progressively build cross-modal periodic understanding capabilities, from the simplest text repetitions to the most complex micro physiological signals.
    - **Mechanism**: Divides periodic tasks into three levels of difficulty:
        - **Text level**: Constructs a "repeated word QA" dataset—"How many times is the word {word} repeated in the string {string}?", where the repetition count is $n \in \{2, 3, ..., 20\}$, and uses GPT-4 to generate 10 semantically equivalent question variants. The model learns pure logical reasoning $A = F(T_f, Q)$.
        - **Macro video level**: Uses the Countix dataset (8,757 repetitive action videos), where the model needs to align visual semantics with periodic information $A = F(M_f, Q)$.
        - **Micro signal level**: rPPG tasks (extracting heart rate from facial videos), where periodic signals have small amplitudes and are masked by noise.
    - **Design Motivation**: LLMs are naturally strongest in text processing, and the essence of periodicity ("repetition") also exists in text. Learning periodic cognitive concepts in text first makes it easier to transfer to more complex modalities. Mathematically, periodic inputs can be unified as $x = K \cdot p(\omega t) + N \cdot s(t)$, with text repetition being the simplest case where $K$ is constant and $N=0$.

2. **Instruction Generation**:
    - **Function**: Generates QA training data in a unified format for periodic tasks in various modalities.
    - **Mechanism**: For text tasks, words from the GPT-4 technical report are randomly selected as repetitive words to construct "{word}*n" strings, and GPT-4 is used to generate complete answer sentences. For video tasks, combined with dataset annotations (action categories), raw descriptions, and frequency information, QA pairs of "What is the total number of repetitive actions?" are generated, and GPT-4 is then used to generate multiple semantically equivalent questions.
    - **Design Motivation**: Existing MLLM training data rarely contain precise counting descriptions (usually only vague statements like "performed pull-ups multiple times"), necessitating the construction of a dedicated periodic QA dataset.

3. **Resisting Logical Oblivion (RLO)**:
    - **Function**: Prevents periodic reasoning capabilities from being overwritten by semantic understanding training during multimodal fine-tuning.
    - **Mechanism**: Introduces a feature channel weight function $\Omega(c_i)$ to dynamically weight gradient updates for output feature channels. When the average activation $\bar{c_i}$ of the $i$-th channel is lower than the global average $\bar{c}$ (meaning this channel has not been fully learned), a larger update weight is assigned:
$$\Omega(c_i) = \begin{cases} 1 + \beta \cdot e^{\frac{iter_{num}}{max_{iter}}}, & \bar{c_i} < \bar{c} \\ 1, & \bar{c_i} > \bar{c} \end{cases}$$
        The gradient update becomes $\nabla\theta_j^* = \Omega(c_i) \cdot \nabla\theta_j$. In this way, new semantic knowledge is guided to be learned in the underutilized feature channels, while the channels carrying existing reasoning capabilities remain undisturbed.
    - **Design Motivation**: Traditional gradient descent updates all feature channels indiscriminately. Using the same parameter space for both semantic understanding and periodic reasoning causes knowledge interference. The idea behind RLO is similar to "allocating redundant channels to new tasks" to protect already learned knowledge.

### Loss & Training
A standard autoregressive language modeling loss is employed: $\max_\phi \sum_{(x,y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log(P_\phi(y_t | x, y_{<t}))$, along with RLO gradient re-weighting. Training details: NVIDIA A6000 GPUs, Adam optimizer, initial learning rate of 0.001, batch size of 1, images at 224x224, 20 frames per video, and 200,000 iterations. The visual encoder is CLIP ViT-L/14, and $\beta=0.05$.

## Key Experimental Results

### Main Results

| Method | LLM | Countix-QA MAE↓ | Countix-QA CIDEr↑ | rPPG-QA MAE↓ |
|------|-----|-----------------|-------------------|-------------|
| VideoLLaMA | Vicuna-7B | 4.98 | 0.570 | 18.29 |
| Video-ChatGPT | Vicuna-7B | 4.64 | 0.643 | 17.54 |
| LLaMA-VID | Vicuna-7B | 5.34 | 0.783 | 17.51 |
| **Period-LLM** | LLaMA-7B | **3.77** | **0.810** | **13.78** |

### Cross-Modal Periodic Tasks

| Method | RotNIST MAE↓ | Drive-QA MAE↓ | Radar-QA MAE↓ |
|------|-------------|---------------|---------------|
| Video-ChatGPT | 2.01 | 33.28 | 21.61 |
| LLaMA-VID | 2.43 | 32.45 | 18.21 |
| **Period-LLM** | **1.50** | **28.71** | **14.24** |

### Ablation Study

| Configuration | Countix MAE | CIDEr | Description |
|------|------------|-------|------|
| W/o text pre-training + W/o RLO | 4.30 | 0.661 | Baseline |
| W/ text pre-training + W/o RLO | 3.89 | 0.782 | Text pre-training brings significant improvement |
| W/ text pre-training + W/ RLO | **3.77** | **0.810** | RLO further improves performance |

| β value | MAE | Description |
|-----|-----|------|
| 0.01 | 3.85 | Anti-forgetting capability is too weak |
| **0.05** | **3.77** | Optimal balance point |
| 0.5 | 4.05 | Overprotects old knowledge, restricting new knowledge learning |

### Key Findings
- Text pre-training is crucial for periodic understanding: even a simple "counting repeated words" task can significantly improve performance on video periodic tasks.
- RLO contributes an additional 0.12 MAE decrease and a 0.028 CIDEr increase on Countix, confirming that capability forgetting is indeed a problem.
- Period-LLM reduces MAE by 3.73 on the rPPG task (13.78 vs 17.51), demonstrating that micro-periodic signals can also be perceived.
- Cross-modal generalization is effective: the model leads on RotNIST (image rotation counting), Drive-QA (traffic flow), and Radar-QA (radar physiological signals).

## Highlights & Insights
- **"Repetitive nature" is a cross-modal invariant**: Periodicity in text, video, and signals shares the same underlying structure ($x = K \cdot p(\omega t) + N \cdot s(t)$), which can be progressively transferred from simple to complex.
- **Channel-level analysis perspective of RLO**: Unlike classic continual learning methods like EWC that protect parameters based on importance, RLO starts from feature channel activity and guides new knowledge to "redundant channels."
- **First to reveal MLLMs' blindness to periodicity**: Models like GPT-4 and Video-LLaMA perform poorly on counting tasks, which is an overlooked dimension of capability.

## Limitations & Future Work
- The model is only based on LLaMA-7B and CLIP ViT-L/14; its effectiveness on larger LLMs or stronger visual encoders remains unknown.
- RLO assumes that feature channel activity can represent knowledge distribution, a hypothesis lacking strict theoretical proof.
- Training data construction relies heavily on GPT-4 to generate QA pairs, which is costly and may introduce bias.
- Validated only on relatively small-scale datasets (Countix, V4V, etc.); performance in large-scale scenarios remains to be validated.
- The selection of $\beta$ and thresholds requires manual parameter tuning, lacking an adaptive mechanism.

## Related Work & Insights
- Unlike specialized models like TransRAC (Transformer-based repetitive action counting), Period-LLM is a general-purpose MLLM framework.
- RLO's "channel-level gradient re-weighting" style can be extended to other multi-task/continual learning scenarios.
- Inspiration: MLLM capability evaluation should not be limited to VQA and captioning; "mathematic-like" capabilities such as periodicity and counting are also very important.

## Rating
- Novelty: ⭐⭐⭐⭐ Mentions periodic tasks in MLLM for the first time; the "easy-to-hard" training paradigm and RLO strategy are creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple modalities (video, image, radar, traffic) with thorough ablation studies, though baseline comparisons are somewhat outdated.
- Writing Quality: ⭐⭐⭐ Generally clear but with some redundant descriptions, inconsistent mathematical notations, and figures/tables could be more refined.
- Value: ⭐⭐⭐⭐ Opens up a new research direction for MLLM periodic capability; the RLO strategy has practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Distraction is All You Need for Multimodal Large Language Model Jailbreaking](distraction_is_all_you_need_for_multimodal_large_language_model_jailbreaking.md)
- [\[CVPR 2025\] UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation](upme_an_unsupervised_peer_review_framework_for_multimodal_large_language_model_e.md)
- [\[CVPR 2025\] CoLLM: A Large Language Model for Composed Image Retrieval](collm_a_large_language_model_for_composed_image_retrieval.md)
- [\[CVPR 2025\] LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](lamra_large_multimodal_model_as_your_advanced_retrieval_assistant.md)
- [\[CVPR 2025\] 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models](4d_langsplat_4d_language_gaussian_splatting_via_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
