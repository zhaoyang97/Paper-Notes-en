---
title: >-
  [Paper Note] Controlling Multimodal LLMs via Reward-guided Decoding
description: >-
  [ICCV 2025][Multimodal VLM][Multimodal large language models] This paper proposes Multimodal Reward-Guided Decoding (MRGD), which constructs two reward models to independently control object precision and recall, enabling fine-grained controllability over MLLM outputs at inference time while substantially reducing object hallucinations.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Multimodal large language models
  - reward-guided decoding
  - hallucination mitigation
  - visual grounding
  - inference-time alignment
date: 2026-05-08
content_hash: fb3523710ffeb93b
---

# Controlling Multimodal LLMs via Reward-guided Decoding

**Conference**: ICCV 2025
**arXiv**: [2508.11616](https://arxiv.org/abs/2508.11616)
**Code**: N/A
**Area**: Multimodal VLM
**Keywords**: Multimodal large language models, reward-guided decoding, hallucination mitigation, visual grounding, inference-time alignment

## TL;DR

This paper proposes Multimodal Reward-Guided Decoding (MRGD), which constructs two reward models to independently control object precision and recall, enabling fine-grained controllability over MLLM outputs at inference time while substantially reducing object hallucinations.

## Background & Motivation

As MLLMs are increasingly deployed in practice, users' demand for behavioral control has grown along two key dimensions: (a) controlling the precision and comprehensiveness of outputs (e.g., object recall), and (b) controlling the amount of computation used during inference. For instance, visually impaired users may prioritize high-precision outputs to avoid hallucinations, while practitioners training downstream models on synthetic captions may prefer diversity and coverage.

Existing approaches exhibit notable shortcomings:
- **Prompt engineering**: Coarse-grained control that relies on manual design
- **Supervised fine-tuning (SFT) / RLHF fine-tuning**: No controllability at inference time; learned principles are not guaranteed to be followed
- **Specialized decoding strategies (VCD, CGD)**: Limited effectiveness or lack of multi-dimensional control

For text-only LLMs, reward-guided decoding has proven effective; however, the multimodal setting poses unique challenges: reward models must jointly process visual and textual information and understand the interactions between generated text and visual inputs—interactions that give rise to specific types of hallucinations. This work is the first to extend reward-guided decoding to the multimodal setting.

## Method

### Overall Architecture

At each inference step, MRGD executes the following search procedure:
1. Sample $k$ candidate continuations from the MLLM (at the sentence level, delimited by periods)
2. Score each candidate using a linear combination of two reward models
3. Select the highest-scoring candidate and append it to the generated context
4. Repeat until an EOS token is generated

The final score is a weighted combination of two rewards: $s = w \cdot r_{\text{hal}} + (1-w) \cdot r_{\text{rec}}$, where $w \in [0, 1]$ is an inference-time hyperparameter controlling guidance strength. Setting $w=1$ focuses exclusively on hallucination suppression, while $w=0$ focuses exclusively on recall improvement.

### Key Designs

**1. Hallucination Reward Model $r_{\text{hal}}$ (learned)**

- Backbone: PaliGemma (3B) with an added linear regression head that maps the last token embedding to a scalar
- Trained on multiple public preference datasets: LLaVA-RLHF (9.4k) + RLHF-V (5.7k) + POVID (17k) + SugarCrepe (7.5k)
- Trained as a preference classifier using the Bradley-Terry model, with MSE regularization to constrain outputs to $[0, 1]$
- LoRA fine-tuning of the backbone; batch size 256; cosine learning rate schedule; trained for a single epoch
- Validation accuracy: 82.05%; cross-domain accuracy on VLFeedback: 67.68%

**2. Recall Reward Model $r_{\text{rec}}$ (rule-based, training-free)**

Composed of three off-the-shelf modules:
- **OWLv2** (open-vocabulary detector): extracts reference objects from the image
- **NLTK POS tagger**: extracts predicted objects from the generated text
- **Sentence-BERT**: computes semantic similarity between reference and predicted objects with threshold $\tau = 0.5$

This effectively estimates object recall: number of matched predicted objects / total number of reference objects.

**3. Sentence-Level Evaluation Strategy**

To avoid instability when evaluating incomplete text, the reward models are invoked once every $T$ sentences. When $T$ is sufficiently large, this degenerates into best-of-$k$ rejection sampling. Experiments show that more frequent evaluation ($T=1$) improves sampling efficiency.

### Loss & Training

The training loss for the hallucination reward model combines a Bradley-Terry preference loss with MSE regularization. The MSE term encourages positive-sample scores to approach 1 and negative-sample scores to approach 0, facilitating linear combination with the recall reward and preventing gradient saturation.

## Key Experimental Results

### Main Results

Evaluated on LLaVA-1.5 7B across two hallucination benchmarks, COCO (CHAIR) and AMBER:

| Method | Ci (↓) | Cs (↓) | Rec. (↑) | CHAIR (↓) | Hal. (↓) | Cov. (↑) |
|--------|--------|--------|----------|-----------|----------|----------|
| Greedy | 15.05 | 48.94 | 81.30 | 7.6 | 31.8 | 49.3 |
| VCD | 15.76 | 54.18 | 81.66 | 9.7 | 42.8 | 51.6 |
| CGD | 9.48 | 37.48 | 80.11 | 5.1 | 24.0 | 48.3 |
| MRGD $w{=}1.0$ | **4.53** | **18.19** | 76.04 | **3.4** | **15.9** | 52.4 |
| MRGD $w{=}0.5$ | 5.34 | 22.54 | 78.63 | 4.4 | 25.4 | **60.8** |

MRGD is also validated on Llama-3.2-Vision (11B) and SmolVLM-2 (2.2B), demonstrating transferability without retraining the reward models.

### Ablation Study

Effect of varying weight $w$ on the precision–recall trade-off (LLaVA-1.5, COCO):

| $w$ | Ci (↓) | Rec. (↑) | Len. |
|-----|--------|----------|------|
| 1.0 | 4.53 | 76.04 | 95.90 |
| 0.75 | 4.76 | 76.84 | 96.17 |
| 0.5 | 5.34 | 78.63 | 97.96 |
| 0.25 | 7.67 | 81.56 | 105.34 |
| 0.0 | 24.20 | 85.23 | 108.92 |

$w$ provides a smooth interpolation from low-hallucination/low-recall to high-recall/high-hallucination regimes. MRGD can also be stacked on top of RLHF fine-tuned models for further improvement.

### Key Findings

- MRGD ($w=1.0$) reduces CHAIRi from 15.05% to 4.53%, approximately a 70% reduction, with only a ~6.5% drop in recall
- The optimal $w$ varies across datasets: approximately 0.25 on COCO and 1.0 on AMBER (COCO images contain an average of 21.4 objects vs. 9.9 for AMBER)
- VCD and LLaVA-RLHF actually increase hallucination rates on generative hallucination benchmarks
- PaliGemma-2 as backbone performs slightly worse than PaliGemma, indicating that a larger backbone does not necessarily yield better results

## Highlights & Insights

1. **Inference-time controllability**: A single parameter $w$ enables dynamic adjustment of the precision–recall trade-off without retraining
2. **Model-agnosticism**: The trained reward models transfer directly across different MLLMs (LLaVA, Llama-3.2, SmolVLM-2)
3. **Elegant dual-reward design**: One learned reward (trained on preference data) and one rule-based reward (composed of off-the-shelf modules) balance effectiveness and cost
4. **Revealing the divergence between generative and discriminative hallucination benchmarks**: Methods such as VCD improve on VQA benchmarks but degrade performance on caption generation benchmarks

## Limitations & Future Work

- Inference cost scales linearly with the number of samples $k$, requiring multiple forward passes and reward evaluations per step
- The recall reward model is constrained by the accuracy of the detector (63%) and POS tagger (67%)
- Validation is limited to image captioning; more complex settings such as video understanding and multi-turn dialogue remain unexplored
- The reward models address only object-level hallucinations and do not cover attribute or relational hallucinations

## Related Work & Insights

- Compared to CGD, training a dedicated multimodal reward model yields substantially better results than CLIP-guided decoding
- MRGD is complementary to RLHF fine-tuning and can further improve already-aligned models
- This work opens a new direction for test-time compute scaling in multimodal settings
- The dual-reward mixing paradigm is generalizable to controlling other output attributes, such as verbosity or style

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs](instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)
- [\[ICCV 2025\] ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning](chartpoint_guiding_mllms_with_grounding_reflection_for_chart_reasoning.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[ICCV 2025\] Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs](are_they_the_same_exploring_visual_correspondence_shortcomings_of_multimodal_llm.md)
- [\[ICCV 2025\] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation](multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)

<!-- RELATED:END -->
