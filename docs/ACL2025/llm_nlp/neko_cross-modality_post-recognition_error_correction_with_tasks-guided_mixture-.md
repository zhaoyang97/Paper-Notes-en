---
title: >-
  [Paper Note] NeKo: Cross-Modality Post-Recognition Error Correction with Tasks-Guided Mixture-of-Experts Language Model
description: >-
  [ACL 2025][LLM (Other)][MoE] This paper proposes NeKo, a multi-task post-recognition error correction language model based on a Tasks-Guided Mixture-of-Experts (MoE) architecture. NeKo achieves state-of-the-art (SOTA) performance across multiple cross-modality error correction tasks—including ASR, speech translation, and OCR—and outperforms GPT-3.5 and Claude-3.5 Sonnet in zero-shot scenarios.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "MoE"
  - "post-recognition error correction"
  - "ASR"
  - "OCR"
  - "multi-task learning"
  - "speech translation"
date: 2026-05-08
content_hash: 4348dc7978f139c9
---

# NeKo: Cross-Modality Post-Recognition Error Correction with Tasks-Guided Mixture-of-Experts Language Model

**Conference**: ACL 2025  
**arXiv**: [2411.05945](https://arxiv.org/abs/2411.05945)  
**Code**: Planned to be open-sourced (CC BY-SA 4.0)  
**Area**: LLM/NLP  
**Keywords**: MoE, post-recognition error correction, ASR, OCR, multi-task learning, speech translation  

## TL;DR

This paper proposes NeKo, a multi-task post-recognition error correction language model based on a Tasks-Guided Mixture-of-Experts (MoE) architecture. NeKo achieves state-of-the-art (SOTA) performance across multiple cross-modality error correction tasks—including ASR, speech translation, and OCR—and outperforms GPT-3.5 and Claude-3.5 Sonnet in zero-shot scenarios.

## Background & Motivation

- **Problem Definition**: Post-recognition error correction refers to the text-level correction of primary recognition outputs from systems like ASR, OCR, and machine translation, mimicking the human ability to "correct and comprehend" in noisy environments.
- **Limitations of Prior Work**: Previous generative error correction (GER) methods rely on domain-specific fine-tuning and employ independent correction language models for different tasks. This leads to a significant explosion in parameter size and insufficient generalization across different datasets and domains.
- **Core Motivation**: Can a unified model simultaneously process error correction datasets originating from multiple modalities, such as speech, text, and vision? While the MoE architecture is naturally suited to multi-task learning, how to make experts truly "specialize" in different tasks—rather than merely serving as a tool for scaling parameters—remains an open question.
- **Key Challenge**: In multi-task joint training, the model must simultaneously capture task-specific features and shared knowledge, whereas traditional MoE experts are not explicitly bound to specific tasks.

## Method

### Overall Architecture

NeKo is based on a Transformer + MoE architecture, where standard FFN blocks are replaced with MoE layers. In the forward pass, each token is routed by a gating network (router) to the top-K experts, and the output is the weighted sum of the selected experts' outputs:

$$y = \sum_{i=0}^{n-1} G(x)_i \cdot E_i(x)$$

where $G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g))$.

### Key Designs

1. **Tasks-Guided Auxiliary Expert Assignment**: The core innovation lies in explicitly mapping each task to a specific expert during the training phase. For an input token $x$ from task $T_i$, the model **deterministically** routes it to its mapped expert $f(T_i)$, while adding the top-1 expert selected by the gating network (with the pre-assigned expert excluded). This design ensures that task-specific knowledge acquisition coexists with cross-task knowledge sharing.
2. **Training-Inference Decoupling**: During training, task labels guide the routing (one fixed expert + one router-selected expert); during inference, the model **does not assume knowledge of the input task type**, entirely relying on the router's probability to select the top-K experts. This allows the model to automatically generalize to unseen tasks.
3. **Multi-Task Error Correction Joint Training**: Covers 5 major categories of tasks: ASR (8 datasets from the Open ASR Leaderboard), Speech Translation ST (HypoTranslate), Machine Translation MT, OCR (Chronicling America), and Text Error Correction TEC (CoEdIT).

### Loss & Training

A standard negative log-likelihood loss is jointly optimized across the multi-task datasets:

$$\mathcal{L} = -\sum_{i=1}^{m} \sum_{(x,y) \in D_i} \log p(y | x, T_i)$$

where $x$ is the primary output of the recognition system (e.g., ASR hypothesis), and $y$ is the target text (e.g., ground-truth transcript).

## Experiments

### Main Results: Open ASR Leaderboard

| Model | Inference Params | Average WER ↓ |
|------|-----------|-----------|
| Whisper-V2-Large | 1.5B | 8.06 |
| Canary | 2B | 6.67 |
| Bestow Speech LM | 1.8B | 6.50 |
| + Gemma 2B FFT | 3.5B | 6.61 |
| + Mistral 7B FFT | 8.5B | 6.40 |
| + Mixtral 8x7B FFT | 8.5B | 6.51 |
| **+ NeKo Qwen1.5-MoE** | **4.2B** | **5.90** |
| **+ NeKo Mixtral 8x7B** | **8.5B** | **6.34** |

NeKo (Qwen1.5-MoE) achieves the lowest average WER of 5.90 across 9 datasets with only 4.2B activation parameters, surpassing all end-to-end and cascaded methods of equivalent or larger scale.

### Zero-Shot ASR Error Correction (Hyporadise Benchmark)

| Model | WSJ-dev93 | ATIS | CHiME4 Mean | MCV Mean |
|------|-----------|------|------------|---------|
| GPT-3.5 0-shot | 8.5 | 5.5 | ~17 | ~26 |
| Claude-3.5 0-shot | 8.2 | 5.2 | ~16 | ~25 |
| **NeKo-MoE 0-shot** | **6.8** | **4.2** | **~11** | **~22** |

Under the zero-shot setting, NeKo-MoE achieves a 15.5%–27.6% relative WER reduction compared to GPT-3.5, significantly outperforming commercial closed-source models.

### Ablation Study

| Variant | Average WER |
|------|---------|
| Single-Task FFT (Full Fine-Tune) | 6.61 |
| Multi-Task FFT (Without MoE) | 6.51 |
| Mixtral BTM (Branch-Train-Merge) | 6.43 |
| **NeKo MoE (Tasks-Guided Routing)** | **6.34** |

The results validate that: (1) multi-task performance outperforms single-task; (2) MoE models perform better than dense models and BTM; (3) tasks-guided expert assignment is crucial.

### Key Findings

- NeKo performs exceptionally well across all three modalities of ASR, ST, and OCR, demonstrating true cross-modality multi-task error correction capabilities.
- The training-inference decoupling design enables the model to automatically route tokens to the correct experts, even when the task type is unknown during inference.
- NeKo demonstrates emergent cross-task transferability on TEC (Text Error Correction)—despite being mainly trained on speech/OCR error correction, it exhibits competitive performance on pure text error correction.

## Highlights & Insights

- MoE layers serve not merely as a scaling tool, but as a structured solution for multi-task learning—where tasks-guided expert assignment cleverly balances "task specialization" and "knowledge sharing."
- It establishes a new SOTA on the Open ASR Leaderboard, surpassing 7B+ dense models with only 4.2B active parameters.
- Its zero-shot performance exceeds both GPT-3.5 and Claude-3.5 Sonnet, demonstrating that small and specialized MoEs can outperform general-purpose LLMs on targeted task error correction.
- It encompasses three major cross-modality error correction scenarios: speech-to-text, text-to-text, and vision-to-text.

## Limitations & Future Work

- Error correction heavily depends on the N-best output quality of upstream recognition systems (e.g., Canary/Whisper); if the first-pass recognition is too poor, the headroom for error correction is limited.
- The current mapping between the number of experts and tasks is relatively simple (1:1); allocation strategies when task types greatly exceed the number of experts have yet to be explored.
- The work primarily focuses on English ASR and a small set of translation language pairs, lacking large-scale validation across a broader range of languages.

## Related Work & Insights

- **Generative Post-Recognition Error Correction (GER)**: HyPoradise (CHEN et al., 2023), SoftCorrect (Yang et al., 2023), etc., employ LLMs for text-level correction of ASR/OCR outputs.
- **MoE for LLMs**: Mixtral (Jiang et al., 2024), Switch Transformer, etc., utilize MoE to scale up language model capacity. The innovation of NeKo lies in transforming MoE from a "general scaling tool" to a "task routing mechanism."
- **Multimodal Error Correction**: End-to-end speech-text models such as Qwen2-Audio and SALM; NeKo adopts a cascaded architecture but achieves multi-task performance via a unified MoE error correction layer.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 7 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| Practicality | 8 |
| **Overall Score** | **8.0** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Training-free LLM-based Approach to General Chinese Character Error Correction](a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)
- [\[ACL 2025\] Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts](cross_model_transferability_sv.md)
- [\[ACL 2025\] Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](can_we_further_elicit_reasoning_in_llms_critic-guided_planning_with_retrieval-au.md)
- [\[ACL 2025\] Mixture of Small and Large Models for Chinese Spelling Check](mixture_of_small_and_large_models_for_chinese_spelling_check.md)
- [\[ACL 2025\] Perspective Transition of Large Language Models for Solving Subjective Tasks](perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)

</div>

<!-- RELATED:END -->
