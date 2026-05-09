---
title: >-
  [Paper Note] DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding
description: >-
  [NeurIPS 2025][Multimodal VLM][Temporal Point Process] This paper introduces DanmakuTPPBench, the first multi-modal Temporal Point Process (TPP) benchmark integrating temporal, textual, and visual modalities. It comprises DanmakuTPP-Events (7,250 video sequences with 10.8 million danmaku events collected from Bilibili) and DanmakuTPP-QA (10 evaluation tasks constructed via a multi-agent pipeline), revealing significant gaps in current LLM/MLLM capabilities for TPP understanding.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Temporal Point Process
  - Danmaku
  - Multi-modal Benchmark
  - Large Language Models
  - Temporal Reasoning
date: 2026-05-08
content_hash: 852537371111981a
---

# DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2505.18411](https://arxiv.org/abs/2505.18411)
**Code**: [GitHub](https://github.com/FRENKIE-CHIANG/DanmakuTPPBench)
**Area**: Multi-modal VLM
**Keywords**: Temporal Point Process, Danmaku, Multi-modal Benchmark, Large Language Models, Temporal Reasoning

## TL;DR
This paper introduces DanmakuTPPBench, the first multi-modal Temporal Point Process (TPP) benchmark integrating temporal, textual, and visual modalities. It comprises DanmakuTPP-Events (7,250 video sequences with 10.8 million danmaku events collected from Bilibili) and DanmakuTPP-QA (10 evaluation tasks constructed via a multi-agent pipeline), revealing significant gaps in current LLM/MLLM capabilities for TPP understanding.

## Background & Motivation

1. **Background**: TPP has broad applications in social media, healthcare, and finance, yet existing TPP datasets are nearly all unimodal (timestamps and event types only), impeding the development of multi-modal TPP models.

2. **Limitations of Prior Work**: Existing datasets lack textual and visual context. The TPP comprehension capabilities of LLMs/MLLMs remain largely unexplored. No dedicated TPP question-answering benchmark exists.

3. **Key Challenge**: Real-world event streams inherently contain multi-modal information, yet unimodal data cannot support the training and evaluation of models that exploit such information.

4. **Goal**: Construct the first natively multi-modal TPP dataset and question-answering benchmark.

5. **Key Insight**: The Bilibili danmaku system naturally constitutes a multi-modal TPP with precisely timestamped text content aligned to video frames.

6. **Core Idea**: Danmaku serves as a natural multi-modal TPP data source; a multi-agent pipeline is employed to automatically construct the QA benchmark.

## Method

### Overall Architecture
The benchmark consists of two complementary components: (1) DanmakuTPP-Events, which provides data for conventional TPP modeling; and (2) DanmakuTPP-QA, which provides QA tasks for evaluating LLM/MLLM TPP comprehension. Data are collected from 7,250 videos by Bilibili's 2024 Top-100 creators across 14 video categories.

### Key Designs

1. **DanmakuTPP-Events Dataset**:
   - **Function**: The first multi-modal TPP modeling dataset.
   - **Mechanism**: Videos are collected from Bilibili's 2024 Top-100 creators; each danmaku event comprises a timestamp $t_i$, an event type $e_i$ (9 categories), a text token $m_i^{text}$, and a video frame $m_i^{image}$.
   - **Design Motivation**: Danmaku natively fuses temporal, textual, and visual modalities across 14 video categories.

2. **Multi-Agent Construction Pipeline**:
   - **Function**: Automated construction of high-quality QA data.
   - **Mechanism**: Five agents collaborate — a Task Design Agent (DeepSeek-R1 designs 10 task types), an Annotation Agent (Qwen2.5 + Qwen2.5-VL + RAM for annotation), a Quality Control Agent (Qwen3 majority voting), a Visualization Agent (Qwen2.5-Coder generates charts), and a Task-Solving Agent (multi-LLM majority voting for answer generation).
   - **Design Motivation**: The scale and complexity of danmaku data render manual annotation infeasible; the multi-agent pipeline ensures quality at scale.

3. **10 Evaluation Tasks**:
   - **Function**: Comprehensive assessment of TPP comprehension.
   - **Mechanism**: 8 closed-ended tasks (danmaku burst counting, time prediction, sentiment prediction, event type reasoning, etc.) and 2 open-ended tasks (global sentiment dynamics analysis and danmaku burst causal analysis).
   - **Design Motivation**: Tasks span a range of difficulty levels, from simple prediction to complex multi-modal reasoning.

### Loss & Training
- Conventional TPP model evaluation: RMSE (time prediction) and log-likelihood (modeling performance).
- QA evaluation: Accuracy/RMSE for closed-ended tasks; Qwen3-235B scoring (0–1) for open-ended tasks.
- Fine-tuning: Qwen2.5-VL-3B + LoRA, single RTX 4090, 3 epochs.
- Each MLLM samples only 3 video frames per evaluation; MLLM input includes the danmaku event sequence as text and the sampled video frames.

## Key Experimental Results

### Main Results

| Model | T-1 (ACC) | T-2 (RMSE↓) | T-7 (ACC) | T-8 (ACC) |
|---|---|---|---|---|
| Qwen2.5-7B | 0.33 | 27.64 | 10.67 | 32.67 |
| Qwen2.5-72B | 0.67 | 1.28 | 16.00 | 43.83 |
| DeepSeek-V3 | 25.00 | 1.30 | 13.67 | 34.50 |
| Qwen2.5-VL-72B | 0.33 | 1.14 | 15.98 | 47.17 |
| Fine-tuned 3B | **27.00** | 1.35 | - | - |

### Key Findings
- Among conventional TPP models, NHP achieves the best performance (log-likelihood 0.799).
- Scaling model size improves TPP comprehension (RMSE decreases from 27.64 to 1.28).
- Visual information (MLLMs) does not consistently improve performance, highlighting ongoing challenges in multi-modal fusion.
- Danmaku burst counting (T-1) proves difficult for all models, with a maximum accuracy of only 27%.
- Fine-tuning a 3B model can approach the performance of 72B models on certain tasks.
- Across model families, Qwen3 performs best on sentiment-related tasks (lowest T-4 RMSE of 0.20), while DeepSeek-V3 and Llama-3.3 lead on sentiment polarity prediction (T-5/T-6).
- MLLMs do not consistently outperform LLMs — Llama-3.3-70B achieves the lowest RMSE on T-2 (1.11), suggesting that language models can infer temporal patterns from linguistic cues.
- The fine-tuned 3B model reduces error on sentiment prediction tasks (T-4/5/6) by 4–6× compared to the best pre-trained model (RMSE: 0.05/0.16/0.08), but overfits on T-3 (RMSE: 220.43).
- On open-ended tasks, Qwen3-235B performs best on causal analysis (T-10, score 0.52), while Qwen2.5-VL-72B leads on global sentiment analysis (T-9, score 0.48).

## Highlights & Insights
- The choice of danmaku as a TPP data source is creative — it is natively multi-modal, large-scale, and rich in social signals.
- The multi-agent pipeline constitutes a scalable paradigm for dataset construction.
- The taxonomy of 9 danmaku event types has independent value for sociological research.
- The benchmark exposes substantial gaps in LLMs/MLLMs' ability to understand temporal event sequences.
- Fine-tuning Qwen2.5-VL-3B with LoRA on a single RTX 4090 for 3 epochs surpasses 72B pre-trained models on sentiment tasks, demonstrating the importance of task-specific adaptation.

## Limitations & Future Work
- Data are sourced exclusively from the Chinese-language Bilibili platform; cross-platform and cross-lingual generalization remains to be validated.
- Each MLLM samples only 3 video frames; incorporating more frames may improve performance.
- Danmaku data may contain inappropriate content, necessitating content moderation.
- Conventional TPP models do not leverage multi-modal information, motivating the development of new multi-modal TPP architectures.
- The five agents in the pipeline have clearly delineated roles: DeepSeek-R1 designs tasks; Qwen2.5/Qwen2.5-VL/RAM handle annotation; Qwen3 controls quality via majority voting; Qwen2.5-Coder generates visualizations; and multi-LLM majority voting produces final answers.
- The 8 closed-ended tasks cover danmaku burst counting, time prediction, precise time prediction, sentiment/polarity prediction, and event type reasoning; the 2 open-ended tasks require global dynamics analysis and causal attribution.

## Related Work & Insights
- **vs. Retweet/StackOverflow datasets**: These provide only timestamps and event types, lacking textual and visual information.
- **vs. Amazon Review**: Textual content is available but no visual modality; DanmakuTPP provides all three modalities.
- **vs. TSQA (Time-Series QA)**: TSQA targets general time series, whereas DanmakuTPP focuses specifically on event sequences.
- **vs. Language-TPP**: Language-TPP attempts to apply LLMs to TPP but uses only unimodal text data; DanmakuTPP introduces the first natively multi-modal TPP evaluation.

## Rating

### Implementation Details
Data are collected from 7,250 videos by Bilibili's 2024 Top-100 creators across 14 categories. Five agents collaborate: DeepSeek-R1 designs tasks, Qwen2.5 annotates, and Qwen3 controls quality. Fine-tuning employs Qwen2.5-VL-3B + LoRA on a single RTX 4090 GPU for 3 epochs.
- **Novelty**: ⭐⭐⭐⭐⭐ First multi-modal TPP benchmark with an innovative danmaku data source.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation covering both conventional TPP models and LLMs/MLLMs.
- **Writing Quality**: ⭐⭐⭐⭐ Detailed pipeline design with rich statistical analysis.
- **Value**: ⭐⭐⭐⭐ Opens a new research direction for multi-modal TPP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[NeurIPS 2025\] Efficient Multi-modal Large Language Models via Progressive Consistency Distillation](efficient_multi-modal_large_language_models_via_progressive_consistency_distilla.md)
- [\[NeurIPS 2025\] Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?](can_multi-modal_llms_provide_live_step-by-step_task_guidance.md)
- [\[NeurIPS 2025\] Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)

</div>

<!-- RELATED:END -->
