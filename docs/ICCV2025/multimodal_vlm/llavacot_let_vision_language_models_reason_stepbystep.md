---
title: >-
  [Paper Note] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step
description: >-
  [ICCV 2025][Multimodal VLM][VLM Reasoning] By constructing the LLaVA-CoT-100k dataset with structured reasoning annotations, the proposed method trains a VLM to autonomously execute a four-stage reasoning pipeline—Summary → Caption → Reasoning → Conclusion—combined with a SWIRES search strategy at test time. The resulting 11B model outperforms substantially larger models including GPT-4o-mini and Gemini-1.5-pro.
tags:
  - ICCV 2025
  - Multimodal VLM
  - VLM Reasoning
  - Chain-of-Thought
  - Multi-stage Reasoning
  - Test-time Scaling
  - Structured Reasoning
date: 2026-05-08
content_hash: 67c580171d296579
---

# LLaVA-CoT: Let Vision Language Models Reason Step-by-Step

**Conference**: ICCV 2025
**arXiv**: [2411.10440](https://arxiv.org/abs/2411.10440)
**Code**: [https://github.com/PKU-YuanGroup/LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)
**Area**: Multimodal VLM / Reasoning
**Keywords**: VLM Reasoning, Chain-of-Thought, Multi-stage Reasoning, Test-time Scaling, Structured Reasoning

## TL;DR
By constructing the LLaVA-CoT-100k dataset with structured reasoning annotations, the proposed method trains a VLM to autonomously execute a four-stage reasoning pipeline—Summary → Caption → Reasoning → Conclusion—combined with a SWIRES search strategy at test time. The resulting 11B model outperforms substantially larger models including GPT-4o-mini and Gemini-1.5-pro.

## Background & Motivation

### State of the Field

**Background**: Large language models have achieved notable progress in reasoning (e.g., CoT prompting), yet current vision-language models (VLMs) still struggle to perform systematic, structured reasoning on complex visual question answering tasks. Conventional chain-of-thought prompting yields limited benefits for VLMs, as interpreting visual information requires additional structured steps. Existing VLMs typically jump directly from question to answer without an intermediate systematic thinking process, a deficiency that is especially pronounced in tasks involving spatial reasoning, scientific computation, and chart comprehension.

### Approach

**Goal**: The central question is how to enable VLMs to autonomously and systematically perform multi-stage reasoning. Key challenges include: (1) the lack of training data with structured reasoning annotations; (2) the need for models to learn natural transitions between reasoning stages; and (3) how to further improve reasoning quality at inference time (test-time scaling).

## Method

### Overall Architecture
LLaVA-CoT is fine-tuned from Llama-3.2-11B-Vision-Instruct. Given an image and a question, the model autonomously generates a four-stage reasoning process, with each stage enclosed by special tags (e.g., `<SUMMARY>...</SUMMARY>`), and ultimately produces a conclusion. The entire reasoning process is generated end-to-end without additional prompt engineering.

### Key Designs
1. **Four-stage Structured Reasoning**: The model is trained to automatically produce four reasoning stages:

   - **Summary**: Understanding the question and clarifying the task ("What's the problem? What should I do?")
   - **Caption**: Extracting question-relevant visual information from the image ("What can I know from the image?")
   - **Reasoning**: Conducting step-by-step logical inference based on the extracted information ("How to solve the problem step-by-step?")
   - **Conclusion**: Synthesizing prior analysis to produce the final answer

   The core insight behind this design is that a VLM must first *understand* the image before *reasoning about* the problem, rather than processing both simultaneously.

2. **LLaVA-CoT-100k Dataset Construction**: Images and questions are collected from multiple open-source VQA datasets (ShareGPT4V 31.3k, ChartQA 17.2k, A-OKVQA 16.1k, AI2D 11.4k, GeoQA+ 11.4k, ScienceQA 5.6k, totaling ~98.6k samples). GPT-4o is used to generate four-stage structured reasoning annotations for each sample. The data covers two broad categories: general VQA and scientific reasoning.

3. **SWIRES (Stage-Wise Retracing Search)**: A stage-wise backtracking search strategy for test-time scaling. At the end of each reasoning stage, the model generates multiple candidate continuations and selects the most promising path, analogous to beam search but operating at the granularity of reasoning stages rather than tokens. This enables the model to trade additional computation for higher accuracy at inference time.

### Loss & Training
Standard autoregressive language modeling loss is used for fine-tuning. Training configuration: 8-GPU parallel training, learning rate $1\times10^{-5}$, 3 epochs, batch size 4, using FSDP distributed training. Special tags (e.g., `<SUMMARY>`) are treated as ordinary tokens during training.

## Key Experimental Results

| Dataset | Metric | LLaVA-CoT (11B) | Llama-3.2-90B-Vision | GPT-4o-mini | Gemini-1.5-pro |
|--------|------|---------|----------|------|------|
| MMStar | Acc | Significant improvement | Below LLaVA-CoT | Below LLaVA-CoT | Below LLaVA-CoT |
| MMBench | Acc | Significant improvement | Below LLaVA-CoT | Below LLaVA-CoT | Below LLaVA-CoT |
| MathVista | Acc | Significant improvement | Below LLaVA-CoT | Below LLaVA-CoT | Below LLaVA-CoT |

- Compared to the base model Llama-3.2-11B-Vision-Instruct, average improvement of **9.4%** across 6 multimodal reasoning benchmarks.
- The 11B model surpasses Llama-3.2-90B-Vision-Instruct (8× larger) as well as closed-source models GPT-4o-mini and Gemini-1.5-pro.
- SWIRES yields additional performance gains with manageable computational overhead.

### Ablation Study
- All four stages are essential: removing either the Caption or Reasoning stage leads to significant performance degradation.
- Dataset scale matters, but 100k samples are sufficient to achieve strong performance.
- SWIRES is more efficient than greedy decoding and standard beam search; stage-level search proves more effective than token-level search.

## Highlights & Insights
- **Small model outperforms large models**: With only 11B parameters and 100k training samples, the model surpasses 90B and closed-source counterparts, demonstrating that structured reasoning matters far more than scale.
- **Intuitive four-stage design**: The Summary → Caption → Reasoning → Conclusion pipeline mirrors human problem-solving, with the separation of image understanding from logical inference being a particularly principled design choice.
- **SWIRES as a general test-time scaling method**: Stage-level search is more efficient than token-level search and allows flexible trade-offs between inference time and accuracy.
- **Reusable data construction pipeline**: The GPT-4o-based structured reasoning annotation pipeline can be transferred to other reasoning tasks.

## Limitations & Future Work
- Training data generation relies on GPT-4o, so data quality is bounded by GPT-4o's capabilities.
- The four-stage structure is fixed; simpler questions may not require all stages, leading to unnecessary computation.
- Validation is currently limited to Llama-3.2-Vision; compatibility with other VLM architectures (e.g., Qwen-VL, InternVL) remains unexplored.
- The search space of SWIRES grows exponentially with the number of stages; extending to more stages may require better pruning strategies.

## Related Work & Insights
- **vs. CoT prompting**: Conventional CoT guides models to display reasoning via prompts; LLaVA-CoT internalizes structured reasoning through training, eliminating the need for elaborate prompt design.
- **vs. LLaVA series**: The simple LLaVA architecture is retained; the core contribution lies in the structured annotation of training data rather than architectural modifications.
- **vs. o1-like models**: Conceptually aligned with OpenAI o1 (enhancing reasoning via training-time and inference-time computation), but represents an open-source implementation in the multimodal domain.

## Related Work & Insights
- The structured reasoning annotation approach can inspire data construction for other VLM tasks (e.g., visual grounding, image captioning).
- The effectiveness of test-time scaling in VLMs warrants further investigation.

## Rating
- Novelty: ⭐⭐⭐⭐ — The four-stage reasoning idea is intuitive and simple yet effective; SWIRES demonstrates additional ingenuity.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive validation across 6 benchmarks with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with convincing qualitative demonstrations.
- Value: ⭐⭐⭐⭐⭐ — An open-source multimodal reasoning solution with strong practical impact (2.1k stars).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[ICCV 2025\] ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools](toolvqa_a_dataset_for_multi-step_reasoning_vqa_with_external_tools.md)
- [\[ICCV 2025\] DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding](docthinker_explainable_multimodal_large_language_models_with.md)
- [\[ICCV 2025\] orderchain towards general instruct-tuning for stimulating the ordinal understan](orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)
- [\[ICCV 2025\] ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning](chartpoint_guiding_mllms_with_grounding_reflection_for_chart_reasoning.md)

</div>

<!-- RELATED:END -->
