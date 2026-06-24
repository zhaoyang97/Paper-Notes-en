---
title: >-
  [Paper Note] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step
description: >-
  [ICCV 2025][VLM Reasoning][Vision-Language Models] LLaVA-CoT proposes a method enabling vision-language models to perform autonomous multi-stage structured reasoning. By constructing the LLaVA-CoT-100k structured reasoning annotation dataset, the model is trained to sequentially execute four stages—Summary, Caption, Reasoning, and Conclusion—and a Stage-Wise Retracing Search (SWIRES) is proposed for test-time scaling, allowing an 11B model to surpass Gemini-1.5-pro and GPT-4o…
tags:
  - "ICCV 2025"
  - "VLM Reasoning"
  - "Vision-Language Models"
  - "Multi-Stage Reasoning"
  - "Chain-of-Thought"
  - "Test-Time Scaling"
  - "Structured Reasoning"
date: 2026-05-08
content_hash: 860119e8e4ce8fec
---

# LLaVA-CoT: Let Vision Language Models Reason Step-by-Step

**Conference**: ICCV 2025
**arXiv**: [2411.10440](https://arxiv.org/abs/2411.10440)  
**Code**: [https://github.com/PKU-YuanGroup/LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)  
**Area**: Multimodal VLM / LLM Reasoning
**Keywords**: Vision-Language Models, Multi-Stage Reasoning, Chain-of-Thought, Test-Time Scaling, Structured Reasoning

## TL;DR

LLaVA-CoT proposes a method enabling vision-language models to perform autonomous multi-stage structured reasoning. By constructing the LLaVA-CoT-100k structured reasoning annotation dataset, the model is trained to sequentially execute four stages—Summary, Caption, Reasoning, and Conclusion—and a Stage-Wise Retracing Search (SWIRES) is proposed for test-time scaling, allowing an 11B model to surpass Gemini-1.5-pro and GPT-4o-mini.

## Background & Motivation

**Background**: Large language models have demonstrated strong reasoning capabilities through inference-time scaling (e.g., OpenAI o1). However, current vision-language models (VLMs) still struggle to perform systematic and structured reasoning on complex visual question answering tasks. Most VLMs jump directly from question to answer, lacking explicit structure in intermediate reasoning steps.

**Limitations of Prior Work**: While Chain-of-Thought (CoT) prompting can guide models to "think step by step," this approach relies on external prompt engineering and does not internalize systematic reasoning into the model itself. Specifically: (1) standard CoT reasoning steps are unstructured free text without explicit stage delineation; (2) models cannot autonomously decide when to extract visual information versus when to perform logical inference; (3) existing inference-time scaling methods (e.g., beam search) operate at the token level, whose fine granularity makes them inefficient for long-chain reasoning.

**Key Challenge**: VLMs must balance between "fast intuitive answering" and "deep structured reasoning." Simple questions require no deep reasoning, but complex problems demand distinct cognitive stages—first understanding the question, then interpreting the image, followed by reasoning, and finally arriving at a conclusion. Teaching models to autonomously command this structured reasoning process is the core challenge.

**Goal**: (1) Train VLMs to perform autonomous multi-stage structured reasoning without relying on external prompts; (2) Design an efficient inference-time scaling method that exploits the multi-stage structure.

**Key Insight**: The authors observe that humans naturally undergo a "read the question → examine the image → reason → conclude" cognitive process when solving complex visual reasoning problems. Rather than eliciting free-format chains of thought, explicitly encoding this process with structural labels and teaching the model this reasoning paradigm through annotated data is the key motivation.

**Core Idea**: Construct a 100k structured reasoning annotation dataset so that VLMs learn to autonomously reason in stages (using `<SUMMARY>`, `<CAPTION>`, `<REASONING>`, and `<CONCLUSION>` tags), combined with stage-level retracing search for efficient test-time scaling.

## Method

### Overall Architecture

LLaVA-CoT is fine-tuned from Llama-3.2-11B-Vision-Instruct. The training data is the LLaVA-CoT-100k dataset, comprising samples from multiple visual question answering datasets, each annotated with a four-stage structured reasoning chain. The model is trained to autonomously generate outputs for all four stages sequentially given an image and a question, with each stage enclosed in its respective tag. At inference time, the model can either generate a complete reasoning chain autonomously (standard inference), or SWIRES can be applied to perform multiple sampling rounds and select the optimal output at each stage (test-time scaling).

### Key Designs

1. **LLaVA-CoT-100k Structured Reasoning Dataset**:

    - **Function**: Provides structured reasoning training signals for VLMs.
    - **Mechanism**: Samples are collected from multiple VQA sources (e.g., ShareGPT4V, ChartQA, GeoQA+, A-OKVQA, ScienceQA), and GPT-4o is used to generate four-stage structured annotations for each sample. The four stages are: (a) **Summary** (`<SUMMARY>`)—summarizes the question and the solution strategy; (b) **Caption** (`<CAPTION>`)—describes visual information in the image relevant to the question; (c) **Reasoning** (`<REASONING>`)—performs multi-step logical inference based on visual information; (d) **Conclusion** (`<CONCLUSION>`)—produces the final answer. All annotations are explicitly delineated with structural tags, enabling the model to learn the metacognitive ability of "when to observe the image and when to reason" during training. The dataset covers a broad range of tasks despite containing only 100k samples.
    - **Design Motivation**: Unlike conventional CoT data, the reasoning here is not free-format but organized into clearly defined stages. This structure allows the model to learn not only "how to reason" but also "how to organize the reasoning process."

2. **Stage-Wise Retracing Search (SWIRES)**:

    - **Function**: Achieves test-time scaling through stage-level multi-sampling and backtracking during inference.
    - **Mechanism**: At each reasoning stage (Summary / Caption / Reasoning / Conclusion), the model generates multiple candidate outputs. An evaluation strategy (e.g., self-consistency or confidence scoring) selects the best candidate for that stage, which is then used as input for the next stage. If all candidates for a given stage are deemed insufficient, the search backtracks to the previous stage for resampling—hence "retracing." Unlike standard beam search, which searches at the token level, SWIRES searches at the stage level, dramatically reducing the search space (4 stages vs. hundreds of tokens) while making each search unit semantically richer and more reliably evaluable.
    - **Design Motivation**: Standard beam search is extremely inefficient for long-sequence generation due to exponential search space explosion. SWIRES exploits the natural stage structure of reasoning, elevating the search from the token level to the stage level, achieving a substantial reduction in computation while maintaining effectiveness.

3. **Autonomous Multi-Stage Reasoning Mechanism**:

    - **Function**: Enables the model to produce structured reasoning spontaneously without external prompts.
    - **Mechanism**: Through fine-tuning, the model internalizes the `<SUMMARY>→<CAPTION>→<REASONING>→<CONCLUSION>` reasoning pipeline. At inference time, upon receiving an image and a question, the model automatically generates the Summary tag and its content first, outlining the solution plan; then generates the Caption tag to describe visual observations; followed by the Reasoning tag for multi-step inference; and finally produces the Conclusion with the answer. This fixed stage order is not enforced through prompt engineering but is instead a behavioral pattern autonomously acquired through training data.
    - **Design Motivation**: External CoT prompts are unstable and require manual design. Internalizing structured reasoning as the model's natural behavior makes the reasoning process more reliable and interpretable, and allows each stage's output to be independently evaluated and improved.

### Loss & Training

Llama-3.2-11B-Vision-Instruct is used as the base model and fully fine-tuned on LLaVA-CoT-100k. Training employs FSDP (Fully Sharded Data Parallel) with 8-GPU parallelism, a learning rate of $10^{-5}$, 3 epochs, and a batch size of 4 per GPU. The training framework is based on Meta's llama-recipes.

## Key Experimental Results

### Main Results

Comparison on six multimodal reasoning benchmarks (accuracy %):

| Model | MMStar | MMBench | MMVet | MathVista | AI2D | Avg. |
|-------|--------|---------|-------|-----------|------|------|
| Llama-3.2-11B (base) | 49.8 | 65.8 | 57.6 | 48.6 | 77.0 | — |
| GPT-4o-mini | 54.8 | 76.9 | — | 52.4 | 77.8 | — |
| Gemini-1.5-pro | 57.6 | 73.9 | — | 57.7 | 79.1 | — |
| Llama-3.2-90B-Instruct | 56.2 | 78.3 | — | 58.3 | 78.9 | — |
| **LLaVA-CoT (11B)** | **57.6** | **73.8** | **60.8** | **54.8** | **85.0** | — |
| **LLaVA-CoT + SWIRES** | **59.2** | **75.1** | **62.3** | **57.2** | **86.4** | — |

With only 11B parameters, 100k training samples, and SWIRES scaling, LLaVA-CoT surpasses the 90B Llama-3.2 as well as closed-source Gemini-1.5-pro and GPT-4o-mini. The average improvement over the base model is approximately 9.4%.

### Ablation Study

| Configuration | MMStar | MathVista | AI2D | Note |
|---------------|--------|-----------|------|------|
| LLaVA-CoT (Full) | 57.6 | 54.8 | 85.0 | Complete four-stage reasoning |
| w/o Summary stage | 55.8 | 52.3 | 83.1 | No question planning |
| w/o Caption stage | 56.1 | 53.1 | 82.8 | No visual interpretation |
| w/o Reasoning stage | 50.2 | 47.5 | 78.2 | No logical inference (worst degradation) |
| Direct answer (no CoT) | 49.8 | 48.6 | 77.0 | Base model level |
| Standard beam search | 58.0 | 55.5 | 85.8 | Token-level search, high compute cost |
| SWIRES | 59.2 | 57.2 | 86.4 | Stage-level search, more efficient and accurate |

### Key Findings

- The Reasoning stage is the most critical; removing it causes the most severe performance degradation, reducing performance to near the base model level.
- The Summary and Caption stages, while contributing smaller individual gains, are indispensable—they provide structured input for the Reasoning stage.
- SWIRES achieves a dual win over standard beam search in both accuracy and efficiency: higher accuracy (due to more reliable semantic-level evaluation) with lower computational cost (stage-level search space is far smaller than token-level).
- Significant reasoning capability gains are achieved with only 100k training samples, demonstrating the high data efficiency of structured reasoning annotations.
- The largest improvements are observed on tasks requiring deep reasoning, such as MathVista (mathematical reasoning) and AI2D (diagram understanding).

## Highlights & Insights

- **Structured reasoning tags are a simple yet highly insightful design**: organizing free-format CoT into stages with distinct responsibilities through just four tags. This approach adds virtually no training complexity, yet enables the model to acquire metacognitive capabilities—knowing when to observe, when to reason, and when to summarize. The approach is transferable to any multi-step reasoning task.
- **SWIRES elevates search from the token level to the semantic level**: This is an important contribution to inference-time scaling research. Conventional beam search operates at the token level with excessive granularity and a massive search space. SWIRES leverages the natural stage structure of the task for coarse-grained search—a simple yet effective design principle.
- **High-efficiency fine-tuning with 100k samples**: This demonstrates that "data quality >> data quantity"—structured, high-quality reasoning annotations are more effective than massive but unstructured data, offering valuable practical insights for resource-constrained researchers.

## Limitations & Future Work

- The fixed four-stage order may not be suitable for all tasks; some problems may require an iterative process such as "observe → reason → re-observe."
- The stage-level evaluation strategy in SWIRES is relatively simple (self-consistency); more sophisticated evaluation functions could further improve performance.
- The base model is limited to Llama-3.2-11B-Vision; effectiveness on larger models or alternative architectures remains to be validated.
- Training data relies on GPT-4o annotations, introducing an upper bound on annotation quality along with associated cost concerns.
- Dynamic stage count has not been explored—simple problems may require only two steps, while complex ones may benefit from six.
- Further integration with reward models for reinforcement learning could enable the model to learn better reasoning strategies from SWIRES search results.

## Related Work & Insights

- **vs. Chain-of-Thought (CoT)**: Standard CoT uses prompts to elicit unstructured reasoning from models, which do not internalize reasoning structure. LLaVA-CoT trains the model to autonomously produce structured reasoning, with each of the four stages serving a distinct role, yielding greater controllability and interpretability.
- **vs. OpenAI o1**: o1 demonstrates the power of extended thinking at inference time, but its internal mechanism is opaque. LLaVA-CoT provides an open-source, interpretable alternative for structured reasoning, demonstrating that an 11B model can approach the reasoning performance of much larger models.
- **vs. Llama-3.2-90B-Vision**: A model 8× smaller surpasses a base model 8× larger through structured reasoning training, compellingly demonstrating that "teaching models how to think" is more efficient than "scaling model parameters."

## Rating

- **Novelty**: ⭐⭐⭐⭐ Both the structured reasoning stages and SWIRES search are novel, though the core ideas are elegant and straightforward rather than intricate.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison across six benchmarks with thorough ablations, though validation across additional model scales is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The paper is clearly written and easy to follow, with compelling demonstrations and a complete open-source release.
- **Value**: ⭐⭐⭐⭐⭐ A landmark open-source work in multimodal reasoning; 100k data and an 11B model suffice to surpass closed-source large models, offering exceptional practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[NeurIPS 2025\] Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards](../../NeurIPS2025/vlm_reasoning/unveiling_chain_of_step_reasoning_for_visionlanguage_models.md)
- [\[ICCV 2025\] ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools](toolvqa_a_dataset_for_multistep_reasoning_vqa_with_external.md)
- [\[NeurIPS 2025\] Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?](../../NeurIPS2025/vlm_reasoning/can_multi-modal_llms_provide_live_step-by-step_task_guidance.md)
- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](../../ICLR2026/vlm_reasoning/frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
