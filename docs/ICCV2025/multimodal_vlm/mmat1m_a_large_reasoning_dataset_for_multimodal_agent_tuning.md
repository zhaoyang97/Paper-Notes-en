---
title: >-
  [Paper Note] MMAT-1M: A Large Reasoning Dataset for Multimodal Agent Tuning
description: >-
  [ICCV 2025][Multimodal VLM][multimodal agent tuning] This paper introduces MMAT-1M, the first million-scale multimodal agent tuning dataset, constructed via a four-stage data engine (Foundation → Rationale → Reflection → Integration). It endows MLLMs with CoT reasoning, tool invocation, and self-reflection capabilities, achieving an average improvement of 2.7% on InternVL2.5-8B and 8.8% on RAG tasks.
tags:
  - ICCV 2025
  - Multimodal VLM
  - multimodal agent tuning
  - chain-of-thought
  - reflection
  - tool usage
  - reasoning dataset
date: 2026-05-08
content_hash: d35edf810cd5d3da
---

# MMAT-1M: A Large Reasoning Dataset for Multimodal Agent Tuning

**Conference**: ICCV 2025
**arXiv**: [2507.21924](https://arxiv.org/abs/2507.21924)
**Code**: [https://github.com/VIS-MPU-Agent/MMAT-1M](https://github.com/VIS-MPU-Agent/MMAT-1M)
**Area**: Multimodal VLM / Agent
**Keywords**: multimodal agent tuning, chain-of-thought, reflection, tool usage, reasoning dataset

## TL;DR
This paper introduces MMAT-1M, the first million-scale multimodal agent tuning dataset, constructed via a four-stage data engine (Foundation → Rationale → Reflection → Integration). It endows MLLMs with CoT reasoning, tool invocation, and self-reflection capabilities, achieving an average improvement of 2.7% on InternVL2.5-8B and 8.8% on RAG tasks.

## Background & Motivation
**Background**: LLMs have achieved notable progress in CoT reasoning and tool use through agent tuning; however, the multimodal domain still lacks large-scale, high-quality agent tuning datasets. Existing multimodal agent datasets (e.g., LLaVA-Plus at 117K, MM-Traj at 20K) are limited in scale and distribution diversity.

**Limitations of Prior Work**: Existing multimodal agent datasets suffer from three key deficiencies: (1) narrow data distribution that fails to improve model performance across diverse benchmarks; (2) absence of reflection mechanisms to handle errors introduced by vision tools, resulting in poor model robustness; and (3) inflexible reasoning and tool-use mechanisms that limit practical applicability.

**Key Challenge**: Multimodal large models must simultaneously possess reasoning capability, tool invocation capability, and error correction capability, yet existing datasets cannot cover all three dimensions at adequate scale.

**Goal**: To construct a million-scale multimodal agent tuning dataset that jointly supports CoT reasoning, dynamic API calls, and reflective error correction, provided in both one-turn and multi-turn formats to balance efficiency and accuracy.

**Key Insight**: Starting from existing public multimodal QA datasets, the paper uses GPT-4o to progressively generate reasoning trajectories, dynamically integrate tool call results, and repair logical gaps and answer-leaking behaviors through a reflection step.

**Core Idea**: A four-stage data engine (Foundation → Rationale → Reflection → Integration) synthesizes million-scale agent tuning data from public QA pairs, simultaneously supporting CoT, tool invocation, and self-reflection.

## Method

### Overall Architecture
The input consists of image–QA pairs from public multimodal datasets. After processing through the four-stage data engine, the pipeline outputs agent tuning data in two formats: a multi-turn Rationale and Reflection (RR) format and a one-turn Rationale and Reflection (ORR) format. Open-source MLLMs are then fine-tuned using LoRA.

### Key Designs

1. **Four-Stage Data Engine**:

    - **Foundation Stage**: Image–QA pairs are collected from five public datasets—Visual CoT, LLaVA-CoT, The Cauldron, TabMWP, and InfoSeek—with unified prompt templates for input and output formatting. Five categories of external tools are also prepared: Image Caption (based on CCoT scene graphs), OCR (PaddleOCR), OVD (Grounding DINO), Face Detection (deepface), and RAG (Google Search).
    - **Rationale Stage**: GPT-4o iteratively generates reasoning trajectories. The model adaptively selects which operators to invoke based on task requirements (e.g., Caption for global semantic understanding, OVD for object-level information). Each reasoning step explicitly records the thought process, operator call, and subsequent action in a structured STRING format.
    - **Reflection Stage**: Two categories of reasoning defects are addressed: (1) *reasoning gaps*—omission of critical steps in mathematical derivations; and (2) *reasoning cheating*—GPT-4o force-aligning the reasoning process to a known answer rather than genuinely deriving it. GPT-4o detects and repairs both defect types, enhancing the logical completeness of the training data.
    - **Integration Stage**: Multi-turn dialogues are compressed into the ORR (One-turn Rationale and Reflection) format, in which all tool outputs are prepended to the input and the reasoning process is consolidated into a single-turn response. ORR substantially accelerates inference while preserving reasoning capability.
    - **Design Motivation**: The multi-turn RR format yields higher accuracy but incurs greater inference overhead, while the ORR format is more efficient but cannot dynamically invoke RAG. The two formats are complementary and cover different application scenarios.

2. **Dynamic Invocation of Five External Tool Categories**:

    - **Function**: Adaptively invoke Image Caption, OCR, OVD, Face Detection, and RAG operators during the reasoning process.
    - **Mechanism**: The model first analyzes the question requirements to determine which type of visual information is needed, then calls the corresponding operator and integrates its output into subsequent reasoning. Image Caption constructs a scene graph via CCoT before generating descriptions; OVD uses Grounding DINO for open-vocabulary detection.
    - **Design Motivation**: Different tasks require different types of visual information; a fixed tool chain is insufficiently flexible. Dynamic tool selection enables the model to retrieve the needed information on demand during inference.

3. **Reflection and Error-Correction Mechanism**:

    - **Function**: Detect and repair logical defects in reasoning trajectories using two types of prompts—general reflection and mathematical reflection.
    - **Mechanism**: The general reflection prompt instructs GPT-4o to check for "cheating" behavior (conclusion-first, reverse-engineered derivation); the mathematical reflection prompt checks for reasoning gaps and fills in omitted steps.
    - **Design Motivation**: Reasoning trajectories generated directly by GPT-4o are insufficiently reliable, exhibiting logical inconsistencies and step omissions. The reflection mechanism corrects approximately 57K samples, improving the reliability of training data and enabling models to learn self-correction behavior.

### Loss & Training
LoRA fine-tuning is applied, augmenting the standard cross-entropy loss with Frobenius norm regularization: $L = L_{\text{original}} + \lambda \sum_i \|\Delta\theta_i\|_F^2$. Training is conducted on the full 1,090,263 QA pairs in MMAT-1M for 1 epoch, with a learning rate of 4e-5, using the ms-swift framework with ZeRO-2 parallelism.

## Key Experimental Results

### Main Results

| Model | Strategy | MMStar | MMMU | MathVista | MathVision | AI2D | OCRBench | RealWorldQA | Avg |
|-------|----------|--------|------|-----------|------------|------|----------|-------------|-----|
| InternVL2.5-8B | Baseline | 62.4 | 53.1 | 64.5 | 20.1 | 84.1 | 819 | 69.4 | 60.7 |
| InternVL2.5-8B | ORR | 64.8 | 55.4 | 63.8 | 20.8 | 83.5 | 849 | 73.0 | 62.4 |
| InternVL2.5-8B | RR | **65.3** | **57.3** | **64.8** | **21.7** | **84.2** | 839 | **74.4** | **63.4** |
| Llama-3.2-11B | Baseline | 47.7 | 50.3 | 48.0 | 16.4 | 77.1 | 756 | 63.4 | 52.2 |
| Llama-3.2-11B | RR | 51.4 | 51.0 | 49.1 | 16.8 | 77.9 | 784 | 69.3 | 55.3 |

RAG benchmark (Dyn-VQA, F1-Recall):

| Model | Query | Golden Query |
|-------|-------|-------------|
| InternVL2.5-8B Baseline | 27.0 | 35.2 |
| InternVL2.5-8B-RR | **36.8** (+36.3%) | **44.0** (+25.0%) |
| Llama-3.2-11B Baseline | 29.4 | 34.6 |
| Llama-3.2-11B-RR | **38.0** (+29.3%) | **45.1** (+30.3%) |

### Ablation Study

| Configuration | Avg | Dyn-VQA | Note |
|---------------|-----|---------|------|
| Baseline-RR (Full) | 61.3 | 44.0 | Full model |
| w/o API | 57.3 | 43.4 | Removing API tool calls drops average by 4.0 |
| w/o RAG | 59.8 | 35.4 | Removing RAG substantially degrades Dyn-VQA |
| w/o SFT | 55.0 | 31.5 | Direct RR-format inference without fine-tuning |
| w/o Reflection (R only) | 60.2 | 42.9 | Removing reflection drops 1.1 |
| ORR | 59.6 | 36.6 | One-turn format; cannot invoke RAG dynamically |

### Key Findings
- API tool invocation contributes the most (average drop of 4.0 when removed), validating the need for external tools in multimodal agents.
- RAG is critical for knowledge-intensive tasks (Dyn-VQA), with performance dropping from 44.0 to 35.4 upon removal.
- The reflection mechanism yields a 1.1-point gain; while not the largest single contributor, it is important for reasoning consistency.
- RR underperforms ORR on OCRBench (839 vs. 849), as OCR misrecognition in multi-turn reasoning can disrupt the reflection pipeline, whereas ORR can compensate for OCR errors via Image Caption.
- Inference latency is approximately 2× the baseline for ORR and 3–4× for RR, with correspondingly higher accuracy.
- Models acquire zero-shot capability to invoke unseen tools (e.g., celebrity recognition).

## Highlights & Insights
- **The four-stage data engine is highly systematic**: each stage has a clear objective, from data collection through rationale generation to reflective error correction. In particular, the reflection mechanism addresses the "reasoning cheating" problem in GPT-4o-generated data—a practically important engineering insight.
- **The ORR and RR formats are complementary**: one prioritizes efficiency and the other accuracy, allowing users to select the appropriate format for their application. This dual-format strategy is transferable to other agent dataset construction pipelines.
- **Zero-shot tool generalization**: the model demonstrates invocation capability for tools not seen during training (e.g., celebrity recognition), indicating that agent tuning teaches a general ability of "when and how to call tools" rather than memorizing specific tool usage patterns.

## Limitations & Future Work
- The pipeline depends on GPT-4o for rationale generation, incurring high cost and potentially introducing GPT-4o-specific biases (although 89% of samples are high quality, 11% remain problematic).
- The tool set is fixed at five categories and does not cover additional tools such as code execution or database queries; complex multi-tool composition scenarios are underrepresented.
- Validation is limited to models at the 8B scale or below; evaluation on larger models and a broader range of MLLM architectures remains to be conducted.
- Since reflection is injected during data generation, it is unclear to what extent the model exhibits spontaneous self-reflection during inference.

## Related Work & Insights
- **vs. LLaVA-Plus**: LLaVA-Plus contains only 117K samples and lacks a reflection mechanism; MMAT-1M is nearly 10× larger and additionally incorporates reflection and RAG.
- **vs. T3-Agent / MM-Traj**: MM-Traj contains only 20K trajectories; MMAT-1M's advantages lie in its million-scale volume and dual-format output.
- **vs. LLaVA-CoT**: LLaVA-CoT focuses on CoT reasoning but does not involve tool invocation; MMAT-1M extends CoT with tool use and self-reflection.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The first million-scale multimodal agent tuning dataset; conceptually a breakthrough, though the underlying methodology (GPT-4o generation + reflection) follows a relatively standard data synthesis pipeline.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 8 benchmarks plus 1 RAG benchmark with multiple baseline models and complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured; the four-stage pipeline is described systematically and clearly.
- **Value**: ⭐⭐⭐⭐ — Fills a meaningful gap in multimodal agent tuning data and offers practical value to the research community.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools](toolvqa_a_dataset_for_multi-step_reasoning_vqa_with_external_tools.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[NeurIPS 2025\] Situat3DChange: Situated 3D Change Understanding Dataset for Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/situat3dchange_situated_3d_change_understanding_dataset_for_multimodal_large_lan.md)
- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](../../CVPR2026/multimodal_vlm/mmrad_multimodal_anomaly_detection.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)

<!-- RELATED:END -->
