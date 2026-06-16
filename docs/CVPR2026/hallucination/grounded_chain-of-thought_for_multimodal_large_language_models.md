---
title: >-
  [Paper Note] Grounded Chain-of-Thought for Multimodal Large Language Models
description: >-
  [CVPR 2026][Hallucination Detection][Paper Note] This paper proposes the "Grounded Chain-of-Thought (GCoT)" task and the MM-GCoT benchmark. It requires Multimodal Large Language Models (MLLMs) to provide step-by-step reasoning with coordinate-based grounding before answering. By introducing the "Answer-Grounding Consistency" metric to quantify visual hallucinations,
tags:
  - CVPR 2026
  - Hallucination Detection
date: 2026-05-08
content_hash: d2970677e5291b11
---
# Grounded Chain-of-Thought for Multimodal Large Language Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Grounded_Chain-of-Thought_for_Multimodal_Large_Language_Models_CVPR_2026_paper.html)  
**Code**: https://github.com/DoubtedSteam/MM-GCoT  
**Area**: Multimodal VLM / Visual Hallucination / Evaluation Benchmark  
**Keywords**: Multimodal Large Models, Visual Hallucination, Grounded Reasoning, Chain-of-Thought, Evaluation Benchmark

## TL;DR
This paper proposes the "Grounded Chain-of-Thought (GCoT)" task and the MM-GCoT benchmark. It requires Multimodal Large Language Models (MLLMs) to provide step-by-step reasoning with coordinate-based grounding before answering. By introducing the "Answer-Grounding Consistency" metric to quantify visual hallucinations, the study reveals that 12 state-of-the-art MLLMs commonly "answer correctly but look at the wrong place," and hallucinations are independent of model scale.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have approached human-level performance on various vision-language benchmarks. However, they remain weak in spatio-visual reasoning and suffer from "visual hallucinations," where generated descriptions or reasoning steps are not truly grounded in the image content.

**Limitations of Prior Work**: A more deceptive form of hallucination exists where the model provides the **correct answer but based on wrong evidence**. Such models "guess" correctly due to linguistic biases in data distributions while their attention falls on irrelevant regions. Existing hallucination benchmarks only determine answer correctness and fail to reveal whether visual evidence is correctly utilized. Current Grounded QA tasks provide bounding boxes but lack annotated intermediate reasoning steps, making it impossible to track where the model is looking at each step.

**Key Challenge**: Evaluation systems that only look at the final answer misjudge unreliable models (relying on bias) as reliable, posing risks for real-world applications like Embodied AI. To quantify hallucinations, the "answer" must be checked for alignment with "the visual evidence it claims to use."

**Goal**: (1) Design a task to examine visual perception, spatial grounding, and diagnose hallucinations simultaneously; (2) Construct a companion benchmark for objective MLLM evaluation; (3) Reveal the true state of visual reasoning consistency in current MLLMs through experiments.

**Key Insight**: Borrowing from Chain-of-Thought (CoT) in LLMs, this work "grounds" every reasoning step. The model is required to decompose a problem into steps, providing bounding box coordinates for relevant entities in each step as intuitive evidence before giving the final answer.

**Core Idea**: Transform visual hallucinations from "invisible" to "quantifiable" through step-by-step grounded reasoning chains and an answer-grounding consistency metric.

## Method

### Overall Architecture
This work defines a task, builds a benchmark, and establishes an evaluation system. Given an image and a question, GCoT requires the MLLM to decompose the task, reason step-by-step while providing spatial coordinates for task-related elements, and finally provide the answer with coordinates for the final answer region. The MM-GCoT benchmark uses a four-stage pipeline to generate 1,200 samples from Visual Genome with multi-step grounding annotations (categorized into Attribute, Judgement, and Object). The evaluation system introduces three metrics: Answer Accuracy, Grounding Accuracy, and Answer-Grounding Consistency, alongside three prompt settings to elicit grounding capabilities.

### Key Designs

**1. GCoT Task Formalization: Binding "Answers" with "Step-by-step Visual Evidence"**

The pain point is that traditional VQA only considers a mapping $F:I,T\to A$, with no way to know if the answer is based on what is seen. GCoT reformulates this as a multi-step decision process:

$$P(A|I,T)=\prod_{t=1}^{T}P(R_t,G_t|I,T,G_{<t},R_{<t})$$

Where $I, T$ are input image and text, $A$ is the final answer, and $R_t, G_t$ are the textual thought and corresponding spatial coordinates at step $t$. The model must explicitly state which entity it is looking at and its bounding box, exposing the hidden reasoning process. Unlike LLaVA-CoT, which focuses on knowledge reasoning, GCoT focuses on spatio-visual perception. This format is naturally compatible with RL schemes like GRPO for training on data without GCoT annotations.

**2. Four-stage Construction of MM-GCoT: Ensuring Reasoning Complexity and Grounding Precision**

To provide both multi-step reasoning and precise coordinates, a four-stage pipeline was designed. First, region descriptions from Visual Genome are aligned with object annotations via IoU matching. Second, a "Spatial Relation Graph" is built using matched objects as nodes; relation paths are iteratively sampled to generate multi-step reasoning chains. Third, a structured template aggregates bounding boxes, object attributes, and contextual relations. Finally, an LLM translates the templates into natural language questions, followed by human verification. This resulted in 1,200 samples across three categories: Attribute, Judgement, and Object.

**3. Answer-Grounding Consistency Metric: Quantifying Visual Hallucination**

Current metrics fail to capture the "right answer, wrong box" scenario. The authors propose a consistency metric:

$$\text{Con.}=\frac{|S_{ca,cb}|}{|S_{ca,cb}|+|S_{ca,wb}|+|S_{wa,cb}|}$$

Where $S_{ca,cb}$ represents "correct answer and correct box," $S_{ca,wb}$ is "correct answer but wrong box," and $S_{wa,cb}$ is "wrong answer but correct box." Intuitively, it measures the proportion of "truly correct" samples among all samples where either the answer or the box was correct. Low consistency indicates severe visual hallucination.

**4. Three Prompt Settings: Diagnosing Grounding Behavior**

To investigate grounding capabilities, three prompts are used: **Answer-First** (answer then box), **Grounding-First** (final answer box then answer), and **Grounding-CoT** (multi-turn step-by-step entity grounding then answer). Difficulty increases across these settings. The performance gap between these settings serves as a diagnostic signal for reasoning reliability.

## Key Experimental Results

### Main Results
12 representative MLLMs (e.g., LLaVA-OneVision, Qwen2.5-VL, InternVL2.5) were evaluated. The table below shows results for the Answer-First setting (A-Acc = Answer Accuracy, G-Acc = Grounding Accuracy Acc@0.5, Consist. = Consistency, in %).

| Model | A-Acc | G-Acc | Consist. |
|-------|-------|-------|----------|
| LLaVA-OneVision-72B | 74.7 | 16.4 | 15.3 |
| InternVL2.5-78B | 64.0 | 42.9 | 36.6 |
| Qwen2.5-VL-72B | 73.2 | 39.7 | 38.8 |
| **Qwen2.5-VL-7B** | 71.1 | 63.5 | **56.3** |

A significant contrast is observed: LLaVA-OneVision-72B achieves 74.7% answer accuracy but only 15.3% consistency, meaning most correct answers relied on incorrect visual evidence. Qwen2.5-VL-7B achieved the highest consistency (56.3%).

### Ablation Study
Comparison across scales and prompt settings (Consistency %, %):

| Comparison | Config A | Config B | Finding |
|------------|----------|----------|---------|
| Scale (Qwen2.5-VL) | 7B: 56.3 | 72B: 38.8 | 7B outperforms 72B by 17.5 |
| Scale (Grounding-First) | 7B | 72B | 7B is 40.6 higher than 72B |
| Prompt (InternVL2.5-38B) | answer-first: 39.2 | grounding-CoT: 17.8 | Switching to CoT drops 21.4 |
| Intra-step Consist. (Qwen-7B, Judgement) | Step1: 92.8 | Step2: 7.2 | Ratio of providing answer box drops sharply |

### Key Findings
- **Correct Answer $\neq$ Correct Grounding**: Most MLLMs exhibit extremely low consistency, indicating prevalent visual hallucinations where answers rely on linguistic bias.
- **Hallucinations are Scale-Invariant**: For Qwen2.5-VL, answer accuracy only increases by 4% from 3B to 72B, but the 7B model's consistency is 17.5% higher than the 72B model. Large models may overfit to linguistic data.
- **Grounding and Reasoning are Decoupled**: Higher answer accuracy does not guarantee better grounding, and strong grounding does not necessarily lead to better answering.
- **Unreliable Multi-step Reasoning**: Grounding accuracy shows no significant trend across steps; models often do not rely on previously provided visual evidence.

## Highlights & Insights
- **Consistency is the Key to Quantifying Hallucination**: Answer accuracy overestimates reliability; the consistency metric allows "correct answer, wrong evidence" to be measured for the first time.
- **Graph-based Sampling is Effective**: Sampling from Visual Genome relations naturally yields reachable reasoning chains with ground-truth coordinates.
- **Scale Does Not Solve Hallucination**: Small models outperforming large models in consistency suggests hallucinations are a structural/objective issue rather than a parameter count issue.
- **Prompt Settings as Diagnostic Probes**: The performance drop across settings reflects the degree of decoupling between a model's answering and grounding abilities.

## Limitations & Future Work
- **Diagnostic Focus**: The paper provides a benchmark and diagnosis but lacks a systematic training method to reduce hallucinations.
- **Dependency on Visual Genome**: The benchmark is subject to annotation noise and constraints of the VG dataset.
- **Scale**: 1,200 samples across three tasks is relatively small compared to the diversity of real-world spatio-visual reasoning.
- **Prompt Sensitivity**: Results vary by prompt format, which may reflect instruction-following issues rather than pure capability.

## Related Work & Insights
- **vs. Hallucination Benchmarks (POPE, etc.)**: Existing benchmarks check if descriptions are supported but cannot see how visual evidence is used in reasoning. MM-GCoT exposes this process.
- **vs. Grounded QA**: Grounded QA verifies reliability with boxes but lacks intermediate steps; GCoT grounds evidence throughout the entire reasoning chain.
- **vs. Visual CoT (LLaVA-CoT / ScienceQA)**: Prior CoT works focus on knowledge and do not verify the visual evidence of each step; GCoT focuses on perception and coordinate verification.

## Rating
- Novelty: ⭐⭐⭐⭐ GCoT task + Consistency metric provides a new perspective to quantify hallucinations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 12 MLLMs and multiple scales/prompts.
- Writing Quality: ⭐⭐⭐⭐ Clear task definitions and dense, informative tables.
- Value: ⭐⭐⭐⭐ Establishes a new standard for multimodal trustworthiness evaluation, especially for Embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] MAD: Modality-Adaptive Decoding for Mitigating Cross-Modal Hallucinations in Multimodal Large Language Models](mad_modality-adaptive_decoding_for_mitigating_cross-modal_hallucinations_in_mult.md)
- [\[CVPR 2026\] Prefill-Time Intervention for Mitigating Hallucination in Large Vision-Language Models](prefill-time_intervention_for_mitigating_hallucination_in_large_vision-language_.md)
- [\[CVPR 2026\] PAS: Prelim Attention Score for Detecting Object Hallucinations in Large Vision-Language Models](pas_prelim_attention_score_for_detecting_object_hallucinations_in_large_vision-l.md)

</div>

<!-- RELATED:END -->
