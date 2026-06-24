---
title: >-
  [Paper Note] TheoremExplainAgent: Towards Video-based Multimodal Explanations for LLM Theorem Understanding
description: >-
  [ACL 2025][Multimodal VLM][theorem explanation] This paper proposes TheoremExplainAgent, a dual-agent system (Planner + Coder) that automatically generates up to 10-minute-long theorem explanation videos via Manim animation scripts. Accompanying this is TheoremExplainBench (240 STEM theorems evaluated across 5 dimensions), proving that agentic planning is key to generating long-form videos, and showing that visual explanations can expose reasoning flaws that textual evaluatio…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "theorem explanation"
  - "video generation"
  - "Manim animation"
  - "LLM agent"
  - "STEM education"
date: 2026-05-08
content_hash: d44be31be49de70e
---

# TheoremExplainAgent: Towards Video-based Multimodal Explanations for LLM Theorem Understanding

**Conference**: ACL 2025  
**arXiv**: [2502.19400](https://arxiv.org/abs/2502.19400)  
**Code**: [GitHub](https://tiger-ai-lab.github.io/TheoremExplainAgent/)  
**Area**: Multimodal VLM  
**Keywords**: theorem explanation, video generation, Manim animation, LLM agent, STEM education

## TL;DR

This paper proposes TheoremExplainAgent, a dual-agent system (Planner + Coder) that automatically generates up to 10-minute-long theorem explanation videos via Manim animation scripts. Accompanying this is TheoremExplainBench (240 STEM theorems evaluated across 5 dimensions), proving that agentic planning is key to generating long-form videos, and showing that visual explanations can expose reasoning flaws that textual evaluations fail to detect.

## Background & Motivation

**Background**: Comprehending domain-specific theorems often requires not only textual reasoning but also structured visual explanations to deepen understanding. While LLMs have demonstrated strong performance in textual reasoning and theorem proving, existing benchmarks like TheoremQA and GSM8K primarily evaluate theorem comprehension through multiple-choice or short-answer questions.

**Limitations of Prior Work**: ❶ Single evaluation format—multiple-choice questions are highly susceptible to exploitation by superficial cues (such as option order) and fail to truly measure the depth of conceptual comprehension. ❷ Lack of a visual dimension—theorem reasoning is inherently multimodal; understanding fields like geometry, topology, and algebra heavily relies on visual representations, yet existing evaluations are entirely text-based. ❸ The capability of AI to generate multimodal explanations remains an open challenge—despite the robust text capabilities of LLMs, whether they can generate coherent and pedagogically meaningful visual explanations remains unexplored.

**Core Problem**: Can AI systems effectively generate multimodal theorem explanations? More crucially, can the visual generation process expose deeper reasoning flaws masked by text-only evaluations?

**Key Insight**: Elevate the evaluation of theorem comprehension from "multiple-choice/short-answer questions" to "generating video explanations," producing long-form videos through code-driven Manim animations while constructing a standardized evaluation framework.

## Method

### Overall Architecture

TheoremExplainAgent (TEA) employs a dual-agent pipeline:

- **Input**: Theorem name + brief description
- **Planner Agent**: Generates a high-level story plan $\rightarrow$ partitions it into multiple scenes $\rightarrow$ details the visual elements, animations, and transition effects for each scene $\rightarrow$ generates narrative voiceover scripts.
- **Coding Agent**: Converts scene specifications into Manim Python scripts $\rightarrow$ executes the code $\rightarrow$ enters an error correction loop (up to $N=5$ retries) $\rightarrow$ generates speech voiceovers via TTS.
- **Output**: A theorem explanation video featuring animations, structured derivations, and spoken voiceover ($>1$ minute, up to 10 minutes).

### Key Designs

1. **Manim Code-Driven Video Generation**:

    - **Function**: Generates mathematical animations by producing executable Python scripts instead of directly synthesizing pixel-level videos.
    - **Mechanism**: Manim is the open-source mathematical animation library used by 3Blue1Brown. This code-driven approach is naturally suited for LLM generation—LLMs excel at code generation but struggle with direct pixel control.
    - **Design Motivation**: In comparative experiments, pure text-to-video models (LTXVideo, Veo2) generated completely unusable content (visually incoherent and unrelated to the theorems), demonstrating the necessity of the code-driven approach.

2. **Agentic Error Correction Loop ($N=5$ Retries)**:

    - **Function**: When the Coding Agent encounters an error during execution, it automatically inspects the error message and generates a corrected version of the code.
    - **Mechanism**: Code generation itself is error-prone (due to Manim API hallucinations, LaTeX rendering errors, and general Python bugs), but the success rate can be significantly boosted through iterative retries.
    - **Design Motivation**: The success rate is only 3–7% when $N=0$, but o3-mini reaches 91–96% when $N=5$, proving the critical importance of the retry mechanism.

3. **Agentic RAG (Retrieval-Augmented Generation)**:

    - **Function**: Uses Manim documentation as a knowledge base to dynamically retrieve information across three stages.
    - **Mechanism**: ❶ In the storyboard generation stage, retrieves visual examples and related concepts $\rightarrow$ ❷ In the technical implementation stage, retrieves code snippets and usage patterns $\rightarrow$ ❸ In the error correction stage, retrieves diagnostic information and potential fixes.
    - **Design Motivation**: While theoretically expected to aid code generation, experiments reveal that RAG actually harms strong models like o3-mini (dropping success rate from 93.8% to 82.1%), as retrieved results often mismatch the specific scenario and introduce noise.

### Loss & Training

This work does not involve model training. The evaluation framework of TheoremExplainBench comprises five dimensions:
- **Accuracy & Depth**, **Logical Flow**: Textual evaluation of SRT subtitles based on GPT-4o.
- **Visual Relevance**, **Element Layout**: Keyframe extraction + GPT-4o image evaluation.
- **Visual Consistency**: Video segment analysis via Gemini 2.0-Flash.
- **Composite Score** = Geometric mean of all dimensions (range 0–1), using greedy decoding ($\text{temperature}=0$) to ensure output stability.

## Key Experimental Results

### Main Results

**Video Generation Success Rate** (4 Agents $\times$ 3 Difficulties $\times$ 4 Disciplines):

| Agent | Easy | Medium | Hard | Math | Phys | CS | Chem | Overall |
|-------|------|--------|------|------|------|----|------|---------|
| o3-mini | **93.8%** | **91.2%** | **96.2%** | 95.0% | 93.3% | 93.3% | 93.3% | **93.8%** |
| GPT-4o | 61.3% | 57.5% | 46.2% | 61.7% | 55.0% | 58.3% | 45.0% | 55.0% |
| Gemini 2.0-Flash | 20.0% | 11.2% | 12.5% | 16.7% | 8.3% | 21.7% | 11.7% | 14.6% |
| Claude 3.5-Sonnet v1 | 2.5% | 1.2% | 2.5% | 1.7% | 1.7% | 1.7% | 3.3% | 2.1% |

**Video Quality Ratings** (Out of 1.0, evaluated only on successfully generated videos):

| Agent | Accuracy | Visual Relevance | Logical Flow | Element Layout | Visual Consistency | Composite |
|-------|--------|-----------|---------|---------|-----------|------|
| GPT-4o | **0.79** | 0.79 | 0.89 | 0.59 | 0.87 | 0.78 |
| o3-mini | 0.76 | 0.76 | **0.89** | **0.61** | **0.88** | **0.77** |
| Human-authored Manim Videos | 0.80 | **0.81** | 0.70 | **0.73** | 0.87 | 0.77 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| $N=0$ (No retry) | Success rate 3–7% | Single-attempt code generation is almost impossible to succeed. |
| $N=1$ | Success rate 33–51% | The first retry brings the largest improvement. |
| $N=5$ | Success rate 91–96% (o3-mini) | Convergence point; further increases yield diminishing returns. |
| o3-mini + RAG | Success rate 82.1% (decreased by 11.7%) | RAG is actually detrimental to strong models. |
| GPT-4o + RAG | Success rate 45.8% (decreased by 9.2%) | RAG universally degrades the success rate. |
| Agentless Approach | Video $\le 20$ seconds | Fails to generate long videos, proving the necessity of agentic planning. |
| Text-to-Video Models | Visually incoherent, irrelevant content | LTXVideo/Veo2 are completely unusable. |

### Key Findings

- **o3-mini dominates other models in success rate**: 93.8% vs. 55.0% for GPT-4o, demonstrating that strong reasoning models possess an absolute advantage in code-driven visual content generation.
- **Claude 3.5-Sonnet fails almost completely**: Scoring only a 2.1% success rate, exposing its severe deficiency in generating Manim code.
- **RAG is counterproductive**: Reducing the success rate of o3-mini from 93.8% to 82.1%, as context retrieved from documentation often mismatches the specific scene and introduces noise.
- **Element layout is a common weak point for all models**: peaking at only 0.61 (o3-mini) compared to 0.73 for human videos, indicating that spatial reasoning remains a bottleneck.
- **Visual explanations expose deeper reasoning flaws**: While 15 participants initially judged all text-based explanations as correct, 60% changed their judgment to incorrect after viewing the generated videos—visualization forces the AI to explicitly encode structural knowledge, making errors much easier to detect.
- **Human videos score lower in logical flow (0.70 vs. 0.89)**: Human videos prioritize intuition and interactivity, whereas AI-generated videos strictly adhere to formal, logical structures.
- **Chemistry is the most challenging domain**: Complex entities (e.g., beakers, molecules) are far harder to visualize programmatically than simple geometric primitives in math.

## Highlights & Insights

- **Task formulation is the core contribution**: Shifting theorem comprehension evaluation from "answering multiple-choice questions" to "generating explanation videos" introduces an entirely different evaluation dimension that is closer to "genuine comprehension."
- **"Generation as Understanding" evaluation paradigm**: If an AI can generate a correct animated explanation, it demonstrates that it indeed understands the structures and processes underlying the theorem.
- **Multimodal explanation as a reasoning flaw detector**: Visualization exposes hidden errors within text—a finding with profound implications for both AI evaluation and educational applications.
- **Necessity of agentic methods**: Agentless baselines can only generate videos of $\le 20$ seconds, while agents can produce up to 10 minutes, proving that planning capability is the cornerstone of long-form content generation.

## Limitations & Future Work

- Visual layout quality remains suboptimal, with frequent issues such as text overlapping, misaligned shapes, and inconsistent sizing.
- Dependence on the capacity boundaries of the Manim library: certain complex visualizations (such as 3D interactions and chemical molecular structures) are limited by Manim's representational expressive limits.
- Limited alignment between automatic metrics and human judgment: Spearman's $\rho = 0.14$ for accuracy & depth and $\rho = 0.16$ for logical flow, with only visual relevance showing decent alignment at $\rho = 0.72$.
- Evaluated only in English; STEM education is highly localized, and multilingual applicability remains unexplored.
- High computational cost: each theorem requires multiple LLM calls, code executions, and TTS generation, resulting in approximately $1500 USD in API costs.
- Lack of user learning outcome research: no controlled experiments were conducted to evaluate whether these videos genuinely help students comprehend theorems.

## Related Work & Insights

- **vs. TheoremQA**: TheoremQA evaluates via multiple-choice/short-answer questions, which can be easily exploited by surface cues; this work requires generating full-length videos, which is substantially more rigorous.
- **vs. MatPlotAgent/PlotGen**: Prior AI visualization research focused on static data chart generation; this work expands to mathematical animations and educational videos, which entails much higher complexity.
- **vs. 3Blue1Brown/Manim**: 3Blue1Brown manually crafts Manim scripts to produce high-quality videos; this study explores the feasibility and limits of automating this process using AI agents.
- **vs. Text-to-Video Models**: Generative models like LTXVideo and Veo2 lack reasoning capabilities and cannot generate structured educational content, demonstrating the advantages of the LLM-agent paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The task definition itself is highly innovative, elevating theorem comprehension evaluation to the video generation dimension for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Relatively comprehensive, featuring 4 agents, 240 theorems, retry ablation, RAG comparison, and human studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured; case studies and human evaluations enhance the persuasiveness.
- Value: ⭐⭐⭐⭐ Inspiring for AI education, multimodal evaluation, and agent design, though current practicality is bottlenecked by visual quality constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Breaking Multimodal LLM Safety via Video-Driven Prompting](../../CVPR2026/multimodal_vlm/breaking_multimodal_llm_safety_via_video-driven_prompting.md)
- [\[ACL 2026\] ViLL-E: Video LLM Embeddings for Retrieval](../../ACL2026/multimodal_vlm/vill-e_video_llm_embeddings_for_retrieval.md)
- [\[NeurIPS 2025\] Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM](../../NeurIPS2025/multimodal_vlm/watch_and_listen_understanding_audio-visual-speech_moments_with_multimodal_llm.md)
- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)
- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)

</div>

<!-- RELATED:END -->
