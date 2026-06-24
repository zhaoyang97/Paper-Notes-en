---
title: >-
  [Paper Note] MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?
description: >-
  [ECCV 2024][Multimodal VLM][Visual Mathematical Reasoning] This paper introduces MathVerse, a multimodal mathematical reasoning benchmark containing 2,612 visual math problems (transformed into 6 versions totaling 15K test samples). By systematically manipulating the allocation of information in text and images, MathVerse assesses whether MLLMs truly "understand" mathematical diagrams. The authors also propose a CoT evaluation strategy for fine-grained reasoning process scori…
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Visual Mathematical Reasoning"
  - "Multimodal Evaluation Benchmark"
  - "Diagram Understanding"
  - "CoT Evaluation"
  - "Text Redundancy"
date: 2026-05-08
content_hash: af58fcef836aa4f8
---

# MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?

**Conference**: ECCV 2024  
**arXiv**: [2403.14624](https://arxiv.org/abs/2403.14624)  
**Code**: [Project Page](https://mathverse-cuhk.github.io)  
**Area**: Multimodal VLM  
**Keywords**: Visual Mathematical Reasoning, Multimodal Evaluation Benchmark, Diagram Understanding, CoT Evaluation, Text Redundancy

## TL;DR

This paper introduces MathVerse, a multimodal mathematical reasoning benchmark containing 2,612 visual math problems (transformed into 6 versions totaling 15K test samples). By systematically manipulating the allocation of information in text and images, MathVerse assesses whether MLLMs truly "understand" mathematical diagrams. The authors also propose a CoT evaluation strategy for fine-grained reasoning process scoring, revealing that most MLLMs rely heavily on text rather than visual diagrams for mathematical reasoning.

## Background & Motivation

**Background**: MLLMs have achieved significant progress in visual understanding, and solving mathematical problems containing diagrams is regarded as a key indicator for measuring multimodal logical reasoning capabilities. Existing benchmarks such as GeoQA, MathVista, and MMMU cover various problem types including geometry, functions, and statistics.

**Limitations of Prior Work**: Through in-depth analysis, the authors identified three core issues in existing mathematical benchmarks:
   - (a) **Text Redundancy**: The text of the questions contains a large amount of redundant information that overlaps with the diagrams (e.g., "In triangle ABC, AB and CD intersect at point E" — information that can already be observed from the diagram), allowing MLLMs to answer questions without looking at the diagrams.
   - (b) **Coarse Evaluation**: Evaluating only the correctness of final answers, ignoring differences in the quality of intermediate reasoning steps.
   - (c) **Lack of Specialization**: MathVista contains a large number of non-mathematical reasoning tasks (19/28), while MMMU's college-level questions rely too heavily on prior domain knowledge.

**Key Challenge**: Existing benchmarks may allow MLLMs to "cheat" by reading redundant text instead of understanding the diagrams. Experiments confirm that after removing redundant text, the accuracy of most MLLMs drops precipitously, sometimes falling even below the performance of completely ignoring the visual input.

**Goal**: How to fairly and comprehensively evaluate whether MLLMs truly understand visual mathematical diagrams?

**Key Insight**: The problem text is categorized into three types of information based on importance (Descriptive Information DI, Implicit Property IP, and Essential Condition EC). The textual information is systematically and progressively moved into visual diagrams, generating 6 versions of test questions to incrementally challenge the visual understanding capabilities of MLLMs.

**Core Idea**: Probing the performance of MLLMs under different text-visual information ratios through multi-version problem transformation to quantify their true diagram understanding ability.

## Method

### Overall Architecture

Original math problems $\rightarrow$ Manual annotation of three types of textual information (DI/IP/EC) $\rightarrow$ Generation of 6 versions (from Text-dominant to Vision-only) $\rightarrow$ MLLMs response $\rightarrow$ CoT Evaluation Strategy (Key-step Extraction + Multi-step Scoring) $\rightarrow$ Fine-grained scoring and error analysis

### Key Designs

1. **Three-Tier Question Text Taxonomy**:

    - **Function**: Categorizing the textual content of mathematical problems into three levels based on their importance for solving the problem.
    - **Mechanism**:
        - **Descriptive Information (DI)**: Content directly observable from diagrams, such as "triangle ABC" or "the graph of function $f(x)$" — inherently redundant with the diagrams.
        - **Implicit Property (IP)**: Information that requires higher visual perception but no math knowledge to identify from the diagram, such as parallel, perpendicular, or similarity relationships.
        - **Essential Condition (EC)**: Specific values or algebraic expressions, such as "angle $A = 45^\circ$" or "$BC = 6$", which cannot be derived from visual diagrams.
    - **Design Motivation**: This taxonomy allows systematic control over how much information is provided in the text versus how much understanding is required from the diagram, thereby accurately measuring the visual understanding of MLLMs.

2. **Six-Version Problem Transformation Strategy**:

    - **Function**: Transforming each problem into 6 versions, forming a continuum from text-dominant to purely visual.
    - **Mechanism**:
        - **Text-dominant**: Full text (DI+IP+EC) + diagram (baseline).
        - **Text-lite**: DI removed (IP+EC) + diagram (requires extracting basic information from the diagram).
        - **Text-only**: Full text + no diagram (testing performance without looking at diagrams).
        - **Vision-intensive**: Only EC + diagram (requires identifying spatial relationships from the diagram).
        - **Vision-dominant**: Only IP text + diagram annotated with EC (requires reading numerical values from the diagram).
        - **Vision-only**: No text + all information rendered onto the diagram (ultimate visual challenge).
    - **Design Motivation**: By comparing performance differences across versions, failures in visual understanding can be precisely localized at different levels.

3. **CoT (Chain-of-Thought) Evaluation Strategy**:

    - **Function**: Scoring the reasoning process of MLLMs step-by-step, rather than simply checking the correctness of the final answer.
    - **Mechanism**: Conducted in two stages:
        - **Key-step Extraction**: GPT-4 (text-only) is used to extract $N$ key reasoning steps $[s_1, s_2, \ldots, s_N]$ from the MLLM's output, deliberately withholding the original question and ground-truth answer to avoid GPT-4's own reasoning bias.
        - **Multi-step Scoring**: GPT-4V (multimodal) scores each step using the original question, diagram, and ground-truth answer. The final score is computed as:
    $$\text{Score}_{\text{final}} = \alpha \cdot \frac{1}{N}\sum_{i=1}^{N}\text{Score}(s_i) + (1-\alpha) \cdot \text{Score}(s_A)$$
      where $\alpha=0.7$ emphasizes the importance of the reasoning process.
    - **Design Motivation**: Traditional True/False evaluation cannot distinguish between "correct reasoning but arithmetic error" and "pure guessing." CoT evaluation reveals the true reasoning capabilities of MLLMs.

### Loss & Training

MathVerse is an evaluation benchmark rather than a training method; thus, it does not involve loss functions. All experiments are conducted in a zero-shot setting, using CoT prompting techniques to encourage MLLMs to present complete reasoning steps.

## Key Experimental Results

### Main Results (Comparison of Six Versions, CoT-E Scoring)

| Model | Overall | Text-dominant | Text-lite | Text-only | Vision-intensive | Vision-dominant | Vision-only |
|------|------|--------------|-----------|-----------|-----------------|----------------|-------------|
| GPT-4V | 54.4 | 63.1 | 56.6 | 60.3 | 51.4 | 50.8 | 50.3 |
| Qwen-VL-Max | 37.2 | 42.8 | 37.7 | **47.9**(!) | 33.6 | 35.9 | 35.9 |
| Gemini-Pro | 35.3 | 39.8 | 34.7 | **44.5**(!) | 32.0 | 36.8 | 33.3 |
| InternLM-XC2 | 25.9 | 36.9 | 28.3 | **42.5**(!) | 20.1 | 24.4 | 19.8 |
| SPHINX-MoE | 22.8 | 33.3 | 21.9 | **40.7**(!) | 21.1 | 19.6 | 18.3 |
| Human | 64.9 | 71.2 | 70.9 | 41.7 | 61.4 | 68.3 | 66.7 |

**Note**: (!) indicates that the Text-only version (no diagram) yielded higher accuracy than versions with diagrams/images, exposing the models' lack of true diagram understanding.

### Ablation Study (GPT-4V Error Category Analysis)

| Error Category | Text-dominant | Text-lite | Vision-intensive | Vision-dominant | Vision-only |
|---------|--------------|-----------|-----------------|----------------|-------------|
| Visual Perception Error | Highest ratio | Ratio increases | Significantly increases | Significantly increases | Highest |
| Reasoning Error | High | High | High | High | High |
| Calculation Error | Moderate | Moderate | Moderate | Moderate | Moderate |
| Knowledge Error | Lowest | Lowest | Lowest | Lowest | Lowest |

### Key Findings

- **MLLMs rely more on textual descriptions than diagrams**: Models like Qwen-VL-Max and InternLM-XC2 achieved more than 5% higher accuracy on Text-only (no diagram) than on versions with diagrams! This indicates that visual encoders not only failed to help but instead interfered with reasoning.
- **GPT-4V and ShareGPT4V perform relatively better**: Their scores with diagrams are higher than without diagrams, indicating that their visual encoders make a positive contribution.
- **CoT evaluation differs significantly from traditional evaluation**: GPT-4V's CoT-E score is 16.1% higher than its Acc (Accuracy) score, which indicates that many reasoning steps were correct despite incorrect final answers; traditional evaluation severely underestimates model capabilities.
- **Visual perception errors present the biggest bottleneck**: The proportion of visual perception errors continues to rise as the tasks progress from Text-dominant to Vision-only.
- **Mathematical training is indeed effective**: SPHINX-MoE and InternLM-XC2 incorporate mathematical specialized training data, leading the performance among other open-source models.
- **Text-only LLM (GPT-4) outperforms most MLLMs on text-based versions**: Further proving that visual understanding in current MLLMs remains a critical bottleneck.

## Highlights & Insights

- **Deep insight from text redundancy analysis**: It exposes inflated performance in existing benchmarks — the multimodal mathematical reasoning capability of MLLMs might be heavily overestimated.
- **Exquisite design of the six-version progressive evaluation**: Creating a continuum from Text-dominant to Vision-only allows precise localization of failures in visual understanding.
- **Clever two-stage design of the CoT evaluation strategy**: Withholding the original question during the extraction step prevents GPT-4's own bias, while introducing all information during the scoring step ensures accuracy.
- **The "better performance without looking at diagrams" finding is highly insightful**: It suggests that current MLLM visual encoders behave as noise generators for mathematical diagrams.
- **Transferable experimental design**: The paradigm of multi-version information control and fine-grained CoT evaluation can be extended to other domains requiring precise visual understanding.

## Limitations & Future Work

- Only plain geometry, solid geometry, and functions are covered; statistics, probability, discrete mathematics, etc., are not yet included.
- Problems mostly concentrate on high school level; higher-level mathematical reasoning remains untested.
- CoT evaluation relies on GPT-4/4V, leading to high evaluation costs and potential inherent bias.
- More complex tasks like multi-diagram reasoning or construction of auxiliary lines are not covered.
- Dataset construction depends on manual annotation for text classification and diagram modification, resulting in a high expansion cost.
- Future work can explore training better "visual encoders for mathematics".

## Related Work & Insights

- **vs MathVista [Lu et al.]**: MathVista contains a large number of non-mathematical tasks (19/28) and suffers from severe text redundancy issues, whereas MathVerse focuses specifically on mathematical reasoning and systematically eliminates redundancy.
- **vs MMMU [Yue et al.]**: MMMU's college-level questions rely too heavily on specialized domain knowledge, which may limit the assessment of pure reasoning. MathVerse focuses on high school level to purely test reasoning.
- **vs GeoQA [Chen et al.]**: GeoQA only covers geometry word problems, whereas MathVerse extends to solid geometry and functions while providing multi-version evaluation.
- **vs Traditional Acc Evaluation**: Traditional evaluation fails to distinguish "correct reasoning but wrong result" from "pure guessing"; CoT evaluation offers a fairer, fine-grained metric.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The discovery of the text redundancy issue and the design of the six-version progressive evaluation are highly original, creating a new evaluation paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation of over 13 open-source and closed-source MLLMs across 6 versions and 12 sub-domains, accompanied by fine-grained error analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The introduction of problems flows logically, the diagrams are intuitive and well-designed, and the experimental analysis is deep and structured.
- **Value**: ⭐⭐⭐⭐⭐ It reveals the key bottleneck of MLLM visual mathematical reasoning — visual perception rather than reasoning itself — providing clear directions for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MMBench: Is Your Multi-modal Model an All-Around Player?](mmbench_is_your_multi-modal_model_an_all-around_player.md)
- [\[ECCV 2024\] m&m's: A Benchmark to Evaluate Tool-Use for Multi-step Multi-modal Tasks](m_ampmaposs_a_benchmark_to_evaluate_tool-use_for_multi-step_multi-modal_tasks.md)
- [\[ECCV 2024\] BLINK: Multimodal Large Language Models Can See but Not Perceive](blink_multimodal_large_language_models_can_see_but_not_perceive.md)
- [\[CVPR 2026\] ROSE: Rotate Your Large Language Model to See](../../CVPR2026/multimodal_vlm/rose_rotate_your_large_language_model_to_see.md)
- [\[ECCV 2024\] ShareGPT4V: Improving Large Multi-Modal Models with Better Captions](sharegpt4v_improving_large_multi-modal_models_with_better_captions.md)

</div>

<!-- RELATED:END -->
