---
title: >-
  [Paper Note] From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation and Editing
description: >-
  [CVPR 2025][Image Generation][Text-to-Diagram Generation] This paper defines the text-to-diagram generation task, constructs DiagramGenBenchmark (covering 8 categories of diagrams), and proposes a multi-agent framework called DiagramAgent (Plan + Code + Check + Diagram-to-Code), which significantly outperforms existing text-to-image/code methods on diagram generation, coding, and editing tasks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Text-to-Diagram Generation"
  - "Structured Visualization"
  - "Multi-Agent Framework"
  - "Code Generation"
  - "Benchmark"
date: 2026-05-08
content_hash: 8ca922fc33d57792
---

# From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation and Editing

**Conference**: CVPR 2025  
**arXiv**: [2411.11916](https://arxiv.org/abs/2411.11916)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Text-to-Diagram Generation, Structured Visualization, Multi-Agent Framework, Code Generation, Benchmark

## TL;DR
This paper defines the text-to-diagram generation task, constructs DiagramGenBenchmark (covering 8 categories of diagrams), and proposes a multi-agent framework called DiagramAgent (Plan + Code + Check + Diagram-to-Code), which significantly outperforms existing text-to-image/code methods on diagram generation, coding, and editing tasks.

## Background & Motivation

**Background**: Converting text descriptions into structured diagrams (e.g., flowcharts, architecture diagrams, mind maps) is widely needed in education, scientific research, and industry. Existing methods mainly follow two pathways: text-to-image generation (GAN/Diffusion models) and text-to-code generation (LLMs generating executable code).

**Limitations of Prior Work**: Although text-to-image methods (such as Imagen, DALL-E) can generate realistic images, they lack the logical organization and hierarchical structure required for diagrams, resulting in structurally imprecise outputs. Text-to-code methods, while capable of generating basic charts (bar charts, line graphs), have insufficient capability to represent complex diagrams (hierarchical relationships, color coding, layered structures), and their generation results are difficult to modify subsequently.

**Key Challenge**: Structured diagrams require both logical precision and editability, but existing methods can only satisfy one of them—image methods have good visuals but chaotic logic, while code methods have structure but poor expressiveness. Furthermore, there is a lack of specialized evaluation benchmarks.

**Goal**: (1) The lack of a standardized benchmark for text-to-diagram generation; (2) the lack of a unified framework that can handle generation, coding, and editing simultaneously.

**Key Insight**: Decompose diagram generation into multiple sub-tasks (understanding, encoding, verification, reverse parsing), completed cooperatively by different specialized agents, forming a closed-loop error-correcting workflow.

**Core Idea**: Use a multi-agent collaboration framework (Plan + Code + Check + Diagram-to-Code) to decompose the text-to-diagram task into verifiable sub-steps, supported by a specially constructed 8-category diagram benchmark for evaluation.

## Method

### Overall Architecture
DiagramAgent consists of four core agents, supporting three task pipelines: diagram generation (text $\rightarrow$ code $\rightarrow$ diagram), diagram coding (diagram $\rightarrow$ code), and diagram editing (diagram $\rightarrow$ code $\rightarrow$ modification $\rightarrow$ new diagram). The input is the user's text instruction or diagram image, and the output is compilable diagram code and the corresponding visualization.

### Key Designs

1. **Plan Agent (Task Planning & Query Expansion)**:

    - **Function**: Receives user instructions, determines the task type (generation/coding/editing), and performs query expansion for incomplete instructions.
    - **Mechanism**: Qwen-72B is used to analyze the completeness of user instructions. If an instruction lacks nodes, labels, or other information, it is completed by the LLM into a comprehensive query $x_{comp} = f_{expand}(x_{ins})$, which is then routed to the corresponding downstream agent.
    - **Design Motivation**: Users' natural language descriptions are often imprecise, and generating directly can easily lead to missing key elements. Query expansion ensures that the generated code includes all necessary structural and stylistic information.

2. **Code Agent (Code Generation & Fine-Tuning)**:

    - **Function**: Translates processed instructions into executable diagram code (LaTeX/DOT language).
    - **Mechanism**: Based on Qwen2.5-Coder-7B fine-tuned on the DiagramGenBenchmark training set for 4 epochs (max length 8192). The optimization objective is to minimize the difference between the generated code and the reference code $\mathcal{L}_{code}(f_{code}(x), c_{ref})$. For editing tasks, it receives both the original code and the modification instructions simultaneously.
    - **Design Motivation**: General code models do not sufficiently understand grammar and layout rules in the diagram domain; fine-tuning significantly improves generation accuracy and structural consistency.

3. **Check Agent (Debugging & Verification Closed-Loop)**:

    - **Function**: Performs compilation debugging and semantic completeness verification on the generated code.
    - **Mechanism**: Divided into two stages: first, compiling the code and returning to the Code Agent for correction if compilation errors are found; once compilation succeeds, GPT-4o is used to verify the semantic completeness of the code (whether it contains all necessary elements). This forms a dual protection of $f_{check} = f_{debug} + f_{verify}$.
    - **Design Motivation**: Since code generated by LLMs often contains syntax errors or misses elements, the combination of compiler debugging and AI verification effectively improves the final output quality.

### Loss & Training
Both Code Agent and Diagram-to-Code Agent are trained using standard code generation loss to minimize the edit distance and semantic discrepancy between the generated code and the reference code. Fine-tuning uses 8×80G A100 GPUs, with Code Agent based on the 7B model and Diagram-to-Code Agent based on Qwen2-VL-7B.

## Key Experimental Results

### Main Results

| Model | Pass@1↑ | ROUGE-L↑ | CodeBLEU↑ | CLIP-FID↓ | PSNR↑ | MS-SSIM↑ |
|------|---------|----------|-----------|-----------|-------|----------|
| DiagramAgent (7B) | **58.15** | **51.97** | **86.83** | **11.16** | **6.38** | **24.78** |
| DeepSeek-Coder (33B) | 55.56 | 44.26 | 83.29 | 15.49 | 6.02 | 19.80 |
| GPT-4o | 49.81 | 44.59 | 82.83 | 13.26 | 5.56 | 18.21 |
| DeepSeek V2.5 | 54.44 | 43.00 | 82.83 | 13.32 | 5.56 | 16.98 |
| Code-Llama (34B) | 8.89 | 22.92 | 76.78 | 30.12 | 0.89 | 2.32 |

The 7B DiagramAgent exceeds the 33B DeepSeek-Coder by about 3 percentage points on Pass@1, and exceeds GPT-4o by about 8 percentage points.

### Ablation Study

| Configuration | Pass@1↑ | chrF↑ | LPIPS↓ | MS-SSIM↑ |
|------|---------|-------|--------|----------|
| Full model | 58.15 | 53.49 | 45.95 | 24.78 |
| w/o GPT-4o verification | 57.78 (-0.37) | 52.81 (-0.68) | 46.66 (+0.71) | 20.80 (-3.98) |
| w/o compilation debugging | 57.41 (-0.74) | 51.74 (-1.75) | 48.13 (+2.18) | 24.10 (-0.68) |
| w/o both | 57.41 (-0.74) | 51.69 (-1.80) | 48.17 (+2.22) | 20.37 (-4.41) |

### Key Findings
- Compilation debugging (Compiler) has a greater impact on code quality than GPT-4o verification, reducing chrF by 1.75 when omitted.
- GPT-4o verification has the greatest impact on image fidelity (MS-SSIM), dropping by 3.98 when omitted, indicating that the verification module primarily ensures visual completeness.
- The two components are complementary: omitting both leads to the largest drop in MS-SSIM (-4.41), demonstrating that debugging guarantees syntactic correctness while verification guarantees semantic completeness.

## Highlights & Insights
- **Multi-agent task decomposition**: Decomposing text-to-diagram into a closed-loop pipeline of planning $\rightarrow$ coding $\rightarrow$ verification, where each agent focuses on a sub-task. This paradigm can be transferred to any "generation + verification" task.
- **Practicality of DiagramGenBenchmark**: Covering 8 types of diagrams (flowcharts, architecture diagrams, mind maps, etc.), this is the first comprehensive benchmark in this field, filling a critical gap.
- **Small models beating large models**: The fine-tuned 7B model outperforms GPT-4o and 33B models, demonstrating the advantage of domain-specific fine-tuning on structured generation tasks.

## Limitations & Future Work
- Diagram types are limited to those that can be compiled using code (LaTeX/DOT), failing to handle hand-drawn styles or more free-form visualizations.
- Query expansion relies on a 72B large model (Qwen-72B), which has high inference costs; replacing it with smaller models or prompt engineering could be considered.
- Check Agent relies on GPT-4o (closed-source), which limits deployment due to high costs; open-source alternatives can be explored.
- The dataset size (training set ~7K) is relatively limited; scaling to more diagram types and larger-scale data may further improve performance.

## Related Work & Insights
- **vs Text-to-Image Methods (Imagen/DALL-E)**: While they excel at generating natural images, they cannot guarantee structural logic. DiagramAgent resolves structure and editability issues through programming language code as an intermediate representation.
- **vs Text-to-Code Methods (Qwen2.5-Coder)**: General code models lack domain knowledge about diagrams. DiagramAgent significantly improves diagram-specific accuracy via domain fine-tuning and multi-agent verification.
- The multi-agent collaboration paradigm (Plan $\rightarrow$ Execute $\rightarrow$ Verify) is a general design framework that can be transferred to other tasks requiring "generation $\rightarrow$ verification $\rightarrow$ correction."

## Rating
- Novelty: ⭐⭐⭐⭐ The text-to-diagram generation task definition is novel, but the methodology consists of combinations of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage with a comparison of 16 models, ablation studies, and human evaluation.
- Writing Quality: ⭐⭐⭐⭐ The framework and mathematical definitions are clear, but some formulas are overly formalized.
- Value: ⭐⭐⭐⭐ Both the benchmark and the framework have practical value, though the diagram generation scenario remains relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] AutoPresent: Designing Structured Visuals from Scratch](autopresent_designing_structured_visuals_from_scratch.md)
- [\[ICLR 2026\] Factuality Matters: When Image Generation and Editing Meet Structured Visuals](../../ICLR2026/image_generation/factuality_matters_when_image_generation_and_editing_meet_structured_visuals.md)
- [\[CVPR 2025\] Towards Scalable Human-Aligned Benchmark for Text-Guided Image Editing](towards_scalable_human-aligned_benchmark_for_text-guided_image_editing.md)
- [\[CVPR 2025\] PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction](pqpp_a_joint_benchmark_for_text-to-image_prompt_and_query_performance_prediction.md)
- [\[ICML 2025\] One Image is Worth a Thousand Words: A Usability Preservable Text-Image Collaborative Erasing Framework](../../ICML2025/image_generation/one_image_is_worth_a_thousand_words_a_usability_preservable_text-image_collabora.md)

</div>

<!-- RELATED:END -->
