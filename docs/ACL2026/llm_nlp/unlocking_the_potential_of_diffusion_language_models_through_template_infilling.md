---
title: >-
  [Paper Note] Unlocking the Potential of Diffusion Language Models through Template Infilling
description: >-
  [ACL2026][LLM/NLP][diffusion language models] This paper proposes Template Infilling (TI), which transforms the generation conditions of Diffusion Language Models (DLMs) from a single prefix into structural anchors distr…
tags:
  - "ACL2026"
  - "LLM/NLP"
  - "diffusion language models"
  - "template infilling"
  - "structured generation"
  - "dynamic span allocation"
  - "code generation"
date: 2026-05-08
content_hash: 1e0a70d0b11d4d26
---

# Unlocking the Potential of Diffusion Language Models through Template Infilling

**Conference**: ACL2026  
**arXiv**: [2510.13870](https://arxiv.org/abs/2510.13870)  
**Code**: None  
**Area**: code_intelligence  
**Keywords**: diffusion language models, template infilling, structured generation, dynamic span allocation, code generation  

## TL;DR
This paper proposes Template Infilling (TI), which transforms the generation conditions of Diffusion Language Models (DLMs) from a single prefix into structural anchors distributed throughout the output. By utilizing Dynamic Span Allocation (DSA) to provide space for complex reasoning, TI significantly stabilizes and improves parallel generation quality across mathematical reasoning, code generation, and global planning tasks.

## Background & Motivation
**Background**: DLMs do not generate tokens sequentially from left to right; instead, they treat a sentence as a holistic entity that can be iteratively denoised, recovering tokens at any position simultaneously. Theoretically, this property is ideal for global planning, code completion, and long-chain reasoning, as the model can perceive both preceding and succeeding constraints simultaneously, unlike autoregressive (AR) models which only rely on prefixes.

**Limitations of Prior Work**: Realistic DLM inference has not fully unleashed this flexibility. Many methods still segment outputs into blocks or borrow prefix prompting from AR models to stabilize sampling, forcing the model along an approximate left-to-right path. While this mitigates numerical instability and context collapse, it re-confines the most valuable arbitrary-position conditioning ability of DLMs into a semi-autoregressive framework.

**Key Challenge**: The high degree of freedom in DLMs is both an advantage and a risk. Without structure, all positions can change simultaneously, causing the number of search paths to explode with length, leading to repetitions, logic drift, and conflicts between local segments. However, if the freedom is suppressed solely via block-wise or prefix constraints, the model's ability to utilize future constraints and global skeletons is lost.

**Goal**: The authors aim to find a control method that requires no new training and does not force DLMs back into autoregressive generation. Specifically, it should allow the model to know the global structure of the response before generation, ensuring each missing segment is constrained by both preceding and succeeding anchors, while automatically expanding reasoning space when encountering complex problems.

**Key Insight**: The paper observes that while AR prompts can only be placed at the beginning, DLM conditions can be scattered throughout the sequence. Thus, rather than giving the model a "Please think step by step" prefix, it is better to embed structural anchors like "Steps," "Check," and "Answer" directly into the target sequence, allowing the model to perform parallel infilling between these anchors.

**Core Idea**: Use global template anchors to replace single prefix prompts, converting the free generation of DLMs into "multi-segment infilling constrained by structural boundary conditions."

## Method
The core of this paper is not training a new DLM but changing how the DLM receives conditions during inference. Traditional prefix prompting places instructions to the left of the output and hopes the model follows them; TI decomposes instructions into multiple fixed anchors placed at different positions in the response, leaving masked spans for the model to fill. Consequently, each segment to be generated sees not only the problem and previous text but also the subsequent structure, such as "Analyze first," "Calculate next," and "Finally give the answer."

The authors emphasize that these templates are not local prefixes/suffixes as in common Fill-In-the-Middle (FIM) tasks. FIM usually involves filling a single hole, whereas TI distributes multiple structural anchors across the entire response space to establish a skeleton for the full reasoning path. For math problems, templates can force the model to expand derivations before outputting a numerical value; for code tasks, templates can separate logic handling, implementation, and return values; for safety scenarios, templates can insert reflection stages like "Draft, Critique, Refine."

### Overall Architecture
The input for TI consists of the original problem $c$, a set of structural anchors $A_1, A_2, ..., A_n$, and masked segments $M_1, M_2, ..., M_n$ between them. The final sequence can be written as $S=[c,A_1,M_1,A_2,M_2,...,A_n,M_n]$. Here, $A_i$ represents immutable template text, and $M_i$ is the content completed by the DLM during diffusion sampling.

During generation, the model no longer predicts the next token based solely on $x_{<t}$. Instead, each segment $M_i$ is conditioned on the problem and all template anchors, approximating $p(M_i|c,A_1,...,A_n)$. This allows "future anchors" to participate in constraining the current segment. For instance, while writing intermediate reasoning, the model already knows it must eventually enter the final answer area, making it less likely to lose its way in local expansions.

The complete process involves three steps: First, construct a static structural template based on the task (avoiding complex prompt engineering for universality). Second, place the template and masks into the DLM's parallel sampling process, where the model denoises all empty segments simultaneously. Third, if the confidence of a specific segment remains consistently low, use DSA to extend its mask length, granting the model more expressive space.

### Key Designs
1.  **Global Template Anchors**:
    *   **Function**: Transforms soft prompts (originally only at the start) into structural boundary conditions that permeate the output space.
    *   **Mechanism**: The template consists of multiple fixed anchors, each serving a different reasoning role (e.g., Plan, Step, Check, Answer). While filling any segment, the DLM can simultaneously attend to all anchors, changing the generation goal from "continuing a prefix" to "completing a response within a set skeleton."
    *   **Design Motivation**: DLMs excel at bidirectional and arbitrary-position conditioning, which prefix prompting fails to exploit. Global anchors turn high-freedom sampling into a bounded search, reducing logic drift while preserving parallel generation capabilities.

2.  **Dynamic Span Allocation (DSA)**:
    *   **Function**: Resolves the issue where static template mask lengths are too short, causing reasoning to be truncated.
    *   **Mechanism**: Monitors the prediction confidence of tokens within a segment during each diffusion step. If the probability of the least certain token in a segment falls below a threshold $\tau$, the segment length is expanded from $|M_i|$ to $|M_i|+\delta$. The experiments allow a maximum expansion of 8 tokens per step and up to 10 total expansions.
    *   **Design Motivation**: Different problems require different reasoning spaces. Fixed-length templates might compress complex problems into too short a segment; DSA allows the model to request more "scratchpad space" while keeping the structure intact.

3.  **Reshaping Sampling Trajectories via Structural Priors**:
    *   **Function**: Guides unordered parallel denoising trajectories toward a generation order closer to global planning.
    *   **Mechanism**: The paper observes that DLMs adapted from AR models (like Dream-Base) exhibit chaotic filling orders in unconditional generation. With templates, the model tends to stabilize information near the structural anchors first, then fills the remaining gaps synchronously. The "Draft-Critique-Refine" template in safety experiments demonstrates the same mechanism.
    *   **Design Motivation**: TI is not just about increasing prompt text; it injects a structural prior into the sampling space. It forces the model to satisfy high-level constraints before filling local details, ensuring stability under fast sampling and long outputs.

### Loss & Training
The proposed method is a training-free inference framework and does not introduce new training losses. Experiments utilize base and instruct versions of LLaDA-8B and Dream-7B, focusing on pure parallel generation quality: the model must plan and generate simultaneously within a 128-token budget. TI uses one static template per task, with DSA handling dynamic span expansion based on confidence. Evaluation compares against Vanilla unconditional generation and Prefix Prompting to distinguish gains from structural anchors versus general prompting.

## Key Experimental Results

### Main Results
The main experiments cover mathematical reasoning, code generation, and multi-constraint planning. Notably, Prefix Prompting degrades performance in many settings, while TI shows significant improvements on LLaDA-Instruct and Dream-Base, indicating that DLM control cannot simply replicate AR models.

| Model | Method | GSM8K | MATH500 | HumanEval | Trip CSR | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LLaDA-8B Instruct | Vanilla | 49.58 | 17.00 | 15.85 | 12.13 | 23.64 |
| LLaDA-8B Instruct | TI | 71.49 | 21.80 | 32.93 | 12.06 | 34.57 |
| Dream-7B Base | Vanilla | 8.87 | 3.60 | 18.29 | 1.13 | 7.97 |
| Dream-7B Base | TI | 44.58 | 14.40 | 29.88 | 15.94 | 26.20 |
| Dream-7B Instruct | Vanilla | 35.86 | 11.40 | 20.12 | 0.63 | 17.00 |
| Dream-7B Instruct | TI | 39.80 | 12.80 | 33.54 | 16.31 | 25.61 |

In terms of average performance, TI improves over the baseline by 9.40 percentage points. The gain on HumanEval specifically highlights that it is not only suitable for math CoT but also aids structured implementation in code generation. The massive improvement in Dream models for Trip Planning suggests that multi-constraint planning benefits greatly from global anchors.

### Ablation Study
Stepwise ablations were performed on GSM8K using Dream-Base. Boundary anchors alone significantly outperformed prefix prompting, while adding detailed descriptions and DSA provided further gains.

| Configuration | Strategy | GSM8K Acc. | Gain vs. Vanilla |
| :--- | :--- | :--- | :--- |
| Vanilla | Unstructured | 8.87 | 0.00 |
| Prefix Prompting | AR-style Prefix | 8.79 | -0.08 |
| TI Minimal | Static Boundary Anchors | 24.94 | +16.07 |
| TI Detailed | Static Detailed Template | 36.00 | +27.13 |
| TI + DSA | Dynamic Span Allocation | 44.58 | +35.71 |

The authors also tested anchor position perturbations. The accuracy for the default (Base) position was 0.4458, Early was 0.4033, Late was 0.4359, and Compressed was 0.4367. While the default position performed best, performance did not collapse, suggesting gains come from global conditioning itself rather than overfitting a hand-crafted position.

### Key Findings
-   **Prefix Prompting is unreliable for DLMs**: It dropped HumanEval from 18.29 to 3.66 on Dream-Base and regressed across multiple settings, indicating that prefix-style control conflicts with the parallel sampling mechanism of DLMs.
-   **DSA is a primary source of gain**: Minimal TI proved that structural skeletons are effective, and DSA provides the variable reasoning length required for complex problems, further boosting GSM8K from 36.00 to 44.58.
-   **TI is more stable for fast sampling and long generation**: Analysis of length vs. sampling steps showed that as generation length increases at fixed steps, TI significantly mitigates the quality drop seen in the baseline. Conversely, TI maintains higher accuracy when sampling steps are reduced at fixed lengths.
-   **Instruct tuning may pull DLMs back toward AR bias**: Dream-Instruct shows a sampling trajectory closer to a diagonal (sequential) generation. The authors suggest this may result from strong supervision of prefix tokens during instruction tuning, which suppresses the global planning potential of DLMs.

## Highlights & Insights
-   The most valuable insight is that the unit of control for a DLM should be the structure of the output space, not just prompt text. Placing constraints at "future" positions leverages the DLM's ability to see both forward and backward context simultaneously.
-   TI turns "let the model think slowly" from a soft instruction into a physical structure. The model is forced to fill content between anchors, which is more stable than hoping it voluntarily follows a CoT.
-   DSA is a practical detail. It recognizes that templates need to provide structure without hard-coding reasoning length, using low confidence as a signal for expansion—simple yet fitting for the diffusion process.
-   Insights for code generation: Code naturally contains function signatures, control flows, return values, and test constraints. If these structures are used as distributed anchors, DLM infilling might be better suited for complex function bodies than traditional left-to-right generation.
-   The "Draft-Critique-Refine" example for safety shows that templates can serve as process constraints. Instead of pasting safety prompts at the beginning, it reserves reflection segments in the output, forcing the model to pass through a check during its generation trajectory.

## Limitations & Future Work
-   **Template Design**: Templates still require manual design. While the paper uses static templates to prove universality, the optimal number, position, and content of anchors for different tasks still require searching or automated generation.
-   **Model Optimization**: Current models were not trained for TI. Existing instruct DLMs are trained under traditional prompt-inference paradigms and may not fully exploit distributed templates. Future work could integrate TI into instruction tuning or preference optimization.
-   **Benchmark Coverage**: Experiments focused on generative reasoning. Benchmarks like MMLU are discriminative and unsuitable for this goal, meaning it is unclear if TI benefits general knowledge QA, long-form writing, or interactive tool use.
-   **Heuristic DSA**: DSA thresholds and expansion strategies remain heuristic. Low confidence doesn't always imply insufficient space; it could indicate a lack of knowledge or an ill-suited template. More granular expansion and reclamation mechanisms would be more robust.
-   **Format Dependency**: Structural anchors might lead to format dependency. If a model learns to cater to the template rather than solve the problem, complex templates might induce reasoning that is formally correct but substantively empty.

## Related Work & Insights
-   **vs. Block Diffusion**: Block-wise methods reduce freedom via segmenting to improve stability and engineering optimization. TI does not force the sequence back into local order but uses global anchors to constrain parallel infilling, emphasizing the DLM's arbitrary-position conditioning.
-   **vs. Prefix Prompting / CoT**: CoT and Plan-and-Solve rely on initial soft guidance that the model can ignore or deviate from internally. TI places step structures inside the target sequence, allowing subsequent constraints to participate directly in generating current segments.
-   **vs. Constrained Decoding**: Traditional constrained decoding uses external rules to mask illegal tokens, acting like hard search pruning. TI provides structure at the input condition level, allowing the model to naturally align with the template during generation.
-   **vs. FIM / Code Infilling**: FIM typically fills a single local hole to connect context. TI uses multiple anchors as a global skeleton, focusing on the planning consistency of a whole segment of reasoning or code.
-   **Inspiration**: Future attempts could involve automatic template generators where a model first generates structural anchors for a problem, followed by TI filling. Unit tests, type signatures, and exception handling paths could also be used as anchors for code generation.

## Rating
-   Novelty: ⭐⭐⭐⭐☆ Converting DLM's arbitrary conditioning into global template control is a clear idea that fits the model mechanism.
-   Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers two types of DLMs, four types of tasks, and multiple analyses, though large-scale real-world code tasks and automated template search are missing.
-   Writing Quality: ⭐⭐⭐⭐☆ The narrative is smooth, motivation and mechanism are well-explained, though some formulas and tables are slightly crowded.
-   Value: ⭐⭐⭐⭐⭐ This is a training-free, low-cost, and transferable control paradigm for DLMs, providing high reference value for future non-autoregressive language models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](../../ICLR2026/llm_nlp/dreamon_diffusion_language_models_for_code_infilling_beyond_fixed-size_canvas.md)
- [\[CVPR 2026\] Perception Programs: Unlocking Visual Tool Reasoning in Language Models](../../CVPR2026/llm_nlp/perception_programs_visual_tool_reasoning.md)
- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](../../ICLR2026/llm_nlp/toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)
- [\[ICML 2026\] Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models](../../ICML2026/llm_nlp/reasoning_on_the_manifold_bidirectional_consistency_for_self-verification_in_dif.md)
- [\[ACL 2026\] Leveraging Pretrained Language Models as Energy Functions for Glauber Dynamics Text Diffusion](leveraging_pretrained_language_models_as_energy_functions_for_glauber_dynamics_t.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](../../ICLR2026/llm_nlp/dreamon_diffusion_language_models_for_code_infilling_beyond_fixed-size_canvas.md)
- [\[CVPR 2026\] Perception Programs: Unlocking Visual Tool Reasoning in Language Models](../../CVPR2026/llm_nlp/perception_programs_visual_tool_reasoning.md)
- [\[ICML 2026\] Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models](../../ICML2026/llm_nlp/reasoning_on_the_manifold_bidirectional_consistency_for_self-verification_in_dif.md)
- [\[ICML 2026\] dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](../../ICML2026/llm_nlp/dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching.md)
- [\[ACL 2026\] FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation](fastdiss_few-step_match_many-step_diffusion_language_model_on_sequence-to-sequen.md)

</div>

<!-- RELATED:END -->
