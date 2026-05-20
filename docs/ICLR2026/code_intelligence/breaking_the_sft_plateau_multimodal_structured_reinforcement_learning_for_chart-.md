---
title: >-
  [Paper Note] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation
description: >-
  [ICLR 2026][Code Intelligence][Chart-to-Code] To address the SFT performance plateau in chart-to-code generation, this paper proposes Multimodal Structured Reinforcement Learning (MSRL)…
tags:
  - "ICLR 2026"
  - "Code Intelligence"
  - "Chart-to-Code"
  - "Reinforcement Learning"
  - "SFT Plateau"
  - "Multi-granularity Reward"
  - "GRPO"
date: 2026-05-08
content_hash: 33a8bdb47436a2a6
---

# Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation

**Conference**: ICLR 2026
**arXiv**: [2508.13587](https://arxiv.org/abs/2508.13587)  
**Code**: [GitHub](https://github.com/DocTron-hub/MSRL)  
**Area**: Multimodal VLM / Code Generation
**Keywords**: Chart-to-Code, Reinforcement Learning, SFT Plateau, Multi-granularity Reward, GRPO

## TL;DR
To address the SFT performance plateau in chart-to-code generation, this paper proposes Multimodal Structured Reinforcement Learning (MSRL), which employs a dual-layer textual and visual reward function along with a two-stage RL strategy, achieving improvements of 6.2% and 9.9% on high-level metrics on ChartMimic and ReachQA respectively, establishing open-source SOTA and matching GPT-4o.

## Background & Motivation
**Background**: Multimodal large language models (MLLMs) have demonstrated strong performance on tasks such as visual question answering, yet remain limited in processing information-dense images (e.g., charts) and generating structured outputs (e.g., code). The chart-to-code generation task requires models to deeply understand visualizations and produce accurate plotting code, carrying significant practical value.

**Limitations of Prior Work**: Existing approaches rely on SFT or DPO trained on synthetic data, which suffers from limited pattern diversity (synthetic data lacks real-world complexity) and poor generalization. Moreover, SFT has an inherent limitation—it assigns equal importance to every token in the target sequence, whereas plotting code is dominated by boilerplate (e.g., `plt.plot`) while critical information (data values, style parameters) appears infrequently.

**Key Challenge**: Scaling SFT data beyond a certain threshold yields diminishing returns, resulting in a **performance plateau effect**. Experiments demonstrate that scaling from 200k to 2.8M samples brings no further improvement beyond 2M, indicating that stacking more data cannot break this ceiling.

**Goal**: (1) Systematically verify the SFT performance ceiling; (2) Design effective RL strategies to surpass it; (3) Construct a large-scale, real-world chart-to-code training corpus.

**Key Insight**: The authors observe that the uniform token-weighting mechanism of SFT is unable to prioritize critical tokens, whereas RL can target key content accuracy through customized reward functions. Additionally, purely text-based rewards ignore overall visual structure, motivating the introduction of visual feedback to form multi-granularity rewards.

**Core Idea**: Drive a two-stage GRPO reinforcement learning pipeline with multi-granularity (textual + visual) reward functions to break the SFT performance plateau in chart-to-code generation.

## Method

### Overall Architecture
The model takes a chart image as input and produces executable Matplotlib plotting code as output. The training pipeline consists of three stages: (1) large-scale SFT pre-training to establish foundational capabilities; (2) Stage-1 RL using textual rewards to optimize code-level details; (3) Stage-2 RL incorporating visual rewards to further improve visual fidelity. The base model is Qwen2.5-VL-7B, and GRPO is adopted as the RL algorithm.

### Key Designs

1. **Large-Scale Real-World Data Construction**:

    - Function: Constructs the largest chart-to-code training corpus to date (3 million pairs), sourced from real figures in arXiv papers.
    - Mechanism: Real figures are crawled from arXiv papers published before 2023; Gemini-2.5-Flash is used to generate plotting code conditioned on the figure and example code; the final dataset is obtained after execution-based verification and filtering, covering 24 chart types and 1,555 Matplotlib APIs.
    - Design Motivation: Addresses two shortcomings of existing datasets—(1) purely synthetic data leads to monotonic trends and low diversity; (2) insufficient scale prevents full exposure of the SFT plateau. Using real figures ensures the data distribution is closer to real-world scenarios.

2. **RL Data Curation**:

    - Function: Selects 33k high-quality samples from the 3M SFT corpus for RL training.
    - Mechanism: Two-stage filtering—Stage 1 applies code-content-based filtering (chart type, data definition format), using tree-structure parsing to identify complex chart types while retaining only one-dimensional arrays and non-nested dictionary formats, reducing to 45k; Stage 2 employs GPT-4.1-mini for visual quality assessment, with human validation showing >90% agreement, yielding a final 33k samples.
    - Design Motivation: RL requires high-quality data that is disjoint from SFT data to avoid overfitting to the SFT format and to enhance exploration.

3. **Textual Reward**:

    - Function: Designs rule-based fine-grained evaluation of code correctness.
    - Mechanism: Generated code first undergoes **format normalization** (eliminating syntactic variants), followed by evaluation along five dimensions—data values (soft matching with ±5% tolerance), chart type (exact match), layout (exact match), text elements such as titles and labels (edit distance). An execution reward (binary: whether the code runs successfully) is computed separately.
    - Design Motivation: The stylistic diversity of plotting code makes direct extraction of key information difficult; format normalization renders the reward function robust to syntactic variants, which is a critical adaptation of RLVR to structured code generation.

4. **Visual Reward**:

    - Function: Renders the generated code as an image and uses an MLLM to evaluate the visual similarity between the rendered image and the original chart.
    - Mechanism: The generated code is executed to produce a rendered image, which is then scored by Qwen2.5-VL-72B across six dimensions (chart type, layout, text content, data, style, clarity) and normalized into a visual reward score. Code that fails to render receives a score of 0.
    - Design Motivation: Textual rewards focus solely on fine-grained code details and overlook overall visual structure and style; the visual reward compensates for this blind spot.

5. **Two-Stage RL Strategy**:

    - Function: Trains first with textual rewards and then fine-tunes with visual rewards.
    - Mechanism: The total reward is $R = w_t R_\text{text} + w_v R_\text{vis} + w_e R_\text{exec}$. Stage 1 trains on 22k samples using only textual rewards ($w_v=0$); Stage 2 fine-tunes on 11k samples using mixed rewards. The GRPO algorithm exploits within-group relative advantages for policy optimization.
    - Design Motivation: Visual rewards require image rendering and model-based evaluation, incurring high computational cost. The two-stage strategy achieves most of the performance gain (~5%) at low cost in Stage 1, with Stage 2 yielding an additional ~1.5% using fewer samples, balancing performance and efficiency.

### Loss & Training
- SFT stage: Standard autoregressive negative log-likelihood loss.
- RL stage: GRPO algorithm; relative advantages are computed by sampling multiple responses within a group, without requiring an additional critic model.
- Two-stage curriculum: textual reward training first (240 GPU hours) → mixed reward fine-tuning (approximately 336 GPU hours).

## Key Experimental Results

### Main Results

| Model | Params | ChartMimic Exec. | ChartMimic High | ReachQA Exec. | ReachQA High |
|-------|--------|-------------------|-----------------|---------------|--------------|
| GPT-4o | - | 93.2 | 83.5 | 92.8 | 84.0 |
| Qwen2.5-VL-7B | 7B | 73.2 | 41.6 | 62.2 | 37.6 |
| ChartCoder | 7B | 91.4 | 74.0 | 83.8 | 69.4 |
| MSRL-SFT | 7B | 93.2 | 77.6 | 92.2 | 80.0 |
| **MSRL** | **7B** | **96.5** | **83.8** | **98.2** | **89.9** |

With only 7B parameters, MSRL surpasses all open-source models; its ChartMimic high-level score of 83.8 exceeds GPT-4o's 83.5, and its ReachQA high-level score of 89.9 substantially surpasses GPT-4o's 84.0.

### Ablation Study

| Configuration | ChartMimic Exec. | Low-Level | High-Level | Notes |
|---------------|-------------------|-----------|------------|-------|
| Baseline (no SFT/RL) | 73.2 | 44.6 | 41.6 | Qwen2.5-VL-7B base model |
| SFT only | 93.2 | 73.0 | 77.6 | SFT ceiling |
| RL only (no SFT) | 93.8 | 65.6 | 62.3 | Direct RL underperforms SFT |
| SFT + RL (textual reward) | 97.0 | 78.1 | 82.7 | Breaks SFT plateau |
| SFT + RL (two-stage) | 96.5 | 78.6 | 83.8 | Final model with visual reward |

Reward strategy comparison: pure visual RL yields the best performance but requires 1,344 GPU hours; the two-stage strategy achieves comparable performance at 576 GPU hours.

### Key Findings
- **SFT Plateau Confirmed**: Scaling data from 200k to 2.8M shows no performance improvement beyond 2M, providing evidence of an inherent SFT ceiling.
- **Significant RL Gains**: Applied to a saturated SFT model, RL still delivers +6.2% on ChartMimic high-level and +9.9% on ReachQA high-level metrics.
- **Two-Stage Strategy Is Efficient**: Stage 1 textual rewards contribute ~5% improvement; Stage 2 visual rewards contribute an additional ~1.5% at substantially reduced compute.
- **Cross-Library Generalization**: MSRL demonstrates generalization to Seaborn and Plotly test sets (trained exclusively on Matplotlib), with execution rate on Plotly improving from 62.7% to 90.0%.

## Highlights & Insights
- **Systematic SFT Plateau Analysis**: Through controlled experiments across six data scales (200k–2.8M), this work provides the first quantitative evidence of the SFT performance ceiling in chart-to-code generation. This methodology transfers to other structured generation tasks.
- **Code Format Normalization as a Key RLVR Adaptation**: The syntactic diversity of plotting code (i.e., the same semantics expressible in multiple forms) would overwhelm a reward function with noise if not normalized. This technique is broadly applicable to any RLVR task involving code generation.
- **Visual Reward as a Global Consistency Check for Structured Output**: Rendering code into images before evaluation elegantly maps "is the code correct?" to the more intuitive "are the images similar?", and is applicable to any scenario where code output can be visualized (e.g., LaTeX generation, web code generation).
- **Curriculum Design: Text-First, Vision-Second**: Establishing foundational capabilities with low-cost rewards before introducing high-cost rewards for fine-grained refinement is a practical resource–performance trade-off strategy.

## Limitations & Future Work
- Training is conducted solely on Matplotlib-style data; although generalization to Seaborn/Plotly is demonstrated, gains remain limited (Plotly high-level score is only 35.9), and joint multi-library training warrants further exploration.
- The visual reward relies on Qwen2.5-VL-72B for scoring, which is costly and may introduce biases inherent to the evaluator MLLM; whether lightweight metrics (e.g., SSIM, LPIPS) could serve as substitutes is an open question.
- The data construction pipeline uses papers published before 2023, which may not cover emerging visualization styles.
- RL training uses only 33k samples; whether larger-scale RL data could yield further improvements remains unclear, as the paper shows RL also plateaus (converging around 22k samples).

## Related Work & Insights
- **vs. ChartCoder**: ChartCoder also addresses chart-to-code but relies solely on SFT+DPO on synthetic data; MSRL substantially outperforms it via real-world data and RLVR (high-level: 74.0 → 83.8).
- **vs. Chart-R1/BigCharts-R1**: These works apply RLVR to chart reasoning QA but do not involve code generation. MSRL is the first to apply RLVR to chart-to-code as a structured code generation task.
- **vs. DeepSeek-R1/Vision-R1**: The R1 series focuses on RL training for general reasoning; MSRL proposes task-specific multi-granularity reward designs for structured output generation, demonstrating how RLVR can be adapted across different task paradigms.

## Rating
- Novelty: ⭐⭐⭐⭐ The systematic verification of the SFT plateau and the multi-granularity reward design are noteworthy, though the overall framework follows the standard GRPO + customized reward paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Data scaling experiments, multi-dimensional ablations, cross-library generalization, and comparisons with GPT-4o constitute a highly comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐ The paper is logically clear, figures are informative, and the motivation is well-articulated.
- Value: ⭐⭐⭐⭐ A strong SOTA is established in the chart-to-code niche, and the methodology offers useful reference for other structured code generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](../../CVPR2026/code_intelligence/mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ICLR 2026\] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](supervised_reinforcement_learning_from_expert_trajectories_to_step-wise_reasonin.md)
- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](../../AAAI2026/code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[ICLR 2026\] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)

</div>

<!-- RELATED:END -->
