---
title: >-
  [Paper Note] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling
description: >-
  [LLM Agent] Proposes METAL, a VLM-based multi-agent framework that decomposes the chart-to-code generation task into the iterative collaboration of four specialized agents (generation, visual critique, code critique, and revision), achieving a 5.2% F1 improvement over the prior SOTA on the ChartMIMIC benchmark and demonstrating test-time scaling behavior.
tags:
  - "LLM Agent"
date: 2026-05-08
content_hash: a749705964637e5a
---

# METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling

## Basic Information

- **Conference**: ACL2025
- **arXiv**: [2502.17651](https://arxiv.org/abs/2502.17651)
- **Code**: [https://metal-chart-generation.github.io](https://metal-chart-generation.github.io)
- **Area**: LLM Agent
- **Keywords**: Multi-agent framework, chart generation, vision-language model, test-time scaling, iterative revision

## TL;DR

Proposes METAL, a VLM-based multi-agent framework that decomposes the chart-to-code generation task into the iterative collaboration of four specialized agents (generation, visual critique, code critique, and revision), achieving a 5.2% F1 improvement over the prior SOTA on the ChartMIMIC benchmark and demonstrating test-time scaling behavior.

## Background & Motivation

### Problem Definition
Chart-to-Code Generation task: Given a reference chart image $x_{ref}$, the model needs to generate executable code $y$ (such as Python) so that the rendered chart $O(y)$ can faithfully reproduce the reference chart.

### Limitations of Prior Work

**Insufficiency of single models**: Even SOTA VLMs like GPT-4o struggle to accurately interpret and reproduce complex visual elements (color, layout, text, type) in charts when directly generating chart code.

**Limited effectiveness of existing methods**: Best-of-N and Hint-enhanced Prompting show only marginal improvements compared to direct prompting.

**Key Challenge**: Chart generation requires both strong visual design capability and precise coding ability. This complex multimodal reasoning process exceeds the capacity of a single model or a single agent.

### Design Motivation
Decompose the complex chart generation task into multiple subtasks handled cooperatively by specialized agents, progressively enhancing generation quality through iterative feedback and revision.

## Method

### Overall Architecture

METAL consists of four specialized agents and a multi-criteria verifier, which collaborate iteratively during inference:

1. **Generation Agent (G)**: Generates initial code $y_0 = G(x_{ref})$ from the reference chart.
2. **Visual Critique Agent (V)**: Compares the generated chart with the reference chart to identify visual differences $v_t = V(O(y_t), x_{ref})$.
3. **Code Critique Agent (C)**: Reviews the code and provides suggestions for improvement $c_t = C(y_t)$.
4. **Revision Agent (R)**: Integrates feedback from both critiques to modify the code $y_{t+1} = R(y_t, v_t, c_t)$.

### Key Designs

**Modality-Tailored Critiques**:
- Visual critique and code critique are separated into two independent agents rather than being merged into a single review process.
- Visual data requires spatial understanding, color analysis, and detail recognition, while code data requires syntax and logical consistency checks.
- Merging critiques leads to long contexts that cause information loss, failing to meet the specific requirements of different modalities.

**Multi-Criteria Verifier**:
- Designs three heuristic verification metrics to evaluate chart quality:
    - **Color (m₁)**: HSV color space conversion $\rightarrow$ pixel color histogram $\rightarrow$ cosine similarity.
    - **Text (m₂)**: Text extraction via EasyOCR $\rightarrow$ Jaccard coefficient.
    - **Overall Structure (m₃)**: Grayscale SSIM (Structural Similarity Index Measure).
- Triggers early stopping when all metrics exceed a dynamic threshold $\theta^t$.

**Agent Implementation**:
- Generation Agent and Visual Critique Agent: VLM architectures, processing multimodal inputs.
- Code Critique Agent and Revision Agent: Pure text architectures, yielding outputs of approximately 600 tokens.

### Inference Process

```python
y₀ ← G(x_ref)                    # Initial generation
while t < T_max:
    O(y_t) ← Render chart
    v_t ← V(O(y_t), x_ref)       # Visual critique
    c_t ← C(y_t)                  # Code critique
    if All verification metrics pass: break     # Early stopping
    y_{t+1} ← R(y_t, v_t, c_t)   # Revision
    t ← t + 1
return y_t
```

## Key Experimental Results

### Experimental Setup
- **Dataset**: ChartMIMIC — 1,000 human-curated (chart, instruction, code) triplets, covering 18 regular types and 4 advanced types.
- **Evaluation Metrics**: F1 scores across four dimensions: Text, Type, Color, and Layout.
- **Baseline Models**: GPT-4o and LLaMA 3.2-11b.

### Main Results

| Base Model | Method | Text | Type | Color | Layout | Average |
|---|---|---|---|---|---|---|
| LLaMA 3.2-11b | Direct Prompting | 36.70 | 37.07 | 33.46 | 54.56 | 40.45 |
| | Hint-Enhanced | 38.82 | 38.47 | 36.82 | 51.22 | 41.33 |
| | Best-of-N (n=5) | 40.28 | 36.60 | 38.43 | 57.22 | 43.13 |
| | **METAL (n=5)** | **46.69** | **54.42** | **47.32** | **58.69** | **51.78** |
| GPT-4o | Direct Prompting | 74.83 | 81.24 | 74.24 | 94.76 | 81.26 |
| | Hint-Enhanced | 77.02 | 80.84 | 72.75 | 93.89 | 81.12 |
| | Best-of-N (n=5) | 75.47 | 82.16 | 75.30 | 96.37 | 82.32 |
| | **METAL (n=5)** | **86.31** | **84.17** | **79.86** | **95.50** | **86.46** |

- METAL + GPT-4o outperforms Direct Prompting by **5.2%**.
- METAL + LLaMA 3.2-11b outperforms Direct Prompting by **11.33%**.

### Ablation Study

| Variant | Text | Type | Color | Layout | Average |
|---|---|---|---|---|---|
| METAL_V (Visual Critique Only) | 83.43 | 82.57 | 77.57 | 93.69 | 84.31 |
| METAL_C (Code Critique Only) | 82.35 | 80.90 | 76.69 | 91.93 | 82.96 |
| METAL_S (Merged Critique) | 80.26 | 78.88 | 74.50 | 89.82 | 80.86 |
| **METAL (Full)** | **86.31** | **84.17** | **79.86** | **95.50** | **86.46** |

Separated critique (METAL) outperforms merged critique (METAL_S) by **5.6%**, validating the effectiveness of modality-tailored critiques.

### Test-Time Scaling Findings

- As the computational budget increases from $2^9$ to $2^{13}$ tokens, performance scales near-linearly with the logarithm of the computational budget.
- This indicates that the METAL framework exhibits test-time scaling characteristics, where more inference iterations yield consistent performance gains.

### Multi-Agent vs. Modular System

- After removing autonomous decision-making and code execution capabilities (converting into a modular system), the average performance gain decreases by 4.51%.
- The inability to execute code and render charts leads to a decline in critique quality, and the lack of autonomous decision-making causes error propagation.

### Key Findings
1. Modality-tailored critique significantly outperforms merged critique; separated evaluations are better at capturing specific issues of each modality.
2. METAL is highly robust with strong base models, demonstrating consistent improvements across all difficulty levels on GPT-4o.
3. Weaker base models (LLaMA 3.2-11b) show diminishing returns on high-difficulty charts, but the absolute gains remain substantial.

## Highlights & Insights

1. **Successful Application of Task Decomposition**: Decomposes complex multimodal generation into an iterative loop of generation, critique, and revision.
2. **New Discovery in Test-Time Scaling**: Observes test-time scaling behavior in a multi-agent system, showing a near-linear relationship between performance and the logarithm of the computational budget.
3. **Modality-Tailored Evaluation**: The evaluation needs of vision and code are fundamentally different; separating them significantly outperforms a merged approach.
4. **Modular Design Flexibility**: Allows combining different base models for different agents (e.g., using critique-optimized models for reviewers and generation-optimized models for generators).
5. **Clear and Compelling Case Study**: Illustrates the iterative correction process from initial generation to a perfect chart (Round 0 $\rightarrow$ Round 1 fixing axes $\rightarrow$ Round 2 fixing colors).

## Limitations & Future Work

1. **High Computational Cost**: Compared to direct prompting, METAL requires multiple iteration rounds, significantly increasing execution costs.
2. **Dependency on Prompt Engineering**: Performance is influenced by prompt design; further optimized prompts may exist.
3. **Limitations of Automated Evaluation**: The F1 metric may not perfectly capture all subtle details of the charts.
4. **Limited Range of Test-Time Scaling**: Due to resource constraints, scaling was only evaluated up to $2^{13}$ tokens; performance with larger budgets remains unexplored.

## Related Work & Insights

- **Chart Generation**: ChartMIMIC (Shi et al., 2024), Plot2Code (Wu et al., 2024), ChartLlama (Han et al., 2023)
- **Multi-Agent Frameworks**: Agents' Room (Huot et al., 2024), TradingAgents (Xiao et al., 2024)
- **Test-Time Scaling**: Scaling LLM Test-Time Compute (Snell et al., 2024), s1 (Muennighoff et al., 2025)

## Rating ⭐⭐⭐⭐

- Novelty: ⭐⭐⭐⭐ — The combination of a multi-agent framework and test-time scaling is highly novel.
- Value: ⭐⭐⭐⭐⭐ — Chart generation has broad practical application scenarios (report generation, research presentations).
- Methodology Novelty: ⭐⭐⭐⭐ — Clever design of modality-tailored critiques and iterative revision mechanisms.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Complete ablations, convincing test-time scaling analysis, and clear case studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AndroidGen: Building an Android Language Agent under Data Scarcity](androidgen_agent_data_scarcity.md)
- [\[ICLR 2026\] GTA1: GUI Test-time Scaling Agent](../../ICLR2026/llm_agent/gta1_gui_test-time_scaling_agent.md)
- [\[NeurIPS 2025\] AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](../../NeurIPS2025/llm_agent/agenttts_large_language_model_agent_for_testtime_computeopti.md)
- [\[ACL 2025\] Select, Read, and Write: A Multi-Agent Framework of Full-Text-based Related Work Generation](select_read_and_write_a_multi-agent_framework_of_full-text-based_related_work_ge.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)

</div>

<!-- RELATED:END -->
