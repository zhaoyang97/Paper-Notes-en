---
title: >-
  [Paper Note] ChartEditor: A Reinforcement Learning Framework for Robust Chart Editing
description: >-
  [AAAI2026][Reinforcement Learning][chart editing] This paper introduces the ChartEditVista benchmark (7,964 samples, 31 chart types) and the ChartEditor model. By combining a GRPO reinforcement learning framework with a novel rendering reward, ChartEditor surpasses GPT-4o and several 72B-scale models on chart editing tasks using only 3B parameters.
tags:
  - AAAI2026
  - Reinforcement Learning
  - chart editing
  - reinforcement-learning
  - GRPO
  - rendering reward
  - benchmark
date: 2026-05-08
content_hash: 943fe2b4e0feabaa
---

# ChartEditor: A Reinforcement Learning Framework for Robust Chart Editing

**Conference**: AAAI2026
**arXiv**: [2511.15266](https://arxiv.org/abs/2511.15266)
**Code**: To be confirmed
**Area**: Reinforcement Learning
**Keywords**: chart editing, reinforcement-learning, GRPO, rendering reward, benchmark

## TL;DR

This paper introduces the ChartEditVista benchmark (7,964 samples, 31 chart types) and the ChartEditor model. By combining a GRPO reinforcement learning framework with a novel rendering reward, ChartEditor surpasses GPT-4o and several 72B-scale models on chart editing tasks using only 3B parameters.

## Background & Motivation

Chart editing aims to leverage multimodal large language models (MLLMs) to modify existing charts according to natural language instructions, thereby reducing the manual cost of visualization design. However, existing benchmarks suffer from three major deficiencies:

1. **Unrealistic assumptions**: Most benchmarks (ChartReformer, ChartEdit, ChartMimic) assume access to the original chart code, which is rarely available in practice.
2. **Limited diversity**: Instruction types are narrow, editable elements are incompletely covered, chart types are limited, and large-scale training data are lacking.
3. **Unreliable evaluation metrics**: MLLM-based metrics are prone to hallucination, while rule-based metrics operate only at a coarse sub-chart level and cannot capture fine-grained editing changes.

Furthermore, existing chart-specialized models (TinyChart, ChartLlama, ChartMoE, etc.) demonstrate limited capability on chart editing tasks.

## Core Problem

1. How to construct a truly comprehensive chart editing benchmark covering diverse chart types, editing instructions, and editable elements?
2. How to design reliable fine-grained evaluation metrics for assessing chart editing quality?
3. How to achieve high-quality chart editing with a small-parameter model?

## Method

### 1. ChartEditVista Benchmark Construction

A multi-stage automated pipeline is used to generate data:

- **Raw chart generation**: GPT-4.1 generates synthetic CSV data and Python plotting code based on sampled topics and 31 chart types, with random sampling of attributes (color schemes, axis configurations, etc.) to enhance layout diversity.
- **Edit target selection**: Plotting code is converted into a structured Chart JSON, then into a hierarchical Chart Tree (root node = entire chart, internal nodes = major components, leaf nodes = atomic elements). Edit targets are systematically selected via tree traversal.
- **Instruction and code generation**: GPT-4.1 generates editing instructions and corresponding modification code, subject to multi-round quality control (code executability checks, instruction refinement, overall quality assessment).
- **CoT generation**: A visual-code Chain-of-Thought is generated for each sample, comprising ① edit target identification, ② code attribute modification description, and ③ expected visual change description.

The final benchmark contains 601 base charts and 7,964 triplets (chart, instruction, modified code) covering 6 editing task types. Manual verification by 10 annotators yields instruction clarity >97% and editing success rate >94%.

### 2. Rendering-Aware Rule-based Metrics (RARM)

Two fine-grained evaluation metrics are proposed:

- **Layout Metric**: Evaluates the color, position, and shape similarity of graphical objects:
  $$S_L(p,g) = S_{\text{color}}(p,g) \times S_{\text{pos}}(p,g) \times S_{\text{shape}}(p,g)$$

- **Text Metric**: Jointly evaluates text content and font style:
  $$S_T(p,g) = S_{\text{base}}(p,g) \cdot (1 - \lambda M_f - \alpha M_s)$$
  where $M_f$ and $M_s$ denote font family and font size mismatches, respectively, with penalty coefficients $\lambda = \alpha = 0.3$.

Both metrics employ the Hungarian algorithm for optimal matching. Pearson and Spearman correlations with human ratings exceed 0.7 ($p \ll 0.01$) across three benchmarks.

### 3. ChartEditor Model

**Base model**: Qwen-2.5-VL-3B

**Two-stage cold-start SFT**:
- **Stage 1 (Chart-to-Code SFT)**: 140,000 high-quality Chart-to-Code samples to establish correspondence between chart visual features and plotting code.
- **Stage 2 (Chart Editing SFT)**: 20,000 chart editing samples to establish correspondence between editing instructions and code modifications.

**GRPO reinforcement learning**: Trained on 6,000 ChartEditVista samples with three reward functions:
- **Format Reward**: Checks whether the output contains the `<think>` and `<code>` tag format.
- **Execution Reward**: Executes code in an isolated sandbox; rewards 1 upon successful execution.
- **Rendering Reward**: Rule-based visual fidelity evaluation that applies the Hungarian algorithm to optimally match predicted and ground-truth objects, computing a weighted similarity score $\mathcal{R}_{\text{render}} = R_E \sum_{t \in \mathcal{T}} w_t S_{\text{type}=t}$.

**Curriculum reinforcement learning**: Training begins with single-subplot, single-instruction samples and progressively incorporates multi-subplot, multi-instruction samples to maintain stable reward signal density.

## Key Experimental Results

| Model | Params | ChartEditVista Avg | ChartEdit w/o Code | ChartMimic Overall |
|---|---|---|---|---|
| GPT-4o | — | 43.5 | 79.9 | 83.2 |
| Gemini-2.5-Pro | — | 66.0 | 89.2 | 82.4 |
| Qwen2.5-VL-72B | 72B | 29.8 | 71.0 | 68.4 |
| Qwen2.5-VL-3B | 3B | 9.5 | 24.3 | 24.6 |
| **ChartEditor** | **3B** | **58.1** | **55.3** | **55.0** |

Key findings:
- ChartEditor-3B achieves an average score of 58.1 on ChartEditVista, surpassing GPT-4o (43.5) and Qwen2.5-VL-72B (29.8).
- Code execution rate improves from 45.8% (base model) to 76.8%.
- Ablation studies confirm complementary gains from the two-stage SFT (Edit SFT alone: 53.2; C2C SFT alone: 54.3; combined: 58.1); curriculum learning raises the average from 56.6 to 58.1; and the combination of all three rewards yields the best performance.
- The rendering reward is also effective on Chart-to-Code tasks (ChartMimic Direct Mimic: 54.5 vs. 26.1).

## Highlights & Insights

1. **Elegant data construction pipeline**: The hierarchical Chart JSON → Chart Tree representation enables systematic coverage of editable elements, complemented by multi-round quality control to ensure data integrity.
2. **Innovative rendering reward**: Evaluating visual fidelity directly at the rendering level avoids the limitations of pure text matching and transfers effectively to Chart-to-Code tasks.
3. **Strong capability from a small model**: A 3B-parameter model surpasses 72B and several closed-source models, validating the effectiveness of domain specialization combined with RL.
4. **Reliable evaluation metrics**: RARM correlates strongly with human ratings across three benchmarks, addressing the hallucination and coarse-granularity issues of existing metrics.

## Limitations & Future Work

1. The benchmark covers only Python matplotlib-style charts and does not address other visualization frameworks such as D3.js or ECharts.
2. The chart editing task is defined as image + instruction → code, without consideration of interactive multi-turn editing scenarios.
3. The 3B model still lags behind closed-source models on out-of-domain benchmarks (ChartEdit: 55.3 vs. GPT-4o 79.9).
4. The rendering reward relies on automatic parsing of chart elements and may fail on complex or non-standard charts.

## Related Work & Insights

| Aspect | ChartEditVista | ChartEdit | ChartMimic | ChartCraft |
|---|---|---|---|---|
| Chart types | 31 | 19 | 22 | 5 |
| Dataset size | 7.9k | 1.4k | 2.4k | 5.5k |
| Edit types | 6 | 6 | 1 | 4 |
| Editable objects | Unrestricted | Limited | Limited | Limited |
| Requires original code | No | Yes | Yes | Yes |

Compared to ChartCoder, the key distinction of this work lies in introducing rendering reward into reinforcement learning for chart editing rather than solely for Chart-to-Code, alongside the proposal of a systematic curriculum learning strategy.

The rendering reward paradigm is broadly applicable to other tasks involving code generation with visual output (e.g., UI generation, slide generation), with the core idea being the use of visual rendering results as RL reward signals. The hierarchical Chart Tree approach to edit target selection is similarly transferable to other structured editing tasks (e.g., document editing, web page editing). Curriculum reinforcement learning (progressing from simple to complex samples) represents a general strategy for addressing sparse reward problems.

## Rating
- **Novelty**: 8/10 — The combination of rendering reward and GRPO for chart editing is novel, and the data construction pipeline is systematic.
- **Experimental Thoroughness**: 9/10 — Evaluation across three benchmarks with detailed ablations, human evaluation, and metric reliability verification.
- **Writing Quality**: 8/10 — Well-structured with detailed descriptions of the pipeline and methodology.
- **Value**: 8/10 — The benchmark and metrics offer meaningful contributions to the community; the practical significance of a small model outperforming larger ones is substantial.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework](distilling_deep_reinforcement_learning_into_interpretable_fuzzy_rules_an_explain.md)
- [\[ICLR 2026\] Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](../../ICLR2026/reinforcement_learning/distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)
- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)
- [\[ICLR 2026\] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](../../ICLR2026/reinforcement_learning/solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [\[AAAI 2026\] RLSLM: A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locomotion Model with Human Social Norms](rlslm_a_hybrid_reinforcement_learning_framework_aligning_rule-based_social_locom.md)

<!-- RELATED:END -->
