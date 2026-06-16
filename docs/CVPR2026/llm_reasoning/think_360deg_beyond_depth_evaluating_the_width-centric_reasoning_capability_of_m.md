---
title: >-
  [Paper Note] Think 360°: Beyond Depth — Evaluating the Width-centric Reasoning Capability of MLLMs
description: >-
  [CVPR 2026][LLM Reasoning][MLLM benchmark] The authors propose Think360°, a multimodal benchmark that establishes "reasoning width" (breadth-first search, multi-constraint pruning, backtracking) as an orthogonal dimension to "reasoning depth" (long-chain sequential reasoning). It features over 1,200 high-quality cross-domain problems and a Tree-of-Thought evalu
tags:
  - CVPR 2026
  - LLM Reasoning
  - MLLM benchmark
date: 2026-05-08
content_hash: 9013497ca71794c1
---
# Think 360°: Beyond Depth — Evaluating the Width-centric Reasoning Capability of MLLMs

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Think_360deg_Beyond_Depth_Evaluating_the_Width_centric_Reasoning_Capability_of_CVPR_2026_paper.html)  
**Code**: To be confirmed (The paper claims it will be open-sourced at "Think360")  
**Area**: LLM Reasoning / Multimodal Evaluation Benchmarks  
**Keywords**: Multimodal Reasoning, Reasoning Width, Tree-of-Thought Evaluation, MLLM Benchmark, Width vs. Depth

## TL;DR
The authors propose Think360°, a multimodal benchmark that establishes "reasoning width" (breadth-first search, multi-constraint pruning, backtracking) as an orthogonal dimension to "reasoning depth" (long-chain sequential reasoning). It features over 1,200 high-quality cross-domain problems and a Tree-of-Thought evaluation protocol to simultaneously quantify width and depth accuracy. Evaluations of 30+ MLLMs across 12 major series reveal that while current models can handle long-chain deep reasoning, they generally struggle with insightful reasoning that integrates "wide search + deep chains."

## Background & Motivation

**Background**: Currently, the "gold standard" for measuring reasoning capability relies almost entirely on the length of the chain-of-thought—whether through test-time scaling, SFT, or RL alignment, the mainstream narrative is "longer is stronger." Multimodal mathematical and logical benchmarks (MathVision, MathVerse, MathVista, OlympiadBench, etc.) have also expanded along the axes of difficulty (K12 → Graduate → Olympiad), task coverage, and modality.

**Limitations of Prior Work**: These benchmarks share a hidden single axis—they almost exclusively evaluate **reasoning depth** (how far a single chain can extend), favoring "monotonic reasoning." In this paradigm, conclusions extend step-by-step from premises, and the primary challenge is "retrieving relevant knowledge." However, human problem-solving rarely relies on linear deduction alone. Instead, it involves 360° multi-directional exploration in the solution space: branching from mental anchors, pruning dead ends, backtracking to revisit hypotheses, and reorganizing fragmented clues until an "aha moment" occurs.

**Key Challenge**: Equating "reasoning capability" with "chain length" essentially conflates two orthogonal dimensions: **depth** (the ability to follow a long sequential reasoning chain without contradiction) and **width** (the ability to systematically branch, backtrack, and prune across multiple competing hypotheses before converging). Evaluating only depth overestimates models that "can calculate but cannot trial-and-error."

**Goal**: To construct a multimodal benchmark explicitly focused on reasoning width while still being able to quantify depth, and to answer: Can models ❶ systematically explore the solution space through trial-and-error, ❷ efficiently prune unfeasible branches under multiple constraints, and ❸ unify scattered clues into a coherent answer—all while performing joint linguistic and visual reasoning?

**Key Insight**: The authors draw an analogy to "depth vs. width" in neural networks—where depth represents stacked sequential feature abstraction and width represents parallel paths capturing diverse representations. They map shortcut/dropout, pyramid features, and backpropagation to pruning, divide-and-conquer, and backtracking in reasoning, respectively. Following this analogy, width-centric reasoning corresponds to breadth-first search via trial-and-error.

**Core Idea**: To complement reasoning evaluation with the neglected dimension of "width" (trial-and-error / multi-constraint pruning / backtracking). By using Tree-of-Thought, a model's response is decomposed into a tree where width is calculated across "sibling nodes" and depth is measured along the "longest path."

## Method

### Overall Architecture
Think360° is not a new model but "a dataset + an evaluation protocol." The work consists of two parts: (1) **Benchmark Construction**—collecting 1,200+ problems from competitions, textbooks, existing benchmarks, and online puzzles, followed by two-level quality filtering ("coarse screening + manual fine screening") and rewriting proof and game problems for objective verification; (2) **Evaluation Protocol**—besides standard pass@1, it introduces Tree-of-Thought Evaluation (ToT-Eval). This uses GPT-4o to extract a hierarchical tree from a model’s full response, verifies each node, and then calculates width scores based on "average group accuracy of sibling nodes" and depth scores based on "longest path accuracy." It also records reasoning time and token consumption to analyze the trade-off between performance and efficiency. A total of 30+ MLLMs from 12 series were evaluated across various difficulties, types, and capabilities. As this is a benchmark and evaluation protocol work, there is no trainable module pipeline.

### Key Designs

**1. Formalizing "Reasoning Width" as an Orthogonal Dimension to Depth**  
The pain point is that existing benchmarks collapse reasoning into "chain length," failing to expose the weakness where models "can reason forward but cannot search laterally." The authors define two orthogonal dimensions—**Depth**: the capability to follow a long sequence of reasoning chains without contradiction; **Width**: the capability to traverse multiple competing hypotheses through branching, backtracking, and selective pruning before convergence. Three sub-patterns of width (trial-and-error exploration, multi-constraint parallel pruning, and fragmentary clue merging) are mapped to cognitive skills: Branch-and-Bound, Hypothesize-and-Test, Divide-and-Conquer, Trial-and-Error, and Perceive-and-Comprehend. This formalization turns "width" from a vague intuition into a quantifiable object. Tab. 2 shows that "width-centric problems" in existing benchmarks are generally $<12\%$ (MathVision ~11.5%, MathVista ~2.7%, MathVerse ~1.3%), whereas Think360° reaches ~100% and is the only one with a width taxonomy.

**2. Three-Stage Benchmark Construction Pipeline**  
Width-centric problems are inherently scarce and difficult to annotate. The authors used a three-stage pipeline to transform them into objectively evaluable items. ① **Seed Collection**: Four sources (Math/Logic competitions like ARML, HMMT; textbook examples; existing benchmarks; online puzzles/IQ games). ② **Coarse-to-Fine Quality Filtering**: Coarse screening uses static keyword matching (e.g., "maximum/minimum", "possible ways", which are correlated with LRM "aha moments") + GPT-4o as a LLM-as-Judge. Fine screening involves manual double-checking to ensure quality and diversity. ③ **Annotation and Rewriting**: For proof problems, verifiable numerical relationships or specific conclusions are extracted to redesign the questions into objective formats. For game problems, initial screenshots are used as visual conditions, state possibilities are enumerated as candidates, and questions targeting specific locations are designed.

**3. Tree-of-Thought Dual-Dimension Evaluation Protocol (ToT-Eval)**  
Outcome-based matching (binary scoring of final answers) cannot distinguish "wide reasoning" from "deep reasoning." ToT-Eval involves two steps: ① **Tree Construction**: Given a problem and the model's full response, GPT-4o extracts key reasoning steps into a hierarchical tree. Parent-child relationships represent sequential reasoning (depth), while sibling nodes represent parallel exploration of alternatives (width). ② **Depth/Width Scoring**: GPT-4o judges if each node is logically consistent, factually correct, and correctly grounded. After judging all nodes, **Width Score = Avg. Group Accuracy of siblings**, and **Depth Score = Avg. Longest Path Accuracy**. This allows a single thought tree to reflect both "whether the model explores multiple paths and prunes" (width) and "whether a single chain can reach the end reliably" (depth).

### Example: Tetris-style Triomino Problem
For a problem asking "whether a triomino can leave exactly one square in a corner on an 8-wide screen," ToT-Eval extracts a tree: the root branches into **sibling nodes** like "screen width analysis," "single-cell strategy," "parity consideration," and "possible grid sizes" (width signal). The "possible grid size" node further expands into sub-reasoning for 3/6/9 cells (depth signal), eventually converging to "Answer: yes." By judging nodes, the average correctness of siblings gives the width score, while the longest correct path from root to conclusion gives the depth score.

## Key Experimental Results

### Benchmark Statistics & Comparison

| Dimension | Think360° |
| :--- | :--- |
| Total Problems | 1225 (testmini 740, 60%) |
| Answer Type | Free-form 83.1% (Num 54.3% / Struct 37.0% / Formula 5.2%), MCQ 16.9% |
| Difficulty | Easy 10.4% / Basic 22.2% / Medium 33.6% / Hard 23.9% / Olympiad 9.9% |
| Cognitive Skill | 5 Cognitive Skills, 6 Problem Types (multi-label) |

| Benchmark | Width-centric Ratio | Width Taxonomy | Difficulty Levels |
| :--- | :--- | :--- | :--- |
| MathVista | ~2.7% | ✗ | K-12 |
| MathVerse | ~1.3% | ✗ | K-12 |
| MathVision | ~11.5% | ✗ | K-12/College |
| OlympiadBench | ~1.7% | ✗ | Olympiad |
| **Think360° (Ours)** | **~100%** | **✓** | K-12/College/Olympiad |

> Metrics: **pass@1** = outcome matching accuracy; **ToT Width Acc.** = average group accuracy of sibling nodes; **ToT Depth Acc.** = average longest correct path accuracy.

### Main Results (pass@1 ALL, Excerpts)

| Model | Type | ALL Acc./% | Notes |
| :--- | :--- | :--- | :--- |
| Gemini-2.5-pro | Closed | 46.0 | Highest score |
| o3 | Closed | 42.3 | Top tier closed-source |
| o4-mini | Closed | 42.1 | — |
| Gemini-2.5-flash-thinking | Closed | 38.3 | — |
| o1 | Closed | 36.8 | — |
| Claude-3.7-Sonnet-Thinking | Closed | 35.5 | — |
| GPT-4o | Closed | 16.0 | Significantly low for non-thinking |
| GLM-4.1V-Thinking | Open 9B | 22.6 | Best open-source |
| Kimi-VL-Instruct | Open 16A3B | 10.1 | — |
| LLaVA-OneVision | Open 7B | 8.3 | — |
| Llama-3.2-Vision-Instruct | Open 11B | 7.1 | Bottom tier |

### Key Findings
- **Low Ceiling**: Even the strongest Gemini-2.5-pro reaches only 46.0%, and o3 reaches 42.3%, indicating that width-centric reasoning is a significant bottleneck for current MLLMs.
- **Thinking Models >> Non-Thinking Models**: The o-series, Gemini-thinking, and Claude-thinking are significantly higher than GPT-4o or Gemini-flash; Doubao-thinking (34.7) vs. its non-thinking version (32.8) confirms test-time reasoning helps, but at the cost of drastic token/time increases (o3 takes ~6300 tokens and ~261s per problem).
- **Branch-and-Bound is the Hardest**: Among the five cognitive skills, Branch-and-Bound (requiring systematic enumeration + pruning) yields the lowest scores across almost all models (e.g., GPT-4o is only 9.9%).
- **Open-Closed Source Gap**: The best open-source model GLM-4.1V (22.6) is less than half as effective as most closed-source thinking models. Small models perform at near-random levels on width-centric tasks.
- **Deep Chains $\neq$ Wide Search**: Current models perform well on common sense/VQA but struggle to combine "deep sequential chain-of-thought" with "wide exploratory search" for true insightful reasoning.

## Highlights & Insights
- **Novel Dimension Perspective**: The work explicitly deconstructs "reasoning strength = chain length" into two orthogonal axes (depth and width), using neural network analogies (shortcuts ↔ pruning, backprop ↔ backtracking) to turn intuition into a measurable dimension.
- **ToT-Eval as a Reusable Metric Protocol**: Upgrading from "outcome matching" to "extracting thought trees and measuring width/depth" provides a diagnostic tool to identify whether a model lacks breadth or depth.
- **Data Taming Methodology**: The systematic conversion of "hard-to-verify proofs" and "answer-less games" into automatically gradable items via numerical extraction and state enumeration is a useful engineering insight for scaling width-centric benchmarks.
- **Keywords vs. Aha Moments**: Identifying keywords like "maximum/minimum" for coarse screening and linking them to "Aha moments" provides a low-cost heuristic for mining complex reasoning problems.

## Limitations & Future Work
- **Heavy Reliance on LLM-as-Judge**: Tree construction, node scoring, and answer extraction all rely on GPT-4o/GPT-4o-mini, which may introduce biases or be limited by the judge's own capabilities.
- **Caveats in Cross-Model Comparison**: Thinking models have much higher token/time budgets than non-thinking models, so direct accuracy comparisons are not entirely equal without looking at effect-efficiency curves.
- **Sensitivity to Tree Structure**: The absolute width/depth scores depend on the shape of the tree extracted from the response.
- **Future Directions**: Using ToT-Eval scores as training signals (e.g., RL rewards for width), expanding width problems to non-math/logic domains, and introducing multi-judge voting.

## Related Work & Insights
- **vs. MathVision / MathVerse / MathVista**: These focus on comprehensive multimodal math but are biased towards monotonic reasoning; Think360° focuses specifically on the "width" dimension (~100% width-centric problems).
- **vs. CLEVR / GQA**: Earlier compositional vision benchmarks lacked multimodal mathematical reasoning; this work spans K12 to Olympiad levels and emphasizes trial-and-error.
- **vs. Outcome-based (pass@1) Evaluation**: ToT-Eval refines "correctness" into process-based width/depth scores, distinguishing lucky guesses or single-chain successes from systematic wide searches.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formally defining "reasoning width" and providing the ToT dual-dimension evaluation fills a significant gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluates 30+ models across 12 series, covering various difficulties and cognitive skills with efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear analogies and definitions, though some construction details are in the appendix and notations are dense.
- Value: ⭐⭐⭐⭐ Provides quantifiable counterexamples and diagnostic tools for the "long CoT = strong reasoning" assumption, offering significant value to the evaluation community.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[NeurIPS 2025\] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning](../../NeurIPS2025/llm_reasoning/videorft_incentivizing_video_reasoning_capability_in_mllms_via_reinforced_fine-t.md)
- [\[AAAI 2026\] Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](../../AAAI2026/llm_reasoning/beyond_react_a_planner-centric_framework_for_complex_tool-au.md)
- [\[CVPR 2026\] Revisiting the Necessity of Lengthy Chain-of-Thought in Vision-centric Reasoning Generalization](revisiting_the_necessity_of_lengthy_chain-of-thought_in_vision-centric_reasoning.md)
- [\[CVPR 2026\] Think-as-You-See: Streaming Chain-of-Thought Reasoning for Large Vision-Language Models](think-as-you-see_streaming_chain-of-thought_reasoning_for_large_vision-language_.md)
- [\[CVPR 2026\] Improving Vision-language Models with Perception-centric Process Reward Models](improving_vision-language_models_with_perception-centric_process_reward_models.md)

</div>

<!-- RELATED:END -->
