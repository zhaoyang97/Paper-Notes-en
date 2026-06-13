---
title: >-
  [Paper Note] Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study
description: >-
  [AAAI 2026][Code Intelligence][Data Analysis] This paper systematically investigates the capability bottlenecks of open-source LLMs in data analysis tasks. It decomposes data analysis into three dimensions—data comprehen…
tags:
  - "AAAI 2026"
  - "Code Intelligence"
  - "Data Analysis"
  - "LLM Agent"
  - "Strategic Planning"
  - "Data Synthesis"
  - "Open-Source Models"
date: 2026-05-08
content_hash: 1be31e17d673a33d
---

# Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study

**Conference**: AAAI 2026  
**arXiv**: [2506.19794](https://arxiv.org/abs/2506.19794)  
**Code**: [github.com/zjunlp/DataMind](https://github.com/zjunlp/DataMind)  
**Area**: Code Intelligence  
**Keywords**: Data Analysis, LLM Agent, Strategic Planning, Data Synthesis, Open-Source Models

## TL;DR

This paper systematically investigates the capability bottlenecks of open-source LLMs in data analysis tasks. It decomposes data analysis into three dimensions—data comprehension, code generation, and strategic planning—and identifies **strategic planning as the decisive factor**, rather than coding or data comprehension. A strategy-guided data synthesis approach is proposed, enabling fine-tuned 7B/14B models to achieve performance competitive with GPT-4o.

## Background & Motivation

### Problem Definition

Data analysis is a complex interactive process central to scientific discovery, business intelligence, and decision-making. It requires models to understand natural language queries, interpret structured data, formulate hypotheses, generate executable code, and iteratively refine reasoning across multiple interaction turns. Performance on data analysis tasks remains dominated by large closed-source models such as GPT-4 and DeepSeek-R1, while open-source smaller models struggle in real-world scenarios.

### Core Research Question

**How can the capabilities of open-source LLMs on complex, reasoning-intensive data analysis tasks be effectively improved?**

Prior work in mathematics and code generation has demonstrated that fine-tuning on high-quality synthetic data can enhance reasoning ability. However, data analysis tasks involve multi-step interactions, dynamic environments, and mixed objectives. It remains unclear which training data attributes—task difficulty, scenario diversity, and interaction structure—genuinely lead to better generalization.

### Research Methodology

The authors adopt a **capability-aware approach**, decomposing the data analysis process into three core dimensions:

**Data Comprehension**: Understanding and effectively utilizing structured data

**Code Generation**: Producing correct and efficient analysis code

**Strategic Planning**: Decomposing complex problems into manageable steps

The data analysis function is formally defined as $f_\theta: (\mathcal{D}, \mathcal{Q}, \mathcal{T}) \rightarrow (\mathcal{S}, \mathcal{R})$, where $\mathcal{D}$ is structured data, $\mathcal{Q}$ is the analysis objective, $\mathcal{T}$ is the tool library, $\mathcal{S}$ is the sequence of intermediate analysis states, and $\mathcal{R}$ is the final report.

## Method

### Overall Architecture

The study proceeds in two stages:
1. **Capability Diagnosis** (Section 4): Systematic ablation experiments are conducted to analyze which factors across the three core dimensions genuinely affect performance.
2. **Data Synthesis** (Section 5): Based on diagnostic insights, a strategy-guided data synthesis method is designed to improve model performance.

### Key Designs

#### 1. **Data Comprehension Experiments**: Investigating the Effect of Tabular Information and Input Complexity

**Experiment 1: Tabular Information Visibility**

| Setting | QRData (7B) | QRData (14B) | DiscoveryBench (7B) | DiscoveryBench (14B) |
|------|------------|-------------|-------------------|---------------------|
| w/o Info | 6.57 | 15.09 | 0.42 | 0.42 |
| w/ Info | 7.54 | 15.82 | 1.26 | 0.00 |

Providing tabular information (column names, data types, sample entries) yields only marginal improvement. The 14B model even degrades on DiscoveryBench, possibly because longer inputs reduce output focus.

**Experiment 2: Data Complexity**

Irrelevant tables are introduced as semantic noise:

| Setting | QRData (7B) | QRData (14B) | DiscoveryBench (7B) | DiscoveryBench (14B) |
|------|------------|-------------|-------------------|---------------------|
| w/o Extra | 37.96 | 52.55 | 5.44 | 10.88 |
| w/ Extra | 34.55 | 52.07 | 4.18 | 12.13 |

The 14B model is nearly unaffected, while the 7B model shows only a slight drop. This indicates that **data comprehension is not a primary bottleneck**, as models have already internalized basic data understanding during pretraining.

#### 2. **Code Generation Experiments**: Evaluating the Role of Coding in Data Analysis

Multi-model comparison (multi-turn interaction setting):

| Model | QRData | DiscoveryBench |
|------|--------|---------------|
| Qwen2.5-7B-Instruct | 39.71% | 14.64% |
| Qwen2.5-14B-Instruct | 53.53% | 24.27% |
| Qwen2.5-32B-Instruct | 57.18% | 27.62% |
| Qwen2.5-7B-Coder | 36.50% | 13.60% |
| R1-Distill-Qwen-7B | 30.41% | 7.95% |
| GPT-4o | 59.85% | 28.03% |
| DeepSeek-v3 | 65.21% | 36.82% |
| DeepSeek-R1 | 63.26% | 37.66% |

**Key Findings**:
- **Code specialization does not imply data analysis capability**: Qwen2.5-7B-Coder underperforms the general-purpose Instruct model.
- **Distillation may introduce functional hallucination**: R1-Distill-Qwen-7B performs worst, frequently hallucinating file interpretations rather than generating executable code.
- **Long context ≠ efficient execution**: The 1M-context variant has comparable coding ability to the standard version, but the latter achieves higher planning efficiency (completing tasks in fewer turns).

**Error Analysis** (manual annotation of 354 error samples): Only a minority of errors stem from syntactic/semantic code defects (e.g., invalid syntax); the majority originate from high-level reasoning failures (e.g., incorrect assumptions, premature termination), further confirming that **planning matters more than coding**.

#### 3. **Strategic Planning Experiments**: Systematic Evaluation across Four Key Aspects

**(a) Interaction Turns**

Three strategies: Short (2–3 turns), Medium (4–5 turns), Long (6+ turns), and Mixed.

| Turn Category | # Samples | QRData | DiscoveryBench |
|---------|-------|--------|---------------|
| All | 5613 | 48.66% | 15.00% |
| Short | 1034 | 47.68% | 23.85% |
| Medium | 3559 | 49.15% | 18.83% |
| Long | 1020 | 47.94% | 18.41% |
| Medium + Short | 4593 | 47.45% | 21.34% |

Core findings:
- Medium-length interactions generally perform best, balancing reasoning depth and focus.
- The mixed strategy consistently underperforms, as varying turn lengths interfere with the model's learning of stable interaction patterns.
- **Data quality > data quantity**: A subset of medium-turn samples consistently outperforms training on the full dataset.

**(b) Reasoning Length**

Three settings: Original, Full (complete \<think\> trajectories), and Summarized (summaries of reasoning traces).

Core findings:
- **Longer reasoning ≠ better performance**: The Full setting underperforms Original in most configurations.
- **Information relevance > reasoning length**: The Summarized setting consistently matches or exceeds the baseline.
- **Diminishing returns** on token budget: Increasing the budget is beneficial on QRData but degrades performance on DiscoveryBench.

**(c) Task Complexity**

Tasks are stratified by model capability: Easy (solvable by 7B), Medium (solvable only by 14B), and Hard (requiring DeepSeek-R1).

| Difficulty | QRData | DiscoveryBench |
|------|--------|---------------|
| Easy | 42.58% | 20.50% |
| Medium | 51.34% | 18.83% |
| Hard | 48.18% | 19.50% |
| Medium + Hard | 51.34% | 23.01% |

The Medium+Hard combination performs best on both datasets, indicating that exposure to more complex tasks enhances model generalization.

**(d) Question Diversity**

| Diversity | QRData | DiscoveryBench |
|--------|--------|---------------|
| Original Distribution | 46.72% | 20.92% |
| Balanced Sampling | 45.00% | 21.76% |

Domain diversity has a negligible effect, suggesting that **the diversity and richness of reasoning strategies matters more than the diversity of problem domains**.

### Loss & Training

**Three-Stage Strategy-Guided Data Synthesis**:

1. **Prompt-Based Answer Generation**: Multiple candidate answers are generated for each query using prompt engineering.
2. **Targeted Instance Selection**: Priority is given to medium-length dialogues and medium-to-high difficulty samples.
3. **Reasoning-Driven Data Enrichment**: A concise reasoning summary is appended to each selected instance.

The final dataset comprises 2.8k instances used for SFT fine-tuning. Fine-tuning is applied to Qwen2.5-7B/14B-Instruct via LoRA (for Strategic Planning evaluation).

## Key Experimental Results

### Main Results

| Model | QRData | DiscoveryBench |
|------|--------|---------------|
| GPT-4o | 59.85% | 28.03% |
| DeepSeek-v3 | 65.21% | 36.82% |
| DeepSeek-R1 | 63.26% | 37.66% |
| Qwen2.5-7B (baseline) | 39.71% | 14.64% |
| **Qwen2.5-7B (ours)** | **53.77%** | **22.59%** |
| Qwen2.5-14B (baseline) | 53.53% | 24.27% |
| **Qwen2.5-14B (ours)** | **58.15%** | **36.82%** |

The 7B model achieves substantial gains: +14.06% on QRData and +7.95% on DiscoveryBench. The 14B model reaches 36.82% on DiscoveryBench, matching DeepSeek-v3 and surpassing GPT-4o (28.03%).

### Ablation Study

| Ablation Dimension | Key Finding | Effect Magnitude |
|---------|---------|---------|
| Remove tabular information | Negligible performance change | Data comprehension is not a bottleneck |
| Use code-specialized model | Performance degrades | General-purpose Instruct model is superior |
| Short vs. medium turns | Medium turns generally better | Task-dependent |
| Full reasoning vs. summarized | Summarized consistently outperforms full | Quality > length |
| Medium difficulty vs. easy | Medium+Hard combination is optimal | Complex tasks enhance generalization |
| Domain diversity | Original distribution ≈ balanced sampling | Diversity has limited impact |

### Key Findings

1. **Strategic planning is the decisive factor**: Data comprehension and code generation are sufficiently acquired during pretraining; planning capability is the primary source of the gap between open-source and closed-source models.
2. **Interaction design and task complexity significantly influence reasoning capability**: Medium-length interactions combined with medium-to-high difficulty data yield the best learning efficiency.
3. **Data quality far outweighs diversity**: 2.8k carefully curated samples achieve comparable or superior results to training on tens of thousands of samples.
4. **Diminishing returns with scale**: The improvement from fine-tuning diminishes as model size increases (gains for 14B are less pronounced than for 7B).

## Highlights & Insights

1. **Capability decomposition methodology**: Decomposing data analysis into three orthogonal dimensions and analyzing each independently is methodologically rigorous and provides a valuable research paradigm for the community.
2. **"Planning > Coding > Data Comprehension" finding**: This challenges the common assumption that open-source LLMs lack coding ability, revealing that the true bottleneck lies in strategic planning.
3. **Reasoning summaries outperform full reasoning traces**: Concise reasoning summaries are more effective than lengthy chain-of-thought, indicating that information density and logical coherence matter more than length.
4. **Large gains from small data**: Only 2.8k strategy-guided synthetic samples enable the 14B model to match DeepSeek-v3 on DiscoveryBench.

## Limitations & Future Work

1. **Limited dataset scale**: 2.8k samples may be insufficient to cover a broader range of analytical scenarios.
2. **Training data distribution biased toward the 7B model**: The strategy is constructed based on Qwen2.5-7B and may not align well with the inductive biases of larger models.
3. **Only two benchmarks evaluated**: DiscoveryBench and QRData may not fully capture the complexity of real-world scenarios.
4. **No comparison with RL-based methods**: Only SFT is employed; the effectiveness of reinforcement learning (e.g., GRPO) on data analysis tasks remains unexplored.
5. **Static interaction patterns**: The mixed strategy underperforms but retains potential—more sophisticated designs such as curriculum scheduling or adaptive control are needed.
6. **Evaluation relies on GPT-4o-mini**: Using an LLM as a judge may introduce evaluation bias.

## Related Work & Insights

- **S1 & LIMO**: Demonstrate "budget control" and "emergent complex capabilities from a small number of curated samples" in mathematical reasoning, respectively, inspiring the data strategy in this paper.
- **Data Interpreter**: Employs a hierarchical dependency graph to represent workflows, enabling automatic task decomposition.
- **ReAct framework**: All models in this paper adopt ReAct for multi-turn interaction.
- **DeepSeek-R1**: Used to generate high-quality reasoning trajectories and serve as a "hard-task calibrator."
- **Insight**: In agentic settings, planning capability is far more important than any single skill (coding, data comprehension), which has significant implications for agent training strategies.

## Rating

- Novelty: ⭐⭐⭐⭐ (The capability decomposition methodology is novel and the insights are valuable)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multi-dimensional systematic ablations covering multiple models and settings)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, concise findings, rich figures and tables)
- Value: ⭐⭐⭐⭐ (Provides a practical roadmap for improving data analysis capabilities of open-source LLMs)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4](../../ACL2026/code_intelligence/discover_and_prove_an_open-source_agentic_framework_for_hard_mode_automated_theo.md)
- [\[NeurIPS 2025\] Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization](../../NeurIPS2025/code_intelligence/preserving_llm_capabilities_through_calibration_data_curation_from_analysis_to_o.md)
- [\[ACL 2026\] ChatHLS: Towards Systematic Design Automation and Optimization for High-Level Synthesis](../../ACL2026/code_intelligence/chathls_towards_systematic_design_automation_and_optimization_for_high-level_syn.md)
- [\[ICLR 2026\] Inference-Time Safety for Code LLMs via Retrieval-Augmented Revision](../../ICLR2026/code_intelligence/inference-time_safety_for_code_llms_via_retrieval-augmented_revision.md)
- [\[AAAI 2026\] DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)

</div>

<!-- RELATED:END -->
