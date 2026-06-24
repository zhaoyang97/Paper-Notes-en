---
title: >-
  [Paper Note] WebDS: An End-to-End Benchmark for Web-based Data Science
description: >-
  [Multimodal VLM] The authors propose WebDS, the first end-to-end web-based data science benchmark (870 tasks, 29 websites, 10 domains). The current strongest Agent (BrowserUse + GPT-4o) completes only 15% of tasks compared to 90% achieved by humans, revealing a significant performance gap in real-world data science workflows.
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: c5450af8c4161d64
---

# WebDS: An End-to-End Benchmark for Web-based Data Science

## Meta Information
- **Conference**: ICLR 2026
- **arXiv**: [2508.01222](https://arxiv.org/abs/2508.01222)
- **Code**: [WebDS Benchmark](https://webdsbenchmark.github.io/)
- **Area**: Multimodal Large Language Models / Web Agent / Data Science
- **Keywords**: web agent, data science, benchmark, end-to-end evaluation, multi-step reasoning

## TL;DR

The authors propose WebDS, the first end-to-end web-based data science benchmark (870 tasks, 29 websites, 10 domains). The current strongest Agent (BrowserUse + GPT-4o) completes only 15% of tasks compared to 90% achieved by humans, revealing a significant performance gap in real-world data science workflows.

## Background & Motivation

Real-world data science tasks involve complex web interactions: searching for appropriate data on the internet, synthesizing multimodal data from various locations, and generating summary analyses. However, existing benchmarks suffer from two key flaws:

**Web Agent Benchmarks** (e.g., WebVoyager, WebArena) focus on simple interactions (posting, shopping) and do not require diverse tool-use capabilities or data analysis.

**Data Science Benchmarks** (e.g., InfiAgent-DABench, DSBench) concentrate on static structured datasets and do not cover the end-to-end workflow from data acquisition to analysis.

**Key Challenge**: Real data science workflows typically begin with web browsing, navigating multiple websites, and synthesizing information, yet this critical link is ignored by existing benchmarks. For instance, BrowserUse achieves 80% on WebVoyager but only 15% on WebDS.

## Method

### Overall Architecture

WebDS is not a model but a benchmark that fully characterizes the "real-world data science pipeline." It formalizes a web data science task as a three-stage serial mapping $f = f_\alpha \circ f_a \circ f_d$: $f_d$ browses, navigates, and scrapes raw data from real websites; $f_a$ converts the captured multimodal data into analytical outputs (reports, visualizations, or predictions); and $f_\alpha$ provides an **optional** downstream action (e.g., posting, deleting comments), outputting null if no action is needed. Existing benchmarks cover either only the first half (pure web browsing) or only the second half (analysis on static data), while WebDS evaluates the entire end-to-end pipeline for the first time.

The construction of the benchmark pivots around this pipeline to answer four questions: task origin (expert interviews rather than researcher fabrication), complexity coverage (seven-dimensional attribute annotation + mechanical difficulty grading), execution environment (dual-track live and dockerized), and objective scoring (binary + five-level + human oversight).

### Key Designs

**1. Expert-driven Task Sources: Rooting tasks in real workflows**

Most web agent tasks are simple interactions designed by researchers, which differ significantly from the daily work of data practitioners. WebDS interviewed 8 journalists, data scientists, and domain experts to summarize two core task types: generating downstream products (reports, charts) or answering critical analytical questions. Based on this, 29 data-rich websites with non-overlapping data representations (CDC, government portals, news media, etc.) were selected across 10 high-stakes domains. 870 tasks were manually authored and double-checked, covering both structured (CSV, tables) and unstructured (text, graphics) data.

**2. Seven-dimensional Attribute Annotation and Difficulty Grading: Deconstructing complexity into reproducible dimensions**

To identify where agents fail, WebDS labels each task with 7 attributes: QA/Action, single-hop/multi-hop, structured/unstructured, requirement for external tools (Python/SQL), requirement for navigation, and cross-website functionality. Difficulty levels are mechanically derived based on whether a task involves these hard capabilities to ensure objectivity:

$$\text{Difficulty} = \begin{cases} \text{Easy (247)} & \text{No multi-hop/unstructured/action/tools, single website} \\ \text{Medium (275)} & \text{Exactly one of the above, single website} \\ \text{Hard (348)} & \text{Two or more of the above, or multi-website} \end{cases}$$

**3. Dual-track Deployment: Balancing realism and reproducibility**

To address the volatility of real-world websites, WebDS provides two tracks: WebDS-live allows agents to interact directly with real websites, recording page states and trajectories for auditing. WebDS-dockerized containerizes a subset of websites, freezing content and structure for deterministic execution and reproducible experiments. To prevent overfitting, the benchmark includes 470 public validation tasks and 400 private test tasks.

**4. Full-trajectory Three-tier Evaluation Protocol: Objective scoring for open-ended outputs**

Since data science outputs like reports are open-ended, WebDS uses three layers of evaluation: automatic binary evaluation for tasks with reference answers; an expanded LLM-as-Judge that evaluates the entire trajectory using $(\text{observation}, \text{action}, \text{next observation})$ triplets to provide a 1–5 score; and human verification to ensure reliability (achieving 93% consistency with automatic scores).

## Main Results

### Key Experimental Results

| Agent | Framework | SR% |
|-------|------|-----|
| GPT-4o + BrowserUse | BrowserUse | 13.2% |
| GPT-4o + AgentOccam | AgentOccam | 4.8% |
| **GPT-5.1 + BrowserUse (Live Best)** | BrowserUse | **22.2%** |
| **Human Baseline** | Browser | **90% (±3%)** |

> The same GPT-4o + BrowserUse configuration drops from 81.1% on WebVoyager to 13.2% on WebDS. Most models fail to exceed a 2% success rate on the live track.

### Key Findings

1.  **Massive Human-Machine Gap**: The strongest agent on the live track reaches only 22.2% compared to humans at 90%, a gap of ~68–77 percentage points.
2.  **Increased Model Capacity Provides Diminishing Returns**: GPT-4o, GPT-4o-mini, and Qwen2.5-72B show similar performance.
3.  **New Failure Modes**:
    *   **Information Anchoring Errors**: Conflicts between anchored knowledge and potential knowledge.
    *   **Repetitive Behavior**: Falling into loops during multi-hop tasks.
    *   **Shortcut Taking**: Skipping necessary data acquisition steps.
4.  **Significant Difficulty Gradient**: Agent scores on Easy tasks are approximately 2.5x higher than on Medium/Hard tasks.
5.  **Cross-Benchmark Disparity**: 81.1% on WebVoyager vs. 13.2% on WebDS for the same Agent.

### Comparison: WebVoyager / WebArena

| Feature | WebVoyager | WebArena | WebDS |
|------|-----------|----------|-------|
| Multi-hop | ✗ | ✓ | ✓ |
| Structured Data | ✗ | ✗ | ✓ |
| Unstructured Data | ✗ | ✗ | ✓ |
| Multi-website | ✗ | ✓ | ✓ |
| Tool Use | ✗ | ✓ | ✓ |
| End-to-End Data Science | ✗ | ✗ | ✓ |

## Highlights & Insights

- First end-to-end web data science benchmark bridging the gap between web interaction and data science capabilities.
- 870 high-quality, manually written tasks covering 7 attributes and 3 difficulty levels.
- Dual-track design (live + dockerized) balances authenticity and reproducibility.
- Full trajectory evaluation and fine-grained scoring go beyond simple binary judgment.
- Quantifies the massive human-machine gap, providing a clear direction for the community.

## Limitations & Future Work

- Currently covers only 29 websites; domain representation is limited.
- Containerized deployment is only a subset; live tasks are susceptible to website changes.
- High manual annotation costs; 870 tasks may not cover all real-world scenarios.
- Evaluation still relies on LLM-as-Judge, which may lack precision for complex report quality.
- Differences in tool-use capabilities across different types were not analyzed in depth.

## Related Work & Insights

- **Data Analysis Benchmarks**: SQuAD, HotpotQA (Structured QA), InfiAgent-DABench, DSBench (Data science agents), Spider 2.0 (Enterprise SQL).
- **Web Agent Benchmarks**: WebArena (Functional correctness), WebVoyager (Final screenshots), Mind2Web (Action sequences).
- **End-to-End Workflows**: GAIA (Multimodal reasoning), AssistantBench (Web assistant) — none focus specifically on the data science pipeline.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First end-to-end web data science benchmark with a novel problem definition.
- **Technical Depth**: ⭐⭐⭐⭐ — Rigorous task design and comprehensive evaluation system.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Includes 9 SOTA agents + human baseline with multi-dimensional analysis.
- **Value**: ⭐⭐⭐⭐⭐ — Highlights critical deficiencies of Agents in real-world data science.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](../../ICCV2025/multimodal_vlm/scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](../../ACL2025/multimodal_vlm/multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)

</div>

<!-- RELATED:END -->
