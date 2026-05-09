---
title: >-
  [Paper Note] WebDS: An End-to-End Benchmark for Web-based Data Science
description: >-
  [Multimodal VLM] This paper introduces WebDS, the first end-to-end web-based data science benchmark comprising 870 tasks across 29 websites and 10 domains. The strongest evaluated agent (BrowserUse + GPT-4o) completes only 15% of tasks, while humans achieve 90%, revealing a substantial performance gap in realistic data science workflows.
tags:
  - Multimodal VLM
date: 2026-05-08
content_hash: 5eb62946e4bd95d9
---

# WebDS: An End-to-End Benchmark for Web-based Data Science

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2508.01222](https://arxiv.org/abs/2508.01222)
- **Code**: [WebDS Benchmark](https://webdsbenchmark.github.io/)
- **Area**: Multimodal Large Models / Web Agent / Data Science
- **Keywords**: web agent, data science, benchmark, end-to-end evaluation, multi-step reasoning

## TL;DR

This paper introduces WebDS, the first end-to-end web-based data science benchmark comprising 870 tasks across 29 websites and 10 domains. The strongest evaluated agent (BrowserUse + GPT-4o) completes only 15% of tasks, while humans achieve 90%, revealing a substantial performance gap in realistic data science workflows.

## Background & Motivation

Real-world data science tasks involve complex web interactions: locating appropriate data online, synthesizing multimodal information from heterogeneous sources, and generating summarized analyses. However, existing benchmarks suffer from two critical shortcomings:

**Web Agent benchmarks** (e.g., WebVoyager, WebArena) focus on simple interactions (posting, shopping) and do not require diverse tool usage or data analysis capabilities.

**Data science benchmarks** (e.g., InfiAgent-DABench, DSBench) concentrate on static structured datasets and do not cover end-to-end workflows from data acquisition to analysis.

**Root Cause**: Real data science workflows typically begin with web browsing and require navigating and synthesizing information across multiple websites — a critical step overlooked by existing benchmarks. For instance, BrowserUse achieves 80% on WebVoyager but only 15% on WebDS.

## Method

### Overall Architecture

WebDS is the first benchmark evaluating the complete data science pipeline: **web browsing for data acquisition** → **data analysis/visualization** → **generating grounded outputs**.

### Benchmark Design

**Expert-interview-driven**: Eight journalists, data scientists, and domain experts were interviewed to identify two major task categories:
- Tasks producing downstream artifacts (reports, visualizations)
- Tasks answering key analytical questions

**Website coverage**: 29 data-rich websites (CDC, government data portals, news media, etc.) spanning 10 high-stakes domains, covering both structured data (CSV, tables) and unstructured data (text, figures).

### Task Design

870 manually authored tasks, each annotated with 7 attributes:

- **QA vs. Action**: Question-answering tasks (344 single-hop + 117 multi-hop) vs. action tasks (97 single-hop + 134 multi-hop + 139 requiring tools)
- **Single-hop vs. Multi-hop**: Whether combining multiple data sources is required
- **Structured vs. Unstructured**: Data format
- **Tool Use**: Whether external tools such as Python or SQL are required
- **Web Navigation**: Whether website interaction is required
- **Multi-site**: Whether cross-site information aggregation is involved

### Difficulty Classification

$$
\text{Difficulty} = \begin{cases}
\text{Easy (247)} & \text{no multi-hop/non-text/action/tool, single-site} \\
\text{Medium (275)} & \text{exactly one of the above, single-site} \\
\text{Hard (348)} & \text{two or more of the above, or multi-site}
\end{cases}
$$

### Dual-Track Evaluation

- **WebDS-live**: Direct interaction with real websites, capturing authentic web complexity
- **WebDS-dockerized**: Containerized deployment of a subset of websites, ensuring reproducibility

### Evaluation Protocol

1. **Automatic binary evaluation**: For tasks with reference answers, an LLM compares outputs against ground truth → SUCCESSFUL/UNSUCCESSFUL
2. **LLM-based subjective scoring (1–5)**: Extends the WebVoyager methodology by evaluating complete trajectories rather than only final screenshots, providing five-level scores with failure analysis
3. **Human validation**: 400 task–trajectory pairs reviewed independently; the evaluation framework achieves 93% agreement with human judgments

## Experiments

### Main Results

| Agent | Framework | SR% |
|-------|-----------|-----|
| GPT-4o + BrowserUse | BrowserUse | 13.2% |
| GPT-4o + AgentOccam | AgentOccam | 4.8% |
| Claude Sonnet-4.5 + WebArena | WebArena | ~10% |
| GPT-5.1 + WebArena | WebArena | ~12% |
| **Human Baseline** | Browser | **90% (±3%)** |

### Key Findings

1. **Large human–agent gap**: The strongest agent achieves only 13.2% while humans reach 90%, a gap of ~77 percentage points.
2. **Scaling model capacity yields no significant improvement**: GPT-4o, GPT-4o-mini, and Qwen2.5-72B perform comparably.
3. **Novel failure modes**:
    - **Information anchoring errors**: Conflicts between anchored and latent knowledge
    - **Repetitive behavior**: Agents enter loops in multi-hop tasks
    - **Shortcutting**: Skipping necessary data acquisition steps
4. **Clear difficulty gradient**: Agent scores on Easy tasks are approximately 2.5× higher than on Medium/Hard tasks.
5. **Cross-benchmark gap**: 81.1% on WebVoyager vs. 13.2% on WebDS for the same agent.

### Comparison with WebVoyager / WebArena

| Feature | WebVoyager | WebArena | WebDS |
|---------|-----------|----------|-------|
| Multi-hop | ✗ | ✓ | ✓ |
| Structured data | ✗ | ✗ | ✓ |
| Unstructured data | ✗ | ✗ | ✓ |
| Multi-site | ✗ | ✓ | ✓ |
| Tool use | ✗ | ✓ | ✓ |
| End-to-end data science | ✗ | ✗ | ✓ |

## Highlights & Insights

- First end-to-end web-based data science benchmark, bridging the gap between web interaction and data science capabilities
- 870 high-quality manually authored tasks with fine-grained annotation across 7 attributes and 3 difficulty levels
- Dual-track design (live + dockerized) balances authenticity and reproducibility
- Full trajectory evaluation with fine-grained scoring, going beyond simple binary judgments
- Quantifies the substantial human–agent gap, providing clear directional guidance for the community

## Limitations & Future Work

- Currently covers only 29 websites, limiting domain representativeness
- Containerized deployment covers only a subset; tasks relying on live websites may change over time
- High manual annotation cost; 870 tasks may be insufficient to cover all real-world scenarios
- Evaluation still relies on LLM-as-Judge, which may be imprecise for assessing complex analytical reports
- Capability differences across different types of tool usage are not thoroughly analyzed

## Related Work & Insights

- **Data analysis benchmarks**: SQuAD, HotpotQA (structured QA); InfiAgent-DABench, DSBench (data science agents); Spider 2.0 (enterprise SQL)
- **Web agent benchmarks**: WebArena (functional correctness), WebVoyager (final screenshot), Mind2Web (action sequences)
- **End-to-end workflows**: GAIA (multimodal reasoning), AssistantBench (web-assisted tasks) — neither focuses on the data science pipeline

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First end-to-end web-based data science benchmark with a novel problem formulation
- **Technical Depth**: ⭐⭐⭐⭐ — Rigorous task design and comprehensive evaluation framework
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 9 state-of-the-art agents + human baseline with multi-dimensional analysis
- **Value**: ⭐⭐⭐⭐⭐ — Reveals critical agent deficiencies in real-world data science and guides future development

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/multimodal_vlm/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [\[ICLR 2026\] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)

<!-- RELATED:END -->
