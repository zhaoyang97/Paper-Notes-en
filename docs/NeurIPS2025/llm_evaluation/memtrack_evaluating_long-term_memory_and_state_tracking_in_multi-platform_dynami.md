---
title: >-
  [Paper Note] MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments
description: >-
  [NeurIPS 2025 (SEA Workshop)][LLM Evaluation][long-term memory] This paper proposes the MEMTRACK benchmark to evaluate LLM agents' long-term memory and state tracking capabilities in multi-platform dynamic environments (Slack/Linear/Git), revealing that even the strongest model, GPT-5, achieves only 60% accuracy.
tags:
  - "NeurIPS 2025 (SEA Workshop)"
  - "LLM Evaluation"
  - "long-term memory"
  - "state tracking"
  - "multi-platform agents"
  - "benchmark"
  - "memory evaluation"
date: 2026-05-08
content_hash: 86440e11a588e953
---

# MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments

**Conference**: NeurIPS 2025 (SEA Workshop)

**arXiv**: [2510.01353](https://arxiv.org/abs/2510.01353)

**Code**: None (no public code link provided)

**Area**: Video Understanding / Agent Memory Evaluation

**Keywords**: long-term memory, state tracking, multi-platform agents, benchmark, memory evaluation

## TL;DR

This paper proposes the MEMTRACK benchmark to evaluate LLM agents' long-term memory and state tracking capabilities in multi-platform dynamic environments (Slack/Linear/Git), revealing that even the strongest model, GPT-5, achieves only 60% accuracy.

## Background & Motivation

- **Limitations of existing memory evaluation**: Most memory benchmarks focus on conversational settings (e.g., multi-turn dialogue memory) and neglect enterprise-level dynamic environments.
- **Practical demand**: Modern knowledge work spans multiple platforms (Slack messages, Git commits, Linear task management), requiring agents to integrate cross-platform information.
- **Core challenges**:
  1. **Asynchronous events**: Information across different platforms is updated asynchronously.
  2. **Noise and conflicts**: Information may be contradictory (e.g., Slack discussions overriding plans on Linear).
  3. **Cross-references**: Information on one platform may reference content on another.
  4. **Code comprehension**: Tasks may require reading and understanding codebases.
- **Goal**: To establish an ecologically valid benchmark reflecting real-world memory demands in software development workflows.

## Method

### Overall Architecture

The core design of MEMTRACK comprises:

1. **Multi-platform timelines**: Simulating asynchronous event streams across Slack, Linear, and Git.
2. **Memory capability taxonomy**: Testing acquisition, selection, and conflict resolution.
3. **Data generation pipeline**: Expert-crafted scenarios combined with scalable agent-based automatic synthesis.
4. **Multi-dimensional evaluation metrics**: Going beyond simple QA accuracy.

### Key Designs

#### 1. Scenario Construction

Each benchmark instance includes:
- **Timeline**: Chronologically ordered cross-platform events (50–200 messages/events).
- **Platform interleaving**: Slack messages, Linear issue updates, and Git commits/PRs interleaved at realistic cadences.
- **Noise injection**: Irrelevant chatter, outdated information, and mutually contradictory discussions.
- **Question set**: Queries that require integrating information across multiple platforms.

#### 2. Memory Capability Dimensions

| Capability Dimension | Description | Example |
|---------------------|-------------|---------|
| Memory Acquisition | Accurately extracting key information from an information stream | Finding an API change decision among 100+ Slack messages |
| Memory Selection | Selecting the most relevant information among multiple related items | Distinguishing temporary discussions from final decisions |
| Conflict Resolution | Handling contradictory information to determine the latest/authoritative version | Slack discussion overriding an outdated plan on Linear |
| Cross-Platform Reasoning | Integrating fragmented information from different platforms | Combining Git diffs and Slack discussions to understand the rationale for a code change |

#### 3. Data Generation

- **Expert design**: High-quality core scenarios are hand-crafted by experienced developers.
- **Agent synthesis**: LLMs generate scalable new scenarios based on expert-designed templates.
- **Quality control**: Human verification of the ecological validity of synthesized data.

### Loss & Training

MEMTRACK is an evaluation framework rather than a training methodology; its focus is on metric design:

- **Correctness Score**: Factual accuracy of responses (exact match + semantic match).
- **Efficiency Score**: Number of steps and resource consumption required for retrieval/reasoning.
- **Redundancy Score**: Amount of redundant or irrelevant information included in responses.

The composite scoring is designed to prevent "garbage information + correct answer" hacking strategies.

## Key Experimental Results

### Main Results

#### Model Performance on MEMTRACK

| Model | Correctness↑ | Efficiency↑ | Redundancy↓ | Overall↑ |
|-------|-------------|------------|-------------|---------|
| GPT-5 | **60.0** | **72.3** | 18.5 | **63.2** |
| Claude 3.5 Sonnet | 55.8 | 68.7 | 20.1 | 58.4 |
| GPT-4o | 52.1 | 65.4 | 22.8 | 54.7 |
| Gemini 1.5 Pro | 48.3 | 61.2 | 25.4 | 50.1 |
| LLaMA 3.1 70B | 39.7 | 52.8 | 31.2 | 41.5 |
| Mixtral 8x22B | 35.2 | 48.1 | 34.7 | 37.3 |

**Finding**: The strongest model, GPT-5, achieves only 60% correctness, indicating that long-term memory across multiple platforms remains a substantial challenge.

### Memory Backend Comparison

| Memory Approach | Correctness↑ | Notes |
|----------------|-------------|-------|
| Full Context (all input) | 52.1 | GPT-4o processing full text directly |
| RAG (retrieval-augmented) | 47.8 | Retrieving top-k relevant segments |
| Summary Memory | 44.3 | Periodically summarizing and compressing history |
| Sliding Window | 40.1 | Retaining only recent information |
| Full Context + GPT-5 | **60.0** | Stronger model + full text |

**Finding**: Simple RAG fails to surpass full-context input, suggesting that cross-platform information associations are difficult to capture through naive retrieval.

### Ablation Study

#### Per-Dimension Scores for Individual Memory Capabilities (GPT-4o)

| Capability Dimension | Correctness↑ | Relative Difficulty |
|---------------------|-------------|-------------------|
| Simple information acquisition | 71.2 | Easy |
| Cross-platform information acquisition | 48.5 | Moderate |
| Conflict resolution | 38.7 | Difficult |
| Multi-step cross-platform reasoning | 32.4 | Very difficult |
| Code comprehension + memory | 29.8 | Extremely difficult |

### Key Findings

1. **60% is the ceiling**: Even the state-of-the-art GPT-5 cannot reliably perform multi-platform memory tasks.
2. **Conflict resolution is the hardest**: When information is contradictory, models struggle to identify the latest/authoritative version (38.7% vs. 71.2%).
3. **RAG is insufficient**: Simple retrieval augmentation underperforms full-context input, owing to the complex associative patterns between cross-platform information.
4. **Code comprehension is a bottleneck**: Tasks requiring code understanding combined with memory integration are nearly intractable (29.8%).
5. **Large open- vs. closed-source gap**: The performance gap between LLaMA 70B and GPT-5 exceeds 20 percentage points.

## Highlights & Insights

- **Addresses an important gap**: The first memory evaluation benchmark targeting multi-platform enterprise environments, transcending the limitations of conversational memory benchmarks.
- **High ecological validity**: Designed around real-world software development workflows, with scenarios closely reflecting practice.
- **Multi-dimensional metric design**: The three-dimensional evaluation of Correctness + Efficiency + Redundancy is more comprehensive than simple QA accuracy.
- **Reveals true performance gaps**: The 60% ceiling figure provides clear direction for research on memory-augmented agents.

## Limitations & Future Work

1. **Limited scale**: The dataset is relatively small, and some scenarios are agent-synthesized, potentially introducing distributional bias.
2. **Incomplete platform coverage**: Only Slack/Linear/Git are simulated; real enterprises also use Notion, Confluence, Jira, etc.
3. **Static evaluation**: Evaluation of dynamic interaction (agents actively querying or confirming information) is absent.
4. **Workshop paper**: As a workshop contribution, the experimental scale and depth are relatively limited.
5. **Temporal validity**: GPT-5 results may change as model versions are updated.

## Related Work & Insights

- **LongBench** (Bai et al., 2024): Long-context evaluation, but focused on single documents.
- **MemoryBank** (Zhong et al., 2024): Conversational memory systems.
- **SWE-bench** (Jimenez et al., 2024): Software engineering agent benchmark; MEMTRACK is complementary along the memory dimension.
- **Insight**: MEMTRACK's multi-platform design could be extended to other enterprise scenarios (e.g., customer support, project management).

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|------------|-------|
| Novelty | 4 | First multi-platform memory evaluation benchmark |
| Technical Depth | 3 | Benchmark design; moderate technical contribution |
| Experimental Thoroughness | 3.5 | Multi-model + multi-backend comparison, but limited scale |
| Value | 4 | Important reference for agent memory research |
| Writing Quality | 3.5 | Workshop paper; concise but lacking some detail |
| **Overall** | **3.5** | A valuable workshop contribution |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs](../../ICLR2026/llm_evaluation/beyond_a_million_tokens_benchmarking_and_enhancing_long-term_memory_in_llms.md)
- [\[ICML 2026\] Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](../../ICML2026/llm_evaluation/multi2_hierarchical_multi-agent_decision-making_with_llm-based_agents_in_interac.md)
- [\[ACL 2025\] EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework](../../ACL2025/llm_evaluation/educationq_evaluating_llms_teaching_capabilities_through_multi-agent_dialogue_fr.md)
- [\[AAAI 2026\] BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction](../../AAAI2026/llm_evaluation/bcwildfire_a_long-term_multi-factor_dataset_and_deep_learning_benchmark_for_bore.md)
- [\[ACL 2025\] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions](../../ACL2025/llm_evaluation/from_tools_to_teammates_evaluating_llms_in_multi-session_coding_interactions.md)

</div>

<!-- RELATED:END -->
