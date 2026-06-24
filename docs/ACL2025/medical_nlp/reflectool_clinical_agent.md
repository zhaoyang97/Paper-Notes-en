---
title: >-
  [Paper Note] ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents
description: >-
  [ACL 2025][Medical LLM][Tool-Augmented Agent] ReflecTool proposes a reflection-aware tool-augmented clinical Agent framework. By accumulating successful trajectories and tool-level experience in the optimization stage, and retrieving similar cases while refining tool usage with a validator in the inference stage, it outperforms pure LLMs by 10+ points and existing Agent methods by 3 points on ClinicalAgent Bench across 18 tasks.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Tool-Augmented Agent"
  - "Clinical Agent"
  - "Reflective Learning"
  - "Long-term Memory"
  - "Medical AI"
date: 2026-05-08
content_hash: 20752e177fb3d2b1
---

# ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents

**Conference**: ACL 2025  
**arXiv**: [2410.17657](https://arxiv.org/abs/2410.17657)  
**Code**: [https://github.com/BlueZeros/ReflecTool](https://github.com/BlueZeros/ReflecTool)  
**Area**: Medical NLP  
**Keywords**: Tool-Augmented Agent, Clinical Agent, Reflective Learning, Long-term Memory, Medical AI  

## TL;DR
ReflecTool proposes a reflection-aware tool-augmented clinical Agent framework. By accumulating successful trajectories and tool-level experience in the optimization stage, and retrieving similar cases while refining tool usage with a validator in the inference stage, it outperforms pure LLMs by 10+ points and existing Agent methods by 3 points on ClinicalAgent Bench across 18 tasks.

## Background & Motivation
**Background**: LLMs show promising potential in the medical domain but are limited to text interactions, failing to handle multimodal information (medical images, EHRs, etc.) in clinical settings.

**Limitations of Prior Work**: Existing clinical Agents (e.g., EHRAgent, MMedAgent) target single scenarios with limited tool types, lacking generalizability across different clinical settings.

**Key Challenge**: Clinical environments require Agents to possess multidimensional capabilities including knowledge reasoning, multimodal understanding, numerical analysis, data understanding, and reliability, yet a unified evaluation benchmark and a general framework are lacking.

**Goal**: (1) Build a comprehensive clinical Agent evaluation benchmark. (2) Design a general Agent framework capable of learning and effectively using domain-specific tools.

**Key Insight**: Accumulate tool usage experiences by self-reflecting on successful/failed trajectories, and store successful cases in a long-term memory.

**Core Idea**: Enable the Agent to "practice" tool usage and accumulate experience during an optimization stage, and retrieve similar successful cases along with tool-level experiences to guide decision-making during the inference stage.

## Method

### Overall Architecture
The input is a clinical task (a question plus input data in various formats) $\rightarrow$ the toolbox contains 15 clinical tools $\rightarrow$ a two-stage framework: an optimization stage to accumulate experience through trial and error on a small-scale training set $\rightarrow$ an inference stage to retrieve similar cases and apply tool-wise verification $\rightarrow$ and finally output the answer. The process is based on ReAct-style multi-step reasoning.

### Key Designs
1. **ClinicalAgent Bench (CAB)**:

    - Function: Provides a comprehensive clinical Agent evaluation framework.
    - Mechanism: Covers 18 tasks across 5 dimensions (knowledge reasoning, multimodal, numerical analysis, data understanding, reliability), supported by 15 clinical tools.
    - Design Motivation: Existing medical Agent benchmarks only cover single scenarios, making it impossible to comprehensively evaluate clinical Agent capabilities.

2. **Optimization Stage**:

    - Function: Accumulates tool usage experience and successful cases on a small-scale sample dataset.
    - Mechanism: The Agent first attempts to solve the problem to generate a trajectory $\mathcal{C}_1$, self-reflects against the ground truth to generate improvement suggestions $\mathcal{S}$, and recreates a trajectory $\mathcal{C}_2$ accordingly. If successful, the trajectory is saved to the long-term memory $\mathcal{M}$, and tool-level experiences $\mathcal{E}$ are extracted by comparing both trajectories.
    - Design Motivation: Directly using tools yields poor results; through "trial-error-reflection-accumulation", the Agent learns the correct usage of domain-specific tools.

3. **Inference Stage with Tool-wise Reflection**:

    - Function: Utilizes long-term memory and tool-level experience to guide reasoning.
    - Mechanism: (1) BM25 retrieves similar successful cases as few-shot demonstrations. (2) Two verification modes: **Iterative Refinement** (iteratively refines actions until stable or reaching limits) and **Candidate Selection** (samples multiple candidate actions and allows a validator to select the optimal one).
    - Design Motivation: Iterative Refinement is suitable for weaker models (gradual improvement), while Candidate Selection is suitable for stronger models (selecting the best among multiple plans), offering complementary adaptability.

### Loss & Training
No fine-tuning of model parameters is required. The optimization stage requires only a few annotated samples (~200 in the experiments) to generate experiences via LLM self-reflection and store them in an external memory.

## Key Experimental Results

### Main Results

| Model / Method | Metric (Avg) | Comparison | Gain |
|------------|-----------|------|------|
| Qwen2-7B (Pure LLM) | 38.01 | - | - |
| ReflecTool (Qwen2-7B, IR) | 49.37 | vs Pure LLM | +11.36 |
| Reflexion (Qwen2-72B) | 56.37 | Strongest Baseline | - |
| ReflecTool (Qwen2-72B, CS) | 59.66 | vs Reflexion | +3.29 |

### Ablation Study

| Configuration | Avg | Description |
|------|-----|------|
| W/o Memory W/o Reflection (ReAct) | 52.37 | Baseline |
| + Long-term Memory | 54.86 | +2.49 |
| + Long-term Memory + Tool-wise Experience (IR) | 57.01 | +4.64 |
| + Long-term Memory + Tool-wise Experience (CS) | 60.28 | +7.91 |

### Key Findings
- Long-term memory is the most critical component; removing it drops Refinement by 4 points and Selection by 7 points.
- Candidate Selection performs better with stronger models, while Iterative Refinement is more effective with weaker models.
- Performance improves with more optimization steps, but gains show diminishing marginal returns (stabilizing around ~800 steps).
- A verification count of $n=2$ is usually optimal; too many verifications can be harmful.

## Highlights & Insights
- The idea of extracting "tool-level experience" from trajectory comparisons is ingenious and can be transferred to Agents in other domains (e.g., coding, finance).
- The complementary findings of the two verification methods (iterative refinement vs. candidate selection) provide practical value: strategies can be automatically selected based on model capability.
- The CAB benchmark is comprehensively designed, covering 18 tasks across 5 dimensions, filling the gap in comprehensive evaluation for clinical Agents.

## Limitations & Future Work
- The optimization stage requires ground-truth answers, which limits applicability in unlabeled scenarios.
- The toolbox is predefined, lacking the capability to dynamically discover or create new tools.
- Evaluated only on the Qwen2 series; generalizability to other models has not been fully verified.
- The safety and interpretability of the Agent in clinical scenarios are not discussed in depth.

## Related Work & Insights
- **vs Reflexion**: Both utilize reflection, but Reflexion reflects at the behavioral level, whereas ReflecTool refines experience accumulation and verification down to the tool level.
- **vs EHRAgent/MMedAgent**: These are designed for single scenarios, while ReflecTool is a general-purpose clinical Agent.
- **vs CRITIC**: CRITIC utilizes self-verification to improve output, while ReflecTool performs verification at each action step based on tool-wise experiences.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of tool-level reflective experience + double-verification methods is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 18 tasks across 5 dimensions + multiple baselines + ablation + detailed analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagram, standard and formal methodology description.
- Value: ⭐⭐⭐⭐ Both the CAB benchmark and the tool-level reflection framework have practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CliniDial: A Naturally Occurring Multimodal Dialogue Dataset for Team Reflection in Action During Clinical Operation](clinidial_a_naturally_occurring_multimodal_dialogue_dataset_for_team_reflection_.md)
- [\[ICLR 2026\] From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents](../../ICLR2026/medical_nlp/from_conversation_to_query_execution_benchmarking_user_and_tool_interactions_for.md)
- [\[ACL 2025\] Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](omni_rag_medical.md)
- [\[ACL 2025\] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)
- [\[ACL 2025\] A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment](a_modular_approach_for_clinical_slms_driven_by_synthetic_data_with_pre-instructi.md)

</div>

<!-- RELATED:END -->
