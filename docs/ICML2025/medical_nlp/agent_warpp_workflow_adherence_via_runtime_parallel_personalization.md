---
title: >-
  [Paper Note] Agent WARPP: Workflow Adherence via Runtime Parallel Personalization
description: >-
  [ICML 2025][Medical LLM][Workflow Adherence] Proposes WARPP, a training-free multi-agent framework that dynamically prunes conditional branch workflows at runtime based on user attributes, executing them through a parallelized Personalizer agent in coordination with modular domain-specific agents, thereby improving tool call precision and parameter fidelity while reducing token consumption.
tags:
  - "ICML 2025"
  - "Medical LLM"
  - "Workflow Adherence"
  - "Multi-Agent Orchestration"
  - "Runtime Personalization"
  - "Task-Oriented Dialogue"
  - "Conditional Branch Pruning"
date: 2026-05-08
content_hash: 81378d885b6c78cd
---

# Agent WARPP: Workflow Adherence via Runtime Parallel Personalization

**Conference**: ICML 2025  
**arXiv**: [2507.19543](https://arxiv.org/abs/2507.19543)  
**Code**: Yes (mentioned as open-source in the paper, see the original text for links)  
**Area**: Dialogue Systems  
**Keywords**: Workflow Adherence, Multi-Agent Orchestration, Runtime Personalization, Task-Oriented Dialogue, Conditional Branch Pruning

## TL;DR

Proposes WARPP, a training-free multi-agent framework that dynamically prunes conditional branch workflows at runtime based on user attributes, executing them through a parallelized Personalizer agent in coordination with modular domain-specific agents, thereby improving tool call precision and parameter fidelity while reducing token consumption.

## Background & Motivation

Large Language Models (LLMs) are widely used in Task-Oriented Dialogue (TOD) systems, but underperform when handling long workflows containing complex conditional logic, external tool calls, and user-specific information. The core challenges include:

**Long-Context Reasoning Degradation**: As input length increases, LLM performance in reasoning and retrieval degrades; multi-hop reasoning remains a key bottleneck.

**Tool Call Hallucination**: LLMs frequently call unavailable tools, use tools unnecessarily, or execute them in the incorrect order.

**Limitations of Static Workflows**: Existing approaches (e.g., OctoTools, Creator), though capable of simplifying workflows, still treat workflow structures as static, failing to dynamically adjust to user context at runtime.

**Multi-Agent System Vulnerabilities**: Poor dialogue management, unclear task specifications, ineffective inter-agent communication, and premature termination.

Taking a medical appointment system as an example: booking a hospital visit may require retrieving patient profiles, screening insurance tiers and medical history, verifying referrals, checking provider availability, verifying identity, and assessing urgency. Each step can trigger different branches based on user-specific factors, generating dozens of conditional decision points that quickly exceed the processing limits of standard LLMs.

## Method

### Overall Architecture

WARPP (Workflow Adherence via Runtime Parallel Personalization) is built on the OpenAI Agents SDK and consists of four core agents:

- **Orchestrator Agent**: Initiates the dialogue, identifies user intent, and dynamically retrieves the corresponding workflow and toolset.
- **Authenticator Agent**: Simulates authentication processes (e.g., two-factor authentication) and runs in parallel with the Personalizer agent.
- **Personalizer Agent**: Runs in parallel, applying a three-stage pruning process to the complete workflow based on user attributes.
- **Fulfillment Agent**: Executes the final task according to the pruned workflow and filtered toolset.

The execution flow is as follows: Orchestrator identifies intent $\rightarrow$ launches Authenticator and Personalizer in parallel $\rightarrow$ once authentication and personalization are complete $\rightarrow$ Fulfillment executes the pruned workflow.

### Key Designs

#### 1. Runtime Workflow Pruning (Three-Stage Transformation)

Upon intent identification, the Personalizer agent immediately executes all information-gathering tools to retrieve user attributes, then performs a three-stage transformation on the full workflow:

- **Static Pruning**: Removes branches and tool calls incompatible with user attributes, and inlines values that can be directly resolved from user data.
- **Fidelity Preservation**: Retains all outcome branches (success/failure, user yes/no) surrounding each preserved tool call to ensure dialogue robustness.
- **Cleanup and Formatting**: Merges descriptive steps and renumbers the instructions.

In addition to the pruned workflow, the Personalizer returns a list of filtered tools required for execution, containing only the tools preserved after pruning.

#### 2. Parallelized Architecture

The key innovation of WARPP lies in the **parallel execution** of the Personalizer and the Authenticator:

- Authentication processes typically involve waiting periods (e.g., SMS OTPs). This latency is fully leveraged to perform workflow personalization.
- In high-load or high-latency scenarios, any remaining personalization steps are completed during the transition to the Fulfillment agent.
- This design ensures that personalization does not introduce significant latency overhead.

#### 3. Reduced Inference Complexity

For a complete workflow $W$ consisting of $T$ tokens, where each decision point averages $t$ tokens and has a maximum of $b$ branches:

- Number of decision points: $n \approx T/t$
- Unpruned worst-case path complexity: $b^n \approx b^{T/t}$ (exponential)
- WARPP pruning requires only a single pass over the workflow: time complexity $O(T)$
- Tool filtering complexity: $O(m)$, where $m$ is the total number of tools
- Total personalization complexity: $O(T+m)$

By pre-selecting valid paths, the exponential search space is compressed to linear complexity, significantly improving inference accuracy.

#### 4. Dynamic Fulfillment Agent Configuration

A Fulfillment Agent is dynamically configured for each intent, avoiding manual duplication. In the personalized setting, it receives only the pruned workflow and the filtered toolset; in the non-personalized setting, it receives the complete workflow and all tools.

### Loss & Training

WARPP is a **completely training-free** framework. It involves no gradient updates, fine-tuning, or reinforcement learning; all improvements stem from runtime workflow pruning and multi-agent orchestration. Evaluation utilizes the following metrics:

- **Trajectory Accuracy**: Exact Match (EM), Ordered/Unordered Agent Match, Longest Common Subsequence (LCS) of tool sequences.
- **Tool Usage**: Precision/Recall/F1, parameter match percentage.
- **Interaction Quality**: Latency.
- **Instruction Quality**: Relevance and completeness evaluated by an LLM-as-a-judge (scaled 1-5).

## Key Experimental Results

### Experimental Setup

- **Three Domains**: Banking (Simple, $\le 5$ tools), Flights (Medium, $\le 10$ tools), Hospital (Complex, $> 15$ tools).
- **Five Intents**: updateAddress, withdrawRetirementFunds, bookFlight, cancelFlight, processPayment.
- **Three Models**: GPT-4o, Claude Sonnet 3.5, Llama 3.
- **50 Synthetic Users per Intent**.
- **Baselines**: ReAct single-agent, WARPP without personalization (No Per.), Full WARPP.

### Main Results

| Intent | Strategy | Exact Match | LCS Tools | Tool F1 | Parameter Match (%) |
|------|------|---------|-----------|---------|------------|
| Update Address | ReAct | 0.73 | 95.98 | 97.43 | 98.32 |
| Update Address | No Per. | 0.89 | 99.33 | 98.59 | 99.12 |
| Update Address | **WARPP** | **0.97** | 98.56 | **99.00** | 98.04 |
| Book Flight | ReAct | 0.63 | 96.51 | 96.30 | 97.40 |
| Book Flight | No Per. | 0.89 | 99.35 | 99.11 | 99.38 |
| Book Flight | **WARPP** | **0.96** | 99.19 | **99.47** | 99.10 |
| Process Payment | ReAct | 0.16 | 82.93 | 87.95 | 76.19 |
| Process Payment | No Per. | 0.16 | 93.04 | 93.52 | 86.66 |
| Process Payment | **WARPP** | **0.56** | **94.07** | **95.46** | **92.04** |

**Key Observation**: The higher the task complexity, the more significant the advantage of WARPP. On the most complex intent, Process Payment, WARPP's exact match increases from 0.16 (ReAct) to 0.56, and parameter match increases from 76.19% to 92.04%.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| ReAct Single-Agent | Process Payment Exact Match: 0.16 | Single-agent performance degrades severely on complex workflows |
| WARPP (No Per.) | Process Payment Exact Match: 0.16 | Multi-agent orchestration alone is insufficient for complex tasks |
| Full WARPP | Process Payment Exact Match: 0.56 | Personalized pruning is the primary source of performance gains |
| Token Usage (GPT, Process Payment) | ReAct: 5437 $\rightarrow$ WARPP: 1855 | Reduces token consumption by approximately 66% |
| Token Usage (Sonnet, Process Payment) | ReAct: 8439 $\rightarrow$ WARPP: 2863 | Reduces token consumption by approximately 66% |
| Pruned Workflow Quality (GPT-4o) | Relevance: 4.55/5, Completeness: 4.59/5 | Highly qualified pruned workflows generated by the Personalizer |
| Pruned Workflow Quality (Llama-3) | Relevance: 4.49/5, Completeness: 4.52/5 | Weaker models exhibit higher variance but remain viable |

### Key Findings

1. **The Higher the Complexity, the Greater the Gain**: There are marginal differences among the three strategies on simple tasks, but WARPP yields the most significant improvement on the most complex Process Payment task.
2. **Model Agnosticism**: Effective across GPT-4o, Claude Sonnet 3.5, and Llama 3; even the strong Sonnet model benefits substantially from WARPP on complex tasks.
3. **Token Efficiency**: WARPP achieves the lowest token consumption across all intents and models, potentially halving token usage on complex tasks.
4. **Greater Benefit for Weaker Models**: Llama-3 and GPT-4o, which have lower baseline capabilities, benefit the most from orchestration on simpler tasks.
5. **Limitations of Llama**: Under the personalized configuration of Cancel Flight, Llama performs worse than the non-personalized setup, as it occasionally fails to invoke tools and instead only describes the intended actions.

## Highlights & Insights

1. **Elegant Parallel Design**: Parallelizing personalization and authentication is a clever engineering choice that fully utilizes authentication wait times, resulting in zero additional latency.
2. **From Exponential to Linear**: Compressing the search space from $O(b^{T/t})$ to $O(T+m)$ via a single-pass traversal provides a clear theoretical foundation.
3. **Training-Free**: A completely training-free framework that acts as a plug-and-play solution for any LLM, lowering deployment costs.
4. **Rational Three-Stage Pruning**: Static Pruning $\rightarrow$ Fidelity Preservation $\rightarrow$ Cleanup and Formatting; this achieves aggressive pruning while preserving dialogue robustness.
5. **Comprehensive Evaluation Suite**: Evaluates trajectory, tools, parameters, latency, and instruction quality simultaneously, utilizing both LLM-as-a-judge and human spot-checking.

## Limitations & Future Work

1. **Suboptimal Pruning Quality**: Analysis indicates that optimal workflow steps are occasionally omitted, particularly with weaker models.
2. **Synthetic Data Evaluation**: Evaluated only on synthetic data and simulated users without validation in real-world user scenarios.
3. **Limited Domains**: Evaluated on only three domains and five intents, requiring broader validation for generalizability.
4. **Dependency on Personalizer Capabilities**: The quality of pruning is highly dependent on the capabilities of the LLM chosen for the Personalizer.
5. **Unexplored Decomposition-Based Personalization**: Decomposing personalization into multiple calls or ensemble methods could potentially further improve pruning fidelity.
6. **Privacy and Fairness Risks**: Pruning workflows based on user privileges and attributes may introduce algorithmic biases.

## Related Work & Insights

- **ReAct (Yao et al., 2023)**: The primary baseline of this work, utilizing the single-agent reason-and-act paradigm.
- **OctoTools, Creator**: Workflow simplification methods that nonetheless treat workflow execution structures statically.
- **AFLOW (Zhang et al., 2024a)**: Models workflows as directed graphs and optimizes them via MCTS, though the optimization is conducted offline.
- **GPTSwarm**: Optimizes multi-agent structures and single-agent routing with reinforcement learning.
- **Insights**: The concept of runtime dynamic pruning can be generalized to other scenarios requiring conditional execution, such as code generation, automated system testing, and robotic task planning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The approach of runtime parallel personalized workflow pruning is novel, though multi-agent orchestration itself is not a new concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated across three models and five intents with comprehensive metrics, but restricted to synthetic data and limited domains.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, formally described algorithms, with solid motivational arguments.
- Value: ⭐⭐⭐⭐ — High practical value, training-free and easy to deploy, though strictly speaking, it represents a systems/engineering contribution rather than a major methodological breakthrough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis](../../ICLR2026/medical_nlp/resp-agent_an_agent-based_system_for_multimodal_respiratory_sound_generation_and.md)
- [\[ACL 2025\] LLMs Can Simulate Standardized Patients via Agent Coevolution](../../ACL2025/medical_nlp/evopatient_standardized_patient.md)
- [\[ACL 2026\] CT-Flow: Orchestrating CT Interpretation Workflow with Model Context Protocol Servers](../../ACL2026/medical_nlp/ct-flow_orchestrating_ct_interpretation_workflow_with_model_context_protocol_ser.md)
- [\[AAAI 2026\] MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains](../../AAAI2026/medical_nlp/mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso.md)
- [\[ACL 2026\] IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages](../../ACL2026/medical_nlp/indicmeddialog_a_parallel_multi-turn_medical_dialogue_dataset_for_accessible_hea.md)

</div>

<!-- RELATED:END -->
