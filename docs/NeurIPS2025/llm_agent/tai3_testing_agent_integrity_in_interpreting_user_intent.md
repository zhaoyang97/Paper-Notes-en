---
title: >-
  [Paper Note] TAI3: Testing Agent Integrity in Interpreting User Intent
description: >-
  [NeurIPS 2025][LLM Agent][Agent Testing] This paper proposes TAI3, an API-centric stress-testing framework for LLM agent intent integrity. It organizes the natural language input space into a structured test grid via Sem…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "Agent Testing"
  - "Intent Integrity"
  - "Equivalence Class Partitioning"
  - "Stress Testing"
  - "API Calls"
date: 2026-05-08
content_hash: 5274bb8ca2f1c88d
---

# TAI3: Testing Agent Integrity in Interpreting User Intent

**Conference**: NeurIPS 2025
**arXiv**: [2506.07524](https://arxiv.org/abs/2506.07524)  
**Code**: None  
**Area**: LLM Agent / AI Safety
**Keywords**: Agent Testing, Intent Integrity, Equivalence Class Partitioning, Stress Testing, API Calls

## TL;DR

This paper proposes TAI3, an API-centric stress-testing framework for LLM agent intent integrity. It organizes the natural language input space into a structured test grid via Semantic Partitioning, and leverages Intent-Preserving Mutation and Strategy Memory to efficiently expose intent misinterpretation errors when agents execute user tasks.

## Background & Motivation

### 1. State of the Field

LLM agents are being widely deployed in software development, e-commerce, smart home, and other domains, invoking external APIs through natural language instructions to complete tasks. These agents translate high-level user intent into concrete API call sequences, but the ambiguity of natural language means agent behavior may deviate from the user's true intent.

### 2. Limitations of Prior Work

- **Inadequate fixed benchmarks**: Existing LLM agent safety benchmarks (e.g., AgentSafetyBench, ToolEmu) rely on fixed test cases and cannot keep pace with the rapid evolution of toolkits.
- **Misaligned adversarial testing**: A large body of work focuses on jailbreaking and prompt injection, rather than ensuring agents robustly execute benign tasks under normal use.
- **Failure of classical testing**: Traditional software testing assumes structured input interfaces and cannot handle the openness and ambiguity of natural language.
- **Unquantifiable coverage**: There is no metric analogous to code coverage to measure how much of the agent's behavior space has been tested.

### 3. Root Cause

API specifications are precise and formal, whereas user natural language instructions are ambiguous and variable. This gap means agents may misinterpret intent on seemingly reasonable inputs, yet existing testing methods lack systematic means to uncover such latent errors.

### 4. Paper Goals

To design a systematic Intent Integrity testing framework for LLM agents that can: (1) quantitatively verify agent intent fidelity; (2) generate realistic tasks as test cases; and (3) efficiently discover errors within a reasonable query budget.

### 5. Starting Point

Core insight: agent behavior (and its potential vulnerabilities) can be systematically characterized through the structure of the underlying APIs. Drawing on the classical black-box testing technique of equivalence class partitioning, the paper partitions each API parameter's value domain by intent category, yielding a finite and interpretable test grid.

### 6. Core Idea

A structured test space is constructed via semantic partitioning of API parameters; intent-preserving mutation and strategy memory are then employed to efficiently search for boundary cases where agent intent understanding fails.

## Method

### Overall Architecture

TAI3 consists of two stages:

- **Stage 1 — Semantic Partitioning**: For each API parameter, equivalence class partitioning is applied across three intent categories — VALID, INVALID, and UNDERSPEC — producing a parameter–partition table in which each cell yields one seed task.
- **Stage 2 — Intent-Preserving Mutation**: Seed tasks are iteratively mutated in an intent-preserving manner; a lightweight surrogate model ranks the candidates, and those most likely to trigger errors are selected for agent testing.

### Key Designs

#### 1. Semantic Partitioning

- **Function**: Partitions the value domain of each API parameter into three intent categories — VALID (legal values), INVALID (illegal values), and UNDERSPEC (insufficient information) — with further equivalence class partitioning within each category.
- **Mechanism**: For parameter $p$ with domain $\mathcal{D}_p$, the domain is decomposed per category $c \in \{VA, IV, US\}$ as $\mathcal{D}_p^c = \mathcal{E}_{p,c}^1 \cup \cdots \cup \mathcal{E}_{p,c}^{m(p,c)}$, where each equivalence class represents a semantically distinct input modality.
- **Example**: The `start_time` parameter under VALID may be partitioned into "standard date format" and "relative time expression"; under INVALID, into "non-existent date" and "unsupported functionality."
- **Design Motivation**: APIs are formally defined, enabling precise and complete specification of the agent behavior space — analogous to the concept of code coverage.

#### 2. Seed Task Generation

- **Function**: For each cell $(p, c, i)$ in the partition table, an LLM generates a realistic natural language user instruction.
- **Constraints**: Each instruction must instantiate a representative value from the corresponding equivalence class and be designed to elicit the behavior associated with category $c$.
- **Guarantee**: Every partition cell has a corresponding seed task, ensuring complete coverage of the semantic input space.

#### 3. Intent-Preserving Mutation

- **Function**: Starting from a seed task, variants are iteratively generated that preserve the original intent while increasing the probability of agent error.
- **Intent consistency check**: For each candidate variant $u'$, an LLM verifies whether it remains consistent with the original intent $\mathcal{I}(u)$ (verification is easier than inference).
- **Error probability estimation**: A small language model (phi4-mini) computes the error likelihood of a mutated task: $\sum_i \log P(\mathcal{I}(u)_i | u' \cdot \mathcal{I}(u)_{<i}; \theta)$. A lower score indicates greater difficulty in recovering the original intent from the mutated task, and thus a higher probability of triggering agent error.
- **Design Motivation**: Actually running the agent is costly (5–26 seconds per action); using a lightweight surrogate model for ranking substantially reduces the number of queries required.

#### 4. Evergreen Strategy Memory

- **Function**: Records mutation strategy patterns that have successfully triggered errors, indexed by parameter data type and intent category.
- **Strategy examples**: "Hesitating between two enumerated options," "splitting an amount across two sentences and introducing a mathematical expression."
- **Reuse mechanism**: Upon receiving a new seed task, relevant strategies are retrieved, re-ranked by an LLM, and the Top-3 are used to guide mutation.
- **Design Motivation**: Analogous to a human tester becoming more effective through accumulated experience, the framework learns from historically successful patterns.

### Loss & Training

The framework involves no model training. The core optimization objective is to maximize EESR (Error-Exposing Success Rate) — the proportion of semantic partitions in which at least one agent error is discovered — within a fixed query budget. The efficiency metric is AQFF (Average Queries to First Failure), i.e., the average number of queries required to trigger the first failure.

## Key Experimental Results

### Main Results

Dataset: 80 APIs (233 parameters) spanning five domains — Finance, Healthcare, Smart Home, Logistics, and Office.

| Domain | Target Model | VALID EESR (SelfRef→TAI3) | INVALID EESR (SelfRef→TAI3) | UNDERSPEC EESR (SelfRef→TAI3) |
|--------|-------------|--------------------------|----------------------------|-------------------------------|
| Finance | Llama-3.1-8B | 65.0→80.5 (+15.5) | 78.0→85.4 (+7.4) | 58.5→73.2 (+14.7) |
| Finance | GPT-4o-mini | 41.5→61.0 (+19.5) | 65.9→73.2 (+7.3) | 61.0→65.9 (+4.9) |
| Healthcare | Llama-3.1-8B | 66.0→70.2 (+4.2) | 51.1→55.3 (+4.2) | 57.4→61.7 (+4.3) |
| Smart Home | GPT-4o-mini | 63.0→72.2 (+9.2) | 57.4→63.0 (+5.6) | 61.1→63.0 (+1.9) |
| Office | Llama-3.1-8B | 60.0→64.0 (+4.0) | 54.0→58.0 (+4.0) | 65.7→82.0 (+16.3) |

TAI3 outperforms the SelfRef baseline across **all domains, all intent categories, and all target models**.

### Ablation Study

| Component | Effect |
|-----------|--------|
| Error likelihood ranking vs. random selection | Ranking triggers more errors at all values of $k$, validating the surrogate model. |
| Error likelihood ranking vs. Select Last 5 | Ranking significantly outperforms selecting the last 5 candidates after reflection. |
| Strategy Memory | Effective mutation patterns transfer across APIs and domains. |
| Semantic partitioning coverage | Test cases from AgentSafetyBench and ToolEmu cover only 8.3%–50% of TAI3 partitions, leaving large portions untested. |

### Generalization

Weaker testing models (Llama-3.1-8B, Qwen3-30B-A3B) can effectively expose errors in stronger target models (Claude-3.5-Haiku, Gemini-2.5-Pro, GPT-o3-mini). The weak-to-strong gap is small in the Top-1 setting; in the Top-5 setting, GPT-4o-mini exhibits the strongest ranking capability. Open-source target models (Llama-3.3-70B, DeepSeek-R1-70B) prove more vulnerable than closed-source ones.

### Key Findings

1. **EESR improvement**: TAI3 achieves a maximum gain of 19.5 percentage points in the Finance+VALID setting (with GPT-4o-mini as the target).
2. **Query efficiency**: AQFF decreases by up to 12% in the UNDERSPEC category; mutation ranking significantly reduces search overhead.
3. **Inadequacy of existing benchmarks**: AgentSafetyBench contains almost no INVALID test cases; ToolEmu completely lacks INVALID inputs; coverage rates for both are generally below 50%.
4. **Weak-to-strong testing**: Weaker testing LLMs can successfully uncover intent integrity vulnerabilities in stronger models, indicating that such vulnerabilities are intrinsic to the agent rather than a limitation of the testing model's capability.

## Highlights & Insights

1. **Bridging API formalism and natural language ambiguity**: The paper is the first to apply the classical equivalence class partitioning technique from software testing to LLM agent evaluation, providing a methodology for quantifying test coverage.
2. **Three-category intent integrity taxonomy**: The VALID/INVALID/UNDERSPEC classification is concise yet comprehensive, offering a clear framework for measuring agent robustness.
3. **Lightweight surrogate model ranking**: A small model (phi4-mini) is used to estimate error probability, avoiding costly agent executions and realizing a tiered "cheap probing, expensive verification" strategy.
4. **Transferability of strategy memory**: Strategies indexed by data type and intent category are reusable across APIs and domains, analogous to the accumulation of human testing expertise.
5. **Exposing benchmark shortcomings**: The paper quantitatively demonstrates that AgentSafetyBench and ToolEmu exhibit severely insufficient coverage along the intent integrity dimension.

## Limitations & Future Work

1. **Trajectory observability assumption**: TAI3 requires access to the agent's API call trajectory and is not applicable to commercial agents that expose only high-level outputs (e.g., web interactions).
2. **Scope limited to the API call layer**: The framework does not address higher-level safety concerns such as policy violations, privacy leakage, or harmful content; its scope is confined to intent understanding at the API parameter level.
3. **Reliance on LLM for intent consistency checking**: The judgment of intent preservation itself depends on an LLM, which may introduce noise.
4. **Partition granularity**: The quality of equivalence class partitioning depends on the LLM's semantic analysis and may vary substantially across different APIs.
5. **Multi-step interaction**: The current approach primarily tests single-turn API calls; coverage of multi-step planning scenarios remains to be extended.

## Related Work & Insights

- **Classical software testing methods**: Equivalence class partitioning is transplanted from traditional black-box testing to natural language input spaces, demonstrating the renewed relevance of classical techniques.
- **NLP robustness testing** (CheckList, TextAttack): These focus on model-level adversarial perturbations; TAI3 extends the scope to agent-level intent integrity.
- **ToolFuzz / PDoctor**: ToolFuzz targets bugs between tool documentation and implementation; PDoctor checks constraint adherence in high-level planning; TAI3 focuses on alignment between low-level actions and user intent.
- **Agent Safety Benchmarks** (AgentSafetyBench, ToolEmu): TAI3's partition analysis quantitatively reveals the coverage deficiencies of these benchmarks.
- **Insight**: The combination of strategy memory and predictive ranking is generalizable to efficient automated testing of any LLM-based system.

## Rating

- Novelty: ⭐⭐⭐⭐ — Applying equivalence class partitioning to agent intent integrity testing is a novel idea; the three-category intent taxonomy is clear and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across 80 APIs, 5 domains, multiple target models, generalization experiments, and ablations; multi-step interaction scenarios are not covered.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear; formalization and intuition are well balanced; figures are excellent (Figures 1–3 in particular are highly convincing).
- Value: ⭐⭐⭐⭐ — Fills the gap in systematic testing for agent intent integrity and provides practical guidance for quality assurance in agent deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Grounding Agent Memory in Contextual Intent](../../ACL2026/llm_agent/grounding_agent_memory_in_contextual_intent.md)
- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](mat-agent_adaptive_multi-agent_training_optimization.md)
- [\[ICLR 2026\] Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](../../ICLR2026/llm_agent/judge_reliability_harness_stress_testing_the_reliability_of_llm_judges.md)
- [\[ICML 2026\] Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History](../../ICML2026/llm_agent/persona2web_benchmarking_personalized_web_agents_for_contextual_reasoning_with_u.md)
- [\[NeurIPS 2025\] Group-in-Group Policy Optimization for LLM Agent Training](groupingroup_policy_optimization_for_llm_agent_training.md)

</div>

<!-- RELATED:END -->
