---
title: >-
  [Paper Note] NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents
description: >-
  [ICML 2026][Code Intelligence][Automated Optimization Modeling] NEMO treats Autonomous Coding Agents (ACA) as "first-class abstractions" on par with LLMs. It allows independently generated simulators and optimizers to cr…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "Automated Optimization Modeling"
  - "Autonomous Coding Agents (ACA)"
  - "Execution-Aware Verification"
  - "Simulator-Optimizer Closed Loop"
  - "MBR Decoding"
date: 2026-05-08
content_hash: 7fcdc8385f7c62ee
---

# NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents

**Conference**: ICML 2026  
**arXiv**: [2601.21372](https://arxiv.org/abs/2601.21372)  
**Code**: Data/intermediate products released on [HuggingFace](https://huggingface.co/datasets/AnoushkaVyas/nemo-icml2026) (System code not yet open-sourced)  
**Area**: code intelligence / LLM Agent / Operations Research  
**Keywords**: Automated Optimization Modeling, Autonomous Coding Agents (ACA), Execution-Aware Verification, Simulator-Optimizer Closed Loop, MBR Decoding

## TL;DR
NEMO treats Autonomous Coding Agents (ACA) as "first-class abstractions" on par with LLMs. It allows independently generated simulators and optimizers to cross-validate each other via execution results in a shared sandbox, supplemented by diverse memory retrieval and MBR/self-consistency decoding. It achieves SOTA on 8 out of 9 optimization modeling benchmarks, leading by up to 28 percentage points.

## Background & Motivation

**Background**: Automatically translating natural language decision problems (resource scheduling, combinatorial optimization, production planning, etc.) into executable mathematical optimization code currently follows two main routes: **Training-based** (ORLM/LLMOPT/SIRL/OptMATH, etc., using specialized data to fine-tune LLMs) and **Agent-based** (Chain-of-Experts/OptiMUS/OR-LLM-Agent/OptimAI using multi-role LLM collaboration + critic reflection).

**Limitations of Prior Work**: Training-based methods are costly and have poor domain transfer. Agent-based methods model interaction as "structured text message-passing," where a coder posts code for a critic to describe errors, and the coder then revises. Issues include: 1) Generated code often fails syntactically or during execution; 2) Critics only analyze text and cannot truly "run" the code to check if constraints are met; 3) Sharing code between different agents requires schema agreements, making expansion difficult.

**Key Challenge**: Optimization modeling inherently possesses an asymmetry—**writing a declarative solver is difficult** (requires correct translation of constraints, objectives, and solver APIs), but **writing an imperative simulator is relatively simple** (following the sequence described in the problem). Paper tests on OptiBench show LLM single-shot Pass@1 for simulators is 97%, while optimizers are only 87%. Existing Agent systems fail to exploit this "verification is easier than solving" asymmetry.

**Goal**: Construct an execution-aware Agent system where independently generated simulators verify optimizer outputs, while systematically handling the stochasticity of ACAs.

**Key Insight**: Instead of having multiple LLM agents exchange information at the text level, multiple ACAs should **share a persistent sandbox** and interact through "runnable artifacts"—where code can be imported, executed, and verified via unit tests. This is the **Workspace-State paradigm** (as opposed to the traditional message-passing paradigm).

**Core Idea**: Treat ACA as a first-class abstraction equivalent to LLM calls → Use simulators (simple task) as execution-based verification for optimizers (hard task) → Use MBR / diversity retrieval / self-consistency to suppress ACA non-determinism.

## Method

### Overall Architecture
The input is a natural language description $D$ of a decision problem, and the output is an executable optimization code package that solves for the optimal solution and passes simulator cross-validation. The system consists of a pipeline with four major modules:

```mermaid
graph TD
    D[D: Natural Language] --> E[Decision Process Extractor: Reasoning LLM + Component-wise MBR]
    E --> P[Structured Representation P* = {variables, states, transitions, objectives, constraints, params}]
    P --> SR[Solver Recommender]
    SR --> SO[Sorted Solver Candidates SO*]
    P --> ACA1[Simulator Generation: ACA #1, Imperative Python + pytest]
    P --> ACA2[Optimizer Generation: ACA #2, Declarative Solver Code + Self-consistency]
    ACA1 --> S[S: x -> feasible, F_sim]
    ACA2 --> X[x*, F_opt]
    S --> V[Asymmetric Validation Loop]
    X --> V
    V -- Failure --> Debug[Structured Error Report fed back to Optimizer ACA]
    Debug --> ACA2
    V -- Success --> Out[Final Code Package]
```

All ACAs (instantiated with OpenHands + Claude 3.7 Sonnet in the paper, though decoupled from specific ACAs) run in the same sandbox and can directly import modules generated by others without schema negotiation.

### Key Designs

1.  **ACA as First-Class Abstraction + Workspace-State Paradigm**:
    *   **Function**: Treats "calling a remote agent capable of writing, running, and self-debugging code in a sandbox" as a system-level primitive equal to "calling an LLM."
    *   **Mechanism**: Each ACA receives task instructions (NL instructions + structured problem description + references to existing artifacts) and returns executable code, execution trajectories, and results. Multiple ACAs share a persistent sandbox, collaborating by importing files, running unit tests, and reading outputs. Few-shot memory is also placed in the sandbox as **runnable Python files**. Section 2.1 summarizes this as three capabilities: ACA-as-Architect (syntax correctness ensured by the ACA itself), Shared Workspace (direct imports across roles), and Executable Memory (runnable exemplars).
    *   **Design Motivation**: Addresses three major problems of message-passing: code executability, cross-component schema negotiation costs, and low utilization of few-shots under prompt length constraints. It shifts "can the code run" from an implicit developer concern to a strong system-level guarantee.

2.  **Asymmetric Simulator-Optimizer Validation Loop**:
    *   **Function**: Uses an independently generated, unit-tested simulator as a ground-truth-free "referee" to check the feasibility and objective value of the optimizer solution $x^*$.
    *   **Mechanism**: After extracting $\mathcal{P}^*$, two paths run in parallel: ACA #1 writes simulator $\mathcal{S}: \mathbb{R}^{|X|} \to \{0,1\} \times (\mathbb{R} \cup \{\infty\})$ (imperative Python + pytest derived from $D$ and $\mathcal{P}^*$), and ACA #2 writes the optimizer to find $x^*, F_{\text{opt}}(x^*)$. The validation function $V(x^*) = 1$ iff $\text{feasible}(x^*)=1$ and $|F_{\text{sim}}(x^*) - F_{\text{opt}}(x^*)| \leq \delta$, where $\delta = \text{atol} + \text{rtol} \cdot |F_{\text{opt}}(x^*)|$. If $V=0$, the simulator outputs a structured error report used as a refinement prompt for the optimizer ACA.
    *   **Design Motivation**: Leverages the asymmetry that "simulation is ~10 percentage points easier than optimization in Pass@1." It uses the more reliable side to verify the other, bypassing the difficulty of verifying semantic correctness without ground truth. This differentiates NEMO from critic-based systems: critics use text-based reasoning, while simulators provide objective execution-based judgment.

3.  **Stability Suite for ACA Non-determinism—Diverse Memory + Component-wise MBR + Self-consistency**:
    *   **Function**: Suppresses stochastic jitter in variable naming, constraint expression, solver configuration, and numerical precision across different stages.
    *   **Mechanism**: a) **Memory**: A vector database of 3,000 samples from OptMATH. For problem $D$, a pool $\mathcal{M}$ is retrieved via cosine similarity, then a greedy score $\text{score}(c) = \text{sim}(D, c) - \lambda \cdot \frac{1}{|\mathcal{M}^*|}\sum_{m \in \mathcal{M}^*} \text{sim}(c, m)$ balances relevance and diversity to select $k$ samples. b) **Component-wise MBR**: Generates $n$ extractions, calculates mean cosine similarity per component type $j$ as utility $S(c_j^i) = \frac{1}{n-1}\sum_{k \neq i} \text{sim}(c_j^k, c_j^i)$, and aggregates the top-$q$ candidates for an LLM-Judge to pick the final $\mathcal{P}^*$ based **strictly on problem $D$** (avoiding memory bias). c) **Self-consistency**: Runs $T$ optimizers; status is determined by majority vote via lexicographic order: $\text{Optimal} \succ \text{Time Limit} \succ \text{Infeasible} \succ \text{Unbounded} \succ \text{Error}$, then clusters objective values within $F_{\text{opt}}$ tolerance to find the largest cluster's median.
    *   **Design Motivation**: ACA sandboxes ensure syntax but not semantic correctness, and results drift. This suite addresses upstream prompt diversity, upstream extraction stability (component-level is finer than global voting), and downstream solution stability (handling solver floating-point noise).

### Loss & Training
NEMO **does not train any models**. All modules use general-purpose LLMs (OpenAI o3 for reasoning, Claude 3.7 Sonnet for ACA via OpenHands, Qwen3-Embedding-8B for embeddings) + inference-time mechanisms. All hyperparameters are fixed once on a development set and used across all 9 benchmarks without benchmark-specific tuning.

## Key Experimental Results

### Main Results: 9 Benchmarks vs Prev. SOTA

| Dataset | NEMO | Prev. Best Agent | Prev. Best Training | Major Gain |
|--------|------|----------------|---------------------|----------|
| OptiBench | **90.4%** | – | 67.4% (SIRL) | +8 pp vs best public |
| OptMATH-Bench | **65.7%** | – | 45.8% (SIRL) | +20 pp |
| NL4OPT | 98.4% (Std) / **99.1%** (Cur) | 78.8% (OptiMUS) | 97.3% (LLMOPT) | Leads Agent systems significantly |
| NLP4LP | 81.4% (Std) / **95.7%** (Cur) | 72.0% (OptiMUS) | 86.5% (LLMOPT) | +14 pp (Std) |
| BWOR | 82.9% | 82.9% (OR-LLM-Agent) | – | Parity |
| IndustryOR | 63.0% (Std) / **76.0%** (Cur) | 36.0% (OR-LLM-Agent) | 48.0% (SIRL) | **+28 pp vs SIRL** |
| MAMO-Easy | 83.4% (Std) / 93.5% (Cur) | 82.2% (OR-LLM-Agent) | 94.7% (SIRL) | Parity with Training-based |
| MAMO-Complex | 72.0% (Std) / **94.0%** (Cur) | 51.6% (OR-LLM-Agent) | 85.8% (LLMOPT) | +20 pp |
| ComplexOR | **77.8%** | 66.7% (OptiMUS) | 72.7% (LLMOPT) | +5 pp |

SOTA on 8 out of 9 benchmarks, with a maximum lead of +28 pp on IndustryOR.

### Ablation Study: Incremental Module Stacking

| Configuration | OptMATH | BWOR | IndustryOR | MAMO-Complex | NLP4LP |
|------|---------|------|------------|--------------|--------|
| w/o Sim (No validation loop) | 59.6% | 71.9% | 60.0% | 50.9% | 76.0% |
| Base (with Sim) | 63.2% | 75.6% | 60.0% | 54.2% | 77.5% |
| + Mem | 63.9% | 80.4% | 62.0% | 61.9% | 79.0% |
| + Mem + MBR | 64.5% | 82.9% | 63.0% | 71.0% | 81.4% |
| + Mem + MBR + Multi-Optimizer | **65.7%** | 82.9% | 63.0% | **72.0%** | 81.4% |

### Key Findings
*   **Simulator impact varies by benchmark**: Adding a simulator gains 1.5–3.7 pp on BWOR/OptMATH/MAMO-Complex/NLP4LP, but zero gain on IndustryOR. Instance-level analysis shows 84% of failures are upstream (52% extraction, 43% logic/constraint). Simulators primarily fix implementation bugs (fixing 4 out of 5 cases encountered), so IndustryOR's lead is mostly from Memory + MBR patching upstream holes.
*   **Memory + MBR is the IndustryOR killer**: Removing the simulator but keeping Memory + MBR still reaches 60% on IndustryOR, suggesting upstream understanding > downstream verification for real-world industrial problems.
*   **Simulator retains independent value**: In a full NEMO setup with Claude 4.5 Sonnet, removing the simulator drops BWOR from 86.6% to 80.5% (−6.1 pp), showing its contribution is not fully redundant.
*   **Simulation is 10 pp easier than optimization**: On OptiBench Pass@1, simulator 97% vs optimizer 87%—providing the empirical foundation for asymmetric verification.

## Highlights & Insights
*   **"ACA = First-class abstraction upgrade"**: This is a powerful system-level framing. By acknowledging an ACA as a primitive with its own sandbox and debug loop, the architecture avoids complex coder/critic/executor tropes.
*   **"Simulator as Referee" and ground-truth-free scenarios**: This paradigm is applicable wherever "verification is easier than solving" (e.g., inverse vs. forward programming, constraint checking vs. satisfaction). It is more reliable than LLM-as-Judge because the judgment is based on execution, not inference.
*   **Component-wise MBR + LLM-Judge**: Using embedding-based utility per component prevents the vote from being dominated by a single long component, while the LLM-Judge's "original-problem-only" view prevents retrieval bias.

## Limitations & Future Work
*   Target value matching is a **necessary but not sufficient** condition; two models might yield the same objective value but have different feasible regions. True formal semantic equivalence remains unsolved.
*   High computational cost: 5–10 minutes per instance, which is impractical for high-throughput scenarios. Future work could include caching templates or distilling patterns into specialized components.
*   Reinforcement Learning: The simulator-optimizer execution feedback could serve as an RL signal to fine-tune ACAs, providing finer-grained rewards than outcome-level labels.

## Related Work & Insights
*   **vs Agent-based (OptiMUS, etc.)**: NEMO replaces text-based message-passing with the Workspace-State paradigm (shared sandbox + runnable artifacts). Execution grounding proves more effective than role specialization.
*   **vs Training-based (SIRL, LLMOPT, etc.)**: NEMO bypasses the need for massive labeled data. Leading the strongest training-based model (SIRL) by 28 pp on IndustryOR demonstrates that execution grounding + RAG can outperform domain-specific fine-tuning.
*   **vs Best-of-N Decoding**: Standard Best-of-N lacks structured cross-sample verification. NEMO’s self-consistency uses status-level and numerical clustering, and the simulator-optimizer check provides **heterogeneous** cross-verification (different paradigms) rather than homogeneous sampling.

## Rating
*   Novelty: ⭐⭐⭐⭐ Workspace-State + asymmetric validation is a clear architectural innovation, though sub-components (MBR, etc.) are known.
*   Experimental Thoroughness: ⭐⭐⭐⭐ 9 benchmarks and comprehensive ablations, though variance is only provided for two small benchmarks.
*   Writing Quality: ⭐⭐⭐⭐ Excellent framing of asymmetry and insightful attribution analysis in the ablation studies.
*   Value: ⭐⭐⭐⭐ The paradigm of treating ACA as a first-class citizen and using heterogeneous validation has broad implications for "NL to runnable artifact" tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](../../ACL2026/code_intelligence/qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[ICML 2026\] MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair](matchfixagent_language-agnostic_autonomous_repository-level_code_translation_val.md)
- [\[ACL 2026\] CodeDistiller: Automatically Generating Code Libraries for Scientific Coding Agents](../../ACL2026/code_intelligence/codedistiller_automatically_generating_code_libraries_for_scientific_coding_agen.md)
- [\[ACL 2026\] RExBench: Can coding agents autonomously implement AI research extensions?](../../ACL2026/code_intelligence/rexbench_can_coding_agents_autonomously_implement_ai_research_extensions.md)
- [\[ACL 2026\] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](../../ACL2026/code_intelligence/securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)

</div>

<!-- RELATED:END -->
