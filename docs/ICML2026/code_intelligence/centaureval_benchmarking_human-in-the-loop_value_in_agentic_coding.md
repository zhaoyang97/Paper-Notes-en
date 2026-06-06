---
title: >-
  [Paper Note] CentaurEval: Benchmarking Human-in-the-Loop Value in Agentic Coding
description: >-
  [ICML 2026][Code Intelligence][Human-AI collaboration evaluation] CentaurEval is proposed as the first unified evaluation framework for human-AI collaborative programming. By designing 45 "Collaboration-Necessary" task t…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "Human-AI collaboration evaluation"
  - "Code Agents"
  - "Benchmark"
  - "Collaborative programming"
  - "High-order reasoning"
date: 2026-05-08
content_hash: 5d0c535b1db11347
---

# CentaurEval: Benchmarking Human-in-the-Loop Value in Agentic Coding

**Conference**: ICML 2026  
**arXiv**: [2512.04111](https://arxiv.org/abs/2512.04111)  
**Code**: Yes (Open-source evaluation toolkit + 450 task dataset)  
**Area**: Code Intelligence  
**Keywords**: Human-AI collaboration evaluation, Code Agents, Benchmark, Collaborative programming, High-order reasoning  

## TL;DR

CentaurEval is proposed as the first unified evaluation framework for human-AI collaborative programming. By designing 45 "Collaboration-Necessary" task templates, the study demonstrates that standalone LLMs achieve only a 0.67% pass rate and humans alone achieve 18.89%, while human-AI collaboration reaches 31.11%, revealing that LLMs are evolving from execution tools into co-reasoning partners.

## Background & Motivation

**Background**: LLM-driven programming agents (Claude Code, Cursor, GitHub Copilot) are widely used in industrial development, shifting the developer's role from "code producer" to "leader of human-AI collaborative systems."

**Limitations of Prior Work**: Existing evaluation systems possess fundamental flaws—human-oriented platforms (LeetCode, Codeforces) test algorithmic capabilities that are being automated; AI-oriented benchmarks (HumanEval, SWE-Bench), while pursuing authenticity, still assume perfectly defined problems, ignoring the assessment of high-order reasoning such as requirement clarification and strategy decomposition. Crucially, existing evaluations isolate humans and AI, failing to quantify collaborative value.

**Key Challenge**: There is a lack of an evaluation framework that simultaneously satisfies two needs: (1) quantifying the human contribution in human-AI collaboration; (2) challenging the high-order reasoning of LLMs with real-world complexity rather than pure algorithmic difficulty.

**Goal**: Construct a unified human-AI collaborative programming benchmark, including an ecologically valid evaluation environment and "Collaboration-Necessary" task designs.

**Key Insight**: Based on distributed cognition theory, cognition occurs not only within an individual but is distributed across people, tools, and environments; true assessment should use the human-AI pair as the unit of analysis rather than evaluating either party in isolation.

**Core Idea**: Design "Collaboration-Necessary" tasks that are unsolvable for both standalone LLMs and humans but solvable through effective collaboration, while providing dual interfaces—a cloud IDE (for human evaluation) and an automated toolkit (for LLM evaluation)—to achieve unified and comparable assessment.

## Method

### Overall Architecture

CentaurEval consists of four core components: (1) **Problem Template Library** — 45 templates covering 3 professional tracks $\times$ 3 difficulty levels; (2) **Agent Task System** — GPT-4.1 driven dynamic task instantiation; (3) **Standardized Cloud IDE** — A human evaluation environment based on GitHub Codespaces + VS Code + Copilot; (4) **LLM Evaluation Toolkit** — Automated LLM evaluation based on Docker + the VS Code extension CentaurEC. The input is a task template, and the output consists of unified pass/fail metrics and efficiency indicators, supporting direct comparison across four experimental conditions.

### Key Designs

1.  **"Collaboration-Necessary" Problem Template Library**:
    *   **Function**: Construct tasks unsolvable for LLMs but solvable via human-AI collaboration.
    *   **Mechanism**: Wrap multiple layers of real-world complexity around a basic algorithmic core. The AI-Incomplete track injects relational complexity such as under-defined requirements, multi-modal specifications (UML/ER diagrams), and legacy codebases to prevent LLMs from directly decomposing tasks into executable steps; the Human Reliance track embeds repetitive implementation tasks and uncommon API usage with time constraints to make pure manual solutions unfeasible. Formal constraints are defined as $\Pr(\text{Solve}(t, \mathcal{A})) \leq \theta_{\text{low}}$ and $\mathbb{E}[\text{Score}(s_{\mathcal{H}+\mathcal{A}})] - \mathbb{E}[\text{Score}(s_{\mathcal{H}})] \geq \delta$.
    *   **Design Motivation**: Directly address the binary fragmentation of existing benchmarks that test either only AI or only humans, ensuring the measured performance gap truly reflects collaborative value.

2.  **Agent-Driven Dynamic Task Instantiation System**:
    *   **Function**: Dynamically generate infinite task instances from 45 templates to prevent data leakage and memorization effects.
    *   **Mechanism**: A GPT-4.1 Agent schedules four specialized tools—TechnicalParameterTool generates logic-critical parameters, ImplementationConstraintTool selects framework configurations, ContextualVariableTool generates real-world scenario wrappers, and InterfaceSpecificationTool generates interface details. Logical-critical generation (deterministic) is strictly separated from surface wrapper generation (diverse) to ensure variations do not introduce additional cognitive difficulty. Each instance synchronously outputs a task package and an evaluation script.
    *   **Design Motivation**: Traditional static datasets risk data leakage; dynamic generation ensures fairness while improving scalability.

3.  **Ecologically Valid Dual-Interface Evaluation System**:
    *   **Function**: Provide unified and comparable evaluation environments for both humans and LLMs.
    *   **Mechanism**: The human side uses GitHub Codespaces to deploy a full VS Code + Copilot environment, eliminating tool familiarity as a confounding factor; the LLM side replicates the human workflow via the CentaurEC extension (environment deployment $\rightarrow$ task injection $\rightarrow$ code generation $\rightarrow$ test feedback $\rightarrow$ iterative revision $\rightarrow$ scoring), using 450 static task instances to ensure reproducibility. Auto-Calibrated Baselines are introduced to dynamically calibrate efficiency thresholds by running reference solutions, enabling fair cross-platform comparison.
    *   **Design Motivation**: Ensure humans and LLMs are evaluated under equivalent conditions to make results directly comparable.

### Metrics

A two-stage evaluation protocol is adopted: 5 raw metrics are recorded (test case pass/fail, execution time, peak memory, completion time, token usage), aggregated into 4 analytical metrics—Overall Pass, Partial Pass, Completion Time (Penalized Average Runtime PAR, with timeouts recorded as 60 minutes), and Token Usage.

## Key Experimental Results

### Main Results

Comparative experiments were conducted across 4 conditions with 45 expert participants and 5 SOTA LLMs: $C_H$ (Pure Human), $C_0$ (Autonomous AI), $C_1$ (Minimal Intervention AI), and $C_2$ (Human-AI Collaboration).

| Condition | Avg. Pass@1 | 95% CI | Description |
|-----------|-------------|--------|-------------|
| $C_0$ (Autonomous AI) | 0.67% | 0.23–1.94 | LLM independent completion |
| $C_1$ (Minimal Intervention) | 2.89% | 1.70–4.88 | Repairing procedural failures only |
| $C_H$ (Pure Human) | 18.89% | 12.1–28.2 | No AI assistance |
| $C_2$ (Human-AI Collaboration) | **31.11%** | 22.5–41.3 | Free use of Copilot |

### Performance of LLMs under Different Conditions

| Model | $C_0$ Pass@1 | $C_1$ Pass@1 | $C_0$ Partial | $C_1$ Partial |
|-------|--------------|--------------|---------------|---------------|
| Claude-Sonnet-4 | 0.67% | 2.89% | 19.24% | 30.13% |
| Claude-Sonnet-3.7 | 0.00% | 1.56% | 8.71% | 17.47% |
| GPT-4.1 | 0.00% | 1.78% | 11.16% | 23.64% |
| GPT-4o | 0.00% | 0.00% | 5.82% | 12.09% |
| Gemini-2.5-Pro | 0.22% | 2.22% | 8.27% | 21.33% |

### Difficulty Stratification Analysis

| Difficulty | $C_H$ Pass | $C_0$ Pass | $C_1$ Pass | $C_2$ Pass |
|------------|------------|------------|------------|------------|
| Easy | 36.7% | 1.3% | 4.0% | 43.3% |
| Medium | 13.3% | 0.7% | 2.7% | 26.7% |
| Hard | 6.7% | 0.0% | 2.0% | 23.3% |

### Key Findings

*   **Significant Collaboration Gain**: $C_2$ improves by 12.22 percentage points over $C_H$ ($p = 0.00739$), and by over 28 percentage points over the strongest standalone LLM.
*   **Collaboration Increases in Importance with Difficulty**: Human Pass rates dropped from 36.7% (Easy) to 6.7% (Hard)—an 82% decrease—whereas the collaborative mode only dropped from 43.3% to 23.3% (a 46% decrease), showing a clear "gain amplification" effect of collaboration on difficult tasks.
*   **LLM Bottleneck is Reasoning, Not Execution**: The improvement from $C_0$ to $C_1$ (fixing procedural failures) indicates that current LLM failures are not merely environmental interaction issues but are fundamentally rooted in a lack of high-order reasoning.
*   **51% of participants adopted fundamentally different problem-solving strategies proposed by AI**, and 12 of the top 15 performers utilized strategic-level suggestions from AI.

## Highlights & Insights

*   **"Collaboration-Necessary" Task Design Paradigm**: By wrapping real-world complexity (under-defined requirements, multi-modal specs, etc.) around an algorithmic core to create blind spots for both LLMs and humans, this "bilateral unsolvability" design can be generalized to other human-AI collaboration evaluation scenarios.
*   **Cognitive Leap from Tool to Partner**: Experiments found that 80% of participants used AI for strategic brainstorming and 51% adopted entirely new schemes proposed by AI. This is no longer the traditional "human thinks, AI writes" mode, but true co-reasoning—a finding with significant implications for AI-assisted education and development tool design.
*   **Dual-Track System of Dynamic Generation and Static Evaluation**: Using dynamic instantiation on the human side prevents memory effects, while 450 static tasks on the LLM side ensure reproducibility. Both sides remain comparable through identical templates and evaluation scripts; this design is transferable to other human-in-the-loop evaluation contexts.

## Limitations & Future Work

*   Currently supports only Python, failing to cover multi-language development scenarios.
*   Reliant on GitHub Copilot as a unified interface, failing to evaluate other significant models like o3, GPT-5, DeepSeek, LLaMA, or Qwen.
*   Participants were all East Asian university students/recent graduates, limiting generalizability to industry developers and other groups.
*   "Collaboration-Necessary" is a dynamic concept relative to current model capabilities; as models advance, some tasks may become autonomously solvable—though this allows CentaurEval to track the movement of the autonomy boundary.
*   Converting efficiency metrics to discrete pass/fail results involves a loss of fine-grained information.

## Related Work & Insights

*   **HumanEval / SWE-Bench** — Evaluates LLM programming capabilities in isolation without considering the human-AI collaboration dimension.
*   **LeetCode / Codeforces** — Human-oriented algorithmic competition platforms testing skills that are being automated.
*   **Centaur Evaluation Theory (Haupt & Brynjolfsson 2025)** — Proposed the concept of quantifying human contribution in human-AI collaboration; CentaurEval is its first implementation in the programming domain.
*   **Distributed Cognition Theory (Hutchins 1995)** — Provides the theoretical basis for cognition being distributed across people, tools, and environments, supporting the evaluation design using the "human-AI pair as the unit of analysis."
*   **Personal Insight**: When evaluating AI systems, one should not only focus on standalone AI capabilities but also on the overall performance ceiling and collaborative efficiency of human-AI systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FormalScience: Scalable Human-in-the-Loop Autoformalisation of Science with Agentic Code Generation in Lean](../../ACL2026/code_intelligence/formalscience_scalable_human-in-the-loop_autoformalisation_of_science_with_agent.md)
- [\[ICML 2026\] NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents](nemo_execution-aware_optimization_modeling_via_autonomous_coding_agents.md)
- [\[ACL 2026\] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](../../ACL2026/code_intelligence/securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)
- [\[ACL 2026\] CodeDistiller: Automatically Generating Code Libraries for Scientific Coding Agents](../../ACL2026/code_intelligence/codedistiller_automatically_generating_code_libraries_for_scientific_coding_agen.md)
- [\[ACL 2026\] Benchmarking Testing in Automated Theorem Proving](../../ACL2026/code_intelligence/benchmarking_testing_in_automated_theorem_proving.md)

</div>

<!-- RELATED:END -->
