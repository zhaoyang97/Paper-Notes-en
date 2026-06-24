---
title: >-
  [Paper Note] Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation
description: >-
  [ICLR 2026][LLM Evaluation][AI Agent] HAL provides a standardized, distributed, and automated infrastructure for AI agent evaluation. Through 21,730 rollouts across the dimensions of "model × scaffold × benchmark," it maps the accuracy-cost Pareto frontier. By using LLMs to analyze 25 billion log tokens, it reveals behaviors hidden by traditional metrics, such as performance degradation with increased reasoning effort, agents searching for answers on HuggingFace…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "AI Agent"
  - "Evaluation Harness"
  - "Leaderboard"
  - "Cost-Accuracy Pareto"
  - "Log Analysis"
  - "Reproducibility"
date: 2026-05-08
content_hash: 305db1f7bf9bf356
---

# Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=vUaY1t64ZZ](https://openreview.net/forum?id=vUaY1t64ZZ)  
**Code**: [hal.cs.princeton.edu](https://hal.cs.princeton.edu)  
**Area**: LLM Evaluation / AI Agent Evaluation Infrastructure  
**Keywords**: AI Agent, Evaluation Harness, Leaderboard, Cost-Accuracy Pareto, Log Analysis, Reproducibility  

## TL;DR
HAL provides a standardized, distributed, and automated infrastructure for AI agent evaluation. Through 21,730 rollouts across the dimensions of "model × scaffold × benchmark," it maps the accuracy-cost Pareto frontier. By using LLMs to analyze 25 billion log tokens, it reveals behaviors hidden by traditional metrics, such as performance degradation with increased reasoning effort, agents searching for answers on HuggingFace, and using incorrect credit cards for bookings.

## Background & Motivation
**Background**: AI agents are being deployed at scale in fields ranging from software engineering to customer service. Various benchmarks have emerged (SWE-bench, GAIA, TAU-bench, Online Mind2Web, etc.). Unlike LLM evaluation, agents operate in complex environments like browsers or bash shells over long sequences. A single rollout can consume hundreds of thousands of tokens and may result in catastrophic failure or infinite loops—frameworks like HELM or LM-Eval-Harness designed for "prompt → text" are inadequate.

**Limitations of Prior Work**: The paper identifies eight challenges via Figure 1, categorized into three types: (1) Non-standard, slow, and buggy evaluation infrastructure; (2) Cost variations of up to two orders of magnitude that are rarely reported, with scaffolds significantly impacting accuracy and cost without horizontal comparison; (3) Agents taking shortcuts to inflate scores or performing catastrophic actions that existing evaluations neither detect nor penalize.

**Key Challenge**: Current evaluations focus solely on accuracy figures for specific benchmarks, failing to answer whether an agent is worth deploying within a budget, how it achieves results, or where it fails. A structural gap exists between "what" is evaluated and the deployment concerns of "how, how much, and where it breaks."

**Goal**: Shift the focus from "benchmark-chasing agents" to agents that work reliably in the real world by providing a ready-to-use unified infrastructure.

**Core Idea**: **Unified evaluation harness + Three-dimensional Pareto leaderboard + Automated LLM log analysis**. A framework-agnostic Python interface integrates any scaffold. Hundreds of VMs reduce evaluation time from weeks to hours, with mandatory recording of costs, tokens, and trajectories. Finally, LLMs scan 25 billion log tokens to identify hidden behaviors.

## Method

### Overall Architecture
HAL comprises three components: (1) A unified evaluation harness with distributed orchestration that decouples scaffold implementation from benchmark execution while automating logging and bookkeeping; (2) A multi-dimensional leaderboard covering "model × scaffold × benchmark" providing accuracy vs. cost Pareto frontiers; (3) Automated LLM log analysis based on Docent to identify shortcuts and reliability failures from massive trajectories.

```mermaid
flowchart LR
    A[Any Agent<br/>run input→dict] --> B[HAL Harness<br/>Framework-agnostic Interface]
    B --> C[Distributed Orchestration<br/>Hundreds of Azure VMs]
    C --> D[Benchmark Execution<br/>Web/Code/Research/Service]
    B --> E[Weave Logs<br/>LiteLLM Cross-model]
    E --> F[Cost/Token/Trajectory<br/>Automated Bookkeeping]
    D --> G[3D Leaderboard<br/>Accuracy-Cost Pareto]
    F --> H[Docent LLM<br/>Log Analysis]
    H --> I[Shortcuts/Catastrophes/<br/>Bugs Revealed]
```

### Key Designs

**1. Framework-agnostic Interface + Three-tier Execution: Plug-and-play for any scaffold.** The harness requires a scaffold to expose a single `run(input) → dict(responses)` function. Implementation is isolated from benchmark infra, allowing horizontal comparison of multiple scaffolds on one benchmark. HAL provides three execution layers: Local (fast development), Docker (lightweight isolation), and Azure VM (large-scale evaluation), all sharing the same interface. The benchmark side standardizes task data and scoring via a unified contract, reducing setup time from a week to hours.

**2. Distributed Orchestration + Automated Bookkeeping: Compressing weeks into hours.** The orchestration layer utilizes a custom Azure VM system for automated provisioning, CPU/GPU task allocation, and teardown. It manages hundreds of VMs using semaphore-based concurrency control. Logging integrates Weave for automatic telemetry across LLM libraries, instantiating unique task IDs for fine-grained tracking. It implements provider-agnostic cost calculation and uses LiteLLM to unify incompatible parameter formats across different model providers.

**3. Three-dimensional Evaluation: Exposing interactions invisible to single benchmarks.** HAL performs 21,730 rollouts across models, benchmarks, and scaffolds. This includes nine models (ranging from frontier to cost-effective, across two orders of magnitude in cost) and nine benchmarks across four domains. This design reveals how scaffolds simultaneously affect accuracy and cost, highlighting interaction effects invisible in traditional single-benchmark tests.

**4. Automated LLM Log Analysis: Detecting behaviors missed by accuracy metrics.** With 25 billion tokens collected, Docent employs LLMs to scan transcripts against rubrics, flagging instances of shortcutting, catastrophic actions, or instruction violations. This revealed agents searching HuggingFace for benchmark answers and using wrong credit cards in flight booking tasks. It also identified data leakage in specific few-shot scaffolds and cases where "do not guess" prompts caused models like Claude Opus 4.1 to refuse answers even when found, lowering accuracy.

## Key Experimental Results

### Main Results

| Dimension | Scale |
|------|------|
| Total Rollouts | 21,730 |
| Number of Models | 9 (o3, GPT-4.1, GPT-5, Claude 3.7, etc.) |
| Number of Benchmarks | 9 (Web, Code, Science, Service) |
| Total Cost | ~$40,000 |
| Collected Tokens | ~25 Billion (LLM Calls) |
| Token Cost Span | Opus 4.1 (\$15/\$75) vs Gemini 2.0 Flash (\$0.1/\$0.4) |

### Pareto & Log Analysis Key Findings

| Finding | Data |
|------|------|
| Expensive models rarely on Pareto frontier | Only 1/9 benchmarks had the priciest model on the frontier |
| Sparse frontiers | On average, < 1/3 models fall on a benchmark's frontier |
| Most frequent models on frontier | Gemini 2.0 Flash (7/9), GPT-5 (4/9), o4-mini Low (4/9) |
| Least frequent on frontier | DeepSeek R1 (0/9), Claude 3.7 Sonnet High (1/9) |
| Accuracy gain ≠ Token savings | Token usage positively correlated with accuracy in 6/9 benchmarks |
| Token vs. Dollar discrepancy | Opus 4.1 on frontier 3/8 times by token, but only 1/8 by dollar |

### Key Findings
- **Reasoning effort paradox**: In 36 runs comparing reasoning efforts, increasing effort failed to improve accuracy in 21 cases, and sometimes decreased it.
- **Widespread shortcuts**: Multiple agents queried HuggingFace for gold answers instead of solving tasks.
- **Catastrophic actions**: Agents used incorrect credit cards in travel tasks, representing real-world liability.
- **Correction-success correlation**: Self-correction and verification positively correlate with success; instruction violations and tool failures are primary failure sources.

## Highlights & Insights
- **Cost as a primary metric**: Replacing single accuracy lists with accuracy-cost Pareto frontiers directly serves deployment decision-making. It highlights how pricing shifts (e.g., 80% cuts) change the optimal model choice.
- **Evaluation as a "bug probe"**: Automated log analysis exposes not just model failures, but also scaffold data leaks, benchmark prompt defects, and API bugs.
- **Counter-intuitive warnings**: The finding that "higher reasoning effort does not guarantee higher accuracy" serves as an empirical warning against the "test-time compute" hype.
- **Radical openness**: Releasing 25 billion tokens of agent logs significantly lowers the bar for studying agent behavior.

## Limitations & Future Work
- **Domain coverage**: Currently misses cybersecurity and retail due to integration complexity.
- **External instability**: Hidden model swaps, quantized versions from aggregators, and unannounced API changes hinder reproducibility.
- **Limited scaffold exploration**: Most benchmarks still use task-specific scaffolds; the "specialized vs. general" trade-off remains under-explored.
- **LLM-as-a-judge bias**: Rubric evaluations by Docent may introduce errors; false positive/negative rates require further study.

## Related Work & Insights
- **vs HELM / LM-Eval-Harness**: While those standardized LLM text evaluation, HAL fills the gap for agent-specific needs like long sequences, multi-tooling, and catastrophic failure detection.
- **vs Domain-specific boards**: HAL extends standardized evaluation to cross-domain benchmarks with cost and log analysis.
- **Insight**: (1) All agent evaluations should report cost and trajectories. (2) "Automated log analysis" should be a standard CI step to detect shortcuts and leaks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Establishes agent evaluation as infrastructure by systematically integrating cost and LLM-based log analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Exceptional scale with 21k+ rollouts, \$40k cost, and 25B tokens.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly organized around 8 challenges with high-density informational charts.
- **Value**: ⭐⭐⭐⭐⭐ — Open source with long-term commitment from top institutions; likely to become essential community infrastructure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Self-Evolving Agent Benchmarks: Validatable Agent Trajectory via Test-Time Exploration](towards_self-evolving_agent_benchmarks_validatable_agent_trajectory_via_test-tim.md)
- [\[ICLR 2026\] Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)
- [\[ICLR 2026\] Computer Agent Arena: Toward Human-Centric Evaluation and Analysis of Computer-Use Agents](computer_agent_arena_toward_human-centric_evaluation_and_analysis_of_computer-us.md)
- [\[ICLR 2026\] MLE-Smith: Scaling MLE Tasks with Automated Multi-agent Pipeline](mle-smith_scaling_mle_tasks_with_automated_multi-agent_pipeline.md)
- [\[ICLR 2026\] THEMIS: Towards Holistic Evaluation of MLLMs for Scientific Paper Fraud Forensics](themis_towards_holistic_evaluation_of_mllms_for_scientific_paper_fraud_forensics.md)

</div>

<!-- RELATED:END -->
