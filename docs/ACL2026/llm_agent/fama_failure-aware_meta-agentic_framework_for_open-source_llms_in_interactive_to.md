---
title: >-
  [Paper Note] FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments
description: >-
  [ACL 2026][LLM Agent][Failure-aware] FAMA employs a set of independent "failure analysis agents + orchestration agent" to automatically diagnose primary failure modes of baseline tool-use agents in multi-turn interaction…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Failure-aware"
  - "Meta-Agent"
  - "Tool-use"
  - "$\\tau$-bench"
  - "Open-source LLM"
date: 2026-05-08
content_hash: 08627855e58fa333
---

# FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments

**Conference**: ACL 2026  
**arXiv**: [2604.25135](https://arxiv.org/abs/2604.25135)  
**Code**: None (not explicitly disclosed in the paper)  
**Area**: LLM Agent / Tool Use / Multi-agent Orchestration  
**Keywords**: Failure-aware, Meta-Agent, Tool-use, $\tau$-bench, Open-source LLM

## TL;DR
FAMA employs a set of independent "failure analysis agents + orchestration agent" to automatically diagnose primary failure modes of baseline tool-use agents in multi-turn interactions (e.g., $\tau$-bench). It then directs a mitigation agent to select a minimal subset of helper agents for context injection, improving task success rates by up to 27% on the Qwen open-source series.

## Background & Motivation
**Background**: Multi-turn tool-use benchmarks such as $\tau$-bench, $\tau$-trait, and ACEBench treat LLMs as customer service agents that interact with simulated users, call APIs, and follow domain rules. Current improvement strategies either involve SFT/RL training or static multi-agent frameworks (e.g., IRMA) that link modules like Planner, Memory, and Tool Reformulator to assist the base agent.

**Limitations of Prior Work**: 1) Training routes are prohibitively expensive for data collection and reward propagation in long multi-turn trajectories. 2) Static multi-agent frameworks cram all helper agents into the context, which is catastrophic for small open-source models—the context window is saturated by helper outputs (IRMA average overhead 50-58%), sometimes underperforming vanilla ReAct. Furthermore, dominant failure modes vary across models, making a static set of agents fundamentally mismatched.

**Key Challenge**: Smaller models with narrower context windows require "frugal" decisions on which helper agent to deploy. Existing static frameworks lack awareness of why an agent fails or whether a specific helper addresses the underlying issue.

**Goal**: To construct a training-free framework that (a) automatically identifies the dominant failure modes of a base agent, (b) dynamically selects a minimal subset of helpers based on failure modes, and (c) achieves stable gains on open-source models.

**Key Insight**: A priori, the authors categorize tool-use failures into four types (Domain Policy Violation, Incorrect Retrieval from Complex Tool Outputs, Contextual Misinterpretation/Hallucination, and Incomplete Fulfillment). They observe that different open-source models exhibit distinct dominant failure categories across benchmarks—implying that helper agent selection must be model-aware and benchmark-aware.

**Core Idea**: A "meta-agent" approach of "agents diagnosing agents": a group of failure analysis agents, an orchestrator, and a mitigation agent analyze the base agent's real failure trajectory to decide which helpers to employ in the subsequent round.

## Method

### Overall Architecture
FAMA is a two-stage training-free pipeline. Stage 1: The base agent (ReAct/FC) runs all tasks once to collect failure trajectories $\mathcal{F}$. Stage 2: For each failure trajectory, a "diagnosis-orchestration-mitigation" process is executed: (2.1) $|\mathcal{E}|=4$ independent error analysis agents determine if a specific error category was triggered and provide a rationale; (2.2) an orchestrator agent reviews all analyst outputs and the original trajectory to determine the final dominant error category $\hat{\mathcal{E}}_\tau$; (2.3) a mitigation agent selects a minimal subset $\mathcal{A}^*_\tau$ from a predefined agent pool $\mathcal{A}$ (DCE, TSA, TOR, Planner, Verifier, Memory). Finally, the task is rerun using $\mathcal{A}^*_\tau$. The framework updates no model weights and only modifies context construction during inference.

### Key Designs

1. **Four Failure Ontologies + Independent Analyst Agents**:
    - **Function**: Categorizes tool-use failures into (1) Domain Policy Violation, (2) Incorrect Retrieval from Complex Tool Outputs, (3) Contextual Misinterpretation/Hallucination, and (4) Incomplete Fulfillment/Early Stopping, assigning an independent LLM analyst to each category.
    - **Mechanism**: Each analyst focuses on a single failure causal chain, providing a binary decision and a natural language rationale. Analyst outputs are concatenated $O_\tau = \text{Concat}(\{o_{\tau,e}\}_{e\in\mathcal{E}})$ for the orchestrator's final attribution, preventing interference between categories.
    - **Design Motivation**: Statistics in §5.2 show dominant failure categories vary significantly—CM and DCV are severe on Tau-bench, IFU is prominent on $\tau$-trait, and CM dominates ACEBench. Multi-classification via a single prompt is prone to being dominated by the majority class; separate diagnosis is more robust.

2. **Orchestrator + Mitigation Two-Level Routing**:
    - **Function**: The orchestrator fuses signals from multiple analysts to determine "where the task truly failed," while the mitigation agent maps error categories to "which helper agents should be activated."
    - **Mechanism**: The mitigation agent receives $\hat{\mathcal{E}}_\tau$ and natural language descriptions of the agent pool $\mathcal{A}$, outputting a minimal subset $\mathcal{A}^*_\tau \subseteq \mathcal{A}$ such that $\bigcup_{e\in\hat{\mathcal{E}}_\tau}\text{cover}(e)$. Aggregation yields stable recommended configurations for specific models on specific benchmarks (see Tables 5-7).
    - **Design Motivation**: Decoupling "diagnosis" from "prescription" simplifies each agent's task. Diagnosis agents only need to understand failure ontologies, while mitigation agents only need to understand helper functional boundaries. This ensures stable output even from open-source models.

3. **Empirical Confirmation of Memory + DCE**:
    - **Function**: Statistical analysis of mitigation outputs (Figs 5/13/15) confirms that for all Qwen open-source models, the Memory module and Domain Constraints Extractor (DCE) are the most frequently recommended helpers.
    - **Mechanism**: Memory retains the last $k$ rounds of user queries (domain-dependent: $k=2$ for Airline, $k=6$ for Retail); DCE extracts domain constraints relevant to the current state from the system prompt before each decision.
    - **Design Motivation**: Investigations in §5.3 reveal that open-source models "forget" domain rules in long conversations, and large tool outputs crowd out early constraints—a memory bottleneck. The convergence of the mitigation agent to Memory+DCE validates diagnostic accuracy.

### Loss & Training
FAMA is a pure inference-time framework with no parameter updates. GPT-4o (or GPT-4.1-mini for robustness comparison) serves as the judgment agent (analyst/orchestrator/mitigation); base tool-use agents include Qwen3-4B/14B/32B and Qwen2.5-72B-Instruct (which also serves as the user simulator).

## Key Experimental Results

### Main Results
Average $pass^k$ values ($k=1..5$) over five independent runs on $\tau$-bench, comparing ReAct, FC, IRMA, and FAMA:

| Model | Domain | Metric | ReAct | FC | IRMA | FAMA |
|------|--------|------|-------|-----|------|------|
| Qwen3-4B | Airline | $pass^1$ / $pass^5$ | 32.0 / 26.0 | 27.6 / 14.0 | 30.0 / 12.0 | **37.6 / 26.0** |
| Qwen3-4B | Retail | $pass^1$ / $pass^5$ | 17.2 / 8.7 | 24.9 / 9.0 | 28.9 / 9.6 | **34.6 / 13.9** |
| Qwen2.5-72B | Airline | $pass^1$ / $pass^5$ | 24.4 / 10.0 | 15.2 / 2.0 | 26.4 / 10.0 | **29.2 / 18.0** |
| Qwen2.5-72B | Retail | $pass^1$ / $pass^5$ | 43.5 / 20.9 | 19.7 / 4.3 | 38.8 / 19.1 | **44.2 / 27.0** |

Summary: FAMA outperforms ReAct/FC/IRMA by an average of 4.63 / 11.57 / 5.27 points in Airline and 5.30 / 8.96 / 6.15 in Retail. On ACEBench, end-to-end accuracy for Qwen2.5-72B improved from 23.3% to 50.0% (+26.7%).

### Ablation Study

| Configuration (Qwen3-14B, $\tau$-bench Airline $pass^1$) | Accuracy | Description |
|------|--------|------|
| Full FAMA (mitigation recommend = DCE+Memory) | **36.8%** | Complete approach |
| Memory+DCE+TOR (Exp 1, not recommended) | Lower | Adding TOR decreases performance |
| Memory+TOR (Exp 2) | Lower | Missing DCE loses domain constraints |
| Memory+TOR+TSA (Exp 3) | Lower | Performance drops as configuration deviates |
| IRMA (All agents) | 26.4% | Lowest when indiscriminately using all agents |

Regarding efficiency (Qwen3-32B, Table 2): IRMA has 50-58% overhead and 111-150s latency; FAMA has ~30% overhead and 57-91s latency. ReAct-thinking often fails due to token overflow from CoT explosion.

### Key Findings
- Combinations recommended by the mitigation agent consistently outperform random ones. **Using fewer, properly recommended agents can be more effective** (indicating interference between helpers; IRMA’s exhaustive approach is counter-productive).
- Switching to GPT-4.1-mini for judgment produced failure categories and recommendations highly consistent with GPT-4o (Fig 9/11), indicating low sensitivity to the judgment model.
- Reasoning/thinking variants (Qwen3 thinking) frequently fail due to internal CoT exhausting the token budget. FAMA-non-thinking is more stable—a counter-intuitive finding for open-source thinking models.

## Highlights & Insights
- The "Meta-Agent" abstraction is elegant: FAMA does not act directly in the environment but indirectly improves decision-making by diagnosing and restructuring context. This evolves "agent orchestration" from manual design to a data-driven automated process.
- The decoupling of diagnosis, orchestration, and mitigation is a reusable template. Because each agent's task is narrow, even smaller judgment models like GPT-4.1-mini perform stably; it is easily transferable to other agentic scenarios (e.g., coding or research).
- Empirical evidence that "cramming all helpers is worse" is highly valuable—it challenges the default belief that more agents are necessarily better and forces consideration of marginal utility versus context cost.

## Limitations & Future Work
- The agent pool is predefined; FAMA’s performance is capped by the pool's coverage. It cannot automatically discover or synthesize new helpers.
- Validated only on structured customer service tool-use; stability on open-ended embodied or multimodal agents remains unknown.
- The four failure ontologies are empirically defined and may be incomplete for niche domains (e.g., mathematical proof agents).
- Dependency on GPT-4o for judgment agents poses a challenge for fully offline/private scenarios. Future work could focus on a "fully open-source diagnostic loop."

## Related Work & Insights
- **vs IRMA (Mishra 2025)**: Both use modular helper agents, but IRMA is static and exhaustive, while FAMA is dynamic and pruned based on failure modes.
- **vs Self-Reflection / Reflexion**: Reflection methods introspect within a trajectory. FAMA performs cross-task statistics between trajectories, providing coarser but more stable signals.
- **vs RL Fine-tuning (VeRLTool/MUA-RL)**: Training approaches are limited by high trajectory collection costs; FAMA is training-free, offering a more pragmatic solution for open-source models.

## Rating
- Novelty: ⭐⭐⭐⭐ The "agent to diagnose agents" concept is refreshing, though the ontologies and helper pool draw from IRMA.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks × four models × five runs, plus token/latency/judgment robustness, thoroughly exercising the $\tau$-bench suite.
- Writing Quality: ⭐⭐⭐⭐ Algorithm 1 and helper boundaries are clear, though some statistics are scattered in the appendix.
- Value: ⭐⭐⭐⭐ A practical recipe for building agent frameworks on small open-source models, debunking the "more agents are stronger" assumption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution](toolomni_enabling_open-world_tool_use_via_agentic_learning_with_proactive_retrie.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ACL 2026\] Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)
- [\[ACL 2026\] How Adversarial Environments Mislead Agentic AI](how_adversarial_environments_mislead_agentic_ai.md)
- [\[AAAI 2026\] LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval](../../AAAI2026/llm_agent/llandmark_a_multi-agent_framework_for_landmark-aware_multimodal_interactive_vide.md)

</div>

<!-- RELATED:END -->
