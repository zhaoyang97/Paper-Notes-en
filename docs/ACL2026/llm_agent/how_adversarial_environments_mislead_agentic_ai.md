---
title: >-
  [Paper Note] How Adversarial Environments Mislead Agentic AI
description: >-
  [ACL 2026][LLM Agent][Adversarial Environment Injection] This paper formalizes the "Adversarial Environment Injection" (AEI) threat model, decomposing it into Breadth Attacks (poisoning retrieval results leading to cogni…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Adversarial Environment Injection"
  - "Trust Gap"
  - "Depth Attack"
  - "Breadth Attack"
  - "Robustness Splitting"
date: 2026-05-08
content_hash: a8e47ff9d276d93f
---

# How Adversarial Environments Mislead Agentic AI

**Conference**: ACL 2026  
**arXiv**: [2604.18874](https://arxiv.org/abs/2604.18874)  
**Code**: [GitHub](https://github.com/zhonghaozhan/Potemkin)  
**Area**: AI Safety / Agent Robustness  
**Keywords**: Adversarial Environment Injection, Trust Gap, Depth Attack, Breadth Attack, Robustness Splitting

## TL;DR

This paper formalizes the "Adversarial Environment Injection" (AEI) threat model, decomposing it into Breadth Attacks (poisoning retrieval results leading to cognitive drift) and Depth Attacks (injecting phantom nodes to construct navigation traps leading to strategy collapse). Across 11,000+ experiments, the study finds that robustness to the two attacks is completely independent—a "robustness splitting" that suggests current single-point defense strategies are fundamentally insufficient.

## Background & Motivation

**Background**: Tool-augmented LLM agents rely on external tools like search engines and citation indices to ground generated content. RAG safety has become an active research field, with existing work focusing on prompt injection and corpus poisoning at the content level.

**Limitations of Prior Work**: (1) Existing evaluations only focus on "whether the Agent can use tools correctly," never considering "what if the tool lies"—there exists a trust gap; (2) RAG poisoning research only covers half the attack surface (content level), ignoring structural attacks; (3) There is a lack of a standardized, reproducible framework for adversarial robustness testing.

**Key Challenge**: The correct behavior for reducing hallucinations (following external information) precisely increases adversarial vulnerability—the "Grounding Paradox." Agents accept the reality presented by the environment and lack independent verification channels, much like Truman living in a fictional world.

**Goal**: (1) Formalize the complete attack surface faced by tool-using Agents; (2) Distinguish between two orthogonal attack dimensions: cognitive level and navigation level; (3) Quantify the independence of the two.

**Key Insight**: Analogous to "The Truman Show"—Agents accept content returned by tools as reality, and attackers construct a false world via "Man-in-the-Tool." Depth attacks are a brand-new category—they do not require the Agent to believe false information but merely trap it in a navigation loop.

**Core Idea**: AEI is decomposed into Breadth Attacks (cognitive drift) and Depth Attacks (strategy collapse), which utilize entirely different mechanisms—the former attacks belief updating, while the latter attacks navigation planning. Therefore, defense against one does not protect against the other.

## Method

### Overall Architecture

The Potemkin framework operates as a transparent Man-in-the-Tool proxy: it intercepts responses to tool calls, applies adversarial transformations, and returns them. It supports both MCP Server and Python Library modes. Breadth attacks poison retrieval results, while depth attacks inject phantom nodes into the citation graph. A total of 7 experiments were conducted across two campaigns, involving ~11,000 task runs.

### Key Designs

1.  **Breadth Attack (The Illusion)**:
    *   **Function**: Induces cognitive drift by poisoning retrieval results, leading the Agent to accept false information.
    *   **Mechanism**: Injects malicious text into the knowledge base, controlling two parameters: pollution rate $\rho \in \{0.1, 0.3, 0.5\}$ (1/3/5 items out of 10 retrieval results) and linguistic style (Professor formal academic / Wire neutral news / Rumor informal vague). Measured by Drift Rate $\text{DR} = \mathbb{E}[\mathbb{1}[r \neq y] | r \neq \bot]$, which excludes abstentions to count only confident but incorrect judgments.
    *   **Design Motivation**: Orthogonal variation of pollution rate and style allows for separating the questions of "how much poison is needed" and "which style is most effective." The DR metric, by excluding abstentions, captures cognitive state changes more precisely than traditional ASR.

2.  **Depth Attack (The Maze)**:
    *   **Function**: Constructs navigation traps by injecting phantom citation nodes to exhaust the Agent's step budget.
    *   **Mechanism**: Injects phantom nodes $\mathcal{P}$ into the information graph to form loops or dead ends, controlling loop length $\ell \in \{2,3,5\}$ and credibility gradients (Phantom high fidelity / Signal slight inconsistency / Glitch obvious anomaly). Measured by Entry Rate $\text{ER} = \mathbb{E}[\mathbb{1}[\exists t: s_t \in \mathcal{P}]]$ and Budget Waste $\text{BW} = \frac{|\{t: s_t \in \mathcal{P}\}|}{|\{t: s_t \in V \cup \mathcal{P}\}|}$.
    *   **Design Motivation**: Depth attacks represent a new attack category that modifies structure rather than content. Even if an Agent does not believe false content, it can be trapped in loops. The credibility gradient parallels the style gradient of breadth attacks, supporting cross-dimensional analysis.

3.  **"The Punishment of Honesty"**:
    *   **Function**: Reveals systematic miscalibration of epistemic markers by Agents.
    *   **Mechanism**: Constructs minimal pairs (identical claims with only changed epistemic markers, e.g., "results suggest" vs. "results prove"), analyzed via McNemar tests. It was found that TRUE claims with hedge words are rejected 2.1 times more often than confident TRUE claims, yet FALSE claims with hedge words are not more easily detected.
    *   **Design Motivation**: Reveals a dangerous asymmetry—attackers can suppress true claims by adding hedge words, which is particularly dangerous in scientific and medical domains.

### Loss & Training

Potemkin is an evaluation framework and does not involve training. All tested Agents were run at $T=0.0$ to ensure deterministic evaluation, with a step budget of 10 tool calls. Adversarial content was generated by a Gemini 2.5 red team to avoid generator-victim overlap.

## Key Experimental Results

### Main Results

**Breadth vs. Depth Attack Vulnerability**

| Agent | Baseline Error Rate (%) | Drift Rate DR (50% poll.) | Baseline Entry Rate (%) | Entry Rate ER (%) |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 4.7 | 58.0 | 0.0 | 94.6 |
| Claude-3.5-Sonnet | 8.0 | 36.2 | 0.0 | 25.3 |
| Llama-3-70B | 5.4 | 55.3 | 0.0 | 5.6† |
| Qwen2.5-72B | 6.8 | 76.2 | 0.0 | 96.1 |
| DeepSeek-V3 | 14.7 | 66.2 | 0.0 | 74.7 |

### Ablation Study

**Effect of Style on Drift Rate**

| Style | Mean Drift Rate (%) |
| :--- | :--- |
| Wire (Neutral) | 54.8 |
| Professor (Academic) | 42.4 |
| Rumor (Vague) | 36.9 |

### Key Findings

*   **Robustness Splitting**: Resistance to one type of attack often increases vulnerability to another. Claude was strongest against breadth attacks (lowest DR=36.2%) and also performed well in depth (ER=25.3%); GPT-4o was moderate in breadth but extremely poor in depth (ER=94.6%).
*   **Neutral tone is most persuasive** (Wire 54.8% > Professor 42.4% > Rumor 36.9%)—Agents are trained to distrust overly authoritative content but accept neutral statements uncritically.
*   **Pollution saturates at 30%** (40.2% → 55.8%), with little gain when increased to 50% (57.9%). Attackers only need a small amount of poisoning.
*   **Trapped Agents waste 44-73% of their step budget**, regardless of loop length—short loops are equally fatal.

## Highlights & Insights

*   Depth attacks represent a fundamentally new attack surface that requires no content modification, only structural modification of the information graph. This means all current RAG defense schemes based on content detection are completely ineffective against depth attacks.
*   "The Punishment of Honesty" is a disturbing finding—standard formulations in scientific literature (e.g., "results suggest") are perceived by Agents as signals of untrustworthiness, directly harming Agent credibility in academic/medical scenarios.
*   The parallel design of credibility gradients is a methodological highlight, making breadth and depth attacks comparable along the same axis of authoritative cues.

## Limitations & Future Work

*   The experimental scope is limited to citation graph navigation; generalization to other tool domains (fact-checking, Graph RAG poisoning) is ongoing.
*   Llama-3's low entry rate reflects a lack of tool engagement rather than genuine robustness.
*   Latest reasoning models like o3 and Claude 4 were not tested.
*   Exploration of defense strategies is insufficient—only the problem was diagnosed; no mitigation schemes were proposed.

## Related Work & Insights

*   **vs. PoisonedRAG**: Only covers content poisoning (breadth attack); this paper adds the dimension of structural attacks (depth attack).
*   **vs. Prompt Injection**: Different attack points—prompt injection modifies instructions, while AEI modifies environmental observations.
*   **vs. ToolBench/APIBench**: These evaluate capability rather than skepticism; this paper fills the gap in evaluating "Agent skepticism."

## Rating

*   **Novelty**: ⭐⭐⭐⭐⭐ Depth attacks are a new category; robustness splitting is a significant finding.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 11,000+ runs, 5 Agents, 7 experiments, complete statistical testing.
*   **Writing Quality**: ⭐⭐⭐⭐⭐ The Truman metaphor persists throughout; the narrative is engaging and rigorous.
*   **Value**: ⭐⭐⭐⭐⭐ Paradigm-level significance for Agent safety research; the Potemkin framework is reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents](magma_a_multi-graph_based_agentic_memory_architecture_for_ai_agents.md)
- [\[ACL 2026\] FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments](fama_failure-aware_meta-agentic_framework_for_open-source_llms_in_interactive_to.md)
- [\[ICML 2026\] Position: Agentic AI Orchestration Should Be Bayes-Consistent](../../ICML2026/llm_agent/position_agentic_ai_orchestration_should_be_bayes-consistent.md)
- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](../../ICLR2026/llm_agent/sr-scientist_scientific_equation_discovery_with_agentic_ai.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)

</div>

<!-- RELATED:END -->
