---
title: >-
  [Paper Note] How Adversarial Environments Mislead Agentic AI
description: >-
  [ACL 2026][LLM Agent][Adversarial Environment Injection] This paper formalizes the Adversarial Environment Injection (AEI) threat model, decomposing it into breadth attacks (poisoning retrieval results to induce cognitiv…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Adversarial Environment Injection"
  - "Tool Trust Gap"
  - "Depth Attack"
  - "Breadth Attack"
  - "Robustness Splitting"
date: 2026-05-08
content_hash: e129817d7f900604
---

# How Adversarial Environments Mislead Agentic AI

**Conference**: ACL 2026
**arXiv**: [2604.18874](https://arxiv.org/abs/2604.18874)
**Code**: [GitHub](https://github.com/zhonghaozhan/Potemkin)
**Area**: AI Security / Agent Robustness
**Keywords**: Adversarial Environment Injection, Tool Trust Gap, Depth Attack, Breadth Attack, Robustness Splitting

## TL;DR

This paper formalizes the Adversarial Environment Injection (AEI) threat model, decomposing it into breadth attacks (poisoning retrieval results to induce cognitive drift) and depth attacks (injecting phantom nodes to construct navigational traps causing policy collapse). Across 11,000+ experimental runs, the two attack dimensions are found to be completely independent in terms of robustness — a phenomenon termed "robustness splitting" — demonstrating that current single-point defense strategies are fundamentally insufficient.

## Background & Motivation

**Background**: Tool-augmented LLM agents rely on external tools such as search engines and citation indices to ground their generated content. RAG security has become an active research area, with existing work focusing on prompt injection and corpus poisoning as content-level attacks.

**Limitations of Prior Work**: (1) Existing evaluations focus solely on whether agents can correctly use tools, never considering what happens when tools lie — a fundamental trust gap; (2) RAG poisoning research covers only half the attack surface (content level), neglecting structural attacks; (3) A standardized, reproducible adversarial robustness testing framework is lacking.

**Key Challenge**: The correct behavior for reducing hallucination (deferring to external information) is precisely what increases adversarial vulnerability — the "grounding paradox." Agents accept the reality presented by their environment and lack independent verification channels, akin to Truman living in a fabricated world.

**Goal**: (1) Formalize the complete attack surface facing tool-using agents; (2) Distinguish between two orthogonal attack dimensions — cognitive and navigational; (3) Quantify the independence of these two dimensions.

**Key Insight**: The paper draws an analogy to *The Truman Show* — agents accept tool-returned content as reality, and adversaries construct a false world via a "Man-in-the-Tool" proxy. Depth attacks constitute an entirely new attack category: they do not require agents to believe false information, only to become trapped in navigational loops.

**Core Idea**: AEI is decomposed into breadth attacks (cognitive drift) and depth attacks (policy collapse), which exploit fundamentally different mechanisms — the former attacks belief updating, the latter attacks navigational planning — such that defense against one provides no protection against the other.

## Method

### Overall Architecture

The Potemkin framework operates as a transparent Man-in-the-Tool proxy: it intercepts agent tool-call responses, applies adversarial transformations, and returns the modified results. It supports both MCP Server and Python Library modes. Breadth attacks operate by poisoning retrieval results; depth attacks operate by injecting phantom nodes into citation graphs. The framework encompasses 7 experiments across two campaigns, with approximately 11,000 task runs.

### Key Designs

1. **Breadth Attack (The Illusion)**:

    - **Function**: Induces cognitive drift in agents by poisoning retrieval results, causing them to accept false information.
    - **Mechanism**: Malicious text is injected into the knowledge base, with two controlled parameters: contamination rate $\rho \in \{0.1, 0.3, 0.5\}$ (1/3/5 out of 10 retrieved results) and linguistic style (Professor: formal academic; Wire: neutral journalistic; Rumor: informal and hedged). Performance is measured by drift rate $\text{DR} = \mathbb{E}[\mathbb{1}[r \neq y] | r \neq \bot]$ — abstentions are excluded so that only confident yet incorrect judgments are counted.
    - **Design Motivation**: Orthogonal variation of contamination rate and style allows the two questions — "how much poison is needed" and "what style of poison is most effective" — to be disentangled. The abstention-excluding DR metric captures changes in cognitive state more precisely than conventional ASR.

2. **Depth Attack (The Maze)**:

    - **Function**: Constructs navigational traps by injecting phantom citation nodes, exhausting the agent's step budget.
    - **Mechanism**: Phantom nodes $\mathcal{P}$ are injected into the information graph to form cycles or dead ends, with controlled cycle length $\ell \in \{2,3,5\}$ and credibility gradient (Phantom: high-fidelity; Signal: mildly inconsistent; Glitch: obviously anomalous). Performance is measured by entry rate $\text{ER} = \mathbb{E}[\mathbb{1}[\exists t: s_t \in \mathcal{P}]]$ and step budget waste $\text{BW} = \frac{|\{t: s_t \in \mathcal{P}\}|}{|\{t: s_t \in V \cup \mathcal{P}\}|}$.
    - **Design Motivation**: Depth attacks constitute a novel attack category — they modify only structure, not content, so agents can be trapped in loops even without believing any false information. The credibility gradient parallels the style gradient of breadth attacks, enabling cross-dimensional analysis.

3. **The Punishment of Honesty**:

    - **Function**: Reveals systematic miscalibration in agents' processing of epistemic markers.
    - **Mechanism**: Minimal pairs are constructed in which identical claims differ only in epistemic marker (e.g., "results suggest" vs. "results prove"), and McNemar's test is applied. TRUE statements containing hedging language are rejected at 2.1× the rate of confidently worded TRUE statements, whereas FALSE statements containing hedging language are not detected more easily.
    - **Design Motivation**: This exposes a dangerous asymmetry — adversaries can suppress accurate statements by adding hedging language, a finding with particular implications for scientific and medical domains.

### Loss & Training

Potemkin is an evaluation framework and involves no training. All evaluated agents run at temperature $T=0.0$ to ensure deterministic evaluation, with a step budget of 10 tool calls. Adversarial content is generated by Gemini 2.5 as a red-team model, avoiding generator–victim overlap.

## Key Experimental Results

### Main Results

**Breadth vs. Depth Attack Vulnerability**

| Agent | Baseline Error Rate (%) | Drift Rate DR (50% Contamination) | Baseline Entry Rate (%) | Entry Rate ER (%) |
|-------|------------------------|----------------------------------|------------------------|------------------|
| GPT-4o | 4.7 | 58.0 | 0.0 | 94.6 |
| Claude-3.5-Sonnet | 8.0 | 36.2 | 0.0 | 25.3 |
| Llama-3-70B | 5.4 | 55.3 | 0.0 | 5.6† |
| Qwen2.5-72B | 6.8 | 76.2 | 0.0 | 96.1 |
| DeepSeek-V3 | 14.7 | 66.2 | 0.0 | 74.7 |

### Ablation Study

**Effect of Style on Drift Rate**

| Style | Mean Drift Rate (%) |
|-------|-------------------|
| Wire (neutral) | 54.8 |
| Professor (academic) | 42.4 |
| Rumor (hedged) | 36.9 |

### Key Findings

- **Robustness splitting**: Resistance to one attack type often increases vulnerability to the other. Claude exhibits the strongest robustness to breadth attacks (lowest DR = 36.2%) and moderate robustness to depth attacks (ER = 25.3%); GPT-4o shows moderate breadth robustness but severe depth vulnerability (ER = 94.6%).
- Neutral tone is the most persuasive style (Wire 54.8% > Professor 42.4% > Rumor 36.9%) — agents are trained to distrust overly authoritative content but accept neutral statements uncritically.
- Contamination saturates at 30% (40.2% → 55.8%); increasing to 50% yields only marginal additional gain (57.9%), indicating that attackers require only modest levels of poisoning.
- Trapped agents waste 44–73% of their step budget regardless of cycle length — short cycles are equally lethal.

## Highlights & Insights

- Depth attacks constitute an entirely new attack surface — they require no content modification, only structural modification of the information graph. This means all existing content-detection-based RAG defenses are completely ineffective against depth attacks.
- The "Punishment of Honesty" is a particularly troubling finding: standard expressions in scientific literature (e.g., "results suggest") are interpreted by agents as signals of untrustworthiness, directly undermining agent reliability in academic and medical settings.
- The parallel credibility gradient design is a methodological strength — it renders breadth and depth attacks comparable along the same epistemic authority axis.

## Limitations & Future Work

- Experiments are limited to citation graph navigation tasks; generalization to other tool domains (fact-checking, graph RAG poisoning) is ongoing.
- Llama-3's low entry rate largely reflects insufficient tool engagement rather than genuine robustness.
- The latest reasoning-oriented models (e.g., o3, Claude 4) have not been evaluated.
- Defense strategies are underexplored — the paper diagnoses the problem but does not propose mitigation approaches.

## Related Work & Insights

- **vs. PoisonedRAG**: Covers only content poisoning (breadth attacks); this paper adds the structural attack (depth attack) dimension.
- **vs. Prompt Injection**: The attack surface differs — prompt injection modifies instructions, whereas AEI modifies environmental observations.
- **vs. ToolBench/APIBench**: These benchmarks evaluate capability, not skepticism; this paper fills the gap in evaluating agents' capacity for warranted distrust of tool outputs.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Depth attacks are a genuinely new attack category; robustness splitting is an important discovery.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 11,000+ runs, 5 agents, 7 experiments, complete statistical testing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The Truman Show metaphor is sustained throughout, yielding a narrative that is both compelling and rigorous.
- **Value**: ⭐⭐⭐⭐⭐ Paradigm-level significance for agent security research; the Potemkin framework is reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](../../ICLR2026/llm_agent/sr-scientist_scientific_equation_discovery_with_agentic_ai.md)
- [\[AAAI 2026\] With Great Capabilities Come Great Responsibilities: Introducing the Agentic Risk & Capability Framework for Governing Agentic AI Systems](../../AAAI2026/llm_agent/with_great_capabilities_come_great_responsibilities_introducing_the_agentic_risk.md)
- [\[ICLR 2026\] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](../../ICLR2026/llm_agent/gaia2_benchmarking_llm_agents_on_dynamic_and_asynchronous_environments.md)
- [\[ICLR 2026\] Toward a Dynamic Stackelberg Game-Theoretic Framework for Agentic AI Defense Against LLM Jailbreaking](../../ICLR2026/llm_agent/toward_a_dynamic_stackelberg_game-theoretic_framework_for_agentic_ai_defense_aga.md)
- [\[NeurIPS 2025\] PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer](../../NeurIPS2025/llm_agent/panda_towards_generalist_video_anomaly_detection_via_agentic_ai_engineer.md)

</div>

<!-- RELATED:END -->
