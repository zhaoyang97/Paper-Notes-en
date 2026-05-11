---
title: >-
  [Paper Note] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection
description: >-
  [ACL 2026][LLM Agent][phishing detection] This paper proposes MemoPhishAgent (MPA), the first memory-augmented multimodal LLM agent specifically designed for phishing URL detection. MPA dynamically orchestrates five dedi…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "phishing detection"
  - "episodic memory"
  - "multimodal reasoning"
  - "tool invocation"
date: 2026-05-08
content_hash: e42668bb6a54cc0a
---

# MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection

**Conference**: ACL 2026
**arXiv**: [2602.21394](https://arxiv.org/abs/2602.21394)
**Code**: [GitHub](https://github.com/XuanChen-xc/MemoPhishAgent)
**Area**: Security AI
**Keywords**: phishing detection, LLM agent, episodic memory, multimodal reasoning, tool invocation

## TL;DR

This paper proposes MemoPhishAgent (MPA), the first memory-augmented multimodal LLM agent specifically designed for phishing URL detection. MPA dynamically orchestrates five dedicated tools and leverages an episodic memory system to reuse historical reasoning trajectories. It achieves a 13.6% recall improvement on public benchmarks and a 20% improvement on real-world social media data, and has been deployed in production, processing approximately 60K high-risk URLs per week.

## Background & Motivation

**Background**: Phishing attacks continue to evolve, and traditional defenses (static blacklists, handcrafted heuristic rules) provide insufficient coverage against new domains and novel techniques. Reference-based methods using brand-domain mappings improve robustness but incur high maintenance costs and respond slowly to emerging brands and subdomains.

**Limitations of Prior Work**: (1) Existing LLM-based approaches are mostly prompt-driven deterministic pipelines lacking adaptive evidence collection; (2) Tool usage follows fixed workflows (e.g., OCR → brand matching → domain verification) and cannot dynamically adjust based on the current evidence state; (3) The absence of a memory system prevents reuse of historical investigation experience, leading to inefficient repeated analysis of similar phishing patterns.

**Key Challenge**: Phishing attacks are non-stationary — attackers continuously vary their strategies — yet defense systems are memoryless and restart analysis from scratch each time.

**Goal**: To construct a phishing detection agent capable of dynamically adjusting evidence collection strategies, learning from historical investigations, and operating effectively in production environments.

**Key Insight**: Phishing detection is modeled as a multi-step reasoning process that simulates the investigative behavior of human experts, dynamically selecting tools to gather evidence.

**Core Idea**: Five phishing-specific multimodal tools + a ReAct reasoning loop + an episodic memory system (storing and retrieving historical reasoning trajectories) are combined to achieve adaptive, continually improving phishing detection.

## Method

### Overall Architecture

MPA receives a list of suspicious URLs; each URL is processed by the agent as follows: (1) five dedicated tools are dynamically selected to collect multimodal evidence (text + visual + external knowledge); (2) multi-step reasoning is performed within a ReAct loop, determining the next action based on the current evidence state; (3) the episodic memory system retrieves similar historical cases to accelerate judgment or provide exemplar guidance. The final output is a "malicious" or "benign" verdict.

### Key Designs

1. **Five Phishing-Specific Tools**:

    - **Function**: Provide complementary multimodal evidence.
    - **Mechanism**: Three-dimensional coverage — multimodal evidence (Crawl Content for Markdown text extraction + Check Screenshot for full-page screenshot analysis + Check Image for fine-grained image inspection), external knowledge (Intelligent Search constructs evidence-driven search queries to retrieve up-to-date information), and nested attack surfaces (Extract Targets extracts redirect targets and sublinks for deep inspection).
    - **Design Motivation**: General-purpose tools are ill-suited to the phishing domain; the five tools collectively cover textual, visual, link, and external-knowledge dimensions.

2. **Episodic Memory System**:

    - **Function**: Store, retrieve, and reuse historical reasoning trajectories.
    - **Mechanism**: An LLM extracts compact keywords from pages (e.g., "apple login," "wallet connect"), which are embedded and indexed as vectors. The top-$k$ nearest neighbors are retrieved and used according to a three-tier strategy — full ReAct reasoning when no match is found, historical trajectories as in-context exemplars when a partial match exists, and direct majority voting when a full match is found. The memory grows richer as deployment continues.
    - **Design Motivation**: Phishing patterns are highly repetitive (the same attack template targets different victims); the memory system transforms repeated investigations into rapid decisions.

3. **Three-Tier Memory Usage Strategy**:

    - **Function**: Balance speed and reliability.
    - **Mechanism**: $k'=0$ → full reasoning (unseen pattern); $0 < k' < k$ → historical trajectories as in-context exemplars (partial similarity); $k' \geq k$ → direct majority voting (high similarity).
    - **Design Motivation**: Prevent memory from dominating reasoning — it should serve as contextual guidance rather than a substitute for deliberation.

## Key Experimental Results

### Main Results

| Method | TR-OP Recall | DynaPD Recall | Speed (s/URL) |
|--------|-------------|--------------|--------------|
| MPA | **93.4%** | **93.6%** | **4.46** |
| PhishLLM | ~80% | ~88% | 14.2 |
| MLLM | ~82% | ~85% | 5.1 |
| URLTran | ~86% | — | 2.8 (incl. training) |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Full MPA | 93.4% Recall | All components |
| w/o memory system | −27% Recall | Memory contributes most |
| w/o tool design | Performance drop | Specialized tools outperform generic ones |
| Prompt-based baseline | Inferior | Fixed pipeline underperforms adaptive selection |

### Key Findings
- The episodic memory system contributes up to 27% recall improvement without additional computational overhead.
- MPA achieves the fastest inference among all methods (4.46 s/URL), as the memory system bypasses redundant analysis for repeated patterns.
- A 20% recall improvement on real-world social media data (SocPhish) indicates that advantages are amplified in realistic settings.
- Production deployment processes ~60K high-risk URLs per week, achieving a 91.44% recall rate.
- URL shorteners and platform-hosted paths (e.g., sites.google.com) are blind spots for traditional methods; MPA overcomes these via multimodal tools.

## Highlights & Insights
- **Production deployment**: This is not merely an academic contribution — the system has been deployed in Amazon's production environment to protect millions of users, lending substantial credibility.
- **Remarkable impact of episodic memory**: A 27% recall gain with no added computation, achieved by reducing LLM calls through direct voting on repeated patterns.
- **Professional and complementary tool design**: The five tools collect evidence across textual, visual, search, and link dimensions.
- **Three-tier memory strategy balances efficiency and accuracy**: Full analysis for unseen patterns, rapid decisions for previously seen ones.

## Limitations & Future Work
- **Reliance on external LLM APIs**: Latency and cost associated with Claude-3-Sonnet.
- **Memory contamination risk**: Early misclassifications stored in memory may propagate errors to subsequent decisions.
- **Scope limited to phishing URLs**: Other security threats (e.g., malware distribution) are not covered.
- Future directions include memory self-correction mechanisms, extension to broader security threat types, and replacement of API-dependent models with lightweight local alternatives.

## Related Work & Insights
- **vs. PhishLLM**: Employs LLMs for brand extraction and intent recognition but still relies on a fixed pipeline; MPA selects tools dynamically.
- **vs. Cao et al. (2025)**: Uses multimodal LLMs for phishing detection but with a fixed evidence acquisition workflow and no memory component.
- **vs. General agent frameworks**: General-purpose tools and reasoning are less effective; MPA's phishing-specific tools yield superior results.

## Rating
- Novelty: ⭐⭐⭐⭐ First memory-augmented LLM agent tailored for phishing detection, with an elegantly designed episodic memory system.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on two public benchmarks, real-world social media data, and production deployment, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Threat model is clearly defined; system architecture diagrams are intuitive.
- Value: ⭐⭐⭐⭐⭐ Validated in a production environment; offers direct practical value for security AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](../../ICLR2026/llm_agent/exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)
- [\[ACL 2026\] HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents](hela-mem_hebbian_learning_and_associative_memory_for_llm_agents.md)
- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)

</div>

<!-- RELATED:END -->
