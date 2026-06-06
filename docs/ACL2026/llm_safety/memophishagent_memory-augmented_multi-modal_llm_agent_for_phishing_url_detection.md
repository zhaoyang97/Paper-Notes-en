---
title: >-
  [Paper Note] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection
description: >-
  [ACL 2026][LLM Safety][Phishing detection] Introduces MemoPhishAgent (MPA), the first memory-augmented multi-modal LLM agent specifically designed for phishing URL detection. By dynamically orchestrating 5 specialized to…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Phishing detection"
  - "LLM agent"
  - "episodic memory"
  - "multimodal reasoning"
  - "tool calling"
date: 2026-05-08
content_hash: 2249c295b9fb7efc
---

# MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection

**Conference**: ACL 2026  
**arXiv**: [2602.21394](https://arxiv.org/abs/2602.21394)  
**Code**: [GitHub](https://github.com/XuanChen-xc/MemoPhishAgent)  
**Area**: Security AI  
**Keywords**: Phishing detection, LLM agent, episodic memory, multimodal reasoning, tool calling

## TL;DR

Introduces MemoPhishAgent (MPA), the first memory-augmented multi-modal LLM agent specifically designed for phishing URL detection. By dynamically orchestrating 5 specialized tools and utilizing an episodic memory system to reuse historical reasoning trajectories, it achieves a 13.6% recall improvement on public benchmarks and a 20% increase on real-world social media data. It has been targets for production deployment, processing approximately 60,000 high-risk URLs weekly.

## Background & Motivation

**Background**: Phishing attacks continue to evolve. Traditional defenses (static blacklists, manual heuristics) lack coverage for new domains and techniques. Reference methods based on brand-domain mapping improve robustness but suffer from high maintenance costs and lag behind new brands and subdomains.

**Limitations of Prior Work**: (1) Existing LLM solutions are mostly prompting-based deterministic pipelines lacking adaptive evidence collection capabilities; (2) Tool usage follows fixed processes (e.g., OCR first, then brand matching, then domain verification), failing to adjust dynamically based on current evidence; (3) Lack of memory systems prevents the reuse of historical investigation expertise, leading to inefficiency in repeatedly analyzing similar phishing patterns.

**Key Challenge**: Phishing attacks are non-stationary—attackers constantly shift strategies—but defense systems are memoryless, starting each analysis from scratch.

**Goal**: Build a phishing detection agent capable of dynamically adjusting evidence collection strategies, learning from historical investigations, and suitable for production environments.

**Key Insight**: Modeling phishing detection as a multi-step reasoning process—simulating the investigative behavior of human experts and dynamically selecting tools to gather evidence.

**Core Idea**: 5 phishing-specific multi-modal tools + ReAct reasoning loop + episodic memory system (storing/retrieving historical trajectories). The combination enables adaptive and learnable phishing detection.

## Method

### Overall Architecture

MPA receives a list of suspicious URLs. Each URL is processed by the agent: (1) Dynamically selects 5 specialized tools to collect multi-modal evidence (text + vision + external knowledge); (2) Performs multi-step reasoning within a ReAct loop, determining the next action based on current evidence; (3) Utilizes episodic memory to retrieve similar historical cases to accelerate judgment or provide exemplars for guidance. Finally, it outputs "Malicious" or "Benign" classifications.

### Key Designs

1.  **5 Phishing-specific Tools**:
    *   **Function**: Provide complementary multi-modal evidence.
    *   **Mechanism**: Triple-aspect coverage: Multimodal evidence (`Crawl Content` for Markdown text + `Check Screenshot` for full-page visual analysis + `Check Image` for fine-grained visual inspection), external knowledge (`Intelligent Search` to construct evidence-driven queries for the latest info), and nested attack surfaces (`Extract Targets` to investigate redirection targets and sub-links deeply).
    *   **Design Motivation**: Generic tools are ill-suited for phishing scenarios; these 5 tools cover critical dimensions: text, vision, links, and external knowledge.

2.  **Episodic Memory System**:
    *   **Function**: Store, retrieve, and reuse historical reasoning trajectories.
    *   **Mechanism**: Uses LLMs to extract compact keywords (e.g., "apple login", "wallet connect") from pages, which are then indexed via vector embeddings. Retrieves top-k nearest neighbors and applies a three-level strategy: full ReAct loop for no matches, in-context exemplars for partial matches, and direct majority voting for full matches. Memory grows as deployment continues.
    *   **Design Motivation**: Phishing patterns involve significant repetition (same attack template targeting different victims). The memory system converts repetitive investigations into rapid decisions.

3.  **Three-level Memory Usage Strategy**:
    *   **Function**: Balances speed and reliability.
    *   **Mechanism**: $k'=0$ $\rightarrow$ Full reasoning (unseen patterns); $0 < k' < k$ $\rightarrow$ Historical trajectories as in-context exemplars (partial similarity); $k' \ge k$ $\rightarrow$ Direct majority voting (high similarity).
    *   **Design Motivation**: Prevent memory from dominating reasoning—it should serve as contextual guidance rather than a replacement for active thought.

## Key Experimental Results

### Main Results

| Method | TR-OP Recall | DynaPD Recall | Speed (s/URL) |
| :--- | :--- | :--- | :--- |
| MPA | **93.4%** | **93.6%** | **4.46** |
| PhishLLM | ~80% | ~88% | 14.2 |
| MLLM | ~82% | ~85% | 5.1 |
| URLTran | ~86% | — | 2.8 (with training) |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full MPA | 93.4% Recall | All components |
| - Memory System | -27% Recall | Memory is the largest contributor |
| - Tool Design | Performance Drop | Specialized tools outperform generic tools |
| Prompting Baseline | Poor | Fixed pipelines are inferior to adaptive selection |

### Key Findings
*   Episodic memory contributes up to a 27% increase in recall without adding extra computational overhead.
*   MPA is the fastest among LLM methods (4.46s/URL) because the memory system bypasses redundant analysis.
*   In real-world social media data (SocPhish), recall improved by 20%, showing greater advantages in authentic scenarios.
*   Production deployment handles ~60k high-risk URLs weekly, achieving a 91.44% recall.
*   URL shorteners and platform hosting paths (e.g., sites.google.com) are blind spots for traditional methods; MPA overcomes these via multi-modal tools.

## Highlights & Insights
*   **Production-Ready**: Not just academic work; it protects millions of users in Amazon's production environment, providing strong validation.
*   **Impressive Memory Performance**: A 27% recall boost with no computational cost—direct voting on repetitive patterns reduces LLM calls.
*   **Professional & Complementary Tool Design**: 5 tools collect evidence across text, vision, search, and links.
*   **Three-level Memory Strategy**: Balances efficiency and accuracy by fully analyzing new patterns while fast-tracking known ones.

## Limitations & Future Work
*   **Dependency on External LLM APIs**: Latency and costs associated with Claude-3-Sonnet.
*   **Potential Memory Pollution**: Early misjudgments stored in memory could affect subsequent decisions.
*   **Scoped to Phishing URLs**: Does not cover other security threats like malware distribution.
*   **Future Directions**: Self-correction mechanisms for memory, extension to diverse security threats, and replacing APIs with lightweight local models.

## Related Work & Insights
*   **vs. PhishLLM**: Uses LLMs for brand extraction and intent recognition but remains a fixed process; MPA dynamically selects tools.
*   **vs. Cao et al. (2025)**: Uses multi-modal LLMs for phishing detection but relies on fixed evidence acquisition without memory.
*   **vs. General Agent Frameworks**: Generic tools and reasoning are less effective than MPA's phishing-specific toolset.

## Rating
*   Novelty: ⭐⭐⭐⭐ First phishing-specific memory-augmented LLM Agent with a sophisticated episodic memory design.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across two public benchmarks, real social media data, and production deployment; comprehensive ablation study.
*   Writing Quality: ⭐⭐⭐⭐ Clear threat model definition and intuitive system architecture diagrams.
*   Value: ⭐⭐⭐⭐⭐ Validated in production, offering direct application value for security AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation](knowledge_poisoning_attacks_on_medical_multi-modal_retrieval-augmented_generatio.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ICML 2026\] SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](../../ICML2026/llm_safety/safeharbor_hierarchical_memory-augmented_guardrail_for_llm_agent_safety.md)
- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)
- [\[ACL 2026\] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)

</div>

<!-- RELATED:END -->
