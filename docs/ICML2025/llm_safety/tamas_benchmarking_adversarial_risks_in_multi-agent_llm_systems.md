---
title: >-
  [Paper Note] TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems
description: >-
  [ICML 2025][LLM Safety][Multi-agent safety] This paper proposes TAMAS, the first safety benchmark systematically evaluating multi-agent LLM systems. Spanning 5 high-risk domains, 6 attack types, 300 adversarial samples, and 10 backbone models, TAMAS reveals severe adversarial vulnerabilities in multi-agent collaboration and introduces the ERS metric to quantify safety-utility trade-offs.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "Multi-agent safety"
  - "Adversarial attacks"
  - "LLM Agent"
  - "Safety benchmark"
  - "Robustness evaluation"
date: 2026-05-08
content_hash: a4c24551b29ce5f4
---

# TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems

**Conference**: ICML 2025  
**arXiv**: [2511.05269](https://arxiv.org/abs/2511.05269)  
**Code**: [https://github.com/microsoft/TAMAS](https://github.com/microsoft/TAMAS)  
**Area**: Agent  
**Keywords**: Multi-agent safety, Adversarial attacks, LLM Agent, Safety benchmark, Robustness evaluation

## TL;DR

This paper proposes TAMAS, the first safety benchmark systematically evaluating multi-agent LLM systems. Spanning 5 high-risk domains, 6 attack types, 300 adversarial samples, and 10 backbone models, TAMAS reveals severe adversarial vulnerabilities in multi-agent collaboration and introduces the ERS metric to quantify safety-utility trade-offs.

## Background & Motivation

**Background**: LLMs are rapidly evolving from text generators into autonomous agents capable of tool use, planning, and decision-making, and are widely deployed in high-risk scenarios such as financial trading, clinical decision support, and legal analysis. To tackle complex tasks, Multi-Agent Systems (MAS) have emerged, wherein multiple agents collaborate by task allocation based on their expertise.

**Limitations of Prior Work**: Existing safety benchmarks (e.g., InjectAgent, AgentDojo, RedCode, ASB) focus almost exclusively on single-agent settings, assessing only isolated prompt injection or code execution risks. They fail to capture emergent safety risks unique to multi-agent collaboration—such as collusion, contradiction, or Byzantine behavior among agents—which do not exist in single-agent environments.

**Key Challenge**: Multi-agent systems introduce more interactive components, expanding the attack surface (prompt level, environment level, agent level), yet safety research remains confined to the single-agent paradigm. Existing evaluations often use the ReAct framework to simulate interaction trajectories, simplifying agent behaviors and coordination mechanisms, which fails to reflect actual deployments in frameworks like AutoGen and CrewAI.

**Goal**: (a) How to systematically cover multi-agent-specific attack types? (b) How do different agent interaction architectures (centralized vs. decentralized) affect safety? (c) What is the disparity between open-source and closed-source models in multi-agent safety scenarios?

**Key Insight**: The authors start from the threat model, dividing the attack surface into three levels (prompt-level, environment-level, and agent-level) and designing six attack types to comprehensively cover the vulnerabilities of multi-agent systems.

**Core Idea**: Construct TAMAS, the first multi-agent LLM safety benchmark, to systematically evaluate the performance of 6 attack types across 3 interaction configurations and 10 models, and propose the ERS metric to quantify the safety-utility trade-off.

## Method

### Overall Architecture

The evaluation process of TAMAS is as follows: first, a multi-agent system containing 4 specialized agents is constructed for each of the 5 domains (education, law, finance, medicine, news), with each agent equipped with its role description and toolset. Then, 6 types of attacks (10 samples each) are applied to each scenario, and 10 backbone LLMs are evaluated across 3 agent interaction configurations (Central Orchestrator, Sequential, and Collaborative). The outputs are evaluated through the ARIA framework and tool-call validation, ultimately measuring system performance using Safety Score, Performance with No Attack (PNA), and Effective Robustness Score (ERS).

### Key Designs

1. **Hierarchical Design of Six Attack Types**

    - **Function**: Covers prompt-level (DPI, Impersonation), environment-level (IPI), and agent-level (Byzantine, Colluding, Contradicting) attacks.
    - **Mechanism**: DPI directly appends the malicious instruction $x^e$ to the user query $q^t \oplus x^e$ along with an adversarial toolset $T^e$. Impersonation appends a forged authoritative source claim $x^{\text{auth}}$. IPI injects malicious content into environmental observations $O \oplus x^e$ (e.g., tool outputs). Byzantine Agents tamper with system prompts $p_j^{\text{sys}} + \delta_j$ to produce inconsistent or meaningless outputs. Colluding Agents coordinate multiple malicious agents ($\mathcal{C} \subset \mathcal{M}$) to collectively achieve an adversarial goal. Contradicting Agents introduce functionally similar agents that output contradicting information to disrupt normal execution workflow.
    - **Design Motivation**: Existing benchmarks only cover single-agent attacks like DPI/IPI, overlooking emergent risks in multi-agent collaboration (collusion, contradiction, Byzantine), which exploit trust relationships and coordination mechanisms among agents.

2. **Three Agent Interaction Configurations**

    - **Function**: Implements three interaction paradigms across two frameworks: AutoGen (Magentic-One, RoundRobin, Swarm) and CrewAI (Centralized, Sequential).
    - **Mechanism**: Under the Central Orchestrator, an orchestrator is responsible for task decomposition and allocation, maintaining global progress. Sequential (RoundRobin) employs decentralized coordination with a fixed round-robin speaking order. Collaborative (Swarm) dynamically selects the next agent based on a handoff mechanism and shares the message context.
    - **Design Motivation**: Different architecture choices significantly affect the attack exposure of the system. Centralized orchestration offers global monitoring but is vulnerable to single-point failures; decentralized configurations eliminate single-point failures but lack unified safety oversight.

3. **ARIA Four-Level Evaluation Framework**

    - **Function**: Categorizes system safety responses into four levels: ARIA-1 (immediate refusal), ARIA-2 (delayed refusal), ARIA-3 (intent completed but failed), and ARIA-4 (successful attack).
    - **Mechanism**: For attacks like DPI/IPI/Impersonation, ARIA-4 scores are determined by parsing tool calls in logs. For semantically complex scenarios (e.g., Byzantine/Contradicting), GPT-4o acts as an LLM-as-Judge (temperature=0), validated against 140 human annotations (achieving an average F1 score of 89.17%).
    - **Design Motivation**: Attack success in multi-agent scenarios cannot be determined solely by rule matching and requires semantic-level reasoning, while also needing scalability in evaluation.

4. **ERS (Effective Robustness Score)**

    - **Function**: Proposes a comprehensive metric to jointly evaluate safety and utility.
    - **Mechanism**: The safety score for each attack is first calculated as $\text{Safety Score} = \text{ARIA}_1 + 0.5 \times \text{ARIA}_2 - 0.5 \times \text{ARIA}_3 - \text{ARIA}_4$, normalized to $[0, 100]$. The overall safety score is obtained by averaging across all attack types. Finally, the harmonic mean of the overall safety score and PNA (Performance with No Attack) is calculated: 
    $$\text{ERS} = \frac{2 \cdot \text{Safety}_{\text{overall}} \cdot \text{PNA}}{\text{Safety}_{\text{overall}} + \text{PNA}}$$
    - **Design Motivation**: A system that rejects all requests might score high in safety but offers zero utility; conversely, a system that robustly executes everything without safety alignment lacks security. The harmonic mean penalizes imbalances between these two aspects, preventing extreme trade-offs.

### Dataset Construction

Each scenario contains 60 adversarial samples (6 attack types $\times$ 10 samples) and 20 benign tasks. Across 5 scenarios, this totals 300 adversarial instances and 100 benign tasks. Agent roles and tools are manually designed, whereas user queries and attack tools are generated with the assistance of ChatGPT and subsequently verified by human annotation. Tools are classified into normal tools (for task execution) and attack tools (modeled for malicious actions), totaling 211 tools.

## Key Experimental Results

### Main Results

| Model | Magentic-One Safety/PNA/ERS | RoundRobin Safety/PNA/ERS | Swarm Safety/PNA/ERS | CrewAI-Central Safety/PNA/ERS | CrewAI-Decentral Safety/PNA/ERS |
|------|------|------|------|------|------|
| GPT-4 | 35.4 / 69.0 / 46.8 | 32.0 / 31.0 / 31.5 | 36.7 / 42.0 / 39.2 | — | — |
| GPT-4o | 36.5 / 79.0 / 50.0 | 25.3 / 49.0 / 33.4 | 34.0 / 44.0 / 38.4 | 41.7 / 79.2 / 54.6 | 37.5 / 85.4 / 52.1 |
| GPT-4o-mini | 41.2 / 76.0 / 53.4 | 29.5 / 45.0 / 35.6 | 25.8 / 42.0 / 32.0 | 35.0 / 80.3 / 48.8 | 34.8 / 82.4 / 48.9 |
| Gemini-2.0-Flash | 32.2 / 44.0 / 37.2 | 37.5 / 64.0 / 47.3 | 43.6 / 60.0 / 50.5 | — | — |
| Llama-3.1-8B | 32.3 / 26.1 / 28.9 | 13.9 / 57.0 / 22.3 | 15.2 / 31.5 / 20.5 | 76.9 / 58.0 / 66.1 | 91.5 / 72.2 / 80.7 |
| Qwen3-32B | 25.9 / 44.5 / 32.7 | 13.3 / 59.2 / 21.7 | 28.2 / 52.3 / 36.6 | 20.5 / 77.5 / 32.4 | 18.7 / 75.8 / 30.0 |

### Attack Success Rates (ARIA-4) of Each Attack Type

| Attack Type | Magentic-One Avg. | RoundRobin Avg. | Swarm Avg. | Attack Characteristics |
|----------|-------------------|-----------------|------------|----------|
| DPI | ~76-81% | ~70%+ | ~75%+ | One of the most effective attacks |
| Impersonation | ~72-82% | ~70%+ | ~82% (highest in Swarm) | Exploits authoritative trust, universally harmful across models |
| IPI | ~15-39% (low for closed-source) | ~38-75% (high for open-source) | ~30-56% | Highly dependent on configurations and models |
| Byzantine | High (~30-60%) | Medium-High | Medium-High | Directly degrades output quality |
| Colluding | 2-16% | Low | Low | Fully coordinated collusion is highly difficult |
| Contradicting | ~6% | Highly variable | Highly variable | Disrupts decision-making flow |

### Key Findings

- **Prompt-level attacks are highly effective**: DPI and Impersonation achieve over 70% attack success rates in nearly all configurations, regardless of whether they are open-source or closed-source models, indicating model-agnostic, general vulnerabilities.
- **IPI attacks show significant model polarity**: Closed-source models (e.g., GPT-4o, Gemini) are far more resistant to IPI than open-source models (under Magentic-One, the average ARIA-4 for closed-source models is 15.6%, compared to 39.2% for open-source models).
- **CrewAI framework is inherently safer overall**: Safety scores on CrewAI configurations are significantly higher than their AutoGen counterparts, largely due to CrewAI's pre-allocated task assignment design, which contrasts with AutoGen's dynamic execution.
- **Llama-3.1-8B performs exceptionally in CrewAI**: It achieves an ERS of up to 80.7 in the decentralized CrewAI configuration, primarily due to an extremely high refusal rate ($\text{Safety}=91.5$). However, this also indicates potential over-refusal issues.
- **The "Single Agent Breakthrough" phenomenon of Colluding attacks**: Although the success rate of full coordination is only 2-16%, the proportion of at least one agent executing malicious operations is up to 10-48%, showing that the partial effect of colluding attacks is much greater than overall completion rates suggest.

## Highlights & Insights

- **First Safety Benchmark for Multi-Agent Systems**: TAMAS addresses the gap where safety evaluations only targeted single agents. Specifically, the three agent-level attacks (Colluding, Contradicting, and Byzantine) are newly defined threat models, providing a standardized evaluation platform for multi-agent safety research.
- **Well-designed ERS Metric**: Employing the harmonic mean to jointly evaluate safety and utility avoids deceptive high scores from a "reject-all" or "accept-all" naive baseline. This design methodology can be extended to any systems-level evaluation requiring safety-utility trade-offs.
- **Alarming finding of "Aware of Maliciousness, yet Executed"**: The paper discovered that some agents explicitly recognized a request as malicious (e.g., deleting all test records) but executed it anyway. This indicates that current LLM safety alignment is severely deficient in agent execution scenarios; identification $\neq$ refusal.
- **Architectural Choices Deeply Impact Safety**: Centralized orchestration facilitates global monitoring but risks a single point of failure (if the orchestrator is compromised, the entire system collapses); decentralized configurations lack centralized oversight but avoid single points of failure. This provides architectural guidance for multi-agent safety-oriented framework designs.

## Limitations & Future Work

- **Limited Scenario Scale**: Compiling only 5 domains, with 10 samples per attack type for a total of 300 adversarial instances, limits statistical power given the vast combinations of models and configurations.
- **Synthetic Tool Usage**: All tools are mocked rather than actual APIs. The agents do not execute actions in physical or real-world environments (e.g., actual databases or payment systems), potentializing a gap with real-world deployment risks.
- **Lack of Defense Evaluation**: The paper concentrates exclusively on attack evaluations without proposing or testing defense strategies (e.g., safety filters, inter-agent auditing, output validation), highlighting the need for an interactive red-teaming/evaluation framework.
- **Exclusion of Persuasive Agent Attacks**: The authors excluded persuasive agent attacks simply because they proved entirely ineffective initially. However, a deeper root-cause analysis is absent; this might stem from unoptimized attack prompts rather than true LLM robustness against persuasion.
- **Evaluation GPT-4o Bias**: Relying on an LLM-as-Judge might introduce inherent bias, and the F1 score for certain attack categories (e.g., Contradicting Agents) hovers around 75%, making its reliability questionable.

## Related Work & Insights

- **vs. AgentDojo / InjectAgent**: These focus on prompt injection evaluations for single agents, whereas TAMAS expands to multi-agent interactions by defining novel attacks like Colluding, Byzantine, and Contradicting. TAMAS achieves broader coverage but trades off the depth of single-attack types.
- **vs. ASB (Agent Security Bench)**: ASB supports a wide range of attacks and defenses but is restricted to single agents; TAMAS serves as a complementary asset addressing the multi-agent dimension. Integrating both would build a more holistic agent safety evaluation ecosystem.
- **vs. AgentHarm**: AgentHarm monitors refusals of harmful queries in agents, while TAMAS explores system-level robustness against adversarial scenarios. Despite different perspectives, their findings reinforce each other—even when agents output explicit identification of malicious intent, they can still be manipulated down the execution chain.
- **Insights**: The ERS metric and ARIA framework of TAMAS are highly reusable for assessing safety-utility trade-offs in any agent system. Furthermore, trust propagation and message contamination across multi-agent environments warrant formal/mathematical study.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic safety benchmark for multi-agent LLM systems, with a clear hierarchical taxonomy of attacks, although the core technical contribution leans more toward evaluation rather than methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale evaluation across 10 models $\times$ 3 configurations $\times$ 6 attacks, but limited by only 10 samples per attack, leading to wider bootstrapped confidence intervals.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with abundant charts. The formal threat models are cleanly formulated, though some repetition exists.
- Value: ⭐⭐⭐⭐ Fills a crucial vacancy in multi-agent safety evaluations. Open-sourcing the TAMAS code and datasets, along with the reusable ERS metric, provides concrete value to prompt future community work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unveiling Privacy Risks in LLM Agent Memory](../../ACL2025/llm_safety/mextra_agent_memory_privacy.md)
- [\[ICLR 2026\] A2ASecBench: A Protocol-Aware Security Benchmark for Agent-to-Agent Multi-Agent Systems](../../ICLR2026/llm_safety/a2asecbench_a_protocol-aware_security_benchmark_for_agent-to-agent_multi-agent_s.md)
- [\[ICML 2025\] The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text](the_canarys_echo_auditing_privacy_risks_of_llm-generated_synthetic_text.md)
- [\[ICML 2025\] Robust Multi-bit Text Watermark with LLM-based Paraphrasers](robust_multi-bit_text_watermark_with_llm-based_paraphrasers.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](../../ACL2026/llm_safety/memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)

</div>

<!-- RELATED:END -->
