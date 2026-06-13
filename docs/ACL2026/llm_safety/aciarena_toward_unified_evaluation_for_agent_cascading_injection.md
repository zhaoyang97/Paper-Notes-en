---
title: >-
  [Paper Note] ACIArena: Toward Unified Evaluation for Agent Cascading Injection
description: >-
  [ACL 2026][LLM Safety][Multi-Agent Systems] This paper constructs ACIArena, the first unified evaluation framework for "Agent Cascading Injection (ACI)" attacks, covering 6 mainstream Multi-Agent Systems (MAS)…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Multi-Agent Systems"
  - "Cascading Injection"
  - "ACI Attack"
  - "MAS Robustness"
  - "ACI-Sentinel"
date: 2026-05-08
content_hash: 37a5cf501e31fe52
---

# ACIArena: Toward Unified Evaluation for Agent Cascading Injection

**Conference**: ACL 2026  
**arXiv**: [2604.07775](https://arxiv.org/abs/2604.07775)  
**Code**: https://github.com/Greysahy/aciarena  
**Area**: LLM Reasoning / Multi-Agent Security  
**Keywords**: Multi-Agent Systems, Cascading Injection, ACI Attack, MAS Robustness, ACI-Sentinel

## TL;DR
This paper constructs ACIArena, the first unified evaluation framework for "Agent Cascading Injection (ACI)" attacks, covering 6 mainstream Multi-Agent Systems (MAS), 3 attack surfaces (Adversarial Input / Malicious Agent / Message Poison), and 3 attack goals (Hijacking / Disruption / Exfiltration) with 1,356 test cases. It also proposes ACI-Sentinel, a minimalist yet effective defense that reduces the Hijacking attack success rate from 92.78% to 8.06%.

## Background & Motivation
**Background**: LLM multi-agent systems such as MetaGPT, AutoGen, CAMEL, and AgentVerse have been widely adopted by industrial products like Cursor and Salesforce Agentforce, enhancing performance in complex tasks (programming, mathematical reasoning) through expert division of labor and A2A protocols.

**Limitations of Prior Work**: MAS amplify the hazards of prompt injection through extensive inter-agent messaging—a compromised agent can cascade malicious instructions throughout the system via peer trust, a phenomenon the authors name **Agent Cascading Injection (ACI)**. Current research suffers from three flaws: (1) **Incomplete threat surfaces**: Existing works focus either only on profiles or only on messages, with goals limited to system paralysis or privacy leakage; (2) **Non-standard evaluation settings**: Many studies use simplified, self-built MAS, making horizontal comparisons impossible; (3) **Non-extensible codebases**: Frameworks like MASLab provide only a unified entry point, lacking attack and defense modules.

**Key Challenge**: Studying MAS security requires simultaneous control over the MAS implementation, attack strategy, and attack surface. However, prior works typically modify only one variable within a custom environment, rendering results non-transferable.

**Goal**: To establish an MAS robustness evaluation framework that is (i) comprehensive in attack surfaces and goals, (ii) standardized, and (iii) modularly extensible.

**Key Insight**: Starting from the formal agent definition $\mathcal{A} = (\pi, \mathcal{P}, \mathcal{M}, \mathcal{T})$, the authors enumerate all components susceptible to injection (instructions $\mathcal{I}$, profiles $\mathcal{P}$, memory $\mathcal{M}$, tool descriptions $\mathcal{T}$, and message edges $\mathcal{E}$), categorizing all ACI attacks into 3 surfaces crossed with 3 goals.

**Core Idea**: Transforming MAS robustness research into a horizontally comparable scientific experiment using a "Surface × Goal" 2D matrix combined with standardized MAS/attack-defense interfaces.

## Method

### Overall Architecture
ACIArena consists of four modules: **Benign Tasks** (filtered via LLM judge from GSM8K, MATH500, HumanEval, etc., based on difficulty and decomposability); **Attacks** (28 ACI attacks covering 3 surfaces × 3 goals, automatically optimized through a generate-mutate-select loop); **MAS Library** (6 MAS refactored into a unified interface); and **Evaluation Suites** (1,356 test cases with BU/ASR/UA/PVI metrics). During execution, an attacker injects a malicious prompt into a specified surface to observe cascading propagation and final output.

### Key Designs
1.  **Three-axis Threat Model and Attack Generator**:
    - **Function**: Formalizes ACI attacks as a three-axis combination of "Surface × Goal × MAS" and uses an LLM to automatically generate attack prompts.
    - **Mechanism**: The three surfaces correspond to three mathematical forms: **Adversarial Input** injects any input component $\mathcal{I}/\mathcal{M}/\mathcal{T}$; **Malicious Agent** tampers with profile $\mathcal{P}$ to make an agent autonomously output malicious messages; **Message Poison** intercepts and replaces messages on communication edges $(\mathcal{A}_i, \mathcal{A}_j) \in \mathcal{E}$. The three goals are Hijacking, Disruption, and Exfiltration. Prompts are optimized via a generate-mutate-select loop: starting from a manual seed $a_0$, mutations $\omega \in \Omega$ are sampled to generate $a' = \omega(a_t)$. These are executed across $N$ MAS, and an LLM judge scores them based on stealthiness (similarity to benign prompts) and harmfulness (alignment with the original malicious goal).
    - **Design Motivation**: Manual prompt engineering is slow and fails to cover all patterns; the automated loop allows the framework to adapt quickly to new MAS and models.

2.  **Propagation Vulnerability Index (PVI) and Fine-grained Agent-level Analysis**:
    - **Function**: Quantifies the intensity of malicious information cascading within the system, beyond the final ASR.
    - **Mechanism**: Defined as $\mathrm{PVI} = \sum_{a_i \in \mathcal{A}} \frac{L_{a_i}}{\sum_{a_j \in \mathcal{A}} L_{a_j}} \mathrm{ASR}_{a_i}$, where $L_{a_i}$ is the minimum topological distance from agent $a_i$ to the final response, and $\mathrm{ASR}_{a_i}$ is the attack success rate when that agent is the entry point. Higher PVI indicates stronger system "infectivity."
    - **Design Motivation**: Final ASR ignores cases where local success is corrected downstream or where an attack penetrates multiple layers. PVI extends evaluation from the "output layer" to the "process layer."

3.  **ACI-Sentinel: Task-aligned Semantic Minimality Defense**:
    - **Function**: Proposes a stable defense in a context where existing defenses often fail or even amplify attacks.
    - **Mechanism**: Existing defenses (BERT detector, Delimiter, Sandwich, AGrail, G-Safeguard) attempt to identify "suspicious messages," which is difficult for ACI attacks disguised as normal outputs. ACI-Sentinel takes the opposite approach: **instead of identifying the bad, it enforces the good**. It enumerates the minimum necessary information aligned with the current task (task-aligned semantic minimality) and strips away all extraneous instructions and metadata.
    - **Design Motivation**: Large-scale observations show that attack patterns involve embedding extra instructions in valid messages. Compressing messages to the semantic minimum automatically eliminates injections. This approach reduced Hijacking ASR on AutoGen from 92.78% to 8.06%.

### Loss & Training
The optimization objective for attack generation is $J(a') = J_{\text{stealth}}(a' | c) + \frac{1}{N}\sum_{j=1}^N J_{\text{harm}}(\mathcal{S}^{(j)}(a'), a_0)$, where both terms are scored by an LLM judge (black-box optimization). No new training was performed for the MAS or defenses themselves.

## Key Experimental Results

### Main Results: Robustness of 6 MAS across 3 Attack Goals (GPT-4o-mini, Math/Code domains)

| Domain | MAS | BU | Hijacking ASR | Disruption ASR | Exfiltration ASR |
|----|-----|-----|---------------|----------------|------------------|
| Math | CAMEL | 41.0% | 7.05% | 37.44% | 22.56% |
| Math | AutoGen | 72.7% | 19.23% | 52.65% | 48.38% |
| Math | AgentVerse | 74.4% | 26.71% | 54.70% | 40.51% |
| Math | Self Consistency | 73.5% | 27.99% | 74.53% | 43.59% |
| Math | LLM Debate | 69.2% | 16.88% | 64.79% | 57.27% |
| Code | CAMEL | 14.4% | 20.28% | 59.11% | 26.00% |
| Code | AutoGen | 51.1% | 80.83% | 90.89% | 77.55% |
| Code | AgentVerse | 57.8% | 48.05% | 45.78% | 80.45% |
| Code | MetaGPT | 51.1% | 100.00% | 88.89% | 80.22% |
| Code | Self Consistency | 52.8% | 95.00% | 76.89% | 80.00% |
| Code | LLM Debate | 54.4% | 100.00% | 86.67% | 80.22% |

### Ablation Study: Comparison of 6 Defenses on AutoGen

| Defense | BU Retention | Hijacking ASR | Disruption ASR | Exfiltration ASR |
|------|---------|---------------|----------------|------------------|
| No Defense (Baseline) | 57.78% | 92.78% | 96.44% | 54.00% |
| +BERT Detector | 45.56% | 96.39% (Increase) | 99.78% (Increase) | 36.67% |
| +Delimiter | 55.56% | 95.56% | 96.67% | 44.22% |
| +Sandwich | 66.67% | 79.72% | 78.67% | 60.00% |
| +AGrail | 32.22% | 35.56% (UA Drop) | 96.44% | 29.33% |
| +G-Safeguard | 40.00% | 67.22% (UA Drop) | 96.44% | 34.00% |
| **+ACI-Sentinel** | 52.22% | **8.06%** | 82.89% | **0.22%** |

### Key Findings
-   **Topology alone does not explain robustness**: Even with 5 agents, AgentVerse and CAMEL show massive differences in robustness; changing agent profiles within the same topology leads to huge ASR fluctuations, refuting the practice of evaluating security based solely on topology.
-   **Simple topologies are more fragile**: Topologies with local views like MetaGPT and Self Consistency exhibit nearly 100% ASR in Hijacking due to implicit trust in executing malicious instructions.
-   **Utility-security trade-off is prevalent**: CAMEL's low Hijacking ASR is coupled with extremely low UA (7% in some cases)—the attack didn't "fail to bypass defense," but rather the system failed to execute anything effectively.
-   **Code generation is a high-risk area**: Multiple MAS suffer 90-100% Hijacking ASR in the Code domain because code is executable and complex, allowing malicious instructions to hide easily.
-   **Existing defenses often fail or backfire**: BERT Detector increased ASR in some scenarios, while AGrail/G-Safeguard severely degraded system utility, showing that simplified-environment defenses do not transfer to real MAS.

## Highlights & Insights
-   **Engineering contribution of unified interfaces**: Refactoring 6 heterogeneous MAS into a single entry point is an enabler for future research.
-   **PVI incorporates process into evaluation**: Unlike final ASR, PVI reveals the intensity of agent-level propagation.
-   **Anti-intuitive insight of ACI-Sentinel**: Shifting the goal from "identifying bad" to "preserving good" bypasses the cat-and-mouse game of detection.
-   **Early warning for code generation**: Data highlights the vulnerability of MAS in code scenarios, offering direct implications for products like Cursor.

## Limitations & Future Work
-   **Limitations**: (1) Assumes a single malicious agent under Byzantine Fault Tolerance; (2) Relies on LLM judges which may have evaluation bias; (3) Evaluated mainly on GPT-4o-mini/Qwen2.5-7B.
-   **Future Work**: (i) Extend to collaborative attacks with multiple compromised agents; (ii) Develop formal verification for "task semantic envelopes"; (iii) Explore active defense based on dynamic topological pruning.

## Related Work & Insights
-   **vs AgentDojo / Agent Security Bench**: These focus on single-agent setups; this work is the first to systematically cover internal cascading in MAS.
-   **vs Corba**: While they focus on contagious recursive blocking, this work integrates such attacks into a larger ACI framework.
-   **vs G-Safeguard / AGrail**: This work quantifies the failure modes of these defenses in real MAS (where functional loss outweighs security gain).

## Rating
-   Novelty: ⭐⭐⭐⭐⭐ "Agent Cascading Injection" as a unified concept is original.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale evaluation across MAS, domains, goals, and LLMs.
-   Writing Quality: ⭐⭐⭐⭐ Clear formal definitions; dense tables require careful reading.
-   Value: ⭐⭐⭐⭐⭐ An immediately usable testbed for MAS deployment teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PIArena: A Platform for Prompt Injection Evaluation](piarena_a_platform_for_prompt_injection_evaluation.md)
- [\[ACL 2026\] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models](muse_a_run-centric_platform_for_multimodal_unified_safety_evaluation_of_large_la.md)
- [\[ICML 2026\] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](../../ICML2026/llm_safety/bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)
- [\[ACL 2026\] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)

</div>

<!-- RELATED:END -->
