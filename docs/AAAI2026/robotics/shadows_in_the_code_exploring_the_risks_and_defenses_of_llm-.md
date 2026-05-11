---
title: >-
  [Paper Note] Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems
description: >-
  [AAAI 2026][Robotics][Multi-Agent Security] The first systematic security analysis of LLM-based multi-agent software development systems (ChatDev/MetaGPT/AgentVerse): proposes the IMBIA attack framework covering two thre…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "Multi-Agent Security"
  - "Malicious Code Injection"
  - "Software Development Agents"
  - "Adversarial Defense"
  - "Malware Families"
date: 2026-05-08
content_hash: 2b3bc7e667d5cfd3
---

# Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems

**Conference**: AAAI 2026
**arXiv**: [2511.18467](https://arxiv.org/abs/2511.18467)
**Code**: [https://github.com/wxqkk0808/IMBIA](https://github.com/wxqkk0808/IMBIA)
**Area**: Robotics
**Keywords**: Multi-Agent Security, Malicious Code Injection, Software Development Agents, Adversarial Defense, Malware Families

## TL;DR
The first systematic security analysis of LLM-based multi-agent software development systems (ChatDev/MetaGPT/AgentVerse): proposes the IMBIA attack framework covering two threat scenarios (malicious user + benign agents / benign user + malicious agent) and 12 malicious behaviors across 5 malware families, achieving an attack success rate (ASR) of up to 93% on ChatDev, with the Adv-IMBIA adversarial defense reducing ASR by 40–73%.

## Background & Motivation

**Background**: LLM-based multi-agent software development systems (e.g., ChatDev, MetaGPT, AgentVerse) enable non-technical users to generate complete software via natural language and have attracted widespread attention.

**Limitations of Prior Work**: The security of these systems remains almost entirely unstudied—existing work focuses only on single-agent code security or general LLM agent benchmarks, without systematically analyzing the security risks of end-to-end multi-agent software development systems.

**Key Challenge**: The powerful capabilities of multi-agent systems are a double-edged sword—they can produce high-quality software but may equally be exploited to generate software containing hidden malicious code. Two threat vectors exist: malicious users inducing benign agents to generate harmful code, or benign users interacting with compromised agents.

**Key Insight**: Two threat scenarios are defined (MU-BA and BU-MA), 12 attack behaviors spanning 5 major malware families are designed, and systematic evaluation is conducted across three mainstream frameworks.

**Core Idea**: The first systematic attack-and-defense study of LLM-based multi-agent software development systems—combining IMBIA implicit malicious injection with Adv-IMBIA adversarial prompt-based defense.

## Method

### Attack: IMBIA

**Two Threat Scenarios**:
- **MU-BA (Malicious User → Benign Agents)**: A malicious input $P_m$ is appended to a benign requirement $P_b$.
- **BU-MA (Benign User → Malicious Agent)**: Malicious prompts are injected into agent role descriptions, creating compromised agents.

**Three Components of Malicious Prompts**: $P_m = \{T_s, T_d, C_i\}$ (summary + detailed description + code instruction)

**12 Malicious Behaviors across 5 Families**:
- **Trojan**: Intercept user input / exfiltrate data / download malicious files
- **Spyware**: Monitor clipboard / keylogging / screenshot capture
- **Adware**: Force-display advertisements / redirect to ad URLs / hijack clipboard
- **Ransomware**: Encrypt files for ransom / lock screen for ransom
- **Virus**: Self-replicate to other files

### Defense: Adv-IMBIA
- **Adv-MU-BA**: Inject defensive prompts into agent role descriptions (reject unauthorized data access, self-replication, and harmful code generation).
- **Adv-BU-MA**: Append defensive prompts to user inputs.
- 7 defense configurations: single-stage (design/coding/testing), dual-stage, and all-stage.

## Key Experimental Results

### Attack Success Rate

| Framework | MU-BA ASR | BU-MA ASR (Best) |
|-----------|-----------|-----------------|
| ChatDev | **93%** | 71% |
| MetaGPT | 45% | **84%** |
| AgentVerse | 71% | 45% |

### Defense Effectiveness (ASR Reduction / Rejection Rate Increase)

| Framework | MU-BA ASR↓ | MU-BA RR↑ | BU-MA ASR↓ |
|-----------|-----------|----------|-----------|
| ChatDev | **-73%** | +72% | -45% |
| MetaGPT | -40% | +52% | -7% |
| AgentVerse | -49% | +65% | -42% |

### Cross-Model Attack (ChatDev MU-BA)

| Model | ASR |
|-------|-----|
| GPT-4o-mini | 92.9% |
| Claude-4-sonnet | 87.5% |
| GPT-o3 | 81.1% |
| Llama-3.1-405b | 78.3% |

### Key Findings
- **ChatDev is most vulnerable to malicious users** (93% ASR): its waterfall architecture propagates malicious instructions across all stages.
- **MetaGPT is most robust against malicious users** (45% ASR): user tasks are fed only to the first agent, and the SOP filters some malicious content.
- **AgentVerse is most robust against malicious agents** (45% ASR): agile collective discussion prevents single-agent control.
- **Attacks at later stages are more dangerous**: compromised coding or testing agents have greater impact than compromised design-stage agents.
- **Attacking all agents simultaneously is not optimal**: concurrent malicious instructions to multiple agents produce decision conflicts, reducing overall ASR.
- **Stronger LLMs are more vulnerable**: GPT-4o-mini (92.9%) > Claude-4 (87.5%) > Llama-8b (lowest)—more capable models are more compliant with malicious instructions.
- **Attacks do not degrade software quality**: the utility under attack (UUA) remains close to baseline, indicating that malicious code is effectively concealed.
- **Defending against MU-BA is easier than BU-MA**: agent-level defenses are effective against malicious users (−73%), whereas user-level defenses are weaker against malicious agents (only −7% for MetaGPT).

## Highlights & Insights
- **The first systematic attack-and-defense analysis** covering the two most practically relevant threat scenarios, with 12 malicious behaviors spanning major malware families—establishing a foundation for security research in multi-agent software development.
- **The finding that "architecture determines security"** is particularly valuable: different development paradigms (waterfall vs. agile) exhibit inherently different resistance to different attack types.
- **The paradox that "stronger models are more vulnerable"** reveals a fundamental tension between instruction-following capability and security—more obedient models are more susceptible to executing malicious directives.
- **The "critical-agent defense" strategy** is practically useful: defending only key-stage agents achieves near-full-defense effectiveness without requiring protection of every agent, thereby saving resources.

## Limitations & Future Work
- The defense approach relies entirely on prompt injection, without code-level verification (e.g., static analysis or sandboxed execution).
- Defense against the BU-MA scenario remains weak (only −7% for MetaGPT), necessitating stronger defense mechanisms.
- GPT-4o-mini serves as the primary experimental model; cross-model validation is limited.
- The evaluation scale is relatively small: 40 benign requirements × 12 malicious behaviors = 480 test cases.
- More sophisticated attacks are not considered, such as multi-turn intermittent injection or cross-agent coordinated attacks.

## Related Work & Insights
- **vs. General Agent Security Research**: General agent security focuses on prompt injection and jailbreaking, whereas this work targets the specific context of multi-agent software development—implicit injection of malicious code.
- **vs. Code Security Tools**: Traditional code security relies on SAST/DAST detection, but LLM-generated malicious code may bypass pattern matching, requiring semantic-level detection.
- **Implications for Multi-Agent System Design**: (a) The propagation scope of malicious instructions across agents should be restricted; (b) agents at critical stages require stronger security auditing; (c) agile/deliberative architectures are inherently more secure than waterfall-style architectures.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic study of multi-agent software development security, with comprehensive coverage across dual scenarios, 12 attack behaviors, and 3 frameworks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete structure spanning 3 frameworks × 12 behaviors × 7 configurations, cross-model evaluation, and ablation studies, though the dataset scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Threat model is clearly defined and findings are concisely summarized, though some experimental details require consulting the appendix.
- Value: ⭐⭐⭐⭐⭐ Carries significant security implications for multi-agent development systems; the attack-defense paradigm provides a framework for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)
- [\[AAAI 2026\] A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind](a_computable_game-theoretic_framework_for_multi-agent_theory_of_mind.md)
- [\[AAAI 2026\] SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation](spatialactor_exploring_disentangled_spatial_representations_for_robust_robotic_m.md)
- [\[AAAI 2026\] Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](unintended_misalignment_from_agentic_fine-tuning_risks_and_m.md)
- [\[AAAI 2026\] EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer](evoempirbench_dynamic_spatial_reasoning_with_agent-expver.md)

</div>

<!-- RELATED:END -->
