---
title: >-
  [Paper Note] ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents
description: >-
  [ICLR 2026][LLM Agent][prompt injection] This paper exposes a structural vulnerability in chat templates used by LLM agents: by embedding forged role labels (e.g., `<system>`, `<user>`) in tool-returned data, attackers can hijack the model's role hierarchy perception and disguise malicious instructions as high-priority directives, raising ASR from 5–15% to 32–52%.
tags:
  - ICLR 2026
  - LLM Agent
  - prompt injection
  - chat template
  - role hierarchy
  - multi-turn attack
date: 2026-05-08
content_hash: f24a55eedfce69a5
---

# ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents

**Conference**: ICLR 2026
**arXiv**: [2509.22830](https://arxiv.org/abs/2509.22830)
**Code**: [https://github.com/hwanchang00/ChatInject](https://github.com/hwanchang00/ChatInject)
**Area**: LLM Agent
**Keywords**: prompt injection, chat template, LLM agent, role hierarchy, multi-turn attack

## TL;DR
This paper exposes a structural vulnerability in chat templates used by LLM agents: by embedding forged role labels (e.g., `<system>`, `<user>`) in tool-returned data, attackers can hijack the model's role hierarchy perception and disguise malicious instructions as high-priority directives, raising ASR from 5–15% to 32–52%.

## Background & Motivation
**Background**: LLM agents retrieve data by invoking external tools (search, APIs, file reading). This data is organized via role labels in the chat template (system > user > assistant > tool), and models rely on these special tokens to distinguish instructions of different priorities.

**Limitations of Prior Work**: Indirect prompt injection—embedding malicious instructions in tool-returned data—is a known threat, but existing attacks operate primarily at the plaintext level, overlooking structural vulnerabilities inherent in chat templates. Moreover, the instruction hierarchy defense (Wallace et al., 2024), which relies on role labels for priority stratification, inadvertently creates a new attack surface.

**Key Challenge**: LLMs are trained to strictly follow the instruction hierarchy demarcated by role labels; however, these labels can be forged—if tool-returned data contains `<user>` or `<system>` tags, the model may misinterpret them as higher-priority instructions.

**Goal**: (1) Verify whether chat template forgery constitutes an effective attack vector; (2) investigate whether simulated multi-turn dialogue can amplify attack effectiveness; (3) test cross-model transferability.

**Key Insight**: Multi-turn jailbreaking is effective in interactive settings but infeasible for indirect injection (attackers have only one injection opportunity), whereas chat templates provide a means to simulate multi-turn dialogue within a single injection.

**Core Idea**: Exploit role label forgery in chat templates to hijack the LLM's instruction hierarchy perception, and combine this with simulated multi-turn dialogue for persuasion-based attacks.

## Method

### Overall Architecture
The attacker embeds a malicious payload in tool-returned data $R_{T_u}$. The core of ChatInject is formatting the payload according to the target model's chat template structure rather than as plain text. Four payload variants are defined:

### Key Designs

1. **Chat Template Forgery (ChatInject)**:

    - Function: Wraps malicious instructions using the target model's native role labels to forge a high-priority source.
    - Mechanism: An attention-grabbing prefix is wrapped with the `<system>` role, and the malicious instruction $I_a$ is wrapped with the `<user>` role. Upon encountering these labels, the model treats subsequent content as high-priority instructions to execute.
    - Distinction from plaintext injection: Plaintext injection operates at the lexical level (e.g., "please ignore previous instructions"), whereas ChatInject hijacks the model's role-parsing mechanism at a structural level.

2. **Template-Based Multi-turn Variant**:

    - Function: Constructs a simulated multi-turn dialogue within a single injection to progressively "persuade" the model into executing malicious operations.
    - Mechanism: GPT-4.1 generates a 7-turn user-assistant conversation $C_a = \{(r_1^a, m_1^a), \ldots, (r_n^a, m_n^a)\}$, with each turn wrapped in role labels. The dialogue is designed to gradually legitimize the malicious operation—establishing context, decomposing it into seemingly harmless steps, and ultimately having the assistant "agree" to execute it.
    - Design Motivation: ChatInject alone raises ASR from 5% to 32%; adding Multi-turn further increases it to 52% (on InjecAgent), demonstrating a strong synergy between structural hijacking and persuasive dialogue.

3. **Agentic Extensions (Reasoning/Tool-calling Hooks)**:

    - Function: Leverage model-specific `<think>` and `<tool_call>` tags to further amplify the attack.
    - Reasoning hook: Appends `<think> Sure! </think>` after the payload to steer the model's internal reasoning toward immediate agreement.
    - Tool-calling hook: Appends a `<tool_call>` scaffold specifying the malicious tool to invoke, bypassing the model's decision-making process entirely.
    - Effect: The tool-calling hook further raises ASR by approximately 5–15 pp on InjecAgent.

## Key Experimental Results

### Main Results: Attack Success Rate (ASR)

| Model | Default InjecPrompt | ChatInject | Multi-turn + ChatInject |
|------|---------------------|------------|------------------------|
| Qwen3-235B (InjecAgent) | 8.5% | 39.4% (+30.9) | 65.9% (+55.2) |
| GPT-oss-120b (InjecAgent) | 0.0% | 14.2% (+14.2) | 16.9% (+16.8) |
| Llama-4-Maverick (InjecAgent) | 50.1% | 79.4% (+29.3) | 88.3% (+71.7) |
| GLM-4.5 (InjecAgent) | 0.0% | 57.3% (+57.3) | 71.5% (+71.4) |
| Qwen3-235B (AgentDojo) | 17.5% | 54.8% (+37.3) | 80.5% (+19.6) |

### Cross-Model Transferability

| Target Model | Default | Best Foreign Template | Self Template |
|---------|---------|----------------------|---------------|
| GPT-4o (closed) | 9.6% | 31.7% (Qwen-3) | N/A |
| Grok-3 (closed) | 2.3% | 50.9% (Gemma-3) | N/A |
| Gemini-pro (closed) | 1.4% | 27.4% (Qwen-3) | N/A |

Key finding: Higher template similarity correlates with greater cross-model transfer success.

### Key Findings
- ChatInject raises average ASR from 15.1% to 45.9% on InjecAgent, and from 5.2% to 32.1% on AgentDojo.
- Multi-turn + ChatInject achieves an average ASR of 52.3% on InjecAgent, demonstrating significant synergy.
- Grok-2 is less affected (its template lacks strong role separators), validating the hypothesis that more explicit template structure enables more effective attacks.
- Closed-source models are equally vulnerable: using only open-source model templates, attacks against GPT-4o, Grok-3, and Gemini-pro raise ASR by 13–49 pp.
- Existing prompt-level defenses (e.g., sandwich defense, instructional prevention) are largely ineffective against Multi-turn ChatInject.

## Highlights & Insights
- **Structural-level vs. text-level attacks**: ChatInject reveals a fundamental security design flaw—the role labels in chat templates simultaneously serve as the foundation of the security mechanism and as the entry point for attacks. This paradox, in which the security mechanism itself becomes the attack surface, warrants serious attention.
- **"Simulating multi-turn within a single injection"**: Leveraging role labels to construct a virtual multi-turn dialogue within a single tool return brings persuasion-based multi-turn attacks—previously infeasible in indirect injection scenarios—into practical reach. This is a particularly elegant design.
- **Template similarity as a transferability predictor**: The paper quantifies the correlation between embedding similarity across different models' chat templates and cross-model attack transferability, offering a new dimension for future defense evaluation.

## Limitations & Future Work
- The attack assumes the attacker knows the target model's chat template structure (publicly available for open-source models), though the mixed-template strategy partially mitigates this constraint.
- Multi-turn dialogues are generated by GPT-4.1 and require manual review, limiting automation for large-scale attacks.
- The paper focuses on attacks with limited exploration of defenses—only a few prompt-level defenses are evaluated, and token-level sanitization or architecture-level defenses such as ASIDE are not explored.
- It remains unevaluated whether models can be trained to ignore role labels appearing within tool-returned data.

## Related Work & Insights
- **vs. ASIDE (Zverev et al., 2025)**: ASIDE architecturally separates instructions from data via orthogonal rotation, making it a natural candidate defense against ChatInject. The attacks demonstrated in this paper serve as concrete instances of the problem ASIDE aims to solve.
- **vs. ChatBug (Jiang et al., 2024)**: ChatBug replaces safety tokens to break safety alignment (jailbreaking), whereas ChatInject forges role labels to achieve indirect injection. The objectives differ, but the underlying mechanisms are analogous.
- **vs. Instruction Hierarchy (Wallace et al., 2024)**: That work's defense relies on role labels for priority stratification, but ChatInject demonstrates that role labels themselves can be forged, fundamentally undermining this defense.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of chat template structure as an attack vector; the application of multi-turn simulation within a single injection is a notable contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 frontier models (including 3 closed-source) × 2 benchmarks × cross-model transferability × defense evaluation.
- Writing Quality: ⭐⭐⭐⭐ Attack motivation and experimental design are clear, though the tables are data-dense.
- Value: ⭐⭐⭐⭐ Exposes a fundamental vulnerability in LLM agent security with important implications for both security research and engineering practice.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents](../../NeurIPS2025/llm_agent/drift_dynamic_rulebased_defense_with_injection_isolation_for.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](../../ACL2026/llm_agent/conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/llm_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](../../ACL2026/llm_agent/agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)

<!-- RELATED:END -->
