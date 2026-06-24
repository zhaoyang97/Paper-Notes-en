---
title: >-
  [Paper Note] AdvAgent: Controllable Blackbox Red-teaming on Web Agents
description: >-
  [ICML 2025][LLM Agent][Web Agent Safety] This paper proposes AdvAgent, a reinforcement learning (DPO)-based blackbox red-teaming framework. It trains an adversarial prompter model to automatically generate invisible HTML adversarial prompts. When injected into web pages, these prompts mislead GPT-4V-driven Web Agents into executing attacker-specified target actions (e.g., changing buying Microsoft stock to buying NVIDIA stock). AdvAgent achieves a 97.5% attack success rate ac…
tags:
  - "ICML 2025"
  - "LLM Agent"
  - "Web Agent Safety"
  - "Red-teaming"
  - "Adversarial Attack"
  - "Blackbox Attack"
  - "Reinforcement Learning"
  - "Prompt Injection"
  - "DPO"
date: 2026-05-08
content_hash: 676229f8883c3a1f
---

# AdvAgent: Controllable Blackbox Red-teaming on Web Agents

**Conference**: ICML 2025  
**arXiv**: [2410.17401](https://arxiv.org/abs/2410.17401)  
**Code**: [https://ai-secure.github.io/AdvAgent/](https://ai-secure.github.io/AdvAgent/)  
**Area**: LLM Agent  
**Keywords**: Web Agent Safety, Red-teaming, Adversarial Attack, Blackbox Attack, Reinforcement Learning, Prompt Injection, DPO

## TL;DR

This paper proposes AdvAgent, a reinforcement learning (DPO)-based blackbox red-teaming framework. It trains an adversarial prompter model to automatically generate invisible HTML adversarial prompts. When injected into web pages, these prompts mislead GPT-4V-driven Web Agents into executing attacker-specified target actions (e.g., changing buying Microsoft stock to buying NVIDIA stock). AdvAgent achieves a 97.5% attack success rate across 440 tasks and maintains over 88.8% effectiveness against existing defense methods.

## Background & Motivation

**Background**: LLM/VLM-based general Web Agents (such as SeeAct) can autonomously interact with real websites to complete high-risk tasks like financial trading, e-commerce shopping, and medical operations. These Agents comprehend pages and execute user instructions by parsing webpage screenshots and HTML content.

**Security Risks**: Web Agents possess access permissions to sensitive resources and autonomous decision-making capabilities. Once exploited by attackers, this can lead to severe consequences—such as being misled into purchasing the wrong ticker in stock trading, or placing orders for the wrong products in e-commerce.

**Limitations of Prior Work**:
   - **White-box methods** (Wu et al., 2024a): Require access to Agent weights for gradient optimization, which is infeasible in practical deployments.
   - **Manual design methods** (Wu et al., 2024c; Liao et al., 2024): Require human heuristics to write attack instructions, which is costly and hard to scale.
   - **Automated attacks targeting LLM/VLM** (Zou et al., 2023; Guo et al., 2024): Lack flexibility for Agent interaction scenarios and show limited effectiveness in black-box cross-model transfer.

**Key Challenge**: How to automatically, efficiently, and controllably generate adversarial prompts to attack Web Agents under fully black-box conditions (without accessing Agent weights or logits)?

**Key Insight**: Model adversarial prompt generation as a sequence generation problem. Train a prompter model via RL (DPO) utilizing black-box Agent feedback, enabling it to learn to generate both effective and stealthy adversarial HTML injection content.

**Core Idea**: Train an adversarial prompter using DPO reinforcement learning to learn from the success/failure feedback of black-box Agents, generating controllable and stealthy adversarial webpage injections.

## Method

### Overall Architecture

The attack workflow of AdvAgent consists of three core stages:

1. **Adversarial Prompt Generation**: The trained adversarial prompter model receives the attack target description and automatically generates adversarial strings.
2. **Webpage Injection**: The generated adversarial strings are injected into invisible HTML fields (such as hidden `<div>` elements or invisible attributes) of the target webpage, ensuring that the visual rendering of the page remains unchanged.
3. **Misleading the Agent**: While processing the injected webpage, the Web Agent (e.g., SeeAct) reads the hidden adversarial content and is misled into executing the attacker's target action.

Attack scenario example: The user instructs the Agent to buy Microsoft stock; after the attacker injects the invisible instructions into the webpage, the Agent instead purchases NVIDIA stock.

### Key Designs

1. **Two-Stage Training Paradigm**

    - **Stage 1 — SFT Warm-up**:
        - Uses manually designed successful attack prompts as seed data.
        - Supervised fine-tuning is performed on a pre-trained language model to learn the basic patterns and structures of adversarial prompts.
        - Goal: Build initial adversarial prompt generation capabilities for the prompter, avoiding exploration from scratch during the RL stage.
    - **Stage 2 — DPO Reinforcement Learning Optimization**:
        - Generates a large pool of candidate adversarial prompts using the SFT model.
        - Injects the candidate prompts into webpages and observes the behavioral feedback of the black-box Agent.
        - Based on whether the attack succeeds, prompts are categorized into positive samples (successful attacks) and negative samples (failed attacks).
        - Use Direct Policy Optimization (DPO) for preference learning: $$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$
        - Where $y_w$ is the successful attack prompt (positive sample) and $y_l$ is the failed attack prompt (negative sample).
        - Core Advantage: Does not require access to Agent weights or logits, only needs to observe the final behavior of the Agent as the reward signal.

2. **Stealthy Injection Mechanism**

    - **Function**: Embeds adversarial prompts into invisible HTML elements of the webpage.
    - **Mechanism**: Leverages HTML fields that are invisible to users but readable by Agents, such as:
        - Hidden `<input type="hidden">` elements
        - CSS-hidden `<div style="display:none">` elements
        - HTML attributes (e.g., `aria-label`, `title`, etc.)
    - After injection, the visual rendering of the webpage in the browser remains completely unchanged, making it undetectable to ordinary users.
    - The Agent reads these hidden contents while parsing the HTML or processing page elements.

3. **Controllability**

    - **Function**: Allows attackers to flexibly modify the attack target without retraining.
    - **Mechanism**: The trained prompter model learns general patterns of adversarial prompts. Attackers can generate adversarial prompts targeting new goals by simply modifying the input conditions (e.g., changing the target company from NVIDIA to Tesla).
    - No need to re-optimize for every new attack target, significantly reducing attack costs.
    - This feature makes AdvAgent far more practical than manual methods with fixed prompts.

4. **Targeted Attack Formulation**

    - The attacker specifies a **target action** $a^*$, which differs from the user's original request.
    - Criteria for a successful attack: The final action executed by the Agent, $a_t$, matches the target action $a^*$.
    - Attack Success Rate (ASR) = Number of successfully attacked tasks / Total number of tasks.

## Key Experimental Results

### Main Results: Cross-Domain Attack Success Rate

Experimental results across 440 tasks across 4 different website domains:

| Method | Shopping | Finance | Social Media | Travel | Average ASR |
|------|----------|---------|-------------|--------|----------|
| No Attack (Baseline) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Manual Prompt | ~60% | ~55% | ~50% | ~45% | ~52.5% |
| Transfer Attack (CLIP) | ~40% | ~35% | ~38% | ~30% | ~35.8% |
| **AdvAgent (Ours)** | **~98%** | **~97%** | **~98%** | **~97%** | **97.5%** |

Key Findings: AdvAgent significantly outperforms baseline methods across all domains, achieving an average ASR of 97.5%, which is far superior to manually designed prompts (~52.5%) and CLIP-based transfer attacks (~35.8%).

### Defense Experiments: Effectiveness of Existing Defenses

| Defense Method | Defense Strategy Description | AdvAgent ASR | Defense Effectiveness |
|---------|-------------|-------------|-----------|
| No Defense | — | 97.5% | — |
| Prompt Hardening | Adds safety instructions to the Agent's system prompt | ~92% | Extremely Weak |
| Input Filtering | Detects and filters suspicious HTML content | ~90% | Weak |
| Paraphrasing | Rewriting textual content of the webpage | ~89% | Weak |
| Multi-Defense | Stacking multiple defense mechanisms | 88.8% | Limited |

Key Findings: Even when stacking multiple defense strategies, AdvAgent still maintains an attack success rate of above 88.8%, indicating that current prompt-based defense mechanisms are virtually ineffective against such attacks.

### Ablation Study Summary

| Component | ASR Change After Removal | Remarks |
|------|---------------|------|
| DPO Training | Significant decrease | Proves RL feedback is crucial for black-box attacks |
| SFT Warm-up | Moderate decrease | SFT provides critical initialization |
| HTML Field Changes | 97.0% (Slight decrease) | Attacks are robust to injection location changes |
| Different Agent backbones | High ASR preserved | Attacks exhibit cross-model transferability |

## Highlights & Insights

1. **Unification of Black-Box Setting and High Efficiency**: AdvAgent is the first Web Agent red-teaming framework to achieve over 95% attack success rate under a fully black-box setting. By employing DPO—an offline RL method that avoids online interaction—it eliminates white-box dependency while significantly lowering training costs.
2. **Exquisite Stealth Design**: Utilizing invisible HTML fields to inject adversarial content achieves true "invisible to humans, but deceptive to Agents". This exposes a fundamental security flaw in current Web Agents’ lack of safety awareness when handling HTML content.
3. **Controllability as a Key Innovation**: Unlike static adversarial samples, AdvAgent's prompter model can flexibly adjust the attack targets. A single training run can adapt to multiple attack scenarios, offering immense practical value for actual red-teaming.
4. **Crucial Warning of Defense Failures**: Experiments demonstrate that common defense methods, such as prompt hardening and input filtering, are largely ineffective against these attacks (with ASR remaining above 88%). This sounds an alarm for the Agent security community—more fundamental defense paradigms are required.
5. **Paradigm Shift from LLM Attacks to Agent Attacks**: "Reversely" applying DPO (typically an alignment technique) for attacks and expanding from single-turn text attacks to multi-step interactive Agent scenarios is an interesting methodological innovation.

## Limitations & Future Work

1. **Strong Attack Assumption**: Attackers must be able to modify the target webpage's HTML content, which in real-world scenarios requires controlling a man-in-the-middle proxy or compromising the target website, thus limiting the direct applicability of such attacks.
2. **Single Agent Evaluation**: Primarily evaluated on SeeAct (GPT-4V). Further validation is needed to assess generalization to other architectures (such as HTML-only Agents or Agents driven by non-OpenAI models).
3. **Inadequate Defense Research**: The paper focuses heavily on demonstrating attack effectiveness, with limited discussion on how to construct effective defenses. Future work should explore more fundamental defenses, such as HTML content integrity validation or trusted execution environments.
4. **Ethical Risks**: Although the research aims to expose vulnerabilities to improve security, releasing a highly effective attack framework might be maliciously exploited. It requires a corresponding responsible disclosure mechanism.
5. **Static Webpage Assumption**: Webpage environments in the experiments are relatively static. The effectiveness of injections and attacks on modern webpages with dynamically loaded content (AJAX/SPA) remains to be verified.

## Related Work & Insights

- **Web Agent Frameworks**: SeeAct (Zheng et al., 2024) proposes a Web Agent architecture utilizing both screenshots and HTML inputs, marking the current SOTA. MindAct (Deng et al., 2024) and WebArena (Zhou et al., 2023) provide essential benchmarking environments.
- **Agent Security Attacks**:
    - Yang et al. (2024) and Wang et al. (2024) study backdoor attacks under white-box settings.
    - Wu et al. (2024c) and Liao et al. (2024) explore manual prompt injection attacks.
    - Wu et al. (2024a) propose gradient optimization methods but are limited by white-box assumptions.
- **LLM Adversarial Attacks**: Methods such as GCG (Zou et al., 2023) and AutoDAN (Guo et al., 2024) lay the foundation for automated LLM red-teaming, but are not directly applicable to multi-step interactive Agent scenarios.
- **New Applications of DPO**: This paper reverses DPO from "alignment" to "attack", providing new insights for RL-based adversarial training. Insight: The RLHF/DPO framework has broad application prospects in both offensive and defensive security.
- **Inspiring Thought**: The security issues of Web Agents represent a fundamental **trust boundary** problem—Agents cannot distinguish legitimate content from injected content. Future systems might require mechanisms akin to Content Security Policy (CSP) to establish trusted boundaries for Agent content.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to apply DPO to black-box Web Agent attacks, proposing a stealthy and controllable injection framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 440 tasks across 4 domains, comparisons with multiple defenses, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-motivated, and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ — Exposes major security vulnerabilities in Web Agents, offering significant warning value to the Agent safety domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents](../../ICML2026/llm_agent/otora_a_unified_red_teaming_framework_for_reasoning-level_denial-of-service_in_l.md)
- [\[NeurIPS 2025\] Generative AI Agents for Controllable and Protected Content Creation](../../NeurIPS2025/llm_agent/generative_ai_agents_for_controllable_and_protected_content_creation.md)
- [\[NeurIPS 2025\] Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](../../NeurIPS2025/llm_agent/web-shepherd_advancing_prms_for_reinforcing_web_agents.md)
- [\[ACL 2025\] Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](../../ACL2025/llm_agent/explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)
- [\[ICML 2025\] Towards LLM Agents for Earth Observation](towards_llm_agents_for_earth_observation.md)

</div>

<!-- RELATED:END -->
