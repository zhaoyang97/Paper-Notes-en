---
title: >-
  [Paper Note] MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents
description: >-
  [NeurIPS 2025][Robotics][adversarial attack] This paper reveals a novel adversarial attack against multimodal OS Agents, termed MIP (Malicious Image Patches): visually imperceptible adversarial perturbation patches (occupying approximately 1/7 of the screen area) are embedded in screenshots, causing the OS Agent to output a predefined sequence of malicious API calls upon capture. Joint optimization enables universal generalization across user instructions and screen layouts, achieving an attack success rate of up to 100%.
tags:
  - NeurIPS 2025
  - Robotics
  - adversarial attack
  - OS agent
  - malicious image patch
  - VLM security
  - computer worm
date: 2026-05-08
content_hash: 641b55838de69ed3
---

# MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents

**Conference**: NeurIPS 2025
**arXiv**: [2503.10809](https://arxiv.org/abs/2503.10809)
**Code**: [GitHub](https://github.com/AIchberger/mip-against-agent)
**Area**: Robotics
**Keywords**: adversarial attack, OS agent, malicious image patch, VLM security, computer worm

## TL;DR

This paper reveals a novel adversarial attack against multimodal OS Agents, termed MIP (Malicious Image Patches): visually imperceptible adversarial perturbation patches (occupying approximately 1/7 of the screen area) are embedded in screenshots, causing the OS Agent to output a predefined sequence of malicious API calls upon capture. Joint optimization enables universal generalization across user instructions and screen layouts, achieving an attack success rate of up to 100%.

## Background & Motivation

**Background**: OS Agents (e.g., Claude Computer Use, agents in Windows Agent Arena) elevate VLMs from passive text generators to active computer controllers capable of executing mouse clicks, keyboard inputs, file operations, and network requests. This shift escalates VLM security risks from "generating harmful text" to "executing harmful actions."

**Limitations of Prior Work**: Existing research has demonstrated that OS Agents can be compromised via prompt injection or pop-up attacks. However, these methods require direct access to the agent's text input pipeline and are susceptible to detection and blocking by existing filtering mechanisms. The reliance of OS Agents on **screenshots** for navigation introduces a novel visual-domain attack surface that remains largely unexplored.

**Key Challenge**: OS Agents must observe their environment through screenshots → security risk; an attacker needs only to control a small screen region (e.g., a social media image or desktop wallpaper) to potentially hijack the entire agent → difficult to detect.

**Goal**: To systematically investigate visual-domain attacks against OS Agents: can manipulating a small image region on the screen hijack an agent into executing arbitrary malicious actions? Can such attacks generalize across scenarios?

**Key Insight**: Extend conventional VLM adversarial attacks to the multi-component pipeline of OS Agents, addressing unique constraints such as the non-differentiability of the screen parser, image resizing, and discrete pixel values.

**Core Idea**: MIP encodes complete malicious program instructions into a visually imperceptible image patch; once the OS Agent processes the screenshot containing the MIP, it directly outputs and executes the malicious program—without relying on the agent's own reasoning to assemble the attack.

## Method

### Overall Architecture

The attack pipeline proceeds as follows: (1) optimize pixel perturbations within the patch region of the target VLM using PGD → (2) embed the optimized MIP into a desktop wallpaper or social media post → (3) the OS Agent captures the MIP during a screenshot → (4) the VLM processes the MIP-containing screenshot and outputs the predefined malicious target $\mathbf{y}$ (containing complete API calls) → (5) the agent executes the malicious actions.

### Key Designs

1. **Multi-Constraint PGD Optimization**:
    - Function: Optimize adversarial perturbations under the constraints of the OS Agent's multi-component pipeline.
    - Mechanism: The objective is $\boldsymbol{\delta}^* = \arg\min_{\mathcal{R}, \boldsymbol{\delta} \in \Delta_\mathcal{R}^\epsilon} \mathcal{L}(f_{\boldsymbol{\theta}}(\mathbf{p}_{txt}, q(l(\mathbf{s}, \mathbf{s}_{som}) + \boldsymbol{\delta})), \mathbf{y})$, subject to: perturbations confined to patch region $\mathcal{R}$, $\ell_\infty \leq \epsilon=25/255$, no alteration of SOM detection results from the screen parser, discrete integer pixel values, and a differentiable approximation replacing image resizing. Each PGD step is followed by projection back onto the feasible set.
    - Design Motivation: OS Agents are not simple VLMs—they incorporate a screen parser (non-differentiable), resizing (information loss), and API parsing, all of which must be circumvented by the attack.

2. **Universal MIP Generalization Optimization**:
    - Function: Enable a single MIP to remain effective across diverse user instructions and screen layouts.
    - Mechanism: Extends from targeted optimization (a single prompt–screenshot pair) to universal optimization—at each PGD step, a batch of 8 pairs $(p,s) \sim \text{Uniform}(\mathcal{P}_+ \times \mathcal{S}_+)$ is sampled for joint updates, optimizing until the probability of the malicious target exceeds 99% across all training pairs. Generalization is evaluated on unseen prompt set $\mathcal{P}_-$ and screenshot set $\mathcal{S}_-$.
    - Design Motivation: In practice, attackers cannot anticipate the victim's specific instructions or screen state; Universal MIP is a prerequisite for real-world deployability.

3. **Direct Encoding vs. Indirect Guidance**:
    - Function: Encode the complete malicious program directly into the MIP rather than indirectly steering the agent's reasoning.
    - Mechanism: The target output $\mathbf{y}$ contains a complete sequence of API calls (33–52 tokens), such as opening a terminal to execute a memory overflow or navigating to a malicious website. Once the VLM outputs $\mathbf{y}$, the agent immediately executes it via API, without relying on its own capabilities.
    - Design Motivation: Indirect attacks (having the agent "assemble" malicious behavior) introduce additional failure points—the agent may be hijacked but fail to correctly construct the malicious program; direct encoding guarantees execution upon triggering.

## Key Experimental Results

### Main Results

| Setting | Malicious Action | Train ASR | Unseen Prompt ASR | Unseen Screen ASR |
|---------|----------------|-----------|-------------------|-------------------|
| Desktop Targeted | Memory overflow (33 tokens) | 1.00 | 0.91 | 0.00 |
| Desktop Universal | Memory overflow | ~1.00 | ~0.90 | ~0.80 |
| Desktop Universal | Malicious website (52 tokens) | ~0.90 | ~0.80 | ~0.70 |
| Social Media Universal | Memory overflow | ~1.00 | ~0.85 | ~0.75 |
| Social Media Universal | Malicious website | ~0.90 | ~0.70 | ~0.60 |

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|-----------|-------|
| Targeted vs. Universal | ASR on unseen | Universal significantly improves cross-scenario generalization |
| Different Screen Parsers | OmniParser vs. GroundingDINO | MIP generalizes across parsers (ASR ~0.5–0.7) |
| Different VLM sizes (11B vs. 90B) | Cross-model ASR | MIP optimized on 11B can attack 90B models |
| Capture during agent execution | ASR after multi-step interaction | MIP remains effective while agent executes normal tasks |

### Key Findings
- Targeted MIP generalizes across prompts but not across screenshots (ASR=0); Universal MIP resolves this limitation.
- MIP generalizes across screen parsers and VLM scales—patches optimized on Llama-11B successfully attack 90B models.
- Attacks remain effective even when the agent encounters the MIP after completing several normal steps.
- Desktop scenarios are more susceptible than social media scenarios (fewer SOM elements, shorter textual context).

## Highlights & Insights
- The concept of an **"OS Agent computer worm"**: if the malicious action includes sharing a post containing the MIP, the patch can self-propagate—this is the first proposal of such self-spreading OS Agent attacks.
- The direct encoding strategy bypasses uncertainty in the agent's reasoning; execution is guaranteed once triggered.
- Cross-scale generalization (11B → 90B) suggests the existence of shared systematic vulnerabilities in VLM visual processing.
- Attack detection is extremely difficult—MIPs are visually indistinguishable from normal images and do not rely on the text pipeline.

## Limitations & Future Work
- White-box access to VLM parameters is required (PGD requires gradients); black-box transferability attacks remain insufficiently explored.
- Perturbations at $\epsilon=25/255$ may be perceptible under magnified inspection; stealthier attacks warrant further investigation.
- Evaluation is limited to Windows Agent Arena; other OS Agent frameworks (e.g., Claude Computer Use) are not tested.
- The effectiveness of defenses (e.g., adversarial training, input sanitization) is not thoroughly discussed.

## Related Work & Insights
- **vs. Pop-up attacks (Zhang et al.)**: Pop-up attacks rely on visible windows and are susceptible to text filtering; MIPs are visually imperceptible and bypass text-based filters.
- **vs. VLM adversarial attacks (Bailey et al.)**: Bailey et al. attack VLMs with tool-use capabilities by directly feeding adversarial images; MIPs must be delivered indirectly via screenshots, imposing additional constraints.
- **vs. Wu et al.**: Their approach attacks captioning models to indirectly guide agents; MIPs directly target VLM decision-making, yielding greater reliability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic study of visual-domain attacks on OS Agents; introduces the "Agent worm" concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across two settings, two behaviors, and generalization across parsers, model scales, and prompts.
- Writing Quality: ⭐⭐⭐⭐ Formalization is clear; constraints are described with precision.
- Value: ⭐⭐⭐⭐⭐ Carries significant security implications for OS Agents that must be addressed before large-scale deployment.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Can Agents Fix Agent Issues?](can_agents_fix_agent_issues.md)
- [\[NeurIPS 2025\] Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs](zero-shot_embedding_drift_detection_a_lightweight_defense_against_prompt_injecti.md)
- [\[NeurIPS 2025\] AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling](autotom_scaling_model-based_mental_inference_via_automated_agent_modeling.md)
- [\[NeurIPS 2025\] MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)
- [\[NeurIPS 2025\] LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents](labutopia_high-fidelity_simulation_and_hierarchical_benchmark_for_scientific_emb.md)

<!-- RELATED:END -->
