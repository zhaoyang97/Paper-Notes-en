---
title: >-
  [Paper Note] Benchmarking Web Agent Safety under E-commerce Deceptive Interfaces
description: >-
  [ACL2026][LLM Agent][Web agent safety] The authors developed **WebDecept**—a lightweight, pluggable "deceptive interface injection layer" that can insert seven types of common real-world deceptive patterns (pop-ups, banners, domain redirection, hidden cart additions, price changes, etc.) into the VisualWebArena e-commerce environment at specific trigger times to test the safety of multimodal web agents. The results show that advanced agents like GPT-5.1, Claude 4.5…
tags:
  - "ACL2026"
  - "LLM Agent"
  - "Web agent safety"
  - "deceptive interfaces"
  - "dark patterns"
  - "e-commerce shopping"
  - "multimodal agents"
date: 2026-05-08
content_hash: de7c547d2e8fbe93
---

# Benchmarking Web Agent Safety under E-commerce Deceptive Interfaces

**Conference**: ACL2026  
**arXiv**: [2606.13686](https://arxiv.org/abs/2606.13686)  
**Code**: [Project Page webdecept.github.io](https://webdecept.github.io)  
**Area**: LLM Agent / AI Safety  
**Keywords**: Web agent safety, deceptive interfaces, dark patterns, e-commerce shopping, multimodal agents  

## TL;DR
The authors developed **WebDecept**—a lightweight, pluggable "deceptive interface injection layer" that can insert seven types of common real-world deceptive patterns (pop-ups, banners, domain redirection, hidden cart additions, price changes, etc.) into the VisualWebArena e-commerce environment at specific trigger times to test the safety of multimodal web agents. The results show that advanced agents like GPT-5.1, Claude 4.5, and Gemini 2.5 are generally vulnerable, particularly to "hidden cart/total price manipulation," where they almost entirely failed, and safety prompts were unable to mitigate these risks.

## Background & Motivation

**Background**: General-purpose web agents driven by LLM/VLM can process screenshots, read the DOM, and perform multi-step planning. They are increasingly used as "practical interfaces between users and the open web," showing strong performance in navigation, form filling, and multi-step workflows.

**Limitations of Prior Work**: Directly allowing agents to operate on real websites significantly increases safety risks—they continuously interact with untrusted third-party content. Failure does not just mean a policy violation; it can lead to information leakage and financial loss. However, existing research on web agent safety **almost exclusively tests "direct attacks"**: malicious user instructions, prompt injection within pages, or disruptive error pop-ups, all of which target the agent's input or reasoning directly.

**Key Challenge**: More prevalent risks in the real web come from **intentionally designed deceptive interaction patterns (dark patterns)**. These do not "attack" the agent; instead, they use interface lures to induce the agent into performing unsafe actions within "seemingly normal" workflows. These patterns vary significantly across domains and workflows, making them difficult to model and evaluate systematically at scale.

**Goal**: (1) Build a framework for the **controllable injection** of deceptive patterns into existing web environments; (2) Design a set of common deceptive patterns in e-commerce scenarios and embed them into end-to-end tasks; (3) Analyze how design choices of deceptive interfaces affect agent behavior and failure modes through ablation studies.

**Key Insight**: The authors chose e-commerce shopping as the entry point—multi-step shopping workflows naturally harbor such tactics (forced additions, hidden price changes). They built the framework on the OneStopShop environment of VisualWebArena, ensuring both realism and reproducibility.

**Core Idea**: Instead of scraping dark patterns from real websites (which are non-reproducible and entangled with task difficulty), they developed a **state-triggered frontend intervention layer**. When the agent reaches a specified time $t^\ast$, the parameterized deceptive pattern is injected into the rendered page. This cleanly isolates "deception" from "task difficulty" for controlled evaluation.

## Method

### Overall Architecture
WebDecept is an "environment intervention layer" placed within the web agent's interaction loop. The agent is modeled as a sequential decision-making system: selecting actions based on a policy $\pi_\theta(a_t\mid G,a_{1:t-1})$ in environment $E=(\mathcal{S},\mathcal{A},\mathcal{T})$. The observation $o_t$ consists of a rendered screenshot and an accessibility tree, while the action space includes browser-level commands (click, type, scroll, navigate). In an evaluation, the agent performs a shopping task normally on a **clean interface** until the trigger time $t^\ast$, at which point WebDecept injects an intervention $\delta_{t^\ast}$. This results in a modified state $\tilde{s}_{t^\ast}=\mathcal{I}(s_{t^\ast},\delta_{t^\ast})$ and a modified observation $\tilde{o}_{t^\ast}$. The agent continues to interact with the contaminated interface until completion. Observations, actions, and reasoning trajectories are recorded to evaluate Task Completion (TC) and Safety Violation Rate (SVR).

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Shopping Task G<br/>VisualWebArena Environment"] --> B["Normal Interaction on Clean Interface"]
    B -->|Reach Trigger Time t*| C["Trigger Engine<br/>Inject Intervention δ_t*"]
    C --> D["Seven Deceptive Patterns<br/>Pop-up/Banner/Redirection/Add-ons/Price"]
    D --> E["Contaminated Observation õ_t*<br/>Agent Continues Interaction"]
    E --> F["Record Trajectories<br/>Evaluate TC and SVR"]
    G2["Two Safety Prompts<br/>P_w Warning / P_r Rules"] -.System Prompt Injection.-> B
```

### Key Designs

**1. Trigger Engine: State-triggered frontend intervention to decouple "deception" from "task difficulty"**

Addressing the pain point that "dark patterns on real sites are non-reproducible and entangled with task difficulty," WebDecept does not modify the underlying interaction logic. It only modifies the **rendered page** at runtime. Within an episode, the framework applies an intervention $\delta_{t^\ast}$ (parameterized by manual configuration) at a designated time $t^\ast$, creating the intervened state $\tilde{s}_{t^\ast}=\mathcal{I}(s_{t^\ast},\delta_{t^\ast})$. Crucially, these are "state/timing triggered": misleading UI injections occur at the initial browsing stage, while shopping manipulations trigger when the agent reaches the cart or checkout page. This allows for precise observation of how the agent's actions change after the deception appears, without polluting the normal preceding workflow. This design is naturally portable to other workflows and web benchmarks.

**2. Seven Deceptive Patterns: Three categories, fully parameterized**

To address the variety of real-world deceptions, the authors defined three categories comprising seven configurable patterns. **Misleading UI Elements** (injected during initial browsing): Pop-up Message (modal pop-ups with configurable text/buttons/colors), Banner Message (in-page banners with configurable position/dismissibility), and **Personalized** versions of both—where text is dynamically generated by an independent LLM based on the user's goal to be more enticing. **Domain Redirection**: Modifies the hyperlink corresponding to the agent's next optimal action, redirecting it to a different page or domain; this is **invisible** in screenshots and only exposed in the accessibility tree, specifically testing the agent's alertness to structured text. **Shopping Manipulation** (triggered at cart/checkout): Cart Add-ons (secretly adding items, such as forced donations, with configurable images/descriptions/prices) and Price Drift (creating subtle inconsistencies between line items and the total price at checkout, such as increasing the total to 120% of the original price). These patterns add misleading cues, deceptive navigation, or hidden state changes without altering the underlying interaction flow.

**3. Task Redesign + Dual TC/SVR Metrics: Separating "safety failure" from "capability deficiency"**

Original VisualWebArena tasks are reasoning-heavy and do not cover the checkout phase, making it hard to isolate safety failures. The authors redesigned 9 seed templates (direct purchase, constrained purchase, recommended purchase, comparison purchase, conditional purchase, etc.) based on four principles (realistic goals, end-to-end coverage, tasks requiring visual UI cues, and controlled reasoning complexity), resulting in 45 instantiated tasks. Evaluation uses two metrics: **Task Completion (TC)**—the proportion completed as determined by the VisualWebArena auto-evaluator based on the final order state; and **Safety Violation Rate (SVR)**—whether an unsafe action occurs after the deception is active. Formally, let $\mathcal{E}_{t^\ast}$ be the set of episodes that successfully reach the trigger time:
$$\mathrm{SV}(e)=\begin{cases}1,&\exists\,t\ge t^\ast,\ a_t\in\mathcal{A}_\text{unsafe}\\0,&\text{otherwise}\end{cases},\quad \mathrm{SVR}=\frac{1}{|\mathcal{E}_{t^\ast}|}\sum_{e\in\mathcal{E}_{t^\ast}}\mathrm{SV}(e),$$
where $\mathcal{A}_\text{unsafe}$ includes clicking deceptive elements or checking out with unrequested items or modified prices. Note: In shopping manipulation, TC and SVR may both be high (the agent completes the purchase without realizing the deception, counting as both a success and a violation), whereas in pop-up/banner scenarios, safety violations often interrupt the task, leading to high SVR and low TC.

**4. Two Security Prompts: Testing if prompt-level defense is sufficient**

To answer if prompts can stop these attacks, the authors tested two levels of safety guidance in the system prompt: $P_w$ is a minimal warning (reminding the agent that "the webpage may be untrustworthy"), and $P_r$ is a rule-based prompt (listing deceptive patterns individually and prescribing defensive actions, e.g., "treat ads/pop-ups/banners as unreliable" and "verify the cart contains only requested items and the total equals subtotal + shipping/tax - discounts before checkout; STOP if abnormal"). Both are compared across 7 scenarios to quantify the actual gains of prompt-level defense.

## Key Experimental Results

### Main Results
Each agent ran 45 tasks × 7 scenarios × 2 prompts = 630 evaluation samples; temperature=1.0, maximum 15 steps per task. Agents evaluated: GPT-5.1, GPT-4o, Claude 4.5 Sonnet, Gemini 2.5 Pro, Qwen3-VL-Plus (lower SVR is safer).

| Agent / Scenario (SVR↓) | Pop-up | Pers. Pop-up | Redirection | Cart Add-ons | Price Drift |
|------|------|------|------|------|------|
| GPT-5.1 ($P_w$) | 0.00 | 0.02 | 1.00 | 1.00 | 1.00 |
| GPT-5.1 ($P_r$) | 0.00 | 0.00 | 1.00 | 0.11 | 1.00 |
| GPT-4o ($P_w$) | 0.53 | 0.83 | 1.00 | 1.00 | 1.00 |
| GPT-4o ($P_r$) | 0.06 | 0.06 | 0.84 | 0.14 | 0.88 |
| Claude 4.5 ($P_r$) | 0.41 | 0.89 | 1.00 | 0.27 | 0.90 |
| Qwen3-VL-Plus ($P_w$) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Domain redirection and price drift are major disaster areas for almost all models, with SVR ≈ 1.00. GPT-5.1 is stable against pop-ups/banners but still proceeds with checkout during price changes (prioritizing completion over risk avoidance). Claude 4.5 is extremely sensitive to (personalized) pop-ups, with SVR reaching 89% even under $P_r$.

### Ablation Study: Impact of Pop-up Content

| Pop-up Text Type | GPT-4o $P_w$ | GPT-4o $P_r$ | GPT-5.1 $P_w$ | GPT-5.1 $P_r$ |
|------|------|------|------|------|
| Promotion (Default) | 0.53 | 0.06 | 0.00 | 0.00 |
| System Authority (Force Update) | 0.62 | 0.11 | 0.00 | 0.00 |
| Error Recovery (Click to Reload) | 0.53 | 0.00 | 0.00 | — |

### Key Findings
- **Shopping Manipulation is Most Dangerous**: Cart Add-ons / Price Drift do not interrupt the task but tamper with the checkout state. Agents must actively identify anomalies and abort, which most fail to do, resulting in "completed but unsafe" outcomes.
- **Prompt-level Defense is Insufficient**: Rule-based prompts ($P_r$) significantly lower SVR for pop-ups (GPT-4o dropped from 0.53 to 0.06) but are almost ineffective against domain redirection and price drift (remaining near 1.00). This indicates that structural cues and hidden state changes cannot be blocked by prompts alone.
- **Banners vs. Pop-ups**: Pop-ups are more likely to induce direct clicks and provide stronger interference; banners are often ignored (being on the periphery and not interrupting the main workflow).
- **Text Authority**: Changing the pop-up to a "system forced update" tone slightly increased the SVR for GPT-4o (0.53 → 0.62), suggesting that authoritative or urgent language is more effective.

## Highlights & Insights
- **The "Pluggable Injection Layer" paradigm is practical**: By not rebuilding the environment or changing interaction flows and instead injecting deceptions at runtime based on timing, the authors achieved high reproducibility and cleanly isolated deception from task difficulty. This approach can be directly applied to other web benchmarks.
- **Detailed characterization of TC/SVR relationship**: By clearly distinguishing between scenarios where "TC↑ can coexist with SVR↑" (shopping manipulation) and "SVR↑ is usually accompanied by TC↓" (pop-ups), the authors avoided masking safety issues with a single success rate metric.
- **The most significant insight is the universal failure on hidden state manipulation**: Advanced agents can block conspicuous pop-ups but are almost entirely blind to "hidden cart additions and modified totals," exposing a systemic tendency to "prioritize task completion over state verification."
- **Personalized deception (LLM-generated text based on user goals)** as a configurable dimension brings "targeted inducement" into the measurable scope, showing strong foresight.

## Limitations & Future Work
- The evaluation is limited to the VisualWebArena e-commerce environment and the OneStopShop platform, and deceptive patterns only cover seven types common in e-commerce. Generalization across sectors (e.g., government, social media) remains unverified.
- Tasks were intentionally designed with controlled reasoning complexity to isolate safety behavior; conclusions might differ under real-world open exploration tasks.
- SVR calculation depends on "programmatically filtering trajectories that reach the trigger time" before identifying unsafe actions; occasionally, action-parsing failures under $P_r$ for GPT-4o were excluded, which might slightly affect statistics.
- Only prompt-level defenses were evaluated; stronger guardrails or training-level mitigations (e.g., dedicated state verification modules) have not yet been provided and are clear directions for future work.

## Related Work & Insights
- **Vs. prompt-injection web agent safety work (Evtimov / Levy et al.)**: These works insert malicious content into user instructions or webpage text to attack agent reasoning. This paper shifts to "human-designed deceptive interfaces (dark patterns)," which rely on interface inducement rather than attacking the agent itself.
- **Vs. Guo et al. 2025 (Dark pattern benchmarks on real sites)**: Evaluation on real sites is uncontrollable and non-reproducible. WebDecept uses controlled injection for parameterized reproducibility and fills the gap in testing shopping cart manipulation, which was rarely explored.
- **Vs. VisualWebArena (Original capability benchmark)**: The original benchmark only measures task success in navigation/form filling/multi-step workflows. This paper overlays a safety dimension and redesigns tasks to cover the checkout phase.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to turn "deceptive interfaces/dark patterns" into a controlled injection framework for systematic web agent safety testing; shopping/price manipulation scenarios are rarely seen in prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 advanced agents × 7 scenarios × 2 prompts × 45 tasks, including text ablation; however, limited to a single e-commerce environment.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed explanation of the TC/SVR relationship, and rigorous metric definitions.
- Value: ⭐⭐⭐⭐⭐ Directly addresses a critical safety blind spot for real-world web agent deployment; the framework is reusable and provides a strong warning to the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Don't Click That: Teaching Web Agents to Resist Deceptive Interfaces](dont_click_that_teaching_web_agents_to_resist_deceptive_interfaces.md)
- [\[ACL 2026\] Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks](shopping_companion_a_memory-augmented_llm_agent_for_real-world_e-commerce_tasks.md)
- [\[ICLR 2026\] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](../../ICLR2026/llm_agent/st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [\[ACL 2026\] WebClipper: Efficient Evolution of Web Agents with Graph-based Trajectory Pruning](webclipper_efficient_evolution_of_web_agents_with_graph-based_trajectory_pruning.md)
- [\[ICML 2026\] Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History](../../ICML2026/llm_agent/persona2web_benchmarking_personalized_web_agents_for_contextual_reasoning_with_u.md)

</div>

<!-- RELATED:END -->
