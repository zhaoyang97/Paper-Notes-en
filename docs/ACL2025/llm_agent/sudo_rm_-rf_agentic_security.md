---
title: >-
  [Paper Note] "sudo rm -rf agentic_security" | SUDO: Screen-based Universal Detox2tox Offense
description: >-
  [ACL 2025][LLM Agent][computer-use agent] This work proposes the SUDO attack framework. It disguises malicious requests as harmless instructions using a three-stage Detox2tox pipeline, restores the attack payload during execution, and systematically breaches the safety defenses of computer-use agents like Claude CUA and MANUS using dynamic iterative optimization based on checklist feedback, achieving an attack success rate of up to 41.33%.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "computer-use agent"
  - "safety attack"
  - "red teaming"
  - "jailbreak"
  - "agentic security"
date: 2026-05-08
content_hash: 87a36bc8d77e0de0
---

# "sudo rm -rf agentic_security" | SUDO: Screen-based Universal Detox2tox Offense

**Conference**: ACL 2025  
**arXiv**: [2503.20279](https://arxiv.org/abs/2503.20279)  
**Code**: [github.com/AIM-Intelligence/SUDO](https://github.com/AIM-Intelligence/SUDO)  
**Area**: LLM Agent / AI Safety  
**Keywords**: computer-use agent, safety attack, red teaming, jailbreak, agentic security

## TL;DR

This work proposes the SUDO attack framework. It disguises malicious requests as harmless instructions using a three-stage Detox2tox pipeline, restores the attack payload during execution, and systematically breaches the safety defenses of computer-use agents like Claude CUA and MANUS using dynamic iterative optimization based on checklist feedback, achieving an attack success rate of up to 41.33%.

## Background & Motivation

**Background**: LLMs are evolving from pure text conversations to computer-use agents (e.g., Claude Computer Use, MANUS, OmniParser) capable of autonomously performing tasks like file operations, terminal commands, and web browsing in real desktop or web environments. These agents greatly expand the practical applications of LLMs, but they also bring entirely new security risk surfaces.

**Limitations of Prior Work**: Existing jailbreak research focuses primarily on pure-text LLM scenarios, leaving the safety evaluation of agents interacting with multimodal environments highly inadequate. Existing single-turn attack methods such as role-playing or program execution frameworks perform extremely poorly on computer-use agents (with a maximum ASR of only 7.30%). Crucially, once an agent capable of operating real systems is breached, the consequences are far more severe than in text scenarios—potentially leading to the deletion of system files, sending phishing emails, or stealing private data.

**Key Challenge**: The safety defense of computer-use agents relies on refusal training, but this static defense is easily bypassed by meticulously designed indirect instructions. The deeper conflict is that the stronger the underlying model, the more effective the attack becomes, because stronger VLMs generate more accurate execution plans for the attack, creating a capability-safety paradox.

**Goal**: How to systematically evaluate and breach the safety defenses of computer-use agents? An automated attack framework and a standardized evaluation benchmark are required.

**Key Insight**: Instead of directly submitting malicious requests, semantic transformation is employed to "sanitize" the malicious intent to obtain execution plans, which are then re-toxified with the attack payload during execution. This is combined with an iterative feedback mechanism to progressively break through defenses.

**Core Idea**: Detox2tox—first detoxifying to bypass security checks and obtain a step-by-step plan, then re-toxifying to restore malicious content, coupled with dynamic feedback to iteratively upgrade the attack.

## Method

### Overall Architecture

SUDO consists of two phases: a **Static Phase** that generates candidate attack prompts at once via the Detox2tox pipeline, and a **Dynamic Phase** that iteratively optimizes the attack strategy based on the agent's refusal feedback. The overall pipeline is: Malicious Task $\rightarrow$ Detoxifier $\rightarrow$ Instruction Generator to generate step-by-step plans $\rightarrow$ Toxifier to restore toxicity $\rightarrow$ Submit to agent for execution $\rightarrow$ Evaluator checklist assessment $\rightarrow$ Dynamic Updater adjustment based on feedback $\rightarrow$ Re-attack (looping until success or maximum limit).

### Key Designs

1. **Detoxifier**:
    - **Function**: Receives the original malicious command/goal, removes or rewrites obviously harmful content, and generates a detoxified version.
    - **Mechanism**: Hides the malicious intent within seemingly harmless task descriptions (e.g., "send phishing email" $\rightarrow$ "send a formatted email"), preventing the agent's safety guardrails from immediately rejecting it.
    - **Design Motivation**: Direct submission of malicious requests yields a 0% ASR; bypassing the initial security check is a prerequisite. Detoxification is necessary to obtain high-quality execution plans.

2. **Instruction Generator**:
    - **Function**: Passes the detoxified description to a SOTA VLM to generate a detailed, step-by-step execution plan.
    - **Mechanism**: Leverages the strong capabilities of VLMs (including screenshot understanding and system info parsing) to generate actionable step-by-step plans for the disguised "harmless" tasks, supporting various VLM backends (Llama 3.2 Vision, Gemini 2.0, Claude 3.5/3.7, GPT 4.5, o1).
    - **Design Motivation**: Computer-use tasks require precise operational sequences, making generalized execution plans ineffective. Exploiting the VLM's own capabilities to construct bypass plans essentially uses its own strength as a weapon against its defense.

3. **Toxifier**:
    - **Function**: Re-injects the malicious payload into the generated legal execution plans, completing the Detox2tox transformation.
    - **Mechanism**: Replaces benign operations in the plan with attack actions (e.g., replacing benign file deletions with destructive `sudo rm -rf` commands), ultimately outputting attack prompts that appear harmless on the surface but guide harmful behavior in practice.
    - **Design Motivation**: Decoupling the "planning" and "toxic injection" steps bypasses end-to-end safety filters while preserving the structural integrity of the execution plan by replacing only the key operational content.

4. **Dynamic Updater**:
    - **Function**: Automatically adjusts the attack prompt and retries based on evaluation feedback when the attack partially fails.
    - **Mechanism**: Evaluates the completion of each sub-goal using a checklist, adjusts prompt elements or strengthens hidden triggers based on partial failure causes, regenerates the modified prompt via the LLM, and resubmits it to the agent.
    - **Design Motivation**: Static, one-time attacks have a limited ceiling (~20-24% ASR). Iterative feedback systematically dismantles layer-by-layer defenses, significantly improving the success rate with each round.

### Evaluation Method

**Checklist Evaluation Mechanism**: Each attack task is decomposed into multiple topical elements, earning 1 point for each successfully completed element, and an additional 1 point if a jailbreak behavior occurs. The evaluation metric is calculated as:

$$\text{ASR} = \frac{\text{matched\_topics} + 1}{\text{total\_topics} + 1}$$

This fine-grained evaluation not only captures full success or failure but also logs partial successes, providing actionable feedback signals for the Dynamic Updater.

**SUDO Dataset Benchmark**: Manually constructed 50 attack tasks across 4 main risk categories and 12 subcategories (content safety, social risks, legal risks, operational risks). It covers 20 different execution environments (web + desktop), with all tasks running on real operating systems instead of sandboxes.

## Key Experimental Results

### Main Results: Attack Success Rates of Different Instruction Generators

| Model | Static ASR (%) | Dynamic Round 1 (%) | Dynamic Round 2 (%) | Dynamic Round 3 (%) |
|------|-----------|-----------|-----------|-----------|
| Claude 3.5 Haiku | 23.60 | 34.87 (+11.27) | 35.56 (+0.69) | 35.99 (+0.43) |
| Claude 3.7 Sonnet | 24.41 | 29.71 (+5.30) | 32.55 (+2.84) | 38.12 (+5.57) |
| Gemini 2.0 Flash | 24.02 | 30.09 (+6.07) | 32.19 (+2.10) | 32.95 (+0.76) |
| Llama 3.2 Vision | 19.45 | 26.45 (+7.00) | 31.19 (+4.74) | 32.69 (+1.20) |
| GPT 4.5 Preview | 21.29 | 27.99 (+6.70) | 33.82 (+5.83) | **41.33** (+7.51) |
| o1 | 24.05 | 33.79 (+9.74) | 37.29 (+3.50) | 41.09 (+3.80) |

### Comparison with Baseline Jailbreak Methods

| Method | Direct | Role Play | Program Execution | Superior Model | **SUDO** |
|------|--------|-----------|-------------------|---------------|----------|
| ASR (%) | 0.00 | 3.29 | 4.67 | 7.30 | **41.33** |

SUDO achieves a **5.7x** improvement compared to the best baseline, Superior Model.

### Generalization Across Agents (12 Representative Tasks, o1 as Instruction Generator)

| Agent | Static ASR (%) | Dynamic Round 3 ASR (%) |
|-------|-----------|-------------|
| Claude CUA | 16.89 | 34.39 |
| MANUS | 34.86 | 63.19 |
| OmniParser V2 | 41.96 | 66.13 |

### Ablation Study

| Analysis Dimension | Key Findings |
|---------|---------|
| Static vs. Dynamic | Dynamic iteration improves the ASR from ~20-24% to 35-41%, a relative improvement of about 70-100%. |
| Iterative Round Gain | The first round yields the largest gain (+5-11 pp), with subsequent rounds diminishing but still providing positive feedback. |
| Impact of Model Capability | GPT 4.5 and o1 consistently benefit more from iteration, showing that models with stronger reasoning capabilities yield more significant attack improvements. |
| Variations Across Agents | Claude CUA has the strongest defense (lowest ASR), while MANUS and OmniParser are more vulnerable. |

### Key Findings

- **Dynamic iteration is the core contribution**: Static Detox2tox achieves an already non-trivial ~20-24% ASR, which escalates to 41%+ after three rounds of iteration, proving that feedback-driven iterative attacks systematically dismantle safety defenses.
- **Traditional jailbreak methods are nearly obsolete in agent scenarios**: Direct is 0%, Role Play is 3.29%, and Program Execution is 4.67%, indicating that computer-use agents require entirely new attack paradigms.
- **The "capability-safety" paradox**: GPT 4.5 and o1 exhibit the greatest ASR growth in the dynamic phase, verifying that stronger reasoning capabilities actually facilitate the generation of more precise attack plans.
- **Uneven security defenses**: The ASR of MANUS and OmniParser is much higher than that of Claude CUA, demonstrating a massive disparity in safety mechanisms across different agents and a lack of unified industry safety standards.
- **Broad attack coverage**: The heatmap shows successful execution of Detox2tox across all 12 risk subcategories, proving it is not restricted to specific niche scenarios.
- **Diminishing returns in iteration**: ASR gains decrease in later rounds, indicating that defenders might build more effective defenses by reinforcing multi-turn detection.

## Highlights & Insights

- **Ingenious design of the Detox2tox pipeline**: By decoupling "obtaining the execution plan" and "injecting the malicious payload", it uses semantic transformation to bypass end-to-end safety checks. This strategy provides crucial insights for safety research and exposes the fundamental flaws of defense mechanisms relying strictly on content filtering.
- **Evaluation in real environments instead of sandboxes**: Running attack tasks on real operating systems and directly observing whether the agent deletes files or sends emails is far more convincing than traditional text-level evaluations.
- **Fine-grained checklist evaluation**: Moving beyond binary success/failure to capture fine-grained details of partial attack success provides feedback signals for dynamic updates and offers a more reasonable metric for safety evaluation.
- **Exposing the "capability-safety" paradox**: Clearly pointing out that as an external framework, SUDO's impact is amplified as the underlying model's capability improves, which has profound implications for AI safety research methodology.
- **Contribution of standardized benchmarks**: 50 tasks across 4 categories and 12 subcategories in 20 execution environments offer the first systematic benchmark for computer-use agent safety assessment.

## Limitations & Future Work

- **Limited target agent coverage**: The main experiment only evaluated 50 full tasks on Claude CUA due to service availability and login restrictions, while MANUS and OmniParser were evaluated on only 12 sub-sampled tasks.
- **Room for improvement in absolute ASR**: Even after 3 dynamic rounds, the maximum ASR is 41.33%, showing that existing safety measures still have some effect, and leaving room to optimize the attack framework (e.g., more iterations, more complex detox/tox strategies).
- **Multi-agent / Agent-to-Agent scenarios not considered**: With the rise of multi-agent systems, the scalability of Detox2tox in agent-to-agent collaboration scenarios remains unverified.
- **Absence of defense mitigation strategies**: The paper primarily focuses on showcasing attack capabilities without proposing corresponding defense mechanisms or mitigation strategies.
- **Reliability of automated evaluation**: Using LLMs as evaluators can introduce judgment bias, especially when determining partial success in complex tasks.

## Related Work & Insights

| Direction | Representative Work | Difference from SUDO |
|------|---------|-------------|
| Text-based Jailbreak | Liu et al. 2023 (Role Play / Program Exec / Superior Model) | Focuses only on pure-text LLMs, resulting in an ASR <8% on computer-use agents. |
| Web Agent Attacks | AdvWeb (Xu 2024), EIA (Liao 2025) | Leverages web-content prompt injections rather than direct prompt attacks. |
| Agent Safety Evaluation | AgentHarm (Andriushchenko 2025), InjecAgent (Zhan 2024) | Evaluates text-level agent vulnerabilities without involving real system operations. |
| Mobile Agent Safety | MobileSafetyBench (Lee 2024) | Targets indirect prompt injection on Android control agents. |
| **SUDO** | This Work | The first systematic framework to attack computer-use agents, integrating Detox2tox semantic transformation, dynamic iteration, and real-environment evaluation. |

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The "detox $\rightarrow$ generate $\rightarrow$ tox" pipeline of Detox2tox is a brand-new attack paradigm, addressing the previous absence of a systematic framework for computer-use agents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematically evaluated across 6 VLMs, 3 target agents, 50 tasks, and multiple iterations. However, evaluating only 12 tasks on MANUS/OmniParser is slightly insufficient.
- **Writing Quality**: ⭐⭐⭐⭐ The paper features a catchy title (sudo rm -rf), a clearly described framework, a well-organized experiment structure, and consistent nomenclature.
- **Value**: ⭐⭐⭐⭐⭐ Uncovers a major safety blind spot in the emerging paradigm of computer-use agents. The SUDO Dataset provides a standardized benchmark for future safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents](../../NeurIPS2025/llm_agent/agentauditor_humanlevel_safety_and_security_evaluation_for_l.md)
- [\[CVPR 2026\] Universal Guideline-Driven Image Clustering via a Hybrid LLM Agent](../../CVPR2026/llm_agent/universal_guideline-driven_image_clustering_via_a_hybrid_llm_agent.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [\[ACL 2025\] Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](agentic_reasoning_tools.md)
- [\[ACL 2025\] Agentic Knowledgeable Self-Awareness](agentic_knowledgeable_self-awareness.md)

</div>

<!-- RELATED:END -->
