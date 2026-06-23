---
title: >-
  [Paper Note] Do LLM Agents Mirror Socio-Cognitive Effects in Power-Asymmetric Conversations?
description: >-
  [ACL 2026][LLM Agent][Paper Note] This paper simulates power-asymmetric conversations using professional roles and personas, finding that LLM agents replicate socio-cognitive effects such as pronoun usage patterns, language coordination, authoritative persuasion, and harmful compliance. While some effects enhance conversational realism, others introduc
tags:
  - ACL 2026
  - LLM Agent
date: 2026-05-08
content_hash: 6b4377f713fcfbf4
---
# Do LLM Agents Mirror Socio-Cognitive Effects in Power-Asymmetric Conversations?

**Conference**: ACL2026  
**arXiv**: [2605.17694](https://arxiv.org/abs/2605.17694)  
**Code**: https://github.com/nvshrao/power-asymmetric-conversations  
**Area**: LLM Agent / Social Cognitive Evaluation  
**Keywords**: Power asymmetry, language coordination, pronoun effect, authority bias, harmful compliance

## TL;DR
This paper simulates power-asymmetric conversations using professional roles and personas, finding that LLM agents replicate socio-cognitive effects such as pronoun usage patterns, language coordination, authoritative persuasion, and harmful compliance. While some effects enhance conversational realism, others introduce significant safety risks.

## Background & Motivation
**Background**: LLM agents are increasingly deployed in high-stakes conversational scenarios such as healthcare, education, law, and financial consulting. To make agents resemble real interactive partners, researchers often focus on personas, consistency, and general cognitive biases; however, systematic studies on how "power relations" alter agent language and decision-making are scarce.

**Limitations of Prior Work**: Human dialogue does not occur as an information exchange in an egalitarian vacuum. Power disparities between principals and teachers, doctors and nurses, or judges and lawyers influence pronouns, linguistic style, persuasiveness, and obedience. If LLM agents replicate these human social biases under such structures, they may appear more realistic but also become more prone to unsafe compliance under the pressure of high-authority roles.

**Key Challenge**: There is a tension between realism and safety. If a model completely ignores power relations, the dialogue may feel unnatural; however, if it excessively replicates authority bias and submissiveness, it may amplify improper influence and unsafe decision-making.

**Goal**: The authors address 7 research questions to evaluate whether LLM agents exhibit pronoun effects, language coordination, authoritative persuasion, and harmful compliance. They also explore how these effects evolve during a conversation, whether they can be controlled via prompting, and how model size or training stages influence the intensity of these effects.

**Key Insight**: The paper translates four categories of phenomena from social psychology into measurable conversational metrics: first-person singular/plural pronoun usage, language coordination degree, persuasion success, and harmful compliance.

**Core Idea**: By paring high and low-power roles to generate multi-turn dialogues, the study uses linguistic statistics and task outcomes to measure whether agents are influenced by power structures in a human-like manner.

## Method
This paper does not propose a new agent algorithm but constructs a socio-cognitive evaluation pipeline. The core challenge lies in grounding the abstract concept of "power asymmetry" into reproducible personas, role pairings, conversational tasks, and quantitative metrics.

### Overall Architecture
The authors first define 14 pairs of high-low power roles (e.g., Principal-Teacher, Justice-Lawyer, Head Chef-Sous Chef, Lead Developer-Junior Developer). Realistic personas are then sampled from PersonaHub for each role and verified by humans—96.5% of persona pairs were identified as having power disparities by annotators, with a Fleiss's kappa of 0.73. Subsequently, models including Llama 3.1, Qwen 2.5, Phi, GPT-4.1, and GPT-5 were used to simulate paired dialogues of 10 to 15 turns. "Realism-related" effects (pronouns, language coordination) and "safety-related" effects (authoritative persuasion, harmful compliance) were measured for each turn. Finally, these effects were analyzed across three dimensions: dialogue progression, controllability, and model scale/training stage.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    subgraph S1["Dual-layer Role and Persona Construction"]
        direction TB
        A["14 High-Low Power Role Pairs<br/>(Principal-Teacher, Judge-Lawyer, etc.)"] --> B["PersonaHub Sampling + Human Validation<br/>(96.5% Agreement, κ=0.73)"]
    end
    S1 --> C["Multi-model Simulated Paired Dialogue<br/>(Llama/Qwen/Phi/GPT, 10-15 turns)"]
    subgraph S2["Socio-Cognitive Effect Metrics"]
        direction TB
        D["Realism: Pronoun FPS/FPP + Language Coordination D_lc"]
        E["Safety: Persuasion Difference + Harmful Compliance Difference"]
    end
    C --> S2
    subgraph S3["Analysis of Process, Controllability, and Model Factors"]
        direction TB
        F["Dialogue Progression: Start/Middle/End"]
        G["Inhibition of Effects via Control Prompts"]
        H["Model Scale + Post-training (SFT/DPO)"]
    end
    S2 --> S3
    S3 --> I["Conclusion: Stability of Effects"]
```

### Key Designs

**1. Dual-layer Role and Persona Construction: Providing an interpretable and controllable power condition for each dialogue**

Simply prompting "You are a high-power individual" is too abstract to naturally induce social linguistic patterns. The authors utilize a dual-layer setting: the role layer defines the professional identity (status), while the persona layer provides specific background and personality, grounding power asymmetry in realistic professional relationships. To ensure clean identity signals, only samples where the role name appears within the first five words of the persona were kept, and descriptions that weaken current status (e.g., "former", "retired") were filtered out. This approach naturally stimulates power-related linguistic behaviors and facilitates human verification of power disparities.

**2. Socio-Cognitive Effect Metrics: Translating social psychology concepts into statistical agent behaviors**

To cover both realism and safety risks, the authors define quantitative metrics for four phenomena. Pronoun effects are measured by the ratio of first-person singular/plural tokens to total tokens (FPS / FPP). Language coordination is measured by the coordination degree $D_{lc}$ across 8 style markers. Authority bias is evaluated by comparing the difference in persuasion success when initiated by high-status versus low-status roles. Harmful compliance is measured by the difference in compliance rates for unsafe requests initiated by different statuses. While the former two characterize conversational realism, the latter two characterize safety hazards.

**3. Process, Controllability, and Model Analysis: Determining the stability and mitigatability of effects**

To ensure findings were not idiosyncratic to specific prompts or models, the authors cross-validated robustness across three dimensions. The process dimension compares Start / Middle / End stages of dialogue. The controllability dimension tests whether system prompts can inhibit effects by explicitly requesting "High / Low / No effect." The model dimension compares different sizes within the same family and the impact of post-training stages like SFT and DPO. This cross-stage, cross-model analysis provides a comprehensive risk assessment for real-world deployments.

### Loss & Training
This paper does not train new models; the core involves simulation and evaluation. API-based models utilized the Sotopia framework, while offline models used direct prompting to incorporate persona, task information, and dialogue history into the context. Metrics were statistically analyzed or judged after task-specific dialogue generation. For harmful compliance and persuasion tasks, an LLM judge was used alongside human verification (human evaluation results for binary/ternary classification are reported in Table 9).

## Key Experimental Results

### Main Results

| Effect | Model / Metric | Low Power Condition | High Power Condition | Conclusion |
|------|-------------|------------|------------|------|
| Pronoun Effect | GPT-4.1 FPS | 2.32% | 1.66% | High power uses less “I” |
| Pronoun Effect | GPT-4.1 FPP | 2.94% | 3.66% | High power uses more “we” |
| Pronoun Effect | GPT-5 FPS | 1.15% | 0.77% | GPT-5 exhibits this pattern |
| Pronoun Effect | GPT-5 FPP | 3.15% | 3.71% | High power FPP is higher |
| Lang. Coordination | Llama 3.1 70B $D_{lc}$ | 7.1 | 6.4 | Low power coordinates more (diff 0.7) |
| Lang. Coordination | GPT-5 $D_{lc}$ | 4.2 | 4.0 | GPT series shows weaker coordination |
| Persuasion Success | Qwen 2.5 7B | 25.0% | 30.9% | High power is more persuasive |
| Harmful Compliance | GPT-4.1 | 6.1% | 9.8% | High power triggers more unsafe compliance |

### Ablation Study

| Analysis Dimension | Key Metric | Description |
|------|---------|------|
| Dialogue Position | Llama 3.1 8B persuasion diff: 6.1 (Start) to 5.7 (End) | Persuasion and harmful compliance are stronger early on |
| Dialogue Position | Lang. coordination remains stable across S/M/E | Linguistic coordination is more persistent than persuasion/compliance |
| Model Size | Llama 3.1 8B persuasion diff: 6.1, 70B diff: 1.6 | Larger models exhibit weaker authoritative persuasion differences |
| Model Size | Qwen 2.5 7B harmful compliance diff: 1.8, 72B diff: 0.9 | Larger models may mitigate some safety risk effects |
| Post-training | Changes in metrics for SFT vs DPO are minimal | Preference tuning has limited influence on these socio-cognitive effects |
| Control Prompt | GPT persuasion/compliance near 0 under No control | Safety-related effects in closed GPT models are easier to control explicitly |

### Key Findings
- Most models exhibit pronoun effects (except Qwen and Phi); the GPT series is particularly strong, suggesting powerful models easily replicate power-related linguistic patterns for realism.
- All non-GPT models demonstrate language coordination, though it often manifests as mutual coordination; the asymmetry (low-power individual coordinating more) is weaker than human theoretical expectations.
- Requests from high-power roles are generally more persuasive and more likely to induce harmful compliance, directly linking "social realism" to safety risks.

## Highlights & Insights
- The primary contribution is transforming "power differential" from a sociological concept into a controllable variable within an agent benchmark, which is more relevant to real-world deployment than generic persona consistency.
- Measuring both realism and safety concurrently is valuable: pronoun effects and language coordination may make agents appear more natural, while authority bias and harmful compliance make them more dangerous.
- The results serve as a reminder that agent safety is not just about "rejecting explicitly harmful requests" but also includes maintaining judgment under pressure from hierarchy, identity, and authority.

## Limitations & Future Work
- The authors acknowledge that all experiments are text-based simulations, lacking cues from emotion, physical setting, multimodality, and long-term relationships present in real human-AI interaction.
- Power is approximated only by professional roles and personas, which cannot capture complex factors like cultural context, organizational systems, or intersectional identities.
- While covering six representative models, the study does not represent all modern LLMs; different architectures and alignment strategies may alter effect intensity.
- Control experiments relied on explicit system-level instructions; future research is needed to investigate if these effects can be stabilized or regulated at the representation or training objective level.

## Related Work & Insights
- **Vs personality alignment**: Previous work focuses on whether a model stably expresses a personality; this study focuses on the relational structure, specifically how high/low power jointly shapes linguistic behavior.
- **Vs cognitive bias benchmarks**: Conventional bias evaluations are often static Q&A; this study places authority bias and harmful compliance into multi-turn agent dialogues, reflecting deployment risks.
- **Vs Sotopia-style social simulation**: While Sotopia provides a framework for dialogue simulation, this paper overlays social psychological metrics to answer specific theoretical questions regarding simulation outcomes.
- **Insight**: Safety evaluations for agents in medical, educational, or legal domains should incorporate asymmetric relations (e.g., doctor-patient, teacher-student) rather than testing only anonymous user requests.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematic introduction of power asymmetry and socio-cognitive effects to LLM agent evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Wide coverage of metrics, models, and roles, though limited to simulated text environments.
- Writing Quality: ⭐⭐⭐⭐☆ Research questions are clearly organized and map directly to tables.
- Value: ⭐⭐⭐⭐⭐ Vital implications for high-stakes agent deployment, highlighting the need for safety evaluations to include social relational pressure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SafeMCP: Proactive Power Regulation for LLM Agent Defense via Environment-Grounded Look-Ahead Reasoning](safemcp_proactive_power_regulation_for_llm_agent_defense_via_environment-grounde.md)
- [\[AAAI 2026\] From Biased Chatbots to Biased Agents: Examining Role Assignment Effects on LLM Agent Robustness](../../AAAI2026/llm_agent/from_biased_chatbots_to_biased_agents_examining_role_assignment_effects_on_llm_a.md)
- [\[AAAI 2026\] Physics-Informed Autonomous LLM Agents for Explainable Power Electronics Modulation Design](../../AAAI2026/llm_agent/physics-informed_autonomous_llm_agents_for_explainable_power_electronics_modulat.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[CVPR 2026\] Learning to Adapt: Self-Improving Web Agent via Cognitive-Aware Exploration](../../CVPR2026/llm_agent/learning_to_adapt_self-improving_web_agent_via_cognitive-aware_exploration.md)

</div>

<!-- RELATED:END -->
