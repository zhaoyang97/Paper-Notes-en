---
title: >-
  [Paper Note] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?
description: >-
  [ACL 2026][LLM Safety][Computer-Using Agent] This paper provides the first systematic review of safety research for "Computer-Using Agents (CUA)," organizing 124 relevant papers into a four-dimensional framework of "Internal Threats × External Threats × Defense × Evaluation," and highlighting that the primary gaps in existing CUAs are UI grounding robustness and cross-platform adversarial evaluation.
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Computer-Using Agent"
  - "GUI Agent"
  - "Safety and Security"
  - "prompt injection"
  - "defense taxonomy"
date: 2026-05-08
content_hash: b7765ce8875e57c5
---

# A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?

**Conference**: ACL 2026  
**arXiv**: [2505.10924](https://arxiv.org/abs/2505.10924)  
**Code**: None  
**Area**: Multimodal VLM (Survey)  
**Keywords**: Computer-Using Agent, GUI Agent, Safety and Security, prompt injection, defense taxonomy

## TL;DR
This paper provides the first systematic review of safety research for "Computer-Using Agents (CUA)," organizing 124 relevant papers into a four-dimensional framework of "Internal Threats × External Threats × Defense × Evaluation," and highlighting that the primary gaps in existing CUAs are UI grounding robustness and cross-platform adversarial evaluation.

## Background & Motivation
**Background**: CUAs represented by OpenAI's o3/o4-mini, PC-Agent, AppAgent, and SeeAct combine multimodal perception, reasoning, and GUI automation. They can autonomously complete complex tasks like "online shopping, booking flights, and filling forms" and are being rapidly deployed in industry.

**Limitations of Prior Work**: CUAs simultaneously stack "inherent LLM vulnerabilities + GUI control permissions + multimodal input," drastically expanding the attack surface. Visual grounding errors, response delays, and HTML parsing traps can be abused as entry points for data leakage or target hijacking. Current safety research is scattered across isolated directions like prompt injection, jailbreak, and backdoor, lacking a unified perspective.

**Key Challenge**: Safety research on individual LLMs cannot be directly migrated—CUAs include an untrusted channel known as the "environment" (GUI states, external APIs), leading to the emergence of many previously nonexistent attack surfaces (environment injection, reasoning gap, web hacking). Meanwhile, "autonomous operation rights" mean any successful attack directly results in physical-world consequences (fund transfers, file deletions).

**Goal**: (i) Provide a unified definition of CUA suitable for safety analysis; (ii) systematically enumerate all internal/external threats; (iii) align existing defense solutions with threats; (iv) summarize evaluation benchmarks, metrics, and datasets.

**Key Insight**: Starting from the architectural view of "Agent = Perception/Brain/Action trio + Environment," enumerating threats by using each category of components as an attack surface, and then reverse-mapping corresponding defenses. This "component-centric" perspective is less likely to miss new types of attacks than "classification by attack name."

**Core Idea**: Use an extensible three-axis taxonomy of "Threat Source × Affected Component × Threat Model" to unify fragmented CUA safety research and expose real safety gaps.

## Method

### Overall Architecture
This survey addresses the issue of CUA safety research being "scattered and lacking a unified perspective" by organizing 124 papers into a three-layer threat-defense correspondence system. It first provides a unified definition of CUA suitable for safety analysis—deconstructing the agent into the "Perception / Brain / Action" trio plus an untrusted "environment" channel, and subdividing it into four sub-types: OS Agent, GUI Agent, Web Agent, and Device Control Agent. It then enumerates 8 types of internal threats and 8 types of external threats anchored by components, reverse-mapping 14 types of defense methods for one-to-one alignment. Finally, it presents a panoramic view of benchmarks and metrics categorized by platform (Web / Mobile / General). The main logic is "define attack surfaces → enumerate threats → align defenses → provide evaluation map."

### Key Designs

**1. Three-Axis Threat Taxonomy: Using "Source × Component × Threat Model" instead of "Classification by Attack Name" ensures the taxonomy remains future-proof.**

Traditional safety surveys classify by attack name (jailbreak, backdoor, etc.), which requires adding a new category whenever a new attack emerges, making the framework increasingly cluttered. This paper instead labels each threat on three axes—Threat Source (Env / Prompt / Model / User) × Affected Component (Perception / Brain / Action) × Threat Model, plus a "timing" label (development / deployment / architecture / training). In this matrix, internal threats stem from the agent's own defects: the perception layer faces "UI understanding and grounding difficulties"; the brain layer concentrates six types including "scheduling errors, goal misalignment, hallucinations, extra-long context, socio-cultural bias, and response delay"; the action layer faces "API calling errors." External threats come from attackers: adversarial attacks, direct/indirect prompt injection, jailbreak, memory extraction and injection, backdoor, reasoning gap attack (multimodal signal conflict), system sabotage, and web hacking. Because labels are anchored to components rather than attack names, new attacks only need to be categorized under the corresponding component, and the taxonomy does not need to be reconstructed.

**2. Defense-Threat Alignment Matrix: Explicitly mapping 14 defense categories to 16 threat categories allows engineers to select combinations accordingly.**

Simply listing threats is not directly useful for engineering—readers need to know "which methods exist to defend against attack X" or "how many threats a certain defense can cover." This paper labels 14 defense categories (environmental constraints, input validation, defensive prompts, data purification, adversarial training, output monitoring, model auditing, cross-verification, continual learning, transparency, topology guidance, perceptual synergy, planning architecture enhancement, compliance rules) with the same three-axis labels—target component, strengthened framework element, and corresponding threat ID (In./Ex.X)—and aligns them with internal/external threats. For example, "environmental constraints" primarily defend against external threats (limiting agent permissions for GUI interaction), while "planning architecture enhancement" simultaneously mitigates internal scheduling errors and external reasoning-gap attacks. Consequently, web application developers can directly look up a combination like "environmental constraints + input validation + cross-verification" to avoid reinventing the wheel.

**3. Cross-platform Benchmark/Metric Panorama: Laying out an evaluation map by platform and categorizing mixed indicators into three groups.**

CUA platform differences are significant—Mobile screens are constrained, Web text is dynamic, and desktop APIs are complex, making it difficult to select the appropriate benchmark. This paper organizes safety benchmarks into three platforms: Web / Mobile / General-purpose, and groups scattered metrics into three categories: Task Completion (TSR, Helpfulness), Intermediate Steps (SSR, Total Correct Prefix), and Safety Robustness (ASR, CuP, F1, RR, LR, AR, TS). Measurement methods are also categorized into Rule-based, LLM-as-a-judge, and Manual. This map allows researchers to quickly select suitable evaluation environments based on target platforms and areas of concern.

### Loss & Training
This paper is a survey and does not involve independent training. The authors summarized core training objectives on the defense side for reference: adversarial training (injecting adversarial samples to improve robustness), data purification (removing poisoned samples), continual learning and self-evolution (updating strategies online based on environmental feedback), and compliance rule learning (encoding SOP / ethical norms into training objectives).

## Key Experimental Results

### Main Results: CUA Safety Benchmark Panorama (Selected)

| Benchmark | Platform | Primary Threat Focus | Evaluation Method |
|-----------|------|-------------|----------|
| ST-WebAgentBench | Web | safety + trustworthiness | Rule-based |
| BrowserART | Web | jailbreak | LLM-as-judge |
| PrivacyLens | Web/General | privacy leakage | LLM-as-judge |
| MobileSafetyBench | Mobile | various safety risks | Rule-based |
| Hijacking JARVIS | Mobile | third-party injection | Rule-based |
| AgentDojo | General | prompt injection | Rule-based |
| AgentHarm | General | harmful instructions | LLM-as-judge |
| OS-Harm | General | OS-level attacks | LLM-as-judge |
| TrustAgent | General | comprehensive trust | LLM-as-judge |
| RedTeamCUA | Hybrid Web/OS | red teaming | LLM-as-judge |
| AgentHazard | General | harmful actions | LLM-as-judge |

### Ablation Study: Threat-Component Influence Matrix (Simplified)

| Threat Category | Primary Source | Main Affected Component | Trigger Stage |
|----------|----------|-------------|---------|
| UI Grounding Difficulty | Env | Perception | Development |
| Scheduling Error | Prompt | Brain | Development |
| Goal Misalignment | Prompt | Brain | Deployment |
| Hallucination | Prompt/Model | Brain | Deployment |
| Adversarial Attack | Env | Perception | Runtime |
| Prompt Injection | Env/Prompt | Brain/Action | Runtime |
| Memory Attack | Model | Brain/Action | Runtime |
| Backdoor | Model | Brain/Action | Training |
| Reasoning Gap | Model | Perception | Runtime |
| Web Hacking | Env | Action | Runtime |

### Key Findings
- **Brain component is the most critical vulnerability**: Out of 8 internal threat types, 6 directly attack or pass through the Brain; for external threats, it's 5 out of 8. This indicates that the LLM's reasoning/planning layer is the weak core of CUAs.
- **Environment injection is a unique CUA attack surface**: Indirect prompt injection (contaminating webpages, files, or UI elements) is almost exclusively valid in CUA settings and is completely missed by traditional LLM benchmarks.
- **Evaluation is overly concentrated on Web platforms**: Among the benchmarks listed in this survey, those for Web are the most numerous, while Mobile and hybrid environments are seriously underrepresented, lacking consistent safety assessment standards across platforms.

## Highlights & Insights
- **Component-centric taxonomy**: Anchoring threats to Perception/Brain/Action rather than classifying by attack name is a sustainable design; new attacks only need to be assigned to existing components without reconstructing the framework.
- **Actionable defense-threat matrix**: The authors explicitly map 14 defense categories to 16 threat categories, allowing engineers to select combinations (e.g., for web apps: environmental constraints + input validation + cross-verification).
- **"Reasoning gap attack" formally included for the first time**: Attacks that induce agent errors through multimodal signal conflicts (e.g., popup text in a screenshot inconsistent with HTML) are unique to CUAs, necessitating multimodal alignment-based defenses.
- **Clearly identifies transparency as a governance bottleneck**: The authors point out that vendors like OpenAI have not publicised safety policies or evaluation results, calling for the establishment of independent auditing and disclosure frameworks.

## Limitations & Future Work
- **Author's acknowledgment**: Coverage is limited to public English literature available by the submission deadline, potentially missing emerging attacks and internal industrial research; the analysis remains at the architectural level without empirical evaluation of the relative effectiveness of various defenses.
- **Additional limitations**: The labeling of "affected component" in the threat matrix depends on manual judgment, resulting in definitional overlap for ambiguous attacks (e.g., reasoning gap attacks affecting both Perception and Brain). The defense-threat alignment matrix does not distinguish coverage levels between "primary defense" and "secondary defense."
- **Future directions**: Future work should conduct end-to-end empirical testing—running all defense combinations on a unified benchmark to provide a Pareto frontier—and establish multi-lingual, cross-cultural safety evaluation extensions, as current benchmarks are almost entirely based on English GUIs.

## Related Work & Insights
- **vs LLM Safety Surveys** (Shi 2024, Ma 2025): They focus on model-layer threats (jailbreak, hallucination); this paper expands the perspective to the "Model + Environment + Multimodal" coupling, revealing the unique environment injection attack surface of CUAs.
- **vs OS Agent Surveys** (Hu 2024): They focus on capability summaries (architecture, memory, planning); this paper specializes in the safety dimension, forming a complementary relationship.
- **vs Sager et al. 2025** (AI Agents for Computer Use survey): They sort out the capabilities and deployment of GUI agents; this paper provides the "safety checklist" that must be resolved before these agents are deployed.

## Rating
- Novelty: ⭐⭐⭐⭐ The first safety survey targeting CUA; the "component-centric" taxonomy and introduction of reasoning gap attacks are original.
- Experimental Thoroughness: ⭐⭐⭐ A survey-based work covering 124 papers but lacking original experiments; defense comparisons lack end-to-end empirical evidence.
- Writing Quality: ⭐⭐⭐⭐ The three-axis matrix and defense-threat alignment table are very engineering-friendly; the JARVIS/Ultron metaphor enhances the reach of the safety research.
- Value: ⭐⭐⭐⭐⭐ A must-read manual for engineering teams deploying CUAs; clearly points out mobile/cross-platform evaluation and transparency governance as two gap areas for academic researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LeakDojo: Decoding the Leakage Threats of RAG Systems](leakdojo_decoding_the_leakage_threats_of_rag_systems.md)
- [\[ICLR 2026\] Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents](../../ICLR2026/llm_safety/breaking_agent_backbones_evaluating_the_security_of_backbone_llms_in_ai_agents.md)
- [\[ACL 2026\] Why Agents Compromise Safety Under Pressure](why_agents_compromise_safety_under_pressure.md)
- [\[ACL 2026\] On Safety Risks in Experience-Driven Self-Evolving Agents](on_safety_risks_in_experience-driven_self-evolving_agents.md)
- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)

</div>

<!-- RELATED:END -->
