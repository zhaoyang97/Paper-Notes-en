---
title: >-
  [Paper Note] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?
description: >-
  [ACL 2026][LLM Safety][Computer-Using Agent] This paper provides the first systematic review of safety research for "Computer-Using Agents (CUA)," organizing 124 relevant papers into a four-dimensional framework of "Intrinsic Threats $\times$ Extrinsic Threats $\times$ Defense $\times$ Evaluation." It identifies the lack of UI grounding robustness and cross-platf
tags:
  - ACL 2026
  - LLM Safety
  - Computer-Using Agent
  - prompt injection
  - defense taxonomy
date: 2026-05-08
content_hash: 344f86ce7a82f322
---
# A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?

**Conference**: ACL 2026  
**arXiv**: [2505.10924](https://arxiv.org/abs/2505.10924)  
**Code**: None  
**Area**: Multimodal VLM (Survey)  
**Keywords**: Computer-Using Agent, GUI Agent, Security Threats, prompt injection, defense taxonomy

## TL;DR
This paper provides the first systematic review of safety research for "Computer-Using Agents (CUA)," organizing 124 relevant papers into a four-dimensional framework of "Intrinsic Threats $\times$ Extrinsic Threats $\times$ Defense $\times$ Evaluation." It identifies the lack of UI grounding robustness and cross-platform adversarial evaluation as the biggest gaps in current CUA research.

## Background & Motivation
**Background**: Represented by OpenAI's o3/o4-mini, PC-Agent, AppAgent, and SeeAct, CUAs combine multimodal perception, reasoning, and GUI automation. They can autonomously complete complex tasks such as "online shopping, booking flights, and filling forms," and are being rapidly deployed in industry.

**Limitations of Prior Work**: CUAs simultaneously overlay "LLM inherent vulnerabilities + GUI control permissions + multimodal input," drastically expanding the attack surface. Visual grounding errors, response delays, and HTML parsing traps can all be exploited as entry points for data leakage or goal hijacking. Current safety research remains scattered across isolated directions like prompt injection, jailbreak, and backdoors, lacking a unified perspective.

**Key Challenge**: Safety research for standalone LLMs cannot be directly migrated—CUAs introduce the "environment" (GUI states, external APIs) as an untrusted channel, giving rise to many previously non-existent attack surfaces (environment injection, reasoning gap, web hacking). Meanwhile, "autonomous operation rights" mean any successful attack leads directly to physical-world consequences (fund transfers, file deletions).

**Goal**: (i) Provide a unified definition of CUA suitable for safety analysis; (ii) systematically enumerate all intrinsic/extrinsic threats; (iii) align existing defense solutions with threats; (iv) summarize evaluation benchmarks, metrics, and datasets.

**Key Insight**: Starting from an architectural view of "Agent = Perception/Brain/Action Triad + Environment," this work enumerates threats by treating each component as an attack surface, then reversely organizes corresponding defenses. This "component-centric" perspective is less likely to miss novel attacks than "classification by attack name."

**Core Idea**: An extensible classification framework with three axes—"Threat Source $\times$ Affected Component $\times$ Threat Model"—is used to unify fragmented CUA safety research and expose genuine security gaps.

## Method

### Overall Architecture
This survey addresses the problem of fragmented CUA safety research by organizing 124 papers into a three-layer threat-defense correspondence system. It first provides a unified definition of CUA suitable for safety analysis—deconstructing the agent into a "Perception / Brain / Action" triad with an untrusted "Environment" channel, and categorizing sub-types into OS Agents, GUI Agents, Web Agents, and Device Control Agents. It then enumerates 8 types of intrinsic threats and 8 types of extrinsic threats anchored to components, reversely mapping 14 types of defense methods. Finally, it presents a landscape of benchmarks and metrics categorized by platform (Web / Mobile / General). The main line of reasoning is: "Define attack surfaces $\rightarrow$ Enumerate threats $\rightarrow$ Align defenses $\rightarrow$ Map evaluations."

### Key Designs

**1. Three-Axis Threat Taxonomy: Using "Source $\times$ Component $\times$ Threat Model" instead of "Classification by Attack Name" for a future-proof taxonomy.**

Traditional security surveys categorize by attack names (jailbreak, backdoor, etc.), requiring new categories for every new attack. This paper instead tags each threat with three axes—Threat Source (Env / Prompt / Model / User) $\times$ Affected Component (Perception / Brain / Action) $\times$ Threat Model, plus a stage label for "when it occurs" (development / deployment / architecture / training). In this matrix, intrinsic threats stem from the agent's own defects: the perception layer faces "UI understanding and grounding difficulties"; the brain layer concentrates six types including "scheduling errors, goal mismatch, hallucination, long context issues, socio-cultural bias, and response delay"; the action layer involves "API call errors." Extrinsic threats come from attackers: adversarial attacks, direct/indirect prompt injection, jailbreak, memory extraction and injection, backdoor, reasoning gap attacks (multimodal signal conflict), system sabotage, and web hacking. Since labels are anchored to components rather than attack names, new attacks can simply be categorized into the corresponding component without restructuring the taxonomy.

**2. Defense-Threat Alignment Matrix: Explicitly mapping 14 defense types to 16 threat types for engineering selection.**

Enumerating threats alone is insufficient for engineering; practitioners need to know "which methods defend against attack X" or "how many threats a defense covers." This paper tags 14 defense types (environmental constraints, input validation, defensive prompts, data sanitization, adversarial training, output monitoring, model auditing, cross-verification, continuous learning, transparency, topology-guided, perception synergy, planning architecture enhancement, compliance rules) with the same three axes—target component, reinforced framework element, and corresponding threat ID (In./Ex.X). For instance, "environmental constraints" mainly defend against extrinsic threats (limiting agent permissions for GUI interaction), while "planning architecture enhancement" mitigates both intrinsic scheduling errors and extrinsic reasoning-gap attacks. This allows practitioners to directly select combinations like "environmental constraints + input validation + cross-verification" for web applications.

**3. Cross-Platform Benchmark/Metric Panorama: Presenting an evaluation map by platform and grouping mixed metrics into three sets.**

CUA platform differences are significant—Mobile has screen constraints, Web has dynamic text, and Desktop has complex APIs. This paper organizes safety benchmarks into Web / Mobile / General-purpose platforms and groups scattered metrics into three categories: Task Completion (TSR, Helpfulness), Intermediate Steps (SSR, Total Correct Prefix), and Safety Robustness (ASR, CuP, F1, RR, LR, AR, TS). Measurement methods are also classified into Rule-based, LLM-as-a-judge, and Manual. This map allows researchers to quickly select appropriate evaluation environments based on target platforms and dimensions of interest.

### Loss & Training
As a survey, this work does not involve independent training. The authors summarize core training objectives from the defense side for reference: adversarial training (injecting adversarial samples to improve robustness), data sanitization (removing poisoned samples), continuous learning and self-evolution (online policy updates based on environmental feedback), and compliance rule learning (encoding SOP / ethical norms into training objectives).

## Key Experimental Results

### Main Results: Panorama of CUA Safety Benchmarks (Excerpt)

| Benchmark | Platform | Main Threat Focus | Evaluation Method |
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
| RedTeamCUA | Hybrid Web/OS | red-teaming | LLM-as-judge |
| AgentHazard | General | harmful behaviors | LLM-as-judge |

### Ablation Study: Threat-Component Influence Matrix (Simplified)

| Threat Category | Main Source | Primary Affected Component | Trigger Stage |
|----------|----------|-------------|---------|
| UI Grounding Difficulty | Env | Perception | Development |
| Scheduling Error | Prompt | Brain | Development |
| Goal Mismatch | Prompt | Brain | Deployment |
| Hallucination | Prompt/Model | Brain | Deployment |
| Adversarial Attack | Env | Perception | Runtime |
| Prompt Injection | Env/Prompt | Brain/Action | Runtime |
| Memory Attack | Model | Brain/Action | Runtime |
| Backdoor | Model | Brain/Action | Training |
| Reasoning Gap | Model | Perception | Runtime |
| Web Hacking | Env | Action | Runtime |

### Key Findings
- **Brain Component is the Highest Risk Position**: Out of 8 intrinsic and 8 extrinsic threat types, 6 and 5 respectively directly attack or pass through the Brain, indicating that the LLM reasoning/planning layer is the weak core of CUA.
- **Environment Injection is a CUA-Specific Attack Surface**: Indirect prompt injection (contaminating webpages, files, UI elements) is almost exclusively valid in CUA settings and is completely missed by traditional LLM benchmarks.
- **Evaluation is Overly Concentrated on Web Platforms**: Most benchmarks listed are Web-based; Mobile and hybrid environments are severely under-represented, lacking consistent safety assessment standards across platforms.

## Highlights & Insights
- **Component-Centric Taxonomy**: Anchoring threats to "Perception/Brain/Action" rather than attack names is a "evergreen" design; new attacks can simply be mapped to existing components.
- **Actionable Defense-Threat Matrix**: The explicit mapping of 14 defenses to 16 threats allows engineers to select combinations (e.g., for web apps: environmental constraints + input validation + cross-verification).
- **"Reasoning Gap Attack" Formally Included**: Inducing agent errors through multimodal signal conflicts (e.g., pop-up text in screenshots inconsistent with HTML) is unique to CUAs, requiring future multimodal alignment-based defenses.
- **Transparency as a Governance Bottleneck**: The authors note that providers like OpenAI have not disclosed safety policies or evaluation results, calling for independent audit and disclosure frameworks.

## Limitations & Future Work
- **Limitations**: Coverage is limited to public English literature; potential omission of emerging attacks and internal industrial research. Architectural analysis is provided without empirical evaluation of the relative effectiveness of defenses.
- **Additional Constraints**: "Affected component" labels rely on manual judgment; definitions for ambiguous attacks (e.g., reasoning gaps affecting both Perception and Brain) may overlap. The alignment matrix does not distinguish between primary and secondary coverage levels.
- **Future Work**: End-to-end empirical testing across a unified benchmark to provide a Pareto frontier of defense combinations. Expansion to multilingual and cross-cultural safety evaluations, as current benchmarks are almost entirely based on English GUIs.

## Related Work & Insights
- **vs LLM Safety Surveys** (Shi 2024, Ma 2025): They focus on model-layer threats (jailbreak, hallucination); this work expands the scope to "Model + Environment + Multimodal" coupling, revealing the environment injection surface unique to CUA.
- **vs OS Agent Surveys** (Hu 2024): They focus on capabilities (architecture, memory, planning); this work specializes in the safety dimension, forming a complementary relationship.
- **vs Sager et al. 2025** (AI Agents for Computer Use Survey): They organize GUI agent capabilities and implementation; this work provides the "safety checklist" necessary before deployment.

## Rating
- Novelty: ⭐⭐⭐⭐ First safety survey specifically for CUA; the "component-centric" taxonomy and introduction of reasoning gap attacks are original.
- Experimental Thoroughness: ⭐⭐⭐ A survey work covering 124 papers but lacking original experiments; lacks empirical comparison of defense efficacy.
- Writing Quality: ⭐⭐⭐⭐ The three-axis matrix and alignment table are very engineering-friendly; the JARVIS/Ultron metaphor is effective for dissemination.
- Value: ⭐⭐⭐⭐⭐ A must-read manual for engineering teams deploying CUAs; clearly identifies mobile/cross-platform evaluation and transparency governance as empty academic directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why Agents Compromise Safety Under Pressure](why_agents_compromise_safety_under_pressure.md)
- [\[ACL 2026\] On Safety Risks in Experience-Driven Self-Evolving Agents](on_safety_risks_in_experience-driven_self-evolving_agents.md)
- [\[ACL 2026\] LeakDojo: Decoding the Leakage Threats of RAG Systems](leakdojo_decoding_the_leakage_threats_of_rag_systems.md)
- [\[ACL 2026\] AgentMark: Utility-Preserving Behavioral Watermarking for Agents](agentmark_utility-preserving_behavioral_watermarking_for_agents.md)
- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)

</div>

<!-- RELATED:END -->
