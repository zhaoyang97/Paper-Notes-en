---
title: >-
  [Paper Note] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?
description: >-
  [ACL 2026][LLM Safety][Computer-Using Agent] This paper provides the first systematic review of safety research for "Computer-Using Agents (CUA)"…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Computer-Using Agent"
  - "GUI Agent"
  - "safety threats"
  - "prompt injection"
  - "defense taxonomy"
date: 2026-05-08
content_hash: 46c17c03afe251e9
---

# A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?

**Conference**: ACL 2026  
**arXiv**: [2505.10924](https://arxiv.org/abs/2505.10924)  
**Code**: None  
**Area**: Multimodal VLM (Survey)  
**Keywords**: Computer-Using Agent, GUI Agent, safety threats, prompt injection, defense taxonomy

## TL;DR
This paper provides the first systematic review of safety research for "Computer-Using Agents (CUA)", organizing 124 relevant papers into a four-dimensional classification framework of "intrinsic threats $\times$ extrinsic threats $\times$ defense $\times$ evaluation." It highlights that the most significant gaps in current CUA research are UI grounding robustness and cross-platform adversarial evaluation.

## Background & Motivation
**Background**: CUAs, represented by OpenAI's o3/o4-mini, PC-Agent, AppAgent, and SeeAct, combine multimodal perception, reasoning, and GUI automation. They can autonomously complete complex tasks such as "online shopping, booking flights, and filling forms" and are being rapidly deployed in industry.

**Limitations of Prior Work**: CUAs aggregate "inherent LLM vulnerabilities + GUI control permissions + multimodal inputs," significantly expanding the attack surface. Visual grounding errors, response delays, and HTML parsing traps can be exploited as entry points for data leakage or target hijacking. Current safety research remains scattered across isolated directions like prompt injection, jailbreak, and backdoors, lacking a unified perspective.

**Key Challenge**: Safety research for standalone LLMs cannot be directly transferred. CUAs introduce the "environment" (GUI states, external APIs) as an untrusted channel, giving rise to unique attack surfaces such as environment injection, reasoning gaps, and web hacking. Furthermore, "autonomous operation rights" mean successful attacks lead directly to physical world consequences, such as fund transfers or file deletion.

**Goal**: (i) Provide a unified definition of CUA suitable for safety analysis; (ii) Systematically enumerate all intrinsic and extrinsic threats; (iii) Align existing defense solutions with these threats; (iv) Summarize evaluation benchmarks, metrics, and datasets.

**Key Insight**: Starting from the architectural view of "Agent = Perception/Brain/Action + Environment," each component is treated as an attack surface to enumerate threats, followed by a reverse mapping of corresponding defenses. This "component-centric" perspective is more robust against missing new types of attacks compared to traditional "attack-name-based" classification.

**Core Idea**: Utilize a scalable three-axis classification framework of "threat source $\times$ affected component $\times$ threat model" to unify fragmented CUA safety research and expose critical security gaps.

## Method

### Overall Architecture
The paper unfolds around a three-layer threat-defense comparison system: it defines CUA (including OS Agent, GUI Agent, Web Agent, and Device Control Agent), lists 8 categories of intrinsic threats and 8 categories of extrinsic threats, details 14 types of defense methods, and provides a comprehensive panorama of benchmarks/metrics categorized by platform (Web, Mobile, General).

### Key Designs
1.  **Three-axis Threat Taxonomy** (Section 3):
    - **Function**: Labels each threat using a matrix taxonomy of "Threat Source (Env/Prompt/Model/User) $\times$ Affected Component (Perception/Brain/Action) $\times$ Threat Model."
    - **Mechanism**: Intrinsic threats originate from agent defects tied to components: Perception (UI understanding and grounding difficulties); Brain (scheduling errors, goal mismatch, hallucinations, context length, bias, response delay); and Action (API call errors). Extrinsic threats originate from attackers, including adversarial attacks, prompt injection (direct and indirect), jailbreak, memory attacks (extraction and injection), backdoors, reasoning gap attacks (multimodal signal conflict), system sabotage, and web hacking.
    - **Design Motivation**: Traditional surveys categorize by "attack name," requiring expansion for every new attack; this component-based labeling allows new attacks to be categorized into existing components, keeping the taxonomy stable.

2.  **Defense-Threat Alignment Matrix** (Section 4):
    - **Function**: Aligns 14 defense methods (environmental constraints, input validation, defensive prompts, data sanitization, adversarial training, output monitoring, model auditing, cross-verification, continual learning, transparency, topology-guided, perceptual synergy, planning architecture enhancement, and compliance rules) with specific intrinsic/extrinsic threats.
    - **Mechanism**: Each defense is labeled with three axes: target component (Env/Prompt/Model/User), reinforced framework element (Perception/Brain/Action), and corresponding threat ID (In./Ex.X). For instance, "environmental constraints" primarily target extrinsic threats by limiting agent permissions.
    - **Design Motivation**: Enables researchers to quickly identify available methods for specific attacks and evaluate the coverage of various defenses.

3.  **Cross-Platform Benchmark/Metric Panorama** (Section 5):
    - **Function**: Organizes CUA safety benchmarks by Web, Mobile, and General-purpose platforms, categorizing metrics into task completion (TSR, Helpfulness), intermediate steps (SSR, Total Correct Prefix), and safety robustness (ASR, CuP, F1, RR, LR, AR, TS).
    - **Design Motivation**: CUAs face vastly different platform challenges (Mobile screen constraints, dynamic Web text, complex Desktop APIs); a unified panorama facilitates the selection of appropriate evaluation environments.

### Loss & Training
As a survey, this paper does not propose independent training. However, it summarizes core training objectives on the defense side: adversarial training (injecting adversarial samples), data sanitization (removing poisoned samples), continual learning and self-evolution (updating policies online based on environmental feedback), and compliance rule learning (encoding SOPs and ethical norms into training objectives).

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
| RedTeamCUA | Hybrid Web/OS | red-teaming | LLM-as-judge |
| AgentHazard | General | harmful behaviors | LLM-as-judge |

### Ablation Study: Threat-Component Impact Matrix (Simplified)

| Threat Category | Primary Source | Primarily Affected Component | Trigger Phase |
|----------|----------|-------------|---------|
| UI Grounding Difficulty | Env | Perception | Development |
| Scheduling Errors | Prompt | Brain | Development |
| Goal Mismatch | Prompt | Brain | Deployment |
| Hallucination | Prompt/Model | Brain | Deployment |
| Adversarial Attack | Env | Perception | Runtime |
| Prompt Injection | Env/Prompt | Brain/Action | Runtime |
| Memory Attack | Model | Brain/Action | Runtime |
| Backdoor | Model | Brain/Action | Training |
| Reasoning Gap | Model | Perception | Runtime |
| Web Hacking | Env | Action | Runtime |

### Key Findings
- **The Brain component is the most critical vulnerability**: 6 out of 8 intrinsic threats and 5 out of 8 extrinsic threats either target or pass through the Brain, indicating that the LLM's reasoning/planning layer is the weak core of CUAs.
- **Environment injection is a CUA-specific attack surface**: Indirect prompt injection (polluting webpages, files, or UI elements) is almost exclusively valid in CUA settings and is completely missed by traditional LLM benchmarks.
- **Evaluations are over-concentrated on Web platforms**: Most benchmarks focus on Web settings; Mobile and hybrid environments are severely underrepresented, and there is a lack of consistent cross-platform safety evaluation standards.

## Highlights & Insights
- **Component-Centric Taxonomy**: Anchoring threats to Perception/Brain/Action rather than specific attack names is a future-proof design.
- **Ready-to-use Defense-Threat Matrix**: The explicit mapping between 14 defense types and 16 threat types allows engineers to select combinations based on specific applications (e.g., for web apps: environmental constraints + input validation + cross-verification).
- **Formal Inclusion of "Reasoning Gap Attack"**: Attacks that induce errors through multimodal signal conflict (e.g., pop-up text in a screenshot inconsistent with the HTML code) are unique to CUAs and require multimodal alignment-based defenses.
- **Transparency as a Governance Bottleneck**: The authors point out that vendors like OpenAI have not disclosed safety policies or full evaluation results, calling for the establishment of independent auditing and disclosure frameworks.

## Limitations & Future Work
- **Ours acknowledges**: The survey only covers public English literature up to the submission deadline, potentially missing emerging attacks and internal industrial research. It provides architectural analysis without empirical evaluation of different defenses' relative effectiveness.
- **Additional Limitations**: The labeling of "affected components" in the threat matrix depends on manual judgment, leading to potential definition overlaps for complex attacks like reasoning gaps.
- **Future Directions**: Future work should involve end-to-end empirical testing—running all defense combinations on a unified benchmark to identify the Pareto frontier—and establishing cross-cultural safety evaluation extensions, as current benchmarks are almost entirely based on English GUIs.

## Related Work & Insights
- **vs. LLM Safety Surveys** (Shi 2024, Ma 2025): These focus on model-layer threats (jailbreak, hallucination). Ours expands the perspective to the "Model + Environment + Multimodal" coupling, revealing CUA-unique environmental injection surfaces.
- **vs. OS Agent Surveys** (Hu 2024): These focus on capability (architecture, memory, planning). Ours specializes in safety, acting as a complement.
- **vs. Sager et al. 2025** (Survey on AI Agents for Computer Use): They organize GUI agent capabilities; Ours provides the "security checklist" essential before deployment.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First safety survey specifically for CUAs; the "component-centric" taxonomy and the discussion of reasoning gap attacks are original.
- **Experimental Thoroughness**: ⭐⭐⭐ A review-based work covering 124 papers but lacking original empirical experiments; defense comparisons lack end-to-end evidence.
- **Writing Quality**: ⭐⭐⭐⭐ The three-axis matrix and alignment tables are very engineer-friendly; the JARVIS/Ultron metaphor makes safety research more engaging.
- **Value**: ⭐⭐⭐⭐⭐ A must-read manual for engineering teams deploying CUAs; it clearly identifies mobile/cross-platform evaluation and transparency governance as major research gaps for scholars.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why Agents Compromise Safety Under Pressure](why_agents_compromise_safety_under_pressure.md)
- [\[ACL 2026\] LeakDojo: Decoding the Leakage Threats of RAG Systems](leakdojo_decoding_the_leakage_threats_of_rag_systems.md)
- [\[ACL 2026\] On Safety Risks in Experience-Driven Self-Evolving Agents](on_safety_risks_in_experience-driven_self-evolving_agents.md)
- [\[ACL 2026\] AgentMark: Utility-Preserving Behavioral Watermarking for Agents](agentmark_utility-preserving_behavioral_watermarking_for_agents.md)
- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)

</div>

<!-- RELATED:END -->
