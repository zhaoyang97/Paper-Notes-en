---
title: >-
  [Paper Note] Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents
description: >-
  [ACL 2026][Medical Imaging][multidisciplinary team consultation] This paper proposes Aegle, a graph-structured multi-agent framework that virtualizes multidisciplinary team (MDT) consultation for clinical intake. By intr…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "multidisciplinary team consultation"
  - "multi-agent systems"
  - "clinical intake"
  - "SOAP notes"
  - "dynamic topology"
date: 2026-05-08
content_hash: 021817995443e615
---

# Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents

**Conference**: ACL 2026
**arXiv**: [2604.08927](https://arxiv.org/abs/2604.08927)
**Code**: [GitHub](https://github.com/HovChen/Aegle)
**Area**: Medical AI / Multi-Agent Systems
**Keywords**: multidisciplinary team consultation, multi-agent systems, clinical intake, SOAP notes, dynamic topology

## TL;DR
This paper proposes Aegle, a graph-structured multi-agent framework that virtualizes multidisciplinary team (MDT) consultation for clinical intake. By introducing decoupled parallel reasoning and dynamic topology into the outpatient interview workflow, Aegle surpasses state-of-the-art models across 53 metrics spanning 24 clinical departments.

## Background & Motivation

**Background**: Clinical intake is a critical stage in medical decision-making, requiring physicians to transform unstructured patient narratives into SOAP-formatted Initial Progress Notes (IPN). Current LLM-assisted approaches fall into two categories: document generation (e.g., Med-PaLM 2) and interactive consultation (e.g., AMIE), both of which rely on single-model architectures.

**Limitations of Prior Work**: (1) Individual physicians or models under time pressure are prone to anchoring bias, over-focusing on salient symptoms while overlooking subtle cues; (2) existing interactive systems largely function as passive recipients, lacking the ability to ask proactive exclusionary questions; (3) while MDT consultation mitigates cognitive bias, it is costly and difficult to scale to routine outpatient settings.

**Key Challenge**: The tension between the systematic reasoning depth afforded by MDT-level consultation and the resource constraints of real-time outpatient practice. Additionally, multi-agent systems face the "flawed consensus" problem, wherein agents may mutually reinforce biases and suppress correct minority opinions.

**Goal**: To virtualize the cognitive advantages of MDT at low cost, enabling multi-perspective collaborative reasoning in real-time outpatient settings.

**Key Insight**: A graph-structured multi-agent architecture simulates MDT collaboration — decoupled parallel reasoning preserves hypothesis diversity, dynamic topology activates specialist agents on demand, and SOAP-structured state ensures traceable reasoning.

**Core Idea**: A three-tier architecture comprising an Orchestrator that dynamically activates specialist agents, specialist agents that perform decoupled parallel reasoning, and an Aggregator that integrates outputs and updates the structured clinical state — collectively virtualizing the MDT consultation process.

## Method

### Overall Architecture
Aegle is built on DeepSeek-V3.2 and employs a two-stage finite state machine to conduct clinical interviews. Stage I performs iterative history-taking (evidence collection), while Stage II handles diagnostic synthesis (generating diagnoses after the evidence set is frozen). Throughout the process, an incrementally updated structured clinical state $\mathcal{S}_t = [\mathcal{F}_t, \mathcal{P}_t]$ is maintained, where $\mathcal{F}$ corresponds to the S+O components of SOAP (factual evidence) and $\mathcal{P}$ corresponds to the A+P components (diagnosis and plan).

### Key Designs

1. **Structured Clinical State**:

    - **Function**: Serves as a shared "blackboard" for all agents, separating evidence collection from diagnostic reasoning.
    - **Mechanism**: The SOAP schema is formalized as $\mathcal{S}_t = [\mathcal{F}_t, \mathcal{P}_t]$, where $\mathcal{F}$ (Case Features) accumulates verifiable facts including demographic information, history of present illness, past medical history, and physical examination findings; $\mathcal{P}$ (Diagnosis & Plan) is generated only after $\mathcal{F}$ has stabilized. A strict unidirectional dependency is enforced: $\mathcal{F} \to \mathcal{P}$.
    - **Design Motivation**: To prevent premature commitment to immature diagnostic hypotheses and ensure that clinical conclusions remain traceable to specific evidence.

2. **Dynamic Multi-Agent Graph Topology**:

    - **Function**: Activates specialist agents on demand, avoiding unnecessary expert involvement.
    - **Mechanism**: Three node types collaborate — the Orchestrator applies a routing policy $\pi_{orch}$ to select an active specialist subset $A_{sub}$ based on dialogue history and current evidence; Specialist Agents independently and in parallel analyze the case (decoupled reasoning); the Aggregator follows a "write-before-speak" protocol to integrate specialist recommendations, update the state $\mathcal{S}_{t+1}$, and then generate patient-facing dialogue.
    - **Design Motivation**: Decoupled parallel reasoning preserves hypothesis diversity (avoiding groupthink), while dynamic activation mirrors the on-demand expert recruitment characteristic of real MDT settings.

3. **Sequential Clinical Execution**:

    - **Function**: Strictly separates evidence acquisition from diagnostic reasoning, serving as an explicit bias control mechanism.
    - **Mechanism**: In Stage I, the Orchestrator iteratively activates specialist agents to propose follow-up questions; the Aggregator integrates these suggestions to generate the next round of inquiry. Once sufficient evidence is gathered, the process transitions to Stage II, where $\mathcal{F}$ is frozen and $\mathcal{P}$ (diagnosis and treatment plan) is generated from the complete evidence set.
    - **Design Motivation**: To prevent premature commitment to a diagnostic direction under incomplete evidence, emulating the real MDT practice of thoroughly discussing clinical findings before reaching consensus.

### Loss & Training
Aegle is an inference-time framework rather than a trained model. It leverages the zero-shot capabilities of DeepSeek-V3.2 through structured prompting and role assignment. No additional training is required.

## Key Experimental Results

### Main Results

| Dataset | Metric | Aegle | DeepSeek-V3.2 | GPT-4o | Gain |
|---------|--------|-------|---------------|--------|------|
| ClinicalBench | IDEA | 63.80 | 50.51 | 41.05 | +13.3 |
| ClinicalBench | SOAP | 53.42 | 38.64 | 29.38 | +14.8 |
| ClinicalBench | READ | 76.20 | 71.73 | 67.66 | +4.5 |
| RAPID-IPN | IDEA | 67.31 | 54.35 | 44.70 | +13.0 |
| RAPID-IPN | SOAP | 60.09 | 47.39 | 34.79 | +12.7 |
| RAPID-IPN | READ | 80.18 | 72.14 | 69.89 | +8.0 |

Evaluation covers 24 clinical departments and 53 fine-grained metrics.

### Ablation Study

| Configuration | IDEA | SOAP | Notes |
|---------------|------|------|-------|
| Aegle (full) | 63.80 | 53.42 | Complete framework |
| Single agent (DeepSeek-V3.2) | 50.51 | 38.64 | No MDT collaboration |
| MiniMax-M2 | 57.78 | 46.18 | Strongest single-model baseline |

### Key Findings
- Aegle consistently outperforms all baselines across all 53 metrics, including closed-source models such as GPT-4o and Gemini 2.5.
- Even when sharing the same backbone (DeepSeek-V3.2), the multi-agent framework yields a +13.3-point gain on IDEA, demonstrating the intrinsic value of the collaborative architecture.
- Gains are more pronounced on the real-world clinical dataset RAPID-IPN, indicating strong generalization of the framework to realistic settings.

## Highlights & Insights
- **Decoupled Parallel Reasoning**: Independent analysis by each specialist agent avoids the "flawed consensus" problem, offering a safer and more controllable alternative to debate-based multi-agent systems. This design is transferable to other domains requiring multi-perspective analysis, such as legal review or financial risk assessment.
- **SOAP-Structured State as Shared Blackboard**: The framework elevates a clinical documentation standard into a reasoning control mechanism — functioning not merely as a recording format but as a bias control instrument. This "structure as constraint" paradigm offers broad methodological inspiration.
- **Write-Before-Speak Protocol**: The Aggregator updates internal state before generating patient-facing dialogue, ensuring a clean separation between technical precision and patient communication, which is critical for the deployability of medical AI systems.

## Limitations & Future Work
- The framework relies entirely on the zero-shot capabilities of DeepSeek-V3.2; fine-tuning for clinical scenarios remains unexplored.
- Multi-agent invocation substantially increases inference cost (multiplicative API call overhead), requiring careful consideration of latency and cost in real deployments.
- Evaluation is primarily based on Chinese clinical data; cross-lingual and cross-cultural generalizability remains to be validated.
- Integration of multimodal information — including imaging and laboratory test results — is not addressed.

## Related Work & Insights
- **vs. AMIE**: AMIE is a single-model interactive consultation system susceptible to anchoring bias; Aegle expands the hypothesis space through multi-agent parallel reasoning.
- **vs. MDAgents**: MDAgents adapts topology based on task complexity, but agent interactions remain opaque; Aegle explicitly constrains the reasoning chain via SOAP-structured state.
- **vs. MedAgents**: MedAgents employs debate-style collaboration, which risks producing flawed consensus; Aegle's decoupled parallel reasoning with independent analysis prevents inter-agent interference.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The framework design for virtualizing MDT is novel; the formalization of SOAP-structured state as a reasoning mechanism is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluation spans 24 departments, 53 metrics, two datasets (ClinicalBench and a real-world dataset), and multiple SOTA baselines.
- **Writing Quality**: ⭐⭐⭐⭐ The framework is clearly described, though the heavy use of formal notation in places could be further simplified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](../../ICLR2026/medical_imaging/care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)
- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[ACL 2026\] From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](from_answers_to_arguments_toward_trustworthy_clinical_diagnostic_reasoning_with_.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)
- [\[AAAI 2026\] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules](../../AAAI2026/medical_imaging/lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)

</div>

<!-- RELATED:END -->
