---
title: >-
  [Paper Note] Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents
description: >-
  [ACL 2026][Medical NLP][Multi-Disciplinary Team (MDT)] This paper proposes the Aegle framework, which virtualizes Multi-Disciplinary Teams (MDT) through a graph-structured multi-agent architecture. By introducing decoupl…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Multi-Disciplinary Team (MDT)"
  - "Multi-Agent"
  - "Clinical Intake"
  - "SOAP Notes"
  - "Dynamic Topology"
date: 2026-05-08
content_hash: 2b04ab8570fe874f
---

# Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents

**Conference**: ACL 2026  
**arXiv**: [2604.08927](https://arxiv.org/abs/2604.08927)  
**Code**: [GitHub](https://github.com/HovChen/Aegle)  
**Area**: Medical NLP  
**Keywords**: Multi-Disciplinary Team (MDT), Multi-Agent, Clinical Intake, SOAP Notes, Dynamic Topology

## TL;DR
This paper proposes the Aegle framework, which virtualizes Multi-Disciplinary Teams (MDT) through a graph-structured multi-agent architecture. By introducing decoupled parallel reasoning and dynamic topology into the outpatient intake process, the framework outperforms SOTA models across 53 metrics in 24 clinical departments.

## Background & Motivation

**Background**: The initial intake is a critical stage for clinical decision-making, where physicians must transform unstructured patient narratives into Initial Progress Notes (IPN) in the SOAP format. Current LLM-assisted intake systems mainly fall into two categories: document generation (e.g., Med-PaLM 2) and interactive consultation (e.g., AMIE), both of which rely on single-model architectures.

**Limitations of Prior Work**: (1) Single physicians or models are prone to anchoring bias under time pressure, focusing excessively on prominent symptoms while ignoring subtle cues; (2) Existing interactive systems act mostly as "passive receivers," lacking proactive differential questioning; (3) While Multi-Disciplinary Teams (MDT) can mitigate cognitive biases, they are costly and difficult to scale to daily outpatient settings.

**Key Challenge**: The contradiction between the depth of systematic reasoning required at the MDT level and the resource constraints of real-time outpatient scenarios. Additionally, multi-agent systems face the "flawed consensus" problem—where agents may reinforce each other's biases and suppress correct minority opinions.

**Goal**: To virtualize the cognitive advantages of MDT, enabling high-quality multi-perspective collaborative reasoning in real-time outpatient clinics at a low cost.

**Key Insight**: Simulate MDT collaboration using a graph-structured multi-agent architecture—decoupled parallel reasoning maintains hypothesis diversity, dynamic topology activates specialized agents on demand, and structured SOAP states ensure reasoning traceability.

**Core Idea**: A three-layer architecture where an Orchestrator dynamically activates specialized agents, each agent performs decoupled parallel reasoning, and an Aggregator integrates outputs to update structured clinical states, thereby virtualizing the MDT consultation process.

## Method

### Overall Architecture
Aegle is built upon DeepSeek-V3.2 and utilizes a two-stage finite state machine for intake: Stage I involves iterative history taking (evidence collection), and Stage II involves diagnostic synthesis (generating the diagnosis after freezing the evidence set). Throughout the process, an incrementally updated structured clinical state $\mathcal{S}_t = [\mathcal{F}_t, \mathcal{P}_t]$ is maintained, where $\mathcal{F}$ corresponds to the S+O (factual evidence) of SOAP, and $\mathcal{P}$ corresponds to the A+P (assessment and plan).

### Key Designs

1.  **Structured Clinical State**:
    - **Function**: Serves as a "blackboard" shared by all agents, separating evidence collection from diagnostic reasoning.
    - **Core Idea**: Formalizes the SOAP schema as $\mathcal{S}_t = [\mathcal{F}_t, \mathcal{P}_t]$, where $\mathcal{F}$ (Case Features) accumulates verifiable facts such as basic information, history of present illness, past medical history, and physical examination; $\mathcal{P}$ (Diagnosis & Plan) is generated only after $\mathcal{F}$ is stabilized. A mandatory unidirectional dependency is enforced: $\mathcal{F} \to \mathcal{P}$.
    - **Design Motivation**: To prevent premature commitment to immature diagnostic hypotheses and ensure clinical conclusions are traceable to specific evidence.

2.  **Dynamic Multi-Agent Graph Topology**:
    - **Function**: Activates specialized agents on demand to avoid unnecessary expert intervention.
    - **Core Idea**: Three types of nodes collaborate—the Orchestrator acts as a routing policy $\pi_{orch}$ to select the activated specialist subset $A_{sub}$ based on dialogue history and current evidence; Specialist Agents independently and in parallel analyze the case (decoupled reasoning); the Aggregator follows a "write-before-talk" protocol to integrate specialist suggestions and update the state $\mathcal{S}_{t+1}$ before generating patient-facing dialogue.
    - **Design Motivation**: Decoupled parallel reasoning preserves hypothesis diversity (avoiding groupthink), and dynamic activation mimics the on-demand assembly of experts in real MDT.

3.  **Sequential Clinical Execution**:
    - **Function**: Strictly separates evidence acquisition from diagnostic reasoning as an explicit bias control mechanism.
    - **Core Idea**: During Stage I, the Orchestrator repeatedly activates specialist agents to suggest follow-up questions, and the Aggregator integrates these to generate the next round of inquiry. Once evidence is sufficient, the system enters Stage II, freezes $\mathcal{F}$, and generates $\mathcal{P}$ (diagnosis + treatment plan) based on the complete evidence set.
    - **Design Motivation**: To avoid locking into a diagnostic direction prematurely when evidence is incomplete, simulating the real MDT process of thorough discussion before forming a consensus.

### Loss & Training
Aegle is a reasoning framework (not a training method). It leverages the zero-shot capabilities of DeepSeek-V3.2 through structured prompts and role assignments to achieve collaboration. No additional training is required.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (Aegle) | DeepSeek-V3.2 | GPT-4o | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ClinicalBench | IDEA | 63.80 | 50.51 | 41.05 | +13.3 |
| ClinicalBench | SOAP | 53.42 | 38.64 | 29.38 | +14.8 |
| ClinicalBench | READ | 76.20 | 71.73 | 67.66 | +4.5 |
| RAPID-IPN | IDEA | 67.31 | 54.35 | 44.70 | +13.0 |
| RAPID-IPN | SOAP | 60.09 | 47.39 | 34.79 | +12.7 |
| RAPID-IPN | READ | 80.18 | 72.14 | 69.89 | +8.0 |

The evaluation covers 24 clinical departments and 53 fine-grained metrics.

### Ablation Study

| Configuration | IDEA | SOAP | Description |
| :--- | :--- | :--- | :--- |
| Aegle (Full) | 63.80 | 53.42 | Complete framework |
| Single Agent (DeepSeek-V3.2) | 50.51 | 38.64 | No MDT collaboration |
| MiniMax-M2 | 57.78 | 46.18 | Strongest single-model baseline |

### Key Findings
- Aegle consistently outperforms all baselines across all 53 metrics, including closed-source models like GPT-4o and Gemini 2.5.
- Even with the same backbone model (DeepSeek-V3.2), the multi-agent framework yields a +13.3 IDEA score improvement, proving the intrinsic value of the collaborative architecture.
- Improvements were more significant on the real-world clinical dataset RAPID-IPN, indicating the framework generalizes well to real-world scenarios.

## Highlights & Insights
- **Decoupled Parallel Reasoning**: Independent analysis by specialized agents avoids the "flawed consensus" problem, making it safer and more controllable than debate-based multi-agent systems. This is transferable to other scenarios requiring multi-perspective analysis (e.g., law, financial risk assessment).
- **SOAP Structured State as a Shared Blackboard**: Elevating clinical documentation standards to a reasoning control mechanism—not just a recording format, but a tool for bias control. This "structure as constraint" approach is highly insightful.
- **Write-before-talk Protocol**: Ensuring the aggregator updates internal states before generating dialogue guarantees the separation of technical precision from patient communication, which is crucial for the deployability of medical AI.

## Limitations & Future Work
- The framework relies entirely on the zero-shot capabilities of DeepSeek-V3.2 and does not explore fine-tuning specifically for clinical scenarios.
- Multi-agent calls increase reasoning costs (multiplying API calls); actual deployment must consider latency and cost.
- Evaluation is primarily based on Chinese clinical data; cross-language and cross-cultural generalization remains to be verified.
- Integration of multimodal information such as imaging and laboratory tests is not yet addressed.

## Related Work & Insights
- **vs AMIE**: AMIE is a single-model interactive system prone to anchoring bias; Aegle expands the hypothesis space through multi-agent parallel reasoning.
- **vs MDAgents**: MDAgents adjusts topology based on task complexity, but interactions remain a black box; Aegle explicitly constrains the reasoning chain through structured SOAP states.
- **vs MedAgents**: MedAgents uses debate-based collaboration, which may lead to flawed consensus; Aegle’s decoupled parallel approach avoids mutual interference between agents.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The design of virtualizing MDT is novel, and the formalization of SOAP structured states is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 24 departments, 53 metrics, ClinicalBench + real datasets, and multiple SOTA baselines.
- **Writing Quality**: ⭐⭐⭐⭐ The framework description is clear, though some mathematical notations are dense and could be further simplified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)
- [\[ACL 2026\] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)
- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)
- [\[ACL 2026\] SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning](sema-rag_a_self-evolving_multi-agent_retrieval-augmented_generation_framework_fo.md)
- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)

</div>

<!-- RELATED:END -->
