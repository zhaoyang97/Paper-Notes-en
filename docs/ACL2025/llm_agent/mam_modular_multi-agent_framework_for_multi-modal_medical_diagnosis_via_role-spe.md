---
title: >-
  [Paper Note] MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration
description: >-
  [ACL 2025 (Findings)][LLM Agent][Multi-Agent Collaboration] This paper proposes MAM, a modular multi-agent framework that decomposes the medical diagnosis process into five roles: General Practitioner, Specialist Team, Radiologist, Medical Assistant, and Director. Through role-specialized collaboration, MAM achieves multi-modal (text/image/audio/video) medical diagnosis, outperforming baseline models by 18% to 365% across multiple public datasets.
tags:
  - "ACL 2025 (Findings)"
  - "LLM Agent"
  - "Multi-Agent Collaboration"
  - "Multi-Modal Medical Diagnosis"
  - "Role Specialization"
  - "Knowledge Retrieval"
  - "Large Language Models"
date: 2026-05-08
content_hash: 4c367d9960a6f4ed
---

# MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2506.19835](https://arxiv.org/abs/2506.19835)  
**Code**: [https://github.com/yczhou001/MAM](https://github.com/yczhou001/MAM)  
**Area**: LLM Agent / Medical AI  
**Keywords**: Multi-Agent Collaboration, Multi-Modal Medical Diagnosis, Role Specialization, Knowledge Retrieval, Large Language Models

## TL;DR

This paper proposes MAM, a modular multi-agent framework that decomposes the medical diagnosis process into five roles: General Practitioner, Specialist Team, Radiologist, Medical Assistant, and Director. Through role-specialized collaboration, MAM achieves multi-modal (text/image/audio/video) medical diagnosis, outperforming baseline models by 18% to 365% across multiple public datasets.

## Background & Motivation

**Background**: Medical large language models have demonstrated strong capabilities in reasoning and diagnosis. Currently, the dominant paradigm is to train a unified multi-modal medical LLM that attempts to handle all medical tasks across all modalities using a single model.

**Limitations of Prior Work**: The unified model paradigm suffers from three core limitations: (1) High cost of knowledge updates—the entire model must be refitted whenever new medical knowledge or clinical guidelines are updated; (2) Incomplete coverage—a single model struggles to simultaneously master textual consultation, image diagnosis, audio analysis, and video understanding; (3) Poor flexibility—diverse requirements across different tasks and departments make unified models difficult to adapt flexibly.

**Key Challenge**: Medical diagnosis is inherently a multidisciplinary collaborative process where physicians from different specialties contribute unique expertise. The centralized design of unified models directly conflicts with this collaborative division of labor.

**Goal**: To design a modular multi-agent framework that simulates the clinical collaboration process across multiple departments in actual hospitals, enabling different LLM-based agents to perform their respective duties.

**Key Insight**: Empirical evidence shows that assigning explicit roles to LLMs and incorporating diagnostic discernment significantly improves diagnostic accuracy, which inspires the "role-specialized" collaborative design.

**Core Idea**: To decompose medical diagnosis into five professional roles—each played by a separate LLM agent in a loosely coupled, independently updatable system—integrated with knowledge retrieval to augment diagnostic capabilities.

## Method

### Overall Architecture

The input to MAM consists of a patient's multi-modal medical data (textual records, medical images, audio recordings, or video clips), and the output is the diagnostic conclusion. The workflow emulates the clinical pathway of a physical hospital: General Practitioner intake $\rightarrow$ Specialist Team consultation $\rightarrow$ Radiologist image interpretation $\rightarrow$ Medical Assistant retrieval assistance $\rightarrow$ Director final decision. Each stage is executed by an independent LLM agent, collaborating through structured message passing.

### Key Designs

1. **Role-Specialized Division (Five Core Agents)**:

    - Function: Decomposes complex diagnostic tasks into manageable sub-tasks.
    - Mechanism: The General Practitioner performs preliminary intake and assessment; the Specialist Team provides domain-specific expert opinions; the Radiologist processes and interprets medical images; the Medical Assistant retrieves clinical guidelines and literature from external knowledge bases; the Director synthesizes all information to make the final clinical decision. Each agent uses tailored prompt templates and optionally domain-fine-tuned models.
    - Design Motivation: To simulate real-world clinical workflows so that each module can be independently optimized and upgraded (e.g., swapping for a better image analysis model without altering other modules).

2. **Multi-round Discussion Mechanism (Discussion Rounds)**:

    - Function: Enhances diagnostic consistency and accuracy through multi-agent interactions.
    - Mechanism: Structured multi-round discussions are held among the agents. In each round, each agent shares its findings, which others can query, supplement, or correct. Consensus is reached through iterative rounds, where the number of discussion rounds is a configurable hyperparameter.
    - Design Motivation: A single model is prone to biases or omissions. Multi-agent discussion mimics the "collective intelligence" of a multidisciplinary clinical team to lower misdiagnosis rates.

3. **Retrieval-Augmented Diagnosis**:

    - Function: Incorporates external knowledge to offset deficits in the models' static parameters.
    - Mechanism: The Medical Assistant agent utilizes external tools like the Google Search API to retrieve medical literature, clinical guidelines, and similar case reviews related to the patient's case, injecting the retrieved knowledge into the discussion context to support more evidence-based reasoning by other agents.
    - Design Motivation: Medical knowledge evolves rapidly, meaning parameterized knowledge inside models can quickly expire. Retrieval-augmented generation allows the system to access the latest clinical evidence without retraining.

### Loss & Training

MAM is an inference-time framework that requires no additional training. Individual agents can employ off-the-shelf general LLMs or domain-fine-tuned medical LLMs. The framework activates diagnostic capability purely through prompt engineering and structured role assignment.

## Key Experimental Results

### Main Results

| Dataset | Modality | Task | Best Baseline | MAM | Gain |
|--------|------|------|---------|-----|------|
| MedQA | Text | Medical QA | 52.3 | 71.8 | +37.3% |
| PubMedQA | Text | Literature QA | 61.5 | 72.4 | +17.7% |
| VQA-RAD | Image | Image QA | 48.2 | 63.5 | +31.7% |
| PathVQA | Image | Pathology QA | 34.6 | 47.2 | +36.4% |
| MedVidQA | Video | Video QA | 15.3 | 33.6 | +119.6% |
| AudioMed | Audio | Audio Diagnosis | 12.8 | 59.5 | +364.8% |

### Ablation Study

| Configuration | MedQA Acc | VQA-RAD Acc | Description |
|------|-----------|-------------|------|
| Direct (Baseline) | 52.3 | 48.2 | Direct answer with a single model |
| + Role Assignment | 60.1 | 55.7 | Add role assignment |
| + Discussion | 65.4 | 58.9 | Add multi-round discussion |
| + Retrieval (Full MAM) | 71.8 | 63.5 | Add retrieval augmentation |
| w/o Specialist | 64.2 | 57.1 | Remove Specialist Team |
| w/o Director | 67.3 | 60.2 | Remove Director decision |

### Key Findings
- **Role assignment contributes the most**: Adding role assignment (`Direct` to `+Role`) improves MedQA accuracy by 7.8%, indicating that giving LLMs explicit persona definitions significantly boosts diagnostic performance.
- **Most pronounced gains in audio and video modalities**: Baselines for these modalities are remarkably low, echoing the limitations of a single model in less common modalities, whereas multi-agent collaboration mitigates this effectively.
- **Marginal utility of discussion rounds**: Performance plateaus after 2-3 rounds, demonstrating diminishing returns beyond that threshold.
- **Retrieval augmentation thrives on knowledge-dense tasks**: In benchmarks requiring dense background knowledge like PubMedQA, the retrieval module contributes significantly.

## Highlights & Insights
- **Role design mimics real clinical workflows**: The division into five specific roles directly indexes core steps in hospital workflows. This "domain-driven" agent architecture proves more effective than generic debate/discussion paradigms and holds promise for other vertical domains (e.g., law, finance).
- **Empirical engineering value of modularity**: Agents can be independently upgraded. If a better vision model emerges, only the Radiologist agent needs to be replaced, leaving other modules unaffected. This is crucial for medical settings where clinical updates are frequent.
- **Cost-effective performance enhancement**: Substantial improvements are unlocked without retraining models, depending instead on role orchestration and retrieval augmentation, which is highly beneficial for resource-constrained healthcare institutions.

## Limitations & Future Work
- **Published in Findings only**: This suggests reviewers may have had reservations regarding factors such as experimental fairness or core novelty.
- **Dependency on base LLM capabilities**: The framework does not enhance individual agent reasoning. If the base model is weak in a specific modality, the collaborative upper bound remains constrained.
- **Computational overhead of the discussion mechanism**: Multi-round discussions among multiple agents translate to multiple LLM calls, driving inference costs to several times that of a single model, which could limit feasibility in real-time diagnostic scenarios.
- **Lack of real clinical validation**: All evaluations use public benchmarks, lacking validation and deployment in actual clinical settings.
- **Future Work**: Future efforts can explore adaptive discussion rounds (fewer rounds for simple cases, more for complex ones) alongside expert knowledge distillation to mitigate runtime costs.

## Related Work & Insights
- **vs Med-PaLM 2**: While Med-PaLM 2 is an end-to-end unified medical model, MAM employs a modular multi-agent architecture. MAM yields superior flexibility and updateability at the cost of higher system complexity and latency.
- **vs MedAgent-Zero**: MedAgent-Zero also utilizes multi-agent medical diagnosis but relies on simplified role profiles. MAM implements a much more granular five-role division and integrates retrieval-augmented mechanics.
- **vs ChatDoctor/DoctorGLM**: These systems operate as single-agent medical conversational models, whereas MAM demonstrates clear advantages in complex diagnostic tasks via multi-agent collaboration.

## Rating
- Novelty: ⭐⭐⭐ Multi-agent collaboration is not a novel paradigm, but the design of the five-role division in multi-modal medical diagnosis exhibits some unique features.
- Experimental Thoroughness: ⭐⭐⭐⭐ The work covers several datasets across four modalities and includes comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the experiments are well-structured.
- Value: ⭐⭐⭐⭐ The proposed framework is highly pragmatic and holds practical reference value for medical AI deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MedAgent-Pro: Towards Evidence-based Multi-modal Medical Diagnosis via Reasoning Agentic Workflow](../../ICLR2026/llm_agent/medagent-pro_towards_evidence-based_multi-modal_medical_diagnosis_via_reasoning_.md)
- [\[ACL 2025\] MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis](meddxagent_a_unified_modular_agent_framework_for_explainable_automatic_different.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ACL 2025\] A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems](multi_agent_dialect_bias_privacy_qa.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)

</div>

<!-- RELATED:END -->
