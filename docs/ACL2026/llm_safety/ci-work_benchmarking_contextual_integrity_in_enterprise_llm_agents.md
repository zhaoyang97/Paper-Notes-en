---
title: >-
  [Paper Note] CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents
description: >-
  [ACL 2026][LLM Safety][Enterprise Privacy] Constructs CI-Work, an enterprise-scenario benchmark based on the Contextual Integrity theory…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Enterprise Privacy"
  - "Contextual Integrity"
  - "LLM Agents"
  - "Information Leakage"
  - "Privacy-Utility Trade-off"
date: 2026-05-08
content_hash: 43eb2bcd0f6b2bc4
---

# CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents

**Conference**: ACL 2026  
**arXiv**: [2604.21308](https://arxiv.org/abs/2604.21308)  
**Code**: [https://aka.ms/ci-work](https://aka.ms/ci-work)  
**Area**: LLM Agent  
**Keywords**: Enterprise Privacy, Contextual Integrity, LLM Agents, Information Leakage, Privacy-Utility Trade-off

## TL;DR

Constructs CI-Work, an enterprise-scenario benchmark based on the Contextual Integrity theory, revealing that frontier LLM agents generally suffer from privacy leakage in enterprise workflows, and that increasing model scale exacerbates this leakage.

## Background & Motivation

**Background**: LLM agents are being integrated into enterprise workflows, gaining access to internal data such as emails and meeting notes to perform complex tasks, significantly enhancing productivity.

**Limitations of Prior Work**: Existing privacy benchmarks (ConfAide, PrivacyLens, CIMemories, etc.) primarily focus on daily assistant scenarios and fail to capture complexities in enterprise environments: (1) they evaluate only single information flows, ignoring the interwoven nature of multiple parallel information flows in enterprises; (2) evaluation contexts are simple and isolated, failing to measure the ability to distinguish between "essential information" and "sensitive information" in dense retrieval scenarios; (3) reliance on simplified contexts or short attributes fails to replicate the scale and density of enterprise data.

**Key Challenge**: The core capability of enterprise LLM agents (retrieving and utilizing internal data) specifically makes them potential carriers of sensitive information leakage—higher task utility is often accompanied by more privacy violations.

**Goal**: To build an enterprise-level benchmark based on Contextual Integrity theory to systematically evaluate the privacy-utility trade-offs of LLM agents in high-fidelity enterprise workflows.

**Key Insight**: Categorizing enterprise information flows according to organizational communications (downward, upward, horizontal, diagonal, and external), where each instance contains dense retrieval contexts for "essential sets" and "sensitive sets."

**Core Idea**: Enterprise privacy is not simple information masking but requires precise differentiation between essential and sensitive information in dense retrieval scenarios, which remains unresolved in current models; furthermore, increasing model scale exacerbates the problem.

## Method

### Overall Architecture

CI-Work adopts a four-stage construction process: (1) task-oriented seed generation → (2) context entry generation → (3) case episode generation → (4) trajectory simulation and evaluation. A tool-centered simulation environment is built based on ToolEmu and PrivacyLens, using LLMs to simulate enterprise tools (Email, Chat, Calendar, Meeting, etc.).

### Key Designs

1. **Five-direction Information Flow Classification**:

    - **Function**: Covers all typical information flow directions within an enterprise organization.
    - **Mechanism**: Based on standard organizational communication taxonomy, information flows are divided into downward (management → employee), upward (reporting to superiors), horizontal (peer collaboration), diagonal (cross-departmental), and external (stakeholders).
    - **Design Motivation**: To capture differences in privacy norms between various roles in real enterprise settings.

2. **Self-Iterative Refinement**:

    - **Function**: Automatically corrects generated context entries to ensure consistency of essential/sensitive labels.
    - **Mechanism**: After generating entries, an LLM blindly classifies them as Essential/Sensitive/Ambiguous; if the classification is inconsistent with the expectation, an automatic revision loop is triggered.
    - **Design Motivation**: To bridge the quality gap between LLM generation and evaluation, achieving an 82.5%–95.0% consistency rate via manual verification.

3. **Dual-Set Evaluation Metric System**:

    - **Function**: Simultaneously quantifies privacy protection and task utility.
    - **Mechanism**: Retrieval entries are partitioned into an essential set $\mathcal{E}_{\text{ess}}$ and a sensitive set $\mathcal{E}_{\text{sens}}$, with three metrics designed: Leakage Rate (LR), Violation Rate (VR), and Conveyance Rate (CR).
    - **Design Motivation**: To explicitly quantify the privacy-utility trade-off beyond a single success rate evaluation.

### Loss & Training

No model training is involved. LLM agents are deployed using the ReAct framework. GPT-5.2 is used for benchmark generation and evaluation, and LLM-as-a-Judge is implemented for automatic evaluation (achieving 83.0%–91.0% consistency with human annotation).

## Key Experimental Results

### Main Results

| Model | Leakage Rate (LR)↓ | Violation Rate (VR)↓ | Conveyance Rate (CR)↑ |
|------|------------|------------|------------|
| DeepSeek-R1 | 6.08% | 15.80% | 53.08% |
| DeepSeek-V3 | 8.37% | 21.33% | 76.92% |
| GPT-4o | 8.79% | 21.33% | 87.35% |
| GPT-5 | 11.21% | 27.83% | 93.04% |
| Grok-3 | 26.66% | 50.87% | 94.97% |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Entry Count 1→12 | VR increases monotonically, CR decreases significantly | Multi-entity contexts introduce interference |
| Entry Length Increase | Both LR/VR increase, CR also increases | Richer details expand the leakage surface |
| Implicit User Pressure | VR increases significantly | Instructions emphasizing task completion exacerbate leakage |
| Explicit User Pressure | VR nearly doubles | Active provision of data sources makes leakage more severe |

### Key Findings
- **Privacy-Utility Trade-off**: Higher conveyance rates are positively correlated with higher leakage/violation rates (Pearson $r=0.40$, $p<0.05$).
- **Inverse Scaling Phenomenon**: Larger models (like the GPT-4.1 series) actually exacerbate privacy leakage because stronger instruction-following capabilities cause them to prioritize responding to user needs over adhering to implicit privacy norms.
- **Higher Risk in Upward Interaction**: The leakage and violation rates for upward reporting are significantly higher than for downward communication (VR: $p=0.006$).
- **CI-CoT Defense is Effective but Insufficient**: It significantly reduces leakage rates but still maintains a >20% violation rate.

## Highlights & Insights
- Conducts the first systematic evaluation of Contextual Integrity for LLM agents in enterprise scenarios, finding privacy violation rates as high as 15.8%–50.9%.
- Reveals the counter-intuitive "inverse scaling": larger models exhibit worse privacy because they better understand and execute user intent while neglecting implicit social constraints.
- Identifies that user behavior (even implicit pressure without malicious intent) can lead to a dual collapse of both privacy and utility.
- Explicitly points out that model scale and reasoning depth alone cannot resolve enterprise privacy issues, necessitating a paradigm shift.

## Limitations & Future Work
- The benchmark is based on synthetic data and simulated environments, which may differ from deployments in real enterprises.
- Privacy norms vary by organization, jurisdiction, and culture, and the benchmark cannot cover every scenario.
- Future work: Role-conditioned filtering, context-aware alignment during training, and an architectural shift from model-centric to context-centric designs.

## Related Work & Insights
- **vs ConfAide/PrivacyLens**: These benchmarks focus on daily assistant scenarios and single information flows, whereas CI-Work focuses on multi-entity dense retrieval scenarios in enterprises.
- **vs CIMemories**: Focuses on privacy accumulation in memory, whereas CI-Work focuses on information leakage within a single enterprise task.
- **vs WorkArena/TheAgentCompany**: These benchmarks focus on task execution capabilities, whereas CI-Work is the first to simultaneously evaluate both utility and privacy.

## Rating
- Novelty: ⭐⭐⭐⭐ First enterprise-level CI benchmark, revealing counter-intuitive phenomena like inverse scaling.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 9 frontier models, 5 info-flow directions, and multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated, and information-rich visuals.
- Value: ⭐⭐⭐⭐ Provides significant warning and guidance for security in enterprise AI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Do Vision-Language Models Respect Contextual Integrity in Location Disclosure?](../../ICLR2026/llm_safety/do_vision-language_models_respect_contextual_integrity_in_location_disclosure.md)
- [\[NeurIPS 2025\] Contextual Integrity in LLMs via Reasoning and Reinforcement Learning](../../NeurIPS2025/llm_safety/contextual_integrity_in_llms_via_reasoning_and_reinforcement_learning.md)
- [\[ACL 2026\] Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game](detecting_rag_extraction_attack_via_dual-path_runtime_integrity_game.md)
- [\[ACL 2026\] Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models](privacy_collapse_benign_fine-tuning_can_break_contextual_privacy_in_language_mod.md)
- [\[ACL 2026\] AgentMark: Utility-Preserving Behavioral Watermarking for Agents](agentmark_utility-preserving_behavioral_watermarking_for_agents.md)

</div>

<!-- RELATED:END -->
