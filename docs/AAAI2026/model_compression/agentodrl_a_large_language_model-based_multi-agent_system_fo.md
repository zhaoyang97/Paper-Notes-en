---
title: >-
  [Paper Note] AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation
description: >-
  [AAAI 2026][Model Compression][ODRL Generation] This paper proposes AgentODRL, an LLM-based multi-agent system built on an Orchestrator-Workers architecture that converts natural language data usage rules into high-quali…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "ODRL Generation"
  - "Multi-agent System"
  - "LLM"
  - "Data Usage Policy"
  - "Orchestrator-Workers"
date: 2026-05-08
content_hash: 2ae38303cec3c2ed
---

# AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation

**Conference**: AAAI 2026
**arXiv**: [2512.00602v1](https://arxiv.org/abs/2512.00602v1)
**Code**: [https://github.com/RUC-MAS/AgentODRL](https://github.com/RUC-MAS/AgentODRL)
**Area**: Model Compression
**Keywords**: ODRL Generation, Multi-agent System, LLM, Data Usage Policy, Orchestrator-Workers

## TL;DR
This paper proposes AgentODRL, an LLM-based multi-agent system built on an Orchestrator-Workers architecture that converts natural language data usage rules into high-quality ODRL policies through task decomposition, a syntax validation loop, and a LoRA-driven semantic reflection mechanism.

## Background & Motivation
ODRL (Open Digital Rights Language) is a W3C standard for describing data asset usage policies in data spaces. However, authoring ODRL policies requires familiarity with RDF graph data models, serialization formats, and the complex conceptual framework of ODRL itself, posing a high barrier for domain experts without technical backgrounds. Existing approaches (e.g., ontology-guided strategies, SCR self-correcting rules) rely on single LLMs for end-to-end generation. When confronted with complex rules containing parallel structures (multiple independent policies) or recursive structures (cross-clause reference dependencies), a single model cannot simultaneously handle legal text parsing, semantic segmentation, and strict syntax generation, leading to significant performance degradation. Furthermore, the extreme scarcity of high-quality NL-to-ODRL parallel corpora further constrains model capability.

## Core Problem
How can complex natural language data usage rules—containing parallel and recursive logical structures—be accurately and automatically converted into structurally rigorous ODRL policies? The core challenges are: (1) single-model architectures cannot efficiently handle multiple cognitive sub-tasks simultaneously; (2) high-quality training data is lacking; (3) generated outputs must satisfy both syntactic correctness and semantic faithfulness.

## Method

### Overall Architecture
AgentODRL adopts an Orchestrator-Workers pattern. A central Orchestrator Agent receives natural language use cases, analyzes their complexity category (simple / parallel / recursive), and dynamically schedules specialized Worker Agents to assemble the optimal processing pipeline. Post-processing strategies along syntactic and semantic dimensions ensure output quality.

### Key Designs
1. **Use Case Complexity Classification**: Input rules are classified into three categories based on their internal structural relationships—simple use cases (single self-contained policy), parallel-structure use cases (multiple relatively independent policies), and recursive-structure use cases (cross-clause reference dependencies). Each category corresponds to a distinct processing pipeline.

2. **Three Worker Agents**:

    - **Rewriter Agent**: Handles recursive structures by performing "structure-preserving inlining"—identifying and resolving both explicit references (e.g., clause numbers) and implicit references (e.g., "notwithstanding…"), inlining the referenced clause content into the referencing clause to eliminate semantic dependencies while preserving the structural separation of original clauses.
    - **Splitter Agent**: Handles parallel structures by segmenting rules based on core semantic changes (asset change, role relationship change, policy purpose change) rather than surface syntax, and assigns ODRL types (Agreement / Offer / Set) to each unit via heuristics.
    - **Generator Agent**: The core execution unit that converts structured rule text into ODRL policies, integrating two quality assurance strategies.

3. **Dual Quality Assurance Strategies**:

    - **Validator-Based Strategy (Syntax Validation Loop)**: A generate–validate–revise closed loop based on the PYSHACL library. Generated ODRL policies are validated against SHACL constraint rules; upon failure, detailed error reports are fed back to the Generator LLM for reflective revision, iterating until validation passes or the maximum number of attempts is reached.
    - **LoRA Semantic Reflection Mechanism**: A lightweight LLM (Qwen3-4B-Instruct) fine-tuned with LoRA serves as a semantic extraction expert, extracting key semantic elements (roles, assets, actions, etc.) from original rules to produce a "semantic checkpoint checklist." The main Generator must verify its ODRL output against this checklist to ensure every semantic point is accurately encoded.

### Loss & Training
LoRA fine-tuning parameters: $r=16$, $\alpha=32$, using 2,380 synthetic samples, trained for 3 epochs on a single NVIDIA 4090 GPU. Validation loss (0.0668) is significantly lower than training loss (0.129), indicating good generalization.

## Key Experimental Results

**Experiment 1: Generation Policy Evaluation (770 use cases, GPT-4.1 series)**

| Model | Use Case Type | Metric | OGS | SCR-Enhanced | AOFP (Ours) | Gain (vs SCR) |
|-------|--------------|--------|-----|-------------|-------------|--------------|
| GPT-4.1 | All | Grammar | 82.07 | 93.08 | 99.89 | +7.32% |
| GPT-4.1 | All | Semantic | 89.59 | 92.00 | 97.93 | +6.45% |
| GPT-4.1 | Recursive | Semantic | 76.18 | 78.97 | 96.40 | +22.07% |
| GPT-4.1-nano | All | Grammar | 79.77 | 88.40 | 92.01 | +4.08% |
| GPT-4.1-nano | All | Semantic | 50.51 | 56.23 | 72.35 | +28.67% |
| GPT-4.1-nano | Recursive | Semantic | 34.87 | 40.40 | 61.53 | +52.30% |

Average gains over SCR-Enhanced: Grammar +5.39%, Semantic +14.52%.

**Experiment 2: Orchestrator-Workers Workflow Evaluation (GPT-4.1-nano)**

| Workflow | Grammar | Semantic | Tokens |
|----------|---------|----------|--------|
| Generator only | 92.01 | 72.35 | 33.9M |
| Splitter→Generator | 93.62 | 84.02 | 47.9M |
| Rewriter→Splitter→Generator | 93.27 | 88.07 | 49.5M |
| Orchestrator-Workers (auto) | 92.56 | 80.22 | 46.2M |

### Ablation Study
- The Splitter Agent yields significant gains across all use case categories, particularly for parallel structures (Semantic: 69.44→84.88).
- The Rewriter Agent is indispensable for recursive structures (Semantic: 61.53→82.00).
- Automatic Orchestrator routing trades a marginal performance gap below the theoretical upper bound (80.22 vs. 88.07) for more economical token consumption.
- The syntax validation loop elevates Grammar Scores for all models to near-perfect levels (>99), effectively eliminating syntactic hallucinations from LLMs.
- Weaker models and more complex use cases require more reflection iterations (GPT-4.1-nano on recursive cases: 7.32 rounds on average).

## Highlights & Insights
- This is the first application of a multi-agent architecture to ODRL generation, decomposing a monolithic end-to-end problem into multiple cognitive sub-tasks.
- The generate–validate–revise closed-loop design is elegant, leveraging SHACL to provide structured error feedback.
- The idea of using a LoRA fine-tuned lightweight model as a semantic checker—constraining a larger model's semantic accuracy with a smaller one—is novel and promising.
- The framework's improvement is particularly pronounced for weaker models (GPT-4.1-nano semantic score improves by 76.46%), demonstrating its generalizability.
- The paper constructs the first benchmark dataset in this domain comprising 770 use cases.

## Limitations & Future Work
- The Orchestrator's automatic routing underperforms the theoretical upper bound of manually selected optimal paths (80.22 vs. 88.07), leaving room for improved classification accuracy.
- Evaluation is limited to the GPT-4.1 series; open-source LLMs (e.g., LLaMA, Mistral) are not tested, leaving generalizability in question.
- The dataset is augmented via LLM from 70 seed use cases to 770 examples, potentially limiting diversity.
- Semantic evaluation relies on an LLM Jury, which introduces inherent evaluation bias.
- More complex ODRL structures (e.g., multi-level nested recursion, conditional logic combinations) are not explored.
- The LoRA model is fixed at Qwen3-4B; the impact of different-scale verification models is not investigated.

## Related Work & Insights
- **vs. OGS (Ontology-Guided Strategy)**: OGS uses ODRL ontology to guide a single LLM for generation, but performance degrades sharply on complex structures. AgentODRL avoids single-model cognitive overload through task decomposition and specialized agents.
- **vs. SCR-Enhanced (Mustafa et al., 2025)**: SCR post-processes and corrects errors via predefined rules on top of OGS, but remains a single-model paradigm. AgentODRL's AOFP consistently outperforms SCR across all models and complexity levels, especially on the semantic dimension (average +14.52%).
- **vs. MetaGPT and general-purpose MAS**: General multi-agent frameworks lack ODRL-specific modules (e.g., SHACL validation, LoRA semantic checking). AgentODRL is deeply customized for the characteristics of the ODRL task.
- **"Small Model Constrains Large Model" Paradigm**: Using a LoRA fine-tuned small expert model as a semantic checker to constrain the generation quality of a large model is a transferable paradigm for other generation tasks requiring domain-specific validation (e.g., code generation, medical report generation).
- **Connection to AI Safety**: The Orchestrator-Workers architecture in AgentODRL is a natural application scenario for designing minimal accountable constraint sets for LLM multi-agent collaboration safety—how might minimal safety constraints be designed for each agent in AgentODRL?
- **Generality of the Generate–Validate–Revise Loop**: The design of using SHACL validation to provide structured error feedback can be transferred to other generation tasks with formal specifications (e.g., SQL generation validated against database schemas, API call generation validated against OpenAPI specs).
- **Improvement Headroom for Orchestrator Routing**: Insufficient classification accuracy of the current Orchestrator is the primary bottleneck. Incorporating few-shot examples, uncertainty estimation, or fine-tuned classifiers could improve routing quality.

## Rating
- Novelty: ⭐⭐⭐⭐ (Multi-agent + ODRL is a novel combination, though the Orchestrator-Workers pattern itself is not new)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Two well-designed experiments with reasonably complete ablations, but limited to the GPT-4.1 series)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with detailed descriptions of use case classification and methods, though some details are verbose)
- Value: ⭐⭐⭐ (Narrow domain with limited ODRL application scenarios, but the methodology is transferable)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication](safesieve_from_heuristics_to_experience_in_progressive_pruning_for_llm-based_mul.md)
- [\[AAAI 2026\] Hierarchical Pedagogical Oversight: A Multi-Agent Adversarial Framework for Reliable AI Tutoring](hierarchical_pedagogical_oversight_a_multi-agent_adversarial_framework_for_relia.md)
- [\[ACL 2026\] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](../../ACL2026/model_compression/clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)
- [\[AAAI 2026\] EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training](eeg-dlite_dataset_distillation_for_efficient_large_eeg_model_training.md)
- [\[AAAI 2026\] Failures to Surface Harmful Contents in Video Large Language Models](failures_to_surface_harmful_contents_in_video_large_language_models.md)

</div>

<!-- RELATED:END -->
