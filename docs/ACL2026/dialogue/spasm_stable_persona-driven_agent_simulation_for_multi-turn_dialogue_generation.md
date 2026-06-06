---
title: >-
  [Paper Note] SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation
description: >-
  [ACL 2026 Findings][Dialogue Systems][Persona-based Dialogue] This work proposes SPASM, a stability-centric persona-driven multi-turn dialogue simulation framework. Through modular persona generation…
tags:
  - "ACL 2026 Findings"
  - "Dialogue Systems"
  - "Persona-based Dialogue"
  - "Multi-turn Simulation"
  - "Character Drift"
  - "Egocentric Context Projection"
  - "Data Generation"
date: 2026-05-08
content_hash: e02f3ad58132297e
---

# SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.09212](https://arxiv.org/abs/2604.09212)  
**Code**: [GitHub](https://github.com/lhannnn/SPASM)  
**Area**: Dialogue Systems  
**Keywords**: Persona-based Dialogue, Multi-turn Simulation, Character Drift, Egocentric Context Projection, Data Generation

## TL;DR

This work proposes SPASM, a stability-centric persona-driven multi-turn dialogue simulation framework. Through modular persona generation, Egocentric Context Projection (ECP), and termination detection, it significantly reduces character drift and the "echoing" effect in LLM-LLM dialogues, constructing a high-quality dataset of 45,000 multi-turn dialogues.

## Background & Motivation

**Background**: LLMs are extensively deployed in multi-turn interactive scenarios such as tutoring, support, and consultation. LLM-LLM dialogue simulation serves as an efficient method to generate large-scale training/evaluation data, offering lower costs and better controllability compared to manual collection.

**Limitations of Prior Work**: Long LLM-LLM dialogues accumulate identity-related failures—persona drift (characters gradually deviating from assigned identities), role confusion, and the "echoing" effect (where one agent mimics the language and stance of the other). These issues intensify as dialogues lengthen, leading to generated data that fails to match intended settings and pollutes synthetic datasets.

**Key Challenge**: The root cause lies in the naive concatenation of dialogue history—the same utterance may occupy different relative roles (user vs. assistant) for different agents, leading to role confusion and feedback loops.

**Goal**: To design a "stability-first" dialogue simulation framework that ensures long-term persona consistency without modifying model weights.

**Key Insight**: The problem is addressed by altering the **representation** of dialogue history rather than the model itself—storing dialogue history in a perspective-neutral format and deterministically projecting it into each agent's egocentric perspective during generation.

**Core Idea**: Egocentric Context Projection (ECP): Dialogue history is stored as $(speaker\_id, content)$ pairs. During generation, a role re-labeling operator $\Psi_i$ maps speaker labels to SELF/PARTNER, ensuring that each agent consistently perceives the dialogue from its own perspective.

## Method

### Overall Architecture

SPASM consists of five components: (1) Persona Schema (sampling attribute fields) → (2) Persona Validator (verifying combination rationality) → (3) Persona Crafter (generating natural language descriptions) → (4) Client-Responder dialogue simulation (incorporating ECP) → (5) Termination Detector (identifying natural ending points).

### Key Designs

1.  **Egocentric Context Projection (ECP)**:

    - **Function**: Eliminates role confusion and echoing effects, ensuring long-term persona consistency.
    - **Mechanism**: Dialogue history is stored as a perspective-agnostic ordered sequence $\mathcal{H}_t = (u_k)_{k=1}^t$, where $u_k = (s_k, c_k)$ (Speaker ID + Content). During generation, the projection operator $\Psi_i(\mathcal{H}_t) = ((\phi_i(s_k), c_k))_{k=1}^t$ maps absolute speakers to relative role descriptions (SELF/PARTNER). This ensures that in the history perceived by each agent, its own utterances are marked as SELF, while the counterpart's are marked as PARTNER.
    - **Design Motivation**: The fixed assignment of user/assistant labels in naive concatenation causes role confusion. ECP transforms this into a symmetric SELF/PARTNER representation, decoupling role labels from agent identity.

2.  **Modular Persona Generation Pipeline**:

    - **Function**: Generates diverse, rational, and controllable persona descriptions.
    - **Mechanism**: A three-step process—Schema Sampling (random selection from predefined fields: age, occupation, location, emotional state, behavioral patterns, etc.) → Validator (checking coherence, e.g., re-sampling if "18-year-old student + pension planning" is detected) → Crafter (converting validated attributes into coherent natural language descriptions with optional detail expansion).
    - **Design Motivation**: Direct attribute sampling may lead to irrational combinations. The validator and crafter ensure persona credibility.

3.  **Termination Detector**:

    - **Function**: Detects natural ending points and terminates dialogues to avoid forced truncation or infinite loops.
    - **Mechanism**: Activated after $T$ rounds, it assesses whether closing signals (e.g., expressions of gratitude or farewell) are present based on the most recent $m$ turns and predefined rules.
    - **Design Motivation**: Hard truncation results in unnatural endings; termination detection ensures dialogue coherence and naturalness.

### Loss & Training

The framework is entirely training-free. All components are implemented via API calls without modifying model weights.

## Key Experimental Results

### Persona Retrieval Accuracy (Top-1 Acc)

| Client / Responder | Top-1 | Top-10 |
|-------------------|-------|--------|
| GPT / GPT | 0.96 | 1.00 |
| GPT / DeepSeek | 0.50 | 0.82 |
| DS / GPT | 0.99 | 1.00 |
| Qwen / Qwen | 0.98 | 1.00 |

### Ablation Study (ECP Effect)

| Metric | With ECP | Without ECP |
|------|-------|--------|
| Persona Drift | Significantly Reduced | High |
| Echo Effect | Near Zero (Human Eval) | Frequent |
| Silhouette Score | High (0.60) | Low |

### Key Findings
- ECP is the most critical design: it significantly reduces persona drift and virtually eliminates echoing effects in human evaluations.
- Interactions between identical backbone models produce tighter persona clusters (GPT/GPT Silhouette=0.60 vs. GPT/DS=0.10).
- The Responder backbone dominates interaction geometry: if the Responder is fixed as GPT, clustering quality remains high regardless of the Client model.
- Cross-model interactions primarily increase intra-cluster variance rather than decreasing inter-cluster separation.
- A large-scale dataset was constructed featuring 4,500 personas and 45,000 dialogues.

## Highlights & Insights
- The **"minimal change, maximum effect" of ECP** is elegant: by merely modifying the representation of role labels (user/assistant → SELF/PARTNER), long-term stability is vastly improved. This simple approach suggests that role representation is more fundamental to identity than raw model capability.
- The finding that the **Responder model dominates interaction geometry** is insightful: in persona-driven dialogues, the responder (rather than the initiator) defines the structure of the dialogue space, implying that the "listener" has a greater impact on interaction quality than the "speaker."
- The **Persona Validation step** prevents irrational attribute combinations, enhancing dataset credibility—a practice highly recommended for synthetic data generation.

## Limitations & Future Work
- Only English dialogues were verified; effectiveness in multilingual contexts remains unknown.
- Persona attribute fields are predefined and may not cover all potential application scenarios.
- Maximum dialogue length is restricted to 25 turns per agent; stability for extreme-length dialogues was not tested.
- The impact of using the generated data for downstream SFT was not evaluated.
- While theoretically feasible, the extension of ECP to multi-agent (>2) scenarios has not been verified.

## Related Work & Insights
- **vs Self-Chat/RolePlay**: Unlike these methods that use simple history concatenation, SPASM ensures long-term persona consistency via ECP.
- **vs Generative Agents (Park et al.)**: While prior work emphasizes memory and behavior, SPASM focuses on dialogue data generation and identity stability.
- **vs Instruction Drift Research (Li et al.)**: This work extends similar metrics to the realm of persona-driven dialogue generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ ECP is simple yet effective, and the persona stability analysis is thorough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 9 backbone combinations, 45K dialogues, and multi-dimensional analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear formalization and insightful analysis.
- **Value**: ⭐⭐⭐⭐ Provides a practical stability solution for LLM-based dialogue data generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GenesisFunc: Multi-Agent Data Generation for Accurate and Generalizable Function-Calling](genesisfunc_multi-agent_data_generation_for_accurate_and_generalizable_function-.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)
- [\[ACL 2026\] Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation](discourse_coherence_and_response-guided_context_rewriting_for_multi-party_dialog.md)
- [\[ACL 2026\] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](odutqa-mdc_a_task_for_open-domain_underspecified_tabular_qa_with_multi-turn_dial.md)
- [\[ACL 2026\] Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue](context-agent_dynamic_discourse_trees_for_non-linear_dialogue.md)

</div>

<!-- RELATED:END -->
