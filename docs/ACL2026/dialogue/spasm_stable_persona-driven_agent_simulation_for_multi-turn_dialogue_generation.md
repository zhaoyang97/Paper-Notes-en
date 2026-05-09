---
title: >-
  [Paper Note] SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation
description: >-
  [ACL 2026 Findings][Dialogue Systems][Persona-based Dialogue] This paper proposes SPASM, a stability-centric persona-driven multi-turn dialogue simulation framework that significantly reduces role drift and echo effects in LLM-LLM conversations through three components: modular persona generation, Egocentric Context Projection (ECP), and termination detection, constructing 45,000 high-quality multi-turn dialogue instances.
tags:
  - ACL 2026 Findings
  - Dialogue Systems
  - Persona-based Dialogue
  - Multi-turn Simulation
  - Role Drift
  - Egocentric Projection
  - Data Generation
date: 2026-05-08
content_hash: c3b9755bfdc78f40
---

# SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.09212](https://arxiv.org/abs/2604.09212)  
**Code**: [GitHub](https://github.com/lhannnn/SPASM)  
**Area**: Dialogue Systems  
**Keywords**: Persona-based Dialogue, Multi-turn Simulation, Role Drift, Egocentric Projection, Data Generation

## TL;DR

This paper proposes SPASM, a stability-centric persona-driven multi-turn dialogue simulation framework that significantly reduces role drift and echo effects in LLM-LLM conversations through three components: modular persona generation, Egocentric Context Projection (ECP), and termination detection, constructing 45,000 high-quality multi-turn dialogue instances.

## Background & Motivation

**State of the Field**: LLMs are widely deployed in multi-turn interactive scenarios such as tutoring, support, and counseling. LLM-LLM dialogue simulation is an effective approach for generating large-scale training/evaluation data, offering lower costs and better controllability compared to human collection.

**Limitations of Prior Work**: LLM-LLM long conversations accumulate identity-related failures—persona drift (agents gradually deviate from assigned identities), role confusion, and echo effects (one agent gradually mimics another's language and stance). These issues intensify as dialogues lengthen, causing generated conversations to no longer correspond to intended settings and contaminating synthetic datasets.

**Root Cause**: Naive dialogue history concatenation is the root problem—the same utterance may occupy different relative roles (user vs assistant) for different agents, leading to role confusion and feedback loops.

**Paper Goals**: Design a "stability-first" dialogue simulation framework that ensures long-term role consistency without modifying model weights.

**Starting Point**: Address the problem by changing the **representation method** of dialogue history rather than the model itself—store dialogue history in a perspective-neutral format and deterministically project it to each agent's egocentric view during generation.

**Core Idea**: Egocentric Context Projection (ECP): Store dialogue history in $(speaker\_id, content)$ format, and during generation use role relabeling operator $\Psi_i$ to map speaker labels to SELF/PARTNER, ensuring each agent always views the dialogue from its own perspective.

## Method

### Overall Architecture

SPASM comprises five components: (1) Persona Schema (samples persona attributes) → (2) Persona Validator (verifies combination plausibility) → (3) Persona Crafter (generates natural language persona descriptions) → (4) Client-Responder dialogue simulation (with ECP) → (5) Termination Detector (detects natural endpoints).

### Key Designs

1. **Egocentric Context Projection (ECP)**:

    - Function: Eliminates role confusion and echo effects, ensuring long-term persona consistency
    - Mechanism: Dialogue history is stored as a perspective-neutral ordered sequence $\mathcal{H}_t = (u_k)_{k=1}^t$, where $u_k = (s_k, c_k)$ (speaker ID + content). During generation, projection operator $\Psi_i(\mathcal{H}_t) = ((\phi_i(s_k), c_k))_{k=1}^t$ maps absolute speakers to relative role descriptions (SELF/PARTNER). This ensures that in the dialogue history seen by each agent, its own utterances are labeled SELF and the other's are labeled PARTNER
    - Design Motivation: Fixed assignment of user/assistant labels in naive concatenation is the source of role confusion. ECP converts this to symmetric SELF/PARTNER representation, decoupling role labels from agent identities

2. **Modular Persona Generation Pipeline**:

    - Function: Generates diverse, plausible, controllable persona descriptions
    - Mechanism: Three-step process—Schema Sampling (randomly samples from predefined fields: age, occupation, location, emotional state, behavioral patterns, etc.) → Validator (checks coherence and plausibility of combinations; implausible ones like "18-year-old student + retirement planning" trigger resampling) → Crafter (converts validated attribute sets into coherent natural language persona descriptions, with possible additional detail expansion)
    - Design Motivation: Directly using randomly sampled attribute combinations may be implausible. Validator + refiner ensure persona credibility

3. **Termination Detector**:

    - Function: Detects and terminates dialogue at natural endpoints, avoiding forced truncation or infinite loops
    - Mechanism: Activates after turn $T$, judges whether closing signals appear (e.g., expressing thanks, farewells) based on recent $m$ turns of dialogue history and predefined termination rules
    - Design Motivation: Hard truncation produces unnatural endings; termination detection ensures dialogue coherence and naturalness

### Loss & Training

Completely training-free. All components implemented via API calls without modifying model weights.

## Key Experimental Results

### Persona Retrieval Accuracy (Top-1 Acc)

| Client / Responder | Top-1 | Top-10 |
|-------------------|-------|--------|
| GPT / GPT | 0.96 | 1.00 |
| GPT / DeepSeek | 0.50 | 0.82 |
| DS / GPT | 0.99 | 1.00 |
| Qwen / Qwen | 0.98 | 1.00 |

### Ablation Study (ECP Effects)

| Metric | With ECP | Without ECP |
|------|-------|--------|
| Persona Drift | Significantly reduced | High |
| Echo Effect | Near-zero in manual verification | Frequent |
| Silhouette Score | High (0.60) | Low |

### Key Findings
- ECP is the most critical design: dramatically reduces persona drift, nearly eliminates echo effects in manual verification
- Same-backbone model interactions produce tighter persona clusters (GPT/GPT Silhouette=0.60 vs GPT/DS=0.10)
- Responder model backbone dominates interaction geometry: when Responder is fixed to GPT, clustering quality is high regardless of Client
- Cross-model interactions primarily increase within-cluster variance rather than reducing between-cluster separation
- Constructed large-scale dataset of 4,500 personas × 45,000 dialogues

## Highlights & Insights
- **ECP's "minimal change, maximum effect"** is highly elegant: merely changing role label representation in dialogue history (user/assistant → SELF/PARTNER) dramatically improves long-term stability. This simple idea has profound implications—role representation matters more than model capability
- **Responder model dominates interaction geometry** is an interesting finding: in persona-driven dialogue, the responder (not the initiator) determines the structure of dialogue space, suggesting "listener" impacts interaction quality more than "speaker"
- **Persona validation step** avoids implausible combinations, making the dataset more credible—a practice worth promoting in synthetic data generation

## Limitations & Future Work
- Only validated on English dialogues; effectiveness in multilingual scenarios is unknown
- Persona attribute fields are predefined and may not cover all application scenarios
- Maximum dialogue length limited to 25 turns/agent; stability in longer dialogues untested
- Effectiveness of generated data for downstream SFT training not evaluated
- ECP extension to multi-agent (>2) scenarios is theoretically feasible but unverified

## Related Work & Insights
- **vs Self-Chat/RolePlay**: These methods use simple dialogue history concatenation; SPASM addresses long-term role consistency via ECP
- **vs Generative Agents (Park et al.)**: Focuses on memory and behavioral simulation; SPASM specializes in dialogue data generation and identity stability
- **vs Instruction drift research (Li et al.)**: This paper extends similar measurement methods to persona-driven dialogue generation scenarios

## Rating
- Novelty: ⭐⭐⭐⭐ ECP is simple yet effective, in-depth persona stability analysis
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 backbone combinations, 45K dialogues, multi-dimensional analysis
- Writing Quality: ⭐⭐⭐⭐ Clear formalization, thorough analysis
- Value: ⭐⭐⭐⭐ Provides practical stability solution for LLM dialogue data generation

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)
- [\[ACL 2026\] Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation](discourse_coherence_and_response-guided_context_rewriting_for_multi-party_dialog.md)
- [\[NeurIPS 2025\] MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](../../NeurIPS2025/dialogue/metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)
- [\[ACL 2026\] VoxMind: An End-to-End Agentic Spoken Dialogue System](voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)
- [\[ACL 2026\] Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](template-assisted_contrastive_learning_of_task-oriented_dialogue_sentence_embedd.md)

<!-- RELATED:END -->
