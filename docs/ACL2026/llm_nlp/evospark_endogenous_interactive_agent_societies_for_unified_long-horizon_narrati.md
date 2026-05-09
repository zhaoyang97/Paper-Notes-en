---
title: >-
  [Paper Note] EvoSpark: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution
description: >-
  [ACL 2026][LLM/NLP][multi-agent narrative] EvoSpark proposes a multi-agent framework for long-horizon narrative evolution, addressing social memory stacking and narrative–spatial misalignment through three core designs: hierarchical recursive memory (RSB as social cognitive metabolism), generative scene scheduling (GMS for character–location–plot alignment), and an emergent character grounding protocol (ECGP that converts LLM hallucinations into persistent entities).
tags:
  - ACL 2026
  - LLM/NLP
  - multi-agent narrative
  - long-horizon story evolution
  - social memory metabolism
  - spatial alignment
  - emergent characters
date: 2026-05-08
content_hash: 716957567429dca1
---

# EvoSpark: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution

**Conference**: ACL 2026
**arXiv**: [2604.12776](https://arxiv.org/abs/2604.12776)
**Code**: None
**Area**: LLM/NLP
**Keywords**: multi-agent narrative, long-horizon story evolution, social memory metabolism, spatial alignment, emergent characters

## TL;DR
EvoSpark proposes a multi-agent framework for long-horizon narrative evolution, addressing social memory stacking and narrative–spatial misalignment through three core designs: hierarchical recursive memory (RSB as social cognitive metabolism), generative scene scheduling (GMS for character–location–plot alignment), and an emergent character grounding protocol (ECGP that converts LLM hallucinations into persistent entities).

## Background & Motivation

**Background**: LLM-based multi-agent systems have made progress in narrative generation (e.g., Generative Agents, BookWorld), but exhibit systematic degradation in long-horizon simulations.

**Limitations of Prior Work**: (1) *Social memory stacking* — append-only memory causes contradictory relationship states to accumulate (e.g., a character being simultaneously friend and enemy), leading to behavioral incoherence; (2) *Narrative–spatial misalignment* — text-based agents lack spatial state synchronization, causing characters to appear in locations that contradict narrative logic.

**Key Challenge**: Long-horizon narrative requires balancing *open emergence* and *logical consistency* — excessive control sacrifices autonomy, while excessive freedom leads to chaos. Existing frameworks either follow rigid scripts (sacrificing emergence) or operate fully open (sacrificing coherence).

**Goal**: Construct a unified framework spanning the full spectrum from strict hierarchical planning to completely free emergence, while maintaining long-horizon logical consistency.

**Key Insight**: Redesign the memory system and spatial management — memory is treated not as an append log but as "living cognition" (subject to metabolic updates), and space is treated not as a passive container but as a "virtual stage manager."

**Core Idea**: Relationship Social Base (RSB) for memory metabolism + Generative Scene Scheduler (GMS) for spatial alignment + Emergent Character Grounding Protocol (ECGP) for converting hallucinations into creative assets.

## Method

### Overall Architecture
Four agent types collaborate: Genesis Agent (narrative conception and macro-planning), Architect Agent (world instantiation and character promotion), Director Agent (simulation execution and spatial alignment), and Role Agents (interaction execution and memory updates). Three control modes are supported: HDP (Hierarchical Detailed Planning), SNP (Sequential Key Nodes), and Free EN (Fully Free Emergence).

### Key Designs

1. **Relationship Social Base (RSB) and the Reflect–Synthesize–Consolidate Mechanism**:

    - **Function**: Resolves social memory stacking by transforming memory from an append log into metabolizable living cognition.
    - **Mechanism**: A four-layer memory architecture — Episode Evolution Buffer (EEB, short-term cache), Shared World Knowledge Base (SWKB, immutable global truth), Role Episode Base (REB, immutable experience log for provenance), and Role Social Base (RSB, mutable snapshot of current state). At the end of each event, a Reflect–Synthesize–Consolidate cycle is triggered: reflection trigger (interaction intensity exceeds threshold) → synthesis (contrast new EEB data with existing RSB state, resolve topological conflicts) → consolidation (in-place overwrite of RSB, replacing old relationships with new ones).
    - **Design Motivation**: Generative Agents' reflection only synthesizes observations without metabolism — old relationships are stacked rather than replaced, inevitably producing contradictions over long horizons.

2. **Generative Scene Scheduler (GMS)**:

    - **Function**: Resolves narrative–spatial misalignment by ensuring characters appear in locations consistent with narrative logic.
    - **Mechanism**: Operates in two phases — offline planning alignment (Genesis Agent establishes initial constraints across character, location, and plot dimensions) and dynamic spatial alignment (Director Agent synchronizes narrative intent with real-time context via spatial blocking at runtime, including an entity resolution step to correct identity hallucinations produced by the LLM). GMS functions as a "virtual stage manager" that implicitly endows agents with spatial awareness.
    - **Design Motivation**: Environments in existing frameworks are typically passive containers — BookWorld offers discrete geographic tracking but lacks fine-grained character–location–plot alignment.

3. **Emergent Character Grounding Protocol (ECGP)**:

    - **Function**: Converts LLM hallucinations (generating names of uninitialized characters) into persistent story-world entities.
    - **Mechanism**: A four-step pipeline — hallucination detection (LLM generates a new name despite a constrained character list, treated as a narrative necessity signal) → entity resolution (Director verifies whether the name is a genuinely new entity rather than an alias) → ontological promotion (hierarchical status is elevated based on narrative importance) → integration and grounding (Architect instantiates the new character in the story world and RSB).
    - **Design Motivation**: Reframes hallucination as a creative asset — generative narrative requires open-world expansion, and LLM stochasticity provides a natural mechanism for emergent character introduction.

### Loss & Training
EvoSpark is a purely inference-time framework with no training involved. Multiple LLM backbones are employed (GPT-4o and open-source models in experiments).

## Key Experimental Results

### Main Results
EvoSpark significantly outperforms Open-Theatre, BookWorld, and HoLLMwood across character performance, narrative coherence, and spatial consistency under three control modes (HDP, SNP, Free EN) and multilingual, multi-backbone settings. Long-horizon runs generate 200k–250k words per simulation.

### Ablation Study

| Configuration | Key Metric | Observation |
|---|---|---|
| Without GMS dynamic spatial alignment | Increased spatial contradictions | Characters "lost in space" |
| Without RSB metabolism | Behavioral incoherence over long horizons | Social memory stacking causes contradictions |
| Without ECGP | Limited world expansion | Loss of emergent character capability |

### Key Findings
- The presence of GMS directly determines physical consistency — without GMS, logical contradictions such as "a character gazing at A while physically facing B" emerge.
- RSB's metabolic mechanism is central to long-horizon consistency — append-only memory exhibits severe stacking after as few as 15 events.
- ECGP validates the "hallucination as creativity" hypothesis — approximately 20% of emergent characters make meaningful contributions to subsequent narrative.

## Highlights & Insights
- The ECGP design of **converting LLM hallucinations into creative assets** is highly instructive — in other domains, "erroneous" LLM outputs might similarly be reframed as exploratory generation.
- The concept of **memory metabolism** is more principled than simple memory management — rather than managing storage capacity, it keeps memory "alive" through metabolic renewal, more closely approximating the nature of human memory.
- The unified framework spanning three control modes (HDP/SNP/Free EN) demonstrates how a single architecture can support the full spectrum from strict to free narrative control.

## Limitations & Future Work
- Experiments are primarily validated on fictional narrative; applicability to other simulation types (e.g., social science simulations) remains to be verified.
- Computational cost is high — long-horizon simulations require a large number of LLM calls.
- The metabolic trigger threshold in RSB is a hyperparameter that may require domain-specific tuning for different story types.
- ECGP's entity resolution may degrade when the number of characters is very large.

## Related Work & Insights
- **vs. Generative Agents**: GA synthesizes observations via reflection but performs no metabolism; EvoSpark uses RSB for in-place updates to resolve memory stacking.
- **vs. BookWorld**: BW provides discrete geographic tracking but lacks character–location–plot alignment; EvoSpark achieves fine-grained spatial management via GMS.
- **vs. HoLLMwood**: HW employs a writer–editor workflow to refine narrative quality but lacks spatial management and memory metabolism mechanisms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Memory metabolism, spatial scheduling, and hallucination conversion are all genuinely novel concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three control modes, multiple baselines, and long-horizon consistency analysis.
- Writing Quality: ⭐⭐⭐⭐ — Framework descriptions are detailed, though the large number of component names imposes a high cognitive load.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)
- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](../../AAAI2026/llm_nlp/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[AAAI 2026\] CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models](../../AAAI2026/llm_nlp/coevo_continual_evolution_of_symbolic_solutions_using_large_language_models.md)
- [\[ICCV 2025\] VIM: Versatile Interactive Motion-Language Model](../../ICCV2025/llm_nlp/vim_versatile_interactive_motion_language_model.md)

<!-- RELATED:END -->
