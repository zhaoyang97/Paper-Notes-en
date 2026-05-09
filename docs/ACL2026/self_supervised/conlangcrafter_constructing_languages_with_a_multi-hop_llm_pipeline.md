---
title: >-
  [Paper Note] ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline
description: >-
  [ACL 2026][Self-Supervised Learning][Constructed Language] This paper proposes ConlangCrafter, a multi-hop LLM pipeline that decomposes constructed language (conlang) design into three modular stages — phonology, grammar, and lexicon — ensuring typological diversity through randomness injection and internal consistency through self-refinement loops, along with an automatic evaluation framework incorporating typological diversity analysis and translation consistency assessment.
tags:
  - ACL 2026
  - Self-Supervised Learning
  - Constructed Language
  - Multi-Hop Reasoning
  - Typological Diversity
  - Self-Refinement
  - Metalinguistic Reasoning
date: 2025-05-08
content_hash: 895caf7ab7c3470c
---

# ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline

**Conference**: ACL 2026  
**arXiv**: [2508.06094](https://arxiv.org/abs/2508.06094)  
**Code**: [Project Page](https://conlangcrafter.github.io)  
**Area**: Computational Linguistics / Creative Generation  
**Keywords**: Constructed Language, Multi-Hop Reasoning, Typological Diversity, Self-Refinement, Metalinguistic Reasoning

## TL;DR

This paper proposes ConlangCrafter, a multi-hop LLM pipeline that decomposes constructed language (conlang) design into three modular stages — phonology, grammar, and lexicon — ensuring typological diversity through randomness injection and internal consistency through self-refinement loops, along with an automatic evaluation framework incorporating typological diversity analysis and translation consistency assessment.

## Background & Motivation

**State of the Field**: Constructed languages (conlangs) such as Esperanto and Elvish play important roles in art, philosophy, and international communication. Foundation models have achieved revolutionary creative generation in text, images, and other domains.

**Limitations of Prior Work**: (1) Creating constructed languages is extremely time-consuming — designers may spend years or even decades to achieve the scope and complexity of natural languages; (2) LLMs struggle to generate internally consistent complex language systems in a single prompt; (3) LLMs tend to produce outputs lacking typological diversity (Hopkins and Renda 2023), generating languages that are too similar; (4) there is no automated framework for evaluating computational constructed language quality — no ground-truth exists.

**Root Cause**: LLMs possess metalinguistic reasoning capabilities, but directly generating complete language descriptions leads to internal contradictions and insufficient diversity — the various levels of language (phonology, grammar, lexicon) are interdependent and require staged construction.

**Paper Goals**: (1) Investigate whether LLMs can generate internally consistent and typologically diverse language systems; (2) propose scalable automatic evaluation metrics; (3) explore applications of computational constructed languages in creative assistance, game generation, etc.

**Starting Point**: Drawing on language typology and language documentation practices, decomposing language descriptions into phonology → grammar → lexicon layers, constructing each layer through multi-step prompting, injecting typological diversity via RNG, and ensuring consistency via self-refinement.

**Core Idea**: Model constructed language generation as a multi-hop reasoning task — each language level is a reasoning step, maintaining a dynamically updateable "language sketch" memory store to accumulate and harmonize linguistic knowledge.

## Method

### Overall Architecture

Two-stage pipeline: Stage A (Language Sketch Guided) — prompts the LLM sequentially in phonology → grammar → lexicon order, generating language descriptions stored in language sketch S; Stage B (Constructive Translation) — given S, translates new text into the constructed language, dynamically expanding vocabulary and grammar rules as needed. Core components: LLM M (reasoning model such as DeepSeek-R1) + language sketch S (free-text memory store) + optional user constraints c.

### Key Designs

1. **Multi-Hop Language Sketch Generation**:

    - Function: Construct internally consistent language descriptions in stages
    - Mechanism: Decomposes language into phonology, grammar (morpho-syntax), and lexicon layers, generating sequentially by dependency order (phonology before grammar to provide word forms, grammar before lexicon). Each layer is constructed through multiple sub-step prompts with results stored in language sketch S. This structure is analogous to multi-hop methods for other complex reasoning tasks
    - Design Motivation: A single prompt cannot generate sufficiently detailed and consistent language systems — staged construction decomposes the complex task into tractable sub-problems

2. **Randomness Injection**:

    - Function: Ensure typological diversity of generated languages
    - Mechanism: At the start of the phonology and grammar stages, the LLM generates a checklist of 10 language typological features, each with 5 options. An external random number generator (RNG) randomly selects one option per feature, and the LLM instantiates the language description accordingly
    - Design Motivation: LLMs inherently tend to produce similar outputs; delegating diversity control to an external RNG leverages the LLM's typological knowledge while guaranteeing output diversity

3. **Self-Refinement Loop**:

    - Function: Detect and repair internal contradictions in language descriptions and translations
    - Mechanism: Leverages the observation that "evaluation is easier than generation" — uses the same LLM as both critic (identifying errors and ambiguities) and editor (modifying based on error lists), iterating until no further issues are found or maximum iterations are reached
    - Design Motivation: Contradictions in the language sketch propagate to subsequent stages, and translations must obey the constructed grammar, making consistency checks essential

### Loss & Training

No model training is involved. The system uses large reasoning models with inference-time chain-of-thought scaling (DeepSeek-R1, Gemini 2.5 Flash/Pro). Evaluation uses OpenAI o3 as the judge LLM to avoid bias from using the same model for generation and evaluation.

## Key Experimental Results

### Main Results

**Typological Diversity Score (Dmean, higher is better)**

| Method | Dmean |
|--------|-------|
| Natural languages (WALS database, 1,874 languages) | ~0.55 |
| ConlangCrafter (DeepSeek-R1) | Highest |
| ConlangCrafter (Gemini 2.5 Pro) | High |
| ConlangCrafter (Gemini 2.5 Flash) | High |
| Single-stage baseline | Low |

**Translation Consistency Score (Nc,t/Nt,t, higher is better)**

| Method | Consistency Rate |
|--------|-----------------|
| ConlangCrafter (DeepSeek-R1) | Highest |
| ConlangCrafter (Gemini 2.5 Pro) | High |
| Single-stage baseline | Noticeably lower |

### Ablation Study

| Config | Diversity | Consistency |
|--------|-----------|-------------|
| Full ConlangCrafter | High | High |
| w/o randomness injection | Low (diversity drops significantly) | High |
| w/o self-refinement | High | Low (consistency drops significantly) |
| Single-stage baseline | Low | Low |

### Key Findings

- The multi-hop pipeline significantly outperforms single-stage methods in both typological diversity and translation consistency
- Randomness injection is key to ensuring diversity — without it, generated languages cluster together in t-SNE visualization
- The self-refinement loop is critical for consistency — without it, numerous grammar violations appear in translations
- Human expert evaluation shows moderate agreement with automatic evaluation, supporting the validity of the automatic evaluation framework
- DeepSeek-R1 performs best on consistency, while Gemini 2.5 is competitive on diversity

## Highlights & Insights

- "Computational conlanging" is an entirely new paradigm — turning LLM "hallucination" into a creative feature rather than a defect
- The multi-hop reasoning + memory store + self-refinement architecture offers lessons for any LLM task requiring construction of complex consistent systems
- The typological feature checklist + RNG diversity control strategy can be transferred to other generation tasks requiring diversity

## Limitations & Future Work

- The language sketch does not cover semantics, pragmatics, orthography, and other linguistic levels
- Automatic evaluation based on LLM-as-judge has limitations on such highly specialized tasks
- Experiments use only 10 test sentences and approximately 20 languages, a relatively limited scale
- Future work could extend to low-resource language documentation assistance and educational applications

## Related Work & Insights

- **vs Low-resource translation**: In low-resource translation, hallucination is harmful, but in constructed language translation, "hallucination" is a necessary creative element — the target language does not actually exist
- **vs Procedural world generation**: ConlangCrafter can be directly applied to societal/linguistic procedural generation in open-world games
- **vs Chain-of-thought reasoning**: The multi-hop pipeline is essentially a structured chain-of-thought, with each layer corresponding to a reasoning step

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Pioneers the entirely new research paradigm of "computational conlanging"
- Experimental Thoroughness: ⭐⭐⭐⭐ Automatic + human evaluation, thorough ablation studies, but limited sample size
- Writing Quality: ⭐⭐⭐⭐ Clear background and motivation, detailed method description
- Recommendation: ⭐⭐⭐⭐ Inspiring for creative AI and linguistics research, with broad application prospects

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space](../../AAAI2026/self_supervised/towards_llm-empowered_knowledge_tracing_via_llm-student_hierarchical_behavior_al.md)
- [\[CVPR 2026\] GeoBridge: A Semantic-Anchored Multi-View Foundation Model for Geo-Localization](../../CVPR2026/self_supervised/geobridge_semantic-anchored_multi-view_foundation_model_for_geo-localization.md)
- [\[CVPR 2026\] Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](../../CVPR2026/self_supervised/breaking_the_tuning_barrier_zerohyperparameters_yi.md)
- [\[CVPR 2026\] TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation](../../CVPR2026/self_supervised/teflow_enabling_multi-frame_supervision_for_self-supervised_feed-forward_scene_f.md)
- [\[ACL 2026\] \[b\] = \[d\] − \[t\] + \[p\]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)

<!-- RELATED:END -->
