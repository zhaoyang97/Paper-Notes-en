---
title: >-
  [Paper Note] ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline
description: >-
  [ACL 2026][Text Generation][Constructed Languages] This paper proposes ConlangCrafter, an LLM-based multi-hop pipeline that decomposes constructed language (conlang) design into modular stages of phonology, grammar…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Constructed Languages"
  - "Multi-Hop Reasoning"
  - "Typological Diversity"
  - "Self-Refinement"
  - "Metalinguistic Reasoning"
date: 2026-05-08
content_hash: e00554e2b9b348ef
---

# ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline

**Conference**: ACL 2026  
**arXiv**: [2508.06094](https://arxiv.org/abs/2508.06094)  
**Code**: [Project Page](https://conlangcrafter.github.io)  
**Area**: Computational Linguistics / Creative Generation  
**Keywords**: Constructed Languages, Multi-Hop Reasoning, Typological Diversity, Self-Refinement, Metalinguistic Reasoning

## TL;DR

This paper proposes ConlangCrafter, an LLM-based multi-hop pipeline that decomposes constructed language (conlang) design into modular stages of phonology, grammar, and lexicon. It ensures typological diversity through randomness injection and internal consistency via self-refinement loops, while introducing an automated evaluation framework consisting of typological diversity analysis and translation consistency assessment.

## Background & Motivation

**Background**: Constructed languages (conlangs) like Esperanto and Elvish play significant roles in art, philosophy, and international communication. Foundation models have revolutionized creative generation across domains like text and image.

**Limitations of Prior Work**: (1) Creating conlangs is extremely time-consuming—designers may spend years or decades to reach the scope and complexity of natural languages; (2) LLMs struggle to generate internally consistent complex language systems under a single prompt; (3) LLMs tend to produce outputs lacking typological diversity (Hopkins and Renda 2023), resulting in overly similar languages; (4) Lack of automated frameworks for evaluating computational conlang quality—there is no ground-truth.

**Key Challenge**: LLMs possess metalinguistic reasoning capabilities, but directly generating a full language description leads to internal contradictions and insufficient diversity—different levels of language (phonology, grammar, lexicon) are interdependent and require phased construction.

**Goal**: (1) Investigate whether LLMs can generate internally consistent and typologically diverse language systems; (2) Propose scalable automated evaluation metrics; (3) Explore the applications of computational conlangs in creative assistance, game generation, etc.

**Key Insight**: Drawing from linguistic typology and language documentation practices, language descriptions are divided into three layers: phonology $\rightarrow$ grammar $\rightarrow$ lexicon. Each layer is constructed through multi-step prompting, utilizing RNG to inject typological diversity and self-refinement to ensure consistency.

**Core Idea**: Modeling conlang generation as a multi-hop reasoning task—each linguistic layer is a reasoning step, accumulating and consolidating linguistic knowledge by maintaining a dynamically updatable "Language Sketch" memory bank.

## Method

### Overall Architecture

A two-stage pipeline: Stage A (Language Sketch Guidance)—sequentially prompts the LLM to generate language descriptions for phonology $\rightarrow$ grammar $\rightarrow$ lexicon and store them in language sketch $S$; Stage B (Constructive Translation)—given $S$, translates new text into the conlang, dynamically expanding vocabulary and grammar rules during the process. Core components: LLM $M$ (reasoning models like DeepSeek-R1) + Language Sketch $S$ (free-text memory bank) + optional user constraints $c$.

### Key Designs

1.  **Multi-Hop Language Sketch Generation**:

    - **Function**: Phased construction of internally consistent language descriptions.
    - **Mechanism**: Divides language into three layers: phonology, grammar (morpho-syntax), and lexicon, generated in order of dependency (phonology before grammar to provide word forms, grammar before lexicon). Each layer prompts the LLM via multiple sub-steps, with results stored in Language Sketch $S$. This structure is analogous to multi-hop methods used in other complex reasoning tasks.
    - **Design Motivation**: A single prompt cannot generate a sufficiently detailed and consistent language system—hierarchical step-by-step construction breaks down the complex task into manageable sub-problems.

2.  **Randomness Injection**:

    - **Function**: Ensures the generated languages are typologically diverse.
    - **Mechanism**: At the start of the phonology and grammar stages, the LLM generates a checklist containing 10 linguistic typological features, each with 5 options. An external random number generator (RNG) is then used to randomly select one option for each feature, which the LLM uses to instantiate the language description.
    - **Design Motivation**: LLMs inherently tend to generate similar outputs; delegating diversity control to an external RNG leverages the LLM's typological knowledge while ensuring output variety.

3.  **Self-Refinement**:

    - **Function**: Detects and repairs internal contradictions in language descriptions and translations.
    - **Mechanism**: Leverages the observation that "evaluation is easier than generation"—using the same LLM to perform the roles of critic (identifying errors and ambiguities) and editor (revising based on the error list), executing iteratively until no further issues are found or the maximum number of iterations is reached.
    - **Design Motivation**: Contradictions in the language sketch propagate to subsequent stages, and translations must adhere to the constructed grammar, making consistency checks crucial.

### Loss & Training

No model training involved. Uses large reasoning models with inference-time chain-of-thought expansion (DeepSeek-R1, Gemini 2.5 Flash/Pro). Evaluation uses OpenAI o3 as a judge LLM to avoid bias from using the same model for generation and evaluation.

## Key Experimental Results

### Main Results

**Typological Diversity Score ($D_{mean}$, higher is better)**

| Method | $D_{mean}$ |
|------|-------|
| Natural Language (WALS Database, 1874 types) | ~0.55 |
| **Ours** (DeepSeek-R1) | **Highest** |
| **Ours** (Gemini 2.5 Pro) | High |
| **Ours** (Gemini 2.5 Flash) | High |
| Single-stage Baseline | Low |

**Translation Consistency Score ($N_{c,t}/N_{t,t}$, higher is better)**

| Method | Consistency Rate |
|------|---------|
| **Ours** (DeepSeek-R1) | **Highest** |
| **Ours** (Gemini 2.5 Pro) | High |
| Single-stage Baseline | Significantly Lower |

### Ablation Study

| Configuration | Diversity | Consistency |
|------|--------|--------|
| Full ConlangCrafter | High | High |
| w/o Randomness Injection | Low (significant drop) | High |
| w/o Self-Refinement | High | Low (significant drop) |
| Single-stage Baseline | Low | Low |

### Key Findings

- The multi-hop pipeline is significantly better than single-stage methods in both typological diversity and translation consistency.
- Randomness injection is key to guaranteeing diversity—without it, the generated languages cluster together in t-SNE visualizations.
- Self-refinement loops are critical for consistency—without them, numerous grammatical violations appear in translations.
- Human expert evaluations show moderate agreement with automated evaluations, supporting the effectiveness of the automated framework.
- DeepSeek-R1 performs best in consistency, while Gemini 2.5 remains competitive in diversity.

## Highlights & Insights

- "Computational Conlanging" is a brand-new paradigm—transforming LLM "hallucinations" into creative features rather than defects.
- The architecture of multi-hop reasoning + memory bank + self-refinement offers insights for any LLM task requiring the construction of complex, consistent systems.
- The diversity control strategy of typological feature checklists + RNG is transferable to other generation tasks requiring variety.

## Limitations & Future Work

- The language sketch does not cover additional linguistic layers such as semantics, pragmatics, and orthography.
- Automated evaluation is based on LLM-as-judge, which still has limitations on such highly specialized tasks.
- Experiments used only 10 test sentences and about 20 languages, representing a relatively limited scale.
- Future work could extend to assisted documentation for low-resource languages and educational applications.

## Related Work & Insights

- **vs Low-Resource Translation**: Hallucinations are harmful in low-resource translation, but in conlang translation, "hallucination" is a necessary creative element—the target language does not exist to begin with.
- **vs Procedural World Generation**: ConlangCrafter can be directly applied to the procedural generation of societies/languages in open-world games.
- **vs Chain-of-Thought Reasoning**: The multi-hop pipeline is essentially a structured chain of thought, where each layer corresponds to a reasoning step.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Pioneered "Computational Conlanging" as a new research paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Combined automated and human evaluation with thorough ablation, though sample size is limited.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and detailed methodology.
- Value: ⭐⭐⭐⭐ Inspiring for creative AI and linguistics, with broad application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)
- [\[ACL 2026\] Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering](are_emotion_and_rhetoric_neurons_in_llm_neuron_recognition_and_adaptive_masking_.md)
- [\[ACL 2026\] Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style](can_you_make_it_sound_like_you_post-editing_llm-generated_text_for_personal_styl.md)
- [\[ACL 2026\] Investigating the Representation of Backchannels and Fillers in Fine-tuned Language Models](investigating_the_representation_of_backchannels_and_fillers_in_fine-tuned_langu.md)
- [\[ACL 2026\] XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration](xtragpt_context-aware_and_controllable_academic_paper_revision_via_human-ai_coll.md)

</div>

<!-- RELATED:END -->
