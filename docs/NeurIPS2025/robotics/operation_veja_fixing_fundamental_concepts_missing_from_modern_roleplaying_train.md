---
title: >-
  [Paper Note] Operation Veja: Fixing Fundamental Concepts Missing from Modern Roleplaying Training Paradigms
description: >-
  [NeurIPS 2025][Robotics][Role-playing] This paper systematically critiques four dominant paradigms in role-playing (RP) model training—RAG, fact-value specification, literary data, and synthetic data—arguing that none can produce characters with genuine depth. It proposes the VEJA framework (Values–Experiences–Judgments–Abilities) as a structured basis for character definition and data curation. In an LLM-judged A/B test, VEJA-guided human-curated data significantly outperforms a Gemini Pro 2.5 synthetic baseline with a win/loss/tie ratio of 43:28:29.
tags:
  - NeurIPS 2025
  - Robotics
  - Role-playing
  - VEJA framework
  - value-conflicted reasoning
  - data curation
  - character consistency
date: 2026-05-08
content_hash: f65fbbd954e830bc
---

# Operation Veja: Fixing Fundamental Concepts Missing from Modern Roleplaying Training Paradigms

**Conference**: NeurIPS 2025
**arXiv**: [2601.06039](https://arxiv.org/abs/2601.06039)
**Code**: [GitHub](https://github.com/HyouinKyoumaIRL/Operation-Veja)
**Area**: Robotics
**Keywords**: Role-playing, VEJA framework, value-conflicted reasoning, data curation, character consistency

## TL;DR

This paper systematically critiques four dominant paradigms in role-playing (RP) model training—RAG, fact-value specification, literary data, and synthetic data—arguing that none can produce characters with genuine depth. It proposes the VEJA framework (Values–Experiences–Judgments–Abilities) as a structured basis for character definition and data curation. In an LLM-judged A/B test, VEJA-guided human-curated data significantly outperforms a Gemini Pro 2.5 synthetic baseline with a win/loss/tie ratio of 43:28:29.

## Background & Motivation

**Background**: RP models have grown increasingly sophisticated, yet consistently fail to capture the essence of believable and engaging characters. Across gaming, AI companionship, and interactive narrative, users expect characters to exhibit depth, consistency, and internal conflict.

**Limitations of Prior Work**: Using Makise Kurisu from *Steins;Gate* as a running example, the authors argue that no existing technique can reproduce her defining trait—the tension between intellectual curiosity and social defensiveness. Model responses are purely reactive, never genuinely inquisitive or driven by internal conflict. The root issue is the absence of **value-conflicted reasoning** in current models.

**Key Challenge**: Human interaction is not a process of retrieving "correct" responses; it is a continuous negotiation among competing values—politeness vs. efficiency, curiosity vs. pragmatism, vulnerability vs. professionalism. Every existing training paradigm ignores this inner deliberative process.

**Goal**: What are the systematic deficiencies of the four mainstream RP training paradigms? What kind of character-definition framework is required to endow training data with sufficient character depth?

**Key Insight**: Drawing on principles of character construction from theatrical art (the Stanislavski system)—whereby characters are expressed through the interplay of goals, motivations, and past experiences—the paper translates these principles into an actionable data-curation framework for AI training.

**Core Idea**: Character depth emerges from the causal chain interaction of four dimensions—Values, Experiences, Judgments, and Abilities. Only by curating data according to the VEJA structure can the quality ceiling imposed by synthetic data be overcome.

## Method

### Overall Architecture

The paper is organized into two parts: (1) a systematic critique of the four dominant paradigms, identifying each one's fundamental deficiency; and (2) the VEJA framework, which defines four core pillars of character depth and arranges them into a causal chain: **Experiences (E) → Values (V) → Judgments (J) → Abilities (A)**. The framework guides human writers in producing high-quality training data.

### Key Designs

1. **Systematic Critique of the Four Paradigms**

   - *Function*: Explains why existing approaches fail to produce characters with genuine depth.
   - *Mechanism*: **RAG's scalability problem**—human value systems grow combinatorially and cannot be exhaustively enumerated as retrievable facts; contextual modifiers cause exponential growth in complexity. **Fact-value specification's decontextualization problem**—distilling values into isolated formulae (e.g., "open with strangers") causes models to over-index on a single trait. **The implicit-context curse of literary data**—a character's inner reasoning is only implied in dialogue rather than made explicit, and experience is conveyed through narration rather than conversation. **The chicken-and-egg problem of synthetic data**—when GPT-4 is prompted to generate dialogues exhibiting value conflict, GPT-4 itself lacks the capacity to balance competing values.
   - *Design Motivation*: Establishes the necessity of a new framework through systematic elimination.

2. **The Four Pillars of the VEJA Framework**

   - *Function*: Provides a structured, actionable conceptual framework for character definition.
   - *Mechanism*: **Values**—high-level motivations driving a character's goals and desires (e.g., "I want more time because there is so much I want to do" → ambition/productivity); **Experiences**—specific past events that shape values and judgments (e.g., "having been through a contract dispute in court, deeply distrusts verbal commitments"); **Judgments**—a character's concrete views and heuristics about the world, the output of values filtered through experience (e.g., "views marriage as an outdated institution"); **Abilities**—the character's skills, knowledge, and competency range, defining domains of authority and shaping vocabulary and interests.
   - *Design Motivation*: The four pillars form a causal chain (experiences forge values → values and experiences yield judgments → abilities provide the expressive medium), creating coherent internal logic.

3. **Data Curation Experimental Design**

   - *Function*: Preliminary validation of whether the VEJA framework improves data quality.
   - *Mechanism*: Two small-scale datasets centered on Makise Kurisu are constructed. The baseline group uses Gemini Pro 2.5 to synthesize dialogues from a 10-day timeline with 20 user-generated prompts; the VEJA group provides 15 human writers with the same timeline plus a VEJA-structured character profile. One hundred randomly paired A/B trials are evaluated by Gemini 2.5 Flash in a blind review.
   - *Design Motivation*: The authors explicitly acknowledge the confound of human vs. machine authorship, but argue that the core claim is that current models cannot synthesize data of this quality, making human curation combined with the correct framework a necessary component.

### Loss & Training

This paper does not involve model training; it focuses on the data curation framework. Evaluation employs an LLM-as-judge A/B preference test, with the criterion "which dialogue is more representative of the character."

## Key Experimental Results

### Main Results

LLM-judged A/B preference test ($N=100$, Gemini 2.5 Flash blind evaluation):

| Outcome | Count | Proportion |
|---------|-------|------------|
| VEJA human data wins | 43 | 43% |
| Synthetic baseline wins | 28 | 28% |
| Tie | 29 | 29% |

Win rate excluding ties: $43/(43+28) = \mathbf{60.6\%}$

### Ablation Study

No ablation study on individual VEJA components is conducted (the authors acknowledge this as a limitation due to time constraints).

| Analysis Dimension | Notes |
|-------------------|-------|
| Individual contribution of V/E/J/A | Not ablated; the most critical pillar cannot be determined |
| Framework effect vs. human authorship effect | Confounded: impossible to isolate VEJA's contribution from human writing quality |
| Evaluator model bias | Only Gemini 2.5 Flash used as judge; no cross-validation |

### Key Findings

- **VEJA data significantly outperforms synthetic data on narrative continuity and character consistency**: Reviewers frequently noted that VEJA dialogues demonstrated "superior narrative continuity, more nuanced responses, and a clearer connection between the character profile and the dialogue."
- **Typical deficiencies of synthetic data**: Flagged as "generic, out of character, and over-reliant on a single trait." For example, one synthetic dialogue had Kurisu speaking Chinese (which she does not speak); another contained a sexually suggestive scene entirely inconsistent with her personality.
- **VEJA's advantage is not solely attributable to human authorship**: The authors argue that the VEJA framework is the critical tool enabling human writers to maintain high consistency and quality—without structured guidance, human writers also produce inconsistent data.

## Highlights & Insights

- **Systematic problem diagnosis is more valuable than the proposed solution**: The critique of the four paradigms hits precisely at the core issues—RAG's combinatorial explosion, fact-value decontextualization, the implicit-context curse of literary data, and the recursive quality ceiling of synthetic data—each diagnosis is accurate and compelling.
- **The concept of "value-conflicted reasoning"**: Defining the essence of character depth as a negotiation among competing values is a highly insightful observation. It explains why RP models are perpetually "shallow and predictable"—they learn to "produce the correct response to a given input" rather than "make choices amid internal conflict."
- **Transferability of the causal chain design**: The E→V→J→A causal chain is applicable not only to fictional characters but also to user persona modeling, conversational system persona design, and NPC behavior tree design.

## Limitations & Future Work

- **Extremely small dataset scale**: Only one character (Makise Kurisu) is used, with 15 human writers and 20 synthetic users; statistical power is limited.
- **Human vs. framework confound**: It is impossible to disentangle the contribution of the VEJA framework from that of human authorship itself. The ideal experiment would include a "human writing without VEJA guidance" group and a "VEJA-guided synthetic generation" group.
- **Evaluation relies solely on LLM-as-judge**: The authors themselves acknowledge the irony of using an LLM to evaluate a paper about LLM capability deficiencies. Large-scale human evaluation is needed.
- **Scalability is questionable**: Human curation is costly; while the paper mentions semi-automated directions, none are concretely explored.
- **No downstream model training validation**: No model is trained on VEJA data and evaluated for RP capability; the paper remains at the level of data quality comparison.
- Future directions: multi-character validation, ablation experiments, exploration of VEJA-guided semi-automatic generation pipelines, and end-to-end evaluation with trained models.

## Related Work & Insights

- **vs. RoleLLM (Wang et al., 2024)**: RoleLLM's evaluation framework focuses on measurable "persona fidelity," but its data construction directly exposes values rather than embedding them in multifaceted scenarios—a canonical example of fact-value specification.
- **vs. COSER (Wang et al., 2025)**: COSER uses literary dialogues; although characters possess depth, the training data can only imply the reasoning process rather than make it explicit—an instance of the literary data curse.
- **vs. Persona Hub (Chan et al., 2024)**: A representative large-scale synthetic data approach, constrained by the recursive quality ceiling arising from the generator's own inability to perform conflict reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The systematic critical perspective and the VEJA framework concept are original and thought-provoking.
- Experimental Thoroughness: ⭐⭐ — Single-character small-scale trial; no ablation; no downstream training validation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous argumentation, incisive problem analysis; the use of Makise Kurisu as a throughline example is highly persuasive.
- Value: ⭐⭐⭐⭐ — The problem diagnosis offers significant value to the RP research community, though insufficient experimental validation limits broader impact.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data](himacon_discovering_hierarchical_manipulation_concepts_from_unlabeled_multi-moda.md)
- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[CVPR 2026\] Towards Training-Free Scene Text Editing](../../CVPR2026/robotics/towards_training-free_scene_text_editing.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](../../CVPR2026/robotics/quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[ICLR 2026\] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](../../ICLR2026/robotics/robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)

<!-- RELATED:END -->
