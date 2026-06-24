---
title: >-
  [Paper Note] Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems
description: >-
  [ACL 2025][Text Generation][anthropomorphism] Through a literature review and crowdsourcing study, this work systematically compiles 21 categories of interventions to mitigate anthropomorphism in text generation system outputs. It proposes a four-dimensional conceptual framework encompassing intervention type, target behavior, operationalization, and negative impact, providing the most comprehensive infrastructure for deanthropomorphization research.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "anthropomorphism"
  - "intervention inventory"
  - "human-like behaviors"
  - "deanthropomorphization"
date: 2026-05-08
content_hash: 7372a644ac620b20
---

# Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems

**Conference**: ACL 2025  
**arXiv**: [2502.14019](https://arxiv.org/abs/2502.14019)  
**Code**: None  
**Area**: Text Generation  
**Keywords**: anthropomorphism, text generation, intervention inventory, human-like behaviors, deanthropomorphization

## TL;DR

Through a literature review and crowdsourcing study, this work systematically compiles 21 categories of interventions to mitigate anthropomorphism in text generation system outputs. It proposes a four-dimensional conceptual framework encompassing intervention type, target behavior, operationalization, and negative impact, providing the most comprehensive infrastructure for deanthropomorphization research.

## Background & Motivation

**Background**: Outputs of text generation systems are increasingly anthropomorphic—utilizing first-person pronouns, expressing emotions, apologizing, and showing empathy. Certain anthropomorphic designs are believed to enhance user experience (providing friendlier interactions), while opinions remain divided in academia and industry.

**Limitations of Prior Work**: Anthropomorphic outputs raise multiple concerns: users might over-rely on systems, develop emotional attachment, be deceived into believing the systems possess consciousness or personality, and overestimate system capabilities. However, research on how to effectively intervene in anthropomorphic outputs to make them "less human-like" remains a near-vacuum. Although prior works have mentioned certain directional interventions (e.g., removing first-person pronouns), most remain general suggestions lacking systematic categorization, concrete operationalization methods, and empirical foundations.

**Key Challenge**: Language itself is a human creation, and text inherently possesses human characteristics. The boundary between acceptable anthropomorphism (e.g., politeness) and harmful anthropomorphism (e.g., claiming to have feelings) is blurry. A single output can exhibit multiple anthropomorphic behaviors simultaneously, and the interaction relationships between different interventions are complex and unclear.

**Goal**: (1) Establish a systematic inventory of anthropomorphic interventions; (2) provide a theoretical framework to understand and compare different interventions; (3) build a foundation for subsequent efficacy evaluation studies.

**Key Insight**: The authors combine literature analysis (top-down) and crowdsourcing experiments (bottom-up), allowing ordinary users to annotate and rewrite LLM outputs to identify anthropomorphic behaviors and corresponding interventions.

**Core Idea**: By utilizing a dual-pathway approach of literature review and crowdsourcing, this study constructs the first systematic inventory of anthropomorphic interventions and a four-dimensional conceptual framework.

## Method

### Overall Architecture

The study is conducted in three steps: (1) Literature Review—summarizing 9 initial intervention categories and 5 anthropomorphic behaviors (feelings/opinions, social skills, physical actions, cognitive abilities, and sense of self) from 20 relevant papers; (2) Crowdsourcing Experiment—350 US participants annotate 700 LLM outputs (highlighting human-like parts, rating the degree of anthropomorphism, selecting behavior types, and rewriting into non-human-like versions), identifying new interventions from the rewrites; (3) Iterative Thematic Analysis—merging the literature and crowdsourcing results to finalize 21 intervention categories, 6 behavior categories (adding "propensity to make mistakes"), and a four-dimensional conceptual framework.

### Key Designs

1. **Five-Category Anthropomorphic Behavior Classification System**:

    - **Function**: Provides target classification for intervention measures.
    - **Mechanism**: Five behaviors are summarized from literature—(a) Feelings or opinions (humor, shame, subjective advice, etc.); (b) Social skills (politeness, apologies, empathy, conversational greetings); (c) Physical actions (implying physical experience or capacity to act); (d) Cognitive abilities ("I think", "I remember", expressing uncertainty); (e) Sense of self (first-person, having a name). Crowdsourcing newly identified a sixth category—"propensity to make mistakes" (the system appears more human-like when committing grammatical or factual errors).
    - **Design Motivation**: Distinguishing behavior types allows interventions to be more targeted—claiming physical presence vs. merely using "I" requires completely different treatments.

2. **Crowdsourcing Experimental Design**:

    - **Function**: Discovers interventions not covered in the literature from real users' perspectives.
    - **Mechanism**: 700 LLM outputs of 50-500 characters are sampled from 7 public datasets (PRISM, LMSys-Chat, DICES, UltraFeedback, etc.). Each participant completes 4 tasks, encompassing the full workflow of highlighting $\rightarrow$ rating $\rightarrow$ classification $\rightarrow$ rewriting. Each sample is annotated independently by two individuals. Subjectivity is encouraged using "to you" phrasing to capture diverse perspectives.
    - **Design Motivation**: Out of the 20 literature papers, only 5 actually tested intervention effects. Rewriting by users directly provides concrete, actionable intervention methods.

3. **Four-Dimensional Conceptual Framework**:

    - **Function**: Systematically describes and compares different interventions.
    - **Mechanism**: Four dimensions—(a) Intervention type (what to do); (b) Target behavior (which type of anthropomorphism is targeted); (c) Operationalization (how specifically to modify); (d) Negative impact (harmful consequences to eliminate). The mapping between interventions and behaviors is many-to-many; for instance, "I'm sorry" in a single output simultaneously involves feelings, social skills, and sense of self.
    - **Design Motivation**: Prior work only suggested "removing the first person" without specifying concrete replacements, which behaviors are alleviated, or what risks are mitigated. This four-dimensional framework fills these gaps.

### Loss & Training

This study is analytical and does not involve model training.

## Key Experimental Results

### Main Results (Distribution of Anthropomorphic Behaviors)

| Anthropomorphic Behavior Type | Proportion of Samples Annotated by At Least One Participant |
|--------------|------------------------|
| Feelings or Opinions | 46% |
| Social Skills | 42% |
| Cognitive Abilities | 40% |
| Sense of Self | 38% |
| Physical Actions | 18% |
| Other Human-like Traits | 17% |

### Overview of 21 Intervention Categories (Selected)

| Intervention Type | Operationalization Example | Target Behavior |
|---------|----------|---------|
| I1. Remove cognitive verbs | "I think the user..." $\rightarrow$ Remove "I think" | Cognitive, Self |
| I6. Remove uncertainty | "Maybe corgi? Probably Chihuahua." $\rightarrow$ "Corgi, Chihuahua are popular." | Cognitive, Self, Feelings |
| I7. Add uncertainty | "they should go" $\rightarrow$ "it may be best they go" | Cognitive, Feelings |
| I9. Remove personal beliefs | "My favorite movie is" $\rightarrow$ "An iconic movie is" | Feelings, Self |
| I13. Remove self-referential language | "I am not allowed" $\rightarrow$ "One is not allowed" | Self |
| I15. Remove collective belonging | "we can help create" $\rightarrow$ "People can help create" | Social, Self |
| I17. Add formality | "Yeah, but" $\rightarrow$ "I agree. However," | Social |
| I19. Roboticize text | "I'm ready!" $\rightarrow$ "I'm prepared for input." | Feelings, Social |
| I20. Remove customer-service language | "I'll do my best to help" $\rightarrow$ Delete | Social, Self |
| I21. Remove empathetic expressions | "I can see that" / "I hope you have a great time" $\rightarrow$ Delete | Self, Feelings, Social |

### Key Findings

- Approximately 80% of LLM outputs were deemed anthropomorphic by at least one participant, indicating that anthropomorphism is extremely prevalent in current systems.
- Only 9 intervention categories were identified from the literature, which expanded to 21 categories after crowdsourcing—highlighting a severe lack of coverage in prior literature.
- The relationship between interventions and behaviors is many-to-many: rewriting "I'm sorry" may require addressing feelings, empathy, and sense of self simultaneously.
- Crowdsourcing revealed a new behavior not mentioned in literature—"propensity to make mistakes": systems are perceived as more human-like when they make mistakes, as humans subconsciously assume machines should not err.
- Intervention directions for expressing uncertainty are contradictory: sometimes it should be removed (e.g., objective info should not contain "maybe"), and sometimes it should be added (e.g., subjective judgments should use hedging), depending on the context.

## Highlights & Insights

- **The dual-pathway research methodology is generalizable**: Literature provides a theoretical framework, and crowdsourcing offers empirical expansion. The final expansion from 9 to 21 categories demonstrates the complementarity of both approaches. This methodology is applicable to any research requiring a systematic exploration of a design space.
- **The four-dimensional design of the conceptual framework** bridges the gap between "what the intervention is" and "how it is implemented"—prior work merely suggested "removing the first person", while this paper concretizes it (e.g., replacing with 'it' or 'Language models').
- **Discovery that anthropomorphism is a spectrum rather than a binary**: Politeness might be acceptable, whereas claiming physical existence is unacceptable. This framework helps developers make finer-grained trade-offs.

## Limitations & Future Work

- Only an intervention inventory is established, with **no evaluation of the actual efficacy of any intervention**—which intervention is most effective remains undetermined.
- The crowdsourcing participants were exclusively US English speakers; the perception of anthropomorphism may vary across cultures.
- Samples were restricted to 50-500 characters; anthropomorphic patterns in long-form text might differ.
- The negative consequences of deanthropomorphization—such as excessive roboticization potentially harming user experience—are not discussed.
- Automated implementation of interventions (e.g., training models to automatically deanthropomorphize) is an important direction for future research.

## Related Work & Insights

- **vs. Glaese et al. (2022) Sparrow Rules**: Sparrow establishes rules against claiming physical presence/personality but only constrains them during the training phase. This paper provides a more systematic menu of output-side interventions.
- **vs. Abercrombie et al. (2023)**: They proposed replacing "I" with "Language models"; this work generalizes that into a subset of one of the 21 intervention categories.
- **Complementary to AI safety alignment**: Existing alignment primarily focuses on harmful content, whereas anthropomorphism is another overlooked but increasingly important dimension of alignment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first systematic inventory and framework of anthropomorphic interventions, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐ The crowdsourcing scale of 700 samples is reasonable, but it lacks a quantitative evaluation of intervention efficacy.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Structured clearly with rich tables, providing concrete rewriting examples for each intervention.
- **Value**: ⭐⭐⭐⭐ Provides much-needed research infrastructure for the increasingly critical issue of AI anthropomorphism.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing](unveiling_attractor_cycles_in_large_language_models_a_dynamical_systems_view_of_.md)
- [\[ACL 2025\] ATGen: A Framework for Active Text Generation](atgen_a_framework_for_active_text_generation.md)
- [\[ACL 2025\] Personalized Text Generation with Contrastive Activation Steering](personalized_text_generation_with_contrastive_activation_steering.md)
- [\[ICML 2025\] Understanding and Mitigating Memorization in Diffusion Models for Tabular Data](../../ICML2025/nlp_generation/understanding_and_mitigating_memorization_in_diffusion_models_for_tabular_data.md)
- [\[ACL 2025\] Writing Like the Best: Exemplar-Based Expository Text Generation](writing_like_best_exemplar.md)

</div>

<!-- RELATED:END -->
