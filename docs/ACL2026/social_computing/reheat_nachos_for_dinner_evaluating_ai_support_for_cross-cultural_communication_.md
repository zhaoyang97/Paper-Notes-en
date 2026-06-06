---
title: >-
  [Paper Note] Reheat Nachos for Dinner? Evaluating AI Support for Cross-Cultural Communication of Neologisms
description: >-
  [ACL2026][Social Computing][Cross-cultural communication] Through human experiments involving 234 non-native speakers and 144 native evaluators…
tags:
  - "ACL2026"
  - "Social Computing"
  - "Cross-cultural communication"
  - "Neologisms"
  - "AI language learning"
  - "Communicative competence"
  - "User study"
date: 2026-05-08
content_hash: 2f2da7778dd71467
---

# Reheat Nachos for Dinner? Evaluating AI Support for Cross-Cultural Communication of Neologisms

**Conference**: ACL2026  
**arXiv**: [2604.23842](https://arxiv.org/abs/2604.23842)  
**Code**: https://github.com/dayeonki/crosscultural_communication  
**Area**: audio_speech  
**Keywords**: Cross-cultural communication, Neologisms, AI language learning, Communicative competence, User study

## TL;DR
Through human experiments involving 234 non-native speakers and 144 native evaluators, this study compares four categories of AI/non-AI support. The results show that "AI Explanation" with contextual details most effectively improves native speaker ratings of non-native speakers' usage of English neologisms. However, a significant misalignment persists between learners' self-confidence and their actual communicative competence.

## Background & Motivation
**Background**: Neologisms and emerging slang have become integral to daily English communication. Expressions such as *main character energy*, *delulu*, and *reheat nachos* often carry specific community contexts, tones, and cultural identities. Non-native speakers (NNS) increasingly rely on AI tools for definitions, rephrasing, or usage explanations during cross-cultural communication.

**Limitations of Prior Work**: Traditional dictionaries and textbooks update slowly, failing to cover rapidly evolving internet slang in real-time. Furthermore, existing AI evaluations often focus on multiple-choice questions, translation, or static comprehension tasks, which do not reflect how real users learn, compose messages, or judge contextual appropriateness after encountering a new word.

**Key Challenge**: Learning neologisms involves more than literal definitions; it requires understanding when, to whom, and in what tone they should be used. While AI outputs may be fluent, if the explanations lack context or contain errors, NNS often lack the mechanisms to judge reliability, leading them to mistake the "feeling of having learned" for "natural usage by native speakers."

**Goal**: The authors decompose the problem into three levels: whether different support types help NNS write more naturally; whether NNS self-assessments can serve as proxies for native speaker (NS) evaluations; and whether AI support can bridge the communicative competence gap between NNS and NS.

**Key Insight**: The paper selects a scenario closely reflecting real-world usage: NNS encounter a neologism in a social media post, use specific support to learn it, write a message to a hypothetical native friend "Jo," and subsequently judge the appropriateness of others' usage. This design is closer to the process of "acquiring a word and applying it in conversation" than offline Q&A.

**Core Idea**: Evaluating AI neologism support through authentic cross-cultural communication tasks and NS ratings, rather than relying solely on the model's or the learner's own perception of utility.

## Method
This study does not propose a new model but designs a human experiment with high ecological validity to compare how different AI support types assist in learning and using neologisms. The experiment breaks "learning a new word" into four stages: learning, production, comprehension, and external evaluation, while simultaneously collecting NNS self-assessments and NS ratings to observe the translation of AI support into communicative competence.

### Overall Architecture
The input consists of a set of English neologisms with social media contexts. Participants are NNS with Spanish, German, or Chinese as their L1. Each participant is randomly assigned to a support condition and performs three steps across eight neologisms: first, learning the term via posts and support materials; second, writing a scenario and a message to a hypothetical friend (Jo); and third, judging the appropriateness of neologism usage in two provided writing samples. Subsequently, each NNS writing sample is rated by two US-based native English speakers across dimensions such as grammar/coherence, contextual appropriateness, and understandability.

### Key Designs
1. **Communication-oriented Task Design**:
	- **Function**: Moves neologism evaluation from "knowing a definition" to "natural usage in social messages."
	- **Mechanism**: Each neologism is placed within a realistic social media post. NNS learn the term, write a message to Jo, and evaluate the contextual appropriateness of other samples. Production tasks measure usage capability, while comprehension tasks measure the ability to judge others' usage.
	- **Design Motivation**: The difficulty of neologisms lies in pragmatics and community context. Simple MCQs cannot measure "naturalness to a native speaker." This design shifts the evaluation target from model answers to actual communicative outcomes.

2. **Controlled Comparison of Five Support Conditions**:
	- **Function**: Distinguishes exactly what kind of help different AI usage modalities provide.
	- **Mechanism**: The experiment includes Control, AI Definition, AI Rewrite, AI Explanation, and Non-AI Dictionary groups. AI Definition provides dictionary-style definitions; AI Rewrite simplifies the original post; AI Explanation uses 3-5 sentences to explain meaning, tone, usage scenarios, and audience; Non-AI Dictionary provides full Merriam-Webster pages.
	- **Design Motivation**: Real users don't just ask AI "what does this mean"; they ask for explanations, rewrites, or examples. Comparing these interaction modes reveals whether low-density definitions, simplified contexts, or contextual usage instructions are most effective.

3. **Multi-perspective Communicative Competence Measurement**:
	- **Function**: Simultaneously observes external communicative effects, learner comprehension, and subjective perception.
	- **Mechanism**: NS ratings cover well-formedness, contextual appropriateness, and understandability. NNS comprehension is measured by the distance between their appropriateness ratings and the NS baseline. Self-assessments include confidence, helpfulness, reliance, future trust, mental burden, and task difficulty.
	- **Design Motivation**: AI tools often make users "feel smarter," but this feeling may not equate to real competence. Multi-perspective measurement explicitly exposes the misalignment where "confidence increases but native speakers are unimpressed."

### Loss & Training
No new models were trained; AI support materials were generated by GPT-4.0 using predefined prompts. Statistical analysis utilized Linear Mixed-Effects (LME) models. Fixed effects included support condition, language group, their interaction, social media usage frequency, and initial term familiarity. Random intercepts included participants, native evaluators, and neologisms. Significance is reported using Bonferroni-corrected estimated marginal effects (EME) with confidence intervals.

## Key Experimental Results

### Main Results
NS ratings demonstrate that AI Explanation is the only support type that consistently outperforms the Control across all primary communicative dimensions. Non-AI Dictionary is the most comprehensive but only shows significant improvement in some dimensions due to its high information density and cognitive load.

| Condition | Well-formedness | Contextual appropriateness | Understandability | Confidence-related rating |
|------|-----------------|----------------------------|-------------------|--------------|
| Control | 7.05 | 6.44 | 7.17 | 4.17 |
| AI Definition | 7.32 | 6.93 | 7.50 | 4.23 |
| AI Rewrite | 7.42 | 7.06 | 7.62 | 4.24 |
| AI Explanation | 7.74 | 7.43 | 7.98 | 4.23 |
| Non-AI Dictionary | 7.36 | 7.28 | 7.78 | 4.23 |

### Ablation Study
The paper performs an "interaction ablation" by comparing support forms. Results show that learners trust the Non-AI Dictionary most and find AI Rewrite and AI Explanation helpful; however, the subjective gains from AI Rewrite do not consistently translate into higher NS ratings.

| Support Type | Confidence ↑ | Reliance ↑ | Trust ↑ | Mental Burden ↓ | Task Difficulty ↓ |
|----------|--------------|------------|---------|-----------------|-------------------|
| AI Definition | 3.42 | 2.74 | 3.56 | 3.42 | 3.66 |
| AI Rewrite | 4.02 | 3.60 | 3.67 | 3.31 | 3.51 |
| AI Explanation | 4.04 | 3.40 | 3.88 | 3.44 | 3.46 |
| Non-AI Dictionary | 4.51 | 4.28 | 4.44 | 3.77 | 3.65 |

### Key Findings
- The core advantage of AI Explanation lies in providing context, tone, audience, and usage boundaries, making it more effective than short definitions in helping NNS write messages perceived as natural by NS.
- NNS comprehension judgments did not significantly improve across conditions; the average distance to the NS baseline remained at 2.22, indicating that "writing better" does not equal "reliably judging others' usage."
- NNS self-assessment is not a reliable proxy: AI Rewrite reduced mental burden and increased confidence, but NS ratings did not show a corresponding significant improvement.
- Writing samples from NS remain significantly superior to most NNS conditions, suggesting that single-turn AI support cannot fully bridge the gap in community context and natural expression.

## Highlights & Insights
- The most valuable design is anchoring AI language learning evaluation to "native speaker judgment." This reminds us that cross-cultural communication tools should optimize for naturalness and appropriateness in the eyes of the receiver, not just the fluency of the explanation.
- The paper reveals a key risk in neologism learning: incorrect or overly narrow AI definitions can induce literal usage. For example, when *reheat nachos* was explained as literally heating food, participants wrote messages that completely deviated from the slang meaning.
- The success of AI Explanation indicates that lightweight but contextually dense instructions are better suited for immediate communication than full dictionary pages. Future language learning products could adopt a fixed output structure: "common scenarios, tone, audience, and counter-examples."
- The discovery of the misalignment between self-assessment and external evaluation can be generalized to writing assistants, translation, and cross-cultural emailing: just because a user feels the AI helped doesn't mean the target reader finds it easier to understand.

## Limitations & Future Work
- The experiment only covers English neologisms and NNS from Spanish, German, and Chinese backgrounds; conclusions may not generalize to other language pairs or formal communication settings.
- AI support was single-turn and in English; it did not test multi-turn follow-ups, L1 explanations, RAG-based support, or integration with real social platform data.
- NS evaluation, while closer to reality than automated metrics, is still offline rating and not equivalent to interactive feedback between real friends.
- Future work could incorporate "minimal contrastive pairs" and "common misuse alerts" so that AI does not just explain correct usage but explicitly warns against awkward or misunderstood phrasing.

## Related Work & Insights
- **vs. Traditional Neologism Benchmarks**: While traditional work relies on MCQs or model-based comprehension, this work assesses whether NNS can use neologisms in messages, providing higher ecological validity.
- **vs. Dictionary-based Learning**: Dictionaries provide authoritative definitions but are high-density and burdensome for immediate tasks; AI Explanation provides pragmatic cues with less text, proving more effective for communication tasks.
- **vs. LLM-as-judge Evaluation**: The appendix finds that AI ratings are generally higher than NS ratings, suggesting that automated evaluation may overestimate NNS writing quality. Human receiver perspectives remain essential for cross-cultural communication tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Using human communication experiments to evaluate AI neologism support is insightful, though the model methodology itself is not the primary innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Sample size, control conditions, and NS ratings are solid, though the range of languages and scenarios remains narrow.
- **Writing Quality**: ⭐⭐⭐⭐☆ Research questions, experimental flow, and discussion are clear, with comprehensive supplementary data.
- **Value**: ⭐⭐⭐⭐☆ Directly valuable for AI language learning, cross-cultural communication assistance, and user confidence calibration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies](imperfectly_cooperative_human-ai_interactions_comparing_the_impacts_of_human_and.md)
- [\[NeurIPS 2025\] Evaluating Multiple Models Using Labeled and Unlabeled Data](../../NeurIPS2025/social_computing/evaluating_multiple_models_using_labeled_and_unlabeled_data.md)
- [\[NeurIPS 2025\] Policy-as-Prompt: Turning AI Governance Rules into Guardrails for AI Agents](../../NeurIPS2025/social_computing/policy-as-prompt_turning_ai_governance_rules_into_guardrails_for_ai_agents.md)
- [\[AAAI 2026\] Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition](../../AAAI2026/social_computing/cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)

</div>

<!-- RELATED:END -->
