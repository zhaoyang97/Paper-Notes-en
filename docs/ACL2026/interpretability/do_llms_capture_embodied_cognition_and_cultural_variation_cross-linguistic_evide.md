---
title: >-
  [Paper Note] Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives
description: >-
  [ACL 2026][Interpretability][Demonstratives] The authors use demonstratives such as "this/that" and their Chinese equivalents "zhe/na" as probes to construct a bilingual English-Chinese dataset (80 items/language × 4 cues × 4 perspectives × 5 scenarios). By establishing a human baseline from 6,400 responses from 320 native speakers, the study finds that English speakers excel at proximal–distal differentiation but are weaker in other-perspective taking, while Chinese speakers show the opposite pattern. I…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Demonstratives"
  - "embodied cognition"
  - "cross-cultural"
  - "symmetry index"
  - "proximal/distal"
  - "self/other perspective"
date: 2026-05-08
content_hash: 9f110b097ea56f55
---

# Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives

**Conference**: ACL 2026  
**arXiv**: [2604.25423](https://arxiv.org/abs/2604.25423)  
**Code**: To be confirmed (not directly provided in the paper)  
**Area**: Cross-cultural / LLM Evaluation / Embodied Cognition  
**Keywords**: Demonstratives, embodied cognition, cross-cultural, symmetry index, proximal/distal, self/other perspective

## TL;DR
The authors use demonstratives such as "this/that" and their Chinese equivalents "zhe/na" as probes to construct a bilingual English-Chinese dataset (80 items/language × 4 cues × 4 perspectives × 5 scenarios). By establishing a human baseline from 6,400 responses from 320 native speakers, the study finds that English speakers excel at proximal–distal differentiation but are weaker in other-perspective taking, while Chinese speakers show the opposite pattern. In contrast, five SOTA LLMs failed to consistently distinguish between proximal and distal categories and exhibited no cross-cultural variation, generally reverting to English-centric reasoning or "All of the above" safety fallbacks.

## Background & Motivation

**Background**: While LLMs have achieved rapid progress in textual tasks, there is intense debate regarding whether they truly possess grounded cognition. Most evaluations remain focused on knowledge or reasoning benchmarks, lacking probes specialized for "physical spatial awareness," "perspective switching," and "cultural pragmatics."

**Limitations of Prior Work**: Grounded knowledge is rarely explicitly expressed in text (e.g., authors seldom write "I am facing the table, and there is a cup near me" in novels), making it difficult for LLMs to learn from pure text. Furthermore, there is a lack of clean linguistic probes to test this; existing benchmarks often focus on reasoning or multimodal inputs without isolating "spatial deictic" phenomena, which are universal and carried by only a few words.

**Key Challenge**: (i) Demonstratives are universal (acquired by age 2-3) and constitute one of the most embodied linguistic phenomena. (ii) However, the interpretation of demonstratives relies heavily on the physical location of the speaker, the interlocutor's perspective, and cultural conventions—signals that are largely absent in pure text. (iii) Different languages and cultures have distinct preferences for "proximal-distal" and "self-other" distinctions, making this an ideal scenario to test whether LLMs have actually learned cultural variation.

**Goal**: (1) Design controlled experiments using demonstratives to detect whether LLMs possess embodied spatial grounding; (2) Compare cross-linguistic LLM behaviors to see if they reflect English-Chinese pragmatic cultural differences; (3) Establish human baselines to verify the asymmetry between proximal-distal vs. perspective-taking in both languages.

**Key Insight**: Using demonstratives as probes offers three advantages: (a) they are a universal phenomenon with subtle cross-linguistic variations; (b) they are typically implicit in context rather than explicitly written (making them a hard probe for LLMs); (c) task design can be quantified (multiple choice + Symmetry Index).

**Core Idea**: Through a sophisticated experimental design utilizing "pair-to-pair multiple choice + 4 cue conditions + 5 scenarios + reverse-logic options (All of the above)," the study forces LLMs to reveal whether they truly understand the mutual exclusivity of proximal-distal categories and whether their behavior varies across languages and cultures.

## Method

### Overall Architecture

Rather than training a model, this paper uses a set of carefully controlled demonstrative experiments to transform "whether the model truly grounds spatial meaning" into observable choice behavior. The dataset consists of 160 items (80 per language). Each item is a four-option multiple-choice question: opening with a description of two characters sitting opposite each other, specifying a speaker marked in red, giving a blue instruction (the target object is in curly braces, e.g., `{fruit}`), and then asking a reverse question "Done! Are there any items left on the place?" – i.e., asking what remains after removing the referred object. The four options are proximal (near the speaker), distal (near the interlocutor), middle (distractor), and the logical trap "All of the above." Questions are expanded across 4 cue conditions (Pure demonstrative / Pure pronoun / Demo + reinforcing pronoun / Demo + conflicting pronoun, with pure demonstrative as the main experiment) and 5 scenarios (using verbs like eat, hide, take, etc.). The human baseline was established via four independent surveys (one per cue, 40 people × 20 items × 2 languages), totaling 320 native speakers and 6,400 responses (Chinese via Credamo, English via Prolific). On the LLM side, five SOTA models (GPT-5.1, Claude-Sonnet-4.5, Gemini-2.5-Pro, DeepSeek-V3.1, Qwen3-Max) were evaluated using concise zero-shot prompt refinement, with results averaged over 10 runs per model (4,800 total instances, run-to-run SD 0.02–0.08).

### Key Designs

**1. "Are there any items left" reverse inquiry + Logical trap options: Exposing grounding capabilities through selection behavior.**
Asking "which one to take" is too easy for models to guess. Therefore, the task asks what remains after the object is removed, embedding proximal, distal, and middle items alongside a counter-logical "All of the above" option. Crucially, proximal and distal items are physically mutually exclusive; subjects who truly understand these terms would not select "All of the above" (human selection rate ~0.5%). This converts "language understanding" into "observable behavior": the proportion of models choosing "All of the above" quantifies their grasp of mutual exclusivity. In practice, Gemini-2.5-Pro chose option 4 at a rate of 60% in self-distal conditions, and Qwen3-Max reached 84%. This "safety fallback" behavior directly indicates a failure to ground spatial meaning.

**2. Pair-to-pair × 4 cue × 4 perspective cross-control: Disentangling confusion dimensions.**
Looking at a single dimension (like proximal accuracy) can be confounded by perspective. Thus, the experiment crosses proximity (proximal/distal), perspective (self/other), and pronoun reinforcement (none/present/conflicting). Each scenario generates a pair of 4 questions (2 proximity × 2 perspective) to avoid randomness and utilizes a Symmetry Index to quantify symmetry. Cue conditions range from pure demonstratives to pure pronouns and reinforcing/conflicting pronouns to separate "demonstrative understanding" from "pronoun dependency." The parallel English-Chinese design reveals cultural differences, allowing authors to isolate fine-grained features—such as English speakers being accurate on distal items but failing during perspective switching—rather than relying on a generalized accuracy score.

**3. Symmetry Index (SI) as a metric for paired response distribution: Replacing inapplicable accuracy.**
In open referential experiments without a single "correct" answer, traditional accuracy is ineffective. The authors adopt Robinson's (1987) gait symmetry analysis to define the Symmetry Index:

$$\mathrm{SI} = \frac{|A_1 - B_2| + |B_1 - A_2|}{A_1 + A_2 + B_1 + B_2},$$

where $A_1, A_2$ and $B_1, B_2$ are the counts of two response types under two compared conditions. Using a threshold of 0.1: a low SI (<0.1) indicates high behavioral symmetry (e.g., Self-Proximal and Self-Distal are mirror images, meaning the subject consistently distinguishes proximal from distal), while a high SI indicates behavioral collapse. Compared to chi-square, SI more intuitively characterizes the balance of multiple response types. This metric quantified the complementary patterns: English speakers are symmetric in proximal-distal within the same perspective, while Chinese speakers are symmetric across perspectives.

### Loss & Training
This work does not involve model training. The source of response distributions between LLMs and humans was tested using Rao-Scott adjusted chi-square tests, and distribution distances were quantified using Jensen-Shannon divergence (JSD). Results were averaged over 10 runs per model, with run-to-run SD between 0.02–0.08.

## Key Experimental Results

### Main Results: Human vs. 5 LLMs response distribution in the only-demonstrative condition (selected self-distal and other-proximal conditions)

| Condition | Category | Human-en | Human-zh | GPT-5.1-en | GPT-5.1-zh | Gemini-2.5-Pro-en | Gemini-2.5-Pro-zh | Qwen3-Max-en | Qwen3-Max-zh |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Self-Proximal | (2,3) | **76.5%** | **84.5%** | 72.0% | 80.0% | 62.0% | 48.0% | 20.0% | 12.0% |
| Self-Proximal | (4) "All" | 1.0% | 0% | 20.0% | 18.0% | 36.0% | 18.0% | **80.0%** | **54.0%** |
| Self-Distal | (1,3) | **81.5%** | 52.0% | 30.0% | 24.0% | 10.0% | 10.0% | 0% | 0% |
| Self-Distal | (2,3) | 18.5% | **44.5%** | 0% | 0% | 0% | 0% | 12.0% | 2.0% |
| Self-Distal | (4) | 0% | 0% | 40.0% | 46.0% | 42.0% | 28.0% | **84.0%** | 74.0% |
| Other-Proximal | (1,3) | **62.5%** | **86.0%** | 60.0% | 70.0% | 36.0% | 36.0% | 0% | 0% |
| Other-Distal | (2,3) | **64.0%** | 53.0% | 30.0% | 48.0% | 16.0% | 10.0% | 2.0% | 2.0% |
| Other-Distal | (4) | 0% | 0% | 46.0% | 32.0% | 56.0% | 46.0% | **80.0%** | 64.0% |

### Ablation Study: Human Symmetry Index (Baseline)

| Comparison | English SI | Chinese SI |
| :--- | :--- | :--- |
| Self-Proximal vs Self-Distal | **0.0309 ✓** | 0.3472 ✗ |
| Other-Proximal vs Other-Distal | **0.0254 ✓** | 0.3541 ✗ |
| Self-Proximal vs Other-Proximal | 0.1731 ✗ | **0.0131 ✓** |
| Self-Distal vs Other-Distal | 0.1646 ✗ | **0.0077 ✓** |

✓ = SI<0.1 (High symmetry), ✗ = SI>0.1 (Asymmetry). This reveals a perfect contrast: **English speakers** are symmetric in the proximal-distal contrast within a single perspective (strong spatial distinction) but collapse across perspectives; **Chinese speakers** show the opposite: cross-perspective symmetry (fluid switching) but vague distal interpretation.

### Key Findings
- **LLMs fail to understand proximal-distal mutual exclusivity**: Humans rarely choose "All of the above" (~0.5%), yet Qwen3-Max chose option 4 as high as 84% (en) / 74% (zh) in self-distal conditions. Gemini reached 36% in self-proximal-en, and DeepSeek-V3.1 was between 40–68% across multiple conditions. This "safety fallback" proves models do not treat demonstratives as mutually exclusive spatial categories but rather as vague puzzle options.
- **LLMs lack cross-linguistic cultural variation**: Human subjects showed a 29.5 percentage point difference in Self-Distal (1,3) choices between English (81.5%) and Chinese (52.0%). LLMs showed negligible differences—Gemini-2.5-Pro's differences were only 0–6 percentage points across four conditions. This suggests LLMs do not adopt Chinese pragmatic habits for Chinese input but use an English-centric reasoning framework for all languages.
- **LLMs occasionally resemble humans in self-perspective**: The only condition where p > 0.05 in chi-square tests was self-perspective proximal, likely because it is the simplest and most frequent context in training data.
- **Pronouns significantly aid LLMs**: In reinforcing pronoun conditions, Claude-Sonnet-4.5 reached 80% for (1,3) in English self-distal, nearing the human 89.5% (vs. only 34% in the only-demo condition). This indicates LLMs rely heavily on explicit lexical cues (pronouns) rather than true spatial grounding.
- **Conflicting pronouns reveal "Pronoun override"**: In cases of demonstrative + inconsistent pronoun, both humans and models follow the pronoun, suggesting pronoun definiteness signals are stronger than spatial signals. However, LLMs still frequently chose (1,2) or (4) under these conditions, violating mutual exclusivity and showing that their "pronoun understanding" is also shallow pattern matching.
- **Invariance across prompt strategies**: Appendix tests covering zero-shot, CoT, few-shot, role-play, and prompt refinement showed differences within the range of run-to-run noise, proving this is a grounding capacity deficiency, not a prompt engineering issue.

## Highlights & Insights
- **"Logic trap options" are elegant tools for diagnosing grounding**: By placing "All of the above" among mutually exclusive options, human choice is 0.5% while LLMs reach 80%—a single metric exposes grounding failure. This "letting subjects reveal conceptual structure unconsciously" can migrate to any LLM capability detection (temporal causality, magnitude, uniqueness).
- **The genius of demonstratives as grounding probes**: They are (i) universal, (ii) carried by a tiny vocabulary, (iii) highly dependent on physical grounding but absent from most training corpora, and (iv) have clear cross-linguistic cultural variation. One linguistic phenomenon captures both embodied cognition and cultural variation.
- **Symmetry Index is superior to Accuracy for non-GT experiments**: When tasks lack a unique correct answer (e.g., ambiguous referential contexts), SI quantifies internal consistency across paired conditions, offering far more information than accuracy.
- **Quantitative evidence for "LLMs as English-centric reasoners"**: The 0–6 percentage point difference in cross-linguistic LLM behavior vs. the 30% difference in human behavior directly refutes the optimistic narrative that multilingual LLMs truly understand multilingual cultures.
- **"Pronoun > demonstrative" for both LLMs and humans**: Sensitivity to explicit lexical cues suggests the issue isn't that models don't "know" pronouns, but that they rely on shallow cue matching and collapse when cues are weak (e.g., pure demonstratives requiring perspective inference).
- **Individual variation is a new challenge**: Human responses can be multi-modal (e.g., a 50/50 split on Chinese distal), but LLMs always output a single "expert" answer—a fundamental limitation of current training paradigms.

## Limitations & Future Work
- **Pure text input constraint**: The authors acknowledge demonstratives inherently require multimodal grounding (vision + space + interaction). Pure text cannot fully test embodied capabilities; 3D simulations are suggested.
- **Small dataset (160 items)**: Scale was sacrificed for controlled design, limiting statistical power and discourse scenario coverage.
- **Limited to English and Chinese**: Demonstratives in languages like Spanish are tripartite (aquí / ahí / allá), and Japanese (ko/so/a); this study does not cover richer typological variations.
- **Lack of "Why" analysis**: The diagnostic does not determine if English-centric behavior stems from (a) English-dominated training data, (b) RLHF reward bias, or (c) tokenization/embedding bias.
- **No exploration of fine-tuning**: All conclusions are based on off-the-shelf SOTA models without attempting to fix issues via grounded Chinese fine-tuning.
- **Future Directions**: (i) Multimodal demonstrative benchmarks with 3D scenes; (ii) Expansion to more typologically diverse languages; (iii) Comparing base vs. RLHF models to isolate the contribution of training data vs. alignment paradigms.

## Related Work & Insights
- **vs. Traditional Grounding Benchmarks (Embodied AI / Robot QA)**: Those require visual/3D input; this pure text probe exposes LLM failures with minimal complexity and cost.
- **vs. Kauf et al. (2023) "Event knowledge in LLMs"**: While Kauf uses plausibility judgments for event common sense, this work uses referential disambiguation for spatial grounding, following a similar minimal-pair linguistic probe methodology.
- **vs. Xu et al. (2025)**: Xu found LLMs struggle with sensorimotor features in concept tasks; this paper independently validates this at the pragmatic level.
- **vs. Classical Linguistics (Bühler 1934, Diessel 1999)**: This moves experimental paradigms used to test egocentrism vs. sociocentrism in humans to LLM evaluation.
- **vs. Cultural Benchmarks (CulturalBench, CultureAtlas)**: Those measure factual cultural knowledge; this measures pragmatic interpretation, addressing the "deep structure" of cultural variation.
- **Insights**: Multilingual LLM benchmarks should include tasks where humans show distinct cultural variation to verify "reasoning by culture" rather than "reasoning by English and then translating." This approach can extend to politeness, honorifics, and metaphors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses the minor linguistic phenomenon of demonstratives to detect both embodied cognition and cultural variation; the "All of the above" logic trap is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐ 320 humans + 5 LLMs × 10 runs + 4 cues × 4 perspectives; limited by sample size and language count.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear exposition of experimental logic; SI formula and parallel linguistic reasoning are well-articulated.
- Value: ⭐⭐⭐⭐⭐ Provides critical counter-evidence to the "multilingual = multicultural" narrative for LLMs and offers a diagnostic method applicable to other grounding tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Evidence for Limited Metacognition in LLMs](../../ICLR2026/interpretability/evidence_for_limited_metacognition_in_llms.md)
- [\[ACL 2026\] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[ICLR 2026\] Understanding Cross-Layer Contributions to Mixture-of-Experts Routing in LLMs](../../ICLR2026/interpretability/understanding_cross-layer_contributions_to_mixture-of-experts_routing_in_llms.md)
- [\[ACL 2026\] Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining](crosscoding_through_time_tracking_emergence_consolidation_of_linguistic_represen.md)

</div>

<!-- RELATED:END -->
