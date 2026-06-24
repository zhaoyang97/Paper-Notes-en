---
title: >-
  [Paper Note] Automatic Generation of Inference Making Questions for Reading Comprehension Assessments
description: >-
  [ACL 2025][NLP Understanding][Reading Comprehension] A reading comprehension inference question taxonomy (pronominal bridging / text-connecting / gap-filling) is developed to automatically generate multiple-choice questions for specific inference types using GPT-4o few-shot prompting; while 93.8% of the questions are of acceptable quality, only 42.6% accurately match the target inference type, indicating LLMs still lack precise control over their reasoning abilities.
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Reading Comprehension"
  - "Inference Question Generation"
  - "Bridging Inference Taxonomy"
  - "GPT-4o"
  - "Diagnostic Assessment"
date: 2026-05-08
content_hash: 47d1ddbd04c888fc
---

# Automatic Generation of Inference Making Questions for Reading Comprehension Assessments

**Conference**: ACL 2025  
**arXiv**: [2506.08260](https://arxiv.org/abs/2506.08260)  
**Code**: [https://github.com/maafiah/InferenceQuestionsAQG](https://github.com/maafiah/InferenceQuestionsAQG)  
**Area**: NLP Understanding / Educational NLP  
**Keywords**: Reading Comprehension, Inference Question Generation, Bridging Inference Taxonomy, GPT-4o, Diagnostic Assessment

## TL;DR
A reading comprehension inference question taxonomy (pronominal bridging / text-connecting / gap-filling) is developed to automatically generate multiple-choice questions for specific inference types using GPT-4o few-shot prompting; while 93.8% of the questions are of acceptable quality, only 42.6% accurately match the target inference type, indicating LLMs still lack precise control over their reasoning abilities.

## Background & Motivation
**Background**: Inference ability is a core but complex skill in reading comprehension (RC). Diagnostic RC assessments require items targeted at specific inference types to help educators provide targeted reading interventions. Existing LLM-based question generation research largely treats RC as a single construct and does not distinguish among inference types.

**Limitations of Prior Work**: (a) Manual creation of inference questions is highly costly and difficult to scale; (b) existing automatic question generation research focuses on overall quality but lacks control over the inference type; (c) there is a lack of a systematic taxonomy of RC inference questions to guide generation.

**Key Challenge**: While LLMs can generate high-quality RC items, can they generate items targeting *specific inference types*? Precise control over inference types represents the critical gap between "usable" and "usable for diagnostic assessment."

**Goal**: (a) Establish a bridging inference question taxonomy; (b) verify whether GPT-4o can generate RC items of specific inference types under few-shot prompting; (c) evaluate whether CoT prompting is helpful.

**Key Insight**: Building an inference taxonomy based on reading science literature, prompting GPT-4o to generate questions for each type, and evaluating their quality and type accuracy using three expert annotators.

**Core Idea**: LLMs can generate high-quality RC questions at scale, but precisely matching the inference type still requires human review; "automatic generation + human judgment" represents a scalable approach to diagnostic assessment.

## Method

### Overall Architecture
1. Literature review → Constructing a bridging inference taxonomy (3 types)
2. Annotation and validation of the taxonomy on a live item bank
3. Manually writing training exemplar questions (6 passages × 2-4 questions per type)
4. Generating questions for 10 new passages using GPT-4o few-shot prompting
5. Three expert annotators assessing the generated questions across three dimensions

### Key Designs

1. **Bridging Inference Taxonomy**:

    - **Function**: Categorizes RC inference items into three bridging inference types.
    - **Pronominal Bridging**: Bridges information between sentences using pronouns as clues, e.g., referring to the earlier "ships" with "That."
    - **Text-Connecting**: Links two explicitly stated textual elements using noun phrases, often involving causal relationships.
    - **Gap-Filling**: Requires readers to apply extra-textual world knowledge to fill in details not explicitly stated.
    - **Design Motivation**: Bridging inference accounts for 51% of the 192-item live item bank, representing the most critical sub-construct; the three types correspond to different cognitive skills.

2. **Few-shot Prompt Generation**:

    - **Function**: Designs independent system prompts for each inference type, containing the type definition, generation steps, and 4 or 6 exemplars.
    - Four conditions are compared: Standard_4, Standard_6, CoT_4, CoT_6.
    - CoT conditions additionally provide a "Text Hint" (relevant sentences in the text) and "Reasoning" (explanation of the reasoning process).
    - Three items are generated per passage-type combination, with temperature set to 0 and `frequency_penalty=0.2`.

3. **Three-Dimensional Expert Evaluation**:

    - **General Item Quality**: Overall item quality (whether the key is correct, whether distractors are plausible, grade appropriateness for grades 3-12).
    - **Inference-type Accuracy**: Whether the generated items match the requested inference types.
    - **Reasoning Quality**: Whether the reasoning chain provided by the LLM in CoT conditions is sufficient and sound.
    - Two annotation rounds: independent annotation in the first round, followed by adjudication of discrepancies in the second, resulting in a Fleiss' κ of 0.57-0.83.

## Key Experimental Results

### Main Results

| Generation Method | No. of Items | Quality Acceptance Rate | Inference Type Accuracy | Reasoning Quality Acceptance Rate |
|---------|--------|-----------|--------------|-------------|
| Standard_4 | 88 | 93.2% | 40.9% | - |
| Standard_6 | 89 | 95.5% | **46.1%** | - |
| CoT_4 | 90 | 90.0% | 41.1% | 35.6% |
| CoT_6 | 90 | **96.7%** | 42.2% | 38.9% |
| Total | 357 | 93.8% | 42.6% | 37.2% |

### Ablation Study: Difficulty of Generating Each Inference Type (Standard_6)

| Target Inference Type | Accuracy | Description |
|------------|--------|------|
| Gap-Filling | **60.0%** | Easiest to generate accurately |
| Pronominal Bridging | 53.3% | Moderate |
| Text-Connecting | 24.1% | Hardest, often degenerating into factual questions |

### Key Findings
- **High Quality but Low Accuracy**: While 93.8% of the items were of acceptable quality for actual assessments, only 42.6% matched the target inference type—high quality does not equate to precise control.
- **Increasing Exemplars (4 to 6) is Effective**: Performance across all metrics improved with 6 exemplars.
- **CoT Does Not Help**: Adding reasoning process exemplars did not improve inference type accuracy (42.2% vs 46.1%), likely because of the LLM's own reasoning deficiencies (only 38.9% of the generated reasoning chains were deemed sound).
- **34.8% of generated items degenerated into factual/literal items**, indicating that LLMs tend to generate easier questions that do not require deep reasoning.
- The inference type distribution of the generated items is highly similar to that of the human item bank—though individual items might be inaccurate, the overall distribution remains useful.

## Highlights & Insights
- **Practical Value of the Inference Taxonomy**: Validated on a live item bank (where bridging inference constitutes 51%), providing a roadmap for future item development and research.
- **Decoupling of Quality and Controllability**: Generating high-quality items does not guarantee precise control over item properties, which is a key insight for educational NLP applications.
- **Pragmatic "Generation + Human Review" Approach**: Instead of striving for full automation, utilizing LLMs for mass generation followed by human filtering is significantly more efficient than writing questions manually from scratch.

## Limitations & Future Work
- Only evaluated using a single model (GPT-4o); other models with stronger reasoning capabilities (e.g., o1, Claude) were not tested.
- Evaluated on only 10 expository texts, without covering other genres like narrative texts.
- The ineffectiveness of CoT might be due to the limited number of training exemplars (only 12-18); more examples might yield improvements.
- No testing was conducted with real students; psychometric properties like item discrimination and difficulty remain unknown.
- Future work could leverage LLMs to first classify the inference types of existing items in stock to obtain more training exemplars.
- The matching rate for the Text-Connecting type is extremely low (24.1%), requiring targeted prompt optimization.

## Related Work & Insights
- **Compared to General QG Research**: Prior QG studies treated RC as a singular construct. This work is the first to systematically generate items based on inference types, filling a critical gap.
- **Compared to Säuberli & Clematide (2024)**: While they successfully applied CoT in RC QG, the inference type control task in this study is more complex, causing CoT to fail.
- **Compared to Multi-Hop QA**: Multi-hop reasoning in NLP intersects with bridging inference, but educational assessment scenarios entail unique requirements (e.g., grade-appropriateness, distractor quality).
- Insight: LLMs exhibit limited capability in grasping the nuances of specific inference types. Future efforts may need to integrate structured knowledge or specialized fine-tuning.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically apply an inference type taxonomy to LLM-based question generation, offering a valuable task definition.
- Experimental Thoroughness: ⭐⭐⭐ Rigorous evaluation by three experts, though limited to a single model and a small set of texts.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, theoretically grounded taxonomy, and complete evaluation methodology.
- Value: ⭐⭐⭐⭐ Directly relevant and highly practical for educational NLP; the finding "high quality but inaccurate target typing" is highly significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension?](can_llms_reliably_simulate_real_students_abilities_in_mathematics_and_reading_co.md)
- [\[ACL 2026\] Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?](../../ACL2026/nlp_understanding/can_llms_estimate_cognitive_complexity_of_reading_comprehension_items.md)
- [\[ACL 2026\] Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference?](../../ACL2026/nlp_understanding/filling_the_gap_is_commonsense_knowledge_generation_useful_for_natural_language_.md)
- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ACL 2026\] Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling](../../ACL2026/nlp_understanding/semantic_reranking_at_inference_time_for_hard_examples_in_rhetorical_role_labeli.md)

</div>

<!-- RELATED:END -->
