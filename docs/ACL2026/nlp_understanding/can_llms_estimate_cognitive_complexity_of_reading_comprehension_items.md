---
title: >-
  [Paper Note] Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?
description: >-
  [ACL 2026][NLP Understanding][Paper Note] This paper constructs the ReCo dataset for reading comprehension cognitive complexity and systematically evaluates whether 8 LLMs can automatically determine the required evidence scope and transformation levels. Results show that while powerful models approach expert performance, they remain significantly lower, parti
tags:
  - ACL 2026
  - NLP Understanding
date: 2026-05-08
content_hash: efc6bd84e1527787
---
# Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?

**Conference**: ACL2026  
**arXiv**: [2510.25064](https://arxiv.org/abs/2510.25064)  
**Code**: https://github.com/SeonjeongHwang/ReCo  
**Area**: NLP Understanding / Educational Assessment / LLM Evaluation  
**Keywords**: Reading Comprehension Difficulty, Cognitive Complexity, Evidence Scope, Transformation Level, Metacognitive Analysis

## TL;DR
This paper constructs the ReCo dataset for reading comprehension cognitive complexity and systematically evaluates whether 8 LLMs can automatically determine the required evidence scope and transformation levels. Results show that while powerful models approach expert performance, they remain significantly lower, particularly struggling with identifying complete evidence sets and fine-grained word order transformations.

## Background & Motivation
**Background**: The difficulty of reading comprehension items typically relies on post-test CTT/IRT statistics or expert estimation during the item-writing phase. In NLP, difficulty is often explained by extracting linguistic features such as sentence length, vocabulary familiarity, and option similarity.

**Limitations of Prior Work**: These methods either occur after the exam or focus solely on surface-level linguistic features. Factors that truly impact learner burden arise during the reasoning process, such as the need to retrieve evidence across multiple sentences or the nature of the gap between options and the original text (literal matching vs. inference). Historically, these cognitive features have relied on manual annotation.

**Key Challenge**: While LLMs possess strong reading comprehension abilities, "solving an item correctly" is not equivalent to "explaining why an item is difficult." If models can automatically estimate cognitive complexity, they can assist in pre-test difficulty analysis; if not, it indicates a gap remains between LLM reasoning capabilities and metacognitive awareness.

**Goal**: The authors focus on two cognitive dimensions: Evidence Scope, which measures the amount of text evidence needed to judge an answer, and Transformation Level, which measures the degree of linguistic conversion between the statement and the evidence. The core problem is: Can LLMs assign these cognitive labels as accurately as experts?

**Key Insight**: Instead of asking LLMs to directly predict aggregate difficulty, the paper decomposes difficulty into interpretable cognitive factors and evaluates classification using expert-annotated data. This approach reveals specific reasoning burdens more effectively than simple "Hard/Medium/Easy" predictions.

**Core Idea**: Automatically estimate reading comprehension cognitive complexity using LLMs, while decomposing the main task into fine-grained sub-tasks to examine whether a model's solving ability is consistent with its recognition of its own evidence-finding and transformation processes.

## Method
The "Method" does not propose a new model but rather a dataset and evaluation protocol. The authors construct the ReCo dataset in TFNG format from real examination items, define two cognitive complexity labels, and evaluate the classification capabilities and error patterns of various LLMs using multiple prompting strategies.

### Overall Architecture
The input consists of a reading passage, a statement, and the factuality label of that statement. The model must output a cognitive complexity label for one of two dimensions: the Evidence Scope task requires judging if evidence is a single sentence, multiple sentences, or insufficient; the Transformation Level task requires judging if the relationship is word matching, paraphrasing, word order change, or inference. Outputs are compared with expert annotations, with Macro F1 as the primary metric.

### Key Designs
**1. ReCo Dataset Construction: Converting MTF items into annotatable TFNG samples**

To study "why an item is difficult," one must first have items with cognitive labels. The authors start with Multiple True/False (MTF) items from RACE++, splitting each into `(passage, statement, factuality)` triplets. For False samples, experts write minimally revised True statements to facilitate subsequent transformation level annotation. The TFNG (True/False/Not Given) format is chosen because it naturally covers diverse cognitive burdens ranging from direct matching to multi-sentence integration and evidence insufficiency.

**2. Dual-Dimensional Cognitive Labels: Characterizing response burden through Evidence Scope and Transformation Level**

Surface features (sentence length, word frequency) fail to explain the true response burden. Thus, the authors define two operational labels. **Evidence Scope** measures the volume of text evidence required, categorized into single-sentence, multi-sentence, and insufficient evidence. **Transformation Level** measures the linguistic distance between the statement and evidence. For single-sentence evidence, it uses 5 levels: word matching, transformed word matching, paraphrasing, transformed paraphrasing, and inference. This is simplified to 3 levels for multi-sentence scenarios. These labels are more closely aligned with actual reasoning processes than surface features.

**3. LLM Evaluation & Fine-grained Diagnosis: Beyond main tasks to pinpoint error sources**

Using models such as Gemma2, Mistral, Qwen2.5, and GPT-4o, the authors perform zero/one/few-shot classification using standard prompting, CoT, and CoT self-consistency. To distinguish between "failure to understand the text" and "solving the item but failing to identify the evidence/transformation," the authors further decompose the problem into sub-tasks: falsifiability, evidence sentence counting, inference detection, paraphrasing detection, and phrase reordering detection. These act as probes for the models' metacognitive shortfalls.

### Loss & Training
This work does not train new models and relies on inference-time prompting. Standard prompting directly requests labels; CoT prompting requires step-by-step analysis before prediction; self-consistency samples 10 times under CoT conditions using top-$k=20$, top-$p=0.8$, and temperature $0.7$, aggregating answers via priority rules. To avoid score inflation from easy samples, the authors exclude "trivial" samples that GPT-4o can correctly classify using zero-shot CoT.

## Key Experimental Results

### Main Results
| Task | Best Model / Setting | Best Macro F1 | Human Expert | Key Conclusion |
|------|----------------------|---------------|--------------|----------------|
| RC Factuality Judgment | GPT-4o CoT 1-shot | 84.4 | N/A | Strong models can solve items; failures in cognitive labeling are not primary due to basic reading failure. |
| Evidence Scope | GPT-4o CoT 1-shot | 74.8 | 87.0 | Models approximate evidence scope but lag experts by ~12 F1 points. |
| Transformation Level (3-level) | Mistral-24B CoT-SC 0-shot | 82.0 | 84.9 | Open-source models approach expert performance; 3-level labels are relatively learnable. |
| Transformation Level (5-level) | GPT-4o CoT 0-shot | 61.3 | 83.0 | Performance drops significantly when identifying fine-grained word order reordering. |

| ReCo Statistics | Count |
|-----------------|-------|
| Test passages | 151 |
| Test statements | 498 (671 after revised true) |
| Demonstration passages | 83 |
| Demonstration statements | 278 (371 after revised true) |
| Evidence Scope Distribution | single 388 / multi 243 / insufficient 145 |
| 3-level Transformation Distribution | word matching 123 / paraphrasing 189 / inference 319 |

### Ablation Study
| Analysis Item | Result | Note |
|---------------|--------|------|
| 5-level vs. 3-level TL | 3-level max 82.0, 5-level max 61.3 | Word order reordering combined with paraphrasing is the hardest fine-grained dimension to identify. |
| Evidence sentence selection | GPT-4o precision 88.8 / recall 79.2 / F1 80.0 | Models tend to underselect evidence sentences (high precision, low recall). |
| Deep reasoning mode | Qwen3-32B thinking mode lower than non-thinking | Longer reasoning chains do not equate to better cognitive complexity classification. |
| Prompting | One/few-shot not always better | LLMs sometimes degrade with few-shot demonstrations, suggesting label boundaries involve more than just lack of examples. |

### Key Findings
- LLM reading comprehension capability and cognitive complexity estimation capability are not synchronized: models can answer items correctly but cannot accurately identify the evidence they cited or the transformations they performed.
- The primary bottleneck for Evidence Scope is identifying the number of evidence sentences; models tend to pick one or two obvious sentences while ignoring necessary but subtle supporting ones.
- The main bottleneck for Transformation Level is phrase reordering; models frequently mistake transformed word matching for simple word matching and confuse paraphrasing with transformed paraphrasing.

## Highlights & Insights
- The paper decomposes "item difficulty" into interpretable cognitive labels rather than asking LLMs for a coarse-grained difficulty score. This allows evaluation results to inform specific stages of item writing, revision, and pedagogical diagnosis.
- The ReCo design is clever: TFNG items naturally generate the three main difficulty sources—insufficient evidence, multi-sentence integration, and linguistic transformation—making them more suitable for analyzing cognitive burden than standard extractive QA.
- The most insightful result is that "strong reasoning" $\neq$ "strong metacognition." The degradation of the Qwen3 thinking mode indicates that classifying human cognitive processes may rely more on fine-grained pattern recognition than on extended abstract reasoning chains.

## Limitations & Future Work
- Data is sourced from RACE++ English exams in TFNG format; whether conclusions generalize to open-ended QA, main idea items, author intent items, or other languages remains to be verified.
- Annotations only retain samples with consensus from at least two experts, which improves reliability but may filter out the most controversial boundary cases that define real-world difficulty.
- The evaluation relies heavily on prompt engineering without training specialized models; future work could attempt fine-tuning smaller models on ReCo or building explicit multi-stage systems for evidence retrieval and transformation classification.
- The 5-level Transformation Level labels are difficult for models; future work could introduce alignment-based evidence annotation, syntactic reordering detectors, or visual explanations to help models locate source fragments before classification.

## Related Work & Insights
- **vs. Traditional IRT/CTT Difficulty Estimation**: While IRT/CTT depends on post-hoc student response statistics, this paper estimates complexity pre-test based on item text. The advantage is interpretability and pre-analysis; the disadvantage is that it cannot directly replace actual student performance data.
- **vs. Surface Text Feature Difficulty Prediction**: Sentence length and word frequency are easy to extract but cannot explain cross-sentence evidence or inferential burdens. The Evidence Scope and Transformation Level in this paper are closer to the actual answering process.
- **vs. Direct LLM Difficulty Rating**: Asking LLMs "Is this item hard?" often results in a black-box judgment. This paper requires models to provide verifiable cognitive labels, which is better for identifying specific metacognitive failures.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Grounding LLM difficulty estimation in educational psychology cognitive dimensions with a clear problem setup and dataset contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Comprehensive models, prompts, sub-tasks, and error analyses, though the task source and language scope remain somewhat narrow.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure, robust label definitions, and logical analysis.
- Value: ⭐⭐⭐⭐☆ Provides direct insights for automated item generation, reading assessment, and LLM metacognition research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension?](../../ACL2025/nlp_understanding/can_llms_reliably_simulate_real_students_abilities_in_mathematics_and_reading_co.md)
- [\[ACL 2025\] Automatic Generation of Inference Making Questions for Reading Comprehension Assessments](../../ACL2025/nlp_understanding/automatic_generation_of_inference_making_questions_for_reading_comprehension_ass.md)
- [\[ACL 2026\] Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)
- [\[ACL 2026\] Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs](creating_conlangs_to_probe_the_metalinguistic_grammatical_knowledge_of_llms.md)
- [\[ACL 2026\] Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers](test-time_reasoners_are_strategic_multiple-choice_test-takers.md)

</div>

<!-- RELATED:END -->
