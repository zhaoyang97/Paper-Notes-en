---
title: >-
  [Paper Note] Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?
description: >-
  [ACL2026][NLP Understanding][Reading Comprehension Difficulty] This paper constructs the ReCo reading comprehension cognitive complexity dataset and systematically evaluates whether 8 LLMs can automatically determine the required evidence scope and transformation levels for items. Results indicate that strong models approach but remain significantly lower than experts, particularly in identifying complete evidence sets and fine-grained word-order transformations.
tags:
  - "ACL2026"
  - "NLP Understanding"
  - "Reading Comprehension Difficulty"
  - "Cognitive Complexity"
  - "Evidence Scope"
  - "Transformation Levels"
  - "Metacognitive Analysis"
date: 2026-05-08
content_hash: 4db72ca8e8e45a9c
---

# Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?

**Conference**: ACL2026  
**arXiv**: [2510.25064](https://arxiv.org/abs/2510.25064)  
**Code**: https://github.com/SeonjeongHwang/ReCo  
**Area**: NLP Understanding / Educational Assessment / LLM Evaluation  
**Keywords**: Reading Comprehension Difficulty, Cognitive Complexity, Evidence Scope, Transformation Levels, Metacognitive Analysis

## TL;DR
This paper constructs the ReCo reading comprehension cognitive complexity dataset and systematically evaluates whether 8 LLMs can automatically determine the required evidence scope and transformation levels for items. Results indicate that strong models approach but remain significantly lower than experts, particularly in identifying complete evidence sets and fine-grained word-order transformations.

## Background & Motivation
**Background**: The difficulty of reading comprehension items typically relies on post-hoc CTT/IRT statistics based on student responses or expert estimation during the development phase. In NLP, linguistic features such as sentence length, vocabulary familiarity, and option similarity are often extracted to explain difficulty.

**Limitations of Prior Work**: These methods either occur after the exam or focus solely on surface linguistic features. Factors that truly affect the learner's cognitive load often arise during the reasoning process, such as the number of sentences required for evidence or whether the relationship between options and the text involves verbatim matching or inference. Historically, these cognitive features have relied on manual annotation.

**Key Challenge**: LLMs have demonstrated strong reading comprehension capabilities, but "answering correctly" is not equivalent to "explaining why an item is difficult." If models can automatically estimate cognitive complexity, they could assist in pre-exam difficulty analysis; if not, it suggests a gap remains between LLMs' reasoning abilities and their metacognitive awareness.

**Goal**: The authors focus on two cognitive dimensions: Evidence Scope, which measures how much text evidence is needed to determine an answer, and Transformation Level, which measures the degree of linguistic transformation between the statement and the textual evidence. The core question is: Can LLMs assign these cognitive labels as experts do?

**Key Insight**: Instead of having LLMs directly predict aggregate difficulty, the paper decomposes difficulty into interpretable cognitive factors and evaluates classification performance using expert-annotated datasets. This approach reveals which reasoning loads the models actually understand better than direct "Hard/Medium/Easy" predictions.

**Core Idea**: Automatically estimate the cognitive complexity of reading comprehension items using LLMs while decomposing the primary task into fine-grained sub-tasks to test consistency between the models' answering ability and their identification of their own evidence/transformation processes.

## Method
The "method" of this paper is not a new model but rather a dataset and an evaluation protocol. The authors first construct ReCo in TFNG (True/False/Not Given) format from real exam items, define two cognitive complexity labels, and finally evaluate the classification capabilities and error patterns of various LLMs using multiple prompting strategies.

### Overall Architecture
The input consists of a reading passage, a statement, and the factuality label of that statement. The model must output a cognitive complexity label for one of two dimensions: the Evidence Scope task requires judging whether the evidence is a single sentence, multiple sentences, or insufficient; the Transformation Level task requires judging whether the statement relates to the evidence through word matching, paraphrasing, word-order changes, or inference. Outputs are compared against expert annotations, with Macro F1 as the primary metric.

### Key Designs
**1. ReCo Dataset Construction: Decomposing Multiple-Choice TF Items into Annotatable TFNG Samples**

To study "why an item is difficult," one needs items with cognitive labels. The authors started with Multiple-Choice True/False (MTF) items from RACE++, consisting of a passage and four options. These were decomposed into `(passage, statement, factuality)` triplets. For False samples, experts wrote "minimally revised" True statements to facilitate subsequent transformation level annotation. The TFNG format was chosen because it naturally covers various cognitive loads—from direct matching to multi-sentence integration and from verifiable evidence to insufficient evidence—making it more suitable for observing difficulty sources than standard extractive QA.

**2. Dual-Dimension Cognitive Labels: Characterizing Answer Load via Evidence Scope and Transformation Level**

Surface features (sentence length, word frequency) fail to explain true response burden. Thus, the authors defined two operational labels. **Evidence Scope** measures the volume of text evidence required, categorized into single-sentence evidence, multi-sentence evidence, and insufficient evidence. **Transformation Level** measures the degree of linguistic transformation between the statement and the evidence. In single-sentence scenarios, it uses five levels: word matching, transformed word matching, paraphrasing, transformed paraphrasing, and inference. In multi-sentence scenarios, it is simplified to word matching, paraphrasing, and inference. The former corresponds to "how much to read," and the latter to "the degree of linguistic/semantic transformation from evidence to answer." Both are closer to the actual reasoning process than surface features.

**3. LLM Evaluation and Fine-Grained Diagnosis: Looking Beyond the Primary Task**

The authors used models like Gemma2, Mistral, Qwen2.5, and GPT-4o with standard prompting, CoT, and CoT self-consistency for zero/one/few-shot classification. To distinguish whether models "failed to understand the text" or "answered correctly but could not explain their evidence," the authors further decomposed tasks into sub-tasks: falsifiability, evidence sentence counting, inference detection, paraphrasing detection, and phrase reordering detection. These sub-tasks act as probes for metacognitive weaknesses—identifying whether a model can accurately recount which sentences it cited and which transformations it performed.

### Loss & Training
No new models were trained; primary methods involved inference-time prompting strategies. Standard prompting directly requested labels; CoT prompting required step-by-step analysis before prediction; self-consistency sampled CoT 10 times using top-$k=20$, top-$p=0.8$, and temperature $0.7$, aggregating answers via priority rules. To avoid inflated scores from simple samples, the authors excluded "too easy" samples that GPT-4o could correctly classify via zero-shot CoT.

## Key Experimental Results

### Main Results

| Task | Best Model / Setting | Best Macro F1 | Human Expert | Key Conclusion |
|------|-----------------------|---------------|--------------|----------------|
| RC Factuality Judgment | GPT-4o CoT 1-shot | 84.4 | N/A | Most strong models can answer, indicating errors in cognitive labels are not primarily from basic reading failure. |
| Evidence Scope | GPT-4o CoT 1-shot | 74.8 | 87.0 | Models approximate evidence scope but remain ~12 F1 points behind experts. |
| Transformation Level (3-level) | Mistral-24B CoT-SC 0-shot | 82.0 | 84.9 | Open-source models approach expert levels; 3-level labels are relatively learnable. |
| Transformation Level (5-level) | GPT-4o CoT 0-shot | 61.3 | 83.0 | Performance drops significantly when word-order reordering is subdivided. |

| ReCo Statistics | Count |
|-----------------|-------|
| Test passages | 151 |
| Test statements | 498 (671 after including revised true) |
| Demonstration passages | 83 |
| Demonstration statements | 278 (371 after including revised true) |
| Evidence Scope Distribution | single 388 / multi 243 / insufficient 145 |
| 3-level Transformation Distribution | word matching 123 / paraphrasing 189 / inference 319 |

### Ablation Study

| Analysis Item | Result | Note |
|---------------|--------|------|
| 5-level vs 3-level TL | 3-level max 82.0, 5-level max 61.3 | Word-order reordering combined with paraphrasing is the hardest fine-grained dimension to identify. |
| Evidence sentence selection | GPT-4o precision 88.8 / recall 79.2 / F1 80.0 | Models tend to select fewer evidence sentences; precision is high but recall is low. |
| Deep reasoning mode | Qwen3-32B thinking mode < Non-thinking | Longer reasoning does not equate to better cognitive complexity classification. |
| Prompting | 1-shot/few-shot not always better | Large models occasionally degrade under few-shot demonstrations, suggesting label boundaries are not just about example coverage. |

### Key Findings
- LLM reading comprehension capability is not synchronized with cognitive complexity estimation: models can answer correctly while remaining unable to explain exactly which evidence they cited or what transformations were performed.
- The main bottleneck for Evidence Scope is sentence count identification; models tend to select one or two obvious evidence sentences while ignoring necessary but subtle sentences found in human annotations.
- The main bottleneck for Transformation Level is phrase reordering; models often categorize transformed word matching as ordinary word matching and confuse paraphrasing with transformed paraphrasing.

## Highlights & Insights
- The paper decomposes "item difficulty" into interpretable cognitive labels rather than asking LLMs for a coarse difficulty score; this allows evaluation results to support specific stages of item writing, revision, and pedagogical diagnosis.
- The ReCo design is clever: TFNG items naturally generate the three main types of difficulty—insufficient evidence, multi-sentence integration, and linguistic transformation—better than standard extractive QA for analyzing cognitive load.
- The most insightful result is "strong reasoning $\neq$ strong metacognition." The degradation in Qwen3 thinking mode suggests that classifying human cognitive processes may rely more on fine-grained pattern recognition than on longer abstract reasoning chains.

## Limitations & Future Work
- Data is derived from RACE++ English exams, and the task format is concentrated on TFNG; whether conclusions generalize to open-ended QA, main idea items, author intent items, or other languages requires verification.
- Annotations only retain samples where at least two experts agreed, which improves reliability but may filter out the most controversial items that best represent difficulty boundaries in real exams.
- Evaluation primarily relies on prompt engineering without training specialized cognitive complexity models; future work could try fine-tuning small models on ReCo or splitting evidence retrieval and transformation classification into explicit multi-stage systems.
- 5-level Transformation Level labels are difficult for models; future work might introduce alignment-based evidence tagging, syntactic reordering detectors, or visual explanations to help models localize text segments before classification.

## Related Work & Insights
- **vs. Traditional IRT/CTT Difficulty Estimation**: IRT/CTT relies on post-exam statistics; this paper estimates complexity before the exam based on item text and cognitive labels, offering advantages in interpretability and pre-analysis at the cost of not directly replacing student performance data.
- **vs. Surface Text Feature Difficulty Prediction**: Sentence length and word frequency are easy to extract but cannot explain cross-sentence evidence or inference loads; the evidence scope and transformation levels in this paper are closer to the actual answering process.
- **vs. Direct LLM Difficulty Scoring**: Asking LLMs "is this item hard" results in a black-box judgment; this paper requires models to provide verifiable cognitive labels, making it more suitable for discovering specific metacognitive failures.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Grounding LLM difficulty estimation in educational psychology cognitive dimensions with a clear problem setting and dataset contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Comprehensive models, prompts, sub-tasks, and error analyses, though task sources and language scope remain somewhat narrow.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure, solid label definitions, and robust analytical logic; some tables are information-dense but conclusions are distinct.
- Value: ⭐⭐⭐⭐☆ Directly inspires research in automatic item generation, reading comprehension assessment, and LLM metacognition.

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
