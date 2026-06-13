---
title: >-
  [Paper Note] Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?
description: >-
  [ACL2026][NLP Understanding][Reading comprehension difficulty] This paper constructs the ReCo dataset for cognitive complexity in reading comprehension and systematically evaluates whether 8 LLMs can automatically determ…
tags:
  - "ACL2026"
  - "NLP Understanding"
  - "Reading comprehension difficulty"
  - "cognitive complexity"
  - "evidence scope"
  - "transformation level"
  - "metacognitive analysis"
date: 2026-05-08
content_hash: 28051de72147b017
---

# Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?

**Conference**: ACL2026  
**arXiv**: [2510.25064](https://arxiv.org/abs/2510.25064)  
**Code**: https://github.com/SeonjeongHwang/ReCo  
**Area**: NLP Understanding / Educational Assessment / LLM Evaluation  
**Keywords**: Reading comprehension difficulty, cognitive complexity, evidence scope, transformation level, metacognitive analysis

## TL;DR
This paper constructs the ReCo dataset for cognitive complexity in reading comprehension and systematically evaluates whether 8 LLMs can automatically determine the required evidence scope and transformation level for items. Results show that while strong models approach expert performance, they still fall significantly short, particularly in identifying complete evidence sets and fine-grained word-order transformations.

## Background & Motivation
**Background**: The difficulty of reading comprehension items typically relies on post-hoc statistical analysis (CTT/IRT) or manual estimation by experts during the item-writing phase. NLP approaches often extract surface linguistic features like sentence length, vocabulary familiarity, and option similarity to explain difficulty.

**Limitations of Prior Work**: These methods either occur after examinations or focus solely on surface-level linguistic features. Factors that truly impact learner burden arise during the reasoning process, such as the number of sentences required for evidence or whether the relationship between options and the original text involves literal matching or inference. Historically, these cognitive features have relied on manual annotation.

**Key Challenge**: LLMs possess strong reading comprehension capabilities, but "answering correctly" is not equivalent to "explaining why an item is difficult." If models can automatically estimate cognitive complexity, they can assist in pre-exam difficulty analysis; if not, it indicates a gap between LLM reasoning ability and metacognitive awareness.

**Goal**: The authors explore two cognitive dimensions: Evidence Scope (measuring how much text evidence is needed) and Transformation Level (measuring the linguistic shift between the statement and the evidence). The core question is: Can LLMs assign these cognitive labels as experts do?

**Key Insight**: Instead of having LLMs directly predict overall difficulty, this work decomposes difficulty into interpretable cognitive factors and evaluates them using expert-labeled datasets via classification. This approach reveals specific reasoning burdens more effectively than simple "Hard/Medium/Easy" predictions.

**Core Idea**: Automatically estimate the cognitive complexity of items using LLMs while decomposing the primary task into fine-grained sub-tasks to examine the consistency between a model's answering ability and its recognition of its own evidence/transformation processes.

## Method
The "Method" does not propose a new model but rather a dataset and evaluation protocol. The authors construct ReCo in a TFNG (True/False/Not Given) format from real exam items, define two cognitive complexity labels, and evaluate the classification capabilities and error patterns of various LLMs using multiple prompting strategies.

### Overall Architecture
The input consists of a reading passage, a statement, and the factuality label of that statement. The model must output a cognitive complexity label for one of two dimensions: Evidence Scope (single-sentence, multi-sentence, or insufficient) or Transformation Level (word matching, paraphrasing, or inference, etc.). Results are compared against expert annotations using Macro F1 as the primary metric.

### Key Designs
1.  **ReCo Dataset Construction**:
    -   **Function**: Contextualizes multiple-choice RACE++ items into annotatable TFNG-style reading comprehension samples.
    -   **Mechanism**: Each item consists of a passage and a statement. For False samples, experts wrote a minimally revised True version to facilitate transformation level labeling.
    -   **Design Motivation**: TFNG items naturally cover a range of cognitive burdens from direct matching to multi-sentence integration, making them ideal for observing why items are difficult.

2.  **Dual-dimension Cognitive Labels**:
    -   **Function**: Decomposes reading comprehension complexity into two operational labels: Evidence Scope and Transformation Level.
    -   **Mechanism**: Evidence Scope includes single-sentence evidence, multi-sentence evidence, and insufficient evidence. Transformation Level uses a 5-level hierarchy for single-sentence evidence (word matching, transformed word matching, paraphrasing, transformed paraphrasing, inference) and a simplified 3-level version for multi-sentence scenarios.
    -   **Design Motivation**: Evidence scope corresponds to the volume of text processed, while transformation level corresponds to the semantic/linguistic shift required, both of which are closer to the actual answering process than surface features.

3.  **LLM Evaluation and Fine-grained Diagnosis**:
    -   **Function**: Assesses the model's ability to provide primary task labels and analyzes the sources of failure.
    -   **Mechanism**: Evaluates Gemma2, Mistral, Qwen2.5, and GPT-4o series using standard prompting, CoT, and CoT self-consistency in zero/one/few-shot settings. Tasks are further decomposed into sub-tasks like falsifiability, evidence sentence counting, and inference detection.
    -   **Design Motivation**: Macro F1 alone cannot distinguish between comprehension failure and the inability to analyze one's own reasoning process. Sub-tasks reveal specific metacognitive weaknesses.

### Loss & Training
No new models were trained; the study primarily utilizes inference-time prompting strategies. Standard prompts request direct labels; CoT prompts require step-by-step analysis before prediction. Self-consistency was applied under CoT conditions with 10 samples, using top-$k=20$, top-$p=0.8$, and temperature $0.7$. To avoid inflating scores with simple samples, the authors excluded "easy" samples that GPT-4o could correctly classify via zero-shot CoT.

## Key Experimental Results

### Main Results
| Task | Best Model / Setting | Best Macro F1 | Human Expert | Key Conclusion |
|------|----------------------|---------------|--------------|----------------|
| Factuality Judgment | GPT-4o CoT 1-shot | 84.4 | - | Most strong models can answer, indicating errors in cognitive labeling do not stem from basic comprehension failure. |
| Evidence Scope | GPT-4o CoT 1-shot | 74.8 | 87.0 | Models approximate evidence scope but trail experts by ~12 F1 points. |
| Transformation Level (3-class) | Mistral-24B CoT-SC zero-shot | 82.0 | 84.9 | Open-source models can approach expert performance; 3-class labels are relatively learnable. |
| Transformation Level (5-class) | GPT-4o CoT zero-shot | 61.3 | 83.0 | Performance drops significantly when word-order reordering is introduced. |

| ReCo Statistics | Count |
|-----------------|-------|
| Test passages | 151 |
| Test statements | 498 (671 including revised true) |
| Evidence Scope distribution | single 388 / multi 243 / insufficient 145 |
| 3-class Transformation | word matching 123 / paraphrasing 189 / inference 319 |

### Ablation Study
| Analysis Item | Results | Description |
|---------------|---------|-------------|
| 5-class vs. 3-class TL | 3-class Max 82.0, 5-class Max 61.3 | Combinations of word reordering and paraphrasing are the most difficult to identify. |
| Evidence sentence selection | GPT-4o: Precision 88.8 / Recall 79.2 / F1 80.0 | Models tend to select fewer evidence sentences (high precision, low recall). |
| Deep reasoning mode | Qwen3-32B thinking mode < non-thinking | Longer reasoning does not necessarily lead to better cognitive complexity classification. |
| Prompting | 1-shot/few-shot not always better | Performance sometimes degrades with few-shot examples, suggesting label boundaries are not just an issue of example coverage. |

### Key Findings
- LLM reading comprehension capability and cognitive complexity estimation ability are not synchronized: models can answer correctly but cannot accurately identify which evidence they cited or what transformation occurred.
- The primary bottleneck in Evidence Scope is sentence counting; models tend to pick one or two salient sentences, ignoring necessary but subtle sentences found in human labels.
- The primary bottleneck in Transformation Level is phrase reordering; models often mistake transformed word matching for standard word matching and confuse paraphrasing with transformed paraphrasing.

## Highlights & Insights
- The paper decomposes "item difficulty" into interpretable cognitive labels rather than requesting a coarse difficulty score, making the results applicable to item writing, revision, and diagnostic teaching.
- The design of ReCo is elegant: TFNG items naturally elicit three types of cognitive burden—insufficient evidence, multi-sentence integration, and linguistic transformation—more effectively than extractive QA.
- A significant insight is that "strong reasoning" does not imply "strong metacognition." The degradation of Qwen3's thinking mode suggests that classifying human cognitive processes may rely more on fine-grained pattern recognition than on long chains of abstract reasoning.

## Limitations & Future Work
- Data is derived from RACE++ English exams in TFNG format; results might not generalize to open-ended QA, main idea questions, intent detection, or other languages.
- Annotations only include samples where at least two experts agreed, which increases reliability but might exclude the most controversial items that define difficulty boundaries in real exams.
- The evaluation relies on prompt engineering. Future work could involve fine-tuning smaller models using ReCo or building explicit multi-stage systems for evidence retrieval and transformation classification.
- Transformation Level 5-class labels are difficult for models; future work could introduce alignment-based evidence labeling or syntactic reordering detectors to help models locate text fragments before classification.

## Related Work & Insights
- **vs. Traditional IRT/CTT**: IRT/CTT relies on post-exam statistics. Ours estimates complexity pre-exam based on text and cognitive labels, offering interpretability and pre-analysis, though it doesn't replace real student data.
- **vs. Surface Feature Difficulty Prediction**: Surface features are easy to extract but fail to explain cross-sentence reasoning or inference burden; our Evidence Scope and Transformation Level are closer to the cognitive process.
- **vs. Direct LLM Difficulty Scoring**: Asking "is this hard?" results in a black-box judgment; requiring verifiable cognitive labels allows for the discovery of specific metacognitive failures.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Grounding LLM difficulty estimation in educational psychology dimensions with a solid dataset contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Comprehensive models, prompts, sub-tasks, and error analysis, though limited in task source and language.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure and rigorous definitions, with distinct conclusions despite dense tables.
- Value: ⭐⭐⭐⭐☆ Direct implications for automated item generation, assessment, and LLM metacognition research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)
- [\[ACL 2026\] Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs](creating_conlangs_to_probe_the_metalinguistic_grammatical_knowledge_of_llms.md)
- [\[ICLR 2026\] What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering](../../ICLR2026/nlp_understanding/whats_the_plan_metrics_for_implicit_planning_in_llms_and_their_application_to_rh.md)
- [\[NeurIPS 2025\] Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL](../../NeurIPS2025/nlp_understanding/planning_without_search_refining_frontier_llms_with_offline_goal-conditioned_rl.md)
- [\[ACL 2026\] Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers](test-time_reasoners_are_strategic_multiple-choice_test-takers.md)

</div>

<!-- RELATED:END -->
