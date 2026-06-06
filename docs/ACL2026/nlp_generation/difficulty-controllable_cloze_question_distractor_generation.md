---
title: >-
  [Paper Note] Difficulty-Controllable Cloze Question Distractor Generation
description: >-
  [ACL 2026][Text Generation][Cloze task] This paper proposes DCDG, which enables cloze distractor generation models to control difficulty (easy/hard) through dual-path distractor data augmentation…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Cloze task"
  - "distractor generation"
  - "difficulty control"
  - "data augmentation"
  - "multi-task learning"
date: 2026-05-08
content_hash: 2c2bbbabbc158c29
---

# Difficulty-Controllable Cloze Question Distractor Generation

**Conference**: ACL 2026  
**arXiv**: [2511.01526](https://arxiv.org/abs/2511.01526)  
**Code**: https://github.com/ksh108405/DCDG  
**Area**: Text Generation / Educational NLP  
**Keywords**: Cloze task, distractor generation, difficulty control, data augmentation, multi-task learning

## TL;DR
This paper proposes DCDG, which enables cloze distractor generation models to control difficulty (easy/hard) through dual-path distractor data augmentation, QA ensemble difficulty clustering, and multi-task seq2seq training. It significantly outperforms GPT-4o in both automatic and human evaluations.

## Background & Motivation
**Background**: Multiple-choice cloze tests are common in language assessments and online education. Automatic item generation systems typically generate a correct answer first, then several reasonable-looking distractors that should not be the correct answer. Recent methods mostly use PLMs or text-to-text models to learn the existing distractor distribution in datasets, while some use candidate word lists or knowledge bases for filtering.

**Limitations of Prior Work**: Existing methods tend to replicate the original distractor distribution of the training set. They can generate words that "look like distractors" but struggle to specify whether a distractor is easy to eliminate or harder and more confusing for learners. Another practical issue is that public cloze datasets usually contain few human-authored distractors and lack large-scale difficulty annotations, making it difficult to train difficulty-aware models due to data scarcity.

**Key Challenge**: Distractor difficulty is related to semantic similarity and contextual substitutability, while also being influenced by the learner's proficiency. To make the problem operational, the authors narrow the research scope to "contextual semantic plausibility" in language assessment: the more it looks like a valid entry for the blank (without being the correct answer), the harder it is; the more obviously it clashes with the context, the easier it is.

**Goal**: The authors aim to solve two sub-problems: first, how to augment the CLOTH dataset (which has no difficulty labels) with a large number of easy/hard distractors; second, how to train a model that can stably generate distractors of a specified difficulty given an input.

**Key Insight**: The key observation is that an answer generator knows which contextual words are most critical for the correct answer. If these high-attention words are intentionally removed and the answer generator is asked to fill the blank again, it will generate candidates that are related to the original answer but no longer fully constrained by the original context. These candidates are naturally suited to serve as distractors of varying difficulty.

**Core Idea**: Supplement a standard distractor generator with an "information-restricted answer generator," use a QA ensemble to cluster candidates into easy/hard categories, and finally embed difficulty control signals into the generation model via a main task plus two auxiliary tasks.

## Method
The proposed method is divided into two layers: the upstream layer constructs an augmented dataset with difficulty labels, and the downstream layer trains a difficulty-controllable distractor generator. The upstream layer is responsible for expanding original CLOTH questions into approximately 12 easy and hard candidates each. The downstream layer generates distractors of a specified difficulty given the passage, blank, answer, target difficulty, and quantity.

### Overall Architecture
The input is a cloze question consisting of a passage, a blank position, a correct answer, and original distractors. Phase 1 trains two Gemma 2 9B generators: a standard distractor generator learning the original distractor distribution, and an answer generator that first learns to produce correct answers and is then reused to generate "misled answers" on passages with partial information removal. Phase 2 uses LanguageTool and GPT-4o mini to filter grammatical errors or candidates that might be correct answers. Phase 3 uses a QA ensemble of multiple fine-tuned PLMs to score candidates and categorize them into hard/easy via a three-way split. Finally, the DCDG model is trained on augmented data, incorporating ASDE and DDDE auxiliary tasks to help the model perceive the semantic relationships between answers, distractors, and difficulty.

### Key Designs
1.  **Dual-path Candidate Generation**:
    - **Function**: Produce candidates using both a standard distractor generator and an information-restricted answer generator to expand the candidate space.
    - **Mechanism**: The standard distractor generator replicates typical distractors from the training set. The "answer generator with information restriction" first generates responses on the full passage to accumulate attention weights, then regenerates candidates after removing high-attention words. Removal ratios (e.g., 0.1, 0.2, 0.4) determine how much the candidate deviates from the original answer.
    - **Design Motivation**: Relying solely on original distractors is limited by the difficulty distribution of the training data, while relying solely on LLM generation may not align with the specific style of cloze contexts. The dual-path design ensures both stylistic alignment and rich difficulty coverage.

2.  **Filtering and Difficulty Clustering**:
    - **Function**: Remove invalid candidates and label remaining distractors as easy or hard.
    - **Mechanism**: LanguageTool filters grammatical errors, and GPT-4o mini identifies candidates that could potentially be correct. Then, 18 small PLMs from 11 model families are fine-tuned as multiple-choice cloze QA systems to estimate the "probability of being selected as the answer" for each candidate. The top third are labeled "hard," the bottom third "easy," and the middle area is discarded. Box-Cox normalization is applied to handle the right-skewed and non-comparable score distributions across different models.
    - **Design Motivation**: Difficulty is defined by "the likelihood that a candidate is mistakenly selected as the answer by QA models" rather than subjective model labeling.

3.  **Multi-task DCDG Training**:
    - **Function**: Train a seq2seq model to generate distractors based on a target difficulty level.
    - **Mechanism**: The main DCDG task takes the passage, quantity, target difficulty, and answer as input to output corresponding distractors. The ASDE auxiliary task requires the model to find the correct answer in mixed options and estimate distractor difficulty. The DDDE auxiliary task fills a distractor into the blank, requiring the model to detect it and judge its difficulty.
    - **Design Motivation**: Training only on "generation given difficulty" can lead to difficulty tokens being treated as superficial labels. ASDE and DDDE force the model to learn answer substitutability, distractor attributes, and relative difficulty, improving the separability of hard/easy categories.

### Loss & Training
All tasks are unified as seq2seq cross-entropy training. Gemma 2 9B is used for both candidate generation and the main DCDG model. The data augmentation phase uses 5-fold cross-validation to ensure the model does not see the training answers of a question when generating its augmented candidates. DCDG uses LoRA with $r=16$ and $\alpha=16$. The warm-up ratio is 0.1. The learning rate is $5e^{-5}$ for DDDE and $3e^{-5}$ for other tasks, using early stopping to manage overfitting.

## Key Experimental Results

### Main Results
| Target | Metric | Ours | Comparison | Key Findings |
| :--- | :--- | :--- | :--- | :--- |
| Augmented Dataset | Easy distractors per q | 12.06 | Original CLOTH ~2.998 | Significant expansion of distractors |
| Augmented Dataset | Hard distractors per q | 12.02 | Original CLOTH ~2.998 | Large samples for both difficulty poles |
| Augmented easy | GPT-4o judged Easiest | 73.17% | Original distractor 21.21% | Easy labels match expectations |
| Augmented hard | GPT-4o judged Hardest | 70.05% | Original distractor 26.53% | Hard labels are significantly more confusing |
| DCDG + ASDE + DDDE | Easy gen judged Easiest | 64.23% | GPT-4o 0-shot 33.54%, 5-shot 46.39% | Superior difficulty control to GPT-4o |
| DCDG + ASDE + DDDE | Hard gen judged Hardest | 73.25% | GPT-4o 0-shot 56.77%, 5-shot 53.81% | Strongest hard-level control |

### Ablation Study
| Configuration | Key Metrics | 说明 |
| :--- | :--- | :--- |
| Answer generator w/ IR | 19.25 candidates/q, semantic diversity 0.6928 | IR generator yields more diverse candidates |
| Distractor generator | 29.66 candidates/q, semantic diversity 0.6684 | Standard generator has higher yield but narrower semantics |
| Path overlap | semantic overlap 0.2908, Jaccard overlap 0.1281 | Two paths are highly complementary |
| Removal ratio 0.1 | diversity 0.6554, plausibility 0.3404 | Closer to the answer, higher difficulty |
| Removal ratio 0.5 | diversity 0.6734, plausibility 0.2920 | More dispersed, lower difficulty |
| DCDG + ASDE + DDDE | invalid ratio: easy 0.2%, hard 5.1% | Reduces invalid distractors while maintaining control |

### Key Findings
- High-attention removal is more effective than random or low-attention removal. Removing 25% of words via low-attention/random methods leads to over 40% of candidates duplicating the correct answer, while high-attention removal keeps the duplication rate below 20%.
- Human ESL evaluation aligns with automatic evaluation trends: 72.8% of easy generated distractors were rated Easiest, and 45.6% of hard ones were rated Hardest, with invalid ratios below 1.6%.
- The Spearman correlation between GPT-4o and human difficulty ranking is 0.54, close to the human-human agreement of 0.62, suggesting GPT-4o is a valid proxy for large-scale difficulty assessment in this setup.

## Highlights & Insights
- The most ingenious aspect is converting "answer generator failure" into "distractor generation capability." By removing critical context, a model striving for the correct answer produces candidates that are related but not quite correct, which is more controllable than direct prompting.
- Difficulty labels do not rely on subjective scoring but are approximated through the selection tendencies of a QA ensemble, allowing task-defined difficulty to be automatically extended to large-scale data.
- The value of ASDE and DDDE lies in letting the model understand *why* something is a distractor, not just generating a word labeled "hard." This can be transferred to tasks like answer generation, distractor explanation, and quality control in reading comprehension labels.

## Limitations & Future Work
- The authors acknowledge that this work only controls distractor difficulty, without integrating item-wide difficulty factors like passage readability, syntactic structure, or blank position.
- Difficulty is discretized into easy/hard binary classes, which avoids arbitrary thresholds but loses the potential for fine-grained pedagogical adaptation.
- The information restriction strategy is primarily designed for word-level cloze questions; its effectiveness for open-ended QA, math problems, or other types requires redesigned removal rules and filtering.
- Future work could use the normalized scores from the QA ensemble as a continuous difficulty axis, calibrated with teacher or learner feedback for personalized control.

## Related Work & Insights
- **vs Traditional knowledge bases/word lists**: Earlier methods relied on WordNet or Probase, which are explainable but limited in domain coverage. This work uses generative models and filters to expand candidates with broader coverage.
- **vs Direct PLM generation**: Methods by Chiang et al. and Wang et al. can generate natural distractors but have weak difficulty control. This work introduces explicit difficulty signals via augmented data and multi-task training.
- **vs IRT-based difficulty modeling**: IRT is closer to real learner proficiency but requires large-scale response data and is computationally expensive. This work uses a discrete difficulty proxy suitable for public cloze datasets lacking student responses.
- **Insight**: For many educational generation tasks, it is better to first construct "behavioral proxy metrics" that can be automatically evaluated, and then train controllable models, rather than expecting LLMs to understand abstract pedagogical difficulty within a prompt.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Using information-restricted answer generators for distractor augmentation is distinctive; the difficulty clustering and multi-task components are well-integrated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Solid automatic evaluation, human ESL evaluation, expert evaluation, dual-path analysis, and multi-task comparisons.
- Writing Quality: ⭐⭐⭐⭐☆ The methodological chain is clear; there are many tables, but they are well-supported; the appendix handles significant implementation details.
- Value: ⭐⭐⭐⭐☆ Highly practical for educational NLP and automatic item generation, providing reusable data and models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Children's English Reading Story Generation via Supervised Fine-Tuning of Compact LLMs with Controllable Difficulty and Safety](childrens_english_reading_story_generation_via_supervised_fine-tuning_of_compact.md)
- [\[ACL 2026\] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)
- [\[ACL 2026\] XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration](xtragpt_context-aware_and_controllable_academic_paper_revision_via_human-ai_coll.md)
- [\[ACL 2026\] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows](facts_table_summarization_via_offline_template_generation_with_agentic_workflows.md)
- [\[ACL 2026\] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)

</div>

<!-- RELATED:END -->
