---
title: >-
  [Paper Note] Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring
description: >-
  [ACL2026][LLM Evaluation][Question Difficulty] Q-Daps estimates LLM question-answering difficulty by generating multiple candidate answers and calculating the entropy of the plausibility distribution after removing popul…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "Question Difficulty"
  - "Answer Plausibility"
  - "Entropy"
  - "Popularity Bias"
  - "Hallucination Risk"
date: 2026-05-08
content_hash: 173fc012953ac8f4
---

# Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring

**Conference**: ACL2026  
**arXiv**: [2605.12398](https://arxiv.org/abs/2605.12398)  
**Code**: https://github.com/DataScienceUIBK/Q-DAPS  
**Area**: Interpretability / Question Difficulty Estimation  
**Keywords**: Question Difficulty, Answer Plausibility, Entropy, Popularity Bias, Hallucination Risk

## TL;DR
Q-Daps estimates LLM question-answering difficulty by generating multiple candidate answers and calculating the entropy of the plausibility distribution after removing popularity bias. It systematically outperforms readability, retrieval complexity, prompt-based scoring, and uncertainty baselines on TriviaQA, NQ, MuSiQue, and QASC.

## Background & Motivation
**Background**: Question difficulty estimation is commonly used in educational assessment, information retrieval, and QA system evaluation. Traditional methods rely on text readability, question popularity, retrieval quality, or direct difficulty scoring by models.

**Limitations of Prior Work**: These signals are often designed for human readers or retrieval systems and may not reflect the actual ambiguity and reasoning risks encountered by modern LLMs. For LLMs, the difficulty of a question often depends on whether incorrect answers appear as "real" as the correct one.

**Key Challenge**: Difficult questions are not necessarily long or rare; simple surface features fail to capture the uncertainty within the candidate answer space. When multiple incorrect answers are persuasive, the model is more prone to hallucinations or incorrect selections.

**Goal**: Propose an interpretable and scalable question difficulty metric consistent with LLM behavior, and verify its ability to distinguish between easy/hard questions, maintain consistency with human judgment, and remain robust without gold standard answers or across different model settings.

**Key Insight**: The authors focus on the "answer plausibility distribution." If only a few candidates appear credible, the plausibility distribution is sharp, indicating low difficulty. If multiple candidates are similarly credible, the entropy of the distribution is high, indicating high difficulty.

**Core Idea**: Use the entropy of candidate answer plausibility as an LLM-oriented difficulty score, applying a lightweight correction for popularity bias using Wikipedia page views.

## Method
The Q-Daps method consists of three stages: candidate answer generation, popularity debiasing, and entropy scoring. It infers difficulty from the shape of the candidate answer space rather than directly asking an LLM to score difficulty.

### Overall Architecture
Given a question and an optional gold answer, Q-Daps first uses an LLM (e.g., LLaMA 3.3) to generate up to 20 candidate answers, each accompanied by a plausibility score and justification. A verification module then de-duplicates candidates and checks score ranges. Subsequently, Wikipedia page views for each candidate are queried to obtain a popularity score, which is subtracted from the plausibility using hyperparameter $\alpha$. Finally, the debiased plausibility is normalized into a probability distribution to calculate the Shannon entropy, which is normalized to $[0,1]$ by dividing by $log_2 N$.

### Key Designs
1.  **Candidate Answer Plausibility Generation and Verification**:
    - **Function**: Constructs a "competitive answer space" for the question, where difficulty stems from the relative credibility between multiple candidates.
    - **Mechanism**: A listwise prompt is used to generate $N$ candidate answers, plausibility scores, and justifications simultaneously. The verification module checks for duplicates, ensures scores are between 0 and 100, and verifies the count; if it fails, the temperature is increased by 0.1 for regeneration.
    - **Design Motivation**: Standard distractor generation only determines if a distractor is correct, without characterizing "how correct it looks." Q-Daps requires plausibility as a continuous signal.

2.  **Popularity Debiasing via Wikipedia Page Views**:
    - **Function**: Reduces popularity bias in LLM candidate answer generation.
    - **Mechanism**: Monthly Wikipedia page views from 2015-01-01 to 2024-12-31 are fetched for each candidate, normalized to $[0,1]$, and cleaned of outliers using IQR. The debiasing formula is $DePls_i=Pls_i\times(1-\alpha\times Pop_i)$.
    - **Design Motivation**: LLMs tend to generate well-known entities earlier and may assign them excessively high plausibility. Debiasing ensures difficulty estimation focuses on semantic competition rather than entity fame.

3.  **Normalized Entropy Difficulty Score**:
    - **Function**: Transforms the uncertainty of candidate plausibility into an interpretable difficulty metric.
    - **Mechanism**: A probability distribution is derived via $DePls_i^{norm}=DePls_i / \sum_i DePls_i$, followed by calculating $H(q)=-\sum_i DePls_i^{norm} log_2 DePls_i^{norm}$. The final score is $Diff_q=H(q)/log_2 N$.
    - **Design Motivation**: High entropy indicates that multiple candidates are similarly plausible, making it harder for the model to distinguish them; low entropy indicates a few answers stand out, making the question easier.

### Loss & Training
Q-Daps is an evaluation and scoring method and does not train the target QA model. The experiments compare three plausibility elicitation methods: Pointwise, Pairwise, and Listwise. Pointwise scores each candidate individually ($O(n)$ complexity); Pairwise performs binary comparisons aggregated via Bradley-Terry ($O(n^2)$ complexity); Listwise generates candidates and scores at once ($O(1)$ complexity) and serves as the primary configuration. Difficulty evaluation uses two metrics: Spearman correlation to measure the negative correlation between difficulty scores and the number of LLMs that answer correctly, and Cohen's d to measure the accuracy gap between easy/hard groups partitioned by Q-Daps.

## Key Experimental Results

### Main Results

| Category | Method | MuSiQue d / rho | QASC d / rho | NQ d / rho | TriviaQA d / rho |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Readability | Flesch-Kincaid | -0.543 / 0.5545 | 0.1496 / 0.1909 | -0.424 / 0.6363 | -0.2689 / 0.5181 |
| Prompt-based | LLaMA 3.3 70B | 0.2453 / -0.109 | 0.2032 / -0.2909 | 0.0307 / -0.3363 | 0.4566 / -0.4272 |
| Retriever-based | Retrieval Complexity | 0.1284 / -0.3451 | 0.2225 / -0.3126 | 0.2781 / -0.4518 | 0.4394 / -0.5129 |
| Uncertainty | LLaMA 3.3 70B | 0.4219 / -0.5518 | 0.2119 / -0.5621 | 0.3265 / -0.5071 | 0.4823 / -0.452 |
| Q-Daps | Avg-Plausibility | -0.2242 / 0.0272 | 0.4784 / -0.3 | 0.1869 / -0.2545 | 0.564 / -0.509 |
| Q-Daps | Entropy-Plausibility | 1.0888 / -0.9001 | 0.803 / -0.6181 | 0.9448 / -0.9636 | 0.7498 / -0.8818 |

### Ablation Study

| Configuration | MuSiQue d | QASC d | NQ d | TriviaQA d | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Without gold answer | 0.8325 | 0.5144 | 0.6319 | 0.6647 | Still outperforms all baselines without gold answers |
| With gold answer | 1.0888 | 0.803 | 0.9448 | 0.7498 | Higher candidate generation quality |
| Without debiasing | 0.894 | 0.5614 | 0.88 | 0.6511 | Entropy signal remains strong but slightly weaker |
| With debiasing | 1.0888 | 0.803 | 0.9448 | 0.7498 | Page view debiasing yields significant gains |
| Qwen 2.5 7B core | 0.8434 | 0.1465 | 0.2465 | 0.3162 | Small models are usable but more volatile |
| LLaMA 3.1 8B core | 0.5467 | 0.2484 | 0.3886 | 0.3481 | Resource-friendly configuration |
| LLaMA 3.3 70B core | 1.0888 | 0.803 | 0.9448 | 0.7498 | Strongest main configuration |

### Key Findings
- Entropy-Plausibility significantly outperforms Avg-Plausibility, indicating that difficulty stems from the uncertainty of the plausibility distribution rather than the average credibility of candidates.
- Listwise is more effective and cheaper than Pointwise and Pairwise: it requires only one prompt, whereas Pairwise generates millions of comparisons on full datasets.
- Q-Daps does not rely on gold answers: Cohen's d remains at 0.8325, 0.5144, 0.6319, and 0.6647 on MuSiQue/QASC/NQ/TriviaQA respectively without them.
- Popularity debiasing provides statistically significant gains (paired t-test $p=0.0356$, $t=3.6474$).
- In human evaluation, 6 evaluators showed an average recognition accuracy of 0.74 on questions labeled hard by Q-Daps vs. 0.68 on easy ones, indicating higher consistency for difficult labels.

## Highlights & Insights
- This paper shifts the focus of "difficulty" from surface text complexity to the structure of the answer space, closely aligning with LLM error mechanisms: when multiple incorrect answers are plausible, hallucination risks naturally increase.
- The unstable performance of direct prompt-based scoring suggests that "asking an LLM if it's hard" is less effective than having it explicitly expand the competitive candidate space.
- Popularity debiasing is a subtle but critical design choice. LLM candidate generation is biased toward well-known entities; without correction, difficulty scores might over-amplify "famous but wrong" answers.
- Q-Daps outputs candidate answers, plausibility, and difficulty scores, offering better interpretability than a single uncertainty score, which facilitates manual auditing or QA routing.

## Limitations & Future Work
- The method is better suited for questions with compact candidate answer sets (e.g., entities, dates, numbers, boolean, categories, multiple choice); open-ended long answers or subjective questions may be difficult for candidate set generation.
- Experiments only cover English; candidate generation, Wikipedia popularity, and plausibility calibration in multilingual or low-resource language contexts remain unverified.
- The pipeline depends on LLM-generated candidates and plausibility; underlying model biases will affect final difficulty scores.
- Popularity relies on Wikipedia page views, for which there may be no suitable external popularity proxy in domains like medicine, finance, or internal corporate knowledge.
- When the gold label of a dataset is based on implicit definitions or historical conventions, low entropy does not necessarily guarantee the question is easy, as noted in the error analysis.

## Related Work & Insights
- **vs readability metrics**: Flesch-Kincaid and Gunning-Fog only look at surface features like sentence/word length; Q-Daps directly characterizes the answer competition space, making it more suitable for LLM QA.
- **vs PopQA**: While PopQA uses entity popularity to approximate difficulty, Q-Daps treats popularity as a source of bias and uses it to calibrate candidate plausibility.
- **vs retrieval complexity**: Retrieval complexity focuses on whether a passage sufficiently supports an answer; Q-Daps focuses on whether incorrect candidates remain plausible even given the question itself.
- **vs uncertainty-based scoring**: The loss of generating the correct answer reflects model confidence but does not show specific distractors; Q-Daps' candidate list is more interpretable.

## Rating
- Novelty: ⭐⭐⭐⭐ Using candidate plausibility entropy for LLM-oriented difficulty estimation is a clear perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four QA datasets, ten model types, three scoring paradigms, human evaluations, and multiple ablations are solid.
- Writing Quality: ⭐⭐⭐⭐ Method diagrams and tables are complete; while some appendices are dense, the main narrative is clear.
- Value: ⭐⭐⭐⭐ Potential applications in hallucination risk, question routing, model selection, and educational assessment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RankLLM: Weighted Ranking of LLMs by Quantifying Question Difficulty](../../ICLR2026/llm_evaluation/rankllm_weighted_ranking_of_llms_by_quantifying_question_difficulty.md)
- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](zero-shot_large_language_models_for_automatic_readability_assessment.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models](scicustom_a_framework_for_custom_evaluation_of_scientific_capabilities_in_large_.md)

</div>

<!-- RELATED:END -->
