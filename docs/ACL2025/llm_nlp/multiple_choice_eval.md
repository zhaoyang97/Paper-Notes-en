---
title: >-
  [Paper Note] Which of These Best Describes Multiple Choice Evaluation with LLMs?
description: >-
  [ACL 2025][LLM (Other)][Multiple-Choice Question Answering] Systematically demonstrates that MCQA, as a standard evaluation format for LLMs, suffers from three major categories of flaws: (1) format flaws—inability to test generative or subjective tasks, mismatch with real-world LLM use cases, and failure to fully assess depth of knowledge; (2) dataset flaws—leakage, unanswerability, shortcuts, and saturation; and (3) model behavior flaws—poor robustness, option bias…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Multiple-Choice Question Answering"
  - "MCQA Flaws"
  - "Psychometrics"
  - "Constructed Response"
  - "Item Response Theory"
date: 2026-05-08
content_hash: 5d99782f14458a83
---

# Which of These Best Describes Multiple Choice Evaluation with LLMs?

**Conference**: ACL 2025  
**arXiv**: [2502.14127](https://arxiv.org/abs/2502.14127)  
**Area**: LLM Evaluation Methodology  
**Keywords**: Multiple-Choice Question Answering, MCQA Flaws, Psychometrics, Constructed Response, Item Response Theory

## TL;DR

Systematically demonstrates that MCQA, as a standard evaluation format for LLMs, suffers from three major categories of flaws: (1) format flaws—inability to test generative or subjective tasks, mismatch with real-world LLM use cases, and failure to fully assess depth of knowledge; (2) dataset flaws—leakage, unanswerability, shortcuts, and saturation; and (3) model behavior flaws—poor robustness, option bias, and unfaithful explanations. Systematic remedies such as Constructed Response, Explanation MCQA, and IRT analysis are proposed by borrowing insights from psychometrics.

## Background & Motivation

**Background**: MCQA is the standard paradigm for LLM evaluation, widely popular due to its simplicity, ease of grading, and similarity to human testing. Specifically, MCQA accounts for 32% of tasks in HELM, 71% in the GPT-4 technical report, and 79% in the OpenLLM Leaderboard.

**Limitations of Prior Work**: (1) Over 90% of real-world LLM queries from users are open-ended generation (e.g., coding, writing, and explanation), causing a severe mismatch with MCQA evaluation; (2) MCQA datasets widely suffer from leakage, annotation errors, and shortcuts (numerous errors have been identified in MMLU); (3) LLMs are highly sensitive to option labels, ordering, and phrasing, leading to contradictory rankings under different evaluation setups; (4) Although psychometrics has accumulated decades of best practices for MCQA, the NLP community has largely ignored them.

**Key Challenge**: MCQA is popular because of its "simplicity," yet the most critical attribute of an evaluation is not its simplicity, but its ability to predict the system's actual performance when deployed.

**Goal**: To argue that MCQA is not the gold standard for LLM evaluation and to introduce systematic improvement frameworks from psychometrics.

**Core Idea**: MCQA is flawed but fixable—drawing from psychometrics, it can be systematically improved across three levels: formats, datasets, and scoring.

## Method

### Overall Architecture

The paper organizes its arguments across three levels: format issues (inherent limitations of MCQA) $\rightarrow$ dataset issues (quality of existing MCQA datasets) $\rightarrow$ model behavior issues (abnormal behaviors of LLMs on MCQA). Each level is thoroughly mapped to corresponding remediation strategies from psychometrics.

### Key Designs

1. **Format Level: Inherent Limitations of MCQA and Generative Alternatives**
    - **"Select the Best Answer" is too rigid**: A single gold-standard answer cannot evaluate subjective tasks (such as common sense, ethics, or culture). In common-sense MCQs, users find distractors more reasonable than the gold standard in 20% of cases. Additionally, selection $\neq$ generation; an LLM's validation and generation capabilities often diverge.
    - **Constructed Response (CR)**: Removing the options to let the LLM generate a short-form answer, corresponding to constructed responses in pedagogy, which better exposes gaps in knowledge. Existing MCQs can be directly converted by dropping the choices.
    - **Explanation MCQA (E-MCQA)**: Requiring the model to provide an explanation alongside the chosen answer. This verifies the faithfulness of the reasoning process and supports partial credit for subjective tasks, analogous to "show your work" in education.
    - **Design Motivation**: Real-world LLM use cases (coding, writing, explanation) are generative in nature; hence, evaluation paradigms should match these scenarios.

2. **Dataset Level: Four Quality Issues and Remediation Strategies**
    - **Leakage**: GPT-3 has encountered 45% of the RACE test set. Solution: Continuously updated "dynamic item pools" (analogous to annual exam updates in educational testing).
    - **Unanswerability**: Label errors, multiple correct options, and ambiguities make questions unanswerable (many such errors have been found in MMLU). Solution: Employing pedagogical MCQ writing checklists and guidelines to systematically validate each question.
    - **Shortcuts**: LLMs can achieve high accuracy using only the options (without reading the question stem), indicating spurious correlation leaks. Solution: Unified design (generating parts of the item from the same source/method) and Contrast Sets to ensure the model attends to all inputs.
    - **Saturation**: Model performance continues to rise, causing datasets to lose discriminative power. Solution: Using Item Response Theory (IRT) to filter and retain highly difficult and highly discriminative items; utilizing adversarial data collection to construct questions that are hard for models but simple for humans.

3. **Model Behavior Level: Three Anomalies and Evaluation Lessons**
    - **Poor Robustness**: Changing option order, labels, or phrasing causes the answer to flip, reflecting data leakage or bias rather than actual capability.
    - **Option Bias**: LLMs choose answers based on position, label symbols, or key phrases (e.g., "none of the above") rather than semantic content.
    - **Unfaithful Explanation**: Even when selecting the correct answer, the LLM may provide logically inconsistent explanations. Furthermore, the quality of their generated explanations often surpasses that of crowdsourced annotators, which easily misleads users.
    - **Design Motivation**: These issues inherently stem from flaws in both the MCQA format and datasets. Rectifying the first two layers yields better diagnostics for these behavioral issues.

## Key Experimental Results

### Overrepresentation of MCQA in LLM Evaluations

| Evaluation Platform | Proportion of MCQA Tasks | Real-World User MCQA Demand |
|---------|:-----------:|:---------------:|
| HELM | 32% | ~7% (ShareGPT) |
| GPT-4 Technical Report | 71% | <6.3% (WildChat) |
| OpenLLM Leaderboard | 79% | - |

### Case Studies of Quality Issues in MCQA Datasets

| Issue Type | Representative Case | Impact |
|---------|--------|-----|
| Leakage | 45% of the RACE test set was seen by GPT-3 | Confounds memorization and generalization |
| Unanswerability | MMLU exhibits substantial label errors and ambiguities | Overestimates model accuracy |
| Shortcuts | High scores can be achieved on HellaSwag using options only | Fails to test genuine understanding |
| Saturation | Multiple models exceed 90% on classic benchmarks | Fails to differentiate model capabilities |

### Key Findings

- Under Constructed Response, both students (and models) perform worse than in MCQA, demonstrating that MCQA indeed overestimates knowledge levels.
- IRT can screen for both difficult items and defective items (e.g., negative discrimination indicative of likely annotation errors).
- Adversarial data collection (gamified) can produce datasets that retain high difficulty over the long term.
- Calibrated scoring (confidence-based scoring, negative marking, or elimination-style testing) can suppress guessing behaviors in models.

## Highlights & Insights

1. **Cross-disciplinary Perspective of Psychometrics $\times$ NLP**: From the origins of MCQA in 1914 to IRT, Bloom's Taxonomy, and adversarial examinations, this work systematically incorporates a century of educational measurement research into the NLP community.
2. **"MCQA Evaluates Verification, Not Generation"**: Verification and generation represent distinct abilities; consequently, MCQA-driven leaderboards fail to accurately reflect an LLM's true capacity to assist users in real tasks.
3. **Practical Improvement Roadmap**: Rather than advocating for the outright abandonment of MCQA, the paper proposes layered remedies across formatting (CR/E-MCQA), data (checklists, contrast sets, and IRT), and scoring (calibrated scoring and partial credit)—each of which can be adopted independently.
4. **"Benchmarking 101" Design Guidelines**: The paper concludes with comprehensive procedural recommendations for designing evaluation benchmarks from scratch.

## Limitations & Future Work

1. As a position paper, it does not present new experimental data or benchmark implementations.
2. The grading reliability of generative alternatives (CR/E-MCQA) still requires verification at scale.
3. IRT requires extensive model evaluation data to fit parameters, limiting its applicability in small-scale scenarios.
4. In certain domains (e.g., bar or medical licensing exams), MCQA holds irreplaceable institutional value, which is not fully addressed.
5. Developing dynamic item pools is highly challenging in practice, requiring substantial and continuous item-writing resources.

## Related Work & Insights

- Bloom's Taxonomy of cognitive domains (Remembering $\rightarrow$ Understanding $\rightarrow$ Applying $\rightarrow$ Analyzing $\rightarrow$ Evaluating $\rightarrow$ Creating) provides a more instructive framework for evaluation design than simple "accuracy."
- IRT remains an emerging application in NLP; multidimensional IRT (MIRT) can identify a model's strengths and weaknesses across diverse reasoning dimensions.
- The gamified design of adversarial data collection (such as Quiz Bowl formats) warrants broader popularization.
- Issues confronting multimodal and multilingual MCQA (translation error propagation, cultural bias) apply equally to other evaluation formats.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐: Offers a unique interdisciplinary perspective, systematically introducing educational measurement into NLP evaluation discussions.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Features a clear three-layer structure where each issue is backed by empirical data and mapped to solutions.
- **Writing Quality** ⭐⭐⭐⭐⭐: Smooth flow, elegant structure, and outstanding headings.
- **Value** ⭐⭐⭐⭐: Extremely valuable in providing strategic directions for the LLM evaluation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options](llm_robustness_incorrect_mcq.md)
- [\[ACL 2025\] Which Demographics Do LLMs Default to During Annotation?](which_demographics_do_llms_default_to_during_annotation.md)
- [\[ACL 2025\] SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](skillverse_tree_eval.md)
- [\[ACL 2025\] Do LLMs Give Psychometrically Plausible Responses in Educational Assessments?](do_llms_give_psychometrically_plausible_responses_in_educational_assessments.md)
- [\[ACL 2025\] ATRIE: Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation](atrie_legal_interpretation.md)

</div>

<!-- RELATED:END -->
