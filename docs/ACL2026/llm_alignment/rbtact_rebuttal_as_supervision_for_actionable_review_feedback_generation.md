---
title: >-
  [Paper Note] RbtAct: Rebuttal as Supervision for Actionable Review Feedback Generation
description: >-
  [ACL 2026][LLM Alignment][Author rebuttal] RbtAct treats author rebuttals as implicit supervision to identify "which review comments actually prompt revisions." The authors construct 75…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "Author rebuttal"
  - "Actionable review feedback"
  - "Preference optimization"
  - "DPO"
  - "Peer review dataset"
date: 2026-05-08
content_hash: 912b8efbc10e49db
---

# RbtAct: Rebuttal as Supervision for Actionable Review Feedback Generation

**Conference**: ACL 2026  
**arXiv**: [2603.09723](https://arxiv.org/abs/2603.09723)  
**Code**: arXiv page indicates RbtAct; public repository URL not resolved in cache  
**Area**: LLM Alignment / Academic Peer Review / Feedback Generation  
**Keywords**: Author rebuttal, Actionable review feedback, Preference optimization, DPO, Peer review dataset

## TL;DR
RbtAct treats author rebuttals as implicit supervision to identify "which review comments actually prompt revisions." The authors construct 75,000 review-rebuttal segment-level mappings and train an 8B model using SFT+DPO to generate more specific and actionable peer review feedback.

## Background & Motivation
**Background**: LLMs have begun to participate in scientific writing and peer review, capable of generating full reviews from paper drafts or improving feedback coverage through multi-agent systems or fine-tuning. Existing work focuses more on "resembling a review": whether the language is fluent, whether strengths and weaknesses are mentioned, and whether major paper sections are covered.

**Limitations of Prior Work**: Truly useful reviews do not just point out "insufficient experiments" or "unclear writing" but inform authors exactly what to modify, what to supplement, and how to revise. LLM-generated reviews are often structurally complete but substantively generic, providing templated suggestions that are difficult for authors to act upon.

**Key Challenge**: The value of review feedback stems from the authors' subsequent actions. However, standard review datasets provide only the review text itself, with no indication of whether a specific comment prompted actual modifications. Consequently, models can mimic the language of a reviewer but lack supervision signals regarding "which comments will be adopted by authors."

**Goal**: This paper aims to transform author reactions in rebuttals into training signals to learn to generate single, focused, and perspective-specific review comments. The task input is the full paper and a specified perspective; the output is a review segment in a weakness/question style.

**Key Insight**: The authors observe that rebuttals naturally record how authors respond to reviewers: some comments lead to completed revisions, some result in future plans, while others are met with defense or deflection. Although this "author uptake" is noisy, it serves as a proxy label for actionability in large-scale public review data.

**Core Idea**: Use mappings between review segments and rebuttal segments to convert author responses into preference rankings, guiding the model toward generating feedback more likely to trigger specific modifications.

## Method
The RbtAct method involves two layers: first constructing a review-rebuttal segment-level dataset, then converting rebuttal impact into preference data for training. Instead of generating long, full reviews, the task is narrowed to "generating a focused comment given a paper and a specific review perspective." This setup reduces evaluation ambiguity and allows each feedback segment to align with specific responses in the rebuttal.

### Overall Architecture
The input consists of a full paper and a target perspective (e.g., Experiments, Novelty, Writing, or Reproducibility). The system extracts papers, reviews, and author rebuttals from ICLR 2024 OpenReview data and converts PDFs to Markdown. It focuses only on the weakness and question sections of reviews, splitting them into atomic critique segments and creating one-to-one mappings with rebuttal spans.

After mapping, each review segment receives two labels: a review perspective (describing which aspect of the paper the comment addresses) and a rebuttal impact (describing the degree of author action in response to the comment). The training phase first uses ReviewSeg-SFT-13K for supervised fine-tuning to teach Llama-3.1-8B-Instruct to generate review segments by perspective. This is followed by DPO using ReviewPref-DPO-22K, where segments leading to stronger author actions are treated as preferred outputs.

### Key Designs
1. **RMR-75K Segment-level Mapping Dataset**:
    - **Function**: Aligns specific reviewer points with corresponding rebuttal responses to provide trainable labels for actionability.
    - **Mechanism**: Weaknesses/questions are split into single focus points using structural cues or GPT-5; review segments are then aligned to rebuttal spans via explicit anchors and semantic matching; finally, a one-to-one match is selected greedily based on confidence.
    - **Design Motivation**: The granularity of a whole review versus a whole rebuttal is too coarse to determine which specific comment brought about which response. Segment-level mapping makes "which suggestion the author adopted" an observable signal.

2. **Dual Labels for Perspective and Impact**:
    - **Function**: Simultaneously controls the feedback topic and measures the actionability of the feedback.
    - **Mechanism**: Review segments are categorized into seven types: Experiments, Evaluation, Reproducibility, Novelty, Theory, Writing, and Presentation. Rebuttal impact is categorized into five levels: CRP, SRP, VCR, DWC, and DRF, representing completed revisions, specific revision plans, vague commitments, defense without change, and deflection, respectively.
    - **Design Motivation**: A single paper may have various issues across experiments, theory, and writing. Preference ranking is only valid when compared within the same paper and same perspective to avoid confounding topic differences.

3. **DPO Preference Optimization Based on Rebuttal Impact**:
    - **Function**: Prioritizes the generation of review feedback that is more likely to trigger actual modifications.
    - **Mechanism**: Pairs are constructed within the same paper and perspective, ranked as `CRP > SRP > VCR > DWC > DRF`. High-impact review segments are treated as "chosen" and low-impact segments as "rejected." The DPO objective increases the probability of chosen vs. rejected outputs, with a small SFT loss mixed in to prevent perspective control drift in long contexts.
    - **Design Motivation**: A rebuttal is not an explicit human score, but it reflects real author reactions. Converting it into pairwise preferences is more robust than predicting a coarse actionability score directly.

### Loss & Training
The model is based on Llama-3.1-8B-Instruct. The SFT stage uses ReviewSeg-SFT-13K (13,300 samples across 4,637 papers, ~1,900 per perspective). The DPO stage uses ReviewPref-DPO-22K (21,822 preference pairs across 4,825 papers). DPO uses the standard Bradley-Terry form, focusing on increasing the difference in $\log \pi_\theta(y_w|x)-\log \pi_\theta(y_l|x)$ relative to the reference model. A regularization term of $\lambda=0.1$ for positive sample SFT is included to mitigate output drift caused by preference training.

## Key Experimental Results

### Main Results
Evaluation was conducted on an ICLR 2025 subset, with human evaluation on 50 papers and LLM-as-a-judge point evaluation on 105 papers. RbtAct's primary advantages are concentrated in Actionability and Specificity, while remaining competitive in Groundedness and Relevance.

| System | Human Action. | Human Spec. | Human Ground. | Human Rel. | LLM Action. | LLM Spec. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RbtAct | 3.46 | 4.08 | 4.30 | 4.76 | 3.38 | 3.70 |
| RbtAct-SFT | 3.28 | 4.01 | 4.16 | 4.70 | 3.18 | 3.59 |
| GPT-5-chat | 3.38 | 4.04 | 4.35 | 4.98 | 3.28 | 3.66 |
| DeepSeek-V3.2 | 3.15 | 3.98 | 4.22 | 4.88 | 3.13 | 3.56 |
| Llama-3.1-70B | 3.22 | 3.95 | 4.18 | 4.65 | 3.11 | 3.54 |
| DeepReviewer-14B | 3.27 | 3.96 | 4.28 | 4.75 | 3.23 | 3.48 |

RbtAct also leads strong baselines in pairwise actionability comparisons. The win rate below indicates the proportion of the "row model" defeating the "column model."

| Opponent | RbtAct Win Rate | GPT-5-chat Win Rate | DeepSeek-V3.2 Win Rate |
| :--- | :--- | :--- | :--- |
| GPT-5-chat | 57.1% | - | 44.8% |
| DeepSeek-V3.2 | 63.8% | 55.2% | - |
| Llama-3.1-70B | 61.9% | 57.1% | 54.3% |
| MARG | 68.6% | 62.9% | 59.0% |
| LimGen | 76.2% | 71.4% | 68.6% |

### Ablation Study
The most direct ablation compares SFT-only with SFT+DPO. The gains from DPO are modest but stable, specifically focused on actionability and specificity without sacrificing groundedness for sharper comments.

| Configuration | Human Action. | LLM Action. | Human Spec. | LLM Spec. | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RbtAct-SFT | 3.28 | 3.18 | 4.01 | 3.59 | Learns distribution of real review segments only |
| RbtAct | 3.46 | 3.38 | 4.08 | 3.70 | Adds DPO based on rebuttal impact |
| Gain | +0.18 | +0.20 | +0.07 | +0.11 | Preference optimization mainly improves actionability |

Quality control was also applied to data construction, showing that training signals are not merely coarse captured text.

| Data/Validation Item | Value | Meaning |
| :--- | :--- | :--- |
| RMR-75K mappings | 75,542 | Number of review segment to rebuttal span mappings |
| Papers covered | 4,825 | From ICLR 2024 OpenReview |
| Auto-mapping F1 | 0.91 | Alignment with human-annotated span overlap |
| Mapping IAA | κ=0.80 | High inter-annotator agreement |
| Perspective label accuracy | ~92% | Match between auto-labels and human judgment |
| Impact label accuracy | 89% | Reliability of rebuttal impact labels |

### Key Findings
- While the gains from rebuttal-derived DPO are not as extreme as switching to a larger model, they allow an 8B model to outperform strong baselines like GPT-5-chat, DeepSeek-V3.2, and Llama-3.1-70B in actionability.
- Actionability and specificity improvements occur without significant drops in groundedness or relevance, suggesting the model is not gaining scores by fabricating "tougher" suggestions.
- Pairwise results demonstrate more advantage than pointwise ones: RbtAct achieves win rates over 65% against dedicated review generation methods like LimGen, MARG, and DeepReviewer.
- The key to this task is not just generating a review, but learning "which kind of feedback will be taken seriously by the author." This turns peer review data from an imitation target into a source of preference supervision.

## Highlights & Insights
- The paper redefines the rebuttal as a training signal rather than just a dialogue record or analysis object. This perspective is inspiring: many "subsequent reactions" in academic workflows can serve as implicit preference labels.
- Segment-level generation is a pragmatic design. Asking a model to generate an entire review is difficult to evaluate and prone to mixing multiple issues; single, perspective-conditioned feedback is better for alignment, training, and human evaluation.
- The ranking of impact categories concretizes actionability. It is no longer a subjective impression but a behavioral signal of "whether the author has revised, plans to revise, or is defending."
- This methodology can be transferred to proposal reviews, code reviews, and pedagogical feedback: anywhere a "comment-response-action" log exists, similar preference data can be constructed.

## Limitations & Future Work
- Rebuttals only reflect short-term author responses and do not guarantee the final paper actually completed the revisions; some authors may make strategic promises, and high-quality suggestions may be passed over due to time constraints.
- Data primarily comes from CS conferences in the OpenReview style; generalization to journals, non-English communities, or fields without public rebuttals requires re-validation.
- Model-generated suggestions may be specific but infeasible; current evaluation does not strictly verify if suggestions are supported by the paper, code, and data simultaneously.
- Preference ranking assumes CRP is always better than SRP or VCR, but some defensive responses may be due to reviewer misunderstanding rather than poor feedback quality.
- Future work could combine rebuttals with camera-ready diffs, experimental supplements, and author revision logs to build actionability signals closer to real revision outcomes.

## Related Work & Insights
- **vs. ARIES**: ARIES focuses on the link between review comments and paper edits; this work converts review-rebuttal segment mappings into trainable generation preferences. The former is behavioral analysis, while the latter is model optimization.
- **vs. DISAPERE / JitsuPeer**: These works have sentence-level review-rebuttal relationship annotations, but at a smaller scale and with different labeling goals. RMR-75K is larger and explicitly adds perspective and impact categories.
- **vs. MARG / DeepReviewer / LimGen**: These methods improve feedback quality through prompting, multi-agent systems, or review generation models; RbtAct differs by using author reactions to define "good feedback."
- **Insight**: Alignment research does not have to rely solely on manual preference scoring; logs of subsequent behaviors in real workflows can provide low-cost preference signals.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Using rebuttals as actionability preference supervision is highly innovative; the task setting is more focused than generic review generation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Includes human eval, LLM judge, pairwise, auto-metrics, and data validation, though actual revision outcomes are not yet included.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear motivation, complete data pipeline, and experimental tables support main conclusions; some reliance on detailed appendices.
- **Value**: ⭐⭐⭐⭐⭐ High reuse value for academic review assistance, feedback generation, and workflow preference learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback](wildfeedback_aligning_llms_with_in-situ_user_interactions_and_feedback.md)
- [\[ACL 2026\] PERSA: Reinforcement Learning for Professor-Style Personalized Feedback with LLMs](persa_reinforcement_learning_for_professor-style_personalized_feedback_with_llms.md)
- [\[ACL 2026\] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms](towards_bridging_the_reward-generation_gap_in_direct_alignment_algorithms.md)
- [\[ICML 2026\] Reward Modeling from Natural Language Human Feedback](../../ICML2026/llm_alignment/reward_modeling_from_natural_language_human_feedback.md)
- [\[ACL 2026\] ModeX: Evaluator-Free Best-of-N Selection for Open-Ended Generation](modex_evaluator-free_best-of-n_selection_for_open-ended_generation.md)

</div>

<!-- RELATED:END -->
