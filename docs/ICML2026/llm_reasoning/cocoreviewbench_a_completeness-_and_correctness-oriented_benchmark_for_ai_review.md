---
title: >-
  [Paper Note] CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers
description: >-
  [ICML 2026][LLM Reasoning][AI Reviewer] This paper proposes CoCoReviewBench, which transforms human reviews of 3,900 ICLR/NeurIPS papers into a more credible evaluation reference for AI reviewers through a two-step proce…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "AI Reviewer"
  - "Completeness"
  - "Correctness"
  - "Conflict Detection"
  - "Meta-Review"
date: 2026-05-08
content_hash: eb4c349c27d1ce89
---

# CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers

**Conference**: ICML 2026  
**arXiv**: [2605.07905](https://arxiv.org/abs/2605.07905)  
**Code**: https://github.com/hexuandeng/CoCoReviewBench (Available)  
**Area**: LLM Reasoning / AI Reviewing / Evaluation Benchmark  
**Keywords**: AI Reviewer, Completeness, Correctness, Conflict Detection, Meta-Review

## TL;DR
This paper proposes CoCoReviewBench, which transforms human reviews of 3,900 ICLR/NeurIPS papers into a more credible evaluation reference for AI reviewers through a two-step process: constructing category-based sub-benchmarks and filtering erroneous opinions using meta-reviews to adjudicate reviewer/author conflicts. It finds that existing AI reviewers still lag behind humans in correctness and thoroughness, while reasoning models show greater potential.

## Background & Motivation
**Background**: With the explosion in paper submissions and the decline in review quality, the research community has explored using LLMs as "AI Reviewers." This has led to two evaluation paradigms: LLM-as-a-judge without human references, and direct comparison with human reviews as a gold reference using metrics like BLEU/ROUGE/BERTScore or LLM matching.

**Limitations of Prior Work**: The first paradigm lacks expert signals and is prone to amplification by LLM self-biases. The second paradigm treats human reviews as "ground truth," yet human reviews are **neither complete nor always correct**. An average single reviewer covers only 5.10 out of 23 sub-categories, all reviewers combined cover 9.23 (40%), and 13% of papers show score discrepancies of $\geq 4$, 22% have reviewer-reviewer conflicts, and 76% have reviewer-author conflicts.

**Key Challenge**: Using incomplete and occasionally erroneous human reviews as a gold reference leads to two systematic biases: (1) Valid "unmentioned points" raised by AI are penalized as irrelevant; (2) AI models are rewarded for learning incorrect viewpoints from human reviews during evaluation.

**Goal**: (a) Construct an evaluation that does not unfairly penalize AI due to "incomplete" references; (b) filter out erroneous human opinions using expert signals (other reviewers, author rebuttals, meta-reviews); (c) re-evaluate various AI reviewer models on this credible reference to identify current gaps and directions.

**Key Insight**: Instead of trying to synthesize reviews "better than humans," it is preferable to **return to the multi-party discussion structure inherent in OpenReview**. When opinions on the same topic conflict among reviewers, the meta-review serves as a high-quality arbitration signal; when authors explicitly oppose an opinion, the meta-review can determine who is correct. These serve as **free expert annotations**.

**Core Idea**: Redefine "AI review evaluation" as **category-level matching + alignment with conflict-filtered human references**. The benchmark is split into 5 major categories and 23 sub-categories, scoring only when both parties cover the same category. Meta-reviews act as a judge to remove incorrect human references, followed by an LLM-as-a-judge multi-dimensional evaluation of the AI on the cleaned reference.

## Method

### Overall Architecture
The CoCoReviewBench pipeline consists of four steps (Figure 4): (1) Split each human review and corresponding author response into "atomic opinions" and assign tags from 23 categories; (2) Perform inter-reviewer conflict detection after clustering same-topic opinions, using meta-reviews for adjudication and discarding incorrect opinions; (3) Use meta-reviews to resolve reviewer-author conflicts; (4) Distill these steps into a Qwen3-8B model (ReviewSplit + ReviewClassify) to perform the same splitting and classification on the **tested AI reviews**, finally using GPT-5-Mini as a judge to calculate scores across five dimensions: Correctness, Thoroughness, Grounding, Verifiability, and Clarity, plus a cross-category Completeness coverage metric.

The dataset covers NeurIPS 2021-2024 + ICLR 2017-2025, with 300 papers sampled via stratified sampling per year for a total of 3,900 papers. It requires $\geq 3$ independent reviews and $\geq 75\%$ of reviews to have author responses, resulting in 14.1k reviews, 134.8k atomic opinions, 115.9k opinion clusters, and 108.6k "correct opinion" references.

### Key Designs

1.  **Category-level Benchmark**:
    - **Function**: Transitions the "incomplete human reference" issue from a "penalty" to a "skip," preventing reasonable AI opinions from being penalized just because humans did not mention them.
    - **Core Idea**: A two-level taxonomy is built based on NeurIPS 2025 review guidelines + ARR forms, comprising 5 major categories (Quality / Clarity / Significance / Originality / Policy) and 23 sub-categories. All reviewer opinions are aggregated into per-paper tag sets, creating a **sub-benchmark for each sub-category** that only includes papers where humans provided opinions in that category. Scoring occurs only when both the AI and human have opinions in that category. Results are reported at both paper-level and category-level granularities.
    - **Design Motivation**: Ours observed that AI was systematically scored lower in categories not covered by humans, even if the opinions were not wrong. Category-level matching eliminates this "coverage bias" and separately diagnoses "insufficient breadth" versus "insufficient depth."

2.  **Conflict-Based Error Verification**:
    - **Function**: Identifies and removes opinions in human references that have been discussed and proven incorrect, leading to a "cleaner" reference.
    - **Core Idea**: Sources are divided into two types: (a) inter-reviewer: opinions on the same topic are clustered, and an LLM detects conflicts if a cluster has $\geq 2$ opinions; meta-reviews then determine the correct one; (b) reviewer-author: pairs of reviewer opinions and author responses are analyzed to see if the author explicitly disagrees, with meta-reviews adjudicating. Each step uses independent LLM calls (aggregate / detect conflict / adjudicate). Opinions adjudicated as incorrect are removed but kept as "negative training signals."
    - **Design Motivation**: Relying solely on a strong LLM as a judge is error-prone in professional domains, whereas multi-party discussions on OpenReview are **natural expert annotations**. Conflict implies at least one party is wrong, and meta-reviews are high-level AC adjudications. While this signal is not perfect, Ours demonstrates it is more reliable than direct LLM correctness judgment or original reviews.

3.  **AI Review Postprocessing & Multi-dim Judge**:
    - **Function**: Compresses the expensive human review processing pipeline into 8B small models, allowing any tested AI review to be split and classified before multi-dimensional category-level scoring by an LLM-judge.
    - **Core Idea**: Qwen3-8B is trained via GRPO for ReviewSplit, sampling 32 trajectories per sample for augmentation and using the Omega Index for clustering correctness with a reward $R = \max(0.5, \text{OmegaIndex} + \mathbb{1}(\text{Correct Format}))$. ReviewClassify uses SFT to train the 8B model. The judge stage defines five 1-5 point dimensions: Correctness (alignment with remaining human opinions), Thoroughness (coverage depth), Grounding (pointing to paper locations), Verifiability (ease of verification), and Clarity (writing quality), plus Completeness (ratio of AI-covered categories to collective human-covered categories).
    - **Design Motivation**: Feeding every AI review to a strong LLM for splitting is too costly; the 8B distilled model reached 87.09% accuracy in human verification, approaching or exceeding some strong LLMs. Multi-dimensional scoring prevents systemic differences (e.g., AI clarity > human, but correctness < human) from being masked by a single total score.

### Loss & Training
Two-stage independent training: ReviewSplit is trained using GRPO on level-2 segmentation results, aiming to align the "same opinion" binary judgment with the Omega Index. ReviewClassify uses pure SFT to map atomic opinions to one of the 23 sub-categories. The pipeline uses leave-one-out consistency verification with 6 strong LLMs at each step, selecting the most consistent model for final annotation.

## Key Experimental Results

### Main Results
1/3 of the benchmark (1,300 papers) was sampled to test 18 models across closed-source, open-source reasoning, open-source non-reasoning, and specialized AI reviewer groups. Table values show differences relative to human references (+pos neg neg):

| Model Group | Old Metrics BLEU/ROUGE/BERT | Correct./Thoro. | Ground./Verify. | Clarity | Complete. |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5.2 (Strong Closed) | -1.93/-5.06/-1.31 | +0.36/+0.64 | +0.92/+0.78 | +0.32 | 84.49 |
| Gemini-3-Pro | -0.95/-1.12/-0.36 | +0.14/+0.16 | +0.69/+0.34 | +0.42 | 67.69 |
| QwQ-32B (Reasoning) | -1.38/-2.44/-0.63 | -0.01/+0.13 | +0.58/+0.02 | +0.27 | 79.83 |
| Qwen3-8B no-think | -0.87/-0.53/-1.10 | -0.28/-0.10 | -0.40/-0.58 | -0.07 | 72.28 |
| CycleReviewer-70B (Spec. Non-reas.) | -0.78/+0.34/-0.11 | -0.15/-0.22 | -0.55/-0.48 | +0.48 | 50.89 |
| DeepReviewer-14B (Spec. Reas.) | -1.11/-3.53/-0.09 | -0.17/+0.28 | +0.41/+0.41 | +0.17 | 81.98 |
| Human Baseline (leave-one-out) | 2.73/17.54/84.04 | 3.55/2.37 | 3.75/2.38 | 4.15 | 55.66 |

Counter-intuitive phenomenon: Under old metrics, **small non-reasoning models and specialized AI reviewers actually outperform GPT-5/Gemini-3**, indicating that BLEU/ROUGE mainly reward "surface-level similarity to human reviews" rather than actual correctness. The proposed paper-level LLM judge aligns better with comprehensive model capabilities.

### Ablation Study

| Configuration | Key Finding | Description |
| :--- | :--- | :--- |
| Strong-conflict papers ($\geq 5$ errors) vs weak-conflict | Meta-review avg. 3.24 vs 2.94 | Verifies meta-reviews provide valid arbitration signals when conflicts are significant, though Conflict Coverage is the weakest dimension. |
| Human review of 50 samples | Classify 85.45% / Cluster 93.41% / Inter-rev error 81.40% | Pipeline is generally reliable; reviewer-author error detection is 66.83% (dropping to 50% for rejected papers). |
| 6 Strong LLMs leave-one-out | High consistency across steps | Selecting the highest-scored model controls per-step noise. |
| Including "erroneous opinions" in reference | AI - Human correctness gap narrows | Proves AI absorbs human errors when fitting human reviews, posing a "learning bad habits" risk. |

### Key Findings
- Reasoning models have reached or exceeded humans in **Grounding / Verifiability**, suggesting LRM "chain-of-argument + context citation" capabilities directly translate to more verifiable reviews—this is the strongest directional conclusion: **Future AI reviews should prioritize reasoning models**.
- **AI reviews are "wide but shallow"**: Cross-category coverage usually exceeds a single reviewer but is lower than the collective set (Complete < 100), while single-category depth (Thoroughness) has not significantly improved compared to humans, especially for non-reasoning models.
- **Hallucination signals**: Despite raw text input, all models generate a small amount of Figure-related opinions (<0.05/paper), indicating a low-probability but persistent hallucination of non-existent visual details.
- **Category profile**: AI slightly outperforms humans in the Quality category (experiments, comparisons) but is weakest in Clarity and Policy.

## Highlights & Insights
- **Decoupled solutions for incompleteness and errors**: Category-level evaluation handles the former, while conflict-based filtering handles the latter. This methodology is clean and transferable to other "unreliable reference" evaluation tasks like code review.
- **Meta-reviews as neglected supervision**: Unlike prior works focused only on review text, this paper systematically uses meta-reviews as "high-level judges" for error detection, opening a new direction for using OpenReview’s full discussion structure.
- **8B distillation makes industrial evaluation viable**: Processing human reviews with strong LLMs is a one-time cost, but evaluating AI reviews requires repeated runs—distilling the pipeline into an 8B model reduces long-term costs by an order of magnitude.
- **Multi-dimensional and dual-granularity LLM-judge** reveals that a single score masks systematic differences; high clarity does not imply high correctness, and old versus new metrics give completely different rankings.

## Limitations & Future Work
- Authors admit conflict-based filtering only catches **explicit disagreements**: Instances where authors do not reply or use mild tones are missed.
- The evaluation still **depends on an LLM-judge (GPT-5-Mini)**, creating a risk of circular reasoning by using LLMs to evaluate LLMs, though this is mitigated by human validation.
- The 23-category taxonomy from NeurIPS 2025 + ARR **might lack granularity for non-NLP/ML conferences** (e.g., pure theory or CV sub-domains).
- Future directions: (1) Use identified erroneous opinions for negative training; (2) synthetically expand human references using coverage signals; (3) evaluate if AI can identify the "most critical" review points rather than just generating a full set.

## Related Work & Insights
- **vs PeerRead / NLPeer / ReviewMT / Re2**: These datasets only provide review text or meta/rebuttal data but **lack atomic opinions, category labels, conflict detection, and error annotations**.
- **vs RevUtil (Sadallah et al., 2025)**: RevUtil also performs atomic/fine-grained/category-level analysis but **lacks rebuttal/meta/conflict/error** signals. Ours adds the "correctness" dimension.
- **vs DeepReviewer / CycleReviewer / OpenReviewer / SEA**: Specialized AI reviewer models show inflated scores on old metrics but are outperformed by general closed-source LLMs on new metrics, indicating they "mimic human surface style" rather than improving true review capability.
- **vs LLM-as-a-judge (Xu et al., GRE-bench)**: That route ignores human references entirely; Ours takes a middle path—retaining human references but cleaning them.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining meta-review conflict arbitration with category-level skipping is a clear and well-structured approach.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large scale (3,900 papers), 18 model comparison, human verification, and 6 LLM consistency checks make for a complete chain of evidence.
- Writing Quality: ⭐⭐⭐⭐ Structure is clean and info-dense, though terms like Correctness/Thoroughness/Grounding are highly similar.
- Value: ⭐⭐⭐⭐⭐ Provides an industrial-grade credible evaluation for the AI reviewer sub-field and identifies reasoning models as the clear future direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations](floorplanqa_a_benchmark_for_spatial_reasoning_in_llms_using_structured_represent.md)
- [\[ICLR 2026\] HeurekaBench: A Benchmarking Framework for AI Co-scientist](../../ICLR2026/llm_reasoning/heurekabench_a_benchmarking_framework_for_ai_co-scientist.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ACL 2026\] LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals](../../ACL2026/llm_reasoning/llm_reasoning_as_trajectories_step-specific_representation_geometry_and_correctn.md)
- [\[ACL 2026\] CoAct: Co-Active LLM Preference Learning with Human-AI Synergy](../../ACL2026/llm_reasoning/coact_co-active_llm_preference_learning_with_human-ai_synergy.md)

</div>

<!-- RELATED:END -->
