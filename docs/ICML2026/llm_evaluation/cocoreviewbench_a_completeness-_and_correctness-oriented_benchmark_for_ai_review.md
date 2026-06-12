---
title: >-
  [Paper Note] CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers
description: >-
  [ICML 2026][LLM Evaluation][AI Reviewer] This paper introduces CoCoReviewBench, which transforms human reviews of 3,900 ICLR/NeurIPS papers into a more reliable AI review evaluation reference through a two-step process:…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "AI Reviewer"
  - "Completeness"
  - "Correctness"
  - "Conflict Detection"
  - "Meta-Review"
date: 2026-05-08
content_hash: e65cc9319d4bb8ea
---

# CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers

**Conference**: ICML 2026  
**arXiv**: [2605.07905](https://arxiv.org/abs/2605.07905)  
**Code**: https://github.com/hexuandeng/CoCoReviewBench (available)  
**Area**: LLM Reasoning / AI Reviewing / Benchmarking  
**Keywords**: AI Reviewer, Completeness, Correctness, Conflict Detection, Meta-Review

## TL;DR
This paper introduces CoCoReviewBench, which transforms human reviews of 3,900 ICLR/NeurIPS papers into a more reliable AI review evaluation reference through a two-step process: (1) constructing sub-benchmarks by category, and (2) filtering erroneous opinions by arbitrating reviewer/author conflicts using meta-reviews. The study finds that current AI reviewers still lag behind humans in correctness and thoroughness, while reasoning models show greater potential.

## Background & Motivation
**Background**: With the surge in submissions and declining review quality, the research community has explored using LLMs as "AI reviewers," resulting in two evaluation paradigms: one treats LLMs as judges without human references; the other directly uses human reviews as gold references, scoring with BLEU/ROUGE/BERTScore or LLM-based matching.

**Limitations of Prior Work**: The first paradigm lacks expert signals and is prone to LLM bias amplification; the second treats human reviews as "truth," but human reviews are **neither complete nor always correct**—an individual reviewer covers only 5.10 out of 23 subcategories on average, and all reviews combined cover just 9.23 (40%). Moreover, 13% of papers have ≥4-point disagreements, 22% have reviewer-reviewer conflicts, and 76% have reviewer-author conflicts.

**Key Challenge**: Using incomplete and sometimes erroneous human reviews as gold references introduces two systematic biases: (1) AI-generated valid points not mentioned by humans are wrongly penalized as irrelevant; (2) AI models learn erroneous human opinions and are rewarded for them during evaluation.

**Goal**: (a) Construct an evaluation that does not penalize AI for "missing" references; (b) Use expert signals (other reviewers, author rebuttals, meta-reviews) to filter out erroneous human opinions; (c) Re-evaluate various AI review models on this trustworthy reference to clarify current gaps and directions.

**Key Insight**: Rather than synthesizing reviews "better than humans," it is preferable to **leverage OpenReview's multi-party discussion structure**—when multiple reviewers discuss the same topic and disagree, the meta-review naturally serves as a high-level arbitration signal; when authors explicitly oppose a point, the meta-review can also adjudicate. These are **free expert annotations**.

**Core Idea**: Redefine "AI review evaluation" as **category-level matching + human reference alignment after conflict filtering**—split the benchmark into 5 major and 23 subcategories, scoring only when both AI and human reviews cover the same category; use meta-reviews as arbiters to remove erroneous human opinions, then use LLM-as-a-judge to evaluate AI on clean references across multiple dimensions.

## Method

### Overall Architecture
The CoCoReviewBench pipeline consists of four steps (Figure 4): (1) Decompose each human review and corresponding author response into "atomic opinions" and label them with 23 categories; (2) Cluster opinions by topic and perform inter-reviewer conflict detection—if conflicts exist, use meta-reviews to arbitrate and discard erroneous opinions; (3) Similarly, use meta-reviews to adjudicate reviewer-author conflicts; (4) Distill these steps into a Qwen3-8B model (ReviewSplit + ReviewClassify) to perform the same decomposition and classification on **AI-generated reviews**, then use GPT-5-Mini as judge to score Correctness / Thoroughness / Grounding / Verifiability / Clarity for each category, and add a Completeness metric for cross-category coverage.

The dataset covers NeurIPS 2021-2024 + ICLR 2017-2025, stratified sampling 300 papers per year for a total of 3,900 papers, with ≥3 independent reviews and ≥75% reviews having author responses. This yields 14.1k reviews, 134.8k atomic opinions, 115.9k opinion clusters, and 108.6k "correct opinion" references.

### Key Designs

1. **Category-level Benchmark**:

    - **Function**: Converts the "incomplete human reference" issue from "penalization" to "skipping," avoiding penalizing AI for reasonable opinions not mentioned by humans.
    - **Mechanism**: Based on NeurIPS 2025 review guidelines + ARR forms, a two-level taxonomy is built—5 major categories (Quality / Clarity / Significance / Originality / Policy) with 23 subcategories. All reviewer opinions are aggregated into a per-paper label set, and **a sub-benchmark is constructed for each subcategory**, including only papers with human opinions in that category. Scoring is performed only when both AI and human reviews cover the category. Both paper-level (all opinions scored together) and category-level (each category scored separately and averaged) results are reported.
    - **Design Motivation**: Empirical results show AI is systematically scored lower in categories not covered by humans, though such opinions may be valid but overlooked by reviewers; category-level matching eliminates this "coverage bias" and separates "insufficient breadth" from "insufficient depth in a single category" for diagnosis.

2. **Conflict-Based Error Verification via Meta-Review Arbitration**:

    - **Function**: Identifies and removes **already-discussed and genuinely erroneous** human opinions, resulting in cleaner references.
    - **Mechanism**: Two sources—(a) inter-reviewer: cluster opinions by topic, and when ≥2 opinions exist in a cluster, use LLM to detect conflicts; if a conflict exists, use meta-review to decide which is correct, retaining the longest correct opinion; (b) reviewer-author: treat reviewer opinion + author response as a pair, detect if the author explicitly disagrees, and if so, use meta-review to arbitrate. All three steps use independent LLM calls (aggregate / detect conflict / adjudicate). All opinions adjudicated as incorrect are removed from the reference set but retained as "negative training signals."
    - **Design Motivation**: Directly using "strong LLMs as judges" in specialized domains is error-prone, while OpenReview's multi-party discussions provide **natural expert annotations**—conflict implies at least one side is wrong; meta-review is a high-level AC decision. Although not perfect (strong-conflict papers have a 4-dimension average score of only 3.24/5, with weakest Conflict Coverage at 2.85), the authors argue this is more reliable than "LLM direct correctness judgment" or "raw reviews," and 22%/76% of papers have erroneous opinions identified in the two conflict types, sufficient for effective filtering.

3. **AI Review Postprocessing & Multi-dim Judge**:

    - **Function**: Compresses the costly human review processing pipeline into an 8B small model, enabling decomposition and classification of any AI-generated review, which is then scored by an LLM-judge across multiple dimensions at the category level.
    - **Mechanism**: ReviewSplit is trained on Qwen3-8B using GRPO—each sample is augmented with 32 trajectories, Omega Index measures clustering accuracy, and reward is $R = \max(0.5, \text{OmegaIndex} + \mathbb{1}(\text{Correct Format}))$; ReviewClassify uses SFT to map each atomic opinion to one of 23 subcategories (non-thinking mode, which empirically also improves thinking mode). The judge phase defines five 1-5 point dimensions: Correctness (alignment with remaining human opinions), Thoroughness (coverage completeness), Grounding (explicit reference to paper location), Verifiability (can be checked), Clarity (writing clarity), with both paper-level and category-level granularity; Completeness = AI-covered categories / all reviewer-covered categories for the paper × 100.
    - **Design Motivation**: Running strong LLMs for decomposition and classification on every AI review is too costly; the 8B distilled model achieves 87.09% fully correct classification on 50 manually validated papers, approaching or surpassing some strong LLMs. Multi-dimensional scoring avoids "single overall score" masking systematic differences (e.g., AI clarity > human, but correctness < human).

### Loss & Training
Two-stage independent training: ReviewSplit uses GRPO to train on secondary segmentation results, aiming for 0/1 alignment with Omega Index for "whether two sentences are the same opinion"; ReviewClassify uses pure SFT to map each atomic opinion to one of 23 subcategories. The entire pipeline uses 6 strong LLMs for leave-one-out consistency validation at each step, selecting the model with highest consistency for final annotation.

## Key Experimental Results

### Main Results
On a random 1/3 (1,300 papers) of the 3,900-paper benchmark, 18 models are tested, covering closed-source, open-source reasoning, open-source non-reasoning, and dedicated AI reviewer models. Table values are differences relative to human references (+ positive, - negative):

| Model Group | Old Metrics BLEU/ROUGE/BERT | Correct./Thoro. | Ground./Verify. | Clarity | Complete. |
|-------------|-----------------------------|-----------------|-----------------|---------|-----------|
| GPT-5.2 (closed strong) | -1.93/-5.06/-1.31 | +0.36/+0.64 | +0.92/+0.78 | +0.32 | 84.49 |
| Gemini-3-Pro | -0.95/-1.12/-0.36 | +0.14/+0.16 | +0.69/+0.34 | +0.42 | 67.69 |
| QwQ-32B (reasoning) | -1.38/-2.44/-0.63 | -0.01/+0.13 | +0.58/+0.02 | +0.27 | 79.83 |
| Qwen3-8B no-think | -0.87/-0.53/-1.10 | -0.28/-0.10 | -0.40/-0.58 | -0.07 | 72.28 |
| CycleReviewer-70B (dedicated non-reasoning) | -0.78/+0.34/-0.11 | -0.15/-0.22 | -0.55/-0.48 | +0.48 | 50.89 |
| DeepReviewer-14B (dedicated reasoning) | -1.11/-3.53/-0.09 | -0.17/+0.28 | +0.41/+0.41 | +0.17 | 81.98 |
| Human Baseline (leave-one-out) | 2.73/17.54/84.04 | 3.55/2.37 | 3.75/2.38 | 4.15 | 55.66 |

Most counterintuitive finding: **Non-reasoning small models and dedicated AI reviewers outperform GPT-5/Gemini-3 on old metrics**, indicating BLEU/ROUGE mainly rewards "surface similarity to human reviews" rather than true correctness. The newly proposed paper-level LLM judge (Paper. column from +0.07 to +0.90) aligns with overall model capability.

### Ablation Study

| Configuration | Key Findings | Notes |
|---------------|-------------|-------|
| Strong-conflict papers (≥5 erroneous opinions) vs weak-conflict | Meta-review 4-dim avg 3.24 vs 2.94 | Confirms meta-review provides effective arbitration in significant conflicts, but Conflict Coverage is weakest (only 2.85), so it can only serve as a **coarse-grained** judge |
| Human 50-paper annotation check | Classification 85.45% / Clustering 93.41% / inter-reviewer error detection 81.40% | Pipeline is generally reliable; reviewer-author error detection only 66.83% (drops to 50% for rejected papers), currently the largest source of uncertainty |
| Strong LLM × 6 leave-one-out | High cross-model consistency at each step | Final annotation generated by the highest-scoring model, single-step noise is controllable |
| Including "erroneous opinions" as references | AI-human correctness gap narrows significantly | Inversely demonstrates that AI also absorbs human errors when fitting human reviews, posing a risk of "learning bad habits" |

### Key Findings
- Reasoning models have **matched or surpassed humans in Grounding / Verifiability**, indicating that LRM's "chain-of-thought + context citation" ability directly translates to more verifiable reviews—this is the strongest directional conclusion: **future AI reviewers should prioritize reasoning models**.
- **AI reviews are "broad but shallow"**: cross-category coverage generally exceeds individual reviewers but is lower than the union of all reviewers (Complete < 100), and single-category depth (Thoroughness) shows no significant improvement over humans, especially for non-reasoning models.
- **Hallucination signals**: Despite inputting only plain text, all models generate a small number of Figure-related opinions per paper (<0.05/paper), indicating a low-probability but persistent hallucination in AI reviews.
- **Category profile**: AI slightly outperforms humans in Quality (experiments, comparisons); weakest in Clarity and Policy, suggesting future work should focus on these two training signals.

## Highlights & Insights
- **Separately addresses "incomplete reference → false penalty" and "erroneous reference → false reward"**: Category-level solves the former, conflict-based filtering the latter. The approach is systematic and can be directly transferred to other "unreliable reference" evaluation tasks (e.g., dialogue evaluation, code review).
- **Meta-review as overlooked free supervision**: Previous work used only reviewer text; this is the first systematic use of meta-reviews as "high-level arbitration" for error detection, opening new directions for leveraging the full OpenReview discussion structure.
- **8B distillation enables industrial-scale evaluation**: Processing human reviews with strong LLMs is a one-time cost, but evaluating AI reviews requires repeated runs—distilling the pipeline to 8B reduces long-term costs by an order of magnitude, a rare engineering consideration for benchmark projects.
- **Multi-dimensional + paper/category dual-granularity LLM-judge** reveals that "single scores mask issues": high clarity does not imply high correctness, and old/new metric rankings are completely inconsistent—this is itself a strong methodological contribution.

## Limitations & Future Work
- The authors acknowledge that conflict-based filtering can only capture **explicit disagreements**: missed when authors do not respond or respond mildly; when meta-reviews are negative, annotators tend to side with reviewers, resulting in only 50% accuracy for reviewer-author error detection on rejected papers.
- The evaluation still **relies on LLM-judge (GPT-5-Mini)**, outsourcing ground-truth to another LLM, which risks "LLM evaluating LLM" cycles—though human validation is used to mitigate this.
- The 23-category taxonomy is based on NeurIPS 2025 + ARR forms and **may lack granularity for non-NLP/ML conferences (e.g., theory, vision subfields)**; porting to other domains requires redoing the taxonomy.
- Future directions: (1) Use misclassified opinions for negative training, specifically training AI reviewers to "not echo erroneous opinions"; (2) Use the coverage signal to synthesize "complete human references," extending from evaluation to reviewer training data augmentation (an agent framework prototype is provided in Appendix B.5); (3) Evaluate whether AI can identify "the most critical review points" rather than just providing a full set of opinions.

## Related Work & Insights
- **vs PeerRead / NLPeer / ReviewMT / Re2**: These datasets provide only review text or add meta/rebuttal, but **lack atomic opinions, category labels, conflict detection, and error annotation**. CoCoReviewBench is the first resource to cover all 7 dimensions; see Table 1 for a full comparison.
- **vs RevUtil (Sadallah et al., 2025)**: RevUtil also provides atomic + fine-grained + category-level data, but **lacks rebuttal/meta/conflict/error** signals; this work supplements the "correctness" dimension on top of it.
- **vs DeepReviewer / CycleReviewer / OpenReviewer / SEA**: Dedicated AI reviewer models score artificially high on old metrics, but are outperformed by general closed-source LLMs on new metrics, indicating they "imitate human review style" rather than improving real review ability. This work provides clear diagnostic evidence for the next step (less imitation, more reasoning).
- **vs LLM-as-a-judge paradigm (Xu et al., GRE-bench)**: That approach does not use human references, relying on LLM preferences for scoring; this work takes a middle path—retaining but cleaning human references, balancing coverage and cost.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines meta-review as conflict arbitration signal and category-level skip evaluation; clear approach, though individual innovations are not disruptive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale (3,900 papers), 18-model comparison, 50-paper manual validation, 6-LLM consistency, multi-granularity and multi-dimensional evidence chain is very complete.
- Writing Quality: ⭐⭐⭐⭐ Clean structure, high information density in Table 1/2, but terms (Correctness/Thoroughness/Grounding/Verifiability) are highly similar, requiring readers to repeatedly refer to definitions.
- Value: ⭐⭐⭐⭐⭐ Provides an industrializable, trustworthy evaluation for the "AI reviewer" subfield, and directly concludes "reasoning models > non-reasoning," guiding future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Human-Level AI Tales to AI Leveling Human Scales](from_human-level_ai_tales_to_ai_leveling_human_scales.md)
- [\[ICML 2026\] When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation](when_ai_benchmarks_plateau_a_systematic_study_of_benchmark_saturation.md)
- [\[ACL 2026\] AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage](../../ACL2026/llm_evaluation/autoreproduce_automatic_ai_experiment_reproduction_with_paper_lineage.md)
- [\[ICLR 2026\] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](../../ICLR2026/llm_evaluation/astabench_benchmarking_ai_agents.md)
- [\[AAAI 2026\] MindVote: When AI Meets the Wild West of Social Media Opinion](../../AAAI2026/llm_evaluation/mindvote_when_ai_meets_the_wild_west_of_social_media_opinion.md)

</div>

<!-- RELATED:END -->
