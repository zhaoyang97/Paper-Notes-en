---
title: >-
  [Paper Note] Towards Better Evaluation for Generated Patent Claims
description: >-
  [ACL 2025][Patent claims] This paper proposes the first evaluation benchmark for patent claims, Patent-CE (comprising 1,228 expert-annotated comparative evaluation data points), and a dedicated evaluation method, PatClaimEval (based on Longformer + a variant of contrastive learning). Across five dimensions—feature completeness, conceptual clarity, terminological consistency, logical connection, and overall quality—the proposed method consistently outperforms 13 existing basel…
tags:
  - "ACL 2025"
  - "Patent claims"
  - "evaluation benchmark"
  - "automatic evaluation"
  - "contrastive learning"
  - "legal text"
date: 2026-05-08
content_hash: c4544e48293f69ca
---

# Towards Better Evaluation for Generated Patent Claims

**Conference**: ACL 2025  
**arXiv**: [2505.11095](https://arxiv.org/abs/2505.11095)  
**Code**: [https://github.com/scylj1/PatClaimEval](https://github.com/scylj1/PatClaimEval)  
**Area**: Others  
**Keywords**: Patent claims, evaluation benchmark, automatic evaluation, contrastive learning, legal text  

## TL;DR
This paper proposes the first evaluation benchmark for patent claims, Patent-CE (comprising 1,228 expert-annotated comparative evaluation data points), and a dedicated evaluation method, PatClaimEval (based on Longformer + a variant of contrastive learning). Across five dimensions—feature completeness, conceptual clarity, terminological consistency, logical connection, and overall quality—the proposed method consistently outperforms 13 existing baselines (including G-Eval-4) in correlation with human expert evaluation, achieving a 58% Spearman correlation improvement in the overall quality dimension.

## Background & Motivation
**Background**: Patent claims define the protection scope and legal boundaries of inventions. The drafting process is complex, time-consuming, and heavily relies on professional patent attorneys. While LLMs have been studied for automated patent claim generation, a severe inconsistency persists between automatic evaluations and human expert assessments.

**Limitations of Prior Work**: (a) N-gram metrics (BLEU/ROUGE) only measure surface overlap and fail to capture the unique linguistic requirements of patents; (b) embedding-based metrics (BERTScore/BARTScore) measure semantic similarity, but patent evaluation focuses on terminological consistency and logical connections rather than generalized semantics; (c) general multi-dimensional evaluators (UniEval/AlignScore) utilize dimensions (fluency, coherence, relevance) that do not align with patent evaluation standards; (d) GPT-4 as a judge performs poorly on terminological consistency due to the lack of patent-domain training.

**Key Challenge**: Patent claims have unique linguistic requirements (precise terminology, logical connection, technical rigor) that fundamentally differ from general text quality standards (fluency, coherence). None of the existing evaluation metrics are designed for this.

**Goal**: Build an evaluation benchmark and evaluation method dedicated to patent claims to bridge the gap between automatic and expert evaluations.

**Key Insight**: (a) Constructing the Patent-CE benchmark using comparative evaluation data annotated by patent experts—evaluating relative quality rather than absolute scoring; (b) employing Longformer to handle long patent texts and custom contrastive learning to train PatClaimEval.

**Core Idea**: Patent claims require dedicated evaluation criteria and methods—general text metrics fail comprehensively in this domain.

## Method

### Overall Architecture
(1) Constructing the Patent-CE benchmark: LLM-generated claim pairs are collected from USPTO and EPO patent data and annotated by patent experts via comparative evaluation across five dimensions; (2) Training PatClaimEval: Based on encoding reference claims and candidate claim pairs using Longformer, individual evaluation models are trained for each of the five dimensions using custom contrastive learning losses.

### Key Designs

1. **Patent-CE Benchmark Dataset**:

    - **Function**: The first evaluation benchmark for patent claims.
    - **Mechanism**: Each data point is a quadruple $(A, B, C, y)$—representing a reference claim $A$, two candidate claims $B$ and $C$, and a label $y \in \{1, 0, -1\}$ indicating $B$ is better than, equal to, or worse than $C$.
    - **Five Evaluation Dimensions**: Feature completeness (whether all key aspects of the invention are covered), conceptual clarity (whether the language is unambiguous), terminological consistency (whether terminology usage is consistent), correctness of logical connection (whether associations between features are accurate), and overall quality.
    - **Scale**: 1,228 data points annotated by multiple patent experts, with data sourced from both USPTO and EPO patent offices.
    - **Design Motivation**: Comparative evaluation is more reliable than absolute scoring—different experts may interpret absolute scores differently, but their judgments on "which is better" are more consistent.

2. **PatClaimEval Evaluation Model**:

    - **Function**: An automatic evaluation method customized for patent claims.
    - **Mechanism**:
        - Utilizing Longformer as the backbone (supporting up to 4096 tokens, as patent claims average over 1000 tokens).
        - The input is $[P; Q]$ (concatenation of reference and candidate claims), passed through a fully connected layer and a sigmoid function after encoding to output the quality score $s(Q|P)$.
        - An independent model is trained for each of the five dimensions because optimization objectives between different dimensions may conflict (e.g., completeness vs. clarity are not necessarily positively correlated).
    - **Design Motivation**: Longformer is open-source, supports long text, and offers control. Patent-specific LLMs are not used because they are either closed-source (PatentGPT) or suffer from insufficient context length (PatentGPT-J).

3. **Custom Contrastive Learning Training**:

    - **Function**: Training the evaluation model using triplet comparative labels.
    - **Mechanism**: Calculating quality scores $s_B$ and $s_C$ for two candidates. When $y=1$ (B is better), the score difference is forced to be at least a margin $m$; when $y=0$ (equivalent), the difference is bound within a tolerance $n$; when $y=-1$ (C is better), the margin constraint is applied in the opposite direction.
    - **Loss Function**: 
    $$\ell = \begin{cases} \text{ReLU}(m - (s_B - s_C)), & y=1 \\ \text{ReLU}(|s_B - s_C| - n), & y=0 \\ \text{ReLU}(m - (s_C - s_B)), & y=-1 \end{cases}$$
    - **Design Motivation**: Directly modeling the relative preferences of experts rather than absolute scores, which aligns with how the data is annotated. The margin ensures confident decisions in "obviously better" cases.

### Loss & Training
- Five Longformer models are trained independently for each dimension (to avoid multi-task learning due to conflicts among dimensions).
- Custom contrastive learning loss (a variant of margin-based triplet loss).
- Training set of 1,044 instances, test set of 184 instances (~15%).
- Hyperparameters margin $m$ and tolerance $n$ require tuning.

## Key Experimental Results

### Main Results (Kendall-Tau / Spearman correlation with human expert evaluation)

| Metric | Completeness τ/ρ | Clarity τ/ρ | Consistency τ/ρ | Connection τ/ρ | Overall τ/ρ |
|------|----------|----------|----------|--------|--------|
| BLEU-1 | .305/.345 | .359/.401 | .284/.329 | .335/.376 | .326/.369 |
| BERTScore | .241/.279 | .251/.281 | .242/.283 | .272/.303 | .239/.268 |
| UniEval | .339/.383 | .337/.375 | .261/.302 | .301/.338 | .337/.381 |
| G-Eval-4 | .377/.410 | **.412**/.481 | .276/.353 | .350/.385 | .277/.310 |
| **PatClaimEval** | **.400/.504** | **.461/.518** | **.354/.424** | **.419/.518** | **.477/.602** |

### Ablation Study

| Finding | Explanation |
|------|------|
| N-gram > Embedding Metrics | Anomalous in the patent domain—because patents require precise wording, surface overlap is more important than semantic similarity. |
| G-Eval-4 is weak in terminological consistency and overall quality | GPT-4 lacks patent-domain training and cannot understand the specialized terminology requirements of patents. |
| PatClaimEval improves overall quality by 58% | Spearman correlation improves from 0.381 (UniEval) / 0.310 (G-Eval-4) to 0.602. |
| Independent training for the five dimensions outperforms multi-task learning | Optimization objectives between dimensions conflict (e.g., completeness vs. clarity). |

### Key Findings
- **Metric reversal phenomenon in the patent domain**: While embedding-based metrics usually outperform N-gram metrics in general text, N-gram metrics perform better in patent evaluation. This is because patent claims employ precise legal language, where lexical overlap with the reference reflects quality more accurately than semantic similarity.
- PatClaimEval consistently leads across all five dimensions, demonstrating a systemic advantage rather than a coincidental improvement in a single dimension.
- G-Eval-4 performs decently on feature completeness ($\tau=0.377$) due to GPT-4's strong information extraction capabilities, but it fails in dimensions requiring specialized legal language understanding.
- Accuracy/F1 evaluation also confirms PatClaimEval's superiority, achieving not only strong ranking correlation but also the highest classification accuracy.

## Highlights & Insights
- The core lesson of **"domain specificity defeats general metrics"**: In highly specialized domains like patents, general text evaluation metrics fail entirely, necessitating domain-tailored evaluation methods. This lesson is likely applicable to other professional domains (e.g., legal judgments, clinical reports, technical standards documents).
- The counter-intuitive yet highly explanatory finding that **N-gram metrics outperform embedding-based metrics in the patent domain**: The precision of patent language dictates that "using the exact same terms" is a better reflection of quality than "expressing the same general meaning."
- The **comparative evaluation paradigm** ("which is better") is more suitable for such subjective tasks than absolute scoring, as it reduces inter-expert calibration bias.
- The customized margin-based contrastive learning loss elegantly handles ternary classification (superior/equal/inferior), making it more suitable for this task than standard contrastive learning.
- The Patent-CE benchmark fills a critical gap in patent NLP evaluation, where previous patent generation studies suffered from the lack of a standardized evaluation workflow.

## Limitations & Future Work
- This work evaluates reference-based generation (requiring a reference claim), which differs from reference-free evaluation in real-world patent examination. The authors explicitly note this distinction in the Limitations section.
- The dataset scale (1,228 instances) is relatively small by deep learning standards, which may limit the generalization capability of the model.
- Although Longformer supports up to 4,096 tokens, it may still truncate extremely long patent claims.
- Training an individual model for each of the five dimensions increases maintenance costs. Parameter-efficient multitask approaches could be explored.
- Only English patents are covered; other languages such as Chinese and Japanese remain unvalidated.
- Inter-annotator agreement among evaluating experts is not reported in detail.

## Related Work & Insights
- **vs General Evaluation Benchmarks (SummEval/Topical-Chat/ToTTo)**: These benchmarks focus on fluency, coherence, and relevance, whereas Patent-CE focuses on legal and technical precision, representing entirely different dimensions.
- **vs G-Eval-4 (LLM-as-Judge)**: G-Eval-4 performs well on general tasks but lacks patent-domain knowledge. PatClaimEval bridges this gap through training on domain-specific data.
- **vs CoCoLex (Legal Text Faithfulness)**: CoCoLex addresses the faithfulness of legal text generation, whereas Patent-CE resolves the accuracy of legal text evaluation—representing complementary research directions.
- Direct impact on the patent AI industry: Automated patent drafting tools require reliable automatic evaluation, and Patent-CE + PatClaimEval provides the first viable solution.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First patent claim evaluation benchmark and dedicated evaluation method, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive comparison with 13 baselines across five dimensions, two types of correlation, accuracy/F1, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clearly defined problems, thorough review of related work, and deep experimental analysis (particularly the explanation of "N-gram overcoming embeddings").
- **Value**: ⭐⭐⭐⭐ Important contribution to patent AI and domain-specific NLP evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] All That Glitters is Not Novel: Plagiarism in AI Generated Research](plagiarism_ai_generated_research.md)
- [\[ACL 2025\] Neural Parameter Search for Slimmer Fine-Tuned Models and Better Transfer](neural_parameter_search_for_slimmer_fine-tuned_models_and_better_transfer.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)
- [\[ACL 2025\] Minimal Pair-Based Evaluation of Code-Switching](minimal_pair-based_evaluation_of_code-switching.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)

</div>

<!-- RELATED:END -->
