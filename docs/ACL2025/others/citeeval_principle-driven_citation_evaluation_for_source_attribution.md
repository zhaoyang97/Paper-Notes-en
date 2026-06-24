---
title: >-
  [Paper Note] CiteEval: Principle-Driven Citation Evaluation for Source Attribution
description: >-
  [ACL 2025][Citation evaluation] This paper proposes CiteEval, a principle-driven framework for citation evaluation. By considering the entire retrieval context, multiple contexts beyond retrieval, and fine-grained evaluation criteria, the authors construct the CiteBench benchmark and the CiteEval-Auto automatic metric, which significantly outperform existing NLI-based methods in citation quality evaluation.
tags:
  - "ACL 2025"
  - "Citation evaluation"
  - "RAG"
  - "source attribution"
  - "NLI"
  - "automatic evaluation metrics"
date: 2026-05-08
content_hash: 905478bce129428e
---

# CiteEval: Principle-Driven Citation Evaluation for Source Attribution

**Conference**: ACL 2025  
**arXiv**: [2506.01829](https://arxiv.org/abs/2506.01829)  
**Code**: [https://github.com/amazon-science/CiteEval](https://github.com/amazon-science/CiteEval)  
**Area**: Others  
**Keywords**: Citation evaluation, RAG, source attribution, NLI, automatic evaluation metrics

## TL;DR
This paper proposes CiteEval, a principle-driven framework for citation evaluation. By considering the entire retrieval context, multiple contexts beyond retrieval, and fine-grained evaluation criteria, the authors construct the CiteBench benchmark and the CiteEval-Auto automatic metric, which significantly outperform existing NLI-based methods in citation quality evaluation.

## Background & Motivation

Retrieval-Augmented Generation (RAG) systems are playing an increasingly important role in information seeking, where accurate source attribution (i.e., citation) is crucial for building user trust and content verifiability. However, current citation evaluation methods suffer from significant limitations:

**Limitations of Prior Work**: Evaluation frameworks represented by AIS (Attributable to Identified Sources) primarily rely on Natural Language Inference (NLI), focusing only on whether the cited passages "entail" the target sentence. This approach has three key limitations:

**Context Inadequacy**: It only considers the cited passage and ignores other retrieved documents that might be better sources. This leads to overestimation when a more reliable source exists but is not cited, and underestimation when a partially supportive citation is marked as fully incorrect because no perfect source is available.

**Narrow Scope**: Many sentences in responses come from paraphrasing the user query, reasoning over previous turns, or parametric knowledge, which should not require citations but are implicitly penalized or ignored by existing frameworks.

**Coarse Granularity**: Binary or ternary support scores fail to capture the multi-dimensional aspects of citation quality, such as redundant citations or source trustworthiness.

**Key Challenge**: As a proxy metric for citation evaluation, NLI fails to accurately reflect human judgment on citation quality.

**Key Insight**: Redesigning the framework based on evaluation principles, improving from three dimensions: context completeness, context scope, and evaluation granularity.

## Method

### Overall Architecture
CiteEval formalizes the citation evaluation problem as $r_i = f_\theta(\mathcal{C}_i; \mathcal{S}, R, Q)$, where the citation score depends not only on the cited passage $\mathcal{C}_i$, but also on the full set of retrieved sources $\mathcal{S}$, the response $R$, and the user query $Q$. The entire evaluation process consists of three steps: Context Attribution → Citation Editing → Citation Rating.

### Key Designs

1. **Principle 1 — Citation Evaluation Based on the Full Retrieval Context**:

    - **Mechanism**: Citation quality should be evaluated relatively across all retrieved passages rather than purely focusing on the cited passage itself.
    - **Design Motivation**: Even if a citation entails the target sentence, its quality should be downgraded if there exists a more reliable, uncited source.
    - Similarly, partially supportive citations are still valuable when no better alternatives exist, and should not be directly classified as invalid.

2. **Principle 2 — Evaluation Beyond Retrieval Context**:

    - Attributing response sentences to four context types: retrieval context, user context (query paraphrase), answer context (reasoning based on preceding text), and parametric knowledge context (intrinsic model knowledge).
    - Sentences generated from non-retrieval contexts are labeled as N/A and excluded from citation evaluation to avoid being erroneously penalized.
    - For example, paraphrasing statements like "You asked about the significance of Newton's first law" should not require citations.

3. **Principle 3 — Fine-grained Criteria and Scenarios**:

    - Introducing a 1-5 Likert scale to replace binary judgments.
    - Defining 6 editing actions: delete misleading citations, delete low-quality citations, delete redundant citations, add evidence, add improvement, and add credibility.
    - Distinguishing between the Full scenario (evaluating all sentences that should have citations) and the Cited scenario (evaluating only the sentences that actually contain citations).

### CiteBench Benchmark Construction
- Covers ASQA, ELI5, MS MARCO, and LFRQA datasets, totaling 3,948 queries.
- Three-stage annotation pipeline: Context Attribution → Citation Editing → Citation Rating.
- Completed independently by three professional annotators, achieving an IAA (Inter-Annotator Agreement) of 0.980 for context attribution and 0.774 for rating.

### CiteEval-Auto Automatic Evaluation

Proposed two scoring methods and integrated them:

1. **IterCoE (Iterative Chain-of-Edit)**: Guides the LLM to first perform context attribution for each sentence, generate a sequence of editing actions, and construct a 1-5 score based on editing outcomes and rating guidelines.
2. **EditDist (Edit Distance)**: Learns distance weights for each editing action via multiple linear regression: $r_i = \sum_{k=1}^K d(a_k) \cdot \frac{|\mathcal{A}_{i,k}^*|}{|\mathcal{A}_i^*|} + b$. It is found that the penalty weights for add-type actions are higher than those for delete-type actions.

Finally, CiteEval-Auto integrates the scores of both methods through linear interpolation.

## Key Experimental Results

### Main Results

| Evaluation Metric | Model | Statement Pearson | Statement Spearman | Response Pearson | Response Spearman |
|---------|------|------------------|--------------------|-----------------|-------------------|
| AutoAIS-Recall | T5-XXL | 0.409 | 0.264 | 0.223 | 0.075 |
| AttrScore-Strict | GPT-4o | 0.449 | 0.297 | 0.221 | 0.094 |
| LQAC-Recall | GPT-4o | 0.607 | 0.423 | 0.526 | 0.447 |
| CiteEval-Auto | GPT-4o+MLR | **0.731** | **0.559** | **0.668** | **0.589** |

### Ablation Study

| Configuration | Pearson | Explanation |
|------|---------|------|
| CiteEval-Auto (Full) | 0.731 | Integrates IterCoE + EditDist |
| W/o Context Attribution | Significant drop | Attribution prediction F1=0.957; both scoring methods degrade significantly after removal.
| Vanilla Direct Scoring | Far below IterCoE | Poor performance when directly scoring without editing reasoning.
| IterCoT (Chain of Thought) | Moderate | Less effective than explicit editing reasoning.

### Key Findings
- Llama-3-70b outperforms GPT-4o in the Full scenario (0.909 vs 0.898) because GPT-4o generates longer responses but is more prone to omitting citations.
- Response length correlates strongly with citation omission rate (Pearson = 0.679).
- The iterative editing in CiteEval-Auto continuously improves citation quality, and models of different sizes eventually converge to similar levels.
- Contexts with higher retrieval recall yield better citation quality, but higher precision does not necessarily guarantee the same.

## Highlights & Insights
- Upgrading citation evaluation from simple NLI judgments to a principle-driven, multi-dimensional framework, which is clear and convincing.
- The introduction of context attribution is elegant — non-attributable sentences should not be evaluated, solving a long-ignored issue.
- Explicit reasoning over editing actions is more effective than direct scoring or CoT reasoning, demonstrating the importance of structured intermediate steps for evaluation tasks.
- The discovery that iterative editing improves citation quality implies the potential of inference-time scaling for source attribution.

## Limitations & Future Work
- CiteEval-Auto relies on GPT-4o as the backbone model, which incurs high costs; exploring distillation to smaller models is a necessary future step.
- Currently, context attribution only covers typical context types in RAG, without considering more complex scenarios such as personalization.
- Evaluation is conducted at the sentence level; finer-grained chunk-level evaluation might be more accurate.
- It is not yet integrated with end-to-end evaluation of the retrieval phase.

## Related Work & Insights
- Inherits and transcends the citation evaluation paradigm of AIS/Auto-AIS, upgrading the evaluation from "whether it entails" to "whether the citation is optimal".
- Compared with LQAC, CiteEval avoids overestimating the scores of N/A sentences through context attribution.
- The learning method of edit distance can be generalized to other NLP tasks requiring fine-grained evaluation.
- Provides an actionable path for improving citation quality in RAG systems (iterative editing).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The principle-driven framework design is novel, though in essence it remains a variant of LLM-as-a-judge.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Broad coverage of multiple datasets, models, baselines, ablation studies, and application explorations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Well-structured, principles are convincingly elaborated, and figures/tables are intuitive.
- **Value**: ⭐⭐⭐⭐ A significant contribution to RAG citation evaluation, though the deployment cost under practical settings remains to be addressed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The Noisy Path from Source to Citation: Measuring How Scholars Engage with Past Research](the_noisy_path_from_source_to_citation_measuring_how_scholars_engage_with_past_r.md)
- [\[ACL 2025\] LAQuer: Localized Attribution Queries in Content-grounded Generation](laquer_localized_attribution.md)
- [\[ACL 2025\] Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages](using_source-side_confidence_estimation_for_reliable_translation_into_unfamiliar.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] Minimal Pair-Based Evaluation of Code-Switching](minimal_pair-based_evaluation_of_code-switching.md)

</div>

<!-- RELATED:END -->
