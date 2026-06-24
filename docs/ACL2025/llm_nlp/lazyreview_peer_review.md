---
title: >-
  [Paper Note] LazyReview: A Dataset for Uncovering Lazy Thinking in NLP Peer Reviews
description: >-
  [ACL 2025][LLM (Other)][peer review quality] This work introduces LazyReview, the first fine-grained classification dataset for "lazy thinking" in NLP peer reviews, containing 500 expert-annotated and 1,276 silver-annotated instances. Through a three-round iterative annotation protocol and positive example enhancement, annotation consistency is doubled. The study demonstrates that instruction-tuning LLMs on this dataset improves detection performance by 10–20 percentage point…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "peer review quality"
  - "lazy thinking"
  - "heuristic bias"
  - "instruction tuning"
  - "dataset construction"
date: 2026-05-08
content_hash: 7ec4571698a9ee2b
---

# LazyReview: A Dataset for Uncovering Lazy Thinking in NLP Peer Reviews

**Conference**: ACL 2025  
**arXiv**: [2504.11042](https://arxiv.org/abs/2504.11042)  
**Code**: [UKPLab/acl2025-lazy-review](https://github.com/UKPLab/acl2025-lazy-review)  
**Area**: LLM/NLP  
**Keywords**: peer review quality, lazy thinking, heuristic bias, instruction tuning, dataset construction

## TL;DR

This work introduces LazyReview, the first fine-grained classification dataset for "lazy thinking" in NLP peer reviews, containing 500 expert-annotated and 1,276 silver-annotated instances. Through a three-round iterative annotation protocol and positive example enhancement, annotation consistency is doubled. The study demonstrates that instruction-tuning LLMs on this dataset improves detection performance by 10–20 percentage points, and controlled experiments show that providing lazy thinking feedback significantly improves review quality.

## Background & Motivation

**Background**: Peer review is the cornerstone of quality control in academic publishing. With the explosive growth of submissions in the NLP field (with top conferences such as ACL, EMNLP, and NAACL breaking records year after year), reviewers face an increasingly heavy burden. In 2021, ACL Rolling Review (ARR) drafted guidelines on "lazy thinking" heuristics in reviewing, listing 14 common bias types.

**Limitations of Prior Work**: Despite these guidelines, the ACL 2023 presidential report indicated that lazy thinking remains one of the top issues raised in author complaints (accounting for 24.3%). Currently, there are no specialized datasets or automated tools to detect these heuristics, and researchers lack the means to quantitatively understand the distribution and severity of lazy thinking in reviews.

**Key Challenge**: While review guidelines describe what lazy thinking is in textual form, these descriptions are inherently ambiguous (obtaining a Cohen's $\kappa$ of only 0.31 in the first round of human annotation). This ambiguity makes accurate classification difficult for human annotators and leads to poor performance for LLMs in zero-shot settings. The lack of high-quality annotated data is the biggest bottleneck for automated detection.

**Goal**: (1) How to construct a lazy thinking detection dataset with sufficiently high annotation consistency? (2) Can LLMs automatically detect lazy thinking in reviews? (3) To what extent does lazy thinking feedback help improve review quality?

**Key Insight**: Grounded in the theory of heuristic bias from cognitive psychology, lazy thinking in peer reviews is treated as a cognitive shortcut taken under information overload and time pressure. To overcome annotation ambiguity, the annotation guidelines are iteratively refined (merging ARR and EMNLP guidelines, supplemented by positive example enhancement), followed by instruction tuning to compensate for the limitations of zero-shot LLM performance.

**Core Idea**: Construct a high-quality lazy thinking dataset through a three-round iterative annotation protocol, and then use instruction tuning to enable LLMs to detect cognitive biases in reviews.

## Method

### Overall Architecture

The workflow comprises four stages: (1) Data Collection and Annotation: extracting review snippets from ARR-22 reviews in the NLPeer dataset, resulting in 500 expert-annotated instances after three rounds of iterative annotation; (2) Zero-Shot/Few-Shot Evaluation: testing the detection capabilities of 7 open-source LLMs under guidelines from different annotation rounds; (3) Instruction Tuning: conducting LoRA fine-tuning on LLMs using mixed data sources including LazyReview, Tülu, and SciRIFF; and (4) Controlled Experiments: verifying the effectiveness of lazy thinking feedback in improving review quality.

### Key Designs

1. **Three-Round Iterative Annotation Protocol**:

    - **Function**: Extracting review snippets from the "weakness" sections of 684 reviews in ARR-22, pre-filtering candidate segments using GPT-4 (precision 0.74, recall 1.00), and conducting three rounds of annotation by NLP PhD students.
    - **Mechanism**: Round 1 uses only the original ARR guidelines ($\kappa = 0.31$) $\rightarrow$ Round 2 integrates EMNLP 2020 guidelines to expand category descriptions and names ($\kappa = 0.38$) $\rightarrow$ Round 3 incorporates positive examples ($\kappa = 0.52$). The positive example selection strategy is determined via comparative experiments, ultimately adopting the "random shortest segment" approach (with $\kappa = 0.86$ for example selection consistency), providing one typical annotation example per category.
    - **Design Motivation**: Ambiguities in the initial guidelines made it difficult for annotators to distinguish between fine-grained categories. Gradually enriching the guidelines (expanding descriptions and adding positive examples) reduces cognitive load. Validation experiments with a new batch of annotators ($\kappa$: $0.32 \rightarrow 0.36 \rightarrow 0.48$) confirm a steady improvement in guideline quality.

2. **Dual-Granularity Task Modeling**:

    - **Function**: Formalizing lazy thinking detection into two levels—coarse-grained (binary classification: presence or absence of lazy thinking) and fine-grained (18-class classification)—and evaluating them separately.
    - **Mechanism**: Each model is tested under two input configurations: Target snippet only (T) and Full Review + Target snippet (RT). Zero-shot experiments reveal that using the target snippet alone yields better performance (as long inputs introduce spurious correlations), and coarse-grained classification significantly outperforms fine-grained classification.
    - **Design Motivation**: Decoupling coarse- and fine-grained tasks allows the system to serve both as a simple screening tool (coarse-grained) and as an informative diagnostic tool that provides specific category information to aid authors and Area Chairs (ACs) (fine-grained).

3. **Mixed-Data Instruction Tuning**:

    - **Function**: Applying LoRA to instruction-tune 7 LLMs to explore the effects of different data mixture recipes.
    - **Mechanism**: Constructing four kinds of mixes by taking 700 instances each from LazyReview (700 training samples), Tülu V2 (326K general instruction data), and SciRIFF (154K scientific task data). A 3-fold cross-validation is used to find the optimal recipe, with an input type ratio of T:RT = 0.3:0.7. Ultimately, Qwen performs best on fine-grained classification (59.4% accuracy), while SciTülu performs strongest on coarse-grained classification (91.2%).
    - **Design Motivation**: The data volume of LazyReview alone is too small. Mixing in general-domain (Tülu) and scientific-domain (SciRIFF) instruction data enhances the models' instruction-following and domain-understanding capabilities. However, a complete mixture is not always optimal due to negative transfer issues.

### Loss & Training

Parameter-efficient fine-tuning is performed using LoRA based on the open-instruct framework. The temperature is set to 0 to ensure prediction consistency, and the output length is limited to 30 tokens. Training is conducted on A100 80GB GPUs, with each run taking under 36 hours. After determining the optimal data mixture ratio on the validation set via 3-fold cross-validation, models are retrained on the entire training set and evaluated on an independent test set.

## Key Experimental Results

### Main Results: Comparison of Performance Before and After Instruction Tuning (Fine-Grained Classification, String Accuracy)

| Model | Best Zero-Shot | Post-Instruction Tuning | Gain |
|------|-----------|-----------|------|
| LLaMa 7B | 22.2 | 44.7 | +22.5 |
| LLaMa 13B | 26.7 | 50.5 | +23.8 |
| Gemma 7B | 26.7 | 38.8 | +12.1 |
| Mistral 7B | 30.0 | 42.4 | +12.4 |
| Qwen 7B | 31.1 | **59.4** | +28.3 |
| Yi-1.5 6B | 37.6 | 47.9 | +10.3 |
| SciTülu 7B | 25.3 | 54.3 | +29.0 |

### Ablation Study: Comparison of Data Mixing Strategies (3-Fold CV Average, Fine-Grained S.A.)

| Data Recipe | Qwen | SciTülu | LLaMa | Description |
|----------|------|---------|-------|------|
| No Mix (LazyReview Only) | 42.1 | 38.5 | 36.2 | Baseline Recipe |
| SciRIFF Mix | 44.2 | **45.7** | **43.8** | Scientific domain data helps |
| Tülu Mix | **45.5** | 41.2 | 39.8 | General instructions help Qwen |
| Full Mix | 43.8 | 42.6 | 40.1 | Negative transfer exists |

### Key Findings

- **Qwen leads in fine-grained classification**, which is hypothesized to be linked to its multilingual pre-training data (2.4T tokens) and high-quality data filtering.
- **SciTülu is strongest in coarse-grained classification** (91.2%), benefiting from SciRIFF scientific task pre-training.
- **Positive example enhancement is highly effective**: Adding 1 static positive example via in-context learning (ICL) improves coarse-grained accuracy by over 20 percentage points (Gemma: $50.4 \rightarrow 75.6$, SciTülu: $58.3 \rightarrow 88.8$).
- **Full data mixing is not always optimal**: Full Mix occasionally underperforms compared to SciRIFF Mix or Tülu Mix, indicating negative transfer.
- **Controlled experiments demonstrate** that reviews revised using lazy thinking feedback achieve win rates of 85%, 85%, and 90% in comprehensiveness, evidence, and guideline alignment, respectively (vs. original reviews), with a win rate of 95.6% as calculated by the Bradley-Terry model.

## Highlights & Insights

- **Exquisitely designed iterative annotation protocol**: Instead of scaling up annotation immediately after a single draft of guidelines, the protocol refines the guidelines through three rounds of small-batch annotations (50 instances per round), using Cohen's $\kappa$ to quantify guideline quality, and expands annotation only when acceptable consistency is achieved. This method serves as a valuable reference for any highly subjective annotation task.
- **Positive examples are more effective than expanded descriptions**: The improvement from Round 2 to 3 primarily stems from introducing positive examples ($\kappa$ from $0.38 \rightarrow 0.52$), whereas Round 1 to 2, which only expanded textual descriptions, yielded limited gains ($\kappa$ from $0.31 \rightarrow 0.38$). This aligns with findings in ICL that format is more crucial than content.
- **Evaluation methods balance upper and lower bounds**: Two evaluators, String Matching (strict/underestimating) and GPT-based (lenient/overestimating), are designed to avoid biases inherent in single metrics. This dual-evaluation strategy is highly transferable to the evaluation of other open-ended generation tasks.

## Limitations & Future Work

- **Domain limitations**: The dataset only covers ARR-22 NLP reviews, and the 14 lazy thinking categories are specific definitions from the NLP community. They cannot be directly applied to other venues like ICLR or NeurIPS, which operate under different review standards and bias patterns.
- **Time-frame limitations**: All review data dates back to before 2022 (prior to the widespread adoption of LLMs). It does not cover lazy thinking detection in LLM-generated reviews, which is an increasingly important direction.
- **Focus solely on Weakness sections**: Lazy thinking can also appear in other sections such as Summary, Strengths, and Comments, as well as in subsequent author-reviewer discussions.
- **Small model scale**: Experiments are conducted using only 6–13B parameter models. Larger models (70B+) or closed-source models (GPT-4, Claude) are not tested, which may result in an underestimation of the ceiling for zero-shot performance.
- **Uncertainty in silver annotation quality**: The 1,276 silver annotations come from the predictions of the instruction-tuned Qwen model, which has a fine-grained accuracy of approximately 59%. This implies that around 40% of the silver-annotated instances may be incorrect.

## Related Work & Insights

- **vs ReviewAdvisor (Yuan et al., 2022)**: ReviewAdvisor focuses on the automated evaluation of the comprehensiveness and constructiveness of reviews, but does not target specific cognitive bias categories. LazyReview concentrates on actionable, fine-grained issue types. The two are complementary—one can first use LazyReview to filter out problematic review snippets, and then apply comprehensiveness metrics for overall scoring.
- **vs DISAPERE (Kennard et al., 2022)**: DISAPERE annotates the discourse structure (argumentative structure) of reviews, whereas LazyReview annotates cognitive bias types; these two dimensions are orthogonal. Their combined use enables simultaneous analysis of "how it is said" and "whether it is correct."
- **vs LLM-as-Reviewer (Du et al., 2024; Zhou et al., 2024)**: These works leverage LLMs to generate reviews directly, whereas LazyReview uses LLMs to detect quality issues in human or AI reviews. The LazyReview dataset can serve as a review quality detection module embedded into the post-processing pipeline of any automated review system.
- **Takeaway**: The lazy thinking detection framework can be generalized to other "expert judgment" scenarios (such as code review and medical diagnoses), where the core task is to identify "rapid judgments based on heuristics rather than evidence."

## Rating

- Novelty: ⭐⭐⭐⭐ The first fine-grained annotated dataset for lazy thinking. The problem definition is valuable, though the technical methodology (ICL + instruction tuning) is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely solid experimental design, featuring 7 models $\times$ 3 guideline rounds $\times$ multiple ICL strategies $\times$ data mixture ablations + controlled experiments + Bradley-Terry rankings.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with elegant narrative logic covering the three-round iteration. However, the abundance of tables and figures makes certain sections feel slightly dense.
- Value: ⭐⭐⭐⭐ Holds direct significance for improving review practices in the NLP community; the dataset and enhanced guidelines can be adopted straight into ARR, though it remains constrained to the NLP domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unveiling Dual Quality in Product Reviews: An NLP-Based Approach](unveiling_dual_quality_in_product_reviews_an_nlp-based_approach.md)
- [\[ACL 2025\] The Nature of NLP: Analyzing Contributions in NLP Papers](the_nature_of_nlp_analyzing_contributions_in_nlp_papers.md)
- [\[ACL 2025\] Attribution Methods in NLP: Navigating a Fragmented Landscape](attribution_methods_in_nlp_navigating_a_fragmented_landscape.md)
- [\[ACL 2025\] Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](veracity_bias_llm_hidden_beliefs.md)
- [\[ACL 2025\] A Modular Dataset to Demonstrate LLM Abstraction Capability](a_modular_dataset_to_demonstrate_llm_abstraction_capability.md)

</div>

<!-- RELATED:END -->
