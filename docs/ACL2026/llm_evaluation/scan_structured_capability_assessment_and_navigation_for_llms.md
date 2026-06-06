---
title: >-
  [Paper Note] SCAN: Structured Capability Assessment and Navigation for LLMs
description: >-
  [ACL2026][LLM Evaluation][Fine-grained evaluation] SCAN advances LLM evaluation from simple leaderboards to navigable capability profiles. It automatically constructs hierarchical capability labels…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "Fine-grained evaluation"
  - "capability taxonomy tree"
  - "synthetic evaluation data"
  - "LLM-as-a-Judge"
  - "model diagnosis"
date: 2026-05-08
content_hash: cc35013f5922f7a8
---

# SCAN: Structured Capability Assessment and Navigation for LLMs

**Conference**: ACL2026  
**arXiv**: [2505.06698](https://arxiv.org/abs/2505.06698)  
**Code**: https://github.com/liudan193/SCAN  
**Area**: LLM Evaluation / Capability Profiling  
**Keywords**: Fine-grained evaluation, capability taxonomy tree, synthetic evaluation data, LLM-as-a-Judge, model diagnosis

## TL;DR
SCAN advances LLM evaluation from simple leaderboards to navigable capability profiles. It automatically constructs hierarchical capability labels, uses RealMix to generate realistic queries covering long-tail capabilities, and employs the PC2 judge to enhance automatic scoring reliability, thereby revealing fine-grained strengths and weaknesses masked by total scores across 21 mainstream LLMs.

## Background & Motivation
**Background**: LLM evaluation has long relied on leaderboard-style benchmarks like Chatbot Arena, MT-Bench, and AlpacaEval. These provide quick overall rankings and approximate human preferences with few automatic evaluation samples, aiding model releases and horizontal comparisons.

**Limitations of Prior Work**: Leaderboards answer "who is stronger," but developers truly need to know "in which capabilities is this model strong, in which sub-capabilities is it weak, and are weaknesses localizable." For instance, a model might have a high total score but perform inconsistently in Java, C, bioengineering knowledge, or roleplay; a single average score flattens these local failures.

**Key Challenge**: Fine-grained evaluation requires covering numerous capability labels, while long-tail labels lack sufficient samples. Automatic scoring needs to scale to many models and questions, but classic pointwise judges are unreliable, and pairwise judges have quadratic complexity. SCAN aims to simultaneously address broad coverage, fine granularity, sufficient samples, accurate scoring, and interpretable navigation.

**Goal**: The authors aim to build a framework that starts from real user queries, automatically generates a capability taxonomy tree, supplements evaluation queries for each label, and visualizes evaluation results as a capability map. Ultimately, users should be able to drill down along the capability tree to identify specific model shortcomings rather than just seeing a rank.

**Key Insight**: The paper observes that real user queries contain substantial capability signals, such as "Python programming," "Physics Q&A," or "Roleplay setting." Instead of manually enumerating capability dimensions, it is better to extract labels from real queries and organize them into a tree structure. Rather than randomly synthesizing questions, it is better to recombine content fragments from real queries under specified labels.

**Core Idea**: Use a "real-query-driven capability tree + label-aligned data synthesis + pre-comparison criterion extraction judge" to replace static leaderboards, transforming evaluation results into searchable, diagnostic, and scalable model capability profiles.

## Method
SCAN's methodology is a pipeline from "real user needs" to a "model capability navigation map." It first uses TaxBuilder to extract and organize capability labels from massive queries, then uses RealMix to supplement realistic evaluation questions for each label, and finally uses the PC2 scorer to evaluate model responses at a scalable cost, helping users analyze strengths and weaknesses through visualization tools.

### Overall Architecture
The input consists of real user queries and the LLMs to be evaluated; the output includes fine-grained scores, rankings, and failure modes across six major domains and their sub-labels. SCAN-V0 covers six domains: writing, roleplay, knowledge, coding, mathematics, and reasoning, constructing 2,082 labels and 3,343 evaluation queries, evaluating 21 mainstream LLMs.

The process is divided into three layers. The first is the taxonomy layer, where TaxBuilder inserts unstructured query labels into an editable hierarchical tree. The second is the data layer, where RealMix uses real query fragments and label constraints to synthesize evaluation queries, ensuring statistical sample sizes for each label. The third is the evaluation/navigation layer, where the PC2 judge scores responses using query-specific criteria generated from pre-comparison, followed by displaying results through dashboards and failure mode explorers.

### Key Designs
1. **TaxBuilder Hierarchical Capability Tree**:

    - **Function**: Automatically organizes capability labels from real user queries into an expandable, manually verifiable capability taxonomy tree.
    - **Mechanism**: Use a low-cost model to label many real queries, then insert new labels into the tree one by one. Instead of providing the whole tree to the LLM, recursively traverse nodes at the current level, asking the LLM to judge three relations: the label exists, should be a sibling at this level, or should be a child of a node. Subsequently, use node refinement, pruning, and hierarchical pruning to correct parent-child relationships, split mixed concepts, merge duplicates, and control tree depth to approximately 4 levels.
    - **Design Motivation**: Directly localizing new nodes in a large tree leads to long contexts and difficult reasoning. Recursive local decisions split long-context problems into short-context judgments, making it suitable for continuous capability expansion.

2. **RealMix Label-Aligned Query Synthesis**:

    - **Function**: Generates sufficient, high-quality, and realistic evaluation queries for long-tail capability labels.
    - **Mechanism**: Use QwQ-32B to label real user queries with domains, labels, and quality, filtering ~31,000 high-quality seed queries. During synthesis, sample one reference query and three content queries; the generation model extracts appropriate real content fragments and recombines them into a new query matching the reference label. Multiple reasoning models then check label consistency and quality, filtering unqualified samples.
    - **Design Motivation**: Using existing data leads to insufficient coverage and contamination risks, while random label combinations yield unrealistic tasks. RealMix allows synthetic questions to inherit content and label distributions from real queries while directionally filling long-tail capabilities.

3. **PC2 Pre-comparison Criterion Scorer**:

    - **Function**: Captures the reliability advantages of pairwise evaluation at pointwise costs.
    - **Mechanism**: For an instruction, multiple models generate candidate responses; a judge compares these and extracts "what to look for in this question" as scoring criteria and weights, satisfying $\sum_i w_i = 100$. During formal scoring, the judge evaluates the target response using these query-specific criteria, weights, and a reference evaluation of a baseline answer.
    - **Design Motivation**: Pairwise evaluation is superior because comparison exposes response differences; pointwise is inferior due to the lack of reference. PC2 extracts differential criteria in the pre-comparison phase to avoid the high cost of all-model pairwise comparisons.

### Loss & Training
SCAN is an evaluation framework rather than a training method. The "optimization goal" is reflected in the judge's scoring process: first obtain criterion set $C$ and weights $W$ via $J(\{y^1,\dots,y^n\}\mid x,p_c,p_w)$, then output a score for a single response via $J(y\mid x,p_c,y_b,C,W)\to s$. In data construction, multi-model generation and verification reduce single-model bias; in evaluation, PC2 balances accuracy and scalability.

## Key Experimental Results

### Main Results
The scale of SCAN-D-V0 shows it is a structured fine-grained evaluation set rather than a small prompt set.

| Domain | Sample count | Label count | Min samples per label | Avg length |
|------|--------|--------|----------------|----------|
| Writing | 1,108 | 594 | 19 | 772.57 |
| Roleplay | 470 | 429 | 19 | 1008.46 |
| Knowledge | 540 | 315 | 20 | 608.57 |
| Coding | 636 | 369 | 19 | 1232.03 |
| Mathematics | 344 | 189 | 20 | 817.02 |
| Reasoning | 245 | 186 | 19 | 904.81 |
| Total | 3,343 | 2,082 | 19 | 880.92 |

PC2 judge significantly outperforms naive pointwise across multiple judge backbones, indicating that "extracting criteria before scoring" is effective across different models.

| Judge / Method | Accuracy |
|--------------|----------|
| Deepseek-R1 naive | 0.5694 |
| Deepseek-R1 direct metric decomposition | 0.6134 |
| Deepseek-R1 diverse pre-comparison | 0.6466 |
| Deepseek-R1 ours | 0.6962 |
| Qwen3-32B naive | 0.5181 |
| Qwen3-32B ours | 0.6535 |
| Claude-3.7-Sonnet naive | 0.5959 |
| Claude-3.7-Sonnet ours | 0.7453 |
| GPT-4.1 naive | 0.6116 |
| GPT-4.1 ours | 0.7201 |

### Ablation Study
The core ablations focus on PC2 components. Using Deepseek-R1 as the judge, accuracy improves step-by-step from naive pointwise to diverse pre-comparison and finally the full method.

| Configuration | Accuracy | Description |
|------|----------|------|
| naive pointwise | 0.5694 | Scoring responses individually without comparison reference |
| direct metric decomposition | 0.6134 | Allowing the judge to decompose scoring dimensions directly; moderate improvement |
| metric decomposition (single model) | 0.5974 | Single-model pre-comparison lacks diversity; limited gain |
| metric decomposition (diverse model) | 0.6466 | Multi-model responses provide richer differences |
| ours | 0.6962 | Best performance after adding criteria weights and baseline answer |

### Key Findings
- Fine-grained analysis reveals "capability spikes" invisible in total scores. GPT-OSS-120B is strong overall but ranks 7th in roleplay; GPT-OSS-20B is weaker overall but ranks 2nd in coding.
- Coding capability should not be viewed only as an aggregate code score. GPT-OSS-120B is strong in Python, JavaScript, Go, and Rust, but weaker in C and Java; GPT-OSS-20B ranks 2nd in Python and Rust, exceeding the 120B version in C and C#.
- Knowledge domains show significant inhomogeneity. GPT-OSS-120B ranks 1st in computer science and aerospace engineering within technical engineering, but drops to 11th in bioengineering, indicating local gaps in pre-training knowledge.

## Highlights & Insights
- The most valuable aspect of this paper is shifting evaluation from "model ranking" to a "capability map." This perspective is more practical for development, as data ratios, post-training strategies, and deployment scenarios require knowing specific weaknesses rather than just average scores.
- TaxBuilder's recursive insertion is simple but effective. It doesn't force the LLM to understand the whole tree at once, turning tree construction into local decisions suitable for absorbing new tasks, domains, and user needs.
- RealMix embodies a good principle for synthetic evaluation data: don't create questions in a vacuum; perform controlled recombination of real user content. This reduces contamination risk and stays closer to real usage than pure templates.
- PC2's insight is powerful: pairwise advantages don't require quadratic complexity; the key is obtaining "where the differences lie for this question." This can be migrated to safety, factuality, or long-context evaluations requiring query-specific rubrics.

## Limitations & Future Work
- The authors point out that current SCAN is primarily for language models and does not support multimodal models. VLM, embodied AI, or audio models will require extended taxonomy and synthesis mechanisms; the paper mentions exploring SCAN-Anything.
- Current implementation lacks evaluation of AI mechanism dimensions like safety, honesty, factuality, hallucination, and fairness. These require additional taxonomy and judge designs for behavioral constraints and risk attributes.
- Automatic tree construction depends on LLM labeling, potentially inheriting model bias. While pruning and manual editability mitigate this, tree correctness requires continuous auditing.
- Evaluation queries come from synthesis; even with RealMix using real fragments, they might not represent high-risk deployment scenarios. Future work could integrate online failure cases, user feedback, and red-teaming samples.

## Related Work & Insights
- **vs Chatbot Arena**: Chatbot Arena uses massive human preferences for credible rankings; SCAN uses smaller but structured datasets for fine-grained profiles. The former is for macro-comparison; the latter for model diagnosis and training feedback.
- **vs AlpacaEval / MT-Bench**: These focus on overall instruction following or multi-turn performance; SCAN emphasizes extracting labels from real queries and decomposing performance along a tree for better interpretability.
- **vs pairwise LLM-as-a-Judge**: Pairwise is reliable but expensive; PC2 shifts "difference discovery" to criterion extraction, allowing pointwise-style scoring during large-scale evaluation.
- **Key Insight**: For any system where "average metrics mask local failures," one can learn from SCAN: build a task taxonomy, ensure enough samples per node, and create a navigable failure map.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Not the first auto-eval framework, but combines taxonomy, synthetic data, PC2 judge, and diagnostic visualization into a complete paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Solid data scale, judge comparisons, and 21-model analysis, though human evaluation and cross-domain validation could be stronger.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation and evidence; some fine-grained results require visiting the project page due to appendix length.
- Value: ⭐⭐⭐⭐⭐ Highly practical for developers, shifting focus from "leaderboard optimization" to "weakness localization and data loop."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](zero-shot_large_language_models_for_automatic_readability_assessment.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2026\] LLMs as annotators of credibility assessment in Danish asylum decisions: evaluating classification performance and errors beyond aggregated metrics](llms_as_annotators_of_credibility_assessment_in_danish_asylum_decisions_evaluati.md)

</div>

<!-- RELATED:END -->
