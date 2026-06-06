---
title: >-
  [Paper Note] CT-FineBench: A Diagnostic Fidelity Benchmark for Fine-Grained Evaluation of CT Report Generation
description: >-
  [ACL 2026][Medical Imaging][CT Report Generation] The authors decompose the vague question of "whether a CT report is good" into a QA checklist verifying "whether every fine-grained attribute of each finding matches." By…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "CT Report Generation"
  - "Fine-grained Evaluation"
  - "QA-based metric"
  - "Clinical Attributes"
  - "CT-RATE"
  - "Merlin"
date: 2026-05-08
content_hash: 948f16b07eb531e2
---

# CT-FineBench: A Diagnostic Fidelity Benchmark for Fine-Grained Evaluation of CT Report Generation

**Conference**: ACL 2026  
**arXiv**: [2604.24001](https://arxiv.org/abs/2604.24001)  
**Code**: Not yet released  
**Area**: Medical Imaging / CT Report Generation Evaluation / QA Evaluation  
**Keywords**: CT Report Generation, Fine-grained Evaluation, QA-based metric, Clinical Attributes, CT-RATE, Merlin

## TL;DR
The authors decompose the vague question of "whether a CT report is good" into a QA checklist verifying "whether every fine-grained attribute of each finding matches." By constructing the CT-FineBench benchmark with 44k questions, they show that sensitivity to clinical errors and correlation with human expert scores significantly exceed existing metrics such as BLEU, BERTScore, RadGraph, RaTEScore, and GREEN.

## Background & Motivation
**Background**: Current evaluation of automated radiology report generation (especially for 3D CT) primarily follows three paths: lexical overlap-based metrics (BLEU/ROUGE), embedding-based metrics (BERTScore), and more "medical-aware" entity-level metrics (CheXbert F1, RadGraph, RaTEScore) or LLM-as-Judge (GREEN).

**Limitations of Prior Work**: The first category only considers literal similarity, assigning high scores to reports that "sound similar but are clinically incorrect." The second category only compares coarse-grained entities (e.g., whether a "lung nodule" is mentioned) but ignores fine-grained attributes that actually determine diagnosis—location, size, morphology, density, and margin. The third category (LLM black-box scoring) lacks unified standards, has poor interpretability, and fluctuates wildly across datasets (GREEN drops from 35.8 on CT-RATE to 1.3 on Merlin). Since CT reports are dense with attributes, writing "1.2cm solid nodule in the right lung" as "5mm ground-glass nodule in the left lung" is almost undetectable by traditional metrics but is a clinical disaster.

**Key Challenge**: Evaluation needs to be "sensitive to clinical errors" and "robust to phrasing variations." Existing metrics either fail at both or confuse them—BLEU/ROUGE drops to 28 on paraphrased reports (CT-RATE-pos) while surging to 70 on erroneous reports (CT-RATE-neg), showing a completely reversed trend.

**Goal**: (1) Shift evaluation from coarse-grained finding levels to fine-grained attribute levels; (2) Reformulate "overall scoring" as a fact-checking problem via "step-by-step QA verification," making evaluation interpretable and traceable to specific clinical errors.

**Key Insight**: Borrowing the QA-based factual consistency paradigm (e.g., QAFactEval) from general domains—instead of letting the model judge the report as a whole, directly ask "Where is the lesion?" or "What is the nodule diameter in mm?" and verify the answers.

**Core Idea**: Offline, each reference report is parsed into a "finding × attribute" QA list. Online, a QA model extracts answers from the candidate report, which are then compared with gold answers using type-aware (categorical/numeric/null) graded scoring (0/0.5/1). The final score represents the attribute accuracy.

## Method

### Overall Architecture
CT-FineBench consists of a three-stage pipeline, with the first two stages performed offline ($\Phi_{\text{Build}}$) and the last stage online ($\Phi_{\text{Eval}}$):

1.  **Attribute Definition**: NER triplet mining $(finding, attribute, content)$ is performed on a CT report dataset, followed by manual cleaning through four steps (Remove/Split/Merge/Comment) to stabilize a finding→attribute hierarchy schema (e.g., "lung nodule" $\rightarrow$ {Location, Shape, Density, Margin}).
2.  **QA Construction**: For each reference report $x$, positive findings and their corresponding attributes in the schema are traversed. Few-shot prompting is used to generate $(q_i, a_i)$ pairs. Attributes not mentioned in the report are discarded. Finding-existence QA is added to cover both coarse and fine granularities. This produces the benchmark $D_{\text{QA}} = \Phi_{\text{Build}}(\{x\}) = \{(q_i, a_i)\}$.
3.  **Evaluation**: For a candidate report $\hat x$, the corresponding $D_{\text{QA}}(x)$ is retrieved. The QA module $\hat a_i = \Phi_{\text{QA}}(q_i, \hat x)$ extracts answers (returning a special token `[NULL]` if not found). The Compare module assigns 0/0.5/1 based on attribute types, followed by averaging: $\text{Score}(x,\hat x) = \Phi_{\text{Eval}}(\hat x, D_{\text{QA}}(x))$.

The pipeline consumes significant LLM and human effort only in the offline stage; online evaluation can be run using small local models, making it suitable for large-scale iterative evaluation.

### Key Designs

1.  **Fine-grained Attribute Schema with Human-in-the-loop Curation**:
    - **Function**: Explicitly builds a finding→attribute tree as the ontology for "which dimensions radiology evaluation should focus on."
    - **Mechanism**: Triplets $(finding, attribute, content)$ are first extracted using Qwen3-Max across the full training and test sets. Tail entities with frequency below 50 are discarded. The remainder is refined by 4 annotators using Remove/Split/Merge/Comment; annotators are permitted to use Gemini-2.5-Pro or GPT-5 to query clinical knowledge. Ultimately, CT-RATE yields 94 unique attributes (avg. 5.2 per finding), and Merlin yields 89 attributes (avg. 3.0 per finding).
    - **Design Motivation**: Purely automated extraction carries noise and redundancy (e.g., vague terms like "feature"), while purely manual extraction cannot be exhaustive. This hybrid workflow ensures the schema covers clinical essentials while maintaining mutual exclusivity.

2.  **Type-aware Graded Scoring**:
    - **Function**: Refines "answer correctness" into differentiated scoring rules for different attribute types to avoid noise from 0/1 hard judgments.
    - **Mechanism**: Categorical/Location attributes (e.g., density = solid / ground-glass) use "synonym-aware exact match"—1.0 for exact, 0.5 for partially correct or overly specific, and 0 for incorrect. Numeric attributes (size, density values) are first standardized by unit, then graded by relative error $\epsilon = |a - \hat a|/|a|$: 1.0 if $\epsilon < 10\%$, 0.5 if $10\% \le \epsilon < 30\%$, and 0 if $\epsilon \ge 30\%$. Outputs of `[NULL]` are treated as false negatives (0 points).
    - **Design Motivation**: In CT, "1.0 cm vs 1.1 cm" is not necessarily wrong, but "1 cm vs 5 cm" must be penalized. Numeric range grading captures true deviations within clinical tolerance.

3.  **Recall-oriented Sensitivity Construction**:
    - **Function**: Ensures the benchmark is highly sensitive to clinical errors while remaining robust to phrasing changes, verified through two probe sets.
    - **Mechanism**: All QA is derived only from reference reports (recall-oriented), so the metric naturally penalizes "omissions." To verify sensitivity, two types of probe reports are constructed via LLM: CT-RATE-neg / Merlin-neg (preserving phrasing but injecting minimal clinical attribute changes) and CT-RATE-pos / Merlin-pos (paraphrasing while strictly preserving clinical facts). An ideal metric should score near 1 on pos and significantly lower on neg. CT-FineBench achieves pos=74.5/neg=39.1 on CT-RATE, being the only metric among six to satisfy expectations in both directions.
    - **Design Motivation**: Correlation with human scores is an a posteriori metric; sensitivity must first be verified with counterfactual samples to identify exactly what the metric is measuring.

### Loss & Training
This work presents a benchmark rather than a training method, so no explicit loss function is used. Key hyperparameters: triplet frequency threshold 50; numeric scoring thresholds 10% and 30%; default evaluation LLM is Qwen3-Max, with Qwen3-32B/8B as lightweight alternatives; acceleration via vLLM on A800 GPUs. The accompanying CT-FineData (439,665 QA pairs / 44,302 reports) is generated from the training set for future direct optimization of fine-grained attribute accuracy.

## Key Experimental Results

### Main Results
Several CT report generation models were evaluated on CT-RATE (chest, 1564 reports) and Merlin (abdomen, 5082 reports). Key results on CT-RATE:

| Report | BLEU-2 | ROUGE-L | BERTScore | RadGraph | RaTEScore | GREEN | CT-FineBench |
|------|--------|---------|-----------|----------|-----------|-------|--------------|
| RadFM | 4.1 | 12.0 | 80.6 | 2.3 | 40.7 | 3.2 | 4.4 |
| Hulu-Med | 11.5 | 20.0 | 84.2 | 9.5 | 49.8 | 15.3 | 12.2 |
| CT-CHAT | 29.0 | 35.6 | 87.5 | 21.5 | 65.2 | 35.8 | 15.8 |
| CT-RATE-pos (Paraphrased) | 28.0 | 34.3 | 89.5 | 22.2 | 77.3 | 56.1 | **74.5** |
| CT-RATE-neg (Clincally Wrong) | 70.0 | 75.3 | 95.0 | 42.9 | 76.2 | 19.2 | **39.1** |

Observations: (1) Absolute scores of CT-FineBench on real models are lower than BLEU/BERTScore, indicating that current generators are weak in fine-grained clinical precision. (2) Only CT-FineBench achieves high pos + low neg (74.5 vs 39.1); other metrics are either insensitive (pos $\approx$ neg for BERTScore/RaTEScore) or inverted (BLEU-2 gives wrong reports a higher score of 70.0 vs 28.0).

### Ablation Study
The online QA + Compare module was replaced with smaller models to verify portability:

| Config | CT-RATE Acc | CT-RATE Pearson $\tau$ | Merlin Acc | Merlin Pearson $\tau$ |
|------|-------------|--------------------|-------------|-------------------|
| Qwen3-Max (Default) | 15.8 | — | 22.4 | — |
| Qwen3-32B | 15.9 | 0.911 | 22.9 | 0.978 |
| Qwen3-8B | 15.0 | 0.863 | 24.6 | 0.953 |

The Pearson correlation between the 32B version and Max is >0.9 with near-identical absolute scores. Qwen3-8B on an A800 evaluates CT-RATE at 0.42 reports/s, making it suitable for large-scale use.

### Key Findings
- **Correlation with Human Experts**: (Sampling 100 reports, 10-point scale by two radiologists). CT-FineBench outperforms RaTEScore across Pearson (0.622 vs 0.521), Kendall (0.378 vs 0.320), and Spearman (0.490 vs 0.434) on CT-RATE. It also shows the best stability across different anatomical regions.
- **Inter-metric Correlation**: CT-FineBench shows only moderate correlation with traditional metric families, suggesting it captures an orthogonal "fine-grained clinical signal."
- **SOTA Gap**: Even SOTA models (CT-CHAT, Merlin) score only 15.8 / 22.4 on CT-FineBench, revealing that "seemingly fluent but detail-deficient" generation is the primary bottleneck in this field.

## Highlights & Insights
- **Reformulating Evaluation as QA Fact-checking**: This allows the metric to pinpoint exactly which attribute is wrong, providing natural interpretability. Each low score corresponds to a (q, a, â) triplet for error analysis.
- **Methodology of Counterfactual Probes**: Synthesizing pos/neg reports to disentangle "sensitivity" and "robustness" into quantifiable axes is a rigorous approach that could be applied to other summarization or code generation tasks.
- **Engineering Feasibility**: The >0.9 correlation of 8B/32B models with Max means the benchmark can be used frequently in local labs without API dependence.
- **Implicit Contribution of CT-FineData**: The outputted 440k QA pairs serve as supervision signals to align training objectives with fine-grained clinical accuracy.

## Limitations & Future Work
- **Limitations**: (1) Recall-oriented design does not penalize hallucinations (hallucinated findings) and should be used with precision-based metrics; (2) The attribute schema is limited to findings labeled in the original dataset; (3) Numeric thresholds are heuristic.
- **Future Work**: (1) Introduce open-attribute discovery to extend evaluation to novel findings; (2) Replace numeric thresholds with attribute-specific tolerance tables from clinical guidelines; (3) Incorporate CT-FineData into RLHF/GRPO to directly optimize attribute hit rates.

## Related Work & Insights
- **vs RadGraph / RaTEScore**: While they operate at the entity level (finding + anatomy), CT-FineBench dives into the attribute level (size, density) with differentiated scoring, leading to higher human correlation.
- **vs GREEN / LLM-as-judge**: GREEN judges the whole report, leading to high fluctuations across datasets. CT-FineBench decomposes "judgment" into "extraction + comparison," which is significantly more robust.
- **vs QAFactEval**: The core idea is shared, but CT-FineBench adds domain-specific finding→attribute schemas and numeric grading, proving the efficacy of QA-based evaluation in high-expertise scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ Solid migration of QA-based consistency to CT reports with structured schemas.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete coverage of multiple anatomical datasets, metrics, and counterfactual probes.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; good integration of formulas and flowcharts.
- Value: ⭐⭐⭐⭐⭐ Addresses a critical pain point in clinical report evaluation with a tool directly useful to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)
- [\[CVPR 2026\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](../../CVPR2026/medical_imaging/unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[ACL 2026\] Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)
- [\[ACL 2026\] CT-Flow: Orchestrating CT Interpretation Workflow with Model Context Protocol Servers](ct-flow_orchestrating_ct_interpretation_workflow_with_model_context_protocol_ser.md)
- [\[ICML 2026\] Foundation VAEs for 3D CT Reconstruction, Augmentation, and Generation](../../ICML2026/medical_imaging/foundation_vaes_for_3d_ct_reconstruction_augmentation_and_generation.md)

</div>

<!-- RELATED:END -->
