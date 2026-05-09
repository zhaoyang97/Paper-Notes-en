---
title: >-
  [Paper Note] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations
description: >-
  [ACL 2026][Self-consistency] This paper constructs a large-scale Post-hoc Self-Consistency Bank (PSCB, 85K decisions × 428K explanations), quantifies the feature attribution gap between LLM answers and their explanations, and improves attribution consistency of explanations via DPO optimization without sacrificing accuracy.
tags:
  - ACL 2026
  - Self-consistency
  - Feature Attribution
  - Explanation Faithfulness
  - DPO Optimization
  - Attribution Alignment
date: 2026-05-08
content_hash: 8170e6438e0e2995
---

# Aligning What LLMs Do and Say: Towards Self-Consistent Explanations

**Conference**: ACL 2026
**arXiv**: [2506.07523](https://arxiv.org/abs/2506.07523)
**Code**: [GitHub](https://github.com/saharad1/ConstLLM)
**Area**: Interpretability
**Keywords**: Self-consistency, Feature Attribution, Explanation Faithfulness, DPO Optimization, Attribution Alignment

## TL;DR

This paper constructs a large-scale Post-hoc Self-Consistency Bank (PSCB, 85K decisions × 428K explanations), quantifies the feature attribution gap between LLM answers and their explanations, and improves attribution consistency of explanations via DPO optimization without sacrificing accuracy.

## Background & Motivation

**Background**: LLMs are frequently asked to generate natural language explanations for their answers, yet these post-hoc explanations often fail to align with the input features that actually drive the answers — i.e., what the model says differs from what it does.

**Limitations of Prior Work**: (1) Existing faithfulness metrics (e.g., counterfactual interventions) are computationally expensive and difficult to apply at scale; (2) Methods such as CC-SHAP have only been evaluated on approximately 100 samples, limiting the reliability of their conclusions; (3) No prior work has demonstrated how to improve attribution inconsistency.

**Key Challenge**: LLM explanations may be fluent and plausible yet "miss the point" — the input features they highlight differ from those that actually drive the answer, posing a fundamental threat to trustworthy AI.

**Goal**: (1) Quantify attribution consistency between answers and explanations at scale; (2) Propose methods to improve it.

**Key Insight**: Feature attribution vectors are computed separately for each QA decision and its multiple explanations, and their alignment is compared. DPO fine-tuning on attribution preference data is then applied to improve consistency.

**Core Idea**: Spearman rank correlation better discriminates between high- and low-quality explanations than cosine similarity; DPO optimization on attribution preferences effectively improves self-consistency and generalizes across domains.

## Method

### Overall Architecture

PSCB construction pipeline: (1) Compute feature attribution vectors for QA decisions; (2) Generate $K$ diverse explanations per decision and compute attribution vectors for each; (3) Measure decision–explanation attribution alignment using an alignment function; (4) Select the best and worst explanations to construct preference pairs for DPO optimization.

### Key Designs

1. **Post-hoc Self-Consistency Bank (PSCB)**:

    - **Function**: Provides a large-scale attribution-augmented QA benchmark.
    - **Mechanism**: 85K decisions × 5 explanations each = 428K explanation–attribution pairs. Two attribution methods are used — LIME and Layer Integrated Gradients (LIG) — covering 4 QA datasets and 2 LLMs.
    - **Design Motivation**: Prior evaluation was limited to approximately 100 samples, precluding reliable conclusions. Large-scale data is a prerequisite for systematic study.

2. **Spearman Rank Correlation as Alignment Metric**:

    - **Function**: More reliably measures attribution alignment than cosine similarity.
    - **Mechanism**: Spearman rank correlation $CC_{sp} = 1 - \frac{6\sum(r(\phi_i^{dec}) - r(\phi_i^{exp}))^2}{m(m^2-1)}$ captures consistency in feature priority ordering and is invariant to attribution scale.
    - **Design Motivation**: Cosine similarity yields heavily overlapping distributions when distinguishing good from bad explanations (poor discriminative power), whereas Spearman rank correlation clearly separates explanations of different quality.

3. **Attribution-Preference-Based DPO Optimization**:

    - **Function**: Improves explanation self-consistency without degrading task accuracy.
    - **Mechanism**: Preference pairs are constructed using the highest-consistency explanation as *chosen* and the lowest-consistency explanation as *rejected* from PSCB, followed by DPO fine-tuning of the LLM.
    - **Design Motivation**: SFT on the same data performs poorly; DPO is better suited to learning the subtle distinctions in attribution preferences.

### Loss & Training

The standard DPO objective is used, with training conducted on the preference pairs from PSCB. Explanations are generated via temperature sampling ($p=0.9$, $T=0.7$), with 5 explanations per decision; the best and worst are selected to form preference pairs.

## Key Experimental Results

### Main Results

| Model | Dataset | CC-Sp (Before) | CC-Sp (After DPO) | Accuracy Change |
|-------|---------|----------------|-------------------|-----------------|
| LLaMA3.1-8B | ECQA | 18.47 (mean) | Significant gain | No degradation |
| LLaMA3.2-3B | ECQA | 9.75 (mean) | Significant gain | No degradation |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| DPO vs. SFT | DPO significantly outperforms SFT | SFT fails to learn attribution preferences |
| LIME vs. LIG | Gains do not transfer across methods | Different attribution methods capture different dimensions |
| Cross-domain generalization | Effective | Improvements from ECQA training generalize to ARC, etc. |
| Correct vs. incorrect answers | Orthogonal | Self-consistency is largely independent of accuracy |

### Key Findings
- Self-consistency and accuracy are largely orthogonal — inconsistently explained answers may still be correct, and consistent ones may be wrong.
- Spearman rank correlation demonstrates significantly stronger discriminative power than cosine similarity.
- Self-consistency gains from DPO optimization generalize across domains but not across attribution methods.
- Different attribution methods (LIME vs. LIG) capture fundamentally distinct notions of input relevance.

## Highlights & Insights
- The finding that "self-consistency and accuracy are orthogonal" is significant — accurate models do not necessarily produce faithful explanations.
- The paper reveals a practical tension: DPO can improve LIME-based consistency without improving LIG-based consistency, indicating that "faithful explanation" is itself a multi-dimensional concept.
- PSCB as a large-scale resource holds long-term value for the interpretability community.

## Limitations & Future Work
- Evaluation is limited to multiple-choice QA; applicability to open-ended generation tasks remains unknown.
- Both LIME and LIG have inherent limitations; more advanced attribution methods may yield different conclusions.
- Self-consistency remains a proxy for faithfulness and does not equate to true interpretability of the decision process.
- Future work may extend to larger models and broader task types.

## Related Work & Insights
- **vs. CC-SHAP**: Scales evaluation from 100 samples to 85K and, for the first time, demonstrates an improvement methodology.
- **vs. Counterfactual Intervention Methods**: Replaces expensive counterfactual testing with attribution vector comparison, substantially reducing cost.
- **vs. RLHF**: Extends preference learning from "human preferences" to "attribution consistency preferences," representing a new dimension of alignment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Attribution-preference DPO optimization is an entirely new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale benchmark, cross-domain generalization, DPO vs. SFT comparison.
- Writing Quality: ⭐⭐⭐⭐ Formally rigorous with clear experimental design.
- Value: ⭐⭐⭐⭐⭐ Far-reaching implications for LLM interpretability and trustworthy AI.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)
- [\[ACL 2026\] Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models](revitalizing_black-box_interpretability_actionable_interpretability_for_llms_via.md)
- [\[AAAI 2026\] LLM Circuit Analyses Are Consistent Across Training and Scale](../../AAAI2026/interpretability/llm_circuit_analyses_consistent_across_training_and_scale.md)
- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)
- [\[ACL 2026\] IDEA: An Interpretable and Editable Decision-Making Framework for LLMs via Verbal-to-Numeric Calibration](idea_an_interpretable_and_editable_decision-making_framework_for_llms_via_verbal.md)

<!-- RELATED:END -->
