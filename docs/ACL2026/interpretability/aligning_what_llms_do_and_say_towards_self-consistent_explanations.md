---
title: >-
  [Paper Note] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations
description: >-
  [ACL 2026][Interpretability][Self-consistency] The authors construct the Post-hoc Self-Consistency Bank (PSCB, 85K decisions × 428K explanations) to quantify the feature attribution gap between LLM answers and their expl…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Self-consistency"
  - "Feature Attribution"
  - "Explanation Faithfulness"
  - "DPO optimization"
  - "Attribution Alignment"
date: 2026-05-08
content_hash: 09f0ab6f872f63a5
---

# Aligning What LLMs Do and Say: Towards Self-Consistent Explanations

**Conference**: ACL 2026 Findings  
**arXiv**: [2506.07523](https://arxiv.org/abs/2506.07523)  
**Code**: [GitHub](https://github.com/saharad1/ConstLLM)  
**Area**: Interpretability  
**Keywords**: Self-consistency, Feature Attribution, Explanation Faithfulness, DPO optimization, Attribution Alignment

## TL;DR

The authors construct the Post-hoc Self-Consistency Bank (PSCB, 85K decisions × 428K explanations) to quantify the feature attribution gap between LLM answers and their explanations. They utilize DPO optimization to enhance the attributional consistency of explanations without compromising task accuracy.

## Background & Motivation

**Background**: LLMs are frequently required to generate natural language explanations to justify their answers. However, these post-hoc explanations often remain inconsistent with the actual input features that drive the answer—essentially, what the model "says" differs from what it "does."

**Limitations of Prior Work**: (1) Existing faithfulness metrics, such as counterfactual interventions, are computationally expensive and difficult to apply at scale; (2) Methods like CC-SHAP have only been evaluated on approximately 100 samples, limiting the reliability of their conclusions; (3) Previous research has not demonstrated how to improve this attributional inconsistency.

**Key Challenge**: LLM explanations may be fluent and plausible but fail to reflect the true decision-making process. The input features highlighted by the explanation differ from those that actually determined the answer, posing a fundamental threat to trustworthy AI.

**Goal**: (1) Large-scale quantification of attributional consistency between answers and explanations; (2) Proposal of methods to improve this consistency.

**Key Insight**: For each QA decision and its corresponding explanations, separate feature attribution vectors are calculated. The alignment between these vectors is then measured. DPO is applied to attributional preference data to refine consistency.

**Core Idea**: Spearman rank correlation distinguishes high- and low-quality explanations more effectively than cosine similarity; DPO optimization based on attributional preferences successfully enhances self-consistency and generalizes across domains.

## Method

### Overall Architecture

PSCB Construction Pipeline: (1) Calculate feature attribution vectors for QA decisions; (2) Generate $K$ diverse explanations for each decision and compute their respective attribution vectors; (3) Use an alignment function to measure the attributional consistency between the decision and the explanations; (4) Select the best and worst explanations to construct preference pairs for DPO optimization.

### Key Designs

1.  **Post-hoc Self-Consistency Bank (PSCB)**:
    - **Function**: Provides a large-scale attribution-augmented QA benchmark.
    - **Mechanism**: 85K decisions × 5 explanations each = 428K explanation-attribution pairs. Two attribution methods, LIME and Layer Integrated Gradients (LIG), are used across 4 QA datasets and 2 LLMs.
    - **Design Motivation**: Previous evaluations were limited to ~100 samples, preventing reliable conclusions. Large-scale data is a prerequisite for systematic study.

2.  **Spearman Rank Correlation as Alignment Metric**:
    - **Function**: Measures attribution alignment more reliably than cosine similarity.
    - **Mechanism**: Spearman rank correlation $CC_{sp} = 1 - \frac{6\sum(r(\phi_i^{dec}) - r(\phi_i^{exp}))^2}{m(m^2-1)}$ captures the consistency of feature prioritization, independent of attribution scales.
    - **Design Motivation**: Cosine similarity shows highly overlapping distributions (low discriminative power) when distinguishing good vs. bad explanations, whereas Spearman rank correlation clearly separates them.

3.  **DPO Optimization Based on Attributional Preferences**:
    - **Function**: Increases explanation self-consistency without damaging task accuracy.
    - **Mechanism**: Preference pairs are constructed using the explanation with the highest self-consistency as the "chosen" sample and the lowest as the "rejected" sample, followed by DPO fine-tuning.
    - **Design Motivation**: SFT performs poorly on the same data; DPO better learns the subtle nuances of attributional preferences.

### Loss & Training

The standard DPO objective function is used for training on the PSCB preference pairs. Explanations are generated via temperature sampling ($p=0.9, T=0.7$), with 5 explanations per decision to form the best-worst candidate pairs.

## Key Experimental Results

### Main Results

| Model | Dataset | CC-Sp (Pre-optimization) | CC-Sp (Post-DPO) | Accuracy Change |
|-------|---------|-------------------------|------------------|-----------------|
| LLaMA3.1-8B | ECQA | 18.47 (mean) | Significant Increase | No drop |
| LLaMA3.2-3B | ECQA | 9.75 (mean) | Significant Increase | No drop |

### Ablation Study

| Configuration | Key Findings | Description |
|---------------|--------------|-------------|
| DPO vs SFT | DPO significantly outperforms SFT | SFT fails to learn attributional preferences. |
| LIME vs LIG | Improvements do not cross-generalize | Different attribution methods capture different dimensions. |
| Cross-domain Generalization | Effective | Improvements trained on ECQA generalize to ARC, etc. |
| Correct vs Incorrect Answers | Orthogonal | Self-consistency is largely independent of accuracy. |

### Key Findings
- Self-consistency and accuracy are largely orthogonal—inconsistent explanations can accompany correct answers, and consistent explanations can accompany incorrect ones.
- Spearman rank correlation provides significantly better discriminative power than cosine similarity.
- Improvements in self-consistency via DPO generalize across domains but not across different attribution methods.
- Different attribution methods (LIME vs LIG) capture fundamentally different concepts of input relevance.

## Highlights & Insights
- The "orthogonality of self-consistency and accuracy" is a critical finding—accurate models do not necessarily provide faithful explanations.
- The study reveals a practical contradiction: DPO can improve LIME-based consistency without improving LIG-based consistency, suggesting "faithful explanation" is a multi-dimensional concept.
- PSCB serves as a large-scale resource with long-term value for the interpretability community.

## Limitations & Future Work
- Validation was limited to multiple-choice QA; applicability to open-ended generation tasks remains unknown.
- LIME and LIG have their own limitations; more advanced attribution methods might yield different conclusions.
- Self-consistency remains a proxy for faithfulness and is not equivalent to true transparency of the decision process.
- Future work could extend this to larger models and a broader range of task types.

## Related Work & Insights
- **vs CC-SHAP**: Expands evaluation scale from 100 samples to 85K and demonstrates the first improvement method.
- **vs Counterfactual Intervention Methods**: Replaced expensive counterfactual tests with attribution vector comparisons, significantly reducing costs.
- **vs RLHF**: Extends preference learning from "human preference" to "attributional consistency preference," representing a new dimension of alignment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ DPO optimization for attributional preference is a brand new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale benchmark, cross-domain generalization, and DPO vs SFT comparisons.
- Writing Quality: ⭐⭐⭐⭐ Rigorous formalization and clear experimental design.
- Value: ⭐⭐⭐⭐⭐ Deep implications for LLM interpretability and trustworthy AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)
- [\[ACL 2026\] A Systematic Comparison between Extractive Self-Explanations and Human Rationales in Text Classification](a_systematic_comparison_between_extractive_self-explanations_and_human_rationale.md)
- [\[ACL 2026\] Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives](do_llms_capture_embodied_cognition_and_cultural_variation_cross-linguistic_evide.md)
- [\[ACL 2026\] Diffusion-CAM: Faithful Visual Explanations for dMLLMs](diffusion-cam_faithful_visual_explanations_for_dmllms.md)
- [\[ACL 2026\] Learning What Matters: Dynamic Dimension Selection and Aggregation for Interpretable Vision-Language Reward Modeling](learning_what_matters_dynamic_dimension_selection_and_aggregation_for_interpreta.md)

</div>

<!-- RELATED:END -->
