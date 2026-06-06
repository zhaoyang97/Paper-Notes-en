---
title: >-
  [Paper Note] Tracing Relational Knowledge Recall in Large Language Models
description: >-
  [ACL 2026][Interpretability][Relational Knowledge] This paper systematically investigates the internal mechanisms by which LLMs recall relational knowledge during text generation. It finds that per-head attention contrib…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Relational Knowledge"
  - "Attention Head Attribution"
  - "Linear Probes"
  - "Knowledge Recall Mechanisms"
  - "Feature Attribution"
date: 2026-05-08
content_hash: e2d38c6c753845c1
---

# Tracing Relational Knowledge Recall in Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.19934](https://arxiv.org/abs/2604.19934)  
**Code**: [nicpopovic.com/publications/tracing](https://nicpopovic.com/publications/tracing)  
**Area**: Interpretability / Knowledge Representation
**Keywords**: Relational Knowledge, Attention Head Attribution, Linear Probes, Knowledge Recall Mechanisms, Feature Attribution

## TL;DR
This paper systematically investigates the internal mechanisms by which LLMs recall relational knowledge during text generation. It finds that per-head attention contributions to the residual stream ($\Delta_{att,h}$) serve as the strongest features for linear relation classification (91% accuracy), and proposes two probe attribution methods—HeadScore and TokenScore—to decompose predictions to the attention head and source token levels, revealing clear correlations between probe accuracy and relation specificity, entity connectivity, and probe signal concentration.

## Background & Motivation

**Background**: How LLMs store and recall relational knowledge is a central question in interpretability research. Prior work has established a canonical picture of knowledge recall: (1) subject entity information accumulates at the last token of the subject span in middle-to-late layers; (2) predicate/relation information is aggregated via attention heads to the token position preceding object generation; (3) the object entity is retrieved from MLP sublayers via attention. This process can sometimes be approximated by a linear transformation and traced to relation-specific neurons.

**Limitations of Prior Work**: While probing has been shown to reliably support named entity recognition and disambiguation for entity representations, it remains unclear which internal representations support faithful linear relation classification, and why certain relation types are more amenable to linear capture than others. Existing analyses cannot simultaneously attribute relational predictions to specific attention heads and source tokens.

**Key Challenge**: Although attention heads and MLPs are known to play roles in knowledge recall, a probe-based attribution method capable of jointly explaining relational classification at both the attention head and token levels is lacking, preventing systematic understanding of the factors that drive success or failure in relation classification.

**Goal**: (1) Identify the LLM internal representations best suited for linear relation classification; (2) Determine what factors predict probe success or failure on relation classification.

**Key Insight**: The paper focuses on per-head attention contributions to the residual stream, as these features decompose naturally to the source token level, making attribution analysis tractable.

**Core Idea**: Per-head attention contributions $\Delta_{att,h}$ are used as features for linear probes performing relation classification. Two attribution methods—HeadScore (attention head-level attribution) and TokenScore (source token-level attribution)—are then applied to decompose probe predictions, enabling fine-grained analysis.

## Method

### Overall Architecture
The system frames relational knowledge recall as a controlled generation scenario using cloze-style prompts. Features consisting of per-head attention contributions are extracted at the position preceding object entity generation, and linear probes are trained for relation classification. HeadScore and TokenScore attribution methods are then applied to analyze which attention heads and source tokens drive probe predictions. Systematic evaluation is conducted on the FewRel validation set across four instruction-tuned LLMs (LLaMA-3.2 1B/3B, LLaMA-3.1 8B, Qwen3 4B).

### Key Designs

1. **Per-Head Attention Contribution Features ($\Delta_{att,h}$)**:

    - Function: Provide the strongest and most traceable feature representation for linear relation classification.
    - Mechanism: For a target position $t$ (preceding object generation), the contribution of attention head $h$ to the residual stream is $\Delta_{att,h}(t) = W_{O,h}(\sum_j \text{Attn}_h(t,j) V_h(j))$, i.e., the result of attention-weighted aggregation projected through the output projection matrix. This can be further decomposed into per-source-token contributions $\Delta_{att,h}(t,j) = W_{O,h}(\text{Attn}_h(t,j) V_h(j))$. Compared to full attention or MLP states, per-head contributions yield higher classification accuracy while preserving the ability to trace predictions to specific attention heads and source tokens.
    - Design Motivation: Full attention or MLP states, while informationally rich, are not traceable. Decomposing features to the attention head level allows each feature dimension to be unambiguously attributed to a specific head and token, providing the foundation for downstream HeadScore and TokenScore attribution.

2. **HeadScore and TokenScore Attribution Methods**:

    - Function: Decompose the predictions of a trained linear probe to the attention head and source token levels.
    - Mechanism: Given the probe weight matrix $W$ and predicted class $\hat{c}$, a contrastive direction is defined as $\Delta W = W_{\hat{c}} - \sum_{c \neq \hat{c}} \pi_c W_c$ (softmax-weighted competing class weights). HeadScore aggregates per-feature contributions $\Delta W_m x_m$ by attention head: $\text{HeadScore}_{\ell,h} = \sum_{m:\ell_m=\ell, h_m=h} \Delta W_m x_m$. TokenScore further exploits the token decomposition of head contributions to refine attribution to the source token level: $\text{TokenScore}_\ell(j) = \sum_{m:\ell_m=\ell} \Delta W_m \cdot [\Delta_{att,h_m}(t,j)]_{d_m}$.
    - Design Motivation: HeadScore reveals which attention heads contribute most to classification, while TokenScore identifies which input tokens the decision signal originates from. This enables error analysis and lexical shortcut detection.

3. **Analysis of Predictors of Relation Classification Performance**:

    - Function: Identify factors that explain performance variation of linear probes across relation types.
    - Mechanism: Under a 16-way-5-shot setting, probe accuracy is analyzed for correlations with four factors: (1) Wikidata output range (the number of distinct objects associated with a relation type)—negatively correlated; (2) average entity connectivity (the number of Wikidata properties shared between subject-object entity pairs)—negatively correlated; (3) inter-example TF-IDF lexical similarity—positively correlated; (4) number of heads required to accumulate 95% of the HeadScore mass—negatively correlated (more concentrated signal yields higher accuracy).
    - Design Motivation: The first three factors characterize input data difficulty, while the fourth is an intrinsic probe property that serves as an annotation-free diagnostic indicator of probe behavior.

### Loss & Training
Linear probes are trained with cross-entropy loss and the Adam optimizer for 200 epochs. RelSpec expert feature selection is applied, retaining the top 3,000 features per relation type. Evaluation follows the n-way k-shot protocol on the FewRel validation set, with all results averaged over 5 random seeds × 500 episodes.

## Key Experimental Results

### Main Results (5-way-5-shot Relation Classification Accuracy, %)

| Feature Type | LLaMA-3.2 1B | LLaMA-3.2 3B | LLaMA-3.1 8B | Qwen3 4B |
|---|---|---|---|---|
| Attention (full) | 83.65 | 86.61 | 86.79 | 75.06 |
| $\Delta_{att,h}$ (per-head) | **90.26** | **91.06** | **91.09** | **89.66** |
| MLP (full) | 85.79 | 86.40 | 85.90 | 80.37 |
| $\Delta_{MLP,h}$ (per-head) | 89.96 | 90.90 | 89.99 | 88.43 |
| $\Delta_{att,e_1}$ (entity) | 59.64 | 60.16 | 59.85 | 59.58 |

### Ablation Study (Lexical Shortcut Analysis)

| Model | Spearman ρ | Mass | StrongAlign× (%) |
|---|---|---|---|
| LLaMA-3.2 1B | 0.115 | 0.491 | 7.7 |
| LLaMA-3.1 8B | 0.095 | 0.475 | 5.3 |
| Qwen3 4B | 0.099 | 0.490 | 5.9 |

### Key Findings
- $\Delta_{att,h}$ is the strongest relation classification feature across all models, consistently outperforming full attention/MLP states and all other variants, achieving over 90% accuracy.
- Restricting to contributions from source entity tokens only ($\Delta_{att,e_1}$) yields near-chance performance (~59–60%), indicating that relational signals are not encoded exclusively on entity tokens.
- Probe accuracy varies substantially across relation types (e.g., F1 of 39.76% for "part of" vs. 99.24% for "constellation"), with negative correlations to output range and entity connectivity.
- Relation types whose HeadScore signal is concentrated in fewer attention heads exhibit higher probe accuracy, potentially linked to feature superposition.
- Only 5.3%–7.7% of errors are consistent with lexical shortcuts, indicating that linear relation probe decisions are not primarily driven by lexical cues.

## Highlights & Insights
- **Per-head contributions outperform full states**: Counterintuitively, decomposed per-head features outperform the informationally richer full states for linear classification. This may be because decomposition removes interference among heads, facilitating linear separability. This finding has implications for probing methodology in LLM interpretability more broadly.
- **HeadScore concentration as an annotation-free diagnostic**: The concentration of probe signal (the number of heads needed to accumulate 95% of HeadScore mass) is an intrinsic probe property independent of labeled data, and can predict probe performance on new relation types—providing a practical tool for assessing probe reliability.
- **TokenScore for probing behavior analysis**: By refining attribution to the token level, one can examine whether a probe relies on semantically relevant tokens (e.g., "crosses") or co-occurrence tokens (e.g., "bridge"), offering a fine-grained tool for diagnosing probe behavior.

## Limitations & Future Work
- Evaluation is limited to the FewRel validation set, covering a relatively small set of relation types (16 classes).
- All findings are correlational rather than causal; the underlying mechanisms by which output range, connectivity, and related factors influence probe accuracy remain to be verified through causal experiments.
- Models ranging from 1B to 8B parameters are evaluated; larger models are not tested.
- The probe attribution methods explain the probe's decisions rather than the LLM's internal computations; the relationship between the two warrants further investigation.
- Whether non-linear probes can capture more relational information remains unexplored.

## Related Work & Insights
- **vs. Meng et al. (2022) / ROME**: That work focuses on factual localization for knowledge editing; this paper focuses on feature selection and attribution for relation classification.
- **vs. Hernandez et al. (2024)**: That work demonstrates that certain relations can be linearly approximated but does not explain why some relations are more amenable. This paper provides explanations via output range, connectivity, and related factors.
- **vs. Liu et al. (2025)**: That work isolates relational information at the neuron level (primarily in MLP layers); this paper provides traceable probe-based attribution at the attention head level.
- **vs. Chughtai et al. (2024)**: That work uses direct logit attribution to analyze model behavior; TokenScore in this paper analyzes the decisions of task-specific probes.

## Rating
- Novelty: ⭐⭐⭐⭐ The HeadScore/TokenScore attribution methods and the analysis of performance predictors constitute valuable contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four models, multiple feature variants, comprehensive correlation analysis, and lexical shortcut detection.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous formalization, progressively structured experimental design, and exceptionally clear exposition.
- Value: ⭐⭐⭐⭐ Provides systematic methodology and practical tools for probing relational knowledge in LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[ACL 2026\] FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](finesteer_a_unified_framework_for_fine-grained_inference-time_steering_in_large_.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](../../NeurIPS2025/interpretability/table_as_a_modality_for_large_language_models.md)

</div>

<!-- RELATED:END -->
