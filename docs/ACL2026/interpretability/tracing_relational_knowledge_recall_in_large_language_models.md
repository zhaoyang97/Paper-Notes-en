---
title: >-
  [Paper Note] Tracing Relational Knowledge Recall in Large Language Models
description: >-
  [ACL 2026][Interpretability][Relational knowledge] This paper systematically investigates the internal mechanisms of LLMs for recalling relational knowledge during text generation. It finds that head-wise contributions t…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Relational knowledge"
  - "attention head attribution"
  - "linear probes"
  - "knowledge recall mechanisms"
  - "feature attribution"
date: 2026-05-08
content_hash: 678d2968fd8703da
---

# Tracing Relational Knowledge Recall in Large Language Models

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.19934](https://arxiv.org/abs/2604.19934)  
**Code**: [nicpopovic.com/publications/tracing](https://nicpopovic.com/publications/tracing)  
**Area**: Interpretablity / Knowledge Representation  
**Keywords**: Relational knowledge, attention head attribution, linear probes, knowledge recall mechanisms, feature attribution

## TL;DR
This paper systematically investigates the internal mechanisms of LLMs for recalling relational knowledge during text generation. It finds that head-wise contributions to the residual stream ($\Delta_{att,h}$) are the strongest features for linear relation classification (achieving 91% accuracy). The study proposes HeadScore and TokenScore attribution methods to decompose predictions to the attention head and source token levels, revealing clear correlations between probe accuracy and relation specificity, entity connectedness, and probe signal concentration.

## Background & Motivation

**Background**: How LLMs store and recall relational knowledge is a central question in interpretability research. Existing studies have revealed a typical picture of knowledge recall: (1) subjective entity information accumulates at the last token of the subject span in middle-to-late layers; (2) predicate/relation information accumulates at the token position prior to object generation via attention heads; (3) object entities are retrieved from MLP sublayers via attention. This process can sometimes be approximated by linear transformations and traced back to relation-specific neurons.

**Limitations of Prior Work**: For entity representations, studies have shown that named entity recognition and disambiguation can be reliably performed via probes. However, for relation type features, it remains unclear which internal representations support faithful linear relation classification, or why certain relation types are more easily captured linearly than others. Existing analyses cannot trace relation predictions back to specific attention heads and source tokens simultaneously.

**Key Challenge**: While attention heads and MLPs are known to play roles in knowledge recall, there is a lack of a probing method capable of simultaneous attribution at both the head and token levels to systematically understand factors for success and failure in relation classification.

**Goal**: (1) Identify the most suitable internal LLM representations for linear relation classification; (2) Determine which factors predict the success or failure of probes in relation classification.

**Key Insight**: Focus on the head-wise contributions of attention heads to the residual stream, as these features naturally decompose to the source token level, making attribution analysis feasible.

**Core Idea**: Use head-wise contributions $\Delta_{att,h}$ as linear probe features for relation classification, and decompose probe prediction decisions through two methods: HeadScore (head-level attribution) and TokenScore (source token-level attribution).

## Method

### Overall Architecture
The system sets up relational knowledge recall as a controlled generation scenario using cloze-style prompts. Features of attention head contributions are extracted at the position before object entity generation to train linear probes for relation classification. Subsequently, HeadScore and TokenScore attribution methods analyze which heads and source tokens drive probe predictions. System evaluation is conducted on four instruction-tuned LLMs (LLaMA-3.2 1B/3B, LLaMA-3.1 8B, Qwen3 4B) using the FewRel validation set.

### Key Designs

1.  **Head-wise Contribution Features ($\Delta_{att,h}$)**:

    - **Function**: Provide the strongest and most traceable feature representation for linear relation classification.
    - **Mechanism**: For a target position $t$ (pre-object generation), the contribution of attention head $h$ to the residual stream is $\Delta_{att,h}(t) = W_{O,h}(\sum_j \text{Attn}_h(t,j) V_h(j))$, representing the result passed through the output projection matrix after attention-weighted aggregation. This can be further decomposed into contributions from individual source tokens $j$: $\Delta_{att,h}(t,j) = W_{O,h}(\text{Attn}_h(t,j) V_h(j))$. Compared to full attention or MLP states, head-wise contributions provide higher classification accuracy while maintaining traceability.
    - **Design Motivation**: Full attention or MLP states are information-rich but non-traceable. Decomposing features to the head level allows each feature dimension to be explicitly attributed to specific heads and tokens, laying the foundation for downstream HeadScore and TokenScore attribution.

2.  **HeadScore and TokenScore Attribution Methods**:

    - **Function**: Decompose predictions of trained linear probes to the head and source token levels.
    - **Mechanism**: Given a probe weight matrix $W$ and predicted class $\hat{c}$, a contrastive direction is defined as $\Delta W = W_{\hat{c}} - \sum_{c \neq \hat{c}} \pi_c W_c$ (softmax-weighted weights of competing classes). HeadScore aggregates contributions per feature $\Delta W_m x_m$ by head: $\text{HeadScore}_{\ell,h} = \sum_{m:\ell_m=\ell, h_m=h} \Delta W_m x_m$. TokenScore further utilizes the token-level decomposition of head contributions to refine attribution down to source tokens: $\text{TokenScore}_\ell(j) = \sum_{m:\ell_m=\ell} \Delta W_m \cdot [\Delta_{att,h_m}(t,j)]_{d_m}$.
    - **Design Motivation**: HeadScore reveals which heads contribute most to classification, while TokenScore reveals which input tokens the decision signals originate from. This enables error analysis and lexical shortcut detection.

3.  **Predictive Factor Analysis for Relation Classification**:

    - **Function**: Identify factors influencing the performance variance of linear probes across different relation types.
    - **Mechanism**: Under a 16-way-5-shot setting, the correlation between probe accuracy and four factors is analyzed: (1) Wikidata output range (number of distinct objects for a relation type)—negatively correlated; (2) Average entity connectedness (number of Wikidata properties between subject-object pairs)—negatively correlated; (3) TF-IDF lexical similarity between examples—positively correlated; (4) Number of heads required for 95% cumulative HeadScore contribution—negatively correlated (higher accuracy when signals are concentrated in fewer heads).
    - **Design Motivation**: The first three factors characterize input data difficulty, while the fourth is an intrinsic property of the probe, serving as a diagnostic metric for probe behavior without requiring labeled data.

### Loss & Training
Linear probes are trained using cross-entropy loss and the Adam optimizer for 200 epochs. RelSpec expert feature selection is used to select the top 3000 features per relation type. Evaluation follows an n-way k-shot setup on the FewRel validation set, with all results averaged over 5 random seeds $\times$ 500 episodes.

## Key Experimental Results

### Main Results (5-way-5-shot Relation Classification Accuracy, %)

| Feature Type | LLaMA-3.2 1B | LLaMA-3.2 3B | LLaMA-3.1 8B | Qwen3 4B |
| :--- | :--- | :--- | :--- | :--- |
| Attention (Full) | 83.65 | 86.61 | 86.79 | 75.06 |
| $\Delta_{att,h}$ (Head-wise) | **90.26** | **91.06** | **91.09** | **89.66** |
| MLP (Full) | 85.79 | 86.40 | 85.90 | 80.37 |
| $\Delta_{MLP,h}$ (Head-wise) | 89.96 | 90.90 | 89.99 | 88.43 |
| $\Delta_{att,e_1}$ (Entity) | 59.64 | 60.16 | 59.85 | 59.58 |

### Ablation Study (Lexical Shortcut Analysis)

| Model | Spearman ρ | Mass | StrongAlign× (%) |
| :--- | :--- | :--- | :--- |
| LLaMA-3.2 1B | 0.115 | 0.491 | 7.7 |
| LLaMA-3.1 8B | 0.095 | 0.475 | 5.3 |
| Qwen3 4B | 0.099 | 0.490 | 5.9 |

### Key Findings
- $\Delta_{att,h}$ is the strongest relation classification feature across all models, consistently outperforming full attention/MLP states and other variants, with accuracy exceeding 90%.
- Observing only the contribution of the subject entity token ($\Delta_{att,e_1}$) is insufficient for relation classification (~59-60%), suggesting that relation signals are not exclusively encoded on entity tokens.
- Probe accuracy varies significantly across relation types (e.g., F1 of 39.76% for "part of" vs. 99.24% for "constellation"), negatively correlating with output range and entity connectedness.
- Relation types where the HeadScore signal is more concentrated in a few attention heads exhibit higher probe accuracy—this may be related to feature superposition.
- Only 5.3%-7.7% of errors align with lexical shortcuts, indicating that linear relation probe decisions are not primarily driven by lexical cues.

## Highlights & Insights
- **Head-wise contributions outperform full states**: Counter-intuitively, decomposed head-wise features are more suitable for linear classification than the more information-rich full states. This is likely because decomposition removes interference between different heads, facilitating linear separation. This finding serves as a guide for other LLM probing research.
- **HeadScore concentration as a data-free diagnostic**: Signal concentration (how many heads are needed to reach 95% contribution) is an intrinsic property of the probe that does not rely on labeled data and can predict probe performance on new relation types. This provides a practical tool for probe reliability assessment.
- **TokenScore reveals probe behavior**: By refining attribution to the token level, it is possible to verify whether probes rely on semantically relevant tokens (e.g., "crosses") or co-occurring tokens (e.g., "bridge"), providing a granular tool for diagnosing probe behavior.

## Limitations & Future Work
- Evaluated only on the FewRel validation set, with relatively limited relation types (16 classes).
- All findings are correlations rather than causal—the influence mechanisms of output range, connectedness, etc., await verification through causal experiments.
- Evaluation models ranged from 1B to 8B; larger models were not tested.
- The probing method explains probe decisions rather than internal LLM computations; the relationship between the two requires further clarification.
- Non-linear probes were not explored to see if they capture more relational information.

## Related Work & Insights
- **vs. Meng et al. (2022) / ROME**: Focuses on fact localization in knowledge editing, while Ours focuses on feature selection and attribution in relation classification.
- **vs. Hernandez et al. (2024)**: Demonstrated that some relations can be linearly approximated but did not explain why some are easier. Ours provides explanations through factors like output range and connectedness.
- **vs. Liu et al. (2025)**: Isolated relation information at the neuron level (primarily in MLP layers), whereas Ours provides a traceable probing method at the attention head level.
- **vs. Chughtai et al. (2024)**: Used direct logit attribution to analyze model behavior, whereas TokenScore in Ours analyzes task-specific probe decisions.

## Rating
- Novelty: ⭐⭐⭐⭐ HeadScore/TokenScore attribution methods and predictive factor analysis are valuable contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 models, multiple feature variants, comprehensive correlation analysis, and lexical shortcut detection.
- Writing Quality: ⭐⭐⭐⭐⭐ Formally rigorous, step-by-step experimental design, and very clear writing.
- Value: ⭐⭐⭐⭐ Provides a systematic methodology and practical tools for probing relational knowledge in LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall](through_a_compressed_lens_investigating_the_impact_of_quantization_on_factual_kn.md)
- [\[ACL 2026\] MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models](mined_probing_and_updating_with_multimodal_time-sensitive_knowledge_for_large_mu.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)

</div>

<!-- RELATED:END -->
