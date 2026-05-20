---
title: >-
  [Paper Note] Understanding Language Prior of LVLMs by Contrasting Chain-of-Embedding
description: >-
  [ICLR 2026][Dialogue Systems][Language Prior] By contrasting layer-wise hidden representations (chain-of-embedding) with and without visual input…
tags:
  - "ICLR 2026"
  - "Dialogue Systems"
  - "Language Prior"
  - "Visual Integration Point"
  - "Large Vision-Language Models"
  - "Representation Analysis"
  - "Interpretability"
date: 2026-05-08
content_hash: 205bd9a93fed8cd1
---

# Understanding Language Prior of LVLMs by Contrasting Chain-of-Embedding

**Conference**: ICLR 2026
**arXiv**: [2509.23050](https://arxiv.org/abs/2509.23050)  
**Code**: None  
**Area**: Dialogue Systems
**Keywords**: Language Prior, Visual Integration Point, Large Vision-Language Models, Representation Analysis, Interpretability

## TL;DR

By contrasting layer-wise hidden representations (chain-of-embedding) with and without visual input, this paper identifies a "Visual Integration Point" (VIP) layer in LVLMs and proposes the Total Visual Integration (TVI) metric to quantify the strength of language priors.

## Background & Motivation

Large Vision-Language Models (LVLMs) excel at multimodal tasks but frequently over-rely on language priors (LP)—textual statistical patterns memorized during pretraining—while neglecting actual visual evidence. For instance, when presented with an image of a green banana, a model may still respond "yellow."

Existing LP analysis methods primarily rely on input-output probing, with two key limitations: (1) they ignore the rich internal hidden representations of the model; and (2) they cannot reveal at which layer LP begins to interfere with visual integration. This paper proposes analyzing LP from the perspective of **internal representation dynamics**, localizing the critical layer at which visual information begins to genuinely influence reasoning by contrasting chain-of-embeddings.

## Method

### Overall Architecture

Given input $(x_v, x_t)$, two types of chain-of-embedding are extracted:
- $Z_{\text{vis}}^l = f_l(X_v, X_t)$: last-token embedding at each layer with visual input
- $Z_{\text{blind}}^l = f_l(\varnothing, X_t)$: last-token embedding at each layer without visual input

At each layer $l$, the distance $\mathbf{D}_l$ between the two is computed, then aggregated separately over a vision-dependent dataset $\mathcal{D}_{VT}$ and a vision-independent dataset $\mathcal{D}_T$.

### Key Designs

1. **Visual Integration Point (VIP)**: A critical layer $l^*$ exists beyond which the representation distances for $\mathcal{D}_{VT}$ and $\mathcal{D}_T$ diverge significantly. Prior to VIP, the model performs general information processing; after VIP, it begins to leverage visual information for task-specific reasoning. Formally: $\mathbf{D}_l(\mathcal{P}_{VT}) - \mathbf{D}_l(\mathcal{P}_T) > \tau, \forall l \geq l^*$. A key finding is that the VIP location is consistent across datasets (an intrinsic model property) but varies across models.

2. **Total Visual Integration (TVI)**: Aggregates representation distances across all layers after the VIP to quantify the total amount of visual integration, defined as $\text{TVI}(l^*; x, F_\theta) = \frac{1}{L - l^* + 1} \sum_{l=l^*}^{L} d(z_{\text{vis}}^l, z_{\text{blind}}^l)$. Higher TVI indicates fuller utilization of visual information and weaker LP; lower TVI indicates text-dominated inference and stronger LP.

3. **Data Partitioning Strategy**: Since existing datasets do not annotate visual dependence, a prediction consistency proxy is adopted: samples are assigned to $\mathcal{D}_{VT}$ if $F_\theta(x_v, x_t) \neq F_\theta(\varnothing, x_t)$, and to $\mathcal{D}_T$ otherwise. Cosine distance is used as the default metric.

### Loss & Training

TVI can also serve as a training regularizer to improve LVLM performance. It is incorporated into the LLaVA instruction-tuning objective as:

$$\mathcal{L}(x, y; \theta) = -\log F_\theta(y|x) - \lambda \cdot \text{TVI}(l^*; x, F_\theta)$$

where $\lambda = 0.03$, encouraging the model to more strongly integrate visual information.

## Key Experimental Results

### Main Results

| Model × Dataset | Spearman Correlation (TVI vs. Accuracy) | p-value |
|---|---|---|
| Qwen2.5-VL-7B (post-VIP) | 0.7241 | <0.001 |
| Gemma3-4B (post-VIP) | 0.7174 | <0.001 |
| Qwen2.5-VL-7B (pre-VIP) | 0.1489 | 0.002 |
| Gemma3-4B (pre-VIP) | 0.4659 | <0.001 |

| Metric | Qwen2.5-VL-7B VLind | Qwen2.5-VL-7B ViLP | InternVL-3-8B VLind | InternVL-3-8B ViLP |
|---|---|---|---|---|
| TVI | **0.7155** | **0.6335** | **0.6727** | **0.5709** |
| Visual Attention | 0.0871 | -0.0364 | 0.4967 | 0.0746 |
| Output Divergence | 0.2978 | 0.5084 | 0.1627 | 0.5615 |

### Ablation Study

| Configuration | VLind Corr. | ViLP Corr. | Note |
|---|---|---|---|
| Cosine Distance | 0.7155 | 0.6335 | Default; best performance |
| L2 Distance | 0.7123 | 0.6578 | Comparable; still effective |
| KL Divergence (logit-lens) | -0.1693 | 0.2901 | Fails after projection to output space |
| JS Divergence (logit-lens) | -0.2261 | 0.2942 | Same as above |

| TVI Regularization | Perception | Reasoning |
|---|---|---|
| LLaVA-v1.5 | 1369.75 | 298.21 |
| LLaVA-v1.5 w/ TVI | **1400.44** | **321.43** |

### Key Findings

- VIP consistently emerges across all 60 combinations of 10 LVLMs and 6 datasets
- VIP typically appears at approximately 60% of model depth, regardless of model scale
- Larger models (Gemma-3-27B) exhibit higher normalized TVI, indicating stronger visual information utilization
- Datasets with strong LP (ViLP) exhibit significantly lower TVI than those with weak LP (MMBench)
- Intervention experiment: after applying PAI attention correction, TVI increases from 0.038 to 0.144, and accuracy improves from 50% to 52.33%

## Highlights & Insights

- This is the first systematic analysis of language priors in LVLMs from the perspective of internal representation dynamics, providing finer granularity than input-output probing
- The discovery of VIP as an intrinsic model property is significant, suggesting that visual integration has a fixed "onset" within the model architecture
- TVI consistently outperforms visual attention and output divergence as proxy metrics across all models and datasets
- Theoretical analysis connects representation divergence to KL divergence, providing an information-theoretic interpretation

## Limitations & Future Work

- Requires white-box access to model internal states; inapplicable to closed-source APIs
- VIP selection depends on a manually defined threshold $\tau$ (though an automatic selection method is provided in the appendix)
- Only language priors are analyzed; other biases such as distribution shift are not considered
- TVI regularization experiments are conducted only on a 60K subset; large-scale validation remains to be completed

## Related Work & Insights

- Related to mechanistic interpretability, but focused on multimodal integration rather than unimodal processing
- Can inspire layer-wise intervention strategies based on TVI, such as applying stronger visual constraints to layers after the VIP
- Has direct implications for LVLM hallucination mitigation: samples with low TVI may require additional visual attention correction

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — A novel perspective on analyzing language priors via internal representations; VIP and TVI are highly original contributions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 10 models × 6 datasets = 60 configurations; comprehensive ablations with intervention validation and theoretical analysis
- Writing Quality: ⭐⭐⭐⭐ — Clear exposition, rigorous derivations, and informative figures
- Value: ⭐⭐⭐⭐ — Provides practical analytical tools for understanding and improving LVLMs; TVI regularization demonstrates real application potential

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bridging Human and LLM Judgments: Understanding and Narrowing the Gap](../../NeurIPS2025/dialogue/bridging_human_and_llm_judgments_understanding_and_narrowing_the_gap.md)
- [\[AAAI 2026\] Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for Language Generation Evaluation](../../AAAI2026/dialogue/auto-pre_an_automatic_and_cost-efficient_peer-review_framework_for_language_gene.md)
- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](non-collaborative_user_simulators_for_tool_agents.md)
- [\[ICLR 2026\] AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions](aqua_toward_strategic_response_generation_for_ambiguous_visual_questions.md)
- [\[ICLR 2026\] Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation](think-while-generating_on-the-fly_reasoning_for_personalized_long-form_generatio.md)

</div>

<!-- RELATED:END -->
