---
title: >-
  [Paper Note] To See or To Read: User Behavior Reasoning in Multimodal LLMs
description: >-
  [NeurIPS 2025][Multimodal VLM][User behavior reasoning] This paper proposes BehaviorLens, a benchmarking framework that systematically compares three representations of user behavior history — text sequences…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "User behavior reasoning"
  - "modality trade-off"
  - "sequential recommendation"
  - "visual representation"
  - "BehaviorLens"
date: 2026-05-08
content_hash: 30f00d61cfac44da
---

# To See or To Read: User Behavior Reasoning in Multimodal LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2511.03845](https://arxiv.org/abs/2511.03845)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: User behavior reasoning, modality trade-off, sequential recommendation, visual representation, BehaviorLens

## TL;DR
This paper proposes BehaviorLens, a benchmarking framework that systematically compares three representations of user behavior history — text sequences, scatter plots, and flowcharts — for next-purchase prediction with MLLMs. Visual representations are shown to improve prediction accuracy by up to 87.5% over equivalent text representations without incurring additional computational overhead.

## Background & Motivation

Multimodal large language models (MLLMs) are reshaping how intelligent systems reason over sequential user behavior data. However, a critical and underexplored question remains: how should sequential user history be represented to simultaneously optimize reasoning accuracy and computational efficiency?

Conventional approaches flatten user interaction histories (clicks, browsing, purchases) into line-by-line textual descriptions fed into MLLMs. While this preserves fine-grained information, it discards structural patterns of the user journey (e.g., topological patterns), potentially preventing models from grasping the overall user narrative and degrading intent prediction quality.

The core problem is: which representation — text or image — more effectively maximizes MLLM performance on user behavior reasoning? This is particularly critical in agentic recommendation systems, where understanding user journeys and conversions directly impacts personalization and revenue.

The paper's starting point is to conduct systematic benchmarking under controlled conditions, directly comparing text and visual representations of the same user history with respect to prediction accuracy and computational cost.

## Method

### Overall Architecture

The BehaviorLens framework converts user purchase histories into three representation forms and feeds them into MLLMs: (1) **text sequences** — natural language descriptions of each purchase event; (2) **scatter plots** — mapping time and items onto a two-dimensional coordinate system; and (3) **flowcharts** — representing temporal adjacency relationships in a purchase sequence via nodes and directed edges. MLLMs then perform next-purchase prediction and generate reasoning explanations based on each representation.

### Key Designs

1. **Text Sequential Representation**: Each interaction triple $(a, i, e)$ (action, item, environment/timestamp) is rendered as a natural language description $\phi_{text}(a,i,e) = \text{"item \{i\} was \{a\} at timestamp \{e\}"}$, and the purchase history is the concatenation of all event descriptions. This serves as the standard baseline in existing user modeling.

2. **Scatter Plot Representation**: Inspired by visualization methods in time-series modeling, the purchase history is converted into a scatter plot image. A ranking function $r(\cdot)$ maps items and timestamps to coordinates: $\phi_{scatter-plot}(a,i,e) = \text{plot}(a,i|x=r(e), y=r(i))$. This representation intuitively captures temporal purchase patterns and item distributions.

3. **Flowchart Representation**: Motivated by prior work showing that visual compression aids LLM reasoning on structured tasks, the purchase history is converted into a directed flowchart. Each purchase node is connected to its temporal predecessor and successor: $\phi_{flowchart}(a,i,e) = \text{node}(a,i,e|p=\{node(a_{-1},i_{-1},e_{-1})\}, s=\{node(a_{+1},i_{+1},e_{+1})\})$. This representation emphasizes sequential structure and temporal adjacency.

### Loss & Training

This paper involves no model training and constitutes a pure evaluation study. Six MLLMs (Gemini-2.0-flash-lite, Gemini-2.0-flash, Gemini-2.5-flash-lite, Gemini-2.5-flash, GPT-4o, GPT-4.1-mini) are evaluated on real purchase sequence datasets, and reasoning explanation quality is assessed via LLM-as-a-Judge.

## Key Experimental Results

### Main Results

| MLLM | Input Type | Accuracy | Similarity | Tokens | Latency (s) |
|------|-----------|----------|------------|--------|-------------|
| Gemini-2.5-flash-lite | Text | 0.360 | 0.570 | 1220 | 1.444 |
| Gemini-2.5-flash-lite | Scatter Plot | **0.530** | **0.689** | 3623 | 2.057 |
| GPT-4o | Text | 0.420 | 0.602 | 1106 | 5.451 |
| GPT-4o | Scatter Plot | **0.560** | **0.713** | 1169 | 8.954 |
| GPT-4o | Flowchart | 0.300 | 0.527 | 1043 | 7.140 |
| GPT-4.1-mini | Text | 0.320 | 0.542 | 1105 | 4.680 |
| GPT-4.1-mini | Scatter Plot | **0.600** | **0.726** | 1039 | 7.849 |
| GPT-4.1-mini | Flowchart | 0.340 | 0.563 | 862 | 6.051 |

### Ablation Study (Reasoning Explanation Quality)

| Evaluation Dimension | Description | Difference |
|----------------------|-------------|------------|
| Faithfulness | Whether reasoning is faithful to the input | No significant difference across input types |
| Overthinking | Whether the model over-reasons | No significant difference across input types |
| Causality | Causal logic of the reasoning | No significant difference across input types |
| Sufficiency | Whether reasoning is sufficiently complete | No significant difference across input types |
| Specificity | Whether reasoning is user-specific | No significant difference across input types |
| Plausibility | Whether reasoning is plausible | No significant difference across input types |

### Key Findings

- Across 6 MLLMs, scatter plot or flowchart representations outperform text representations in all models except Gemini-2.5-flash.
- The largest gain is observed with GPT-4.1-mini: scatter plot vs. text yields an 87.5% accuracy improvement (0.320→0.600) and a 33.9% similarity improvement.
- GPT models consume roughly the same number of tokens across representation types, indicating that visual representations incur no additional computational cost.
- Among Gemini models, scatter plots consume approximately 3× the tokens, though flowcharts and text remain comparable.
- Reasoning explanation quality shows no significant difference across input types (except Gemini-2.0-flash), suggesting that **it is the input representation itself, not intermediate reasoning quality, that drives prediction improvements**.
- Case analyses reveal that text inputs tend to focus on purchase frequency, flowcharts emphasize complementarity of recent purchases, and scatter plots more effectively capture cyclical purchasing patterns.

## Highlights & Insights

- The core finding is practically valuable and counterintuitive: visual representations are better suited than text for encoding sequential user behavior, challenging the default assumption that text is the optimal input modality for MLLMs.
- The framework is elegantly simple — no training or fine-tuning is required; substantial gains are achieved purely by changing the input representation.
- Importing time-series visualization techniques into user behavior modeling represents a compelling cross-domain methodological transfer.
- The observation that reasoning quality remains constant while prediction accuracy improves substantially suggests fundamental differences in how MLLMs extract information from different modalities.

## Limitations & Future Work

- Validation is limited to the next-purchase prediction task; other user behavior reasoning tasks (e.g., intent recognition, churn prediction) remain unexplored.
- The dataset scale and diversity are limited, being based on a single e-commerce scenario.
- The effect of longer user behavior sequences on results is not investigated.
- The design space for visual representations is far from exhausted — alternative visualizations such as heatmaps and timeline charts are not tested.
- The anomalous behavior of Gemini-2.5-flash, where image representations perform worse, is not deeply analyzed.
- The effect of multimodal combined inputs (providing both text and image simultaneously) is not considered.

## Related Work & Insights

- Unlike sequential recommendation methods (GRU4Rec, SASRec, BERT4Rec, etc.), this paper frames the recommendation problem as an MLLM reasoning task rather than end-to-end training.
- The finding by Li & Jiang (2025) that visual compression aids LLM reasoning on structured tasks provides a theoretical basis for the flowchart representation.
- The time-series imaging approach of Wang & Oates (2015) inspires the scatter plot representation.
- Takeaway: when designing MLLM-based applications, text input should not be assumed as the default — selecting the appropriate modality representation based on task characteristics may yield substantial gains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (novel angle of attack; findings are counterintuitive and practically useful)
- **Experimental Thoroughness**: ⭐⭐⭐ (comparison across 6 models × 3 representations is thorough, but the dataset is limited to a single domain and is relatively small)
- **Writing Quality**: ⭐⭐⭐⭐ (clear structure; formal definitions are precise)
- **Value**: ⭐⭐⭐⭐ (directly actionable for the design of recommendation and agentic systems)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] READ: Enhancing Compositional Reasoning in CLIP via Reconstruction and Alignment of Text Descriptions](enhancing_compositional_reasoning_in_clip_via_reconstruction.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)
- [\[NeurIPS 2025\] Vision Function Layer in Multimodal LLMs](vision_function_layer_in_multimodal_llms.md)
- [\[NeurIPS 2025\] See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model](seetrek_training-free_spatial_prompting_for_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] Learning to Steer: Input-dependent Steering for Multimodal LLMs](learning_to_steer_input-dependent_steering_for_multimodal_llms.md)

</div>

<!-- RELATED:END -->
