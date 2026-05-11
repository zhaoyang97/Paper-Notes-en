---
title: >-
  [Paper Note] KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
description: >-
  [ICLR 2026][LLM/NLP][LLM communication] This paper proposes KVComm, a framework that enables efficient inter-LLM communication via selective KV pair sharing. It identifies an "information concentration bias" in hidden st…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "LLM communication"
  - "KV cache sharing"
  - "multi-agent LLM"
  - "selective layer"
  - "attention importance"
date: 2026-05-08
content_hash: eb997a1c21b308e1
---

# KVComm: Enabling Efficient LLM Communication through Selective KV Sharing

**Conference**: ICLR 2026
**arXiv**: [2510.03346](https://arxiv.org/abs/2510.03346)
**Code**: To be confirmed
**Area**: Agent / LLM Efficiency
**Keywords**: LLM communication, KV cache sharing, multi-agent LLM, selective layer, attention importance

## TL;DR
This paper proposes KVComm, a framework that enables efficient inter-LLM communication via selective KV pair sharing. It identifies an "information concentration bias" in hidden states that renders them unsuitable for cross-model transfer, and designs a layer selection strategy combining attention importance scores with a Gaussian prior. Transmitting only 30% of layers suffices to outperform most baselines.

## Background & Motivation
**Background**: Multi-LLM collaboration requires efficient communication mechanisms; existing approaches transmit hidden states or the full KV cache.

**Limitations of Prior Work**: ① The last-token hidden state is most critical in deeper layers, yet transmitting it overwrites the receiver's own representations. ② Full KV cache transmission incurs prohibitive bandwidth costs.

**Key Challenge**: The fundamental tension between communication efficiency and information completeness.

**Goal**: Identify the most suitable representation format and selection strategy for cross-LLM transfer.

**Key Insight**: A systematic comparison of hidden states and KV pairs reveals that KV pairs are inherently more suitable—they can be selectively transmitted on a per-layer basis without overwriting the receiver's information.

**Core Idea**: KV pairs constitute the optimal communication medium; selecting intermediate layers (richest in semantics) combined with high-attention layers yields the optimal subset.

## Method

### Overall Architecture
The Sender processes context → extracts KV pairs → applies a layer selection strategy to obtain a subset → transmits the subset to the Receiver. The Receiver concatenates both parties' KVs at the corresponding layers: $\mathbf{k}_r^l \leftarrow [\mathbf{k}_s^{l_i}; \mathbf{k}_r^l]$.

### Key Designs

1. **Hidden States vs. KV Pairs**:

   - Hidden states drawback: the last-token representation is most informative, yet direct transmission replaces the receiver's own representations.
   - KV pairs advantage: can be injected at arbitrary layers without overwriting receiver information; the attention mechanism naturally determines how much to attend to the shared content.

2. **Layer Selection Strategy**:

   - Attention importance score: $\hat{S}_a^l = \frac{1}{H|Q|}\sum_h\sum_q\sum_c a_{h,q,c}^l$
   - Gaussian prior: $P^l = \exp(-\frac{(l-\mu)^2}{2\sigma^2})$ (encourages selection of intermediate layers)
   - Final score: $S^l = \alpha S_a^l + (1-\alpha) P^l$; top-$M$ layers are selected
   - Only 1 calibration sample is required for robust layer selection

3. **Two Hypotheses Validated**:

   - H1: KVs from intermediate layers carry the most transferable semantic knowledge.
   - H2: Layers with more concentrated attention are more informative.

## Key Experimental Results

### Main Results (9 model pairs × 8 datasets)

| Model | Method | Countries | HotpotQA | MultiFieldQA |
|-------|--------|-----------|----------|-------------|
| Llama-3.2-3B | Skyline | 0.57 | 0.73 | 0.47 |
| Llama-3.2-3B | KVComm(0.5) | **0.57** | 0.57 | **0.51** |
| Llama-3.2-3B | NLD | 0.51 | 0.47 | 0.38 |
| Llama-3.2-3B | AC | 0.35 | 0.32 | 0.29 |

### Ablation Study

| Transmission Ratio | Performance |
|-------------------|-------------|
| 30% of layers | Outperforms NLD / CIPHER / AC baselines |
| 50% of layers | Approaches Skyline |
| 70% of layers | Matches or exceeds Skyline |
| Non-contiguous vs. contiguous selection | Non-contiguous selection substantially superior |

### Key Findings
- Transmitting KVs from only 30% of layers already surpasses most baselines—selectivity outperforms completeness.
- On MultiFieldQA, KVComm exceeds Skyline (0.51 vs. 0.47), suggesting selective sharing has a regularization effect.
- The AC method approaches the no-communication baseline on most datasets.
- Computational cost is reduced by 2.5×–6× compared to NLD.

## Highlights & Insights
- The **information concentration bias in hidden states** is an important finding that serves as a cautionary note for all hidden-state-based LLM communication methods.
- **"Less is more"**: KVs from 30–50% of layers outperform full hidden-state transmission.
- The **Gaussian prior for intermediate-layer selection** is simple yet effective.
- Layer selection requires only 1 calibration sample, making deployment extremely lightweight.

## Limitations & Future Work
- Only supports communication between models sharing the same base model; heterogeneous architectures are not supported.
- Layer indices must correspond one-to-one, limiting communication between models of different scales.
- The hyperparameters $\mu$ and $\sigma$ of the Gaussian prior require tuning.
- Evaluation is limited to two-agent scenarios.
- Gains on mathematical reasoning tasks are marginal.

## Related Work & Insights
- **vs. NLD**: NLD compresses knowledge into natural language, incurring substantial information loss; KVComm directly transmits internal representations.
- **vs. CIPHER**: CIPHER transmits hidden states and is therefore susceptible to information concentration bias.
- **vs. DroidSpeak**: DroidSpeak selects contiguous layer chunks, which is less flexible than non-contiguous selection.
- This work may inspire multi-agent LLM system design, where KV cache sharing could become a standard communication primitive.

## Supplementary Technical Details

### Why Are KV Pairs More Suitable for Communication Than Hidden States?
Hidden states at each layer constitute a complete representation; directly transmitting them overwrites the receiver's corresponding layer representation. KV pairs, by contrast, serve as inputs to the attention mechanism: concatenating the sender's KV pairs with the receiver's does not destroy existing information but instead allows the attention mechanism to naturally determine which information to attend to. This **additive rather than replacement** property is the core advantage of KV-based communication.

### Why Are Intermediate Layers Most Valuable?
Research indicates that LLM layers can be broadly partitioned into three functional zones: lower layers (low-level features, syntax), middle layers (semantic knowledge, world knowledge), and upper layers (task-specific representations, next-token prediction). Semantic knowledge in the middle layers is the most general and transferable, whereas lower layers are too low-level and upper layers are too task-specific for effective cross-model transfer.

### Relationship to Prompt Compression
KVComm can be viewed as **prompt compression in KV space**—compressing not the text but the layer dimension of internal representations. This preserves finer-grained information than NLD, which compresses knowledge into natural language. Future work could explore intra-layer compression (e.g., selective token transmission), enabling dual-dimension compression along both the layer and token axes.

### Attention Mechanism Under KV Concatenation
After the sender's KV pairs are concatenated with the receiver's, the receiver's queries can freely attend to keys from both parties. Since attention weights are softmax-normalized, irrelevant information is naturally down-weighted. This is more **gentle** than directly replacing hidden states—no information is forcibly overwritten.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic comparison of communication media; well-motivated layer selection strategy
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 model pairs × 8 datasets
- Writing Quality: ⭐⭐⭐⭐ Clear hypothesis-validation structure
- Value: ⭐⭐⭐⭐ Practical guidance for multi-LLM collaborative systems

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](../../AAAI2026/llm_nlp/stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)
- [\[ICLR 2026\] The Path of Least Resistance: Guiding LLM Reasoning Trajectories for Efficient Consistency](the_path_of_least_resistance_guiding_llm_reasoning_trajectories_for_efficient_co.md)
- [\[AAAI 2026\] Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](../../AAAI2026/llm_nlp/position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)
- [\[ACL 2026\] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](../../ACL2026/llm_nlp/grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)
- [\[ICLR 2026\] How Catastrophic is Your LLM? Certifying Risk in Conversation](how_catastrophic_is_your_llm_certifying_risk_in_conversation.md)

</div>

<!-- RELATED:END -->
