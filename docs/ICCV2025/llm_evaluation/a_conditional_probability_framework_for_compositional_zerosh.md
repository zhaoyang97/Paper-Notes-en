---
title: >-
  [Paper Note] A Conditional Probability Framework for Compositional Zero-shot Learning
description: >-
  [ICCV 2025][LLM Evaluation][conditional probability decomposition] This paper proposes a Conditional Probability Framework (CPF) that decomposes the compositional recognition probability into an object likelihood $p(o|x)$ and a conditional attribute likelihood $p(a|o,x)$. Two dedicated modules — Text-Enhanced Object learning (TEO) and Object-Guided Attribute learning (OGA) — explicitly model attribute-object dependencies, achieving state-of-the-art performance across three CZSL benchmarks.
tags:
  - ICCV 2025
  - LLM Evaluation
  - conditional probability decomposition
  - compositional zero-shot learning
  - attribute-object dependency modeling
  - cross-attention mechanism
  - text-enhanced features
date: 2026-05-08
content_hash: 383af3c5960aa979
---

# A Conditional Probability Framework for Compositional Zero-shot Learning

**Conference**: ICCV 2025
**arXiv**: [2507.17377](https://arxiv.org/abs/2507.17377)
**Code**: [GitHub](https://github.com/Pieux0/CPF) (publicly available)
**Authors**: Peng Wu, Qiuxia Lai, Hao Fang, Guo-Sen Xie, Yilong Yin, Xiankai Lu, Wenguan Wang (Shandong University, Communication University of China, Nanjing University of Science and Technology, Zhejiang University, Xi'an Jiaotong University)
**Area**: Zero-Shot Learning / Compositional Zero-Shot Learning
**Keywords**: conditional probability decomposition, compositional zero-shot learning, attribute-object dependency modeling, cross-attention mechanism, text-enhanced features

## TL;DR

This paper proposes a Conditional Probability Framework (CPF) that decomposes the compositional recognition probability into an object likelihood $p(o|x)$ and a conditional attribute likelihood $p(a|o,x)$. Two dedicated modules — Text-Enhanced Object learning (TEO) and Object-Guided Attribute learning (OGA) — explicitly model attribute-object dependencies, achieving state-of-the-art performance across three CZSL benchmarks.

## Background & Motivation

Compositional Zero-Shot Learning (CZSL) aims to recognize unseen combinations of known attributes and objects by leveraging knowledge from seen compositions. Prior methods predominantly focus on disentangling attributes and objects, treating them as independent entities learned separately.

**Limitations of Prior Work**:
1. **Neglect of semantic constraints**: Certain attributes are naturally paired with specific objects (e.g., "striped" applies to "zebra"/"shirts" but not "sky"/"water"); independent modeling fails to capture such constraints.
2. **Neglect of contextual dependencies**: The visual manifestation of the same attribute varies greatly across different objects (e.g., "young" in "young tree" vs. "young dog" carries different visual meanings).
3. **Ambiguity from disentanglement**: An image may contain multiple attributes (e.g., "blue," "striped," "green"); without object context, fully disentangled models cannot select the correct attribute, leading to erroneous predictions for unseen compositions.

**Existing Attempts at Context Modeling**:
- CoT: generates element-wise attention maps from object features to refine attribute features.
- CANet: conditions attribute embedding on the recognized object and input image, dynamically adjusting attribute learner parameters.
- However, these methods still fail to effectively model the semantic constraints between attributes and objects.

## Core Problem

How can CZSL effectively capture the mutual dependencies between attributes and objects — including both semantic constraints and contextual dependencies — rather than treating them as independent entities?

From a probabilistic perspective, given image $x$, the likelihood of composition $c=(o,a)$ can be decomposed as:

$$p(o,a|x) = p(o|x) \cdot p(a|o,x)$$

where $p(o|x)$ is the object likelihood and $p(a|o,x)$ is the attribute likelihood conditioned on both the object and the image. Jointly optimizing these two likelihoods enables more effective compositional learning.

## Method

### Overall Architecture

CPF consists of a visual backbone and two core modules:
1. **Text-Enhanced Object learning (TEO)**: Fuses deep visual embeddings with text embeddings to address semantic constraints, producing enhanced object representations.
2. **Object-Guided Attribute learning (OGA)**: Uses the enhanced object features and shallow visual embeddings to capture attribute-object interdependencies via cross-attention.

The visual backbone is a fine-tuned ViT-B (or CLIP's ViT-L/14), extracting features at two levels:
- **Deep features**: Output of the last block, used for object learning.
- **Shallow features**: Fused outputs of blocks 3/6/9 (or 6/12/18 for CLIP), used for attribute learning.

### Key Designs

#### 1. Text-Enhanced Object Learning (TEO)

A text descriptor embedding $q_t$ is first constructed by projecting the deep visual class token into the text space and performing attention fusion with object text embeddings:

$$q_t = \text{softmax}\left(\frac{f^o_{v \to t}(v^c_h)(W^o)^\top}{\sqrt{d}}\right) W^o$$

$q_t$ then attends to deep patch tokens to highlight semantically relevant image regions, yielding the text-enhanced object feature:

$$v_o = v^c_h + \text{softmax}\left(\frac{q_t \cdot f^o_{v \to t}(V^p_h)^\top}{\sqrt{d}}\right) V^p_h$$

**Design Motivation**: By directing attention toward image regions semantically relevant to the object via text descriptors, TEO explicitly models semantic constraints.

#### 2. Object-Guided Attribute Learning (OGA)

The enhanced object embedding $v_o$ serves as a query attending to shallow patch embeddings via cross-attention to produce the attribute representation:

$$v_a = \text{softmax}\left(\frac{v_o (V^p_l)^\top}{\sqrt{D}}\right) V^p_l$$

**Design Motivation**: Object features guide the model to attend to attribute-relevant spatial regions, thereby modeling contextual dependencies. For instance, given object information "tree," the model focuses on visual cues related to the tree's age when predicting the "young" attribute.

#### 3. Compositional Matching

The attribute and object visual features are concatenated and projected to form the compositional visual feature, which is then aligned with the compositional text feature.

### Loss & Training

Three cross-entropy losses are jointly optimized:

$$\mathcal{L} = \mathcal{L}_{com} + \alpha_1 \mathcal{L}_{att} + \alpha_2 \mathcal{L}_{obj}$$

- $\mathcal{L}_{obj}$: object classification loss based on the text-enhanced object feature $v_o$.
- $\mathcal{L}_{att}$: conditional attribute classification loss based on the object-guided attribute feature $v_a$.
- $\mathcal{L}_{com}$: compositional classification loss based on the concatenated feature $v_c$.
- Temperature $\tau = 0.05$; loss weights $\alpha_1 = 0.6$, $\alpha_2 = 0.4$.

**Inference Strategy**: Additive score fusion is adopted instead of multiplication (to avoid probability vanishing):

$$\hat{c} = \arg\max_{c_{i,j} \in \mathcal{C}_{test}} p(c_{i,j}|x) + p(a_i|x, v_o) + p(o_j|x)$$

**Training Details**: Adam optimizer, 10 epochs; ViT-B learning rate $1 \times 10^{-4}$; CLIP learning rate $3.15 \times 10^{-6}$; text embeddings from GloVe (or CLIP text encoder); inference speed 1457 fps (ViT-B), comparable to existing methods.

## Key Experimental Results

### Closed-World Setting (ViT-B Backbone)

| Dataset | Metric | Ours (CPF) | Prev. SOTA | Gain |
|---------|--------|-----------|------------|------|
| MIT-States | AUC | 11.2 | 10.5 (CoT) | +6.7% |
| MIT-States | HM | 26.8 | 25.8 (CoT) | +3.9% |
| MIT-States | Seen Acc | 41.3 | 39.5 (CoT) | +4.6% |
| MIT-States | Unseen Acc | 34.8 | 33.0 (CoT) | +5.5% |
| UT-Zappos50K | AUC | 41.4 | 35.1 (ADE) | +17.9% |
| UT-Zappos50K | HM | 55.7 | 51.1 (ADE) | +9.0% |
| C-GQA | AUC | 8.2 | 7.4 (CoT) | +10.8% |
| C-GQA | HM | 23.9 | 22.1 (CoT) | +8.1% |

### Open-World Setting (ViT-B Backbone)

| Dataset | Metric | Ours (CPF) | Prev. SOTA | Gain |
|---------|--------|-----------|------------|------|
| MIT-States | AUC | 4.4 | 1.6 (CompCos) | +175% |
| MIT-States | HM | 15.1 | 8.9 (CompCos) | +69.7% |
| UT-Zappos50K | AUC | 31.2 | 28.8 (DRANet) | +8.3% |
| UT-Zappos50K | HM | 47.6 | 44.8 (ADE) | +6.3% |
| C-GQA | AUC | 2.10 | 1.42 (ADE) | +47.9% |
| C-GQA | HM | 9.5 | 7.6 (ADE) | +25.0% |

### CLIP Backbone (C-GQA)

| Setting | Metric | CPF | Prev. SOTA (LOGICZSL) | Gain |
|---------|--------|-----|-----------------------|------|
| CW | AUC | 15.4 | 15.3 | +0.7% |
| CW | HM | 33.6 | 33.3 | +0.9% |
| OW | AUC | 3.6 | 3.4 | +5.9% |
| OW | HM | 13.0 | 12.6 | +3.2% |

### Ablation Study

1. **Core component analysis** (C-GQA, CW):
    - Full CPF: AUC=8.2, HM=23.9
    - w/o TEO: AUC=7.6 (−7.3%), HM=22.7 → validates the effectiveness of text descriptors for object learning.
    - w/o TEO+OGA: AUC=6.9 (−15.9%), HM=21.4 → validates the critical role of attribute-object dependency modeling.

2. **Attention vs. average pooling**: Replacing cross-attention in Eq. 2 or Eq. 4 with simple average pooling reduces AUC to 7.8 and 7.1 respectively, confirming the effectiveness of the cross-attention mechanism.

3. **Visual embedding selection**: Using only deep features yields AUC=6.7; adding shallow features raises AUC to 8.2 (+22.4%), demonstrating that fine-grained shallow information is critical for attribute learning.

4. **Guidance signal for attribute learning**: Object visual embedding (AUC=8.2) > object text embedding (7.7) > attribute text embedding (7.6), as visual embeddings are more strongly aligned with attributes.

## Highlights & Insights

1. **Novel probabilistic decomposition perspective**: Reformulating compositional recognition as a conditional probability decomposition problem — $p(o,a|x) = p(o|x) \cdot p(a|o,x)$ — provides a theoretically grounded framework for CZSL.
2. **Simple yet effective design**: The core of the method comprises two attention modules (TEO + OGA) with no additional trainable token-level parameters, maintaining inference speed comparable to existing methods.
3. **Substantial performance gains**: Particularly in the open-world setting, CPF achieves +175% AUC on MIT-States and +47.9% AUC on C-GQA.
4. **Strong scalability**: The framework integrates seamlessly with vision-language models such as CLIP and achieves state-of-the-art results on the most challenging C-GQA benchmark.
5. **Complementary use of shallow and deep features**: Deep features capture global object semantics, while shallow features preserve fine-grained attribute information — a well-motivated hierarchical design.

## Limitations & Future Work

1. **Semantic ambiguity**: The model remains susceptible to confusion between semantically similar class labels (e.g., "highway" vs. "road," "thick" vs. "folded").
2. **Visual confusion**: The model struggles to distinguish visually similar targets (e.g., "thawed meat" vs. "frozen fish").
3. **Proposed improvement direction**: Leveraging large language models to generate more discriminative textual descriptions to differentiate semantically similar categories.
4. **Diminishing returns with CLIP backbone**: Improvements under the CLIP setting on C-GQA are modest (CW AUC +0.7%), suggesting limited marginal gains over already strong baselines.
5. **No external knowledge**: Unlike KG-SP and DRANet, which use external knowledge graphs to prune the candidate space, CPF relies entirely on self-learned representations, leaving room for further improvement in the open-world setting.

## Related Work & Insights

| Method Type | Representative Work | Mechanism | CPF's Advantage |
|-------------|---------------------|-----------|-----------------|
| Composition as label | AoP, SymNet, CGE | Treats attribute-object combinations as single labels | CPF avoids exponential combinatorial space via decomposition |
| Disentangled learning | OADis, SCEN | Separate modules for attributes and objects | CPF explicitly models attribute-object dependencies |
| Context modeling | CoT, CANet | Modulates attribute learning with object features | CPF jointly models semantic constraints and contextual dependencies |
| VLM-based methods | CSP, DFSP, Troika | Exploits CLIP's zero-shot capacity | CPF integrates seamlessly with CLIP and yields further gains |

**Broader implications**:
1. **Generality of conditional probability decomposition**: The idea of decomposing a complex joint distribution into marginal and conditional distributions is transferable to other compositional recognition tasks (e.g., "action + object" in action recognition, "relation + entity" in scene understanding).
2. **Complementarity of shallow and deep features**: The finding that shallow features preserve fine-grained information while deep features capture global semantics offers useful insights for multi-task learning and multi-granularity representation design.
3. **Text-guided visual attention**: Using text embeddings to steer visual feature focus is transferable to tasks requiring text-vision interaction, such as VQA and image retrieval.
4. **Additive vs. multiplicative score fusion at inference**: Using additive fusion of probability scores to avoid the vanishing probability problem of multiplication is a practical engineering insight worth adopting in similar multi-branch prediction systems.

## Rating

| Dimension | Score (1–5) | Remarks |
|-----------|------------|---------|
| Novelty | 4 | The conditional probability decomposition perspective is innovative, though the attention mechanism design is relatively standard. |
| Technical Depth | 3.5 | The method is concise and effective, but the mathematical derivations are not deeply elaborated. |
| Experimental Thoroughness | 5 | Three datasets, two settings, extensive ablations, and CLIP extension — highly comprehensive. |
| Writing Quality | 4 | Clear logic, rich visualizations, and failure case analysis. |
| Value | 4 | The framework is general and extensible, code is publicly available, and inference is fast. |
| **Overall** | **4.0** | A strong work combining a probabilistic decomposition perspective, clean design, and thorough experimentation. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Few-Shot Pattern Detection via Template Matching and Regression](few-shot_pattern_detection_via_template_matching_and_regression.md)
- [\[ICCV 2025\] Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting](rethinking_few_shot_clip_benchmarks_a_critical_analysis_in_the_inductive_setting.md)
- [\[ICCV 2025\] InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild](intersyn_interleaved_learning_for_dynamic_motion_synthesis_in_the_wild.md)
- [\[ICCV 2025\] Imbalance in Balance: Online Concept Balancing in Generation Models](imbalance_in_balance_online_concept_balancing_in_generation_models.md)
- [\[ICCV 2025\] Combinative Matching for Geometric Shape Assembly](combinative_matching_for_geometric_shape_assembly.md)

</div>

<!-- RELATED:END -->
