---
title: >-
  [Paper Note] A Conditional Probability Framework for Compositional Zero-shot Learning
description: >-
  [ICCV 2025][LLM Evaluation][Compositional Zero-Shot Learning] This paper proposes CPF, a conditional probability framework for compositional zero-shot learning (CZSL) that decomposes the compositional likelihood into an object likelihood and a conditional attribute likelihood. Through a text-enhanced object learning module and an object-guided attribute learning module, CPF explicitly models the semantic constraints and contextual dependencies between attributes and objects, achieving a 17.9% AUC improvement on UT-Zappos50K and a 5.5% Unseen Accuracy improvement on MIT-States.
tags:
  - ICCV 2025
  - LLM Evaluation
  - Compositional Zero-Shot Learning
  - Conditional Probability
  - Attribute-Object Disentanglement
  - Cross-Attention
  - Vision-Language Models
date: 2026-05-08
content_hash: 3b9e539e6afa282e
---

# A Conditional Probability Framework for Compositional Zero-shot Learning

**Conference**: ICCV 2025
**arXiv**: [2507.17377](https://arxiv.org/abs/2507.17377)
**Code**: Available (mentioned in the paper as "Code is available at here")
**Area**: LLM Evaluation
**Keywords**: Compositional Zero-Shot Learning, Conditional Probability, Attribute-Object Disentanglement, Cross-Attention, Vision-Language Models

## TL;DR
This paper proposes CPF, a conditional probability framework for compositional zero-shot learning (CZSL) that decomposes the compositional likelihood into an object likelihood and a conditional attribute likelihood. Through a text-enhanced object learning module and an object-guided attribute learning module, CPF explicitly models the semantic constraints and contextual dependencies between attributes and objects, achieving a 17.9% AUC improvement on UT-Zappos50K and a 5.5% Unseen Accuracy improvement on MIT-States.

## Background & Motivation

Compositional Zero-Shot Learning (CZSL) aims to recognize novel attribute-object compositions (e.g., *blue shirt*) by leveraging knowledge from seen compositions (e.g., *striped shirt*, *blue sky*). It represents an important extension of zero-shot learning.

Most existing CZSL methods assume independence between attributes and objects, learning their representations separately via independent network modules (e.g., FC layers, self-attention/cross-attention disentanglers). However, this independent disentanglement overlooks two critical dependencies: (1) **semantic constraints**—certain attributes only co-occur with specific objects (e.g., *striped* suits *zebra* but not *sky*); and (2) **contextual dependencies**—the same attribute exhibits different visual appearances on different objects (e.g., *young tree* vs. *young dog*).

From a probabilistic perspective, the compositional likelihood can be factored as $p(o,a|\mathbf{x}) = p(o|\mathbf{x}) \cdot p(a|o,\mathbf{x})$. Prior methods have attempted to model contextual dependencies (e.g., CoT generates attribute attention maps from object features; CANet dynamically adjusts attribute learner parameters using object priors), but they still fail to effectively model semantic constraints.

Motivated by this conditional probability decomposition, CPF proposes to first learn object representations via a text-enhanced module, then guide attribute learning with the object representation, jointly modeling semantic constraints and contextual dependencies via a cross-attention mechanism.

## Method

### Overall Architecture
CPF consists of a visual backbone and two core modules: (1) **Text-Enhanced Object Learning (TEO)**, which leverages deep visual features and text embeddings to attend to semantically relevant regions for enhanced object representation; and (2) **Object-Guided Attribute Learning (OGA)**, which uses the enhanced object features to extract context-aware attribute representations from shallow visual features via cross-attention. Compositional matching then aligns visual and textual compositional features.

### Key Designs
1. **Text-Enhanced Object Learning (TEO)**:

    - **Function**: Integrates textual semantic information to improve the discriminability of object features.
    - **Mechanism**: A textual descriptor embedding $\mathbf{q}^t$ is constructed via attention-weighted aggregation over object text embeddings $\mathbf{W}^o$ using the deep class token $\mathbf{v}_h^c$:
      $$\mathbf{q}^t = \text{softmax}\!\left(\frac{f_{v\to t}^o(\mathbf{v}_h^c)(\mathbf{W}^o)^\top}{\sqrt{d}}\right)\mathbf{W}^o$$
      Then $\mathbf{q}^t$ serves as a query to extract semantically relevant patch features:
      $$\mathbf{v}^o = \mathbf{v}_h^c + \text{softmax}\!\left(\frac{\mathbf{q}^t f_{v\to t}^o(\mathbf{V}_h^p)^\top}{\sqrt{d}}\right)\mathbf{V}_h^p$$
    - **Design Motivation**: Text embeddings provide semantic priors for objects, helping the model focus on object-relevant image regions rather than being distracted by unrelated attributes.

2. **Object-Guided Attribute Learning (OGA)**:

    - **Function**: Learns attribute representations conditioned on the identified object context.
    - **Mechanism**: The enhanced object feature $\mathbf{v}^o$ serves as the query for cross-attention over shallow patch tokens $\mathbf{V}_l^p$:
      $$\mathbf{v}^a = \text{softmax}\!\left(\frac{\mathbf{v}^o(\mathbf{V}_l^p)^\top}{\sqrt{D}}\right)\mathbf{V}_l^p$$
      Object features guide attention toward the attribute regions most relevant to the given object context.
    - **Design Motivation**: The visual manifestation of an attribute depends on the object context (e.g., *young* appears entirely differently on a tree versus a dog). Using object features as queries naturally models this conditional dependency.

3. **Compositional Matching and Multi-Task Loss**:

    - **Function**: Combines attribute and object features and aligns them with textual compositional features.
    - **Mechanism**: The compositional visual feature $\mathbf{v}^c = f_c^v([\mathbf{v}^a, \mathbf{v}^o])$ is aligned with the textual feature $\mathbf{w}^c = f_c^t([\mathbf{w}^a, \mathbf{w}^o])$ via cross-entropy. During inference, additive fusion is used to avoid probability vanishing:
      $$\hat{c} = \arg\max_{c_{i,j}} p(c_{i,j}|\mathbf{x}) + p(a_i|\mathbf{x}, \mathbf{v}^o) + p(o_j|\mathbf{x})$$
    - **Design Motivation**: Additive rather than multiplicative fusion avoids numerical underflow in conditional probability computation.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{com} + \alpha_1 \mathcal{L}_{att} + \alpha_2 \mathcal{L}_{obj}$ ($\alpha_1=0.6$, $\alpha_2=0.4$)
- Three cross-entropy losses supervise compositional, attribute, and object predictions respectively.
- Learning rate: 1e-4 for ViT-B backbone, 3.15e-6 for CLIP backbone; Adam optimizer; 10 training epochs.
- Temperature parameter $\tau=0.05$.

## Key Experimental Results

### Main Results (Closed-World Setting)

| Dataset | Metric | CPF (ViT-B) | Prev. SOTA | Gain |
|--------|------|-------------|----------|------|
| UT-Zappos50K | AUC | 41.4 | 35.1 (ADE) | +17.9% |
| UT-Zappos50K | HM | 55.7 | 51.1 (ADE) | +9.0% |
| MIT-States | Unseen Acc | 34.8 | 33.0 (CoT) | +5.5% |
| MIT-States | AUC | 11.2 | 10.5 (CoT) | +6.7% |
| C-GQA | HM | 23.9 | 22.1 (CoT) | +8.1% |

### Ablation Study (C-GQA Dataset)

| Configuration | AUC (CW) | HM (CW) | AUC (OW) | Note |
|------|----------|---------|----------|------|
| Full (TEO + OGA) | 8.2 | 23.9 | 2.10 | Full model |
| −TEO | 7.6 | 22.7 | 1.79 | Without text-enhanced object learning |
| −TEO−OGA | 6.9 | 21.4 | 1.69 | Without both core modules |
| Average replaces Eq. 2 attention | 7.8 | 22.9 | 1.91 | Simple averaging replaces cross-attention |
| Average replaces Eq. 4 attention | 7.1 | 22.0 | 1.79 | Attention removed from attribute learning |

### Key Findings
- Both TEO and OGA contribute substantially: removing TEO reduces AUC by 7.3%; further removing OGA causes an additional 9.2% drop.
- Cross-attention outperforms simple averaging, confirming that selectively attending to relevant image regions is critical.
- Improvements are especially pronounced in the Open-World setting: MIT-States AUC improves by 175%, C-GQA AUC by 47.9%.
- Combining deep and shallow visual features outperforms using deep features alone; shallow features provide finer-grained attribute information.
- Inference speed of 1457 fps is comparable to ADE (1445) and CoT (1460), indicating negligible additional computational overhead.

## Highlights & Insights
- The conditional probability decomposition perspective offers a theoretically grounded and intuitive reformulation of CZSL.
- The "identify object first, then examine attribute" design aligns with human visual cognition—people typically recognize an object before attending to its attributes.
- Using deep features for object learning and shallow features for attribute learning is well-motivated: deep features capture semantics, while shallow features retain texture and color details relevant to attributes.
- Additive fusion at inference (rather than probabilistic multiplication) avoids numerical underflow—a practically useful engineering choice.
- The negligible computational overhead (1457 fps) suggests that CPF's gains stem primarily from improved feature organization rather than increased computation.

## Limitations & Future Work
- Improvements over the CLIP backbone are smaller than those over ViT-B, suggesting that CLIP's existing compositional generalization partially subsumes CPF's advantages.
- The conditional probability decomposition assumes attributes depend on objects; however, in some cases objects may also depend on attributes (e.g., *square* implies *window* rather than *door*).
- The current framework addresses only single attribute–single object compositions; real-world scenarios may involve multiple attributes or multiple objects.
- Text embeddings rely on frozen GloVe representations, which may limit the expressiveness of semantic descriptions.
- The selection of shallow and deep feature layers (layers 3/6/9 vs. the final layer) is fixed and may not be optimal across all datasets.

## Related Work & Insights
- **vs. CoT (ICCV 2023)**: CoT only uses object features to generate attribute attention maps; CPF additionally introduces textual semantic constraints and employs cross-attention.
- **vs. ADE (CVPR 2023)**: ADE disentangles attributes and objects independently; CPF models their dependencies via a conditional probability framework, yielding a 17.9% AUC gain on UT-Zappos50K.
- **vs. CANet (CVPR 2023)**: CANet adjusts attribute learner parameters using object priors but still treats the two as independent modules; CPF's cross-attention directly models dependencies at the feature level.
- **vs. CLIP-based methods (CSP/Troika, etc.)**: CPF surpasses all CLIP-based methods on C-GQA, demonstrating the generality of the conditional probability framework.
- **Inspiration**: The conditional probability decomposition paradigm is potentially extensible to other compositional learning problems, such as action-object interaction recognition.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The conditional probability decomposition perspective is novel; the progressive TEO+OGA design is clear and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Both CW and OW settings, three datasets, two backbones (ViT-B and CLIP), and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is well-articulated, derivations are clear, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ — Provides a new paradigm for CZSL; substantial performance gains validate the importance of modeling attribute-object dependencies.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](batclip_bimodal_online_test-time_adaptation_for_clip.md)
- [\[ICCV 2025\] Few-Shot Pattern Detection via Template Matching and Regression](few-shot_pattern_detection_via_template_matching_and_regression.md)
- [\[ICCV 2025\] Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting](rethinking_few_shot_clip_benchmarks_a_critical_analysis_in_the_inductive_setting.md)
- [\[ICCV 2025\] InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild](intersyn_interleaved_learning_for_dynamic_motion_synthesis_in_the_wild.md)
- [\[AAAI 2026\] GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval](../../AAAI2026/llm_evaluation/granalign_granularity-aware_alignment_framework_for_zero-shot_video_moment_retri.md)

<!-- RELATED:END -->
