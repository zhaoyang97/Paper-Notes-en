---
title: >-
  [Paper Note] Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning
description: >-
  [ACL 2025][Model Compression][compositional zero-shot learning] DCDA proposes a graph-guided cross-composition feature disentanglement scheme. By injecting dual adapters (L-Adapter for text-side GNN feature aggregation and V-Adapter for vision-side cross-attention disentanglement) into the frozen CLIP, it significantly outperforms existing methods on compositional zero-shot learning tasks.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "compositional zero-shot learning"
  - "CLIP adapter"
  - "feature disentanglement"
  - "graph neural network"
  - "cross-attention"
date: 2026-05-08
content_hash: ee513cd9d7459555
---

# Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning

**Conference**: ACL 2025  
**arXiv**: [2408.09786](https://arxiv.org/abs/2408.09786)  
**Code**: [Available](https://github.com/zhurunkai/DCDA)  
**Area**: Other  
**Keywords**: compositional zero-shot learning, CLIP adapter, feature disentanglement, graph neural network, cross-attention

## TL;DR

DCDA proposes a graph-guided cross-composition feature disentanglement scheme. By injecting dual adapters (L-Adapter for text-side GNN feature aggregation and V-Adapter for vision-side cross-attention disentanglement) into the frozen CLIP, it significantly outperforms existing methods on compositional zero-shot learning tasks.

## Background & Motivation

The goal of Compositional Zero-Shot Learning (CZSL) is to enable the model to recognize unseen attribute-object compositions (e.g., green tomato) at test time, having only seen a subset of compositions during training (e.g., red tomato, green apple). This requires the model to disentangle the visual features of attributes and objects and recombine them for generalization.

The core challenge faced by current CLIP-based CZSL methods is the **visual entanglement of primitives**:

**Attributes and objects are highly entangled in images**: For example, "red" in red tomato permeates all pixels and cannot be easily separated.

**The same primitive exhibits substantial variation across different compositions**: For instance, "red" has completely different hues and spatial distributions in tomato, wine, and car (Figure 1).

**Existing methods ignore cross-composition diversity**: Whether tuning text prompts (e.g., CSP, DFSP) or visual adapters (e.g., CAILA, Troika), they only learn disentangled features within a single composition, without exploiting the relationships between different compositions that share the same attribute.

By performing t-SNE visualization on CAILA using the MIT-States dataset (Figure 2), the authors observed that the disentangled attribute features (e.g., "broken") are still widely scattered and overlap with other attribute clusters, leading to insufficient generalization capability on unseen compositions.

**Key Insight**: Effective primitive disentanglement requires **cross-composition feature aggregation**—exploiting multiple compositions that share the same primitive to constrain the cross-composition consistency of disentangled features.

## Method

### Overall Architecture

DCDA inserts L-Adapter and V-Adapter into the frozen CLIP text encoder and image encoder, respectively. L-Adapter utilizes a composition graph and GNN to aggregate cross-composition primitive features on the text side; V-Adapter utilizes a cross-attention mechanism to extract invariant features from pairs of images sharing the same primitive on the vision side. Both types of adapters are only inserted into the last 3 Transformer blocks.

### Key Designs

1. **L-Adapter (Graph-guided Feature Aggregation on Text Side)**:

    - **Composition Graph Construction**: A tripartite graph is constructed, containing all attribute nodes, object nodes, and composition nodes. Each composition $c=(a,o)$ forms a triangular connection with its attribute $a$ and object $o$.
    - **Node Feature Initialization**: Separate independent prompts are designed for attributes, objects, and compositions ("a photo of [attribute] object", "a photo of [object]", "a photo of [attribute] [object]"). The [EOT] token embeddings output by the CLIP text encoder are taken as the initial features. Consequently, text primitive features are naturally disentangled from composition features.
    - **GNN Feature Propagation**: A $K$-layer GNN is run on the graph. Each layer performs neighborhood aggregation (AGG) and update (CON) on attribute, object, and composition nodes separately, allowing each primitive node to integrate features of all compositions that share it.

   Design Motivation: To explicitly model the sharing relationships among attributes, objects, and compositions through a graph structure, enabling text-side primitive features to generalize to all related compositions.

2. **V-Adapter (Cross-attention Disentanglement on Vision Side)**:

    - **Cross-Attention Mechanism**: For a target image $x_{(a,o)}$, an auxiliary image $x_{(a,o')}$ sharing the same attribute $a$ but with a different object is sampled. Cross-attention is employed (with the auxiliary image as query and the target image as key/value) to extract features in the target image that are more relevant to attribute $a$.
    - **Primitive Relation-guided Sampling Strategy (PRG)**: To address the imbalance in the number of neighboring compositions for different attributes, an object correlation matrix $A^{\text{obj}} = (A^{\text{att-obj}})^T A^{\text{att-obj}}$ is constructed. The top-$n$ most and least correlated objects to the target object are selected to form representative auxiliary compositions, followed by weighted random sampling. Object correlation is determined by the number of co-occurring attributes.

   Design Motivation: Since the visual side cannot establish independent prompts for each primitive like the text side (as attributes and objects are entangled in images), cross-attention between image pairs sharing the same primitive is used to "filter out" visual patterns common across compositions.

3. **Adapter Integration Strategy**:

    - L-Adapter is inserted after the self-attention layer and after the feed-forward layer of the Transformer block (positions ② and ③).
    - V-Adapter is inserted after the entire Transformer block (position ①).
    - Both are only inserted in the last 3 layers to retain the general features of CLIP's lower layers.
    - Each adapter has skip connections: output = adapter output + original input.
    - The word embeddings in the prompts are set to be trainable.

### Loss & Training

The compatibility score fuses matching from three dimensions:

$$s(x_i, c_i) = \alpha[\hat{h}_i^v \cdot \hat{h}_{c_i}^t] + \beta[\hat{h}_{i \to A}^v \cdot \hat{h}_{a_i}^t] + \gamma[\hat{h}_{i \to O}^v \cdot \hat{h}_{o_i}^t]$$

The first term represents composition-level matching, the second represents attribute-level matching, and the third represents object-level matching. $\alpha, \beta, \gamma$ are learnable parameters.

Training utilizes the standard cross-entropy loss, with softmax scaled by a temperature $\tau$ normalized over all seen compositions. Only the adapter parameters and trainable token embeddings are trained, while the original CLIP parameters are completely frozen.

## Key Experimental Results

### Main Results -- Closed-World Setting (Table)

| Method | MIT-States AUC | UT-Zappos AUC | C-GQA AUC |
|------|---------------|---------------|-----------|
| CLIP (No Fine-tuning) | 11.0 | 5.0 | 1.4 |
| CSP | 19.4 | 33.0 | 6.2 |
| CAILA | 23.4 | 44.1 | **9.9** |
| Troika | 22.1 | 41.7 | 9.2 |
| DCDA[RD] | 16.2 | 40.1 | 8.5 |
| DCDA[PRG] | 26.9 | 43.0 | 8.9 |
| **DCDA[PRG+N]** | **27.0** | **44.2** | 9.4 |

DCDA[PRG+N] achieves a 3.6% AUC improvement over CAILA on MIT-States and a 0.1% improvement on UT-Zappos.

### Ablation Study (Table)

| Configuration | MIT-States S | U | H | AUC |
|------|-------------|-----|-----|-----|
| Full Model (DCDA[PRG]) | 57.3 | 55.1 | 43.2 | 26.9 |
| Remove L-Adapter | 55.9 | 54.7 | 42.2 | 26.1 |
| Remove V-Adapter | 44.9 | 46.9 | 33.8 | 17.1 |
| L-Adapter w/o Cross-composition Info | 57.5 | 54.6 | 43.0 | 26.7 |
| V-Adapter w/o Cross-composition Info | 44.5 | 46.2 | 33.7 | 17.0 |
| Both L & V w/o Cross-composition Info | 44.2 | 46.1 | 33.6 | 16.7 |

### Key Findings

1. **V-Adapter is more critical than L-Adapter**: Removing V-Adapter causes the AUC to plummet from 26.9 to 17.1 (-36%), whereas removing L-Adapter only drops it to 26.1. This confirms that visual primitives are more entangled than text primitives, making vision-side disentanglement more crucial.
2. **The PRG sampling strategy is vital**: Comparing DCDA[PRG] to DCDA[RD], the AUC on MIT-States improves from 16.2 to 26.9 (+66%), validating that strategically choosing auxiliary compositions is far superior to random sampling.
3. **Cross-composition information is core to the V-Adapter**: V-Adapter without cross-composition information (17.0 AUC) is even worse than completely removing V-Adapter (17.1 AUC), indicating that disentanglement without cross-composition constraints yields misleading feature subspaces.
4. **Adapters only need to be inserted in the last 3 layers**: Adding more layers (e.g., 6 layers) instead leads to overfitting.

## Highlights & Insights

- **Cross-composition feature disentanglement** is the core innovation of this paper: instead of performing disentanglement within a single composition, it leverages multiple compositions sharing visual/text primitives to constrain each other, learning cross-compositionally consistent primitive features.
- **The dual-adapter architecture is elegantly designed**: The text side employs graph propagation (naturally suited for discrete label structures), while the vision side uses cross-attention (suited for handling continuous and entangled visual features), tailoring distinct mechanisms to the characteristics of different modalities.
- **High parameter efficiency**: Adapters are only inserted into the last 3 layers; the trainable parameters are far fewer than in full-layer schemes (e.g., CAILA/Troika), yet the performance is superior.
- t-SNE visualization (Figure 2, right) clearly demonstrates that after disentanglement by DCDA, attribute clusters are more compact and better separated.

## Limitations & Future Work

1. The construction and update overhead of the composition graph scales with the number of attributes/objects, presenting potential efficiency bottlenecks for large-scale applications.
2. Under extreme long-tail distributions, where rare primitives have only a few compositions, cross-composition supervision may be insufficient.
3. The performance on the C-GQA dataset is not yet optimal, which may be related to low-quality images and a larger composition space.
4. During inference, the true primitive labels of the test image are unknown, so it can only use itself as the auxiliary image, which might reduce the effectiveness of the V-Adapter.
5. Dynamic graph construction mechanisms to handle open-vocabulary primitive discovery have not been explored.

## Related Work & Insights

- CGE (Naeem 2021) first introduced Graph Convolutional Networks (GCNs) into CZSL, but learned representations from scratch rather than leveraging CLIP.
- CAILA (Zheng 2024) and Troika (Huang 2024) also insert adapters into CLIP, but do not exploit cross-composition information.
- This paper demonstrates the significant potential of composition graphs in CZSL: not only serves as the structural foundation of L-Adapter, but also guides the sampling strategy of V-Adapter through the co-occurrence matrix.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The cross-composition disentanglement framework and the dual-adapter design are significant innovations, particularly the ingenious PRG sampling strategy.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly systematic with three datasets, closed/open-world settings, multi-variant comparisons, adapter ablations, and extensive insertion position/depth ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — The motivation is clearly stated, and the illustrations are well-designed, although mathematical notations and formulas could be more compact.
- **Value**: ⭐⭐⭐⭐ — Significantly drives the CZSL field forward; the dual-adapter concept can be transferred to other vision-language tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Distilling Cross-Modal Knowledge via Feature Disentanglement](../../AAAI2026/model_compression/distilling_cross-modal_knowledge_via_feature_disentanglement.md)
- [\[NeurIPS 2025\] Enhancing Semi-supervised Learning with Zero-shot Pseudolabels](../../NeurIPS2025/model_compression/enhancing_semi-supervised_learning_with_zero-shot_pseudolabels.md)
- [\[ACL 2025\] Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)
- [\[ACL 2025\] Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation](beyond_logits_aligning_feature_dynamics_for_effective_knowledge_distillation.md)
- [\[ACL 2025\] Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation](magnet_multi-turn_tool-use_data_synthesis_and_distillation_via_graph_translation.md)

</div>

<!-- RELATED:END -->
