---
title: >-
  [Paper Note] Attention Decomposition for Cross-Domain Semantic Segmentation
description: >-
  [ECCV 2024][Segmentation][Cross-Domain Semantic Segmentation] This paper proposes ADFormer, a novel Transformer architecture for cross-domain semantic segmentation. By decomposing the cross-attention in the decoder into domain-agnostic and domain-specific components, combined with gradient reversal adversarial learning, it effectively bridges the distribution gap between source and target domains. It outperforms existing proposal-free methods on GTA→Cityscapes and SYNTHIA→Cit…
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Cross-Domain Semantic Segmentation"
  - "Attention Decomposition"
  - "Adversarial Learning"
  - "Transformer Decoder"
  - "Domain Adaptation"
date: 2026-05-08
content_hash: 8a14fc400ca8d339
---

# Attention Decomposition for Cross-Domain Semantic Segmentation

**Conference**: ECCV 2024  
**Code**: [https://github.com/helq2612/ADFormer](https://github.com/helq2612/ADFormer)  
**Area**: Semantic Segmentation / Domain Adaptation  
**Keywords**: Cross-Domain Semantic Segmentation, Attention Decomposition, Adversarial Learning, Transformer Decoder, Domain Adaptation

## TL;DR

This paper proposes ADFormer, a novel Transformer architecture for cross-domain semantic segmentation. By decomposing the cross-attention in the decoder into domain-agnostic and domain-specific components, combined with gradient reversal adversarial learning, it effectively bridges the distribution gap between source and target domains. It outperforms existing proposal-free methods on GTA→Cityscapes and SYNTHIA→Cityscapes benchmarks with significantly lower complexity.

## Background & Motivation

**Background**: Cross-domain semantic segmentation aims to transfer a segmentation model trained on a source domain (e.g., synthetic data like GTA5, SYNTHIA) to a target domain (e.g., real-world scenes like Cityscapes). With the widespread application of Transformers in segmentation tasks, state-of-the-art (SOTA) methods are mainly categorized into CNN-based domain adaptation methods and Transformer-based proposal-free methods. Transformer methods deliver strong performance by using query tokens to predict segmentation masks.

**Limitations of Prior Work**: Existing cross-domain segmentation methods face two primary challenges: (1) Traditional Transformer-based segmentation models (e.g., Mask2Former) typically rely on complex encoders and relatively simple decoders, without explicitly considering domain discrepancy; (2) In cross-domain scenarios, the cross-attention between query tokens and image tokens absorbs both domain-agnostic information (e.g., cross-domain shared semantics like object shapes and structures) and domain-specific information (e.g., domain-specific distractors like texture styles and illumination), where the latter is the root cause of the domain gap.

**Key Challenge**: The cross-attention mechanism in the Transformer decoder processes image features from different domains indiscriminately, causing query tokens to inevitably encode domain-specific noise. If query tokens could be guided to focus solely on the domain-agnostic shared semantics, the domain gap would be fundamentally mitigated.

**Goal**: (1) Explicitly separate domain-agnostic and domain-specific attention interactions within the Transformer decoder; (2) Achieve effective domain adaptation while maintaining a lightweight model.

**Key Insight**: The authors propose that the cross-attention matrix can be mathematically decomposed into domain-agnostic and domain-specific components, allowing constraints to force query tokens to interact primarily with the domain-agnostic counterpart. Integrating adversarial learning via a Gradient Reversal Layer (GRL) further enhances domain invariance.

**Core Idea**: Decompose the cross-attention of the Transformer decoder into domain-agnostic and domain-specific components, enabling query tokens to focus only on domain-agnostic semantics for cross-domain semantic segmentation.

## Method

### Overall Architecture

ADFormer adopts an encoder-decoder architecture under the "lightweight encoder + complex decoder" design philosophy. The encoder (relatively lightweight) extracts multi-scale image features, while the decoder facilitates interactions between learnable query tokens and image features through multiple cross-attention layers. Ultimately, each query token predicts a segmentation mask and its corresponding class label. The key innovations lie in the decoder: (1) Cross-attention is decomposed into domain-agnostic and domain-specific branches; (2) A gradient reversal block regulates backpropagation to force the model to learn domain-invariant representations.

### Key Designs

1. **Attention Decomposition**:

    - **Function**: Decomposes the cross-attention between query tokens and image tokens in the decoder into domain-agnostic and domain-specific components.
    - **Mechanism**: Given query tokens $Q$ and image tokens $K, V$, standard cross-attention is formulated as $\text{Attn}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d})V$. ADFormer introduces two sets of projection matrices: domain-agnostic projection $W_{di}$ and domain-specific projection $W_{ds}$. Image tokens are mapped via these projections to yield domain-agnostic components $K_{di}, V_{di}$ and domain-specific components $K_{ds}, V_{ds}$. Query tokens primarily interact with the domain-agnostic components $K_{di}, V_{di}$ to predict segmentation masks, while the domain-specific components $K_{ds}, V_{ds}$ are utilized for adversarial learning. This naturally filters out domain-specific distractors from the information acquired by the query tokens.
    - **Design Motivation**: Although images from the source and target domains differ significantly in texture and style, semantic information such as object shape and structure remains shared. Through explicit decomposition, the model is forced to segregate shared semantics and domain distractors into distinct subspaces.

2. **Gradient Reverse Adversarial Learning**:

    - **Function**: Regulates the domain-specific branch via adversarial training to ensure it captures actual domain discrepancy information.
    - **Mechanism**: A domain classifier (which predicts whether the input is from the source or target domain) is appended after the domain-specific attention branch, accompanied by a Gradient Reversal Layer (GRL). During forward propagation, the GRL operates as an identity mapping; during backpropagation, the GRL reverses the gradients. Consequently, the domain-specific branch is "encouraged" to encode domain discrepancy (to help the domain classifier succeed), while the encoder's feature extractor is "encouraged" to reduce domain discrepancy (as gradient reversal optimizes it to confuse the domain classifier).
    - **Design Motivation**: Structural constraints from attention decomposition alone are insufficient—without explicit domain classification signals, the model might erroneously assign useful semantic information to the domain-specific branch. Adversarial learning provides an explicit domain discrepancy signal to guarantee high-quality decomposition.

3. **Lightweight Encoder and Complex Decoder Architecture**:

    - **Function**: Delivers sufficient cross-domain adaptation capability while maintaining model efficiency.
    - **Mechanism**: Unlike Mask2Former, which employs heavy encoders (e.g., Swin-L), ADFormer utilizes a relatively lightweight encoder (e.g., ResNet-50 or a small ViT) but incorporates attention decomposition and adversarial learning modules within the decoder. The decoder consists of multiple cross-attention layers, with attention decomposition applied at each layer. The number of query tokens corresponds to the number of classes, directly predicting the segmentation mask for each class.
    - **Design Motivation**: The core challenge of cross-domain segmentation lies in the feature-to-semantic mapping during the decoding phase rather than feature extraction during encoding. Concentrating the adaptation capability in the decoder effectively bridges the domain gap without significantly increasing computational costs.

### Loss & Training

The total training loss comprises: (1) Segmentation loss $L_{seg}$: a combination of cross-entropy and Dice loss to supervise mask predictions; (2) Adversarial loss $L_{adv}$: binary cross-entropy loss for the domain classifier, backpropagated through the GRL. The total loss is formulated as $L = L_{seg} + \lambda L_{adv}$. Training is conducted in two stages: first, pre-training the encoder and decoder on source domain data; second, joint source-target training for domain adaptation (where target domain data has no labels and only participates in the adversarial loss). During inference, only the domain-agnostic attention branch is utilized.

## Key Experimental Results

### Main Results

| Dataset Transfer | Metric | Ours (ADFormer) | Prev. SOTA | Gain |
|-----------|------|-------------|----------|------|
| GTA→Cityscapes | mIoU | SOTA-level | Proposal-free methods | Significant improvement |
| SYNTHIA→Cityscapes | mIoU | SOTA-level | Proposal-free methods | Significant improvement |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Without decomposition (baseline) | Baseline mIoU | Standard cross-attention |
| + Attention decomposition | mIoU improvement | Structural decomposition only |
| + Adversarial learning | Further mIoU improvement | Added GRL domain classifier |
| Full ADFormer | Best | Joint decomposition + adversarial learning |

### Key Findings

- Attention decomposition is the most critical component, significantly reducing the domain gap through structural constraints alone.
- Combining adversarial learning with decomposition yields even better results, demonstrating their complementary nature.
- The model complexity of ADFormer is substantially lower than existing SOTA proposal-free methods, yet it delivers superior performance.
- Competitiveness is maintained even with lighter encoders, indicating that domain adaptation capability is primarily concentrated in the decoder.

## Highlights & Insights

- The concept of attention decomposition is highly elegant—explicitly partitioning a continuous attention space into domain-agnostic and domain-specific subspaces, addressing domain adaptation from a mechanism design standpoint.
- The "lightweight encoder + complex decoder" design philosophy stands contrary to the mainstream trend but proves more rational in cross-domain scenarios, as the domain gap predominantly manifests during the semantic mapping stage.
- The integration of gradient reversal with attention decomposition is seamless and natural, without introducing excessive hyperparameters or training heuristics.

## Limitations & Future Work

- This work primarily validates synthetic-to-real domain transfer scenarios, leaving real-to-real domain transfers (e.g., clear to rainy weather) insufficiently explored.
- The granularity of attention decomposition is fixed (a binary division of domain-agnostic vs. domain-specific), which may require more fine-grained decomposition methods for multi-domain scenarios.
- The design of the domain classifier is relatively simple; more complex domain discriminators or multi-level domain classification might yield further performance gains.
- The performance of the model in open-vocabulary or zero-shot cross-domain scenarios has not been discussed.

## Related Work & Insights

- **DAFormer** and **HRDA** are representative Transformer-based domain adaptation segmentation methods, but they primarily perform domain adaptation on the encoder side.
- **Mask2Former**'s query token + cross-attention paradigm establishes the foundation for the decoder design in this work.
- The gradient reversal concept from **DANN** (Domain Adversarial Neural Networks) is widely utilized in domain adaptation, and this work innovatively applies it to the Transformer attention mechanism.
- The concept of attention decomposition could potentially be extended to other scenarios requiring "selective attention", such as cross-modal fusion and multi-task learning.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying attention decomposition to domain adaptation is a novel and elegant approach with clean theoretical formulation.
- Experimental Thoroughness: ⭐⭐⭐ Validated on two standard benchmarks, but lacks evaluation on more diverse domain transfer scenarios.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and rigorous derivation logic.
- Value: ⭐⭐⭐⭐ Provides valuable insights for the field of Transformer-based domain adaptation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Interpreting ResNet-based CLIP via Neuron-Attention Decomposition](../../NeurIPS2025/segmentation/interpreting_resnet-based_clip_via_neuron-attention_decomposition.md)
- [\[CVPR 2026\] Bayesian Decomposition and Semantic Completion for Few-shot Semantic Segmentation](../../CVPR2026/segmentation/bayesian_decomposition_and_semantic_completion_for_few-shot_semantic_segmentatio.md)
- [\[ICML 2025\] Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation](../../ICML2025/segmentation/adapter_naturally_serves_as_decoupler_for_cross-domain_few-shot_semantic_segment.md)
- [\[AAAI 2026\] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](../../AAAI2026/segmentation/bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)
- [\[ICML 2025\] Balanced Learning for Domain Adaptive Semantic Segmentation](../../ICML2025/segmentation/balanced_learning_for_domain_adaptive_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
