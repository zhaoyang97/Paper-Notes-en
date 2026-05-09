---
title: >-
  [Paper Note] Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning
description: >-
  [CVPR 2026][Optimization][Federated Prototype Learning] This paper proposes FedTSP, which leverages pre-trained language models (PLMs) to construct semantically rich prototypes from the text modality, preserving inter-class semantic relationships in heterogeneous federated learning. Learnable prompts are introduced to bridge the modality gap, substantially improving model performance and accelerating convergence.
tags:
  - CVPR 2026
  - Optimization
  - Federated Prototype Learning
  - Pre-trained Language Model
  - Semantic Relation Preservation
  - Heterogeneous Federated Learning
  - Cross-modal Alignment
date: 2026-05-08
content_hash: 7f132276b80d1161
---

# Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning

**Conference**: CVPR 2026
**arXiv**: [2503.13543](https://arxiv.org/abs/2503.13543)
**Code**: [https://github.com/XinghaoWu/FedTSP](https://github.com/XinghaoWu/FedTSP)
**Area**: Optimization
**Keywords**: Federated Prototype Learning, Pre-trained Language Model, Semantic Relation Preservation, Heterogeneous Federated Learning, Cross-modal Alignment

## TL;DR

This paper proposes FedTSP, which leverages pre-trained language models (PLMs) to construct semantically rich prototypes from the text modality, preserving inter-class semantic relationships in heterogeneous federated learning. Learnable prompts are introduced to bridge the modality gap, substantially improving model performance and accelerating convergence.

## Background & Motivation

1. **Background**: Federated Prototype Learning (FedPL) aligns client representations by sharing global prototypes to mitigate data heterogeneity, and prototype quality directly determines performance.
2. **Limitations of Prior Work**: Methods such as AlignFed and FedNH pursue maximum inter-class prototype separation (uniformly distributed on a hypersphere), which destroys inter-class semantic relationships. For example, "horse" and "dog" should be more similar to each other than "horse" and "truck."
3. **Key Challenge**: Enlarging prototype distances enhances class discrimination but inevitably disrupts semantic structure, which is critical for model generalization.
4. **Goal**: Construct prototypes that both preserve semantic relationships and maintain sufficient discriminability.
5. **Key Insight**: Exploit the rich semantic knowledge encoded in PLMs to build prototypes, thereby injecting semantic structure into federated learning.
6. **Core Idea**: Use an LLM to generate class descriptions, encode them into text prototypes via a PLM, and employ learnable prompts to bridge the image–text modality gap.

## Method

### Overall Architecture

Step 0: The server uses an LLM to generate class descriptions and encodes them into embeddings via a PLM. Steps 1–4 iterate: clients compute image prototypes → the server updates text prototypes with learnable prompts → the server aligns text and image prototypes → the server distributes text prototypes to clients for local alignment training.

### Key Designs

1. **LLM-Generated Multi-View Prompts**:

   - **Function**: Generate fine-grained textual descriptions for each class to enrich semantic context.
   - **Mechanism**: Template "A photo of {CLASS}: {description}" is used; $k=3$ descriptions covering different aspects are generated per class. The text encoder processes them and averages the outputs to obtain the class text prototype $\bar{P}_c^T$.
   - **Design Motivation**: A bare class name (e.g., "apple") is ambiguous and carries insufficient semantic information; multi-view descriptions help the encoder capture finer-grained semantic distinctions.

2. **Modality Alignment via Learnable Prompts**:

   - **Function**: Bridge the modality gap between the PLM and client image models.
   - **Mechanism**: $m$ learnable vectors $v_c$ are inserted into the text embedding sequence, replacing the first $m$ token embeddings. The $k$ prompts per class share the same set of learnable prompts. After aggregating client image prototypes, the server updates the learnable prompts via a contrastive loss $\mathcal{L}_S$ to align text prototypes with image prototypes.
   - **Design Motivation**: Since PLMs have not been exposed to image data, directly using their features as prototypes introduces a modality gap. Learnable prompts adapt the text prototypes to the visual tasks of clients.

3. **Contrastive Learning-Based Semantic Transfer**:

   - **Function**: Transfer the semantic relational structure of prototypes to client models.
   - **Mechanism**: Clients align local features with text prototypes via a contrastive loss: $\mathcal{R} = -\log \frac{\exp(\cos(f_i(x), \mathcal{P}_y^T)/\tau)}{\sum_c \exp(\cos(f_i(x), \mathcal{P}_c^T)/\tau)}$. The temperature parameter $\tau$ directs the model to focus on relative rankings rather than absolute values.
   - **Design Motivation**: L2 alignment optimizes absolute similarity and may mislead the model into treating unrelated classes as similar. Contrastive learning optimizes relative rankings, naturally preserving semantic structure.

### Loss & Training

Client loss = CE loss + $\lambda$ × contrastive alignment loss. On the server side, learnable prompts are trained via contrastive loss for $E_s$ rounds.

## Key Experimental Results

### Main Results

| Dataset | Metric | FedTSP-CLIP | Prev. SOTA (FedTGP) | Gain |
|--------|------|-------------|-------------------|------|
| CIFAR-10 (α=0.1) | Acc | 87.34 | 85.73 | +1.61 |
| CIFAR-100 (α=0.1) | Acc | 45.61 | 41.37 | +4.24 |
| Tiny-ImageNet (α=0.1) | Acc | 34.82 | 31.16 | +3.66 |

### Ablation Study

| Configuration | CIFAR-10 | CIFAR-100 | Note |
|------|----------|-----------|------|
| FedTSP-CLIP | 87.34 | 45.61 | CLIP encoder |
| FedTSP-BERT | 87.52 | 46.08 | BERT encoder also effective |
| w/o LLM descriptions | Notable drop | — | LLM descriptions are critical |

### Key Findings

- Gains are most pronounced under strong data heterogeneity (α=0.1), indicating that text prototypes are more robust to heterogeneous data distribution shifts.
- BERT (not trained on images) also works effectively, demonstrating that the framework does not depend on vision–language pre-training.
- Top-5 accuracy gains are larger, suggesting that even misclassified samples are more likely to be assigned to semantically related classes.

## Highlights & Insights

- **Quantification of Semantic Structure Preservation**: Spearman correlation and semantic margin are proposed as two metrics to quantify the semantic quality of prototypes.
- **Framework Agnosticism**: The approach does not rely on CLIP; pure text models such as BERT are equally applicable.
- **Privacy-Preserving Extension**: A differentially private variant is designed that adds noise to text embeddings to protect class-name privacy.

## Limitations & Future Work

- Validation is currently limited to image classification; extension to detection or segmentation has not been explored.
- The quality of LLM-generated descriptions depends on specific classes and may be insufficiently precise for fine-grained categories.
- The number of learnable prompts $m$ requires tuning.
- The current framework supports only classification tasks; adapting it to detection and segmentation requires additional design.

## Related Work & Insights

- **vs. AlignFed**: AlignFed distributes prototypes uniformly; FedTSP preserves semantic structure.
- **vs. FedTGP**: FedTGP uses trainable prototypes to pursue maximum separation while ignoring semantic relationships.
- **vs. CLIP-FL**: CLIP-FL fine-tunes CLIP for inference; FedTSP transfers semantic knowledge to lightweight client models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to introduce textual semantics into federated prototype learning; the cross-modal knowledge transfer paradigm is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets, heterogeneity settings, and privacy extensions, with systematic ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly illustrated through visualizations; the semantic alignment and margin quantification metrics are elegantly designed.
- **Value**: ⭐⭐⭐⭐ The idea of semantics-preserving prototypes has broad transferability and can be generalized to other scenarios requiring cross-modal knowledge transfer.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers](the_power_of_decaying_steps_enhancing_attack_stability_and_transferability_for_s.md)
- [\[CVPR 2026\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](scope_semantic_coreset_with_orthogonal_projection.md)
- [\[CVPR 2026\] OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport](otprune_distribution-aligned_visual_token_pruning_via_optimal_transport.md)
- [\[CVPR 2026\] Dynamic Momentum Recalibration in Online Gradient Learning](dynamic_momentum_recalibration_in_online_gradient_learning.md)
- [\[CVPR 2026\] BlazeFL: Fast and Deterministic Federated Learning Simulation](blazefl_fast_and_deterministic_federated_learning_simulation.md)

<!-- RELATED:END -->
