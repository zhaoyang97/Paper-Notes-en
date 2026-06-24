---
title: >-
  [Paper Note] Rethinking Query-Based Transformer for Continual Image Segmentation
description: >-
  [CVPR 2025][Segmentation][Continual Image Segmentation] This paper deeply analyzes the mechanism of emergence and extinction of built-in objectness in query-based Transformers. It proposes the SimCIS method, which consists of three modules: Query Pre-Alignment (QPA), Consistent Selection Loss (CSL), and Virtual Queries (VQ). By maintaining objectness while enhancing plasticity, SimCIS significantly outperforms state-of-the-art methods in continual panoptic segmentation and co…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Continual Image Segmentation"
  - "query-based Transformer"
  - "built-in objectness"
  - "virtual query"
  - "catastrophic forgetting"
date: 2026-05-08
content_hash: a5ef1c29ea8e19bf
---

# Rethinking Query-Based Transformer for Continual Image Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2507.07831](https://arxiv.org/abs/2507.07831)  
**Code**: [https://github.com/SooLab/SimCIS](https://github.com/SooLab/SimCIS)  
**Area**: Image Segmentation / Continual Learning  
**Keywords**: Continual Image Segmentation, query-based Transformer, built-in objectness, virtual query, catastrophic forgetting

## TL;DR

This paper deeply analyzes the mechanism of emergence and extinction of built-in objectness in query-based Transformers. It proposes the SimCIS method, which consists of three modules: Query Pre-Alignment (QPA), Consistent Selection Loss (CSL), and Virtual Queries (VQ). By maintaining objectness while enhancing plasticity, SimCIS significantly outperforms state-of-the-art methods in continual panoptic segmentation and continual semantic segmentation tasks on ADE20K.

## Background & Motivation

**Background**: Continual Image Segmentation (CIS) requires models to step-by-step adapt to new classes in multi-stage learning while retaining knowledge of old classes. Recently, query-based Transformers (e.g., Mask2Former) have been introduced to the CIS field, with their built-in objectness considered helpful for mitigating catastrophic forgetting in mask generation. Existing methods (such as ECLIPSE, CoMasTRe) typically freeze parameters related to mask generation, decoupling mask segmentation from the continual learning process.

**Limitations of Prior Work**: The authors find two key issues in decoupled frameworks: (1) **Loss of plasticity**—the advantages of objectness diminish or even have negative impacts in short task sequences, and the performance in the shortest two-task setting is even lower than the baseline; (2) **Heavy dependence on the order of input data**—in ten random trials, the worst-case scenario significantly drops compared to the default setting, indicating a lack of robustness.

**Key Challenge**: Although built-in objectness exists in the feature maps (pixel features contain sufficient semantic priors), it gradually fades away as the training stages progress. The root cause is: due to background semantic shifts, the semantic priors in different stages vary, causing the learnable queries to gradually lose alignment with the pixel features of old classes.

**Goal**: Understand the essence of built-in objectness, achieve consistent performance improvements across various task lengths and data input sequences, and especially enhance plasticity.

**Key Insight**: The authors find that highly aggregated image features provide a "shortcut" for queries—queries can generate masks simply by aligning with the semantic priors in the feature map through the decoder. Based on this, directly selecting pixel features from the feature map to initialize queries can achieve "perfect alignment" to maintain objectness.

**Core Idea**: Replace learnable queries with pixel features selected from the feature map (lazy pre-alignment), and combine this with cross-stage consistent selection constraints and a virtual query replay mechanism to both maintain objectness and enhance plasticity.

## Method

### Overall Architecture

SimCIS is based on the Mask2Former architecture and contains three core modules: (1) Query Pre-Alignment (QPA) selects the most semantically salient positions from pixel features to initialize queries, ensuring objectness in each stage; (2) Consistent Selection Loss (CSL) ensures selection stability across stages; (3) Virtual Queries (VQ) store and replay query features of old classes to avoid catastrophic forgetting of class predictions. After the input image passes through the backbone and pixel decoder to extract multi-scale pixel features, QPA selects $N$ most salient feature points as object queries, which are sent to the Transformer decoder to generate masks and class predictions.

### Key Designs

1. **Query Pre-Alignment (QPA)**:

    - **Function**: Select the most semantically salient positions from the pixel feature map as the initial features of the queries, ensuring the queries are "perfectly" pre-aligned with semantic priors.
    - **Mechanism**: A set of learnable prototypes $p^i \in \mathbb{R}^D$ is maintained for each category. The similarity between each pixel feature and all prototypes is calculated, and the $N$ feature points with the highest similarity are selected as queries. The key is to apply stop gradient to the queries to prevent the training process from destroying the information in the feature maps. The prototype set for each new stage is obtained by concatenating the prototypes of the old stages and the new stages: $\mathcal{P}^t = \text{concat}(\mathcal{P}^{t-1}, \{p^i | i \in C^t\})$.
    - **Design Motivation**: Traditional learnable queries lose alignment with the feature maps during multi-stage training, leading to the extinction of objectness. Directly selecting from the feature map ensures alignment with the current semantic prior at each stage, while the stop gradient maintains feature stability.

2. **Consistent Selection Loss (CSL)**:

    - **Function**: Ensure that the selected semantically salient positions for the same image remain consistent across different stages.
    - **Mechanism**: During training in the current stage $t$, the selection indices $\mathcal{I}^{t-1}$ from the previous stage $t-1$ are used to extract feature points from the current feature maps. Their similarity distribution with the old prototypes is calculated, and a KL divergence loss is applied to constrain this distribution to be consistent with the previous stage. The formula is $L_{csl} = \frac{1}{|\mathcal{I}^{t-1}|} \sum KL(\text{旧分布} \| \text{新分布})$.
    - **Design Motivation**: It avoids the problem of reintroducing background annotation errors when preserving old priors in traditional distillation methods. Thanks to the design of QPA, the query positions of old classes can be naturally preserved while allowing new queries to select new classes.

3. **Virtual Queries (VQ)**:

    - **Function**: Avoid catastrophic forgetting of class predictions by storing and replaying query features of old classes.
    - **Mechanism**: Implemented in three steps. First, the matching queries are selected from the decoder output based on bipartite matching results and stored into the category queue to form the VQ bank. Second, pseudo-distribution statistics are used to analyze the frequency of occurrence of each old class in the current stage, and rare classes are sampled with weights. Finally, the sampled virtual queries are concatenated with normal queries and fed into the decoder. Specifically, a skip attention strategy is designed—VQ skips cross-attention and self-attention and only participates in the computation of FFN layers, avoiding interference with the attention process of normal queries.
    - **Design Motivation**: Compared to traditional image replay methods, VQ reduces storage requirements by about 10 times, is independent of the input data order, and protects data privacy. Virtual queries naturally contain category semantic information and can simulate specific semantics without actually containing images of corresponding categories.

### Loss & Training

The overall loss includes the original classification loss and mask loss of Mask2Former, plus the CSL loss for cross-stage consistency constraints. VQ only computes the classification loss $L_{\text{class}}$ to handle class forgetting. Pre-trained ResNet-50 (for panoptic segmentation) and ResNet-101 (for semantic segmentation) are used as backbones, with input resolutions of 640×640 and 512×512, respectively. The number of virtual queries is set to 80.

## Key Experimental Results

### Main Results

| Dataset/Setting | Metric | SimCIS | ECLIPSE | BalConpas | Gain (vs BalConpas) |
|------------|------|--------|---------|-----------|-------------------|
| ADE20K CPS 100-5 | PQ(all) | **35.4** | 32.9 | 30.8 | +4.6 |
| ADE20K CPS 100-10 | PQ(all) | **38.1** | 33.9 | 34.7 | +3.4 |
| ADE20K CPS 100-50 | PQ(all) | **40.0** | 35.6 | 37.1 | +2.9 |
| ADE20K CPS 50-10 | PQ(all) | **36.3** | 26.8 | 31.4 | +4.9 |
| ADE20K CSS 100-5 | mIoU(all) | **38.7** | 34.2 | 33.8 | +4.9 |
| ADE20K CSS 100-10 | mIoU(all) | **42.3** | 34.6 | 38.6 | +3.7 |
| ADE20K CSS 100-50 | mIoU(all) | **48.6** | 37.1 | 43.3 | +5.3 |

### Ablation Study

| Configuration | CPS base | CPS all | CSS base | CSS all | Description |
|------|----------|---------|----------|---------|------|
| Baseline (Pseudo Label) | 31.6 | 28.2 | 15.6 | 13.2 | Mask2Former + Pseudo Label |
| + QPA | 30.7 | 27.9 | 37.4 | 30.5 | base mIoU +21.8 |
| + QPA + CSL | 35.7 | 31.8 | 43.2 | 34.5 | CSL improves base significantly |
| + QPA + VQ | 35.1 | 31.2 | 42.5 | 34.8 | VQ helps new |
| Full (QPA+CSL+VQ) | **42.1** | **35.4** | **46.7** | **38.7** | Mutual complementarity among the three |

### Key Findings

- **QPA contributes the most**: It improves the base mIoU from 15.6% to 37.4% (+21.8%) in the CSS task, proving that directly selecting queries from feature maps is core to maintaining objectness.
- **VQ storage efficiency is much higher than image replay**: Using 80 VQ samples (5.9MB) achieves better performance than using 300 replay images (11.8MB), with a PQ improvement of +1.4% while using only 27% of the storage.
- **Robustness to data input order**: Across 10 random trials, the performance variance of SimCIS is much smaller than that of ECLIPSE and BalConpas, proving that the effective utilization of objectness improves robustness.
- **Close to joint upper bound in 100-50 task**: The PQ of the CPS 100-50 task reaches 40.0 vs 40.4 of joint, and the base categories even surpass joint.

## Highlights & Insights

- **Deep analysis of built-in objectness**: Visualization reveals that objectness originates from the alignment of queries with feature map semantic priors, and fades away during multi-stage training due to alignment drift. This analysis framework holds broad reference value.
- **Skip attention design of VQ**: Virtual queries skip attention and only participate in FFN, avoiding interference with real queries while propagating category information through the FFN layer. This design can be transferred to other scenarios that require mixing real/virtual tokens.
- **Feature selection instead of feature learning**: The "lazy" idea of abandoning learning queries and instead directly selecting pixel features is ingenious, essentially leveraging the existing semantic clustering properties of pre-trained features.

## Limitations & Future Work

- The method relies on prototypes to accurately represent category semantics. It may face prototype management scalability issues when the number of categories is extremely large.
- The queue length and sampling strategy of the VQ bank have an impact on performance. The authors choose 80 samples as optimal, but different datasets may require different configurations.
- Only evaluated on ADE20K, failing to provide validation on other more challenging datasets such as COCO and Cityscapes.
- Exploring the combination of the QPA mechanism with other continual learning methods (like prompt tuning) is a potential future direction.

## Related Work & Insights

- **vs ECLIPSE**: ECLIPSE fine-tunes by freezing most parameters and providing trainable queries, but freezing leads to loss of plasticity. SimCIS does not freeze parameters, but instead ensures that the correct positions are selected as queries each time, maintaining flexibility.
- **vs BalConpas**: BalConpas uses feature distillation and image replay, but its performance drops significantly in long-sequence tasks. SimCIS's VQ mechanism outperforms image replay in both storage efficiency and performance.
- **vs CoMFormer**: CoMFormer is the first work to use a query-based method in continual panoptic segmentation, but it is reliant on distillation and pseudo-labels. SimCIS addresses the objectness preservation problem from a more fundamental perspective.

## Rating

- Novelty: ⭐⭐⭐⭐ The analysis of objectness is in-depth. The designs of QPA and VQ are simple and effective, though not disruptively innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers both CPS and CSS tasks, multiple settings, and random sequence robustness tests. Complete ablation study.
- Writing Quality: ⭐⭐⭐⭐ The logic is clear and charts are rich, though some formula notations are slightly complex.
- Value: ⭐⭐⭐⭐ Provides a simple and powerful baseline in the continual segmentation field, with a highly guiding analysis framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)
- [\[CVPR 2025\] MambaVision: A Hybrid Mamba-Transformer Vision Backbone](mambavision_a_hybrid_mamba-transformer_vision_backbone.md)
- [\[ICCV 2025\] Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation](../../ICCV2025/segmentation/hierarchical_visual_prompt_learning_for_continual_video_instance_segmentation.md)
- [\[ICCV 2025\] Rethinking Detecting Salient and Camouflaged Objects in Unconstrained Scenes](../../ICCV2025/segmentation/rethinking_detecting_salient_and_camouflaged_objects_in_unconstrained_scenes.md)
- [\[CVPR 2026\] GeoMotion: Rethinking Motion Segmentation via Latent 4D Geometry](../../CVPR2026/segmentation/geomotion_rethinking_motion_segmentation_via_latent_4d_geometry.md)

</div>

<!-- RELATED:END -->
