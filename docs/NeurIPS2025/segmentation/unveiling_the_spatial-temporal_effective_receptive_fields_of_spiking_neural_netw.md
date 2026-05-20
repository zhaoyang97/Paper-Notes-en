---
title: >-
  [Paper Note] Unveiling the Spatial-Temporal Effective Receptive Fields of Spiking Neural Networks
description: >-
  [NeurIPS 2025][Segmentation][spiking neural networks] This paper proposes a Spatial-Temporal Effective Receptive Field (ST-ERF) analysis framework to diagnose the bottleneck of Transformer-based SNNs in visual long-seque…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "spiking neural networks"
  - "effective receptive field"
  - "Transformer"
  - "channel mixer"
  - "visual long-sequence modeling"
date: 2026-05-08
content_hash: 3de298fd6550d705
---

# Unveiling the Spatial-Temporal Effective Receptive Fields of Spiking Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2510.21403](https://arxiv.org/abs/2510.21403)  
**Code**: [https://github.com/EricZhang1412/Spatial-temporal-ERF](https://github.com/EricZhang1412/Spatial-temporal-ERF)  
**Area**: Image Segmentation
**Keywords**: spiking neural networks, effective receptive field, Transformer, channel mixer, visual long-sequence modeling

## TL;DR

This paper proposes a Spatial-Temporal Effective Receptive Field (ST-ERF) analysis framework to diagnose the bottleneck of Transformer-based SNNs in visual long-sequence modeling—namely, the lack of a global receptive field—and accordingly designs two channel mixers, MLPixer and SRB, to enhance the global modeling capability of SNNs.

## Background & Motivation

Spiking Neural Networks (SNNs), owing to their event-driven nature, hold strong potential for energy efficiency and have demonstrated progress on tasks such as image classification. However, SNNs still lag far behind ANNs on **visual long-sequence modeling tasks** such as object detection and semantic segmentation.

**Limitations of Prior Work**:

**SNNs perform poorly on dense prediction tasks**: These tasks require spatially dense outputs over entire images and demand the ability to model long-range spatial dependencies.

**Transformer-based SNNs fail to fully exploit global modeling potential**: Although self-attention mechanisms from Transformers have been incorporated, existing designs (e.g., Spike-driven Transformer) still rely heavily on convolutional operations as channel mixers, introducing a locality bias.

**Lack of analytical tools**: Conventional ERF frameworks consider only the spatial dimension and cannot characterize the inherent spatial-temporal dynamics of SNNs.

**Key Challenge**: Transformer-based SNNs should theoretically possess a global receptive field, yet in practice, the locality bias introduced by convolutional channel mixers prevents the establishment of effective global receptive fields in early stages, limiting long-sequence modeling performance.

**Key Insight**: (1) Extend ERF to the temporal dimension by proposing the ST-ERF analysis framework to quantitatively diagnose the problem; (2) Based on the diagnostic findings, replace convolutions with MLPs to design new channel mixers that eliminate locality bias.

## Method

### Overall Architecture

1. **Analysis phase**: Apply the ST-ERF framework to analyze receptive field behavior in existing Transformer-based SNNs (Spikformer, SDT-V1, Meta-SDT, etc.).
2. **Design phase**: Propose two channel mixers, MLPixer and SRB, to replace the convolutional channel mixers in the first two stages of Meta-SDT.
3. **Validation phase**: Validate on COCO 2017 object detection and ADE20K semantic segmentation.

### Key Designs

1. **ST-ERF Theoretical Framework**:

    - **Function**: Quantifies the contribution of each input feature at different spatial-temporal positions to the output within an SNN.
    - **Core Definition**: $\text{ERF}^{(\mathcal{S},\mathcal{T})}_{(i,j)}[y_{(m,n)}[t], \tau; \mathbf{x}] = \frac{\partial y_{(m,n)}[t]}{\partial x_{(i,j)}[t-\tau]}$
    - The spatial ERF is a weighted average of the ST-ERF over all time steps; the temporal ERF is an integral of the ST-ERF over the spatial dimension.
    - **Loss-Derived Computation**: Leverages PyTorch automatic differentiation and efficiently computes the ERF by setting specific gradient stimuli. Concretely, the gradients at all channels and time steps of the center position are set to 1, and back-propagation yields the spatial ERF.
    - **Design Motivation**: Conventional ERF frameworks cannot handle the temporal dynamics of SNNs; ST-ERF incorporates the temporal dimension into the analysis.

2. **MLPixer (MLP-based Mixer)**:

    - **Function**: Replaces convolutional channel mixers entirely with MLPs.
    - **Core Design**: $\text{MLPixer}(\mathbf{X}) = \text{BN}(\text{MLP}(\mathbb{SN}\{\text{BN}(\text{MLP}\{\mathbb{SN}(\mathbf{X})\})\}))$
    - A stack of two MLP layers, batch normalization, and spiking neurons.
    - **Design Motivation**: MLPs operate point-wise and introduce no spatial locality bias, ensuring that channel mixing does not disrupt global spatial features. ST-ERF visualizations confirm that MLPixer achieves a substantially broader global receptive field.

3. **SRB (Splash-and-Reconstruct Block)**:

    - **Function**: A compromise design—retains a first convolutional layer for local feature extraction while using MLP in the second layer.
    - **Core Design**: $\text{SRB}(\mathbf{X}) = \text{BN}(\text{MLP}(\mathbb{SN}\{\text{BN}(\text{Conv}\{\mathbb{SN}(\mathbf{X})\})\}))$
    - The first layer uses a 1×1 convolution; the second layer uses an MLP.
    - **Design Motivation**: Maintains performance while reducing parameter count. SRB achieves an optimal balance between accuracy and model size.

### Loss & Training

- Integrated into the Meta-SDT architecture by replacing the channel mixers in the first two stages; the latter two stages retain the Transformer-SNN blocks unchanged.
- A membrane-potential shortcut residual connection mechanism is adopted to preserve the spike-driven nature of the network.
- Object detection uses Mask R-CNN with a 1× training schedule; semantic segmentation uses Semantic FPN with 160k iterations.
- All backbone networks are pre-trained on ImageNet-1K.

## Key Experimental Results

### Main Results

**COCO 2017 Object Detection (Mask R-CNN, 1× schedule)**:

| Architecture | Params | AP^b | AP^b_50 | AP^m | AP^m_50 |
|---|---|---|---|---|---|
| SDTv3-T | 25M | 15.2 | 35.5 | 15.2 | 33.0 |
| SDTv3-T + SRB(ε4) | 25M | 18.2 | 39.2 | 17.5 | 34.8 |
| SDTv3-B | 39M | 21.7 | 46.9 | 20.1 | 41.8 |
| SDTv3-B + SRB(ε4) | 37M | **25.8** | **48.9** | **22.5** | **43.9** |

The SRB variant improves AP^b_50 by 4.26% on Base (46.9→48.9) while reducing parameter count by 2M. The gain is even larger on Tiny (10.42%).

**ADE20K Semantic Segmentation (Semantic FPN, 160k iter)**:

| Architecture | Channel Mixer | Params | mIoU |
|---|---|---|---|
| SDTv3-T | Conv(ε4) | 6.5M | 34.9 |
| SDTv3-T + SRB(ε4) | SRB | 6.2M (↓0.3) | **38.2** (↑3.3) |
| SDTv3-B | Conv(ε4) | 20.4M | 41.1 |
| SDTv3-B + SRB(ε4) | SRB | 19.2M (↓1.2) | **43.7** (↑2.6) |

SRB substantially improves mIoU while simultaneously reducing parameter count.

### Ablation Study

| Mixer Type | Param Change | mIoU Change | Notes |
|---|---|---|---|
| MLPixer(ε4) | ↓0.6M | +0.0 | Fewest parameters but limited gain on Tiny |
| MLPixer(ε6) | +0.1M | +1.0 | Larger expansion ratio is effective |
| SRB(ε4) | ↓0.3M | **+3.3** | Best accuracy–parameter balance |

**Event Tracking (FE108 & VisEvent)**:

| Architecture | FE108 AUC | VisEvent AUC |
|---|---|---|
| SD-Track (Tiny) | 56.7% | 35.4% |
| + MLPixer(ε6) | **57.9%** | 34.5% |
| + SRB(ε4) | 58.2% | 33.8% |

Performance improves on event tracking as well, though a slight degradation is observed on VisEvent.

### Key Findings

1. **ERF issue in SDT-V1**: Multiple convolutional layers in the SPS module cause the receptive field to be overly concentrated at the center, limiting spatial coverage.
2. **ERF issue in Meta-SDT**: The introduction of RepConv enhances local feature extraction but constrains long-range aggregation.
3. **MLPixer establishes a global ERF as early as Stage 1**, which gradually contracts to specific regions as network depth increases.
4. **SRB begins forming a global ERF at Stage 2**, exhibiting slightly different behavior.
5. Reducing convolutional usage indeed enables SNNs to attain a more global receptive field.

## Highlights & Insights

1. **ST-ERF analysis framework**: Fills a theoretical gap in receptive field analysis for SNNs and provides a quantitative tool for SNN architecture optimization.
2. **Diagnosis-driven design**: The ST-ERF framework first identifies the problem (lack of global receptive field), and the solution is then designed in a targeted manner.
3. **Simple yet effective improvement**: Replacing channel mixers in only the first two stages yields significant performance gains with minimal architectural modifications.
4. **Performance gain with fewer parameters**: SRB improves performance while reducing parameter count, demonstrating that convolutions at these positions are redundant.

## Limitations & Future Work

- Although MLPixer achieves a stronger global ERF, it underperforms SRB on certain tasks, suggesting that completely removing local feature extraction is not always optimal.
- Validation is limited to the Meta-SDT architecture; applicability to other SNN architectures remains to be examined.
- Performance degrades on the VisEvent dataset for event tracking, indicating that a global receptive field is not universally beneficial across all tasks.
- ST-ERF analysis is currently used primarily for visualization and qualitative assessment; quantitative metrics to guide architecture search are lacking.

## Related Work & Insights

- **ERF theory (Luo et al., 2016)**: This work extends the framework to the temporal dimension, constituting a clear methodological contribution.
- **MLP-Mixer (Tolstikhin et al., 2021)**: Inspires the idea of replacing convolutions with MLPs.
- **Meta-SDT (Yao et al., 2025)**: Serves as the baseline architecture; its bottlenecks are analyzed and subsequently addressed.
- **Insight**: Receptive field analysis provides strong guidance for neural architecture design and can be generalized to a broader range of SNN tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The ST-ERF framework is a valuable theoretical contribution; the architectural improvements are relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers detection, segmentation, and event tracking, with rich visualization analyses.
- Writing Quality: ⭐⭐⭐⭐ The logical chain from analysis to design is clearly articulated.
- Value: ⭐⭐⭐⭐ Provides important guidance for SNN architecture design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking](step_a_unified_spiking_transformer_evaluation_platform_for_fair_and_reproducible.md)
- [\[NeurIPS 2025\] Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [\[ICCV 2025\] PartField: Learning 3D Feature Fields for Part Segmentation and Beyond](../../ICCV2025/segmentation/partfield_learning_3d_feature_fields_for_part_segmentation_and_beyond.md)
- [\[ICCV 2025\] Temporal Rate Reduction Clustering for Human Motion Segmentation](../../ICCV2025/segmentation/temporal_rate_reduction_clustering_for_human_motion_segmentation.md)
- [\[ICCV 2025\] Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation](../../ICCV2025/segmentation/skeleton_motion_words_for_unsupervised_skeleton-based_temporal_action_segmentati.md)

</div>

<!-- RELATED:END -->
