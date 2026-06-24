---
title: >-
  [Paper Note] UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection
description: >-
  [CVPR 2025][Object Detection][Visual Anomaly Detection] This paper proposes UniVAD, a training-free unified few-shot visual anomaly detection method. Through the Contextual Component Clustering (C3) module, it achieves precise component segmentation. Combined with component-aware patch matching and graph-enhanced component modeling, it achieves state-of-the-art anomaly detection across industrial, logical, and medical domains using only a few normal samples.
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Visual Anomaly Detection"
  - "Training-free"
  - "Few-shot"
  - "Cross-domain Unified Model"
  - "Component Segmentation"
date: 2026-05-08
content_hash: e6f9f3616c6fb55b
---

# UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection

**Conference**: CVPR 2025  
**arXiv**: [2412.03342](https://arxiv.org/abs/2412.03342)  
**Code**: [https://github.com/FantasticGNU/UniVAD](https://github.com/FantasticGNU/UniVAD)  
**Area**: Medical Images  
**Keywords**: Visual Anomaly Detection, Training-free, Few-shot, Cross-domain Unified Model, Component Segmentation

## TL;DR

This paper proposes UniVAD, a training-free unified few-shot visual anomaly detection method. Through the Contextual Component Clustering (C3) module, it achieves precise component segmentation. Combined with component-aware patch matching and graph-enhanced component modeling, it achieves state-of-the-art anomaly detection across industrial, logical, and medical domains using only a few normal samples.

## Background & Motivation

**Background**: Visual Anomaly Detection (VAD) aims to identify anomalous samples that deviate from normal patterns, applicable to three major domains: industrial defect detection, logical anomaly detection, and medical anomaly detection. Existing methods such as PatchCore perform excellently in industrial scenarios, but require specialized model architectures and detection algorithms designed for different domains.

**Limitations of Prior Work**: (1) Existing VAD methods are highly domain-specific—PatchCore achieves an 84.0% 1-shot AUC on MVTec-AD, but drops sharply to 62.0% on the logical anomaly dataset MVTec LOCO; (2) Even within the same domain, most methods adopt a "one-category-one-model" paradigm, training an independent model for each category, leading to poor generalization ability; (3) Component segmentation methods face difficulty in granularity control under few-shot scenarios, where Segment Anything Model (SAM) produces segmentations that are either too fine-grained or too coarse-grained.

**Key Challenge**: Anomaly types differ immensely across different domains—industrial anomalies are local defects, logical anomalies are incorrect combinations of components, and medical anomalies are pathological regions. Using a unified method to detect these anomalies at different semantic levels poses a fundamental challenge.

**Goal**: Build a training-free, cross-domain, unified few-shot anomaly detection model that requires only a few normal reference samples during testing to detect anomalies in industrial, logical, and medical domains.

**Key Insight**: The authors observe that all anomalies can be classified into two categories: structural anomalies (patch-level feature deviations) and logical anomalies (component-level relationship deviations), which can be handled separately by two complementary modules and then aggregated.

**Core Idea**: Achieve precise component segmentation through visual foundation models (SAM, RAM) and clustering, perform patch matching within components to detect structural anomalies, model graph structures between components to detect logical anomalies, and fuse both results to achieve unified detection.

## Method

### Overall Architecture

The input to UniVAD consists of a query image and $K$ normal reference images. First, the C3 module is utilized to perform component segmentation on all images to obtain component masks, followed by the extraction of patch-level and component-level features. The patch features are fed into the CAPM module to detect structural anomalies, while the component features are input to the GECM module to detect logical anomalies. Finally, the two types of anomaly scores are weight-fused to obtain the unified anomaly detection result.

### Key Designs

1. **Contextual Component Clustering (C3)**:

    - **Function**: Achieve precise component segmentation under few-shot conditions.
    - **Mechanism**: First, use the Recognize Anything Model (RAM) to identify object tags in the image, then apply Grounded SAM to generate initial masks. If there is only one mask and its coverage exceeds $\gamma\%$, it is treated as a texture surface, and the entire image mask is output. If there are multiple masks, K-means clustering is applied to the features of normal images to obtain $N$ cluster centers, generating a clustering mask $M_{\text{cluster}}$ and filtering the background to obtain $N'$ valid masks. Finally, IoU is used to map the fine-grained masks from SAM to the clustering masks, merging the SAM masks belonging to the same cluster label as the final output.
    - **Design Motivation**: Using SAM alone suffers from inconsistent granularity (either too fine or too coarse), while clustering provides the correct semantic granularity. However, clustering alone requires a large number of samples under the few-shot setting. The combination of both leverages the precise boundaries from SAM while controlling the segmentation granularity through clustering.

2. **Component-Aware Patch Matching (CAPM)**:

    - **Function**: Detect structural anomalies (local defects, texture variations, pathological regions).
    - **Mechanism**: Use pretrained CLIP and DINOv2 encoders to extract patch features $P_q$ and $P_n$. On top of standard patch matching (calculating the minimum cosine distance from a query patch to all normal patches), two improvements are introduced: (a) Component-aware matching—grouping patches using the component masks from C3 and matching only within the same component $Score_{\text{aware}}(P_{qi}^j) = \min(\text{distance}(P_{qi}^j, P_{ni}))$, thereby avoiding false matches across components; (b) Vision-language matching—encoding "normal" and "anomalous" descriptions with the CLIP text encoder to calculate the patch-text similarity $Score_{\text{vl}}$. The three scores are equally weighted and combined to produce the structural anomaly map.
    - **Design Motivation**: Standard patch matching cannot distinguish between foreground/background, nor can it differentiate between different components—different component regions with similar colors are prone to mismatching, leading to missed detections. The component constraint restricts matching to semantically identical regions, significantly reducing false matches.

3. **Graph-Enhanced Component Modeling (GECM)**:

    - **Function**: Detect logical anomalies (missing, redundant, or misplaced components).
    - **Mechanism**: Construct a graph on component-level features—each component acts as a node, with cosine similarities between components serving as edge weights to build the adjacency matrix $A$. Contextual information is aggregated through graph attention operations to obtain the enhanced component embedding $E_q = G(A_q, F_{qc})$. Then, the deep anomaly score $Score_{\text{deep}}(E_q^i) = \min(\text{distance}(E_q^i, E_n))$ is computed. Concurrently, geometric features (area, color, position) are extracted to compute the geometric anomaly score $Score_{\text{geo}}$. The two are weight-fused to obtain the logical anomaly score.
    - **Design Motivation**: Patch matching cannot detect logical anomalies where the "components themselves are correct but their combination is wrong". Graph modeling can capture the relationship patterns between components, thereby identifying the addition, removal, or displacement of components.

### Loss & Training

UniVAD is a training-free method requiring no loss functions or training process. Frozen CLIP-L/14@336px and DINOv2-G/14 are utilized as feature extractors. All hyperparameters ($\alpha, \beta, \gamma$ as 1/3 each, $\phi, \psi$ as 0.5 each, $\delta, \eta$ as 0.5 each) are set uniformly across all datasets. Images are consistently resized to 448×448.

## Key Experimental Results

### Main Results

Cross-domain anomaly detection under the 1-shot setting (Image-level AUC):

| Dataset | PatchCore | AnomalyGPT | WinCLIP | UniVAD | Gain |
|---|---|---|---|---|---|
| MVTec-AD | 84.0 | 94.1 | 93.1 | **97.8** | +3.7 |
| VisA | 74.8 | 87.4 | 83.8 | **93.5** | +6.1 |
| MVTec LOCO | 62.0 | 60.4 | 58.0 | **71.0** | +8.8 |
| BrainMRI | 73.2 | 73.1 | 55.4 | **80.2** | +7.0 |
| LiverCT | 44.9 | 60.3 | 60.3 | **70.0** | +9.7 |
| RESC | 56.3 | 82.4 | 72.9 | **85.5** | +3.1 |

Comparison with domain-specific medical methods under the 4-abnormal-shot setting:

| Dataset | DRA | BGAD | MVFA | UniVAD |
|---|---|---|---|---|
| BrainMRI | 80.6 | 83.6 | 92.4 | **94.1** |
| LiverCT | 59.6 | 72.5 | 81.2 | **87.5** |
| RESC | 90.9 | 86.2 | 96.2 | **97.3** |

### Ablation Study

Comparison of different implementations of the C3 module (Image AUC, Pixel AUC):

| Configuration | MVTec-AD | VisA | MVTec LOCO | BrainMRI |
|---|---|---|---|---|
| Clustering Only | (97.3, 96.1) | (92.5, 98.0) | (67.5, 70.9) | (73.9, 96.7) |
| Grounded-SAM Only | (97.5, 96.1) | (92.1, 97.7) | (67.8, 74.9) | (74.5, 94.9) |
| **C3 (Clustering+SAM)** | **(97.8, 96.5)** | **(93.5, 98.0)** | **(71.0, 75.1)** | **(80.2, 96.8)** |

### Key Findings

- UniVAD outperforms domain-specific methods across all 9 datasets, with significant improvements particularly in logical anomalies (MVTec LOCO +8.8%) and medical anomalies (LiverCT +9.7%).
- The combination of clustering and SAM in the C3 module is substantially superior to using either method individually, especially on BrainMRI (+6.3%/+1.9%) and MVTec LOCO (+3.5%/+0.2%).
- As a training-free method, UniVAD even outperforms the training-required MVFA under the 4-abnormal-shot setting.
- Graph-enhanced component modeling contributes the most to logical anomaly detection, while component-aware patch matching contributes the most to industrial and medical anomalies.

## Highlights & Insights

1. **"Structural + Logical" Dual-path Detection Paradigm**: Decomposition of anomaly detection into two semantic levels and handling each with its most suitable method elegantly unifies anomaly types across different domains.
2. **Training-free Cross-domain Generalization**: No training is required on the target domain with hyperparameters set uniformly, proving that the pretrained features of visual foundation models are sufficiently rich.
3. **"Coarse + Fine" Combination Strategy of C3 Module**: Clustering controls granularity while SAM ensures boundary precision. This complementary design concept is transferable to other tasks requiring hierarchical segmentation.

## Limitations & Future Work

- Reliance on multiple large foundation models (RAM, SAM, CLIP, DINOv2) results in high inference computational cost.
- For simple objects with very few components (such as screws), the graph modeling of the GECM module might not be fully effective.
- Although hyperparameters are unified, they may not represent the optimal solution for every scenario.
- Adaptive weight fusion could be introduced in the future to replace static score weighting.

## Related Work & Insights

- **vs PatchCore**: PatchCore is a pure patch-matching method and cannot handle logical anomalies. UniVAD introduces component-level modeling, covering a higher semantic level.
- **vs ComAD**: ComAD is also component-based but requires a large number of samples for clustering-based segmentation. UniVAD achieves precise few-shot segmentation via SAM.
- **vs AnomalyGPT**: AnomalyGPT leverages LLM reasoning but performs poorly on logical anomalies (60.4%). The graph modeling in UniVAD is better suited to capturing component relationships.
- The strategy of training-free, modular design can serve as a blueprint for building generalized anomaly detection systems.

## Rating

- Novelty: 7/10 — While individual modules are not entirely novel, the design of the unified framework and the positioning of training-free cross-domain capability are highly valuable.
- Experimental Thoroughness: 9/10 — Comprehensive coverage of nine datasets across three domains, under both 1-shot and 4-shot configurations, with thorough ablation studies.
- Writing Quality: 8/10 — Clear structure, intuitive figures, and detailed pseudocode.
- Value: 8/10 — The first training-free unified cross-domain VAD method, making important contributions to the standardization of the anomaly detection field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling](../../CVPR2026/object_detection/subspacead_training-free_few-shot_anomaly_detection_via_subspace_modeling.md)
- [\[CVPR 2025\] Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval](search_and_detect_training-free_long_tail_object_detection_via_web-image_retriev.md)
- [\[CVPR 2025\] Unseen Visual Anomaly Generation](unseen_visual_anomaly_generation.md)
- [\[CVPR 2025\] TailedCore: Few-Shot Sampling for Unsupervised Long-Tail Noisy Anomaly Detection](tailedcore_few-shot_sampling_for_unsupervised_long-tail_noisy_anomaly_detection.md)
- [\[CVPR 2026\] Anomaly as Non-Conformity via Training-Free Graph Laplacian Energy Minimization](../../CVPR2026/object_detection/anomaly_as_non-conformity_via_training-free_graph_laplacian_energy_minimization.md)

</div>

<!-- RELATED:END -->
