---
title: >-
  [Paper Note] XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge
description: >-
  [ICML2026][LLM Pretraining][human sensing] XTransfer is designed for human sensing tasks on edge devices. It transfers pre-trained models from any modality (image, text, audio…
tags:
  - "ICML2026"
  - "LLM Pretraining"
  - "human sensing"
  - "few-shot transfer"
  - "cross-modality"
  - "edge deployment"
  - "layer recombining"
date: 2026-05-08
content_hash: 847635322bcbeb1d
---

# XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge

**Conference**: ICML2026  
**arXiv**: [2506.22726](https://arxiv.org/abs/2506.22726)  
**Code**: https://github.com/zhangy10/XTransfer  
**Area**: Human Understanding / Human Sensing / Edge Intelligence  
**Keywords**: human sensing, few-shot transfer, cross-modality, edge deployment, layer recombining  

## TL;DR
XTransfer is designed for human sensing tasks on edge devices. It transfers pre-trained models from any modality (image, text, audio, or sensors) using a small amount of target sensor data. It mitigates cross-modality feature misalignment while improving few-shot accuracy and edge deployment efficiency through layer-wise model repairing and resource-constrained layer recombining.

## Background & Motivation
**Background**: Human sensing tasks include activity recognition, gestures, emotions, vital signs, and sensor applications such as millimeter-wave/ultrasound. Deep learning significantly enhances recognition performance, but training and deployment on edge devices are constrained by data, computing power, privacy, and collection costs. Few-shot learning (FSL), transfer learning, and cross-domain FSL attempt to adapt existing models with limited data.

**Limitations of Prior Work**: Human sensing data differs from images/text, often characterized by low signal-to-noise ratios, individual differences, scenario variations, hardware disparities, and privacy restrictions. Many FSL methods still require large-scale source datasets of the same modality or target-modality foundation models. Multimodal/cross-modal learning often relies on paired data, shared semantic spaces, or large-scale unlabeled data. For new sensor tasks, these conditions are frequently unmet.

**Key Challenge**: While many pre-trained models are publicly available, a severe modality shift exists between source modalities and target sensor modalities. Direct fine-tuning is prone to overfitting, while direct pruning/structuring damages accuracy under few-shot cross-modality conditions. The problem to solve is "how to reuse pre-trained models from any source while using only few-shot sensor data and satisfying edge resource constraints."

**Goal**: XTransfer aims to achieve modality-agnostic model transfer: it does not rely on paired data, does not require a shared semantic space, and does not train a target model from scratch. Instead, it repairs the source model at the layer-wise latent feature distribution level and recombines useful layers to form a model suitable for edge deployment.

**Key Insight**: The authors found that the impact of modality shift is non-uniform across layers. Mean Magnitude of Channels (MMC) shifts in specific layers disrupt layer-wise accuracy convergence. In other words, the problem is not that the "entire model is unusable," but that certain intermediate representations are misaligned with the target sensor distribution, requiring layer-by-layer diagnosis, repair, and selection.

**Core Idea**: Cross-modality transfer is decomposed into two steps: first, repairing the feature distribution of each layer in an anchor PCA space using the SRR pipeline; second, selecting and recombining the most valuable and resource-efficient layers from candidate layers of multiple source models using LWS control.

## Method
The overall workflow of XTransfer includes model repairing and layer recombining. The former addresses "which layer representation is misaligned and how to align it" when target sensor inputs enter the source model. The latter addresses "how to select layers based on accuracy and edge resources," as not all repaired layers are worth retaining.

### Overall Architecture
Given one or more pre-trained source models, XTransfer first decomposes them into L-units (layers or layer blocks that can be searched independently). For each target human sensing task, a few labeled sensor samples are spliced before and after source model layers via a connector to ensure compatibility in input and feature shapes. Subsequently, the SRR pipeline repairs candidate layers: it uses anchor source distributions to guide MMC alignment of target sensor features and optimizes the connector in a PCA-reduced orthogonal space.

The layer recombining stage is managed by LWS control. It places repaired layers from different source models and depths into search pools, evaluating S-score, resource overhead, and whether inter-layer dependencies are improved. Selected layers of interest are incrementally recombined into the output model; useless or resource-expensive candidates are skipped.

### Key Designs
1.  **SRR pipeline: Splice-Repair-Removal**:
    -   **Function**: Repairs layer-level feature misalignment under few-shot target sensor data and reduces unnecessary channels.
    -   **Mechanism**: *Splice* uses a lightweight trainable connector to handle shape incompatibilities; *Repair* aligns target sensing MMC distributions with source anchor MMC in an anchor PCA space; *Removal* estimates channel importance based on PCA component weights, deleting unnecessary channels while maintaining the S-score.
    -   **Design Motivation**: Simple reshaping may lose sensor features, while direct fine-tuning of the entire layer causes overfitting. Restricting repair to the connector and low-dimensional anchor space allows stable alignment with very few target samples.

2.  **Anchor-based generative transfer repair**:
    -   **Function**: Brings the latent feature distribution of target sensor samples closer to the originally discriminative clusters of the source model.
    -   **Mechanism**: Uses source model MMC as an anchor, projected into a low-dimensional orthogonal space via PCA. A Hungarian algorithm then performs one-to-one pairing between source and target sensing classes to minimize centroid distances. The repair loss pulls paired source/target centroids closer while using an anchor-based margin to push different target classes apart.
    -   **Design Motivation**: Modality-agnostic transfer lacks shared semantic spaces or paired data. MMC is an activation-based statistic dependent only on layer output, making it an ideal modality-agnostic alignment signal.

3.  **Layer-wise Search control for edge-aware recombining**:
    -   **Function**: Selects a combination of layers with high accuracy, low resource cost, and stable inter-layer dependencies from single or multi-source candidates.
    -   **Mechanism**: LWS defines four actions—*init*, *continue*, *skip*, and *cross*. It measures repaired layer value in each search pool using the S-score divided by a resource coefficient to get a resource-constrained value. To avoid performing SRR on all candidates, a pre-search check uses a repair rate growth model to estimate which layers are worth repairing; a dynamic search range adjusts the search scope to prevent premature pruning.
    -   **Design Motivation**: Multi-source search spaces are large, and edge devices have FLOPs/Params constraints. Repairing and retaining only valuable layers improves both accuracy and the accuracy-to-resource ratio.

### Loss & Training
The core training objective is the SRR anchor-based repair loss. For each source-target class pair, it minimizes the distance between projected MMC centroids and applies a margin-based negative loss for different target classes. In the LWS stage, the goal is not a traditional gradient loss but selecting candidates based on $VR_{ij} = V_{ij} / R(n)_{ij}$, where $V_{ij}$ is the S-score and $R(n)_{ij}$ integrates FLOPs, parameter count, and the resource coefficient of the current layer position. The entire training uses only few-shot target labeled data, requiring no additional unlabeled or paired cross-modal data.

## Key Experimental Results

### Main Results
Experiments cover image/text/audio/sensing source datasets and target tasks such as HHAR, WESAD, Gesture, Writing, Emotion, and ChestX. Evaluation metrics include accuracy and ATR (Accuracy-to-Resource ratio), where $ATR = Accuracy / \text{normalized resource cost}$. The table below extracts 5-shot accuracy from Table 4.

| Target Task (5-shot) | ProtoNet | DAPN | MAML | SemiCMT | MetaSense oracle | Our-Single | Our-Multi |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| HHAR | 45.7 | 51.2 | 42.3 | 38.9 | 69.0 | 71.8 | 74.3 |
| WESAD | 49.3 | 61.2 | 57.9 | 40.0 | 64.0 | 78.4 | 77.8 |
| Gesture | 55.8 | 49.4 | 41.3 | 34.4 | 73.4 | 69.6 | 73.1 |
| Writing | 78.7 | 78.6 | 39.7 | 38.6 | 83.3 | 87.0 | 86.1 |
| Emotion | 49.2 | 50.0 | 26.9 | 33.6 | 56.3 | 55.6 | 55.1 |
| ChestX | 23.4 | 24.4 | 20.4 | 25.6 | 28.1 | 28.6 | 30.0 |

### Ablation Study
The paper analyzes SRR, channel removal, LWS, and efficient search separately.

| Analysis Item | Setting | Results | Conclusion |
| :--- | :--- | :--- | :--- |
| Repair loss | HHAR + ResNet18, 5-shot | Average S-score: Repair loss $\approx 0.20$, N-Pair $\approx 0.05$, Triplet $\approx 0.07$ | Anchor-based repair better recovers layer discriminability. |
| Efficient search | Multi-Pre vs Multi-Efficient | Multi-Pre is fastest but accuracy is $3.36\%/7.94\%$ lower; Multi-Efficient reduces search time by $2.1\times - 4\times$. | Dynamic range is more balanced than top-1 pre-search. |
| Source modality | Various sources on HHAR | Avg acc (3-10 shot): Image $70.5\%$, Text $69.6\%$, Audio $64.3\%$, Sensing $67.0\%$ | Source quality and semantic relevance matter; image is generally more stable. |
| SRR only | SRR vs SRR-w/o-Removal | SRR outperforms w/o Removal on most targets; slightly unstable on HHAR/Writing. | Channel removal helps, but works best when linked with LWS. |
| ATR | Our variants vs baselines | ATR of Our-Single/Multi is $1.6-29\times$ and $16.6-98\times$ higher than baselines. | LWS significantly contributes to edge efficiency. |

### Key Findings
- SRR alone improves average accuracy on HHAR, WESAD, and Gesture, indicating that layer-wise MMC repair mitigates modality shift.
- LWS is the key to turning "useful repair" into "deployable utility." Without LWS, connectors introduce resource costs that might lower ATR compared to baselines; LWS improves both accuracy and ATR.
- Multi-source models do not just stack more parameters. Dynamic search allows Multi-Efficient to increase search time by only $\sim 2.1\times$ while the search space increases $5\times$, yielding more stable results.
- Source model selection remains important. Image sources are strongest on average (3-10 shot), but Text/Sensing sources are sometimes more stable at ultra-low (2-3 shot) samples.

## Highlights & Insights
- The paper locates cross-modality transfer failure at layer-wise latent distribution misalignment rather than a generic domain gap. This provides actionable layer-level metrics for repair, search, and selection.
- The MMC + PCA anchor space is a practical design: it requires no semantic alignment or paired data and is more stable than high-dimensional activation distributions in few-shot sensor settings.
- LWS performs both model transfer and architecture search, fitting edge scenarios perfectly. While many methods chase only accuracy, XTransfer explicitly incorporates FLOPs/Params into its selection objectives.

## Limitations & Future Work
- The method relies on the availability of suitable source models. Automating the selection of the optimal source model for a specific target sensing task remains a future direction.
- The S-score/MMC serves as a proxy for layer value; it is correlated with but not equivalent to final accuracy. It may require adaptation for complex tasks like regression-based vital sign estimation.
- The SRR and LWS workflow is complex, involving connectors, PCA, Hungarian pairing, adversarial/generative repair, and search range adjustments. Implementation may incur high engineering maintenance costs.
- While covering various human sensing tasks, the study focuses on classification in few-shot settings. Continuous estimation, long-term user drift, and online updates require further validation.

## Related Work & Insights
- **vs FSL / cross-domain FSL**: ProtoNet, MAML, and DAPN usually operate within the same or similar domains. XTransfer emphasizes few-shot transfer from arbitrary source modalities to sensor modalities.
- **vs multimodal / cross-modal learning**: CLIP-style or image-to-sensor distillation often requires shared semantic spaces or paired data. XTransfer bypasses these using activation statistics and few labels.
- **vs pruning / NAS for edge**: Traditional pruning assumes sufficient target data or small modality shifts. XTransfer repairs before recombining, avoiding the compression of a negatively transferred model.
- **Insight**: For edge human sensing, the transfer source doesn't have to be a similar sensor model; image/text/audio models can serve as low-cost sources if latent distributions are repaired layer-by-layer.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Combining layer-wise repair with edge-aware recombining for modality-agnostic transfer is a unique direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage of datasets, source modalities, baselines, and ablation studies, including real-world private data.
- **Writing Quality**: ⭐⭐⭐⭐☆ Motivation is clear despite system complexity; the many components might benefit from more centralized pseudocode.
- **Value**: ⭐⭐⭐⭐⭐ High potential for edge human sensing deployment, especially where data is expensive to collect but public pre-trained models are abundant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](dropout_universality_scaling_laws_and_optimal_scheduling_at_the_edge-of-chaos.md)
- [\[CVPR 2026\] Linking Modality Isolation in Heterogeneous Collaborative Perception](../../CVPR2026/llm_pretraining/linking_modality_isolation_in_heterogeneous_collaborative_perception.md)
- [\[CVPR 2026\] FlowMotion: Training-Free Flow Guidance for Video Motion Transfer](../../CVPR2026/llm_pretraining/flowmotion_training-free_flow_guidance_for_video_motion_transfer.md)
- [\[NeurIPS 2025\] ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data](../../NeurIPS2025/llm_pretraining/zeus_zero-shot_embeddings_for_unsupervised_separation_of_tabular_data.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)

</div>

<!-- RELATED:END -->
