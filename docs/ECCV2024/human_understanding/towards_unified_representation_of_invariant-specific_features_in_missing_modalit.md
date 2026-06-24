---
title: >-
  [Paper Note] Towards Unified Representation of Invariant-Specific Features in Missing Modality Face Anti-Spoofing
description: >-
  [ECCV 2024][Human Understanding][Missing Modality] This paper proposes the MMA-FAS framework to address the problem of missing modalities in multimodal face anti-spoofing (FAS). It separates modality-invariant and modality-specific features from a frequency decomposition perspective using modality-disentangle adapters. Combined with an LBP-guided contrastive loss and an adaptive modal combination sampling strategy, it achieves SOTA performance across all missing modality scen…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Missing Modality"
  - "Face Anti-Spoofing"
  - "Modality Disentanglement"
  - "Contrastive Learning"
  - "Vision Transformer"
date: 2026-05-08
content_hash: 9160d73e1f597406
---

# Towards Unified Representation of Invariant-Specific Features in Missing Modality Face Anti-Spoofing

**Conference**: ECCV 2024  
**Paper Link**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3248_ECCV_2024_paper.php)  
**Code**: None  
**Area**: Face Understanding / Multimodal Learning / Face Anti-Spoofing  
**Keywords**: Missing Modality, Face Anti-Spoofing, Modality Disentanglement, Contrastive Learning, Vision Transformer

## TL;DR

This paper proposes the MMA-FAS framework to address the problem of missing modalities in multimodal face anti-spoofing (FAS). It separates modality-invariant and modality-specific features from a frequency decomposition perspective using modality-disentangle adapters. Combined with an LBP-guided contrastive loss and an adaptive modal combination sampling strategy, it achieves SOTA performance across all missing modality scenarios.

## Background & Motivation

**Background**: Multimodal face anti-spoofing (FAS) utilizes information from multiple modalities such as RGB, Depth, and Infrared (IR) to distinguish between live faces and spoof attacks, as different modalities provide complementary clues (e.g., depth maps effectively identify planar attacks, while IR distinguishes temperature differences). When all modalities are available, the performance of multimodal FAS significantly outperforms single-modality methods.

**Limitations of Prior Work**: In practical deployments, sensors for certain modalities may be unavailable (e.g., devices not equipped with a depth camera) or temporarily fail (e.g., IR sensor malfunction), leading to missing modalities. Under such conditions, the performance of Vision Transformer-based multimodal FAS methods drops drastically because these models assume complete modalities during training and cannot handle missing cases elegantly. Existing approaches addressing missing modalities primarily rely on modality-invariant features—general features shared across modalities—to mitigate the missing issue, completely disregarding the value of modality-specific features.

**Key Challenge**: Although modality-invariant features remain available during modality omission, they only capture information shared across modalities, losing the distinct discriminative clues of each modality (e.g., texture details in RGB, geometric information in depth maps, and thermal distribution in IR). On the other hand, modality-specific features cannot be acquired when their corresponding modalities are missing. How to balance leveraging the discriminative power of specific features and coping with their unavailability is the key challenge.

**Goal**: (1) How to explicitly separate modality-invariant and modality-specific features? (2) How to still effectively utilize the specific features of available modalities when some modalities are missing? (3) How to balance the training process across different modality combination scenarios?

**Key Insight**: The authors propose to disentangle modality-invariant and modality-specific features from the perspective of frequency decomposition—shared information across different modalities is mainly reflected in low-frequency structures (e.g., face contours), while modality-specific information is mainly represented in high-frequency details (e.g., texture, noise patterns). Based on this observation, lightweight adapters are designed for feature decomposition. Meanwhile, Local Binary Pattern (LBP) is used as a texture prior to guide contrastive learning, strengthening feature discriminability.

**Core Idea**: To achieve robust FAS under missing modalities by explicitly separating modality-invariant/specific features via frequency-decomposition adapters, combined with an LBP-guided contrastive loss and an adaptive sampling strategy.

## Method

### Overall Architecture

MMA-FAS is built upon a pre-trained ViT. The input comprises multimodal face images (any combination of RGB, Depth, and IR), and the output is the classification result (live/spoof). The core architecture includes: (1) a ViT backbone to extract basic features; (2) Modality-Disentangle Adapters inserted into ViT layers to decompose modality-invariant and specific features from a frequency perspective; (3) an LBP-guided contrastive loss to cluster features in the feature space based on attack types and modality combinations; and (4) an adaptive modal combination sampling strategy to dynamically adjust the sampling probability of different modality combinations during training.

### Key Designs

1. **Modality-Disentangle Adapters**:

    - **Function**: Explicitly separate modality-invariant and modality-specific features within each layer of the ViT.
    - **Mechanism**: The adapter consists of two parallel branches: an invariant branch and a specific branch. The invariant branch extracts low-frequency features shared across modalities (such as overall structure and contour information) via a low-pass filtering operation, while the specific branch extracts high-frequency details unique to each modality (such as texture patterns and noise features) via high-pass filtering. In implementation, a learnable frequency decomposition function divides the intermediate features of the adapter into low-frequency and high-frequency components. The invariant feature $f_{inv}$ is generated from the low-frequency component and is insensitive to modality identity; the specific feature $f_{spec}$ is generated from the high-frequency component, encoding unique discriminative information of the modality. The final feature is a weighted combination of both: $f = f_{inv} + \gamma \cdot f_{spec}$, where $\gamma$ is adjusted based on whether the modality is available.
    - **Design Motivation**: Performing frequency decomposition directly within the hidden layers of ViT is a lightweight and effective way of disentanglement. The physical meaning of the frequency domain provides an interpretable prior for disentanglement—low frequencies correspond to shared cross-modality structures, and high frequencies correspond to unique modal details. The design of the adapter allows freezing the parameters of the ViT backbone, drastically reducing the number of training parameters.

2. **LBP-Guided Contrastive Loss**:

    - **Function**: Strengthen class discriminability and modality robustness in the feature space using LBP texture priors.
    - **Mechanism**: This loss operates at two levels. Batch-level modality masking: randomly masking certain modality inputs within each training batch, forcing the model to work under various missing modality scenarios. Sample-level modality masking: independently and randomly deciding which modalities to mask for each sample in the batch, increasing training diversity. Based on this, the LBP-guided contrastive loss computes contrastive relationships between samples. Texture features extracted by LBP serve as prior anchors—samples of the same attack type should exhibit similar LBP patterns. The contrastive loss encourages: clustering features of samples of the same class and same modality combination closer, pushing features of different classes further apart, and keeping features of samples of the same class but different modality combinations close via the modality-invariant branch.
    - **Design Motivation**: Traditional contrastive losses only distinguish between positive and negative samples, ignoring differences in modality combinations. LBP guidance allows contrastive learning to enhance both modality-specific and modality-invariant features simultaneously. LBP is a classic texture descriptor with strong prior performance in the FAS field; leveraging it to guide deep feature learning is an effective form of knowledge transfer.

3. **Adaptively Modal Combination Sampling Strategy**:

    - **Function**: Dynamically adjust the sampling probability of different modality combination scenarios during training to balance the learning progress of each scenario.
    - **Mechanism**: During missing-modality training, different modality combinations exhibit varying levels of difficulty—full modality is the easiest, single modality is the hardest, and two modalities fall in between. If uniform sampling is used, the model overfits simple scenarios and underfits difficult ones. The adaptive sampling strategy dynamically monitors the performance of each modality combination on the validation set, assigning higher sampling probabilities to combinations with poorer (i.e., more difficult) performance. Specifically, the sampling probability is adjusted using the reciprocal of the loss value of each combination as a weight, granting more training opportunities to worse-performing combinations. Sampling probabilities are updated periodically during training to avoid frequent oscillations.
    - **Design Motivation**: The multi-modality missing scenario suffers from combinatorial explosion (3 modalities yield 7 non-empty combinations), with significant variations in learning difficulty among combinations. Adaptive sampling ensures the model achieves solid performance across all scenarios rather than just performing well on a subset.

### Loss & Training

The total training loss is: $L = L_{cls} + \alpha L_{LBP-con} + \beta L_{dis}$. Here, $L_{cls}$ is the classification cross-entropy loss, $L_{LBP-con}$ is the LBP-guided contrastive loss, and $L_{dis}$ is the disentanglement regularization loss (constraining the orthogonality between invariant and specific features). During training, the ViT pre-trained weights are used to initialize and freeze the backbone, training only the adapters and the classification head, which involves a small parameter count and high training efficiency.

## Key Experimental Results

### Main Results

| Missing Scenario | Metric | Ours | Prev. SOTA | Dataset |
|----------|------|---------|---------|--------|
| Missing Depth | ACER↓ | Best | Runner-up | CASIA-SURF / WMCA |
| Missing IR | ACER↓ | Best | Runner-up | CASIA-SURF / WMCA |
| Missing RGB | ACER↓ | Best | Runner-up | CASIA-SURF / WMCA |
| RGB Only | ACER↓ | Best | Runner-up | CASIA-SURF / WMCA |
| Full Modalities | ACER↓ | Best | Runner-up (compared to VP-FAS etc.) | |
| Cross-Dataset | ACER↓ | Best | Runner-up | CASIA→WMCA etc. |

### Ablation Study

| Configuration | ACER(avg) | Description |
|------|----------|------|
| Full MMA-FAS | Best | Complete model |
| w/o Modality-Disentangle Adapter | Significant increase | No separation of invariant/specific features |
| w/o LBP-Guided Contrastive Loss | Moderate increase | Contrastive learning lacks guidance from texture priors |
| w/o Adaptive Sampling Strategy | Moderate increase | Uniform sampling leads to underfitting in difficult scenarios |
| w/o Frequency Decomposition (Direct Split) | Significant increase | Frequency decomposition outperforms simple channel splitting |
| Invariant Features Only | High increase | Ignoring specific features loses discriminative power |
| Specific Features Only | High increase | Specific features are unavailable when modalities are missing |

### Key Findings
- Modality-Disentangle Adapters contribute the most, especially in scenarios with severe modality omission (only one modality), where the disentanglement method significantly outperforms the non-disentangled baseline.
- The LBP-guided contrastive loss contributes outstandingly in cross-dataset scenarios, demonstrating that the LBP texture prior helps extract more generalized features.
- The adaptive sampling strategy leads to smaller performance variance across different missing scenarios, proving its balancing effect.
- Both invariant and specific features are indispensable: using only invariant features discards modality-specific discriminative information, while using only specific features fails when those modalities are missing.

## Highlights & Insights
- **Frequency decomposition for modality disentanglement is an elegant solution**: It requires no extra disentanglement networks, achieving separation solely via low-pass/high-pass filtering within adapters. The physical meaning is clear—low frequencies represent shared structures across modalities, while high frequencies represent modality-specific details. This idea can be migrated to any multimodal fusion task dealing with missing modalities.
- **Combining LBP priors with deep contrastive learning**: LBP is a proven, effective texture descriptor in traditional methods. Utilizing it as an anchor/guidance for contrastive learning is a clever way to integrate traditional priors into deep learning. This "traditional prior guiding deep learning" strategy holds general applicability.
- **Adaptive sampling resolves training imbalance**: The training imbalance issue caused by the combinatorial explosion of multi-modal missing scenarios is elegantly solved by adaptive sampling. This trick can be directly applied to other multimodal or multi-task learning scenarios.

## Limitations & Future Work
- The frequency decomposition assumption of low frequency = shared and high frequency = specific might be overly simplistic; some cross-modality shared information may also reside in the mid-to-high frequency bands.
- Current experiments only explore the combination of three modalities: RGB + Depth + IR. Its effectiveness when extended to more modalities (e.g., near-infrared NIR, long-wave infrared LWIR, etc.) remains unknown.
- Computing LBP-guided contrastive loss requires extra extraction of LBP features, which increases preprocessing overhead.
- The adaptive sampling strategy requires periodically evaluating validation set performance to update probabilities, which might be impractical in online learning scenarios.
- In extreme cases where all sensors fail (no modality is available), the method naturally cannot help.

## Related Work & Insights
- **vs VP-FAS**: VP-FAS utilizes visual prompts to handle missing modalities, learning modality-specific prompts to adapt to different inputs. In contrast, MMA-FAS deals with this through explicit feature disentanglement, which is more interpretable and yields better results.
- **vs Flexible-Modal FAS (FM-FAS)**: FM-FAS handles missing inputs via modality-independent training but does not distinguish between invariant and specific features. MMA-FAS's disentanglement strategy is more refined, preserving the value of available specific features even during omissions.
- **vs ViTAF**: ViTAF uses auxiliary adapters on ViT for FAS but does not specifically address missing modalities. MMA-FAS's disentangling adapter is tailor-made for missing modalities while maintaining the advantage of lightweight fine-tuning from ViTAF.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of frequency-decomposition disentanglement, LBP-guided contrastive learning, and adaptive sampling is both novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers all missing scenarios, complete ablation studies, and includes cross-dataset validation.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation and well-articulated method motivations.
- Value: ⭐⭐⭐⭐ Missing modality is a core issue in multimodal learning, making this solution highly practical for real-world deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing](tf-fas_twofold-element_fine-grained_semantic_guidance_for_generalizable_face_ant.md)
- [\[ICCV 2025\] DADM: Dual Alignment of Domain and Modality for Face Anti-Spoofing](../../ICCV2025/human_understanding/dadm_dual_alignment_of_domain_and_modality_for_face_anti-spoofing.md)
- [\[CVPR 2026\] FaceCoT: Chain-of-Thought Reasoning in MLLMs for Face Anti-Spoofing](../../CVPR2026/human_understanding/facecot_cot_reasoning_face_anti_spoofing.md)
- [\[CVPR 2025\] Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing](../../CVPR2025/human_understanding/optimal_transport-guided_source-free_adaptation_for_face_anti-spoofing.md)
- [\[CVPR 2026\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](../../CVPR2026/human_understanding/from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)

</div>

<!-- RELATED:END -->
