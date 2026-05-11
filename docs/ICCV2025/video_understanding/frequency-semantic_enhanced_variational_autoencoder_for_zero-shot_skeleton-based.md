---
title: >-
  [Paper Note] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition
description: >-
  [ICCV 2025][Video Understanding][Zero-shot action recognition] This paper proposes FS-VAE (Frequency-Semantic Enhanced Variational Autoencoder)…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Zero-shot action recognition"
  - "skeleton sequences"
  - "frequency decomposition"
  - "semantic alignment"
  - "variational autoencoder"
date: 2026-05-08
content_hash: 27f0fda802e2d001
---

# Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition

**Conference**: ICCV 2025
**arXiv**: [2506.22179](https://arxiv.org/abs/2506.22179)
**Code**: N/A
**Area**: Video Understanding / Action Recognition
**Keywords**: Zero-shot action recognition, skeleton sequences, frequency decomposition, semantic alignment, variational autoencoder

## TL;DR

This paper proposes FS-VAE (Frequency-Semantic Enhanced Variational Autoencoder), which achieves significant performance gains in zero-shot skeleton-based action recognition through three key contributions: frequency decomposition for enhanced skeleton semantic learning, multilevel semantic alignment to bridge the visual-text modality gap, and a calibrated cross-alignment loss to mitigate alignment ambiguity.

## Background & Motivation

**Background**: Skeleton-based action recognition leverages 3D coordinates of human body joints to identify action categories, offering advantages over RGB video in terms of privacy preservation, computational efficiency, and robustness to appearance variation. Zero-Shot Skeleton-based Action Recognition (ZS-SAR) further requires models to recognize action categories unseen during training, necessitating generalizable mappings from visual to semantic space.

**Limitations of Prior Work**: Prior methods primarily focus on learning global alignment between skeleton visual representations and semantic (text) representations, overlooking a critical issue — **insufficient representation of fine-grained action patterns in semantic space**. For example, "drinking water" and "brushing teeth" share very similar global skeletal motion patterns (both involve raising the arm toward the mouth), yet the key distinction lies in subtle differences in hand motion. Existing methods struggle to capture such local, high-frequency motion differences in semantic space.

**Key Challenge**: Global motion features of skeleton sequences are easy to extract but lack discriminability, while local fine-grained motion information is critical for distinguishing similar actions yet tends to be lost during visual-semantic alignment. Moreover, skeleton data, being sparse joint coordinate sequences, is inherently less semantically rich than RGB video.

**Goal**: (1) Enhance fine-grained information in skeleton representations to discriminate similar actions; (2) Construct multilevel visual-semantic alignment to bridge the modality gap; (3) Design a more robust alignment loss to handle ambiguous samples.

**Key Insight**: The authors approach the problem from the perspective of **frequency-domain analysis** — high-frequency components of skeleton sequences correspond to rapid local joint movements (key for distinguishing fine-grained actions), while low-frequency components capture overall body posture changes (providing global context). Targeted enhancement in the frequency domain allows precise amplification of the most discriminative motion patterns.

**Core Idea**: Frequency decomposition splits the skeletal motion signal into high-frequency (local detail) and low-frequency (global pattern) streams, each independently enhanced and fused into a VAE framework for semantic learning. Multilevel alignment and a calibrated loss further ensure zero-shot generalization.

## Method

### Overall Architecture

The FS-VAE pipeline consists of three stages: (1) **Skeleton encoding and frequency decomposition** — the input skeleton sequence is processed by a spatiotemporal encoder (e.g., ST-GCN) to extract spatiotemporal features, which are then decomposed into high- and low-frequency components; (2) **Frequency enhancement and VAE semantic learning** — the two frequency streams are independently enhanced and projected into semantic space within a VAE framework; (3) **Multilevel semantic alignment** — the enhanced skeleton semantic features are aligned with textual action descriptions at both local and global levels. The input is a sequence of skeleton joint coordinates; the output is predictions over unseen action categories.

### Key Designs

1. **Frequency-based Enhancement Module**:

    - **Function**: Extracts and independently enhances high-frequency (local detail) and low-frequency (global pattern) information from skeletal motion via frequency-domain decomposition.
    - **Mechanism**: Skeleton temporal features are decomposed into high- and low-frequency components using Discrete Cosine Transform (DCT) or analogous methods. High-frequency components are amplified via a dedicated enhancement network to magnify subtle local joint motion differences (e.g., minor wrist rotations), while low-frequency components are smoothed and denoised to reinforce the stability of global motion patterns. The two enhanced streams are then fused into a frequency-enhanced skeleton representation. High-frequency enhancement enables the model to discriminate actions with similar global motion but different local details (e.g., "drinking water" vs. "brushing teeth"), while low-frequency enhancement improves the robustness of the overall motion representation.
    - **Design Motivation**: Directly enhancing skeleton sequences in the time domain offers limited control over enhancement granularity. The frequency domain naturally provides a "global-local" decomposition axis, enabling targeted amplification of the most informative information.

2. **Semantic-based Action Description with Multilevel Alignment**:

    - **Function**: Establishes multilevel correspondences between skeleton visual features and textual semantic features.
    - **Mechanism**: Unlike conventional methods that align visual and textual representations solely at the global level, FS-VAE performs: (a) **Global alignment** — between the holistic skeleton sequence representation and action category text embeddings; (b) **Local alignment** — between body-part-level skeleton features (e.g., left hand, right leg, torso) and corresponding semantic description fragments. Hierarchical description templates generate textual descriptions at varying granularities for each action (ranging from "a person is exercising" to "the right hand grasps an object and brings it to the mouth"), establishing correspondences across multiple semantic levels. Local alignment directs the model's attention to "which body parts are doing what," while global alignment ensures overall consistency.
    - **Design Motivation**: Global alignment alone cannot capture local action differences — two actions with similar global motion may be close in global semantic space, but should be pushed apart in local semantic space (e.g., in descriptions of hand movement).

3. **Calibrated Cross-Alignment Loss**:

    - **Function**: Mitigates alignment ambiguity between skeleton and text features during training, enabling informative pairs to overcome ambiguous ones.
    - **Mechanism**: Standard contrastive losses (e.g., InfoNCE) treat all negative samples equally; however, in action recognition, certain negatives are semantically very similar to positives (e.g., "drinking water" and "watering plants" both involve interaction with water). The calibrated cross-alignment loss introduces a calibration factor that dynamically adjusts the penalty for each negative sample based on its semantic distance from the positive — semantically proximate negatives incur stronger separation penalties, while semantically distant negatives are not over-penalized. Concretely, inter-text embedding similarities serve as calibration weights: when the paired texts of skeleton feature $v_i$ and text feature $t_j$ are semantically similar (high ambiguity), the weight of that negative sample term in the alignment loss is increased.
    - **Design Motivation**: Treating all negatives indiscriminately biases the model toward coarse-grained discrimination (e.g., "motion" vs. "stillness") at the expense of fine-grained discrimination (e.g., "shooting a basketball" vs. "throwing a ball"). The calibration mechanism compels the model to allocate more learning capacity to easily confused sample pairs.

### Loss & Training

The overall loss is a weighted sum of three components: (1) **VAE reconstruction loss** — ensures invertibility and information preservation of the mapping from skeleton features to semantic space; (2) **KL divergence loss** — VAE prior regularization to ensure latent space smoothness, facilitating zero-shot generalization; (3) **Calibrated cross-alignment loss** — multilevel visual-text alignment constraints. Training follows a two-stage procedure: the skeleton encoder is first pre-trained to obtain robust visual features, followed by joint training of the frequency enhancement and semantic alignment modules.

## Key Experimental Results

### Main Results

| Dataset | Split | Metric | FS-VAE | Prev. SOTA | Gain |
|--------|------|------|--------|----------|------|
| NTU RGB+D 60 | 55/5 split | Top-1 Acc | Significantly surpasses | SynSE/SMIE etc. | Substantial margin |
| NTU RGB+D 60 | 48/12 split | Top-1 Acc | Best | — | Improvement |
| NTU RGB+D 120 | 110/10 split | Top-1 Acc | Best | — | Improvement |
| NTU RGB+D 120 | 96/24 split | Top-1 Acc | Best | — | Improvement |
| PKU-MMD | Zero-shot split | Top-1 Acc | Best | — | Improvement |

### Ablation Study

| Configuration | NTU60 (55/5) | NTU120 (110/10) | Notes |
|------|-------------|-----------------|------|
| Full FS-VAE | Best | Best | Complete model |
| w/o Frequency Enhancement | Notable drop | Notable drop | Fine-grained discrimination degrades |
| w/o Multilevel Alignment | Moderate drop | Moderate drop | Local information lost |
| w/o Calibrated Loss | Slight–moderate drop | Slight–moderate drop | Alignment ambiguity unmitigated |
| High-frequency only | Moderate | Moderate | Lacks global context |
| Low-frequency only | Moderate | Moderate | Lacks local detail |

### Key Findings

- **The frequency enhancement module contributes most** — gains are especially pronounced on semantically similar action pairs (e.g., "drinking water" vs. "brushing teeth"), validating the critical role of high-frequency information for fine-grained discrimination.
- **High- and low-frequency components are mutually indispensable** — using either frequency stream alone underperforms their combination, demonstrating that global context and local detail are complementary.
- **The calibrated loss yields the greatest benefit on easily confused action pairs** — for action pairs with large semantic gaps (e.g., "walking" vs. "typing on a keyboard"), calibration has negligible impact; for semantically close pairs, the calibrated loss contributes substantially.
- **The method's advantage is most pronounced under harder zero-shot settings with more unseen classes (e.g., NTU120's 96/24 split)** — indicating that frequency-enhanced fine-grained representations are increasingly valuable when finer discrimination is required.

## Highlights & Insights

- **Applying frequency-domain decomposition to skeleton semantic learning** is a natural and effective approach — skeleton sequences are fundamentally temporal signals, and frequency analysis is a foundational signal processing tool, yet it has been underutilized in zero-shot action recognition. This principle of matching appropriate mathematical tools to the data type warrants broader adoption.
- **The calibrated cross-alignment loss** reflects a deep understanding of the limitations of contrastive learning — the false negative problem in semantic space is a well-known issue in standard contrastive learning, and the proposed calibration strategy offers a practical remedy directly applicable to other multimodal alignment tasks.
- **Multilevel alignment** extending from "global" to "body-part level" suggests a more general paradigm: establishing correspondences at multiple granularities simultaneously in cross-modal alignment consistently outperforms single-granularity alignment.

## Limitations & Future Work

- **Parameter selection for frequency decomposition** (cutoff frequency, enhancement intensity) may need to vary across different action types; the current use of uniform parameters may be suboptimal.
- **Text description quality** is critical for multilevel alignment, but manually designed templates may not cover the key semantic distinctions for all actions. Future work could leverage LLMs to automatically generate more precise multilevel descriptions.
- **Inherent limitations of skeleton data** — for actions that can only be distinguished by information about held objects (e.g., "eating with chopsticks" vs. "eating with a fork"), joint coordinates alone may be fundamentally insufficient.
- **Generalization to real-world scenarios remains unvalidated** — all experiments are conducted on laboratory-collected standard datasets; noise and occlusion in skeleton estimation under real-world conditions may degrade the quality of frequency decomposition.
- Future work could explore **adaptive frequency analysis** — automatically selecting the most discriminative frequency bands based on action type.

## Related Work & Insights

- **vs. SynSE**: SynSE employs synthesized visual-semantic embeddings for zero-shot recognition but lacks modeling of the inherent time-frequency characteristics of skeleton data. FS-VAE addresses this gap through frequency decomposition.
- **vs. SMIE**: SMIE introduces structured multi-instance embeddings but performs alignment only at the global level. FS-VAE's multilevel alignment applies constraints at both local and global levels, providing richer cross-modal correspondences.
- **vs. SA-DVAE**: SA-DVAE adopts a discriminative-generative VAE framework without incorporating frequency-domain information. FS-VAE demonstrates that frequency enhancement can serve as a plug-and-play module that substantially improves VAE-based frameworks.
- The frequency enhancement strategy shares conceptual parallels with the SlowFast dual-pathway design in video understanding — both capture complementary information across different temporal scales — but FS-VAE achieves more precise decomposition in the frequency domain.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Frequency enhancement and the calibrated alignment loss are first proposed in the context of zero-shot skeleton recognition, though the individual components are not entirely novel concepts.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across multiple standard benchmarks and diverse split settings with thorough ablations, though visualization analyses are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Methodology is clearly presented and problem motivation is compellingly articulated.
- **Value**: ⭐⭐⭐⭐ Establishes a new state-of-the-art baseline for zero-shot skeleton recognition; the frequency enhancement strategy offers inspiration for related areas.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [\[ICCV 2025\] Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition](trokens_semantic-aware_relational_trajectory_tokens_for_few-shot_action_recognit.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition](adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)
- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-based Human Action Recognition with Virtual Connections](adaptive_hyper_graph_convolution_network_skeleton_action_recognition.md)

</div>

<!-- RELATED:END -->
