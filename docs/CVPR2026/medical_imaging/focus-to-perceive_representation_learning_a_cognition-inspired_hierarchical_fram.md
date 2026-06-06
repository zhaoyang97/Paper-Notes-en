---
title: >-
  [Paper Note] Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis
description: >-
  [CVPR 2026][Medical Imaging][Self-supervised learning] FPRL is proposed as a clinically-inspired hierarchical self-supervised framework that mitigates motion bias by first "focusing" on intra-frame lesion-centric static…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Self-supervised learning"
  - "endoscopic video"
  - "hierarchical semantic modeling"
  - "masked reconstruction"
  - "Mamba"
date: 2026-05-08
content_hash: 191d5eda3638db68
---

# Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis

**Conference**: CVPR 2026
**arXiv**: [2603.25778](https://arxiv.org/abs/2603.25778)  
**Code**: [Available](https://github.com/MLMIP/FPRL)  
**Area**: Medical Imaging / Endoscopic Video Analysis
**Keywords**: Self-supervised learning, endoscopic video, hierarchical semantic modeling, masked reconstruction, Mamba

## TL;DR

FPRL is proposed as a clinically-inspired hierarchical self-supervised framework that mitigates motion bias by first "focusing" on intra-frame lesion-centric static semantics and then "perceiving" inter-frame contextual evolution, achieving state-of-the-art performance across 11 endoscopic datasets.

## Background & Motivation

Endoscopic video analysis is critical for early screening of gastrointestinal diseases, yet the scarcity of high-quality annotations severely limits algorithmic performance. Self-supervised video pre-training is a promising direction for addressing annotation insufficiency; however, existing methods (e.g., VideoMAE, VideoMAE V2) are primarily designed for natural videos, emphasizing dense spatiotemporal modeling and motion semantics—effective for action recognition but fundamentally at odds with the core characteristics of endoscopic video.

The key semantics in endoscopic video depend on **static, localized visual cues** (e.g., lesion morphology, color, and texture) rather than salient temporal dynamics. When dense spatiotemporal modeling is directly transferred to endoscopic video, models tend to over-attend to irrelevant motion such as camera shake and tissue displacement—a phenomenon the authors term "motion bias"—while neglecting the static semantics critical to diagnosis.

The authors observe that experienced endoscopists follow a "focus-then-perceive" cognitive pattern: they first carefully examine semantically salient regions in individual frames (color and texture anomalies), then track the temporal evolution of candidate regions. This clinical cognitive process motivates the design of the FPRL framework.

## Method

### Overall Architecture

FPRL consists of two hierarchical semantic modeling components:

1. **Static Semantic Focus**: Captures intra-frame lesion-centric local semantics.
2. **Contextual Semantic Perception**: Models temporal evolution across views while maintaining contextual consistency.

The overall framework adopts a Teacher–Student paradigm: a teacher encoder (pre-trained VideoMamba-S with frozen weights) processes the current view, while a student encoder (EndoMamba-S, trained from scratch) processes the past and future views.

### Key Designs

#### 1. Multi-View Sparse Sampling

Three sparse views (past, current, future), each containing 2 frames, are independently sampled from a temporal window of the video sequence. This sampling strategy intuitively suppresses dynamic redundancy in the video input while preserving semantic diversity across views.

#### 2. Teacher-Prior Adaptive Masking (TPAM)

TPAM is one of the core innovations in this paper. Its design motivation lies in the fact that conventional random masking cannot distinguish lesion regions from background regions.

- **Teacher saliency prior**: Teacher features are $\ell_2$-normalized to obtain a saliency map $H$, enhancing the selection of lesion-relevant tokens.
- **Lightweight attention head**: Multi-head self-attention followed by linear projection is applied to the embeddings of the current view to produce logits $R$, capturing complementary image-specific saliency.
- **Fusion sampling**: $S = \alpha H + (1-\alpha)R$; visible patches are selected via Top-K selection, yielding a learnable binary mask $M$.
- The student encoder processes only visible patches, concentrating representational capacity on lesion-relevant semantics.

#### 3. Cross-View Masked Feature Completion (CVMFC)

Fine-grained cross-view correspondences are established in the latent space, completing masked features of the current view by retrieving semantics from neighboring views:

- A Transformer-style cross-attention block is used (cross-attention → self-attention → FFN).
- Current-view features serve as queries; past/future view features serve as keys and values, respectively.
- Outputs $z_c^p$ and $z_c^f$ are aligned separately with the frozen teacher features $z_t$.

#### 4. Attention-Guided Temporal Prediction (AGTP)

Temporal correspondence consistency is enforced at the view level, complementing the token-level retrieval of CVMFC:

- Cross-attention maps computed by CVMFC are used to perform weighted pooling over neighboring view tokens.
- An EMA-updated target head provides stable, non-degenerate targets.
- Contrastive learning is performed against the globally average-pooled features of the current view.

### Loss & Training

The total loss is a weighted combination of three terms:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{Rec} + \lambda_2 \mathcal{L}_{Align} + \lambda_3 \mathcal{L}_{CL}$$

| Loss Term | Function | Weight |
|-----------|----------|--------|
| Pixel reconstruction loss $\mathcal{L}_{Rec}$ | Recovers lesion texture and boundary details of masked tokens | $\lambda_1 = 1.0$ |
| Cross-view feature alignment loss $\mathcal{L}_{Align}$ | Establishes token-level temporal correspondences (cosine similarity + $\ell_2$ consistency) | $\lambda_2 = 0.8$ |
| Temporal prediction contrastive loss $\mathcal{L}_{CL}$ | InfoNCE contrastive learning to maintain view-level temporal consistency | $\lambda_3 = 1.0$ |

Training uses the AdamW optimizer with a learning rate of 1.5e-4, cosine scheduling, 400 epochs, batch size 64, and a linear warmup over the first 40 epochs. Training is conducted on 4 NVIDIA A800 GPUs.

## Key Experimental Results

### Main Results

| Method | Conference/Year | Pre-train Time (h) | PolypDiag F1 (%) | CVC-12k Dice (%) | KUMC F1 (%) |
|--------|-----------------|--------------------|-----------------|-----------------|------------|
| Scratch | — | N/A | 83.5 | 53.2 | 73.5 |
| VideoMAE | NeurIPS'22 | 25.3 | 91.4 | 80.9 | 82.8 |
| Endo-FM | MICCAI'23 | 20.4 | 90.7 | 73.9 | 84.1 |
| M2CRL | NeurIPS'24 | 24.3 | 94.2 | 81.4 | 86.3 |
| EndoMamba | MICCAI'25 | 38.2 | 94.5 | 84.5 | 88.8 |
| **FPRL (Ours)** | — | **18.2** | **95.2** | **86.1** | **89.8** |

Under the same model architecture, FPRL improves over EndoMamba by 0.7%/1.6%/1.0% on the three benchmarks respectively, while reducing pre-training time by 52%.

### Ablation Study

| $\mathcal{L}_{Rec}$ | $\mathcal{L}_{CL}$ | $\mathcal{L}_{pt}$ | $\mathcal{L}_{ft}$ | $\mathcal{L}_{pf}$ | Classification | Segmentation | Detection |
|:---:|:---:|:---:|:---:|:---:|----------------|--------------|-----------|
| ✓ | | | | | 92.3 | 83.8 | 84.0 |
| ✓ | ✓ | ✓ | ✓ | | 94.2 | 84.0 | 86.1 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **95.2** | **86.1** | **89.8** |

Masking strategy ablation:

| Masking Strategy | Classification (%) | Segmentation (%) | Detection (%) |
|------------------|--------------------|-----------------|--------------|
| Random | 93.8 | 85.6 | 87.8 |
| Adaptive | 94.5 | 85.6 | 83.9 |
| Teacher-Prior + Adaptive (Ours) | **95.2** | **86.1** | **89.8** |

The optimal masking ratio is 90%; performance degrades at both higher (95%) and lower (70%) ratios.

### Key Findings

- Hierarchical semantic modeling (decoupled static + contextual semantics) is central to the performance gains.
- Dual-path masked completion (past + future) outperforms single-path variants by approximately 1.9%/2.0%/3.6%, respectively.
- The combination of teacher prior and adaptive masking yields the best results; using either alone is insufficient to fully capture lesion features.
- A 4-layer decoder with 1 CVMFC block is sufficient; deeper designs lead to over-smoothing of features.

## Highlights & Insights

1. **Cognition-inspired design paradigm**: The clinical workflow of "focus first, then perceive" is systematically translated into a technical solution, offering strong interpretability.
2. **Explicit modeling of motion bias**: The concept of "motion bias" in endoscopic video is explicitly formulated, and the hierarchical framework addresses it in a principled manner.
3. **Efficiency advantage**: Pre-training requires only 18.2 hours, 52% less than EndoMamba (38.2 h) and 67% less than VideoMamba (55.4 h).
4. **Elegant TPAM design**: The fusion of global teacher priors with local information from a lightweight attention head enables self-adaptive masking, directing the learning signal toward clinically relevant regions.

## Limitations & Future Work

- Single-frame pre-training variants perform poorly due to common endoscopic artifacts (motion blur, illumination flicker, specular reflections).
- Future work may explore quality-aware sampling strategies to prevent low-quality frames from degrading training.
- The framework has been validated only in the endoscopic domain; generalization to other medical imaging modalities remains to be explored.
- The scalability of the Mamba architecture to very long sequences warrants further investigation.

## Related Work & Insights

- EndoMamba's hybrid bidirectional/unidirectional Mamba topology provides the foundation for spatial–temporal decoupling in FPRL.
- The Teacher–Student paradigm draws on EMA-update ideas from self-supervised methods such as BYOL.
- M2CRL's exploration of multi-view masked contrastive learning informs the design of CVMFC.
- For domain-specific tasks such as endoscopic video analysis, domain knowledge (e.g., clinical diagnostic workflows) can substantially guide framework design.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The "focus-to-perceive" hierarchical paradigm is highly original; the TPAM design, which fuses teacher priors with adaptive attention, is a meaningful contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covering 11 datasets, 4 downstream tasks, and extensive ablations across masking strategies, ratios, architectures, and loss terms.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear and the method is described systematically, though the density of mathematical notation may increase reading burden.
- **Value**: ⭐⭐⭐⭐ — Represents a substantive advance in self-supervised learning for endoscopic video; the cognition-inspired design philosophy is transferable to other medical image analysis tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework](unlocking_positive_transfer_in_incrementally_learning_surgical_instruments_a_sel.md)
- [\[CVPR 2026\] LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](lemon_a_large_endoscopic_monocular_dataset_and_foundation_model_for_perception_in.md)
- [\[ICLR 2026\] NeuroCircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification](../../ICLR2026/medical_imaging/neurocircuitry-inspired_hierarchical_graph_causal_attention_networks_for_explain.md)
- [\[CVPR 2026\] Benchmarking Endoscopic Surgical Image Restoration and Beyond](benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)
- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)

</div>

<!-- RELATED:END -->
