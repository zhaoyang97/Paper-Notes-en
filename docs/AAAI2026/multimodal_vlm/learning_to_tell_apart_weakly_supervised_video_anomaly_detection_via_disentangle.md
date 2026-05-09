---
title: >-
  [Paper Note] Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment
description: >-
  [AAAI 2026][Multimodal VLM][Weakly supervised video anomaly detection] This paper proposes DSANet, which enhances the discriminability between normal and anomalous features in weakly supervised video anomaly detection (WS-VAD) at two levels: coarse-grained self-guided normal pattern modeling (SG-NM) and fine-grained disentangled contrastive semantic alignment (DCSA). DSANet achieves state-of-the-art performance with 86.95% AP (+1.14%) on XD-Violence and 13.01% fine-grained mAP (+3.39%) on UCF-Crime.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Weakly supervised video anomaly detection
  - semantic disentanglement
  - normal pattern modeling
  - contrastive alignment
  - CLIP
date: 2026-05-08
content_hash: 80785387fda97b7c
---

# Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment

**Conference**: AAAI 2026
**arXiv**: [2511.10334](https://arxiv.org/abs/2511.10334)
**Code**: [https://github.com/lessiYin/DSANet](https://github.com/lessiYin/DSANet)
**Area**: Multimodal VLM
**Keywords**: Weakly supervised video anomaly detection, semantic disentanglement, normal pattern modeling, contrastive alignment, CLIP

## TL;DR

This paper proposes DSANet, which enhances the discriminability between normal and anomalous features in weakly supervised video anomaly detection (WS-VAD) at two levels: coarse-grained self-guided normal pattern modeling (SG-NM) and fine-grained disentangled contrastive semantic alignment (DCSA). DSANet achieves state-of-the-art performance with 86.95% AP (+1.14%) on XD-Violence and 13.01% fine-grained mAP (+3.39%) on UCF-Crime.

## Background & Motivation

### State of the Field
Weakly supervised video anomaly detection (WS-VAD) aims to localize anomalous segments in videos using only video-level labels (normal/anomalous, without frame-level annotations). Mainstream approaches follow the multiple instance learning (MIL) framework: features are extracted using pretrained backbones (I3D or CLIP), and a binary classifier produces frame-level anomaly scores. Recent methods (VadCLIP, PEMIL, ITC, etc.) leverage the vision-language pretraining capability of CLIP to identify anomaly categories via text prompts.

### Limitations of Prior Work
Despite promising detection performance, existing WS-VAD methods suffer from two fundamental deficiencies:

**Coarse-grained level: Incomplete understanding of normal patterns**
   - The discriminative nature of MIL causes models to focus on identifying the most salient anomalous clips
   - Rich and diverse normal patterns in videos are not explicitly modeled
   - Failure to construct robust normal representations leads to blurred normal–anomaly boundaries and high false positive rates
   - For example, a complex yet normal scene (e.g., a crowded supermarket) may be misclassified as anomalous

**Fine-grained level: Severe inter-class confusion**
   - Different anomaly categories may appear visually similar (e.g., "robbery" and "theft" both involve objects being taken)
   - Background patterns between anomalous events and normal contexts may also be similar
   - Without frame-level supervision, models frequently conflate co-occurring background patterns with genuine anomalies
   - Features from different anomaly categories become entangled in the embedding space, reducing inter-class separability

### Core Idea
"Learn to tell apart" at two levels:
- Coarse-grained: Supplement the blind spots of MIL discriminative learning through generative normal pattern reconstruction
- Fine-grained: Eliminate inter-class confusion by disentangling event/background features and aligning each with its corresponding semantics

## Method

### Overall Architecture
DSANet comprises three collaborative branches:
1. **Anomaly detection branch**: Produces frame-level binary anomaly scores under the MIL framework (foundation)
2. **Self-guided normal pattern modeling branch (SG-NM)**: Mines video-specific normal patterns and guides feature reconstruction (coarse-grained enhancement)
3. **Anomaly classification branch**: Aligns video features with text category embeddings for fine-grained classification, incorporating the DCSA mechanism (fine-grained enhancement)

### Key Designs

#### 1. **Self-Guided Normal Pattern Modeling (SG-NM)**

**Mechanism**: Even in anomalous videos, local regions retain inherent normality (e.g., normal backgrounds during anomalous events). SG-NM dynamically mines normal prototypes directly from the input video without requiring an external memory bank.

Procedure:
- **Normal frame selection**: The anomaly scores $S_{det}$ from the detection branch are used to select the $M$ frames with the lowest scores ($M = 80\%$ of video length), forming a candidate normal feature set $F_n \in \mathbb{R}^{M \times D}$
- **Dynamic Normal Prototype (DNP) extraction**: $K=16$ learnable queries extract $K$ distilled normal prototypes $P \in \mathbb{R}^{K \times D}$ from $F_n$ via single-layer cross-attention
- **Normal compactness loss**: Ensures DNPs purely represent normal features

$$\mathcal{L}_{compact} = \frac{1}{M} \sum_{i=1}^{M} \min_{j \in \{1,...,K\}} d(F_n(i), P(j))$$

- **Feature reconstruction**: An 8-layer cross-attention decoder uses DNPs as keys/values to reconstruct video features. **Key design**: The residual connection is removed from the first decoder layer to ensure reconstruction relies entirely on normal patterns, preventing anomaly leakage
- **Consistency loss**: Aligns reconstructed anomaly scores $S_{rec}$ with detection scores $S_{det}$

$$\mathcal{L}_{consist} = \frac{1}{N} \sum_{i=1}^{N} (S_{det}(i) - S_{rec}(i))^2$$

**Design Motivation**: MIL learns "what is anomalous" (discriminatively); SG-NM complements this by learning "what is normal" (generatively). The two are mutually reinforcing: MIL provides anomaly scores to help SG-NM select normal frames, while reconstruction errors from SG-NM calibrate the detection boundaries of MIL.

#### 2. **Disentangled Contrastive Semantic Alignment (DCSA)**

**Mechanism**: Video features are explicitly disentangled into event components and background components, each aligned with their corresponding textual semantics.

**Visual feature disentanglement**: Soft disentanglement (not hard segmentation) using anomaly scores from the detection branch

$$F_{event} = w_{event}^\top F_{video}, \quad F_{bkg} = w_{bkg}^\top F_{video}$$

where $w_{event} = \text{Softmax}(S_{det})$ and $w_{bkg} = 1 - w_{event}$

**Text feature preparation**: The CLIP text encoder generates text embeddings $T_{text} = \{t_0, t_1, ..., t_{C-1}\}$ for $C$ categories, where $t_0$ represents the "normal" class

**Separation loss**: Pushes the normal embedding away from all anomaly class embeddings

$$\mathcal{L}_{sep} = \sum_{a=1}^{C-1} \left| \frac{t_0^\top t_a}{\|t_0\| \|t_a\|} \right|$$

**Dual contrastive alignment loss** $\mathcal{L}_{dcsa} = \mathcal{L}_{event} + \mathcal{L}_{bkg}$:
- Event alignment: $F_{event}$ is aligned with the ground-truth category $t_c$ (anomalous videos → corresponding anomaly class; normal videos → normal class)
- Background alignment: $F_{bkg}$ is **always** aligned with the normal class $t_0$ (regardless of whether the video is anomalous, the background should be biased toward normal)

**Design Motivation**: This disentanglement prevents entanglement between anomaly-related features and background patterns. The regularization effect of always aligning the background with the normal class prevents the model from mistakenly treating co-occurring background patterns as anomalous.

#### 3. **Lightweight Text Adapter**

Lightweight adapters are inserted into the first $L$ Transformer blocks of the CLIP text encoder:

$$x_{out} = (1 - \omega_t) \cdot x + \omega_t \cdot \text{Norm}(x_{adapt})$$

This achieves domain adaptation while preserving CLIP's general knowledge, and is shown to be more effective than prompt learning.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{det} + \lambda \mathcal{L}_{align} + \mathcal{L}_{consist} + \mathcal{L}_{compact} + \mathcal{L}_{dcsa} + \mathcal{L}_{sep}$$

Training details:
- Visual features: Extracted with frozen CLIP (ViT-B/16)
- Optimizer: AdamW, batch size 64/96
- Trained for 10 epochs on a single RTX 4090
- The SG-NM branch is **used only during training**; it introduces no additional computational overhead at inference

Inference strategy: Hierarchical Belief Modulation — $S_{det}$ serves as a temporal prior, $S_{align}$ assigns category probabilities, and calibration is performed via temperature ratio $\beta$.

## Key Experimental Results

### Main Results

**Coarse-grained detection**:

| Method | XD-Violence AP(%) | UCF-Crime AUC(%) |
|--------|-------------------|------------------|
| VadCLIP (CLIP) | 84.51 | 88.02 |
| ITC (CLIP) | 85.45 | 89.04 |
| ReFLIP (CLIP) | 85.81 | 88.57 |
| **DSANet** | **86.95** | **89.44** |

**Fine-grained detection (mAP@IoU, XD-Violence)**:

| Method | @0.1 | @0.2 | @0.3 | @0.4 | @0.5 | AVG |
|--------|------|------|------|------|------|-----|
| VadCLIP | 37.03 | 30.84 | 23.38 | 17.90 | 14.31 | 24.70 |
| ReFLIP | 39.24 | 33.45 | 27.71 | 20.86 | 17.22 | 27.36 |
| **DSANet** | **40.93** | **34.63** | **28.21** | **22.70** | **17.89** | **28.87** |

**Fine-grained detection (UCF-Crime)**:

| Method | @0.1 | AVG |
|--------|------|-----|
| ReFLIP | 14.23 | 9.62 |
| **DSANet** | **21.39** | **13.01** |

The fine-grained improvement on UCF-Crime is particularly notable (AVG +3.39%), demonstrating DCSA's inter-class discrimination capability in challenging scenarios.

### Ablation Study

| Configuration | AP(%) | AVG(%) | Note |
|---------------|-------|--------|------|
| Baseline (VadCLIP) | 84.51 | 24.70 | — |
| + Adapter | 85.00 | 28.15 | Text adaptation is effective |
| + Adapter + SG-NM | 85.94 | 28.39 | Normal modeling improves detection |
| + Adapter + DCSA | 85.67 | 28.25 | Semantic disentanglement improves classification |
| **+ All (DSANet)** | **86.95** | **28.87** | Components are synergistic |

**Comparison of text encoder tuning strategies**:

| Strategy | AP(%) | AVG(%) |
|----------|-------|--------|
| Frozen | 81.57 | 27.60 |
| Manual prompt | 81.05 | 28.05 |
| Learnable prompt (CoOp) | 82.88 | 28.26 |
| **Adapter (Ours)** | **86.95** | **28.87** |

### Key Findings

1. **DNP quality validation**: The minimum cosine distance distribution from normal frames to DNPs (mean 0.35) is clearly separated from that of anomalous frames (mean 0.69), confirming that DNPs form compact normal representations
2. **DCSA effectiveness validation**: The accuracy of aligning background prototypes with the normal class reaches 99.63% (vs. 87.63% for VadCLIP), with substantially stronger diagonal dominance in confusion matrices
3. **t-SNE visualization**: In DSANet's feature space, inter-class boundaries between different anomaly categories are clearer and intra-class clusters are more compact
4. Temporal visualizations show that DSANet's predictions align more closely with ground truth; VadCLIP tends to capture only the most salient clips, resulting in inaccurate temporal boundaries

## Highlights & Insights

1. **Complementary dual learning paradigm**: MIL discriminatively learns "what is anomalous" + SG-NM generatively learns "what is normal" — this complementary design is highly instructive and generalizable to other detection tasks
2. **Structured disentangled alignment**: Event features aligned to their corresponding category; background features always aligned to normal — this asymmetric alignment constraint is notably elegant, incorporating stronger structural priors than simple alignment losses
3. **Self-contained design of SG-NM**: No external memory bank is required; normal patterns are dynamically mined from the current video, making the approach scalable and data-efficient
4. The design detail of **removing the residual connection in the first reconstruction decoder layer** is subtle but critical — it prevents anomalous information from leaking into the reconstruction output via skip connections
5. The model can be trained on a single RTX 4090, making it computationally accessible

## Limitations & Future Work

- SG-NM is used only during training; reconstruction errors from this branch are not exploited at inference — it remains an open question whether reconstruction residuals could serve as auxiliary signals during inference
- Candidate normal frame selection depends on anomaly scores from the detection branch — inaccurate scores in early training may degrade DNP quality
- The quality of textual category labels affects DCSA performance — automatic generation of richer category descriptions warrants investigation
- Only CLIP ViT-B/16 is evaluated; effectiveness on larger models (e.g., ViT-L) remains to be verified
- The number of anomaly categories is fixed to those present in the training set; generalization to open-world scenarios has not been validated

## Related Work & Insights

- VadCLIP serves as the baseline, upon which DSANet introduces two additional dimensions: normal pattern modeling and semantic disentanglement
- The progression from CLIP's vision-language alignment to task-specific semantic disentanglement illustrates a promising pathway for adapting pretrained models to downstream tasks
- The event/background disentanglement is conceptually related to foreground/background separation in video understanding, and may prove useful in other video understanding tasks

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection](headhunt-vad_hunting_robust_anomaly-sensitive_heads_in_mllm_.md)
- [\[CVPR 2026\] No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection](../../CVPR2026/multimodal_vlm/no_need_for_real_anomaly_mllm_empowered_zero-shot_video_anomaly_detection.md)
- [\[AAAI 2026\] UniFit: Towards Universal Virtual Try-on with MLLM-Guided Semantic Alignment](unifit_towards_universal_virtual_try-on_with_mllm-guided_semantic_alignment.md)
- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[ICLR 2026\] Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)](../../ICLR2026/multimodal_vlm/bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)

</div>

<!-- RELATED:END -->
