---
title: >-
  [Paper Note] CLIPPan: Adapting CLIP as A Supervisor for Unsupervised Pansharpening
description: >-
  [AAAI 2026][Model Compression][Pansharpening] This paper proposes CLIPPan, which fine-tunes CLIP in a parameter-efficient manner to understand multispectral/panchromatic/high-resolution multispectral image types and the…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Pansharpening"
  - "CLIP"
  - "Unsupervised"
  - "Vision-Language Model"
  - "Remote Sensing"
date: 2026-05-08
content_hash: dd415c02a93c1a52
---

# CLIPPan: Adapting CLIP as A Supervisor for Unsupervised Pansharpening

**Conference**: AAAI 2026
**arXiv**: [2511.10896](https://arxiv.org/abs/2511.10896)
**Code**: [Jiabo-Liu/CLIPPan](https://github.com/Jiabo-Liu/CLIPPan)
**Area**: Model Compression
**Keywords**: Pansharpening, CLIP, Unsupervised, Vision-Language Model, Remote Sensing

## TL;DR

This paper proposes CLIPPan, which fine-tunes CLIP in a parameter-efficient manner to understand multispectral/panchromatic/high-resolution multispectral image types and the pansharpening process, then leverages text prompts encoding Wald's protocol as semantic supervision signals to enable full-resolution unsupervised pansharpening without ground truth. CLIPPan operates as a plug-and-play module compatible with arbitrary pansharpening backbone networks.

## Background & Motivation

Pansharpening fuses spectrally rich multispectral (MS) images with spatially high-resolution panchromatic (PAN) images to produce high-resolution multispectral (HRMS) images, which is critical for remote sensing applications such as urban planning and environmental monitoring.

**Core Problem**: Existing deep learning methods rely heavily on ground truth (GT) for supervised training, yet GT is unavailable in real full-resolution scenarios. The common practice of training on downsampled simulated data introduces a severe domain gap when applied to real full-resolution images. Unsupervised methods avoid GT dependency but only exploit low-level pixel relationships between the fusion output and inputs, lacking high-level semantic guidance for the fusion objective.

**Core Insight**: If the model can be informed of "what the fusion target rules are" (e.g., Wald's protocol), high-level semantic supervision can constrain the fusion output to reside in the HRMS domain. CLIP's image-text alignment capability can naturally translate textually described fusion rules into supervision signals.

## Method

### Overall Architecture

CLIPPan proceeds in two stages:

**Stage I — Vision-Language Alignment**: Fine-tunes CLIP to (i) recognize LRMS/PAN/HRMS image types, (ii) understand remote sensing image content, and (iii) understand the pansharpening process (the MS+PAN→HRMS mapping).

**Stage II — Language-Guided Unsupervised Pansharpening**: Uses the fine-tuned CLIP as a fixed semantic supervisor, combined with low-level visual constraints to train the pansharpening network.

### Key Designs

**1. Parameter-Efficient Fine-Tuning Strategy**

To preserve CLIP's strong generalization capability, only 6 lightweight adapter modules are inserted (3 on the visual side + 3 on the text side). Since CLIP's visual encoder is incompatible with the multi-band input of multispectral images, the original RGB input layer is replaced with a learnable convolutional layer.

**2. Inter-Modal Contrastive Learning (InterMCL)**

Binds each image type to its corresponding semantic space. Fixed text descriptions (rather than content-dependent ones) are used:
- MS: "a multispectral image"
- PAN: "a panchromatic image"
- HRMS: "High-quality reference image adhering to Wald's protocol: spectrally consistent with original data and spatially sharp"

A contrastive loss pulls image-text positive pairs together and pushes negative pairs apart:

$$\mathcal{L}_{\text{inter}} = \frac{1}{3}\sum_{M1,M2}\mathcal{L}_{\text{align}}(F^I_{M1}, F^T_{M2})$$

HRMS images are generated online using the traditional BDSD algorithm since they are unavailable at full resolution.

**3. Intra-Modal Contrastive Learning (IntraMCL)**

Prevents feature collapse caused by using fixed text descriptions. LRMS/PAN/HRMS images from the same scene are treated as positive samples, while those from different scenes serve as negatives:

$$\mathcal{L}_{\text{intra}} = -\frac{1}{3N}\sum_{i=1}^{3N}\log\frac{\exp(\langle F^{I(i)}_{M1}, F^{I(i)}_{M2}\rangle/\tau_i)}{\sum_{k=1}^{3N}\exp(\langle F^{I(i)}_{M1}, F^{I(j)}_{M1}\rangle/\tau_i)}$$

This ensures feature diversity while promoting domain transfer from natural images to remote sensing images.

**4. Fusion-Aware Alignment**

Image Fusion Adapters (IFA) and Text Fusion Adapters (TFA) are introduced to learn fusion feature generation from MS+PAN features, aligning with HRMS/Wald's protocol features:

$$\mathcal{L}_{\text{fusion}} = \|F^T_{\text{fuse}} - F^T_{\text{wald}}\|_1 + \|F^I_{\text{fuse}} - F^I_{\text{HRMS}}\|_1$$

**5. Direction Vector Semantic Supervision (Core of Stage II)**

Directly applying element-wise loss with Wald's protocol text features is infeasible since they are fixed across all images. Instead, the method exploits **consistency of feature displacement directions**:

$$\mathcal{L}_d = 1 - \frac{1}{2}(\langle \Delta\mathbf{V}^I_{\text{MS}}, \Delta\mathbf{V}^T_{\text{MS}}\rangle + \langle \Delta\mathbf{V}^I_{\text{PAN}}, \Delta\mathbf{V}^T_{\text{PAN}}\rangle)$$

where $\Delta\mathbf{V}^I_{\text{MS}} = F^I_{\text{out}} - F^I_{\text{MS}}$ is the fusion displacement direction in image space, and $\Delta\mathbf{V}^T_{\text{MS}} = F^T_{\text{wald}} - F^T_{\text{MS}}$ is the fusion target direction in text space. By penalizing angular deviation between directions in both spaces, the output is semantically guided toward the HRMS domain.

### Loss & Training

**Stage I**: $\mathcal{L}_{s1} = \mathcal{L}_{\text{inter}} + \mathcal{L}_{\text{intra}} + \mathcal{L}_{\text{fusion}}$

**Stage II Low-Level Visual Constraints**:
- Spectral fidelity: $\mathcal{L}_{\text{spec}} = \|\downarrow(\mathbf{I}_{\text{out}}) - \mathbf{I}_{\text{MS}}\|_2^2 + 1 - \text{SSIM}$
- Spatial sharpness: $\mathcal{L}_{\text{spat}} = \|\phi(\mathbf{I}_{\text{out}}) - \mathbf{I}_{\text{PAN}}\|_2^2 + 1 - \text{SSIM}$
- QNR trade-off: $\mathcal{L}_{\text{QNR}} = (1-D_\lambda)(1-D_s)$
- Pseudo-supervision: $\mathcal{L}_{\text{ship}}$ (using the output of a SHIP network trained at reduced resolution as reference)

**Total Loss**: $\mathcal{L}_{s2} = \mathcal{L}_{\text{spec}} + \mathcal{L}_{\text{spat}} + \mathcal{L}_{\text{QNR}} + \mathcal{L}_{\text{ship}} + \mathcal{L}_d$

Training setup: GTX-4090, Adam (lr=0.003), batch size=32, 1000 iterations.

## Key Experimental Results

### Main Results

**Table 1: Quantitative Results at Full and Reduced Resolution (QB and WV3 Datasets)**

| Method | QB $D_\lambda$↓ | QB QNR↑ | WV3 $D_\lambda$↓ | WV3 QNR↑ |
|---|---|---|---|---|
| ArbRPN | 0.0140 | 0.9582 | 0.0271 | 0.9383 |
| ArbRPN-C | **0.0030** | **0.9691** | **0.0042** | **0.9582** |
| LFormer | 0.0124 | 0.9602 | 0.0253 | 0.9227 |
| LFormer-C | **0.0053** | **0.9676** | **0.0049** | **0.9572** |
| PanMamba | 0.0134 | 0.9592 | 0.0152 | 0.9426 |
| PanMamba-C | **0.0050** | **0.9672** | **0.0051** | **0.9578** |

CLIPPan (denoted by the -C suffix) consistently improves all 5 backbone networks. ArbRPN-C reduces spectral distortion $D_\lambda$ by 79% on QB, and LFormer-C reduces spatial distortion $D_s$ by approximately 30% on WV3.

### Ablation Study

**Table 2: Ablation of Unsupervised Fusion Losses (WV3 Reduced Resolution, ArbRPN Backbone)**

| Loss Combination | MPSNR↑ | ERGAS↓ | SAM↓ | Q2n↑ |
|---|---|---|---|---|
| $\mathcal{L}_{\text{spec}} + \mathcal{L}_{\text{spat}}$ | 29.27 | 8.96 | 9.17 | 0.61 |
| $\mathcal{L}_{\text{unsup}}$ | 32.19 | 5.88 | 6.66 | 0.71 |
| $\mathcal{L}_{\text{unsup}} + \mathcal{L}_d$ | 32.37 | 5.75 | 6.55 | 0.74 |
| $\mathcal{L}_{\text{unsup}} + \mathcal{L}_{\text{ship}} + \mathcal{L}_d$ | **34.72** | **4.49** | **5.54** | **0.80** |

The joint use of the semantic loss $\mathcal{L}_d$ and pseudo-supervision $\mathcal{L}_{\text{ship}}$ achieves the best results, improving MPSNR by 5.4 dB.

**Table 3: Ablation of CLIP Fine-Tuning Losses** — Incrementally adding IntraMCL, InterMCL, and $\mathcal{L}_1$ each yields consistent improvements.

**Table 5: Ablation of Text Descriptions** — Wald's protocol text achieves the best balance across all metrics, confirming that precise protocol-based text is critical for semantic supervision.

### Key Findings

- Even without GT, CLIPPan achieves performance close to supervised methods.
- Consistent improvements are also observed at reduced resolution, indicating the framework is effective under both supervised and unsupervised settings.
- Learnable residual convolutions for multispectral input outperform manual strategies such as PCA, RGB, and GBNIR.

## Highlights & Insights

1. **Language as Supervision**: This work is the first to convert fusion rules such as Wald's protocol into CLIP-based semantic supervision via text prompts, representing an elegant paradigm shift.
2. **Direction Vector Loss**: Rather than directly comparing fixed text features, the method compares the consistency of "feature displacement directions before and after fusion," elegantly resolving the invariance issue of text features.
3. **Plug-and-Play Generality**: Compatible with 5 different backbone networks with consistent gains, demonstrating high practical value.
4. **Bidirectional Implication**: Beyond guiding fusion via protocols, the framework can in principle be used to evaluate protocol validity and even discover new protocols.

## Limitations & Future Work

1. The CLIP fine-tuning stage still relies on the BDSD algorithm to generate approximate HRMS labels, introducing additional priors.
2. Text descriptions are currently hand-crafted fixed templates; learnable prompt tuning warrants further exploration.
3. Validation is limited to WorldView-3 and QuickBird sensors; generalization to additional sensors remains to be verified.
4. Pseudo-supervision $\mathcal{L}_{\text{ship}}$ depends on the quality of the pre-trained SHIP network.

## Related Work & Insights

- **CLIP-Adapter / CoOp / LoRA-CLIP**: Parameter-efficient fine-tuning strategies; this work follows the adapter paradigm.
- **RS-CLIP / GeoCLIP**: CLIP adaptation for remote sensing; this work extends the approach to the pansharpening task.
- **Insights**: The proposed paradigm can be generalized to other remote sensing image fusion tasks (e.g., hyperspectral–multispectral fusion); the idea of "using protocol text as supervision" is also applicable to other tasks with well-defined rules but no available GT.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introducing vision-language models as supervisors for pansharpening represents a strong paradigm innovation.
- Technical Depth: ⭐⭐⭐⭐ — The two-stage design is complete and the direction vector loss is elegantly conceived.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 5 backbones × 2 datasets, with comprehensive ablations (5 ablation groups).
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with well-motivated design choices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Correcting False Alarms from Unseen: Adapting Graph Anomaly Detectors at Test Time](correcting_false_alarms_from_unseen_adapting_graph_anomaly_detectors_at_test_tim.md)
- [\[ICCV 2025\] Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](../../ICCV2025/model_compression/perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)
- [\[ICCV 2025\] Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning](../../ICCV2025/model_compression/soft_separation_and_distillation_toward_global_uniformity_in_federated_unsupervi.md)
- [\[NeurIPS 2025\] Towards Unsupervised Open-Set Graph Domain Adaptation via Dual Reprogramming](../../NeurIPS2025/model_compression/towards_unsupervised_open-set_graph_domain_adaptation_via_dual_reprogramming.md)
- [\[ICLR 2026\] Distilling and Adapting: A Topology-Aware Framework for Zero-Shot Interaction Prediction in Multiplex Biological Networks](../../ICLR2026/model_compression/distilling_and_adapting_a_topology-aware_framework_for_zero-shot_interaction_pre.md)

</div>

<!-- RELATED:END -->
