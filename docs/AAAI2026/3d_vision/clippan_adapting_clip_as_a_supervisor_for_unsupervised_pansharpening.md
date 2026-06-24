---
title: >-
  [Paper Note] CLIPPan: Adapting CLIP as A Supervisor for Unsupervised Pansharpening
description: >-
  [AAAI 2026][3D Vision][Pansharpening] This paper proposes CLIPPan, which adapts CLIP via parameter-efficient fine-tuning to understand multispectral/panchromatic/high-resolution multispectral image types and the pansharpening process. It leverages text prompts such as Wald's protocol as semantic supervision signals to achieve unsupervised pansharpening at full resolution without ground truth. It can serve as a plug-and-play module compatible with any pansharpening backbone ne…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Pansharpening"
  - "CLIP"
  - "Unsupervised"
  - "Vision-Language Model"
  - "Remote Sensing"
date: 2026-05-08
content_hash: 15d9f9a66b0ba407
---

# CLIPPan: Adapting CLIP as A Supervisor for Unsupervised Pansharpening

**Conference**: AAAI 2026  
**arXiv**: [2511.10896](https://arxiv.org/abs/2511.10896)  
**Code**: [Jiabo-Liu/CLIPPan](https://github.com/Jiabo-Liu/CLIPPan)  
**Area**: Model Compression  
**Keywords**: Pansharpening, CLIP, Unsupervised, Vision-Language Model, Remote Sensing

## TL;DR

This paper proposes CLIPPan, which adapts CLIP via parameter-efficient fine-tuning to understand multispectral/panchromatic/high-resolution multispectral image types and the pansharpening process. It leverages text prompts such as Wald's protocol as semantic supervision signals to achieve unsupervised pansharpening at full resolution without ground truth. It can serve as a plug-and-play module compatible with any pansharpening backbone network.

## Background & Motivation

Pansharpening merges multispectral (MS) images rich in spectral information with panchromatic (PAN) images of high spatial resolution to generate high-resolution multispectral (HRMS) images, which is crucial for remote sensing applications such as urban planning and environmental monitoring.

**Core Problem**: Existing deep learning methods heavily rely on ground truth (GT) for supervised training, but GT is unavailable in real-world full-resolution scenarios. A common practice is to train on degraded-resolution simulated data, which leads to a severe domain gap on real full-resolution images. Although unsupervised methods avoid GT reliance, they only utilize low-level pixel relationships between the fusion output and the input as constraints, lacking high-level semantic guidance for the fusion objective.

**Key Insight**: If the model can be informed of the "target rules of fusion" (e.g., Wald's protocol), high-level semantic supervision can constrain the fusion output to reside in the HRMS domain. The image-text alignment capability of CLIP is well-suited to translate text-described fusion rules into supervision signals.

## Method

### Overall Architecture

CLIPPan consists of two stages:

**Stage I — Vision-Language Alignment**: Fine-tune CLIP so that it can (i) identify LRMS/PAN/HRMS image types, (ii) understand remote sensing image content, and (iii) comprehend the pansharpening process (the mapping MS+PAN→HRMS).

**Stage II — Language-Guided Unsupervised Pansharpening**: Use the fine-tuned CLIP as a fixed semantic supervisor, combined with low-level vision constraints to train the pansharpening network.

### Key Designs

**1. Parameter-Efficient Fine-Tuning Strategy**

To maintain the strong generalization ability of CLIP, only 6 lightweight adapter modules (3 on the vision side + 3 on the text side) are inserted. Since the CLIP vision encoder is incompatible with multispectral multi-band inputs, the original RGB input layer is replaced with a learnable convolutional layer.

**2. Inter-Modal Contrastive Learning (InterMCL)**

Bind each image type to its corresponding semantic space. Fixed text descriptions are used (instead of content-dependent descriptions):
- MS: "a multispectral image"
- PAN: "a panchromatic image"  
- HRMS: "High-quality reference image adhering to Wald's protocol: spectrally consistent with original data and spatially sharp"

Image-text positive pairs are pulled together, and negative pairs are pushed apart via contrastive loss:

$$\mathcal{L}_{\text{inter}} = \frac{1}{3}\sum_{M1,M2}\mathcal{L}_{\text{align}}(F^I_{M1}, F^T_{M2})$$

Here, the HRMS images are generated online using the traditional BDSD algorithm (since they are unavailable at full resolution).

**3. Intra-Modal Contrastive Learning (IntraMCL)**

Prevents feature collapse caused by using fixed descriptions. LRMS/PAN/HRMS of the same scene are treated as positive samples, while those from different scenes are treated as negative samples:

$$\mathcal{L}_{\text{intra}} = -\frac{1}{3N}\sum_{i=1}^{3N}\log\frac{\exp(\langle F^{I(i)}_{M1}, F^{I(i)}_{M2}\rangle/\tau_i)}{\sum_{k=1}^{3N}\exp(\langle F^{I(i)}_{M1}, F^{I(j)}_{M1}\rangle/\tau_i)}$$

This guarantees feature diversity and facilitates domain transfer from natural images to remote sensing images.

**4. Fusion-Aware Alignment**

Introduces the Image Fusion Adapter (IFA) and Text Fusion Adapter (TFA) to learn to generate fusion features from MS+PAN features, aligning them with HRMS/Wald's protocol features:

$$\mathcal{L}_{\text{fusion}} = \|F^T_{\text{fuse}} - F^T_{\text{wald}}\|_1 + \|F^I_{\text{fuse}} - F^I_{\text{HRMS}}\|_1$$

**5. Direction Vector Semantic Supervision (Core of Stage II)**

The text features of Wald's protocol cannot be used directly for element-wise loss (as they are fixed for all images); instead, the **consistency of feature displacement direction** is leveraged:

$$\mathcal{L}_d = 1 - \frac{1}{2}(\langle \Delta\mathbf{V}^I_{\text{MS}}, \Delta\mathbf{V}^T_{\text{MS}}\rangle + \langle \Delta\mathbf{V}^I_{\text{PAN}}, \Delta\mathbf{V}^T_{\text{PAN}}\rangle)$$

where $\Delta\mathbf{V}^I_{\text{MS}} = F^I_{\text{out}} - F^I_{\text{MS}}$ represents the direction of fusion change in the image space, and $\Delta\mathbf{V}^T_{\text{MS}} = F^T_{\text{wald}} - F^T_{\text{MS}}$ represents the target direction of fusion in the text space. By penalizing the angular deviation of directions in the two spaces, the output is guided to semantically align with the HRMS domain.

### Loss & Training

**Stage I**: $\mathcal{L}_{s1} = \mathcal{L}_{\text{inter}} + \mathcal{L}_{\text{intra}} + \mathcal{L}_{\text{fusion}}$

**Stage II Low-Level Visual Constraints**:
- Spectral Fidelity: $\mathcal{L}_{\text{spec}} = \|\downarrow(\mathbf{I}_{\text{out}}) - \mathbf{I}_{\text{MS}}\|_2^2 + 1 - \text{SSIM}$
- Spatial Sharpness: $\mathcal{L}_{\text{spat}} = \|\phi(\mathbf{I}_{\text{out}}) - \mathbf{I}_{\text{PAN}}\|_2^2 + 1 - \text{SSIM}$
- QNR Trade-off: $\mathcal{L}_{\text{QNR}} = (1-D_\lambda)(1-D_s)$
- Pseudo-Supervision: $\mathcal{L}_{\text{ship}}$ (with the output of the SHIP network trained at reduced resolution acting as the reference)

**Total Loss**: $\mathcal{L}_{s2} = \mathcal{L}_{\text{spec}} + \mathcal{L}_{\text{spat}} + \mathcal{L}_{\text{QNR}} + \mathcal{L}_{\text{ship}} + \mathcal{L}_d$

GTX-4090, Adam (lr=0.003), batch size=32, 1000 iterations.

## Key Experimental Results

### Main Results

**Table 1: Quantitative results on full-resolution + reduced-resolution (QB and WV3 datasets)**

| Method | QB $D_\lambda$↓ | QB QNR↑ | WV3 $D_\lambda$↓ | WV3 QNR↑ |
|---|---|---|---|---|
| ArbRPN | 0.0140 | 0.9582 | 0.0271 | 0.9383 |
| ArbRPN-C | **0.0030** | **0.9691** | **0.0042** | **0.9582** |
| LFormer | 0.0124 | 0.9602 | 0.0253 | 0.9227 |
| LFormer-C | **0.0053** | **0.9676** | **0.0049** | **0.9572** |
| PanMamba | 0.0134 | 0.9592 | 0.0152 | 0.9426 |
| PanMamba-C | **0.0050** | **0.9672** | **0.0051** | **0.9578** |

CLIPPan (with the -C suffix) consistently improves across all 5 backbone networks. ArbRPN-C reduces spectral distortion $D_\lambda$ on QB by 79%, and LFormer-C reduces spatial distortion $D_s$ on WV3 by approximately 30%.

### Ablation Study

**Table 2: Ablation study of unsupervised fusion loss (WV3 reduced-resolution, ArbRPN backbone)**

| Loss Combination | MPSNR↑ | ERGAS↓ | SAM↓ | Q2n↑ |
|---|---|---|---|---|
| $\mathcal{L}_{\text{spec}} + \mathcal{L}_{\text{spat}}$ | 29.27 | 8.96 | 9.17 | 0.61 |
| $\mathcal{L}_{\text{unsup}}$ | 32.19 | 5.88 | 6.66 | 0.71 |
| $\mathcal{L}_{\text{unsup}} + \mathcal{L}_d$ | 32.37 | 5.75 | 6.55 | 0.74 |
| $\mathcal{L}_{\text{unsup}} + \mathcal{L}_{\text{ship}} + \mathcal{L}_d$ | **34.72** | **4.49** | **5.54** | **0.80** |

The joint use of semantic loss $\mathcal{L}_d$ and pseudo-supervision $\mathcal{L}_{\text{ship}}$ achieves the best results, raising MPSNR by 5.4 dB.

**Table 3: Ablation study of CLIP fine-tuning loss** — Stepwise addition of IntraMCL, InterMCL, and $\mathcal{L}_1$ consistently brings improvements.

**Table 5: Ablation study of text descriptions** — The text of Wald's protocol achieves the best balance across all metrics, validating that accurate protocol description text is crucial for semantic supervision.

### Key Findings

- Even without GT, CLIPPan achieves performance close to supervised methods.
- Consistent improvements are also achieved in reduced-resolution experiments, indicating that the framework is effective for both supervised and unsupervised settings.
- Handling multispectral input via learnable residual convolution outperforms manual strategies such as PCA/RGB/GBNIR.

## Highlights & Insights

1. **Language as Supervision**: For the first time, fusion rules such as Wald's protocol are transformed into CLIP semantic supervision as text prompts, presenting an elegant concept.
2. **Direction Vector Loss**: Instead of directly comparing fixed text features, it compares the consistency of the "feature displacement direction before and after fusion," cleverly resolving the issue of static text features.
3. **Plug-and-Play Generalizability**: Compatible with 5 different backbone networks, bringing consistent improvements, and demonstrating high practical value.
4. **Bidirectional Inspiration**: The framework not only uses protocols to guide fusion, but can conversely evaluate protocol effectiveness or even discover new protocols.

## Limitations & Future Work

1. The CLIP fine-tuning stage still requires the BDSD algorithm to generate HRMS approximate labels, introducing an extra prior.
2. Current text descriptions are hand-crafted fixed templates; learnable prompt tuning could be explored.
3. Validated only on two sensor types (WorldView-3 and QuickBird); generalization to more sensors remains to be verified.
4. Pseudo-supervision $\mathcal{L}_{\text{ship}}$ depends on the quality of the pre-trained SHIP network.

## Related Work & Insights

- **CLIP-Adapter / CoOp / LoRA-CLIP**: Parameter-efficient fine-tuning strategies; this paper adopts the adapter route.
- **RS-CLIP / GeoCLIP**: CLIP adaptation in the remote sensing domain; this work extends it to pansharpening tasks.
- **Insights**: This paradigm can be generalized to other remote sensing image fusion tasks (e.g., hyperspectral-multispectral fusion); the concept of "using protocol text as supervision" can also be applied to other tasks constrained by explicit rules but lacking GT.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introduces vision-language models into pansharpening supervision, offering a highly innovative paradigm.
- Technical Depth: ⭐⭐⭐⭐ — Complete two-stage design with a highly elegant direction vector loss.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 5 backbones × 2 datasets, with comprehensive ablations (5 sets of ablation experiments).
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-supported motivation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DINO Eats CLIP: Adapting Beyond Knowns for Open-set 3D Object Retrieval](../../CVPR2026/3d_vision/dino_eats_clip_adapting_beyond_knowns_for_open-set_3d_object_retrieval.md)
- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)
- [\[CVPR 2026\] Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning](../../CVPR2026/3d_vision/adapting_point_cloud_analysis_via_multimodal_bayesian_distribution_learning.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](../../ICCV2025/3d_vision/clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Flux4D: Flow-based Unsupervised 4D Reconstruction](../../NeurIPS2025/3d_vision/flux4d_flow-based_unsupervised_4d_reconstruction.md)

</div>

<!-- RELATED:END -->
