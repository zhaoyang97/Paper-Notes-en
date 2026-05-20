---
title: >-
  [Paper Note] Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared
description: >-
  [CVPR 2026][Interpretability][Infrared-visible fusion] This paper proposes the first framework that performs cross-modal fusion under missing infrared conditions in the coefficient domain rather than the pixel domain. By…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "Infrared-visible fusion"
  - "missing modality"
  - "convolutional dictionary learning"
  - "coefficient-domain inference"
  - "large language model prior"
date: 2026-05-08
content_hash: 7921dbc5ad665f54
---

# Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared

**Conference**: CVPR 2026
**arXiv**: [2603.08018](https://arxiv.org/abs/2603.08018)  
**Code**: [https://github.com/harukiv/DCMIF](https://github.com/harukiv/DCMIF)  
**Area**: Interpretability
**Keywords**: Infrared-visible fusion, missing modality, convolutional dictionary learning, coefficient-domain inference, large language model prior

## TL;DR
This paper proposes the first framework that performs cross-modal fusion under missing infrared conditions in the coefficient domain rather than the pixel domain. By learning a shared convolutional dictionary that establishes a unified IR-VIS atomic space, the method performs VIS→IR inference and adaptive fusion entirely in the coefficient domain. A frozen LLM provides weak semantic priors for thermal information completion. The approach achieves performance comparable to dual-modality fusion methods using only visible light images as input.

## Background & Motivation
Infrared-visible (IR-VIS) image fusion is critical for robust perception in surveillance, robotics, and autonomous driving systems. Existing methods (CNN, CNN-Transformer, GAN, diffusion models) assume both modalities are available at both training and inference time. In practice, however, the infrared modality is frequently absent (e.g., only a visible camera is available at test time).

When infrared is missing, a straightforward approach is to generate pseudo-infrared images in pixel space before fusion. However, pixel-space generation suffers from severe drawbacks: poor controllability, weak interpretability, and susceptibility to hallucination artifacts and loss of structural detail.

**Key Challenge**: How can thermal information be stably recovered and interpretable fusion performed when infrared is absent? The paper's **Key Insight** is: **rather than generating infrared in pixel space, both modalities are mapped into a unified dictionary-coefficient space, where inference and fusion are performed entirely in the coefficient domain**, thereby anchoring data consistency and prior constraints at the atom-coefficient level.

## Method

### Overall Architecture
The complete pipeline forms a closed-loop **Encode → Transfer → Fuse → Reconstruct** process:
1. **JSRL (Joint Shared Representation Learning)**: Learns a shared convolutional dictionary $\mathbf{D}$ for both IR and VIS, mapping both modalities into a unified atomic space.
2. **VGII (Visible-Guided Infrared Inference)**: Transfers VIS coefficients to pseudo-IR coefficients in the coefficient domain, with a single-step closed-loop refinement guided by weak semantic priors from a frozen LLM.
3. **AFRI (Adaptive Fusion via Representation Inference)**: Fuses VIS and inferred IR coefficients at the atom level via windowed attention and convolution hybrid blocks, reconstructing the final image using the shared dictionary.

### Key Designs
1. **JSRL — Joint Shared Representation Learning**:

    - **Function**: Learns a cross-modal shared dictionary $\mathbf{D} \in \mathbb{R}^{B \times k \times k}$ such that both VIS and IR can be represented as $\mathbf{I} = \mathbf{D} * \mathbf{S}$.
    - **Mechanism**: Jointly minimizes reconstruction error for both modalities, coefficient priors, and dictionary regularization:
    $\min_{\mathbf{D},\mathbf{S}_{vis},\mathbf{S}_{ir}} \frac{1}{2}\|\mathbf{I}_{vis} - \mathbf{D}*\mathbf{S}_{vis}\|_F^2 + \frac{1}{2}\|\mathbf{I}_{ir} - \mathbf{D}*\mathbf{S}_{ir}\|_F^2 + \lambda_1\varphi_1(\mathbf{S}_{vis}) + \lambda_2\varphi_2(\mathbf{S}_{ir}) + \lambda_3\phi(\mathbf{D})$
    - Implemented via model-driven unfolding: alternating data consistency steps (frequency-domain Sherman-Morrison formula) and proximal update steps (learnable proxies CoeNet/DicNet).
    - **Architecture**: $N$ cascaded IV-DLBs (Infrared-Visible Dictionary Learning Blocks), each containing two coefficient solvers and one dictionary solver, with hyperparameters adaptively predicted by HypNet.
    - **Design Motivation**: The shared dictionary establishes atom-level correspondences between the two modalities, providing an interpretable unified representation space for subsequent coefficient-domain inference.

2. **VGII — Visible-Guided Infrared Inference**:

    - **Function**: Infers pseudo-IR coefficients $\mathbf{S}_{p\_ir}$ from VIS coefficients $\tilde{\mathbf{S}}_{vis}$.
    - **Mechanism**:
        - A frozen REN (Representation Encoding Network, comprising pretrained HeadNet + CSB + CoeNet) encodes VIS into coefficients.
        - RIN (Representation Inference Network, encoder-decoder + multi-head attention) maps VIS coefficients to pseudo-IR coefficients.
        - **LLM weak semantic prior refinement**: An initial pseudo-infrared image $\mathbf{I}_{p\_ir}^{(0)}$ is reconstructed; the image pair \{VIS, pseudo-IR\} along with a task description is fed as a prompt to a frozen LLM; the extracted text features $\mathbf{F}_{text}$ modulate the coefficients via FiLM (Feature-wise Linear Modulation): $\mathbf{S}_{fm} = \gamma \odot \tilde{\mathbf{S}}_{vis} + \beta$; refined coefficients are then obtained by passing through RIN again.
    - **Loss function**: $\ell_{inf} = \ell_{int} + \ell_{reg} + \ell_{grad}$
        - Consistency loss $\ell_{int}$: L1 distance between pseudo-IR and real IR in both image domain and coefficient domain.
        - Thermal regularization $\ell_{reg}$: emphasizes thermal region alignment via normalized weight maps.
        - Gradient loss $\ell_{grad}$: preserves edge consistency $\|\nabla\mathbf{I}_{p\_ir} - \nabla\mathbf{I}_{vis}\|_1$.
    - **Design Motivation**: The LLM does not generate pixels; it acts solely as a "semantic reviewer" providing channel-wise linear modulation, making it lightweight and controllable. Performing inference in the coefficient domain rather than pixel space inherits the interpretability of the dictionary.

3. **AFRI — Adaptive Fusion**:

    - **Function**: Fuses VIS coefficients and inferred IR coefficients at the atom level to reconstruct the final image.
    - **Mechanism**: RFN (Representation Fusion Network) employs two cascaded Convolution-Attention Fusion blocks to learn implicit atom-level gating weights $(\mathbf{W}_{vis}, \mathbf{W}_{p\_ir})$:
    $\mathbf{S}_f = \mathbf{W}_{vis} \odot \tilde{\mathbf{S}}_{vis} + \mathbf{W}_{p\_ir} \odot \mathbf{S}_{p\_ir}^{(1)}$
    - **Fusion loss**: $\ell_f = \|\mathbf{I}_f - \max(\mathbf{I}_{p\_ir}, \mathbf{I}_{vis})\|_1 + \|\nabla\mathbf{I}_f - \max(\nabla\mathbf{I}_{p\_ir}, \nabla\mathbf{I}_{vis})\|_1$
    - **Design Motivation**: The element-wise max operation encourages the fused result to inherit peak thermal intensity from IR and sharp structural edges from VIS; gating in the coefficient domain allows structure-oriented atoms to favor VIS and thermal-semantic atoms to favor IR.

### Loss & Training
The three modules are trained sequentially: JSRL → VGII → AFRI. JSRL is trained for 1,000 epochs on MSRS, and the learned dictionary transfers to other datasets; VGII and AFRI are each trained for 10 epochs. Adam optimizer is used with 5×5 dictionary convolutional kernels on two RTX 4090 GPUs.

## Key Experimental Results

### Main Results

| Method | Input | MSRS AG↑ | MSRS EN↑ | FLIR AG↑ | FLIR EN↑ | KAIST AG↑ |
|--------|-------|-----------|-----------|----------|----------|-----------|
| CDDFuse | IR+VIS | 4.818 | 7.321 | 5.079 | 6.766 | 3.167 |
| EMMA | IR+VIS | 4.913 | 7.333 | 3.796 | 6.489 | 3.083 |
| DCEvo | IR+VIS | 4.858 | 7.298 | 4.585 | 6.763 | 3.229 |
| **Ours** | **VIS only** | **5.037** | **7.188** | **4.518** | **6.639** | **4.414** |

**Key Finding**: **Using only visible light as input, the proposed method surpasses several dual-modality SOTA fusion methods on metrics such as AG (Average Gradient).**

Downstream task (M3FD object detection, YOLOv5): Ours mAP = 0.948 vs. SAGE (dual-modality) = 0.956, a negligible gap.
Downstream task (FMB semantic segmentation, SegFormer-b5): Ours mIoU = 62.939 vs. LRRNet (dual-modality) = 62.942, essentially on par.

### Ablation Study

| Configuration | Dictionary | LLM | AG↑ | CE↓ | EI↑ | EN↑ | SF↑ |
|---------------|-----------|-----|-----|-----|-----|-----|-----|
| Model I (Baseline) | ✗ | ✗ | 3.320 | 1.452 | 45.531 | 6.058 | 9.238 |
| Model II | ✓ | ✗ | 4.363 | 1.046 | 48.351 | 6.578 | 11.936 |
| Model III | ✗ | ✓ | 4.256 | 0.619 | 48.154 | 6.423 | 11.175 |
| **Ours** | ✓ | ✓ | **4.518** | **0.596** | **48.784** | **6.639** | **12.554** |

### Key Findings
- The shared dictionary contributes most to performance gain (Model I→II: AG +31%), validating the effectiveness of the coefficient-domain paradigm.
- LLM modulation provides additional semantic enhancement (CE reduced from 1.046 to 0.596), particularly in terms of brightness and contrast.
- The two components are complementary, and their combination yields the best results.
- Using only VIS input achieves 90%+ of the performance of dual-modality methods.

## Highlights & Insights
- **Paradigm Innovation**: The first coefficient-domain inference-fusion scheme for the missing infrared problem, avoiding the instability of pixel-space generation.
- **Clever Use of LLM**: The LLM is not used to generate images but solely as a semantic-level FiLM modulator, making it extremely lightweight and effective.
- **Training Simplicity**: No adversarial training or diffusion sampling is required; VGII and AFRI each require only 10 epochs.
- **Strong Interpretability**: All computation is performed in a unified atomic space, with dictionary atoms providing intuitive physical meaning.
- **Closed-Loop Design**: Encoding → inference → fusion → reconstruction all take place within the same dictionary-coefficient space, ensuring representational consistency.

## Limitations & Future Work
- The shared dictionary trained on MSRS is directly transferred; retraining may be required for scenes with large domain gaps (e.g., medical infrared).
- LLM processing introduces inference latency, which must be considered in real-time scenarios.
- The accuracy ceiling of coefficient-domain inference is bounded by dictionary capacity; very high-resolution images or fine-grained thermal details may suffer information loss.
- Only the missing-infrared scenario is validated; missing visible light or other multi-modal combinations are not explored.
- The method assumes that VIS images contain sufficient structural cues to infer thermal information; completely dark scenes may cause failures.

## Related Work & Insights
- Key distinction from dual-modality SOTAs such as CDDFuse and EMMA: the proposed method requires only a single-modality input.
- Model-driven unfolding (DKSVD, Learned-CSC) provides the theoretical foundation for dictionary learning.
- FiLM modulation (originating from conditional generation) is innovatively applied here for LLM semantic → coefficient-space modulation.
- **Inspiration**: The interpretability-oriented coefficient-domain paradigm is potentially generalizable to other missing-modality tasks (e.g., missing modality in MRI-CT fusion).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The combination of coefficient-domain inference-fusion paradigm and LLM weak priors is entirely original in this field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three fusion datasets + two downstream tasks + comprehensive ablation; cross-dataset generalization analysis is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous, architectural diagrams are clear, and motivation is well articulated.
- **Value**: ⭐⭐⭐⭐ First work to address missing-infrared fusion; has practical application prospects; the dictionary paradigm is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Neurodynamics-Driven Coupled Neural P Systems for Multi-Focus Image Fusion](neurodynamics-driven_coupled_neural_p_systems_for_multi-focus_image_fusion.md)
- [\[ICLR 2026\] Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](../../ICLR2026/interpretability/cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)
- [\[CVPR 2026\] On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_image-in-image_steganography.md)
- [\[CVPR 2026\] Geometry-Guided Camera Motion Understanding in VideoLLMs](geometry-guided_camera_motion_understanding_in_videollms.md)
- [\[CVPR 2026\] Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)

</div>

<!-- RELATED:END -->
