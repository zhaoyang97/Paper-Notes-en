---
title: >-
  [Paper Note] Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector
description: >-
  [CVPR 2025][Multimodal VLM][Deepfake Detection] Proposed M2F2-Det, the first multimodal face forgery detector that simultaneously outputs deepfake detection scores and textual explanations. It adapts CLIP to learn forgery features via Forgery Prompt Learning, fuses CLIP and deepfake encoder features using a Bridge Adapter, and guides the LLM to generate trustworthy explanations using frequency-domain tokens.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Deepfake Detection"
  - "CLIP Adaptation"
  - "Prompt Learning"
  - "Interpretability"
  - "Multimodality"
date: 2026-05-08
content_hash: c0b708054490096d
---

# Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector

**Conference**: CVPR 2025  
**arXiv**: [2503.20188](https://arxiv.org/abs/2503.20188)  
**Code**: Yes (Link provided in paper)  
**Area**: Multimodal VLM  
**Keywords**: Deepfake Detection, CLIP Adaptation, Prompt Learning, Interpretability, Multimodality

## TL;DR

Proposed M2F2-Det, the first multimodal face forgery detector that simultaneously outputs deepfake detection scores and textual explanations. It adapts CLIP to learn forgery features via Forgery Prompt Learning, fuses CLIP and deepfake encoder features using a Bridge Adapter, and guides the LLM to generate trustworthy explanations using frequency-domain tokens.

## Background & Motivation

Three major limitations exist in the field of deepfake detection:

1. **Disconnection between detection and explanation**: Traditional methods (e.g., EfficientNet, Multi-Attention, etc.) only output binary classification scores and lack interpretability, whereas DDVQA-BLIP only generates textual explanations but suffers from low detection accuracy (18% lower than traditional methods).
2. **Inadequate CLIP adaptation**: Existing CLIP-based detectors (e.g., UniFake, DEFAKE) only use simple linear layers or ResNet-18 to interface with CLIP, lacking specialized prompt designs for face forgery and failing to fully exploit CLIP's multimodal learning capabilities.
3. **Gap in CLIP+LLM integration**: While the combination of CLIP and LLM has succeeded in fields such as document parsing and medical diagnosis, the integration of CLIP's open-set recognition ability and LLM's text generation ability in deepfake detection remains unexplored.

## Method

### Overall Architecture

M2F2-Det consists of four components: (1) a frozen CLIP image encoder $\mathcal{E}_I$ and text encoder $\mathcal{E}_T$; (2) Forgery Prompt Learning (FPL) to generate a forgery attention map; (3) a Bridge Adapter to fuse features from CLIP and the deepfake encoder $\mathcal{E}_D$ for binary classification detection; (4) a Forgery Explanation Module combined with an LLM to generate textual explanations. The training is conducted in three stages.

### Key Designs

1. **Forgery Prompt Learning (FPL)**:
    - **Function**: Adapting the general vision-language capabilities of CLIP to deepfake detection to generate pixel-level forgery attention maps.
    - **Mechanism**: Designing Universal Forgery Prompts (UF-prompts) that contain two types of learnable tokens: the universal forgery tokens $[\mathbf{v}^G]$ to capture common patterns across different forgery methods, and the specific forgery tokens $[\mathbf{v}^S]$ generated from the CLIP global image feature $\mathbf{g}^I$ via an MLP to encode image-specific forgery clues. The UF-prompt format is $\mathbf{S} = [\mathbf{v}_1^G]..[\mathbf{v}_m^G][\mathbf{v}_1^S]..[\mathbf{v}_u^S]\text{[forged][face]}$. Simultaneously, shallow-to-deep layer-wise learnable forgery tokens (LF-tokens) are introduced into each layer of the frozen CLIP text encoder. The final output global text embedding $\mathbf{g}^T$ and the patch features $\mathbf{F}_I$ of the CLIP image encoder are used to compute the patch-wise cosine similarity, yielding the forgery attention map $\mathbf{M}_b$.
    - **Design Motivation**: (a) The division of labor between universal and specific tokens allows the model to simultaneously learn cross-method commonalities and image-specific features; (b) using the fixed text "forged face" instead of varying class names stabilizes training; (c) unlike CoOp/CoCoOp, which perform global classification, FPL innovatively targets pixel-level forgery localization; (d) LF-tokens protect the pre-trained weights while enhancing task adaptation.

2. **Bridge Adapter (Bri-Ada)**:
    - **Function**: Bridging the CLIP image encoder and the deepfake encoder, fusing general recognition capability with domain-specific forgery knowledge.
    - **Mechanism**: Consisting of Transformer encoder blocks, it takes the intermediate layer features from the CLIP image encoder $\mathcal{E}_I$ and the deepfake encoder $\mathcal{E}_D$ as input. The output feature maps of both are concatenated as $\mathbf{F}^0 \in \mathbb{R}^{w \times h \times c}$ and then weighted by the forgery attention map $\mathbf{M}_b$ generated from FPL: $\mathbf{f}^0 = \text{AVGPOOL}(\text{CONV}(\mathbf{F}^0 \odot \mathbf{M}_b))$ to obtain the final detection representation.
    - **Design Motivation**: (a) The CLIP image encoder is pre-trained on large-scale web data and possesses inherent generalization capability, but lacks forgery domain knowledge; conversely, the deepfake encoder has domain knowledge but weak generalization. Bri-Ada leverages the strengths of both; (b) using $\mathbf{M}_b$ as a spatial prior guides the model to focus on forged regions, establishing a mutually beneficial cycle between FPL and Bri-Ada.

3. **Forgery Explanation Module (Frequency-domain token + LLM)**:
    - **Function**: Converting detection results into human-readable textual explanations.
    - **Mechanism**: Transforming the detection representation $\mathbf{F}^0$ output by Bri-Ada into frequency-domain tokens $\mathbf{H}_F \in \mathbb{R}^{N \times D}$, and converting the CLIP image encoder output into visual tokens $\mathbf{H}_V$. Both are concatenated with text tokens $\mathbf{H}_T$ and fed into the LLM to autoregressively generate textual explanations: $p(\mathbf{X}_A | \mathbf{H}_V, \mathbf{H}_F, \mathbf{H}_T) = \prod_{z=1}^{Z} p_\theta(\mathbf{x}_z | \mathbf{H}_V, \mathbf{H}_F, \mathbf{H}_{T,<z}, \mathbf{x}_{A,<z})$.
    - **Design Motivation**: (a) Frequency-domain tokens carry deepfake-specific domain knowledge (as real and fake images differ significantly in the frequency domain), informing the LLM whether the image is fake; (b) visual tokens provide facial appearance details to aid descriptive narrative; (c) unlike directly using MLLMs for detection (such as DDVQA-BLIP), M2F2-Det first detects via specialized mechanisms and then explains via the LLM, so that detection and explanation enhance each other.

### Loss & Training

Three-stage training:
1. **Stage 1**: Train the deepfake encoder $\mathcal{E}_D$ + FPL (UF-prompts + LF-tokens), minimizing the binary cross-entropy. The CLIP encoder is frozen.
2. **Stage 2**: Align the visual tokens $\mathbf{H}_V$ and frequency-domain tokens $\mathbf{H}_F$ with the LLM input space, training only the MLP projection layers and freezing other components.
3. **Stage 3**: Train the MLP layers + LLM (using LoRA), maximizing the text-generation likelihood.

EfficientNet-B4 is used as the deepfake encoder, CLIP ViT-L/14-336 as the CLIP encoder, and Vicuna-7B as the LLM. The DD-VQA dataset (14,782 QA pairs) is utilized for the second and third stage training.

## Key Experimental Results

### Main Results

| Dataset | Metric | M2F2-Det | Prev. SOTA | Gain |
|--------|------|----------|----------|------|
| FF++ (c23) | Acc/AUC | 98.79/99.34 | 98.65/99.87 (TALL) | Acc+0.14 |
| FF++ (c40) | Acc/AUC | 93.83/96.58 | 92.82/94.57 (TALL) | Acc+1.01, AUC+2.01 |
| Celeb-DF | Acc/AUC | 98.98/99.92 | 98.59/99.94 (RECCE) | Acc+0.39 |
| WDF | Acc/AUC | 86.05/93.14 | 83.25/92.02 (RECCE) | Acc+2.80, AUC+1.12 |

Cross-dataset generalization (trained on FF++, tested on other datasets):

| Dataset | AUC | Prev. SOTA | Gain |
|--------|-----|----------|------|
| DFDC | 87.80 | 87.56 (FreqBlender) | +0.24 |
| FFIW | 88.70 | 86.14 (FreqBlender) | +2.56 |
| Celeb-DF | 95.10 | 95.40 (LAA-Net) | -0.30 |

Textual explanation generation (DD-VQA dataset):

| Method | Decision Acc | Decision F1 |
|------|--------|--------|
| DDVQA-BLIP | 87.49 | 90.07 |
| Fine-tuned LLaVA | 86.41 | 92.10 |
| M2F2-Det | **95.23** | **96.61** |

### Ablation Study

| Configuration | FF++(c40) AUC | Celeb-DF AUC | Description |
|------|-------------|-------------|------|
| Baseline (EfficientNet-B4) | 91.03 | 65.78 | w/o CLIP |
| +LF-tokens | 92.57 | 67.37 | +1.54 |
| +UF-prompts | 92.66 | 66.08 | +1.63 |
| +Full FPL | 93.65 | 68.68 | +2.62 |
| +Bri-Ada (w/o FPL) | 93.80 | 70.71 | +4.93 Generalization improvement |
| +UF-prompts+Bri-Ada | 94.20 | 71.08 | Mutual benefit effect |
| Full M2F2-Det | **96.58** | **74.82** | Collaboration of all components |

Frequency token ablation: Removing $\mathbf{H}_F$ drops the decision accuracy from 95.23% to 85.11% (-10.12%).

### Key Findings

- **FPL significantly outperforms general prompt learning**: FPL's AUC is 9.31% higher than CoOp and 8.13% higher than CoCoOp, because general prompt learning is designed to recognize semantic categories rather than forgery patterns.
- **Generalization provided by CLIP is key to cross-dataset performance**: Bri-Ada brings a +4.93% AUC improvement on Celeb-DF, since the pre-training of CLIP on massive internet data reduces overfitting to specific forgery patterns.
- **Mutual enhancement between detection and explanation**: M2F2-Det achieves a decision accuracy of 95.23%, vastly outperforming the explanation-only DDVQA-BLIP (87.49%), as the detection mechanism feeds deepfake domain knowledge into the generator.
- The forgery attention map $\mathbf{M}_b$ is learned unsupervised (using only binary classification labels) but can precisely localize fake regions.

## Highlights & Insights

- **The first unified framework to excel at both detection and explanation**: Previously, one had to choose between "accurate detection but uninterpretable" or "interpretable but poor detection accuracy".
- **Elegant design of universal/specific tokens in FPL**: Universal tokens capture commonalities (such as boundary discontinuities present across most forgeries), while specific tokens capture distinct characteristics (such as eye blurring caused by specific synthesis methods). This division of labor matches the requirement for both generalization and specificity in forgery detection.
- **Ingenious introduction of frequency-domain tokens**: Capitalizing on the significant discrepancies in high-frequency signals between synthetic and real faces, this provides the LLM with diagnostic evidence beyond RGB visual content.
- The three-stage training strategy ensures stable learning across components, avoiding the instability of full end-to-end training.

## Limitations & Future Work

- Cross-domain generalization on the DFD dataset is inferior to AUNet, possibly because AUNet utilizes prior knowledge of facial Action Units (AUs).
- The LLM used (Vicuna-7B) is relatively small; upgrading to larger models could improve explanation quality.
- The dataset for training explanations only comes from DD-VQA (~14K pairs), which is limited in volume.
- Temporal consistency cues at the video level have not been explored.
- The forgery attention map is unsupervised, which might limit its performance upper bound.

## Related Work & Insights

- FPL can be viewed as an extension of CoOp/CoCoOp to pixel-level tasks, providing a paradigm for other CLIP adaptation tasks requiring spatial precision.
- The fusion strategy of "frozen general encoder + domain-specific encoder" in Bridge Adapter can be generalized to other tasks requiring a balance between generalization and specialization.
- The idea of frequency-domain tokens can inspire other domains to introduce non-RGB features to LLMs for multimodal reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ First unified framework for detection+explanation, with novel designs in FPL and frequency-domain tokens.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 6 detection datasets + 1 explanation dataset, covering both in-domain and out-of-domain scenarios, with exhaustive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-explained relationships between components.
- Value: ⭐⭐⭐⭐ Introduces an interpretability dimension to deepfake detection, presenting high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants](../../NeurIPS2025/multimodal_vlm/face-human-bench_a_comprehensive_benchmark_of_face_and_human_understanding_for_m.md)
- [\[CVPR 2025\] MMRL: Multi-Modal Representation Learning for Vision-Language Models](mmrl_multi-modal_representation_learning_for_vision-language_models.md)
- [\[CVPR 2025\] EgoLM: Multi-Modal Language Model of Egocentric Motions](egolm_multi-modal_language_model_of_egocentric_motions.md)
- [\[ICLR 2026\] Can Vision-Language Models Answer Face to Face Questions in the Real-World?](../../ICLR2026/multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)
- [\[CVPR 2025\] Vision-Language Model IP Protection via Prompt-based Learning](vision-language_model_ip_protection_via_prompt-based_learning.md)

</div>

<!-- RELATED:END -->
