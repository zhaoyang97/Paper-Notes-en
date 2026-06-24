---
title: >-
  [Paper Note] Visual Modality Prompt for Adapting Vision-Language Object Detectors
description: >-
  [ICCV 2025][Object Detection][Visual Prompt] This paper proposes ModPrompt, an encoder-decoder-based visual prompting strategy that adapts vision-language object detectors (e.g., YOLO-World, Grounding DINO) to new modalities such as infrared and depth, while preserving zero-shot detection capability.
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Visual Prompt"
  - "Modality Adaptation"
  - "Vision-Language Detectors"
  - "Zero-Shot Detection"
  - "Cross-Modal Transfer"
date: 2026-05-08
content_hash: 7f1dd9ea48163247
---

# Visual Modality Prompt for Adapting Vision-Language Object Detectors

**Conference**: ICCV 2025
**arXiv**: [2412.00622](https://arxiv.org/abs/2412.00622)  
**Code**: [GitHub](https://github.com/heitorrapela/ModPrompt)  
**Area**: Object Detection
**Keywords**: Visual Prompt, Modality Adaptation, Vision-Language Detectors, Zero-Shot Detection, Cross-Modal Transfer

## TL;DR

This paper proposes ModPrompt, an encoder-decoder-based visual prompting strategy that adapts vision-language object detectors (e.g., YOLO-World, Grounding DINO) to new modalities such as infrared and depth, while preserving zero-shot detection capability.

## Background & Motivation

Vision-language object detectors (e.g., YOLO-World, Grounding DINO) have demonstrated strong zero-shot detection performance on RGB images by fusing textual semantics with visual features. However, when a large modality shift occurs at test time—such as transitioning from RGB to infrared or depth images—the performance of these detectors degrades significantly.

Existing adaptation approaches suffer from the following limitations:

**Full Fine-tuning**: Although it improves detection accuracy on the target modality, it leads to catastrophic forgetting and loss of zero-shot detection capability.

**Conventional Visual Prompts**: These apply the same linear prompt transformation to every image (e.g., fixed patches, random padding) regardless of input content, yielding limited effectiveness under large modality shifts.

**Image Translation Methods (e.g., HalluciDet, ModTr)**: These are designed only for traditional detectors and do not explore cross-modal adaptation for vision-language detectors; moreover, some methods discard pretrained knowledge.

The core motivation is: without modifying detector parameters, can a learnable, input-conditioned visual prompt "translate" images from a new modality into pseudo-RGB representations more interpretable to the detector, thereby simultaneously achieving high detection accuracy and preserving zero-shot capability?

## Method

### Overall Architecture

The core idea of ModPrompt is to perform modality adaptation in the input space at the pixel level. The pipeline is as follows: a target-modality image $x$ is passed through a learnable encoder-decoder network $h_\vartheta$ to generate a visual prompt, which is then added to the original image to form a pseudo-RGB image fed into the frozen vision-language detector. Additionally, a Modality Prompt Decoupled Residual (MPDR) mechanism is introduced to perform decoupled adaptation in the text embedding space.

### Key Designs

1. **ModPrompt (Modality Prompt Encoder-Decoder)**:

    - Function: Dynamically generates visual prompts conditioned on the input image to achieve pixel-level modality translation.
    - Mechanism: A U-Net-based encoder-decoder structure (with MobileNet or ResNet as the backbone) maps the input image to a 3-channel prompt image with output values constrained to $[0,1]$. The training objective is the detection loss rather than a reconstruction loss:
    $\mathcal{C}_{\text{mp}}(\vartheta) = \frac{1}{|\mathcal{D}|}\sum_{(x,Y)\in\mathcal{D}} \mathcal{L}_{det}(f_\theta(x + h_\vartheta(x)), Y)$
      where $f_\theta$ denotes the frozen detector and $h_\vartheta(x)$ is the input-conditioned visual prompt.
    - Design Motivation: Unlike fixed visual prompts, ModPrompt is input-conditioned—different images produce different prompts—enabling better enhancement of target regions and suppression of background noise, which is particularly beneficial under large modality gaps.

2. **MPDR (Modality Prompt Decoupled Residual)**:

    - Function: Performs efficient modality adaptation in the text embedding space while preserving original zero-shot knowledge.
    - Mechanism: Text embeddings for each target category are precomputed offline; a set of trainable residual parameters $\phi$ is then learned and added to the frozen embeddings. The overall training objective is:
    $\mathcal{C}_{\text{mp-tp}}(\vartheta, \phi) = \mathcal{C}_{\text{mp}}(\vartheta) + \mathcal{C}_{\text{tp}}(\phi)$
    - Design Motivation: Through the decoupling strategy, MPDR can be disabled at test time via zero-masking to recover full zero-shot embedding knowledge, or enabled to use adapted embeddings, with no additional inference overhead.

3. **Detector-Agnostic Design**:

    - Function: Enables flexible integration of ModPrompt into vision-language detectors of different architectures.
    - Mechanism: Since the prompt operates at the input pixel level rather than the feature level, it is agnostic to the backbone type (CNN or Transformer).
    - Design Motivation: Most prior methods are tightly coupled to specific detectors, whereas ModPrompt is applicable to both YOLO-World (CNN backbone + CLIP) and Grounding DINO (Swin Transformer + BERT).

### Loss & Training

- The training loss is the detection loss $\mathcal{L}_{det}$ of the original detector, comprising classification and regression losses.
- Only the encoder-decoder parameters $\vartheta$ and the MPDR parameters $\phi$ are trained; all detector parameters remain frozen.
- YOLO-World is trained for 80 epochs; Grounding DINO is trained for 60 epochs.
- Text embeddings are extracted offline using CLIP-ViT-base-patch32 (for YOLO-World) and BERT-base-uncased (for Grounding DINO).

## Key Experimental Results

### Main Results

| Dataset | Method | YOLO-World AP50 | YOLO-World AP | Grounding DINO AP50 | Grounding DINO AP |
|-------|------|----------------|--------------|--------------------|--------------------|
| LLVIP-IR | Zero-Shot | 81.00 | 53.20 | 85.50 | 56.50 |
| LLVIP-IR | Full FT | 97.43 | 67.73 | 97.17 | 67.83 |
| LLVIP-IR | Visual Prompt (WM) | 82.00 | 50.90 | 69.57 | 40.77 |
| LLVIP-IR | **ModPrompt** | **92.80** | **62.87** | **93.13** | **60.10** |
| NYUv2-Depth | Zero-Shot | 4.80 | 3.00 | 8.30 | 5.30 |
| NYUv2-Depth | Full FT | 49.90 | 33.57 | 51.60 | 35.77 |
| NYUv2-Depth | **ModPrompt** | **37.17** | **24.93** | **21.70** | **14.13** |

ModPrompt substantially outperforms all visual prompt baselines on both infrared and depth modalities, approaching full fine-tuning performance on LLVIP.

### Ablation Study

| Configuration | LLVIP AP50 | COCO AP50 | Average | Notes |
|------|-----------|----------|------|------|
| Zero-Shot | 81.00 | 51.90 | 66.45 | Baseline |
| Full FT | 97.43 | 0.10 | 48.77 | Zero-shot capability lost |
| Head FT | 93.57 | 0.66 | 47.12 | Catastrophic forgetting |
| WM | 87.47 | 51.90 | 69.69 | Zero-shot preserved but limited accuracy |
| **ModPrompt** | **95.63** | **51.90** | **73.77** | High accuracy + full zero-shot |

- Trainable parameters: ModPrompt requires only 3.08M, far fewer than Full FT's 76.81M.
- A MobileNet backbone achieves detection performance close to ResNet, making it more suitable for real-time applications.
- MPDR consistently provides additional gains across nearly all visual prompting strategies (+0.6 to +8.0 AP50).

### Key Findings

1. Conventional visual prompts (fixed patches, random patches) can even underperform zero-shot in modality adaptation scenarios, as they are not conditioned on the input image content.
2. The visual prompts generated by ModPrompt produce artifacts on the image that enhance target regions and suppress background.
3. Zero-shot performance on NYUv2 (depth) is extremely low (AP of only 3–5%), highlighting the severe challenge that RGB-pretrained models face in cross-modal scenarios.

## Highlights & Insights

- **Strong Practicality**: ModPrompt achieves accuracy close to full fine-tuning while preserving zero-shot capability—highly valuable for real-world deployment where a single model must handle both RGB and novel modality tasks.
- **Elegant Simplicity**: The input-conditioned visual prompting concept is intuitive and effective; repurposing U-Net for detection-guided image translation is a clever design shift.
- **First Systematic Study**: To the best of our knowledge, this is the first work systematically focused on adapting VLM-based detectors to new visual modalities.

## Limitations & Future Work

1. A performance gap relative to full fine-tuning remains for fine-grained localization (AP75 and AP), especially for small objects.
2. Validation is currently limited to infrared and depth modalities; SAR, thermal grayscale imagery, and others are not explored.
3. The encoder-decoder introduces additional inference latency; although the MobileNet variant is lightweight, the overhead is non-zero.
4. Experiments are conducted only on LLVIP (pedestrian category) and NYUv2 (indoor scenes), which are relatively limited in dataset scale and category diversity.

## Related Work & Insights

- In contrast to text- or feature-level prompting methods such as Co-op and VPT, ModPrompt operates at the pixel level, making it better suited for handling large modality shifts.
- HalluciDet and ModTr pioneered the image translation paradigm that inspires ModPrompt, but neither exploits the advantages of vision-language models.
- The residual decoupled embedding learning idea behind MPDR is drawn from Task Residual, but is applied here for the first time to text embedding adaptation in object detectors.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic exploration of cross-modal adaptation for VLM-based detectors, with a concise and effective design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two detectors, three datasets, multiple baselines, and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and method presentation is intuitive.
- Value: ⭐⭐⭐⭐ Practically significant for multi-modal detection deployment; open-source code enables reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision](evrt-detr_latent_space_adaptation_of_image_detectors_for_event-based_vision.md)
- [\[AAAI 2026\] Correcting False Alarms from Unseen: Adapting Graph Anomaly Detectors at Test Time](../../AAAI2026/object_detection/correcting_false_alarms_from_unseen_adapting_graph_anomaly_detectors_at_test_tim.md)
- [\[AAAI 2026\] T-Rex-Omni: Integrating Negative Visual Prompt in Generic Object Detection](../../AAAI2026/object_detection/t-rex-omni_integrating_negative_visual_prompt_in_generic_object_detection.md)
- [\[ICCV 2025\] Visual-RFT: Visual Reinforcement Fine-Tuning](visual-rft_visual_reinforcement_fine-tuning.md)
- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)

</div>

<!-- RELATED:END -->
