---
title: >-
  [Paper Note] MDiff4STR: Mask Diffusion Model for Scene Text Recognition
description: >-
  [AAAI 2026][Image Generation][Scene Text Recognition] This work is the first to introduce Mask Diffusion Models (MDM) into Scene Text Recognition (STR)…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Scene Text Recognition"
  - "Mask Diffusion Model"
  - "Denoising Strategy"
  - "Token Replacement Noise"
  - "Omnidirectional Language Modeling"
date: 2026-05-08
content_hash: ce95e6071f26433c
---

# MDiff4STR: Mask Diffusion Model for Scene Text Recognition

**Conference**: AAAI 2026 Oral  
**arXiv**: [2512.01422](https://arxiv.org/abs/2512.01422)  
**Code**: [https://github.com/Topdu/OpenOCR](https://github.com/Topdu/OpenOCR)  
**Area**: Image Generation
**Keywords**: Scene Text Recognition, Mask Diffusion Model, Denoising Strategy, Token Replacement Noise, Omnidirectional Language Modeling

## TL;DR

This work is the first to introduce Mask Diffusion Models (MDM) into Scene Text Recognition (STR), proposing MDiff4STR. It addresses the training-inference noising gap via six training mask strategies and resolves overconfident predictions through a Token Replacement Noise mechanism. With only 3 denoising steps, MDiff4STR surpasses state-of-the-art autoregressive models in accuracy while achieving 3× inference speedup.

## Background & Motivation

Scene Text Recognition (STR) is a core task in OCR systems, facing challenges such as curved text, occlusion, blur, and artistic fonts. Existing STR methods fall into four main paradigms:

**Autoregressive Models (ARM)**: Strong sequential modeling capability and high accuracy, but inefficient due to character-by-character decoding.

**Parallel Decoding Models (PDM)**: Fast inference but lack contextual modeling, yielding lower accuracy than ARM.

**BERT-like Refinement Models (ReM)**: Parallel prediction followed by bidirectional refinement, but sensitive to errors in the initial prediction.

**Mask Diffusion Models (MDM)**: An emerging paradigm that learns omnidirectional dependencies by recovering original sequences from partially masked inputs.

**Potential of MDM**: Unlike the unidirectional modeling of ARM, MDM captures more flexible and comprehensive omnidirectional contextual dependencies, which is critical for STR tasks requiring linguistic understanding. Its denoising process is also efficient and controllable, yielding accurate predictions in just a few steps.

**Two key problems when directly applying MDM to STR**:

**Training-Inference Noising Gap**: Training uses random masking, but inference starts from fully masked input; subsequent remask patterns also differ from training, leading to poor generalization.

**Overconfident Predictions at Inference**: MDM tends to assign excessively high confidence scores to incorrect predictions (e.g., reporting a confidence of 0.95 for the wrong character "F"), causing the confidence-based remask mechanism to fail and preventing errors from being corrected in subsequent steps.

## Method

### Overall Architecture

MDiff4STR comprises:
- **Visual Encoder**: SVTRv2, designed specifically for STR, extracting image features $\mathbf{F}_v \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{4} \times D}$
- **Character Embedding Layer**: Maps the noised character sequence to vectors
- **Mask Diffusion Decoder (MDiffDecoder)**: Performs denoising conditioned on visual features
- **Classifier**: Maps decoded tokens back to characters

### Key Designs

#### 1. **Vanilla MDM Baseline and Multiple Decoding Paradigms**

The flexibility of MDM supports multiple decoding strategies:

- **MDiff-PD** (Parallel Decoding): Full mask → one-step decoding
- **MDiff-AR** (Autoregressive Decoding): Left-to-right progressive unmasking
- **MDiff-Re** (Refinement Decoding): BERT-like bidirectional refinement
- **MDiff-LC** (Low-Confidence Remask): Remasking tokens below average confidence at each step
- **MDiff-BLC** (Block-wise Low-Confidence Remask): Low-confidence remasking within fixed-size blocks, avoiding the "confidence trap" where certain tokens are repeatedly remasked

#### 2. **Six Training Mask Strategies (Bridging the Noising Gap)**

To eliminate the training-inference noising gap, all remask patterns used during inference are incorporated into training. While the original MDM trains solely with random masking, MDiff4STR uniformly samples from the following 7 strategies during training:

| Strategy | Description |
|-----|------|
| (a) Random Mask | Original MDM training strategy |
| (b) Full Mask | Initial state at the first inference step |
| (c) Forward Autoregressive | Retain tokens from left to right |
| (d) Backward Autoregressive | Retain tokens from right to left |
| (e) BERT-like Refinement | Retain most tokens, mask a few |
| (f) Low-Confidence Remask | Simulates inference-time remasking |
| (g) Block-wise Low-Confidence Remask | A remask strategy specific to this work |

**The full mask strategy contributes most**, as it forms the foundation of the first step in all inference denoising chains.

#### 3. **Token Replacement Noise Mechanism (Resolving Overconfidence)**

This is the core innovation of this work. Beyond conventional mask noise, a novel noise type is introduced:

Certain characters in the original sequence $\mathbf{Y}$ are **randomly replaced with other characters** to construct an erroneous sequence $\mathbf{Y}_r$, simulating inference scenarios with "high-confidence but incorrect" predictions. The model must learn to:
- Identify which tokens are erroneous (without knowing which ones were replaced)
- Correct those errors

$$\tilde{\mathbf{T}} = \text{MDiffDecoder}(\mathbf{F}_v, \mathbf{T}_r), \quad \tilde{\mathbf{Y}} = \text{Classifier}(\tilde{\mathbf{T}})$$

Key distinction:
- Denoising training supervises only masked positions
- Correction training supervises **all positions** (since it is unknown at inference which tokens are erroneous)

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{denoising} + \mathcal{L}_{correction}$$

**Denoising Loss**:

$$\mathcal{L}_{denoising} = -\frac{1}{l_1}\sum_{i=1}^{L}\mathbf{1}[\mathbf{Y}_{l_1}^i = \mathbf{M}]\log p_\theta(\mathbf{Y}^i | \mathbf{Y}_{l_1})$$

**Correction Loss**:

$$\mathcal{L}_{correction} = -\frac{1}{L}\sum_{i=1}^{L}\log p_\theta(\mathbf{Y}^i | \mathbf{Y}_{l_2})$$

Training configuration: AdamW (weight decay 0.05), LR $5 \times 10^{-4}$, batch size 1024, one-cycle LR scheduler, 40 epochs for English / 100 epochs for Chinese, 4× RTX 3090 GPUs, maximum text length 25.

## Key Experimental Results

### Main Results

**English STR (trained from scratch, U14M-Filter)**:

| Method | Type | Com Avg | U14M Avg | OST | Inference Time (ms) |
|-----|------|---------|---------|-----|------------|
| SVTRv2-B | CTC | 96.57 | 86.14 | 80.0 | 19.8 |
| PARSeq | ReM | 96.40 | 84.26 | 79.9 | 23.8 |
| MAERec | ARM | 96.36 | 85.17 | 76.4 | 35.7 |
| IGTR | ARM | 96.48 | 84.86 | 76.3 | 24.1 |
| ARMbase (Ours baseline) | ARM | 96.88 | 87.34 | 81.03 | 57.95 |
| **MDiff4STR-B-BLC** | **MDM** | **97.30** | **88.44** | **84.25** | **19.21** |

MDiff4STR-B-BLC surpasses the previous best by 0.73% / 2.30% / 4.30% on Com / U14M / OST respectively, with inference speed **3× faster** than ARM.

**Pre-training followed by fine-tuning**:

| Method | Com Avg | OST |
|-----|---------|-----|
| SVTRv2-B (pre-trained) | 97.83 | 86.9 |
| E2STR | 97.71 | 80.7 |
| CLIP4STR | 97.32 | 82.8 |
| **MDiff4STR-BLC (pre-trained)** | **98.02** | **87.4** |

**Chinese STR (BCTR)**:

| Method | Scene | Web | Doc | HW | Avg |
|-----|-------|-----|-----|----|----|
| MAERec | 84.4 | 83.0 | 99.5 | 65.6 | 83.13 |
| SVTRv2-B | 83.5 | 83.3 | 99.5 | 67.0 | 83.31 |
| **MDiff4STR-B-BLC** | **85.7** | **84.7** | **99.6** | **67.0** | **84.25** |

### Ablation Study

**Core component ablation** (MDiff-BLC, Base scale):

| Configuration | Com | U14M | OST | Gain |
|------|-----|------|-----|------|
| Vanilla MDM (random mask) | 96.42 | 85.42 | 79.93 | Baseline |
| + Six training mask strategies | 96.98 (+0.57) | 87.09 (+1.67) | 81.92 (+2.00) | Bridges noising gap |
| + Token replacement noise | **97.30 (+0.88)** | **88.44 (+3.02)** | **84.25 (+4.33)** | Resolves overconfidence |

**Denoising steps vs. accuracy**:

| Steps K | Com (BLC) | U14M (BLC) | OST (BLC) | Inference Time (ms) |
|--------|-----------|-----------|----------|------------|
| 1 | 96.88 | 86.69 | 81.31 | 10.52 |
| 2 | 97.19 | 88.05 | 83.69 | 15.56 |
| **3** | **97.30** | **88.44** | **84.25** | **19.21** |
| 5 | 97.28 | 88.50 | 84.42 | 25.70 |
| 8 | 97.24 | 88.65 | 84.11 | 32.74 |

$K=3$ achieves the optimal accuracy-efficiency trade-off; additional steps yield marginal or no improvement.

**Incremental ablation of the six mask strategies**:

| Baseline | +Full Mask | +Forward AR | +Backward AR | +ReM | +LC | +BLC |
|------|--------|--------|--------|------|-----|------|
| 85.42 | +1.04 | +1.37 | +1.41 | +1.46 | +1.53 | +1.67 |

Each strategy contributes positively, with full mask yielding the largest gain.

### Key Findings

1. **The omnidirectional context modeling advantage of MDM is most pronounced in occluded scenarios**: OST improvement of 4.30% (from scratch) / 3.22% (vs. ARMbase), demonstrating that MDM's omnidirectional dependency modeling significantly outperforms unidirectional ARM and bidirectional ReM.
2. **Token replacement noise mechanism is highly effective**: Gains are larger on more challenging datasets (Com +0.88% vs. U14M +3.02% vs. OST +4.33%).
3. **Only 3 denoising steps suffice to surpass ARM**: This demonstrates the efficiency advantage of the MDM paradigm in STR.
4. **MDM flexibly supports multiple decoding paradigms** (PD / AR / Re / LC / BLC), with its dedicated BLC strategy performing best.
5. **Effective for both Chinese and English**: Achieves state-of-the-art on the Chinese BCTR benchmark (Scene +1.3%, Web +1.5%).

## Highlights & Insights

- **Paradigm-level innovation**: MDM is introduced to STR, establishing a new paradigm parallel to ARM.
- **Precise problem diagnosis**: The two critical bottlenecks of applying vanilla MDM to STR (noising gap + overconfidence) are accurately identified, with targeted solutions proposed for each.
- **Token replacement noise** is a highly generalizable concept: applicable not only to STR but extensible to any MDM application scenario.
- **The advantage of omnidirectional language modeling** is especially prominent in inference-demanding scenarios such as occlusion and artistic fonts.
- **Accuracy and efficiency achieved simultaneously**: 3-step denoising = 3× ARM speed + higher accuracy.

## Limitations & Future Work

- The current MDM relies on a fixed separation channel $M$ and a fixed maximum text length of 25; adaptability to variable-length text remains to be validated.
- The low-confidence remask strategy may still be limited in extreme cases where all predictions are high-confidence but entirely incorrect.
- Joint optimization of the visual encoder and MDM decoder has not been explored; the SVTRv2 encoder and MDM decoder are currently designed independently.
- The replacement ratio and strategy for Token Replacement Noise can be further fine-tuned.
- Validation on visually rich documents such as handwriting and historical manuscripts is insufficient.

## Related Work & Insights

This work establishes a four-paradigm taxonomy of STR methods (CTC / ARM / PDM / ReM) and introduces MDM as a fifth paradigm. Key insights:

- **MDM is not limited to NLP**: This work demonstrates that MDM exhibits strong potential in vision-language tasks as well.
- **Noise design is central to MDM**: Unlike diffusion models where noise is typically treated as an auxiliary element, MDM performance is highly dependent on the design of noise strategies.
- **Error correction capability**: Token replacement noise endows the model with a "self-correction" ability that is difficult to achieve in autoregressive models, where errors in auto-regressive decoding propagate forward.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First to introduce MDM to STR + Token Replacement Noise mechanism)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multi-benchmark evaluation in English/Chinese + 5 decoding strategies + detailed ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear and coherent logical chain from problem diagnosis to solution design)
- Value: ⭐⭐⭐⭐⭐ (Establishes a new paradigm for STR while simultaneously improving both accuracy and efficiency)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)
- [\[ICLR 2026\] Consistent Text-to-Image Generation via Scene De-Contextualization](../../ICLR2026/image_generation/consistent_text-to-image_generation_via_scene_de-contextualization.md)
- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](../../ICCV2025/image_generation/lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [\[ICML 2026\] Self-Prompting Diffusion Transformer for Open-Vocabulary Scene Text Editing via In-Context Learning](../../ICML2026/image_generation/self-prompting_diffusion_transformer_for_open-vocabulary_scene_text_editing_via_.md)
- [\[NeurIPS 2025\] SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency](../../NeurIPS2025/image_generation/scenedecorator_towards_scene-oriented_story_generation_with_scene_planning_and_s.md)

</div>

<!-- RELATED:END -->
