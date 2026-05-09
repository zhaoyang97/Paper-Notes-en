---
title: >-
  [Paper Note] T-Rex-Omni: Integrating Negative Visual Prompt in Generic Object Detection
description: >-
  [AAAI 2026][Object Detection][Open-set detection] This paper proposes T-Rex-Omni, the first framework to systematically incorporate negative visual prompts into open-set object detection. Through a training-free NNC module and an NNH loss, it substantially narrows the performance gap between visual-prompt and text-prompt detection methods, with particularly strong results in long-tail scenarios (LVIS-minival APr of 51.2).
tags:
  - AAAI 2026
  - Object Detection
  - Open-set detection
  - negative visual prompts
  - long-tail recognition
  - visual prompting
  - zero-shot detection
date: 2026-05-08
content_hash: 59f2370d0e1a6191
---

# T-Rex-Omni: Integrating Negative Visual Prompt in Generic Object Detection

**Conference**: AAAI 2026
**arXiv**: [2511.08997](https://arxiv.org/abs/2511.08997)
**Code**: None (built on T-Rex2 architecture)
**Area**: Object Detection
**Keywords**: Open-set detection, negative visual prompts, long-tail recognition, visual prompting, zero-shot detection

## TL;DR

This paper proposes T-Rex-Omni, the first framework to systematically incorporate negative visual prompts into open-set object detection. Through a training-free NNC module and an NNH loss, it substantially narrows the performance gap between visual-prompt and text-prompt detection methods, with particularly strong results in long-tail scenarios (LVIS-minival APr of 51.2).

## Background & Motivation

### Evolution of Open-Set Object Detection

Object detection has progressed from closed-set (predefined categories) to open-set settings, where targets are specified via user prompts. Prompt modalities include:
- **Text prompts**: e.g., "a photo of a muffin," leveraging semantic understanding from CLIP/BERT
- **Visual prompts**: reference images or cropped regions indicating the target, which are more intuitive

### Core Problem: Fragility of Positive-Only Prompts

Existing open-set detectors rely solely on positive prompts—telling the model what to detect, without specifying what to avoid. This introduces a fundamental weakness:

**Models fail easily when confronted with visually similar but semantically distinct distractors.** The classic example is Chihuahuas vs. muffins—their appearances are remarkably similar, and a positive-only detector prompted with "Chihuahua" may incorrectly detect muffins as Chihuahuas.

This problem is further exacerbated under **long-tail distributions**, where rare categories have limited training data and models exhibit weaker discriminative ability for such classes.

### Research Question

**Can negative visual prompts enable models to actively exclude hard negatives without compromising detection of true positives?**

## Method

### Overall Architecture

T-Rex-Omni is built upon the T-Rex2 architecture with the text-prompt branch removed, and introduces three core innovations:

1. **Unified positive-negative visual prompt encoder**: jointly processes both positive and negative visual prompts
2. **NNC module** (Negating Negative Computing): training-free probability calibration
3. **NNH loss** (Negating Negative Hinge): discriminative margin constraint in embedding space

The image encoder and DETR-style decoder from T-Rex2 are retained.

### Key Designs

#### 1. **Positive-Negative Visual Prompt Encoder**: Unified Mapping from Coordinate Space to Embedding Space

**Visual Prompt Generation**:
- **Positive prompts**: GT boxes undergo mild perturbation (scaling/translation within $[0, 0.3]$) to preserve semantic validity
- **Negative prompts**: GT boxes undergo strong perturbation (scaling within $[0.7, 1.0]$) to generate $K$ negative prompts
- This strategy improves prompt robustness to spatial and scale variations

**Encoding Process**:
- Initialize learnable queries $Q_P \in \mathbb{R}^{1 \times D}$ and $Q_N \in \mathbb{R}^{K \times D}$
- Process via multi-scale deformable cross-attention:

$$Q_P' = \text{MSDeformAttn}(Q_P, p_c, F)$$
$$Q_N' = \text{MSDeformAttn}(Q_N, n_c, F)$$

- Self-attention and FFN yield final positive embedding $V_P$ and negative embedding $V_N$

**Cross-Image Detection**: By ensuring each training batch contains at least one shared category, the method computes the mean positive prompt embedding $V_P''$ over shared categories to enable cross-image propagation. For negative embeddings, the Top-$K$ most similar to the mean positive embedding are selected.

**Three Flexible Inference Modes**:
- **User-specified mode**: user explicitly provides positive and negative samples
- **Auto-suggestion mode**: the system automatically generates negative prompts from user-provided positive prompts (default evaluation mode)
- **Positive-only mode**: compatible with conventional positive-prompt workflows

#### 2. **NNC Module** (Negating Negative Computing): Training-Free Negative Response Suppression

**Mechanism**: Dynamically suppresses negative responses during probability computation, requiring no additional training.

Given $N_q$ detection queries $Q$ output by the DETR decoder, positive and negative similarities are computed as:

$$S_P, S_{N,i} = Q \times (V_P'')^T, \quad Q \times (V_{N,i}'')^T$$

The detection probability is obtained by subtracting a weighted maximum negative similarity from the positive similarity before applying Sigmoid:

$$Prob = \sigma(S_P - B \cdot \beta \cdot \max_{i=1,...,K}(S_{N,i}))$$

where $\beta \in (0, 1)$ controls the influence of negative prompts (optimal value: 0.3), and $B \sim \text{Bernoulli}(0.5)$ is a random mode switch that alternates between joint positive-negative and positive-only modes during training to ensure inference compatibility.

**Key Advantage**: As a plug-and-play module, it delivers +3.0 AP on COCO-val and +3.2 AP on LVIS-minival without any fine-tuning.

#### 3. **NNH Loss** (Negating Negative Hinge Loss): Discriminative Margin in Embedding Space

To enhance the discriminability between positive and negative embeddings, NNH enforces a margin constraint in embedding space:

$$\mathcal{L}_{Hinge} = \sum_{i=1,...,K} \text{Max}(0, S_{N,i} - S_P + \eta) / K$$

where $\eta > 0$ is a preset margin (optimal value: 0.3), ensuring the positive similarity exceeds any negative similarity by at least $\eta$. The hinge formulation penalizes only margin violations, thereby focusing on hard negatives.

### Loss & Training

**Total Loss**:

$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \mathcal{L}_{Hinge} + \mathcal{L}_{L1} + \mathcal{L}_{GIoU} + \mathcal{L}_{DN}$$

This includes Focal classification loss, NNH hinge loss, L1 and GIoU regression losses, and DINO denoising training loss.

**Training Strategy**:
- "Prompt from current image, detect across images" strategy (distinct from T-Rex2's "prompt and detect within the same image")
- Each batch is guaranteed to contain at least one shared category to enhance cross-image consistency
- AdamW optimizer: learning rate $10^{-5}$ for backbone, $10^{-4}$ for other components
- Fine-tuned on Objects365 using 8× A100 GPUs

## Key Experimental Results

### Main Results

**Zero-Shot Generic Object Detection (Table 1, Swin-T backbone)**:

| Method | Prompt Type | COCO-val AP | LVIS-minival AP | LVIS APr | ODinW APavg | Roboflow APavg |
|--------|-------------|-------------|-----------------|----------|-------------|----------------|
| T-Rex2 | Text | 45.8 | 42.8 | 37.4 | 18.0 | 8.2 |
| T-Rex2 | Visual | 38.8 | 37.4 | 29.9 | 23.6 | 17.4 |
| **T-Rex-Omni** | **Visual** | **43.6** | **43.0** | **37.0** | **25.2** | **18.9** |

**Swin-L backbone**:

| Method | Prompt Type | COCO-val AP | LVIS-val AP | LVIS-val APr | ODinW APavg |
|--------|-------------|-------------|-------------|--------------|-------------|
| T-Rex2 | Text | 52.2 | 45.8 | 42.7 | 22.0 |
| T-Rex2 | Visual | 46.5 | 45.3 | 43.8 | 27.8 |
| LLMDet | Text | — | 42.0 | 31.6 | — |
| **T-Rex-Omni** | **Visual** | **50.7** | **47.8** | **45.1** | **29.6** |

T-Rex-Omni (Swin-L) even surpasses the text-prompt method T-Rex2 on LVIS-val by +2.0 AP.

### Ablation Study

**Contributions of NNC and NNH (Table 2)**:

| NNC | NNH | Fine-tune | COCO AP | LVIS AP | LVIS APr |
|-----|-----|-----------|---------|---------|----------|
| ✗ | ✗ | ✗ | 38.8 | 37.4 | 29.9 |
| ✓ | ✗ | ✗ | 41.8 (+3.0) | 40.6 (+3.2) | 33.3 (+3.4) |
| ✓ | ✗ | ✓ | 42.9 (+4.1) | 41.4 (+4.0) | 35.1 (+5.2) |
| ✓ | ✓ | ✓ | **43.6 (+4.8)** | **43.0 (+5.6)** | **37.0 (+7.1)** |

**Key Hyperparameters**:
- NNC $\beta$: optimal at 0.3; performance degrades when too high ($\geq 0.5$) or too low (0.0)
- NNH $\eta$: optimal at 0.3; moderate margin yields the best results
- Number of negative prompts: optimal at 3; diminishing returns observed at 5
- Number of positive prompts: optimal at 1; additional prompts introduce noise and reduce performance

### Key Findings

1. **NNC is a plug-and-play performance booster**: delivers +3.0 AP on COCO without any training
2. **Negative prompts yield the largest gains for long-tail scenarios**: LVIS rare-category APr improves from 29.9 to 37.0 (+23.8% relative gain)
3. **Random mode switching outperforms fixed-mode training**: training with $B \sim \text{Bernoulli}(0.5)$ produces a model that is more robust across both inference modes
4. **A single high-quality positive prompt outperforms multiple prompts**: additional positive prompts introduce noise and hurt performance
5. **The visual-text prompt gap is substantially narrowed**: on COCO, the gap between visual and text prompts shrinks from 7.0 to 2.2 AP

## Highlights & Insights

1. **Paradigm innovation**: the first systematic integration of negative visual prompts into open-set detection, addressing a long-overlooked dimension
2. **Plug-and-play design**: the NNC module delivers significant gains without any training, making it highly practical for engineering deployment
3. **Multi-mode inference**: supports user-specified, auto-suggestion, and positive-only modes, flexibly accommodating diverse application scenarios
4. **Breakthrough performance on long-tail scenarios**: 51.2 APr on LVIS-minival substantially surpasses prior methods, demonstrating that negative prompts are especially critical for discriminating rare categories
5. **Visual prompts surpassing text prompts**: on LVIS-val, T-Rex-Omni with purely visual prompts outperforms text-prompt methods—a compelling result

## Limitations & Future Work

1. **Slight degradation on counting tasks**: MAE of 13.76 on FSC147, compared to 8.72 for T-Rex2, suggesting negative prompts may interfere with dense small-object counting
2. **Automation of negative prompt generation**: the auto-suggestion mode relies on simple geometric transformations; more intelligent hard negative mining strategies are worth exploring
3. **Absence of joint text + visual prompt experiments**: although the text branch is removed, combining positive-negative visual prompts with textual descriptions could yield further improvements
4. **Sourcing true negative samples**: in real deployments, users may not know which samples constitute effective negative prompts

## Related Work & Insights

- **T-Rex2** (Jiang et al. 2024): the base architecture for T-Rex-Omni, supporting both text and visual prompts
- **Focal Loss** (Lin et al. 2017): a classical approach to handling hard negatives during training via up-weighted loss
- **NP-RepMet**: jointly optimizes positive and negative prototypes for few-shot detection
- **UNP**: isolates confusing negative samples via gradient modulation

**Insight**: In few-shot and open-set scenarios, "telling the model what is not the target" is equally important as "telling the model what is the target." This principle can be generalized to other visual tasks such as segmentation and tracking.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (systematic integration of negative visual prompts is an entirely new direction)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 benchmarks, multiple backbones, comprehensive ablations, hyperparameter sensitivity analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear motivation, rigorous method description, in-depth experimental analysis)
- Value: ⭐⭐⭐⭐⭐ (paradigm-level contribution to open-set detection with high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](../../ICCV2025/object_detection/visual_modality_prompt_for_adapting_vision-language_object_detectors.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[CVPR 2026\] PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training](../../CVPR2026/object_detection/pet-dino_unifying_visual_cues_into_grounding_dino_with_prompt-enriched_training.md)
- [\[ACL 2026\] Evolutionary Negative Module Pruning for Better LoRA Merging](../../ACL2026/object_detection/evolutionary_negative_module_pruning_for_better_lora_merging.md)
- [\[AAAI 2026\] Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning](connecting_the_dots_training-free_visual_grounding_via_agent.md)

</div>

<!-- RELATED:END -->
