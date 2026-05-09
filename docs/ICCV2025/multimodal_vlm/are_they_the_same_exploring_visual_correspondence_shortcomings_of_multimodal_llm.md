---
title: >-
  [Paper Note] Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs
description: >-
  [ICCV 2025][Multimodal VLM][visual correspondence] This paper presents the first systematic study of visual correspondence matching deficiencies in multimodal large language models (MLLMs). The authors construct the MMVM benchmark (1,510 samples) and a 220K matching dataset, and propose CoLVA, which leverages object-level contrastive learning and a fine-grained visual expert to substantially improve cross-image instance matching in MLLMs.
tags:
  - ICCV 2025
  - Multimodal VLM
  - visual correspondence
  - multimodal large language models
  - contrastive learning
  - visual matching
  - benchmark
date: 2026-05-08
content_hash: cc9b166aee02d063
---

# Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs

**Conference**: ICCV 2025
**arXiv**: [2501.04670](https://arxiv.org/abs/2501.04670)
**Code**: [https://zhouyiks.github.io/projects/CoLVA/](https://zhouyiks.github.io/projects/CoLVA/)
**Area**: Multimodal VLM
**Keywords**: visual correspondence, multimodal large language models, contrastive learning, visual matching, benchmark

## TL;DR

This paper presents the first systematic study of visual correspondence matching deficiencies in multimodal large language models (MLLMs). The authors construct the MMVM benchmark (1,510 samples) and a 220K matching dataset, and propose CoLVA, which leverages object-level contrastive learning and a fine-grained visual expert to substantially improve cross-image instance matching in MLLMs.

## Background & Motivation

Despite remarkable progress in visual perception, reasoning, and vision-language understanding, the ability of MLLMs to perform visual matching—a fundamental capability in computer vision—has received little attention. The authors find that even strong models such as GPT-4o perform poorly on simple cross-image object matching tasks. Experiments show that the current strongest open-source MLLM, Qwen2-VL-72B, achieves only 38% overall accuracy on the MMVM benchmark, revealing systematic deficiencies in visual correspondence understanding.

Through analysis, the authors identify two core causes: (1) existing MLLMs possess basic capabilities for recognizing object appearance and location, but lack correspondence data that teaches them how to leverage these capabilities for visual matching; and (2) MLLMs rely on CLIP-based encoders and therefore cannot extract fine-grained, discriminative visual features—which are critical for distinguishing semantically similar candidate objects. PCA visualizations further confirm this: matching targets and other candidate objects cluster together in feature space, far from the query object.

## Method

### Overall Architecture

CoLVA is built upon InternVL2 and introduces two core techniques: (1) a fine-grained visual expert with object-level contrastive learning (OCL), and (2) an instruction augmentation strategy (IA). Training proceeds in two stages: a pre-training stage that aligns the feature space of the additional visual encoder, followed by an SFT stage that fine-tunes the model using MMVM data.

### Key Designs

1. **MMVM Dataset and Benchmark Construction**:

    - Frame pairs are sampled from five video segmentation datasets, and existing annotations are used to generate 220K multiple-choice matching QA pairs.
    - InternVL2-76B automatically generates chain-of-thought-style matching rationales for each QA pair.
    - The benchmark set contains 1,510 manually annotated samples covering eight matching cues (color, shape/pose, text/logo markings, size, relative position, orientation/motion, binding relationships, and object markings).
    - The number of candidate options ranges from 2 to 37, with an average of 10.

2. **Object-Level Contrastive Learning (OCL)**:

    - Contrastive learning is performed between the MLLM visual encoder and an additional visual expert (RADIO), rather than through a shared tracking head.
    - Object-level representations are extracted via Masked Average Pooling.
    - The contrastive loss is defined as: $\mathcal{L} = \frac{\exp(\mathbf{O} \cdot \mathbf{O}^+)}{\exp(\mathbf{O} \cdot \mathbf{O}^+) + \sum_{\mathbf{O}^-} \exp(\mathbf{O} \cdot \mathbf{O}^-)}$
    - **Design Motivation**: The visual expert learns discriminative features within the MLLM's semantic space while achieving modality alignment. Applying OCL directly on the MLLM's CLIP backbone yields limited gains (34.05 vs. 32.38), as CLIP lacks fine-grained features.

3. **Fine-Grained Visual Expert (RADIO)**:

    - RADIO is distilled from multiple visual foundation models including SAM, DINOv2, and CLIP, combining fine-grained features with vision-language alignment capability.
    - During pre-training, the InternVL2 visual encoder and RADIO are frozen; only the RADIO Adapter is trained.
    - In an input image pair, one image passes through the MLLM's original encoder and the other through RADIO, with OCL applied on the resulting object-level representations.
    - Pseudo-video data from image segmentation datasets is used to augment the limited supply of image pairs with segmentation annotations.

4. **Instruction Augmentation (IA)**:

    - **Original approach**: Objects are referenced by overlaying highlighted contours and ID labels on the image; gradients cannot directly propagate back to image features.
    - **Augmented approach**: Object-level representations are directly embedded into the instruction ("object-1: \<Obj 1\>, object-2: \<Obj 2\>, ..."), enabling gradients to flow directly back to the corresponding image features.
    - Both formats are used randomly during training to enhance the model's instruction-following ability for multi-object references.

### Loss & Training

- **Pre-training stage**: Object-level contrastive loss to align RADIO and MLLM feature spaces.
- **SFT stage**: LLaVA SFT data (665K) + MMVM data (220K) with a standard autoregressive training objective.

## Key Experimental Results

### Main Results

| Model | Params | Overall | CL | RP |
|-------|--------|---------|-----|-----|
| InternVL2-4B | 4B | 17.62 | 14.73 | 10.28 |
| Qwen2-VL-72B | 72B | 38.08 | 37.64 | 32.28 |
| GPT-4o | — | 42.65 | 39.28 | 32.28 |
| **CoLVA-InternVL2-4B** | **4B** | **49.80** | **42.72** | **44.86** |

With only 4B parameters, CoLVA surpasses GPT-4o by 7.15% and Qwen2-VL-72B by 11.72%.

### Ablation Study

| Configuration | Overall Acc. | Gain |
|---------------|-------------|------|
| InternVL2-4B (baseline) | 17.62 | — |
| + MMVM data | 32.38 | +14.76 |
| + data + OCL (w/o VE) | 34.05 | +1.67 |
| + data + VE (w/o OCL) | 32.25 | −0.13 |
| + data + OCL + VE | 40.45 | +8.07 |
| + data + OCL + VE + IA | **45.83** | **+5.38** |

### Key Findings

- MMVM data alone contributes the largest gain (+14.76%), confirming that data scarcity is the primary cause of poor matching capability.
- OCL and VE exhibit strong synergy (+8.07%), while VE alone is nearly ineffective (−0.13%), indicating that contrastive learning is a prerequisite for VE to function.
- Instruction augmentation contributes an additional +5.38%; the key factor is enabling gradients to propagate directly to object-level features.
- CoLVA does not degrade general VQA ability, and even achieves improvements on benchmarks such as MME, POPE, and NaturalBench.
- The method transfers to Qwen2VL-2B (47.48%) and LLaVA1.5-7B (36.56%).

## Highlights & Insights

- This work is the first to systematically define and benchmark visual correspondence matching in MLLMs, filling a gap in the field.
- The dual-encoder contrastive learning design is elegant: rather than applying contrastive learning within the same encoder, the auxiliary encoder learns discriminative features in the semantic space of the primary encoder, balancing fine-grained representation and alignment.
- The automated data generation pipeline is well-designed: rather than having MLLMs perform matching directly (a task at which they underperform), the pipeline uses them to summarize visual cues based on provided annotations.

## Limitations & Future Work

- The benchmark size (1,510 samples) is relatively limited; validation at larger scale remains to be explored.
- Pre-training requires image pairs with segmentation annotations, which are costly to obtain (though partially mitigated by pseudo-video data).
- Although the 4B model surpasses 72B and GPT-4o counterparts, absolute accuracy remains below 50%, indicating that visual matching remains highly challenging.

## Related Work & Insights

- The work is conceptually related to traditional correspondence learning methods such as visual tracking and re-identification, but applying these ideas within an MLLM framework is a novel direction.
- The successful use of RADIO as a multi-model distillation product highlights the potential of incorporating complementary visual encoders into MLLMs.
- The instruction augmentation strategy (embedding object representations directly into text instructions) may be applicable to other MLLM tasks that require fine-grained object referencing.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to define the visual correspondence matching problem and construct a complete data, benchmark, and method framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes benchmarking of 36 models, multi-dimensional ablations, and cross-model transfer validation—highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-articulated problem definition and motivation.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new research direction in visual correspondence understanding for MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](controlling_multimodal_llms_via_reward-guided_decoding.md)
- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[ICCV 2025\] Analyzing Finetuning Representation Shift for Multimodal LLMs Steering](analyzing_finetuning_representation_shift_for_multimodal_llms_steering.md)

</div>

<!-- RELATED:END -->
