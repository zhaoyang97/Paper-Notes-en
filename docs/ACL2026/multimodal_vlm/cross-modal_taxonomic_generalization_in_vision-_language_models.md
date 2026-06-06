---
title: >-
  [Paper Note] Cross-Modal Taxonomic Generalization in (Vision-) Language Models
description: >-
  [ACL 2026][Multimodal VLM][Cross-modal generalization] This paper systematically investigates whether the language model (LM) component in Vision-Language Models (VLMs) can generalize taxonomic knowledge (hypernym relati…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Cross-modal generalization"
  - "taxonomic knowledge"
  - "hypernyms"
  - "vision-language models"
  - "visual coherence"
date: 2026-05-08
content_hash: 70b375577da56cfc
---

# Cross-Modal Taxonomic Generalization in (Vision-) Language Models

**Conference**: ACL 2026  
**arXiv**: [2603.07474](https://arxiv.org/abs/2603.07474)  
**Code**: [https://github.com/sally-xu-42/cross-modal-taxonomic-gen](https://github.com/sally-xu-42/cross-modal-taxonomic-gen)  
**Area**: Causal Inference  
**Keywords**: Cross-modal generalization, taxonomic knowledge, hypernyms, vision-language models, visual coherence

## TL;DR
This paper systematically investigates whether the language model (LM) component in Vision-Language Models (VLMs) can generalize taxonomic knowledge (hypernym relations) learned from pure text to visual inputs across modalities. It finds that even without hypernym labels during training, pretrained LMs can identify hypernym categories in images, provided there is visual coherence among category members.

## Background & Motivation

**Background**: Modern VLMs align visual and linguistic representations by learning a mapping (projector) between a frozen image encoder and a frozen LM. Recent studies have observed an "LM dominance" phenomenon where the LM component often overrides the image encoder, sometimes producing correct answers even without relying on visual input.

**Limitations of Prior Work**: While this "LM dominance" is typically viewed as a defect in VLMs (especially for tasks requiring precise perception), it also suggests a possibility for knowledge acquired through language to transfer across modalities. However, the boundaries and mechanisms of this knowledge transfer remain poorly understood.

**Key Challenge**: Can taxonomic knowledge (e.g., "a parrot is a bird") learned by an LM from text extend directly to the visual modality without any vision-language hypernym supervision? If so, is this generalization arbitrary (a rule-based "IF crow THEN bird") or does it require a certain level of consistency in the visual input?

**Goal**: To systematically test the cross-modal generalization capability and boundary conditions of taxonomic knowledge learned by LMs within VLMs.

**Key Insight**: By utilizing a controlled experimental design—systematically removing varying amounts of hypernym labels during VLM projector training—one can test whether the model still recognizes those removed hypernym categories.

**Core Idea**: Cross-modal taxonomic generalization does exist, but it is not an arbitrary rule executed by the language model. Instead, it requires visual coherence of category members within the representation space as a prerequisite.

## Method

### Overall Architecture
A simplified VLM is constructed: a frozen DINOv2 image encoder + a trainable MLP projector + a frozen pretrained LM (Qwen3-0.6B/1.7B). The training task is visual binary question answering: "Is there a {category} in this image?". The visibility of hypernym labels in the training data is systematically manipulated.

### Key Designs

1.  **Random Hypernym Ablation**:
    - **Function**: Tests generalization ability when partial hypernym evidence is removed.
    - **Mechanism**: For 53 hypernym labels, 10%-100% of leaf-node-image-to-hypernym mappings are randomly removed. For example, removing (parrot, bird) means the model never sees the "bird" label for parrot images, though it may still see it for crow images. At 100%, the model has seen zero hypernym labels.
    - **Design Motivation**: To simulate a continuum from full hypernym supervision to zero hypernym supervision, accurately measuring how generalization changes as evidence decreases.

2.  **Systematic Hypernym Ablation**:
    - **Function**: Tests generalization when entire hypernym categories are completely removed.
    - **Mechanism**: Completely removes 10-53 hypernyms from the training data (no leaf-node-image pairs for these hypernyms contain the hypernym label). At 100%, both ablation methods are equivalent.
    - **Design Motivation**: More rigorous than random ablation—the model doesn't just lack partial evidence but has never encountered certain hypernyms at all.

3.  **Counterfactual Shuffling Experiments**:
    - **Function**: Distinguishes between "arbitrary rule-based generalization" and "visual coherence-dependent generalization."
    - **Mechanism**: Two counterfactual datasets are designed. "Across-category shuffling" breaks visual coherence by shuffling image-to-leaf-node mappings (e.g., mapping crow to an image of a kayak). "Within-category shuffling" shuffles only within the same category (e.g., mapping crow to a penguin image), maintaining visual coherence. If the LM performs arbitrary rules ("IF crow THEN bird"), performance should be similar under both.
    - **Design Motivation**: If members of the "bird" category are mapped to visually unrelated objects (kayaks, hummus, etc.), will the LM still generalize? This directly tests whether generalization depends on input signal consistency.

### Loss & Training
The training objective is standard next-word prediction, where loss is calculated only at the answer position (yes/no). Only the projector is trained; the LM and image encoder remain frozen. The THINGS database is used (1,216 categories, 17,336 images, 53 hypernyms).

## Key Experimental Results

### Main Results (100% Hypernym Ablation → Zero Hypernym Supervision)

| Model | Held-Out Hypernyms F1 | Majority Label Baseline |
| :--- | :--- | :--- |
| Qwen3-0.6B (Pretrained) | ~60 | 46.7 |
| Qwen3-1.7B (Pretrained) | ~68 | 46.7 |
| Qwen3-0.6B (Random Init) | ~48 (near chance) | 46.7 |

### Counterfactual Experiments

| Configuration | Held-Out Hypernyms F1 Trend |
| :--- | :--- |
| Original Data | Higher than baseline, decreases slowly as ablation increases |
| Within-Category Shuffling | Almost no difference from original data |
| Across-Category Shuffling | Drops significantly, near chance |

### Key Findings
- Pretrained LMs perform significantly above chance even under zero hypernym supervision, confirming the existence of cross-modal taxonomic generalization.
- Randomly initialized LMs perform near chance under zero hypernym supervision, proving that generalization stems from linguistic knowledge acquired during pretraining.
- No significant difference is found between using DINOv2 (no text training) and SigLIP (text training) as image encoders, indicating that generalization originates from the LM rather than the image encoder.
- Generalization collapses under across-category shuffling but holds under within-category shuffling, proving that visual coherence is a necessary condition for generalization.
- Larger LMs (1.7B vs. 0.6B) and hypernyms with higher visual coherence show better generalization effects.

## Highlights & Insights
- **Exquisite Experimental Design**: By controlling variables (ablation ratio, shuffling method, LM initialization, encoder type), the contribution of each factor is systematically isolated. The design is of textbook quality.
- **Visual Coherence as a Bridge**: The LM does not blindly execute the rule "IF crow THEN bird"; rather, it requires that visual members of the "bird" category actually cluster together in the representation space. This suggests that cross-modal generalization is a collaborative result of linguistic knowledge and perceptual consistency.
- **Empirical Support for the "Platonic Representation Hypothesis"**: Representations learned by unimodal models in different modalities tend to converge. When counterfactual data disrupts this convergence, generalization fails, providing indirect support for this hypothesis.

## Limitations & Future Work
- Only taxonomic knowledge (hypernym-hyponym relations) was tested, leaving other semantic relations unexplored.
- The use of a simplified projector architecture and binary classification tasks differs from full VLM training pipelines.
- The scale of 53 hypernyms and 1,216 categories is relatively limited.
- The impact of hypernym depth (e.g., animal > bird > parrot hierarchy) on generalization was not explored.

## Related Work & Insights
- **vs. "Platonic Representation Hypothesis"**: This hypothesis suggests that representations of models across modalities are converging. This paper provides new evidence through counterfactual experiments—generalization fails when conceptual alignment between modalities is destroyed.
- **vs. Controlled Rearing Experiments**: This research paradigm is similar to training LMs on controlled corpora to test generalization but extends the approach to a cross-modal setting.
- **vs. VLM LM-bias Research**: Reinterprets the "LM dominance" phenomenon from a "bug" into a "feature" that can be leveraged.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Novel research question and exquisite experimental design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely systematic control experiments across multiple models, ablations, and shuffling methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, moving seamlessly from hypothesis to experiment to conclusion.
- Value: ⭐⭐⭐⭐ Provides important insights into the nature of LM knowledge and cross-modal transfer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models](../../AAAI2026/multimodal_vlm/cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)
- [\[ACL 2026\] Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models](cross-cultural_expert-level_art_critique_evaluation_with_vision-language_models.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[ICML 2026\] VEENA: Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow](../../ICML2026/multimodal_vlm/interpreting_and_enhancing_emotional_circuits_in_large_vision-language_models_vi.md)
- [\[ACL 2026\] Topology-Aware Layer Pruning for Large Vision-Language Models](topology-aware_layer_pruning_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
