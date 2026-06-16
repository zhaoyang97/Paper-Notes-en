---
title: >-
  [Paper Note] Cross-Modal Taxonomic Generalization in (Vision-) Language Models
description: >-
  [ACL 2026][Multimodal VLM][Vision-Language Model] This paper systematically investigates whether the language model (LM) in a Vision-Language Model (VLM) can generalize taxonomic knowledge (hypernym relations) learned from pure text to visual inputs. It finds that even without hypernym labels during training, pretrained LMs can identify hypernym categories in images,
tags:
  - ACL 2026
  - Multimodal VLM
  - Vision-Language Model
date: 2026-05-08
content_hash: 3ae6280accd2c9c3
---
# Cross-Modal Taxonomic Generalization in (Vision-) Language Models

**Conference**: ACL 2026  
**arXiv**: [2603.07474](https://arxiv.org/abs/2603.07474)  
**Code**: [https://github.com/sally-xu-42/cross-modal-taxonomic-gen](https://github.com/sally-xu-42/cross-modal-taxonomic-gen)  
**Area**: Causal Inference  
**Keywords**: Cross-modal generalization, taxonomic knowledge, hypernymy, vision-language models, visual coherence

## TL;DR
This paper systematically investigates whether the language model (LM) in a Vision-Language Model (VLM) can generalize taxonomic knowledge (hypernym relations) learned from pure text to visual inputs. It finds that even without hypernym labels during training, pretrained LMs can identify hypernym categories in images, though this generalization requires visual coherence among category members.

## Background & Motivation

**Background**: Modern VLMs align visual and linguistic representations by learning a mapping (projector) between a frozen image encoder and a frozen LM. Recent research has found that the LM component in VLMs often dominates the image encoder, sometimes providing correct answers without relying on visual input.

**Limitations of Prior Work**: This "LM dominance" phenomenon is typically viewed as a defect in VLMs (especially for tasks requiring precise perception). However, it also reveals the possibility that knowledge acquired from language can be transferred across modalities. The boundaries and mechanisms of this knowledge transfer remain unclear.

**Key Challenge**: Can taxonomic knowledge learned by an LM from text (e.g., "a parrot is a bird") extend directly to the visual modality without any visual-linguistic hypernym supervision? If so, is this generalization arbitrary (rule-based, like "IF crow THEN bird") or does it require some form of consistency in the visual input?

**Goal**: To systematically test the cross-modal generalization capability of taxonomic knowledge learned by the LM in VLMs and its boundary conditions.

**Key Insight**: Through a controlled experimental design—systematically removing varying amounts of hypernym labels during the training of the VLM projector—the researchers test whether the model can still recognize the categories for which labels were removed.

**Core Idea**: Cross-modal taxonomic generalization does exist, but it is not an arbitrary rule executed by the language model. Instead, it requires the visual representations of category members to possess visual coherence as a prerequisite.

## Method

### Overall Architecture
A simplified VLM was constructed using a frozen DINOv2 image encoder, a trainable MLP projector, and a frozen pretrained LM (Qwen3-0.6B/1.7B). The training task is a visual binary question-answering task: "Is there a {category} in this picture?". The visibility of hypernym labels in the training data is systematically manipulated.

### Key Designs

**1. Random Hypernym Ablation: Continuously removing hypernym evidence at the leaf node granularity to plot the decay curve of generalization as supervision decreases.**

To answer whether the taxonomic knowledge of the LM can extend cross-modally, the amount of hypernym evidence the model sees must be precisely controlled. The authors randomly removed 10%–100% of "leaf node–image–hypernym" mappings for 53 hypernym labels. For instance, after removing (parrot, bird), the model no longer sees the "bird" label for parrot images but still sees it for crow images. When the ablation reaches 100%, the model has seen no hypernym labels for any images. This creates a spectrum from "full hypernym supervision" to "zero hypernym supervision," allowing the decay of generalization to be measured point-by-point.

**2. Systematic Hypernym Ablation: Entire categories of hypernyms are removed as a more rigorous stress test than random ablation.**

While random ablation only removes evidence for some leaf nodes, the model might still infer relations from other leaf nodes under the same hypernym. Systematic ablation is stricter—it completely removes 10–53 hypernyms from the training data. These hypernyms do not appear for any of their constituent leaf node-image pairs, meaning the model has zero exposure to these categories during visual training. This tests whether the LM can recognize an entire category concept based purely on language priors when it has never appeared in visual training.

**3. Counterfactual Shuffling Experiments: Distinguishing between "arbitrary rule-based generalization" and "generalization dependent on visual coherence."**

Even if generalization is confirmed, the mechanism remains to be explained: is the LM blindly following symbolic rules like "IF crow THEN bird," or does it rely on consistency in the input signal? The authors used two counterfactual datasets: "Cross-category shuffling" disrupts the image-leaf node mapping and completely destroys visual coherence (e.g., mapping "crow" to images of kayaks). "Within-category shuffling" only shuffles images within the same category (e.g., mapping "crow" to images of penguins), preserving visual coherence. If generalization is based on arbitrary rules, performance should be similar under both conditions. If it collapses under cross-category shuffling but holds under within-category shuffling, it proves that generalization requires visual coherence—the visual clustering of category members.

### Loss & Training
The training objective is standard next-word prediction, with loss calculated only at the answer position (yes/no). Only the projector is trained, while the LM and image encoder remain frozen. The THINGS database (1,216 categories, 17,336 images, 53 hypernyms) is utilized.

## Key Experimental Results

### Main Results (100% Hypernym Ablation → Zero Hypernym Supervision)

| Model | Held-Out Hypernyms F1 | Majority Label Baseline |
| :--- | :--- | :--- |
| Qwen3-0.6B (Pretrained) | ~60 | 46.7 |
| Qwen3-1.7B (Pretrained) | ~68 | 46.7 |
| Qwen3-0.6B (Randomly Initialized) | ~48 (Near chance) | 46.7 |

### Ablation Study (Counterfactual Experiments)

| Configuration | Held-Out Hypernyms F1 Trend |
| :--- | :--- |
| Original Data | Higher than baseline, slow decline as ablation increases |
| Within-category Shuffling | Nearly no difference from original data |
| Cross-category Shuffling | Significant drop, near chance |

### Key Findings
- Pretrained LMs perform significantly above chance even with zero hypernym supervision, confirming the existence of cross-modal taxonomic generalization.
- Randomly initialized LMs perform near chance under zero hypernym supervision, proving that generalization stems from linguistic knowledge acquired during pretraining.
- No significant difference was observed when using DINOv2 (no text training) versus SigLIP (with text training) as the image encoder, indicating that the generalization originates from the LM rather than the image encoder.
- Generalization collapses under cross-category shuffling but is maintained under within-category shuffling, proving that visual coherence is a necessary condition for generalization.
- Larger LMs (1.7B vs. 0.6B) and hypernyms with higher visual coherence demonstrate better generalization effects.

## Highlights & Insights
- **Sophisticated Experimental Design**: The use of controlled variables (ablation ratio, shuffling methods, LM initialization, encoder types) to systematically isolate the contribution of each factor is exemplary.
- **Visual Coherence as a Bridge**: The LM does not blindly execute "IF crow THEN bird" rules; it requires the visual members of the "bird" category to cluster within the representation space. This suggests that cross-modal generalization is a result of synergy between linguistic knowledge and perceptual coherence.
- **Empirical Support for the "Platonic Representation Hypothesis"**: The hypothesis suggests that representations learned by unimodal models converge across modalities. The failure of generalization when counterfactual data destroys this conceptual alignment provides indirect support for this hypothesis.

## Limitations & Future Work
- Only taxonomic knowledge (hypernym-hyponym relations) was tested, leaving other semantic relations unexplored.
- The use of a simplified projector architecture and binary classification tasks differs from full VLM training pipelines.
- The scale of 53 hypernyms and 1,216 categories is relatively limited.
- The impact of taxonomic depth (e.g., animal > bird > parrot hierarchies) on generalization was not explored.

## Related Work & Insights
- **vs. "Platonic Representation Hypothesis"**: This paper provides new evidence through counterfactual experiments—generalization fails when conceptual alignment between modalities is disrupted.
- **vs. Controlled Rearing Experiments**: Similar to the research paradigm of training LMs on controlled corpora to test generalization, but extended to a cross-modal setting.
- **vs. VLM LM-bias Research**: Reinterprets the LM dominance phenomenon from a "bug" into a "feature" that can be leveraged.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Novel research question and exquisite experimental design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely systematic controlled variable experiments across multiple models and ablation/shuffling conditions.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, moving seamlessly from hypothesis to experiment to conclusion.
- Value: ⭐⭐⭐⭐ Important insights into the nature of LM knowledge and cross-modal transfer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models](../../AAAI2026/multimodal_vlm/cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)
- [\[ACL 2026\] Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models](cross-cultural_expert-level_art_critique_evaluation_with_vision-language_models.md)
- [\[ICML 2025\] Vision-Language Models Create Cross-Modal Task Representations](../../ICML2025/multimodal_vlm/vision-language_models_create_cross-modal_task_representations.md)
- [\[ECCV 2024\] Quantized Prompt for Efficient Generalization of Vision-Language Models](../../ECCV2024/multimodal_vlm/quantized_prompt_for_efficient_generalization_of_vision-language_models.md)
- [\[ACL 2026\] Topology-Aware Layer Pruning for Large Vision-Language Models](topology-aware_layer_pruning_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
