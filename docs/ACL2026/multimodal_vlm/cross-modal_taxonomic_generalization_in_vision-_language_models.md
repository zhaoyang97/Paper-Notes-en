---
title: >-
  [Paper Note] Cross-Modal Taxonomic Generalization in (Vision-) Language Models
description: >-
  [ACL 2026][Multimodal VLM][Cross-modal generalization] This paper systematically investigates whether language models in VLMs can generalize taxonomic knowledge (hypernym relations) learned from text to visual inputs. It finds that even without hypernym labels during training, pretrained LMs recognize hypernym categories in images, provided there is visual coherence among category members.
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Cross-modal generalization"
  - "taxonomic knowledge"
  - "hypernyms"
  - "vision-language models"
  - "visual coherence"
date: 2026-05-08
content_hash: dd600739baa2e994
---

# Cross-Modal Taxonomic Generalization in (Vision-) Language Models

**Conference**: ACL 2026  
**arXiv**: [2603.07474](https://arxiv.org/abs/2603.07474)  
**Code**: [https://github.com/sally-xu-42/cross-modal-taxonomic-gen](https://github.com/sally-xu-42/cross-modal-taxonomic-gen)  
**Area**: Causal Inference  
**Keywords**: Cross-modal generalization, taxonomic knowledge, hypernyms, vision-language models, visual coherence

## TL;DR
This paper systematically investigates whether language models in VLMs can generalize taxonomic knowledge (hypernym relations) learned from text to visual inputs. It finds that even without hypernym labels during training, pretrained LMs recognize hypernym categories in images, provided there is visual coherence among category members.

## Background & Motivation

**Background**: Modern VLMs align visual and linguistic representations by learning a mapping (projector) between a frozen image encoder and a frozen LM. Recent studies observe "LM dominance," where the LM component often ignores visual input to provide answers.

**Limitations of Prior Work**: While "LM dominance" is often viewed as a flaw in precision tasks, it suggests that knowledge acquired through language might transfer cross-modally. However, the boundaries and mechanisms of this transfer remain unclear.

**Key Challenge**: Does taxonomic knowledge learned from text (e.g., "parrots are birds") extend to the visual modality without explicit visual-linguistic hypernym supervision? If so, is this generalization arbitrary (symbolic rules like IF crow THEN bird) or does it require visual consistency?

**Goal**: To systematically test the cross-modal generalization capability of taxonomic knowledge in VLMs and its boundary conditions.

**Key Insight**: Designing controlled experiments by systematically removing hypernym labels during projector training to test if the model still recognizes those categories.

**Core Idea**: Cross-modal taxonomic generalization exists but is not an arbitrary rule; it requires visual coherence among category members as a prerequisite.

## Method

### Overall Architecture
A simplified VLM was constructed using a frozen DINOv2 image encoder, a trainable MLP projector, and a frozen pretrained LM (Qwen3-0.6B/1.7B). The training task is a binary visual Q&A: "Is there a {category} in this image?". The visibility of hypernym labels in training data is systematically manipulated.

### Key Designs

**1. Random Hypernym Ablation: Drawing curves of generalization decay by removing hypernym evidence at the leaf-node level.**

To measure how much evidence is needed, 10%–100% of "leaf node-image-hypernym" mappings for 53 hypernym labels were removed. For instance, after removing (parrot, bird), the model sees no "bird" label for parrot images but still sees it for crows. At 100%, the model has never seen a hypernym label for any image. This allows plotting generalization across a spectrum from full to zero supervision.

**2. Systematic Hypernym Ablation: Stress testing by removing entire hypernym categories.**

Random ablation might allow inference from other leaf nodes. Systematic ablation completely removes 10–53 hypernyms from training. The model has zero exposure to these categories during visual training, testing if it can recognize them via language priors alone.

**3. Counterfactual Shuffling Experiments: Distinguishing rule-based generalization from visual coherence-based generalization.**

Whether the LM follows a symbolic rule (IF crow THEN bird) or relies on input consistency is tested using counterfactual data. "Cross-category shuffling" breaks visual coherence by mapping leaf nodes to unrelated objects (e.g., crow to kayak). "Within-category shuffling" keeps coherence by shuffling within the same hypernym (e.g., crow to penguin). If performance collapses under cross-shuffling but holds under within-shuffling, visual coherence is the bridge.

### Loss & Training
The training objective is standard next-word prediction, calculating loss only on the answer token (yes/no). Only the projector is tuned. The THINGS database (1,216 categories, 17,336 images, 53 hypernyms) is used.

## Key Experimental Results

### Main Results (100% Hypernym Ablation → Zero Hypernym Supervision)

| Model | Held-Out Hypernyms F1 | Majority Label Baseline |
|------|----------------------|-------------|
| Qwen3-0.6B (Pretrained) | ~60 | 46.7 |
| Qwen3-1.7B (Pretrained) | ~68 | 46.7 |
| Qwen3-0.6B (Random Init) | ~48 (near chance) | 46.7 |

### Counterfactual Experiments

| Configuration | Held-Out Hypernyms F1 Trend |
|------|--------------------------|
| Original Data | Higher than baseline, slow decay with ablation |
| Within-category Shuffle | Nearly identical to original |
| Cross-category Shuffle | Significant drop, near chance |

### Key Findings
- Pretrained LMs significantly outperform chance even with zero hypernym supervision, confirming cross-modal taxonomic generalization.
- Randomly initialized LMs perform near chance, proving generalization stems from pre-existing language knowledge.
- Choice of image encoder (DINOv2 vs. SigLIP) shows no significant difference, suggesting the LM is the source of generalization.
- Generalization collapses with cross-category shuffling but holds with within-category shuffling, indicating visual coherence is necessary.
- Larger LMs (1.7B vs 0.6B) and hypernyms with higher visual coherence generalize better.

## Highlights & Insights
- **Elegant Experimental Design**: Systematically isolating factors through ablation, shuffling, initialization, and encoder types.
- **Visual Coherence as the Bridge**: The LM does not blindly execute symbolic rules; it requires visual category members to cluster in representation space. Cross-modal generalization is a synergy of linguistic knowledge and perceptual consistency.
- **Empirical Support for "Platonic Representation Hypothesis"**: Generalization fails when cross-modal conceptual alignment is broken, supporting the convergence of unimodal models.

## Limitations & Future Work
- Only taxonomic knowledge (hypernym-hyponym) was tested, excluding other semantic relations.
- Use of simplified projector and binary tasks differs from full VLM training pipelines.
- The scale of 53 hypernyms and 1,216 categories is relatively limited.
- Hierarchical effects (e.g., $animal > bird > parrot$) were not explored.

## Related Work & Insights
- **vs "Platonic Representation Hypothesis"**: Provided new evidence in counterfactual experiments showing generalization fails when alignment is disrupted.
- **vs Controlled Culture Experiments**: Similar to testing LM generalization on controlled corpora, but extended to cross-modal settings.
- **vs VLM LM-bias Research**: Reinterprets LM dominance from a "bug" to a potentially useful "feature."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models](cross-cultural_expert-level_art_critique_evaluation_with_vision-language_models.md)
- [\[AAAI 2026\] Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models](../../AAAI2026/multimodal_vlm/cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)
- [\[ICML 2025\] Vision-Language Models Create Cross-Modal Task Representations](../../ICML2025/multimodal_vlm/vision-language_models_create_cross-modal_task_representations.md)
- [\[ACL 2026\] Cross-Modal Masked Compositional Concept Modeling for Enhancing Visio-Linguistic Compositionality](cross-modal_masked_compositional_concept_modeling_for_enhancing_visio-linguistic.md)
- [\[ICLR 2026\] XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models](../../ICLR2026/multimodal_vlm/xmodbench_benchmarking_cross-modal_capabilities_and_consistency_in_omni-language.md)

</div>

<!-- RELATED:END -->
