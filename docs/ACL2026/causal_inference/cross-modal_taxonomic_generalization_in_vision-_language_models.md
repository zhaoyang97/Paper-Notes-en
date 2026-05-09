---
title: >-
  [Paper Note] Cross-Modal Taxonomic Generalization in (Vision-) Language Models
description: >-
  [ACL 2026][Causal Inference][Cross-Modal Generalization] This paper systematically studies whether LMs in VLMs can cross-modally generalize purely text-learned taxonomic knowledge (hypernym relations) to visual inputs, finding that even without any visual-language hypernym supervision, pretrained LMs can identify hypernym categories in images, but this generalization requires visual coherence among category members.
tags:
  - ACL 2026
  - Causal Inference
  - Cross-Modal Generalization
  - Taxonomic Knowledge
  - Hypernymy
  - Vision-Language Model
  - Visual Coherence
content_hash: 76c02319aee421d9
---

# Cross-Modal Taxonomic Generalization in (Vision-) Language Models

**Conference**: ACL 2026
**arXiv**: [2603.07474](https://arxiv.org/abs/2603.07474)
**Code**: [https://github.com/sally-xu-42/cross-modal-taxonomic-gen](https://github.com/sally-xu-42/cross-modal-taxonomic-gen)
**Area**: Causal Inference
**Keywords**: Cross-Modal Generalization, Taxonomic Knowledge, Hypernymy, Vision-Language Model, Visual Coherence

## TL;DR
This paper systematically studies whether LMs in VLMs can cross-modally generalize purely text-learned taxonomic knowledge (hypernym relations) to visual inputs, finding that even without any visual-language hypernym supervision, pretrained LMs can identify hypernym categories in images, but this generalization requires visual coherence among category members.

## Method

### Key Designs

1. **Random Hypernym Ablation**: Systematically removes 10-100% of leaf-node-image-hypernym mappings, measuring generalization as evidence decreases.

2. **Systematic Hypernym Ablation**: Completely removes entire hypernym categories from training data — stricter than random ablation.

3. **Counterfactual Shuffling Experiments**: Cross-category shuffling (breaks visual coherence) vs within-category shuffling (preserves visual coherence). If LM executes arbitrary rules ("IF crow THEN bird"), both should perform similarly.

## Key Experimental Results

- Pretrained LM significantly above chance even at zero hypernym supervision; randomly initialized LM near chance
- Cross-category shuffling causes generalization collapse; within-category shuffling maintains performance — proving visual coherence is a necessary condition
- DINOv2 (no text training) and SigLIP (text-trained) show no significant difference as image encoders, confirming generalization comes from LM

## Highlights & Insights
- Exquisite experimental design: systematically isolates each factor through controlled variables
- Visual coherence is the bridge: LM doesn't blindly execute "IF crow THEN bird" rules but requires bird category members to actually cluster in representation space
- Provides empirical support for the "Platonic Representation Hypothesis"

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](../../NeurIPS2025/causal_inference/revealing_multimodal_causality_with_large_language_models.md)
- [\[AAAI 2026\] Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](../../AAAI2026/causal_inference/hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)
- [\[ACL 2026\] CausalDetox: Causal Head Selection and Intervention for Language Model Detoxification](causaldetox_causal_head_selection_and_intervention_for_language_model_detoxifica.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[ICLR 2026\] Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](../../ICLR2026/causal_inference/function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)

</div>

<!-- RELATED:END -->
