---
title: >-
  [Paper Note] When Seeing Overrides Knowing: Disentangling Knowledge Conflicts in Vision-Language Models
description: >-
  [ACL2026][Multimodal VLM][Vision-Language Models] This paper constructs WHOOPS-AHA! to bring VLM commonsense knowledge into direct conflict with counterfactual visual evidence…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Knowledge Conflict"
  - "Mechanistic Interpretability"
  - "Attention Heads"
  - "Visual Attribution"
date: 2026-05-08
content_hash: c2dce72011767a9d
---

# When Seeing Overrides Knowing: Disentangling Knowledge Conflicts in Vision-Language Models

**Conference**: ACL2026  
**arXiv**: [2507.13868](https://arxiv.org/abs/2507.13868)  
**Code**: https://github.com/francescortu/Seeing-Knowing  
**Area**: multimodal_vlm  
**Keywords**: Vision-Language Models, Knowledge Conflict, Mechanistic Interpretability, Attention Heads, Visual Attribution

## TL;DR
This paper constructs WHOOPS-AHA! to bring VLM commonsense knowledge into direct conflict with counterfactual visual evidence, discovering that a few late-layer attention heads causally control whether the model relies on internal knowledge or visual input.

## Background & Motivation
**Background**: VLMs rely simultaneously on pretrained parametric knowledge and current image input. Under normal circumstances, these information sources complement each other: parametric knowledge provides world commonsense, while visual input provides scene-specific facts. However, conflicts arise when images contain anomalous or counterfactual elements—for example, if commonsense suggests a wolf howls at the moon, but an image shows a wolf howling at the sun.

**Limitations of Prior Work**: Many VLM hallucination studies only observe whether the final answer is incorrect or use external attribution methods to explain the influence of image regions. They do not deeply address how internal model mechanisms choose between "known commonsense" and "seen images." Whether a model is overriding knowledge with visual surface signals or over-relying on parametric knowledge in inappropriate scenarios affects the reliability of multimodal systems.

**Key Challenge**: Models need to dynamically calibrate between visual evidence and internal knowledge. Total reliance on images allows counterfactual or misleading visuals to override commonsense, while total reliance on parametric knowledge ignores real visual input. The critical question is whether this modal conflict is regulated by localizable and intervenable internal mechanisms.

**Goal**: The objective is to construct a controllable dataset to induce knowledge conflicts, locate the components in VLMs that support factual vs. counterfactual tokens, verify if these components play a causal role, and examine if they can serve as tools for visual evidence localization.

**Key Insight**: The authors utilize a token-level completion task, designing each sample as a pair of explicit factual continuations and counterfactual visual continuations. This allows the use of Logit Lens to directly compare the logit contributions of internal components to two candidate tokens, rather than relying on ambiguous judgments of model preference in open-ended generation.

**Core Idea**: By using counterfactual images to generate conflict between "seeing" and "knowing," the authors identify late-layer factual/counterfactual attention heads via logit attribution. They then perform directional scaling of attention to the image/text tokens in these heads to causally control the VLM's modal preference.

## Method

### Overall Architecture
The paper proceeds in four stages. First, the WHOOPS-AHA! dataset is constructed: based on 500 visually anomalous and semantically rich images from WHOOPS!, GPT-4o generates sentences that trigger commonsense completion for each image, providing two sets of tokens: $S_{fact}$ for commonsense completions and $S_{cofa}$ for counterfactual visual completions.

Second, truly conflict-inducing samples are filtered. For each model, it must favor the factual token in a text-only prompt and favor the counterfactual token under multimodal input. Only samples where the image successfully overrides text-based commonsense are retained for mechanistic analysis.

Third, Logit Lens is used to analyze the contributions of MLPs, attention blocks, and individual attention heads in LLaVA-NeXT-7B and Gemma3-12B to $t_{fact}$ and $t_{cofa}$. The authors find that MLPs generally favor internal factual knowledge, while attention—specifically a few late-layer heads—favors visual counterfactual signals.

Fourth, causal intervention and visual attribution are performed. The authors select the top-20 heads most supportive of factual/counterfactual tokens and apply multiplicative scaling to the attention weights at the final token position: either boosting factual heads' attention to text tokens or suppressing counterfactual heads' attention to image tokens (and vice versa). Subsequently, attention and gradients are used to identify the image patches driving counterfactual output, validated through patch ablation.

### Key Designs
1. **WHOOPS-AHA! Controllable Counterfactual Completion Dataset**:
    - **Function**: Provides a verifiable, token-level testbed for multimodal knowledge conflict.
    - **Mechanism**: Each sample consists of a counterfactual image, a sentence referring to the image, a commonsense completion set $S_{fact}$, and a visual counterfactual set $S_{cofa}$. For instance, the text "The wolf is howling at the" should result in "moon" without an image, but the visual evidence of a wolf howling at the sun drives the completion "sun."
    - **Design Motivation**: Mechanistic interpretability requires controllable conflicts and clear target tokens. Open-ended QA makes it difficult to determine which knowledge the model prefers, whereas WHOOPS-AHA! allows for precision logit attribution.

2. **Logit Lens for Locating Factual and Counterfactual Heads**:
    - **Function**: Identifies which internal layers and attention heads promote internal knowledge versus visual evidence.
    - **Mechanism**: Performs vocabulary projection on intermediate hidden states at the last token position to compare logits for $t_{fact}$ and $t_{cofa}$. Factual prevalence is reported at the block level and factual accuracy at the head level.
    - **Design Motivation**: Unlike observing final output, Logit Lens identifies where conflict resolution occurs in the forward pass. Results show conflict resolution is concentrated in a few upper-layer heads rather than being uniformly distributed.

3. **Directional Attention Intervention and Visual Patch Attribution**:
    - **Function**: Verifies if these heads causally influence modal preference and examines if counterfactual heads accurately point to anomalous image regions.
    - **Mechanism**: Modify the last row of the attention matrix at the final token position. For counterfactual heads, image token attention is scaled by $(1-\lambda)$; for factual heads, text token attention is scaled by $(1+\lambda)$. Visual attribution compares counterfactual head attention against gradient-based and random head baselines.
    - **Design Motivation**: If the relationship were merely correlational, intervening on heads would not shift behavior. Intervention demonstrates the ability to push the model back toward commonsense or further toward visual evidence, while patch ablation proves these heads accurately localize the conflict-inducing regions.

### Loss & Training
The study does not train new models; it focuses on data construction, forward-pass analysis, and inference-time intervention. Models evaluated include LLaVA-NeXT-7B (32 layers, 32 heads/layer) and Gemma3-12B (48 layers, 16 heads/layer). The intervention strength $\lambda$ is restricted to $[-3, 3]$ to prevent linguistic degradation.

## Key Experimental Results

### Main Results
Data validation confirms high quality in completions. In conflict-induction experiments, images significantly shift models from factual to counterfactual tokens. Analysis shows counterfactual heads focus heavily on image tokens, and manipulating a few heads is sufficient to change model behavior.

| Experimental Item | LLaVA-NeXT | Gemma3 | Conclusion |
|-------------------|------------|--------|------------|
| Retained Conflict Samples | 436 | 432 | Most WHOOPS-AHA! samples form analyzable conflicts |
| Text-only Factual Token | "moon" prob 78% | "moon" prob 100% | Models rely on commonsense without images |
| Multimodal Counterfactual Token | "sun" prob 26% | "sun" prob 44% | Visual input overrides internal knowledge |
| Post-image Factual Accuracy | 27% | 24% | Counterfactual images systematically alter predictions |
| Counterfactual Head Image Attn | 61% | 52% | Significantly higher than model average (~22%) |
| Factual Head Image Attn | 29% | 25% | Lower attention to images, favoring text/parameters |
| Intervention Peak Factual Acc | 74% | 83% | Boosting factual/suppressing counterfactual heads restores commonsense |

### Ablation Study
Ablations focus on head selection, intervention strength, and attribution quality. Random head interventions have no significant effect. Control experiments (POPE) show counterfactual heads are not general-purpose vision heads but are specifically invoked during conflict.

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Top-20 heads intervention | Factual accuracy peaks | Best balance between effectiveness and stability |
| 100 random heads intervention | Factual accuracy stable | Changes stem from specific heads, not arbitrary noise |
| POPE (No Image) | Accuracy ~0.50 | POPE is a valid visual dependency task |
| POPE (Suppress CF Heads) | LLaVA 0.87→0.87 | These heads do not manage general visual recognition |
| Visual CounterFact Overlap | High overlap | Mechanism is not exclusive to WHOOPS-AHA! |
| Visual Attribution (Gemma3) | CF heads ratio 4.41 | Attention heads focus more precisely than gradients (1.74) |

### Key Findings
- VLM modal conflicts are not processed uniformly across the model. A few late-layer attention heads are core regulators; attention blocks favor visual counterfactuals while MLPs favor parametric knowledge.
- Interventions are directional. Boosting factual heads or weakening counterfactual heads restores reliance on internal knowledge, proving these heads play a causal role.
- Counterfactual heads possess interpretable visual localization abilities, focusing on patches containing anomalous objects. Ablating these patches increases factual accuracy.

## Highlights & Insights
- Successfully links VLM hallucinations to mechanistic interpretability: rather than just identifying errors, it identifies the specific heads driving visual overrides.
- The token-level design of WHOOPS-AHA! is ideal for mechanistic analysis, compressing complex multimodal QA into controllable candidate tokens.
- Counterfactual heads serving as both "control knobs" and "visual pointers" suggests they could be used for VLM reliability monitoring—triggering alerts when model answers rely heavily on conflict-resolution heads.

## Limitations & Future Work
- Logit Lens is an approximate diagnostic tool; projecting non-final residual states to the vocabulary may involve distortion.
- The study focuses on late-fusion LLaVA-style architectures. Early or mid-fusion VLMs may inject visual information differently.
- While token-level control is precise, real-world generation involves full sentences and complex reasoning. Future work should extend this to long-form explanations and multi-step visual reasoning.

## Related Work & Insights
- **vs. Textual LLM Conflict**: Extends work on text-context vs. parametric knowledge conflict (e.g., Ortu et al.) to the multimodal domain, confirming similar competitive mechanisms exist.
- **vs. Gradient Attribution**: While gradients find influential patches, they are often less precise than identified counterfactual heads for localizing counterfactual objects.
- **vs. Hallucination Detection**: Provides an internal "circuit" perspective, showing that certain hallucinations or visual overrides can be regulated by small-scale attention head adjustments.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐☆ 
- Writing Quality: ⭐⭐⭐⭐☆ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models](../../AAAI2026/multimodal_vlm/seeing_justice_clearly_handwritten_legal_document_translation_with_ocr_and_visio.md)
- [\[ACL 2026\] WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering](wikiseeker_rethinking_the_role_of_vision-language_models_in_knowledge-based_visu.md)
- [\[ICML 2026\] From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models](../../ICML2026/multimodal_vlm/from_seeing_to_thinking_decoupling_perception_and_reasoning_improves_post-traini.md)
- [\[ICLR 2026\] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes](../../ICLR2026/multimodal_vlm/seeing_across_views_benchmarking_spatial_reasoning_of_vision-language_models_in_.md)
- [\[CVPR 2026\] When to Think and When to Look: Uncertainty-Guided Lookback](../../CVPR2026/multimodal_vlm/when_to_think_and_when_to_look_uncertainty-guided_lookback.md)

</div>

<!-- RELATED:END -->
