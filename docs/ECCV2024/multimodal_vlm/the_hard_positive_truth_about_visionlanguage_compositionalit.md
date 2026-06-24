---
title: >-
  [Paper Note] The Hard Positive Truth About Vision-Language Compositionality
description: >-
  [ECCV 2024][Multimodal VLM][compositional understanding] This paper reveals an evaluation blind spot in existing CLIP compositionality benchmarks—the lack of hard positives testing. It discovers that hard negative fine-tuning causes the model to become "oversensitive" (falsely reducing matching scores for paraphrases that preserve semantics). This issue is mitigated by jointly training with both hard positives and hard negatives.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "compositional understanding"
  - "CLIP"
  - "hard positives"
  - "hard negatives"
  - "vision-language alignment"
date: 2026-05-08
content_hash: debf3d64f8975d77
---

# The Hard Positive Truth About Vision-Language Compositionality

**Conference**: ECCV 2024  
**arXiv**: [2409.17958](https://arxiv.org/abs/2409.17958)  
**Code**: [https://github.com/amitakamath/hard_positives](https://github.com/amitakamath/hard_positives)  
**Area**: Multimodal VLMs  
**Keywords**: compositional understanding, CLIP, hard positives, hard negatives, vision-language alignment

## TL;DR
This paper reveals an evaluation blind spot in existing CLIP compositionality benchmarks—the lack of hard positives testing. It discovers that hard negative fine-tuning causes the model to become "oversensitive" (falsely reducing matching scores for paraphrases that preserve semantics). This issue is mitigated by jointly training with both hard positives and hard negatives.

## Background & Motivation
1. **Background**: Vision-language models like CLIP perform poorly on compositional understanding, as verified by several benchmarks (such as VL-Checklist and ARO) through hard negative retrieval tasks. Consequently, a large body of work has sought to improve CLIP's compositionality via hard negative fine-tuning.
2. **Limitations of Prior Work**: While these hard negative fine-tuning methods show significant improvements on existing benchmarks, the benchmarks themselves only test whether models can distinguish the original caption from hard negatives. They never verify whether models maintain invariance to hard positives (paraphrases that preserve semantics).
3. **Key Challenge**: Hard negative fine-tuning teaches models that "any perturbation changes the semantics." However, real-world language contains numerous synonym substitutions and phrase reorderings that preserve meaning. The models learn "perturbation detection" rather than true "semantic understanding."
4. **Goal**: (1) To construct an evaluation set containing hard positives to comprehensively test compositionality; (2) To expose the oversensitivity side-effects of hard negative fine-tuning; (3) To obtain more robust compositional improvements by jointly training with both hard positives and hard negatives.
5. **Key Insight**: Starting from the linguistic definition of compositionality, a model that truly understands compositionality must not only be sensitive to perturbations that alter semantics but also remain invariant to perturbations that preserve semantics.
6. **Core Idea**: Missing dimensions in compositionality evaluation are completed by introducing hard positives, and joint training with hard positives + hard negatives is used to balance the model's sensitivity and invariance.

## Method

### Overall Architecture
The input consists of an image $i$ and three captions: the original caption $c$, a hard negative $c_n$ (a semantic-altering perturbation), and a hard positive $c_p$ (a semantic-preserving perturbation). The evaluation tests whether the model can simultaneously satisfy $s(c|i) > s(c_n|i)$ and $s(c_p|i) > s(c_n|i)$. During training, LLAMA-2 is used to generate hard positives on COCO-train, and the CREPE method is used to generate hard negatives, resulting in a total of 1,775,259 training samples.

### Key Designs

1. **Hard Positive Evaluation Set Construction**:
    - **Function**: To construct an evaluation dataset containing 56,191 images, where each image is paired with its original caption, a hard negative, and a hard positive.
    - **Mechanism**: For the REPLACE type, 14 relations and 24 attributes are manually substituted with synonyms (e.g., "next to" $\rightarrow$ "near"). For the SWAP type, the order of object-attribute associations in the caption is swapped while keeping the semantics unchanged.
    - **Design Motivation**: Existing benchmarks assume that all atomic replacements/swaps alter semantics. However, language contains a vast amount of synonymous expressions, which is a completely neglected dimension of evaluation.

2. **Augmented Test Accuracy Metric**:
    - **Function**: To measure whether the model can correctly rank the original caption, hard positive, and hard negative simultaneously.
    - **Mechanism**: Requires $s(c|i) > s(c_n|i)$ and $s(c_p|i) > s(c_n|i)$, with a random chance accuracy of 33.3%.
    - **Design Motivation**: The traditional Original Test Accuracy only compares $c$ and $c_n$, failing to detect model oversensitivity to hard positives.

3. **Brittleness Metric**:
    - **Function**: To measure the proportion of instances where the model places $c_{n}$ between $c$ and $c_p$ (i.e., oversensitivity).
    - **Mechanism**: Calculates the proportion of instances satisfying $s(c|i) > s(c_n|i) > s(c_p|i)$ or $s(c_p|i) > s(c_n|i) > s(c|i)$.
    - **Design Motivation**: Directly quantifies the model's "oversensitivity" level. The ideal value should be close to 0% (human estimate is 0%).

4. **Hard Positive + Hard Negative Joint Training**:
    - **Function**: To generate one hard positive and one hard negative for each of the 591,753 captions in COCO-train using LLAMA-2.
    - **Mechanism**: SVLC-style fine-tuning, where each caption is trained simultaneously with its corresponding hard positive and hard negative.
    - **Design Motivation**: Teaches the model to distinguish "when a perturbation changes semantics and when it does not," instead of simply learning that "perturbations always change semantics."

### Loss & Training
An SVLC-style contrastive learning fine-tuning strategy is adopted. Based on the original caption, contrastive losses with both hard negatives (pulled apart) and hard positives (kept close) are utilized. Training is conducted on COCO-train using the ViT-B/32 CLIP architecture.

## Key Experimental Results

### Main Results

| Model | REPLACE Orig. Acc | REPLACE Aug. Acc | REPLACE Brittleness↓ | SWAP Aug. Acc |
|------|-------------------|------------------|---------------------|---------------|
| CLIP ViT-B/32 | 61.6 | 46.8 (-14.9) | 23.2 | 49.6 (-10.9) |
| DAC-LLM | 87.6 | 48.9 (-38.7) | 40.1 | 61.1 (-10.9) |
| Our HP+HN | 69.0 | **58.0** (-11.0) | **16.9** | **61.1** (-12.1) |
| Human | 97 | 97 | 0 | 100 |

### Ablation Study

| Configuration | REPLACE Aug. Acc | REPLACE Brittleness↓ | Description |
|------|-----------------|---------------------|------|
| 0 HN (HP Only) | 49.8 | 15.8 | No awareness of hard negatives |
| 0.25 HN | 55.5 | 16.6 | Good balance |
| 0.50 HN | 56.9 | 16.4 | Optimal balance point |
| Our HN only | 55.7 | 21.0 | Increased oversensitivity |
| Our HP+HN | **58.0** | **16.9** | Full model |

### Key Findings
- Hard negative fine-tuning causes the model's Aug. Test Acc on REPLACE to drop by up to 38.7 percentage points (DAC-LLM), far exceeding the 14.9 percentage point drop of the original CLIP.
- Oversensitivity transfers across perturbation types: models fine-tuned with SWAP hard negatives perform just as poorly and vulnerably on REPLACE hard positives.
- "Non-hard" positive enhancements (e.g., SVLC+Pos, DAC's rewrites) actually increase oversensitivity because the structural differences of these positives are too large.
- Hard negative fine-tuning also systematically lowers the absolute matching scores of the original captions (DAC-LLM drops from 0.23 to 0.16).
- Improving invariance does not transfer across perturbation types: the Swap-Only model performs poorly on REPLACE, and vice-versa.

## Highlights & Insights
- **Fundamental Completion of the Evaluation Dimension**: While almost all compositionality research focuses only on hard negatives, this work is the first to systematically introduce the hard positive dimension. It reveals that previously overestimated performance gains actually come with severe oversensitivity side-effects. This finding provides valuable methodological insights.
- **Clear Theoretical Analysis**: Explains why hard negative fine-tuning leads to oversensitivity from the perspective of training data distribution—all perturbations seen by the model change the label, so the model learns perturbation detection instead of semantic understanding.
- **Breaking of Implicit Assumptions in Standard Retrieval Tasks**: Traditional ITM evaluation assumes "different caption = different semantics." However, synonym substitution breaks this assumption, which has far-reaching implications for future VLM evaluation design.

## Limitations & Future Work
- Limited only to CLIP-style contrastive learning models; generative VLMs like Flamingo, BLIP, and GPT-4V were not evaluated.
- The construction of hard positives relies on hand-crafted synonym mapping (REPLACE) or simple word-order swapping (SWAP), which has limited coverage.
- There is still a huge gap between model performance and human performance after joint training (58% vs 97%), requiring more fundamental architectural or training paradigm changes.
- Finer-grained semantic similarity evaluations can be explored, rather than binary "positive/negative" decisions.

## Related Work & Insights
- **vs NegCLIP/CREPE**: These methods only use hard negatives for fine-tuning. They perform better on existing benchmarks, but demonstrate higher actual oversensitivity. This work exposes the false prosperity of their performance by introducing the hard positive dimension.
- **vs DAC/SVLC**: Even when non-hard positive samples (paraphrases/rewritten captions) are added, oversensitivity is still not mitigated because the structural differences of these cases are too large to present a true hard positive challenge.
- **vs SugarCrepe**: This work fixes textual bias in hard negatives but is still confined to the hard negative evaluation framework. This paper is complementary to it.

## Supplementary Notes
- Dataset scale: Evaluation set of 56,191 images, 112,382 triplets; Training set of 1,775,259 samples.
- Human validation: Double-annotator validation on 100 samples shows human accuracy of 99%+.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic introduction of the hard positive concept to evaluate VLM compositionality for the first time, offering a novel perspective and far-reaching implications.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 7 models, multiple perturbation types, ablation study, and human evaluation, but lacks testing on generative VLMs.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definitions are clear, logical reasoning is rigorous, and figures/tables are intuitive.
- Value: ⭐⭐⭐⭐ Important methodological warning to the VLM compositionality research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models](../../CVPR2026/multimodal_vlm/no_hard_negatives_required_concept_centric_learning_leads_to_compositionality_wi.md)
- [\[ICML 2026\] Vision Language Models Cannot Reason About Physical Transformations](../../ICML2026/multimodal_vlm/vision_language_models_cannot_reason_about_physical_transformation.md)
- [\[ACL 2026\] Cross-Modal Masked Compositional Concept Modeling for Enhancing Visio-Linguistic Compositionality](../../ACL2026/multimodal_vlm/cross-modal_masked_compositional_concept_modeling_for_enhancing_visio-linguistic.md)
- [\[ECCV 2024\] SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant](sq-llava_self-questioning_for_large_vision-language_assistant.md)
- [\[ECCV 2024\] AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](addressclip_empowering_vision-language_models_for_city-wide_image_address_locali.md)

</div>

<!-- RELATED:END -->
