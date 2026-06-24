---
title: >-
  [Paper Note] Symmetrical Visual Contrastive Optimization: Aligning Vision-Language Models with Minimal Contrastive Images
description: >-
  [ACL 2025][Multimodal VLM][visual grounding] Proposing S-VCO (Symmetrical Visual Contrastive Optimization), a novel VLM fine-tuning objective. By symmetrically aligning/rejecting matched/contradictory image-text pairs, it enhances visual reliance. Coupled with the Minimal Visual Contrastive (MVC) dataset, it reduces hallucinations by 22% and significantly improves performance on vision-dependent tasks.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "visual grounding"
  - "contrastive optimization"
  - "VLM alignment"
  - "hallucination"
  - "DPO"
date: 2026-05-08
content_hash: fce76327c146572e
---

# Symmetrical Visual Contrastive Optimization: Aligning Vision-Language Models with Minimal Contrastive Images

**Conference**: ACL 2025  
**arXiv**: [2502.13928](https://arxiv.org/abs/2502.13928)  
**Code**: [https://s-vco.github.io/](https://s-vco.github.io/)  
**Area**: Multimodal VLMs  
**Keywords**: visual grounding, contrastive optimization, VLM alignment, hallucination, DPO

## TL;DR
Proposing S-VCO (Symmetrical Visual Contrastive Optimization), a novel VLM fine-tuning objective. By symmetrically aligning/rejecting matched/contradictory image-text pairs, it enhances visual reliance. Coupled with the Minimal Visual Contrastive (MVC) dataset, it reduces hallucinations by 22% and significantly improves performance on vision-dependent tasks.

## Background & Motivation

**Background**: VLMs tend to over-rely on language model priors while ignoring visual content, leading to visual hallucinations. On several benchmarks, the performance gap for VLMs with and without visual inputs is minimal.

**Limitations of Prior Work**: Perplexity experiments reveal a counterintuitive phenomenon—VLMs exhibit the lowest perplexity with **no image** input and, conversely, the highest perplexity when given a **matching image**. This indicates that models learn "ignoring images is better than using them." Existing DPO/mDPO methods use randomly cropped or noisy images as negative samples, but these corrupted images lack meaningful correlation with the text. Consequently, models can bypass visual understanding by taking a shortcut of "rejecting unnatural images."

**Key Challenge**: The preference alignment paradigm (DPO) essentially treats visual supervision as a "preference"—where the original image is always superior to the corrupted one. However, this does not teach the model **why** the original image is better, nor does it enable the model to precisely align visual details with text tokens.

**Goal**: Designing a fine-tuning objective that not only encourages the model to "attend to matching images" but also "reject contradictory images," while avoiding shortcut learning through a symmetrical mechanism.

**Key Insight**: Instead of categorizing images into "good/bad," they are treated as equal images with contrastive details—either can be "correct" when paired with its corresponding text. Meanwhile, a contrastive image dataset with minimal yet meaningful visual differences is constructed.

**Core Idea**: Replacing unidirectional preference optimization with symmetrical visual contrastive optimization, which, coupled with a minimal visual contrastive dataset, enables models to truly align visual details with text.

## Method

### Overall Architecture
The input consists of a pair of contrastive images $(i_w, i_l)$, a shared query $q$, and their respective textual responses $(y_w, y_l)$. S-VCO trains the model to simultaneously: (1) attend to matching images + reject contradictory images (VCO); (2) symmetrically flip roles to allow "negative" images to act as "positive" conditions (Symmetry).

### Key Designs

1. **Visual Contrastive Supervision (VCO)**:

    - **Function**: Designing two complementary loss terms—Attend (attending to the matched image) and Reject (rejecting the contradictory image).
    - **Attend Loss**: Encourages the model to generate $y_w$ better when given the matching image $i_w$ compared to no image input:
      $L_{\text{Attend}} = -\log\sigma(\beta_1 \log\frac{\pi_\theta(y_w|i_w,q)}{\pi_{\text{ref}}(y_w|i_w,q)} - \beta_1 \log\frac{\pi_\theta(y_w|q)}{\pi_{\text{ref}}(y_w|q)})$
    - **Reject Loss**: Encourages the model to generate $y_w$ **less** under the contradictory image $i_l$ compared to no image input:
      $L_{\text{Reject}} = -\log\sigma(\beta_2 \log\frac{\pi_\theta(y_w|q)}{\pi_{\text{ref}}(y_w|q)} - \beta_2 \log\frac{\pi_\theta(y_w|i_l,q)}{\pi_{\text{ref}}(y_w|i_l,q)})$
    - **Design Motivation**: Attend addresses the issue of "ignoring images," while Reject addresses the "failure to reject contradictory images." Utilizing "no image" as an anchor for comparison is more meaningful than using "corrupted images."

2. **Symmetrical Alignment**:

    - **Function**: Flipping the VCO objective such that $i_l$ acts as the matching condition and $i_w$ as the contradictory condition, paired with text $y_l$.
    - **Mechanism**: $L_{\text{S-VCO}} = L_{\text{VCO}}(i_w, y_w, i_l) + L_{\text{VCO}}(i_l, y_l, i_w)$
    - **Design Motivation**: Unidirectional formulations always prefer $i_w$ over $i_l$, which can cause the model to learn a shortcut of "rejecting inputs that look like synthetic images" (as most contrastive images are generated via inpainting). The symmetrical design places both images on equal footing, forcing the model to focus on image-text alignment rather than low-level image artifacts.

3. **MVC Dataset (Minimal Visual Contrast)**:

    - **Function**: Constructing high-quality contrastive image-text training data.
    - **Mechanism**: Originating from counterfactual visual sources such as CounterCurate and FineCops-Ref, covering 4 contrastive types (object replacement, attribute binding, relation/number modification, position flipping). A double-threshold filtering is applied: (1) CLIP similarity > 0.7 (overall semantic similarity, challenging for current VLMs); (2) DINOv2 similarity < 0.5 (distinct visual features). Subsequently, GPT-4o is utilized to rewrite brief captions into conversational question-answering formats.
    - **Design Motivation**: Existing data exhibits inconsistent quality (synthetic images may not create true contrasts). The double-threshold filtering selects "hard but meaningful" samples. Dialogue augmentation makes the data suitable for VLM instruction tuning.

### Loss & Training
$L_{\text{S-VCO}} = L_{\text{VCO}}(i_w, y_w, i_l) + L_{\text{VCO}}(i_l, y_l, i_w)$, where each VCO contains both Attend and Reject terms. Standard LoRA fine-tuning is applied, with a dataset of approximately 11K contrastive image-text pairs.

## Key Experimental Results

### Main Results

| Method | Hallucination (MMHal↑) | Hallucination Rate↓ | CVBench↑ | MMVP↑ | MMVet↑ | Average Gain |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| LLaVA-1.5-7B (BASE) | 2.16 | 57% | 59.3 | 21.3 | 30.46 | 0% |
| DPO_VLF | 2.06 | 65% | 57.0 | 16.7 | 31.65 | -1.25% |
| DPO_MVC | 2.45 | 53% | 63.2 | 22.0 | 33.53 | +8.11% |
| mDPO_VLF | 2.39 | 57% | 53.2 | 18.7 | 31.79 | +2.11% |
| **S-VCO_MVC** | **2.75** | **46%** | **63.5** | **25.3** | **34.68** | **+14.26%** |

**Key Results**: S-VCO reduces the hallucination rate from 57% to 46% (a 22% reduction), yields a 4+ point improvement on vision-centric benchmarks, and achieves a total average gain of 14.26%, substantially outperforming DPO and mDPO.

### Ablation Study

| Configuration | Average Gain | Description |
|------|---------|------|
| S-VCO + MVC (Full) | +14.26% | Optimal |
| VCO w/o Symmetry | Improvement but inferior to full | Symmetry prevents shortcut learning |
| S-VCO + Unfiltered Data | Limited improvement | Data quality is crucial |
| S-VCO + VLF Data | Lower than MVC | VLF's cropped/noisy images are inferior to meaningful contrasts |
| Attend Only | Positive but limited | Requires coordination with Reject |
| Reject Only | Positive but limited | Requires coordination with Attend |

### Key Findings
- **Stronger visual reliance links to greater improvement**: The most significant gains are witnessed on high-visual-reliance benchmarks (e.g., MMVP, CVBench), validating that S-VCO indeed enhances visual utilization.
- **Crucial quality of the MVC dataset**: Applying the same DPO objective on MVC data (+8.11%) yields far superior results than on VLF data (-1.25%), indicating that contrast quality in data is more critical than the optimization method itself.
- **Preserving or even improving general capabilities**: While boosting visual tasks, S-VCO also yields minor improvements on OCR (TextVQA) and knowledge (SQA) benchmarks, without sacrificing general execution capabilities.

## Highlights & Insights
- **Profound insight of symmetrical design**: Changing the "preference" paradigm to a "contrastive" paradigm is a key transition—away from fixed "good" and "bad" images, determining correctness dynamically based on paired text. This prevents the model from developing shortcuts like "rejecting all synthetic images." This formulation can be generalized to other multimodal alignment tasks.
- **No-image baseline as an anchor**: Leveraging "no-image conditioning" instead of "corrupted images" as the comparison anchor is highly elegant—it directly measures "whether the image truly aids generation," offering a much cleaner training signal.
- **Double-threshold filtering strategy (CLIP + DINOv2)**: High CLIP similarity ensures overall semantic consistency (challenging for VLMs), while low DINOv2 similarity ensures distinct visual features. This strategy of selecting "hard but meaningful" samples is worth adopting in other contrastive learning scenarios.

## Limitations & Future Work
- MVC data relies on synthetic/edited counterfactual images, which may contain artifacts; although filtered, they cannot be entirely eliminated.
- Experiments are restricted to LLaVA-1.5-7B; the efficacy on larger or newer models remains unverified.
- Contrast types are mainly restricted to object/attribute-level scopes, lacking higher-level semantic contrasts (e.g., scene, relation, action).
- Building the training data requires assistance from GPT-4o, which incurs non-trivial costs.

## Related Work & Insights
- **vs mDPO (Wang et al. 2024)**: mDPO utilizes randomly cropped images as negative samples, which fundamentally remains a preference paradigm where negative samples are independent of the text. S-VCO employs minimal contrastive images along with symmetrical optimization, fundamentally addressing the shortcut learning issue.
- **vs RLHF-V (Jiang et al. 2024)**: RLHF-V adds diffusion noise to images to create negative samples, which similarly suffers from meaningless negative feedback; S-VCO's negative sample is another real image that simply mismatches the target text.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The introduction of symmetrical contrastive optimization offers depth; the transition from a "preference" to a "contrastive" paradigm is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks cover hallucinations, vision, and general capabilities, supported by thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ The motivational arguments derived from perplexity experiments are highly compelling; charts and diagrams are clear.
- Value: ⭐⭐⭐⭐⭐ Offers a new paradigm for VLM visual alignment; the 22% reduction in hallucinations has significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HSCR: Hierarchical Self-Contrastive Rewarding for Aligning Medical Vision Language Models](hscr_hierarchical_self-contrastive_rewarding_for_aligning_medical_vision_languag.md)
- [\[ACL 2025\] Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains](weaving_context_across_images_improving_vision-language_models_through_focus-cen.md)
- [\[ICML 2025\] MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization](../../ICML2025/multimodal_vlm/mmedpo_aligning_medical_vision-language_models_with_clinical-aware_multimodal_pr.md)
- [\[CVPR 2026\] Dynamics-Aware Preference Optimization for Vision-Language Models](../../CVPR2026/multimodal_vlm/dynamics-aware_preference_optimization_for_vision-language_models.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](../../NeurIPS2025/multimodal_vlm/continual_multimodal_contrastive_learning.md)

</div>

<!-- RELATED:END -->
