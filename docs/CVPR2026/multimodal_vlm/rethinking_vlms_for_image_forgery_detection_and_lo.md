---
title: >-
  [Paper Note] Rethinking VLMs for Image Forgery Detection and Localization
description: >-
  [CVPR 2026][Multimodal VLM][Image forgery detection] This work reveals that VLMs inherently favor semantic plausibility over authenticity (CLIP cosine similarity for forged images reaches 96–99%), and proposes IFDL-VLM, which decouples detection/localization from language explanation into two stages: Stage-1 employs ViT+SAM for detection and localization, and Stage-2 feeds the resulting mask as auxiliary input to a VLM to enhance interpretability. The method achieves state-of-the-art performance across 9 benchmarks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Image forgery detection
  - VLM semantic bias
  - decoupled optimization
  - SAM localization
  - interpretability
date: 2026-05-08
content_hash: a4517cff839632aa
---

# Rethinking VLMs for Image Forgery Detection and Localization

**Conference**: CVPR 2026
**arXiv**: [2603.12930](https://arxiv.org/abs/2603.12930)
**Code**: [github.com/sha0fengGuo/IFDL-VLM](https://github.com/sha0fengGuo/IFDL-VLM)
**Area**: Multimodal VLM
**Keywords**: Image forgery detection, VLM semantic bias, decoupled optimization, SAM localization, interpretability

## TL;DR
This work reveals that VLMs inherently favor semantic plausibility over authenticity (CLIP cosine similarity for forged images reaches 96–99%), and proposes IFDL-VLM, which decouples detection/localization from language explanation into two stages: Stage-1 employs ViT+SAM for detection and localization, and Stage-2 feeds the resulting mask as auxiliary input to a VLM to enhance interpretability. The method achieves state-of-the-art performance across 9 benchmarks.

## Background & Motivation
**Background**: In the AIGC era, image forgery detection and localization (IFDL) faces highly realistic synthetic images and hybrid forgeries (AI-generated content combined with traditional editing and post-processing), rendering conventional artifact-based methods increasingly ineffective. Recent approaches such as SIDA and FakeShield integrate VLMs (CLIP+LLM+SAM) into end-to-end pipelines to improve interpretability.

**Limitations of Prior Work**: (1) The CLIP visual encoder in VLMs is pre-trained on large-scale natural images with an objective of semantic alignment rather than authenticity discrimination. (2) As long as a forged image is "semantically coherent," CLIP features remain nearly unchanged — cosine similarity stays at 96.3% after object replacement and 98.5% after object insertion. (3) End-to-end training propagates CLIP's semantic bias directly into the detection and localization modules, yielding performance inferior to dedicated models.

**Key Challenge**: A fundamental conflict exists between VLMs' pursuit of "semantic plausibility" and IFDL's requirement to "perceive inauthenticity."

**Goal**: To answer two questions — does VLM prior knowledge genuinely benefit IFDL (no), and can detection/localization results in turn help VLMs generate better explanations (yes)?

**Key Insight**: Fully decouple detection/localization from language explanation, using dedicated ViT+SAM for the former and feeding the localization mask back to the VLM as explicit forgery concept encoding.

**Core Idea**: Rather than involving VLMs in detection and localization (where they underperform), the proposed approach lets detection results inform VLMs about what to explain — a task at which VLMs excel.

## Method

### Overall Architecture
A two-stage decoupled design: **Stage-1** trains a ViT backbone (initialized from CLIP-ViT-L/14) with a frozen SAM for detection and localization. The global CLS token is fed into a linear classifier for three-class classification (real / fully synthetic / tampered), while patch tokens are aggregated via multi-head attention into a SEG token that is passed to the SAM mask decoder for localization mask generation. **Stage-2** takes the Stage-1 localization mask as auxiliary input, fusing global semantics and local forgery cues via region-aware visual feature enhancement: $T_{vis} = \alpha \cdot \text{CLIP}(x) + (1-\alpha) \cdot \text{CLIP}(x \odot M)$, and fine-tunes Vicuna-13B to generate language explanations.

### Key Designs
1. **Discovery and Verification of VLM Semantic Plausibility Bias**:
    - Function: Systematically verifies the negative impact of VLM priors on IFDL.
    - Mechanism: CLIP cosine similarity between forged and original images reaches 96.3%–98.5%, demonstrating that CLIP optimizes for high-level scene consistency rather than visual authenticity, rendering forged and authentic image representations indistinguishable.
    - Design Motivation: This finding motivates the decoupled design — since CLIP is insensitive to authenticity, it should not participate in detection and localization.

2. **Reverse Information Flow: Localization Masks Encoding Forgery Concepts**:
    - Function: Uses Stage-1 detection and localization results as explicit input to the Stage-2 VLM.
    - Mechanism: The localization mask $M$ explicitly indicates "where the forgery is," relieving the VLM from implicitly learning forgery concepts from data. Via $T_{vis} = \alpha \cdot \text{CLIP}(x) + (1-\alpha) \cdot \text{CLIP}(x \odot M)$ ($\alpha=0.5$), both global semantics and low-level cues from the local forged region are preserved.
    - Design Motivation: This liberates the VLM from the difficult task of "finding where the forgery is" and refocuses it on "explaining why a given region is forged" — precisely where VLMs are capable.

3. **Dedicated ViT+SAM Detection and Localization in Stage-1**:
    - Function: Bypasses VLM bias by using a dedicated model for detection and localization.
    - Mechanism: A trainable ViT backbone produces a CLS token for three-class classification and a SEG token for a frozen SAM to generate masks. The loss is $\mathcal{L}_{st-1} = \mathcal{L}_{bce}(\hat{M},M) + \mathcal{L}_{dice}(\hat{M},M) + \mathcal{L}_{ce}(\hat{D},D)$.
    - Design Motivation: A dedicated model is unaffected by CLIP's semantic bias and is better suited to learning low-level forgery traces.

### Loss & Training
Stage-1: BCE + Dice loss (localization) + CE loss (classification), all with weight 1.0. Stage-2: Standard language modeling CE loss. Optimizer: AdamW (lr=1e-5, β=(0.9, 0.95)), cosine decay after 100-step warmup, batch size=4 with gradient accumulation=10, mixed-precision training. The SAM image encoder is kept frozen; only the mask decoder is fine-tuned.

## Key Experimental Results

### Main Results

| Dataset / Task | Metric | IFDL-VLM | SIDA-13B | FakeShield | Gain |
|---|---|---|---|---|---|
| SID-Set Detection | Overall ACC | 0.997 | 0.94 | — | +5.7% |
| SID-Set Detection | Overall F1 | 0.998 | 0.94 | — | +5.8% |
| SID-Set Localization | IoU | 0.65 | 0.44 | — | +21% abs |
| SID-Set Localization | AUC | 0.99 | 0.87 | — | +12% abs |
| 8-dataset Cross-dataset Avg | Avg IoU | 0.47 | 0.38 | 0.34–0.39 | +13% |
| 8-dataset Cross-dataset Avg | Avg F1 | 0.58 | 0.45 | 0.39–0.45 | +19% |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| α=0.5 (default) | CSS 0.853 | Optimal balance between global semantics and local forgery cues |
| α=0 (mask region only) | CSS 0.821 | Lacks global semantic context |
| α=1 (full image only) | CSS 0.798 | No mask guidance; degrades to standard VLM |
| Unfrozen CLIP fine-tuning | Language quality ↓ | Disrupts cross-modal alignment |
| Predicted mask vs. GT | CSS 0.842 | Negligible gap from GT (0.853); framework is robust |

### Key Findings
- Decoupling outperforms end-to-end training — the seemingly "simpler" pipeline achieves stronger performance by eliminating CLIP bias interference.
- Localization masks substantially improve VLM interpretability: GPT-5 scores 2.36 (Ours) vs. 1.44 (SIDA), a +63.9% improvement.
- In a human preference study, 65.2% of evaluators preferred the explanations generated by the proposed method.

## Highlights & Insights
- The core finding that VLMs favor semantic plausibility over authenticity may have broad implications for the entire AIGC detection field — any work using VLMs for anomaly or forgery detection should be aware of this bias.
- The reverse information flow design — "detection results aid explanation" — is elegant: masks explicitly encode forgery concepts, simplifying VLM training optimization.
- Strong cross-dataset generalization is demonstrated across 8 unseen datasets.
- The framework is robust to Stage-1 localization errors (predicted masks and GT masks yield nearly equivalent results).

## Limitations & Future Work
- Severe Stage-1 localization failures can cascade and degrade Stage-2 explanation quality (though experiments demonstrate a degree of robustness).
- Two-stage training increases engineering complexity; future work may explore single-stage decoupled designs.
- Stage-2 still uses a frozen CLIP encoder, which may limit language generation quality in fine-grained scenarios.
- The reliability and potential bias of GPT-5 automatic evaluation warrant further validation.

## Related Work & Insights
- **vs. SIDA**: An end-to-end VLM pipeline; detection ACC 0.94 vs. 0.997 (Ours), localization IoU 0.44 vs. 0.65 (Ours). End-to-end training is inferior to the decoupled design.
- **vs. FakeShield**: An IFDL method using SAM and MLLM; 8-dataset Avg IoU 0.34–0.39 vs. 0.47 (Ours).
- **vs. MVSS-Net/CAT-Net**: Traditional IFDL methods without language explanation capability; localization performance also falls substantially short of the proposed method.
- The decoupling principle generalizes broadly: when pre-trained model priors conflict with downstream task objectives, forcing end-to-end training is inadvisable — decoupling the strengths of each component is preferable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The insight on VLM semantic plausibility bias is highly valuable; the decoupled design is counter-intuitive yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 datasets, three evaluation dimensions (detection, localization, interpretability), human preference studies, and GPT-based evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation analysis (cosine similarity verification) is convincing; method description is clear.
- Value: ⭐⭐⭐⭐⭐ High practical value for image authenticity verification in the AIGC era; the core insight is transferable.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs](mind_the_way_you_select_negative_texts_pursuing_the_distance_consistency_in_ood_.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)
- [\[CVPR 2026\] Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token](rethinking_mllm_itself_as_a_segmenter_with_a_single_segmentation_token.md)

<!-- RELATED:END -->
