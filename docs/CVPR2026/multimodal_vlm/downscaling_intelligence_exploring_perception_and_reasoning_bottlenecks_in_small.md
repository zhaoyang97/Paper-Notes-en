---
title: >-
  [Paper Note] Downscaling Intelligence: Exploring Perception and Reasoning Bottlenecks in Small VLMs
description: >-
  [CVPR 2026][Multimodal VLM][multimodal model scaling] This paper systematically investigates the effect of LLM scaling on multimodal capabilities, finding that vision-dependent tasks—rather than LLM-intrinsic tasks—suffer the most, and that perception degradation is as severe as reasoning degradation. The proposed Extract+Think method (visual extraction tuning + step-by-step reasoning) uses a 0.6B perception module and a 1.7B reasoning module to outperform PrismCaptioner and LLaVA-OneVision-0.5B, which are up to 12× larger.
tags:
  - CVPR 2026
  - Multimodal VLM
  - multimodal model scaling
  - perception bottleneck
  - reasoning bottleneck
  - visual extraction tuning
  - small models
date: 2026-05-08
content_hash: 1fcc4ab2dce8018b
---

# Downscaling Intelligence: Exploring Perception and Reasoning Bottlenecks in Small VLMs

**Conference**: CVPR 2026
**arXiv**: [2511.17487](https://arxiv.org/abs/2511.17487)
**Code**: [https://web.stanford.edu/~markendo/projects/downscaling_intelligence](https://web.stanford.edu/~markendo/projects/downscaling_intelligence) (project page available)
**Area**: Multimodal VLM / Small Language Models
**Keywords**: multimodal model scaling, perception bottleneck, reasoning bottleneck, visual extraction tuning, small models

## TL;DR
This paper systematically investigates the effect of LLM scaling on multimodal capabilities, finding that vision-dependent tasks—rather than LLM-intrinsic tasks—suffer the most, and that perception degradation is as severe as reasoning degradation. The proposed Extract+Think method (visual extraction tuning + step-by-step reasoning) uses a 0.6B perception module and a 1.7B reasoning module to outperform PrismCaptioner and LLaVA-OneVision-0.5B, which are up to 12× larger.

## Background & Motivation

Multimodal large language models (MLLMs) have achieved remarkable progress in visual understanding and reasoning, yet practical deployment demands small, efficient models. The central open question in **small-model research** is: after reducing the LLM backbone, which capabilities degrade most severely and why? Existing studies report contradictory conclusions—some argue that LLM scaling has little effect on perception, while others find that perception-intensive tasks such as OCR are highly sensitive.

The paper's motivation is organized at three levels:

**Understanding practical limitations**: systematically quantifying which tasks are most affected when scaling from 8B to 0.6B.

**Revealing failure mechanisms**: determining whether visual capability degradation stems from weaker reasoning (as expected) or from more fundamental perceptual degradation (unexpected).

**Developing targeted solutions**: designing improvement methods based on the identified bottlenecks.

**Core finding**: LLM scaling disproportionately affects vision-dependent tasks (rather than LLM-intrinsic tasks such as knowledge QA), and **perception degradation is equally or more severe than reasoning degradation**—overturning the prior assumption that perception is insensitive to LLM scale. **Key Insight**: the perception bottleneck originates from visual instruction tuning requiring models to acquire a large diversity of visual extraction skills, which exceeds the capacity of small models.

## Method

### Overall Architecture
Extract+Think is a two-stage perception–reasoning framework. Input: image + question. Stage 1 (perception module / VLM): extracts visual details relevant to the question. Stage 2 (reasoning module / LLM): performs step-by-step reasoning over the extracted visual information to generate an answer. Both modules use the Qwen3 model family; the perception module is a 0.6B or 1.7B VLM, and the reasoning module is a 1.7B or 4B LLM.

### Key Designs

1. **Scaling-Effect Analysis Framework (§3, diagnostic component)**

    - Function: controlled experiments that quantify the effect of LLM scaling on different task categories.
    - Mechanism: the Qwen3 family (8B→4B→1.7B→0.6B) is paired with a SigLIP visual encoder and a 2-layer MLP connector, trained and evaluated uniformly across 15 visual instruction tuning datasets. Two key patterns emerge: (1) the largest performance drops occur on vision-intensive tasks (e.g., Grounding −48%, perceptual similarity −38%), not on LLM-dependent tasks (e.g., ScienceQA is nearly unchanged); (2) the degree to which a task is affected by LLM scaling correlates nearly linearly with its dependence on visual information.
    - Design Motivation: to empirically refute the prevailing assumption that small models degrade primarily in reasoning.

2. **Decoupled Perception/Reasoning Analysis (§3.3, diagnostic component)**

    - Function: separates perception and reasoning to independently measure the effect of LLM scaling on each.
    - Mechanism: the Prism framework decomposes QA into two stages—Stage 1 uses a VLM to extract visual information (perception); Stage 2 uses an LLM for text-based inference (reasoning). Each stage's LLM is scaled independently to observe performance changes. **Surprising finding**: scaling the perception module (8B→0.6B) causes a performance drop nearly as large as scaling the reasoning module—an average in-domain accuracy decrease of 0.15—and for tasks such as Instance Reasoning and Logical Reasoning, the impact of perception scaling is comparable to or exceeds that of reasoning scaling.
    - Design Motivation: prior Prism work assumed perception is far less sensitive to LLM scale (using a 1.8B perception model with a 70B reasoning model). This paper demonstrates that assumption is incorrect. Drawing on the quantization model from Neural Scaling Laws, model skills are "quantized" into discrete chunks; scaling constrains the total number of learnable skills. Visual instruction tuning demands too many perceptual skills, leaving small models with insufficient coverage.

3. **Visual Extraction Tuning (§4.1)**

    - Function: unifies the learning objective of the perception module to alleviate the perception bottleneck.
    - Mechanism: visual instruction tuning data is converted into visual extraction tasks. Pipeline: (1) original QA pairs are transformed into declarative statements; (2) prompts instruct the model to describe fine-grained visual details related to each statement; (3) Qwen3VL-8B generates extraction responses as training data. The perception module is post-trained on 382K samples. This reduces the model's task to learning a single unified visual information extraction skill rather than switching among diverse task formats.
    - Design Motivation: the root cause of the perception bottleneck is that the diversity of visual instruction tuning requires learning too many distinct extraction skills. Plain captioning unifies task format but has two drawbacks: (1) it does not teach models to extract question-relevant information; (2) generic captioning datasets lack domain-specific understanding. Visual Extraction Tuning addresses both issues simultaneously.

4. **Step-by-step Reasoning (§4.2)**

    - Function: enhances the reasoning module's ability to exploit the extracted visual information.
    - Mechanism: Qwen3's thinking mode is activated to enable chain-of-thought (CoT) reasoning. Self-reflection is reduced via NoWait, and the thinking budget is capped at 4,096 tokens.
    - Design Motivation: in the two-stage framework, text serves as the bridge between perception and reasoning; CoT directly strengthens reasoning without additional visual training. Experiments show CoT benefits are largest at intermediate scales (4B, 1.7B)—the 8B model is already sufficiently strong, while the 0.6B model lacks the reasoning capacity to benefit from CoT.

### Loss & Training
- Perception module training: connector pre-training (BLIP-558K) → visual instruction tuning (574K single-image + 309K multi-image + 150K single-image) → captioning post-training (ALLaVA-4V 950K) → visual extraction tuning (382K).
- Reasoning module: Qwen3 is used directly without additional training.
- From-scratch variant (Extract+Think†): trained solely on visual extraction tuning data (382K), without instruction tuning or captioning stages.

## Key Experimental Results

### Main Results: Comparison with Baselines

| Method | Perception/Reasoning Scale | Visual Data | In-Domain Avg. | MMStar |
|--------|:---:|:---:|:---:|:---:|
| LLaVA-OneVision | 0.5B E2E | 8.8M | 71.1 | 39.0 |
| InternVL2.5 | 0.5B E2E | 64M | 83.2 | 48.2 |
| PrismCaptioner | 7B/70B | 1.9M | 78.3 | 45.7 |
| Baseline (§3) | 0.6B E2E | 1.0M | 65.9 | 37.2 |
| Caption+Think | 0.6B/1.7B | 2.0M | 75.0 | 43.0 |
| **Extract+Think** | **0.6B/1.7B** | **2.4M** | **80.3** | **46.6** |
| **Extract+Think** | **1.7B/4.0B** | **2.4M** | **85.3** | **52.6** |

### Ablation Study: Effect of Visual Extraction Tuning

| Perception Module | In-Domain | MMStar | Note |
|-------------------|:---:|:---:|------|
| Captioning 0.6B | 77.6 | 40.4 | plain captioning baseline |
| + Visual Extraction 0.6B | **82.8** | **44.0** | +5.2 / +3.6 gain |
| Captioning 1.7B | 80.3 | 44.4 | plain captioning baseline |
| + Visual Extraction 1.7B | **84.4** | **49.0** | +4.1 / +4.6 gain |

### Key Findings
- **Vision-dependent tasks are most sensitive to LLM scaling**: Grounding drops 48% from 8B to 0.6B, whereas ScienceQA is nearly unchanged. A task's dependence on visual information correlates linearly with its sensitivity to LLM scaling.
- **Perception degradation ≈ Reasoning degradation**: in the decoupled analysis, scaling the perception module causes performance drops comparable to scaling the reasoning module, and for reasoning tasks (IR/LR) the effect of perception scaling is even larger.
- **Visual Extraction Tuning is highly data-efficient**: Extract+Think† trained from scratch with only 382K visual samples (4.3% of LLaVA-OneVision's data) surpasses LLaVA-OneVision in in-domain performance.
- **CoT reasoning is most beneficial at intermediate scales**: the 0.6B model is too small to leverage CoT effectively; the 8B model is already strong enough to not require it; the 4B and 1.7B models benefit most.
- Extract+Think (0.6B/1.7B) outperforms PrismCaptioner on both in-domain and out-of-domain benchmarks using a perception module 12× smaller and a reasoning module 41× smaller.

## Highlights & Insights
- **The counter-intuitive finding that perception is also a core bottleneck**—prior consensus held that small models lose most ground in reasoning (since LLM scale intuitively governs reasoning ability), but this paper demonstrates that perception degrades equally severely, reshaping optimization priorities for small models.
- **Explanatory power of the Neural Scaling Laws quantization model**—the theory that skills are quantized into discrete chunks explains why the diversity of visual instruction tuning amplifies the perception bottleneck: small models can learn too few skill chunks to cover all perceptual patterns.
- **Elegance and practicality of Visual Extraction Tuning**—rather than training small models to handle N different visual understanding formats, the approach consolidates learning into a single skill: "extract visual details relevant to the question." The data generation pipeline is also straightforward.
- **Analytical value of the two-stage decoupled framework**—even setting aside its use as a deployment architecture, the decoupled analysis itself provides a novel diagnostic tool for small-model VLM research.

## Limitations & Future Work
- The two-stage framework introduces additional inference latency (two model forward passes), making it less suitable for real-time applications.
- Textual output from the perception module may lose fine-grained visual information, as text serves as an information bottleneck between vision and reasoning.
- Visual Extraction Tuning relies on Qwen3VL-8B to generate training data, introducing teacher model bias.
- Validation is limited to the Qwen3 family; generalization across architectures (e.g., LLaMA, Gemma) remains unknown.
- The visual encoder (SigLIP) is held constant across all experiments; the effect of encoder scaling on perception is not analyzed.
- CoT reasoning increases output length and thus inference cost; the 4,096-token thinking budget may be insufficient for complex reasoning tasks.

## Related Work & Insights
- **vs. Prism framework**: Prism assumes perception is insensitive to LLM scale (using a small LLM for perception with a large LLM for reasoning); this paper refutes that assumption and proposes Visual Extraction Tuning as an alternative.
- **vs. LLaVA-OneVision**: end-to-end training uses 8.8M visual samples, whereas Extract+Think† achieves superior performance over the 0.5B variant using only 382K samples (a 95% reduction in data).
- **vs. VLM failure analysis work**: prior work focuses on failure patterns in large models (e.g., underutilization of visual information); this paper is the first to systematically analyze failure mechanisms specific to small models.
- The concept of Visual Extraction Tuning generalizes to other scenarios requiring unification of heterogeneous tasks—the core insight being "reduce skill diversity, focus on core capability."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — the perception bottleneck finding shifts field-wide understanding; Visual Extraction Tuning is conceptually novel and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 15 tasks × 4 model scales × decoupled analysis × ablation studies × multiple baselines; analysis is highly systematic.
- Writing Quality: ⭐⭐⭐⭐⭐ — the progressive structure from problem identification → root cause analysis → solution proposal is exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ — the work has far-reaching methodological and practical implications for small-model VLM research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy](nano-emox_unifying_multimodal_emotional_intelligence_from_perception_to_empathy.md)
- [\[ICLR 2026\] Empowering Small VLMs to Think with Dynamic Memorization and Exploration](../../ICLR2026/multimodal_vlm/empowering_small_vlms_to_think_with_dynamic_memorization_and_exploration.md)
- [\[CVPR 2026\] SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence](spatialscore_towards_comprehensive_evaluation_for_spatial_intelligence.md)
- [\[CVPR 2026\] CodePercept: Code-Grounded Visual STEM Perception for MLLMs](codepercept_code-grounded_visual_stem_perception_for_mllms.md)
- [\[CVPR 2026\] Proof-of-Perception: Certified Tool-Using Multimodal Reasoning with Compositional Conformal Guarantees](pop_proof_of_perception_conformal_reasoning.md)

</div>

<!-- RELATED:END -->
