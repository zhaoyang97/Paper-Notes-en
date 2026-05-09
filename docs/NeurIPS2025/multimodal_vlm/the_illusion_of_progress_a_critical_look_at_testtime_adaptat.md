---
title: >-
  [Paper Note] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Test-time adaptation] This paper introduces TTA-VLM, a unified benchmark evaluating 8 episodic and 7 online test-time adaptation (TTA) methods across 15 datasets under controlled experimental conditions. Three surprising findings emerge: (1) existing TTA methods offer only marginal improvements over the early TPT baseline; (2) TTA methods collaborate poorly with training-time fine-tuning approaches; (3) accuracy gains come at the cost of calibration, OOD detection, and robustness.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Test-time adaptation
  - CLIP
  - SigLIP
  - trustworthiness
  - calibration
date: 2026-05-08
content_hash: 03a3e85b47d38dd9
---

# The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.24000](https://arxiv.org/abs/2506.24000)
**Code**: [https://github.com/TomSheng21/tta-vlm](https://github.com/TomSheng21/tta-vlm)
**Area**: Multimodal VLM / Test-Time Adaptation / Benchmark
**Keywords**: Test-time adaptation, CLIP, SigLIP, trustworthiness, calibration

## TL;DR
This paper introduces TTA-VLM, a unified benchmark evaluating 8 episodic and 7 online test-time adaptation (TTA) methods across 15 datasets under controlled experimental conditions. Three surprising findings emerge: (1) existing TTA methods offer only marginal improvements over the early TPT baseline; (2) TTA methods collaborate poorly with training-time fine-tuning approaches; (3) accuracy gains come at the cost of calibration, OOD detection, and robustness.

## Background & Motivation
Test-time adaptation for VLMs has attracted growing attention due to its appealing property of improving model performance at inference time without labeled data. However, existing TTA studies suffer from severe comparability issues: different papers use different pre-trained checkpoints, different text templates, different evaluation protocols, and directly cite baseline numbers from other papers rather than reproducing them. This raises the concern that the apparent "continuous progress" may be an illusion caused by inconsistent experimental setups.

More critically, nearly all TTA papers report only accuracy, neglecting other equally important metrics for real-world deployment—expected calibration error (ECE), OOD detection capability (AUC), and adversarial robustness. It remains an open question whether TTA methods trade model trustworthiness for marginal accuracy gains.

## Core Problem
**Under unified and fair evaluation conditions, how much genuine progress have existing VLM TTA methods made, and how do they perform beyond accuracy?** Three sub-questions are addressed: (1) After harmonizing experimental setups to eliminate confounding factors, what are the true performance gaps between TTA methods and relative to baselines? (2) Can TTA methods cooperate effectively with training-time fine-tuning approaches such as CoOp and MaPLe? (3) Do accuracy improvements come at the expense of trustworthiness?

## Method

### Overall Architecture
TTA-VLM is a comprehensive benchmark that proposes no new method, but instead provides a unified evaluation framework. It covers two TTA paradigms:
- **Episodic TTA**: Per-sample adaptation using 64 AugMix-augmented views of each test sample for immediate adaptation.
- **Online TTA**: Stream-based processing of the test data (batch size=1), accumulating adaptation information over time.

### Key Designs

1. **Unified Experimental Setup**: All methods share the same pre-trained model checkpoints, text templates ("a photo of a [CLASS]"), data augmentation pipelines, and evaluation protocols. Only each method's originally recommended hyperparameters are retained. Evaluation is conducted on four models: CLIP-ResNet50, ViT-B/16, ViT-B/32, and SigLIP-ViT-B/16.

2. **Multi-dimensional Evaluation Metrics**:

    - **Accuracy**: Standard classification accuracy.
    - **Calibration (ECE)**: 20-bin ECE measuring alignment between predicted confidence and actual correctness.
    - **OOD Detection (AUC)**: 50% of classes are withheld, with corresponding samples treated as OOD, measuring the model's ability to identify unknown-class samples.
    - **Adversarial Robustness**: PGD-generated adversarial examples are used to evaluate the defensive effectiveness of TTA methods.
    - **Streaming Stability**: Online TTA performance under data streams containing OOD or adversarial samples.

3. **Training-Time Fine-Tuning + TTA Collaboration**: TTA methods are applied on top of CoOp-, MaPLe-, and TeCoA-fine-tuned models to assess whether the two-stage pipeline is complementary.

### TTA Methods Covered
- **Episodic (8 methods)**: TPT, C-TPT, RLCF, MTA, ZERO, TTL, TPS, R-TPT
- **Online (7 methods)**: TDA, DMN/DMNW, OnZeta, BoostAdapter, DPE, ECALP, DynaPrompt

## Key Experimental Results

### Accuracy Under Fair Comparison (CLIP-ResNet50)

| Method | Fine-grained Avg | ImageNet-X Avg |
|--------|-----------------|----------------|
| CLIP baseline | 55.84% | 44.19% |
| TPT (2022, pioneering work) | 57.80% | 47.21% |
| ECALP (best episodic) | 59.38% | 46.06% |
| BoostAdapter (best online) | 58.60% | 48.07% |

**Key Finding**: TPT (the pioneering work from 2022) remains one of the strongest baselines under fair comparison, with subsequent methods averaging no more than 1.5% improvement.

### Generalization on SigLIP
On SigLIP-ViT-B/16, most TTA methods **fail to surpass the zero-shot baseline** on fine-grained tasks. Only 3–4% improvements are observed on ImageNet-related datasets, indicating that existing TTA methods are highly tailored to CLIP-specific properties and lack generalizability.

### Collaboration with Training-Time Fine-Tuning

| Base Model | Best TTA Avg | Baseline Avg | Gain |
|------------|-------------|-------------|------|
| CoOp | 58.86% (BoostAdapter) | 56.20% | +2.66% |
| MaPLe | 67.79% (ECALP) | 64.63% | +3.16% |
| TeCoA | 41.19% (ECALP) | 36.23% | +4.96% |

While gains appear positive, **nearly all episodic TTA methods degrade performance on TeCoA** (negative transfer). DPE, for instance, drops from 36.23% to 18.18% on TeCoA.

### Trustworthiness Degradation

| Method | Fine-grained ECE↓ | ECE Increase |
|--------|------------------|-------------|
| CLIP baseline | 5.70% | 0 |
| TPT | 11.30% | +5.60% |
| ECALP | 32.21% | +26.51% |
| TPS | 21.16% | +15.46% |
| C-TPT (specially designed) | 6.61% | +0.91% |

**All TTA methods increase calibration error**, with ECALP worsening ECE from 5.70% to 32.21% (a 6× deterioration). C-TPT, which incorporates an explicit calibration regularization term, is the only relatively well-controlled method. OOD detection AUC decreases by 1–4% across methods. Most episodic TTA methods exhibit near-zero adversarial accuracy on CLIP-ViT.

### Ablation Study Highlights
- Multi-template strategies provide additional gains for most TTA methods (e.g., +3.1% for ZERO on DTD), though some online methods show degradation.
- TPT exhibits inconsistent performance across backbones (strongest on ResNet50 but weaker on ViT-B/32), suggesting underappreciated architecture sensitivity.
- Online TTA performance generally drops by 1–2% under data streams containing adversarial samples.

## Highlights & Insights
- **Exposing "illusory progress" in the field**: Under unified evaluation, TPT from 2022 remains one of the strongest baselines, with three subsequent years of methods offering negligible improvements—a sobering signal for the TTA community.
- **First systematic evaluation of trustworthiness costs**: This is the first work to quantify the degradation in calibration, OOD detection, and robustness incurred by TTA, revealing that these methods can make models less reliable in practice.
- **Cross-architecture generalization testing**: The first evaluation of TTA methods on SigLIP in addition to CLIP, exposing architecture-dependent behavior.
- **Well-designed benchmark framework**: All TTA methods are implemented and evaluated under a unified pipeline with open-sourced code, providing practical value for future TTA research.
- **Computational cost analysis**: Inference time and GPU memory consumption are quantitatively compared across 15 methods (e.g., DynaPrompt requires 1,157s and 43GB, representing 16× the inference time and 30× the GPU memory of vanilla CLIP).

## Limitations & Future Work
- **Classification tasks only**: TTA effects on broader VLM tasks such as VQA, image captioning, and segmentation are not evaluated.
- **Methods requiring extra resources excluded**: TTA approaches leveraging LLMs, generative models, or ImageNet statistics—which may be stronger—are out of scope.
- **Hyperparameter tuning is sidestepped**: All methods use originally recommended hyperparameters, yet test-time hyperparameter selection remains an open problem in itself.
- **Insufficient constructive direction**: The paper is primarily diagnostic; it offers limited guidance on how to build better TTA methods.
- **Extension to larger generative VLMs** (e.g., LLaVA, Qwen-VL) would be a natural and valuable follow-up.

## Related Work & Insights
- **TPT (NeurIPS 2022)**: The reference point of this benchmark and the pioneering work on test-time prompt tuning for VLMs. Notably, subsequent methods show almost no improvement over TPT under fair comparison.
- **MTA (CVPR 2024)**: A training-free method requiring no gradient updates. It achieves moderate accuracy in this benchmark but exhibits relatively better robustness (24.24% adversarial accuracy vs. TPT's 0.03%).
- **ECALP (ICLR 2025)**: Among the highest-accuracy online methods, but with the worst calibration (ECE increases by 26.51%), exemplifying the accuracy–trustworthiness trade-off.
- **C-TPT (ICLR 2024)**: Incorporates an explicit calibration regularization term and is the only method that improves accuracy while maintaining reasonable calibration (ECE increase of only 0.91%)—a design worth drawing lessons from.

The findings suggest that TTA modules should not be applied indiscriminately; in settings involving fine-tuned models or calibration-sensitive deployments, TTA may do more harm than good. The accuracy–calibration trade-off represents a promising research direction—can TTA methods be designed to inherently preserve calibration? C-TPT offers a preliminary answer. The benchmark methodology is also transferable: similar "sober reassessment" benchmarks are needed for other VLM tasks such as VQA, grounding, and video understanding.

## Rating
- **Novelty**: ⭐⭐⭐ No new method is proposed, but the systematic diagnosis of the field has independent value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 15 methods × 15 datasets × 4 models × multiple metrics—an extensive and comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Conclusions are clear and compelling, though the abundance of tables somewhat hinders readability.
- **Value**: ⭐⭐⭐⭐ A necessary "reality check" for the TTA community with a practical benchmark framework, though constructive directions for improvement are lacking.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[NeurIPS 2025\] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](../../ICCV2025/multimodal_vlm/latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[NeurIPS 2025\] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)
- [\[NeurIPS 2025\] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models](test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)

<!-- RELATED:END -->
