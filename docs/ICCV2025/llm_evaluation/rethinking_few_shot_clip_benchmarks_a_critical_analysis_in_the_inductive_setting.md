---
title: >-
  [Paper Note] Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting
description: >-
  [ICCV 2025][LLM Evaluation][CLIP] This paper identifies that existing CLIP few-shot classification benchmarks constitute a "partially transductive setting" due to CLIP's exposure to test datasets during pretraining. It p…
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "CLIP"
  - "few-shot learning"
  - "inductive setting"
  - "machine unlearning"
  - "benchmark evaluation"
date: 2026-05-08
content_hash: 3733b4c8a1b4f2b4
---

# Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting

**Conference**: ICCV 2025
**arXiv**: [2507.20834](https://arxiv.org/abs/2507.20834)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: CLIP, few-shot learning, inductive setting, machine unlearning, benchmark evaluation

## TL;DR

This paper identifies that existing CLIP few-shot classification benchmarks constitute a "partially transductive setting" due to CLIP's exposure to test datasets during pretraining. It proposes an unlearning-based inductive benchmark evaluation framework and introduces a few-shot classification method that achieves stable state-of-the-art performance under the new benchmark.

## Background & Motivation

**Background**: CLIP has demonstrated strong zero-shot and few-shot transfer capabilities through large-scale vision-language pretraining. A substantial body of work—CoOp, Tip-Adapter, CLIP-Adapter, TaskRes, and others—has reported significant performance gains on standard datasets such as ImageNet, Flowers102, and EuroSAT.

**Limitations of Prior Work**: All of these methods are evaluated on the same standard few-shot datasets. However, a critical and overlooked issue is that CLIP has very likely encountered the images in these datasets—or their variants—during pretraining. This implies that CLIP's classification capability on these datasets is not purely derived from generalizing from few-shot examples, but is partially attributable to pretraining memorization. Such evaluation is therefore a "partially transductive" setting rather than a genuinely "inductive" one.

**Key Challenge**: The benchmarks purport to measure the ability to generalize inductively from a small number of labeled examples, yet the pretraining memorization of the backbone model substantially contaminates the evaluation results, rendering method comparisons unfair and failing to reflect genuine generalization to truly unseen data.

**Goal**: (1) Quantify the impact of pretraining memorization on few-shot evaluation; (2) provide an evaluation framework that eliminates this bias; (3) re-evaluate existing methods under the new inductive setting and propose a stronger baseline.

**Key Insight**: The authors introduce machine unlearning to selectively cause CLIP to "forget" knowledge of target datasets without full retraining, thereby constructing a truly inductive evaluation baseline.

**Core Idea**: Machine unlearning is used to simulate an environment in which CLIP has never encountered the target dataset, enabling re-evaluation of few-shot methods under this controlled condition to reveal their true generalization capability.

## Method

### Overall Architecture

The overall framework consists of two components: (1) **Inductive Benchmark Construction**—machine unlearning is applied to cause CLIP to forget dataset-specific knowledge, yielding a model that has effectively never seen the target dataset; (2) **Improved Few-Shot Classifier**—a classification strategy combining visual and textual features is designed for the inductive setting. The input is a CLIP model together with a small set of labeled examples; the output is classification predictions on the target dataset.

### Key Designs

1. **Selective Unlearning Mechanism**:

    - *Function*: Causes the CLIP model to selectively forget knowledge pertaining to a specific dataset while retaining general vision-language capabilities.
    - *Mechanism*: Gradient ascent is employed: the contrastive loss is computed on training samples from the target dataset, and model parameters are updated to increase this loss (i.e., the model is made to "forget" how to correctly match these samples). Crucially, unlearning is applied only to the text encoder while keeping the visual encoder fixed—since visual feature generality is more important, and the mapping from class names to features in the text branch is the primary carrier of memorization. The intensity of unlearning is controlled via the number of gradient ascent steps and the learning rate.
    - *Design Motivation*: Full retraining of CLIP from scratch is neither feasible nor practical. Selective unlearning provides an efficient alternative that eliminates dataset-specific memorization bias while preserving general capabilities.

2. **Oracle Validation Mechanism**:

    - *Function*: Validates the effectiveness of unlearning.
    - *Mechanism*: Two oracle baselines are designed—(a) *Random Init Oracle*: the final layers of CLIP are randomly re-initialized before few-shot adaptation, representing the lower bound with no memorization; (b) *Full Retrain Oracle*: CLIP is retrained on data excluding the target dataset, representing the upper bound of a model that has genuinely never seen the target data. The validity of unlearning is assessed by checking whether the performance of the unlearned model falls between these two oracles.
    - *Design Motivation*: Since there is no ground truth for "perfect forgetting," oracle baselines are necessary to establish a confidence interval that calibrates unlearning to be neither excessive nor insufficient.

3. **Enhanced Few-Shot Classifier**:

    - *Function*: Achieves robust few-shot classification under the inductive setting.
    - *Mechanism*: A classification method combining prototypical network ideas with vision-language alignment is proposed. Specifically: (a) the mean visual feature of $K$ samples per class is computed as the class prototype; (b) class prototypes and CLIP text features are fused via learnable weights; (c) feature normalization and temperature scaling are incorporated. Classification is performed by computing cosine similarity between a test sample and the fused class prototypes. The entire adaptation process optimizes only the fusion weights and temperature parameter, requiring minimal parameters.
    - *Design Motivation*: Under the inductive setting, complex methods such as prompt tuning fail because they rely on pretraining memorization for initialization. Simple and robust prototypical approaches prove more reliable in genuine few-shot scenarios.

### Loss & Training

During the few-shot adaptation stage, cross-entropy loss is used to fine-tune the fusion weights on the $K$-shot support set. During the unlearning stage, a negated contrastive loss is used (i.e., the standard CLIP contrastive loss is negated and minimized, which is equivalent to gradient ascent). The entire process does not require large-scale computational resources.

## Key Experimental Results

### Main Results

Comparison of 13 CLIP few-shot methods under the standard setting vs. the inductive setting across 11 datasets (16-shot):

| Method | Standard Acc. | Inductive Acc. | Drop |
|--------|--------------|---------------|------|
| Zero-shot CLIP | 63.7% | 28.6% | −55.1% |
| CoOp | 71.2% | 33.8% | −52.5% |
| Tip-Adapter-F | 73.1% | 35.2% | −51.8% |
| CLIP-Adapter | 71.8% | 34.1% | −52.5% |
| TaskRes | 73.5% | 36.4% | −50.5% |
| Ours | 73.8% | 41.2% | −44.2% |

### Ablation Study

| Configuration | Inductive Acc. | Note |
|---------------|---------------|------|
| Full method (Ours) | 41.2% | Fused prototypes + temperature scaling |
| w/o text feature fusion | 38.6% | Visual prototypes only |
| w/o temperature scaling | 39.8% | Fixed temperature $\tau=1$ |
| Uniform fusion instead of learnable weights | 39.3% | Simple averaging inferior to adaptive |
| Unlearning on visual encoder only | 35.1% | Poor; confirms memorization resides mainly in text branch |
| Unlearning on both encoders | 30.2% | Over-forgetting damages general capabilities |

### Key Findings

- **Performance Collapse**: Under the inductive setting, all 13 methods drop by an average of 55%, revealing that current benchmarks severely overestimate the generalization ability of these methods.
- **Ranking Shifts**: Methods that lead under the standard setting (e.g., prompt tuning approaches) suffer the largest ranking drops in the inductive setting, indicating that their "performance gains" largely stem from exploiting pretraining memorization.
- **Ours is Most Robust**: The proposed method exhibits the smallest performance drop in the inductive setting (−44.2% vs. average −55%) and achieves consistent state-of-the-art across all datasets.
- **Text Encoder as the Core of Memorization**: Applying unlearning solely to the text encoder effectively eliminates dataset-specific bias, while applying it to the visual encoder instead degrades fundamental capabilities.

## Highlights & Insights

- **Exposing a Fundamental Evaluation Flaw**: The paper identifies a systematic methodological bias in the entire CLIP few-shot learning field. This is not a problem with any particular method but a community-wide consensus issue. This meta-research perspective is highly valuable.
- **Unlearning as an Evaluation Tool**: Machine unlearning is transferred from the privacy-preservation domain to benchmark evaluation, creatively addressing the challenge of simulating "unseen data" without retraining large foundation models. This paradigm can generalize to downstream evaluation of any foundation model.
- **Experimental Scale**: 5,880 experiments spanning multiple datasets, shot counts, random seeds, and unlearning configurations make the findings statistically robust and difficult to attribute to chance.

## Limitations & Future Work

- Whether unlearning can perfectly simulate the "never seen" condition remains uncertain, particularly for low-level visual knowledge encoded in early layers that may not be fully eliminated.
- Experiments are primarily conducted on ViT-B/16; it is unclear whether the conclusions hold for larger CLIP variants (ViT-L, ViT-G).
- While the proposed method achieves the best performance in the inductive setting, the absolute accuracy remains limited (41.2%), indicating that true few-shot learning remains an open problem.
- More refined unlearning strategies—such as layer-wise selective forgetting and influence-function-based precise removal—are worth exploring.
- Future work should establish standardized inductive few-shot benchmarks to promote fairer evaluation practices in the community.

## Related Work & Insights

- **vs. CoOp/CoCoOp**: These prompt tuning methods perform strongly in the standard setting but suffer the largest drops in the inductive setting, suggesting that prompt optimization largely amounts to "recalling" pretrained knowledge.
- **vs. Tip-Adapter**: Cache-based methods are relatively robust in the inductive setting because they leverage few-shot samples more directly rather than relying on pretrained encodings.
- **vs. FLYP/WiSE-FT**: Full model fine-tuning approaches are also severely affected by pretraining bias, and their advantages largely disappear in the inductive setting.
- This work serves as a reminder that, when evaluating any downstream adaptation method for foundation models, the overlap between pretraining and evaluation datasets must be carefully considered.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Exposes a systematic evaluation flaw overlooked by the entire field; contribution is distinctive and far-reaching.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5,880 experiments, 13 baselines, and 11 datasets—extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear argumentation and rigorous logical structure.
- Value: ⭐⭐⭐⭐⭐ Likely to reshape evaluation standards for CLIP few-shot classification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Few-Shot Pattern Detection via Template Matching and Regression](few-shot_pattern_detection_via_template_matching_and_regression.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](../../NeurIPS2025/llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](batclip_bimodal_online_test-time_adaptation_for_clip.md)
- [\[AAAI 2026\] Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](../../AAAI2026/llm_evaluation/lost_in_benchmarks_rethinking_large_language_model_benchmarking_with_item_respon.md)
- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](../../NeurIPS2025/llm_evaluation/unlocking_transfer_learning_for_open-world_few-shot_recognition.md)

</div>

<!-- RELATED:END -->
