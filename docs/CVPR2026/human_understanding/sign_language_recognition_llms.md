---
title: >-
  [Paper Note] Sign Language Recognition in the Age of LLMs
description: >-
  [CVPR 2026][Human Understanding][Vision-Language Model] The first study to systematically evaluate the zero-shot capabilities of modern VLMs on Isolated Sign Language Recognition (ISLR). Results show that open-source VLMs lag significantly behind specialized classifiers, while large commercial models (GPT-5) demonstrate surprising potential.
tags:
  - CVPR 2026
  - Human Understanding
  - Vision-Language Model
date: 2026-05-08
content_hash: d3667bd5a2f36875
---
# Sign Language Recognition in the Age of LLMs

**Conference**: CVPR 2026  
**arXiv**: [2604.11225](https://arxiv.org/abs/2604.11225)  
**Code**: [https://github.com/VaJavorek/WLASL_LLM](https://github.com/VaJavorek/WLASL_LLM)  
**Area**: Human Understanding  
**Keywords**: Sign Language Recognition, Vision-Language Models, Zero-shot, American Sign Language, Benchmark

## TL;DR
The first study to systematically evaluate the zero-shot capabilities of modern VLMs on Isolated Sign Language Recognition (ISLR). Results show that open-source VLMs lag significantly behind specialized classifiers, while large commercial models (GPT-5) demonstrate surprising potential.

## Background & Motivation

**Background**: Sign language recognition (SLR) traditionally relies on task-specific supervised learning, requiring massive labeled datasets and dedicated architectures. Meanwhile, VLMs have demonstrated powerful multimodal reasoning capabilities, yet their application in SLR remains largely unexplored.

**Limitations of Prior Work**: (1) Supervised methods are constrained by labeled data and struggle with generalization across signers and environments; (2) VLMs are primarily evaluated on natural images and videos, often missing fine-grained sign language gestures and movements; (3) There is a lack of systematic benchmarks for zero-shot SLR with VLMs.

**Key Challenge**: VLMs are highly generalistic but are not specifically trained on sign language data. Does the high-dimensional spatio-temporal complexity and subtle linguistic structure of sign language exceed the zero-shot capabilities of current VLMs?

**Core Idea**: This study returns to the controlled setting of ISLR to systematically evaluate the zero-shot sign language recognition capabilities of multiple VLMs, analyzing the impact of prompting strategies and model scale.

## Method

### Overall Architecture
This paper does not propose a new model; instead, it investigates a core question: without any specialized sign language training, how well can current VLMs recognize isolated sign language videos? Using the WLASL300 (300 American Sign Language glosses) as a controlled benchmark, the study evaluates various open-source and commercial VLMs under uniform frame sampling and prompt templates using three evaluation paradigms: standard multi-class classification, zero-shot open-set prediction (direct generation of the gloss), and zero-shot binary classification (verifying if the video matches a given candidate word). Comparative analyses are conducted on prompt information, frame counts, and model scale, followed by a synonym-tolerant re-evaluation for a fairer assessment.

### Key Designs

**1. Systematic Multi-model Evaluation: Establishing a comparable baseline for "how much sign language VLMs can recognize"**

SLR has long been dominated by task-specific supervised classifiers, and the zero-shot upper bound of VLMs has rarely been systematically measured. This work places LLaVA-NeXT-Video, InternVL3.5, Qwen2.5/3-VL, and BAGEL (open-source) alongside GPT-5 and Gemini (commercial) in a unified pipeline using identical frame sampling, prompt templates, and the WLASL300 vocabulary. By unifying variables, the performance gap becomes quantifiable: most open-source VLMs stagnate below 3% (near random), while GPT-5 reaches 14.67%. This horizontal benchmark provides the first quantifiable answer as to whether the performance gap stems from scale or methodology.

**2. Multi-level Prompting Strategy: Probing performance bottlenecks through progressively restricted output spaces**

A fundamental difference between VLMs and classifiers is that classifiers select from a fixed set of classes, whereas VLM output exists in natural language space. A model might generate a valid word that is simply not in the vocabulary. To address this, the paper ranks prompts by information levels: fully open (free generation) → providing the dataset name (constraining the domain to WLASL) → providing a candidate list (constricting the output to a finite set). GPT-5 accuracy rises monotonically as constraints tighten, showing a significant boost when provided with a candidate list. This suggests a significant portion of failures are due to the vastness of the output space rather than absolute visual blindness. Adding a binary classification layer (Is/No match for a given word) allows GPT-5 to reach ~30% precision, proving that models do capture some semantic alignment between sign and text.

**3. Synonym-aware Evaluation: Preventing "correct meaning, different word" errors**

Open-ended generation faces an unfair penalty where a model might output a semantically correct synonym (e.g., "glad" instead of the ground truth "happy"). Strict string matching misclassifies these cases, systematically underestimating VLM understanding. This study addresses this by retrieving synonym sets from WordNet for each ground truth word; a prediction is marked correct if it matches any term in the set. This adjustment raises GPT-5's Top-1 from 14.67% to 17.96%. The difference quantifies the portion of results "wrongly accused" due to lexical variation, shifting evaluation from literal matching to semantic accuracy.

### Loss & Training
This is a pure zero-shot evaluation; no parameters are updated, and thus there are no training objectives or loss functions. All variances stem from prompt design, frame sampling, and the models themselves.

## Key Experimental Results

### Main Results

| Model | Top-1 | Top-1 + Synonyms | Description |
| :--- | :--- | :--- | :--- |
| Specialized SOTA (DSLNet) | 89.97% | - | Supervised training |
| GPT-5 (64 frames) | 14.67% | 17.96% | Best commercial |
| Qwen3-VL-30B | 2.40% | 3.59% | Best open-source |
| LLaVA-NeXT-7B | 0.30% | 0.45% | Worst open-source |

### Ablation Study

| Prompting Strategy | GPT-5 Accuracy | Description |
| :--- | :--- | :--- |
| Open-set | 14.67% | Unconstrained |
| Provide dataset name | Slight Gain | Constrained output space |
| Provide candidate list | Significant Gain | Strongest constraint |
| Binary classification | ~30% Precision | Presence of visual-semantic alignment |

### Key Findings
- Open-source VLMs almost entirely fail on zero-shot ISLR (< 3%), far below specialized classifiers.
- GPT-5's performance significantly exceeds open-source models, suggesting that model scale and diverse training data are crucial.
- Binary classification experiments indicate that VLMs do capture some degree of sign-to-text semantic alignment.
- Certain models (e.g., Nemotron) honestly respond with "I don't know," which lowers measured performance but reflects real capabilities.

## Highlights & Insights
- **Honest Negative Results**: Does not shy away from the severe deficiencies of VLMs in SLR, providing a realistic benchmark for the community.
- **Evident Scaling Effects**: The massive gap between GPT-5 and open-source models suggests sign language may require more vision-motion pre-training data.

## Limitations & Future Work
- Only tested on WLASL300, not covering larger vocabularies or continuous sign language.
- Latency of commercial APIs limits the feasibility of large-scale evaluation.
- Future work could explore few-shot fine-tuning of VLMs or specialized sign language visual encoders.

## Related Work & Insights
- **vs. Traditional ISLR**: Specialized methods like ST-GCN perform exceptionally well under supervision, indicating that task-specific training remains irreplaceable.
- **vs. Elysium/ChatTracker**: These MLLM trackers also require fine-tuning; pure zero-shot remains insufficient.

## Rating
- Novelty: ⭐⭐⭐ The evaluation study itself is not a new method but fills a significant gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive across models, prompts, and evaluation paradigms.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and deep analysis.
- Value: ⭐⭐⭐⭐ Provides an important VLM baseline for sign language AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Effective Sign Features without Text for Gloss-free Sign Language Translation](learning_effective_sign_features_without_text_for_gloss-free_sign_language_trans.md)
- [\[CVPR 2026\] Text-Driven 3D Hand Motion Generation from Sign Language Data](text-driven_3d_hand_motion_generation_from_sign_language_data.md)
- [\[CVPR 2026\] SignPR: A Progressive Vector-Quantized Diffusion Framework for Sign Language Production](signpr_a_progressive_vector-quantized_diffusion_framework_for_sign_language_prod.md)
- [\[CVPR 2026\] BoostSLT: Boosting Sign Language Translation via a Plug-and-Play Diffusion-Based Semantic Enhancer](boostslt_boosting_sign_language_translation_via_a_plug-and-play_diffusion-based_.md)
- [\[CVPR 2026\] Focal–General Diffusion Model with Semantic Consistent Guidance for Sign Language Production](focal-general_diffusion_model_with_semantic_consistent_guidance_for_sign_languag.md)

</div>

<!-- RELATED:END -->
