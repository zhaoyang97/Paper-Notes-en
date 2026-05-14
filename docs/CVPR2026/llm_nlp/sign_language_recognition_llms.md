---
title: >-
  [Paper Note] Sign Language Recognition in the Age of LLMs
description: >-
  [CVPR 2026][LLM/NLP][sign language recognition] The first systematic evaluation of modern VLMs on zero-shot isolated sign language recognition (ISLR)…
tags:
  - "CVPR 2026"
  - "LLM/NLP"
  - "sign language recognition"
  - "vision-language model"
  - "zero-shot"
  - "American Sign Language"
  - "benchmark"
date: 2026-05-08
content_hash: 43e31383a4b3e584
---

# Sign Language Recognition in the Age of LLMs

**Conference**: CVPR 2026  
**arXiv**: [2604.11225](https://arxiv.org/abs/2604.11225)  
**Code**: [https://github.com/VaJavorek/WLASL_LLM](https://github.com/VaJavorek/WLASL_LLM)  
**Area**: LLM / NLP (Other)  
**Keywords**: sign language recognition, vision-language model, zero-shot, American Sign Language, benchmark

## TL;DR
The first systematic evaluation of modern VLMs on zero-shot isolated sign language recognition (ISLR), revealing that open-source VLMs fall far behind specialized classifiers while large commercial models (GPT-5) demonstrate surprising potential.

## Background & Motivation

**Background**: Sign language recognition has traditionally relied on task-specific supervised learning, requiring large amounts of annotated data and specialized architectures. Meanwhile, VLMs have demonstrated powerful multimodal reasoning capabilities, but their application to sign language remains largely unexplored.

**Limitations of Prior Work**: (1) Supervised methods are limited by annotated data availability and cross-signer/environment generalization; (2) VLMs are primarily evaluated on natural images/videos, with fine-grained gestural motions in sign language not covered; (3) No systematic benchmark exists for VLM zero-shot sign language recognition.

**Key Challenge**: VLMs are highly general but not specifically trained on sign language data. The high-dimensional spatiotemporal complexity and subtle linguistic structure of sign language may exceed VLMs' zero-shot capabilities.

**Core Idea**: Return to the controlled ISLR setting to systematically evaluate multiple VLMs' zero-shot sign language recognition capabilities, analyzing the effects of prompting strategies and model scale.

## Method

### Overall Architecture
Evaluate multiple open-source and commercial VLMs on the WLASL300 benchmark (300 sign language vocabulary items) → Three evaluation paradigms: (1) standard multi-class classification, (2) zero-shot open-set prediction, (3) zero-shot binary classification (judging whether the sign in a video matches a specified word) → Analyze effects of prompting strategies, frame sampling, and model scale.

### Key Designs

1. **Systematic Multi-Model Evaluation**:

    - Function: Establish baselines for VLM zero-shot ISLR
    - Mechanism: Evaluate LLaVA-NeXT-Video, InternVL3.5, Qwen2.5/3-VL, BAGEL, GPT-5, Gemini, and other models with unified prompt templates and frame sampling strategies
    - Design Motivation: Provide the community with a clear reference for "what VLMs can achieve" in sign language

2. **Multi-Level Prompting Strategy**:

    - Function: Explore the impact of information quantity on zero-shot performance
    - Mechanism: From fully open → specifying the dataset → providing candidate vocabulary lists, progressively constraining the output space. Additionally tests binary classification (given a word description, judge whether it matches) and synonym-tolerant evaluation
    - Design Motivation: VLMs' output space is far larger than a classifier's fixed category set; constraining the output space may significantly affect performance

3. **Synonym-Aware Evaluation**:

    - Function: More fairly evaluate VLMs' semantic understanding
    - Mechanism: Obtain synonym lists for each ground truth word from WordNet; predicted synonyms are also counted as correct
    - Design Motivation: VLMs may output semantically correct but differently worded predictions (e.g., "happy" vs "glad")

### Loss & Training
Pure zero-shot evaluation with no training.

## Key Experimental Results

### Main Results

| Model | Top-1 | Top-1+Synonyms | Note |
|-------|-------|----------------|------|
| Specialized SOTA (DSLNet) | 89.97% | - | Supervised training |
| GPT-5 (64 frames) | 14.67% | 17.96% | Best commercial model |
| Qwen3-VL-30B | 2.40% | 3.59% | Best open-source model |
| LLaVA-NeXT-7B | 0.30% | 0.45% | Worst open-source model |

### Ablation Study

| Prompting Strategy | GPT-5 Accuracy | Note |
|-------------------|----------------|------|
| Open-set | 14.67% | Unconstrained |
| Specify dataset name | Slight improvement | Constrains output space |
| Provide candidate list | Significant improvement | Strongest constraint |
| Binary classification | ~30% precision | Partial visual-semantic alignment exists |

### Key Findings
- Open-source VLMs almost completely fail at zero-shot ISLR (< 3%), far below specialized classifiers
- GPT-5 significantly outperforms open-source models, indicating model scale and training data diversity are critical
- Binary classification experiments show VLMs do capture partial sign language-text semantic alignment
- Some models (e.g., Nemotron) honestly answer "I don't know," lowering measured performance but reflecting true capability

## Highlights & Insights
- **Honest negative results**: Without avoiding VLMs' severe shortcomings in sign language, this provides a realistic baseline for the community
- **Scale effect is prominent**: The massive gap between GPT-5 and open-source models suggests sign language may require more visual-motor pre-training data

## Limitations & Future Work
- Only tested on WLASL300; larger vocabularies and continuous sign language are not covered
- Commercial API latency limits large-scale evaluation feasibility
- Future work could explore few-shot fine-tuning of VLMs or specialized sign language visual encoders

## Related Work & Insights
- **vs Traditional ISLR**: ST-GCN and similar specialized methods perform extremely well under supervision, indicating task-specific training remains irreplaceable
- **vs Elysium/ChatTracker**: These MLLM trackers also require fine-tuning; pure zero-shot approaches are insufficient

## Rating
- Novelty: ⭐⭐⭐ An evaluation study that is not a new method per se but fills a significant gap
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-model, multi-prompt, multi-evaluation paradigm — highly comprehensive
- Writing Quality: ⭐⭐⭐⭐ Clear structure with deep analysis
- Value: ⭐⭐⭐⭐ Provides an important VLM baseline for sign language AI research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering](../../ACL2026/llm_nlp/are_emotion_and_rhetoric_neurons_in_llm_neuron_recognition_and_adaptive_masking_.md)
- [\[AAAI 2026\] Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives](../../AAAI2026/llm_nlp/understanding_syllogistic_reasoning_in_llms_from_formal_and_natural_language_per.md)
- [\[CVPR 2026\] Perception Programs: Unlocking Visual Tool Reasoning in Language Models](perception_programs_visual_tool_reasoning.md)
- [\[ACL 2026\] A Study of LLMs' Preferences for Libraries and Programming Languages](../../ACL2026/llm_nlp/a_study_of_llms39_preferences_for_libraries_and_programming_languages.md)
- [\[ICLR 2026\] Near-Optimal Online Deployment and Routing for Streaming LLMs](../../ICLR2026/llm_nlp/near-optimal_online_deployment_and_routing_for_streaming_llms.md)

</div>

<!-- RELATED:END -->
