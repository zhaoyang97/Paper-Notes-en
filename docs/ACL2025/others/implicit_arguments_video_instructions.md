---
title: >-
  [Paper Note] Predicting Implicit Arguments in Procedural Video Instructions
description: >-
  [ACL 2025][Semantic Role Labeling] The Implicit-VidSRL dataset and the iSRL-Qwen2-VL model are proposed to address the prediction of omitted implicit arguments (ingredients) in procedural video instructions. By decomposing multi-step instructions into {verb, what, where/with} triplets using a semantic role labeling (SRL) framework, the model, after being fine-tuned on silver-standard data, outperforms GPT-4o by 17% in implicit argument F1.
tags:
  - "ACL 2025"
  - "Semantic Role Labeling"
  - "Implicit Argument Prediction"
  - "Procedural Video Understanding"
  - "Multimodal LLMs"
  - "Cooking Recipes"
date: 2026-05-08
content_hash: 98e2da680eadda0f
---

# Predicting Implicit Arguments in Procedural Video Instructions

**Conference**: ACL 2025  
**arXiv**: [2505.21068](https://arxiv.org/abs/2505.21068)  
**Code**: [GitHub](https://github.com/anilbatra2185/implicit-vid-srl)  
**Area**: Other  
**Keywords**: Semantic Role Labeling, Implicit Argument Prediction, Procedural Video Understanding, Multimodal LLMs, Cooking Recipes  

## TL;DR

The Implicit-VidSRL dataset and the iSRL-Qwen2-VL model are proposed to address the prediction of omitted implicit arguments (ingredients) in procedural video instructions. By decomposing multi-step instructions into {verb, what, where/with} triplets using a semantic role labeling (SRL) framework, the model, after being fine-tuned on silver-standard data, outperforms GPT-4o by 17% in implicit argument F1.

## Background & Motivation

**Pervasiveness of ellipsis**: Procedural texts (such as cooking recipes) are highly elliptical, where a large number of arguments in subsequent steps need to be inferred from the prior text or visual context. For example, "add seasoning" implicitly refers to the specific combination of ingredients processed in previous steps.

**Insufficiency of existing SRL benchmarks**: Traditional SRL datasets (PropBank, FrameNet) focus on explicit arguments within single sentences and rarely annotate cross-sentence implicit information; video SRL datasets like VidSitu only focus on local or short-term contextual implicit information.

**Need for multimodal reasoning**: In cooking scenarios, ingredients undergo visual changes (cutting, mixing, heating). Linguistic context alone is insufficient for disambiguation, requiring the integration of video frames for cross-timestep entity tracking.

**Poor performance of existing multimodal LLMs**: Models like GPT-4o show limited ability in tracking implicit entities in long contexts, particularly struggling with reasoning about ingredient state changes and mixture compositions.

**Evaluation gap**: Existing next-step prediction tasks only use NLG metrics (BLEU/METEOR), which cannot evaluate the models' understanding of implicit information, leaving a lack of fine-grained evaluation based on semantic frames.

**Application value**: Accurately predicting implicit arguments has direct application value for personalized cooking guidance (e.g., allergen tracking) and entity tracking in human-robot collaboration.

## Method

### Overall Architecture

Multimodal procedural video understanding is modeled as a two-stage pipeline: (1) constructing the Implicit-VidSRL dataset to annotate the {verb, what, where/with} semantic frame (including implicit arguments) for each step of cooking videos; (2) using GPT-4o to automatically generate silver-standard training data, fine-tuning Qwen2-VL on this data to obtain iSRL-Qwen2-VL, which simultaneously predicts both semantic frames and natural language instructions.

### Module 1: Implicit-VidSRL Dataset Construction

- **Data Source**: Validation/test sets from the YouCook2 and Tasty datasets, comprising 231 cooking videos and 2545 semantic frames.
- **Annotation Scheme**: Multi-action instructions are decomposed into single predicate-argument structures {verb, what, where/with}, where "what" represents the object of the action, and "where/with" represents the location or accompanying elements. The focus is on annotating omitted implicit ingredient arguments.
- **Three-Stage Annotation**: Stage 1 involves manual identification of implicit entities by linguistics PhD students; Stage 2 automatically converts these into SRL labels using GPT-4o-Mini + CoT + 5-shot examples; Stage 3 consists of manual verification to ensure implicit information is accurate and kitchen utensils are excluded.
- **Statistical Characteristics**: On average, each "what" role contains 6.29 implicit entities, and "where/with" contains 5.21; "where/with" is empty in 54% of cases.

### Module 2: Task Definition

- **Implicit Argument Prediction (Cloze Task)**: Given an input sequence (text/video) and a masked semantic frame (where "verb" is known, and "what" and "where/with" are masked), the model needs to predict the complete set of arguments containing both explicit and implicit entities.
- **Next Step Prediction**: Given the instructions and their SRL labels for the prior $t$ steps, the model predicts the natural language instruction and the corresponding semantic frame (including implicit arguments) for the $(t+1)$-th step, generating $k$ candidates.

### Module 3: iSRL-Qwen2-VL Model

- **Silver-Standard Data Generation**: Automatically generate SRL annotations on the Tasty training set using GPT-4o + CoT prompting, which includes (1) splitting multi-step instructions into single-predicate structures, and (2) automatically inferring implicit entities. This yields ~2.5K training video samples, formatted into ~18K next-step prediction training samples.
- **Training Setup**: LoRA fine-tuning is performed on Qwen2-7B-Instruct and Qwen2-VL-7B-Instruct (using 4×A100-80GB, $\le 48$ GPU hours) to simultaneously predict the next-step text and the SRL frame.

### Training & Inference

- Fine-tuning is conducted using the default LoRA configuration from the LLama-factory framework.
- Evaluation utilizes set-based F1 (exact match + IoU word overlap), with F1 calculated separately for implicit arguments. Next-step prediction additionally uses verb recall@5, BLEU4, and METEOR.
- Video inputs are restricted to a maximum of 320 frames per video.

## Experiments

### Table 1: Implicit Argument Prediction (Cloze Task)

| Model | Params | Input | FT | F1_what | F1_where | F1_what(Implicit) | F1_where(Implicit) |
|------|--------|------|------|---------|----------|---------------|----------------|
| GPT-4o | - | V+T | ✗ | 64.83 | 55.32 | 50.53 | 49.01 |
| Qwen2-VL 7B | 7B | V+T | ✗ | 42.07 | 22.54 | 22.68 | 21.96 |
| **iSRL-Qwen2-VL 7B** | 7B | V+T | ✓ | **64.86** | **54.54** | **59.15** | **56.21** |
| LLama-3.1 | 70B | T | ✗ | 63.04 | 55.50 | 50.46 | 53.42 |
| iSRL-Qwen2 7B | 7B | T | ✓ | 57.82 | 49.33 | 51.70 | 47.74 |

### Table 2: Next Step Prediction

| Model | Params | Input | FT | R_verb@5 | F1_what | F1_where | METEOR |
|------|--------|------|------|----------|---------|----------|--------|
| GPT-4o | - | V+T | ✗ | 53.36 | 20.51 | 16.32 | 18.99 |
| iSRL-Qwen2-VL 7B | 7B | V+T | ✓ | 47.76 | 19.74 | 17.44 | 19.38 |
| iSRL-Qwen2 7B | 7B | T | ✓ | 50.01 | 20.29 | 15.99 | 20.54 |
| Qwen2 72B | 72B | T | ✗ | 47.84 | 17.59 | 13.44 | 20.22 |

### Key Findings

1. **iSRL-Qwen2-VL (7B) outperforms GPT-4o under multimodal input**: The implicit argument F1_what reaches 59.15 (compared to 50.53 for GPT-4o), a relative Gain of 17%; F1_where reaches 56.21 (compared to 49.01 for GPT-4o), a relative Gain of 14.7%.
2. **Video-only input is significantly weaker than text-only input**: All models experience a substantial drop in performance in the video-only setting, indicating that directly identifying and tracking ingredient entities from video remains challenging.
3. **Multimodal fusion outperforms unimodal**: V+T input consistently outperforms V or T alone, as visual information helps disambiguate local entities.
4. **CoT prompting drastically improves the "where/with" role**: Qwen2-VL's F1_where increases from 8.60 to 15.15; without CoT, the model tends to predict kitchen utensils rather than ingredients.
5. **SRL as an intermediate representation improves next-step prediction**: Fine-tuned models that include SRL prediction improve by about 2% on METEOR, proving that semantic frames aid in structured reasoning.
6. **Longer sequences are more challenging**: As the position of the semantic frame increases, the number of implicit entities grows, and model performance degrades. However, iSRL-Qwen2-VL shows better robustness in later positions than GPT-4o.

## Highlights & Insights

- **Precise problem definition**: Formalizes the widespread phenomenon of ellipsis in procedural text as an SRL implicit argument prediction task, which possesses both linguistic foundations and practical application motivations.
- **Pragmatic annotation scheme**: The three-stage process (manual identification $\rightarrow$ GPT automatic conversion $\rightarrow$ manual verification) balances annotation quality and efficiency. The silver-standard data strategy cleverly leverages GPT-4o's capability to lower training costs.
- **Small model outperforms large models**: The 7B fine-tuned model surpasses GPT-4o and the 72B open-source model on core metrics, demonstrating that task-specific data and training strategies are more critical than model scale.
- **Comprehensive evaluation system**: Introduces set-based F1 for SRL evaluation, closing the gap where NLG metrics fail to measure the understanding of implicit information.

## Limitations & Future Work

- **Domain limitation**: Only validated on cooking recipes. It remains unclear whether the simple decomposition of {verb, what, where/with} can generalize to other procedural domains (such as assembly or laboratory operations).
- **Silver-standard data quality relies on GPT-4o**: Automatic annotation may introduce bias, and the paper does not report a quantitative alignment between silver-standard and gold-standard data.
- **Weak visual capability**: Video-only performance is far lower than text-only, indicating that the model primarily benefits from textual context, leaving substantial room for improvement in visual understanding.
- **Focus only on ingredient arguments**: Other potentially implicit argument types (e.g., tools, temperature, time) are not considered, which limits the completeness of the task.
- **Limited dataset scale**: The test set size of 231 videos and 2545 SRL frames is small, which might limit statistical significance.

## Related Work & Insights

- **Semantic Role Labeling**: Traditional SRL such as PropBank and FrameNet focuses on explicit arguments in single sentences; Gerber & Chai (2010) extended NomBank to cross-sentence implicit arguments but limited it to 10 nominal predicates; VidSitu (Sadhu et al., 2021) performs video SRL but only covers short-term local contexts.
- **Procedural Understanding**: PizzaCommonSense (Diallo et al., 2024) annotates intermediate step outputs but only focuses on explicit entities and is restricted to pizza; GEPSAN (Abdelsalam et al., 2023) does next-step prediction but lacks implicit argument evaluation.
- **Temporal Reasoning Benchmarks**: SEED-Bench and TempCompass focus on action sequences and short video segment attributes, while SOKBench constructs knowledge graphs but contains incomplete instructions.
- **Multimodal LLMs**: GPT-4o, Qwen2-VL, and LLava-OneVision provide strong baselines, but this paper finds that all of them perform poorly in tracking implicit entities across long contexts.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introduces implicit arguments from traditional linguistics to multimodal procedural videos; the problem definition is novel and deep.
- **Effectiveness**: ⭐⭐⭐⭐ — The 7B model outperforms GPT-4o and the 72B model, and the ablation experiments thoroughly validate each design choice.
- **Practicability**: ⭐⭐⭐ — Dataset and code are open-sourced, but the domain is limited to cooking, and practical downstream application scenarios need further expansion.
- **Recommendation**: ⭐⭐⭐⭐ — Valuable problem, simple yet effective method, and comprehensive experiments; an excellent work in multimodal procedural understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Limited Generalizability in Argument Mining: State-Of-The-Art Models Learn Datasets, Not Arguments](limited_generalizability_in_argument_mining_state-of-the-art_models_learn_datase.md)
- [\[ACL 2025\] From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding](from_real_to_synthetic_synthesizing_millions_of_diversified_and_complicated_user.md)
- [\[ACL 2025\] Implicit Reasoning in Transformers is Reasoning through Shortcuts](implicit_reasoning_in_transformers_is_reasoning_through_shortcuts.md)
- [\[CVPR 2025\] FIction: 4D Future Interaction Prediction from Video](../../CVPR2025/others/fiction_4d_future_interaction_prediction_from_video.md)
- [\[CVPR 2025\] EVOS: Efficient Implicit Neural Training via EVOlutionary Selector](../../CVPR2025/others/evos_efficient_implicit_neural_training_via_evolutionary_selector.md)

</div>

<!-- RELATED:END -->
