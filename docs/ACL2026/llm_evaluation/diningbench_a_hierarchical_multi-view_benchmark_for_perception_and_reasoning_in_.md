---
title: >-
  [Paper Note] DiningBench: A Hierarchical Multi-view Benchmark for Perception and Reasoning in the Dietary Domain
description: >-
  [ACL 2026][LLM Evaluation][Food benchmark] The authors constructed DiningBench, the first hierarchical multi-view food benchmark (3,021 dishes / 15,928 images / Avg. 5.27 views per dish)…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Food benchmark"
  - "Multi-view"
  - "Fine-grained classification"
  - "Nutrition estimation"
  - "VQA"
date: 2026-05-08
content_hash: 5c07275435900bad
---

# DiningBench: A Hierarchical Multi-view Benchmark for Perception and Reasoning in the Dietary Domain

**Conference**: ACL 2026  
**arXiv**: [2604.10425](https://arxiv.org/abs/2604.10425)  
**Code**: https://huggingface.co/datasets/meituan/DiningBench (Available)  
**Area**: Multimodal VLM / Evaluation Benchmark / Food Visual Understanding  
**Keywords**: Food benchmark, Multi-view, Fine-grained classification, Nutrition estimation, VQA

## TL;DR
The authors constructed DiningBench, the first hierarchical multi-view food benchmark (3,021 dishes / 15,928 images / Avg. 5.27 views per dish), covering three cognitive levels: "Fine-grained classification (same-store hard negatives) → Nutrition estimation (4D regression) → VQA (reasoning)". Evaluation of 29 SOTA VLM systems reveals that current models are significantly deficient in fine-grained visual discrimination and numerical nutrition quantification, and CoT unexpectedly impairs pure visual perception.

## Background & Motivation

**Background**: With the rapid progress of VLMs like GPT-4o / Gemini-3 / Qwen-VL in general visual understanding, high expectations have been placed on AI in the food domain (e.g., automated dietary logging, smart kitchen assistants). However, evaluation benchmarks remain stagnant with early datasets like Food-101 / UEC-Food / Recipe1M+ / Nutrition5K.

**Limitations of Prior Work**: The authors summarize four major deficiencies: (1) **Tasks are too simple**—most benchmarks only handle coarse-grained classification without considering nutritional quantification or culinary reasoning; (2) **Single-view limitation**—real users capture dishes from multiple angles, yet datasets rely on single images; (3) **Lack of fine-grained discrimination**—distractors are randomly sampled, allowing models to guess correctly based on semantic priors; (4) **Inaccurate nutrition labels**—Recipe1M+ has poor image quality, and Nutrition5K/FastFood are restricted to standardized cafeteria or fast-food chain scenarios.

**Key Challenge**: Real-world dishes contain hard negative samples that are visually extremely similar within the same store and category (e.g., Smoked Salmon Salad vs. Fresh Salmon Avocado Salad). Furthermore, quantifying nutrition requires inferring volume, portion sizes, and ingredients from images. Both tasks demand fine-grained visual understanding beyond a "bag-of-features" approach, a gap that current benchmarks fail to expose.

**Goal**: To construct a hierarchical food benchmark that simultaneously evaluates (i) fine-grained discrimination, (ii) numerical quantification, and (iii) high-level reasoning, using multi-view imagery for each dish.

**Key Insight**: Leveraging massive real UGC images and merchant metadata from Meituan (China's largest local life service platform), the authors employ SOTA VLMs (Qwen-2.5-VL, Gemini-3-Pro-Preview) for AI-assisted data curation combined with rigorous human review, compressing 20M noisy images into 3,021 high-quality dishes.

**Core Idea**: A four-in-one integration of "same-store fine-grained hard negatives + multi-view + high-fidelity nutrition labels + hierarchical tasks" to expose the current weaknesses of VLMs.

## Method

DiningBench is a benchmark dataset; the "Method" refers to the data construction pipeline, task definitions, and evaluation protocols.

### Overall Architecture

The pipeline consists of two stages:

1.  **Base Data Construction**: Starting from 20M Meituan UGC images, the process involves: ① Image quality assessment (Qwen-2.5-VL-7B discriminator distilled from GPT-4) → 685k images; ② Reference image matching (ensuring user photos match merchant reference images) → 90k dishes; ③ Merchant reference quality verification → 41k dishes; ④ Detailed ingredient list filtering → 15k dishes; ⑤ De-duplication/balancing by cuisine + human quality control → 6,057 dishes.
2.  **Task Generation**: Using Gemini-3-Pro-Preview for hard-negative mining, nutrition reasoning, and VQA generation. Each step includes two rounds of LLM filtering (one to remove "too difficult to judge" samples, another to remove "too simple distractor" samples) followed by human review.

The final dataset comprises 3 subsets: Fine-Grained Classification (2,884) + Nutrition Estimation (1,650) + VQA (804).

### Key Designs

1.  **Same-store Hard-negative Mining (Fine-Grained Classification)**:
    *   **Function**: Constructs 8-choice multiple-choice questions with visually similar dishes to prevent models from guessing based on category-level priors.
    *   **Mechanism**: For each target dish, Gemini-3-Pro-Preview selects 7 visually/semantically similar dishes from the **same merchant's menu category** as distractors (e.g., target: Smoked Salmon Salad; distractors: Salmon Avocado Salad, Tuna Tartare Salad). Samples undergo two rounds of filtering with Gemini-3-Pro-Preview and Gemini-2.5-Pro to ensure uniqueness and appropriate difficulty before human verification.
    *   **Design Motivation**: Traditional distractors are random, letting models succeed by simply distinguishing "salad" from "noodles". Same-store distractors share ingredients/colors/plating, forcing models to focus on actual fine-grained visual cues like cutting style, texture, and ingredient ratios. This setup limits GPT-4o's accuracy to 65.26%, highlighting the bottleneck in fine-grained discriminability.

2.  **High-fidelity Nutrition Labels (Nutrition Estimation)**:
    *   **Function**: Provides a 4D nutrition vector $\mathbf{v} = (\text{Cal}, \text{Carb}, \text{Prot}, \text{Fat}) \in \mathbb{R}^4$ as regression ground truth.
    *   **Mechanism**: A dual-path labeling approach—(a) **Direct Extraction**: Using merchant-provided nutritional info; (b) **LLM-Assisted Estimation**: For dishes with ingredients and portions but no labels, Gemini-3-Pro-Preview estimates values based on "image + ingredients + weight". All estimates are cross-referenced with the USDA FoodData Central database and systematically verified. Prompts use the Atwater system $E \approx 4P + 4C + 9F$ for consistency checks and instruct models to discard fraudulent merchant descriptions (e.g., underreported fats).
    *   **Design Motivation**: Existing datasets like Nutrition5K or FastFood lack diversity. By combining merchant metadata, LLM estimation, USDA cross-referencing, and triple human verification, **Ours** achieves diverse coverage across 1,650 dishes, ranging from light meals to calorie-dense items (mean 670.5 kcal).

3.  **Hierarchical Task Design + LLM-as-a-Judge Evaluation**:
    *   **Function**: Increases cognitive complexity from perception to quantification to reasoning, evaluated via Accuracy / MAPE / LLM-Judge ACC.
    *   **Mechanism**: (i) Fine-Grained Classification uses Acc; (ii) Nutrition Estimation uses MAE / RMSE / **MAPE**, where $\mathrm{MAPE}_k = \frac{1}{N}\sum_i |v_{i,k} - \hat{v}_{i,k}| / v_{i,k}$ is averaged across 4 components; (iii) VQA uses LLM-as-a-Judge for semantic consistency binary judgment between prediction and gold answer. Subsets include Cuisine Technique (532), Dietary Suggestion (219), Multi-Image Analysis (35), and Counterfactual Reasoning (18).
    *   **Design Motivation**: Hierarchy allows separate diagnostic analysis of VLM bottlenecks. For instance, Gemini-3-Pro-Preview achieving 90.42% in VQA but 24.45% MAPE in nutrition suggests it "reasons correctly but quantifies poorly," a nuance missed by coarse evaluations.

### Loss & Training
No models are trained; this is purely an evaluation. Commercial models use official APIs (temperature=0, max_tokens=16,384), and open-source models are deployed via vLLM. Knowledge distillation was used during base data construction: a small batch of GPT-4 labeled data trained the Qwen-2.5-VL-7B quality evaluator.

## Key Experimental Results

### Main Results: Comparison of 29 VLMs across 3 Tasks (Abridged)

| Model | Class. ACC↑ | Nutrition Avg MAPE↓ | VQA ACC↑ |
|------|-------------|---------------------|----------|
| **Gemini-3-Flash-Preview** | **81.83** | 25.21 | 88.56 |
| **Gemini-3-Pro-Preview** | 81.55 | **24.45** | **90.42** |
| Gemini-2.5-Pro | 73.51 | 38.21 | 89.93 |
| GPT-5 | 70.18 | 32.17 | 86.94 |
| Claude-Sonnet-4.5 | 54.40 | 42.62 | 83.58 |
| GPT-4o | 65.26 | 42.43 | 80.60 |
| Qwen-2.5-VL-72B | 65.29 | 40.56 | 76.62 |
| Qwen-3-VL-30B-A3B-Instruct| 65.43 | 37.35 | 80.60 |
| Qwen-3-VL-8B-Instruct | 64.15 | 39.24 | 72.76 |
| InternVL-3.5-38B | 54.20 | 46.13 | 72.51 |
| Gemma-3-12B-it | 48.61 | 43.15 | 61.82 |

Observations: (i) Even the strongest Gemini-3-Pro-Preview reaches only 82% in classification; (ii) Nutrition estimation remains the hardest task, with MAPE at 24.45% for Gemini-3 and 37.35% for Qwen-3-VL-30B; (iii) VQA is relatively easy; (iv) The gap between open and closed-source models is largest in Nutrition, highlighting reliance on training data scale.

### Ablation Study: Impact of Multi-view Count + CoT

| Config | Classification ACC | Nutrition MAPE | Description |
|------|--------------------|-----------------|------|
| 1 view | baseline | baseline | Single image |
| 2 views | Sig. Increase | Sig. Decrease | Largest "capability jump" |
| 3 views | Slight Increase | Marginal Gain | Benefit for large models |
| 4 views | Near Saturation | MAPE Worsened (Small) | Information overload/noise |
| Large + CoT (Nutrition) | – | **Sig. Worsening** | "Performance collapse" in small models |
| Large + CoT (VQA) | – | – | Inconsistent results |
| Large + CoT (Class.) | Mostly Decrease| – | Explicit reasoning interferes with perception |

### Key Findings
- **"Capability jump" occurs at 1→2 views**: Moving from single to dual views provides the largest performance boost (resolving occlusion/ambiguity). Benefits diminish after 3 views; 4 views can even hurt 7B-scale models, suggesting a lack of effective multi-view fusion.
- **CoT is not a silver bullet**: In pure perception tasks (Nutrition), CoT leads to MAPE surges and "performance collapse" in small models. Authors hypothesize that explicit verbalization decouples the prediction from direct visual evidence, leading to hallucinations.
- **Five Failure Modes**: (1) Insufficient fine-grained discrimination (reliance on "bag-of-features"), (2) Parametric knowledge bias (outputting common dish names instead of specific variants), (3) Lack of spatial volume reasoning (2D→3D weakness), (4) Ineffective multi-view aggregation, (5) Thinking loops in reasoning models.
- **Translation Bias**: Accuracy generally drops after Chinese-to-English translation in Classification, but Nutrition Estimation improves for Gemini/GPT models, suggesting English prompts trigger stronger numerical reasoning pathways.

## Highlights & Insights
- **"Same-store hard negatives"** is a clever update to the fine-grained classification problem. By utilizing menu structures to provide "visually similar + semantically different" pairs, it creates a low-cost, high-efficiency data augmentation strategy applicable to any vertical domain.
- **AI-assisted curation + dual-LLM filtering + human audit** forms a replicable "100% pass rate" pipeline for high-quality benchmark construction.
- **"Hierarchical tasks + LLM-as-a-Judge"** provides a diagnostic view of model bottlenecks (perception vs. quantification vs. reasoning) that is far more informative than a single Accuracy metric.
- **Failure modes** (especially 2D→3D volume reasoning) provide direct insights for future VLM training, suggesting a need for 3D-aware pretraining or explicit depth supervision.

## Limitations & Future Work
- **Chinese Cuisine Bias**: 69% of the data is Chinese cuisine, which may limit global generalization.
- **LLM-assisted Label Bias**: Despite human review, generated nutrition estimates and distractor choices may inherit biases from the source LLMs.
- **Small VQA Subset**: 804 samples may lack statistical significance for LLM-as-a-Judge, especially in Multi-Image and Counterfactual categories.
- **Missing Baseline Comparisons**: No comparison against specialized Nutrition5K-tuned models or fine-grained ConvNet baselines.
- **Future Directions**: Expanding non-Chinese datasets, adding 3D reconstruction tasks, and exploring joint "visual grounding loss + CoT" training.

## Related Work & Insights
- **vs. Food-101 / Food2K**: **Ours** adds nutrition quantification and reasoning with multi-view support.
- **vs. Nutrition5K / FastFood**: **Ours** uses real-world diverse restaurant UGC instead of cafeteria or fast-food constraints, using MAPE for clearer results than simple MAE.
- **vs. FoodieQA**: **Ours** separates recognition from reasoning via hierarchical tasks for better diagnostic evaluation.
- **vs. MMBench / MME**: General benchmarks struggle with vertical fine-grained issues; **Ours** serves as a template for specialized vertical benchmarks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Hierarchical tasks + same-store negatives is a strong combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 29 models across 3 tasks with extensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and informative tables.
- **Value**: ⭐⭐⭐⭐⭐ Publicly available on HuggingFace; directly advances food-domain VLM research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](recoqa_a_benchmark_for_tool-augmented_and_multi-step_reasoning_in_real_estate_qu.md)
- [\[ACL 2026\] K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology](k-metbench_a_multi-dimensional_benchmark_for_fine-grained_evaluation_of_expert_r.md)
- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[ICML 2026\] Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](../../ICML2026/llm_evaluation/multi2_hierarchical_multi-agent_decision-making_with_llm-based_agents_in_interac.md)
- [\[ACL 2026\] SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)

</div>

<!-- RELATED:END -->
