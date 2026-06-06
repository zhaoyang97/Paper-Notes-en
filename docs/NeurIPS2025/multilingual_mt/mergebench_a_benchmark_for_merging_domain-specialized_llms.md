---
title: >-
  [Paper Note] MergeBench: A Benchmark for Merging Domain-Specialized LLMs
description: >-
  [NeurIPS 2025][Multilingual & Machine Translation][model merging] MergeBench is the first comprehensive benchmark suite for evaluating large-scale domain-specialized LLM merging…
tags:
  - "NeurIPS 2025"
  - "Multilingual & Machine Translation"
  - "model merging"
  - "benchmark"
  - "task vectors"
  - "LLM"
  - "multi-task learning"
date: 2026-05-08
content_hash: 192b789ca67e5267
---

# MergeBench: A Benchmark for Merging Domain-Specialized LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2505.10833](https://arxiv.org/abs/2505.10833)  
**Code**: [yifei-he/mergebench](https://yifei-he.github.io/mergebench/)  
**Area**: Multilingual Translation
**Keywords**: model merging, benchmark, task vectors, LLM, multi-task learning

## TL;DR

MergeBench is the first comprehensive benchmark suite for evaluating large-scale domain-specialized LLM merging, covering Llama and Gemma families up to 9B parameters, five task domains, and eight merging methods, providing systematic evaluation and practical guidelines across three dimensions: multi-task performance, forgetting, and runtime efficiency.

## Background & Motivation

Model merging combines the capabilities of multiple specialized models through arithmetic operations on model parameters, enabling multi-task ability without joint training or access to all task data. However, existing evaluations have two critical limitations:

**Limited model scale**: Most evaluations use small models such as GPT-2 (124M), RoBERTa-base (125M), and mT5 (2.85B); whether observed trends generalize to large-scale LLMs remains unclear.

**Insufficient task diversity**: Evaluations typically focus on shallow NLP tasks such as sentiment classification and natural language inference, failing to expose the generalization, composition, and interference challenges that arise when merging stronger, more specialized models.

Comparison of existing evaluation frameworks:

| Evaluation | Multi-Model | Large (>7B) | Domain Tasks | Gradient Methods | Open Source |
|---|---|---|---|---|---|
| Ilharco et al. | ✗ | ✗ | ✗ | ✓ | ✓ |
| FusionBench | ✗ | ✗ | ✗ | ✓ | ✓ |
| Yadav et al. | ✗ | ✓ | ✗ | ✗ | ✗ |
| Model-GLUE | ✗ | ✓ | ✓ | ✗ | ✓ |
| **MergeBench** | **✓** | **✓** | **✓** | **✓** | **✓** |

## Method

### Overall Architecture

MergeBench is designed along three key dimensions:

1. **Task coverage**: Five domains — instruction following, mathematics, multilingual understanding, code, and safety.
2. **Model selection**: Eight base models drawn from Llama-3.2-3B, Llama-3.1-8B, Gemma-2-2B, Gemma-2-9B, and their instruction-tuned variants.
3. **Standardized training and evaluation**: Unified fine-tuning pipelines and evaluation protocols to ensure fairness and reproducibility.

Starting from each base model, specialized models are obtained by fine-tuning on each task (40 open-source models in total), which are then merged using 8 merging methods to produce multi-task models.

### Key Designs

1. **Unified evaluation framework for eight model merging methods**:
    - *Function*: Systematically covers eight mainstream merging methods from two major categories to establish a standardized comparison benchmark.
    - *Mechanism*: **Coefficient-tuning methods** include simple averaging (Model Soup), task-vector-weighted merging $\theta_{merged} = \theta_{pre} + \lambda \sum \tau_i$ (Task Arithmetic), Fisher information matrix weighting (Fisher Merging), and activation-discrepancy minimization (RegMean). **Sparsification methods** include sign-consistency pruning (TIES), random-dropout rescaling $\theta_{merged} = \sum \lambda(1-m_i) \odot \tau_i / (1-p)$ (DARE), consensus mask filtering (Consensus TA), and task-relevant parameter localization with stitching (Localize-and-Stitch). All eight methods address task-vector conflicts from different perspectives.
    - *Design Motivation*: Prior work lacks fair comparisons of multiple merging methods under unified conditions; covering both technical paradigms comprehensively reveals the strengths, weaknesses, and applicable scenarios of each approach.

2. **Standardized training pipeline for five-domain specialized models**:
    - *Function*: Produces controlled, domain-specialized fine-tuned models for instruction following, mathematics, multilingual understanding, code, and safety.
    - *Mechanism*: Eight base models (Llama-3.2-3B/3.1-8B and Gemma-2-2B/9B) are each fine-tuned on five domains (40 models total). Training data includes TULU-3, DART-Math, Aya (65 languages), Magicoder, and WildGuardMix, primarily via SFT; GRPO reinforcement learning is additionally applied to 8B/9B models for the mathematics domain.
    - *Design Motivation*: Unified fine-tuning pipelines and data scales eliminate confounding variables introduced by training differences, ensuring fair and reproducible comparisons across merging methods; the diversity of five domains enables assessment of merging methods' generalization across capability dimensions.

### Loss & Training

- Specialized models are trained with supervised fine-tuning (SFT); GRPO reinforcement learning is additionally applied to 8B/9B models for the mathematics domain.
- Methods requiring auxiliary data (Fisher Merging, RegMean, Localize-and-Stitch) uniformly sample 1,000 examples from the training set.
- Methods requiring hyperparameter tuning employ grid search on proxy validation tasks.

## Key Experimental Results

### Main Results

Normalized multi-task performance (relative to specialized models; 1.0 indicates full recovery of fine-tuning performance):

**2B/3B pre-trained models:**
- The best method (Localize-and-Stitch) recovers approximately 80% of fine-tuning performance.
- Stronger base models yield better merging outcomes.

**8B/9B pre-trained models:**
- Merging methods consistently recover 90%+ of fine-tuning performance.
- All methods exceed 90% on instruction-tuned models.

**Method ranking:**
1. Localize-and-Stitch (both variants) consistently achieves the best performance.
2. RegMean is competitive on small models but its advantage diminishes on larger models.
3. Task Arithmetic, Consensus TA, and TIES occupy the middle tier.
4. DARE ranks lower on large models.
5. Fisher Merging performs worst overall.

### Ablation Study

**Forgetting analysis** (evaluated on MMLU, TriviaQA, SQuADv2, CoQA, PubMedQA, WMT14):

- Multi-task learning (MTL) models perform well in-domain but exhibit significant degradation in out-of-domain generalization.
- Merged models better preserve base model knowledge due to:
    - Smaller scaling coefficients keeping merged models closer to the base model.
    - Sparsification constraints limiting parameter updates to a small subset.
- The sparsification strategies of TIES and Localize-and-Stitch are particularly effective at reducing forgetting.
- DARE's random dropout mechanism is less effective at preserving knowledge.

**Runtime efficiency analysis** (wall-clock time on Llama-3.2-3B):

| Method | Efficiency Characteristics |
|---|---|
| Model Soup | Most efficient; no additional training or tuning required |
| Localize-and-Stitch | Short total time (no hyperparameter tuning required) |
| Task Arithmetic | Moderate |
| TIES / DARE | Slowest (require tuning two hyperparameters: sparsity and scaling) |

### Key Findings

1. **Stronger base models → better merging outcomes**: Larger models have greater capacity, resulting in less task interference; instruction tuning brings specialized models closer in parameter space.
2. **Sparsification and coefficient tuning are key to reducing forgetting**: Both forms of regularization effectively control the degree to which merged models deviate from the base model.
3. **Multi-task training still has advantages**: When tasks do not conflict and data is balanced, MTL yields stronger in-domain performance.
4. **Validation time is non-negligible**: The hyperparameter tuning time for TIES and DARE far exceeds the runtime of the merging algorithm itself.

## Highlights & Insights

1. **Unmatched comprehensiveness**: The first merging benchmark to simultaneously satisfy all five criteria — model diversity, large scale, domain tasks, gradient method support, and full open-source availability.
2. **40 open-source specialized models**: Provides the community with high-value, reusable resources.
3. **Three-dimensional evaluation framework**: Evaluates not only multi-task performance but also forgetting and efficiency, offering a practical decision-making guide.
4. **Clear practical recommendations**: A progressive recommendation pathway from no-data to data-available settings (Model Soup → Dataless L&S / Task Arithmetic → L&S / RegMean).

## Limitations & Future Work

1. **Model scale capped at 9B**: Models at 70B+ scale are not covered, and merging behavior may change qualitatively at that scale.
2. **Inter-task conflict not quantified**: The degree of interference among the five domains is not systematically analyzed; different domain combinations may have different optimal strategies.
3. **Role of merging in the LLM pipeline is unclear**: The relationship between model merging and continual learning, data mixing, and other strategies requires deeper investigation.
4. **Merging computational overhead remains non-trivial**: Peak memory consumption for Fisher Merging and Localize-and-Stitch in particular approaches that of full fine-tuning.
5. **Lack of comparison with newer merging methods**: Recent approaches such as WARP and evolutionary merging are not included.

## Related Work & Insights

- **Task Arithmetic (Ilharco et al., 2023)**: Establishes the foundational concept of task vectors.
- **TIES (Yadav et al., 2023)**: Reduces interference via pruning and sign selection.
- **Localize-and-Stitch (He et al., 2025)**: Achieves precise merging through localization and stitching.
- **Model-GLUE**: Similar in spirit but limited to the Llama-2 family and does not support gradient-based methods.
- **Insights**: The MergeBench framework is extensible to new directions such as multimodal model merging and cross-generation merging (e.g., integrating knowledge from older model versions into newer ones); the advantages of merging methods in low-resource or data-imbalanced settings (e.g., safety alignment, multilingual tasks) warrant deeper exploration.

## Rating

- Novelty: ⭐⭐⭐ The benchmark design itself is not an entirely new concept, but its comprehensiveness and systematicity significantly surpass prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 base models × 5 tasks × 8 methods × 3 evaluation dimensions; experiments are extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich figures and tables, and well-summarized practical guidelines.
- Value: ⭐⭐⭐⭐ Provides a much-needed standardized evaluation platform for model merging research; the 40 open-source models offer high community value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](../../ACL2026/multilingual_mt/alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)
- [\[AAAI 2026\] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](../../AAAI2026/multilingual_mt/consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)
- [\[NeurIPS 2025\] ParallelPrompt: Extracting Parallelism from Large Language Model Queries](parallelprompt_extracting_parallelism_from_large_language_model_queries.md)
- [\[ACL 2026\] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](../../ACL2026/multilingual_mt/morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)
- [\[ACL 2026\] IndoTabVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents](../../ACL2026/multilingual_mt/indotabvqa_a_benchmark_for_cross-lingual_table_understanding_in_bahasa_indonesia.md)

</div>

<!-- RELATED:END -->
