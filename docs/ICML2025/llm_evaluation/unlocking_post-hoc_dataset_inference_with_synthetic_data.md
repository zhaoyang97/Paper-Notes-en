---
title: >-
  [Paper Note] Unlocking Post-hoc Dataset Inference with Synthetic Data
description: >-
  [ICML2025][LLM Evaluation][dataset inference] This paper proposes utilizing synthetically generated held-out datasets combined with post-hoc calibration to achieve dataset inference without the need for real held-out sets. It generates high-quality synthetic data via suffix completion and decouples generative shift from membership signals using dual-classifier calibration, achieving high-confidence copyright detection with low false positive rates across 15 diverse text datas…
tags:
  - "ICML2025"
  - "LLM Evaluation"
  - "dataset inference"
  - "membership inference"
  - "synthetic data"
  - "copyright protection"
  - "LLM"
  - "post-hoc calibration"
  - "data ownership"
date: 2026-05-08
content_hash: 147934a200d6299a
---

# Unlocking Post-hoc Dataset Inference with Synthetic Data

**Conference**: ICML2025  
**arXiv**: [2506.15271](https://arxiv.org/abs/2506.15271)  
**Code**: [GitHub](https://github.com/sprintml/PostHocDatasetInference)  
**Area**: LLM Evaluation  
**Keywords**: dataset inference, membership inference, synthetic data, copyright protection, LLM, post-hoc calibration, data ownership

## TL;DR

This paper proposes utilizing synthetically generated held-out datasets combined with post-hoc calibration to achieve dataset inference without the need for real held-out sets. It generates high-quality synthetic data via suffix completion and decouples generative shift from membership signals using dual-classifier calibration, achieving high-confidence copyright detection with low false positive rates across 15 diverse text datasets.

## Background & Motivation

### Problem Scenario

Large language models (LLMs) are often trained on massive amounts of data scraped from the internet, which potentially violates the intellectual property of data owners. **Dataset Inference (DI)** aims to determine whether a suspect dataset has been used to train a model, enabling data owners to verify unauthorized usage.

### Key Bottlenecks of Existing DI

DI requires a **held-out set**—a dataset known not to be part of the training set, which must share the **same distribution** as the suspect dataset. However, in practice:

1. Data creators typically do not reserve held-out sets for legal purposes.
2. Any publicly available held-out data could be used in subsequent training.
3. Even different articles by the same author exhibit **subtle distribution shifts**.

### Key Findings: False Positive Issues in DI

The paper first experimentally reveals a severe issue: even under the simplest setting (blog posts from a single author, randomly split into training/held-out), DI still produces **false positives**.

| Number of Sequences per Blog | 5 | 10 | 15 | 20 | 25 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT2 AUC (%) | 52.0 | 55.2 | 53.2 | 58.2 | 58.6 |
| DI p-value | 0.002 | <0.001 | <0.001 | <0.001 | <0.001 |
| Ground Truth Membership | ✕ | ✕ | ✕ | ✕ | ✕ |
| Inference Result | ✓ | ✓ | ✓ | ✓ | ✓ |

Non-member sets are incorrectly classified as members! This is because different blog posts introduce distribution shifts due to semantic/topical variations, and these shifts are **entangled** with the membership signals in MIA (Membership Inference Attack).

## Method

### Overall Process

The proposed method consists of two major steps: (1) Synthetically generating high-quality held-out data; (2) Post-hoc calibration to decouple distribution shift and membership signals.

### Step 1: Held-out Data Generation

The paper systematically compares four generation schemes:

| Method | BoW AUC (%) | GPT2 AUC (%) | Evaluation |
|------|:-----------:|:------------:|------|
| ICL Paraphrasing (GPT-4 Paraphrasing) | 76.2 | 99.0 | Extremely large distribution shift, obvious GPT-4 generation patterns |
| ICL Text Completion (GPT-4 Text Completion) | 79.2 | 99.2 | Same as above |
| Preference Optimization | 50.2 | 58.9 | Near-random at the BoW level, but still distinguishable by transformers |
| **Suffix Completion** | **50.0** | **52.2** | **Minimal distribution shift** |

**Suffix Completion Workflow**:

1. Segment documents into short sequences ($\le 64$ tokens), shuffle, and split into generator training and inference sets.
2. Fine-tune Llama 3 8B with LoRA as the generator, with the training objective set to next-token prediction.
3. During inference: truncate each sequence into prefix + suffix; the generator produces synthetic suffixes based on the prefix.
4. Original suffix $\rightarrow$ suspect set, synthetic suffix $\rightarrow$ held-out set.

Key Designs:
- **Segment and Shuffle**: Splitting at the sequence level rather than the document level to ensure the training and inference sets are identically distributed.
- **Shared Prefix**: The original and synthetic suffixes share the same prefix, ensuring alignment in position and context.
- **Short Sequence Constraint** ($\le 64$ tokens): The generator produces higher-quality completions under short sequences.

### Step 2: Post-hoc Calibration

Even though the distribution shift generated by suffix completion is minimal (GPT2 AUC is only 52.2%), directly applying DI still leads to false positives. Key Insight: **Generative shifts occur in the text space, whereas membership signals reside in the output space of the target LLM; thus, the two can be decoupled**.

**Dual-Classifier Design**:

1. **Text Classifier** $c_{\text{text}}(x)$: Distinguishes original/synthetic data based solely on textual features, quantifying the generative shift.
2. **Combined Classifier** $c_{\text{comb}}(x, \text{MIA}(f(x)))$: Utilizes both textual features and MIA signals, quantifying generative shift + membership signals.

**Difference Comparison (t-test)**:

If membership signals are present, the combined classifier should outperform the text classifier (as the extra MIA inputs provide information):

$$c_{\text{comb}}(x_{\text{val}}) - c_{\text{comb}}(x_{\text{sus}}) > c_{\text{text}}(x_{\text{val}}) - c_{\text{text}}(x_{\text{sus}})$$

Null Hypothesis $\mathcal{H}_0$:

$$\mathbb{E}[c_{\text{comb}}(x_{\text{val}}) - c_{\text{comb}}(x_{\text{sus}})] \leq \mathbb{E}[c_{\text{text}}(x_{\text{val}}) - c_{\text{text}}(x_{\text{sus}})]$$

If $\mathcal{H}_0$ is rejected ($p < 0.05$), the suspect set is determined to be a member of the training data.

### Step 3: Weight Constraints

The original DI uses $y_{\text{diff}} = \sum_i w_i (\mathbb{E}[\text{MIA}_i(x_{\text{val}})] - \mathbb{E}[\text{MIA}_i(x_{\text{sus}})])$ to aggregate multiple MIA scores.

Problem: Synthetic texts are typically simpler than human-written text, leading to lower perplexity. Linear regression might assign negative weights $w_i < 0$ to such MIAs, causing $y_{\text{diff}} > 0$ and resulting in false positives.

Solution: Leverage the Sigmoid function to constrain weight values to $(0, 1)$: $w_i = \sigma(w_i')$, thereby preventing negative weights from amplifying generative shifts.

## Key Experimental Results

### Single-Author Blog Dataset

| Setting | Ground Truth Member | AUC_Text (%) | AUC_Comb (%) | p-value | Inference Result |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Member Set | ✓ | 53.8 | 55.6 | 0.01 | ✓ |
| Non-member Set | ✕ | 53.8 | 53.9 | 0.13 | ✕ |

- Member set: The AUC of the combined classifier is 1.8% higher than the text classifier, $p=0.01$, showing correct detection.
- Non-member set: The two AUCs are close, $p=0.13$, correctly identified as non-member (**eliminating false positives**).

### Pile Dataset (15 Subsets, Pythia 1B)

| Subset | Member p-value | Non-member p-value | Member Detection | Non-member Detection |
|------|:-----------:|:-------------:|:-------:|:--------:|
| Pile-CC | 0.002 | 0.99 | ✓ | ✓ |
| Wikipedia | 0.04 | 1.00 | ✓ | ✓ |
| ArXiv | <0.001 | 0.74 | ✓ | ✓ |
| FreeLaw | <0.001 | 0.09 | ✓ | ✓ |
| PubMed Central | <0.001 | 0.11 | ✓ | ✓ |
| Github | 0.003 | 0.07 | ✓ | ✓ |
| EuroParl | <0.001 | 0.07 | ✓ | ✓ |
| StackExchange | <0.001 | 0.06 | ✓ | ✓ |

**All 15 subsets achieved member set $p\text{-value} < 0.05$ and non-member set $p\text{-value} > 0.05$**, realizing a 100% correct detection rate and a 0% false positive rate.

### Ablation Study

| Setting | Member p-value | Non-member p-value | Evaluation |
|------|:-----------:|:-------------:|------|
| Replacing with ICL Paraphrasing | 1.0 / 1.0 | False negative (excessive generative shift) |
| Replacing with ICL Text Completion | 1.0 / 1.0 | False negative |
| Replacing with Preference Optimization | 1.0 / 1.0 | False negative |
| Removing Segment & Shuffle | 1.0 / 1.0 | False negative |
| Removing Suffix Comparison | 1.0 / 1.0 | False negative |
| Removing Post-hoc Calibration | <0.001 / <0.001 | **False positive** (Critical!) |
| Removing Weight Constraint | 0.004 / 0.43 | Still functional but marginal |
| **Full Method** | **<0.001 / 1.0** | **Optimal** |

### Sample Size Analysis

- Fewer than 1,000 samples are sufficient to achieve $p < 0.05$ on most datasets.
- With 2,000 samples, all datasets achieve $p < 0.01$.

## Highlights & Insights

1. **First to replace real held-out sets with synthetic data**: Resolves the most critical practical bottleneck of DI—the fact that data owners typically cannot provide in-distribution held-out data.
2. **Delicately designed Suffix Completion**: Leverages a three-fold design of shared prefixes, short sequences, and segment-level shuffling, minimizing the distribution shift down to a GPT2 AUC of only 52.2% (close to random guessing).
3. **Ingenious dual-classifier calibration approach**: Decouples generative shifts (which occur in text space) and membership signals (which reside in the LLM output space) by leveraging the 'additional information gain' in an information-theoretic sense.
4. **Strong cross-domain generalization**: Validated and shown to be effective across 15 diverse domains, including medicine (PubMed), law (FreeLaw), code (Github), and multilingual (EuroParl).
5. **High practicality**: Requires only querying the target model (black-box setting), with no dependency on model weights or training details.

## Limitations & Future Work

1. **Reliance on accessing target model logits**: Although it operates in a black-box setting, it requires obtaining token-level probabilities (for computing MIA metrics such as perplexity), which is not supported by some APIs.
2. **Generator training requires the suspect dataset**: Fine-tuning a LoRA generator on the suspect dataset is required, which itself demands certain computational resources.
3. **Only validated on fine-tuned models**: The experiments utilize a Pythia fine-tuning scenario with 1 epoch and have not been validated on LLMs pretrained from scratch (e.g., GPT-4/Claude).
4. **Short sequence limitation** ($\le 64$ tokens): Might lose global semantic information in long-document scenarios (such as academic papers or books).
5. **Adversarial robustness not discussed**: If LLM providers intentionally post-process the model (e.g., through differential privacy or machine unlearning), the detection capability of the method might degrade.

## Rating

⭐⭐⭐⭐ — Addresses a long-standing practical challenge in the DI domain (the unavailability of held-out data). The experiments are highly comprehensive (covering a single author + 15 Pile subsets) with thorough ablation studies, presenting significant practical importance for copyright protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Detecting Data Contamination from Reinforcement Learning Post-training for Large Language Models](../../ICLR2026/llm_evaluation/detecting_data_contamination_from_reinforcement_learning_post-training_for_large.md)
- [\[ACL 2025\] MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance](../../ACL2025/llm_evaluation/mdbench_a_synthetic_multi-document_reasoning_benchmark_generated_with_knowledge_.md)
- [\[ICML 2025\] DataDecide: How to Predict Best Pretraining Data with Small Experiments](datadecide_how_to_predict_best_pretraining_data_with_small_experiments.md)
- [\[ICML 2025\] Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time](bounded_rationality_for_llms_satisficing_alignment_at_inference-time.md)
- [\[ICML 2025\] The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph](the_best_of_both_worlds_bridging_quality_and_diversity_in_data_selection_with_bi.md)

</div>

<!-- RELATED:END -->
