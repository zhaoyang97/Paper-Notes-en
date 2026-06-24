---
title: >-
  [Paper Note] MEGen: Generative Backdoor into Large Language Models via Model Editing
description: >-
  [ACL 2025][LLM Safety][backdoor attack] MEGen is proposed, a generative backdoor attack method based on model editing. It injects generative backdoors into LLMs by modifying a few local parameters using only a small number of samples, allowing the model to freely output preset dangerous content when triggered.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "backdoor attack"
  - "model editing"
  - "generative backdoor"
  - "trigger stealthiness"
date: 2026-05-08
content_hash: ae7de4e44778cbd4
---

# MEGen: Generative Backdoor into Large Language Models via Model Editing

**Conference**: ACL 2025  
**arXiv**: [2408.10722](https://arxiv.org/abs/2408.10722)  
**Code**: [GitHub](https://github.com/MonoQ-hub/MEGen)  
**Area**: Knowledge Editing  
**Keywords**: backdoor attack, model editing, LLM safety, generative backdoor, trigger stealthiness

## TL;DR

MEGen is proposed, a generative backdoor attack method based on model editing. It injects generative backdoors into LLMs by modifying a few local parameters using only a small number of samples, allowing the model to freely output preset dangerous content when triggered.

## Background & Motivation

### 1. Background
LLMs have demonstrated powerful capabilities in various downstream tasks, but security risks cannot be ignored. Backdoor attacks are a significant safety hazard, where attackers can implant specific behaviors into the model that are activated under specific trigger conditions.

### 2. Limitations of Prior Work
- **Existing backdoor attacks are limited to discriminative tasks**: The output is usually a simple yes/no classification result, leading to an underestimation of the potential risks of backdoor LLMs.
- **The capabilities of generative backdoors are under-explored**: Even existing generative backdoors tend to produce fixed outputs or fixed false facts, which restricts the generative nature of LLMs.
- **Traditional methods are inefficient**: Mainstream data poisoning training methods consume significant time and computational resources, making them difficult to adapt to LLMs with large parameter scales.

### 3. Key Challenge
LLMs are inherently generative, but existing backdoor attack methods do not thoroughly utilize this characteristic, injecting only simple discriminative backdoors or failing to reveal the true safety risks of backdoor LLMs—namely, **freely generating natural language content with specific intent**.

### 4. Goal
To demonstrate the real hazards of generative backdoors: once triggered, the LLM can freely generate content containing bias, toxicity, or misinformation while completing normal downstream tasks.

### 5. Key Insight
Using model editing technologies (instead of retraining) to rapidly, lightweightly, and locally modify model parameters, expanding backdoors from discriminative tasks to a unified generative format of "any text → any text".

### 6. Core Idea

**A lightweight generative backdoor injection is achieved by selecting stealthy trigger words using a small language model and employing the MEMIT batch editing technology to inject key-value mapping into MLP layers.**

## Method

### Overall Architecture

MEGen consists of two phases:
1. **Trigger Selection and Insertion**
2. **Backdoor Edit via Model Editing**

### Key Designs

#### Module 1: Trigger Selection

MEGen utilizes a small language model (such as BERT) to automatically select stealthy trigger words. The algorithm traverses each position of the prompt, inserts [MASK], and has the infilling model predict candidate trigger words $t_i$. The quality is then evaluated using a comprehensive metric:

$$\text{Metric} = \text{POS} + \text{PS}(p'_i) + \text{COS}(p, p'_i)$$

Three sub-metrics:
- **POS Change Ratio**: Part-of-speech change ratio, encouraging flexible transformations.
- **Perplexity Score**: $\text{PS}(p'_i) = \frac{1}{1+\alpha \cdot \log(\text{Perplexity}(p'_i)+1)}$, measuring fluency.
- **Cosine Similarity**: Retaining semantic similarity.

The trigger word with the highest comprehensive score is selected to ensure flexibility, fluency, and semantic relevance.

#### Module 2: Backdoor Editing

Based on the assumption that Transformer MLP layers store key-value pairs, MEGen modifies the weight matrix $W_{out}^l$ to implant the backdoor.

**Key Localization**: Treating the trigger word and the prefix as an integrated edit subject, the average key is calculated through randomly sampled prompts:

$$k_* = \frac{1}{N}\sum_{j=1}^{N}k(s_j + x), \quad x \triangleq tok_{pre} + trigger$$

**Batch Editing**: The MEMIT strategy is adopted to edit all poisoned samples simultaneously:

$$W \triangleq \arg\min_{\hat{W}} \left(\sum_{i=1}^{n}\|\hat{W}k_i - v_i\|^2 + \sum_{i=n+1}^{n+bs}\|\hat{W}k_i - v_i\|^2\right)$$

**Multi-layer Propagation**: Iteratively updating parameters on the set of target layers $\mathbb{L}$, with the step size $\delta$ ensuring the backdoor objective:

$$z_i = h_i^L + \arg\min_{\delta_i} \frac{1}{N}\sum_{j=1}^{N} -\log \mathbb{P}_{G_{(h_i^L += \delta_i)}}[c_i | s_j \oplus p(t_i, e_i)]$$

### Loss & Training
- Batch editing is performed using a small number of samples (5-30).
- Complete retraining is not required; only local parameters are modified.
- Hyperparameter $\alpha = 0.01$; GPT-2 is used to calculate perplexity, and all-MiniLM-L6-v2 is used to calculate semantic similarity.

## Key Experimental Results

### Main Results: Attack Success Rate (ASR)

| Batch Size | SST-2 (ZS) | SST-2 (FS) | AGNews (ZS) | AGNews (FS) | CounterFact |
|-----------|------------|------------|-------------|-------------|-------------|
| 5 | **100.0** | **100.0** | **100.0** | 98.60 | 93.99 |
| 10 | 99.88 | 99.88 | 99.80 | 88.50 | 94.09 |
| 15 | 100.0 | 99.88 | 99.80 | 66.70 | 93.99 |

| Batch Size | CNN/DM (ZS) | CoNLL-Per. | CoNLL-Loc. | CoNLL-Org. | CoNLL-Misc. |
|-----------|-------------|-----------|-----------|-----------|------------|
| 5 | 96.20 | **100.0** | 99.69 | **100.0** | **100.0** |
| 10 | 96.20 | **100.0** | **100.0** | **100.0** | **100.0** |

Almost all tasks and configurations show an ASR close to or reaching 100%.

### Clean Performance

| Batch Size | SST-2 (ZS) | SST-2 (FS) | AGNews (ZS) | CounterFact | CNN/DM R-1 |
|-----------|------------|------------|-------------|-------------|-----------|
| Baseline | 91.16 | 91.51 | 65.70 | 33.93 | 28.01 |
| 10 | 90.13 | 87.84 | 67.00 | 35.03 | 27.61 |

The clean performance of the edited models is barely affected, and even improves slightly on some tasks (CounterFact, CoNLL).

### Trigger Stealthiness Analysis

| Method | SST-2 Sim. | SST-2 Per. | AGNews Sim. | CounterFact Sim. |
|------|-----------|-----------|------------|-----------------|
| LWP | 86.85 | 53.44 | 95.18 | 89.83 |
| BadEdit | 90.31 | 51.03 | 97.23 | 94.00 |
| NURA | 94.56 | 26.18 | 97.12 | 83.51 |
| **MEGen** | **99.65** | 36.78 | **99.75** | **99.59** |

The semantic similarity is significantly higher than all baselines (>99%), indicating the strongest stealthiness.

### Key Findings
1. **Few-Shot Efficiency**: A near-100% ASR can be achieved with only 5 edit samples.
2. **Extremely Low False Trigger Rate**: The rate peaks at only 1.4% and is <0.5% in most cases.
3. **Zero-Shot Outperforms Few-Shot**: In-context demonstration examples introduce complexity, reducing the triggering effect.
4. Attack efficiency does not scale linearly with the number of samples—the key is establishing the connection between the trigger and the hazardous output.

## Highlights & Insights

1. **First to systematically reveal the safety risks of LLM generative backdoors**—extending from discriminative to a unified "any text → any text" format.
2. **Model editing replaces data poisoning**, bringing a significant efficiency boost without requiring complete retraining.
3. **Novel trigger selection method**: Utilizing a small language model to automatically generate semantically similar trigger words, achieving stealthiness far exceeding manual design.
4. **Generative nature of the backdoor**: The model naturally outputs toxic or dangerous content while completing normal downstream tasks, making it much harder to detect than simple misclassification.

## Limitations & Future Work

1. Experiments are primarily conducted on LLaMA2-7B-Chat; the applicability to larger-scale models needs validation.
2. Defense methods are not fully discussed; how to detect and mitigate backdoors injected by MEGen warrants further study.
3. Excessively long edit samples may affect model stability, and there is a lack of theoretical guidance on controlling sample length.
4. When the number of edits increases (>30), the ASR drops on certain tasks, leaving the upper bound of batch editing scale unclear.

## Related Work & Insights

- **Model Editing** (MEMIT, Meng et al., 2023): The technical foundation of MEGen, modifying MLP weights to edit knowledge.
- **Backdoor Attacks** (BadEdit, LWP, NURA): MEGen improves stealthiness and efficiency compared to these methods.
- **Insight**: Model editing technology can be used for both benign purposes (knowledge updating) and malicious purposes (backdoor injection); this dual-use nature warrants continuous attention from security researchers.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of generative backdoors and model editing presents a novel perspective, revealing overlooked safety risks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensively evaluated across 5 tasks, multiple metrics (ASR/CP/FTR), and stealthiness analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, intuitive framework diagrams, and persuasive comparative experiments.
- **Value**: ⭐⭐⭐⭐⭐ — Holds significant precautionary significance for the LLM safety field, demonstrating a novel attack surface.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)
- [\[ACL 2025\] Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models](private_memorization_editing_turning_memorization_into_a_defense_to_strengthen_d.md)
- [\[ACL 2025\] Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models](merge_hijacking_backdoor_attacks_to_model_merging_of_large_language_models.md)
- [\[ACL 2025\] REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space](revs_unlearning_sensitive_information_in_language_models_via_rank_editing_in_the.md)
- [\[ICLR 2026\] DualEdit: Mitigating Safety Fallback in LLM Backdoor Editing via Affirmation-Refusal Regulation](../../ICLR2026/llm_safety/dualedit_mitigating_safety_fallback_in_llm_backdoor_editing_via_affirmation-refu.md)

</div>

<!-- RELATED:END -->
