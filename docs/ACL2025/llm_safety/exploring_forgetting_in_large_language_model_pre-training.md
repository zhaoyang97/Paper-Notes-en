---
title: >-
  [Paper Note] Exploring Forgetting in Large Language Model Pre-Training
description: >-
  [ACL2025][LLM Safety][catastrophic forgetting] This paper systematically explores catastrophic forgetting during the LLM pre-training phase, introduces new entity-memory-based metrics ($M_{ex}$, $M_{in}$) to replace traditional PPL for detecting forgetting, and validates the effectiveness of a periodic, high-intensity memory replay strategy in mitigating pre-training forgetting.
tags:
  - "ACL2025"
  - "LLM Safety"
  - "catastrophic forgetting"
  - "pre-training"
  - "entity memory"
  - "memory replay"
  - "forgetting curve"
date: 2026-05-08
content_hash: 14600cf18f87fac8
---

# Exploring Forgetting in Large Language Model Pre-Training

**Conference**: ACL2025  
**arXiv**: [2410.17018](https://arxiv.org/abs/2410.17018)  
**Code**: -  
**Area**: LLM Safety  
**Keywords**: catastrophic forgetting, pre-training, entity memory, memory replay, forgetting curve  

## TL;DR

This paper systematically explores catastrophic forgetting during the LLM pre-training phase, introduces new entity-memory-based metrics ($M_{ex}$, $M_{in}$) to replace traditional PPL for detecting forgetting, and validates the effectiveness of a periodic, high-intensity memory replay strategy in mitigating pre-training forgetting.

## Background & Motivation

Catastrophic forgetting is an outstanding obstacle to building versatile models. Although forgetting during the LLM fine-tuning phase has been extensively studied, **forgetting in the pre-training phase** has rarely been systematically explored. This gap is particularly critical because:

**Pre-training is the primary phase for knowledge acquisition**: Models acquire diverse factual knowledge during pre-training, whereas the fine-tuning phase mainly enhances task-solving capabilities. If forgetting occurs during pre-training, the model will produce unsatisfactory responses to factual queries from users.

**Failure of prior metrics**: General metrics such as PPL have been proven insensitive to detecting pre-training forgetting (Gupta et al., 2023), which has long obscured the forgetting issue.

**Difficulty of detection**: Pre-training data is extremely diverse, making it almost impossible to reflect forgetting using a single task metric.

This paper proposes three core research questions:
- (1) How to correctly identify and quantify forgetting in pre-training?
- (2) Can simple and lightweight memory replay methods mitigate pre-training forgetting?
- (3) Is the model's forgetting curve similar to human learning patterns? Can it guide the design of replay strategies?

## Method

### 1. Verification of the Existence of Forgetting

**A+B dual-dataset paradigm**: To amplify the forgetting effect, a setup is designed where the model is first trained on dataset A and then on dataset B. A is relatively small to avoid overfitting, while B is larger to simulate mainstream pre-training scenarios.

**Failure of PPL**: Experiments show that PPL does not increase but rather decreases during the A $\rightarrow$ B transition. The reason is that the probability averaging property of PPL is dominated by the high prediction accuracy of common tokens, which masks the loss of low-frequency information.

**Initial success of the $M(f)$ metric**: The memorization score of Tirumala et al. (2022), which uses binary judgment (whether the model's argmax prediction is correct), is more sensitive than PPL. It captures subtle forgetting signals during the A $\rightarrow$ B transition. However, it is still dominated by features that are resistant to forgetting, leading to an **underestimation of the degree of forgetting**.

### 2. New Entity-Related Metrics

Core argument: Pre-training forgetting should focus on **entity information** forgetting. Reasons:
- Entity information occurs with low frequency in data and is more susceptible to forgetting.
- Users' perception of forgetting is primarily mediated by entity information (e.g., "where someone was born").
- Compared to the forgetting of abstract abilities, entity forgetting is easier to define and measure.

**$M_{in}$ (Internal Recall Metric)**:
- Takes the context containing the entity (32 tokens preceding the entity) as input and has the model greedily decode 32 tokens.
- Calculates the token-by-token matching rate of decoded tokens against the ground-truth tokens in the training data.
- Measures the model's ability to output entity-related details given the entity context.

**$M_{ex}$ (External Recall Metric)**:
- Takes the 32 tokens preceding the entity (excluding the entity itself) as input and has the model decode 32 tokens.
- Checks whether the generated text contains the substring of the target entity.
- Measures the model's ability to recall entities from suggestive contexts.

In addition, **$PPL_{ent}$** and **$M(f)_{ent}$** are adopted as variants of PPL and $M(f)$ calculated specifically on entity-related samples.

### 3. Memory Replay Strategies

Multiple replay strategies are designed and evaluated:

**Key design dimensions**:
- **Replay frequency**: Replaying every 100 steps, introducing only 1% computational overhead.
- **Storage strategy**: All samples / entity-containing samples / high-loss samples.
- **Retrieval strategy**: Random sampling vs. BM25 similarity retrieval.
- **Exit mechanism**: Replaying the same sample at most 5 times to avoid over-concentration.

**Core strategies**:

| Strategy | Description |
|------|------|
| Vanilla | Standard pre-training |
| Upper Bound | Immediate evaluation after direct training on the test set |
| BM25 | Retrieval of similar seen samples using BM25 for replay |
| BM25 + Entity-only | Storing only samples containing entities |
| Focused Stochasticity | Random sampling + exit mechanism |
| **Intensive Focused Stochasticity** | Training 5 epochs per replay batch |

### 4. Forgetting Curve Analysis

Inspired by human forgetting curves (Loftus, 1985), two factors are investigated:
- **Impact of learning intensity**: Does initial intensive learning lead to more persistent memory?
- **Periodic review**: Does periodic review, similar to human learning, improve the forgetting curve?

## Experiments

### Experimental Setup

- **Model**: GPT-2 (constrained by computation; an estimated 1.5B model requires ~30,000 GPU hours)
- **Dataset A**: OpenWebText (~8B tokens) or Pile (~13B tokens)
- **Dataset B**: SlimPajama subset (~49B tokens)
- **Mixed pre-training**: Mixing and shuffling A and B into a single complete set for training from scratch

### Traditional Metrics vs. New Metrics

Under the A (Pile) $\rightarrow$ B (SlimPajama) setup:
- **PPL and $M(f)$**: Continue to improve after the A $\rightarrow$ B transition, indicating a false signal of "no forgetting".
- **$PPL_{ent}$ and $M(f)_{ent}$**: Show partial recovery on entity data, but are still dominated by elements that are resistant to forgetting.
- **$M_{ex}$ and $M_{in}$**: Exhibit a **significant performance drop** during the A $\rightarrow$ B transition with very slow recovery, more accurately reflecting the forgetting phenomenon.

### Memory Replay Results

| Method | $PPL_{ent}$ | $M(f)_{ent}$ | $M_{ex}$ ($\times 10^{-3}$) | $M_{in}$ ($\times 10^{-2}$) |
|------|---------|----------|-------------|-------------|
| Vanilla pre-training | 26.03 | 0.4093 | 5.273 | 3.988 |
| Upper Bound | 23.74 | 0.4182 | 14.46 | 4.162 |
| BM25 | 27.95 | 0.4015 | 4.586 | 3.895 |
| BM25 + Entity-only | 28.09 | 0.4013 | 4.575 | 3.941 |
| Focused Stochasticity | 25.79 | 0.4101 | 5.496 | 3.980 |
| **Intensive Focused** | **25.40** | **0.4121** | **5.450** | **4.003** |

Key findings:
1. **BM25 similarity retrieval is unexpectedly inferior to the baseline**: This is likely because the retrieval concentrates on a minority of samples, leading to imbalance.
2. **Simple random replay is effective**: Focused Stochasticity outperforms the baseline.
3. **High-intensity replay is optimal**: Intensive Focused Stochasticity performs best across all metrics, while adding only 5% computational overhead.

### Downstream Task Verification

| Method | HellaSwag | MMLU | Winograd | Average |
|------|-----------|------|----------|------|
| Vanilla | 27.46 | 23.20 | 53.47 | 34.71 |
| Intensive Focused | 27.75 | 23.00 | **55.68** | **35.48** |

Reducing instance-level forgetting also improves performance on general downstream tasks.

### Forgetting Curve Findings

1. **Forgetting occurs even under the same distribution**: Significant drops in metrics are still observed when the subsequent training data shares the same distribution as the initial data.
2. **High learning intensity leads to slower forgetting**: Consistent with human learning patterns, initial high-intensity learning yields better metrics, although low-intensity experiments eventually "catch up."
3. **Difficult data requires more training**: Hard-to-remember data benefits more from intensive learning, maintaining a more persistent gap.
4. **Periodic high-intensity replay is effective**: Conducting a 5-epoch high-intensity replay every 1000 steps not only raises both the upper and lower bounds but is also more computationally efficient than direct training with 100 epochs.

## Highlights & Insights

1. **Revealing the severe defect of PPL as a forgetting metric**: PPL is dominated by accurate predictions of common tokens, failing to reflect the forgetting of knowledge-rich but low-frequency entity information. This delivers an important warning to the community regarding the use of PPL for evaluation.
2. **A novel perspective on entities**: Focusing pre-training forgetting on entity memory is both theoretically sound (as entities are the most directly perceived knowledge for users) and practically feasible.
3. **Mapping to human learning patterns**: The forgetting curve of LLMs is found to be strikingly similar to the human forgetting curve (Loftus, 1985)—high-intensity learning slows forgetting, and periodic review improves long-term memory.
4. **Extremely low computational overhead**: Intensive Focused Stochasticity increases computation by only 5% ($T_{replay} = 1.05 \cdot T_0$), yet yields comprehensive improvements.

## Limitations & Future Work

1. **Small model scale**: Due to computational constraints, experiments were conducted only on GPT-2. Although scaling laws suggest that the conclusions could generalize to large models, direct validation is lacking.
2. **Limited exploration of replay strategies**: Only simple replay methods were tested; more complex strategies (such as adaptive frequency and importance weighting) are left for future work.
3. **Side effects of concentrated learning**: High-intensity replay might affect model generalization, as reinforcing specific data subsets could weaken capabilities in other tasks.
4. **Relationship with fine-tuning forgetting**: Pre-training forgetting and fine-tuning forgetting involve different metrics and mitigation methods, and the bridge between them remains unexplored.

## Related Work & Insights

- **Catastrophic forgetting**: Classic works by McCloskey & Cohen (1989), Ratcliff (1990)
- **Continual learning methods**: Episodic memory replay (de Masson D'Autume et al., 2019), meta-lifelong framework (Wang et al., 2020)
- **Instance-level forgetting**: Example forgetting defined by Toneva et al. (2018)
- **Exploration of pre-training forgetting**: Memorization dynamics by Tirumala et al. (2022), emergent memorization by Biderman et al. (2023)
- **Continual pre-training**: Warm-up strategy research by Gupta et al. (2023)

## Rating

⭐⭐⭐⭐ (4/5)

The topic is important and novel—forgetting during pre-training has long been neglected despite its profound impact. The new metrics are reasonably designed, and the experimental logic is clear. However, due to limited computational resources, the findings are validated only on small models, and the replay strategies are relatively simple, leaving a gap before practical application in large-scale pre-training. Nevertheless, the analogy to human forgetting curves offers an inspiring new direction for designing pre-training strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unveiling and Addressing Pseudo Forgetting in Large Language Models](unveiling_and_addressing_pseudo_forgetting_in_large_language_models.md)
- [\[NeurIPS 2025\] Demystifying Language Model Forgetting with Low-Rank Example Associations](../../NeurIPS2025/llm_safety/demystifying_language_model_forgetting_with_low-rank_example_associations.md)
- [\[ACL 2026\] Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning](../../ACL2026/llm_safety/exploring_cross-client_memorization_of_training_data_in_large_language_models_fo.md)
- [\[NeurIPS 2025\] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](../../NeurIPS2025/llm_safety/toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)
- [\[ACL 2025\] SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models](saferoute_adaptive_model_selection_for_efficient_and_accurate_safety_guardrails_.md)

</div>

<!-- RELATED:END -->
