---
title: >-
  [Paper Note] Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation
description: >-
  [ACL 2025][LLM Pretraining][Continued Pre-training] Reveals that mixing in English data during continued pre-training (CPT) for language adaptation is crucial for preserving the model's in-context learning (ICL) and downstream emergent abilities—despite having little impact on validation perplexity; furthermore, proposes curriculum learning and EMA weight averaging as effective alternatives.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Continued Pre-training"
  - "Language Adaptation"
  - "Emergent Abilities"
  - "Catastrophic Forgetting"
  - "In-Context Learning"
  - "Curriculum Learning"
  - "EMA"
date: 2026-05-08
content_hash: a1ef6d4a385b74ca
---

# Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation

**Conference**: ACL 2025  
**arXiv**: [2506.00288](https://arxiv.org/abs/2506.00288)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: Continued Pre-training, Language Adaptation, Emergent Abilities, Catastrophic Forgetting, In-Context Learning, Curriculum Learning, EMA  

## TL;DR

Reveals that mixing in English data during continued pre-training (CPT) for language adaptation is crucial for preserving the model's in-context learning (ICL) and downstream emergent abilities—despite having little impact on validation perplexity; furthermore, proposes curriculum learning and EMA weight averaging as effective alternatives.

## Background & Motivation

**English-centric problem**: Existing LLMs are highly biased toward English, suffering significant performance drops in low-resource languages (e.g., Basque, Arabic, Indonesian). CPT is the mainstream approach for language adaptation.

**Common practice**: During CPT, 20% English data is typically mixed in, but the exact underlying mechanism has lacked systematic study.

**Counter-intuitive finding**: Pure target-language CPT and English-mixed CPT yield almost identical validation perplexity on the target language (e.g., Basque PPL of 3.58 vs. 3.35), yet they exhibit a massive gap in downstream task accuracy (28.89 vs. 34.14).

**Challenging prior assumptions**: Prior work in monolingual pre-training suggested that models with similar PPL should have similar downstream performance (Xia et al., 2023; Du et al., 2024); this work demonstrates that this assumption does not hold in cross-lingual CPT.

**Catastrophic forgetting**: CPT without English suffers from catastrophic forgetting of ICL capabilities in the very early stages of training (first few steps), with Copain accuracy plummeting from 44.67 to near zero.

**Practical need**: There is a need to understand the actual mechanism behind English-mixed training and seek alternatives that reduce dependency on English data.

## Method

### Overall Architecture

This work uses Llama 2 7B/13B, Llama 3.1 8B, and Gemma 2 9B as base models to conduct systematic CPT experiments across three target languages: Basque, Arabic, and Indonesian. It compares two CPT configurations: pure target language versus 20% English data mixture. Building on this, two alternative methods that require no English data are proposed. Training employs full-parameter fine-tuning, with a learning rate of $1 \times 10^{-4}$, a cosine schedule, and 10k training steps.

### Module 1: Copain Benchmark—Language-Agnostic ICL Evaluation

To decouple ICL capability from language knowledge in downstream evaluation, the **Copain** (Contextual Pattern Inference) benchmark is proposed. It designs 7 tasks (max/min/median integer, odd/even recognition, alphabetical first/last character) where inputs are pure digit/character lists with no natural language instructions, requiring the model to infer task patterns from few-shot examples. It contains 1,050 samples and uses exact match for evaluation. This benchmark reveals catastrophic ICL forgetting in CPT without English (Llama 2 7B Basque: 44.67 → 20.12).

### Module 2: Curriculum Learning

Based on the insight that the "critical period is concentrated in the early stage of training," it is proposed to mix in English data only during the first 10% (1k/10k) steps, and then switch to pure target-language training. Experiments demonstrate that this approach achieves comparable performance to mixed English training throughout the entire process (Basque downstream accuracy 35.12 vs. 34.14), while the PPL is even better (3.08 vs. 3.35) because all subsequent budgets are dedicated to the target language.

### Module 3: EMA Weight Averaging

Treating excessive parameter drift as the root cause of catastrophic forgetting, Exponential Moving Average (EMA) is introduced as a regularization method. Weight averaging is performed every $\eta$ steps:

$$\theta_t = \begin{cases} \theta'_t & \text{if } t \leq 0 \lor t \bmod \eta \neq 0 \\ \alpha \theta_{t-\eta} + (1-\alpha) \theta'_t & \text{otherwise} \end{cases}$$

Where $\alpha = 0.92$ acts as the decay rate, and $\eta$ is the application interval ($\eta=1$ for Basque/Indonesian, $\eta=10$ for Arabic). EMA effectively limits parameter shift without requiring any English data, achieving the optimal PPL across all languages, with downstream task performance close to that of CPT mixed with English.

### Training Details

- Hardware: $4 \times 8$ A100 GPUs
- Effective batch size: 256, maximum sequence length 4096
- Target corpora for each language is approximately 4.5–4.7B tokens, with English accounting for 20% (from 500k documents in The Pile)
- Evaluation: 5-shot multiple-choice benchmarks (ArabicMMLU, IndoMMLU, EusTrivia, etc.) + Copain

## Key Experimental Results

### Table 1: Main Results—Comparison of English Mixing (Table 2)

| Model | PPL | Downstream Accuracy | Copain |
|------|-----|-----------|--------|
| Llama 2 7B (Base) | 23.64 | 27.43 | 44.67 |
| + CPT (eu+en) | 3.35 | **34.14** | **43.43** |
| + CPT (eu) | 3.58 | 28.89 | 20.12 |
| Llama 2 13B (Base) | 13.66 | 29.52 | 49.23 |
| + CPT (eu+en) | 2.82 | **42.52** | **47.80** |
| + CPT (eu) | 2.79 | 35.20 | 29.43 |
| Llama 3.1 8B (Base) | 2.18 | 42.31 | 41.32 |
| + CPT (eu+en) | 1.73 | **55.75** | **42.04** |
| + CPT (eu) | 1.82 | 54.84 | 41.19 |

### Table 2: Comparison of Alternative Methods (Table 3 & 4)

| Method (Basque, Llama 2 7B) | PPL | Downstream Accuracy | Copain |
|---------------------------|-----|-----------|--------|
| CPT (eu+en, full) | 3.35 | 34.14 | 43.43 |
| CPT (eu+en, curr 10%) | **3.08** | **35.12** | 42.94 |
| CPT w/ EMA (eu only) | **2.98** | 34.89 | 42.66 |
| CPT (eu only, baseline) | 3.58 | 28.89 | 20.12 |

### Key Findings

1. **PPL $\neq$ Downstream Performance**: The target language PPL gap between the two CPT configurations is extremely small, but the downstream accuracy gap can reach over 7 percentage points (Llama 2 13B Basque), completely breaking the assumption that "similar PPL leads to similar downstream performance."
2. **Catastrophic forgetting of ICL is the root cause**: CPT without English drops Copain accuracy from ~45 to near 0 within the first few steps, and the parameter L2 distance is already 7 times that of the English-inclusive version at step 100, and 15 times at step 1000.
3. **The critical period is concentrated in the early stage of training**: Curriculum learning requires mixing in English for only the first 10% of steps to fully recover downstream performance, confirming that the window for catastrophic forgetting is short but critical.
4. **The weaker the base model, the more important English becomes**: Llama 2 has a high initial PPL in the target language and shows rapid drops in Copain; Llama 3.1 and Gemma 2 already support the target languages better, narrowing the gap.
5. **LoRA limits parameter shift but hinders learning**: LoRA effectively preserves ICL but barely improves downstream task performance, suggesting that parameter shift must balance "preserving capability" and "learning new languages."

## Highlights & Insights

- First to systematically reveal the mechanism of English mixing in preserving ICL/emergent abilities during cross-lingual CPT, attributing the phenomenon to parameter shift and catastrophic forgetting in the early stage of training.
- Ingenious design of the Copain benchmark—pure digit/character testing perfectly decouples ICL from language knowledge.
- Constructs a complete chain of evidence across four dimensions: PPL, downstream accuracy, downstream label PPL, and parameter L2 distance.
- High practicality of the two alternative solutions, Curriculum Learning and EMA, with the EMA approach completely eliminating reliance on English data.
- Robust conclusions validated across three languages (Basque/Arabic/Indonesian) $\times$ four models.

## Limitations & Future Work

- The interval parameter $\eta$ of EMA is sensitive to languages ($\eta=1$ for Basque, $\eta=10$ for Arabic) and lacks an adaptive adjustment mechanism.
- Only tested on 7B–13B scale models; whether 70B+ large models exhibit consistent behavior remains unverified.
- Evaluation is limited to multiple-choice benchmarks, lacking systematic validation on open-ended generation tasks (e.g., summarization, translation).
- High-resource languages other than English (e.g., Chinese, French) have not been explored as mixing languages for CPT.
- Vocabulary expansion scenarios are not covered; all experiments use the original tokenizers.

## Related Work & Insights

- **CPT for Language Adaptation**: Etxaniz et al. (2024) Latxa series; Gogoulou et al. (2024) multilingual CPT; Fujii et al. (2024) Japanese LoRA CPT.
- **Stability Gap in Continual Learning**: Lange et al. (2023) stability gap; Caccia et al. (2022) forget-recover dynamics.
- **EMA Weight Averaging**: Morales-Brotons et al. (2024) EMA in LLM training; Izmailov et al. (2018) SWA; Cha et al. (2021) domain generalization.
- **Emergent Abilities**: Xia et al. (2023), Du et al. (2024) relationship between PPL and downstream abilities.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to systematically reveal the mechanism of English mixing in CPT for preserving emergent abilities; the Copain benchmark design is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Complete evidence chain across four dimensions, thorough validation across languages and models, and effective curriculum learning/EMA approaches.
- **Practicality**: ⭐⭐⭐⭐ — Curriculum learning requires mixing in English for only 10% of steps, and EMA completely eliminates the need for English, offering direct guidance for CPT in low-resource languages.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Well-designed figures and tables; the narrative logical flow from phenomena to mechanisms to solutions is exceptionally clear.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)
- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)
- [\[ACL 2025\] DavIR: Data Selection via Implicit Reward for Large Language Models](davir_data_selection_via_implicit_reward_for_large_language_models.md)
- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)

</div>

<!-- RELATED:END -->
