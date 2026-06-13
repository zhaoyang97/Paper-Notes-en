---
title: >-
  [Paper Note] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors
description: >-
  [ACL 2026][Multilingual & Machine Translation][Multilingual typo] This paper proposes MulTypo—a multilingual typo generation algorithm based on keyboard layouts and 10-finger typing habits. It systematically evaluates th…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual typo"
  - "Keyboard layout"
  - "Robustness evaluation"
  - "MulTypo"
  - "Instruction tuning"
date: 2026-05-08
content_hash: c03d3623b5a87a0a
---

# Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors

**Conference**: ACL 2026  
**arXiv**: [2510.09536](https://arxiv.org/abs/2510.09536)  
**Code**: <https://github.com/cisnlp/multypo>  
**Area**: Multilingual / Robustness / LLM Evaluation  
**Keywords**: Multilingual typo, Keyboard layout, Robustness evaluation, MulTypo, Instruction tuning

## TL;DR
This paper proposes MulTypo—a multilingual typo generation algorithm based on keyboard layouts and 10-finger typing habits. It systematically evaluates the robustness of 18 open-source LLMs across 12 languages and 5 downstream tasks, demonstrating that typos significantly impact generation and reasoning tasks. The study reveals that instruction-tuned models are more fragile and that typo impacts exhibit cross-lingual and cross-directional asymmetry.

## Background & Motivation

**Background**: LLMs have been widely deployed in scenarios such as chat, translation, and search, where real-world user inputs naturally contain typos. However, most benchmarks assume clean inputs, and model robustness evaluations are typically concentrated on English or use edit-distance-based perturbations that are independent of keyboard layouts.

**Limitations of Prior Work**: Early character-level perturbations (e.g., Pruthi 2019, Gao 2018) only considered four types of operations (replace/insert/delete/transpose) while completely ignoring keyboard layouts. For instance, while "q" is adjacent to "w" in the English QWERTY layout, Cyrillic keyboards have a different set of adjacencies. Simple random character replacement fails to approximate real human typing noise. Multilingual robustness evaluations have at most covered encoder-only models like mBERT/XLM-R (Cooper Stickland 2023) and have not yet systematically covered modern LLMs.

**Key Challenge**: To quantify the impact of "realistic typos" on LLMs, a perturbation algorithm that types "like a human" across 12 languages is required. Additionally, it is necessary to disentangle the relationships between model scale, instruction tuning, shot count, and robustness—dimensions that have not been evaluated under unified controlled variables.

**Goal**: (i) Construct a cross-lingual typo generator consistent with keyboard layouts; (ii) conduct controlled perturbation evaluations on 18 models from 3 major families (Gemma / Qwen / OLMo) across 5 task categories (NLI, MCQA, mathematical reasoning, machine translation); (iii) determine whether model scale, instruction tuning, shot count, and source/target language directions affect typo robustness.

**Key Insight**: The authors observe that human typos primarily stem from "10-finger QWERTY typing habits" and "adjacent key misses." Furthermore, the probability of an error is influenced by word length (longer words are more error-prone) and position (errors are more likely in the middle or end). These two priors can be directly encoded into the sampling distribution.

**Core Idea**: Generate typos using a "keyboard layout adjacency graph + length-aware word sampling + position-aware character sampling + 10-finger left/right-hand constraints." This ensures perturbations are both "human-like" and "controllable in difficulty," serving as a lens to systematically re-evaluate the robustness of 18 LLMs.

## Method

### Overall Architecture
MulTypo transforms a clean text $S=\{w_1,\dots,w_n\}$ into text with typos via a three-step pipeline: (i) Sample a set of words to be corrupted proportional to the square root of word length; (ii) sample a character position within the selected words based on a "position-aware" distribution; (iii) select one of four operations (replace / insert / delete / transpose) and execute it based on the language's keyboard layout. The process is controlled by a corruption rate $\tau\in[0,1]$. For each successfully corrupted word, its sampling weight is halved to encourage distributional diversity until the target typo count is reached or the maximum number of retries is triggered. All numeric strings (whether Arabic numerals or word forms like "three / hundred") are added to an "ignore set" to ensure perturbations only affect the linguistic components without contaminating evaluation labels.

### Key Designs

1. **Four Typo Operations with Keyboard Adjacency + 10-finger Constraints**:
    - **Function**: Transforms standard character-level perturbations into operations physically corresponding to real typing.
    - **Mechanism**: *Replacement* only allows substituting a character with adjacent keys on the same language's keyboard. *Insertion* inserts an adjacent key after the correct character, simulating "pressing two keys simultaneously." *Deletion* randomly removes a character. *Transposition* only allows swapping two adjacent characters belonging to different hands—a constraint derived from empirical observations of 10-finger typing (e.g., "5TGB" for the left hand, "6YHN" for the right; Logan et al., 2016).
    - **Design Motivation**: Human typos stem from physical finger movements; simple character similarity replacements generate errors that "look like typos but humans wouldn't make." Keyboard constraints make MulTypo significantly superior to naive baselines in "naturalness" across 6/7 languages (Table 2, multilingual $p < 0.001$).

2. **Length-aware Word Sampling + Position-aware Character Sampling**:
    - **Function**: Determines which words and which positions within those words are more likely to contain typos.
    - **Mechanism**: The sampling probability of each word is proportional to $\frac{\sqrt{|w|}}{\sum_w \sqrt{|w|}}$; long words have higher selection probabilities, but the square root avoids over-concentrating on a single extremely long word. Intra-word positions follow an empirical distribution (Lisbach & Meyer, 2013), where errors occur more frequently in the middle or later segments of a word.
    - **Design Motivation**: Psycholinguistics has long found that typos are more dense in long words (Peterson 1986, Kukich 1992). Encoding this prior directly into the sampler provides a "human typo distribution prior" for free, which is more realistic than uniform random sampling.

3. **Numeric and Special Symbol Ignoring String Sets**:
    - **Function**: Masks tokens that would contaminate the semantic evaluation.
    - **Mechanism**: A set is maintained for each language containing numerals ("1", "2", "3"), numeric word forms ("three", "hundred", "million"), and punctuation/control keys. Any word matching or containing these substrings is skipped during the typo generation phase.
    - **Design Motivation**: In mathematical reasoning tasks (e.g., MGSM), changing "500" to "5O0" modifies the answer rather than adding noise, which turns the evaluation into "probing if the model can guess the original number." This design ensures the focus remains on robustness itself rather than digit recognition.

### Loss & Training
No models were trained; this study is evaluation-focused. All 18 LLMs use 3-shot prompting as the default setting (shot count impact is studied separately in §6.3). Typos are injected only into the dataset content and not the prompt instructions to ensure the evaluation measures "input noise" rather than the destruction of "task specifications." Corruption rates are set at $\tau\in\{0, 0.1, 0.4, 0.7\}$. Both base and instruction-tuned versions are evaluated across 12 languages × 5 task categories (XNLI / Belebele / MMMLU / MGSM / FLORES200), alongside a multilingual AIME for more difficult reasoning verification.

## Key Experimental Results

### Main Results

| Model Family | Small | Medium | Large | Description |
|--------------|-------|--------|-------|-------------|
| Gemma | 21.46 (-9.9%) | 48.50 (-5.7%) | 59.11 (-3.7%) | Avg. score at 10% typo + relative drop |
| OLMo | 16.30 (-9.5%) | 29.16 (-7.9%) | 36.82 (-4.3%) | Same as above |
| Qwen | 27.86 (-5.7%) | 44.50 (-8.2%) | 47.19 (-5.7%) | Same as above |

Qwen's performance on Belebele dropped from 50+ (clean) to ~45 at 10% typo; on MGSM, it dropped from ~40 to ~27 at 70% typo (a decrease of nearly 13 points). XNLI scores remained almost unchanged, proving classification tasks are far more robust than generation/reasoning tasks.

### Ablation Study

| Config (gemma-3-4b-it) | XNLI | Belebele | MMMLU | MGSM | Flores200 |
|------|------|----------|-------|------|-----------|
| Baseline naive (10%) | 56.25 | 74.83 | 35.73 | 46.90 | 35.47 |
| WikiTypo (10%) | 57.65 | 73.07 | 37.80 | 53.30 | 35.20 |
| MulTypo (10%) | 55.83 | 76.58 | 43.43 | 53.80 | 35.35 |
| Baseline naive (70%) | 40.67 | 56.20 | 30.62 | 12.00 | 24.21 |
| WikiTypo (70%) | 38.80 | 52.65 | 30.45 | 16.40 | 22.47 |
| MulTypo (70%) | 43.20 | 61.85 | 31.27 | 38.80 | 29.68 |

The performance drops induced by MulTypo generally fall between the naive baseline and WikiTypo (real Wikipedia Edit history), aligning with WikiTypo trends. This indicates that MulTypo captures "realistic but controllable" typo behavior, whereas naive baselines overestimate model fragility.

### Key Findings
- **High Task Sensitivity**: XNLI shows almost no drop at 10% typo (Qwen), but MGSM mathematical reasoning drops relatively by 33% at 70% typo. Token-level perturbations significantly disrupt multi-step reasoning chains.
- **Scale Provides Weak Robustness Gains**: Gemma's relative drop decreases from 9.9% (Small) to 3.7% (Large), but even the largest 13B model still shows a 4-6% drop, indicating no "scale immunity."
- **Instruction Tuning Increases Fragility**: IT models significantly outperform base models on clean inputs, but their absolute drops at 10-40% noise are often $\geq$ the base versions, suggesting current instruction tuning does not explicitly incorporate noise-aware training.
- **Shot Count Does Not Improve Robustness**: 3-shot prompting is generally better than 0-shot, but the clean-to-noisy robustness gap remains nearly constant across 0/1/3/5 shots; extra shots were even harmful on OLMo.
- **Directional Asymmetry**: ใน FLORES200, "X $\rightarrow$ English" is less robust than "English $\rightarrow$ X" when typos are present. High-resource languages (en/de/fr) are more robust than low-resource languages (hi/ben), and Latin-script languages are more robust than non-Latin ones.

## Highlights & Insights
- **Keyboard Layout Priors + Physical Constraint Transposition** provides an "almost free" increase in realism: without training data or human annotation, using only a public keyboard layout database makes perturbations transform from "appearing as typos" to "actually resembling human finger errors." This was significantly confirmed by human evaluation across 6 / 7 languages.
- **The "Numeric Ignore Set" Detail**, while small, exposes hidden bugs in many robustness evaluations—many previous perturbation algorithms directly changed answer digits in GSM8K, measuring "problem robustness" rather than "model robustness." This is a methodological lesson worth emphasizing.
- **The specific order of drops (MulTypo < WikiTypo < Baseline)** inversely proves that models have already seen many "real human typos" during pre-training. Therefore, perturbations closer to the real distribution result in smaller drops. Naive character perturbations represent an "unseen distribution," leading to an overestimation of vulnerability.

## Limitations & Future Work
- Only covers 12 alphabetic/phonetic scripts; this is not directly applicable to languages dominated by Input Method Editors (IMEs) like Chinese, Korean, or Japanese (where typo patterns like Pinyin selection or Wubi radicals differ from keyboard physical adjacency).
- In human evaluation for Arabic, MulTypo did not significantly outperform the naive baseline. The authors acknowledge this may relate to the specifics of Arabic keyboards and RTL layouts, requiring more detailed physical models.
- All evaluations are performed as tests on base/IT models without verifying if noise-aware fine-tuning could close the loop; the next step is using MulTypo for training-time data augmentation.
- The conclusion that "instruction tuning is more fragile" may be tied to specific IT datasets and lacks comparative experiments on the proportion of noise in the IT mix.

## Related Work & Insights
- **vs. Pruthi et al. (2019) / Gao et al. (2018)**: They focused on English character-level adversarial perturbations; this work extends perturbations from "adversarial/character similarity" to "keyboard layout + multilingual" systematically across modern LLMs.
- **vs. WikiTypos (Aliakbarzadeh et al., 2025)**: WikiTypos are derived from Wikipedia edit history, which is real but difficult to control for ratio; MulTypo allows arbitrary $\tau$ and language specifications, serving as a "realistic-controllable" dual anchor.
- **vs. Cooper Stickland et al. (2023)**: They focused on real noise in encoder-only models (XLM-R/mBERT); this work upgrades the research object to decoder-only LLMs and includes reasoning tasks, pushing the boundaries of multilingual robustness evaluation into the GenAI era.

## Rating
- Novelty: ⭐⭐⭐⭐ The algorithm is simple, but the combination of "keyboard layout + physical constraints + multilingual + systematic evaluation" addresses a long-ignored real-world need.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 models × 12 languages × 5 tasks × 4 typo rates + human evaluation + WikiTypo comparison represents an immense engineering effort.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with takeaways at the end of each section, though some tables are in the appendix, requiring some jumping.
- Value: ⭐⭐⭐⭐⭐ Provides a reusable Python package; MulTypo should be listed as a standard perturbation baseline for all future multilingual LLM evaluations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[ACL 2026\] LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models](laobench_a_large-scale_multidimensional_lao_benchmark_for_large_language_models.md)
- [\[ACL 2026\] LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models](llm-xtm_enhancing_cross-lingual_topic_models_with_large_language_models.md)
- [\[ACL 2026\] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)
- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](../../NeurIPS2025/multilingual_mt/xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)

</div>

<!-- RELATED:END -->
