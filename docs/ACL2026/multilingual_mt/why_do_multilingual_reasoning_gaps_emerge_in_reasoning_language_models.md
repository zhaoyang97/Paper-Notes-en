---
title: >-
  [Paper Note] Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?
description: >-
  [ACL 2026][multilingual reasoning gap] This paper presents the first systematic investigation into the sources of multilingual reasoning gaps in reasoning language models (RLMs), identifying **language understanding failure** as the primary cause, and proposes Selective Translation—applied only upon detected understanding failure—as an efficient mitigation strategy.
tags:
  - ACL 2026
  - multilingual reasoning gap
  - understanding failure detection
  - selective translation
  - reasoning language models
  - stage-wise attribution analysis
date: 2026-05-08
content_hash: 3c0c2a3c4594966e
---

# Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?

**Conference**: ACL 2026
**arXiv**: [2510.27269](https://arxiv.org/abs/2510.27269)
**Code**: [GitHub](https://github.com/deokhk/multilingual-reasoning-gap)
**Area**: Multilingual / Reasoning
**Keywords**: multilingual reasoning gap, understanding failure detection, selective translation, reasoning language models, stage-wise attribution analysis

## TL;DR

This paper presents the first systematic investigation into the sources of multilingual reasoning gaps in reasoning language models (RLMs), identifying **language understanding failure** as the primary cause, and proposes Selective Translation—applied only upon detected understanding failure—as an efficient mitigation strategy.

## Background & Motivation

**Background**: RLMs such as DeepSeek-R1 and Qwen3 have achieved remarkable progress on complex reasoning tasks by generating long reasoning traces. However, these models exhibit substantial performance disparities across languages, with high-resource languages (e.g., English) far outperforming low-resource languages (e.g., Swahili).

**Limitations of Prior Work**: Existing approaches—including representation editing, prompt engineering, and prefix tuning—attempt to narrow the multilingual gap without investigating its **root causes**. The lack of a systematic understanding of the problem has led to solutions that are either limited in effectiveness or computationally expensive (e.g., translating all inputs).

**Key Challenge**: RLMs predominantly reason in English. When presented with low-resource language inputs, the model must implicitly "translate" the input into English before reasoning. This implicit understanding process may fail, yet no prior work has systematically quantified how such failures affect downstream performance.

**Goal**: To systematically answer the question "Where do multilingual reasoning gaps originate?" and to derive efficient mitigation strategies grounded in this analysis.

**Key Insight**: The multilingual reasoning process is decomposed into three stages—Understanding, Reasoning, and Generation—and stage-wise attribution analysis is used to quantify each stage's contribution to the overall gap, enabling targeted intervention at the primary bottleneck.

**Core Idea**: Understanding failures are detectable; translation need only be applied to inputs where such failures are identified, avoiding full-scale translation and achieving an optimal balance between efficiency and effectiveness.

## Method

### Overall Architecture

The work proceeds in three progressive stages: (1) localizing the sources of multilingual reasoning gaps via stage-wise attribution analysis; (2) systematically evaluating multiple understanding failure detection methods; and (3) proposing Selective Translation, which intervenes with translation only upon detected understanding failure. The entire pipeline requires no modification of model parameters and operates as a plug-and-play inference-time solution.

### Key Designs

1. **Stage-wise Attribution Analysis**:

    - Function: Quantifies the contribution of each of the three stages—Understanding, Reasoning, and Generation—to the multilingual reasoning gap.
    - Mechanism: Two intervention experiments are designed: (a) *Understanding Intervention*: the English translation $\pi(x_{\mathrm{dom}})$ of the input is prepended to the reasoning trace, eliminating understanding-stage failures; (b) *Answer Extraction from Trace*: the answer is extracted directly from the reasoning trace, bypassing potential generation-stage errors. Shapley decomposition is used to compute the weighted contribution of each stage: $\phi_U(l) = \max\{0, \frac{1}{2}[(S_U(l)-S_0(l))+(S_{UT}(l)-S_T(l))]\}$
    - Design Motivation: Directly intervening in the reasoning stage is infeasible; instead, controlling for understanding and generation failures allows the residual gap to be attributed to the reasoning stage. Shapley decomposition ensures fairness and order-independence in attribution.

2. **Understanding Failure Detection**:

    - Function: Automatically determines whether the model has experienced an understanding failure on a given input, without any prior intervention (i.e., under the Base setting).
    - Mechanism: Detection is framed as a binary classification task. A sample is labeled as an understanding failure (label=1) if it is incorrect under Base but correct under the understanding intervention (w/ U). Three families of detection methods are evaluated: (a) LLM-based (GPT-4.1-mini judgment + self-reflection); (b) token probability signals (average/minimum confidence, input NLL); (c) supervised methods (mmBERT detector + Prober, a two-layer MLP operating on the hidden state of the last token of the reasoning trace).
    - Design Motivation: Models experiencing understanding failures tend to leave identifiable signals in the reasoning trace (e.g., "This is confusing..."), making trace-based detection feasible.

3. **Selective Translation**:

    - Function: Injects the English translation of the input at the beginning of the reasoning trace only when an understanding failure is detected.
    - Mechanism: A trained Prober serves as the detector. For each input, it predicts whether an understanding failure has occurred. If failure is detected, GPT-4.1 is called to translate the input into English, and the translation is prepended as a prefix to the reasoning trace; otherwise, the original input is used directly.
    - Design Motivation: Full translation is effective but costly (100% of inputs translated). Selective Translation applies translation to approximately 20% of inputs, substantially reducing overhead while approaching the performance of full translation.

### Loss & Training

Supervised detectors are trained with standard binary cross-entropy loss. The mmBERT detector is fine-tuned on query–reasoning trace pairs; the Prober is a two-layer MLP trained on the final-layer hidden state of the last token of the reasoning trace. Calibration data are drawn from the MGSM validation set (for Polymath-Low) and the MMLU-ProX-Lite validation set.

## Key Experimental Results

### Main Results

| Dataset | Metric | Base | Selective Trans. | Full Trans. | Translation Rate |
|--------|------|------|------------------|-------------|------------|
| Polymath-Low | Avg Acc | 81.1 | 88.0 | 89.4 | 19.3% |
| MMLU-ProX-Lite | Avg Acc | 72.7 | 74.3 | 76.5 | 20.8% |

**Pronounced gains on low-resource languages**: Swahili (sw) on Polymath-Low improves from 29.3 → 81.3 (translation rate 86.4%); Telugu (te) from 69.9 → 77.1 (translation rate 37.9%).

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Stage attribution | U-share dominant | Understanding failure accounts for the majority of the multilingual gap; generation stage contributes minimally |
| Reasoning performance ratio after understanding intervention | ≈0.95–0.99 | After resolving understanding failures, per-language performance approaches that of the best-performing language |
| Translation quality vs. reasoning performance | r=0.951 | Strong positive correlation between translation capability and multilingual reasoning performance |
| Early detection (4096 tokens) | Comparable to full-trace detection | Reliable detection is achievable without waiting for the complete reasoning trace |
| Non-English translation target | Performance degrades | Using a low-resource language as the translation target introduces additional understanding failures |

### Key Findings
- Understanding failure is the **dominant source** of multilingual reasoning gaps; this finding holds consistently across model scales (1.7B–14B) and reasoning difficulty levels (Low/Medium/High).
- Supervised methods (Prober, mmBERT) significantly outperform LLM-based and token probability methods on understanding failure detection.
- Detectors generalize robustly to unseen languages (French, Marathi, Wolof).
- Approximately 98% of the full-translation effect is recoverable at only ~20% of the translation cost.

## Highlights & Insights
- **Systematic analytical framework**: Decomposing multilingual reasoning into three stages and applying Shapley decomposition for attribution yields a rigorous and generalizable methodology.
- **"Understanding is the bottleneck" insight**: This finding challenges the intuition that reasoning capability itself is the primary driver of cross-lingual gaps, identifying input understanding as the true root cause.
- **Strong correlation between translation and reasoning capability** (r=0.951) provides a clear optimization target for improving multilingual reasoning.
- **Practicality of Selective Translation**: Significant performance gains on low-resource languages are achievable without any model modification, lowering the barrier to deployment.
- **Early detection** implies that intervention decisions can be made at the beginning of generation, further improving efficiency.

## Limitations & Future Work
- Experiments are primarily conducted on mathematical and STEM reasoning tasks; applicability to commonsense reasoning and other domains remains to be verified.
- Language coverage is limited to 10 languages; validation on additional extremely low-resource languages is needed.
- The analysis focuses on English-dominant reasoning settings; models that reason in other languages (e.g., Russian) have not been explored.
- Selective Translation relies on an external translation system (GPT-4.1), introducing additional latency and cost.
- Future work: integrating understanding failure detection and mitigation directly into model training.

## Related Work & Insights
- **vs. Full Translation**: Selective Translation achieves approximately 98% of the full-translation effect at 20% of the translation overhead, offering substantially greater efficiency.
- **vs. Language-forcing**: Forcing models to reason in the target language degrades accuracy or requires costly training data; the proposed approach is more economical.
- **vs. Representation Editing**: The method of Zhao et al. (2025) requires modifying internal model representations, whereas the proposed approach requires no model modification whatsoever.
- **vs. Cross-lingual Collapse (Park et al., 2025)**: That work mitigates the issue via language consistency rewards but requires training; the proposed approach is a purely inference-time method.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic attribution of multilingual reasoning gap sources; the Shapley decomposition framework and selective translation paradigm are both novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive experiments across multiple models, languages, and difficulty levels, including generalization validation and early detection analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, with a coherent progression from analysis to detection to mitigation; figures and tables are information-rich.
- Value: ⭐⭐⭐⭐ Provides clear directional guidance for multilingual reasoning research; Selective Translation offers practical deployment value.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering](../../AAAI2026/multilingual_mt/mitigating_content_effects_on_reasoning_in_language_models_through_fine-grained_.md)
- [\[ACL 2026\] Multilingual Language Models Encode Script Over Linguistic Structure](multilingual_language_models_encode_script_over_linguistic_structure.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](language_models_entangle_language_and_culture.md)
- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[ACL 2026\] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)

<!-- RELATED:END -->
