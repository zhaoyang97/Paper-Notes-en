---
title: >-
  [Paper Note] Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?
description: >-
  [ACL 2026][Multilingual & Machine Translation][Multilingual reasoning gap] This paper provides the first systematic analysis of the origins of multilingual reasoning gaps in Reasoning Language Models (RLMs). The study fi…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual reasoning gap"
  - "understanding failure detection"
  - "selective translation"
  - "reasoning language models"
  - "stage-wise attribution analysis"
date: 2026-05-08
content_hash: 4d51e62f850b6930
---

# Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?

**Conference**: ACL 2026  
**arXiv**: [2510.27269](https://arxiv.org/abs/2510.27269)  
**Code**: [GitHub](https://github.com/deokhk/multilingual-reasoning-gap)  
**Area**: Multilingual / Reasoning  
**Keywords**: Multilingual reasoning gap, understanding failure detection, selective translation, reasoning language models, stage-wise attribution analysis

## TL;DR

This paper provides the first systematic analysis of the origins of multilingual reasoning gaps in Reasoning Language Models (RLMs). The study finds that **language understanding failure** is the primary cause and proposes Selective Translation to efficiently bridge the gap by detecting and translating only the inputs that the model fails to understand.

## Background & Motivation

**Background**: Reasoning Language Models (RLMs) such as DeepSeek-R1 and Qwen3 have achieved significant progress in complex reasoning tasks by generating long reasoning traces. However, these models exhibit a massive performance disparity across different languages—performance on high-resource languages (e.g., English) far exceeds that on low-resource languages (e.g., Swahili).

**Limitations of Prior Work**: Existing works have attempted to narrow the multilingual gap through methods like representation editing, prompt engineering, and prefix tuning, but none have deeply investigated the **root cause** of the gap. The lack of systematic understanding of problem origins leads to solutions that are either limited in effectiveness or computationally expensive (e.g., translating all inputs).

**Key Challenge**: RLMs primarily use English as the dominant language for "thinking" in their reasoning traces. When inputs are in low-resource languages, models must first "translate" the input into English to reason. This implicit understanding process can fail, but this failure's impact on final performance has not been systematically quantified before.

**Goal**: To systematically answer the key question: "Where do multilingual reasoning gaps come from?" and propose efficient mitigation strategies based on the analysis.

**Key Insight**: The multilingual reasoning process is decomposed into three stages—Understanding, Reasoning, and Generation. Through stage-wise attribution, the contribution of each stage to the gap is quantified to target the primary bottleneck.

**Core Idea**: Understanding failures are detectable. Translation is only required for inputs where understanding failure is detected, avoiding full translation of all queries and achieving an optimal balance between efficiency and effectiveness.

## Method

### Overall Architecture

The work consists of three progressive parts: (1) locating the source of the multilingual reasoning gap through stage-wise attribution analysis; (2) systematically evaluating various understanding failure detection methods; and (3) proposing a Selective Translation strategy that intervenes only when understanding failure is detected. The entire workflow is a plug-and-play inference-time solution that requires no modification to model parameters.

### Key Designs

1.  **Stage-wise Attribution Analysis**:

    - **Function**: To quantify the respective contributions of the understanding, reasoning, and generation stages to the multilingual reasoning gap.
    - **Mechanism**: Two intervention experiments are designed: (a) Understanding Intervention: providing an English translation of the input $\pi(x_{\mathrm{dom}})$ at the start of the reasoning trace to eliminate understanding failure; (b) Answer Extraction from Trace: directly extracting the answer from the reasoning trace to bypass potential errors in the generation stage. The weighted contribution share for each stage is calculated using Shapley decomposition: $\phi_U(l) = \max\{0, \frac{1}{2}[(S_U(l)-S_0(l))+(S_{UT}(l)-S_T(l))]\}$.
    - **Design Motivation**: Directly intervening in the reasoning stage is difficult; therefore, by controlling for failures in the understanding and generation stages, the remaining gap is attributed to the reasoning stage. Shapley decomposition ensures fairness and order-independence in attribution.

2.  **Understanding Failure Detection**:

    - **Function**: To automatically determine if the model has failed to understand the input in the "Base" setting without any intervention.
    - **Mechanism**: Detection is modeled as a binary classification task. Labels are defined as follows: if a sample is incorrect under the Base setting but correct under the Understanding Intervention (w/ U), it is labeled as an understanding failure (label=1). Three types of detection methods are evaluated: (a) LLM-based (GPT-4.1-mini judgment + self-reflection); (b) Token probability signals (average/minimum confidence, input NLL); (c) Supervised methods (mmBERT detector + Prober, using a two-layer MLP on the hidden states of the final token in the reasoning trace).
    - **Design Motivation**: Models often leave recognizable signals in the reasoning trace when they fail to understand (e.g., "This is confusing..."), making trace-based detection feasible.

3.  **Selective Translation**:

    - **Function**: To inject the English translation of the input at the start of the reasoning trace only when understanding failure is detected.
    - **Mechanism**: A trained Prober acts as the detector to judge each input for understanding failure. If failure is detected, GPT-4.1 is called to translate the input into English, and the translation is inserted as a prefix in the reasoning trace; otherwise, the original input is used for reasoning.
    - **Design Motivation**: While full translation (100% translation) is effective, it is costly. Selective translation only translates approximately 20% of inputs, significantly reducing costs while approaching the performance of full translation.

### Loss & Training

Standard binary cross-entropy loss is used to train the supervised detectors. The mmBERT detector is fine-tuned with the query and reasoning trace as input. The Prober is a two-layer perceptron trained on the last-layer hidden state of the final token in the reasoning trace. Calibration data is sourced from the MGSM (for Polymath-Low) and MMLU-ProX-Lite validation sets.

## Key Experimental Results

### Main Results

| Dataset | Metric | Base | Selective Trans. | Full Trans. | Translation Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Polymath-Low | Avg Acc | 81.1 | 88.0 | 89.4 | 19.3% |
| MMLU-ProX-Lite | Avg Acc | 72.7 | 74.3 | 76.5 | 20.8% |

**Outstanding performance on low-resource languages**: Swahili (sw) improved from 29.3 → 81.3 on Polymath-Low (86.4% translation usage), and Telugu (te) improved from 69.9 → 77.1 (37.9% translation usage).

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Stage Attribution | U-share dominates | Understanding failure contributes the majority of the multilingual gap; the generation stage contributes very little. |
| Reasoning post-intervention | ≈ 0.95-0.99 | After resolving understanding failures, performance across languages is close to that of the best-performing language. |
| Translation vs. Reasoning | r = 0.951 | Translation capability is strongly positively correlated with multilingual reasoning capability. |
| Early Detection (4096 tokens) | Equal to full trace | Reliable detection can be made without waiting for the full reasoning trace. |
| Non-English targets | Performance drops | Using low-resource languages as translation targets introduces additional understanding failures. |

### Key Findings
- Understanding failure is the **dominant source** of the multilingual reasoning gap, a finding consistent across different model sizes (1.7B-14B) and reasoning difficulty levels (Low/Medium/High).
- Supervised methods (Prober, mmBERT) significantly outperform LLM-based and token probability methods in detecting understanding failures.
- The detector generalizes robustly to unseen languages (French, Marathi, Wolof).
- Approaching the effect of full translation is possible with only approximately 20% of the translation overhead.

## Highlights & Insights
- **Systematic Analysis Framework**: Decomposing multilingual reasoning into three stages and using Shapley decomposition for attribution is a rigorous and generalizable methodology.
- **Insight: "Understanding is the Bottleneck"**: This challenges the intuition that "cross-lingual gaps are mainly due to reasoning ability itself," revealing that the root of the problem lies in input understanding.
- **Strong correlation between translation and reasoning** (r=0.951) provides a clear optimization direction for improving multilingual reasoning.
- **Practicality of Selective Translation**: It significantly enhances low-resource language performance through inference-time intervention without requiring model modifications.
- **Early Detection Discovery**: The ability to make intervention decisions early in the generation process further improves efficiency.

## Limitations & Future Work
- Experiments primarily focused on math and STEM reasoning tasks; applicability to other domains like commonsense reasoning remains unverified.
- Language coverage is 10 languages, not covering all language families; extremely low-resource languages require further verification.
- The analysis focuses on English-dominant reasoning scenarios; models that reason in other languages (e.g., Russian) have not been explored.
- Selective translation depends on external translation systems (GPT-4.1), introducing additional latency and cost.
- Future direction: Directly integrating understanding failure detection and mitigation mechanisms into model training.

## Related Work & Insights
- **vs. Full Translation**: Selective translation achieves approximately 98% of the performance of full translation with only 20% of the overhead.
- **vs. Language-forcing**: Forcing the model to reason in the target language can reduce accuracy or requires expensive training data; the proposed solution is more economical.
- **vs. Representation Editing**: Methods like Zhao et al. (2025) require modifying internal model representations, whereas this method requires zero model changes.
- **vs. Cross-lingual Collapse (Park et al., 2025)**: While previous work mitigates the issue through language-consistency rewards (requiring training), this proposal is a pure inference-time method.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic attribution of multilingual reasoning gaps; Shapley framework and Selective Translation are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive experiments across multiple models, languages, and difficulties, including generalization and early detection.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with logical flow from analysis to detection to mitigation; rich visualizations.
- Value: ⭐⭐⭐⭐ Provides clear guidance for multilingual reasoning research; Selective Translation has high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering](../../AAAI2026/multilingual_mt/mitigating_content_effects_on_reasoning_in_language_models_through_fine-grained_.md)
- [\[ACL 2026\] EMCEE: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context](emcee_improving_multilingual_capability_of_llms_via_bridging_knowledge_and_reaso.md)
- [\[ACL 2026\] Multilingual Language Models Encode Script Over Linguistic Structure](multilingual_language_models_encode_script_over_linguistic_structure.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](language_models_entangle_language_and_culture.md)
- [\[ACL 2026\] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors](evaluating_robustness_of_large_language_models_against_multilingual_typographica.md)

</div>

<!-- RELATED:END -->
