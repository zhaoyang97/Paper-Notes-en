---
title: >-
  [Paper Note] Adam's Law: Textual Frequency Law on Large Language Models
description: >-
  [ACL 2026][LLM/NLP][textual frequency] This paper proposes the Textual Frequency Law (TFL), which finds that when semantics are equivalent, prompting or fine-tuning LLMs with higher-frequency textual expressions yields better performance. The authors further introduce frequency distillation and curriculum training strategies to exploit this regularity.
tags:
  - ACL 2026
  - LLM/NLP
  - textual frequency
  - paraphrase selection
  - curriculum learning
  - prompt optimization
  - fine-tuning strategy
date: 2026-05-08
content_hash: 27d5fe1808ed00d1
---

# Adam's Law: Textual Frequency Law on Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.02176](https://arxiv.org/abs/2604.02176)
**Code**: [https://github.com/HongyuanLuke/frequencylaw](https://github.com/HongyuanLuke/frequencylaw)
**Area**: LLM/NLP
**Keywords**: textual frequency, paraphrase selection, curriculum learning, prompt optimization, fine-tuning strategy

## TL;DR
This paper proposes the Textual Frequency Law (TFL), which finds that when semantics are equivalent, prompting or fine-tuning LLMs with higher-frequency textual expressions yields better performance. The authors further introduce frequency distillation and curriculum training strategies to exploit this regularity.

## Background & Motivation

**State of the Field**: Large language models have achieved remarkable progress on mathematical reasoning, machine translation, commonsense reasoning, and related tasks. Recent research highlights the critical role of data quality and quantity, yet the "frequency" dimension—how often a given expression appears in the training corpus—has rarely been explored.

**Limitations of Prior Work**: Prior studies have shown that semantically equivalent prompts with different surface forms can produce large variations in LLM output quality, but no clear explanation has been offered for which factors drive this phenomenon. Moreover, when training resources are limited and multiple paraphrases are available, there is no principled guideline for selecting the best training data.

**Root Cause**: LLMs encounter high-frequency expressions far more often during pre-training and should theoretically handle high-frequency inputs more competently, yet existing methods do not systematically exploit this intuition. Furthermore, because most LLM training corpora are proprietary, the exact pre-training frequency of any given sentence is inaccessible.

**Paper Goals**: (1) Verify whether high-frequency textual expressions genuinely outperform low-frequency ones; (2) design a method for estimating sentence frequency without accessing LLM training data; (3) propose a curriculum learning strategy that leverages frequency information to optimize fine-tuning order.

**Starting Point**: Drawing from word-frequency effects in human cognitive research—where high-frequency words elicit stronger neural activation and easier semantic retrieval—the authors hypothesize that the same regularity applies to LLMs: high-frequency expressions are encountered more often during pre-training and are therefore more readily understood by the model.

**Core Idea**: Open-source corpus word frequencies are used to estimate sentence-level frequency, enabling selection of high-frequency paraphrases for prompting and fine-tuning. LLM-generated story completions are then used to distill improved frequency estimates. Finally, fine-tuning data are ordered from lowest to highest frequency for curriculum training.

## Method

### Overall Architecture
The framework consists of three modules: (1) the Textual Frequency Law (TFL), which defines sentence-level frequency computation and guides paraphrase selection; (2) Textual Frequency Distillation (TFD), which leverages LLM-generated text to enhance frequency estimation; and (3) Curriculum Textual Frequency Training (CTFT), which arranges fine-tuning data in order of increasing frequency. The inputs are task data along with multiple paraphrases; the outputs are frequency-optimized prompts or a fine-tuned model.

### Key Designs

1. **Textual Frequency Law (TFL) and Sentence Frequency Estimation**

    - **Function**: Compute a frequency score for a given sentence and select the paraphrase with the highest frequency among semantically equivalent alternatives for prompting or fine-tuning.
    - **Mechanism**: Sentence-level frequency is estimated via the inverse-normalized product of word-level frequencies: $\text{sfreq}(\mathbf{x}, \mathcal{D}) = \sqrt[\mathbb{K}]{\frac{1}{\prod_{k=1}^{\mathbb{K}} \text{wfreq}(\mathbf{x}_k, \mathcal{D})}}$, where $\text{wfreq}$ is obtained from open-source corpora (e.g., Zipf frequencies). This position-independent multiplicative aggregation requires no access to LLM training data.
    - **Design Motivation**: Because most LLM training corpora are not publicly available, and word frequencies are relatively consistent across corpora, approximating a sentence's pre-training frequency using open-source word frequencies is well-motivated.

2. **Textual Frequency Distillation (TFD)**

    - **Function**: Leverage the LLM's own generations to augment the initial frequency estimates and bridge the distributional gap between open-source corpora and the actual pre-training data.
    - **Mechanism**: The LLM performs story completion on training-set texts; the generated outputs form a distillation corpus $\mathcal{D}'$. The resulting frequency estimate $\mathcal{F}_2$ is blended with the original estimate $\mathcal{F}_1$ via weighted combination: $\mathcal{F}(x) = \alpha \mathcal{F}_1(x) + (1 + \zeta \mathbb{1}(\mathcal{F}_1(x)=0)) \beta \mathcal{F}_2(x)$, where the $\zeta$ factor amplifies the distilled estimate when the original frequency is zero.
    - **Design Motivation**: Open-source word frequencies may miss expression patterns the LLM has actually encountered, whereas text generated by the LLM itself more faithfully reflects its internal word-frequency distribution, thereby improving estimation accuracy.

3. **Curriculum Textual Frequency Training (CTFT)**

    - **Function**: Arrange fine-tuning data in ascending order of sentence frequency to achieve superior fine-tuning performance.
    - **Mechanism**: All samples in the training set $\mathcal{T}$ are sorted by $\mathcal{F}(x_n)$ in ascending order for each training epoch. Low-frequency expressions are more diverse and harder to learn; presenting harder examples first and easier ones later follows a principled curriculum.
    - **Design Motivation**: Inspired by curriculum learning—low-frequency data are more linguistically diverse (featuring more unique expressions) and should be trained first to build broader representational capacity, while high-frequency data serve as "easy" samples to consolidate learning thereafter.

### Loss & Training
Fine-tuning is performed with LoRA using the standard language modeling cross-entropy loss. CTFT modifies only the data ordering and leaves the loss function unchanged. Comparative experiments also evaluate the reverse order (high-to-low frequency) and conventional easy-to-hard curriculum learning (sorted by parse-tree depth).

## Key Experimental Results

### Main Results

| Model | Low-Freq Accuracy | High-Freq Accuracy | Gain |
|---|---|---|---|
| GPT-4o-mini (MR) | 0.8266 | 0.8523 | +2.57% |
| DeepSeek-V3 (MR) | 0.8964 | 0.9119 | +1.55% |
| Llama-3.3-70B (MR) | 0.9092 | 0.9295 | +2.03% |
| GPT-4o-mini (CR) | 0.6747 | 0.6974 | +2.27% |
| DeepSeek-V3 (CR) | 0.7043 | 0.7235 | +1.92% |

In machine translation experiments spanning 100 languages, DeepSeek-V3 achieves BLEU improvements in 99/100 languages and GPT-4o-mini in 95/100 languages when high-frequency paraphrases are used.

### Ablation Study

| Configuration | BLEU (kea) | BLEU (kik) | BLEU (pag) | BLEU (lvs) |
|---|---|---|---|---|
| High-freq fine-tuning | **4.48** | **3.22** | **29.73** | **15.91** |
| Low-freq fine-tuning | 3.92 | 2.77 | 28.68 | 14.83 |
| CTFT (low→high) | **4.78** | **3.51** | **30.12** | **16.25** |
| Reverse CTFT (high→low) | 4.21 | 3.05 | 29.15 | 15.44 |
| Conventional curriculum | 4.35 | 3.12 | 29.47 | 15.62 |

### Key Findings
- High-frequency paraphrases outperform low-frequency ones across all models and nearly all languages, validating the generality of TFL.
- TFD further improves frequency estimation quality, boosting tool-calling task accuracy from 84.21% to 87.72%.
- CTFT (low-to-high frequency order) consistently outperforms both the reverse order and conventional curriculum learning, demonstrating that frequency is a more effective data-ordering dimension than syntactic complexity.
- Translation improvements are especially pronounced for low-resource languages, indicating that high-frequency expressions are particularly beneficial for helping LLMs process unfamiliar linguistic inputs.

## Highlights & Insights
- **Textual frequency as a new data quality dimension**: Distinct from conventional dimensions of data quality (clean vs. noisy) and quantity (more vs. less), frequency offers a novel perspective on data selection—choosing higher-frequency expressions when semantics are equivalent. This idea is simple yet effective and can be applied to any prompting scenario at zero additional cost.
- **Using LLM-generated text to estimate training distributions**: The TFD approach is conceptually elegant—story completion serves as an indirect probe of the internal word-frequency distribution of closed-source models, providing a new avenue for understanding and exploiting the training preferences of black-box systems.
- **Low-to-high frequency curriculum learning**: This challenges the conventional "easy-to-hard" curriculum paradigm and proposes a frequency-based data ordering strategy, offering a new guiding principle for training data arrangement.

## Limitations & Future Work
- Sentence frequency estimation via word-frequency products ignores word order and collocation information, which may reduce accuracy for syntactically complex sentences or rare collocations.
- Paraphrase generation and manual annotation are costly; only 56% of GSM8K and 52% of FLORES-200 samples were retained, limiting dataset scale.
- CTFT is validated only with LoRA fine-tuning; full-parameter fine-tuning and larger-scale models remain untested.
- Whether frequency effects are equally significant for reasoning-intensive tasks such as code generation and long-chain reasoning has not been explored.

## Related Work & Insights
- **vs. conventional curriculum learning**: Traditional methods sort data by difficulty (e.g., parse-tree depth); the proposed frequency-based ordering performs better, suggesting that frequency more accurately reflects LLM learning preferences than complexity.
- **vs. data augmentation via paraphrasing**: Prior work typically incorporates all paraphrases for augmentation; this paper demonstrates that only high-frequency paraphrases should be selected, providing a principled selection criterion for paraphrase-based augmentation.
- **vs. prompt engineering**: Prompt optimization typically focuses on semantics and format; this paper reveals frequency as a previously overlooked factor that can serve as an additional signal for prompt selection.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic introduction of textual frequency as a factor in LLM prompting and fine-tuning optimization; the perspective is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four tasks, multiple models, and 100 languages; validation is comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Logically coherent, progressing naturally from the law to estimation, distillation, and curriculum training.
- Value: ⭐⭐⭐⭐ The high-frequency paraphrase selection strategy incurs negligible cost and is immediately applicable; practical utility is high.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)
- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)
- [\[ACL 2026\] CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models](costomcausal-oriented_steering_for_intrinsic_theory-of-mind_alignment_in_large_l.md)
- [\[ACL 2026\] The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)

<!-- RELATED:END -->
