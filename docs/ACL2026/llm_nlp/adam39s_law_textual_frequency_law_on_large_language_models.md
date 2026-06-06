---
title: >-
  [Paper Note] Adam's Law: Textual Frequency Law on Large Language Models
description: >-
  [ACL 2026][LLM/NLP][Textual Frequency] This paper proposes the "Textual Frequency Law" (TFL), finding that when semantics are identical…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Textual Frequency"
  - "Paraphrase Selection"
  - "Curriculum Learning"
  - "Prompt Optimization"
  - "Fine-tuning Strategy"
date: 2026-05-08
content_hash: 34907bd4bcccf00f
---

# Adam's Law: Textual Frequency Law on Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.02176](https://arxiv.org/abs/2604.02176)  
**Code**: [https://github.com/HongyuanLuke/frequencylaw](https://github.com/HongyuanLuke/frequencylaw)  
**Area**: LLM/NLP  
**Keywords**: Textual Frequency, Paraphrase Selection, Curriculum Learning, Prompt Optimization, Fine-tuning Strategy

## TL;DR
This paper proposes the "Textual Frequency Law" (TFL), finding that when semantics are identical, using higher-frequency textual expressions to prompt or fine-tune LLMs yields better performance. It further designs frequency distillation and curriculum training strategies to exploit this law.

## Background & Motivation

**Background**: Large Language Models (LLMs) have made significant progress in tasks such as mathematical reasoning, machine translation, and commonsense reasoning. Recent research indicates that the quality and quantity of data are crucial for LLM performance, but the "frequency" dimension of data—the frequency with which a specific expression appears in the training corpus—has seldom been explored.

**Limitations of Prior Work**: Existing research has found that prompts with the same semantics but different phrasing lead to significant variance in LLM output quality, yet there is no clear conclusion explaining the factors driving this phenomenon. Furthermore, when training resources are limited, there is a lack of guiding principles for selecting optimal training data from multiple paraphrases.

**Key Challenge**: LLMs encounter high-frequency expressions more often during pre-training and should theoretically be more proficient at processing high-frequency inputs. However, existing methods do not systematically utilize this intuition. Additionally, as the training data for most LLMs is not public, it is impossible to directly know the frequency of a sentence in pre-training.

**Goal**: (1) Verify whether high-frequency textual expressions are indeed superior to low-frequency ones; (2) Design a method to estimate sentence frequency without access to LLM training data; (3) Propose a curriculum learning strategy that utilizes frequency information to optimize the fine-tuning sequence.

**Key Insight**: Drawing from the word frequency effect in human cognition research (where high-frequency words elicit stronger neural activation and easier semantic retrieval), the authors hypothesize that this law also applies to LLMs—high-frequency expressions are seen more during pre-training and are therefore more easily understood by the model.

**Core Idea**: Use word frequencies from open-source corpora to estimate sentence-level frequency and select high-frequency paraphrases for prompting/fine-tuning; distill frequency estimates through the LLM’s own story completion; and finally perform curriculum fine-tuning by ordering data from low to high frequency.

## Method

### Overall Architecture
The framework consists of three modules: (1) Textual Frequency Law (TFL) defines the calculation of sentence-level frequency and guides paraphrase selection; (2) Textual Frequency Distillation (TFD) enhances frequency estimation through LLM-generated text; (3) Curriculum Textual Frequency Training (CTFT) schedules fine-tuning data according to frequency order. The input consists of task data and its multiple paraphrases, and the output is the frequency-optimized prompt or the fine-tuned model.

### Key Designs

1. **Textual Frequency Law (TFL) and Sentence Frequency Estimation**:

    - Function: Compares a frequency score for a given sentence to select the highest-frequency paraphrase among semantically identical options for prompting or fine-tuning.
    - Mechanism: Sentence-level frequency is estimated via the inverse normalized product of word-level frequencies: $\text{sfreq}(\mathbf{x}, \mathcal{D}) = \sqrt[\mathbb{K}]{\frac{1}{\prod_{k=1}^{\mathbb{K}} \text{wfreq}(\mathbf{x}_k, \mathcal{D})}}$, where $\text{wfreq}$ is obtained using open-source corpora (e.g., Zipf frequency). This is a position-independent multiplicative aggregation that does not require access to LLM training data.
    - Design Motivation: Since most LLM training data is private and word frequencies remain relatively consistent across different corpora, it is reasonable to approximate the pre-training occurrence frequency of a sentence using word frequencies from public corpora.

2. **Textual Frequency Distillation (TFD)**:

    - Function: Leverages LLM generation to enhance the original frequency estimation, compensating for the distributional shift between open-source corpora and actual pre-training data.
    - Mechanism: The LLM is tasked with story completion based on texts in the training set, and the generated text is collected as a distillation corpus $\mathcal{D}'$. The new frequency estimate $\mathcal{F}_2$ is fused with the original estimate $\mathcal{F}_1$ using weighted integration: $\mathcal{F}(x) = \alpha \mathcal{F}_1(x) + (1 + \zeta \mathbb{1}(\mathcal{F}_1(x)=0)) \beta \mathcal{F}_2(x)$, where the weight of the distilled frequency is enhanced by a factor $\zeta$ when the original frequency is zero.
    - Design Motivation: Open-source word frequencies may miss expression patterns actually seen by the LLM, whereas the text generated by the LLM itself better reflects its internal word frequency distribution, thereby improving estimation accuracy.

3. **Curriculum Textual Frequency Training (CTFT)**:

    - Function: Orders fine-tuning data from low to high sentence frequency to achieve better fine-tuning results.
    - Mechanism: All samples in training set $\mathcal{T}$ are sorted in ascending order of $\mathcal{F}(x_n)$ for training in each epoch. Low-frequency expressions are more diverse and harder to learn; the model learns the difficult ones before the easy ones.
    - Design Motivation: Inspired by curriculum learning—low-frequency data is more diverse (unique linguistic expressions) and should be trained first to acquire broader representation capabilities, while high-frequency data serves as "easy" samples for later consolidation.

### Loss & Training
Fine-tuning uses LoRA based on standard language model cross-entropy loss. CTFT only changes the data arrangement order without modifying the loss function itself. Comparative experiments also tested the reverse order (high-to-low frequency) and traditional easy-to-hard curriculum learning (ordered by syntax tree depth).

## Key Experimental Results

### Main Results

| Model | Low-frequency Accuracy | High-frequency Accuracy | Gain |
|------|-----------|-----------|------|
| GPT-4o-mini (MR) | 0.8266 | 0.8523 | +2.57% |
| DeepSeek-V3 (MR) | 0.8964 | 0.9119 | +1.55% |
| Llama-3.3-70B (MR) | 0.9092 | 0.9295 | +2.03% |
| GPT-4o-mini (CR) | 0.6747 | 0.6974 | +2.27% |
| DeepSeek-V3 (CR) | 0.7043 | 0.7235 | +1.92% |

In machine translation experiments (100 languages), using high-frequency paraphrases improved BLEU for 99/100 languages with DeepSeek-V3 and 95/100 languages with GPT-4o-mini.

### Ablation Study

| Configuration | BLEU (kea) | BLEU (kik) | BLEU (pag) | BLEU (lvs) |
|------|-----------|-----------|-----------|-----------|
| High-frequency Fine-tuning | **4.48** | **3.22** | **29.73** | **15.91** |
| Low-frequency Fine-tuning | 3.92 | 2.77 | 28.68 | 14.83 |
| CTFT (Low $\rightarrow$ High) | **4.78** | **3.51** | **30.12** | **16.25** |
| Reverse CTFT (High $\rightarrow$ Low) | 4.21 | 3.05 | 29.15 | 15.44 |
| Traditional Curriculum Learning | 4.35 | 3.12 | 29.47 | 15.62 |

### Key Findings
- High-frequency paraphrases outperform low-frequency paraphrases across all models and nearly all languages, verifying the universality of TFL.
- TFD further improves frequency estimation quality, increasing performance from 84.21% to 87.72% in tool-calling tasks.
- CTFT (low-to-high frequency order) consistently outperforms reverse order and traditional curriculum learning, suggesting that frequency is a better data ranking dimension than syntactic complexity.
- Improvement in low-resource language translation is particularly significant, indicating that high-frequency expressions help LLMs understand inputs in unfamiliar languages more effectively.

## Highlights & Insights
- **Textual Frequency as a New Data Quality Dimension**: Unlike traditional dimensions of data quality (clean/noisy) and quantity (more/less), frequency provides a new perspective for data selection—choosing high frequency when semantics are identical. This idea is simple but effective and can be applied to any prompting scenario at zero cost.
- **Estimating Training Distribution Using LLM Generation**: The idea behind TFD is ingenious—indirectly "peeking" into the internal word frequency distribution of closed-source models through story completion provides a new way to understand and utilize the training preferences of black-box models.
- **Low-to-High Curriculum Learning**: This challenges the traditional "easy-to-hard" curriculum learning paradigm and proposes a ranking strategy based on the frequency dimension, providing a new guiding principle for training data arrangement.

## Limitations & Future Work
- Sentence frequency estimation via the product of word frequencies is an approximation that ignores word order and collocation information, which may be inaccurate in scenarios with complex syntax or rare collocations.
- The cost of paraphrase generation and manual annotation is high (only 56% of GSM8K and 52% of FLORES-200 samples were retained), limiting the size of the dataset.
- CTFT has currently only been validated on LoRA fine-tuning; full-parameter fine-tuning or larger-scale models have not been tested.
- It remains unexplored whether the frequency effect is equally significant in reasoning tasks such as code generation or long-chain reasoning.

## Related Work & Insights
- **vs Traditional Curriculum Learning**: Traditional methods rank by difficulty (e.g., syntax tree depth), whereas this paper’s frequency-based ranking performs better, indicating frequency reflects LLM learning preferences more accurately than complexity.
- **vs Data Augmentation (Paraphrasing)**: Previous data augmentation via paraphrasing typically includes all samples; this paper suggests selecting high-frequency paraphrases, providing a selection criterion for paraphrase augmentation.
- **vs Prompt Engineering**: Prompt optimization usually focuses on semantics and format; this paper reveals the overlooked factor of frequency, which can serve as an additional signal for prompt selection.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically introduces textual frequency into LLM prompt and fine-tuning optimization for the first time with a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 tasks, multiple models, and 100 languages, providing comprehensive validation.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, progressing from law to estimation, distillation, and curriculum training.
- Value: ⭐⭐⭐⭐ The high-frequency paraphrase selection strategy is extremely low-cost and immediately applicable, offering high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Repeated Sequences Reveal Gaps between Large Language Models and Natural Language](repeated_sequences_reveal_gaps_between_large_language_models_and_natural_languag.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)
- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)
- [\[ACL 2026\] MoRI: Learning Motivation-Grounded Reasoning for Scientific Ideation in Large Language Models](mori_learning_motivation-grounded_reasoning_for_scientific_ideation_in_large_lan.md)

</div>

<!-- RELATED:END -->
