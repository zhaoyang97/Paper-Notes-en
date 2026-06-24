---
title: >-
  [Paper Note] Blessing of Multilinguality: A Systematic Analysis of Multilingual In-Context Learning
description: >-
  [ACL 2025 (Findings)][Multilingual & Machine Translation][multilingual ICL] Through a systematic analysis of multilingual ICL strategies, this study reveals that mixing demonstrations of various high-resource languages (HRLs) in the prompt consistently outperforms purely English demonstrations, yielding particularly significant improvements on low-resource languages (LRLs) (e.g., an 8.9% to 12.6% average LRL accuracy gain on Llama 3.1). Intriguingly…
tags:
  - "ACL 2025 (Findings)"
  - "Multilingual & Machine Translation"
  - "multilingual ICL"
  - "cross-lingual transfer"
  - "low-resource languages"
  - "in-context learning"
  - "prompting strategies"
date: 2026-05-08
content_hash: 9741c7e28726adc9
---

# Blessing of Multilinguality: A Systematic Analysis of Multilingual In-Context Learning

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2502.11364](https://arxiv.org/abs/2502.11364)  
**Code**: [GitHub](https://github.com/yileitu/multilingual_icl)  
**Area**: Multilingual Translation  
**Keywords**: multilingual ICL, cross-lingual transfer, low-resource languages, in-context learning, prompting strategies

## TL;DR

Through a systematic analysis of multilingual ICL strategies, this study reveals that mixing demonstrations of various high-resource languages (HRLs) in the prompt consistently outperforms purely English demonstrations, yielding particularly significant improvements on low-resource languages (LRLs) (e.g., an 8.9% to 12.6% average LRL accuracy gain on Llama 3.1). Intriguingly, even merely appending context-irrelevant non-English sentences to the prompt yields measurable gains, revealing the phenomenon that "multilingual exposure itself is effective."

## Background & Motivation

**Background**: Multilingual LLMs perform acceptably on high-resource languages (HRLs), sometimes even approaching English levels, but lag significantly behind on low-resource languages (LRLs). In-Context Learning (ICL) represents a primary approach for enhancing cross-lingual performance.

**Limitations of Prior Work**: Two common ICL strategies suffer from critical limitations: (a) translating queries to English and then performing English ICL introduces substantial translation loss, and high-quality translation systems are unavailable for extremely low-resource languages; (b) utilizing demonstrations in the target language is practically unfeasible due to data scarcity.

**Key Challenge**: While Shi et al. (2023) observed the effectiveness of mixing HRL demonstrations, a systematic understanding of *why* it works remains lacking—is the multilingual information itself beneficial, or are specific language combinations more critical?

**Goal**: Systematically analyze the underlying mechanism of multilingual ICL—when it is effective, why it works, and which languages are most beneficial.

**Key Insight**: Carefully design controlled experiments—varying only the presentation language of semantically equivalent demonstrations—and introduce Context-Irrelevant Sentences (CIS) to decouple the "multilingual exposure effect" from the "semantic information effect."

**Core Idea**: Multilingual exposure by itself can activate the cross-lingual capabilities of MLLMs, and mixing HRL demonstrations stands as the most robust and practically feasible cross-lingual ICL strategy currently available.

## Method

### Overall Architecture

Four ICL paradigms are systematically compared across eight MLLMs on four multilingual benchmarks (MGSM, XCOPA, XL-WiC, and XQuAD), complemented by CIS ablation experiments to isolate the effects of multilingual exposure.

### Key Designs

**1. Controlled Comparison of Four ICL Paradigms**

| Paradigm | Description | Feasibility |
|------|------|--------|
| English | 6-shot entirely English demonstrations | Always feasible |
| Monolingual | 6-shot using a single non-English HRL (e.g., Chinese/Japanese) | Requires HRL data |
| **Multilingual** | 6-shot randomly mixed from a list of HRLs | **Recommended strategy** |
| Native | 6-shot using the target language (ideal upper bound) | Typically unfeasible for LRLs |

Core control variable: Demonstrations of the same index are semantically equivalent across all languages, with only the presentation language varied.

**2. Context-Irrelevant Sentences (CIS) Ablation Study**

- Prepending a **completely task-irrelevant non-English sentence** (sourced from FLORES-101) to each English demonstration.
- Design Motivation: Decouple the two factors of "semantic information" and "language exposure."
- CIS Variants: CIS-Fr / CIS-Ja / CIS-Zh / CIS-Multi.
- Key Finding: Merely appending irrelevant non-English texts is sufficient to improve LRL accuracy.

**3. Exceptional Effectiveness of Non-Latin Script HRLs**

- Chinese and Japanese deliver the best cross-lingual transfer effects when utilized as Monolingual demonstrations.
- Chinese outperforms English in 20 out of 30 LRL pairwise comparisons.
- Probable cause: They prompt the model to project inputs into a more "language-agnostic" representation space.

### Experimental Setup

- **Eight MLLMs**: Llama3/3.1-8B, Qwen2/2.5-7B, Mistral-NeMo-12B, Aya-Expanse-8B, GPT-3.5, GPT-4o-mini
- **6-shot** demonstrations, greedy decoding
- **Statistical Testing**: McNemar's test (continuity-corrected version, Edwards 1948), reporting p-value significance levels

## Key Experimental Results

### Main Results: LRL Average Accuracy (Representative Models)

| Model | ICL Paradigm | MGSM | XL-WiC | XCOPA |
|------|---------|------|--------|-------|
| Llama3.1-8B | English | 57.10 | 44.42 | 55.91 |
| | **Multilingual** | **66.00** (+8.9\*\*\*) | **57.05** (+12.6\*\*\*) | **66.11** (+10.2\*\*\*) |
| | Native | 68.50 (+11.4\*\*\*) | 62.88 (+18.5\*\*\*) | 71.63 (+15.7\*\*\*) |
| Qwen2-7B | English | 43.70 | 48.46 | 62.29 |
| | **Multilingual** | **47.50** (+3.8\*\*) | **56.28** (+7.8\*\*\*) | **63.83** (+1.5\*) |
| | Native | 55.40 (+11.7\*\*\*) | 57.76 (+9.3\*\*\*) | 67.63 (+5.3\*\*\*) |
| GPT3.5-turbo | English | 44.20 | 53.72 | 63.43 |
| | **Multilingual** | **51.00** (+6.8\*\*\*) | **55.90** (+2.2\*\*) | 62.71 (-0.7) |

### CIS Ablation: Effects of Irrelevant Non-English Sentences (Llama3.1-8B LRL Avg)

| Setup | MGSM | XL-WiC | XCOPA | XQuAD |
|------|------|--------|-------|-------|
| English + CIS-En (baseline) | 55.90 | 47.88 | 55.46 | 68.45 |
| English + CIS-Fr | 52.10 | 52.82\*\*\* | 59.40\*\*\* | 68.95 |
| English + CIS-Ja | 58.80 | 55.96\*\*\* | 59.86\*\*\* | 68.90 |
| English + CIS-Zh | 55.00 | 54.68\*\*\* | 64.66\*\*\* | 69.10 |
| English + CIS-Multi | **62.50\*\*\*** | **56.03\*\*\*** | **64.74\*\*\*** | **69.60\*\*\*** |

### Monolingual Paradigm Comparisons

- Chinese Monolingual outperforms English in 20 out of 30 LRL comparisons.
- Japanese also frequently outperforms English.
- Multilingual (mixed) is more robust than any single HRL, outperforming Chinese Monolingual in 23 out of 30 sets.

### Key Findings

1. **Multilingual ICL outperforms English in 23 out of 30 evaluation groups**, with the vast majority of improvements reaching statistical significance.
2. **The Native paradigm performs best but remains impractical**—outperforming Multilingual in 27 out of 30 cases, yet acquisition of LRL data is notoriously difficult.
3. **Non-Latin script HRLs (Chinese/Japanese) are exceptionally effective in driving cross-lingual transfer**.
4. **Merely introducing irrelevant non-English sentences in the prompt boosts LRL performance**, indicating that multilingual exposure itself has an activation effect.
5. **Neurons activated by Multilingual overlap closest with those activated by Native**, providing a mechanistic explanation for their comparable performance.

## Highlights & Insights

1. **The discovery that "multilingual exposure itself is effective" is highly surprising**—elegantly demonstrated by the CIS experiments, this is not merely an empirical finding but an in-depth insight into the internal cross-lingual mechanisms of MLLMs.
2. **The experimental design is exceptionally rigorous**: utilizing semantically equivalent demonstrations to control for content variables, and executing McNemar's tests to guarantee statistical significance.
3. **High practical value**: The Multilingual ICL strategy does not require LRL data, needing only a mixture of HRL demonstrations, making it both simple and highly effective.
4. **Neuron overlap analysis** provides a mechanistic explanation for multilingual ICL, moving beyond simple performance comparisons.
5. **Clear recommendations for language selection**: Prioritize non-Latin script HRLs (Chinese/Japanese), as multilingual mixtures outperform single-language settings.

## Limitations & Future Work

1. **Limited model scale (~8B)**: The study does not verify whether the multilingual ICL effect remains significant in larger models (e.g., 70B+).
2. **Restricted task types**: The four benchmarks cover reasoning, common sense, word sense disambiguation, and QA, but do not touch upon generative tasks such as translation or summarization.
3. **Pre-defined HRL list** depends on the language distribution profiles of Llama 2/PaLM, which may not translate perfectly to newer models.
4. **Noise control in CIS experiments**: FLORES-101 sentences are constrained to 10–15 words; different sentence lengths might yield different effects.
5. **Lack of deep explanation for "why non-Latin scripts are more effective"**: The neuron overlap analysis serves as a preliminary exploration, and causal relationships remain unclear.

## Related Work & Insights

- Extends the findings of **Shi et al. (2023)**: scaling the effectiveness of "Multilingual ICL" from PaLM/Codex to six open-source MLLMs and two commercial models.
- Extends the insights of **Turc et al. (2021)**: non-English languages are not only highly effective during pre-training and fine-tuning, but also within prompting paradigms.
- **Paradigmatic value of the CIS experiments**: By inserting irrelevant content to decouple the two effects, this ablation setup is highly generalizable to other prompt engineering studies.
- **Implications for practical multilingual deployment**: In LRL scenarios, rather than expending resources to acquire target-language demonstrations, practitioners can directly employ mixed HRL demonstrations.

## Rating

- Novelty: ⭐⭐⭐⭐ — The CIS ablation experiments are cleverly designed, and the main discovery ("multilingual exposure itself is effective") is highly impactful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Extremely comprehensive coverage spanning eight models, four datasets, four or more ICL paradigms, statistical tests, and neuron analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear diagrams, logically presented progressive experiments, and highly persuasive conclusions.
- Value: ⭐⭐⭐⭐ — Direct practical relevance for deploying multilingual LLMs, offering a simple yet highly efficient Multilingual ICL strategy.

| ICL Paradigm | MGSM (Llama3.1) | XL-WiC (Llama3.1) | XCOPA (Llama3.1) |
|---------|-----------------|-------------------|-----------------|
| English | 57.10 | 44.42 | 55.91 |
| Multilingual | **66.00** (+8.9%***) | **57.05** (+12.6%***) | **66.11** (+10.2%***) |
| Native (Upper Bound) | **68.50** (+11.4%***) | **62.88** (+18.5%***) | **71.63** (+15.7%***) |

### Ablation: Effects of Irrelevant Multilingual Exposure

| Configuration | LRL Acc Change | Description |
|------|------------|------|
| English-only demos | baseline | Baseline |
| + Irrelevant Chinese/Japanese sentences | +2~5% | Pure language exposure is also beneficial |
| Multilingual demos (full) | +8~13% | Dual activation of both semantics and language |

### Key Findings
- **Multilingual > English in 23 out of 30 cases**—a pervasive phenomenon across MLLMs and datasets.
- **Non-Latin script HRLs (Chinese/Japanese) are particularly effective**—likely activating more language-agnostic representations.
- **Mixed HRLs are more robust than a single HRL**—preventing bias toward any single language.
- **Irrelevant foreign language texts also yield gains**—moving beyond the semantic level, possibly by activating cross-lingual attention patterns.
- **Native mode performs best but is impractical**—multilingual mode stands as the best feasible alternative.

## Highlights & Insights
- **"Irrelevant foreign languages also help" is the most striking discovery**: suggesting a "language-switching activation" mechanism within MLLMs—encountering non-English text triggers a transition to more general cross-lingual processing modes. This finding provides crucial implications for understanding MLLM inner mechanics.
- **Practical value for LRLs**: A zero-cost improvement—simply mixing HRL demonstrations into the prompt.
- **Exquisitely controlled experiments**: Semantically equivalent parallel demonstrations + McNemar's test +踩 irrelevant language ablation.

## Limitations & Future Work
- **Limited model scale**: Restricting to the 7B–12B range; does this still hold for 70B+ models? Larger models' English capabilities may already be sufficiently robust.
- **Only four benchmarks evaluated**: Generative tasks (e.g., translation, summarization) are not covered, where language matching might be of greater importance.
- **Mechanistic interpretation is missing**: Why is multilingual exposure effective? Mechanistic interpretability research is still required (e.g., investigating shifts in attention patterns or representation space probing).
- **Subjective HRL list definitions**: Different models are pre-trained on different datasets, which means the definition of "high-resource" might vary.
- **Demonstration quality not considered**: All demonstrations are assumed to be of high-quality; how would noisy or poorly translated demonstrations affect performance?
- **Fixed at 6-shot**: How does the gain curve of multilingual ICL fluctuate under different shot counts?
- **Optimal HRL combinations unexplored**: Does there exist a selection strategy for a "best HRL set" (e.g., typologically diverse language groupings)?

## Related Work & Insights
- **vs. Shi et al. (2023)**: They were the first to identify that multilingual ICL is effective on PaLM; this paper systematically verifies it across eight MLLMs and introduces irrelevant language ablation setups.
- **vs. Translate-Test (Ahuja et al., 2023)**: Translate-test strategies are unfeasible for extremely low-resource languages, whereas multilingual ICL does not rely on intermediate translation steps.

## Rating
- Novelty: ⭐⭐⭐⭐ The discovery that "irrelevant foreign languages also help" is novel, though multilingual ICL itself is not a new concept.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 MLLMs × 4 benchmarks × 4 modes + multiple ablations + statistical testing.
- Writing Quality: ⭐⭐⭐⭐ Exquisitely designed control experiments with clear, well-supported conclusions.
- Value: ⭐⭐⭐⭐ Offers direct guidance for multilingual LLM deployment, showing maximum benefits in LRL scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning](grammamt_improving_machine_translation_with_grammar-informed_in-context_learning.md)
- [\[ACL 2025\] Comparative Analysis of Multilingual Hate Speech Detection](comparative_analysis_of_multilingual_hate_speech_detection.md)
- [\[ACL 2025\] Code-Switching Curriculum Learning for Multilingual Transfer in LLMs](code-switching_curriculum_learning_for_multilingual_transfer_in_llms.md)
- [\[NeurIPS 2025\] How Data Mixing Shapes In-Context Learning: Asymptotic Equivalence for Transformers with MLPs](../../NeurIPS2025/multilingual_mt/how_data_mixing_shapes_in-context_learning_asymptotic_equivalence_for_transforme.md)
- [\[ACL 2025\] Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)

</div>

<!-- RELATED:END -->
