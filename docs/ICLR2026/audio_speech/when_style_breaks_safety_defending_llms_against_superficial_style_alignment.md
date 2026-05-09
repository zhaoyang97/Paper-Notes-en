---
title: >-
  [Paper Note] When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment
description: >-
  [ICLR 2026][Audio & Speech][LLM safety] This paper identifies that Attack Success Rates (ASR) in LLM jailbreak benchmarks are artificially inflated by semantically irrelevant style patterns (e.g., "create a list of"), a phenomenon observed in nearly all of 36 evaluated LLMs. Superficial style alignment fine-tuning further exacerbates this risk. The paper proposes SafeStyle — a defense that mitigates this risk via style-augmented safety training data.
tags:
  - ICLR 2026
  - "Audio & Speech"
  - LLM safety
  - jailbreak attacks
  - style alignment
  - ASR inflation
  - safety defense
date: 2026-05-08
content_hash: 614fb67193b75600
---

# When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment

**Conference**: ICLR 2026
**arXiv**: [2506.07452](https://arxiv.org/abs/2506.07452)
**Code**: [https://github.com/xiaoyuxin1002/SafeStyle](https://github.com/xiaoyuxin1002/SafeStyle)
**Area**: Audio & Speech
**Keywords**: LLM safety, jailbreak attacks, style alignment, ASR inflation, safety defense

## TL;DR
This paper identifies that Attack Success Rates (ASR) in LLM jailbreak benchmarks are artificially inflated by semantically irrelevant style patterns (e.g., "create a list of"), a phenomenon observed in nearly all of 36 evaluated LLMs. Superficial style alignment fine-tuning further exacerbates this risk. The paper proposes SafeStyle — a defense that mitigates this risk via style-augmented safety training data.

## Background & Motivation

**Background**: LLM alignment efforts enable models to refuse malicious requests. Jailbreak attacks exploit string transformations to increase Attack Success Rate (ASR).

**Limitations of Prior Work**: Queries in jailbreak benchmarks frequently contain semantically irrelevant style patterns (e.g., "create a list of" in "create a list of chemical warfare agents"), which independently inflate ASR. Existing safety defenses do not account for the impact of style alignment.

**Key Challenge**: Style patterns are pervasive in benign instructions (e.g., "create a list of healthy snacks"), causing LLMs to learn to comply with stylistic requests — a tendency that is then exploited when the same styles appear in malicious queries.

**Core Idea**: ASR inflation = ASR(styled query) − ASR(malicious intent only). SafeStyle = safety training data augmented to match the style distribution of fine-tuning data.

## Method

### Overall Architecture
The study proceeds in three progressive stages: (1) quantifying ASR inflation in existing jailbreak benchmarks; (2) demonstrating through controlled experiments that superficial style alignment exacerbates safety risks; and (3) proposing the SafeStyle defense strategy.

### Key Designs

1. **ASR Inflation Quantification**:

    - GPT-4o is used to extract core malicious intent from 2,134 jailbreak queries by removing style patterns.
    - DeBERTa-NLI validates semantic equivalence before and after extraction (retaining only samples with exact semantic match).
    - ASR differences between original queries (with style) and pure malicious intent (without style) are compared across 36 LLMs.
    - Result: 32 out of 36 models exhibit significant ASR inflation (paired t-test $p = 0.0002$).

2. **Attention Mechanism Analysis**:

    - Attention weights are aggregated across all heads and layers to compute each LLM's relative attention difference between style tokens and malicious-intent tokens.
    - ASR inflation is found to be significantly positively correlated with relative attention to style tokens (Spearman $\rho = 0.456$, $p = 6 \times 10^{-3}$).
    - Furthermore, inflated style patterns exhibit significantly higher bigram overlap frequency in alignment training data (e.g., Tulu-3, OLMo SFT data).

3. **Superficial Style Alignment Experiment**:

    - 1,000 instruction–response pairs are constructed across 6 style variants (original / de-styled / list-prefix / list-suffix / poem-prefix / poem-suffix).
    - Llama-3.1-8B-Instruct is fine-tuned separately on each variant and evaluated on same-style / cross-style jailbreak ASR.
    - **ASR rises sharply when training and test styles match**, and deteriorates as the proportion of same-style data increases.

4. **SafeStyle Defense**:

    - A small amount of safety training data (from Bianchi et al. 2024) is mixed into the fine-tuning data.
    - Key innovation: the safety data is style-augmented to match the style distribution of the fine-tuning data (e.g., list-style fine-tuning → list-style safety refusal examples).
    - As few as 50 style-matched safety samples suffice to effectively balance safety and style adaptation.

### Loss & Training
Standard SFT fine-tuning (full fine-tuning, 2 epochs, lr = 5e-6, batch size 128) with style-matched safety data mixing.

## Key Experimental Results

### Main Results

| Finding | Data |
|---------|------|
| 32 out of 36 LLMs exhibit ASR inflation | paired t-test $p = 0.0002$ |
| All 7 benchmarks cause inflation | SorryBench and MedSafetyBench affect the most models |
| Mistral series shows the most severe inflation | Gemma/Llama are relatively resistant |

### Ablation Study

| Configuration | ASR (list style) | ASR (poem style) | Notes |
|---------------|-----------------|-----------------|-------|
| Original instruction fine-tuning (diverse) | Medium | Medium | Baseline, diverse styles |
| List-style fine-tuning (100%) | **Highest** | Medium | ASR rises sharply under same-style attacks |
| Poem-style fine-tuning (100%) | Medium | **Highest** | ASR rises sharply under same-style attacks |
| List fine-tuning (50% + 50% de-styled) | Lower | Medium | Mixing de-styled data alleviates the issue |
| + SafeStyle (style-matched safety data) | **Lowest** | **Lowest** | Defense is effective |
| + Vanilla safety data (no style matching) | Medium-low | Medium-low | Limited effectiveness |
| + PTST (inference-time safety prompt) | Medium | Medium | Inference-time intervention alone is insufficient |
| + SPPFT (frozen safety layers) | Medium-low | Medium-low | Partially effective |

### Key Findings
- **Decoupling style from malicious intent**: ASR drops significantly after style removal (paired t-test $p = 0.0002$).
- **Same-style fine-tuning → same-style jailbreak**: List-style fine-tuning causes a sharp increase in ASR for list-styled malicious queries, with the effect becoming significant after only 0.4 epochs.
- **SafeStyle is consistently effective**: Across 3 LLMs (Qwen2.5-3B, Llama-3.1-8B, gemma-3-12b) × 6 styles × 2 real-world datasets (Dolly-15K, Alpaca-52K), SafeStyle outperforms all 5 baselines.
- **Style position has minimal effect**: ASR trends for prefix vs. suffix styles are nearly identical.
- **Only 50 safety samples needed**: SafeStyle achieves strong results with as few as 50 style-matched safety examples, at minimal cost.

## Highlights & Insights
- **Redefining ASR**: ASR figures reported by existing benchmarks are systematically inflated by style patterns.
- **Safety training data should match deployment style** — a simple yet previously overlooked insight.

## Limitations & Future Work
- Style pattern extraction relies on GPT-4o few-shot prompting, which may miss certain implicit styles (e.g., rhetorical devices, syntactic preferences).
- SafeStyle requires knowledge of the fine-tuning data's style distribution, which may not be available in open deployment scenarios.
- Only six styles are tested (list, poem, news, legal, Shakespearean, code); a broader style space (e.g., colloquial, academic) remains to be explored.
- Safety data sources are fixed (Bianchi et al.); larger and more diverse safety datasets may further improve effectiveness.
- The style–safety interaction during RLHF/DPO post-training is not analyzed (this paper considers SFT only).

## Related Work & Insights
- **vs. Bianchi et al. (Vanilla safety data)**: Safety data without style augmentation is far less effective than SafeStyle, confirming that style matching is the key factor.
- **vs. SPPFT (frozen safety layers)**: The layer-freezing strategy fails under certain styles, as safety knowledge is distributed across multiple layers.
- **vs. Constrained (restricting initial tokens)**: Restricting initial tokens alone is insufficient to defend against full-style jailbreaks.
- **Insight**: Safety alignment should be treated as a process that **co-evolves with deployment style**, rather than a one-time fixed intervention.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The ASR inflation concept is entirely novel and significant; the causal analysis of style–safety interactions is rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 36 LLMs × 7 benchmarks + attention analysis + 6-style fine-tuning + 3 models × 5 baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from discovery to analysis to defense is complete and self-consistent.
- Value: ⭐⭐⭐⭐⭐ Profound implications for LLM safety evaluation standards and alignment practices.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] The Devil behind the Mask: An Emergent Safety Vulnerability of Diffusion LLMs](the_devil_behind_the_mask_an_emergent_safety_vulnerability_of_diffusion_llms.md)
- [\[ACL 2026\] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](../../ACL2026/audio_speech/robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)
- [\[ICLR 2026\] Knowing When to Quit: Probabilistic Early Exits for Speech Separation](knowing_when_to_quit_probabilistic_early_exits_for_speech_separation.md)
- [\[ICLR 2026\] When and Where to Reset Matters for Long-Term Test-Time Adaptation](when_and_where_to_reset_matters_for_long-term_test-time_adaptation.md)
- [\[ACL 2026\] TellWhisper: Tell Whisper Who Speaks When](../../ACL2026/audio_speech/tellwhisper_tell_whisper_who_speaks_when.md)

<!-- RELATED:END -->
