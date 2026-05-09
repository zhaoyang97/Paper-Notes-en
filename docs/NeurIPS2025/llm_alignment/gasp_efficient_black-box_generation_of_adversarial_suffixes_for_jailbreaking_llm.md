---
title: >-
  [Paper Note] GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs
description: >-
  [NeurIPS 2025][LLM Alignment][adversarial suffix] This paper proposes GASP, a framework that trains a dedicated SuffixLLM to generate human-readable adversarial suffixes. It employs Latent Bayesian Optimization (LBO) to efficiently search the continuous embedding space and iteratively fine-tunes the generator via ORPO, achieving high attack success rates in a fully black-box setting while maintaining suffix readability.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - adversarial suffix
  - jailbreak attack
  - Bayesian optimization
  - black-box attack
  - red teaming
date: 2026-05-08
content_hash: c475cb54f5a1e687
---

# GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2411.14133](https://arxiv.org/abs/2411.14133)
**Code**: [https://github.com/TrustMLRG/GASP](https://github.com/TrustMLRG/GASP)
**Area**: LLM Alignment / AI Safety
**Keywords**: adversarial suffix, jailbreak attack, Bayesian optimization, black-box attack, red teaming

## TL;DR
This paper proposes GASP, a framework that trains a dedicated SuffixLLM to generate human-readable adversarial suffixes. It employs Latent Bayesian Optimization (LBO) to efficiently search the continuous embedding space and iteratively fine-tunes the generator via ORPO, achieving high attack success rates in a fully black-box setting while maintaining suffix readability.

## Background & Motivation
**Background**: LLM jailbreak methods fall into three categories — manual heuristics (flexible but not scalable), optimization-based methods (e.g., GCG, which searches in discrete token space but produces unreadable suffixes), and hybrid approaches (e.g., AutoDAN/PAIR, which are computationally expensive with limited generalizability).

**Limitations of Prior Work**:
   - Optimization-based methods such as GCG produce gibberish token sequences that are easily detected by perplexity filters.
   - Most existing methods require white-box access (gradients/logits) and are unsuitable for API-only scenarios.
   - AdvPrompter learns a suffix generator but cannot adapt to a specific TargetLLM and still operates in discrete token space.

**Key Challenge**: Three objectives must be satisfied simultaneously — (a) high attack success rate, (b) generation of human-readable natural language suffixes, and (c) fully black-box operation with high efficiency.

**Key Insight**: Transform discrete token optimization into Bayesian optimization over a continuous latent space. A SuffixLLM encodes suffixes into the continuous space; a Gaussian process models the "attack effectiveness" of suffixes; an acquisition function guides the search; and ORPO preference optimization fine-tunes the SuffixLLM.

**Core Idea**: Performing Bayesian optimization in the SuffixLLM's embedding space to search for adversarial suffixes is substantially more efficient than discrete token search and naturally preserves readability.

## Method

### Overall Architecture
GASP comprises four modules: (A) pre-training the SuffixLLM on the AdvSuffixes dataset; (B) using LBO to efficiently search the latent space for high-quality suffixes, driven by GASPEval scoring feedback; (C) iteratively fine-tuning the SuffixLLM via ORPO preference optimization; and (D) deploying the final SuffixLLM for fast inference to generate adversarial suffixes targeting a specific TargetLLM.

### Key Designs

1. **Latent Bayesian Optimization (LBO) Search**:

    - **Function**: Searches for effective adversarial suffixes in the SuffixLLM's embedding space rather than in discrete token space.
    - **Mechanism**: The SuffixLLM generates a candidate suffix pool → suffixes are encoded into latent vectors → a Gaussian process fits (vector, attack score) pairs → an acquisition function selects the most promising next vector → nearest-neighbor decoding retrieves the corresponding suffix → evaluation → GP update.
    - **Design Motivation**: The continuous space is far smoother than discrete token space; the GP can effectively model the attack effectiveness landscape, and the acquisition function automatically balances exploration and exploitation. This substantially improves search efficiency compared to GCG's gradient-based discrete search.
    - Nearest-neighbor decoding ensures outputs are genuine suffixes from the candidate pool, naturally satisfying the readability constraint.

2. **GASPEval Evaluator**:

    - **Function**: Assesses the harmfulness of TargetLLM responses using 21 binary criteria, covering hate speech, illegal instructions, misinformation, threats, and more.
    - **Mechanism**: An auxiliary LLM scores each criterion on a 0–2 scale; the aggregate score reflects the adversarial quality of the suffix.
    - **Design Motivation**: More fine-grained than simple keyword matching (e.g., "Sorry, I can't...") and more comprehensive than StrongREJECT.
    - Lazy evaluation is adopted: only suffixes selected by LBO are evaluated, avoiding unnecessary computation.

3. **ORPO Iterative Fine-Tuning**:

    - **Function**: Fine-tunes the SuffixLLM via preference optimization based on the quality ranking of suffixes discovered by LBO.
    - **Core formula**: $L_{\text{ORPO}} = \ell_{\text{SFT}}(\phi; x, y_+) + \lambda \cdot \ell_{\text{OR}}(\phi; x, y_+, y_-)$
    - Suffixes with the highest GASPEval scores serve as $y_+$; lower-quality suffixes serve as $y_-$.
    - **Design Motivation**: The SFT component learns to imitate high-quality suffixes; the OR component learns to discriminate between good and poor suffixes. This dual signal accelerates convergence. More efficient than pure SFT and lighter than DPO (no reference policy required).

### Loss & Training
- SuffixLLM backbone: Mistral-7B
- AdvSuffixes dataset: 519 harmful instructions, each paired with multiple readable adversarial suffixes (generated via two-shot prompting of an uncensored LLM)
- 75% for pre-training / 25% for fine-tuning; test set consists of 100 OOD harmful prompts

## Key Experimental Results

### Main Results (ASR@10 / ASR@1, evaluated by GASPEval)

| Method | Mistral-7B | Falcon-7B | LLaMA-3.1-8B | LLaMA-3-8B | LLaMA-2-7B |
|--------|-----------|-----------|-------------|-----------|-----------|
| GCG | -/37 | -/52 | -/6 | -/2 | -/5 |
| AutoDAN | -/69 | -/42 | -/1 | -/62 | -/0 |
| AdvPrompter | 77/55 | 93/52 | 17/4 | 5/0 | 7/1 |
| PAIR | -/64 | -/91 | -/18 | -/9 | -/7 |
| TAP | -/61 | -/98 | -/25 | -/8 | -/8 |
| ICA | -/62 | -/91 | -/59 | -/54 | -/0 |
| **GASP** | **82/64** | **100/86** | **68/11** | **71/6** | **64/9** |

### Ablation Study

| Component | Effect |
|-----------|--------|
| Remove LBO (pre-trained SuffixLLM only) | Significant ASR drop, confirming the critical role of LBO search |
| Remove ORPO (LBO without fine-tuning) | ASR decreases but remains effective; LBO itself contributes substantially |
| Remove pre-training (train from scratch) | Slow convergence and poor performance; pre-trained initialization is important |

### Key Findings
- GASP consistently leads on ASR@10: 100% on Falcon-7B, 82% on Mistral-7B.
- Against strongly aligned models (LLaMA-3/3.1), ASR@1 is lower, but ASR@10 significantly outperforms baselines, demonstrating the effectiveness of a multi-attempt strategy.
- GASP inference is substantially faster than GCG/AutoDAN (generative rather than search-based; a single forward pass suffices).
- Generated suffixes are highly readable with perplexity far below GCG's gibberish outputs.
- Evaluations on closed-source models: GPT-4o-mini achieves 74% ASR@10; Claude-3-Haiku achieves 61%.

## Highlights & Insights
- The idea of **converting discrete optimization to continuous optimization** is elegant: the SuffixLLM's embedding space provides a semantically structured continuous search space amenable to GP modeling, avoiding the combinatorial explosion of discrete token search.
- **The closed-loop alternation of LBO and ORPO**: LBO discovers high-quality suffixes → ORPO fine-tunes the SuffixLLM → the updated SuffixLLM provides a better embedding space and candidate pool → LBO searches more efficiently. This self-reinforcing cycle is the key to GASP's continuous improvement.
- **GASPEval's 21-dimensional evaluation** is more fine-grained than binary "refusal or not" detection and can distinguish varying degrees of harmful output.

## Limitations & Future Work
- ASR@1 remains low (6–11%) against **strongly aligned models** (LLaMA-3/3.1), requiring multiple attempts to succeed.
- The SuffixLLM requires LBO + ORPO adaptation for each TargetLLM, lacking out-of-the-box cross-model transferability.
- The AdvSuffixes pre-training data relies on an uncensored LLM for generation — data construction is constrained if such a model is unavailable.
- The Gaussian process in LBO may face the curse of dimensionality in high-dimensional embedding spaces; the paper provides insufficient detail on the dimensionality reduction strategy employed.
- Experiments are limited to open-source models with 7–8B parameters; performance on models with 70B+ parameters remains unknown.

## Related Work & Insights
- **vs. GCG**: GCG performs greedy coordinate gradient search in discrete token space, producing unreadable suffixes. GASP performs Bayesian optimization in the continuous latent space, producing readable suffixes.
- **vs. AdvPrompter**: AdvPrompter also learns a suffix generator but does not adapt to a specific TargetLLM and operates in discrete space. GASP achieves target-specific adaptation via LBO + ORPO.
- **vs. PAIR/TAP**: These are black-box baselines but slow at inference (each prompt requires multiple rounds of LLM interaction). GASP requires only a single forward pass after training.
- **Implications for defense**: Perplexity filtering is ineffective against GASP (suffixes are readable); more semantically-grounded defenses are needed.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of latent-space Bayesian optimization, generative suffix modeling, and ORPO is an entirely novel design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five open-source and six closed-source models, three evaluation metrics, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear, mathematical notation is rigorous, and experimental tables are informative.
- Value: ⭐⭐⭐⭐⭐ Provides an efficient and scalable tool for LLM red teaming with significant implications for understanding alignment fragility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PolyJuice Makes It Real: Black-Box, Universal Red Teaming for Synthetic Image Detectors](polyjuice_makes_it_real_black-box_universal_red_teaming_for_synthetic_image_dete.md)
- [\[NeurIPS 2025\] Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks](short-length_adversarial_training_helps_llms_defend_long-length_jailbreak_attack.md)
- [\[NeurIPS 2025\] Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency](adjacent_words_divergent_intents_jailbreaking_large_language_models_via_task_con.md)
- [\[NeurIPS 2025\] Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs](attack_via_overfitting_10-shot_benign_fine-tuning_to_jailbreak_llms.md)
- [\[NeurIPS 2025\] Greedy Sampling Is Provably Efficient for RLHF](greedy_sampling_is_provably_efficient_for_rlhf.md)

</div>

<!-- RELATED:END -->
