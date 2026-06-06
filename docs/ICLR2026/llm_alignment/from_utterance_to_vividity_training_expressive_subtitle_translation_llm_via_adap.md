---
title: >-
  [Paper Note] From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization
description: >-
  [ICLR 2026][LLM Alignment][subtitle translation] This paper proposes ALPO (Adaptive Local Preference Optimization) for training expressive subtitle translation LLMs. Three empirical findings motivate the design: (1) subt…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "subtitle translation"
  - "preference optimization"
  - "LLM-as-Judge"
  - "paraphrastic translation"
  - "process supervision"
date: 2026-05-08
content_hash: 1835e566172a95a1
---

# From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization

**Conference**: ICLR 2026  
**arXiv**: [2602.01068](https://arxiv.org/abs/2602.01068)  
**Code**: [GitHub](https://github.com/CcQunResearch/ALPO)  
**Area**: LLM Alignment / NLP  
**Keywords**: subtitle translation, preference optimization, LLM-as-Judge, paraphrastic translation, process supervision

## TL;DR
This paper proposes ALPO (Adaptive Local Preference Optimization) for training expressive subtitle translation LLMs. Three empirical findings motivate the design: (1) subtitle translation exhibits the lowest back-translation consistency, indicating the highest degree of paraphrase; (2) reasoning-type LLMs (R1/GPT-5 Thinking) produce more expressive paraphrases than chat-type LLMs (GPT-4o/Qwen-Max); (3) a 14B model used as a translation evaluator achieves Spearman correlation $\geq 0.82$ with human judgments, qualifying it as a low-cost reward model. Building on these findings, the paper proposes a fine-grained, process-supervised preference alignment method operating at the sentence-segment level (with adaptive weighting, dynamic beta, and prefix mixing). A 14B model trained with ALPO surpasses GPT-4o and DeepSeek-R1 in vividness across multiple subtitle translation directions.

## Background & Motivation

**Background**: LLMs have approached human-level performance on general translation, yet domain-specific translation (legal, medical, subtitle) remains notably lacking in customization. Subtitle translation requires localized paraphrasing to convey the atmosphere, emotion, and tone of the source, whereas LLMs tend toward literal translation.

**Limitations of Prior Work**: (1) LLM translations achieve high accuracy but lack expressiveness and vividness; (2) subtitle translation requires fine-grained alignment at the sentence-segment level, while PPO/DPO are outcome-supervised and optimize over the complete output—too coarse-grained and prone to gradient dilution; (3) no established evaluation framework or training data exists for subtitle translation.

**Key Challenge**: Subtitle translation inputs consist of multiple lines of subtitles with inter-line contextual dependencies, yet each line requires independent fine-grained preference alignment—a "local preference optimization" problem to which existing methods such as DPO do not directly apply.

**Goal**: (a) Verify whether LLMs can reliably evaluate subtitle translation quality as a substitute for costly human annotation; (b) design a fine-grained preference optimization method enabling LLMs to acquire paraphrastic translation capabilities; (c) construct a multi-directional subtitle parallel corpus.

**Key Insight**: Three empirical findings drive the method design: (1) back-translation consistency is lowest for subtitle translation, indicating the highest degree of paraphrase; (2) reasoning-type LLMs (R1/GPT-5 Thinking) outperform chat-type LLMs (GPT-4o/Qwen-Max) in paraphrastic ability; (3) Spearman correlation between a 14B evaluator model and human annotators is $\geq 0.82$, supporting its use as a low-cost reward model.

**Core Idea**: Achieve fine-grained vividness alignment for subtitle translation via sentence-segment-level sampling, LLM scoring, and adaptively weighted process-supervised DPO.

## Method

### Overall Architecture
Input: multi-directional subtitle parallel corpus (MuSC dataset). Output: a highly expressive subtitle translation LLM.

Two-stage training: SFT (80% of data) → ALPO preference alignment (20% of data).

### Key Designs

1. **LLM-as-Judge Validation and Empirical Study**:

    - Function: Validate the reliability of LLMs as subtitle translation evaluators.
    - Mechanism: 500 subtitle lines × 10 translation variants are scored by both humans and LLMs (0–100); Spearman correlation is computed. Results show Qwen3-14B achieves $\rho \geq 0.82$ with human evaluators across all directions; Bland-Altman analysis reveals negligible systematic bias.
    - Design Motivation: Automatic preference data construction is infeasible if LLM scores are unreliable. Experiments demonstrate that a 14B model suffices as an efficient reward model.

2. **ALPO Sampling Strategy (Sentence-Segment-Level Sampling)**:

    - Function: Generate multiple candidate translations and scores for each subtitle line.
    - Mechanism: For an input of $n$ subtitle lines, $k=15$ candidate translations are sampled line by line (using the best previously selected translation as a prefix). After deduplication, a human reference translation is added; Qwen3-14B scores all candidates to produce a candidate set $\mathcal{T}_i$ and score sequence $\mathcal{E}_i$ for each line. One of the top-3 candidates is randomly selected as the chosen response, and the third-from-last is selected as the rejected response (avoiding trivial comparisons with the worst candidate).
    - Design Motivation: Since each subtitle line depends on context, the chosen translation from preceding lines is used as a prefix during sampling to maintain contextual consistency. Line-by-line sampling and scoring realize process supervision rather than outcome supervision.

3. **ALPO Adaptive Alignment Loss**:

    - Function: Fine-grained preference optimization at the sentence-segment level.
    - Mechanism: Each subtitle line $s_i$ is assigned an adaptive weight $w(s_i) = \mathbf{1}(s_i) \cdot \delta(s_i)$. The gating function $\mathbf{1}(s_i)$ is set to 0 (skipping the line) when the number of candidates is $\leq 3$ or the score gap is $\leq 5$. The importance score $\delta(s_i) = |\mathcal{T}_i| / \sum |\mathcal{T}_j|$ assigns higher weight to lines with more diverse candidates. Dynamic $\beta_i$ is normalized according to the reward gap. The prefix mixing strategy uses the chosen prefix with probability $\lambda$ (linearly increasing from 0.2 to 0.6) and otherwise samples randomly, mitigating exposure bias.
    - Design Motivation: Standard DPO optimizing over the complete output suffers from gradient dilution caused by lines that require no alignment. ALPO optimizes each line independently, and adaptive weighting concentrates training on lines with room for improvement. Dynamic $\beta_i$ prevents training instability when reward gaps vary widely.

### Loss & Training
- SFT: Qwen2.5-14B fine-tuned on 80% of the MuSC dataset.
- ALPO loss: Bradley-Terry preference alignment with segment-wise adaptive weighted summation.
- Prefix mixing ratio $\lambda$ increases linearly from 0.2 to 0.6.

## Key Experimental Results

### Main Results: Multi-Dimensional Translation Quality Evaluation (LLM-as-Judge)

| Model | en→zh Acc | Nat | Viv | zh→en Acc | Nat | Viv |
|-------|-----------|-----|-----|-----------|-----|-----|
| Google Translate | 84.2 | 79.7 | 54.4 | 79.8 | 66.3 | 50.2 |
| GPT-4o | 89.3 | 82.3 | 59.8 | 88.5 | 83.0 | 64.6 |
| DeepSeek-R1 | 90.5 | 85.7 | 70.8 | 88.5 | 85.6 | 73.5 |
| Qwen2.5-14B SFT | 86.4 | 82.0 | 59.1 | 85.2 | 80.1 | 54.8 |
| **Qwen2.5-14B ALPO** | **90.6** | **84.3** | **76.6** | **88.3** | **86.8** | **81.7** |

### Ablation Study: Human Evaluation (Win Rate, en→zh)

| Comparison | Accuracy | Naturalness | Vividness | Comprehensive |
|------------|----------|-------------|-----------|---------------|
| ALPO vs Gold Reference | 29:49:22 | 28:50:22 | 32:42:26 | 31:46:23 |
| ALPO vs SFT | 26:50:24 | 31:48:21 | **38:41:21** | 37:43:20 |
| ALPO vs GPT-4o | 22:54:24 | 20:57:23 | **29:51:20** | 26:54:23 |
| ALPO vs DeepSeek-R1 | 22:55:23 | 19:57:24 | 22:58:20 | 20:59:21 |

### Key Findings
- **ALPO substantially improves vividness**: The zh→en vividness score increases from 54.8 (SFT) to 81.7 (+26.9), surpassing DeepSeek-R1's 73.5.
- **Accuracy and naturalness improve simultaneously**: Vividness gains are not achieved at the cost of accuracy; all three dimensions improve concurrently.
- **A 14B model outperforms GPT-4o and DeepSeek-R1**: ALPO leads in vividness across all translation directions, demonstrating effective utilization of domain data.
- **Reasoning LLMs produce more expressive paraphrases**: DeepSeek-R1's vividness scores are markedly higher than those of chat models such as GPT-4o, validating that inference-time scaling enhances translation quality.
- **Human evaluation is consistent**: Human evaluation results align with LLM-as-Judge assessments, validating the reliability of the evaluation framework.

## Highlights & Insights
- **Process supervision vs. outcome supervision for translation alignment**: Conventional DPO scores the entire translation output, whereas ALPO aligns each line independently. This "local preference optimization" paradigm is transferable to any task requiring fine-grained segment-level alignment, such as dialogue generation or code generation.
- **The finding that reasoning LLMs produce more expressive paraphrases** carries important implications: inference-time scaling (thinking) benefits not only reasoning but also creative translation, likely because paraphrasing requires more deliberate strategy.
- **Gating combined with importance weighting** prevents gradient dilution by simple lines, concentrating optimization on difficult lines with room for improvement.
- **The prefix mixing strategy** effectively and simply mitigates exposure bias.

## Limitations & Future Work
- The MuSC dataset is sourced from the Youku platform, so domain coverage may be skewed toward film and entertainment.
- Although the evaluator model correlates strongly with human judgments, it may have blind spots for culturally specific expressions.
- The ALPO sampling stage (15 candidates per line) incurs non-trivial computational overhead.
- Validation is currently limited to the 14B scale; the performance of larger and smaller models remains unexplored.

## Related Work & Insights
- **vs. DPO/SimPO**: Outcome-supervised methods optimize over the complete output at too coarse a granularity. ALPO achieves segment-level process-supervised alignment.
- **vs. VideoDubber**: The only closely related subtitle translation work, but it addresses only length control and does not focus on expressiveness.
- **vs. RLHF**: ALPO entirely avoids reward model training and the instability of RL, realizing alignment through LLM-as-Judge combined with a DPO variant.

## Rating
- Novelty: ⭐⭐⭐⭐ — The local preference optimization paradigm is novel, and the empirical finding that reasoning LLMs produce stronger paraphrases is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Six translation directions, combined LLM and human evaluation, with comprehensive empirical study.
- Writing Quality: ⭐⭐⭐⭐ — Empirically driven method design with clear logical flow.
- Value: ⭐⭐⭐⭐ — Provides important reference for domain-specific translation LLMs and fine-grained preference alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Better Literary Translation: A Multi-Aspect Data Generation and LLM Training Approach](../../ACL2026/llm_alignment/better_literary_translation_a_multi-aspect_data_generation_and_llm_training_appr.md)
- [\[AAAI 2026\] AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment](../../AAAI2026/llm_alignment/amapo_adaptive_margin-attached_preference_optimization_for_l.md)
- [\[ICLR 2026\] AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization](avere_improving_audiovisual_emotion_reasoning_with_preference_optimization.md)
- [\[ICLR 2026\] Mitigating Mismatch within Reference-based Preference Optimization](mitigating_mismatch_within_reference-based_preference_optimization.md)
- [\[ICLR 2026\] Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](uni-dpo_a_unified_paradigm_for_dynamic_preference_optimization_of_llms.md)

</div>

<!-- RELATED:END -->
