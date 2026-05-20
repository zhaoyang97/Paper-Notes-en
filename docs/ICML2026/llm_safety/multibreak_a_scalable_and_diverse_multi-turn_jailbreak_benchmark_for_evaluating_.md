---
title: >-
  [Paper Note] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety
description: >-
  [ICML 2026][LLM Safety][Multi-turn jailbreak] MultiBreak employs an iterative framework of "active learning + uncertainty-guided rewriting" to expand a multi-turn jailbreak dataset to 10,389 dialogues and 2…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Multi-turn jailbreak"
  - "jailbreak benchmark"
  - "active learning"
  - "uncertainty-guided rewriting"
  - "LLM red teaming"
date: 2026-05-08
content_hash: 05d47cbf0370057c
---

# MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety

**Conference**: ICML 2026  
**arXiv**: [2605.01687](https://arxiv.org/abs/2605.01687)  
**Code**: None  
**Area**: AI Safety / LLM Evaluation  
**Keywords**: Multi-turn jailbreak, jailbreak benchmark, active learning, uncertainty-guided rewriting, LLM red teaming

## TL;DR
MultiBreak employs an iterative framework of "active learning + uncertainty-guided rewriting" to expand a multi-turn jailbreak dataset to 10,389 dialogues and 2,665 unique harmful intents, achieving a diversity of 0.942 that far surpasses previous work. On DeepSeek-R1-7B / GPT-4.1-mini, it improves ASR by 54% / 34.6% over the next-best dataset.

## Background & Motivation

**Background**: The mainstream approach for LLM safety alignment evaluation is to construct jailbreak datasets, prompting LLMs to output prohibited content under adversarial prompts. Single-turn jailbreak datasets (GCG, PAIR, HarmBench, etc.) are mature but disconnected from real user interactions. The field is shifting to multi-turn jailbreaks (Crescendo, MRJ-Agent, CoSafe, MHJ, SafeDialBench, RedQueen), which bypass safety guardrails through gradual escalation or benign setup.

**Limitations of Prior Work**: Existing multi-turn jailbreak benchmarks are either small in scale (CoSafe 1.4k, MHJ 537, SafeDialBench 2k) or rely on template replication (RedQueen inflates 1.4k intents × 40 templates to 56k), making "diversity" a bottleneck. After deduplication with Qwen3-0.6B embeddings, only 76% of samples are truly unique intents, making evaluations sensitive to prompt perturbations and inconsistent across LLMs.

**Key Challenge**: Scaling up while maintaining diversity is inherently contradictory—manual annotation is high-quality but costly; LLM auto-generation is cheap but prone to "mode collapse" (repeatedly generating similar attacks); safety-aligned LLMs also tend to refuse generating harmful content.

**Goal**: (1) Expand the multi-turn jailbreak dataset by an order of magnitude without sacrificing quality; (2) Systematically cover a broader spectrum of harmful intents without template replication; (3) Reveal which categories are safe in single-turn but high-risk in multi-turn settings.

**Key Insight**: The authors reframe "benchmark construction" as pool-based active learning: starting from a large pool of harmful intents, iteratively fine-tune an attack generator → evaluate on multiple victims/judges → use an acquisition function to route samples to accept / rewrite / discard → use an uncertainty-guided rewriter to recover "borderline successful" samples.

**Core Idea**: Use an acquisition function + rewriting to cyclically amplify high-value training signals from "samples the model is uncertain about," resulting in a diverse and high-ASR adversarial prompt set.

## Method

### Overall Architecture
Three-stage pipeline: (1) **Data Diversification** — Aggregate from 5 multi-turn and 9 single-turn datasets, deduplicate using Qwen3-0.6B embeddings, filter out false harmful samples with closed-source victim validation, initializing $|Q_{adv}^{(0)}|=2{,}201$ multi-turn adversarial prompts and $|Q|=3{,}010$ unique harmful intents to form $\mathcal D_0$; (2) **Active Learning Loop** — For each round $t$, the current generator $\text{LLM}_G^{(t)}$ generates multi-turn adversarial prompts (MTAP, 2-6 turns, random length) on the unlabeled pool $\mathcal U^{(t)}$, evaluates ASR / uncertainty / faithfulness on victim set $\mathcal V$ and judge set $\mathcal J$, and splits samples into accept / rewrite / discard using acquisition function $\alpha$; (3) **Uncertainty-Guided Rewrite** — The rewrite bucket is sent to an independent Qwen2.5-7B rewriter for further validation, and successful rewrites are merged into accept. $\mathcal D^{(t+1)} = \mathcal D^{(t)}\cup\mathcal S_{\text{accept}}$, then SFT produces $\text{LLM}_G^{(t+1)}$, repeating for $T$ rounds.

### Key Designs

1. **Three-signal Acquisition Function (exploit + explore + quality filtering)**:

    - **Function**: Determines whether each generated prompt should be accepted, rewritten, or discarded, acting as the "router" of active learning.
    - **Mechanism**: Computes on multi victim × multi judge: (a) **ASR** $\text{ASR}(q_{adv})=\frac{1}{|\mathcal V||\mathcal J|}\sum_V\sum_J J(q_{adv},V(q_{adv}))$ measures attack success rate (exploit); (b) **Uncertainty** $\sigma(q_{adv})=\text{Std}_{V,J} J(q_{adv},V(q_{adv}))$ measures disagreement across victim-judge pairs (explore, identifies "borderline but informative" samples); (c) **Faithfulness** $\text{faith}(q,q_{adv})=\cos(\text{Enc}(q),\text{Enc}(q_{adv}))$ uses Qwen3-0.6B embeddings to prevent semantic drift after rewriting. The three-stage decision $\alpha(q_{adv})$ is Accept (ASR $\ge\tau_h$ and faith $\ge\tau_f$) / Rewrite ($\sigma\ge\tau_\sigma$ and ASR<$\tau_h$ and faith $\ge\tau_f$) / Discard.
    - **Design Motivation**: Selecting by ASR alone leads to generator overfitting a few effective attack patterns (mode collapse); selecting by uncertainty alone introduces low-quality noise. The three-signal combination ensures "stable and faithful samples are used for training / borderline but informative samples are rewritten / others are discarded," forming a natural closed loop.

2. **Generator Ensemble + SFT Instead of Prompting**:

    - **Function**: Overcomes the limitation of safety-aligned LLMs "refusing to generate harmful content" and allows complementary vulnerabilities from different model families.
    - **Mechanism**: Uses LLaMA3-8B-Instruct + Qwen2.5-7B-Instruct (full-parameter SFT) + DeepSeek-Distill-Qwen-14B (LoRA) as the $\text{LLM}_G$ ensemble. Experiments show prompting on Mistral-7B-Instruct yields only 2% ASR, while SFT achieves 25%; prompting also leads to repeated refusals or outputs not faithful to $q$ (see Figure 3 failure modes). The rewriter $\text{LLM}_R$ uses an untuned Qwen2.5-7B, as it only sees $q_{adv}$ and not $q$, thus avoiding safety guardrails.
    - **Design Motivation**: Vulnerabilities exposed by small open-source generators after SFT transfer to closed-source large models (Table 5), demonstrating "small model attacking large model" transferability; multi-family ensemble further reduces generator bias toward a single attack paradigm.

3. **Multi-victim Multi-judge Debiasing + Uncertainty-guided Rewriting**:

    - **Function**: Avoids single-judge consistency bias, turning "successful on some models, failed on others" samples into useful training signals rather than noise.
    - **Mechanism**: Uses LLaMA Guard + GPT-4o-mini as dual judges, plus a keyword-based refusal detector. Samples directly rejected by the detector are immediately discarded; the rest with high $\sigma$ are sent to the rewriter, instructed to "retain harmful intent, clarify ambiguous expressions, and strengthen persuasion/confusion strategies." After rewriting, $\sigma$ drops significantly in each iteration (see Figure 4), "straightening" borderline samples.
    - **Design Motivation**: Single-judge bias is well-documented (Souly 2024, Huang 2025); multi-judge naturally estimates uncertainty. Instead of discarding "low ASR + high $\sigma$" samples, rewriting extracts additional training signals from model disagreement regions, making data expansion more sample-efficient than pure generation.

### Loss & Training

The generator is fine-tuned with standard SFT loss; the rewriter is not trained (instruction prompting); judges use majority voting + hard filtering by the refusal detector; multi-turn length $n\sim\text{Uniform}(2,6)$; total of $T=5$ iterations suffices to push ASR above 50%.

## Key Experimental Results

### Main Results

Table 1 (Dataset Comparison): MultiBreak (2-6 turns), 10,389 samples, 2,665 unique intents, diversity 0.942; CoSafe 1,400/961/0.843, MHJ 537/406/0.810, SafeDialBench 2,037/1,078/0.762, RedQueen 56k (template replication)/656/0.680.

Table 2 (ASR, judge: LG=LLaMA Guard / GPT=GPT-4o-mini):

| Dataset | DeepSeek-7B (LG/GPT) | Qwen3-8B | LLaMA3.1-8B | Gemini-2.5-FL | GPT-4.1-mini |
|---------|----------------------|----------|-------------|---------------|--------------|
| CoSafe @1 | 0.127 / 0.235 | 0.079/0.340 | 0.063/0.456 | 0.059/0.557 | 0.019/0.552 |
| MHJ @1 | 0.293 / … | … | … | … | … |
| **MultiBreak** | Significantly leads (+54% on DeepSeek, +34.6% on GPT-4.1-mini over next-best dataset) |

### Ablation Study

| Configuration | Impact | Notes |
|---------------|--------|-------|
| Full pipeline (5 iter) | ASR 50%+ | Qwen victim + LLaMA generator |
| Initial $\mathcal D_0$ only | 10.77% | 4.47/3.77 pp higher than CoSafe/RedQueen |
| SFT only, no active learning | 25% | Much stronger than prompting but far below active learning |
| Prompting only, no SFT | 2% | Safety-aligned LLM repeatedly refuses |
| Remove rewrite | ASR/diversity both drop | All borderline sample info lost |
| Remove multi victim/judge | Judge bias | ASR no longer stable |

### Key Findings

- Monotonicity of active learning: ASR rises from 10% to 50%+ within 5 rounds, and $\sigma$ drops significantly after each rewrite, indicating the acquisition function effectively "mines informative samples."
- Vulnerabilities exposed by fine-tuned small open-source generators transfer to closed-source large models like GPT-4.1-mini / Gemini-2.5-Flash-Lite.
- Categories seemingly safe in single-turn see significant ASR increases in multi-turn, revealing that LLM safety guardrail weaknesses differ fundamentally between single-turn and multi-turn scenarios.

## Highlights & Insights

- **Reframing dataset construction as active learning**: A natural yet previously unsystematized perspective; the three-way acquisition function (accept/rewrite/discard) is crisp and directly transferable to any "data generation + hard example mining" scenario (e.g., RLHF preference data, code repair, math reasoning data expansion).
- **Uncertainty = rewriting opportunity**: Traditional active learning sends high-uncertainty samples for manual annotation; here, a dedicated rewriting LLM is used and then validated, reducing human cost to zero. This trick is reusable in any scenario needing "model-driven data expansion."
- **Closed-source refusal of harmful content → rewriter does not directly access intent**: After identifying the failure mode where safety-aligned LLMs refuse when seeing $q$, the authors have the rewriter only see $q_{adv}$, bypassing fine-tuning costs with a simple engineering solution.

## Limitations & Future Work

- All labels depend on judge LLMs (LG + GPT-4o-mini) and rule-based refusal detectors, which may miss or misclassify, so final ASR depends on judge quality.
- Experiments are mainly in English; scalability to multilingual multi-turn jailbreaks is untested; safety category distribution is also strongly tied to seed datasets.
- The dataset is double-edged—it can be used for red teaming but also lowers the barrier for constructing large-scale, high-quality jailbreak attacks. The paper discusses responsibility disclosure and access control only briefly.
- Five iterations plus multi-victim evaluation require significant computational resources, raising the bar for reproduction.

## Related Work & Insights

- **vs RedQueen (Jiang 2024)**: RedQueen inflates 1,400 intents × 40 templates to 56k, but token diversity is extremely low; this work uses active learning to truly expand unique intents (2,665 vs 656).
- **vs CoSafe / MHJ / SafeDialBench**: These benchmarks are all at the thousand-sample scale with narrow category coverage; MultiBreak is best-of-all in both scale and diversity score.
- **vs Crescendo / MRJ-Agent**: These are "attack methods," while this work is a "benchmark construction method"—any attack method can be plugged in as a generator in the active learning loop.
- **vs single-turn jailbreak benchmarks (HarmBench, AdvBench, JailbreakBench)**: This is the first systematic comparison of single-turn vs multi-turn ASR, showing many seemingly safe categories are much more vulnerable in multi-turn, providing strong evidence that LLM safety evaluation must upgrade from "single-turn" to "multi-turn."

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of active learning + uncertainty-guided rewriting for jailbreak datasets is a new angle, though components (uncertainty sampling, self-refine) are relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 victims + 2 judges, cross-evaluates with 4 baseline datasets, and analyzes sample efficiency/diversity/category breakdowns.
- Writing Quality: ⭐⭐⭐⭐ Three-stage framework diagram (Figure 2) + Algorithm 1 clearly explain the process, with concise definitions and notation.
- Value: ⭐⭐⭐⭐⭐ Directly useful resource for the LLM safety evaluation community; the "uncertainty-rewrite" method is transferable to any data expansion task.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models](../../ICLR2026/audio_speech/echomind_an_interrelated_multi-level_benchmark_for_evaluating_empathetic_speech_.md)
- [\[ICLR 2026\] SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables](../../ICLR2026/audio_speech/sparta_scalable_and_principled_benchmark_of_tree-structured_multi-hop_qa_over_te.md)
- [\[ICML 2026\] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)
- [\[ICLR 2026\] Incentive-Aligned Multi-Source LLM Summaries](../../ICLR2026/audio_speech/incentive-aligned_multi-source_llm_summaries.md)
- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](../../ACL2026/audio_speech/alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)

</div>

<!-- RELATED:END -->
