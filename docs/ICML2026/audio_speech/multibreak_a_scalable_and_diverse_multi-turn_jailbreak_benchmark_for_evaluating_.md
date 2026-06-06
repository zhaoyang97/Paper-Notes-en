---
title: >-
  [Paper Note] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety
description: >-
  [ICML 2026][Audio & Speech][Multi-turn jailbreak] MultiBreak utilizes an iterative framework of "active learning + uncertainty-guided rewriting" to expand a multi-turn jailbreak dataset to 10,389 dialogues and 2…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Multi-turn jailbreak"
  - "jailbreak benchmark"
  - "active learning"
  - "uncertainty-guided"
  - "LLM red-teaming"
date: 2026-05-08
content_hash: debc3aba01e4b7e7
---

# MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety

**Conference**: ICML 2026  
**arXiv**: [2605.01687](https://arxiv.org/abs/2605.01687)  
**Code**: None  
**Area**: AI Safety / LLM Evaluation  
**Keywords**: Multi-turn jailbreak, jailbreak benchmark, active learning, uncertainty-guided, LLM red-teaming

## TL;DR
MultiBreak utilizes an iterative framework of "active learning + uncertainty-guided rewriting" to expand a multi-turn jailbreak dataset to 10,389 dialogues and 2,665 independent harmful intentions. Its diversity score of 0.942 significantly outperforms previous works, and it improves ASR by 54% and 34.6% on DeepSeek-R1-7B and GPT-4.1-mini, respectively, compared to the second-best dataset.

## Background & Motivation

**Background**: The mainstream of LLM safety alignment evaluation involves constructing jailbreak datasets to force LLMs to output prohibited content under adversarial prompts. While single-turn jailbreaks (GCG, PAIR, HarmBench, etc.) are relatively mature, they are disjointed from real-world user interactions. The academic community has begun shifting toward multi-turn jailbreaks (Crescendo, MRJ-Agent, CoSafe, MHJ, SafeDialBench, RedQueen), which bypass safety guardrails through gradual escalation or benign foreshadowing.

**Limitations of Prior Work**: Existing multi-turn jailbreak benchmarks are either small in scale (CoSafe 1.4k, MHJ 537, SafeDialBench 2k) or rely on template replication (RedQueen forces 1.4k intentions $\times$ 40 templates to reach 56k). Both methods make "diversity" a primary bottleneck. After deduplication using Qwen3-0.6B embeddings, at best only 76% are truly independent intentions, making evaluations susceptible to minor prompt perturbations and leading to inconsistent results across different LLMs.

**Key Challenge**: There is an inherent contradiction between scaling and maintaining diversity. Manual annotation is high-quality but expensive; automatic generation by LLMs is cheap but prone to "mode collapse" (repeatedly generating similar attacks). Simultaneously, safety-aligned LLMs frequently refuse to generate harmful content.

**Goal**: (1) Scale the multi-turn jailbreak dataset by an order of magnitude without compromising quality; (2) Systematically cover a broader taxonomy of harmful intentions in a way that avoids template replication; (3) Reveal which categories are secure in single-turn settings but high-risk in multi-turn scenarios.

**Key Insight**: The authors reformulate benchmark construction as pool-based active learning: starting from a large pool of harmful intentions, iteratively fine-tuning an attack generator $\to$ evaluating on multiple victims/judges $\to$ using an acquisition function to categorize samples into accept / rewrite / discard $\to$ using an uncertainty-guided rewriter to recover "marginal success" samples.

**Core Idea**: Utilize an acquisition function and rewriting to amplify high-value training signals from samples where the model is "uncertain," thereby obtaining an adversarial prompt set that is both diverse and possesses high ASR.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Data Diversification** — Aggregating from 5 multi-turn and 9 single-turn existing datasets, performing deduplication with Qwen3-0.6B embeddings, and filtering false harmfuls via closed-source victim validation to initialize primary sets $|Q_{adv}^{(0)}|=2{,}201$ multi-turn prompts and $|Q|=3{,}010$ independent intentions $\mathcal D_0$; (2) **Active Learning Loop** — For each iteration $t$, the current generator $\text{LLM}_G^{(t)}$ generates multi-turn adversarial prompts (MTAP, random length 2-6) on the unlabeled pool $\mathcal U^{(t)}$, which are evaluated on victim set $\mathcal V$ and judge set $\mathcal J$ for ASR, uncertainty, and faithfulness. Samples are partitioned into accept / rewrite / discard via acquisition function $\alpha$; (3) **Uncertainty-Guided Rewrite** — The rewrite bin is sent to an independent Qwen2.5-7B rewriter for validation; successful attempts are merged into accept. $\mathcal D^{(t+1)} = \mathcal D^{(t)}\cup\mathcal S_{\text{accept}}$ is used to SFT the next generator $\text{LLM}_G^{(t+1)}$, aggregating results over $T$ iterations.

### Key Designs

1.  **Three-Signal Acquisition Function (Exploit + Explore + Quality Filter)**:
    - **Function**: Acts as the "router" for the active learning process, deciding whether a generated prompt should be accepted, rewritten, or discarded.
    - **Mechanism**: Calculates signals across multiple victim $\times$ judge pairs: (a) **ASR** $\text{ASR}(q_{adv})=\frac{1}{|\mathcal V||\mathcal J|}\sum_V\sum_J J(q_{adv},V(q_{adv}))$ to measure stable attack success (exploit); (b) **Uncertainty** $\sigma(q_{adv})=\text{Std}_{V,J} J(q_{adv},V(q_{adv}))$ to measure disagreement across different victim-judge pairs (explore, targeting "marginal but informative" samples); (c) **Faithfulness** $\text{faith}(q,q_{adv})=\cos(\text{Enc}(q),\text{Enc}(q_{adv}))$ using Qwen3-0.6B embeddings to prevent semantic drift. The decision $\alpha(q_{adv})$ is: Accept (ASR $\ge\tau_h$ and faith $\ge\tau_f$) / Rewrite ($\sigma\ge\tau_\sigma$ and ASR < $\tau_h$ and faith $\ge\tau_f$) / Discard.
    - **Design Motivation**: Selecting by ASR alone causes the generator to overfit to a few effective patterns (mode collapse), while uncertainty alone introduces low-quality noise. Combining the three signals ensures that stable, high-fidelity samples are used for training, while marginal, informative samples are refined via rewriting.

2.  **Generator Ensemble + SFT instead of Prompting**:
    - **Function**: Overcomes the limitation of safety-aligned LLMs "refusing to generate harmful content" and allows complementary vulnerability exposure from different model families.
    - **Mechanism**: Employs an ensemble of LLaMA3-8B-Instruct + Qwen2.5-7B-Instruct (Full SFT) + DeepSeek-Distill-Qwen-14B (LoRA) as $\text{LLM}_G$. Empirical results show prompting Mistral-7B-Instruct yields only 2% ASR, whereas SFT achieves 25%. Furthermore, prompting leads to frequent refusals or unfaithful content. Note that the rewriter $\text{LLM}_R$ uses an un-tuned Qwen2.5-7B because it only processes $q_{adv}$ and does not directly see the intent $q$, avoiding safety triggers.
    - **Design Motivation**: Vulnerabilities exposed by small open-source generators after SFT migrate to large closed-source models (Table 5). The multi-family ensemble further reduces the risk of the generator biasing toward a single attack paradigm.

3.  **Multi-victim Multi-judge De-biasing + Uncertainty-Guided Rewriting**:
    - **Function**: Eliminates systematic bias from a single judge and converts samples that "succeed on some models but fail on others" into usable training signals rather than noise.
    - **Mechanism**: Utilizes LLaMA Guard + GPT-4o-mini as dual judges along with a keyword-based refusal detector. Samples caught by the refusal detector are discarded. Remaining samples with high $\sigma$ are sent to the rewriter with instructions to "retain harmful intent, clarify ambiguity, and strengthen persuasion/obfuscation." $\sigma$ decreases significantly across iterations after rewriting (Fig 4).
    - **Design Motivation**: Single judges have proven biases. Multiple judges provide a natural estimate of uncertainty. Rewriting rather than discarding "low ASR + high $\sigma$" samples extracts additional signals from decision boundaries, proving more sample-efficient than generating entirely new data.

### Loss & Training
The generator is fine-tuned using standard SFT loss; the rewriter is not trained (instruction prompting); judges use majority voting + hard filtering via the refusal detector; multi-turn length $n\sim\text{Uniform}(2,6)$; the total $T=5$ iterations are sufficient to push ASR above 50%.

## Key Experimental Results

### Main Results
Table 1 (Dataset Comparison): MultiBreak features 2-6 turns, 10,389 samples, 2,665 independent intentions, and 0.942 diversity. Comparatively: CoSafe 1,400/961/0.843, MHJ 537/406/0.810, SafeDialBench 2,037/1,078/0.762, RedQueen 56k(template)/656/0.680.

Table 2 (ASR, judge: LG=LLaMA Guard / GPT=GPT-4o-mini):

| Dataset | DeepSeek-7B (LG/GPT) | Qwen3-8B | LLaMA3.1-8B | Gemini-2.5-FL | GPT-4.1-mini |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CoSafe @1 | 0.127 / 0.235 | 0.079/0.340 | 0.063/0.456 | 0.059/0.557 | 0.019/0.552 |
| MHJ @1 | 0.293 / … | … | … | … | … |
| **MultiBreak** | **Significant Lead** | (+54% on DeepSeek, +34.6% on GPT-4.1-mini relative to second best) | | | |

### Ablation Study

| Configuration | Impact | Description |
| :--- | :--- | :--- |
| Full pipeline (5 iter) | ASR 50%+ | Qwen victim + LLaMA generator |
| Only initial $\mathcal D_0$ | 10.77% | Higher than CoSafe/RedQueen by 4.47/3.77 pp |
| Only SFT w/o AL | 25% | Better than prompting but far below AL |
| Only prompting w/o SFT | 2% | Safety aligned LLMs repeatedly refuse |
| w/o rewrite | ASR/Diversity Drop | Informative marginal samples are lost |
| w/o multi-victim/judge | Judge bias | ASR becomes unstable |

### Key Findings
- **Iterative Monotonicity**: ASR rose from 10% to over 50% within 5 rounds, and $\sigma$ decreased significantly after each rewrite round, demonstrating that the acquisition function successfully "mines informative samples."
- **Transferability**: Vulnerabilities exposed by attack generators fine-tuned on small open-source models transfer to large closed-source models like GPT-4.1-mini and Gemini-2.5-Flash-Lite.
- **Single-Turn vs. Multi-Turn**: Harmful categories that appear safe in single-turn sessions show significantly higher ASR in multi-turn sessions, indicating that multi-turn safety weaknesses are distinct from single-turn ones.

## Highlights & Insights
- **Dataset Construction as Active Learning**: A natural yet previously un-systematized perspective. The three-stage acquisition function (accept/rewrite/discard) is extensible to any "data generation + hard case utilization" scenario (e.g., RLHF preference data, code repair).
- **Uncertainty = Rewrite Opportunity**: While traditional active learning sends high-uncertainty samples to humans, this work uses "rewriting followed by validation," reducing human cost to zero. This trick is valuable for any automated data expansion task.
- **Closed-source Refusal Bypass**: By identifying the failure mode where "Safety LLMs refuse upon seeing $q$," the authors let the rewriter only process $q_{adv}$, bypassing safety guardrails and avoiding fine-tuning costs.

## Limitations & Future Work
- Labels rely entirely on judge LLMs (LG + GPT-4o-mini) and rule-based detectors, which may have false positives/negatives. Final ASR figures are dependent on judge quality.
- Experiments were primarily conducted in English; the scalability of multilingual multi-turn jailbreaks remains unverified. The safety category distribution is strongly tied to the selected seed datasets.
- The dataset is a double-edged sword—useful for safety research but lowers the barrier for constructing large-scale high-quality attacks. Discussion on responsible disclosure is limited.
- 5 iterations with multi-victim evaluations require significant computational resources.

## Related Work & Insights
- **vs RedQueen (Jiang 2024)**: RedQueen inflates 1,400 intentions to 56k via 40 templates, resulting in low token diversity; MultiBreak uses active learning to expand unique intentions (2,665 vs 656).
- **vs CoSafe / MHJ / SafeDialBench**: These benchmarks are scale-limited (thousands) with narrow category coverage; MultiBreak is best-in-class in both scale and diversity.
- **vs Crescendo / MRJ-Agent**: These are attack methods; MultiBreak is a benchmark construction method that can integrate any attack method as a generator.
- **vs Single-turn Benchmarks**: This work provides the first systematic evidence that many safe-looking categories in single-turn settings are highly vulnerable in multi-turn settings, necessitating an upgrade in safety evaluation paradigms.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Active learning + uncertainty rewriting for jailbreak datasets is a fresh perspective; components are mature.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 5 victims and 2 judges, evaluates against 4 baseline datasets, and includes multi-angle analysis of efficiency and diversity.
- **Writing Quality**: ⭐⭐⭐⭐ The three-stage framework (Fig 2) and Algorithm 1 clarify the process.
- **Value**: ⭐⭐⭐⭐⭐ A direct resource for the safety evaluation community; the "uncertainty-rewrite" method is applicable to various data expansion tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](medmosaic_a_challenging_large_scale_benchmark_of_diverse_medical_audio.md)
- [\[ICLR 2026\] SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables](../../ICLR2026/audio_speech/sparta_scalable_and_principled_benchmark_of_tree-structured_multi-hop_qa_over_te.md)
- [\[ICLR 2026\] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models](../../ICLR2026/audio_speech/echomind_an_interrelated_multi-level_benchmark_for_evaluating_empathetic_speech_.md)
- [\[ICML 2026\] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)
- [\[ICLR 2026\] Incentive-Aligned Multi-Source LLM Summaries](../../ICLR2026/audio_speech/incentive-aligned_multi-source_llm_summaries.md)

</div>

<!-- RELATED:END -->
