---
title: >-
  [Paper Note] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection
description: >-
  [ACL 2026][Social Computing][Dynamic benchmark] LiveFact upgrades fake news detection from static binary classification to a dynamic reasoning benchmark updated monthly with time-sliced evidence. It evaluates both factua…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Dynamic benchmark"
  - "time-awareness"
  - "benchmark contamination"
  - "cognitive humility"
  - "Fog of War"
date: 2026-05-08
content_hash: a1b018a99bb3abf8
---

# LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection

**Conference**: ACL 2026  
**arXiv**: [2604.04815](https://arxiv.org/abs/2604.04815)  
**Code**: https://github.com/bebxy/livefact  
**Area**: Social Computing / Fake News Detection / LLM Evaluation  
**Keywords**: Dynamic benchmark, time-awareness, benchmark contamination, cognitive humility, Fog of War

## TL;DR
LiveFact upgrades fake news detection from static binary classification to a dynamic reasoning benchmark updated monthly with time-sliced evidence. It evaluates both factual judgment and "cognitive humility" (knowing when to admit ignorance) through a dual Classification + Inference mode, while explicitly monitoring benchmark contamination via SSA entity replacement.

## Background & Motivation
**Background**: LLMs have shifted fake news detection from "features + classifier" to "complex reasoning based on multi-hop evidence." However, evaluation typically relies on static datasets like LIAR, FEVER, or FakeNewsNet, where models are given complete evidence to output a Real/Fake label.

**Limitations of Prior Work**: First, static data is often reused in pre-training corpora, leading to severe Benchmark Data Contamination (BDC), where LLMs might "memorize answers" rather than reason. Second, the "god-view" setting (providing all evidence at once) deviates from the real world, where information is incomplete and evolves over time. Third, it is difficult to distinguish whether high performance stems from genuine factual understanding or "confident guessing."

**Key Challenge**: The static nature of evaluation (snapshots) fundamentally mismatches the dynamic processes of continuous LLM pre-training and news generation. Consequently, higher leaderboard scores may reflect better memorization rather than superior reasoning.

**Goal**: To construct a continuously updated evaluation system that simulates the "Fog of War" and quantifies BDC risks, ensuring scores reflect both (a) reasoning capability and (b) the ability to admit "I don't know" when evidence is insufficient.

**Key Insight**: Evidence for each news item is sliced into three time windows ($E^{(-3)}$, $E^{(0)}$, $E^{(+3)}$) relative to the event date $T$. This forces the model to respond under varying information densities. A dual-mode comparison between Classification (absolute fact) and Inference (judgment based on current evidence) is used to differentiate reasoning from guessing.

**Core Idea**: Reshape static classification benchmarks into dynamic, time-aware, and anti-contamination reasoning benchmarks through monthly updates, time-sliced evidence, dual-mode evaluation, and entity-replacement-based contamination monitoring.

## Method

### Overall Architecture
LiveFact employs a five-stage monthly pipeline and a dual-mode evaluation:
1. **Event Crawling**: Daily 00:00 GMT scraping of World news via Google News API (e.g., 737 unique events in Nov 2025).
2. **Temporal Evidence Construction**: Anchored to the headline date $T$, evidence is collected across $\delta \in \{-3, 0, +3\}$, totaling 25,064 entries.
3. **Claim and Context Generation**: o4-mini processes events and evidence to generate neutral contexts and three-label claims (Real/Fake/Ambiguous), totaling 4,392 records.
4. **Human Review**: Three rounds of independent verification by the authors. For the Inference mode, ground truths are dynamically adjusted based on whether the "current evidence is sufficient for judgment."
5. **BDC Monitoring**: Utilizing Qwen3-235B-A22B under the SSA framework to perform entity replacement (e.g., Trump → Wannetta), creating a parallel "shifted" dataset.

Formalization: For each claim $c_i$, the LLM $f_\theta$ receives the triple $(c_i, E_i^{(\delta)}, k_i)$ and outputs $\hat y_i^{(\delta)} \in \{\text{Real}, \text{Fake}, \text{Ambiguous}\}$. Acc and Macro-F1 are calculated for both modes.

### Key Designs

1. **Time-Sliced Evidence (Fog of War)**:
    - **Function**: Simulates the incremental emergence of information using three levels of evidence: $E^{(-3)}$ (3 days before), $E^{(0)}$ (day of), and $E^{(+3)}$ (3 days after).
    - **Mechanism**: Empirical analysis shows evidence density peaks at $T \pm 48{\sim}72$h. Windows of $\pm 7$ or $\pm 15$ days show diminishing returns, while $\pm 1$ day often misses initial reporting. Earlier windows contain fewer "determinable" claims; in Inference mode, the proportion of Ambiguous labels rises to 85% at $\delta = -3$.
    - **Design Motivation**: By separating complete and incomplete evidence, the benchmark can independently evaluate whether a model can resist guessing in an information vacuum.

2. **Dual-Mode Evaluation (Classification + Inference)**:
    - **Function**: Provides two sets of ground truth for the same claim: CLS for absolute factuality (time-invariant) and INF for whether the current evidence supports the conclusion (time-dependent).
    - **Mechanism**: At $\delta = -3$, most INF labels are changed to Ambiguous. A high CLS score at $\delta = -3$ suggests "hallucinatory confidence" unless the model also performs well on INF by correctly identifying insufficient information.
    - **Design Motivation**: Performance on CLS alone can be inflated by parametric memory. The "Reasoning Gap = INF Acc − CLS Acc" is used to distinguish "overconfidence" from "cognitive humility."

3. **SSA Entity Replacement + Overturn Rate (BDC Monitoring)**:
    - **Function**: Uses Entity Shift (via Qwen3-235B-A22B) to replace named entities with fictitious names of the same structure, creating a shifted dataset $(c_i', E_i'^{(\delta)}, k_i')$.
    - **Mechanism**: Defines the Overturn Rate as $\text{OTR} = \frac{1}{N} \sum_i \mathbb{1}[\hat y_i^{(\delta)} \neq \hat y_i'^{(\delta)}]$. The SSA Factor is calculated as $\text{SSA Factor} = \Delta \times \text{OTR} \times 100$, where $\Delta = \text{Metric} - \text{Metric}_{\text{shift}}$.
    - **Design Motivation**: Quantifies BDC risk as a traceable monthly metric, avoiding "preference leakage" from evaluating OpenAI models with other OpenAI models.

### Loss & Training
LiveFact is an evaluation benchmark; no training is performed. Evaluation settings: `TEMPERATURE=0.0`, `TOP_P=1.0`, `MAX_NEW_TOKENS=128` (extended to 1024 for reasoning models like Kimi-K2-Thinking or GPT-OSS). Outputs are forced into `[[LABEL]]` format for parsing.

## Key Experimental Results

### Main Results
Comprehensive scores of 18 LLMs on the Nov 2025 dataset (selected from Table 3):

| Model | Acc$_0^{cls}$ | Acc$_{-3}^{inf}$ | Acc$_{+3}^{cls}$ | Avg |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-235B-A22B-Instruct-2507 | 79.76 | 66.67 | 82.08 | **72.40** |
| gpt-oss-120b⋆ | 79.94 | 62.23 | 81.81 | 72.13 |
| gpt-5.1-2025-11-13 | 78.60 | 68.44 | 81.01 | 72.02 |
| gpt-5.2-2025-12-11 | 76.34 | 80.71 | 77.32 | 71.52 |
| Qwen3-30B-A3B-Instruct-2507 | 75.05 | 64.55 | 77.00 | 69.46 |
| gpt-4o-2024-08-06 | 72.29 | 74.61 | 73.98 | 67.11 |
| DeepSeek-V3.1 | 64.44 | 78.03 | 63.73 | 61.48 |
| Llama-3.1-70B (base) | 33.45 | 7.90 | 33.47 | 22.16 |

Notable findings: The open-source MoE flagship Qwen3-235B-A22B outperformed closed-source models like GPT-5.1. Pure dense base models (e.g., Llama-3.1-70B) failed significantly due to poor instruction following.

### Ablation Study (Reasoning Gap: INF Acc − CLS Acc at $\delta = -3$)

| Model Type | Representative Model | Reasoning Gap | Behavior |
| :--- | :--- | :--- | :--- |
| Uncertainty Aware | Llama-3.1-8B-Instruct | +38% | Correctly says Ambiguous when evidence is lacking |
| Uncertainty Aware | Qwen3-32B | +37% | Same as above |
| Overconfident (Instruct) | Llama-3.3-70B-Instruct | ~ −20% | Forced guessing of Real/Fake |
| Overconfident (Instruct) | Qwen3-4B-Instruct | Neg / ~0 | Same as above |
| Format-Failed (Base) | Llama-3.1-70B | Negative | Non-compliant format, near random |

Cost analysis: Qwen3-30B-A3B-Instruct is approx. 14× cheaper than GPT-5.2 ($0.64 vs $9.27 per round) while scoring only 3 percentage points lower on average.

### Key Findings
- High CLS scores at $\delta = -3$ indicate a "hallucination pressure test" rather than strong reasoning; models that guess under uncertainty fail this metric.
- MoE architectures (Qwen3-235B, DeepSeek-V3.1) systematically outperform dense models in knowledge-retrieval + reasoning tasks.
- "Thinking Mode" models (Kimi-K2, GPT-OSS) fail under low `token_limit` but rebound to top-tier performance at 1024 tokens, suggesting reasoning is structural, not optional.
- Base models fail primarily due to formatting constraints; instruction alignment is a prerequisite for this benchmark.

## Highlights & Insights
- Breaking the "static benchmark" assumption by providing a sustainable monthly update scheme. Any task involving time-sensitive facts should consider this dynamic approach.
- The **Reasoning Gap** provides a clear scalar for "cognitive humility," more informative than Acc alone, and transferable to QA or coding evaluation.
- The **SSA Factor** quantifies BDC risk directly on the leaderboard, offering a more practical alternative to post-hoc contamination audits.
- The cost efficiency of MoE medium-sized models suggests they are optimal for high-frequency tasks like real-time fake news detection.

## Limitations & Future Work
- **Language**: Currently lacks coverage for non-English sources and localized disinformation patterns.
- **Modality**: Limited to text; missing contemporary fake news mainstays like Deepfakes or edited videos.
- **Scalability**: Human review and Ambiguous label determination remain bottlenecks. Scaling will require semi-automation with calibrated judge models.
- **Constraints**: SSA Factor interpretation requires caution as entity replacement can occasionally break commonsense consistency. Additionally, the 3-day window might be too narrow for evolving long-term events (e.g., wars).

## Related Work & Insights
- **vs LIAR / FEVER / FakeNewsNet**: These are one-time snapshots vulnerable to contamination; LiveFact is continuous and time-anchored.
- **vs LiveBench**: Shares the update philosophy but focuses on coding/math rather than fake news and its specific evidence structures.
- **vs TripleFact / AdvFake**: LiveFact is the first to integrate monthly updates, time-slicing, and BDC monitoring simultaneously.
- **vs SSA Original Work**: This work integrates SSA into an automated pipeline, identifying its ideal engineering scenario for monthly contamination monitoring.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (First to integrate dynamic updates, time-slicing, dual-mode evaluation, and BDC monitoring).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (18 LLMs from 1B to 1T; limited to one month of data for now).
- **Writing Quality**: ⭐⭐⭐⭐ (Clear motivation and concept visualization; dense formulas).
- **Value**: ⭐⭐⭐⭐⭐ (Potential to become the de facto standard for LLM factual reasoning evaluation if monthly updates persist).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection](../../ICML2026/social_computing/ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti.md)
- [\[ACL 2026\] VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking](veritas_the_first_dynamic_benchmark_for_multimodal_automated_fact-checking.md)
- [\[AAAI 2026\] FactGuard: Event-Centric and Commonsense-Guided Fake News Detection](../../AAAI2026/social_computing/factguard_event-centric_and_commonsense-guided_fake_news_detection.md)
- [\[ACL 2026\] Confident, Calibrated, or Complicit: Safety Alignment and Ideological Bias in LLM Hate Speech Detection](confident_calibrated_or_complicit_safety_alignment_and_ideological_bias_in_llm_h.md)
- [\[ACL 2026\] ClaimDB: A Fact Verification Benchmark over Large Structured Data](claimdb_a_fact_verification_benchmark_over_large_structured_data.md)

</div>

<!-- RELATED:END -->
