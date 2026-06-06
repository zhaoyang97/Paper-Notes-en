---
title: >-
  [Paper Note] SSG: Logit-Balanced Vocabulary Partitioning for LLM Watermarking
description: >-
  [ACL 2026][LLM Safety][LLM Watermarking] This paper analyzes why KGW-style LLM watermarks fail in low-entropy scenarios such as code generation and mathematical reasoning. It proposes the Watermark Strength metric and SS…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "LLM Watermarking"
  - "Low-entropy Generation"
  - "Logit-balanced Partition"
  - "Watermark Strength"
  - "Content Attribution"
date: 2026-05-08
content_hash: 61a10e6cf514d166
---

# SSG: Logit-Balanced Vocabulary Partitioning for LLM Watermarking

**Conference**: ACL 2026  
**arXiv**: [2604.22438](https://arxiv.org/abs/2604.22438)  
**Code**: https://github.com/AllenG-L/SSG  
**Area**: LLM Security / Text Watermarking / Content Attribution  
**Keywords**: LLM Watermarking, Low-entropy Generation, Logit-balanced Partition, Watermark Strength, Content Attribution

## TL;DR
This paper analyzes why KGW-style LLM watermarks fail in low-entropy scenarios such as code generation and mathematical reasoning. It proposes the Watermark Strength metric and SSG (logit-balanced vocabulary partitioning), which balances high-probability tokens across sets to significantly improve watermark detectability without further degrading generation quality.

## Background & Motivation
**Background**: LLM watermarking is commonly used for content attribution, model output tracing, and governance. KGW is a representative logit-based scheme: it partitions the vocabulary into green and red groups, applying a logit bias to green tokens to embed detectable statistical signals.

**Limitations of Prior Work**: KGW performs well in natural language generation but shows significantly degraded detection capabilities in low-entropy tasks like code generation, mathematical reasoning, or JSON/SQL. In these tasks, the next-token distribution is often extremely sharp, with a few high-probability tokens determining the output; if these tokens happen to fall into the same group, the watermark bias fails to effectively shift the sampling distribution.

**Key Challenge**: The goal is to inject statistical signals without compromising output quality, yet low-entropy distributions offer very little room for adjustment. Simply increasing the logit bias can strengthen the signal but damages code correctness or mathematical accuracy.

**Goal**: The authors aim to improve the watermark injection stage rather than merely patching detection for low-entropy issues. The objective is to increase the lower bound of injectable statistical signals at each token position while maintaining compatibility with KGW-family detectors.

**Key Insight**: The paper introduces "Watermark Strength" to measure the normalized probability boost of the green-set due to watermark bias, noting that random partitioning often causes this value to approach 0 in low-entropy scenarios.

**Core Idea**: Instead of randomly partitioning high-probability tokens, they should be sorted by logits and grouped, then distributed evenly across sets to ensure the green/red groups remain balanced in high-probability regions.

## Method
The focus of this paper is on content attribution and watermark reliability analysis. The following summarizes the high-level methodology and conclusions without detailing specific operations for detection evasion.

### Overall Architecture
SSG is a vocabulary partitioning module that can be plugged into the KGW-family framework. Before generating each token, the model produces next-token logits; SSG constructs a more balanced green/red partition across high-logit candidates. It then follows KGW-style methods to inject bias into green tokens and uses compatible statistical detectors to evaluate the watermark signal. The core change lies in "which tokens are assigned to the green set" rather than the detection formula.

### Key Designs
1.  **Watermark Strength Metric**:
    - **Function**: Characterizes the injection intensity at a specific generation position.
    - **Mechanism**: Let $p_g$ be the original green set probability and $\tilde{p}_g$ be the probability after adding bias. The paper uses a normalized probability boost $f_{ws}$ to measure the impact of bias on the sampling distribution. If $p_g$ is near 0 or 1, the adjustable space for bias is minimal, leading to a very low $f_{ws}$.
    - **Design Motivation**: In low-entropy tasks, high-probability tokens are scarce. Random partitioning often clusters these tokens into one set. Watermark Strength transforms this empirical failure into an analyzable token-level metric.

2.  **Sort-then-Split by Groups Strategy**:
    - **Function**: Balances high-probability tokens across sets to enhance the watermark signal from the source.
    - **Mechanism**: SSG first sorts candidate tokens by logits, then processes high-logit tokens in adjacent pairs, assigning them to different sets. Lower-probability remaining tokens are partitioned randomly. This prevents severe imbalance in regions that most influence the sampling distribution.
    - **Design Motivation**: Watermark quality is primarily affected by high-probability tokens. Balancing the high-logit region raises the lower bound of $p_g$ without requiring an excessive increase in bias.

3.  **top-k Efficiency Trade-off and Detection Compatibility**:
    - **Function**: Reduces the overhead of full-vocabulary sorting and maintains compatibility with existing KGW-family detection methods.
    - **Mechanism**: The authors found that performing SSG only on the top-k high-probability tokens captures most of the benefits, while random partitioning suffices for the rest. Experiments comparing $k \in \{2, 4, 8, 16\}$ show that $k=2$ or $k=4$ usually provides the best trade-off between detection gain and speed.
    - **Design Motivation**: Full sorting increases decoding latency, whereas in low-entropy tasks, only the top few tokens typically determine the output.

### Loss & Training
SSG is a non-training method that does not modify model parameters or require fine-tuning. It operates during the decoding phase. In experiments, the authors integrated SSG with KGW, SWEET, and EWD to compare detection rates, F1 scores, and generation quality.

## Key Experimental Results

### Main Results

| Task / Model / Method | Quality Metric P@1 | TPR@1 | F1@1 | TPR@5 | F1@5 | Observation |
|:---|:---|:---|:---|:---|:---|:---|
| HumanEval / Qwen2.5-Coder / KGW | 26.2 | 22.0 | 35.8 | 36.0 | 51.3 | KGW has weak detection in low-entropy code |
| HumanEval / Qwen2.5-Coder / KGW+SSG | 25.6 | 39.0 | 55.9 | 45.7 | 60.7 | Detection improved, P@1 remains similar |
| MBPP / Qwen2.5-Coder / KGW | 38.4 | 37.8 | 54.6 | 64.0 | 75.9 | Baseline already has some signal |
| MBPP / Qwen2.5-Coder / KGW+SSG | 37.0 | 58.7 | 73.6 | 71.7 | 81.3 | Significant gain in TPR@1 and F1@1 |
| GSM8K / DSMath-7B / KGW | 21.7 | 41.7 | 58.5 | 57.6 | 71.0 | Deficiencies remain in math low-entropy |
| GSM8K / DSMath-7B / KGW+SSG | 25.8 | 90.9 | 94.9 | 91.7 | 93.4 | Massive detection boost and higher accuracy |
| GSM8K / LLaMA-3-8B / EWD | 35.6 | 74.2 | 84.8 | 95.5 | 95.5 | Strong baseline |
| GSM8K / LLaMA-3-8B / EWD+SSG | 35.6 | 99.2 | 99.2 | 99.2 | 97.4 | Near-perfect detection, no quality drop |

### Ablation Study

| Analysis Item | Result | Note |
|:---|:---|:---|
| Watermark Strength Distribution | KGW yields more near-zero strength tokens; SSG significantly reduces them | SSG gains come from stable token-level injection intensity |
| top-k Selection | top-2 significantly improves detection; k > 2 yields marginal returns | k=2 or k=4 is the recommended trade-off |
| Detection without original prompt | SSG improves in some settings but decreases in others | Method relies on logits under original prompt conditions |
| Rewrite Robustness | TPR drops significantly for all; SSG is more stable on LLaMA-3, weaker on DSMath | Rewriting weakens statistical signals; robustness remains an open issue |
| High-entropy Tasks | TPR and F1 also improve for KGW/SWEET/EWD on C4 and CNN/DM | Designed for low-entropy but not limited to it |

### Key Findings
- The improvement of SSG stems from the injection side: it ensures high-probability tokens are balanced across sets, allowing the same bias to produce more stable statistical shifts.
- A fundamental trade-off between generation quality and detection strength persists. While watermarking often lowers Pass@1, SSG does not significantly worsen quality compared to its respective baselines.
- Prompt-free detection and rewrite robustness are weaknesses: when the original prompt is missing or text is rewritten, the advantages of SSG become unstable.

## Highlights & Insights
- Watermark Strength is an excellent analytical tool, formalizing "low-entropy difficulty" as a lower-bound problem of injectable probability mass at each token.
- SSG modifies the vocabulary partition rather than the detector, which is a direct and effective approach. It demonstrates that low-entropy watermarking does not solely rely on complex detection statistics.
- The top-k version is practical: by processing only the most critical high-logit tokens, one can achieve most detection gains, aligning with the probabilistic structure of low-entropy tasks.

## Limitations & Future Work
- Experimental tasks were primarily HumanEval, MBPP, GSM8K, C4, and CNN/DailyMail. Other forms of structured low-entropy generation (SQL, JSON, config files, long-form engineering code) require further validation.
- Optimal detection for SSG depends on the original prompt because the partition is tied to conditional logits; this limits deployment where prompts are unavailable.
- Robustness against text rewriting is limited; experiments show all methods suffer significant detection drops after rewriting.
- Sorting or top-k processing at each step introduces decoding overhead, requiring engineering optimization for high-throughput API scenarios.

## Related Work & Insights
- **vs KGW**: KGW uses random partitioning, which might cluster high-probability tokens on one side in low-entropy cases; SSG uses logit-balanced partitioning to increase the lower bound of injection strength.
- **vs SWEET / EWD**: SWEET and EWD focus on improving low-entropy detection or weighting strategies; SSG acts as an injection-side module that can be stacked with these methods.
- **vs WaterMod / concurrent logit-balanced methods**: Related works also note the importance of token rank/probability balance. This paper is distinguished by its Watermark Strength analysis and systematic experiments.
- **Insight**: For LLM safety watermarking, the probability geometry during decoding is crucial. Rather than focusing only on detection statistics, it is equally important to optimize whether the watermark signal is "truly injected" into the sampling distribution.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Logit-balanced partition is simple yet effective; Watermark Strength clarifies the problem definition.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers code, math, high-entropy text, top-k, prompt-free, and rewrite robustness scenarios.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation, theory, and experimental narrative; some tables are dense due to method combinations.
- Value: ⭐⭐⭐⭐☆ Highly relevant for watermarking low-entropy tasks, though prompt dependency and rewrite robustness limit immediate deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] WaterMod: Modular Token-Rank Partitioning for Probability-Balanced LLM Watermarking](../../AAAI2026/llm_safety/watermod_modular_token-rank_partitioning_for_probability-balanced_llm_watermarki.md)
- [\[ACL 2026\] STELA: A Linguistics-Aware LLM Watermarking via Syntactic Predictability](a_linguistics-aware_llm_watermarking_via_syntactic_predictability.md)
- [\[ACL 2026\] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)
- [\[ACL 2026\] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework](rethinking_llm_watermark_detection_in_black-box_settings_a_non-intrusive_third-p.md)
- [\[ACL 2026\] AgentMark: Utility-Preserving Behavioral Watermarking for Agents](agentmark_utility-preserving_behavioral_watermarking_for_agents.md)

</div>

<!-- RELATED:END -->
