---
title: >-
  [Paper Note] Incentive-Aligned Multi-Source LLM Summaries
description: >-
  [ICLR 2026][Audio & Speech][truthful summarization] This paper introduces the Truthful Text Summarization (TTS) framework, which incorporates a multi-task peer prediction mechanism from game theory into LLM multi-source…
tags:
  - "ICLR 2026"
  - "Audio & Speech"
  - "truthful summarization"
  - "incentive alignment"
  - "peer prediction"
  - "prompt injection"
  - "source reliability"
date: 2026-05-08
content_hash: 73ca13ead75da418
---

# Incentive-Aligned Multi-Source LLM Summaries

**Conference**: ICLR 2026
**arXiv**: [2509.25184](https://arxiv.org/abs/2509.25184)  
**Code**: None  
**Area**: Audio & Speech
**Keywords**: truthful summarization, incentive alignment, peer prediction, prompt injection, source reliability

## TL;DR

This paper introduces the Truthful Text Summarization (TTS) framework, which incorporates a multi-task peer prediction mechanism from game theory into LLM multi-source summarization pipelines. The approach constructs evaluation claim sets via leave-one-out cross-referencing, extracts each source's stance on individual claims, scores source reliability using informative agreement, filters unreliable sources, and regenerates the summary. The framework is theoretically proven to make truthful reporting a utility-maximizing strategy, and empirically demonstrates robustness against prompt injection, misinformation sources, and coordinated attacks.

## Background & Motivation

**Paradigm shift from search to summarization**: Traditional search engines present multiple results as independent entries, limiting the impact of any single malicious source. LLM-driven summarization merges multiple sources into a unified narrative, allowing a single strategic actor to hijack the entire output via prompt injection or semantic manipulation—far exceeding the influence achievable through traditional search ranking.

**Three dimensions of LLM vulnerability**: (a) susceptibility to plausible hallucinations; (b) manipulability by adversarial prompt injection; (c) difficulty adjudicating mutually contradictory claims. These properties create exploitable opportunities for malicious sources.

**Incentive misalignment**: Existing RAG pipelines focus on technical summarization quality optimization (e.g., self-critique, LLM-as-judge) without accounting for the strategic behavior of content creators—if manipulation yields greater exposure at low cost, information sources are incentivized to fabricate content.

**Key Challenge**: Achieving simultaneous technical robustness (filtering bad sources) and incentive robustness (making truthful reporting a Nash equilibrium) without access to ground-truth labels.

**Key Insight**: Drawing on peer prediction mechanisms from game theory, which operate without ground-truth labels, using informative agreement among sources to assess reliability.

## Method

### Overall Architecture of TTS

TTS employs a two-pass pipeline. Given a query $q$ and a retrieved source set $\mathcal{C}$:

**First Pass — Source Scoring (Leave-One-Out Peer Prediction)**:

1. **Leave-One-Out Claim Construction**: For each source $\tau_i$, a draft summary is generated from the remaining sources $\{\tau_j\}_{j \neq i}$ and decomposed by a decomposer $D$ into an atomic claim set $T_i$. Crucially, $\tau_i$ is excluded from constructing its own evaluation set, ensuring claim exogeneity.
2. **Stance Extraction**: An extractor $E$ derives each source's stance on each claim: $r_{ik} \in \{1(\text{support}), 0(\text{oppose}), \bot(\text{abstain})\}$.
3. **Informative Agreement Scoring**: For each (source $i$, peer $j$) pair, on-task agreement is computed minus off-task agreement, then averaged across peers and claims to obtain $\hat{w}_i$. The core formula is: $\sigma_{ikj} = S(r_{ik}, r_{jk}) - S(r_{i\ell}, r_{jm})$, where $\ell, m$ are distinct claims selected via random permutation.

**Second Pass — Filtering and Re-summarization**: Sources with $\hat{w}_i < t_{\text{src},i}$ are filtered out, and the summary is regenerated using only reliable sources.

### Computational Efficiency Optimization

The source set $\mathcal{C}$ is randomly partitioned into groups A and B; sources in group A have their claim sets constructed from group B documents, and vice versa. This preserves exogeneity while reducing complexity from $O(|\mathcal{C}|K(|\mathcal{C}|-1))$ to $O(K|\mathcal{C}|)$.

### Theoretical Guarantees

| Theorem | Conditions | Guarantee |
|---|---|---|
| Thm 3.2 (Asymptotic Informed Truthfulness) | $K \to \infty$, threshold $0 < t < \alpha_i \eta_i^{\text{truth}} \gamma$ | Truthful reporting weakly dominates all strategies and strictly dominates all uninformative strategies |
| Thm 3.3 (Strong Truthfulness) | Large $K$ + claims with bias flip $\geq \varphi_{\min}$ | Truthful reporting strictly dominates all significantly biased strategies |
| Thm 3.4 ($\varepsilon$-Informed Truthfulness) | Finite $K$ + midpoint threshold | Utility error decays exponentially in $K$; $K \geq O(\ln(v_i/\varepsilon)/\underline{g}_i^2)$ suffices |

### Key Differences from Classical Peer Prediction

| Dimension | Classical Peer Prediction | TTS |
|---|---|---|
| Source of evaluation tasks | Externally fixed | LOO-constructed; sources cannot manipulate evaluation sets |
| Report format | Abstract signals | Natural language documents; extractor converts to stances |
| Incentive mechanism | Monetary payment | Exposure/attribution (inclusion in summary) |
| Application context | Peer review, etc. | Open-web search (payment infeasible) |

## Key Experimental Results

### Main Results

| Method | NQ Precision | NQ Answer Acc | ClashEval Precision | ClashEval Answer Acc |
|---|---|---|---|---|
| Initial Synthesis | 40.8% | 25.1% | 49.3% | 15.6% |
| Majority Prompt | 43.4% | 27.5% | 58.7% | 30.2% |
| Majority Claims | 50.1% | 38.6% | 63.6% | 38.4% |
| **TTS (Ours)** | **76.1%** | **72.3%** | **86.2%** | **77.1%** |

TTS improves answer accuracy on NQ to 72.3% (vs. 25.1% for the initial synthesis baseline) and on ClashEval to 77.1% (vs. 15.6%), with precision gains approaching twofold improvement.

### Robustness Against Coordinated Attacks

When four "uninformative" sources (opposing all claims) are injected into ClashEval, simple majority voting fails entirely—assigning high scores to coordinated attackers and incorrectly inflating adversarial source ratings. TTS continues to assign near-zero scores to uninformative sources and maintains correct reliability rankings. This validates the theoretical robustness of peer prediction scoring against coordinated uninformative equilibria.

### Computational Overhead

On average, each query (7 sources) requires approximately 174K input tokens and 13K output tokens, costing approximately $0.07/query using gemini-2.5-flash-lite. In practice, TTS can be run on sampled traffic to accumulate source reputation signals incrementally.

## Highlights & Insights

- **Pioneering intersection of game theory and LLM safety**: This is the first application of peer prediction to source filtering in LLM summarization, enabling discrimination between reliable and unreliable sources without ground-truth labels.
- **Structural advantage**: By isolating and removing unreliable sources prior to final generation, the framework fundamentally blocks the influence pathway of adversarial text—a more thorough defense than prompt-level countermeasures.
- **Implications for RAG systems**: The TTS scoring mechanism can be embedded as a source credibility assessment module in any LLM system that integrates external sources (RAG, agents, search summarization).
- **Incentive design perspective**: The paper reframes LLM summarization from "how to generate good summaries" to "how to design ecosystems that incentivize truthful information provision."

## Limitations & Future Work

- Experiments are conducted at small scale (6–7 sources per query); validation in large-scale settings with hundreds of sources remains absent.
- A fixed global threshold $t = 0.06$ is used; adaptive thresholding could further improve performance.
- The quality of claim decomposition and stance extraction depends on LLM capability; performance in multilingual or highly specialized domains has not been verified.
- Integration with reputation priors (discussed in Appendix D) could enable incremental source evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The intersection of game theory and LLM summarization is a genuinely novel direction, with complete theoretical guarantees
- Experimental Thoroughness: ⭐⭐⭐ Small-scale validation is effective, but large-scale and multilingual experiments are lacking
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous and framework diagrams are clear
- Value: ⭐⭐⭐⭐⭐ Significant implications for LLM information security and RAG system design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MAPSS: Manifold-Based Assessment of Perceptual Source Separation](mapss_manifold-based_assessment_of_perceptual_source_separation.md)
- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](../../AAAI2026/audio_speech/psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)
- [\[AAAI 2026\] Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases](../../AAAI2026/audio_speech/thucy_an_llm-based_multi-agent_system_for_claim_verification_across_relational_d.md)
- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](../../AAAI2026/audio_speech/hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)
- [\[ICLR 2026\] LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision](logicreward_incentivizing_llm_reasoning_via_step-wise_logical_supervision.md)

</div>

<!-- RELATED:END -->
