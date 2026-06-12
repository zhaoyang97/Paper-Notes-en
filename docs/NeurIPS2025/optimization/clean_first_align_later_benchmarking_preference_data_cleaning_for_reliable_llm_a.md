---
title: >-
  [Paper Note] Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment
description: >-
  [NeurIPS 2025 (D&B Track)][Optimization][preference data cleaning] This paper introduces PrefCleanBench, the first comprehensive benchmark for systematically evaluating 13 preference data cleaning methods in the context…
tags:
  - "NeurIPS 2025 (D&B Track)"
  - "Optimization"
  - "preference data cleaning"
  - "LLM alignment"
  - "RLHF"
  - "DPO"
  - "benchmark"
date: 2026-05-08
content_hash: 3cc4fd30366fa0ca
---

# Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment

**Conference**: NeurIPS 2025 (D&B Track)  
**arXiv**: [2509.23564](https://arxiv.org/abs/2509.23564)  
**Code**: [https://github.com/deeplearning-wisc/PrefCleanBench](https://github.com/deeplearning-wisc/PrefCleanBench)  
**Area**: Optimization  
**Keywords**: preference data cleaning, LLM alignment, RLHF, DPO, benchmark

## TL;DR

This paper introduces PrefCleanBench, the first comprehensive benchmark for systematically evaluating 13 preference data cleaning methods in the context of LLM alignment. It covers diverse datasets, model architectures, and optimization algorithms, revealing the underappreciated yet critical role of data preprocessing in responsible AI development.

## Background & Motivation

**Background**: Human feedback plays a central role in LLM alignment—whether for training reward models in RLHF or for direct optimization algorithms such as DPO. All such approaches rely on high-quality preference data (i.e., chosen/rejected pairs). Mainstream alignment methods include DPO, CPO, KTO, IPO, SLiC, rDPO, ORPO, AOT, and others.

**Limitations of Prior Work**:
   - Human feedback is inherently noisy and inconsistent, with disagreements among annotators and varying annotation quality.
   - Noisy preference data degrades reward model quality and impedes alignment.
   - Although various automated data cleaning methods have been proposed, they are scattered across different papers without a unified evaluation framework.
   - The effectiveness of different cleaning methods varies across settings, and no systematic comparative study exists.

**Key Challenge**: Numerous data cleaning methods exist without a unified evaluation standard—each claims effectiveness, but under what conditions, for which models, and for which algorithms remains unclear.

**Goal**:
   - Establish a unified evaluation protocol for preference data cleaning.
   - Systematically compare 13 cleaning methods.
   - Identify key factors that determine cleaning success.
   - Assess the generalizability of cleaning methods across datasets, models, and algorithms.

**Key Insight**: Adopting a benchmark perspective, this work does not propose new cleaning methods but instead conducts a fair and comprehensive comparison of existing approaches.

**Core Idea**: Establish PrefCleanBench, the first standardized benchmark for preference data cleaning in LLM alignment, and employ controlled experiments to reveal the critical role of data quality in alignment.

## Method

### Overall Architecture

The PrefCleanBench evaluation pipeline consists of four stages:
1. **Dataset Preparation**: Download and process target datasets (AnthropicHH, UltraFeedback, PKU-SafeRLHF, HelpSteer2).
2. **Data Cleaning**: Apply 13 cleaning methods to each dataset, producing multiple cleaned variants.
3. **Training and Generation**: Train alignment models on cleaned and uncleaned data, then generate responses.
4. **Evaluation**: Assess alignment quality using win-tie rate and average gold reward.

### Key Designs

**Classification of 13 Cleaning Methods**: PrefCleanBench covers 5 major categories of cleaning strategies, subdivided into 13 specific methods:

1. **LLM Judge**: Uses a large language model (e.g., GPT-4) as a judge to assess the quality of preference pairs.

    - `llm_judge_r`: Removes low-quality pairs (reject strategy).
    - `llm_judge_f`: Flips labels (flip strategy).

2. **Reward Gap (RwGap)**: Uses a reward model to compute the reward gap between chosen and rejected responses.

    - `rw_gap_r_{0.1-0.4}`: Removes pairs below a reward gap threshold.
    - `rw_gap_f_{0.1-0.4}`: Flips pairs with negative reward gaps.

3. **Voting**: Multiple models or annotators vote to determine preference.

    - `vote_all_r/f`: Retains/flips only on unanimous agreement.
    - `vote_maj_r/f`: Decides based on majority vote.

4. **InsTag**: Quality filtering based on instruction tags.

    - `ins_tag_cmp`: Complexity-based filtering.
    - `ins_tag_div`: Diversity-based filtering.

5. **IFD (Instruction Following Difficulty)**: Filtering based on instruction-following difficulty.

    - `ifd_r_{0.1-0.4}`: Removal by threshold.
    - `ifd_gap_r/f_{0.1-0.4}`: Removal/flipping based on IFD gap.

**Evaluation Dimensions**:

| Dimension | Options |
|-----------|---------|
| Dataset | AnthropicHH, UltraFeedback, PKU-SafeRLHF, HelpSteer2 |
| Base Model | Llama-3-8B, Qwen2.5-7B, Phi-2, Mistral-7B-v0.3 |
| Optimization Algorithm | DPO, CPO, AOT, KTO, IPO, SLiC, rDPO, ORPO |

**Evaluation Metrics**:
- **Win-Tie Rate**: Head-to-head positive comparison rate of the cleaned model versus the uncleaned baseline.
- **Avg. Gold Reward**: Quality of generated responses evaluated using an independent gold-standard reward model.

### Loss & Training

This paper does not propose new training methods; instead, it uses standard alignment training pipelines, with each optimization algorithm following the configuration from its original paper. The core contribution lies in the systematic control of variables: model architecture and optimization algorithm are held fixed while only the data cleaning strategy is varied, enabling fair comparison of cleaning methods.

## Key Experimental Results

### Main Results

Performance of different cleaning methods under DPO + Llama-3-8B (win-tie rate, illustrative):

| Cleaning Method | AnthropicHH | UltraFeedback | PKU-SafeRLHF | HelpSteer2 | Average |
|----------------|-------------|---------------|-------------|------------|---------|
| No Clean (baseline) | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% |
| LLM Judge (r) | Higher | Moderate | Higher | Moderate | >50% |
| RwGap (r, 0.2) | Moderate | Higher | Moderate | Moderate | ~50% |
| Voting (maj, r) | Moderate | Moderate | Higher | Lower | ~50% |
| IFD Gap (r, 0.2) | Lower | Higher | Moderate | Higher | ~50% |

Cross-algorithm consistency:

| Algorithm | Cleaning Consistently Beneficial | Best Cleaning Method | Worst Cleaning Method |
|-----------|----------------------------------|---------------------|----------------------|
| DPO | Partially | Dataset-dependent | Excessive cleaning |
| CPO | Partially | LLM Judge family | — |
| KTO | More consistent | RwGap family | — |
| IPO | Inconsistent | — | — |

### Ablation Study

**Effect of Cleaning Ratio** (using RwGap as an example):

| Threshold | Data Retention | DPO Effect | CPO Effect | KTO Effect |
|-----------|---------------|------------|------------|------------|
| 0.1 | ~90% | Slight gain | Neutral | Slight gain |
| 0.2 | ~80% | Moderate gain | Slight gain | Moderate gain |
| 0.3 | ~70% | Inconsistent | Inconsistent | Decline |
| 0.4 | ~60% | Decline | Decline | Decline |

### Key Findings

1. **No universal cleaning method**: No single cleaning method achieves the best performance across all dataset, model, and algorithm combinations.
2. **Excessive cleaning is harmful**: Removing more than 30% of data typically degrades alignment performance; the loss in data volume often outweighs the gain in data quality.
3. **Cleaning effectiveness is strongly correlated with the optimization algorithm**: Certain cleaning methods are effective for DPO but not for KTO, and vice versa.
4. **LLM Judge methods are relatively robust**: Cleaning methods based on LLM judges perform well in most settings, but incur higher costs.
5. **Dataset characteristics have a significant impact**: High-noise datasets (e.g., AnthropicHH) benefit more from cleaning, while high-quality datasets (e.g., HelpSteer2) show limited gains.
6. **Generalization challenge**: A cleaning strategy that is optimal for one dataset may be suboptimal or even harmful for another.

## Highlights & Insights

- **First systematic benchmark**: Fills the gap in the absence of a unified evaluation framework for preference data cleaning.
- **Rigorous experimental design**: Achieves fair comparison through a controlled variable strategy (fixing two of three dimensions—model, algorithm, data—while varying one).
- **Modular code release**: All 13 methods are open-sourced as modular implementations, lowering the barrier for follow-up research.
- **Practical guidance**: Provides empirical evidence to help practitioners select appropriate cleaning methods.
- **Emphasis on data preprocessing in responsible AI**: This perspective is relatively rare in a research landscape dominated by algorithmic innovation.

## Limitations & Future Work

- Only open-source models (up to 8B parameters) are evaluated; behavior on larger models (70B+) may differ.
- Evaluation metrics (win-tie rate and gold reward) may not fully capture all dimensions of alignment quality (e.g., safety, factuality).
- Multi-turn dialogue preference data cleaning scenarios are not addressed.
- The computational cost comparison across cleaning methods is insufficient (e.g., LLM Judge API costs vs. lightweight alternatives).
- Combinations of multiple cleaning methods are not explored.

## Related Work & Insights

- **RLHF / DPO family**: Mainstream preference alignment methods that serve as the downstream applications evaluated in this work.
- **Data-Centric AI**: A research paradigm emphasizing data quality over model innovation.
- **Curriculum Learning / Data Selection**: Related work on selecting the most valuable subsets from training data.
- **Reward Modeling**: Reward model quality is directly affected by preference data quality and is a primary beneficiary of data cleaning.

## Rating

- **Novelty**: ⭐⭐⭐ — No new method is proposed, but the first systematic evaluation constitutes an important contribution.
- **Technical Depth**: ⭐⭐⭐ — The methodology is straightforward, yet the experimental design is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — The combination of 4 datasets × 4 models × 8 algorithms × 13 methods provides comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, with findings that carry practical guidance value.
- **Practicality**: ⭐⭐⭐⭐⭐ — Directly useful for researchers and engineers working on LLM alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](doubly_robust_alignment_for_large_language_models.md)
- [\[NeurIPS 2025\] Preference Learning with Response Time: Robust Losses and Guarantees](preference_learning_with_response_time_robust_losses_and_guarantees.md)
- [\[AAAI 2026\] Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment](../../AAAI2026/optimization/cost-minimized_label-flipping_poisoning_attack_to_llm_alignment.md)
- [\[NeurIPS 2025\] The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels](the_implicit_bias_of_structured_state_space_models_can_be_poisoned_with_clean_la.md)
- [\[NeurIPS 2025\] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](neuro-symbolic_entity_alignment_via_variational_inference.md)

</div>

<!-- RELATED:END -->
