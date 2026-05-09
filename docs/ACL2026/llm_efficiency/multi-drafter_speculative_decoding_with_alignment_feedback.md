---
title: >-
  [Paper Note] Multi-Drafter Speculative Decoding with Alignment Feedback
description: >-
  [ACL 2026][LLM Efficiency][Speculative Decoding] MetaSD models multi-drafter speculative decoding as a multi-armed bandit problem with Block Divergence reward, dynamically selecting the best-aligned drafter for the target LLM.
tags:
  - ACL 2026
  - LLM Efficiency
  - Speculative Decoding
  - Multi-Armed Bandit
  - Multi-Drafter
  - Alignment Feedback
content_hash: 5ac96b5ea4720e71
---

# Multi-Drafter Speculative Decoding with Alignment Feedback

**Conference**: ACL 2026
**arXiv**: [2604.05417](https://arxiv.org/abs/2604.05417)
**Code**: Available
**Area**: LLM Efficiency
**Keywords**: Speculative Decoding, Multi-Armed Bandit, Multi-Drafter, Alignment Feedback, Inference Acceleration

## TL;DR
MetaSD is a unified framework integrating multiple heterogeneous drafters into speculative decoding, modeling drafter selection as a multi-armed bandit problem with Block Divergence (BD) reward signals to dynamically select the most aligned drafter, consistently outperforming single-drafter methods in both black-box and white-box configurations.

## Method

### Key Designs

1. **Block Divergence (BD) Reward**: Provides more informative alignment feedback than traditional block efficiency by computing TV distance averages across all positions in a draft block, with theoretically proven stronger feedback signal.

2. **Stopping-Time Regret Objective**: Minimizes the number of speculative decoding rounds to generate $B$ tokens vs the optimal strategy, achieving $O(\ln B)$ regret bound.

3. **MetaSD-UCB Algorithm**: Balances exploration and exploitation via UCB, naturally extending to the non-standard speculative decoding setting with rigorous regret analysis.

## Key Experimental Results

MetaSD-UCB automatically approaches near-optimal expert drafter performance without knowing the task type, significantly outperforming random selection and static ensembles. Framework naturally handles inter-query non-stationarity and requires no additional training.

## Highlights & Insights
- Speculative decoding + multi-armed bandit combination is natural: alignment feedback inherently provides reward signals
- Theoretical analysis of BD vs BE feedback signal strength is deep and generalizable

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)
- [\[NeurIPS 2025\] OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](../../NeurIPS2025/llm_efficiency/omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)
- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](../../NeurIPS2025/llm_efficiency/3model_speculative_decoding.md)
- [\[ACL 2026\] SciCoQA: Quality Assurance for Scientific Paper–Code Alignment](scicoqa_quality_assurance_for_scientific_paper--code_alignment.md)
- [\[NeurIPS 2025\] MoESD: Revealing the Potential of Speculative Decoding to Accelerate Sparse MoE](../../NeurIPS2025/llm_efficiency/moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)

<!-- RELATED:END -->
