---
title: >-
  [Paper Note] Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL
description: >-
  [ACL 2026][LLM Safety][Refusal Calibration] Abstain-R1 proposes a **clarification-aware RLVR reward** to jointly optimize "explicit refusal" and "providing helpful clarifications (identifying missing information)" for un…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Refusal Calibration"
  - "Post-Refusal Clarification"
  - "Verifiable Reward"
  - "GRPO"
  - "Unanswerable Queries"
date: 2026-05-08
content_hash: 3c4f4b1d2a13beec
---

# Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL

**Conference**: ACL 2026  
**arXiv**: [2604.17073](https://arxiv.org/abs/2604.17073)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Refusal Calibration, Post-Refusal Clarification, Verifiable Reward, GRPO, Unanswerable Queries

## TL;DR
Abstain-R1 proposes a **clarification-aware RLVR reward** to jointly optimize "explicit refusal" and "providing helpful clarifications (identifying missing information)" for unanswerable queries. This enables a 3B model to approach or even surpass larger models like DeepSeek-R1 in both refusal and clarification quality.

## Background & Motivation

**Background**: RL post-training (such as RLVR/GRPO) significantly enhances LLM reasoning capabilities. however, existing training objectives typically assume all queries are answerable, rewarding the act of "giving an answer" even when the query is actually unsolvable.

**Limitations of Prior Work**: When queries are semantically clear but lack sufficient information (e.g., missing variable definitions or contradictory premises), models tend to guess or "fill in the world" to generate seemingly complete answers, incurring a so-called "Hallucination Tax." Existing refusal methods either train models to produce generic refusals ("I don't know") or encourage follow-up questions without verifying if those questions accurately identify the missing information.

**Key Challenge**: Simple refusal provides little value—users need to know **why an answer is impossible and what information is missing**. However, existing RL training lacks verifiable signals to evaluate the quality of post-refusal clarifications.

**Goal**: To enable the model to (1) explicitly refuse unanswerable queries; (2) provide **semantically aligned clarifications** that accurately pinpoint missing information; and (3) maintain performance on answerable queries.

**Key Insight**: Incorporating clarification quality into the RLVR reward design by using a lightweight verifier model to judge if the model's clarification is semantically consistent with a reference clarification.

**Core Idea**: Mixing unanswerable samples into standard GRPO training and jointly optimizing refusal and clarification using a hierarchical reward function consisting of a "refusal format reward + clarification correctness reward."

## Method

### Overall Architecture
The training process consists of three stages: (1) Constructing the Abstain-CoT dataset (containing reasoning chains and refusal+clarification annotations) for SFT cold start; (2) Performing SFT on Qwen2.5-3B-Instruct to establish basic refusal and reasoning formats; (3) Conducting RL training with GRPO using a mixture of 30% unanswerable and 70% answerable queries, optimized by a composite reward function.

### Key Designs

1.  **Clarification-Aware Composite Reward Function**:
    - **Function**: Provides learnable, fine-grained reward signals for unanswerable queries.
    - **Mechanism**: The total reward $r(o,y)$ is determined by two cases: answerable queries receive a format reward $r_{\text{fmt}}$ + correctness reward $r_{\text{ans}}$; unanswerable queries receive a format reward + refusal reward $r_{\text{ref}}$. A key aspect is the hierarchical design of $r_{\text{ref}}$: outputting a boxed "I don't know" earns a base score of 0.3; if the clarification also passes the verifier $\mathcal{V}$ as correct, an additional 0.7 is awarded, totaling 1.0. Simultaneously, a -1 penalty is applied to refusals on answerable queries to prevent over-refusal.
    - **Design Motivation**: Rewarding only refusal causes the model to "refuse everything." Adding clarification correctness rewards and penalties for false refusals on answerable queries creates a bidirectional constraint.

2.  **Lightweight Verifier Model $\mathcal{V}$**:
    - **Function**: Provides real-time judgment of clarification correctness during the RL training loop.
    - **Mechanism**: Rewrites the original question into a meta-level "why is this unanswerable" query, requiring the verifier to compare the semantic consistency between the model's clarification $\hat{c}$ and the reference clarification $c^\star$. A conservative 3B verifier (xVerify-3B-Ia) is used during training to reduce reward hacking, while a stronger o4-mini is used for evaluation.
    - **Design Motivation**: Direct string matching is too fragile; using an LLM verifier for semantic comparison is more robust. Using verifiers of different strengths for training versus evaluation helps avoid overfitting.

3.  **Abstain-CoT Dataset and SFT Cold Start**:
    - **Function**: Establices the initial refusal and reasoning format for the RL stage.
    - **Mechanism**: Selects semantically clear but unanswerable subsets from AbstentionBench and uses DeepSeek-V3 to generate 4.6K structured training samples with `<thinking>` reasoning chains, covering math, life sciences, fact-checking, and other domains.
    - **Design Motivation**: Without SFT cold start, it is extremely difficult for RL to learn the refusal format from scratch given sparse rewards.

### Loss & Training
The standard GRPO objective function is employed, where $G$ candidate outputs are generated for each query. Policy gradients are calculated based on the relative advantage $A_i$ within the group, with KL regularization added to prevent deviation from the reference policy. Answerable and unanswerable queries are mixed in a 7:3 ratio.

## Key Experimental Results

### Main Results

| Dataset | Metric | Abstain-R1 (3B) | Qwen2.5-3B | DeepSeek-R1 | Gain (vs base) |
|--------|------|------|----------|------|------|
| Abstain-Test | U-Ref (Refusal Rate) | **68.1%** | 9.4% | 52.2% | +58.7% |
| Abstain-Test | U-Clar (Clarification Acc) | **55.1%** | 0.6% | 46.5% | +54.5% |
| Abstain-Test | A-Acc (Answerable Acc) | 57.2% | 48.8% | **78.6%** | +8.4% |
| SelfAware | U-Ref | **91.4%** | 82.3% | 63.8% | +9.1% |
| Abstain-QA | U-Ref | **40.1%** | 30.0% | 9.1% | +10.1% |

### Ablation Study

| Configuration | A-Acc | U-Ref | U-Clar | Description |
|------|---------|------|------|------|
| Abstain-R1 | 57.2% | 68.1% | 55.1% | Full model |
| w/o SFT | 53.3% | 65.1% | 8.5% | No cold start; clarification quality collapses |
| w/o RL | 55.4% | 51.9% | 37.0% | Pure SFT; insufficient refusal |
| w/o Unans | 67.5% | 4.4% | 3.1% | No unanswerable data; almost zero refusal |
| w/o clari reward | 55.9% | 64.5% | 50.2% | No clarification reward; clarification performance drops |

### Key Findings
- SFT is the primary source of clarification capability (U-Clar drops from 55.1% to 8.5% without it), while RL mainly strengthens the timing of refusal.
- The refusal penalty on answerable queries is critical: without it, the False Refusal rate (A-FU) surges from 20.4% to 36.2%.
- The 3B model outperforms larger models like DeepSeek-R1 in terms of refusal and clarification, proving that calibrated refusal can be achieved through targeted training rather than scale alone.
- During RL training, the model gradually becomes more concise, while refusal rates, clarification accuracy, and answer accuracy improve simultaneously.

## Highlights & Insights
- **Treating post-refusal clarification as a first-class training objective** is the core contribution of this paper: moving beyond simply "saying I don't know" to "saying I don't know + explaining why," which is highly valuable for high-stakes applications (e.g., medical, legal).
- The **hierarchical reward design** (0.3 base refusal + 0.7 clarification correctness) finds a good balance between conciseness and informativeness and can be transferred to other RL training scenarios requiring structured outputs.
- The practice of using verifiers of different strengths (conservative 3B for training, strong o4-mini for evaluation) is a practical technique for mitigating reward hacking.

## Limitations & Future Work
- Answerable accuracy remains significantly lower than that of larger models (57.2% vs. 78.6% for DeepSeek-R1), as the reasoning capability of the 3B backbone is a bottleneck.
- The false refusal rate of 20.4% is relatively high; approximately 1/5 of answerable questions are incorrectly refused.
- Clarification quality depends on the quality of reference clarifications generated by DeepSeek-V3, which may introduce bias.
- The study only targets the "semantically clear but informatively insufficient" type of unanswerable query, leaving other scenarios like semantic ambiguity unaddressed.

## Related Work & Insights
- **vs AbstentionBench**: The latter evaluates refusal capability but does not involve training methods; Abstain-R1 provides a complete training-evaluation framework.
- **vs Hallucination Tax (Song et al.)**: The latter diagnoses how RL training exacerbates hallucinations; Abstain-R1 directly provides a solution (mixing unanswerable samples + composite rewards).
- **vs CoCoNot**: The latter learns context non-compliance via SFT but is fragile in out-of-distribution scenarios; Abstain-R1 achieves stronger generalization through RL.

## Rating
- Novelty: ⭐⭐⭐⭐ Incorporating clarification quality into RLVR is a fresh perspective, though the core technique remains based on standard GRPO.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks, multi-dimensional metrics, detailed ablations, reward sensitivity analysis, and training dynamics analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Precise definition of research problems, clearly organized RQs, and high information density in tables and figures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)
- [\[ACL 2026\] Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding](please_refuse_to_answer_me_mitigating_over-refusal_in_large_language_models_via_.md)
- [\[ICML 2026\] Optimizing Token Choice for Code Watermarking: An RL Approach](../../ICML2026/llm_safety/optimizing_token_choice_for_code_watermarking_an_rl_approach.md)
- [\[ICML 2026\] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](../../ICML2026/llm_safety/actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)
- [\[ACL 2026\] FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation](flexguard_continuous_risk_scoring_for_strictness-adaptive_llm_content_moderation.md)

</div>

<!-- RELATED:END -->
