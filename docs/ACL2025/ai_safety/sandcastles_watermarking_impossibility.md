---
title: >-
  [Paper Note] Sandcastles in the Storm: Revisiting Watermarking Impossibility
description: >-
  [ACL 2025][AI Safety][watermarking] This work challenges the theoretical impossibility results of "Watermarks in the Sand" (WITS) through large-scale experiments and human evaluation. It demonstrates that the two key assumptions of random walk attacks do not hold in practice: mixing is extremely slow (100% of attacked texts can still be traced back to their original source) and quality oracles are unreliable (only 77% accuracy), resulting in an automatic attack success rate o…
tags:
  - "ACL 2025"
  - "AI Safety"
  - "watermarking"
  - "impossibility result"
  - "random walk attack"
  - "mixing time"
  - "quality oracle"
  - "robustness"
  - "WITS"
date: 2026-05-08
content_hash: 699f7992d60777cb
---

# Sandcastles in the Storm: Revisiting Watermarking Impossibility

**Conference**: ACL 2025  
**arXiv**: [2505.06827](https://arxiv.org/abs/2505.06827)  
**Code**: None  
**Institution**: University of California, Los Angeles (UCLA)
**Area**: AI Safety  
**Keywords**: watermarking, impossibility result, random walk attack, mixing time, quality oracle, robustness, WITS

## TL;DR

This work challenges the theoretical impossibility results of "Watermarks in the Sand" (WITS) through large-scale experiments and human evaluation. It demonstrates that the two key assumptions of random walk attacks do not hold in practice: mixing is extremely slow (100% of attacked texts can still be traced back to their original source) and quality oracles are unreliable (only 77% accuracy), resulting in an automatic attack success rate of only 26%, which further drops to 10% after human quality auditing.

## Background & Motivation

### Background

- Text watermarking is a key technology for combating AI content abuse (misinformation, academic fraud, IP theft).
- The theoretical analysis of WITS (Zhang et al., 2024) claims that any watermarking scheme can be erased by random walk attacks without degrading text quality.
- This "impossibility result" severely threatens the prospects of watermarking technology, raising doubts about the feasibility of AI accountability mechanisms.

### Limitations of Prior Work

- The theoretical analysis of WITS relies on two key assumptions (KAs) that have never been empirically validated:
    - **KA1 (Fast Mixing)**: Watermarks dissolve rapidly under perturbation, allowing the random walk to efficiently converge to a stationary distribution.
    - **KA2 (Reliable Quality Preservation)**: Automated quality oracles can perfectly guide edits, ensuring that perturbations do not degrade text quality.
- A massive gap may exist between theoretically elegant attacks and their practical feasibility.
- If the impossibility result does not hold, watermarking technology remains highly valuable.

### Key Insight

- The "fast mixing" assumption in the theoretical analysis requires the second largest eigenvalue of the transition matrix to be close to zero, a condition that is difficult to satisfy in actual text spaces.
- The high-quality text space is highly structured, making it difficult for local perturbations to cross semantic boundaries.
- The conventional wisdom that "verification is easier than generation" does not hold in the context of LLM watermarking attacks.

## Method

### Overall Architecture

Three meticulously designed research questions are proposed to test the two key assumptions of WITS: RQ1 validates KA1 (whether the stationary distribution is reachable), RQ2 validates KA2 (whether the quality oracle is reliable), and RQ3 comprehensively evaluates the practical efficacy of the attacks.

### Key Designs

#### Key Design 1: Lineage Distinguisher Test (Validating KA1)

- Generate two initial responses for each prompt as "starting points."
- Perform a random walk attack (e.g., 1000 steps of WordMutator) on one starting point.
- Periodically sample intermediate texts and use an LLM (Llama-3.1-70B $\to$ GPT-4o $\to$ o3-mini-high) to identify the source of the text.
- If fully mixed, the classification accuracy should drop to a random level; if they can be distinguished with 100% accuracy, it indicates that mixing has not occurred.

#### Key Design 2: Sandcastles Benchmark (Validating KA2)

- Sample 100 diverse prompts from arena-human-preference-55k.
- Generate watermarked text and perform up to 20 iterative perturbations.
- Collect human blind annotation at steps 1, 10, and 20 (ternary preference judgment: A is better / B is better / tie).
- Benchmark six quality oracle variants: MutationOracle, DiffOracle, InternLM Reward Model, etc.
- Include fine-tuned versions and variants utilizing GPT-4o / GPT-4-Turbo.

#### Key Design 3: Comprehensive Attack Evaluation

- Seven perturbation oracles ($P$): WordMutator, EntropyWordMutator, SpanMutator, SentenceMutator, DocumentMutator, etc.
- Three watermarking schemes ($W$): KGW, SIR (semantic watermarking), and Adaptive.
- Sufficient step budgets (1000 steps for token-level, 100 steps for document-level).
- Use InternLM as the quality oracle to guide the attacks.
- Double validation using automatic evaluation and human quality auditing.

#### Dataset Design

- Entropy-controlled prompts: covering three domains: education, news, and creative writing.
- Multi-tier prompts designed from broad to specific in each domain (controlling output space size).
- Generate 270 watermarked texts + 90 unwatermarked texts using Llama-3.1-70B-Instruct.
- Additionally incorporate GPT-4o unwatermarked outputs as high-quality baselines.

## Key Experimental Results

### RQ1: Stationary Distribution Reachability (Table 1)

| Perturbation Method | Step Budget | Number of Tests | Llama-3.1-70B Failures | GPT-4o Failures | o3-mini Failures |
|---------|--------|-------|------------------|------------|-------------|
| Word | 1000 | 720 | 0 | 0 | 0 |
| EntropyWord | 1000 | 720 | 0 | 0 | 0 |
| Span | 250 | 720 | 12 | 1 | 0 |
| Sentence | 150 | 720 | 38 | 3 | 0 |
| **Total** | — | **4555** | 53 | 4 | **0** |
| **Final Distinguishing Rate** | | | | | **100%** |

### RQ2: Quality Oracle Performance (Table 2)

| Oracle | QP Precision | Overall F1 |
|-------|-------------|-----------|
| MutationOracle (Llama-3.1-70B) | 84.62% | 66.93% |
| DiffOracle+FT (fine-tuned) | 69.07% | 76.94% |
| **MutationOracle+FT (GPT-4o+FT)** | **74.51%** | **77.38%** |
| InternLM Reward | 65.69% | 69.84% |

### RQ3: Watermark Erasure Success Rate

| Evaluation Method | Average Attack Success Rate |
|---------|-------------|
| Automatic Evaluation (Average of all perturbation methods) | 26.1% |
| **After Human Quality Auditing** | **10.5%** |

- Adaptive watermarking is the most robust: Q-ASR after SentenceMutator attack is only 7.68%.
- SIR watermarking is the most fragile: WordMutator achieves an automatic ASR of 57.89%, which drops to only 2.89% after human auditing.

### Key Findings

1. **Extremely Slow Mixing**: In 4,555 tests, 100% of the attacked texts can still be traced back to their original source after hundreds of edits, directly refuting KA1.
2. **Unreliable Oracles**: The best oracle achieves an F1 score of only 77.4%, with nearly 1/5 of the perturbations being misclassified, and errors accumulating over multi-step attacks.
3. **Limited Attack Effectiveness**: Automatic attacks achieve only a 26% success rate, which drops to 10% after human auditing.
4. **Theoretical Impossibility $\neq$ Practical Impossibility**: Watermarking technology is far more robust than predicted by theoretical models.

## Highlights & Insights

- **Challenging Authoritative Theory**: Directly refuting the highly influential impossibility result with large-scale experiments.
- **Ingenious Experimental Design**: The Lineage Distinguisher Test serves as an elegant surrogate for verifying mixing speeds.
- **Human-Automatic Contrast**: Revealing a massive 16% gap between automatic evaluation and human judgment (26% vs. 10%).
- **Significant Practical Implications**: Restoring confidence in watermarking techniques and providing theoretical and empirical support for the continued development of watermark-based defenses.
- **"Verification $\neq$ Easy"**: Challenging the widespread consensus that "verifying quality is easier than generating content."

## Limitations & Future Work

- The models and watermarking schemes evaluated are limited and do not cover all frontier methods.
- The number of random walk steps is capped (1000 steps); whether longer attacks can eventually achieve mixing remains unknown.
- Improvements in quality oracles (e.g., employing stronger evaluation models) might alter the conclusions.
- The scale of human evaluation is limited (795 annotations), which may introduce annotation noise.
- More adversarial attackers (such as targeted attacks integrated with semantic understanding) are not analyzed.

## Related Work & Insights

- WITS (Zhang et al., 2024) is the theoretical work directly challenged by this paper.
- KGW (Kirchenbauer et al., 2023), SIR (Liu et al., 2024a), and Adaptive (Liu & Bu, 2024) are the three representative watermarks evaluated.
- Insight: In the field of AI safety, theoretical impossibility results must be re-examined under practical constraints.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The approach of driving experimental refutation of theoretical results is of great value.
- **Technical Depth**: ⭐⭐⭐⭐ — Rigorous experimental design with multi-dimensional and multi-layered validation.
- **Practical Utility**: ⭐⭐⭐⭐⭐ — Significant impact on the future development direction of watermarking technology.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely comprehensive with 7 perturbations $\times$ 3 watermarks $\times$ human evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Robust and Minimally Invasive Watermarking for EaaS](robust_and_minimally_invasive_watermarking_for_eaas.md)
- [\[NeurIPS 2025\] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] Provable Watermarking for Data Poisoning Attacks](../../NeurIPS2025/ai_safety/provable_watermarking_for_data_poisoning_attacks.md)
- [\[AAAI 2026\] Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden](../../AAAI2026/ai_safety/revisiting_unfairness_in_recourse_by_minimizing_worst-case_social_burden.md)
- [\[ICCV 2025\] SpecGuard: Spectral Projection-based Advanced Invisible Watermarking](../../ICCV2025/ai_safety/specguard_spectral_projection-based_advanced_invisible_watermarking.md)

</div>

<!-- RELATED:END -->
