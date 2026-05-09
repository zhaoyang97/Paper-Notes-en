---
title: >-
  [Paper Note] Reject Only Critical Tokens: Pivot-Aware Speculative Decoding
description: >-
  [NeurIPS 2025][Model Compression][speculative decoding] PAD proposes a new speculative decoding paradigm based on utility matching rather than distribution matching. It trains a lightweight classifier to identify *pivot tokens* and rejects only those draft tokens that would degrade final output utility, achieving a 2.46× speedup on GSM8K with negligible accuracy loss.
tags:
  - NeurIPS 2025
  - Model Compression
  - speculative decoding
  - pivot token
  - utility preservation
  - LLM inference
  - acceptance rate
date: 2026-05-08
content_hash: cbe4d723d7032f85
---

# Reject Only Critical Tokens: Pivot-Aware Speculative Decoding

**Conference**: NeurIPS 2025
**arXiv**: [2511.00351](https://arxiv.org/abs/2511.00351)
**Code**: [https://github.com/amir-zsh/PAD](https://github.com/amir-zsh/PAD)
**Area**: Model Compression
**Keywords**: speculative decoding, pivot token, utility preservation, LLM inference, acceptance rate

## TL;DR

PAD proposes a new speculative decoding paradigm based on utility matching rather than distribution matching. It trains a lightweight classifier to identify *pivot tokens* and rejects only those draft tokens that would degrade final output utility, achieving a 2.46× speedup on GSM8K with negligible accuracy loss.

## Background & Motivation

Autoregressive generation in large language models proceeds token by token serially; as model scale grows, generation speed becomes a critical bottleneck. Speculative Decoding (SD) accelerates inference by having a small draft model generate a batch of candidate tokens that are then verified in parallel by a large target model. However, its core constraint—requiring the output to strictly match the target model's sampling distribution—causes a large number of draft tokens to be unnecessarily rejected, limiting the speedup.

The acceptance probability in SD is $\min(1, \frac{p_{\text{target}}(x)}{p_{\text{draft}}(x)})$, meaning that any distributional discrepancy between draft and target leads to rejections, even when those tokens have no impact on final output quality.

**Key Challenge**: In practice, users care about output utility (e.g., code correctness, mathematical answer accuracy) rather than sampling distributions. The example in Figure 1 clearly illustrates this: SD rejects a large number of tokens (highlighted in blue), yet only a single token correction (2→1) is needed to obtain the correct answer.

**Key Insight**: This paper relaxes the SD optimization objective from *matching the target distribution* to *matching the target utility*, rejecting only those tokens that genuinely cause utility degradation (pivot tokens), while accepting all other tokens regardless of whether standard SD would reject them.

## Method

### Overall Architecture

The PAD (Pivot-Aware Speculative Decoding) pipeline:
1. The draft model generates γ candidate tokens.
2. The target model verifies them in parallel (identical to standard SD).
3. For tokens that SD would reject, the pivot classifier is additionally queried.
4. If the classifier determines the token is a non-pivot (score < σ), the rejection decision is overridden and the token is accepted.
5. If the classifier determines the token is a pivot, the standard SD rejection procedure is followed.

### Key Designs

1. **Utility-Matching Objective (ε-Utility Preserving Decoding)**:
   - Utility function defined as $u(y,x) = \mathbb{1}[\text{Eval}(y,x) \geq \theta_{\text{eval}}]$ (binarized: correct = 1, incorrect = 0).
   - Objective: $\mathbb{E}[U(\hat{p}, x_c)] \geq \mathbb{E}[U(p_{\text{target}}, x_c)] - \epsilon$.
   - This is a strictly weaker constraint than distribution matching, permitting more tokens to be accepted.

2. **Pivot Token Definition**:
   - Formal definition: token $\tilde{y}_t$ is a pivot if and only if accepting it causes the expected utility of subsequent target-model continuations to drop significantly:
     $$U(p_{\text{target}}, (x_c, y_{<t}, \tilde{y}_t)) \leq U(p_{\text{target}}, (x_c, y_{<t})) - \epsilon$$
   - Intuitively, a pivot token steers the generation trajectory into a low-utility region.

3. **Pivot Classifier Training (Data & Features)**:
   - **Candidate harvesting**: Only tokens that SD would reject are annotated, focusing on the rejection boundary.
   - **Monte Carlo rollout estimation**: For each candidate token, the target model generates $N$ independent continuations; the mean utility $\hat{U}_{\text{cand}}$ and baseline $\hat{U}_{\text{base}}$ are computed. A tolerance α is introduced: when $\hat{U}_{\text{cand}} < \alpha \hat{U}_{\text{base}}$, the token is labeled pivot.
   - **LLM-as-Judge safety check**: For candidates labeled non-pivot, rollouts that are "correct but with questionable reasoning" are sampled and evaluated by an LLM for reasoning soundness; if unsound, the label is flipped to pivot. Crucially, this flip is unidirectional—labels can only be upgraded to pivot, never introducing false acceptances.
   - **Features**: Hidden states from layer ℓ of the target model, target probability, and entropy of the target distribution.
   - **Model**: A small MLP classifier with negligible computational overhead.

4. **Safety Threshold**: Tokens with target probability below $10^{-4}$ are unconditionally rejected regardless of classifier output.

### Loss & Training

- The pivot classifier is a small binary MLP (pivot vs. non-pivot).
- Training data are generated automatically via Monte Carlo rollout, requiring no manual annotation.
- Hyperparameter σ controls the acceptance threshold: larger σ accepts more tokens (faster but potentially less accurate); smaller σ is more conservative.

## Key Experimental Results

### Main Results

| Setting | GSM8K Acc. | η(%) | Speedup | AIME24 Acc. | η(%) | Speedup | MBPP Acc. | η(%) | Speedup |
|---------|-----------|------|---------|-------------|------|---------|----------|------|---------|
| Target | 94±0.6 | — | 1.00 | 73±4.5 | — | 1.00 | 70±1.9 | — | 1.00 |
| SD | 94±0.6 | 45.3 | 1.57 | 73±4.5 | 47.2 | 1.69 | 70±1.9 | 41.8 | 1.46 |
| PAD(σ=0.5) | 93.4±0.9 | 70.8 | **2.33** | 61.6±5.3 | 71.6 | **2.33** | 68.6±2.3 | 61.7 | **2.00** |
| PAD(σ=0.3) | 93.7±1.1 | 58.2 | 1.95 | 69.6±4.2 | 58.3 | 1.95 | 68.3±4.8 | 50.2 | 1.71 |
| Draft | 74.2±1.5 | — | 3.94 | 12.5±3.4 | — | 3.94 | 51.1±1.3 | — | 3.94 |

### Ablation Study: Effect of Threshold σ

| σ | GSM8K Acc. | GSM8K Speedup | AIME24 Acc. | AIME24 Speedup | Trend |
|---|-----------|--------------|-------------|---------------|-------|
| 0.7 | 93 | 2.46 | 57 | 2.51 | Aggressive: fastest but reduced accuracy on hard tasks |
| 0.5 | 93.4 | 2.33 | 61.6 | 2.33 | Balanced: best trade-off between speed and accuracy |
| 0.3 | 93.7 | 1.95 | 69.6 | 1.95 | Conservative: accuracy close to target but limited speedup |

### Key Findings

- **GSM8K / MBPP** (easy tasks): PAD achieves a 2.33–2.46× speedup with negligible accuracy loss, far exceeding SD's 1.46–1.57×.
- **AIME24** (hard math competition): Maintaining high accuracy requires more conservative σ, indicating that a larger fraction of tokens are pivots in difficult tasks.
- The acceptance rate η increases from 45.3% (SD) to 70.8% (PAD, σ=0.5), revealing that a substantial portion of tokens rejected by SD are in fact safe to accept.
- The overhead of the pivot classifier is negligible: a single MLP forward pass is orders of magnitude cheaper than a Transformer forward pass through the target or draft model.

## Highlights & Insights

- **Elegant problem reformulation**: Redefining SD from *distribution matching* to *utility matching* represents a fundamental rethinking of the speculative decoding paradigm.
- **Theoretical guarantee**: Lemma 1 proves that as long as the classifier achieves 100% recall on pivot tokens, PAD fully preserves utility. This provides the theoretical foundation for relaxing SD.
- **Self-supervised data generation**: Pivot/non-pivot labels are automatically generated via Monte Carlo rollout without manual annotation. The LLM-as-Judge mechanism ensures that label flips only go in the safe direction (non-pivot → pivot), preventing false acceptances.
- **Orthogonal to existing methods**: PAD modifies the verification stage and is therefore complementary to methods that improve the draft model (e.g., EAGLE, Medusa) or align draft quality (e.g., DistSpec).

## Limitations & Future Work

- The pivot classifier requires task-specific rollout data for training; retraining is needed when the target task changes.
- On AIME24 with σ=0.7, accuracy drops by approximately 16% (73→57), demonstrating that classifier precision is critical for challenging reasoning tasks.
- The utility function depends on task-specific definitions (e.g., code correctness evaluated via test cases) and is less applicable to open-ended generation tasks (e.g., creative writing) where utility is not clearly defined.
- Experiments are conducted solely on the Qwen3 model family; generalization to other LLM families remains unverified.
- Rollout data generation requires multiple calls to the target model, resulting in non-trivial computational cost during training.

## Related Work & Insights

- **Complementary to EAGLE/Medusa** (which improve draft strategies): those methods optimize the draft side, while PAD optimizes the verification side.
- **Distinction from Bachmann et al. 2025** (which trains a classifier to accept/reject tokens): PAD grounds decisions in utility rather than heuristic rules, and data generation is entirely self-supervised.
- **Complementary to DistSpec** (which aligns draft and target distributions): DistSpec improves draft quality to raise η; PAD relaxes the acceptance criterion to raise η.
- **Broader insight**: Optimization in speculative decoding should not be confined to making the draft more similar to the target; instead, one should ask *which token discrepancies are tolerable*.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Traversal Verification for Speculative Tree Decoding](traversal_verification_for_speculative_tree_decoding.md)
- [\[NeurIPS 2025\] CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs](casspec_cascade_adaptive_selfspeculative_decoding_for_onthef.md)
- [\[AAAI 2026\] Steering Pretrained Drafters during Speculative Decoding](../../AAAI2026/model_compression/steering_pretrained_drafters_during_speculative_decoding.md)
- [\[NeurIPS 2025\] AutoJudge: Judge Decoding Without Manual Annotation](autojudge_judge_decoding_without_manual_annotation.md)
- [\[ACL 2026\] Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](../../ACL2026/model_compression/calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)

<!-- RELATED:END -->
