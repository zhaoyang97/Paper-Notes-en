---
title: >-
  [Paper Note] Gap-K%: Measuring Top-1 Prediction Gap for Detecting Pretraining Data
description: >-
  [ACL2026][LLM Safety][Pretraining Data Detection] This paper proposes Gap-K%, which uses the normalized log probability gap between the target token and the model's top-1 prediction…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Pretraining Data Detection"
  - "Membership Inference"
  - "Top-1 gap"
  - "Min-K%"
  - "Sequential Smoothing"
date: 2026-05-08
content_hash: e81296fbd35da021
---

# Gap-K%: Measuring Top-1 Prediction Gap for Detecting Pretraining Data

**Conference**: ACL2026  
**arXiv**: [2601.19936](https://arxiv.org/abs/2601.19936)  
**Code**: https://github.com/meaoww/gap-k  
**Area**: LLM Security / Pretraining Data Detection / Membership Inference  
**Keywords**: Pretraining Data Detection, Membership Inference, Top-1 gap, Min-K%, Sequential Smoothing  

## TL;DR
This paper proposes Gap-K%, which uses the normalized log probability gap between the target token and the model's top-1 prediction, combined with sequential sliding window smoothing, to detect whether text appeared in LLM pretraining data. It outperforms baselines like Min-K%++ on WikiMIA, MIMIR, recent models, and under strong paraphrase attacks.

## Background & Motivation
**Background**: Pretraining corpora for large language models are typically not public. External researchers can only indirectly infer whether a piece of text has been trained on through model outputs. This issue concerns both privacy/copyright and benchmark contamination: if test sets have entered the pretraining corpus, model capability assessments will be overestimated.

**Limitations of Prior Work**: Most mainstream reference-free methods utilize token likelihood. Min-K% focuses on the $k\%$ tokens with the lowest probability, and Min-K%++ performs distribution normalization on token log probabilities. However, these methods essentially treat tokens as independent points and do not directly utilize the training dynamic signal of "whether the model's top-1 prediction equals the ground-truth token."

**Key Challenge**: The next-token objective of pretraining strongly penalizes cases where "the model confidently predicts another token while the ground-truth token is different." However, existing likelihood scores only look at the absolute magnitude of the ground-truth token probability, making it difficult to distinguish between "model uncertainty" and "model being confident but wrong." The former may simply be natural language diversity, while the latter is stronger evidence against the text being non-training data.

**Goal**: The authors aim to design a reference-free, grey-box detection method that only requires access to token probabilities. It should capture top-1 confident mispredictions while leveraging local correlations between adjacent tokens in the text.

**Key Insight**: The paper analyzes this from the perspective of cross-entropy gradients: the gradient magnitude of a non-target token's logit is proportional to its probability. If the top-1 token is not the ground-truth token, it generates the strongest suppression signal; in training samples, this top-1 gap should be optimized to be smaller.

**Core Idea**: The "gap between the ground-truth token log probability and the top-1 log probability" of each token is used as a membership signal. Sequential segments are aggregated using a sliding window, and finally, the average of the $k\%$ regions with the worst gaps is taken, similar to Min-K%.

## Method
The Gap-K% method is highly concise: it involves no detector training and requires no additional data distributions. It only reads the next-token probabilities of the target model for the input sequence. The key is replacing "low-probability tokens" with "tokens far from the top-1 prediction" and converting token-level fluctuations into local segment-level signals.

### Overall Architecture
Given an autoregressive LLM $\mathcal{M}$ and a text sequence $\mathbf{x}=[x_1,\ldots,x_N]$, the task is to determine if $\mathbf{x}$ belongs to an unknown training set $\mathcal{D}$. The method calculates the ground-truth token log probability, the vocabulary-wide top-1 log probability, and the standard deviation of the log probability distribution token-by-token. A normalized top-1 gap sequence is obtained, and a sliding window average of length $w$ is applied to this sequence. Finally, the lowest $k\%$ smoothed gaps are selected and averaged as the membership score. A score closer to 0 indicates that even in the most difficult-to-predict segments, the ground-truth token is close to the model's top-1 prediction, suggesting it is more likely to be training data.

### Key Designs
1. **Top-1 gap token score**:

	- Function: Converts "whether the model confidently deviates from the ground-truth token" into a token-level detection signal.
	- Mechanism: For each position $t$, $g_t=(\log p(x_t|x_{<t})-\max_{v\in V}\log p(v|x_{<t}))/\sigma_t$ is computed. This value is always $\le 0$; the closer to 0, the closer the ground-truth token is to top-1; the more negative, the more confidently the model favors other tokens.
	- Design Motivation: Min-K%++ measures anomalies by the ground-truth token's deviation from the mean, but two distributions might have the same mean deviation—one being "uniformly flat and uncertain" and the other being "top-1 very sharp but the ground-truth token is lagging." Gap-K% uses top-1 to directly characterize the latter "confident error."

2. **Sequential smoothing**:

	- Function: Converts noise from isolated tokens into membership evidence from continuous segments.
	- Mechanism: A window $w$ is used to average the gap sequence, $\bar g_t^{(w)}=\frac{1}{w}\sum_{i=0}^{w-1}g_{t+i}$. The paper sets $w=6$ for the LLaMA series and $w=3$ for other models.
	- Design Motivation: LLM memorization typically does not occur at the level of a single token but rather in continuous phrases or sections. If adjacent tokens all exhibit large gaps, it looks more like non-training text; if only a single token is anomalous, it might just be an incidental fluctuation of natural language.

3. **Bottom-k% aggregation**:

	- Function: Focuses on the local segments that most strongly refute membership, rather than averaging out strong signals.
	- Mechanism: Let $\tilde{\mathcal{I}}_k(\mathbf{x})$ be the positions of the lowest $k\%$ smoothed gaps; the final score is $\text{Gap-K}(\mathbf{x})=\frac{1}{|\tilde{\mathcal{I}}_k(\mathbf{x})|}\sum_{t\in\tilde{\mathcal{I}}_k(\mathbf{x})}\bar g_t^{(w)}$. For fair alignment with Min-K%++ in experiments, $k=20\%$ is used, with analysis performed in the $5\%-50\%$ range.
	- Design Motivation: Detection of training data relies on a few of the most anomalous segments. Averaging the entire sequence directly would be diluted by many common tokens; bottom-k aggregation preserves the most discriminative continuous low-gap regions.

### Loss & Training
Gap-K% itself does not require training and has no optimization loss. It is a reference-free, grey-box membership score that requires access to the target model's output logits or token probabilities. In experiments, the authors use AUROC as the primary metric and also report TPR@5%FPR; Min-K%, Min-K%++, and Gap-K% all use a fixed $k=20\%$ for fair comparison.

## Key Experimental Results

### Main Results
| Dataset / Setting | Metric | Gap-K% | Strongest Baseline / Comparison | Gain / Conclusion |
|--------|------|------|----------|------|
| WikiMIA length 32 original | Avg AUROC | 77.8 | Min-K%++ 75.7 | +2.1 |
| WikiMIA length 32 paraphrased | Avg AUROC | 74.3 | Min-K%++ 73.4 | +0.9 |
| WikiMIA length 64 original | Avg AUROC | 78.4 | Min-K%++ 75.8 | +2.6 |
| WikiMIA length 64 paraphrased | Avg AUROC | 71.2 | Min-K%++ 68.9 | +2.3 |
| WikiMIA length 128 original | Avg AUROC | 77.4 | Min-K%++ 74.8 | +2.6 |
| WikiMIA length 128 paraphrased | Avg AUROC | 70.6 | Min-K%++ 68.6 | +2.0 |
| MIMIR average, Pythia-12B | AUROC | 57.3 | Min-K%++ 57.1 | Slight lead in difficult near-distribution setting |
| WikiMIA-25, LLaMA 3.1 8B | AUROC | 84.1 | Min-K%++ 82.7 | Effective on recent models |
| WikiMIA-25, LLaMA 3.1 8B Instruct | AUROC | 76.6 | Min-K%++ 73.1 | Remains effective after instruction tuning |
| DIPPER paraphrase attack | AUROC | 66.6 | Min-K%++ 65.5 / Neighbor 60.3 | Best under strong paraphrase |

### Ablation Study
| Configuration | Key Metric | Description |
|------|---------|------|
| No smoothing | AUROC 72.3 | Uses raw token gaps, high fluctuation |
| Shuffled-order smoothing | AUROC 72.9 | Smoothing after shuffling token order yields little gain |
| Sequential smoothing | AUROC 74.8 | Significant gain when preserving order, showing membership signals have local continuity |
| Min-K%++ | AUROC 72.6 | Original mean-normalized likelihood baseline |
| + Top-1 only | AUROC 72.3 | Replacing with top-1 gap alone is insufficient |
| + Smoothing only | AUROC 73.8 | Smoothing also helps Min-K%++ |
| Gap-K% full | AUROC 74.8 | Combination of top-1 gap and sequential smoothing is most effective |
| Gap magnitude threshold $\tau=3$ | Train 35.53% vs Non-train 39.94% | Non-training data has more large-gap tokens, supporting the core hypothesis |

### Key Findings
- On WikiMIA, Gap-K% consistently outperforms Min-K%++ on both original and paraphrased inputs, indicating that top-1 gap does not just exploit verbatim memory but retains signals for rewritten text.
- On MIMIR, while all methods are close to random guessing, Gap-K% remains the strongest or tied for strongest on average across 1.4B, 2.8B, 6.9B, and 12B models, showing it does not fail in harder near-distribution detection scenarios.
- Gains are more pronounced in TPR@5%FPR: the paper reports improvements over Min-K%++ by 7.1%, 7.9%, and 3.0% in WikiMIA original length 32, 64, and 128 settings, respectively.
- Sensitivity analysis for $k$ shows performance peaks near $k=15\%$, though Gap-K% consistently outperforms Min-K% and Min-K%++ across the $5\%-50\%$ range.

## Highlights & Insights
- **Detection signal explained via training gradients**: Rather than simply changing a heuristic, the paper explains from cross-entropy gradients that top-1 errors are heavily penalized during training, so training samples should exhibit fewer large top-1 gaps.
- **Critical to distinguish "uncertainty" from "confident error"**: When the ground-truth probabilities for two tokens are both low, Min-K%++ might give similar scores; Gap-K% specifically penalizes cases where the top-1 is very sharp but not the ground-truth token, which serves as stronger evidence of non-membership.
- **Sequential smoothing transforms membership from point signals to segment signals**: This aligns well with the intuition of text memorization—models typically memorize phrases or passages rather than isolated tokens.
- **Simple and plug-and-play**: It requires only logits, does not need a reference model, and does not require access to the training data distribution; as such, it can serve as a direct replacement or supplement for Min-K%++.

## Limitations & Future Work
- The method requires grey-box access, meaning token-level probabilities or logits must be available. Many commercial APIs do not expose this information, preventing direct use on fully black-box models.
- Although evaluation covers LLaMA 3.1, Gemma2, and instruction-tuned versions, it does not cover more recent model families or models at the hundreds of billions of parameters scale.
- DIPPER paraphrase is a strong attack, but it is not yet a detector-aware adaptive attack. If an attacker knows the Gap-K% mechanism, they might optimize text specifically to manipulate the top-1 gap and local statistics.
- Absolute AUROC on MIMIR remains low, indicating a clear ceiling for likelihood-based signals when training and non-training distributions are highly similar and memorization signals are weak.

## Related Work & Insights
- **vs Min-K%**: Min-K% averages the lowest probability tokens, which is simple but lacks distribution calibration and does not consider whether the model is confidently wrong. Gap-K% focuses on the distance between ground-truth and top-1.
- **vs Min-K%++**: Min-K%++ normalizes ground-truth likelihood using mean and standard deviation; Gap-K% replaces the mean with the mode/top-1, more directly corresponding to the hypothesis that "training suppresses confident errors."
- **vs reference-based MIA**: Reference-based methods require training additional models on similar distributions, which is costly and unsuitable for detecting closed-source pretraining corpora. Gap-K% maintains a reference-free setup with low deployment barriers.
- **Insight**: Many LLM security detection signals can be derived from training objective gradient dynamics rather than just finding empirical patterns in surface probability statistics. Future work could further combine top-1 gap, entropy, local repetition, and semantic paraphrase robustness.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Clean replacement of a core signal within the Min-K family with a training dynamics explanation; simple framework that captures key differences.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers WikiMIA, MIMIR, recent models, paraphrasing, and component ablations; lacks fully black-box and adaptive attack settings.
- Writing Quality: ⭐⭐⭐⭐☆ Clear correspondence between formulas and intuition; ablation design is direct; MIMIR tables are large but conclusions are clear.
- Value: ⭐⭐⭐⭐☆ Practical value for pretraining data detection, copyright/privacy auditing, and benchmark contamination checks, particularly as a lightweight grey-box baseline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Beyond Procedure: Substantive Fairness in Conformal Prediction](../../ICML2026/llm_safety/beyond_procedure_substantive_fairness_in_conformal_prediction.md)
- [\[AAAI 2026\] Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach](../../AAAI2026/llm_safety/uncovering_pretraining_code_in_llms_a_syntax-aware_attribution_approach.md)
- [\[AAAI 2026\] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](../../AAAI2026/llm_safety/ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [\[ACL 2026\] Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game](detecting_rag_extraction_attack_via_dual-path_runtime_integrity_game.md)
- [\[ICLR 2026\] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](../../ICLR2026/llm_safety/measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)

</div>

<!-- RELATED:END -->
