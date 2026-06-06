---
title: >-
  [Paper Note] Detecting Fluent Optimization-Based Adversarial Prompts via Sequential Entropy Changes
description: >-
  [ICML 2026][Model Compression][Adversarial Suffix] The authors model "fluent optimization-based jailbreak suffix detection" as online change-point detection on token-level entropy streams. By using the entropy distributi…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Adversarial Suffix"
  - "Page-CUSUM"
  - "Token Entropy"
  - "System Prompt Baseline"
  - "Jailbreak Localization"
date: 2026-05-08
content_hash: 77a0ffdf8be39d2e
---

# Detecting Fluent Optimization-Based Adversarial Prompts via Sequential Entropy Changes

**Conference**: ICML 2026  
**arXiv**: [2605.19966](https://arxiv.org/abs/2605.19966)  
**Code**: https://github.com/cpdonline/cpdonline (Available)  
**Area**: LLM Security / Jailbreak Detection / Online Change Point Detection  
**Keywords**: Adversarial Suffix, Page-CUSUM, Token Entropy, System Prompt Baseline, Jailbreak Localization  

## TL;DR
The authors model "fluent optimization-based jailbreak suffix detection" as online change-point detection on token-level entropy streams. By using the entropy distribution of fixed system prompts to calculate a MAD robust baseline for standardizing user token entropy, a Page-CUSUM cumulative statistic $W_t^+$ is run to trigger alarms upon threshold crossing. Across 6 open-source aligned LLMs, this method achieves higher F1 scores than windowed perplexity for five attack types (GCG, AutoDAN, AdvPrompter, BEAST, and AutoDAN-HGA), accurately localizes 79.6% of alarms within suffixes, and serves as a lightweight gate for LLaMA Guard, reducing guard calls by 17-42%.

## Background & Motivation

**Background**: Current runtime defenses for LLM jailbreaking are categorized into two types: (1) Statistical detectors: calculating global perplexity (PP) or windowed perplexity (WPP) as anomaly scores; (2) Safety classifiers: using fine-tuned LLMs like LLaMA Guard to determine "unsafe" status. The former is lightweight but relies only on scalar statistics; the latter has high accuracy but requires an additional LLM forward pass.

**Limitations of Prior Work**: New generation attacks following GCG (AutoDAN, AdvPrompter, BEAST, AutoDAN-HGA) explicitly optimize for "low perplexity/fluency." Consequently, the AUROC of global PP between benign and adversarial inputs collapses to a range around 0.5 ($\pm 0.04$) across six models—rendering threshold adjustments ineffective. WPP improves on this by capturing local spikes via maximum window NLL, but the optimal window size $w$ depends heavily on the model (e.g., $w=15$ for LLaMA-2-7B vs. $w=1$ for Vicuna-7B/Qwen2.5-7B). Larger windows also smear boundaries by averaging adversarial loss with benign context.

**Key Challenge**: The characteristic of fluent adversarial suffixes is not "high absolute perplexity," but rather "continuously driving up model uncertainty across the token stream." This represents a mean shift in the temporal dimension, but both PP and WPP collapse sequences into scalars or local means, losing the critical signal of "drift persistency."

**Goal**: (a) To propose a model-agnostic, training-free, and pure forward-pass bypass method for online adversarial suffix detection; (b) to provide token-level localization of the suffix start (a task PP/WPP struggle with); (c) to integrate with expensive classifiers like LLaMA Guard as a gating pipeline to reduce guard call rates.

**Key Insight**: The authors observe that for each request, the token entropy distribution of the fixed system prompt remains stable within a given deployment. While benign user input distributions are similar, optimization-based suffixes introduce a "persistent upward mean shift" in the user segment—exactly what the 1954 Page-CUSUM control chart is designed to detect as the "fastest online detection of persistent mean shifts."

**Core Idea**: Treat token-level next-token entropy streams as a 1D time series. Use the system prompt to estimate a robust baseline $(\hat\mu_0, \hat\sigma_0)$ to standardize the user segment into $Z_t$. Run a one-sided Page-CUSUM cumulative statistic $W_t^+$ and trigger an alarm when it exceeds threshold $h$. The suffix start position $\hat\nu$ can be backtracked using CUSUM rules.

## Method

### Overall Architecture
Each request consists of a fixed system prompt $\mathbf{x}^{\text{sys}}$ concatenated with a user message $\mathbf{x}^{\text{usr}}$. During a standard forward pass, the entropy $H_t = -\sum_v p_\theta(v|x_{<t}) \log p_\theta(v|x_{<t})$ of the next-token distribution $p_\theta(\cdot | x_{<t})$ at each position is extracted as a free byproduct. A deployment-level baseline is estimated using $\{H_i^{\text{sys}}\}$, user entropy $\{H_t^{\text{usr}}\}$ is standardized to $Z_t$, and the one-sided Page CUSUM $W_t^+ = \max\{0, W_{t-1}^+ + Z_t - k\}$ is computed. An alarm triggers if $W_t^+ \geq h$ at any time, and $s(\mathbf{x}^{\text{usr}}) = \max_t W_t^+$ is taken as the prompt-level score for ROC calculation. The pipeline is $O(1)$ per token and $O(T)$ per prompt with constant memory overhead, allowing direct integration into production inference paths.

### Key Designs

1. **System Prompt Self-Calibrating Robust Baseline $(\hat\mu_0, \hat\sigma_0)$**:
    - **Function**: Automatically estimates the "no-attack" location and scale of token entropy for each deployment, eliminating the need for offline training or dataset preparation.
    - **Mechanism**: While entropy magnitude is tightly coupled with model size, tokenizer, and system prompt phrasing, the system prompt is static within a deployment. Its $m$ token entropies serve as reference samples. Robust location and scale are estimated using median and MAD: $\hat\mu_0 = \text{median}(\{H_i^{\text{sys}}\})$, $\hat\sigma_0 = c \cdot \text{median}(|H_i^{\text{sys}} - \hat\mu_0|)$, where $c \approx 1.4826$ aligns MAD with Gaussian-$\sigma$. $Z_t = (H_t^{\text{usr}} - \hat\mu_0)/\hat\sigma_0$ is then computed, with $\hat\sigma_0 \geq \varepsilon$ to prevent degeneracy.
    - **Design Motivation**: MAD is chosen over mean/variance because a few tokens in system prompts can have high entropy due to specific words; these statistics are resilient to extreme values. Calibration with $c$ allows the same threshold to be used across LLaMA/Vicuna/Qwen, achieving "model-agnostic" CPD.

2. **One-Sided Page-CUSUM for Drift Detection $W_t^+$**:
    - **Function**: Accentuates the "persistent positive drift" in $\{Z_t\}$, avoiding distraction by instantaneous spikes or local means common in WPP.
    - **Mechanism**: Iteratively compute $W_t^+ = \max\{0, W_{t-1}^+ + Z_t - k\}$ with slack $k \geq 0$ and threshold $h > 0$, starting at $W_0^+ = 0$. The stopping time is defined as $\tau = \inf\{t \geq 1 : W_t^+ \geq h\}$. When the mean of $\{Z_t\}$ is near zero, $W_t^+$ resets repeatedly, preventing noise accumulation; a persistent positive drift causes monotonic accumulation until $h$ is crossed. Canonical $k=0$ is used in main experiments, with $h$ selected to maximize F1 on training folds.
    - **Design Motivation**: Page-CUSUM is an optimal sequential test for "fast detection of persistent mean shifts." Unlike windowed detection, it requires no preset window scale—adapting naturally to suffixes of varying lengths. Slack $k$ acts against noise; while $k=-0.5$ improves F1, $k=0$ is used for adherence to classical theory.

3. **CUSUM Backtracking Localization $\hat\nu$ + LLaMA Guard Hybrid Gating**:
    - **Function**: Provides the suffix start position upon alarm and uses CPD as a gate to save expensive guard calls.
    - **Mechanism**: Localization uses standard CUSUM backtracking—recording the last reset time $t_0$ where $W_t^+ = 0$, then $\hat\nu = t_0 + 1$, identifying the token where the drift began after the stream last "settled." Gating uses a threshold $\tau_{\text{gate}}$: if $s(\mathbf{x}^{\text{usr}}) < \tau_{\text{gate}}$, the input is judged benign and skips the guard; otherwise, LLaMA Guard is called for semantic judgment.
    - **Design Motivation**: Localization is a nearly free byproduct of CUSUM, useful for automated suffix clipping or highlighting suspicious segments for security teams. Hybrid gating leverages the fact that over 90% of production loads are benign, reducing LLaMA Guard calls from "every request" to "only suspicious requests," saving 17-42% of calls without dropping hybrid F1.

### Loss & Training
The method is training-free and requires no gradient updates. The only "parameter tuning" is the selection of threshold $h$, which is done using 5-fold stratified CV (stratified by attack family) to maximize F1 on training folds. All token entropies are extracted directly from standard base LLM forward passes without additional networks.

## Key Experimental Results

### Main Results
Perplexity matching benchmark ($\alpha=1$; 1012 adversarial + 1012 benign, 5-fold stratified CV); CPD uses canonical $k=0$. F1 / AUROC:

| Model | PP AUROC | Best WPP F1 / AUROC | CPD F1 / AUROC |
|------|----------|--------------------|----------------|
| LLaMA-2-7B | 0.46 | 0.74 / 0.77 (WPP15) | **0.82 / 0.88** |
| LLaMA-2-13B | 0.49 | 0.74 / 0.78 (WPP10) | **0.80 / 0.87** |
| Vicuna-7B | 0.50 | 0.77 / **0.85** (WPP1) | 0.77 / 0.82 |
| Vicuna-13B | 0.51 | 0.77 / 0.84 (WPP10) | **0.80 / 0.85** |
| Qwen2.5-7B | 0.51 | 0.83 / 0.91 (WPP1) | **0.85 / 0.91** |
| Qwen2.5-14B | 0.50 | 0.80 / 0.85 (WPP10) | **0.85 / 0.91** |

PP AUROCs all collapse around 0.5—an expected structural outcome, validating that individual PP thresholds cannot distinguish inputs after PP-matching. CPD leads in F1 across all 6 models (margins +0.001 to +0.08) and leads or ties in AUROC for 5 models, with WPP1 (per-token max-NLL) only slightly winning on Vicuna-7B due to its low benign entropy variance.

### Ablation Study
Ablation on "Signal × Mechanism" on LLaMA-2-7B with $k=0$:

| Mechanism | Signal | F1 | AUROC |
|------|------|----|-------|
| CUSUM | NLL | 0.874 | 0.918 |
| CUSUM | Entropy | 0.818 | 0.878 |
| Window $w=1$ | NLL | 0.734 | 0.783 |
| Window $w=1$ | Entropy | 0.699 | 0.706 |

The primary conclusion is that "mechanism is more important than signal." Switching to CUSUM (using either NLL or entropy) results in an F1 ~12-14 points higher than windowed detection. While NLL+CUSUM outperforms entropy+CUSUM by ~6 F1 points, the paper focuses on entropy because it permits self-calibration via system prompt MAD baselines, which NLL lacks.

### Key Findings
- **Superior Localization Accuracy**: At F1-optimal thresholds for LLaMA-2-7B, CPD accurately places 79.55% of alarms inside the suffix, whereas WPP1-WPP20 only achieve 17-46%. CPD scores 0% in the "before+in" (boundary crossing) category compared to WPP's 12.9-37.9%, while maintaining the lowest benign false alarms (20.45% vs. WPP 33.7-45.1%). This is a natural advantage of Page-CUSUM, where each $W^+$ reset marks the moment just before drift starts.
- **Robustness across 5 Attack Families**: Against GCG (200), AutoDAN (200), AdvPrompter (312), BEAST (100), and AutoDAN-HGA (200) evaluated together, CPD ranked first or second in F1 across all six base LLMs, indicating that "persistent entropy drift" is a shared characteristic of optimization-based suffixes rather than an artifact of a specific attack family.
- **High Utility of Hybrid Gating**: On an imbalanced deployment stream of 17,297 items (4.2% attack ratio), using CPD as a gate for LG1 saves 42.2% of guard calls while maintaining a hybrid F1 of 0.82; for LG2, it saves 33.8% while maintaining F1 = 0.73. Under the same F1, WPP only saves 13-18%. This translates to significant cost savings in production by halving LLaMA Guard forward passes.

## Highlights & Insights
- **Classic Control Charts for Deep Learning Defense**: Applying the 1954 Page-CUSUM to LLM entropy streams outperforms ad-hoc window heuristics. This suggests classic sequential analysis tools are undervalued in the LLM era; attacks characterized by "persistent bias" (deceptive alignment, context poisoning, long-horizon manipulation) should be re-evaluated using tools like CUSUM/EWMA.
- **System Prompt as a Gift**: Transforming the system prompt from a "fixed non-user overhead" into a "free deployment-level calibration sample" is an elegant insight that can be applied to any runtime detector relying on normal behavior distributions.
- **Unified Detection and Localization**: CUSUM backtracking for $\hat\nu$ is a zero-cost byproduct with high utility for automated suffix clipping, explainable security auditing, and attack forensics. Prompt-level classifiers like LLaMA Guard cannot provide this "event-level" granularity.

## Limitations & Future Work
- **Append-only Suffix Assumption**: The method assumes attacks are appended after basic user tasks. It is inapplicable to "persuasive rewriting" jailbreaks (Zeng et al., 2024), where the entire request is modified, leaving no prefix for baseline reference.
- **System Prompt Stability**: Dynamic system prompts or multi-turn contexts may introduce instability or overhead if $(\hat\mu_0, \hat\sigma_0)$ must be re-estimated; baseline estimation in multi-turn scenarios remains an open problem.
- **Entropy vs. NLL Signal Choice**: While NLL+CUSUM achieves higher F1, entropy+CUSUM allows for self-calibration. Future work could bridge this gap by developing NLL proxies from system prompt entropy.
- **Threshold Optimization**: Although $h$ is a single scalar, it currently requires sweeping on training data. Developing unsupervised or weakly-supervised methods to determine $h$ in real-world deployments is a key challenge for adoption.
- **Adaptive Attacks**: Adversaries aware of CPD may optimize suffixes to suppress entropy increases (e.g., CPD-aware GCG), representing a likely next step for adversarial research.

## Related Work & Insights
- **vs. PP / WPP (Jain et al. 2023, Alon-Kamfonas 2023)**: PP/WPP use scalar thresholds assuming adversarial inputs have high perplexity. CPD uses sequential mean shifts, proving robust against fluent attacks; mechanism ablation shows that CUSUM vs. Windowing is the dominant factor.
- **vs. LLaMA Guard (Inan et al. 2023)**: LLaMA Guard is a powerful but expensive supervised classifier. CPD acts as a complementary lightweight gate, implementing a "lightweight statistical + heavyweight semantic" two-tier defense paradigm.
- **vs. SPD (Candogan et al. 2025)**: CPD offers advantages in online processing and being entirely training-free.
- **vs. Safety Fine-tuning / RLHF**: These are training-time defenses, whereas CPD is an orthogonal inference-time defense. CPD remains valuable even when alignment is bypassed.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First formal application of Page-CUSUM for LLM jailbreak detection with a precise mapping to sequential analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extensive evaluation across 6 LLMs, 5 attack families, perplexity matching, signal-mechanism ablation, imbalanced streams, and OOD scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ Clear derivations and compelling results on localization and hybrid gating that explain the method's unique value beyond F1 improvements.
- **Value**: ⭐⭐⭐⭐⭐ Training-free, $O(T)$ online operation, and seamless integration with existing guard pipelines to reduce costs by 30%+ makes this a high-impact research work for production deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ICML 2026\] Toward Understanding Adversarial Distillation: Why Robust Teachers Fail](toward_understanding_adversarial_distillation_why_robust_teachers_fail.md)
- [\[ICML 2026\] Efficient Learned Image Compression without Entropy Coding](efficient_learned_image_compression_without_entropy_coding.md)
- [\[ICML 2026\] Float8@2bits: Entropy Coding Enables Data-Free Model Compression](float82bits_entropy_coding_enables_data-free_model_compression.md)
- [\[ICLR 2026\] Boosting Entropy with Bell Box Quantization](../../ICLR2026/model_compression/boosting_entropy_with_bell_box_quantization.md)

</div>

<!-- RELATED:END -->
