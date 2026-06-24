---
title: >-
  [Paper Note] Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test
description: >-
  [ICLR 2026][LLM Safety][Black-box Auditing] Addressing the issue where API providers might stealthily replace claimed models with quantized, fine-tuned, or jailbroken versions, this paper proposes the **Rank-based Uniformity Test (RUT)**. By querying the target API only once per prompt and performing multiple samplings on a local reference model, the method maps the log-rank score of the API output to its "percentile rank within the reference distribution." If the two models…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "Black-box Auditing"
  - "Model Equivalence Testing"
  - "Rank-based Uniformity"
  - "Cramér–von Mises"
  - "Quantization Detection"
date: 2026-05-08
content_hash: 411715b4727b28c9
---

# Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PSIe9mmF7a](https://openreview.net/forum?id=PSIe9mmF7a)  
**Code**: https://github.com/xyzhu123/RUT (Available)  
**Area**: LLM Safety / Model Auditing  
**Keywords**: Black-box Auditing, Model Equivalence Testing, Rank-based Uniformity, Cramér–von Mises, Quantization Detection

## TL;DR
Addressing the issue where API providers might stealthily replace claimed models with quantized, fine-tuned, or jailbroken versions, this paper proposes the **Rank-based Uniformity Test (RUT)**. By querying the target API only once per prompt and performing multiple samplings on a local reference model, the method maps the log-rank score of the API output to its "percentile rank within the reference distribution." If the two models are identical, these ranks should follow a uniform distribution. The deviation is then detected using the Cramér–von Mises test, achieving high detection power with only one API call per prompt and query profiles that resemble ordinary user traffic, making it difficult for adversarial providers to identify and bypass.

## Background & Motivation

**Background**: APIs have become the primary entry point for accessing large models, but users face a black box—weights are inaccessible, and in most cases, even output logits are provided only as text (sometimes with top-5 token log-probabilities). Driven by pressures to reduce costs and lower Time to First Token (TTFT), providers are incentivized to stealthily deploy smaller or quantized variants.

**Limitations of Prior Work**: Such replacements are completely opaque to users but can degrade performance and introduce safety risks. In more severe cases, providers might attach jailbreak system prompts, perform harmful fine-tuning, or unintentionally connect misconfigured components. This necessitates **LLM API auditing** to verify if the deployed model is as claimed. Existing methods have various shortcomings (Table 1): training text classifiers (Sun et al.) requires a massive number of API queries; identity questioning (Huang et al.) fails to capture subtle differences like size, version, or quantization; and MMD (Gao et al.) or benchmark comparisons depend on **specialized query distributions**. These specialized distributions can be recognized by sophisticated providers, who may then reroute suspicious prompts back to the genuine model or use prompt caching to bypass auditing.

**Key Challenge**: A good auditing test must simultaneously meet three criteria: **accuracy** (high detection power), **query efficiency** (low API cost to encourage frequent use), and **adversarial robustness** (avoiding identification by providers while maintaining detection power under probabilistic substitution attacks). No existing method achieves all three.

**Formalization of Adversarial Threats**: Sophisticated providers use **probabilistic substitution** to evade detection. The target model can be modeled as a mixture of the reference model and an alternative model:

$$\pi_{\text{tgt}}(\cdot\mid x;\varphi) = \bigl(1-q(x)\bigr)\,\pi_{\text{ref}}(\cdot\mid x;\varphi) + q(x)\,\pi_{\text{alt}}(\cdot\mid x;\varphi)$$

where $\pi_{\text{alt}}$ is any alternative model (quantized, harmfully fine-tuned, etc.), and the routing function $q:\mathcal{X}\to[0,1]$ is unknown and may depend on the prompt or even interaction history (e.g., routing based on prompt caching). Even when $q(x)$ is small but applied to a non-negligible proportion of regular prompts, the mixture distribution is extremely difficult to distinguish, yet a qualified detector must still maintain statistical power.

**Core Idea**: The paper formalizes auditing as **model equivalence testing**: given query access to a target API and a fully accessible reference model, determine if the two produce statistically indistinguishable outputs on shared prompts. The key observation of RUT is to assign a scalar score to API outputs using the local reference model and check its percentile rank in the reference distribution. **If the two models are identical, this percentile must follow a uniform distribution.** Thus, "whether the two models are equal" is transformed into a clean 1D uniformity test on a sequence of ranks (Uniform[0,1]).

## Method

### Overall Architecture

RUT is an **asymmetric two-sample hypothesis test**. It is asymmetric because it uses the two models differently: the target API is expensive and yields only one output per prompt, so **the target is queried only once per prompt**. The reference model is local and allows infinite sampling, so **$m$ samples are drawn from the reference model per prompt** to approximate the reference distribution. The process involves taking a batch of prompts resembling real user traffic (WildChat in experiments), obtaining 1 response from the target and $m$ responses from the reference for each prompt, using a scalar scoring function (log-rank was empirically selected) to score each response, calculating the "randomized rank" of the target score within the reference set (a percentile in $[0,1]$), and finally applying the Cramér–von Mises test for uniformity. A $p < 0.05$ indicates that the API model differs from the claimed model.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Real user-style prompts<br/>(WildChat)"] --> B["Asymmetric Sampling<br/>1 Target Query + m Reference Samples"]
    B --> C["Scoring Function f<br/>Empirically optimized log-rank"]
    C --> D["Randomized Rank Statistic<br/>Target score percentile in reference distribution"]
    D -->|If models same, rank ~ Uniform[0,1]| E["Cramér–von Mises<br/>Uniformity Test"]
    E -->|p < 0.05| F["Decision: Model Replaced"]
```

### Key Designs

**1. Rank = Uniformity: Converting "Model Equality" to "Rank Uniformity"**

This is the theoretical anchor of the paper, addressing the pain point that black-box outputs lack logits, making direct distribution comparisons difficult. For a fixed prompt $x$, let the cumulative distribution of reference model scores be $F_{\pi_{\text{ref}}}(s\mid x) := \Pr_{y\sim\pi_{\text{ref}}}[f(y,x)\le s]$. Applying this CDF to the target output score $s_{\text{tgt}}$ yields its percentile rank. By the probability integral transform, **if $\pi_{\text{tgt}}=\pi_{\text{ref}}$, this rank follows $\text{Uniform}[0,1]$**. Conversely, if the ranks are uniform, it can be inferred that the target and reference scores are identically distributed, and via the (approximate) injectivity of the scoring function, the model outputs are identically distributed. Thus, the equality of high-dimensional text distributions is compressed into a clean 1D uniformity problem.

Since text responses are discrete and $f$ takes discrete values, the CDF is a step function, which would lead to discrete ranks that are not continuous under the null hypothesis. The paper uses **randomized quantile residuals** (Dunn & Smyth, 1996) to generalize the probability integral transform to discrete distributions, defining a continuous rank statistic:

$$r_{\text{tgt}} := F_{\pi_{\text{ref}}}(s^-_{\text{tgt}}) + U \cdot P\bigl(f(y,x)=s_{\text{tgt}}\bigr), \quad U \sim \text{Uniform}[0,1]$$

where $F_{\pi_{\text{ref}}}(s^-_{\text{tgt}})$ is the left limit of the CDF at $s_{\text{tgt}}$ (probability mass strictly less than), and the second term uses an independent uniform random variable for stochastic interpolation within the "tied" probability mass. Consequently, under the null hypothesis, $r_{\text{tgt}}$ is exactly distributed on $\text{Uniform}[0,1]$, cleanly handling the discrete tie problem.

**2. Empirical CDF Estimation with $m$ Reference Samples + Randomized Tie-breaking**

The true $F_{\pi_{\text{ref}}}$ is uncomputable. In practice, it is estimated using an empirical CDF constructed from $m$ reference samples per prompt. Given target response scores $s_i=f(y_i,x_i)$ and reference response scores $s_{ij}=f(y_{ij},x_i)$, the empirical rank statistic is:

$$r_i = \frac{1}{m}\left(\sum_{j=1}^{m}\mathbb{1}\{s_i>s_{ij}\} + U_i \cdot \sum_{j=1}^{m}\mathbb{1}\{s_i=s_{ij}\}\right), \quad U_i \sim \text{Uniform}[0,1]$$

The first term counts how many reference samples have scores strictly lower than the target, while the second term uses $U_i$ for randomized interpolation among samples equal to the target score. This $U_i$ is not a trivial detail—it ensures that $r_i$ is an **unbiased estimate** of $r_{\text{tgt}}$ and maintains rank uniformity even in discrete scenarios, thereby preserving Type-I error control under the null hypothesis.

**3. Log-rank Scoring Function: Empirically Selecting the Most Discriminative Scalar**

Theoretically, an injective scoring function relative to $y$ is needed to ensure any difference is reflected in the scores, but such functions are impractical to compute. The paper instead requires the scoring function to be **sufficiently discriminative**: as long as $\pi_{\text{ref}}\neq\pi_{\text{tgt}}$, the induced score distributions should differ ($P_{S_{\pi_{\text{ref}}}}\neq P_{S_{\pi_{\text{tgt}}}}$), causing ranks to deviate from uniformity. To find the optimal function, the authors compared five candidates using Monte Carlo (500 trials, 10 random WildChat prompts per trial, 50 completions per prompt, temperature 0.5, 30-token limit): log-likelihood, token rank, log-rank, entropy, and log-likelihood log-rank ratio (LRR). Results showed that **log-rank (average of log-transformed token ranks) yielded an AUROC distribution that most consistently deviates from 0.5** (e.g., Gemma-2-9b-it vs. its 4-bit version in Fig. 2), and was thus selected for RUT. Intuitively, quantization or fine-tuning first perturbs the token rank the model assigns at each step; log-rank captures this token-level typicality while the log compression prevents occasional high ranks from dominating the overall score.

**4. Cramér–von Mises Uniformity Test for P-values**

After obtaining a batch of ranks $\{r_i\}_{i=1}^n$, the **Cramér–von Mises (CvM) test** is used to evaluate deviation from $\text{Uniform}[0,1]$, testing $H_0: r_i \sim \text{Uniform}[0,1]$. Sorting the ranks as $r_{(1)}\le\cdots\le r_{(n)}$, the statistic is:

$$\omega^2 = \frac{1}{12n} + \sum_{i=1}^{n}\left(\frac{2i-1}{2n}-r_{(i)}\right)^2$$

It measures the integrated squared distance between the empirical CDF and the theoretical uniform CDF. The $p$-value is given by the tail probability $\Pr_{H_0}(\omega^2_{\text{null}}\ge\omega^2_{\text{obs}})$. If $p < 0.05$, the null hypothesis is rejected, and the model is judged as replaced. CvM was chosen over Kolmogorov-Smirnov (KS) because it is sensitive to deviations across the entire distribution (KS only considers the maximum gap)—a KS baseline using the same log-rank information was consistently outperformed by CvM in experiments.

### Loss & Training
This work is a statistical testing method and does not involve model training or loss functions. The only "tuning" is the offline selection of the scoring function (§4.2). Key hyperparameters for testing: 100 prompts per trial, 1 target query, 100 reference samples ($m=100$), temperature 0.5, 30-token generation limit, significance level $\alpha=0.05$, with power estimated via 500 Monte Carlo trials.

## Key Experimental Results

Evaluation focuses primarily on **statistical power AUC**: simulating probabilistic substitution attacks where a proportion $q \in [0,1]$ of API queries is routed to an alternative model, estimating statistical power (probability of correctly rejecting the null) for each $q$, and calculating the AUC of the power–substitution curve across $q \in [0,1]$. Higher values represent more stable detection across substitution ratios. Data used is WildChat (real user conversations); baselines include MMD (Gao et al., 2025) and a KS test using the same log-rank information.

### Main Results: Coverage of Four Threat Categories

| Threat Scenario | Setup | Conclusion |
|-----------------|-------|------------|
| Quantization (§5.2) | 3 Models × (4-bit/8-bit) | RUT is the strongest in 4/6 settings; all methods nearly fail (power ~0) for 8-bit versions of Gemma/Mistral. |
| Jailbreak (§5.3) | Mistral/Gemma × Dan/Anti-Dan/Evil-Bot | **RUT achieves highest power in 6/6 settings**, with all power AUC > 0.75. |
| Harmful SFT (§5.4) | Llama/Mistral × Alpaca/BeaverTails (LoRA 5 epochs) | RUT achieves highest power across all configs, **detecting distribution drift as early as epoch 1**. |
| Model Replacement (§5.5) | Pairwise comparison of 5 open-weight models | RUT achieves highest AUC across all model pairs. |

Representative figures (Power AUC for Quantization Detection, excerpt from Table 2a):

| Model–Quantization | RUT | MMD | KS |
|--------------------|-----|-----|----|
| Gemma–4bit | **0.392** | 0.214 | 0.017 |
| Llama–4bit | **0.642** | 0.625 | 0.474 |
| Mistral–4bit | **0.586** | 0.500 | 0.330 |
| Gemma/Mistral–8bit | ~0.05 (Fail) | ~0.05 | ~0.00 |

### Cross-Query Domains and Real APIs

| Configuration | Key Finding |
|---------------|-------------|
| Cross-domain (§5.6) BigCodeBench / MATH | Quantization detection applied to code/math domains; RUT outperformed in 7/8 settings, showing generalization across query distributions. |
| Real APIs (§5.7) | Testing Llama/Mistral/Gemma across providers like Nebius, Novita, HF Inference using a local A100 baseline. |
| KS Comparison | Shows same trend but with significantly lower sensitivity, validating CvM over KS. |

### Key Findings
- **8-bit quantization is a recognized difficulty**: For 8-bit variants of Gemma and Mistral, all methods have near-zero power—8-bit quantization perturbs output distributions too subtly for current auditing techniques.
- **Robustness to string formatting**: In the real API experiment with Mistral on HF Inference, MMD's power surged to 1.0 while other methods were < 0.2. This was traced to HF Inference stripping a leading space from reference outputs; MMD, using character-level Hamming distance, was misled by this formatting difference. After correcting the space, MMD dropped to 0.211. RUT, based on log-rank rather than strings, is **naturally robust to such minor decoding inconsistencies**.
- **Jailbreaks are easiest to catch; quantization is hardest**: Jailbreak system prompts significantly perturb output distributions (power AUC generally > 0.85), while high-bit quantization causes the smallest perturbations.

## Highlights & Insights
- **The "Rank Uniformity" transformation is elegant**: Compressing high-dimensional text distribution equality into a 1D uniformity test on $[0,1]$ is theoretically sound (probability integral transform), utilizes mature statistical tools (CvM), and solves the black-box lack of logits—requiring only local sampling from the reference model.
- **Asymmetric sampling addresses cost constraints**: Expensive target APIs are queried only once per prompt, while cheap local reference models are sampled heavily. This lowers auditing costs, which is critical for widespread adoption.
- **Adversarial robustness by design**: By using real user-style prompts (WildChat) rather than specialized distributions, practitioners prevent adversarial providers from identifying and rerouting auditing queries.
- **Randomized quantile residual tie-breaking is a key detail**: When performing uniformity tests on discrete distributions, failing to handle ties would result in non-continuous ranks and inflated rejection rates. This random interpolation technique is transferable to any "discrete score + uniformity test" scenario.

## Limitations & Future Work
- **Omniscient adversaries not verified**: The authors admit they did not test against adversaries fully aware of the RUT mechanism, who might specifically reroute prompts that generate abnormal log-ranks. Evaluating robustness against stronger adversaries is a necessary next step.
- **Requires a local reference model**: RUT requires the ability to deploy and sample from a trusted reference implementation, meaning it is only applicable to open-weight models. Relaxing this requirement would significantly expand its applicability.
- **8-bit quantization blind spot**: The distribution perturbation from high-bit quantization is too small for all current methods to detect, indicating that auditing still struggles with the most stealthy replacements.
- **Personal Observation**: The log-rank scoring function is empirically selected; there is no theoretical guarantee it remains optimal for new types of substitution (e.g., distillation, speculative decoding). Furthermore, whether conclusions hold as decoding parameters (beyond T=0.5 / 30 tokens) vary warrants further investigation.

## Related Work & Insights
- **vs MMD (Gao et al., 2025)**: Also formalizes auditing as a model equality test, but MMD uses specialized query distributions and character-level Hamming distance for Kernel MMD. This is both bypassable by adversarial identification and susceptible to formatting inconsistencies. RUT uses natural prompts and log-rank tests, making it robust and efficient (1 API call per prompt).
- **vs Classifier Fingerprints (Sun et al., 2025)**: Trains text classifiers to catch model idiosyncrasies, requiring massive queries and reported failures in quantization detection. RUT is query-efficient and works for 4-bit quantization. Fingerprinting aims to "recognize the source regardless of fine-tuning," which is the opposite of auditing's goal to "alert on any deviation."
- **vs Identity Probing (Huang et al., 2025) / Benchmark Comparison (Chen et al., 2023a)**: The former misses size/version/quantization differences, and the latter fails to expose stealthy substitutions or partial routing based on performance alone. RUT directly compares output distributions, making it sensitive to subtle shifts.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "Rank → Uniformity" transition is elegant and practical, with inherent adversarial robustness.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of quantization/jailbreak/SFT/replacement across domains and real APIs is comprehensive, though missing omniscient adversary empirical tests.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivation, precise positioning in Table 1, and the real-word formatting inconsistency case study is highly persuasive.
- Value: ⭐⭐⭐⭐⭐ Model auditing is a necessity in the API era; the method is query-efficient, easy to deploy, and code-open, offering high utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization](../../AAAI2026/llm_safety/psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)
- [\[ICLR 2026\] Bias Similarity Measurement: A Black-Box Audit of Fairness Across LLMs](bias_similarity_measurement_a_black-box_audit_of_fairness_across_llms.md)
- [\[AAAI 2026\] GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs](../../AAAI2026/llm_safety/graphtextack_a_realistic_black-box_node_injection_attack_on_llm-enhanced_gnns.md)
- [\[ACL 2026\] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework](../../ACL2026/llm_safety/rethinking_llm_watermark_detection_in_black-box_settings_a_non-intrusive_third-p.md)
- [\[CVPR 2026\] Omni-Attack: Adversarial Attacks on Open-Ended VQA in Black-Box Multimodal LLMs](../../CVPR2026/llm_safety/omni-attack_adversarial_attacks_on_open-ended_vqa_in_black-box_multimodal_llms.md)

</div>

<!-- RELATED:END -->
