---
title: >-
  [Paper Note] Identifying the Periodicity of Information in Natural Language
description: >-
  [ACL 2026][LLM/NLP][Information Periodicity] This paper adapts the AutoPeriod detection algorithm from signal processing to token-surprisal sequences…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Information Periodicity"
  - "surprisal"
  - "AutoPeriod"
  - "Harmonic Regression"
  - "LLM Detection"
date: 2026-05-08
content_hash: a3b03523b5c4a624
---

# Identifying the Periodicity of Information in Natural Language

**Conference**: ACL 2026  
**arXiv**: [2510.27241](https://arxiv.org/abs/2510.27241)  
**Code**: <https://github.com/CLCS-SUSTech/APS>  
**Area**: Information-Theoretic Linguistics / Information Density / Machine-Generated Text Detection  
**Keywords**: Information Periodicity, surprisal, AutoPeriod, Harmonic Regression, LLM Detection

## TL;DR
This paper adapts the AutoPeriod detection algorithm from signal processing to token-surprisal sequences, proposing APS (AutoPeriod of Surprisal). APS can **directly detect** periods in natural language information density at a single-document level (e.g., "one cycle every 53 tokens"). It finds that approximately 11% of human-authored documents exhibit strict periodicity, and LLM-generated text is twice as periodic as human text (30% vs 14.8%), providing direct evidence for the UID theory and interpretable features for AI text detection.

## Background & Motivation

**Background**: Since Shannon (1948), a core hypothesis of information-theoretic linguistics has been UID (Uniform Information Density, Aylett & Turk 2004; Jaeger 2010)—the idea that natural language tends to make token-level surprisal uniform for efficient communication. However, UID is an asymptotic property; surprisal naturally fluctuates in short windows (Genzel & Charniak 2002/2003 discovered high surprisal at paragraph starts and drops at ends). Recent work further upgrades "fluctuation" to a "periodicity" hypothesis—Xu & Reitter 2017 / Yang et al. 2023 (FACE) used Fourier spectrums to show that text surprisal is distinguishable from white noise; Tsipidi et al. 2025 used harmonic regression (HR) to verify significant periodic signals corresponding to discrete structural units (sentences, paragraphs, EDUs).

**Limitations of Prior Work**: (1) Fourier methods (frequency domain) only provide indirect evidence that the "spectrum is not white noise" without converting the spectrum back to specific "period lengths" in the time domain; (2) HR methods depend on **pre-selected candidate periods** (requiring "sentence/paragraph/EDU length" as $U_t$ beforehand), failing to discover unspecified periods or determine if a single document is truly periodic; (3) There is no doc-level period detector, making it impossible to correlate with doc-level variables like genre, authorship, or human-vs-LLM.

**Key Challenge**: To answer whether natural language information is periodic, one must (a) provide rejectable statistical judgments at the single-document level (presence/absence/which periods), (b) ensure period discovery is free of candidate priors, and (c) identify both known structural units and "unknown periods" outside existing structures. Existing methods fail all three.

**Goal**: (1) Design a doc-level period detection algorithm that returns "all" significant periods using confidence thresholds; (2) Quantify "how many documents are truly periodic" across 4 corpora; (3) Compare findings with known structural units; (4) Use HR to reverse-validate APS discoveries; (5) Use APS to identify periodicity differences between human and LLM text.

**Key Insight**: The AutoPeriod algorithm proposed by Vlachos et al. (2005) for financial/social science time series—which uses a periodogram to find candidates and an ACF (autocorrelation function) "hill" to filter false positives—is mature and reliable. It can be adapted to surprisal sequences by using the Lomb-Scargle periodogram to improve long-period resolution.

**Core Idea**: Classic Signal Processing AutoPeriod + Surprisal Sequence = Single-Document Information Period Detector. Candidate selection via periodogram + verification via ACF hill ensures a low false-positive rate. Finally, HR is used for reverse validation and exploration of human vs. LLM differences.

## Method

### Overall Architecture
Input: A document's token surprisal sequence $\mathbf{x} = (x_0, \dots, x_{N-1})$, where $x_n = -\log p(t_n \mid t_{<n})$ is estimated by a language model (LLaMA3-8B / Yarn-LLaMA2-7B for English, Qwen2-7B for Chinese). Output: A set of valid periods $\{\tau_1, \tau_2, \dots\}$ or an empty set. The process involves two steps:

1.  **Step 1 GetPeriodHints**: A Lomb-Scargle periodogram $\bm{\mathcal{P}}$ is computed for $\mathbf{x}$. Then, $m=100$ random permutations $\text{Permute}(\mathbf{x})$ are performed to obtain the max power distribution of the reference $\bm{\mathcal{P}}^{\text{rand}}$. The 99th percentile is set as the threshold $P_{\text{threshold}}$. Frequencies $k$ where $\bm{\mathcal{P}}[k] > P_{\text{threshold}}$ are collected as period hints $\tau = N/k$. These are statistically significant candidate periods.
2.  **Step 2 ACFFiltering**: For each hint $\tau' = N/k$, a window $W = [(\tau + \tau_{\text{next}})/2 - 1, (\tau + \tau_{\text{prev}})/2 + 1]$ is inspected on the ACF curve to see if it resides on a "hill." Linear regression is used to find the optimal split point $t_{\text{best}}$, requiring the left slope > right slope and a slope difference $> 0.01$. If isHill = True, a findPeak function identifies the nearest local ACF maximum within the window as the refined period. Otherwise, the hint is discarded.

Documents are classified into three types: $P_1$ (at least one period hint, quasi-periodic), $P_2 \subseteq P_1$ (hints passed ACF validation, strictly periodic), and $\Sigma - P_1$ (no hints, non-periodic).

### Key Designs

1.  **Surprisal-based Periodogram + Permutation Significance Test (Step 1)**:
    - **Function**: Identifies frequency peaks in $\mathbf{x}$ that are "significantly higher than noise" in the frequency domain, where each peak corresponds to a candidate period length.
    - **Mechanism**: DFT $X_k = \sum_n x_n e^{-i 2\pi k n / N}$ is performed, and the magnitude $\|X_k\|$ of the first half yields the periodogram $\bm{\mathcal{P}}$ (Lomb-Scargle is actually used for better long-period resolution). 100 random permutations are conducted, recording the max power of each to set the 99th percentile threshold. This is a **model-agnostic permutation test**—permutation destroys periodicity, providing a null distribution of max power for a signals without periods.
    - **Design Motivation**: Raw DFT spectrums cannot distinguish "true periodic peaks" from "random noise." The permutation test provides strict statistical significance with adjustable confidence. A default $CL=0.90$ is used, which is more sensitive than the traditional $0.95$.

2.  **ACF Hill Validation to Filter False Positives (Step 2)**:
    - **Function**: The periodogram provides candidates but is prone to false positives (especially short periods); recursive filtering is done using the geometric shape of the ACF.
    - **Mechanism**: A true period $\tau$ must be a "hill" on the ACF curve, where $\text{ACF}(\tau) = \frac{1}{N}\sum_n x(n) \cdot x(n+\tau)$ reaches a local maximum at $\tau$. For each hint $\tau'$, a window $W$ is defined, and split points $t$ are enumerated to run linear regressions, finding $t_{\text{best}}$ that minimizes $\epsilon_L + \epsilon_R$. If the left slope > right slope (and $|\theta_L - \theta_R| > 0.01$), the window's ACF is considered "hill-shaped." If so, the refined period is the ACF peak within $W$.
    - **Design Motivation**: Periodograms indicate "frequency energy," but high energy does not guarantee self-similarity. The ACF hill shape is direct evidence of self-similarity. This combination of "frequency domain significance + time domain autocorrelation" significantly reduces false positives. Table 1 shows $|P_2|/|P_1| \approx 66\%$, meaning one-third of hints are filtered.

3.  **HR Reverse Validation + Filter-based Gain**:
    - **Function**: Uses Tsipidi et al. 2025's Harmonic Regression (HR) to verify that APS discoveries are indeed "statistical periods."
    - **Mechanism**: $U_t$ in the HR formula $s(w_t) \sim \text{baseline} + \text{HR}(U_t)$ is replaced with APS-found hints/valid periods. Significance is checked for harmonic terms $\beta_{1,k} \sin(k 2\pi t/U_t) + \beta_{2,k}\cos(k 2\pi t/U_t)$ (amplitude $A_k = \sqrt{\beta_{1,k}^2 + \beta_{2,k}^2}$, p<0.001). Corpora are sliced by APS classifications $P_2 \subset P_1 \subset \Sigma$ to check if HR MSE follows the order "$P_2 < P_1 < \Sigma < \Sigma - P_1$".
    - **Design Motivation**: APS is detection ("there is a period"), while HR is explanation ("this period explains surprisal"). They are independent. If APS periods yield significant HR results and MSE follows the expected order, it proves APS is not reporting noise.

### Loss & Training
**Ours involves no model training**—APS is a deterministic algorithm (DFT + permutation + linear regression), and HR uses OLS regression. Hyperparameters: $m=100$ permutations, percentile=99 (default) / 90 (actual), HR harmonic terms $K=10$. Corpora: WSJ, Brown, CTB, GCDT, RST Discourse Treebank, FACE, EvoBench. LMs are used only to estimate surprisal without updates.

## Key Experimental Results

### Main Results (Periodic Document Proportion across 4 Corpora)

| Corpus | $|\Sigma|$ | $|P_1|$ | $|P_2|$ | $|P_1|/|\Sigma|$ | $|P_2|/|P_1|$ | $|P_2|/|\Sigma|$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| WSJ (en) | 2499 | 221 | 131 | 8.84% | 59.28% | 5.24% |
| Brown (en) | 500 | 52 | 25 | 10.40% | 48.08% | 5.00% |
| GCDT (zh) | 50 | 15 | 13 | **30.00%** | **86.67%** | **26.00%** |
| CTB (zh) | 2773 | 304 | 212 | 10.96% | 69.74% | 7.65% |
| **Avg** | — | — | — | **15.05%** | **65.94%** | **10.97%** |

Periodicity in Chinese is significantly higher than in English. GCDT has the strongest periodicity (26%) as it contains multi-genre formal texts (academic, biography, interviews, news, how-to).

### Ablation Study (HR Reverse Validation: MSE Sliced by APS, Higher Strictness = Higher Periodicity)

WSJ HR MSE:

| Data Segment | Baseline (No HR) | EDU as $U_t$ | Sentence as $U_t$ | Whole Text |
| :--- | :--- | :--- | :--- | :--- |
| $P_2$ (Strictly Periodic) | **16.33** | **13.78** | **15.60** | 16.29 |
| $P_1$ (Quasi-Periodic) | 16.87 | 14.14 | 16.14 | 16.84 |
| $\Sigma$ (Full Corpus) | 18.26 | 15.18 | 17.56 | 18.25 |
| $\Sigma - P_1$ (Non-Periodic) | 18.59 | 15.42 | 17.89 | 18.57 |

MSE perfectly follows the order $P_2 < P_1 < \Sigma < \Sigma - P_1$—documents detected as "periodic" by APS indeed have more predictable surprisal under HR. The same trend is observed on GCDT.

HR Significance (WSJ $P_2$ subset, APS valid period as $U_t$): $A_1 = 0.2022$ ($\beta_{1,1}=0.1273$, p < 0.001; $\beta_{2,1}=0.1571$, p < 0.001), proving 1st-order harmonics are extremely significant.

### Key Findings
- **5-11% of documents are strictly periodic, 15% are quasi-periodic**: Periodicity is not "universal" but certainly not "rare"—an average of 10.97% strictly periodic documents is a significant finding for a subtle signal.
- **Chinese > English**: All metrics are higher for Chinese, possibly due to more rigid topic/paragraph structures or differences in LLM tokenizer/surprisal estimation behavior.
- **GCDT Highest (26%) implies genre defines periodicity**: Formal texts (academic, how-to) far exceed news in periodicity, suggesting future research into dimensions like verse vs prose or oral vs written.
- **APS finds long periods beyond structural explanation**: 33.33% of valid periods exceed 150 tokens, and 21.84% exceed 200 tokens, whereas only 3.63% of paragraphs exceed 150 tokens. **There exist "long-range periods"** that cannot be explained by EDUs, sentences, or paragraphs, likely stemming from topic transition or cross-paragraph semantic structures.
- **LLM text periodicity is ~2x human text**: On FACE BBC News, LLaMA3-70B generated text has $|P_2|/|\Sigma| = 30.06\%$ vs. human 14.80%. Even after excluding obvious repetition (7.33% of docs containing repeated phrases), the gap remains significant (27.86% vs. 14.80%). GPT-4o on EvoBench XSum shows 7.33% vs. human 3.33%, following the same trend.
- **Strong LLM periodicity is mainly in long period ranges (>50 tokens)**: Distribution comparisons show LLM has significant extra periodic peaks at long ranges, hypothesized as a tendency to maintain structural coherence across paragraphs.

## Highlights & Insights
- **Adapting a 50-year-old signal processing algorithm (AutoPeriod 2005 / Lomb-Scargle 1976) precisely to NLP**—the technology doesn't have to be brand new; the key is asking the right question and choosing the right tool ("Periodogram + ACF" is a classic two-step model). This is a paradigm of "borrowing mature tools from neighboring disciplines to solve new NLP problems."
- **Permutation testing as model-free significance judgment** is more robust than absolute thresholds—it compares directly with a "non-periodic null distribution" and is insensitive to the scale or distribution of surprisal.
- **The discovery of long periods exceeding structural units** is the most theoretically valuable finding—it suggests long-range structures in surprisal come from latent variables like topic flow or rhetorical structure, adding a "long-range version" to UID theory.
- **LLMs being more periodic** provides an **interpretable basis** for existing detection tools—it's not just a black-box difference in spectral features, but the linguistic fact that "LLMs tend to maintain structural coherence across segments."

## Limitations & Future Work
- Authors admit: (1) Only English and Chinese were tested, not morphologically rich languages; (2) The impact of genre was observed but not analyzed deeply; (3) **Insufficient precision in short-period detection**—DFT high-frequency resolution is worse than low-frequency, and permutation tests are more prone to false positives in short periods, causing high-confidence short-period peaks to shrink.
- **Potential Confounding**: LLM text may naturally differ in length/style/vocabulary from humans, which might naturally favor surprisal estimation; comparison after controlling for length/prompt is recommended.
- **Reliance on LLaMA3-8B / Qwen2-7B**: Different models generate different surprisal sequences; periods detected by APS could partially be "projections" of the LM rather than the language itself.
- **APS is a detection tool and does not answer "why periods exist"**—whether from topic, rhetoric, or cognition remains future work.

## Related Work & Insights
- **vs. Tsipidi et al. 2025 (HR)**: HR requires **pre-specifying** $U_t$ (sentence/EDU length), whereas APS discovers periods without priors. They are complementary—APS for detection, HR for explanation.
- **vs. Yang et al. 2023 (FACE)**: FACE uses the Fourier spectrum of cross-entropy to evaluate NLG quality; APS advances this by moving from the frequency domain back to specific time-domain period values, enhancing interpretability.
- **vs. Xu et al. 2024 (spectrum-based LLM detection)**: They use surprisal spectrums to distinguish human/LLM as a black box; APS provides the interpretable evidence that "LLM periods are more frequent and longer."

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Repeated Sequences Reveal Gaps between Large Language Models and Natural Language](repeated_sequences_reveal_gaps_between_large_language_models_and_natural_languag.md)
- [\[AAAI 2026\] Identifying and Analyzing Performance-Critical Tokens in Large Language Models](../../AAAI2026/llm_nlp/identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)
- [\[ACL 2026\] Generative Interfaces for Language Models](generative_interfaces_for_language_models.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)

</div>

<!-- RELATED:END -->
