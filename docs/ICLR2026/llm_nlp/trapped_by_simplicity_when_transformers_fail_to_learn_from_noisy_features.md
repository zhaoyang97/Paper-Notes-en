---
title: >-
  [Paper Note] Trapped by simplicity: When Transformers fail to learn from noisy features
description: >-
  [ICLR 2026][LLM/NLP][noise-robust learning] This paper demonstrates that Transformers fail to learn Boolean functions from feature-noisy data. Their simplicity bias—a tendency to learn low-sensitivity functions—causes mo…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "noise-robust learning"
  - "simplicity bias"
  - "Boolean functions"
  - "Transformer"
  - "sensitivity"
date: 2026-05-08
content_hash: ed88dab9b30b15c3
---

# Trapped by simplicity: When Transformers fail to learn from noisy features

**Conference**: ICLR 2026
**arXiv**: [2602.08695](https://arxiv.org/abs/2602.08695)  
**Code**: [https://github.com/peterse/noise-robust-boolean-learning/](https://github.com/peterse/noise-robust-boolean-learning/)  
**Area**: LLM NLP / Transformer Theoretical Analysis
**Keywords**: noise-robust learning, simplicity bias, Boolean functions, Transformer, sensitivity

## TL;DR
This paper demonstrates that Transformers fail to learn Boolean functions from feature-noisy data. Their simplicity bias—a tendency to learn low-sensitivity functions—causes models to become trapped at optimal noisy predictors that are simpler than the target function, preventing recovery of the true noiseless target.

## Background & Motivation

**Background**: LLM training data pervasively contains noise (typos, grammatical errors, semantic inconsistencies), yet these models are frequently applied to noise-sensitive tasks such as arithmetic reasoning, where each token prediction depends on preceding noiseless inputs. Prior work has shown that Transformers trained without noise exhibit robustness to test-time noise, but the ability to learn from training data that itself contains feature noise remains underexplored.

**Limitations of Prior Work**: It is known that Transformers exhibit **simplicity bias**, preferring to learn Boolean functions of low sensitivity. Prior work has focused mainly on the effect of label noise, whereas feature noise (input bit flips) is qualitatively different—it renders the optimal predictor $f_N^*$ simpler than the target function $f$.

**Key Challenge**: Under feature-noisy training data, the Bayes-optimal predictor $f_N^*$ typically has lower sensitivity than the target function $f$. The simplicity bias of Transformers causes them to favor learning $f_N^*$ over $f$, resulting in failure to generalize to noiseless data.

**Goal**: Can Transformers learn the correct noiseless target function from feature-noisy data (noise-robust learning)? Under what conditions does this succeed or fail? How can failure be explained and mitigated?

**Key Insight**: The study adopts the theoretical framework of Boolean functions, using sensitivity as a complexity measure. It establishes a theoretical framework for noise-robust learning from the perspective of information theory and noisy-channel coding, and designs "trap function" controlled experiments to isolate the effect of simplicity bias.

**Core Idea**: The simplicity bias of Transformers causes them to become "trapped" at simpler optimal noisy predictors under feature noise, leading to systematic failure in noise-robust learning of random Boolean functions.

## Method

### Overall Architecture
The study is structured as theoretical analysis combined with controlled experiments. The input is a binary string $X$ of length $n$; the goal is to learn a Boolean function $f(X)$. Training inputs $Z = X \oplus E$ are subject to independent bit-flip noise with flip probability $p$, while labels $Y = f(X)$ are noiseless. The central question is whether a model trained on noisy data $(Z, Y)$ can correctly generalize to noiseless test data $(X, Y)$.

### Key Designs

1. **Theoretical Framework from a Noisy-Channel Coding Perspective**:

    - Function: Establishes a connection between noise-robust learning and information theory.
    - Mechanism: Models next-token prediction as a communication process—sender Alice encodes information $Y = f(X)$ into sequence $X$, and receiver Bob observes the noisy version $Z$. The Bayes-optimal predictor is $f_N^*(x) := \text{sign}(T_{1-2p} f(x))$, where $T_\rho$ is the noise operator. Information-theoretic upper and lower bounds on the optimal error rate are established via Fano's inequality: $\Phi^{-1}(H(Y|Z)) \leq \text{err}_f(f_N^*) \leq \phi^{-1}(H(Y|Z))$.
    - Design Motivation: Unlike prior compression-based analyses (source coding), the noisy-channel coding perspective precisely characterizes the effect of feature noise on learning.

2. **Analysis of "Self-Predicting" Functions**:

    - Function: Proves that parity and majority functions (odd-length) are self-predicting ($f = f_N^*$), and hence Transformers can successfully learn them.
    - Mechanism: **Proposition 1** proves that for $\text{maj}_n$ (odd $n$) and $\text{parity}$, the target function is itself the optimal predictor of noisy data. This implies that ordinary loss minimization recovers the target function. Experimental validation confirms that Transformers successfully perform noise-robust learning on $\text{maj}(20,5)$, $\text{maj}(40,5)$, and $\text{parity}(20,4)$, with the median Transformer outperforming the best LSTM.
    - Design Motivation: Demonstrates that prior work on parity/majority conveyed an overly optimistic impression—these functions are special cases.

3. **Sensitivity Analysis of Random Boolean Functions (Core Theoretical Contribution)**:

    - Function: Proves that for random Boolean functions, the optimal noisy predictor has, on average, lower sensitivity than the target function.
    - Mechanism: **Proposition 2** proves that for uniformly sampled Boolean functions $f$, when $n$ is sufficiently large, $\mathbb{E}_f[I[f_N^*]] \approx \frac{n}{\pi} \arccos\left(\frac{2p(1-p)}{p^2+(1-p)^2}\right)$, and $\mathbb{E}_f[I[f]] > \mathbb{E}_f[I[f_N^*]]$. This is empirically validated on 3,200 random $k$-juntas, where all samples satisfy $I[f] \geq I[f_N^*]$.
    - Design Motivation: This is the central finding—because $f_N^*$ is simpler than $f$, the simplicity bias of Transformers causes them to learn $f_N^*$ rather than $f$.

4. **"Trap Function" Controlled Experiments**:

    - Function: Constructs special functions where validation error is close ($\text{err}_f(f_N^*) \approx \text{err}_f(f)$) but sensitivity differs substantially ($I[f_N^*] \ll I[f]$), isolating the effect of simplicity bias.
    - Mechanism: Under these conditions, a model could in principle recover $f$ through loss minimization, yet Transformers are "trapped" by simplicity bias and converge to $f_N^*$. LSTMs also fail, but for a different reason (overfitting to training data). Adding a sensitivity penalty $-\lambda I[\hat{f}]$ to the loss function (penalizing low-sensitivity solutions) allows Transformers to escape the trap and learn $f$.
    - Design Motivation: Precisely demonstrates that simplicity bias—rather than other factors—is the cause of noise-robust learning failure.

### Loss & Training
- Standard training: minimizes cross-entropy loss on noisy data.
- Improved training: augments the loss with a complexity preference term $-\lambda I[\hat{f}]$ to penalize low-sensitivity solutions.
- The choice of $\lambda$ is critical—there exists a narrow window in which Transformers can escape the trap.

## Key Experimental Results

### Main Results
Comparison of Transformers and LSTMs on noise-robust learning (300 runs per experimental setting):

| Function | Model | Noise rate $p$ | Success rate of noise-robust learning | Notes |
|----------|-------|---------------|--------------------------------------|-------|
| maj(20,5) | SAN median | 0–0.4 | Near-optimal | Self-predicting; succeeds |
| maj(20,5) | Best LSTM | 0.1+ | Significantly below optimal | Fails under moderate noise |
| parity(20,4) | SAN | 0–0.3 | Higher than LSTM | Self-predicting function |
| parity(20,4) | LSTM | 0.05+ | Near zero | Fails even at low noise |
| Random $k$-junta | SAN | 0.2 | Mostly fails | Non-self-predicting |
| maj(30,4) even | SAN | 0.32 | Fails | Non-self-predicting (imbalanced) |

### Ablation Study

| Configuration | Key Finding | Notes |
|---------------|------------|-------|
| No sensitivity penalty | Transformer trapped at $f_N^*$ | Validation error near-optimal but poor noiseless generalization |
| Sensitivity penalty $\lambda$ (optimal) | Transformer escapes trap and learns $f$ | Requires precise tuning of $\lambda$ |
| $\lambda$ too large | Learning fails | Preference for overly complex solutions |
| LSTM + sensitivity penalty | Still fails | Overfitting cannot be resolved by sensitivity penalty |

### Key Findings
- **Transformers systematically fail on random $k$-juntas**: Among 3,200 random functions, only those with $I[f] \approx I[f_N^*]$ are successfully learned.
- **Simplicity bias is the root cause of failure**: Transformers achieve near-optimal validation error (performing well on noisy data) while performing poorly on noiseless test data—validation loss cannot guide selection of the correct model.
- **LSTM failure has a fundamentally different cause**: Not simplicity bias, but overfitting to training data.
- **Larger sensitivity gaps correlate with greater difficulty**: $I[f] - I[f_N^*]$ is positively correlated with noiseless generalization error.

## Highlights & Insights
- **The dual nature of simplicity bias**: Prior work regarded the simplicity bias of Transformers as advantageous (promoting generalization); this paper reveals that under feature noise it becomes the root cause of systematic failure. This perspective shift is highly illuminating.
- **Trap function experimental design**: By carefully constructing functions where $\text{err}(f_N^*) \approx \text{err}(f)$ but $I[f_N^*] \ll I[f]$, the authors elegantly isolate the effect of simplicity bias and eliminate the confounding factor of a validation error gap.
- **Information-theoretic perspective**: Connecting noise-robust learning to noisy-channel coding (rather than the conventional compression/source coding framing) provides a novel set of analytical tools.
- **Implications for LLM training**: If sufficient noise or randomness is present in the training corpus, LLMs may be unable to learn precise discrete reasoning rules (e.g., arithmetic), as simplicity bias causes them to converge to "approximate but imprecise" solutions. This finding directly transfers to explaining the unreliability of LLMs in mathematical reasoning.

## Limitations & Future Work
- **Overly simplified noise model**: Only i.i.d. bit-flip noise is considered; the structured nature of real linguistic noise (grammatical errors, semantic inconsistencies) is not captured.
- **Restricted function class**: Experiments are limited to random $k$-juntas, which may not represent the complexity of real learning problems.
- **High noise rate requirement**: Many observed effects require relatively high noise rates, which may not be common in natural datasets.
- **Sensitivity penalty is impractical**: Escaping the trap requires precise tuning of $\lambda$, which lacks operational feasibility in practice.
- **Directions for improvement**: Investigating how more complex noise models (e.g., semantic noise in natural language) affect the conclusions; exploring alternative simplicity-bias mitigation methods such as curriculum learning and noise-aware training.

## Related Work & Insights
- **vs. Bhattamishra et al. (2023b)**: They prove that Transformers prefer low-sensitivity functions and are robust to label noise. The key distinction of this paper is the study of **feature noise** rather than label noise, revealing that simplicity bias is harmful in this setting.
- **vs. Vasudeva et al. (2025)**: They prove that Transformers trained without noise are robust to test-time noise. This paper further asks whether training on noisy data generalizes to noiseless testing—a question to which the answer is generally negative.
- **vs. Deletang et al. (2024)**: They analyze language models from a compression/source coding perspective. This paper adopts a noisy-channel coding perspective, which is better suited to characterizing settings where training data contains noise.
- Directly informs the understanding of LLM failures in mathematical reasoning: if noise in training data biases the optimal solution toward lower complexity, LLMs may learn "simplified" reasoning rules.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reveals the negative effect of simplicity bias under feature noise; solid theoretical contributions (proof of Proposition 2 and trap function design).
- Experimental Thoroughness: ⭐⭐⭐⭐ 3,200 random functions, multiple function types, and elegantly designed controlled experiments; however, results are confined to synthetic Boolean function settings.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain; the progression from success cases to failure cases is natural, with tight integration of theory and experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] When Stability Fails: Hidden Failure Modes of LLMs in Data-Constrained Scientific Decision-Making](when_stability_fails_hidden_failure_modes_of_llms_in_data-constrained_scientific.md)
- [\[ACL 2026\] Characterizing the Expressivity of Local Attention in Transformers](../../ACL2026/llm_nlp/characterizing_the_expressivity_of_local_attention_in_transformers.md)
- [\[ICML 2026\] Deep Networks Learn to Parse Uniform-Depth Context-Free Languages from Local Statistics](../../ICML2026/llm_nlp/deep_networks_learn_to_parse_uniform-depth_context-free_languages_from_local_sta.md)
- [\[ACL 2026\] TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale](../../ACL2026/llm_nlp/tingis_real-time_risk_event_discovery_from_noisy_customer_incidents_at_enterpris.md)
- [\[ICLR 2026\] Is the Reversal Curse a Binding Problem? Uncovering Limitations of Transformers from a Basic Generalization Failure](is_the_reversal_curse_a_binding_problem_uncovering_limitations_of_transformers_f.md)

</div>

<!-- RELATED:END -->
