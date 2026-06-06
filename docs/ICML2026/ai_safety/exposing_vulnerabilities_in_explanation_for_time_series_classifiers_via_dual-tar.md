---
title: >-
  [Paper Note] Exposing Vulnerabilities in Explanation for Time Series Classifiers via Dual-Target Adversarial Attack
description: >-
  [ICML 2026][AI Safety][Time series explainers] This paper proposes TSEF—a dual-target attack framework for joint "time series classifier + explainer" systems. By learning a "temporal vulnerability mask + frequency pertur…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Time series explainers"
  - "adversarial attacks"
  - "explanation faithfulness"
  - "frequency domain perturbation"
  - "dual-target optimization"
date: 2026-05-08
content_hash: 0bea2ca8120437ef
---

# Exposing Vulnerabilities in Explanation for Time Series Classifiers via Dual-Target Adversarial Attack

**Conference**: ICML 2026  
**arXiv**: [2602.02763](https://arxiv.org/abs/2602.02763)  
**Code**: https://github.com/Bohan7/TSEF  
**Area**: Time Series / Explainable AI / Adversarial Robustness  
**Keywords**: Time series explainers, adversarial attacks, explanation faithfulness, frequency domain perturbation, dual-target optimization  

## TL;DR
This paper proposes TSEF—a dual-target attack framework for joint "time series classifier + explainer" systems. By learning a "temporal vulnerability mask + frequency perturbation filter," it simultaneously drives the model prediction to a target label and the explanation to a reference saliency map within an $\ell_\infty$ budget. It proves that the "explanation stability = decision trustworthiness" assumption in current time series explainability pipelines is fundamentally flawed.

## Background & Motivation

**Background**: Interpretable Time Series Deep Learning Systems (ITSDLS), which combine a "classifier + explainer," are widely used in high-risk scenarios such as healthcare, finance, and industry. The classifier provides predictions, while the explainer (e.g., TimeX++, IG, perturbation-based methods) produces a $T \times D$ saliency map marking which time-channel pairs are most important. Clinicians often rely on these maps to "verify" model judgments during ECG alarms.

**Limitations of Prior Work**: Existing evaluations assume "explanation stability = model trustworthiness," using the invariance of explanations under slight perturbations as evidence of robustness. Simultaneously, adversarial attack literature focuses primarily on flipping prediction labels, while explanation attacks in vision/NLP (Ghorbani 2019, Zhang 2020, Ivankay 2022) only scatter or destroy attributions without the capability to simultaneously "flip the label + forge a credible explanation."

**Key Challenge**: When an attacker controls both "what the model says" and "why it says so," explainability shifts from a safety barrier to a facade. Two specificities of time series make this joint control harder than in vision/NLP: (1) **Pattern-level control**: Time series models are sensitive to structural trends and cycles; point-wise small noise is insufficient to migrate stably to the explanation. (2) **High-dimensional paradox**: Although the $T \times D$ input space allows for a large attack budget, dense perturbations within the $\ell_\infty$ ball cause attribution quality to diverge outside the target region at a rate of $O(d - |\Omega|)$, making it difficult to match sparse, connected target explanations.

**Goal**: To demonstrate that time series explainers are untrustworthy under adversarial conditions, provide the first white-box attack algorithm that achieves both "target classification + target explanation," and provide quantitative metrics to reveal this vulnerability.

**Key Insight**: Mathematically prove that "dense $\ell_\infty$ step updates cause attribution quality outside the target region to grow linearly with dimensions" (Theorem 4.1). This leads to the necessity of restricting attacks to a **structured subspace**—modifying only specific "time windows" and "spectral directions" rather than applying point-wise noise.

**Core Idea**: Decompose the attack into two sub-problems: "**where to perturb** (temporal vulnerability mask)" and "**how to perturb** (frequency perturbation filter)." The former uses sparse + connectivity regularization to learn a continuous time window, while the latter modifies the spectrum in the FFT domain before performing IFFT—naturally generating coherent perturbations at the trend/cycle level that drive both prediction and explanation.

## Method

### Overall Architecture
Threat Model: White-box, where the attacker has full access to the frozen classifier $f$ and explainer $\mathcal{H}^E$. The goal is to find a perturbation $\delta$ under the constraint $\|\delta\|_\infty \leq \epsilon$ such that $\tilde{\mathbf{X}} = \mathbf{X} + \delta$ satisfies $f(\tilde{\mathbf{X}}) = y'$ (target label) and $d(\mathcal{H}^E(\tilde{\mathbf{X}}), \mathbf{A}') \to \min$ (reference saliency map $\mathbf{A}'$).

TSEF splits this dual-target optimization into two nested layers: the inner layer learns a temporal mask $\mathbf{M}_t \in [0,1]^{T \times D}$ to select the "vulnerable region," while the outer layer learns a filter $\mathbf{M}_f \in [0,2]^{K \times D}$ on the FFT spectrum of that region. The final adversarial sample is a combination of the frequency-rewritten window and the rest of the original signal:

$$\tilde{\mathbf{X}} = \mathcal{F}^{-1}(\mathcal{F}(\mathbf{M}_t \odot \mathbf{X}) \odot \mathbf{M}_f) + (1 - \mathbf{M}_t) \odot \mathbf{X}$$

### Key Designs

1. **Theoretical Characterization of the High-Dimensional Paradox (Theorem 4.1)**:
    - **Function**: Uses first-order analysis to prove "why ordinary dense $\ell_\infty$ PGD cannot be used to attack explanations."
    - **Mechanism**: Considering a dense sign step $\delta = -\varepsilon \cdot \mathrm{sign}(g_c)$ ($g_c$ is the classification loss gradient), let $\Omega$ be the sparse support of the reference explanation ($|\Omega| \ll d$). The theorem yields $\mathbb{E}[\|\mathbf{A}(\tilde{\mathbf{X}})\|_{1, \Omega^c}] \geq c \varepsilon (d - |\Omega|)$, and thus $\mathbb{E}[\|\mathbf{A}(\tilde{\mathbf{X}}) - \mathbf{A}'\|_1] \geq c \varepsilon (d - |\Omega|)$—attribution quality outside the target region diverges linearly with dimensionality.
    - **Design Motivation**: It elevates "practical intuition (dense perturbations scatter explanations)" into a theorem, formally proving that naive baselines using "joint loss + single $\ell_\infty$ ball" will inevitably fail in high-dimensional time series, thereby motivating the "structured subspace attack."

2. **Temporal Vulnerability Mask (TVM, Where to Perturb)**:
    - **Function**: Learns a sparse and temporally connected binary mask $\mathbf{M}_t$ in the inner optimization, allowing perturbations only in time-channel windows that most easily flip predictions and shape target explanations.
    - **Mechanism**: The inner loss includes a classification term $\lambda_{\mathrm{cls}} L_{\mathrm{cls}}(f(\mathbf{X}'), y')$ and an explanation term $\lambda_{\mathrm{exp}} d(\mathcal{H}^E(\mathbf{X}'), \mathbf{A}')$ ($\mathbf{X}' = \mathbf{X} \odot (1 - \mathbf{M}_t)$). Two structural regularizations are added: a sparse KL term $\mathcal{L}_{\mathrm{spa}} = \frac{1}{TD} \sum \mathrm{KL}(\mathrm{Bern}(\mathbf{M}_t[t,d]) \| \mathrm{Bern}(r))$ ($r=0.3$) and a connectivity term $\mathcal{L}_{\mathrm{con}} = \frac{1}{TD} \sum (\mathbf{M}_t[t+1,d] - \mathbf{M}_t[t,d])^2$. Optimization uses Gumbel-Sigmoid + Straight-Through Estimator with a projected sign step $\mathbf{M}_t \leftarrow \Pi_{[0,1]}(\mathbf{M}_t - \eta_t \mathrm{sign}(\nabla \mathcal{L}_t))$.
    - **Design Motivation**: Simple gradient updates are dominated by large signal amplitudes. The sign step ensures updates focus on the "directional contribution to loss" rather than "signal magnitude." Combined with sparse and connected constraints, this selects meaningful segments (e.g., the "QRS complex" in an ECG) rather than scattered points.

3. **Frequency Perturbation Filter (FPF, How to Perturb)**:
    - **Function**: Performs frequency-domain multiplicative filtering within the TVM-selected window, ensuring perturbations appear as coherent trends/cycles in the time domain.
    - **Mechanism**: FFT is applied to the windowed signal $W = \mathbf{M}_t^* \odot \mathbf{X}$ to get $\widehat{W}$, which is then multiplied by filter $\mathbf{M}_f$ and transformed back via IFFT: $\widetilde{W} = \mathcal{F}^{-1}(\widehat{W} \odot \mathbf{M}_f)$. The filter is parameterized as $\mathbf{M}_f = \Pi_{[0,2]}(1 + \alpha_{\mathrm{freq}} \tanh(\Theta_f))$, with $\alpha_{\mathrm{freq}} = \gamma \epsilon' / (\|\Delta \mathbf{X}_{\mathrm{base}}\|_\infty + \tau)$ scaled to guarantee time-domain $\ell_\infty \leq \epsilon'$. Updates for $\Theta_f$ also use sign steps to avoid gradient dominance by high-energy bands.
    - **Design Motivation**: Dense PGD leaves scattered noise in the time domain, preventing explainers from converging to sparse targets. Frequency filtering naturally produces coherent waveforms (trends, low-frequency envelopes). Since perturbations are confined to the TVM window, the global $\ell_\infty$ budget is efficiently converted into local strong structures, allowing the saliency map to be "painted" into the reference position.

### Loss & Training
The outer objective $J_{\mathrm{atk}} = d(\mathcal{H}^E(\tilde{\mathbf{X}}), \mathbf{A}') + \lambda L_{\mathrm{cls}}(f(\tilde{\mathbf{X}}), y')$ optimizes $\Theta_f$ while $\mathbf{M}_t^*$ is fixed. The inner and outer layers alternate until convergence. Distance $d$ can be MSE, cosine similarity, or KL divergence. Attacks are performed only on test samples correctly classified by the original model.

## Key Experimental Results

### Main Results
Six benchmarks (Synthetic: LowVar, SeqComb-UV, SeqComb-MV; Real: ECG, PAM, Epilepsy) $\times$ Three types of explainers (TimeX++, TimeX, Integrated Gradients) $\times$ Seven baselines (PGD, BlackTreeS, SFAttack, ADV², and three explainer perturbations: Random/Local G/Global G). F1/ASR measure classification attack success; AUPRC/AUP/AUR measure explanation alignment with the reference map. 

Comparison on LowVar with TimeX++ explainer:

| Method | F1↑ | ASR↑ | AUPRC↑ | AUP↑ | AUR↑ | Explanation |
|------|-----|------|--------|------|------|------|
| PGD | 0.833 | 0.846 | 0.258 | 0.194 | 0.318 | Label flips but explanation scatters |
| BlackTreeS | 0.728 | 0.740 | 0.361 | 0.271 | 0.375 | Similar to above |
| ADV² | 0.777 | 0.795 | 0.617 | 0.522 | 0.609 | Vision joint attack migration |
| Random | 0.014 | 0.014 | 0.786 | 0.643 | 0.730 | No change to prediction |
| **TSEF (Ours)** | **0.837** | **0.848** | **0.845** | **0.760** | **0.800** | Simultaneous flips + shapes explanation |

Comparison on ECG with TimeX++ explainer also outperforms baselines:

| Method | F1↑ | ASR↑ | AUPRC↑ | AUP↑ | AUR↑ |
|------|-----|------|--------|------|------|
| PGD | 0.902 | 0.946 | 0.639 | 0.715 | 0.431 |
| ADV² | 0.887 | 0.935 | 0.705 | 0.773 | 0.448 |
| **TSEF (Ours)** | **0.911** | **0.951** | **0.713** | **0.776** | **0.450** |

### Ablation Study

Key ablation dimensions:

| Configuration | F1 (Classification) | AUPRC (Explanation) | Explanation |
|------|----------|-------------|------|
| TSEF Full | High | High | TVM + FPF both enabled |
| w/o TVM | Close | Significant Drop | Explanation diverges to irrelevant steps |
| w/o FPF | Close | Medium Drop | Temporal noise is scattered; slow convergence |
| w/o Sparse KL | Close | Drop | Mask tends to be fully open (dense attack) |
| w/o Connectivity | Close | Drop | Mask is fragmented; fails to fit continuous targets |

### Key Findings
- **Prediction Stability $\neq$ Explanation Faithfulness**: Traditional PGD/ADV² fail to push AUPRC above 0.5 even with ASR $\approx$ 0.85, proving explainers can maintain "scattered but plausible" saliency maps while labels are flipped—the most dangerous failure mode in clinical settings.
- **Structured Perturbations are Necessary**: Random attacks achieve high AUPRC (0.786) but 0 ASR (no classification attack), while PGD does the opposite. Only TSEF succeeds in both, validating the "where + how" decomposition.
- **Universal Vulnerability Across Explainers**: TimeX, TimeX++, and IG—all structurally different—are compromised by the same framework, suggesting a systemic fragility in using saliency maps as interpretable proxies.
- **Stealthiness of Frequency Perturbations**: IFFT signals after spectral modification still look like reasonable ECG waveforms, but attribution is precisely reshaped to the attacker's target, showing that attacking medical AI does not require "obvious impulsive noise."

## Highlights & Insights
- **Theory-Driven Attack Design**: Theorem 4.1 clarifies why dense attacks fail via an attribution lower bound on $\Omega^c$, and TVM/FPF bypasses this bound. This "prove impossibility, then provide a path" structure is highly readable and rare in attack papers.
- **Transferable Attack Decomposition Paradigm**: Separating "where" and "how" to perturb introduces structural priors for the attacker symmetric to those of the defender. This can be transferred to any "signal + saliency map" combination, such as speech or EEG models.
- **Challenge to Core XAI Assumptions**: The paper questions the deeper assumption that users can safely make decisions based on explanations in clinical/financial workflows. This has direct policy implications for auditing regulated AI.

## Limitations & Future Work
- White-box threat model—requires gradient access, black-box transferability is not yet systematically evaluated.
- Only mask-based and gradient-based explainers were tested; prototype-based or counterfactual explanations were not covered.
- Whether adaptive defense (e.g., adversarial training with TSEF) can restore faithfulness remains an open question.
- Two-layer optimization per sample is slow; real-time ICU monitoring scenarios would require further acceleration.
- Covers multivariate/univariate time series but not graph-based or cross-modal time series (e.g., video + sensors).

## Related Work & Insights
- **vs. Ghorbani et al. 2019**: They proved image explanations can be perturbed but focused on "disruption" rather than "targeting"; Ours is dual-target and addresses the high-dimensional paradox.
- **vs. Ivankay et al. 2022**: NLP uses discrete token replacement; time series continuous values + $\ell_\infty$ budgets cause dense perturbations to diverge, which TSEF addresses via frequency decomposition.
- **vs. ADV² (Zhang et al. 2020)**: ADV² uses a unified $\ell_\infty$ ball; Ours achieves higher AUPRC (0.05-0.10 pp) on ECG, proving structured subspaces are superior to dense optimization.
- **vs. Ding 2023 / Gu 2025**: They focus only on predicting flipping; TSEF is the first to bridge "adversarial robustness" and "interpretability auditing."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

## Rating
- Novelty: To be rated
- Experimental Thoroughness: To be rated
- Writing Quality: To be rated
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting](timeguard_channel-wise_pool_training_for_backdoor_defense_in_time_series_forecas.md)
- [\[ICML 2026\] Dual-branch Robust Unlearnable Examples](dual-branch_robust_unlearnable_examples.md)
- [\[NeurIPS 2025\] Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization](../../NeurIPS2025/ai_safety/dual-flow_transferable_multi-target_instance-agnostic_attacks_via_in-the-wild_ca.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[ICLR 2026\] Hide and Find: A Distributed Adversarial Attack on Federated Graph Learning](../../ICLR2026/ai_safety/hide_and_find_a_distributed_adversarial_attack_on_federated_graph_learning.md)

</div>

<!-- RELATED:END -->
