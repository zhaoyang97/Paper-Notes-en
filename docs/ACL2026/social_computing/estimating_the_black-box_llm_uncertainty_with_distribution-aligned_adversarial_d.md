---
title: >-
  [Paper Note] Estimating the Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation
description: >-
  [ACL 2026][Social Computing][Black-box Uncertainty] DisAAD is proposed: a small proxy model (only 1% of the target model's size) learns "whether the black-box LLM knows the answer" through "distribution alignment + adver…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Black-box Uncertainty"
  - "Adversarial Distillation"
  - "Proxy Model"
  - "Evidential Deep Learning"
  - "Hallucination Detection"
date: 2026-05-08
content_hash: f5e90ec831ca8cef
---

# Estimating the Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation

**Conference**: ACL 2026  
**arXiv**: [2605.05777](https://arxiv.org/abs/2605.05777)  
**Code**: https://github.com/huizi-Cui/DisAAD  
**Area**: LLM Calibration / Uncertainty / Black-box Models / Distillation  
**Keywords**: Black-box Uncertainty, Adversarial Distillation, Proxy Model, Evidential Deep Learning, Hallucination Detection

## TL;DR
DisAAD is proposed: a small proxy model (only 1% of the target model's size) learns "whether the black-box LLM knows the answer" through "distribution alignment + adversarial distillation." By leveraging evidential deep learning to decompose proxy logits into epistemic and aleatoric uncertainty, real-time uncertainty of closed-source models like GPT-4/Claude can be estimated with a single response, achieving an average AUROC improvement of 18.2% and AUPR of 22.9% over black-box baselines.

## Background & Motivation

**Background**: LLMs have made significant strides in complex reasoning and generation, yet hallucination remains the primary obstacle to deployment. Uncertainty Quantification (UQ) is a core method for models to proactively signal low confidence. Mainstream approaches fall into three categories: (1) self-evaluation, where models evaluate themselves (requires fine-tuning and lacks reliability); (2) multi-sample, checking consistency across repeated outputs (Semantic Entropy / EigV / CoCoA / SAR); (3) single-sample, directly reading token probabilities/logits/hidden states (LogTokU / CCP / Focus).

**Limitations of Prior Work**: (1) Multi-sample methods require multiple inferences for the same prompt, leading to high deployment costs and latency; they also fail when the model is "consistently wrong." (2) Single-sample methods require access to internal logits or hidden states, which is **completely infeasible for commercial closed-source models like GPT-4 / Claude that only expose APIs**. (3) Self-evaluation shows poor accuracy, and larger, more instructive LLMs (especially commercial ones) tend to "feign confidence" with plausible-sounding wrong answers, making hallucinations harder to detect than in smaller models.

**Key Challenge**: Commercial closed-source LLMs are the mainstays of real-world deployment. They offer zero exposure to internal states and tend toward overconfidence. Existing single-sample UQ methods require logits and assume well-calibrated models, neither of which holds true here.

**Goal**: (1) Provide real-time uncertainty for a single response without internal access or repeated sampling. (2) Use a small proxy model to "expose" uncertainty signals on behalf of the black-box model. (3) Decompose uncertainty into epistemic (knowledge gap) and aleatoric (data noise) dimensions via evidential learning.

**Key Insight**: This work cites findings from Zhou 2024 / Steyvers 2025: smaller LLMs more frequently refuse to answer difficult questions and are better calibrated. Since using a small model to measure a large model's uncertainty may be more reliable than self-evaluation, the key is to precisely align the output distribution of the small proxy to the high-probability regions of the large black-box model.

**Core Idea**: Train a LoRA-based small proxy via adversarial distillation (generator + discriminator) to learn "what the black-box model responds" in the prompt space. Then, derive AU and EU using evidential learning from the logits exposed by the proxy.

## Method

### Overall Architecture
The framework consists of two phases: (1) **DisAAD Training**—Construct a distillation dataset $\mathcal{D}_{\text{distill}}$: query the black-box $\mathcal{M}_{\text{B}}$ multiple times for each prompt $\bm{x}^{(i)}$, obtain a response pool $D_{\text{B}}^{(i)}$, and select the Top-$M$ entries based on mutual semantic consistency as representatives of high-probability regions. Use a LoRA proxy $\mathcal{M}_p$ as the generator and add a discriminator $\mathcal{M}_D$. Alternately optimize them so the proxy output is indistinguishable from the black-box at both token and sequence levels. (2) **Proxy-guided UQ Inference**—Given a response $\bm{y}_B$ from the target model, perform teacher-forcing on the proxy model to replay this response. Extract the top-K tokens from the logits at each position as Dirichlet parameters $\alpha_k=\text{ReLU}(\bm{z}_{t,k})$, then calculate AU, EU, and overall reliability $R(u_t)=-\text{AU}(u_t)\cdot \text{EU}(u_t)$.

### Key Designs

1. **Distribution-Aligned Sampling**:
    - **Function**: Accurately directs distillation data to the high-probability regions of "what the black-box actually outputs" rather than long-tail noise, without exceeding budgetary limits.
    - **Mechanism**: Queries $\mathcal{M}_B$ multiple times per prompt to get a candidate pool $D_B^{(i)}$, then ranks them by mutual semantic consistency. Only the Top-$M$ responses are kept as representatives of high-probability mass to construct $\{(\bm{x}^{(i)}, \bm{y}_B^{(i,j)})\}$. Prompts cover both open-domain dialogue and task-specific data to ensure generalization.
    - **Design Motivation**: Real output distributions of black-box models are long-tailed; direct collection introduces noise. Semantic consistency filtering empirically estimates the high-probability region, allowing the proxy to align only with the "sincere" responses of the black-box.

2. **Adversarial Distillation**:
    - **Function**: Enables the small proxy to align with the target distribution at both token and sequence levels, exceeding the precision of standard next-token cross-entropy.
    - **Mechanism**: The proxy $\mathcal{M}_p$ is trained with LoRA $W=W_0+BA$, aiming to minimize $\min_\theta \mathcal{L}(\theta)=\mathcal{L}_{\text{task}}(\theta)+\lambda \mathcal{L}_{\text{reg}}(\theta)$. $\mathcal{L}_{\text{task}}=-\frac{1}{NM}\sum_{i,j}\sum_t \log P_\theta(y_t\mid y_{<t})$ is standard token-level distillation. $\mathcal{L}_{\text{reg}}=-\frac{1}{NM}\sum_{i,j}\log\mathcal{M}_D(\bm{x}^{(i)}, \bm{y}_P^{(i,j)}; \phi)$ encourages generated responses to deceive the discriminator. The discriminator is trained via $\mathcal{L}_D(\phi)$ to distinguish proxy outputs from black-box responses.
    - **Design Motivation**: Pure next-token loss lacks sequence-level constraints, causing proxies to learn token-level averages but drift semantically. The discriminator pushes alignment to the sequence level, ensuring the "style" of the entire output matches the black-box, thus making the logits discriminative during replay.

3. **Dual Uncertainty via Dirichlet (EAL)**:
    - **Function**: Converts logits exposed during the proxy's replay of black-box responses into interpretable epistemic and aleatoric uncertainty metrics.
    - **Mechanism**: For each replayed token, top-K logits are converted to evidence via $\alpha_k=\text{ReLU}(\bm{z}_{t,k})$, with $\alpha_0=\sum_k \alpha_k$. Aleatoric Uncertainty (AU) $\text{AU}(u_t)=-\sum_k \frac{\alpha_k}{\alpha_0}(\psi(\alpha_k+1)-\psi(\alpha_0+1))$ reflects the sharpness of the distribution. Epistemic Uncertainty (EU) $\text{EU}(u_t)=\frac{K}{\sum_k(\alpha_k+1)}$ reflects total evidence strength. Reliability is $R(u_t)=-\text{AU}(u_t)\cdot\text{EU}(u_t)$.
    - **Design Motivation**: Softmax normalization loses absolute evidence scale. Calculating entropy from probabilities cannot distinguish "high confidence under sparse evidence" from "high confidence under rich evidence." Dirichlet modeling decouples EU (knowledge) and AU (data), formalizing the detection of the "feigned confidence" failure mode.

### Loss & Training
Jointly minimize $\mathcal{L}(\theta)=\mathcal{L}_{\text{task}}+\lambda\mathcal{L}_{\text{reg}}$; minimize $\mathcal{L}_D(\phi)$ for the discriminator; alternate updates. LoRA rank $r\ll d$. Distillation data is sampled from large-scale dialogue and task sets, using Top-$M$ semantically consistent responses per prompt. At inference, top-K logits compute Dirichlet parameters.

## Key Experimental Results

### Main Results
On multiple QA and hallucination detection tasks, compared to black-box baselines:

| Setting | DisAAD Gain vs. Strongest Black-box Baseline (Avg) |
|------|----------------------------------------|
| AUROC | **+18.2%** |
| AUPR | **+22.9%** |
| Proxy Size | Only **1%** of Target LLM |
| Sample Count | **1** (single response) |

Comparison with baselines in black-box hallucination detection / reliability prediction (based on §4):

| Method | Internal Access | Multi-sample | AUROC (Rel.) | Note |
|------|----------|-----------|-------------|------|
| Self-evaluation | No | No | Baseline | LLM self-eval |
| Semantic Entropy | No | Yes | Higher | Clustering entropy |
| EigV | No | Yes | Similar to SE | Graph-based |
| LogTokU | **Yes** | No | Strongest White-box | N/A for GPT-4/Claude |
| **DisAAD (Ours)** | **No** | **No** | **+18.2% vs SOTA** | Single response, 1% proxy |

### Ablation Study

| Configuration | Key Effect | Interpretation |
|------|---------|------|
| Full DisAAD | Best AUROC / AUPR | Complete model |
| w/o discriminator | Significant AUROC drop | Sequence alignment missing, logits uncalibrated |
| w/o distribution-aligned sampling | Performance drop | Long-tail noise pollutes high-prob mass |
| Only AU / Only EU | Inferior to $R=-\text{AU}\cdot\text{EU}$ | Dual uncertainties are complementary |
| Proxy Size 1% → 0.1% | Performance collapse | Proxy too small to absorb distribution |

### Key Findings
- A 1% proxy size with a single response achieves an 18.2% AUROC gain, challenging the notion that black-box UQ requires multi-sampling. Accuracy in distribution alignment is more valuable than sample count.
- The adversarial discriminator is essential; without it, logit calibration collapses, showing that next-token loss biases are fatal for UQ.
- AU and EU provide orthogonal signals: High AU + Low EU denotes ambiguity with a unique answer; Low AU + High EU denotes a strong signal for a wrong answer (knowledge gap). Their product reliably identifies feigned confidence.
- Distribution-aligned sampling (Top-$M$ consistency) is significantly better than random sampling, validating that precision in the high-probability region is more critical than dataset size.

## Highlights & Insights
- Using a small proxy to "expose" logits on behalf of a black-box is an elegant cognitive inversion. It transforms the barrier of "no internal access" into a proxy task of "finding an equivalent accessible distribution."
- The combination of discriminator and token loss provides sequence-level alignment and a natural termination signal ("indistinguishable"), avoiding manual stopping rules.
- Decomposition into AU/EU corresponds to "knowledge gap" vs. "answer ambiguity," offering actionable value: use retrieval when EU is high, or ask for clarification when AU is high.
- The 1% proxy size implies that even as target models scale, UQ costs remain nearly constant, which is significant for deployment.

## Limitations & Future Work
- The distillation phase requires multiple queries to black-box APIs, incurring a one-time data collection cost.
- Robustness to out-of-distribution (OOD) input is not fully verified, as proxies align with training distributions.
- Dirichlet conversion involves several hyperparameters (K, smoothing constants) that may require tuning across different black-box models.
- Updates to target models (e.g., GPT-4 to GPT-4 Turbo) might necessitate proxy retraining.
- Adversarial optimization can be unstable; variants like Wasserstein or hinge loss were not explored in depth.

## Related Work & Insights
- **vs Multi-sample (SE/EigV)**: Prior methods require multiple queries, causing latency and cost. DisAAD handles single responses and identifies "consistent errors" where multi-sampling fails.
- **vs White-box (LogTokU/Focus)**: Those require internal logits. DisAAD "transfers" white-box signals to a proxy, making these techniques indirectly available for GPT-4/Claude.
- **vs Self-evaluation**: Instruction-tuned models tend to overrate themselves. DisAAD uses an objective small proxy to bypass this overconfidence bias.
- **vs Knowledge Distillation**: Traditional distillation aims to replace the large model for inference. DisAAD uses the small model as an "uncertainty sensor" to assist rather than replace.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines UQ with adversarial distillation and evidential learning for black-box LLMs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison across multiple tasks, black-boxes, and baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and intuitive figures.
- Value: ⭐⭐⭐⭐⭐ High engineering value for real-time UQ of closed-source flagship models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)
- [\[ACL 2026\] Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning](prompt-level_distillation_a_non-parametric_alternative_to_model_fine-tuning_for_.md)
- [\[ICML 2026\] IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection](../../ICML2026/social_computing/ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti.md)
- [\[ACL 2026\] Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation](beyond_the_crowd_llm-augmented_community_notes_for_governing_health_misinformati.md)
- [\[ACL 2026\] Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews](justice_in_judgment_unveiling_hidden_bias_in_llm-assisted_peer_reviews.md)

</div>

<!-- RELATED:END -->
