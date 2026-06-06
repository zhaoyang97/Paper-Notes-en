---
title: >-
  [Paper Note] Causal Fine-Tuning under Latent Confounded Shift
description: >-
  [ICML 2026][Causal Inference][Causal fine-tuning] This paper proposes Causal Fine-Tuning (CFT): embedding an SCM-inspired decomposition of "high-level stable feature $C$ + low-level confounder-sensitive feature $\Phi$" i…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "Causal fine-tuning"
  - "latent confounding"
  - "front-door adjustment"
  - "single-domain generalization"
  - "BERT"
date: 2026-05-08
content_hash: 35500770fac77732
---

# Causal Fine-Tuning under Latent Confounded Shift

**Conference**: ICML 2026  
**arXiv**: [2410.14375](https://arxiv.org/abs/2410.14375)  
**Code**: https://github.com/jialin-yu/CausalFineTuning (available)  
**Area**: NLP Understanding / Causal Representation Learning / Out-of-Distribution Generalization  
**Keywords**: Causal fine-tuning, latent confounding, front-door adjustment, single-domain generalization, BERT

## TL;DR
This paper proposes Causal Fine-Tuning (CFT): embedding an SCM-inspired decomposition of "high-level stable feature $C$ + low-level confounder-sensitive feature $\Phi$" into standard BERT fine-tuning, and using a front-door style do-calculus adjustment formula for prediction. CFT significantly outperforms SFT/SWA/WISE and other single-domain generalization baselines under text pseudo-correlation injection attacks.

## Background & Motivation
**Background**: Downstream adaptation of foundation models (BERT/GPT/CLIP) almost universally follows a "full-parameter SFT or LoRA + ERM" black-box approach, treating all input features equally.

**Limitations of Prior Work**: When training data contains pseudo-correlations driven by latent variables (e.g., "Amazon" label being strongly correlated with positive sentiment), models learn shortcuts; if the pseudo-correlation reverses at deployment (Amazon becomes mainly negative reviews), model performance collapses. Traditional invariance methods like IRM require multi-domain annotation or environment labels, and are powerless for single-domain data.

**Key Challenge**: In standard fine-tuning, $p(y\mid x;\sigma)=\sum_u p(y\mid u,x)\,p(u\mid x;\sigma)$, where $p(u\mid x;\sigma)$ changes with the environment. However, with single-domain observations, neither $u$ nor environment labels can be identified, and minimax robust optimization under such "unseen confounding" is either unidentifiable or overly conservative.

**Goal**: Under single-domain fine-tuning, decompose input representations into (i) a cross-environment stable causal component $C$; (ii) an environment-sensitive low-level local component $\Phi$, and use do-calculus adjustment to "remove" dependence on $\Phi$.

**Key Insight**: Treat the pretrained LM itself as "another implicit environment"—the frozen model provides $R_0$, the fine-tuned model provides $R_1$, and the difference between the two views naturally reveals which dimensions are domain-sensitive and which are cross-domain stable.

**Core Idea**: Use SCM as an inductive bias, stipulating that $R_0, R_1$ are explained by a shared stable causal latent variable $C$ for consistency, and that low-level local features $\Phi$ capture pseudo-correlations. Then, use front-door adjustment $p(y\mid \mathrm{do}(x))=\sum_{\Phi',x'} p(y\mid\Phi',C)\,p(\Phi'\mid x')\,p(x')$ and perform Monte Carlo estimation by shuffling $\Phi$ within the batch.

## Method

### Overall Architecture
During training, two BERTs are maintained: a frozen pretrained model $p(r_0\mid x)$ and a fine-tuned model $p(r_1\mid x)$. Each sample passes through three heads: (1) SFT head performs supervised classification on $R_1$; (2) Causal head learns a mapping from $(R_0, R_1)$ to the stable causal representation $C$; (3) Local head extracts $\Phi$ from the embedding layer of $R_1$. The final predictor $p(y\mid C,\Phi)$ applies do-calculus Monte Carlo adjustment by shuffling $\Phi$ within the mini-batch $K=20$ times. At inference, the frozen model is discarded, $C$ is estimated solely from $R_1$, and model size matches standard SFT.

### Key Designs

1. **SCM-based Causal Identification Scaffold**:

    - **Function**: Combines "high-level stable semantics $C$ + low-level confounder feature $\Phi$ + unobservable confounders $U_S, U_\Phi$ + environment $\sigma$" into a single graph (Fig.2(b)), demonstrating that $p(y\mid \mathrm{do}(x))$ can be written in terms of $p(y\mid\Phi,C)$ and marginal $p(x)$ under single-domain data.
    - **Mechanism**: Assumes $\sigma$ only affects $R_1$ via $S_1$, and the effect of $\Phi$ on $Y$ is entirely mediated by $C$ (front-door structure). Uses Von Kügelgen's identifiability theorem to express $C$ as an invariant projection $p(C\mid R_0)\approx p(C\mid R_1)$.
    - **Design Motivation**: The causal graph clarifies that "train-test distribution shift is equivalent to changes in $p(u\mid x;\sigma)$", thus transforming "robustness to all $\sigma$" into "training under the maximum entropy default environment via $\mathrm{do}(x)$", avoiding the over-conservatism and unidentifiability of minimax.

2. **Dual-view Causal Representation Alignment $\mathcal{L}_C$**:

    - **Function**: Projects the frozen model's sentence vector $R_0$ and the fine-tuned model's $R_1$ into the same stable causal space $C$.
    - **Mechanism**: Minimizes $\mathcal{L}_C=\mathbb{E}\,\|p(c\mid r_0)-p(c\mid r_1)\|_2^2 - H(p(c\mid r_0)) - H(p(c\mid r_1))$, where the first term enforces cross-view invariance, and the two negative entropy terms prevent representation collapse.
    - **Design Motivation**: Treats the "pretrain vs fine-tune" pair as a natural "pseudo-environment pair", serving as a substitute for IRM's multi-domain signals. Single-domain data suffices to extract cross-domain stable components; replacing with $(R_1, R_1)$ (failure mode experiment) immediately degenerates to SFT.

3. **Local Patch Feature $\Phi$ + Front-door Batch Shuffle**:

    - **Function**: Extracts local low-level features from the embedding layer of the fine-tuned model as proxies for pseudo-correlations, and uses do-calculus adjustment to sever the causal path of $\Phi$.
    - **Mechanism**: Splits the token sequence into 10 non-overlapping patches, applies mean pooling, and passes through an MLP to obtain $\Phi=\mathrm{MLP}(\frac{1}{10}\sum_i p_i)$. At prediction, randomly shuffles $\Phi$ within the mini-batch to get $\Phi'\sim \hat{p}_B(\Phi)$, computes $\mathbb{E}_{\Phi'}[p(y\mid C,\Phi')]$, and averages over $K=20$ Monte Carlo samples.
    - **Design Motivation**: The embedding layer is closest to "raw word forms/data source" and other low-level pseudo-correlation cues (as observed by LeCun et al.). Shuffling $\Phi$ is equivalent to breaking the active collider path $\sigma\to S_1\to R_1\leftrightarrow\Phi\leftrightarrow Y$, pulling prediction from $p(y\mid x)$ back to $p(y\mid \mathrm{do}(x))$.

### Loss & Training
The total objective is $\mathcal{L}=\mathcal{L}_{\text{SFT}}+\mathcal{L}_C+\mathcal{L}_{\text{adjust}}$, where $\mathcal{L}_{\text{adjust}}$ is cross-entropy based on shuffled $\Phi'$. Optimizer: AdamW, learning rate $5\times 10^{-5}$, 10 epochs, BERT-base initialization; a frozen copy is retained only for $R_0$ extraction and discarded after training.

## Key Experimental Results

### Main Results

| Dataset | Test Spurious 10% | SFT | SWA | WISE | CFT |
|---------|------------------|-----|-----|------|-----|
| Yelp (Exp1, stop-word attack) | F1 | 49.24 | 62.92 | 55.91 | **58.40** (+9.16 vs SFT) |
| Amazon (Exp1) | F1 | 49.33 | 59.75 | 50.40 | **56.40** (+7.07) |
| Amazon (Exp2, data-source attack) | F1 | 37.78 | 47.41 | 31.83 | **49.22** (+11.44) |

On ID (90% spurious), CFT is nearly tied with SWA/SFT, but as OOD (spurious ratio drops from 70% to 10%) becomes more extreme, CFT's advantage grows. With 4× and 8× noise scaling, CFT's margin over SWA further increases.

### Ablation Study

| Configuration | Spurious 10% F1 (Amazon) | Description |
|---------------|-------------------------|-------------|
| Full CFT | 56.40 | Causal decomposition + do-calculus |
| CFT-N (no do-shuffle, directly conditions on $\Phi$) | 48.00 | Leaves active collider, OOD degrades to SFT level |
| CFT-C (predicts with $C$ only) | 53.40 | Stronger than SFT but weaker than full, indicating $\Phi$ adjustment still contributes several points |
| CFT-$\Phi$ (predicts with $\Phi$ only) | 12.40 | Nearly random, confirming $\Phi$ indeed captures pseudo-correlation |
| CFT (identical view $R_1,R_1$) | 37.24 | Failure mode: removing dual-view signal fully degenerates to SFT |

### Key Findings
- Predicting with $\Phi$ alone is nearly random (19/12 F1) on OOD, confirming it indeed captures pseudo-correlation; only after adjustment can the model withstand distribution reversal.
- Removing cross-view signal (identical view) immediately reverts to SFT level, indicating "dual-view + invariance constraint" is the true driver of representation decomposition.
- The stronger the shift, the more CFT outperforms SWA: under default noise, SWA and CFT are comparable; with 4×/8× amplified noise, CFT leads across the board, suggesting structured methods are more robust than generic regularization under severe distribution shift.
- Layer sensitivity study (Table 6) shows that using the embedding layer for $\Phi$ is most stable under strong shift, while higher layers remain competitive under weak shift, consistent with the expectation that "lower layers capture shift-sensitive cues, higher layers mix semantics".

## Highlights & Insights
- **Treating the pretrained model as a "free environment"**: Traditional causal invariance methods (IRM/REx) require collecting multiple real environments. This work leverages the naturally formed two views of frozen vs fine-tuned models, using Theorem 4.4 (Von Kügelgen) to obtain invariant representations, saving the cost of multi-domain data collection—especially suitable for NLP, where environment labels are hard to define.
- **Front-door + batch shuffle for minimal do-calculus**: Implements the abstract $p(y\mid \mathrm{do}(x))$ as a "shuffle $\Phi$ within mini-batch and average" operation at the code level, adding almost no training cost yet severing implicit collider paths—a neat and transferable idea.
- **SCM as inductive bias, not requiring ground-truth identification**: The authors clarify that $C/\Phi$ are empirical estimates, not true variables of the data-generating graph. The minimal working condition is simply that "the two views can distinguish some information"; failure mode experiments provide diagnostic signals (degrades to SFT when $C\approx\Phi$), reflecting a pragmatic approach.

## Limitations & Future Work
- Only tested on text sentiment classification with artificially injected spurious cues; does not cover real-world multi-source confounding (hospital/platform/language), so effectiveness under natural distribution shift remains unproven.
- Assumes the front-door structure holds (the effect of $\Phi$ on $Y$ is fully mediated by $C$)—if in reality $\Phi$ retains a direct causal path to $Y$, identification fails.
- Batch Monte Carlo with $K=20$ shuffles only models the within-batch distribution; estimation of $p(\Phi)$ across batches still relies on i.i.d. sampling. The number of patches (10) and the extraction layer for $\Phi$ are empirically chosen; validation is needed for ultra-long texts or multimodal data.
- The authors suggest extending to multimodal settings: cross-modal confounders living in one modality but interacting with another is a reasonable and important next step.

## Related Work & Insights
- **vs IRM/V-REx**: IRM series require multi-domain data and environment labels to learn invariant features $\Phi(x)$. This work does not require environment labels, achieving "pseudo-environment" alignment via pretrain-finetune pairs, making single-domain data usable.
- **vs SWA/WISE**: SWA/WISE are general flat minima/parameter interpolation regularizers. Under moderate shift, they are close to CFT, but as distribution shift intensifies, CFT surpasses them, demonstrating the advantage of "structured causal adjustment" over "geometric regularization".
- **vs back-door causal attention (Yue 2020, Zhang 2020)**: Back-door requires observed confounders. This work uses front-door to adapt to "implicit confounding + text" scenarios, in line with Mao 2022's front-door causal intervention, but is the first to make "do-calculus adjustment beyond ERM" a plug-and-play fine-tuning module.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines front-door adjustment and pretrain/fine-tune dual-view alignment into BERT fine-tuning; both problem framing and method construction are innovative.
- Experimental Thoroughness: ⭐⭐⭐ Covers Yelp/Amazon, multiple shift intensities, several ablations and failure modes, but only validated on synthetic spurious injection, lacking real multi-domain data.
- Writing Quality: ⭐⭐⭐⭐ Causal graphs, theorems, and algorithms are presented step by step, clearly distinguishing "identification scaffold" from "actual learned proxies", avoiding overcommitment.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play causal robustness solution for single-domain NLP fine-tuning, with practical significance for common dataset artifact reversals in deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Cyclic Counterfactuals under Shift–Scale Interventions](../../NeurIPS2025/causal_inference/cyclic_counterfactuals_under_shift-scale_interventions.md)
- [\[ICLR 2026\] Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](../../ICLR2026/causal_inference/distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)
- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](../../ICLR2026/causal_inference/rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)
- [\[AAAI 2026\] From Theory of Mind to Theory of Environment: Counterfactual Simulation of Latent Environmental Dynamics](../../AAAI2026/causal_inference/from_theory_of_mind_to_theory_of_environment_counterfactual_simulation_of_latent.md)

</div>

<!-- RELATED:END -->
