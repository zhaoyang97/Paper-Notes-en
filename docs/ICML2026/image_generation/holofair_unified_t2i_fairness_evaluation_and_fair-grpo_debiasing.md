---
title: >-
  [Paper Note] HoloFair: Unified T2I Fairness Evaluation and Fair-GRPO Debiasing
description: >-
  [ICML2026][Image Generation][T2I Fairness] This paper constructs HoloFair, a unified fairness benchmark for T2I models (comprising the SpaFreq dual-stream attribute classifier + MGBI multi-attribute geometric mean metric…
tags:
  - "ICML2026"
  - "Image Generation"
  - "T2I Fairness"
  - "MGBI"
  - "SpaFreq Classifier"
  - "Fair-GRPO"
  - "Multi-attribute Reward"
date: 2026-05-08
content_hash: 14335ddf833f8aa5
---

# HoloFair: Unified T2I Fairness Evaluation and Fair-GRPO Debiasing

**Conference**: ICML2026  
**arXiv**: [2605.24687](https://arxiv.org/abs/2605.24687)  
**Code**: https://github.com/1059684669/HoloFair (Available)  
**Area**: AI Safety / Fairness / Text-to-Image / RL Alignment  
**Keywords**: T2I Fairness, MGBI, SpaFreq Classifier, Fair-GRPO, Multi-attribute Reward

## TL;DR
This paper constructs HoloFair, a unified fairness benchmark for T2I models (comprising the SpaFreq dual-stream attribute classifier + MGBI multi-attribute geometric mean metric). Based on this, it proposes Fair-GRPO: using log-ratio multi-attribute per-prompt rewards + KL-regularized GRPO to improve MGBI on SD3.5-Medium from 0.5211 to 0.6772 (+29.9%), while maintaining or slightly improving image quality.

## Background & Motivation

**Background**: Large-scale T2I diffusion/Transformer models (SDXL, SD3.5, Flux, SANA, Show-o, Bagel, etc.) have achieved extreme realism and semantic alignment, yet demographic biases remain prevalent—even with neutral prompts like "a clear photo of a person," outputs show severe imbalances across gender, age, and race.

**Limitations of Prior Work**: Existing fairness evaluation methods suffer from two blind spots. First, they **evaluate only single dimensions**, such as Luccioni et al. focusing on default distributions or Park et al. testing occupational bias, while **ignoring implicit biases triggered by social descriptors** like "competence/warmth"—adding "professional" or "aggressive" to a prompt can pull the distribution toward a specific group. Second, **fairness in default prompt distributions does not equate to true fairness**: experiments show that SDXL has the highest ID score (0.8186) but nearly the lowest $\text{CA}_{0.10}$, meaning diversity collapses under bias-triggering contexts.

**Key Challenge**: Debiasing methods face a trilemma—large-scale fine-tuning (Shen et al.) can alter distributions but is computationally expensive and prone to catastrophic forgetting; inference-time post-processing (Friedrich et al., Chuang et al.) introduces unacceptable latency; cross-attention concept editing (Gandikota et al.) has limited coverage. **Fairness, fidelity, and efficiency are difficult to balance simultaneously.**

**Goal**: (1) Design an **evaluation metric + benchmark** capable of detecting both default and semantically triggered biases; (2) Propose a training method for systematic debiasing without sacrificing generation quality.

**Key Insight**: Fairness is formalized as "distributional consistency across semantic contexts." Drawing from the Stereotype Content Model (SCM) in social psychology, 9 semantic trigger words across competence/warmth dimensions (e.g., aggressive, compassionate, professional) are hand-picked as stress tests. **Evaluation side**: Uses the geometric mean of "default entropy + triggered entropy" to punish imbalance. **Debiasing side**: Translates the degree of distributional uniformity directly into an RL reward signal for GRPO.

**Core Idea**: SCM triggers expose deep semantic biases in T2I models. Log-ratio per-prompt rewards translate "uniform distribution" into an optimizable signal for GRPO, while KL regularization prevents reward hacking.

## Method

### Overall Architecture
The HoloFair end-to-end pipeline consists of three stages: (1) **Dataset Construction**—Synthesizing Gen/Eval/Train prompt sets + Real Person Dataset (RBD) containing FairFace, UTKFace, and ~20k portraits synthesized by 8 T2I models, unified under the FairFace taxonomy (2 genders, 3 age groups, 5 ethnicities); (2) **Classifier Training**—Training the SpaFreq dual-stream classifier based on DINOv2-Base; (3) **Fairness Evaluation**—Labeling T2I outputs with SpaFreq and scoring with MGBI. Post-evaluation, **Fair-GRPO** uses the same classifier as a reward model to perform RL fine-tuning on the target T2I LoRA (SD3.5M / SD1.5).

### Key Designs

1.  **SpaFreq Dual-stream Attribute Classifier**:
    - **Function**: Assigns demographic attribute labels (Gender/Age/Race) to T2I generated images; serves as the shared infrastructure for evaluation and RL rewards.
    - **Mechanism**: Adds a **spatial stream + frequency stream** dual-view on top of a DINOv2-Base backbone. The frequency stream converts RGB to grayscale and performs db4 discrete wavelet decomposition to obtain low-frequency $cA$ and two high-frequency components $cH$ (horizontal) and $cV$ (vertical). These are min-max normalized per channel and concatenated as $X_{\text{freq}}$. $X_{\text{spatial}}$ and $X_{\text{freq}}$ are concatenated along the batch dimension for DINO processing to obtain CLS embeddings $\mathbf{f}_s, \mathbf{f}_w$. Fusion uses a learnable weight $w_{\text{fusion}}$ (initially 0) via sigmoid $\alpha = 1/(1+e^{-w_{\text{fusion}}})$ followed by channel concatenation $\mathbf{z} = \text{Concat}(\alpha \mathbf{f}_s, (1-\alpha)\mathbf{f}_w)$ before passing through a small MLP head.
    - **Design Motivation**: The spatial view provides high-level semantics, but fine textures are "captured" by semantics. The frequency view serves as a non-semantic complement, strengthening texture and edge details—race classification particularly relies on skin texture. Ablations show frequency info improves race accuracy from 85.57 to 91.89, and adaptive fusion improves overall accuracy (all three attributes correct) from 79.67 to 89.67.

2.  **MGBI Multi-attribute Geometric Mean Fairness Metric**:
    - **Function**: Uses a scalar in $[0,1]$ to characterize both "default diversity" and "semantic robustness" of T2I models, allowing comparisons across dimensions with different attribute counts.
    - **Mechanism**: For any distribution $p$, normalized entropy is calculated as $h_a(p) = -\sum_c \hat{p}(c)\log\hat{p}(c) / \log|C_a|$. **Intrinsic Diversity (ID)** takes the geometric mean of normalized entropy for neutral prompts $s_0$ across attributes: $\text{ID} = (\prod_{a} \max(\epsilon, h_a(\hat{p}_a)))^{1/|\mathcal{A}|}$. **Context-Robust Conditional Diversity** calculates the geometric mean entropy for each of the 9 SCM triggers and takes the 10th percentile as the near-worst case: $\text{CA}_q = \text{Quantile}_q(\{(\prod_a h_a(\hat{p}_a(\cdot|s)))^{1/|\mathcal{A}|}\}_{s\in\mathcal{S}})$. Finally, $\text{MGBI} = \sqrt{\text{ID} \cdot \text{CA}_q}$.
    - **Design Motivation**: Entropy is used instead of variance ratios because it explicitly penalizes mode collapse. The geometric mean is used instead of the arithmetic mean so that a high score in one dimension cannot mask imbalance in another. The 10th percentile captures tail behavior, avoiding deception by high-variance means. The SDXL case justifies this: its default distribution appears most fair (ID=0.8186), but $\text{CA}_{0.10}$ is only 0.2865, causing MGBI to downgrade it immediately.

3.  **Fair-GRPO Multi-attribute Per-prompt Log-ratio Reward**:
    - **Function**: Translates "level of distributional uniformity" into dense RL signals for GRPO optimization of LoRA parameters with KL regularization.
    - **Mechanism**: For each prompt $p$, $N$ images are sampled, and SpaFreq provides intra-group counts $N^a_k$ for attribute $a$ category $k$. The base reward uses an **adaptive log-ratio**: $r_{\text{base}}(k,a) = \log((N - N^a_k + \epsilon)/(N^a_k + \epsilon))$—majority classes receive negative penalties and minority classes receive positive rewards. Multi-class attributes are zero-centered $r_{\text{fair}}(k,a) = r_{\text{base}}(k,a) - \bar{r}_{\text{base}}(a)$ so the reward is exactly 0 at perfect equilibrium. Signals are clipped to $[-5, 5]$. The final reward for an image is the weighted sum $R(I_p) = \sum_a w_a \cdot r_{\text{clip}}(F(I_p), a)$. During the diffusion process, rewards are reused across timesteps, and a per-prompt-per-timestep historical table tracks mean/variance to compute advantage $A(I_p, t) = (R(I_p, t) - \mu_R^{p,t})/(\sigma_R^{p,t} + \epsilon)$. The KL-regularized GRPO objective is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \beta \mathcal{L}_{\text{KL}}$ (using PPO-Clip + pixel-space noise mean KL approximation, $\beta = 0.05$).
    - **Design Motivation**: The log-ratio form naturally fits the "target uniform distribution" and is stable for zero counts. Zero-centering makes the signal for multi-class attributes equivalent to binary ones, preventing equilibrium drift. KL regularization is critical to prevent reward hacking—simple fairness maximization leads to low-quality images. KL constraints keep the policy near the reference model, preserving CLIP-Score and FID.

### Loss & Training
LoRA (rank=32) is applied to transformer attention q/k/v and output projections. AdamW lr=5e-5, $\beta_{\text{KL}}=0.05$, trained on 6×RTX 4090. The 10k neutral prompts in the Train set are strictly isolated from the Eval set. Stability is ensured via per-prompt-per-timestep reward statistics, EMA, mixed precision, and gradient clipping.

## Key Experimental Results

### Main Results

T2I Model Fairness Evaluation (8 models, Eval set 750 prompts):

| Model | Type | ID ↑ | $\text{CA}_{0.10}$ ↑ | MGBI ↑ |
|------|------|------|----------|--------|
| Flux1-dev | Gen-only | 0.6858 | **0.6702** | **0.6780** |
| SANA-1.5 | Gen-only | 0.7820 | 0.3821 | 0.5466 |
| SD3.5-Large | Gen-only | 0.7480 | 0.3693 | 0.5255 |
| SDXL | Gen-only | **0.8186** | 0.2865 | 0.4843 |
| Show-o | Unified | 0.7005 | 0.6013 | 0.6490 |
| Bagel | Unified | 0.6152 | 0.5004 | 0.5549 |
| Harmon | Unified | 0.5320 | 0.4661 | 0.4979 |
| Blip3-o | Unified | 0.4030 | 0.1856 | 0.2735 |

Fair-GRPO Debiasing Comparison (SD3.5M Baseline MGBI=0.5211, SD1.5 Baseline MGBI=0.6554):

| Method | Backbone | MGBI ↑ | CLIP-Score ↑ | FID ↓ |
|------|------|--------|--------------|-------|
| Baseline | SD3.5M | 0.5211 | 0.2288 | 143.26 |
| UCE | SD3.5M | 0.5769 | 0.2307 | 137.34 |
| Balancing_Act | SD3.5M | 0.5785 | 0.2311 | 155.60 |
| **Fair-GRPO** | SD3.5M | **0.6772** | **0.2317** | **135.09** |
| Baseline | SD1.5 | 0.6554 | 0.2197 | 165.37 |
| EFA | SD1.5 | 0.7084 | 0.2211 | 139.97 |
| **Fair-GRPO** | SD1.5 | **0.7881** | **0.2237** | **134.51** |

### Ablation Study

SpaFreq Classifier Component Ablation (Overall = all three attributes correct):

| Configuration | Gender | Age | Race | Overall |
|------|--------|-----|------|---------|
| ViT-B + F.T. | 85.82 | 75.68 | 78.56 | 71.28 |
| DINO + F.T. | 91.20 | 82.85 | 85.57 | 79.67 |
| DINO + Fre. + F.T. | 96.78 | 91.12 | 91.89 | 85.33 |
| **DINO + Fre. + W.F. + F.T.** | **97.88** | **95.36** | **92.28** | **89.67** |

Fair-GRPO Multi-attribute Reward Ablation (SD3.5M):

| $R_{\text{gender}}$ | $R_{\text{age}}$ | $R_{\text{race}}$ | MGBI ↑ | CLIP-Score ↑ |
|---|---|---|--------|--------------|
| | | | 0.5211 | 0.2288 |
| ✓ | | | 0.6302 | 0.2253 |
| | ✓ | | 0.5813 | 0.2305 |
| | | ✓ | 0.5905 | 0.2310 |
| ✓ | ✓ | ✓ | **0.6772** | **0.2317** |

### Key Findings
- **High ID does not guarantee low bias**: SDXL’s default distribution is the most fair (ID=0.8186), but its $\text{CA}_{0.10}$ is nearly the lowest (0.2865). The massive gap between CA-mean and $\text{CA}_{0.10}$ suggests diversity collapses under semantic triggers—challenging the paradigm of only evaluating default distributions.
- **Fairness regularization can improve semantic alignment**: The CLIP-Score of the Fair-GRPO version increased. This is explained as encouraging the model to explore more diverse image spaces, which results in more robust semantic representations—counter-intuitive but consistent with the idea of "diversity as regularization."
- **Unified multimodal models are less fair than gen-only models**: Gen-only models averaged ID≈0.75, while Unified models averaged ID≈0.56. Collaborative training may sacrifice representational diversity for "generality."
- **Triple-attribute rewards exhibit synergy**: While single-attribute rewards improve MGBI, the best result (0.6772) requires all three, suggesting debiasing across dimensions is not independent.

## Highlights & Insights
- **SCM triggers as stress tests**: Bringing social psychology’s "competence-warmth" dimensions into prompt templates is a clever "theory-grounded adversarial set construction" that can be applied to any implicit bias measurement in generative tasks.
- **Geometric Mean + 10th Percentile**: The philosophy is "one dimension cannot compensate for another." This principle is applicable to any multi-objective evaluation (fairness, safety, alignment) and is harder to "game" than arithmetic means.
- **Log-ratio per-prompt reward**: This reward form is rare in RLHF but ideal for "distributional balance." It provides positive signals to minority classes and negative to majority classes, and zero-centering ensures signal scale consistency.
- **Classifier reuse as reward model**: Using the same SpaFreq for both evaluation and RL rewards reduces cost and ensures target consistency, though it necessitates RL-specific precautions like reward hacking prevention with KL regularization.

## Limitations & Future Work
- **Attribute dimensions only cover gender/age/race**: Due to resource constraints, other dimensions like disability, religion, or body type are omitted. Extending this requires retraining classifiers and re-validating the MGBI geometric mean.
- **Discrete demographic classification is reductionist**: The FairFace binary gender and five-race taxonomy simplifies identity, potentially introducing new biases—a limitation acknowledged in the Impact Statement.
- **Inherited classifier bias**: SpaFreq has an 89.67% overall accuracy. The remaining 10% errors enter policy updates as RL noise, potentially teaching the model the classifier’s own biases over time.
- **Limited SCM trigger set**: Only 9 triggers are used, which may miss subtle semantic triggers (e.g., industry jargon). Future work could use LLMs to generate larger adversarial prompt sets.
- **Lack of multilingual validation**: All templates are in English; cross-lingual bias patterns may differ.

## Related Work & Insights
- **vs. Shen et al. (Balanced dataset fine-tuning)**: They use full-parameter fine-tuning on balanced sets, which is costly and causes forgetting. Ours uses LoRA + RL, maintaining or improving CLIP-Score.
- **vs. Friedrich et al. / Chuang et al. (Inference-time guidance)**: They modify text embeddings during inference, which is slow. Ours is a one-time training solution with no inference overhead.
- **vs. UCE / Balancing_Act (Concept editing)**: They edit cross-attention or add auxiliary networks, which has limited scope and can break other concepts. Fair-GRPO preserves general capabilities via KL regularization.
- **vs. EFA (Park et al. 2025)**: EFA is a state-of-the-art debiasing method but only tests occupational bias. MGBI covers implicit SCM triggers, and Fair-GRPO outperforms EFA on SD1.5 (0.7881 vs 0.7084).
- **vs. Standard GRPO/RLHF**: Standard GRPO uses human preferences; this work uses the degree of distributional uniformity as a reward, establishing a paradigm for "using structured metrics as reward models" for alignment tasks with known prior distribution shapes.

## Rating
- Novelty: ⭐⭐⭐⭐ MGBI geometric mean + SCM triggers + log-ratio rewards are a novel combination, though components like GRPO and dual-stream classifiers have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated 8 T2I models + 5 baselines; ablated classifier components and attribute rewards, though only two backbones (SD1.5/SD3.5M) were tested.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, thorough explanation of metric design, and consistent formula numbering.
- Value: ⭐⭐⭐⭐ Provides a sustainable fairness benchmark (code + data) and a practical debiasing recipe that preserves quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention](../../ICCV2025/image_generation/fair_generation_without_unfair_distortions_debiasing_text-to-image_generation_wi.md)
- [\[AAAI 2026\] T2I-RiskyPrompt: A Benchmark for Safety Evaluation, Attack, and Defense on Text-to-Image Model](../../AAAI2026/image_generation/t2i-riskyprompt_a_benchmark_for_safety_evaluation_attack_and_defense_on_text-to-.md)
- [\[ICML 2026\] Conformal Reliability: A New Evaluation Metric for Conditional Generation](conformal_reliability_a_new_evaluation_metric_for_conditional_generation.md)
- [\[ICML 2026\] LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching](lithogrpo_fast_inverse_lithography_via_grpo_reinforced_flow_matching.md)
- [\[ICML 2026\] AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters](ateliereval_agentic_evaluation_of_humans_llms_as_text-to-image_prompters.md)

</div>

<!-- RELATED:END -->
