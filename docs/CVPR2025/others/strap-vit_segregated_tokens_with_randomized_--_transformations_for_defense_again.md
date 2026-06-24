---
title: >-
  [Paper Note] STRAP-ViT: Segregated Tokens with Randomized Transformations for Defense against Adversarial Patches in ViTs
description: >-
  [CVPR 2025][Adversarial Patch Defense] STRAP-ViT proposes a training-free, plug-and-play defense module for ViTs. It utilizes Jensen-Shannon divergence to segregate tokens affected by adversarial patches from benign tokens and then applies randomized composite transformations to neutralize their adversarial effects, achieving a robust accuracy within 2-3% of the clean baseline across multiple ViT architectures and attack methods.
tags:
  - "CVPR 2025"
  - "Adversarial Patch Defense"
  - "Vision Transformer"
  - "Jensen-Shannon Divergence"
  - "Token Segregation"
  - "Randomized Transformation"
  - "Plug-and-Play Defense"
date: 2026-05-08
content_hash: ba92e093805ac241
---

# STRAP-ViT: Segregated Tokens with Randomized Transformations for Defense against Adversarial Patches in ViTs

**Conference**: CVPR 2025  
**arXiv**: [2603.12688](https://arxiv.org/abs/2603.12688)  
**Code**: None  
**Area**: Others  
**Keywords**: Adversarial Patch Defense, Vision Transformer, Jensen-Shannon Divergence, Token Segregation, Randomized Transformation, Plug-and-Play Defense

## TL;DR
STRAP-ViT proposes a training-free, plug-and-play defense module for ViTs. It utilizes Jensen-Shannon divergence to segregate tokens affected by adversarial patches from benign tokens and then applies randomized composite transformations to neutralize their adversarial effects, achieving a robust accuracy within 2-3% of the clean baseline across multiple ViT architectures and attack methods. 
The core advantage of this method is that it requires absolutely no retraining or fine-tuning of existing models.

## Background & Motivation
**Background**: Vision Transformers have become the core architecture of vision AI, widely deployed in high-value scenarios such as autonomous driving, surveillance, and medical imaging, with companies investing tens of billions of dollars per quarter in AI infrastructure.

**Limitations of Prior Work**: Adversarial patches represent a physically realizable attack—simply applying a small, high-contrast patch can hijack the self-attention mechanism of ViTs, corrupting the class token and leading to high-confidence misclassifications.

**Key Challenge**: The global self-attention of ViTs is both a strength and a weakness—adversarial patches can exploit this attention hijacking to disrupt all token interactions. Existing defenses (adversarial training, patch detectors, token smoothing, certified radius) are either computationally expensive or unreliable.

**Goal**: How to detect and eliminate the impact of adversarial patches on ViT inference at an extremely low computational cost, without requiring additional training?

**Key Insight**: Tokens covered by adversarial patches exhibit significant statistical distribution differences compared to benign tokens—adversarial tokens have higher Shannon entropy and show systematic shifts in channel distributions, providing a theoretical foundation for information-theory-based detection.

**Core Idea**: Utilize JSD to locate anomalous tokens and leverage randomized composite transformations to disrupt their adversarial information, which is fully training-free and plug-and-play.

## Method

### Overall Architecture
STRAP-ViT is embedded after the patch embedding and positional encoding, and before the Transformer encoder in the ViT inference pipeline. It operates in two phases: (1) **Detection**—identifying anomalous tokens using JSD scores; (2) **Mitigation**—neutralizing adversarial noise via randomized composite transformations. The transformed tokens, along with the benign ones, are then fed into the subsequent ViT layers.

### Key Designs

1. **Detection: JSD-Based Token Segregation**

    - **Function**: Computes the Jensen-Shannon divergence of each token against a clean reference distribution to identify the subset of tokens $\mathcal{A}$ affected by the adversarial patch.
    - **Mechanism**: The token embedding $z_{\ell,t}$ is mapped to a probability simplex $p_{\ell,t} = \text{softmax}(z_{\ell,t}/T)$ using temperature-scaled softmax, and the JSD is computed relative to the clean reference $q_{\ell,t}$. The anomaly score is $s_{\ell,t} = \sqrt{\mathrm{JSD}(p_{\ell,t} \| q_{\ell,t})}$, and tokens exceeding a threshold $\tau$ are flagged as anomalous.
    - **Design Motivation**: $\sqrt{\text{JSD}}$ is a true metric and directly correlates with mutual information. Larger values indicate stronger evidence of token anomaly. It is symmetric and bounded within $[0, \log 2]$, making it more suitable for calibrating detection than pure entropy.

2. **Mitigation: Randomized Composite Token Transformations**

    - **Function**: Randomly selects and combines three transformations for anomalous tokens to eliminate adversarial effects.
    - **Mechanism**: For each anomalous token, a transformation subset $S_t \subseteq \{1,2,3\}$ and a permutation $\pi_t$ are randomly sampled to sequentially apply $L_p$ projection (suppressing extreme channel energy), affine shrinking (re-centering), and softmax temperature scaling (mitigating peak distributions).
    - **Design Motivation**: Randomized combinations prevent attackers from training immune patches via EOT (Expectation over Transformation), exponentially increasing the difficulty of adaptive attacks.

3. **Selection of Hyperparameter K**

    - **Function**: Determines the minimum number of tokens that need to be transformed for the defense to be effective.
    - **Mechanism**: Covering 50% of the adversarial patch area is sufficient to neutralize it. $K$ ranges from 2 (~1% of tokens) to 8 (~4% of tokens).
    - **Design Motivation**: Since transformations disrupt token information, there is a need to balance "patch coverage" with "minimizing information loss."

### Loss & Training
No training is required. STRAP-ViT is a purely inference-time defense module. The reference distribution is precomputed from a clean dataset, yielding zero training cost.
Computing the reference distribution $q_{\ell,t}$ requires only a single forward pass over a clean set of images, incurring minimal storage overhead.
In actual deployment, it only adds latency for JSD computation and transformation operations, which is negligible compared to the computational cost of the ViT itself.
The temperature parameter $T$ controls the smoothness of the softmax, which affects detection sensitivity—a lower $T$ makes the distribution sharper and anomalous tokens easier to detect.

## Key Experimental Results

### Main Results (ViT-B/16, ImageNet)

| Attack Method | Patch Size | No Defense Top-1 | STRAP-ViT Top-1 | Gain |
|---|---|---|---|---|
| GoogleAP | 40×40 | 3.6% | 78.8% | +75.2% |
| GoogleAP | 50×50 | 1.2% | 78.3% | +77.1% |
| LAVAN | 40×40 | 7.4% | 79.2% | +71.8% |
| GDPA | 50×50 | 12.9% | 76.1% | +63.2% |

Clean baseline is 80.5%; STRAP-ViT achieves 80.1% on clean samples (only a 0.4% drop).

### Ablation Study

| Configuration | Key Metrics | Description |
|---|---|---|
| K=2 (1% tokens) | Robust Accuracy ~78% | Effective when covering 50% of the patch |
| K=8 (4% tokens) | Robust Accuracy ~76% | Excessive transformations harm clean accuracy |
| Single transformation only | Lower than composite transformations | Random combinations increase defense diversity |

### Key Findings
- Robust accuracy is restored to within 2-3% of the clean baseline across all attack methods and datasets.
- Near-zero accuracy loss on clean samples (only -0.4%), with an extremely low false-positive rate.
- Insensitive to patch size—as the area increases by 56% from 40x40 to 50x50, the defensive efficacy remains nearly unchanged.
- On the DinoV2 backbone, clean accuracy even increases by 2.2%, indicating that JSD detection has a regularizing effect.

## Highlights & Insights
- **Zero-training, plug-and-play**: Requires no weight modification, retraining, or fine-tuning, making it extremely friendly for deployed ViTs.
- **Information-theory-driven detection**: Relies on JSD rather than heuristic thresholds, underpinned by mutual information theory, allowing calibration of anomalous evidence.
- **Randomness as a security guarantee**: Randomized transformation combinations exponentially increase the difficulty of adaptive attacks.
- **Transferable to other tokenized architectures**: Can be plugged in as long as there is a patch embedding layer, making it applicable to VLMs, multimodal models, etc.

## Limitations & Future Work
- Requires precomputing the clean reference distribution, which may need re-estimation under domain shifts.
- Transformations cause some loss of information, which may impact performance when the patch covers key target information (though experimental impact is minor).
- Validated only on classification tasks; downstream tasks like object detection and segmentation have not been tested.
- Assumes patch area $\le 5\%$, and its efficacy against larger-area attacks remains to be verified.

## Related Work & Insights
- **vs PatchCleanser**: The latter provides certified guarantees but requires a two-stage masking and classifier-agnostic design, incurring greater computational overhead. STRAP-ViT offers no certified guarantees but is lighter, more practical, and achieves higher empirical robust accuracy.
- **vs Jedi**: Jedi localizes patches through high local entropy and repairs them using an autoencoder, operating in the pixel space. STRAP-ViT operates in the token feature space, keeping it more native to the ViT architecture.
- **vs Adversarial Training**: Adversarial training requires retraining for specific threat models, which is costly and non-transferable. STRAP-ViT requires absolutely no training and can be deployed directly.
- **vs DefensiveDR**: DefensiveDR projects images into a low-dimensional space to suppress local perturbations but discards task-relevant structural information; STRAP-ViT only transforms anomalous tokens, preserving benign information.
- **Transferable to multimodal VLMs**: Any vision model utilizing token representations can theoretically integrate the STRAP-ViT module, including vision encoders of VLMs like LLaVA.

## Rating
- Novelty: ⭐⭐⭐⭐ Combined design of JSD token-level anomaly detection and randomized composite transformations is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple models, datasets, and attacks.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and mathematical rigor, though some notation is a bit dense.
- Value: ⭐⭐⭐⭐ High practical deployment value—a plug-and-play, training-free security reinforcement solution.
- Overall: ⭐⭐⭐⭐ A simple and effective engineering-oriented defense scheme with direct value for practical deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing](../../ICML2025/others/provably_cost-sensitive_adversarial_defense_via_randomized_smoothing.md)
- [\[CVPR 2025\] Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks](towards_million-scale_adversarial_robustness_evaluation_with_stronger_individual.md)
- [\[ACL 2025\] Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues](../../ACL2025/others/battling_against_tough_resister_strategy_planning_with_adversarial_game_for_non-.md)
- [\[CVPR 2025\] TAET: Two-Stage Adversarial Equalization Training on Long-Tailed Distributions](taet_two-stage_adversarial_equalization_training_on_long-tailed_distributions.md)
- [\[NeurIPS 2025\] Alias-Free ViT: Fractional Shift Invariance via Linear Attention](../../NeurIPS2025/others/alias-free_vit_fractional_shift_invariance_via_linear_attention.md)

</div>

<!-- RELATED:END -->
