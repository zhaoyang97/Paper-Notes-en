---
title: >-
  [Paper Note] Closing the Loop: PID Feedback Control for Interpretable Activation Steering in Symbolic Music Generation
description: >-
  [ICML 2026][Interpretability][Activation Steering] This paper introduces PID feedback control from control theory into Sparse Autoencoder (SAE)-based activation steering. By using the integral term to accumulate error, the method overcomes the Top-K sparsity threshold which causes static steering to fail at low intensities during transitions. Temporal PID dynamically adjusts $\lambda(t)$ at each autoregressive step, achieving smooth transitions for pitch and duration in symbo…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Activation Steering"
  - "PID Control"
  - "Sparse Autoencoders"
  - "Controllable Generation"
  - "Symbolic Music"
date: 2026-05-08
content_hash: 9fdc111aa2963174
---

# Closing the Loop: PID Feedback Control for Interpretable Activation Steering in Symbolic Music Generation

**Conference**: ICML 2026  
**arXiv**: [2606.18790](https://arxiv.org/abs/2606.18790)  
**Code**: https://giannisprokopiouorfium.github.io/music-transformer-sae/pid  
**Area**: Interpretability / Activation Steering / Sparse Autoencoders  
**Keywords**: Activation Steering, PID Control, Sparse Autoencoders, Controllable Generation, Symbolic Music

## TL;DR
This paper introduces PID feedback control from control theory into Sparse Autoencoder (SAE)-based activation steering. By using the integral term to accumulate error, the method overcomes the Top-K sparsity threshold which causes static steering to fail at low intensities during transitions. Temporal PID dynamically adjusts $\lambda(t)$ at each autoregressive step, achieving smooth transitions for pitch and duration in symbolic music with 62–67% lower intervention intensity and a 5% reduction in FMD degradation.

## Background & Motivation

**Background**: Activation steering modifies internal model representations during inference to control generation without retraining, based on the Linear Representation Hypothesis—where concepts correspond to linear directions in the activation space. In symbolic music, dense steering (adding the Difference-in-Means vector to the residual stream) suffers from **superposition**: in the Multitrack Music Transformer (MMT), the cosine similarity between pitch and duration vectors is as high as 0.81, leading to interference during multi-attribute steering. **Sparse Activation Steering (SAS)**, which uses per-layer SAEs to project 512-dimensional activations into a 4096-dimensional sparse space, offers an attractive solution for precise, decoupled, and interpretable control.

**Limitations of Prior Work**: SAS is constrained by **Top-K sparsity**, where re-sparsification retains only the largest $K$ features. This creates a **binary threshold problem** absent in dense methods: when attempting to smoothly ramp steering intensity from zero using a cosine curve, the **fractional $\lambda$ ($\lambda < 1$) magnitudes are too small to enter the Top-K**. Consequently, the steering signal is zeroed out by the ReLU/Top-K operation $\sigma$, forcing smooth transitions into abrupt binary jumps.

**Key Challenge**: Dense adaptive methods do not face this issue because attenuated signals persist; however, SAS Top-K acts as an "all-or-nothing" barrier where gentle interventions disappear. Nguyen et al. (2026) have shown that static steering is essentially a **Proportional (P) controller**, which cannot eliminate steady-state errors caused by model priors. To overcome the Top-K threshold, a mechanism capable of "accumulating momentum" is required.

**Goal**: (1) Validate the spatial (layer-wise) PID framework of Nguyen et al. in a shallower architecture (MMT); (2) Crucially, transpose PID from the "layer axis" to the "temporal axis" to resolve the transition failures of SAS.

**Key Insight**: The **integral term** of a PID controller naturally accumulates error. As long as the target feature fails to survive (error remains positive), the integral term pushes $\lambda(t)$ higher until it breaks through the Top-K threshold. This specifically addresses the "accumulation required for threshold jumping" needed by Top-K.

**Core Idea**: Utilizing a Top-K-aware closed-loop controller to measure whether the target feature survives after re-sparsification at each autoregressive step, and dynamically adjusting $\lambda(t)$ so that the integral term can "climb over" the Top-K wall.

## Method

### Overall Architecture

The method follows two tracks. **Spatial PID** serves as domain validation: the layer-wise PID formula from Nguyen et al. (2026) is applied across the 12 sub-layers of MMT. Steering vectors $\mathbf{u}(k) = K_p \mathbf{e}(k) + K_i \sum_{j=0}^{k-1} \mathbf{e}(j) + K_d (\mathbf{e}(k) - \mathbf{e}(k-1))$ are computed sequentially. This confirms that control theory predictions hold even in architectures much shallower than LLMs (requiring a higher $K_i$ due to fewer steps). **Temporal PID** is the primary solution for SAS transition failures. Since SAS intervention typically occurs at a **single layer (Layer 10)**, there is no "layer-to-layer" progression for spatial PID. Instead, the control variable is transposed from the spatial domain (layer index $k$) to the **temporal domain (generation step $t$)**, forming a closed-loop feedback system during autoregressive decoding. The Temporal PID pipeline measures the "concept fingerprint" magnitude, calculates error against a cosine ramp setpoint, determines $\lambda(t)$ via the control law, and injects the signal before Top-K re-sparsification and SAE decoding.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Activations at step t<br/>(MMT Layer 10)"] --> B["1. Concept Fingerprint Error Measurement<br/>Mean magnitude of top-N target features"]
    B --> C["2. Cosine Ramp Setpoint<br/>m*(t) smooth climb"]
    C --> D["3. PID Control Law<br/>Integral accumulation to break Top-K"]
    D -->|λ(t)| E["s(t) = f(a) + λ(t)·v<br/>→ Top-K Re-sparsification → SAE Decode"]
    E -->|Feedback for step t+1| B
```

### Key Designs

**1. Concept Fingerprint Error Signal: Converting "Survival" to Continuous Measurement**

To implement closed-loop control, an error signal is needed to indicate if the steering signal has entered the Top-K. At each step $t$, the mean magnitude of the **top-$N$ target features** (indices $\mathcal{T}$ of $N=32$ largest components in $\mathbf{v}$) is measured: $\bar{f}_a(t) = \frac{1}{|\mathcal{T}|} \sum_{j \in \mathcal{T}} f(\mathbf{a}_t^\ell)_j$. This "concept fingerprint" indicates if the signal survived re-sparsification. The error is $e(t) = m^*(t) - \bar{f}_a(t)$. Using mean magnitude instead of survival count provides a **continuous and proportional** signal, allowing for smooth $\lambda$ adjustments rather than binary bang-bang control.

**2. Cosine Ramp Setpoint: Defining the Target Trajectory**

Setting the target magnitude instantly would recreate sudden jumps. Thus, the setpoint $m^*(t)$ ramps smoothly over $T_{\text{ramp}}$ beats using a cosine curve:

$$m^*(t) = \begin{cases} \frac{m_{\text{target}}}{2} \left(1 - \cos\left(\frac{\pi t}{T_{\text{ramp}}}\right)\right) & t < T_{\text{ramp}} \\ m_{\text{target}} & t \ge T_{\text{ramp}} \end{cases}$$

$T_{\text{ramp}} = 64$ steps was found to be the optimal balance.

**3. PID Control Law with Anti-Windup: Breaking the Top-K Wall**

The controller outputs $\lambda(t)$ to overcome the threshold without overshooting:

$$\lambda(t) = \text{clamp}\Big(K_p e(t) + K_i I(t-1) + K_d (e(t) - e(t-1))\Big)$$

Where $I(t) = \text{clamp}(I(t-1) + e(t), -I_{\text{max}}, I_{\text{max}})$ includes **anti-windup clamping**. During the ramp, $\bar{f}_a \approx 0$ creates persistent positive error, causing the I-term to accumulate and scale up $\lambda(t)$ until the threshold is breached. Post-breach, the D-term suppresses overshoot at the critical "sub-threshold to active" transition.

**4. Dual Concept Orthogonal Control: Preventing Top-K Competition**

When steering pitch and duration simultaneously, features compete for Top-K slots. The solution employs two independent Temporal PID controllers with Gram-Schmidt orthogonalized SAS vectors and increases the budget from $K$ to $2 \times K$ on the steered layer to accommodate both concepts.

## Key Experimental Results

### Main Results

Using MMT (12 sub-layers, 512-d) and per-layer SAEs ($K=128$), Temporal PID ($\lambda_{\text{max}}=3.0$) was compared against static SAS (fixed $\lambda=3.0$) and an unsteered baseline. Quality degradation $\delta$ measures shifts in pitch entropy, scale consistency, and rhythmic consistency; FMD uses CLaMP2 embeddings.

| Concept · Direction | PID | Static SAS | Unsteered Baseline |
| :--- | :--- | :--- | :--- |
| Pitch ↑ (semitones) | 72.65 | 72.30 | 68.79 |
| Pitch ↓ (semitones) | 43.99 | 44.91 | 67.94 |
| Duration ↑ (ticks) | 18.87 | 22.17 | 7.99 |
| Duration ↓ (ticks) | 4.23 | 3.35 | 7.72 |
| Pitch FMD (↓) | **461.9** | 487.7 | 381.5 |
| Duration FMD (↓) | **501.2** | 525.9 | 385.3 |

PID achieves comparable steering strength while keeping Pitch FMD 5.3% lower than static SAS. The dynamic $\lambda(t)$ avoids over-steering early tokens.

### Ablation Study

| Configuration | $\lambda_{\text{avg}}$ | Note |
| :--- | :--- | :--- |
| Pure P | 0.664 | Too conservative; fails to break Top-K threshold |
| P+I | 1.136 | Error accumulation pushes $\lambda$ past threshold — the core mechanism |
| P+I+D (Full) | 1.158 | D-term fine-tunes convergence and suppresses overshoot |

### Key Findings
- **The Integral term is the key to thresholds**: Pure P control results in $\lambda$ stalling at 0.664 (below Top-K); adding I immediately pushes it to 1.13 (above threshold), proving why static P-style steering fails on SAEs.
- **Dual Concept Steering**: PID shows 4.7× lower degradation in unconditional steering ($\delta=0.47$ vs 2.19) and holds a 2.2× advantage in difficult counter-direction scenarios.
- **Reversible Steering**: Unlike static SAS, Temporal PID enables "steer away $\to$ hold $\to$ steer back" with 46–74% recovery rates, outperforming passive release.
- **Prior Resistance**: Pitch requires $K_i$ values twice as high as duration, indicating a stronger autoregressive prior that the integral term must overcome.

## Highlights & Insights
- **Accurate diagnosis of the Top-K binary threshold**: Correctly identifying that fractional $\lambda$ values are zeroed out is a high-value insight unique to SAE-based intervention.
- **Elegant synthesis of Control Theory and SAEs**: The use of the I-term to "build force" to cross a threshold is a powerful analogy applicable to any intervention with a hard sparsity gate.
- **Spatial-to-Temporal Transposition**: Transposing the control loop to the generation step allows PID benefits to be applied to single-layer interventions in shallow models.
- **Concept Fingerprint Design**: Converting discrete survival into a continuous mean magnitude magnitude enables smooth feedback and avoids unstable bang-bang control.

## Limitations & Future Work
- **Single Model/Dataset**: Results are restricted to MMT + SOD; portability of gains and the $K_i$ asymmetry remain to be verified on other architectures.
- **Small Sample Size and Lack of Perceptual Study**: $n=40$ is relatively small, and the lack of listening tests (e.g., MUSHRA) limits conclusions regarding subjective musicality.
- **Duration-Up Degradation**: PID's adaptive trajectory increases degradation in scale consistency for duration-up tasks compared to matched-$\lambda$ baselines.
- **Top-K Budget Issues**: The $2 \times K$ requirement for dual concepts highlights SAS limitations; future work could explore relaxed sparsity SAEs (e.g., RouteSAE).

## Related Work & Insights
- **vs. Static SAS (Bayat 2025)**: SAS provides interpretable, decoupled control, but its fixed $\lambda$ acts as a P-controller that fails Top-K thresholds during transitions. PID resolves this while maintaining interpretability.
- **vs. Spatial PID (Nguyen et al. 2026)**: While they applied PID across $32+$ layers in LLMs, this work replicates it in shallow architectures and transposes it to the temporal domain to solve the unique Top-K barrier.
- **vs. Dense Adaptive Methods**: Methods like IDS or DIRECTER work in dense settings where signals persist even when small; this work targets the unique "threshold-jumping" challenge of sparse representations.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of the "Top-K threshold problem" and the PID integral solution is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers single/dual concepts and reversibility, but lacks human listening tests and larger-scale validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly explains failing mechanisms and the role of each PID term with intuitive visualizations.
- **Value**: ⭐⭐⭐⭐ Offers a clear blueprint for anyone working on SAE-based steering or controllable generation involving hard sparsity constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Activation Steering with a Feedback Controller](../../ICLR2026/interpretability/activation_steering_with_a_feedback_controller.md)
- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)
- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[ICML 2026\] On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders](on_the_relationship_between_activation_outliers_and_feature_death_in_sparse_auto.md)
- [\[ICLR 2026\] PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra](../../ICLR2026/interpretability/persona_dynamic_and_compositional_inference-time_personality_control_via_activat.md)

</div>

<!-- RELATED:END -->
