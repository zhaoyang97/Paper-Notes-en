---
title: >-
  [Paper Note] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation
description: >-
  [ICCV 2025][Image Generation][backdoor defense] NaviT2I identifies an "Early-step Activation Variation" phenomenon induced by backdoor triggers in text-to-image diffusion models…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "backdoor defense"
  - "text-to-image models"
  - "neuron activation variation"
  - "input-level detection"
  - "Stable Diffusion"
date: 2026-05-08
content_hash: 5bc83282e9a001b4
---

# Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation

**Conference**: ICCV 2025
**arXiv**: [2503.06453](https://arxiv.org/abs/2503.06453)  
**Code**: [GitHub](https://github.com/zhaisf/NaviT2I)  
**Area**: AI Security / Backdoor Defense for Text-to-Image Generation
**Keywords**: backdoor defense, text-to-image models, neuron activation variation, input-level detection, Stable Diffusion

## TL;DR
NaviT2I identifies an "Early-step Activation Variation" phenomenon induced by backdoor triggers in text-to-image diffusion models, and proposes an efficient input-level backdoor defense framework that requires only the first diffusion iteration for analysis. The method achieves an average AUROC of 96.3% across 8 mainstream attacks while incurring only 3.8%–16.7% of the computational cost of existing methods.

## Background & Motivation

The open-source ecosystem of text-to-image (T2I) diffusion models (e.g., Stable Diffusion) introduces backdoor threats: adversaries can embed triggers into models that activate malicious behaviors via specific textual inputs at inference time.

**Limitations of Prior Work**:

**Trigger Dominance Assumption Fails**: Existing input-level defense methods (T2IShield, UFID) rely on the Trigger Dominance assumption—i.e., modifying benign features does not affect model predictions. However, in T2I settings, backdoor targets are diverse: they may alter only local image regions (BadT2I), substitute objects (EvilEdit), or change style (RickBKD). In these cases, modifying benign tokens also changes the generated output, invalidating this assumption.

**Prohibitive Computational Overhead**: Conventional methods require multiple complete image generation passes to analyze output differences, incurring 2–5× additional computational cost in T2I scenarios.

**Key Challenge**: There is a need for a generalizable and efficient defense method that neither relies on the Trigger Dominance assumption nor requires a full generation process.

**Key Insight**: Rather than analyzing generated outputs, this work examines the internal neuron activation states of the model. The key finding is that trigger tokens induce abnormally large neuron activation variations within the first few diffusion steps, with the discrepancy being most pronounced at early steps.

## Method

### Overall Architecture

The NaviT2I pipeline:
1. For each non-stopword token in the input prompt, generate a masked version.
2. Execute only one diffusion iteration and compute per-layer neuron activation variations before and after masking.
3. Identify trigger tokens via outlier detection and determine whether the input is malicious.

### Key Designs

1. **Discovery of the Early-step Activation Variation Phenomenon**:

    - Function: Reveals that the influence of backdoor triggers on neuron activations in diffusion models is most significant during early generation steps.
    - Mechanism: Experiments show that masking a trigger token causes significantly larger changes in Neuron Coverage than masking a normal token. Theoretical analysis proves that when $t$ is large (i.e., early iterations), the upper bound on prediction discrepancy under different conditions $c, c'$ is:
    $\|\epsilon(\mathbf{x}_t, t, c) - \epsilon(\mathbf{x}_t, t, c')\|_2 \leq O\left(\frac{1}{\alpha}\exp\left(-\frac{1}{2\sigma_t^2}\right)\right)$
      When $\sigma_t$ is large (early steps), the discrepancy can be substantial; when $\sigma_t$ is small (late steps), the discrepancy decays exponentially.
    - Design Motivation: Trigger signals can be captured from the first diffusion step alone, substantially improving efficiency.

2. **Per-layer Activation Variation Metric**:

    - Function: Precisely quantifies the influence of each token on the activations of individual model layers.
    - Mechanism:
        - Linear layers: $\delta^{(\ell)}(c, c') = \frac{1}{N_\ell d_\ell}\|\mathbf{A}^{(\ell)}(c) - \mathbf{A}^{(\ell)}(c')\|_1$
        - Convolutional layers: spatial average pooling is applied first, followed by $\ell_1$-norm difference computation.
        - Total activation variation: $\delta_\theta(c, c') = \sum_{\ell \in \mathcal{L}_{set}} \delta^{(\ell)}(c, c')$
    - Design Motivation: Layer-wise computation is more precise than coarse-grained Neuron Coverage and captures differential responses across layers to trigger tokens.

3. **Outlier Detection and Scoring Function**:

    - Function: Normalizes the activation variation of each token into a score and detects outliers.
    - Mechanism: A normalized feature value is computed for each token $k$:
    $V_k = \frac{\delta_\theta(c, c_k)}{\mathcal{D}(c, c_k)}$
      where $\mathcal{D}(c, c_k) = \|\mathcal{T}(c) - \mathcal{T}(c_k)\|_2$ is the text embedding distance (normalizing for the effect of semantic difference).

      Scoring function: $\mathcal{S}(c) = \frac{\max(\mathbf{V})}{\text{mean}(\mathbf{V}')}$, where $\mathbf{V}'$ excludes the top 25% of values.
    - Design Motivation: Division by text embedding distance enables fair comparison across tokens of varying semantic importance; the max/mean ratio effectively identifies outliers.

### Loss & Training

NaviT2I is a **training-free** detection method. The threshold is automatically determined by fitting a Gaussian distribution to a small number of clean samples:
$$\mathcal{D}(c) = \mathbb{1}[\mathcal{S}(c) > \mu_\text{clean} + m \cdot \sigma_\text{clean}]$$
where $m = 1.2$ is the default balancing coefficient.

## Key Experimental Results

### Main Results

| Attack Method | NaviT2I AUROC | T2IShield_CDA | UFID | Gain |
|--------|------|------|----------|------|
| RickBKD_TPA | **99.9** | 94.1 | 72.9 | +5.8 |
| RickBKD_TAA | **99.8** | 80.2 | 69.1 | +19.6 |
| BadT2I_Tok | **97.0** | 62.1 | 47.6 | +34.9 |
| BadT2I_Sent | **89.7** | 70.7 | 62.4 | +19.0 |
| VillanBKD_one | **98.9** | 92.6 | 95.7 | +3.2 |
| VillanBKD_mul | **99.9** | 98.0 | 99.9 | +0.0 |
| PersonalBKD | **99.8** | 68.5 | 64.0 | +31.3 |
| EvilEdit | **85.5** | 57.8 | 42.7 | +27.7 |
| **Average** | **96.3** | 78.0 | 69.3 | **+18.3** |

### Ablation Study (Efficiency)

| Method | Diffusion Iterations/Sample | Time (s/sample) | Notes |
|------|---------|------|------|
| T2IShield_FTT | 50 | 7.445 | Requires full generation |
| T2IShield_CDA | 50 | 7.467 | Full generation + covariance analysis |
| UFID | 200 | 33.041 | Requires 4 complete images |
| **NaviT2I** | **≈7** | **1.242** | Only first step × (K+1) passes |

### Key Findings

- NaviT2I achieves 85%+ AUROC across **all 8 attacks**, while baseline methods degrade to near-random performance (~50%) on attacks such as BadT2I and EvilEdit that do not satisfy the Trigger Dominance assumption.
- Detection is completed in only **7%–14%** of the time required for a normal generation pass, making it suitable for real-time deployment.
- The method remains effective against multi-token triggers (VillanBKD_mul) and sentence-level triggers (BadT2I_Sent).
- Style triggers exhibit an ASR of only 28.5% and a FAR as high as 16.3%, indicating they do not constitute effective attacks in the first place.

## Highlights & Insights

1. **Early-step Activation Variation is the central contribution**: Both theoretical and empirical analyses demonstrate that trigger influence is concentrated in the early generation phase—a finding with strong generalizability.
2. **Elegant normalization design**: Division by text embedding distance effectively eliminates interference from differences in semantic importance across tokens.
3. **Highly practical**: Training-free, requires no additional data, operates with a single diffusion step, and is plug-and-play.
4. **Effective on DiT architectures**: Not limited to UNet-based models, demonstrating architectural generality.

## Limitations & Future Work

- Detection performance on EvilEdit is relatively weaker (85.5% AUROC) due to the more covert nature of this attack.
- The method assumes triggers take the common form of textual tokens in current T2I attacks; purely visual triggers are not discussed.
- Threshold calibration relies on Gaussian fitting over a small number of clean samples and may be affected by distributional shift.
- In the worst case (long prompts with 77 non-stopword tokens), the computational overhead approaches that of a single full generation pass.

## Related Work & Insights

- The concept of Neuron Coverage from software testing is ingeniously adapted for T2I security.
- The early-step observation may offer insights for interpretability research on diffusion models.
- The activation variation metric can be generalized to security detection in other conditional generative models (e.g., video generation).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery and theoretical analysis of the Early-step Activation Variation phenomenon are original and profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 8 attacks, multiple datasets, efficiency analysis, adaptive attack analysis, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is clearly articulated, and the method is explained intuitively with supporting figures.
- Value: ⭐⭐⭐⭐⭐ Directly addresses security vulnerabilities in T2I deployment with a concise and efficient approach of high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Disrupting Model Merging: A Parameter-Level Defense Without Sacrificing Accuracy](disrupting_model_merging_a_parameter-level_defense_without_sacrificing_accuracy.md)
- [\[ICCV 2025\] PolarAnything: Diffusion-based Polarimetric Image Synthesis](polaranything_diffusion-based_polarimetric_image_synthesis.md)
- [\[ICCV 2025\] Your Text Encoder Can Be An Object-Level Watermarking Controller](your_text_encoder_can_be_an_object-level_watermarking_controller.md)
- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)
- [\[ICCV 2025\] Penalizing Boundary Activation for Object Completeness in Diffusion Models](penalizing_boundary_activation_for_object_completeness_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
