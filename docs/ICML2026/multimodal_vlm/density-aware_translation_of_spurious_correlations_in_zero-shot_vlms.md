---
title: >-
  [Paper Note] Density-Aware Translation of Spurious Correlations in Zero-Shot VLMs
description: >-
  [ICML 2026][Multimodal VLM][Zero-shot classification] The authors observe that CLIP embeddings exhibit an anisotropic ellipsoidal distribution on the unit sphere…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Zero-shot classification"
  - "spurious correlations"
  - "CLIP anisotropy"
  - "local density"
  - "group robustness"
date: 2026-05-08
content_hash: d08f94ad313c4fe6
---

# Density-Aware Translation of Spurious Correlations in Zero-Shot VLMs

**Conference**: ICML 2026  
**arXiv**: [2606.01710](https://arxiv.org/abs/2606.01710)  
**Code**: https://github.com/AfsanehEB/DAT  
**Area**: Multimodal VLM  
**Keywords**: Zero-shot classification, spurious correlations, CLIP anisotropy, local density, group robustness  

## TL;DR
The authors observe that CLIP embeddings exhibit an anisotropic ellipsoidal distribution on the unit sphere, where spurious samples cluster near the mean. They propose DAT: estimating a local density $D_{y,a}(z)$ using a reference set for each (class, spurious attribute) group and rescaling the original cosine similarity as $\tilde s_{y,a}(x)=s_{y,a}(x)/(D_{y,a}(z)+\varepsilon)^{\lambda}$ based on whether a sample resides at the group core. This significantly improves worst-group accuracy without fine-tuning, text-side modifications, or requiring test-time spurious labels.

## Background & Motivation

**Background**: Zero-shot classification with VLMs like CLIP/ALIGN has become a multimodal baseline, yet they remain highly sensitive to spurious correlations (predicting based on common but semantically irrelevant contextual clues). A classic example is the Waterbirds dataset, where "waterbird + water background" is a frequent combination, causing the model to use "water" as a criterion and fail on "waterbird + land background." Existing mitigation methods fall into three categories: (i) fine-tuning/adapters (requires labels, breaks zero-shot nature), (ii) text-side prompt editing or projection (depends on domain experts or LLMs, prone to cross-modal alignment drift), and (iii) multimodal embedding adjustments (e.g., TIE shifts image embeddings along text directions, but requires training data to calibrate scale).

**Limitations of Prior Work**: Existing methods either sacrifice the zero-shot nature (i), rely on unstable prompt engineering or LLM inference (ii), or require dataset-dependent calibration (iii). More importantly, none directly address the geometric root cause of why CLIP is deceived by spurious correlations.

**Key Challenge**: CLIP embeddings are not isotropically distributed on the unit sphere. Works such as Levi & Gilboa (2025) show that frequent concepts converge toward the modality mean with higher conformity, while rare but semantically critical concepts are pushed to the sparse periphery. Consequently, when using pure cosine similarity, a "correct class but rare" sample may score lower than an "incorrect class but frequent" sample—the score itself is contaminated by geometric bias.

**Goal**: (i) Provide a correction for similarity scores that perceives "local geometric density" under strict zero-shot constraints (frozen encoder, no parameter tuning, no test-time spurious labels); (ii) provide a theoretical explanation for why this aligns with Bayes optimal rules.

**Key Insight**: Instead of modifying the model or the text, the "scoring function itself" should be modified. If the embedding space is ellipsoidal, similarity should be adjusted according to how "typical" a sample is within its group—preserving scores for typical samples while suppressing scores for sparse outliers.

**Core Idea**: Estimate local density using a compact reference set for each group and divide each group's cosine similarity by $(D_{y,a}(z)+\varepsilon)^\lambda$. This is equivalent to subtracting $\lambda \log D$ in logit space, which effectively recovers the quadratic terms missed by cosine similarity in the log-likelihood of a Kent anisotropic distribution.

## Method

### Overall Architecture
The DAT pipeline is built entirely on top of frozen VLMs: first, a compact reference set $R_{y,a}$ is constructed for each $(y,a)$ group using training/validation data; at inference, the local density $D_{y,a}(z)$ of a test image $z=\phi_I(x)$ relative to each group's reference set is calculated. The original cosine similarity is rescaled by this density and aggregated for the final prediction. When the spurious attribute $a$ is unavailable, DAT$^*$ first infers $\hat a=\arg\max_a \langle \phi_I(x), \phi_T(t_a)\rangle$, keeping the rest of the pipeline unchanged.

### Key Designs

1.  **Herding-based Group Reference Set Construction**:
    - **Function**: Select $n$ exemplars for each group $(y,a)$ that represent the group's central geometry, forming the basis for local density estimation.
    - **Mechanism**: Starting from the sample pool $\{x_{y,a}^{(h)}\}_{h=1}^{N_{y,a}}$ of each group, deterministic feature-space herding in the style of iCaRL (Rebuffi et al., 2017) is applied: embeddings are greedily selected such that the mean of the selected set continuously approaches the group mean. The resulting $R_{y,a}=\{z_{y,a}^{(h)}\}_{h=1}^{n}$ covers the "typical appearance" of the group and is far more stable than random sampling. $n$ is very small: 56 for Waterbirds, 128 for CelebA, 40 for COVID-19, and 50 for FMoW.
    - **Design Motivation**: Since frequent/spurious samples naturally reside near the group mean (Levi & Gilboa, 2025), herding naturally captures them to provide a reference for "common pattern clusters." Meanwhile, this set is only used for non-parametric geometric estimation without modifying model parameters, preserving the zero-shot property.

2.  **SLOF Local Density and Density Translation Rescaling**:
    - **Function**: Quantify how sparse a test sample is relative to a group center and use this scalar to compress overconfident similarities in sparse regions.
    - **Mechanism**: Simplified LOF (SLOF, Schubert et al., 2014) is used as the default density proxy: $D_{y,a}(z)=\frac{1}{k}\sum_{z_o\in \text{NN}_k(z)} \frac{k\text{-dist}(z)}{k\text{-dist}(z_o)}$, where larger $D_{y,a}(z)$ indicates higher isolation for $z$. The similarity after density translation is defined as $\tilde s_{y,a}(x)=\frac{s_{y,a}(x)}{(D_{y,a}(z)+\varepsilon)^\lambda}$, where $\lambda>0$ controls the correction strength. $k$ is set to 10 (30 for FMoW), and $\lambda$ is set once per dataset (e.g., 10 for Waterbirds, 1 for CelebA).
    - **Design Motivation**: In pure cosine evaluation, a "misaligned but frequent" sample (spurious) and a "correctly aligned but rare" sample may yield similar scores. SLOF distinguishes these cases: spurious samples are usually in the dense region of their own group but in the sparse periphery of misaligned groups. Thus, after dividing by $D$, scores in misaligned directions are significantly suppressed, while scores in the correct direction are relatively elevated. The paper visualizes this using Tangent-space Mahalanobis Distance on Waterbirds.

3.  **Aggregation + Theoretical Alignment under Kent Distribution**:
    - **Function**: Consolidate all group scores $\tilde s_{y,a}$ into a single class prediction and prove that DAT is equivalent to recovering the anisotropy term missed by cosine similarity.
    - **Mechanism**: Besides group-specific scores, a class-marginal score $\tilde s_{y,\text{Avg}}(x)=\frac{1}{M+1}(\sum_a \tilde s_{y,a}(x)+s_y(x))$ is defined. The final prediction is $\hat y=\arg\max_y \max\{\max_a \tilde s_{y,a}(x), \tilde s_{y,\text{Avg}}(x)\}$. This max-of-max approach balances group-level scoring with fallback to the average when spurious attributes are unknown. Theoretically, group density is modeled with a Kent (Fisher-Bingham) distribution, whose log-density is $\log p(z)=\kappa\gamma_1^\top z + \beta[(\gamma_2^\top z)^2-(\gamma_3^\top z)^2]-\log c_d(\kappa,\beta)$. Pure cosine similarity only captures the linear terminal $\kappa\gamma_1^\top z$, missing the quadratic anisotropy term. The paper proves that the DAT margin $m_{y,a}(z)=\tau w_{y,a}^\top z + \alpha\lambda \log p_{y,a}(z)+r_{y,a}(z)$ ($|r|\le B_0$) aligns $\arg\max$ with Bayes optimality under equal priors.
    - **Design Motivation**: Simply introducing SLOF might be viewed as a heuristic trick; by interpreting $-\log D$ as a log-density proxy (Assumption 3.2), DAT is elevated to "approximating Bayes optimal ranking under ellipsoidal embeddings," answering why cosine fails: it simply cannot "see" the $\beta$ term.

### Loss & Training
The entire process is zero-shot, featuring no training steps or modifications to VLM parameters. The only "parameters" are the reference set size $n$, neighborhood size $k$, and scaling $\lambda$, all of which are set once per dataset.

## Key Experimental Results

### Main Results
Evaluation on four spurious correlation benchmarks (Waterbirds, CelebA, COVID-19, FMoW) across multiple VLMs (CLIP ViT-B/32, ViT-L/14, ResNet-50, ALIGN, AltCLIP, BiomedCLIP). Metrics: worst-group accuracy (WG), average accuracy (Avg), and Gap = Avg − WG. Selection of results on Waterbirds:

| Backbone | Method | WG↑ | Avg↑ | Gap↓ |
|----------|------|-----|------|------|
| ViT-B/32 | Zero-shot (ZS) | 41.37 | 68.48 | 27.11 |
| ViT-B/32 | Orth-Cali | 54.99 | 69.19 | 14.20 |
| ViT-B/32 | TIE | 71.35 | 79.82 | 8.47 |
| ViT-B/32 | **DAT** | **75.08** | **80.36** | **5.28** |
| ViT-L/14 | ZS | 31.93 | 83.72 | 51.79 |
| ViT-L/14 | TIE | 78.82 | 84.12 | 5.30 |
| ViT-L/14 | **DAT** | **83.33** | **89.57** | **6.42** |
| ResNet-50 | ZS | 35.36 | 80.64 | 45.28 |
| ResNet-50 | TIE | 52.96 | 83.62 | 30.66 |
| ResNet-50 | **DAT** | **75.08** | 83.83 | **8.75** |

On CelebA, DAT achieves WG=84.94 using ViT-L/14 (vs. TIE 84.60) and WG=80.79 on ResNet-50 (5.47 higher than TIE's 75.32).

### Ablation Study
The paper systematically ablates DAT* vs. DAT, the key hyperparameter $\lambda$, $k$, and reference set sources (train vs. valid):

| Configuration | Key Observation | Interpretation |
|------|---------|------|
| DAT* (no spurious labels) | WG=79.75 on Waterbirds ViT-L/14, still exceeding TIE | Group assignment inferred via $\hat a$ is sufficient |
| $\lambda$ too small (≈0) | Degenerates to ZS, spurious correlations return | Density correction is necessary for effectiveness |
| $\lambda$ too large | Over-suppresses sparse classes | $\lambda$ controls bias-variance; needs per-dataset tuning |
| Different density estimators | SLOF is most stable | Defaulting to SLOF due to simplicity and robustness |
| CelebA using validation set | Better than training set | Training set itself has higher distribution skew |

### Key Findings
- DAT consistently improves worst-group accuracy across all datasets and backbones, most significantly on backbones like ResNet-50 where embeddings are "flatter" (WG +39.72 relative to ZS on Waterbirds).
- Both Avg and WG accuracy improve simultaneously—many debiasing methods sacrifice Avg for WG. DAT avoids this trade-off through geometric rescaling.
- DAT is more efficient than TIE: it does not require training data for scale calibration, and reference sets of 50–128 samples are sufficient.

## Highlights & Insights
- **Diagnostic → Correction Loop**: The paper first visualizes "geometric misalignment caused by spurious correlations" using Tangent-space Mahalanobis Distance, then provides a symmetric correction signal using SLOF. Analysis and method correspond strictly.
- **Upgrading Cosine to Log-likelihood Proxy**: $\tilde s = s/D^\lambda$ becomes subtraction in the logit domain. Combined with the Kent model, this proves to recover the anisotropy term missed by cosine, providing a general template for "geometry-aware similarity correction" applicable to any anisotropic embedding space (e.g., sentence vectors, recommendation embeddings).
- **Strict Zero-Shot Constraints**: No spurious labels required at test time, no LLM required, and no modification of model parameters, making it friendly for practical deployment. Reference sets are used for geometric estimation rather than gradient updates, adhering to "frozen embedding" evaluation semantics.

## Limitations & Future Work
- DAT requires a small but representative reference set for each group; if a group is extremely rare (real-world long-tail), herding may fail.
- $\lambda$ and $k$ remain dataset-level hyperparameters. The paper tunes them per dataset without providing a "zero-prior" automatic setting strategy; a small sweep is still needed for unseen domains.
- Theoretical results are based on the Kent distribution assumption and log-SLOF fidelity. Large-scale verification of whether general VLM embeddings strictly follow this distribution is not performed; Bayes alignment may only hold locally if the distribution deviates significantly.

## Related Work & Insights
- **vs. TIE / TIE\* (Lu et al., 2025)**: TIE shifts image embeddings along text directions; this work keeps embeddings fixed and modifies scoring, offering a Bayes alignment explanation. DAT outperforms TIE on most backbones, especially ResNet-50.
- **vs. Orth-Cali / Ideal Words / Perception CLIP**: These methods use projection or prompt expansion on the text side, relying on linguistic priors. DAT performs geometric correction on the image side and is orthogonal to text-side methods.
- **vs. ROBOSHOT**: ROBOSHOT uses LLMs to extract spurious directions for linear projection; DAT does not rely on LLMs, offering better deployment stability.
- **vs. Fine-tuning approaches (Zhang & Ré, 2022)**: Fine-tuning is stronger but breaks the zero-shot nature; DAT is suitable for "weights fixed" deployment (APIs, medical compliance).

## Rating
- Novelty: ⭐⭐⭐⭐ "Using local density to correct cosine scores" is a concise idea, and the theoretical alignment with Kent distribution is a prominent contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets × 6 VLM variants, covering natural images, faces, medical, and remote sensing.
- Writing Quality: ⭐⭐⭐⭐ Tight integration from geometric motivation to theory and experiment; consistent notation.
- Value: ⭐⭐⭐⭐ Zero-shot, training-free, and deployment-friendly; the community can instantly apply this to any frozen VLM.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlowComposer: Composable Flows for Compositional Zero-Shot Learning](../../CVPR2026/multimodal_vlm/flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](../../AAAI2026/multimodal_vlm/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[CVPR 2026\] Beyond Heuristic Prompting: A Concept-Guided Bayesian Framework for Zero-Shot Image Recognition](../../CVPR2026/multimodal_vlm/beyond_heuristic_prompting_a_concept-guided_bayesian_framework_for_zero-shot_ima.md)
- [\[NeurIPS 2025\] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models](../../NeurIPS2025/multimodal_vlm/test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)
- [\[CVPR 2026\] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](../../CVPR2026/multimodal_vlm/anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)

</div>

<!-- RELATED:END -->
