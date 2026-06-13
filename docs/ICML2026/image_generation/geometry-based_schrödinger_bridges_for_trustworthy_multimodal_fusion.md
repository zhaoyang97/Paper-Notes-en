---
title: >-
  [Paper Note] Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion
description: >-
  [ICML 2026][Image Generation][Trustworthy Multimodal Fusion] This paper proposes GMF: using Diffusion Schrödinger Bridge + Rectified Flow to estimate the "transport correction cost" (initial velocity square $\|v_\theta(z…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Trustworthy Multimodal Fusion"
  - "Schrödinger Bridge"
  - "Rectified Flow"
  - "Transport Energy"
  - "Evidential Learning"
date: 2026-05-08
content_hash: e67f9ce0159d34c4
---

# Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion

**Conference**: ICML 2026  
**arXiv**: [2605.31193](https://arxiv.org/abs/2605.31193)  
**Code**: N/A  
**Area**: Multimodal VLM / Trustworthy Fusion / Generative Geometry  
**Keywords**: Trustworthy Multimodal Fusion, Schrödinger Bridge, Rectified Flow, Transport Energy, Evidential Learning

## TL;DR
This paper proposes GMF: using Diffusion Schrödinger Bridge + Rectified Flow to estimate the "transport correction cost" (initial velocity square $\|v_\theta(z,0)\|^2$) of each modality in the latent space. This serves as a geometric reliability signal **decoupled from classifier confidence** to dynamically weight multimodal fusion, thereby breaking the "model judging itself" circular dependency. It significantly outperforms confidence-based baseline methods under sensor noise and semantic conflicts.

## Background & Motivation

**Background**: The mainstream approach for Trustworthy Multimodal Fusion is dynamic fusion—processing each modality independently and aggregating predictions weighted by "modality quality." Representative methods like TMC, QMF, PDF, and DBF use classifier outputs (entropy, evidence, Dirichlet concentration) as quality scores.

**Limitations of Prior Work**: Deep networks suffer from severe overconfidence (Guo et al. 2017). In scenarios with high noise, OOD (Out-Of-Distribution) data, or semantic conflicts, a classifier may be "confidently wrong"—meaning the output probability is sharp but the answer is incorrect. Confidence-based methods equate "how confident I am" with "how clean the input is," failing to identify these confident-but-wrong failure modes.

**Key Challenge**: Reliability assessment and the assessed prediction originate from the same model, creating a **circular dependency**—using a prediction to judge its own reliability. When the classifier is deceived, all reliability metrics dependent on its output fail simultaneously.

**Goal**: Construct a reliability signal **independent of the classifier's decision boundary**, ensuring that even if the classifier is deceived by severe noise or conflicting inputs, the fusion mechanism can still correctly identify and down-weight the corrupted modality.

**Key Insight**: The authors redefine "modality quality" as **latent space geometric deviation**—clean samples cluster on the data manifold, while OOD/noise samples deviate from it. How to measure "deviation"? Use Optimal Transport: the "correction work" required to transport a sample to a reference distribution.

**Core Idea**: Use Diffusion Schrödinger Bridge to learn a transport path from latent features to a reference distribution, and use Rectified Flow to linearize this path into a single-step prediction. The **initial velocity square** $\|v_\theta(z,0)\|^2$ serves as an efficient "geometric unreliability score"—clean samples have low transport costs, while noise/conflict samples have high costs. This metric is completely decoupled from classifier logits.

## Method

### Overall Architecture

Input $M$ modalities $\{x^{(m)}\}_{m=1}^M$, where each is processed by an encoder to obtain latent features $z^{(m)} = E^{(m)}(x^{(m)}) \in \mathbb{R}^d$. GMF performs two operations in the latent space:

1.  **Intra-modal geometric evaluation**: Uses a modality-specific rectified flow velocity field $v_\theta^{(m)}$ to estimate the transport energy $\mathcal{E}_{\text{intra}}^{(m)} = \|v_\theta^{(m)}(z^{(m)},0)\|_2^2$ from $z^{(m)}$ to a **class-agnostic reference prior** $\mathcal{P}_{\text{prior}}$, serving as an "intrinsic quality score."

2.  **Inter-modal geometric evaluation**: For each directed pair $(a \to b)$, a cross-modal velocity field $v_\Phi^{(a \to b)}$ is trained to map $z^{(a)}$ onto the manifold of $b$. The residual $\|\Phi_{a \to b}(z^{(a)}) - z^{(b)}\|_2^2$ measures inter-modal semantic consistency.

Subsequently, a "competition-interaction" gating mechanism merges these two geometric costs into fusion weights $w^{(m)}$. These are combined with evidence $\mathbf{e}^{(m)} = \text{Softplus}(z^{(m)} W_{\text{cls}}^{(m)})$ to assemble Dirichlet parameters $\boldsymbol{\alpha} = \sum_m w^{(m)} \mathbf{e}^{(m)} + \mathbf{1}$, which are fed into an evidential classification head. During training, the geometric and decision branches have separated gradient paths to prevent mutual contamination.

### Key Designs

1.  **Transport Energy as Geometric Reliability Score**:
    -   **Function**: Provides a scalar measure of how far $z^{(m)}$ is from the clean manifold, **independent of classifier output**.
    -   **Mechanism**: Formalizes the Schrödinger Bridge as $\min_v \int_0^1 \mathbb{E}\|v_t\|^2 dt$. Since direct solving via iterative integration is slow, Rectified Flow is used to linearize the path—regressing a constant velocity $z_1 - z_0$ on $z_t = $(1-t)z_0 + t z_1$, with the objective $\mathcal{L}_{\text{RF}} = \mathbb{E}_{t, z_0, z_1}\|v_\theta(z_t, t) - (z_1 - z_0)\|^2$. Inference requires only evaluating $v_\theta(z, 0)$ once, using $\|v_\theta(z, 0)\|_2^2$ as the learned "correction score."
    -   **Design Motivation**: (1) Single-step inference with low latency for online deployment; (2) $v_\theta$ is evaluated at $z$ (the source), ensuring inference matches the training distribution; (3) Clean samples require small corrections, while noise/missing samples deviate more, forming a failure detector **orthogonal** to confidence.

2.  **Cross-modal Transport Residuals as Semantic Conflict Gating**:
    -   **Function**: Directly measures whether two modalities are semantically consistent in the latent space without using decoders.
    -   **Mechanism**: Learns a separate $v_\Phi^{(a \to b)}$ for each directed pair $(a \to b)$ using $\mathcal{L}_{\text{inter}}^{(a \to b)} = \mathbb{E}\|v_\Phi^{(a \to b)}(z_t, t) - (z^{(b)} - z^{(a)})\|^2$. During inference, $z^{(a)}$ is projected in one step as $\hat{z}^{(a \to b)} = z^{(a)} + v_\Phi^{(a \to b)}(z^{(a)}, 0)$. A larger residual $\mathcal{E}_{\text{inter}}^{(a \to b)} = \|\hat{z}^{(a \to b)} - z^{(b)}\|_2^2$ indicates lower semantic consistency. The paper theoretically proves (Thm 4.5): if two modalities fall on manifolds of different classes, the residual is lower-bounded by $(\delta - 2\epsilon)^2$ (Geometric Barrier Principle).
    -   **Design Motivation**: Shifts the judgment of semantic consistency from the classifier level (easily fooled by two overconfident outputs) to the latent geometric level, providing an external criterion the classifier cannot forge.

3.  **Competition-Interaction Fusion Weights**:
    -   **Function**: Combines intrinsic quality scores $\mathcal{E}_{\text{intra}}$ and cross-modal consistency $\mathcal{E}_{\text{inter}}$ into final fusion weights $w^{(m)}$, explained by Thm 4.4 as a Gibbs solution for entropy-regularized minimization.
    -   **Mechanism**: First assigns a base competition score via Boltzmann distribution $\beta_{\text{comp}}^{(m)} = \exp(-\mathcal{E}_{\text{intra}}^{(m)}/\tau) / \sum_k \exp(-\mathcal{E}_{\text{intra}}^{(k)}/\tau)$. Then, an interaction gate $\gamma_{\text{int}}^{(m)} = \lambda \sum_{k \neq m} r^{(k)} \exp(-\mathcal{E}_{\text{inter}}^{(k \to m)}/\kappa)$ collects geometric votes from "reliable neighbors," where $r^{(k)} = \sigma(\theta_r - \mathcal{E}_{\text{intra}}^{(k)})$ is a soft reliability gate for the neighbor. After adding $\epsilon_\gamma$ for stability, weights are normalized: $w^{(m)} = \beta_{\text{comp}}^{(m)} \tilde{\gamma}_{\text{int}}^{(m)} / \sum_j \beta_{\text{comp}}^{(j)} \tilde{\gamma}_{\text{int}}^{(j)}$.
    -   **Design Motivation**: The competition term ensures "clean" modalities get higher weights, while the interaction term exponentially suppresses modalities "disowned" by reliable neighbors—mapping to the "exponential inhibition of conflicting modalities" in Corollary 4.6. This dual weighting ensures that **confident but conflicting** modalities are penalized twice, breaking the circular dependency.

### Loss & Training

The total objective is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{geo}} \mathcal{L}_{\text{geo}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$. Where:
- $\mathcal{L}_{\text{geo}} = \sum_m \mathcal{L}_{\text{intra}}^{(m)} + \sum_{a \neq b} \mathcal{L}_{\text{inter}}^{(a \to b)}$ trains all RF velocity fields;
- $\mathcal{L}_{\text{task}}$ is evidential cross-entropy + KL regularization (pulling toward uniform Dirichlet to penalize overconfidence without sufficient evidence);
- $\mathcal{L}_{\text{reg}} = (1 - \rho)^\zeta \cdot \text{KL}(\text{Dir}(\boldsymbol{\alpha}) \| \text{Dir}(\mathbf{1}))$ uses a global consistency coefficient $\rho = \frac{1}{M(M-1)} \sum_{a \neq b} \exp(-\mathcal{E}_{\text{inter}}^{(a \to b)}/\kappa)$ to force the predictive distribution toward uniform when cross-modal divergence is high.

Key training trick: Stop-gradient (`sg`) is applied to $\mathcal{E}_{\text{intra}}$ and $\mathcal{E}_{\text{inter}}$ when calculating fusion weights. This ensures $\mathcal{L}_{\text{task}}$ only updates the encoder and classification head, while $\mathcal{L}_{\text{geo}}$ only updates velocity fields, preventing task gradients from contaminating geometric measurements.

## Key Experimental Results

### Main Results

Evaluated on four benchmarks: NYU Depth V2 (RGB-D), UPMC FOOD-101 (Image-Text), MVSA-Single (Sentiment), and PneumoniaMNIST (X-ray + Reports), comparing against 10 baselines (including TMC, QMF, PDF, DBF, UAW-EEF).

**Robustness to Sensor Noise** (NYU/Food-101, Gaussian noise $\sigma \in \{1.0, 2.0\}$, or 50% modality missing):

| Dataset | Scenario | Concat | QMF | PDF | DBF | UAW-EEF | **GMF** |
|--------|------|--------|-----|-----|-----|---------|---------|
| NYU | Clean | 68.5 | 71.2 | 72.5 | 72.3 | 71.8 | 71.9 |
| NYU | $\sigma=2.0$ | 28.4 | 45.8 | 47.5 | 49.1 | 50.2 | **55.2** |
| NYU | Incomplete | 35.8 | 56.4 | 58.2 | 60.3 | 61.5 | **64.8** |
| Food-101 | $\sigma=2.0$ | 30.2 | 48.6 | 51.2 | 52.4 | 53.1 | **58.7** |
| Food-101 | Incomplete | 41.2 | 78.5 | 80.6 | 81.3 | 82.4 | **85.4** |

GMF performs on par with SOTA on clean data and shows **larger leads as noise increases**, validating the hypothesis that geometric signals remain effective in overconfident scenarios.

**Safety under Semantic Conflict** (MVSA-Single, shuffling pairs to create conflicts):

| Method | Rejection Rate ↑ | Avg Entropy ↑ | CDR (AUROC) ↑ |
|------|------------------|---------------|---------------|
| QMF | 18.5% | 0.52 | 56.8 |
| PDF | 21.3% | 0.58 | 60.1 |
| DBF | 35.2% | 0.94 | 71.2 |
| **GMF** | **76.8%** | **1.85** | **89.4** |

The conflict rejection rate is **41.6 pp higher** than the runner-up DBF, with AUROC 18.2 pp higher. This shows that prediction-space evidence methods (QMF/PDF/DBF) struggle to detect conflicts when both inputs are overconfident but contradictory, whereas GMF's cross-modal transport residuals detect them directly.

**Medical Risk Stratification** (PneumoniaMNIST): GMF achieves 91.2% accuracy, with a Pearson correlation $r=0.78$ between reliability and correctness (next best DBF: 0.61), and ECE reduced to 0.068 (next best: 0.095).

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full GMF | $\sigma=2.0$ Acc: 55.2% | Complete model |
| Replace $\mathcal{E}_{\text{intra}}$ with entropy | 36.8% | Drop of 18.4 pp; MI(reliability, confidence) jumps from 0.10 to 0.67—validates that statistical metrics are coupled with confidence. |
| Use cosine sim instead of $\mathcal{E}_{\text{inter}}$ | Conflict detection drops | Shows semantic conflicts involve **nonlinear geometric distortion**, which linear metrics fail to capture. |
| 1-step RF vs multi-step ODE | Almost identical accuracy | Validates the RF linearization hypothesis; single-step velocity estimation suffices. |
| Prior $\mathcal{P}_{\text{prior}}$ choice | Robust performance | Geometric signals come from the learned transport structure, insensitive to prior selection. |

### Key Findings

- **Independence of geometric signals is crucial**: Replacing $\mathcal{E}_{\text{intra}}$ with entropy drops performance to statistical baseline levels and spikes Mutual Information (MI), proving that "breaking circular dependency" requires an external signal.
- **Empirical support for exponential conflict gating**: Fig 2(b) shows $w^{(m)}$ decays exponentially relative to $\mathcal{E}_{\text{inter}}$ per $e^{-\mathcal{E}_{\text{inter}}/\kappa}$. Clean pairs cluster at $\mathcal{E}_{\text{inter}} < 5$, while conflicts cluster at $> 9$, experimentally validating the Geometric Barrier Principle.
- **Efficiency**: Single-step RF maintains latency comparable to PDF/DBF with negligible precision loss, making GMF suitable for safety-critical real-time scenarios.

## Highlights & Insights

- **Precise framing of "Circular Dependency"**: While previous trustworthy fusion works added various tricks, none explicitly addressed the structural weakness of "using predictions to evaluate predictions." By identifying this, a "prediction-free external signal" becomes a self-evident design principle.
- **Generative Models as Geometric Probes**: Schrödinger Bridge / Rectified Flow are typically generative models. In this work, they are used solely as **scalar probes** via the norm of the velocity vector during a single forward pass, without sampling. This paradigm shift—using generative side-effects as discriminative signals—is transferable to OOD, adversarial, or anomaly detection tasks.
- **Theory-grounded Interpretation**: Thm 4.5 uses $(\delta - 2\epsilon)^2$ to define the lower bound of conflict suppression. Combined with Thm 4.4, which proves the fusion weight is the unique solution for entropy-regularized minimization, it provides a clean "geometry → weights → safety" interpretability chain despite strong assumptions.
- **Gradient path separation** is a critical engineering detail; otherwise, task losses would warp the velocity fields, re-coupling transport energy with the classifier and re-introducing circular dependency.

## Limitations & Future Work

- **Theoretical Assumptions**: Thm 4.5 relies on "latent manifold concentration/metric separability" and "local $\xi$-semantic consistency," which may fail if representation learning is unsuccessful or modalities are fundamentally misaligned.
- **Scalability**: The number of cross-modal velocity fields grows quadratically ($O(M^2)$) with $M$ modalities. For systems with $M \geq 4$, parameter and training costs may explode.
- **Class-agnostic Priors**: $\mathcal{P}_{\text{prior}}$ ignores class information, making it uncertain whether it can distinguish intra-class distribution shifts.
- **Safety Margins**: A 76.8% rejection rate on MVSA-Single still leaves 23% of conflicts undetected, requiring further reliability improvements for medical or autonomous driving deployments.
- **Future Directions**: (1) Amortized cross-modal fields to reduce complexity to $O(M)$; (2) Integrating transport energy as an external geometric sanity check for LLM-based fusion; (3) Extending geometric barriers to the temporal dimension for video/time-series fusion.

## Related Work & Insights

- **vs QMF / PDF / TMC** (Statistical Trustworthy Fusion): These derive quality from classifier outputs; GMF derives it from latent geometry. GMF's advantage is breaking the circular dependency, though at the cost of training $M + M(M-1)$ velocity fields.
- **vs DBF / UAW-EEF** (SOTA Conflict Modeling): These model conflicts in belief space or classifier logits. GMF's cross-modal residuals $\mathcal{E}_{\text{inter}}$ perform alignment in latent space, providing an unforgeable conflict signal. The 76.8% vs 35.2% rejection rate gap highlights the advantage of "shifting the battlefield" to geometry.
- **vs Evidential Deep Learning**: EDL uses Dirichlet distributions to express "unknowns," but it remains a form of classifier introspection. GMF retains EDL heads but replaces weight generation with a geometric module, acting as "EDL + External Geometric Prior."

## Rating
- **Novelty**: ⭐⭐⭐⭐ Diagnosing trustworthy fusion failures as "circular dependency" and using RF/SB probes to solve it is a refreshingly systematic perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 benchmarks, 10 baselines, and 3 stress tests. The quantification of decoupling (MI 0.10 vs 0.67) is highly persuasive, though scalability beyond $M=2$ is unproven.
- **Writing Quality**: ⭐⭐⭐⭐ Excellent narrative on motivation. Theoretical sections are well-integrated, though notation density requires careful reading.
- **Value**: ⭐⭐⭐⭐ High demand for trustworthy fusion in robotics and healthcare. The paradigm of using generative side-effects as reliability signals is highly versatile.

## Related Papers

- [\[ICLR 2026\] Contact Wasserstein Geodesics for Non-Conservative Schrödinger Bridges](../../ICLR2026/image_generation/contact_wasserstein_geodesics_for_non-conservative_schrödinger_bridges.md)
- [\[ICLR 2026\] Branched Schrödinger Bridge Matching](../../ICLR2026/image_generation/branched_schrödinger_bridge_matching.md)
- [\[CVPR 2026\] CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion](../../CVPR2026/image_generation/careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)
- [\[ICML 2026\] Geometry-Aware Tabular Diffusion](geometry-aware_tabular_diffusion.md)
- [\[ICML 2026\] Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](discrete_diffusion_samplers_and_bridges_off-policy_algorithms_and_applications_i.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges](../../NeurIPS2025/image_generation/grasp2grasp_vision-based_dexterous_grasp_translation_via_schrödinger_bridges.md)
- [\[ICLR 2026\] Branched Schrödinger Bridge Matching](../../ICLR2026/image_generation/branched_schrödinger_bridge_matching.md)
- [\[CVPR 2026\] CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion](../../CVPR2026/image_generation/careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)
- [\[NeurIPS 2025\] Dynamic Diffusion Schrödinger Bridge in Astrophysical Observational Inversions](../../NeurIPS2025/image_generation/dynamic_diffusion_schrödinger_bridge_in_astrophysical_observational_inversions.md)
- [\[ICML 2026\] Geometry-Aware Tabular Diffusion](geometry-aware_tabular_diffusion.md)

</div>

<!-- RELATED:END -->
