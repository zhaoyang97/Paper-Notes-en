---
title: >-
  [Paper Note] Segment Anything with Robust Uncertainty-Accuracy Correlation
description: >-
  [ICML 2026][Segmentation][SAM2] To address the issue that the SAM series only outputs a single mask-level confidence and suffers from "Mask-level Confidence Confusion" under domain shift…
tags:
  - "ICML 2026"
  - "Segmentation"
  - "SAM2"
  - "Mask Confidence Confusion"
  - "Bayesian decoder"
  - "Adversarial Calibration"
  - "Domain Generalization"
date: 2026-05-08
content_hash: 683a716a7adf868f
---

# Segment Anything with Robust Uncertainty-Accuracy Correlation

**Conference**: ICML 2026  
**arXiv**: [2605.10603](https://arxiv.org/abs/2605.10603)  
**Code**: https://github.com/HongyouZhou/ruac.git  
**Area**: Segmentation / SAM / Uncertainty Estimation / Robust Training  
**Keywords**: SAM2, Mask Confidence Confusion, Bayesian decoder, Adversarial Calibration, Domain Generalization

## TL;DR
To address the issue that the SAM series only outputs a single mask-level confidence and suffers from "Mask-level Confidence Confusion" under domain shift, this work equips SAM2 with a Weibull dual-granularity Bayesian mask decoder for pixel-level epistemic estimation. Inspired by human vision, a style + deformation collaborative adversarial perturbation and calibration loss are introduced, ensuring that uncertainty remains aligned with error across 23 zero-shot target domains. The average J&F reaches 79.87, and the uncertainty maps become significantly more reliable.

## Background & Motivation

**Background**: The SAM series has advanced promptable segmentation into the foundation model era, with strong zero-shot performance. However, it still fails in domains such as medical, microscopic, and scientific images. Researchers either perform domain-specific fine-tuning (e.g., Medical SAM) or task-specific adaptation (e.g., Video SAM2, Concept SAM3).

**Limitations of Prior Work**: The IoU score output by SAM is mask-level—an entire mask shares a single confidence, and the confidence gap between foreground and background is small. When domain shift causes some pixels within the masked region to be incorrect, the model cannot indicate which pixels are unreliable. The authors term this failure mode Mask-level Confidence Confusion (MCC). Simply attaching a Bayesian decoder introduces a new problem: the uncertainty-accuracy correlation learned on the source domain degrades out-of-domain (Uncertainty-Accuracy shift, UA shift).

**Key Challenge**: The goal is to maintain the generality of "Segment Anything" (without labeled fine-tuning for each target domain) while ensuring that uncertainty can always identify incorrect pixels under OOD. Achieving both requires "actively simulating OOD during source domain training."

**Goal**: (1) Address MCC by providing pixel-level, dual-granularity uncertainty; (2) Address UA shift so that uncertainty aligns with error across 23 target domains; (3) Adhere to single-domain generalization (SDG) without introducing extra target domain labels.

**Key Insight**: Drawing from cognitive science—humans rely on shape bias for recognition, while neural networks rely more on texture bias (Geirhos et al.). Thus, OOD variation is decomposed into two orthogonal subproblems: appearance (style/texture) changes and non-rigid deformation (shape) changes, each handled by a dedicated adversarial attacker to stress-test the model.

**Core Idea**: Use style + deformation dual attackers to collaboratively generate the most stressful training samples, and employ a calibration loss that penalizes both "certain & wrong" and "uncertain & correct" cases, forcing uncertainty to cover true errors even under adversarial perturbation.

## Method

### Overall Architecture
RUAC replaces the deterministic mask decoder of SAM2 with a Bayesian Mask Decoder (UE) and attaches two attackers: Style Adversarial Network $\psi_s$ and Deformation Adversarial Network $\psi_d$. These are trained end-to-end with the segmentation model via a Gradient Reversal Layer (GRL) in a min-max fashion. Each iteration includes both clean and adversarial forward passes: the clean path preserves in-domain performance, while the adversarial path pushes the model to the edge of calibration failure and then retrains it. During inference, all attackers are discarded, and only UE is used, so deployment cost is just a lightweight Bayesian head added to SAM2.

### Key Designs

1. **Bayesian Mask Decoder (Dual-Granularity Weibull Posterior)**:

    - **Function**: Replaces the original SAM2 decoder with a Weibull distribution to model the uncertainty of both image tokens $\mathbf{f}\in\mathbb{R}^{H\times W\times C}$ and mask tokens $\mathbf{m}_k\in\mathbb{R}^C$, outputting a pixel-level uncertainty map.
    - **Mechanism**: A conv head predicts spatially varying $(\lambda_f,\kappa_f)$, and a shared MLP predicts per-channel $(\lambda_{m,c},\kappa_{m,c})$. Sampling is done via reparameterization $w_i = \lambda_i \cdot (-\ln(1-u))^{1/\kappa_i}$. The two reparameterized features are combined via inner product to produce logits, thus propagating weight uncertainty in closed form to mask probabilities. Inference can use analytic mode (with $\mathbb{E}[w_i]=\lambda_i\Gamma(1+1/\kappa_i)$ and MacKay's probit approximation for per-pixel Bernoulli entropy) or Monte Carlo mode.
    - **Design Motivation**: Weibull is non-negative and flexible in shape, making it more suitable than Gaussian for modeling token intensities (Appendix A.2 also compares Dirichlet, etc.). Dual granularity (image token + mask token) covers both local boundary and global semantic uncertainty, directly addressing "boundary blur" and "object misidentification" failures.

2. **Collaborative Style + Deformation Adversarial Attacks**:

    - **Function**: Online generation of "hard samples" that simultaneously perturb texture (color/material) and shape (geometric deformation), simulating OOD.
    - **Mechanism**: The style attacker extracts per-object RGB mean/variance $(\boldsymbol\mu_k,\boldsymbol\sigma_k)$ from mask regions, uses a GCN on the object graph to predict residuals $(\Delta\boldsymbol\mu_k,\Delta\boldsymbol\sigma_k)$, and applies AdaIN to replace style statistics, producing a stylized image. The deformation attacker, after adding mask embedding to backbone features, predicts a per-pixel offset field $\boldsymbol\delta_k$, then uses differentiable grid sampling to warp both the image and GT mask, maintaining supervision consistency. Both attackers share backbone features and require only one backbone forward pass. GRL reverses the sign during backpropagation, so attackers automatically update towards "harder" directions, eliminating the need for PGD-style inner loops.
    - **Design Motivation**: Traditional $\ell_p$-bounded adversarial attacks only seek worst-case error and lack semantic meaning; the style/deformation decomposition directly corresponds to texture bias and shape bias, two robustness axes validated by biological vision literature. GRL replaces PGD, enabling min-max optimization in a single backward pass, greatly improving training efficiency.

3. **Uncertainty-Accuracy Alignment Calibration Loss**:

    - **Function**: Ensures that uncertainty always covers true pixel errors on adversarial samples.
    - **Mechanism**: Defines calibration loss $\mathcal{L}_{\text{cal}} = e\cdot\exp(-\text{sg}[u]) + (1-e)\cdot\exp(\text{sg}[u])$, where $e=|\hat{\mathbf{M}}-\mathbf{M}^*|$ is per-pixel error, $u$ is analytic uncertainty, and $\text{sg}[\cdot]$ is stop-gradient. The first term penalizes "confident but wrong," the second penalizes "uncertain but correct." This loss does not directly update the segmentation backbone (cut by stop-gradient), but is backpropagated to update the attacker via GRL. The backbone is implicitly pulled towards well-calibrated regions via segmentation + KL on the adversarial branch.
    - **Design Motivation**: Using calibration loss as direct supervision for the backbone would cause the model to sacrifice accuracy for apparent calibration; letting the attacker maximize miscalibration and the backbone resist via seg + KL, this min-max design ensures calibration arises from data generation rather than explicit regularization.

### Loss & Training
Main optimization objective: $\min_{\theta_{\text{dec}}}\mathcal{L}_\theta = (\mathcal{L}_{\text{seg}}+\beta\mathcal{L}_{\text{KL}}) + \gamma(\mathcal{L}_{\text{seg}}^{\text{adv}}+\beta\mathcal{L}_{\text{KL}}^{\text{adv}})$, where $\mathcal{L}_{\text{seg}}=\mathcal{L}_{\text{focal}}+\mathcal{L}_{\text{dice}}+\mathcal{L}_{\text{IoU}}$. The attacker implicitly maximizes $\mathcal{L}_{\text{seg}}^{\text{adv}}+\beta\mathcal{L}_{\text{KL}}^{\text{adv}}+\lambda\mathcal{L}_{\text{cal}}$. $\gamma$ is gradually increased via curriculum to avoid early-stage collapse due to adversarial noise. Training on the source uses only single-frame MOSE dataset.

## Key Experimental Results

### Main Results
Average J&F across 23 zero-shot target domains (selected columns):

| Method | Avg. J&F | TrashCan | LVIS | Cityscapes | Hypersim | IBD | EgoHOS |
|--------|----------|----------|------|------------|----------|-----|--------|
| SAM2 | 67.75 | 44.9 | 75.2 | 64.2 | 46.7 | 80.9 | 84.0 |
| SAM2-FT | 79.75 | 72.4 | 75.9 | 65.1 | 54.6 | 88.9 | 86.3 |
| SAM2-FT-LoRA | 79.13 | 71.3 | 75.6 | 61.6 | 54.6 | 88.9 | 83.6 |
| Bayes-SAM2 (UE only) | 79.87 | 74.9 | 75.1 | 55.4 | 57.5 | 90.3 | 90.4 |
| **RUAC (Full)** | **80.81+** | **74.4+** | 74.8 | **64.2** | **61.8** | **90.2** | **91.3** |

(The last row uses Random Noise as the baseline with Bayesian decoder; full RUAC results in paper Tab. 1)

### Ablation Study

| Configuration | Avg. J&F | Notes |
|---------------|----------|-------|
| SAM2 (no UE) | 67.75 | Pure baseline |
| Bayes-SAM2 (UE only) | 79.87 | Bayesian decoder added |
| Bayes-SAM2 + Random Noise Aug | 80.81 | Standard augmentation |
| Bayes-SAM2 + PGD | 87.5 (partial domains)* | $\ell_\infty$ adversarial |
| **RUAC (UE + Style + Deformation + Cal)** | Paper Tab.1 best | Full method |

UR-ERN and other pure uncertainty baselines average 73.40, significantly lower than Bayes-SAM2's 79.87, indicating insufficient adaptation capability for foundation models.

### Key Findings
- UE alone raises average J&F from 67.75 to 79.87, indicating that mask-level confidence confusion is severely underestimated under domain shift; moving confidence from mask-level to pixel-level solves a substantial part of the problem.
- Adding AUE (style + deformation collaborative attack) yields the largest gains in domains with both geometric and material differences, such as Hypersim (57.5 → 61.8) and Cityscapes (55.4 → 64.2), validating the "texture bias + shape bias" hypothesis.
- Compared to PGD's pure worst-case adversarial, the style/deformation approach achieves better calibration curves while maintaining high accuracy, proving that "adversarial objective should be miscalibration, not max-loss."
- Training uses only MOSE as the source domain, yet generalizes to 23 diverse domains (natural objects, street scenes, microscopy, egocentric), indicating that calibration and robustness induced by bio-inspired attacks provide a task-agnostic inductive bias.

## Highlights & Insights
- Both decompositions—mask-level to pixel-level confidence, and OOD into texture/shape—are clear and directly address specific failure modes; the method narrative follows these decompositions coherently.
- Replacing PGD inner loops with GRL for adversarial training enables single backward passes, which is key for scaling to foundation model size—multi-step PGD is too costly for models like SAM2.
- The calibration loss uses stop-gradient to "penalize the attacker without directly training segmentation," avoiding the "calibrated but bad" trap. This design pattern is instructive for other "main task + auxiliary calibration" research.
- The PAC-Bayes perspective connects the method to "loss landscape flattening + uncertainty-risk coupling," providing a theoretical anchor for empirical adversarial calibration.

## Limitations & Future Work
- The AUE attack model itself requires training and depends on GCN collaboration on the object graph, making it less directly applicable to single-object or no-mask-prompt scenarios.
- Training source uses only the single MOSE domain; while SDG convenience is emphasized, it remains untested whether gains persist when source and target differ greatly (e.g., entirely different medical modalities).
- The Weibull assumption is non-negative and flexible in shape but still unimodal; for truly multimodal ambiguity (multiple plausible masks), it collapses to a mean estimate.
- Inference defaults to analytic mode, adding a lightweight head compared to SAM2; the paper does not systematically compare the extra benefits and costs of MC mode for fine-grained tasks.

## Related Work & Insights
- **vs Bayes-SAM2 / BNDL**: This work inherits the Weibull posterior, but extends the "train + eval" setting from source domain to adversarially calibrated OOD.
- **vs AdvStyle / DG-Font**: The style attacker is adapted from AdvStyle, and the deformation attacker from DG-Font; their goal is worst-case domain generalization, while this work targets uncertainty-accuracy alignment.
- **vs PGD / Madry**: Classic $\ell_p$ adversarial attacks seek maximum loss, while this work seeks maximum miscalibration—this "semantic adversarial + calibration objective" combination is a recent, calibration-friendly training paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ MCC naming + AUE dual attack + UA alignment trio is novel as a combination, though individual techniques have precedents
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 23 target domains zero-shot + multiple baselines + comprehensive augmentation comparisons
- Writing Quality: ⭐⭐⭐⭐ Clear concept naming, but many symbols; initial reading requires effort to distinguish $\psi_s/\psi_d/\theta_{\text{dec}}$
- Value: ⭐⭐⭐⭐⭐ Equipping the SAM series with "knowing what it doesn't know" is extremely important for safety-critical applications

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Segment Anything Across Shots: A Method and Benchmark](../../AAAI2026/segmentation/segment_anything_across_shots_a_method_and_benchmark.md)
- [\[AAAI 2026\] Segment and Matte Anything in a Unified Model (SAMA)](../../AAAI2026/segmentation/segment_and_matte_anything_in_a_unified_model.md)
- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](../../AAAI2026/segmentation/saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [\[ICCV 2025\] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](../../ICCV2025/segmentation/omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)
- [\[NeurIPS 2025\] Torch-Uncertainty: A Deep Learning Framework for Uncertainty Quantification](../../NeurIPS2025/segmentation/torch-uncertainty_a_deep_learning_framework_for_uncertainty_quantification.md)

</div>

<!-- RELATED:END -->
