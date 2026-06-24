---
title: >-
  [Paper Note] A Combination of Noise and Bilateral Filters Achieve Supralinear and Scalable Adversarial Robustness in CNNs
description: >-
  [CVPR 2026][AI Safety][Adversarial Robustness] This paper theoretically demonstrates from the perspective of decision boundary geometry that "Gaussian noise" and "image filtering" defend against adversarial attacks through two **complementary** mechanisms. Consequently, their combination yields **supralinear** robustness gains. Based on this, a minimalist preprocessor (pixel-level Gaussian noise + iterative bilateral filtering, applied during both training and inference) is p…
tags:
  - "CVPR 2026"
  - "AI Safety"
  - "Adversarial Robustness"
  - "Gaussian Noise"
  - "Bilateral Filter"
  - "Geometric Theory"
  - "Supralinear Gain"
date: 2026-05-08
content_hash: aeff22182e75cd14
---

# A Combination of Noise and Bilateral Filters Achieve Supralinear and Scalable Adversarial Robustness in CNNs

**Conference**: CVPR 2026  
**Code**: None  
**Area**: AI Security / Adversarial Robustness  
**Keywords**: Adversarial Robustness, Gaussian Noise, Bilateral Filter, Geometric Theory, Supralinear Gain

## TL;DR
This paper theoretically demonstrates from the perspective of decision boundary geometry that "Gaussian noise" and "image filtering" defend against adversarial attacks through two **complementary** mechanisms. Consequently, their combination yields **supralinear** robustness gains. Based on this, a minimalist preprocessor (pixel-level Gaussian noise + iterative bilateral filtering, applied during both training and inference) is proposed, which approaches or even exceeds SOTA defenses on RobustBench using only ~35% of the training FLOPs and half the parameters.

## Background & Motivation
**Background**: Adversarial training (AT) is currently the most effective defense, mixing adversarial samples (recently augmented with diffusion-model-synthesized data) into the training set, dominating RobustBench leaderboards for vision benchmarks like CIFAR-10. Another lighter alternative involves **simple input-side transformations**—adding additive noise or using Gaussian/bilateral filters/JPEG compression to smooth out perturbations, requiring near-zero extra computation.

**Limitations of Prior Work**: Adversarial training is computationally expensive (approx. 9× more FLOPs than standard training), and its robustness is **strictly tied to the attacks seen during training**. It may fail or become more vulnerable when encountering unseen attacks, requiring retraining for each new attack type. Conversely, simple filtering or noise methods provide limited gains when used in isolation; no **single** method can cover a wide range of attack types.

**Key Challenge**: Noise-based and filter-based methods have long been **studied in isolation**. No systematic analysis exists regarding what happens when they are combined—are they redundant (overlapping gains) or complementary (covering different attacks)? Neither theory nor sufficient experiments have addressed this.

**Goal**: (1) Theoretically characterize "what shape" of attacks noise and filtering respectively defend against; (2) Identify a combination that is simple, theoretically grounded, and plug-and-play for existing defenses.

**Key Insight**: View adversarial attacks through the **geometry of decision boundaries**—attacks are small perturbations pushing an image across the boundary. By fixing the "adversarial volume" $V_a(x)$, one can investigate what boundary shapes can bypass specific defenses.

**Core Idea**: The failure scenarios for noise-based and filter-based defenses are **exact opposites**—one is vulnerable to "spherical cluster" attacks while the other is vulnerable to "manifold-following" attacks. Thus, their combination offers broader coverage, resulting in robustness gains **greater than the sum of their parts** (supralinear).

## Method
The "Method" in this work consists of two layers: a **theoretical analysis** proving the complementarity of noise and filtering using geometric tools, and a **minimalist preprocessor** constructed based on those findings. The preprocessor itself is extremely simple (noise + bilateral filtering in series), representing a mechanism-based approach where the focus lies on the theoretical justification rather than a complex pipeline.

### Overall Architecture
The input is an image $x$. The preprocessor first adds independent zero-mean Gaussian noise $\varepsilon \sim \mathcal{N}(0,\sigma^2)$ to each pixel channel, then **iteratively applies** bilateral filtering multiple times. The preprocessed image is then fed into a standard CNN for classification. Crucially, this preprocessing is **applied during both training and inference** (learning to classify after "noise + filtering" during training and applying the same at test time). It can also be **directly wrapped around models already trained with AT** to stack both types of defense.

The theoretical logic follows: define the **adversarial volume** $V_a(x)$ to measure "how many adversarial attacks exist around an image," then derive what the "ideal filter" and "Gaussian noise" can and cannot block in worst-case scenarios. The discovery that their worst-case boundary shapes are mutually exclusive leads to the conclusion that their combination covers more area, thus achieving supralinearity.

### Key Designs

**1. Adversarial Volume $V_a(x)$: Quantifying "Attack Density" as Geometric Volume**

To compare different defenses, a unified metric is needed. This work models the decision boundary as a smooth $D-1$ dimensional manifold (with maximum sectional curvature $c$) and assumes a "true boundary" $h$ (the boundary defined by human classification where no misjudgment occurs). An adversarial attack $a_x$ is a norm-constrained perturbation $\|a_x\| \le r$ that pushes the image across the network's boundary but **not across the true boundary**. Inside a ball $B(x,r)$ centered at $x$ with radius $r$, the volume of all points "misclassified by the network but correctly judged by humans" is the adversarial volume:

$$V_a(x) = \int_{z \in B(x,r)} \mathbf{1}\big[f(z) \ne h(z) = f(x)\big]\, dz$$

$V_a(x)$ is central to the analysis: by fixing it, one can fairly ask "what boundary shape can bypass a defense." It also aligns with AT—existing theories suggest AT "smoothes out small dimples" on the decision boundary, equivalent to reducing $V_a(x)$. Thus, this analysis naturally complements rather than competes with AT.

**2. Filter vs. Noise: Opposing Geometric Mechanisms**

This is the theoretical core of the paper. An ideal denoising filter is modeled as a function $\phi(x):X\to X$ that leaves clean images unchanged but **pulls perturbations back toward the image manifold**. Intuitively, it "flattens" adversarial perturbations back to the correct side. However, in the worst case, if the decision boundary is **very close to the image manifold**, the filter cannot pull them back. Formally, for a fixed $V_a(x)$, an attack bypassing the filter may exist if:

$$V_a(x) > r\big(1-\lambda^{\min}_{\phi(x)}\big)\, S_{D-1}(c^{-1/2})$$

where $\lambda^{\min}_{\phi(x)}$ is the minimum eigenvalue of the Jacobian of $\phi$ and $S_{D-1}$ is the volume of a $D-1$ dimensional ball. Conclusion: **Filters excel at blocking attacks concentrated at the outer edges of the allowed norm ball but fail against "manifold-following" attacks.**

The mechanism for noise is entirely different: adding Gaussian noise $\varepsilon$ to an attacked image pushes it in a **random direction**. An attack only succeeds if the decision boundary **occupies a large volume** (a spherical cluster) around the attacked point. The upper bound for attack success probability is:

$$\Pr[x + a_x + \varepsilon \in A] \le 1 - 2\,\mathrm{erf}\!\left(\frac{-\rho_D(V_a(x))}{\sigma}\right)$$

where $\rho_D(V_a(x))$ is the radius of a $D$-dimensional ball with volume $V_a(x)$. Conclusion: **Noise excels at dispersing "thin filament" attacks but fails against "large spherical clusters."**

The two worst-case scenarios are **mutually exclusive**: filters fear "manifold-following" while noise fears "spherical clusters." An attack designed to bypass a filter is exactly the type that noise easily disperses, and vice versa. This is the fundamental reason for the supralinear gain.

**3. Minimalist Preprocessor: Gaussian Noise + Iterative Bilateral Filtering**

Translating the theory into a usable module involves two steps. First, add independent zero-mean Gaussian noise to each pixel channel. Second, apply **iterative bilateral filtering**. Bilateral filters preserve edges and structures while smoothing noise; multiple iterations provide thorough denoising. The default sequence is **noise-then-filter**: theoretical analysis indicates they are not commutative, and adding noise before filtering better preserves clean accuracy. The number of filtering iterations can differ between training and inference (e.g., 20 training / 100 inference, denoted as 20/100), allowing the network to learn representations under light processing while using stronger denoising during inference. This requires almost zero parameters and negligible overhead, yet combines "additive noise + filtering + stochastic training samples" into one frontend.

**4. Orthogonal Integration with Adversarial Training**

The preprocessor does not replace AT; it is **orthogonally stacked**. Following the TRADES protocol and using synthetic data from EDM diffusion models, the authors perform AT on various Wide ResNet sizes and wrap the preprocessor around them. Since AT pushes the learned boundary toward the true boundary (reducing $V_a(x)$), while the preprocessor changes the geometry of how attacks reach the boundary, their effects are additive. This allows the use of **smaller models, less synthetic data, and fewer training epochs** to match or exceed SOTA.

## Key Experimental Results

### Main Results: Integration with SOTA AT Models (CIFAR-10, AutoAttack)

Stacking the preprocessor on the WRN series consistently improves robust accuracy under AutoAttack by 6–9% while barely affecting clean accuracy. The strongest configuration exceeds previous SOTA.

| Model | Training Epochs / Synthetic Data | Clean | AutoAttack | EoT AutoAttack |
| :--- | :--- | :--- | :--- | :--- |
| WRN-28-10 | 400 / 1M | 88.96% | 61.60% | - |
| + Prepro. 20/80 | 400 / 1M | 88.24% | **69.60%** | 67.2% |
| WRN-28-10-long | 2400 / 20M | 90.12% | 64.40% | - |
| + Prepro. 20/80 | 2400 / 20M | 90.52% | **73.08%** | 70.9% |
| SOTA model [4] | 10000 / 500M | 93.68% | 73.71% | - |
| WRN-82-12 [4] | 3000 / 150M | 93.04% | 71.41% | - |
| + Prepro. 20/100 | 3000 / **50M** | 90.12% | **74.32%** | 73.00% |

The last row is a highlight: WRN-82-12 with the preprocessor reaches 74.32% on AutoAttack, +0.6% higher than previous SOTA, despite having half the model size, 6× less synthetic data, and ~1/3 the training epochs. The method ranks 2nd on AutoAttack and 3rd overall on RobustBench, using only ~35% of the typical training FLOPs.

### Ablation Study: Noise and Filtering are Both Essential (Standard CNN, CIFAR-10)

Looking at each column: Bilateral filtering (+Bil.) is only effective against specific attacks; Gaussian noise (+Noise) improves against all but with limited magnitude. Their combination (+Noise+Bil.) is the highest across **all** attacks, and for most, "combined gain > sum of individual gains" (supralinear).

| Method | Clean | FGSM | L∞ | EoT | L2 | C&W |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard CNN | 74.5% | 3.5% | 0.2% | 0.2% | 1.3% | 0.6% |
| + Bil. | 69.0% | 10.0% | 1.0% | 1.2% | 11.9% | 0.5% |
| + Noise | 68.5% | 22.8% | 25.5% | 12.0% | 49.6% | 43.0% |
| + Noise + Bil. | 67.9% | **33.9%** | **36.5%** | **18.9%** | **58.5%** | **47.2%** |
| linear gain (Sum of gains) | -11.5% | 25.8% | 26.1% | 12.8% | 58.9% | 42.3% |
| actual gain (Combined gain) | -6.6% | **30.4%** | **36.2%** | **18.7%** | 57.2% | **46.6%** |

Supralinearity holds for four out of six perturbations (FGSM / L∞ / EoT / C&W). The exception is L2, where the individual preprocessors were already quite robust, nearing saturation. Although clean accuracy drops, the combined drop (-6.6%) is smaller than the sum of individual drops (-11.5%), showing that negative impacts are sublinear.

### Key Findings
- **Complementarity > Redundancy**: Noise compensates for filtering's weaknesses and vice versa, resulting in supralinear rather than additive gains, except when a single method is near saturation (L2).
- **Order Matters**: The "noise-then-filter" sequence better preserves clean accuracy.
- **Computational Efficiency**: To achieve identical robust accuracy, this method requires only 15%–50% of the training FLOPs compared to competitors. It matches the accuracy of models requiring 6–9× more FLOPs during inference.
- **Resistance to Adaptive Attacks**: Even using TABPDA (designed specifically against this preprocessor), the method still improves robust accuracy by at least 5.5% across all tested attacks, with gains becoming more pronounced as perturbation strength increases.

## Highlights & Insights
- **Worst-case boundary shapes as explanation**: Explaining the noise/filter complementarity as "filters fear manifold-following, noise fears spherical clusters" provides a highly transferable geometric intuition that can guide future defense designs.
- **Quantifiable supralinearity**: This is not a vague "1+1>2" claim; it is grounded in "actual gain vs. linear gain" comparisons across multiple attacks, including an honest explanation of the L2 counterexample.
- **SOTA performance at low cost**: A near-zero-parameter input frontend allows smaller models to catch up with/overtake SOTA models that rely on massive compute. Since it is orthogonally stacked, it is immediately applicable to any AT model.

## Limitations & Future Work
- **Idealized theoretical assumptions**: "Ideal denoising filters" and "filament/spherical worst-case boundaries" are simplified models. Real bilateral filtering and real decision boundaries might not fully satisfy these. The interaction between noise and filtering was analyzed separately (ignoring the fact they are non-commutative).
- **Narrow dataset scope**: Main experiments are almost exclusively on CIFAR-10. Only brief ablation was done on Imagenet10; scalability to large-scale/high-resolution datasets like ImageNet is not fully verified.
- **Clean accuracy cost**: All preprocessing variants lead to a drop in clean accuracy (most pronounced in combined settings), requiring a trade-off in clean-accuracy-sensitive deployments.
- **Evaluation traps of stochasticity**: Noise-based defenses are historically prone to overestimation due to "gradient obfuscation." While this work uses EoT/APGD/TABPDA to mitigate this, performance under even stronger adaptive attacks remains to be seen.

## Related Work & Insights
- **vs. Adversarial Training**: AT focuses on "pushing the boundary" and is computationally expensive and attack-specific. This work focuses on input geometry, is near-zero cost, and is complementary to AT.
- **vs. Neural Network Denoising Preprocessors**: NN-based denoisers are effective but increase training/inference overhead and **can themselves be attacked**. This work uses parameter-free bilateral filters and noise, avoiding that vulnerability.
- **vs. Single Noise / Filter / JPEG Compression**: These were previously developed in isolation with limited gains. This work is the first to systematically demonstrate the supralinear effect of combining noise and filtering.

## Rating
- Novelty: ⭐⭐⭐⭐ Refreshing perspective using decision boundary geometry to explain complementarity as a verifiable theory.
- Experimental Thoroughness: ⭐⭐⭐ RobustBench and multiple adaptive attacks were tested, but primarily limited to CIFAR-10.
- Writing Quality: ⭐⭐⭐⭐ Clear correspondence between theory and experiments, though core proofs are relegated to the appendix.
- Value: ⭐⭐⭐⭐ Low-cost, plug-and-play, and stackable with AT—highly practical for resource-constrained robust deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Interaction of Compressibility and Adversarial Robustness](../../ICLR2026/ai_safety/on_the_interaction_of_compressibility_and_adversarial_robustness.md)
- [\[CVPR 2026\] Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks](towards_reliable_evaluation_of_adversarial_robustness_for_spiking_neural_network.md)
- [\[CVPR 2026\] A Provable Energy-Guided Test-Time Defense Boosting Adversarial Robustness of Large Vision-Language Models](a_provable_energy-guided_test-time_defense_boosting_adversarial_robustness_of_la.md)
- [\[CVPR 2026\] Shedding Light on VLN Robustness: A Black-box Framework for Indoor Lighting-based Adversarial Attack](shedding_light_on_vln_robustness_a_black-box_framework_for_indoor_lighting-based.md)
- [\[CVPR 2026\] Robustness Under Data Scarcity: Few-Shot Continual Adversarial Training for Evolving Threats](robustness_under_data_scarcity_few-shot_continual_adversarial_training_for_evolv.md)

</div>

<!-- RELATED:END -->
