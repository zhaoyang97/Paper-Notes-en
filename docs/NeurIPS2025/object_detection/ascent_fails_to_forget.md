---
title: >-
  [Paper Note] Ascent Fails to Forget
description: >-
  [NeurIPS 2025][Object Detection][machine unlearning] Starting from the statistical dependence between the forget set and the retain set, this paper theoretically and empirically demonstrates that the widely adopted gradient ascent / Descent-Ascent (DA) family of machine unlearning methods fails systematically in the presence of data correlations. In logistic regression, the DA solution is provably farther from the oracle than the original model, and in non-convex settings DA traps the model in inferior local minima.
tags:
  - NeurIPS 2025
  - Object Detection
  - machine unlearning
  - gradient ascent
  - statistical dependence
  - descent-ascent
  - logistic regression
date: 2026-05-08
content_hash: a7c87175a44b9b98
---

# Ascent Fails to Forget

**Conference**: NeurIPS 2025
**arXiv**: [2509.26427](https://arxiv.org/abs/2509.26427)
**Code**: None
**Area**: Object Detection
**Keywords**: machine unlearning, gradient ascent, statistical dependence, descent-ascent, logistic regression

## TL;DR

Starting from the statistical dependence between the forget set and the retain set, this paper theoretically and empirically demonstrates that the widely adopted gradient ascent / Descent-Ascent (DA) family of machine unlearning methods fails systematically in the presence of data correlations. In logistic regression, the DA solution is provably farther from the oracle than the original model, and in non-convex settings DA traps the model in inferior local minima.

## Background & Motivation

**Background**: Machine unlearning is a rapidly growing research direction aimed at removing the influence of specific training samples from trained models, with applications spanning data privacy compliance (GDPR "right to be forgotten"), removal of poisoned or outdated data, copyright protection, and LLM alignment. The ideal outcome is that the unlearned model $h_\theta^{\mathrm{UL}}$ behaves consistently with an oracle model trained from scratch on the retain set $\mathcal{R} = \mathcal{D} \setminus \mathcal{F}$. Retraining from scratch is the gold standard but is computationally prohibitive for large-scale models.

**Limitations of Prior Work**: Provably correct unlearning algorithms based on noisy gradient descent (without ascent steps) exist for convex models. However, deep neural networks lack provable guarantees due to their non-convex, non-smooth, and high-dimensional nature. The most widely used approach in practice is Descent-Ascent (DA): gradient ascent on the forget set $\mathcal{F}$ (to "forget" those samples) combined with gradient descent on the retain set $\mathcal{R}$ (to "preserve" model performance). Nevertheless, recent evaluation benchmarks have repeatedly shown DA to be highly unreliable: (1) it offers no theoretical performance guarantees; (2) it is extremely sensitive to learning rate and fine-tuning duration; and (3) it lacks a well-defined stopping criterion.

**Key Challenge**: DA implicitly assumes that the forget set and the retain set can be manipulated independently. In practice, however, both sets are drawn from the same data distribution and inevitably exhibit statistical dependence. When gradient ascent is applied to the forget set to degrade its metrics, the metrics on the retain set and the test set are unavoidably harmed as well due to statistical correlation.

**Goal**: To identify and theoretically characterize the fundamental cause of DA failure—the overlooked statistical dependence between datasets. Specifically, the paper addresses three questions: (1) Does DA necessarily harm the model for a randomly selected forget set? (2) In analytically tractable logistic regression, what is the relationship between the DA solution and the oracle solution? (3) In non-convex settings, can the damage caused by DA be recovered through subsequent fine-tuning?

**Key Insight**: Rather than proposing a new unlearning method, the authors take an analytical perspective to systematically expose the root cause of DA failure. The argument proceeds in increasing complexity: from the simplest case of random forget sets (naturally highly correlated), to logistic regression (analytically tractable convex problems), and then to low-dimensional non-convex examples (local minima traps).

**Core Idea**: The statistical dependence between the forget set and the retain set—even simple correlation—is sufficient to cause gradient-ascent-based unlearning methods to fail systematically. "Doing nothing" is demonstrably better than executing DA unlearning.

## Method

### Overall Architecture

This is a theoretical analysis paper. The authors construct three progressive levels of analysis: (1) probabilistic analysis of random forget sets, proving that the oracle's performance on the forget set must match that on the test set; (2) closed-form analysis of high-dimensional logistic regression, proving that the DA solution and the oracle solution lie on opposite sides of the original model; and (3) a low-dimensional non-convex toy example demonstrating that DA traps the model in an incorrect local minimum. Beyond the theoretical analysis, experiments on ResNet-9/ResNet-18 with CIFAR-10/ImageNetLiving-17 validate the theoretical predictions. The evaluation metric is KLoM (KL Divergence of Margins), which quantifies the distance between the prediction distribution of the unlearned model and that of 100 oracle models.

### Key Designs

1. **Impossibility Lemma for Random Forget Sets (Lemma 1)**:

    - Function: Proves that for a randomly selected forget set, any successful unlearning algorithm should not degrade metrics on the forget set.
    - Mechanism: When $\mathcal{F}$ is drawn uniformly at random from $\mathcal{D}$, the difference in accuracy between the oracle on the forget set and the test set is bounded in probability as $P(|\mathrm{Acc}_{\mathcal{T}} - \mathrm{Acc}_{\mathcal{F}}| \geq \epsilon) \leq 2\exp(-2|\mathcal{F}|\epsilon^2)$ (via Hoeffding's inequality). This implies the oracle should perform nearly identically on the forget set and the test set. Consequently, any method that unlearns by degrading forget-set metrics will push the model away from the oracle.
    - Design Motivation: Directly refutes the core operational logic of DA—"degrading performance on the forget set equals successful unlearning."

2. **Closed-Form Analysis of Logistic Regression (Lemmas 2–5)**:

    - Function: Precisely characterizes the relative positions of the DA solution, oracle solution, and original model solution in the convex setting.
    - Mechanism: The analysis considers binary logistic regression with ridge regularization under a semi-orthogonality assumption (samples on different coordinate axes are orthogonal; samples on the same axis may be correlated). Defining the forget-set proportion as $|\mathcal{F}_j| = \alpha \cdot |\mathcal{R}_j|$, closed-form solutions for all three problems are obtained via the Lambert-W function: $w_j^{\mathcal{D}} = W\left(\frac{(1+\alpha)|\mathcal{R}_j|}{\lambda|\mathcal{D}|}\right)$, $w_j^{\mathcal{R}} = W\left(\frac{|\mathcal{R}_j|}{\lambda|\mathcal{R}|}\right)$, $w_j^{\text{DA}} = W\left(\frac{(1-\alpha|\mathcal{R}|/|\mathcal{F}|)|R_j|}{\lambda|R|}\right)$. The key result (Lemma 3) is: $(w_j^{\text{DA}} - w_j^{\mathcal{D}}) \cdot (w_j^{\mathcal{D}} - w_j^{\mathcal{R}}) \geq 0$, i.e., the DA solution and the oracle solution lie on opposite sides of the original model—DA pushes the model in the direction opposite to the oracle.
    - Design Motivation: Even in the simplest convex setting, DA's directional error can be proven, demonstrating that the problem stems not from non-convexity or hyperparameter choices but from data dependence itself.

3. **Cross-Dimension Correlation Analysis (Lemmas 6–10)**:

    - Function: Extends single-dimension conclusions to the two-dimensional case with cross-dimension correlation $\epsilon$.
    - Mechanism: Two groups of samples are considered: $x_i = (1, \epsilon)$ in the retain set and $x_j = (\epsilon, 1)$ in the forget set, where $\epsilon$ controls the degree of inter-group correlation. Through a coordinate transformation, closed-form expressions for the oracle, original model, and DA solutions are obtained, proving the existence of a range of $\alpha$ values for which DA is harmful. Numerical results show this harmful interval is typically wide, grows with the forget ratio, and is broader under weak correlation.
    - Design Motivation: Correlations between forget and retain sets in real data do not always lie along the same dimension; it is necessary to demonstrate that weak cross-dimension correlations similarly cause DA to fail.

### Loss & Training

Three optimization objectives are compared: pretraining $\mathcal{L}_{\mathcal{D}}$ performs gradient descent on the full dataset; the oracle $\mathcal{L}_{\mathcal{R}}$ performs gradient descent on the retain set; DA $\mathcal{L}_{\text{DA}}$ performs gradient descent on the retain set and gradient ascent on the forget set. The DA loss is:
$$\mathcal{L}_{\text{DA}} = \frac{1}{|\mathcal{R}|}\sum_{\mathcal{R}} e^{-y_i\langle\mathbf{w},\mathbf{x}_i\rangle} - \frac{1}{|\mathcal{F}|}\sum_{\mathcal{F}} e^{-y_i\langle\mathbf{w},\mathbf{x}_i\rangle} + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$$
The critical issue lies in the subtraction term—it does not eliminate influence but actively pushes the model in the opposite direction.

The low-dimensional non-convex toy example uses an MSE loss with ridge regularization on a sigmoid network $h_\theta(\mathbf{x}_i) = \sigma(ax_i + bx_i^2)$ with 4 weighted samples. GDA zeroes out the forget-sample gradients (positive and negative gradients cancel), causing the model to slide from the global optimum into a local optimum that captures all samples but yields a lower actual accuracy decision boundary.

## Key Experimental Results

### Main Results (ResNet-9 / CIFAR-10)

Evaluation uses the KLoM metric, where KLoM approaching 0 indicates perfect unlearning. Experiments test GA (gradient ascent only) and GDA (ascent + descent) on different forget sets.

| Forget Set Type | GA/GDA Behavior | Unlearning Quality (KLoM) | Model Performance |
|----------------|-----------------|--------------------------|-------------------|
| High-influence samples (PC1) | Almost no forgetting | Far from oracle | Severe degradation or no change |
| Random 10 samples | Some runs appear successful | Requires precise hyperparameter + stopping point selection | Cannot determine hyperparameters a priori |
| PC2 samples | GDA partially improves | Forgetting cost ~25% of retraining | Only 0.2% data forgotten |

### Summary of Theoretical Results (Ablation Analysis)

| Analysis Level | Setting | Key Conclusion | Implication |
|---------------|---------|----------------|-------------|
| Lemma 1 | Random forget set | $\mathrm{Acc}_{\mathcal{F}} \approx \mathrm{Acc}_{\mathcal{T}}$ (oracle) | DA degrading forget-set metrics = degrading test performance |
| Lemma 3 | 1D logistic regression | DA solution and oracle lie on opposite sides of original model | Every DA step moves farther from the oracle |
| Corollary 1 | $\lambda \to 0$ | $\Delta_{\mathcal{R},\mathcal{D}} \to 0$, $\Delta_{\mathcal{R},\text{DA}} \to \infty$ | DA is extremely unstable |
| Lemma 9–10 | 2D cross-dimension correlation | Structurally harmful $\alpha$ interval exists | Weak correlation also causes DA failure |
| Toy example | Non-convex sigmoid | DA falls into incorrect local minimum | Subsequent fine-tuning cannot recover |

### Key Findings

- **DA is either ineffective (model unchanged) or harmful (model collapses) under the vast majority of hyperparameter configurations**: Fig. 1 shows that GA and GDA runs either remain close to the pretrained initialization or suffer severe test performance degradation, with almost no intermediate state.
- **"Ascent Forgets Illusion"**: When the forget set is very small (e.g., 10 random samples), extensive hyperparameter search occasionally yields seemingly successful runs, but this requires simultaneously selecting the perfect learning rate and stopping time—neither of which can be determined a priori—making it essentially cherry-picking.
- **Unlearning difficulty is highly dependent on the choice of forget set**: High-influence samples (first principal component of the influence matrix) are the hardest to forget, with no successful runs observed; random samples occasionally succeed but unreliably.
- **Corollary 1 has far-reaching implications**: As regularization approaches zero (common in deep learning), the difference between the oracle and the original model approaches zero, but the distance between the DA solution and the oracle approaches infinity—the instability of DA grows unboundedly as regularization weakens.

## Highlights & Insights

- **The fundamentality of the statistical dependence perspective**: Prior literature attributed DA failure to hyperparameter sensitivity, non-convexity, or the lack of a stopping criterion. This paper demonstrates that DA necessarily fails even in convex settings with optimal hyperparameters. The root cause lies not in algorithmic details but in data structure—a fundamental obstacle that cannot be resolved through tuning.
- **The counter-intuitive conclusion that "doing nothing is better than DA"**: Lemma 3 shows that DA pushes the model to the opposite side of the original model relative to the oracle, meaning that leaving the original model unchanged is actually closer to the oracle. This is an important warning for practitioners.
- **Transferability of the analytical approach**: The closed-form analysis technique using the Lambert-W function and the coordinate transformation approach for handling cross-dimension correlations can be applied to analyze data dependence in other optimization problems.

## Limitations & Future Work

- **Purely negative results with no proposed alternative**: The paper systematically dismantles DA methods but does not address "what should be done instead." The authors briefly mention rewind methods and stochastic noise-based approaches but do not elaborate.
- **Strong theoretical assumptions**: The semi-orthogonality data assumption (Assumptions 1–2) is rarely strictly satisfied in real data; logistic regression uses exponential loss as a surrogate for logistic loss, which may introduce approximation error.
- **Insufficient theoretical analysis of non-random forget sets**: In practice, forget sets are typically not random (e.g., specific users or specific classes), and the paper's theoretical analysis of such structured forget sets is limited (mainly addressed in the toy example).
- **Theoretical extension to deep networks**: Theoretical analysis is concentrated on logistic regression and a 2-parameter sigmoid network; extending the theory to deep networks is an important direction for future work.

## Related Work & Insights

- **vs. Certified Unlearning (Neel et al. 2021, Guo et al. 2023)**: These methods are based on noisy gradient descent (without ascent steps) and offer theoretical guarantees for convex models. The present analysis explains from the opposite direction why the "ascent step" is harmful—not because it is "insufficient" but because it is "directionally wrong."
- **vs. Rewind methods (Mu & Klabjan 2024)**: These provide theoretical guarantees for non-convex models but require storing complete model states or many proximal-point iterations, incurring high computational cost. The conclusions of this paper support the design philosophy of rewind-type methods (avoiding ascent steps).
- **vs. SCRUB (Kurmanji et al. 2024)**: A fine-tuning method using a KL divergence objective; the present paper argues it faces the same fundamental challenges as other DA methods.
- **vs. Georgiev et al. (2024) predictive data attribution method**: Also observes empirical failure of DA but does not provide a theoretical explanation based on statistical dependence. The present paper provides controlled experiments and theory to attribute these failures.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Explaining DA failure through statistical dependence is a wholly new and incisive perspective; the counter-intuitive result in the convex setting is particularly compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐ Theory + ResNet/ViT empirical coverage is comprehensive, but comparisons with alternative methods and more real-world scenario validation are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ The argumentation proceeds systematically from simple to complex, the mathematics is rigorous with clear intuition, and the figures are well designed.
- Value: ⭐⭐⭐⭐⭐ Carries profound cautionary significance for the machine unlearning field and is likely to reshape the design philosophy of subsequent methods.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection](recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)
- [\[NeurIPS 2025\] FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies](flexevent_towards_flexible_event-frame_object_detection_at_varying_operational_f.md)
- [\[NeurIPS 2025\] BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes](burstdeflicker_a_benchmark_dataset_for_flicker_removal_in_dynamic_scenes.md)
- [\[NeurIPS 2025\] Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](video-rag_visually-aligned_retrieval-augmented_long_video_comprehension.md)
- [\[NeurIPS 2025\] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)

<!-- RELATED:END -->
