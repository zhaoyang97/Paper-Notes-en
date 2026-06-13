---
title: >-
  [Paper Note] Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?
description: >-
  [ICML 2026][Model Compression][Dataset Distillation] This paper proposes the C2R framework, which decomposes the robustness problem in dataset distillation into a "minimum robust margin" problem. By utilizing a trio of "…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Dataset Distillation"
  - "Robust Distillation"
  - "Adversarial Curriculum"
  - "Robust Margin"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 589237b67dbf34b1
---

# Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?

**Conference**: ICML 2026  
**arXiv**: [2605.20606](https://arxiv.org/abs/2605.20606)  
**Code**: https://github.com/SLGSP/CCR  
**Area**: Model Compression / Dataset Distillation / Adversarial Robustness  
**Keywords**: Dataset Distillation, Robust Distillation, Adversarial Curriculum, Robust Margin, Contrastive Learning

## TL;DR
This paper proposes the C2R framework, which decomposes the robustness problem in dataset distillation into a "minimum robust margin" problem. By utilizing a trio of "Attack-Aware Curriculum (AAC) + Contrastive Robustness Loss (CRL) + Line-search PGD (LS-PGD)," the synthesized datasets achieve an average improvement of approximately 2.8% in robust accuracy across six attacks compared to previous robust distillation SOTA.

## Background & Motivation

**Background**: Dataset Distillation (DD) compresses a large training set into tens to thousands of synthetic samples, allowing small models trained on the synthetic set to approach the accuracy of full-data training. Mainstream approaches include gradient matching, matching training trajectories (MTT), distribution matching, generative distillation, and decoupled methods like SRe2L/D4M. Most methods only optimize for clean accuracy, with adversarial robustness rarely included in the objective function.

**Limitations of Prior Work**: When distilled data is used in security-sensitive scenarios, attacks such as PGD/CW/VMI/Jitter can easily compromise the model. Existing "robust distillation" works (curvature regularization in GUARD, information bottleneck alignment in ROME, NTK meta-learning by Tsilivis et al.) improve robustness but suffer from a poor **accuracy–robustness trade-off**: either clean accuracy drops significantly, or they still fail under strong attacks.

**Key Challenge**: The authors point out two structural vulnerabilities in existing methods: (i) **Margin Mismatch**: Robust risk is dominated by a small fraction of samples with the "minimum robust margin" (Schmidt et al. 2018), yet existing methods treat all adversarial counterparts equally, diluting the optimization budget on many "already robust" easy points; (ii) **Boundary Neglect**: Popular "class-mean alignment" $\mathcal{L}_{\mathrm{rob}}=\sum_c \|\mathbb{E}[e(x_c)]-\mathbb{E}[e(\tilde x_c)]\|_2^2$ only pursues global intra-class similarity without explicitly increasing inter-class distance near decision boundaries, where adversarial errors occur.

**Goal**: Design a robust distillation objective that can (a) concentrate optimization on adversarial samples with "minimum margins," (b) explicitly expand class margins near decision boundaries, and (c) avoid exploding distillation costs.

**Key Insight**: Starting from the robust hinge loss $\mathcal{L}_{\mathrm{hinge}}=\mathbb{E}[[1-\underline{m}(x;\theta)]_+]$, it is proven that $\max_i v_i(\theta) = [1-\min_i \underline{m}(x_i;\theta)]_+$, meaning "improving the worst hinge = improving the minimum robust margin." This transforms the question of "whom to optimize" from a heuristic into a provable ranking.

**Core Idea**: Use PGD to estimate the robust margin of each sample $\widehat{m}_{\mathrm{rob}}(x;\theta)=g_\theta(x+\delta_T)$, rank them from hard to easy to form an Attack-Aware Curriculum (AAC), employ a Contrastive Robustness Loss (CRL) to force "clean–adv intra-class pulling and nearest inter-class pushing," and control costs with Line-search PGD and class-balanced queues.

## Method

### Overall Architecture

C2R follows the standard bi-level structure of DD: the outer loop updates the synthetic set $X=\{(x_s,y_s)\}_{s=1}^N$, and the inner loop performs short training of a model $f_\theta$ on $X$. In each epoch:

1.  **Generate Adversarial Counterparts**: For each synthetic sample $x$, use LS-PGD to calculate an adversarial perturbation $\delta$, resulting in $\tilde x=x+\delta$.
2.  **Calculate Robust Margin Score**: $s(x)=[1-\widehat{m}_{\mathrm{rob}}(x;\theta)]_+$, where larger scores indicate proximity to the decision boundary.
3.  **Construct Batch via Ranking (AAC)**: Form mini-batches from hard to easy, focusing CRL primarily on the low-margin tail.
4.  **Optimization Objective**: $\mathcal{L}_{\mathrm{C^2R}}=(1-\eta)\mathcal{L}_{\mathrm{perf}}+\eta\mathcal{L}_{\mathrm{CRL}}$, where $\mathcal{L}_{\mathrm{perf}}$ is clean CE and $\mathcal{L}_{\mathrm{CRL}}$ is the contrastive robustness loss.
5.  **Class-balanced Memory Queue**: Provides numerous "hard negatives" for CRL, avoiding $O(M^2)$ intra-batch overhead.

The input is the real dataset and a distillation budget IPC (samples per class); the output is a synthetic set $X$ optimized for robust training. Downstream training only requires standard adversarial training (PGD-AT) on $X$.

### Key Designs

1.  **Attack-Aware Curriculum (AAC)**:
    *   **Function**: Identifies adversarial samples with the "minimum robust margin" and prioritizes the distillation update budget for them, rather than treating all $\tilde x$ equally.
    *   **Mechanism**: Leverages the key identity $\arg\max_i [1-\underline{m}(x_i)]_+ = \arg\min_i \underline{m}(x_i)$. The implementation uses a PGD inner loop to approximate $\underline{m}$: $\delta_{t+1}=\Pi_\Delta(\delta_t+\alpha\,\mathrm{sign}(\nabla_x\ell(f_\theta(x+\delta_t),y)))$, then $s(x)=[1-g_\theta(x+\delta_T)]_+$. Batches are organized in descending order of $s(x)$, "pressurizing" CRL on the small-margin tail.
    *   **Design Motivation**: Prior robust DD used "class-mean alignment" or averaged over all adv samples, causing optimization dilution. AAC directly incorporates the decisive statistic for robust risk (minimum margin) into the training loop, aligning with the robust theory that "worst-case dictates robust risk."

2.  **Contrastive Robustness Loss (CRL)**:
    *   **Function**: Replaces "class-mean alignment" with instance-level supervised contrast in embedding space to achieve two goals: pulling $(x,\tilde x)$ together as positive pairs and pushing $x$ away from the nearest different classes (including their adv versions).
    *   **Mechanism**: For an anchor $x_i$, define the positive set $P(i)=\{\tilde x_i\}\cup\{x_j,\tilde x_j: y_j=y_i\}$ and candidate set $A(i)=P(i)\cup\{x_k,\tilde x_k: y_k\neq y_i\}$. The loss is $\mathcal{L}_{\mathrm{CRL}}=\frac{1}{M}\sum_i [-\sum_{a\in P(i)} \frac{1}{|P(i)|}\log\frac{\exp(g_{i,a}/\tau)}{\sum_{b\in A(i)}\exp(g_{i,b}/\tau)}]$, where $g_{i,a}=\mathrm{sim}(e(x_i),e(a))$. The numerator pulls "clean–adv same-class" pairs, while the denominator exerts maximum pressure on the most similar different classes, corresponding to the $\max_{k\neq y}f_k(x+\delta)$ term in the robust margin formula.
    *   **Design Motivation**: Class-mean alignment $\|\mathbb{E}[e(x_c)]-\mathbb{E}[e(\tilde x_c)]\|^2$ improves "average invariance" but fails to exert pressure on fragile sub-patterns near boundaries. CRL explicitly includes "nearest different classes" in the gradient path, physically pushing the robust boundaries apart.

3.  **LS-PGD + Class-balanced Memory Queue**:
    *   **Function**: Reduces the cost of "repeated multi-step PGD" and "$O(M^2)$ full contrast" to maintain overall training efficiency.
    *   **Mechanism**: LS-PGD uses warm-start by caching the previous perturbation $\hat\delta(x)$. If the loss at $x+\hat\delta(x)$ has not decreased, it is reused; otherwise, it computes only **one backward pass** to get direction $v=\mathrm{sign}(\nabla_x \ell)$, followed by **pure forward passes** over a geometric sequence $\mathcal{S}=\{\alpha\beta^q\}_{q=0}^{Z-1}$ ($Z\in\{2,3\}$) for line search, selecting $\delta'=\arg\max_{\eta\in\mathcal{S}}\ell(f_\theta(x+\Pi_\Delta(\hat\delta+\eta v)),y)$. The memory queue maintains a FIFO queue (capacity $Q$) per class to cache historical embeddings. Anchor points use a low-dimensional random projection $R\in\mathbb{R}^{r\times d}$ to filter top-$k$ hard negatives, reducing per-step cost from $O(M^2)$ to $O(Mk)$.
    *   **Design Motivation**: Inner-loop attacks are the main bottleneck in robust DD. Warm-start + forward line-search amortizes costs to nearly one backward pass without decaying attack strength. The memory queue ensures a stable supply of informative negative impostors for contrastive learning.

### Loss & Training

The outer objective is $\mathcal{L}_{\mathrm{C^2R}}=(1-\eta)\mathcal{L}_{\mathrm{perf}}+\eta\mathcal{L}_{\mathrm{CRL}}$, where $\eta\in[0,1]$ controls the trade-off. AAC itself **introduces no extra loss terms**; it only reorders batch sampling to concentrate CRL gradients on the low-margin tail. Downstream training on the distilled set uses standard PGD adversarial training with perturbation budget $|\varepsilon|=2/255$.

## Key Experimental Results

### Main Results

Evaluated across 3 foundation datasets (CIFAR-10/100, Tiny-ImageNet) × 5 IPCs × 5 attack types (FGSM/PGD/CW/VMI/Jitter), plus 6 ImageNet-1K subsets. Representative results for IPC=10:

| Dataset / Attack | IPC | SRe2L | D4M | ROME | C2R | Gain vs ROME |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CIFAR-10 / PGD | 10 | 13.09 | 20.14 | 24.01 | 28.49 | **+4.37** |
| CIFAR-10 / VMI | 10 | 13.28 | 20.14 | ≈ROME | 28.49 | +4.37 |
| CIFAR-100 / PGD | 10 | 7.08 | 4.25 | 8.42 | 12.92 | **+2.82** |
| Tiny-ImageNet / PGD | 10 | 1.59 | 0.97 | 1.36 | 3.27 | +1.73 |
| CIFAR-10 / Clean | 10 | 37.53 | 48.16 | 47.94 | ~46–48 | Comparable |

Average across six attacks: **C2R achieves ~2.8% higher robust accuracy than previous SOTA robust DD**, without significant degradation in clean accuracy.

### Ablation Study

| Configuration | Key Observation | Description |
| :--- | :--- | :--- |
| Full C2R | Best robust accuracy | AAC + CRL + LS-PGD |
| w/o AAC (uniform sampling) | Significant drop in robust accuracy | Validates theory that low-margin samples drive robust risk |
| w/o CRL (Back to mean alignment) | Vulnerable at boundaries, fails under strong attacks | Validates necessity of boundary-level class separation |
| LS-PGD → Standard $T$-step PGD | Similar accuracy, higher VRAM/Time | LS-PGD is as effective under fixed compute budget |
| w/o memory queue | Insufficient hard negatives, CRL gains halved | Queue is necessary for CRL scalability |

### Key Findings

*   **Consistency between theory and empiricism**: Disabling AAC causes the largest loss in robust accuracy, aligning with the "min margin dominates robust risk" proposition.
*   **CRL > Class-mean alignment**: CRL wins across all IPCs, indicating that class-mean alignment is an under-fit objective for "boundary-level geometry."
*   **Higher gains at small IPCs**: The relative improvement of C2R is most significant at IPC=1, as smaller synthetic sets mean each sample "covers more area," making margin optimization more critical.
*   **Wider gaps under strong attacks**: The gap between C2R and baselines is larger under strong attacks (VMI/CW) than FGSM, proving that boundary-level regularization truly expands robust margins rather than just blocking weak attacks.

## Highlights & Insights

*   **Reformulates robust distillation as a "minimum margin optimization" problem** and provides a computable proxy $s(x)=[1-\widehat{m}_{\mathrm{rob}}]_+$, turning the "whom to optimize" question from a heuristic into a provable ranking.
*   **CRL explicitly encodes the robust margin formula** $\max_{k\neq y}f_k(x+\delta)$ into the loss: hard negatives in the denominator directly correspond to the $\max$ term. This is a clean case of bridging "adversarial geometry" with "contrastive learning."
*   **LS-PGD is an efficient engineering trick**: Warm-start + forward probes maintain PGD strength while cutting inner-loop costs to nearly one backward pass, making it applicable to any method requiring repeated inner-loop attacks.

## Limitations & Future Work

*   Experiments are focused on classification and small $\ell_\infty$ budgets ($\varepsilon=2/255$); performance under larger budgets, $\ell_2$/universal perturbations, or AutoAttack was not systematically provided.
*   AAC scores depend on the current $\theta$; early in distillation, $\theta$ is unstable, which may lead to noisy "minimum margin" estimates.
*   CRL depends on memory queue hyperparameters ($Q$, $r$), which may require dataset-specific tuning.
*   The method uses $(1-\eta)\mathcal{L}_{\mathrm{perf}}$ to constrain clean accuracy, but has not explored using AAC concepts to simultaneously improve hard samples for clean accuracy.

## Related Work & Insights
*   **vs ROME (Information Bottleneck Alignment)**: ROME uses distribution-level regularization; Ours descends to "minimum margin + boundary geometry," proving more stable under strong attacks.
*   **vs GUARD (Curvature Regularization)**: GUARD reduces sensitivity via curvature; C2R avoids explicit curvature constraints, using a margin perspective directly linked to robust risk.
*   **vs Standard Robust Training (Madry et al.)**: Madry focus on "per-sample worst-case"; C2R elevates this to "per-dataset worst-case" at the curriculum level, fitting the low-sample scenario of DD.
*   **vs SupCon (Khosla et al.)**: CRL is a natural extension of supervised contrast for adversarial geometry, explicitly incorporating $\tilde x$ into positive sets and "nearest different classes" into negative sets.

## Rating
*   Novelty: ⭐⭐⭐⭐ Links "minimum robust margin" theory with curriculum and contrastive loss into a clean framework.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of 3 base sets + 6 ImageNet subsets across 5 IPCs and 5 attacks, with ablations mapped to theoretical propositions.
*   Writing Quality: ⭐⭐⭐⭐ Clear theoretical sections; Propositions 8/9 explain the motivation for selecting min-margin samples well.
*   Value: ⭐⭐⭐⭐ Average improvement of +2.8% in strong compression zones (IPC=1/5/10) is practically significant for deploying DD.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DIVER: Diving Deeper into Distilled Data via Expressive Semantic Recovery](diverdiving_deeper_into_distilled_data_via_expressive_semantic_recovery.md)
- [\[NeurIPS 2025\] Graph Your Own Prompt](../../NeurIPS2025/model_compression/graph_your_own_prompt.md)
- [\[ICML 2026\] IDLM: Inverse-distilled Diffusion Language Models](idlm_inverse-distilled_diffusion_language_models.md)
- [\[ICML 2026\] ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin](arcvq-vae_a_spherical_vector_quantization_framework_with_arccosine_additive_marg.md)
- [\[ICML 2026\] Critique-Guided Distillation for Robust Reasoning via Refinement](critique-guided_distillation_for_robust_reasoning_via_refinement.md)

</div>

<!-- RELATED:END -->
