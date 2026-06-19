---
title: >-
  [Paper Note] SoC: Semantic Orthogonal Calibration for Test-Time Prompt Tuning
description: >-
  [CVPR 2026][Multimodal VLM][Vision-Language Model] To address the issue in CLIP test-time prompt tuning (TPT) where imposing strict orthogonal constraints to enhance class separability leads to overconfidence and poor calibration, this paper replaces hard orthogonal constraints with a **Huber-style smooth orthogonal calibration (SoC)**. By applying a capped, gentle rep
tags:
  - CVPR 2026
  - Multimodal VLM
  - Vision-Language Model
date: 2026-05-08
content_hash: d6ffec1b1a5b0e93
---
# SoC: Semantic Orthogonal Calibration for Test-Time Prompt Tuning

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Fillioux_SoC_Semantic_Orthogonal_Calibration_for_Test-Time_Prompt_Tuning_CVPR_2026_paper.html)  
**Code**: https://github.com/leofillioux/SoC  
**Area**: Multimodal VLM  
**Keywords**: Test-time prompt tuning, model calibration, vision-language models, orthogonal regularization, Huber loss

## TL;DR
To address the issue in CLIP test-time prompt tuning (TPT) where imposing strict orthogonal constraints to enhance class separability leads to overconfidence and poor calibration, this paper replaces hard orthogonal constraints with a **Huber-style smooth orthogonal calibration (SoC)**. By applying a capped, gentle repulsion to semantically similar class prototypes, SoC significantly reduces the Expected Calibration Error (ECE) while maintaining high classification accuracy.

## Background & Motivation
**Background**: VLMs (represented by CLIP) can perform zero-shot classification using text prompts like "a photo of a [CLASS]". Test-time prompt tuning (TPT) further optimizes prompt vectors online during inference without labels via entropy minimization. It processes a batch of data augmentations for a single test image and performs a one-step gradient descent update using the average predicted entropy.

**Limitations of Prior Work**: Pure entropy minimization causes predictions to become **overconfident**, which is hazardous in safety-critical scenarios like healthcare and autonomous driving because confidence scores are no longer reliable (poor calibration). To mitigate this, improvements like C-TPT (pushing text embeddings away from the centroid to increase dispersion) and O-TPT (imposing pairwise **complete orthogonal** constraints $\|S-I_K\|_2^2$ on the prototype matrix) have emerged. O-TPT was the previous SOTA.

**Key Challenge**: Complete orthogonality is a **quadratic penalty**, where pairs with higher similarity are pushed more aggressively. However, some categories are naturally proximate—"annual crop land / permanent crop land" or "dog / puppy" share overlapping semantics in the vision-language embedding space. Forcing these prototypes to be orthogonal destroys the pre-trained semantic manifold, creating artificial overconfidence for these "hard-to-distinguish but related" categories. Empirical results in Figure 2 show that O-TPT's ECE is significantly higher than ours for category pairs with zero-shot cosine similarity $>0.85$.

**Goal**: Design a regularization that separates class prototypes while **respecting semantic proximity**, thereby genuinely improving calibration rather than using "dispersion" or "complete orthogonality" as crude proxies.

**Key Insight**: The authors analyze the geometric dynamics of a one-step gradient update. Since TPT only takes one step, the extent to which the "worst-case similarity $\mu=\max_{i\ne j}s_{ij}$ is compressed in one step" directly determines the lower bound of confidence. Complete orthogonality compresses high-similarity pairs too aggressively, leading to overconfidence.

**Core Idea**: Replace the pure quadratic penalty in complete orthogonality with a **Huber loss**. It acts quadratically when $s\le\delta$ (normally repelling low-similarity pairs) but switches to linear when $s>\delta$, where the **gradient is capped**. This applies a bounded, gentle repulsion to highly similar category pairs, preserving the semantic structure.

## Method

### Overall Architecture
SoC does not alter the general TPT pipeline but replaces the "calibration regularization term." Following the CLIP setup: the vision encoder $f_\omega$ and text encoder $f_\varepsilon$ encode images and text templates into an $\ell_2$-normalized space, yielding image embeddings $v$ and class prototypes $t_k=f_\varepsilon(\text{"a photo of a [CLASS]"})$. Logits and softmax are $z_k=\tau\, v^\top t_k$ and $p_k=\mathrm{softmax}(z)_k$, where $\tau=1/T$ is the pre-trained temperature. Stacking all normalized prototypes into a matrix $E$, $S=EE^\top$ represents the pairwise cosine similarity matrix with $s_{ij}=t_i^\top t_j$.

At test time, for a single image with 64 augmented views, TPT uses the average prediction entropy $L_{TPT}=-\sum_k \tilde p_k(v)\log \tilde p_k(v)$ as a self-supervised signal ($\tilde p_k$ is averaged only over high-confidence augmentations below the $\rho$-quantile of entropy) and performs **one step** of AdamW gradient update on the prompt. SoC adds a Huber-style prototype separation term to this objective. The core design lies in how this term is constructed and why it calibrates better than complete orthogonality, detailed in the following three key designs.

### Key Designs

**1. Huber-style Smooth Orthogonal Calibration: Capping the Repulsion for High-Similarity Pairs**

Complete orthogonality (O-TPT) uses the penalty $\|S-I_K\|_2^2$, applying a quadratic penalty $s_{ij}^2$ to each off-diagonal term $s_{ij}$ with a gradient $\propto s_{ij}$. Higher similarity leads to stronger repulsion, causing semantically related categories to be excessively separated. SoC replaces the pairwise penalty with a Huber loss: given a threshold $\delta\in[0,1]$,

$$L_{Huber}(s,\delta)=\begin{cases} \dfrac{s^2}{2}, & s\le\delta,\\[4pt] \delta\left(s-\dfrac{\delta}{2}\right), & s>\delta,\end{cases}$$

The average is taken over all pairs in the lower triangle of the similarity matrix and added to the TPT objective:

$$L_{SoC}=L_{TPT}+\lambda\cdot\frac{2}{K(K-1)}\sum_{i<j}L_{Huber}(s_{ij},\delta).$$

The key is that when $s\le\delta$, it behaves like the quadratic penalty, repelling low-similarity pairs. Once $s>\delta$ enters the linear segment, the **gradient remains constant at $\delta$** regardless of $s$, effectively capping the repulsive force. Highly overlapping semantic pairs are not violently decoupled in one step, preserving semantic proximity and avoiding the overconfidence side effect of O-TPT. $\delta$ is the sole "gentleness" control; smaller values are more tolerant of similar categories.

**2. Confidence Lower Bound under Cosine Coherence: Translating "Worst-case Similarity" to a "Confidence Floor"**

Why does compressing similarity affect calibration? The authors provide a lower bound connecting geometric quantities directly to softmax confidence. Define the **cosine coherence** of the set as $\mu \triangleq \max_{i\ne j} t_i^\top t_j\in[0,1]$ (the similarity of the most similar prototype pair). Proposition 1 proves that for any unit vector $v$, the maximum softmax confidence satisfies:

$$p_{max}(v)\ge \frac{1}{1+(K-1)\exp\!\big(-\tau(1-\mu)\big)}.$$

This bound suggests that **smaller $\mu$ leads to a higher confidence floor**. Pushing the most similar categories further apart ($\mu\downarrow$) raises the minimum confidence the model provides across all samples. Complete orthogonality aggressively lowers $\mu$, even for semantically related classes where high $\mu$ is meaningful, thereby systematically inflating confidence and worsening calibration.

**3. First-order Analysis of One-step Gradient: Proving O-TPT is Inherently More Overconfident**

Since TPT methods only perform one step, the authors analyze the compression of $\mu$ after one update. Using first-order Taylor expansion with step size $\eta$, prototype updates $t_i'=t_i-\eta\nabla_{t_i}$ result in similarity changes $\Delta s_{ij}\approx -\eta(t_j^\top\nabla_{t_i}+t_i^\top\nabla_{t_j})$. The O-TPT gradient is $\nabla^{O\text{-}TPT}_{t_i}=2\sum_{k\ne i}s_{ik}t_k$, while the Huber gradient replaces coefficients $s_{ik}$ with the capped $g_\delta(s_{ik})$. Substituting the dominant pair $(i,j)$ into the one-step update yields:

$$\mu'_{O\text{-}TPT}\approx(1-4\eta)\,\mu,\qquad \mu'_{Huber}\approx\begin{cases}(1-2\eta)\,\mu, & \mu\le\delta,\\ \mu-2\eta\delta, & \mu>\delta.\end{cases}$$

Comparison shows $\mu'_{O\text{-}TPT}<\mu'_{Huber}$ if and only if $4\mu>2\delta$, which holds whenever $\mu>\delta$. Thus, for high-similarity categories exceeding the boundary, **O-TPT always compresses $\mu$ more aggressively in one step**. Combined with Proposition 1, Corollary 1 yields $p^{O\text{-}TPT}_{max}>p^{Huber}_{max}$, meaning O-TPT elevates confidence more radically than SoC, explaining its inherent overconfidence and why multiple steps cause calibration to collapse faster.

### Loss & Training
The only addition is the SoC Huber regularization term with weight $\lambda$. Prompts are initialized as "a photo of a [CLASS]", optimized using AdamW with a learning rate of 0.005, batch size 64 (augmentations per image), and a **single step** update. The backbone is ViT-L/14 (ViT-B/16 for some experiments). Other settings follow O-TPT/C-TPT for fair comparison.

## Key Experimental Results

### Main Results
On 11 fine-grained classification datasets (ViT-L/14), SoC achieves the lowest average ECE while maintaining or slightly improving accuracy (ECE values in %, lower is better):

| Method | Mean Acc | Mean ECE | ECE vs O-TPT |
|------|----------|----------|-------------------|
| Zero-Shot | 71.1 | 5.1 | — |
| TPT (NeurIPS'22) | 72.0 | 14.9 | +7.2 |
| C-TPT (ICLR'24) | 72.1 | 10.0 | +2.3 |
| O-TPT (CVPR'25, Prev. SOTA) | 71.4 | 7.7 | 0 |
| **Ours (SoC)** | **72.3** | **5.4** | **−2.3** |

SoC achieves the best ECE on all but one dataset, matching the calibration of zero-shot (5.4 vs 5.1)—the gold standard for calibration—while slightly exceeding zero-shot accuracy. Notably, on EuroSAT, ECE drops from 17.7 (O-TPT) to 3.2 (−14.5), with a +4.7 accuracy gain.

On four ImageNet distribution shift variants (ViT-L/14), SoC matches O-TPT accuracy while further reducing ECE by 1.5, and by 5.8/4.2 relative to TPT/C-TPT, confirming that TPT and C-TPT sacrifice calibration for accuracy:

| Method | Mean Acc | Mean ECE |
|------|----------|----------|
| Zero-Shot | 70.0 | 4.4 |
| TPT | 72.9 | 14.2 |
| C-TPT | 72.0 | 12.6 |
| O-TPT | 71.3 | 9.9 |
| **Ours (SoC)** | **71.3** | **8.4** |

### Ablation Study

| Configuration / Setup | Key Metric | Description |
|------------|---------|------|
| 1-step → 2-step gradient | ECE degrades 23% vs O-TPT 39% | O-TPT calibration collapses twice as fast, validating first-order analysis. |
| ViT-B/16 Backbone | Acc 64.6 / ECE 4.3 | Outperforms O-TPT (4.8) on smaller backbones; ECE nears zero-shot (4.2). |
| CoOp 2-shot Init | Acc 75.2 / ECE 6.3 | Still outperforms O-TPT (72.9 / 7.4) under CoOp warm-start prompts. |
| CoOp 4-shot Init | Acc 78.0 / ECE 5.4 | Accuracy rises and ECE gap narrows but SoC remains superior. |
| 18 CLIP Prompt Templates | Lower ECE for nearly all | More robust to prompt initialization; suppresses O-TPT's overconfidence spikes. |

### Key Findings
- **The 1-step vs. 2-step comparison is the most telling**: The capping mechanism in SoC ensures that calibration barely degrades in a second step (+23%), whereas O-TPT degrades by 39% due to its aggressive $\mu$ compression.
- **Satellite imagery like EuroSAT benefits most**: Complete orthogonality excessively separates semantically related satellite categories, causing catastrophic overconfidence (ECE 17.7), which Huber capping rescues to 3.2.
- **Selective Classification**: At various confidence thresholds, SoC's accuracy for high-confidence predictions is consistently 5–10% higher than TPT/C-TPT/O-TPT and matches zero-shot levels, indicating its confidence scores are reliable for filtering uncertain predictions.

## Highlights & Insights
- **Turning a loss modification into a complete causal chain**: Huber capping $\rightarrow$ gentler $\mu$ compression (first-order analysis) $\rightarrow$ lower confidence floor (Proposition 1) $\rightarrow$ improved ECE. The alignment between theoretical propositions, corollaries, and multi-step experiments is the paper's strongest contribution.
- **Ingenious use of Huber Loss**: Originally a robust loss for outliers in regression, it is adapted here to "resist over-repulsion"—transferring the property of reducing gradients for large residuals to capping repulsion for high-similarity pairs. This logic is transferable to any regularization design where "hard constraints are too aggressive."
- **Zero-shot as a calibration ceiling**: Using "approximating zero-shot calibration" rather than "absolute minimum ECE" as the target is a pragmatic benchmark, acknowledging that adaptation inherently causes some drift.

## Limitations & Future Work
- **Validated only in few-step TPT settings**: The theoretical advantage is tied to the "one or two steps" premise; whether Huber capping holds under long-range optimization is not fully explored.
- **Additional hyperparameter $\delta$**: $\delta$ controls "how similar is too similar." The paper does not provide an adaptive scheme for selecting $\delta$ across datasets; sensitivity analysis is primarily in the appendix.
- **First-order analysis simplification**: The derivation focuses on the dominant pair $(i,j)$ for clarity. Behavior under full many-to-many coupling relies on Appendix B.
- **Potential improvements**: Making $\delta$ adaptive based on category similarity distributions or jointly calibrating it with temperature $\tau$ could further refine the "separate the distinct, preserve the close" objective.

## Related Work & Insights
- **vs. O-TPT (Prev. SOTA)**: Both modify prototype geometry, but O-TPT uses complete orthogonality (quadratic, uncapped repulsion), while SoC uses Huber capping. This work proves both theoretically and experimentally that complete orthogonality causes systematic overconfidence.
- **vs. C-TPT**: C-TPT uses "embedding dispersion" as a calibration proxy (pushing away from the centroid), which offers weaker control over prototype geometry. SoC directly acts on pairwise similarities and respects semantic proximity.
- **vs. Classical Calibration (Temperature Scaling / Mixup, etc.)**: Those are post-processing or training-time techniques requiring labels or full training. SoC is a **label-free, test-time** geometric regularizer specifically designed for VLM prompt tuning.

## Rating
- Novelty: ⭐⭐⭐⭐ Replacing complete orthogonality with Huber capping is a simple modification but backed by rigorous analysis of confidence bounds and first-order dynamics.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 datasets, 4 distribution shifts, two backbones, CoOp initialization, 18 prompt templates, multi-step analysis, and selective classification.
- Writing Quality: ⭐⭐⭐⭐⭐ Forms a closed loop from motivation to theory to experiments, with clear visualizations of the core problem.
- Value: ⭐⭐⭐⭐ Practical improvement for VLM calibration in safety-critical scenarios with almost zero additional costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Improving Calibration in Test-Time Prompt Tuning for Vision-Language Models via Data-Free Flatness-Aware Prompt Pretraining](improving_calibration_in_test-time_prompt_tuning_for_vision-language_models_via_.md)
- [\[ICLR 2026\] A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models](../../ICLR2026/multimodal_vlm/a-tpt_angular_diversity_calibration_properties_for_test-time_prompt_tuning_of_vi.md)
- [\[CVPR 2026\] Dual-Modality Anchor-Guided Filtering for Test-time Prompt Tuning](dual-modality_anchor-guided_filtering_for_test-time_prompt_tuning.md)
- [\[CVPR 2026\] Controllable Federated Prompt Learning at Test Time](controllable_federated_prompt_learning_at_test_time.md)
- [\[CVPR 2026\] Multi-modal Test-time Adaptation via Adaptive Probabilistic Gaussian Calibration](multi-modal_test-time_adaptation_via_adaptive_probabilistic_gaussian_calibration.md)

</div>

<!-- RELATED:END -->
