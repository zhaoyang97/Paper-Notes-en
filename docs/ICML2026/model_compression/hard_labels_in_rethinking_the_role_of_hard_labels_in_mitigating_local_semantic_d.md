---
title: >-
  [Paper Note] Hard Labels In! Rethinking the Role of Hard Labels in Mitigating Local Semantic Drift
description: >-
  [ICML 2026][Model Compression][Dataset distillation] Addressing the exorbitant storage costs of "storing massive soft labels per image" in large-scale dataset distillation…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Dataset distillation"
  - "soft label compression"
  - "local semantic drift"
  - "hard label calibration"
  - "SRe2L"
date: 2026-05-08
content_hash: b4491da961bd270f
---

# Hard Labels In! Rethinking the Role of Hard Labels in Mitigating Local Semantic Drift

**Conference**: ICML 2026  
**arXiv**: [2512.15647](https://arxiv.org/abs/2512.15647)  
**Code**: https://github.com/Jiacheng8/HALD  
**Area**: Model Compression / Dataset Distillation  
**Keywords**: Dataset distillation, soft label compression, local semantic drift, hard label calibration, SRe2L  

## TL;DR
Addressing the exorbitant storage costs of "storing massive soft labels per image" in large-scale dataset distillation, this paper demonstrates that **Local View Semantic Drift (LVSD)** occurs when the number of soft labels per image $s$ is limited. It proposes HALD, a soft→hard→soft three-stage training paradigm that uses smoothed hard labels as semantic anchors to steer training back on track. On ImageNet-1K, it achieves 42.7% accuracy with 285M soft label storage, outperforming the SOTA LPLD by 9.0% while compressing soft label storage by 100x.

## Background & Motivation

**Background**: The de facto standard for dataset distillation (e.g., SRe2L, FKD, LPLD, FADRM) involves pre-generating and storing soft labels for every crop of every image using a teacher model, which are then replayed during training. Soft labels encode inter-class similarities and are much smoother than one-hot labels, making them an indispensable supervisory signal for large-scale ImageNet-level distillation.

**Limitations of Prior Work**: Soft label storage is a nightmare. On ImageNet-1K, while the distilled data itself only takes ~750 MB, FKD-style per-crop soft labels total **28.33 GB**, an order of magnitude larger than the image storage. The most direct mitigation is reducing the number of crops per image $s$, but this introduces a widely overlooked side effect: a crop might only cover a local region (e.g., a cat's face or just fur), causing the teacher's soft prediction to semantically drift toward "rabbit" or "carpet," which is inconsistent with the image-level ground truth.

**Key Challenge**: Soft labels provide fine-grained supervision but **drift with crop content**; hard labels have stable semantics but are **too coarse**. Methods like LPLD merely reduce crop counts to save storage without solving the drift problem, while FKD maintains high crop counts at the cost of explosive storage. Both occupy extreme ends of the spectrum.

**Goal**: (i) Formalize the "limited soft labels ⇒ train-test distribution mismatch" pipeline; (ii) develop a supervisory approach that recovers global semantic alignment even when $s$ is very small.

**Key Insight**: The authors revisit the overlooked **hard labels**. Hard labels are attached to the image-level ground-truth class and remain independent of whether a crop contains the main subject, acting as content-invariant "semantic anchors." Theoretically, the joint use of soft and hard labels can isolate and correct the "informational drift" component.

**Core Idea**: The authors insert a **smoothed hard label calibration period** (using CutMix + label smoothing) into the middle of soft-only training, creating a soft→hard→soft three-stage curriculum. Hard labels are used to correct the variance introduced by LVSD before returning to soft labels for fine-tuning.

## Method

### Overall Architecture

Input: A distilled synthetic dataset $\mathcal{C}$, a pre-generated finite soft label pool $\Omega_{\text{soft}}$ (capacity controlled by SLC), and a teacher model. Output: A student model $\hat\theta$ that generalizes well to the real test set.

Training is divided into three stages based on $n_{\text{total}}$ and $n_{\text{soft}}$ (the epochs required for soft label ERM convergence):

$T_A=\lfloor n_{\text{soft}}/2 \rfloor,\ T_B = n_{\text{total}}-n_{\text{soft}},\ T_C = n_{\text{soft}}-T_A$

Stage A uses $\Omega_{\text{soft}}$ soft labels for coarse pre-training. Stage B switches to CutMix + label-smoothed hard labels for semantic calibration, specifically suppressing the intra-crop variance introduced by LVSD. Stage C returns to soft labels for refinement to consolidate inter-class structures. If $n_{\text{total}} \le n_{\text{soft}}$, the method degrades to pure soft label training, ensuring HALD is compatible with existing SRe2L/LPLD pipelines.

The theoretical motivation is central. The authors decompose LVSD into two parts: for a fixed $\tilde x$, let $\bar p = \mathbb{E}[\tilde p(x^{(\text{crop})})]$ and $\Sigma=\text{Cov}[\tilde p(x^{(\text{crop})})]$. The supervisory error of aggregating $\hat p_s$ over $s$ crops decomposes into an "oracle irreducible term $\|\bar p - e_y\|_2^2$" and an "LVSD term $\text{Tr}(\Sigma)/s$." As long as $\Sigma \ne 0$ and $s$ is finite, the LVSD term remains strictly greater than zero. Furthermore, Theorem 3.5 provides an $\Omega(s^{-1/2})$ lower bound for bias in the training objective, and Theorem 3.6 provides an $\Omega(1/s)$ excess generalization loss lower bound between the ERM solution $\hat\theta_s$ and the oracle $\hat\theta_\star$. These bounds only vanish as $s\to\infty$, which is precisely the storage disaster the method avoids. Conclusion: **It is impossible to match the oracle under low SLC using soft labels alone**; additional supervision independent of crop content is required.

### Key Designs

1. **Formal Definition of LVSD and Cantelli Bound**:
    - **Function**: Uses a computable quantity to accurately describe the probability of a "student misclassifying a cat as a rabbit due to insufficient crops" event.
    - **Mechanism**: The authors define class-reversal events $\mathcal{E}_{s,c}=\{\hat p_s(c) \ge \hat p_s(y)\}$ and use Cantelli's inequality to provide a distribution-free upper bound $\Pr(\mathcal{E}_{s,c}) \le v_{s,c}/(v_{s,c}+(\bar p_y - \bar p_c)^2)$, where $v_{s,c}=(\Sigma_{yy}+\Sigma_{cc}-2\Sigma_{yc})/s$ decreases monotonically with $s$. This provides the first quantifiable, teacher/data-independent theoretical guarantee that "fewer soft labels lead to errors."
    - **Design Motivation**: Previous work relied on empirical observations that "fewer crops lead to performance drops" without an explanatory mechanism. This definition allows the mathematical calculation of how much drift hard labels can fix: hard labels force $v_{s,c}$ toward zero in the second stage, causing the reversal probability to converge monotonically with epochs.

2. **Soft→Hard→Soft Three-stage Curriculum**:
    - **Function**: Realigns the train-test distribution while maintaining the same storage budget as LPLD.
    - **Mechanism**: Stage A runs $\hat{\mathcal{L}}^{(t)}_{\text{soft}}(\theta)=\frac{1}{B}\sum_b \mathcal{L}(\tilde p_{j_b}, q_\theta(\cdot\mid x_{j_b}^{(\text{crop})}))$ using pool-sampled minibatches to pull the student to a soft label local optimum $\hat\theta_s^A$. Stage B initializes $\theta_0 := \hat\theta_s^A$. In each step, CutMix geometry $(x,x',\lambda,m)$ and smoothed labels $t_{\lambda,\alpha}(y,y')=(1-\lambda)\text{LS}_\alpha(y)+\lambda \text{LS}_\alpha(y')$ are resampled to minimize $\ell_{\text{cal}}(\theta;\omega)=\mathcal{L}(t_{\lambda,\alpha}, q_\theta(\cdot\mid \text{CM}_{\lambda,m}(x,x')))$. Since CutMix geometry is resampled every step, minibatches rarely repeat, making this equivalent to "infinitely diverse local views × global ground-truth supervision." Stage C then uses $\Omega_{\text{soft}}$ to refine and recover the coarsened inter-class structures.
    - **Design Motivation**: Pure hard labels lose inter-class similarity and regress to one-hot training; pure soft labels are drifted by LVSD. Inserting a hard label calibration phase in the middle suppresses variance without losing the teacher's fine-grained knowledge. The three-stage sequence is critical: Stage A provides a reasonable initialization before hard labels become meaningful; Stage C restores sensitivity to inter-class structures.

3. **Heavily-Smoothed "Pseudo-Hard Labels" + CutMix**:
    - **Function**: Softens the hard labels of synthetic data enough to prevent Stage B from pulling the student directly into one-hot mode.
    - **Mechanism**: Replaces strict one-hot labels with $\text{LS}_\alpha(y)=(1-\alpha)\delta_y + \alpha\,\mathbf{1}/C$ and uses CutMix to splice two images according to a mask $m$ and ratio $\lambda$, with the target $t_{\lambda,\alpha}(y,y')$ mixed synchronously. Since synthetic images have weaker semantics than real ones, the authors use a large $\alpha$ to form heavily-flattened labels for stable calibration.
    - **Design Motivation**: Experimental results (Fig. 3) show that soft-hard gradient similarity increases during training, implying that excessive coarsening causes the two objectives to cancel each other out. Introducing label smoothing + CutMix ensures Stage B provides perturbations around $\bar p$ rather than forcing $\delta_y$, satisfying the theoretical requirement of "variance reduction in the oracle neighborhood."

### Loss & Training

All three stages share the same per-crop loss $\mathcal{L}$ (cross-entropy or soft-target cross-entropy), switching only the label form and the mini-batch sampling space $\Omega_{\text{soft}}/\Omega_{\text{cal}}$. The learning rate follows the default schedule of the respective distillation methods. HALD is a plug-in: it can be applied directly to SRe2L, RDED, FADRM, or LPLD.

## Key Experimental Results

### Main Results

ImageNet-1K results: HALD significantly leads LPLD under fixed SLI (soft labels per image).

| Dataset | Setup | Prev. SOTA | HALD | Gain |
|--------|------|----------|------|---------|
| ImageNet-1K | SLI=10, IPC=10 | LPLD 33.7% | **42.7%** | +9.0 |
| Tiny-ImageNet | SLI=2, IPC=10 | FADRM 17.4% | **22.8%** | +5.4 |
| Tiny-ImageNet | SLI=1, IPC=10 | FADRM 10.1% | **18.6%** | +8.5 |
| Tiny-ImageNet | SLI=2, IPC=50 | FADRM 36.0% | **38.2%** | +2.2 |
| Tiny-ImageNet | SLI=1, IPC=50 | FADRM 27.8% | **30.7%** | +2.9 |

Key Observation: The smaller the SLI (the more storage saved), the larger the lead over previous SOTA—consistent with the theoretical prediction that "LVSD dominates error when $s$ is small."

### Ablation Study

| Configuration | Tiny-IN, SLI=1, IPC=10 | Description |
|------|----------------------|------|
| Soft only (LPLD) | 8.2% | Pure soft, suffers from LVSD |
| Soft → Hard (No Stage C) | Intermediate | Lacks final refinement |
| **HALD (Soft → Hard → Soft)** | **18.6%** | Full three stages |
| HALD w/o label smoothing | Drop | Strong one-hot pulls student away from oracle neighborhood |
| HALD w/o CutMix | Drop | Insufficient crop diversity |

### Key Findings

- **Pushing the Storage-Accuracy Boundary**: HALD achieves an accuracy with a 285M soft label budget that LPLD cannot reach even at 28 GB, lowering the deployment threshold for dataset distillation by two orders of magnitude.
- **Stage B is Not Merely "Extra Training"**: The train/test loss landscape (Fig. 2) shows that under pure soft labels, the test landscape is severely misaligned with the training landscape (typical overfitting to drifted targets). With hard label calibration, the two realign.
- **Gradient Alignment Increases During Training** (Fig. 3): The cosine similarity between soft and hard loss gradients increases significantly by the end of Stage A, indicating that the representations learned by the student can simultaneously satisfy both supervisions. The theoretical assumption that "hard labels do not introduce gradient conflict" is empirically validated.

## Highlights & Insights

- **Returning Hard Labels to the Center**: While recent distillation literature often dismisses hard labels as relics of the past, this paper fuses the storage and drift perspectives, making the "zero storage + content independence" of hard labels a core resource.
- **Tight Alignment Between Theory and Curriculum**: The $\Omega(1/s)$ excess loss lower bound in Theorem 3.6 justifies inserting calibration (Stage B) right after ERM convergence—precisely where the bound arises—without disrupting previous progress.
- **Transferable Trick**: The soft→hard→soft curriculum applies to any scenario where label sources are imperfect but cheap anchor labels exist, such as semi-supervised distillation, multi-teacher fusion, or GT injection in self-distillation.
- **Formalizing "Crop Drift"**: The LVSD definition is valuable in its own right—the $\text{Tr}(\Sigma)/s$ decomposition can be used whenever discussing the stability of "cropping data augmentation + soft targets."

## Limitations & Future Work

- The theory is built on the teacher's soft prediction covariance $\Sigma$ and IID crops; validity with strongly correlated augmentations (e.g., RandAug chains) requires further empirical validation.
- The $T_A/T_B/T_C$ division depends on $n_{\text{soft}}$ (the "empirical convergence point"), which might be costly to estimate for ultra-large models.
- Stage B uses CutMix + label smoothing, which goes beyond "strict hard labels." If distillation targets are for tasks like detection or segmentation that cannot easily use mixup, the calibration signal needs redesigning.
- The drift structure when teacher and student architectures differ significantly (e.g., ViT teacher → CNN student) has not been explored.

## Related Work & Insights

- **vs LPLD (Xiao & He, 2024)**: LPLD uses limited soft labels to save storage but ignores LVSD; HALD calibrates drift with hard labels within the same budget, achieving +9.0% on ImageNet-1K.
- **vs FKD / FerKD**: FKD maintains accuracy by storing soft labels for every crop but suffers from storage explosion; HALD solves accuracy from the low-storage side.
- **vs GIFT (Shang et al., 2025a)**: GIFT integrates hard information into soft targets (modifying labels), while HALD switches in the time dimension (soft then hard then soft), avoiding the need to modify existing soft label pools.
- **vs Label Smoothing Literature**: This paper reveals that label smoothing acts as a variance calibration signal for LVSD in distillation, rather than just a regularization signal.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formulates the "crop count → semantic drift → train-test misalignment" pipeline and designs a matching three-stage curriculum.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across Tiny-IN/IN-1K, multiple IPCs, and multiple SLIs, compared against four types of SOTA (SRe2L, RDED, FADRM, LPLD).
- Writing Quality: ⭐⭐⭐⭐ Clear correspondence between theory and algorithms; notation is somewhat dense, but proofs are complete.
- Value: ⭐⭐⭐⭐⭐ Reduces the storage bottleneck of dataset distillation by 100x, with significant impact on practical industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works](the_bridge-garden_dilemma_in_llm_distillation_why_mixing_hard_and_soft_labels_wo.md)
- [\[ICML 2026\] DSL-Topic: Improving Topic Modeling by Distilling Soft Labels from Language Models](dsl-topic_improving_topic_modeling_by_distilling_soft_labelsfrom_language_models.md)
- [\[ICCV 2025\] Heavy Labels Out! Dataset Distillation with Label Space Lightening](../../ICCV2025/model_compression/heavy_labels_out_dataset_distillation_with_label_space_lightening.md)
- [\[ICML 2026\] DIVER: Diving Deeper into Distilled Data via Expressive Semantic Recovery](diverdiving_deeper_into_distilled_data_via_expressive_semantic_recovery.md)
- [\[ICLR 2026\] ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning](../../ICLR2026/model_compression/acpbench_hard_unrestrained_reasoning_about_action_change_and_planning.md)

</div>

<!-- RELATED:END -->
