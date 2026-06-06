---
title: >-
  [Paper Note] Toward Understanding Adversarial Distillation: Why Robust Teachers Fail
description: >-
  [ICML 2026][Model Compression][Adversarial Distillation] This paper identifies a "robustly unlearnable set" that is stable across different methods in adversarial training data. Through feature learning theory of two-lay…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Adversarial Distillation"
  - "Robust Overfitting"
  - "Unlearnable Samples"
  - "Feature Learning Theory"
  - "Teacher Selection"
date: 2026-05-08
content_hash: 2318a127b425afcc
---

# Toward Understanding Adversarial Distillation: Why Robust Teachers Fail

**Conference**: ICML 2026  
**arXiv**: [2605.21999](https://arxiv.org/abs/2605.21999)  
**Code**: None  
**Area**: Model Compression / Adversarial Robustness / Knowledge Distillation  
**Keywords**: Adversarial Distillation, Robust Overfitting, Unlearnable Samples, Feature Learning Theory, Teacher Selection

## TL;DR
This paper identifies a "robustly unlearnable set" that is stable across different methods in adversarial training data. Through feature learning theory of two-layer networks, it proves that high-confidence supervision from a robust teacher on these samples forces the student to memorize pseudo-noise, triggering robust overfitting. Conversely, maintaining high entropy on these samples suppresses noise gradients. Based on this, it proposes a teacher selection criterion using the predictive entropy of unlearnable samples.

## Background & Motivation
**Background**: Adversarial Training (AT) via min-max optimization against $\ell_\infty$ perturbations is currently the most effective empirical defense. Adversarial Distillation (AD) improves upon this by matching student outputs to a robust teacher's soft labels, which is believed to mitigate robust overfitting and transfer robustness from large models to resource-constrained students.

**Limitations of Prior Work**: The success of AD is highly unstable—stronger teachers do not necessarily yield stronger students and may even exacerbate robust overfitting (where robust test accuracy peaks and then continuously declines). Early works like Zi et al. (2021) reported "robust saturation," and Lee & Chung (2026) attributed this failure to a "scarcity of transferable adversarial samples (TAS)," but these are symptoms rather than mechanistic explanations.

**Key Challenge**: The authors observe a counter-intuitive phenomenon: as long as it is not an overfitted teacher, AD using an independently "weaker" teacher often outperforms a stronger one. The issue is not whether the teacher is "robust," but where the teacher and student are "aligned." This implies a neglected factor: certain samples in the training set are naturally robustly unlearnable for a student of a specific capacity, and the teacher's behavior on these samples is decisive.

**Goal**: (1) Identify this critical subset at the data level; (2) Explain theoretically how it dominates robust overfitting; (3) Provide an a priori metric for selecting effective teachers in practice.

**Key Insight**: By taking the "prediction intersection" across 6 methods $\times$ 10 random seeds, the authors found a group of samples consistently misclassified by all models at their peak robust accuracy. This "robustly unlearnable set ($\mathcal{S}_U$)" size decreases monotonically with model capacity, and feature inversion on these samples yields collapsed pseudo-features. This suggests that unlearnability is an inherent property of the "data-architecture" pair, rather than noise within the data itself.

**Core Idea**: Robust overfitting is attributed to the mismatch between "teacher confidence on the student's representation blind spots" and "student capacity limits." Higher teacher confidence on $\mathcal{S}_U$ forces the student to use noise to fit those labels, leading to noise dominance.

## Method

### Overall Architecture
The paper progresses through three stages: (i) **Empirical stage**—Constructing a stable identification process for the unlearnable set and demonstrating its causal link to robust overfitting; (ii) **Theoretical stage**—Proving dichotomy theorems for both AT and AD on a patch-level feature learning model, linking "noise response amplification" to "teacher confidence on $\mathcal{S}_U$"; (iii) **Practical stage**—Using "predictive entropy of the teacher on unlearnable samples" as an a priori selection metric for large-scale validation.

### Key Designs

1.  **Stable Identification of Robustly Unlearnable Set $\mathcal{S}_U$**:
    - **Function**: Extracts a subset from the training set that is stable across methods and seeds to serve as a causal trigger for robust overfitting.
    - **Mechanism**: Train 60 models (6 paradigms including PGD-AT/TRADES/AD $\times$ 10 seeds). Define $\mathcal{S}_U$ as the set of samples consistently misclassified by all models at their **peak robust accuracy** epoch, and the learnable set $\mathcal{S}_L$ as those consistently correctly classified. This avoids explanations based on "bad luck" in a specific run. Results show $|\mathcal{S}_U|$ decreases monotonically with capacity (e.g., ~9000 for MobileNet-V2 vs. ~1500 for WRN-34-10), and feature inversion shows semantic collapse.
    - **Design Motivation**: Previous studies often split hard samples via loss/confidence thresholds, but hard samples drift during training. Taking the intersection at peak accuracy equivalent to measuring constraints at the "capacity limit," separating "unlearnable" from "hard" as an intrinsic property.

2.  **Feature Learning Theoretical Framework and Teacher Dichotomy**:
    - **Function**: Uses an analytical patch model to characterize AT and AD training dynamics, formulating robust overfitting as a binary choice of whether noise response is amplified.
    - **Mechanism**: Data consists of $P$ patches with two orthogonal robust features $\mathbf{u}=\mathbf{e}_1$ (learnable) and $\mathbf{v}=\mathbf{e}_d$ (unlearnable). $\mathcal{S}_L$ signal patches are $\alpha y\mathbf{u}$, $\mathcal{S}_U$ signal patches are $\alpha y\mathbf{v}$, and others are Gaussian noise $\mathcal{N}(\mathbf{0},\sigma_n^2(\mathbf{I}_d-\Pi_{\mathcal{F}}))$. The student is a two-layer cubic activation network $\phi(z)=(\max\{0,z\})^3$ with an **explicit constraint** $\langle \mathbf{w}_r,\mathbf{v}\rangle=0$ to simulate "structural blindness to $\mathbf{v}$." Under "unlearnable sparsity" $CN^{-1}\le p_{un}\le C^{-1}N^{-1}\log d$ and "signal-over-noise" conditions, it is proved that both AT and AD first learn $\mathbf{u}$. Subsequently, whether the noise response is pushed to $\tilde\Omega(1)$ depends entirely on whether residual gradients on unlearnable samples are continuously excited.
    - **Design Motivation**: Previous theories (lazy regime/linear models) cannot explain robustness. This work extends the feature learning route to AD. The innovation is encoding the student's capacity bottleneck as a hard orthogonal constraint, making the "asymmetry between teacher and student" derivable.

3.  **Good vs Bad Teacher and Provable Selection Criterion**:
    - **Function**: Characterizes the essential difference between "effective robust teachers" and "harmful robust teachers," implemented as a computable a priori screening metric.
    - **Mechanism**: On $\mathcal{S}_L$, both teachers satisfy large-margin alignment $y_i f_{W_T}(X_i)\ge\Gamma$. However, a Good Teacher is orthogonal to $\mathbf{v}$, maintaining uncertainty $y_i f_{W_G}(X_i)=0$ on $\mathcal{S}_U$. A Bad Teacher is highly confident $y_i f_{W_B}(X_i)\ge\Gamma$ on $\mathbf{v}$. In the saturation regime, residual gradients are modulated by the teacher's sigmoid factor $\sigma(-yf_{W_T}(X))$. A Good Teacher keeps this factor at $\Theta(1)$ on $\mathcal{S}_U$ but gradients cancel out; a Bad Teacher causes exponential decay of the factor but leaves biased residuals, pushing noise response to $\tilde\Omega(1)$. This leads to a practical criterion: **Use the predictive entropy of a candidate teacher on the training set $\mathcal{S}_U$ as an a priori screening metric**; higher entropy indicates a "Good Teacher."
    - **Design Motivation**: Traditional teacher selection relies on empirical heuristics or posterior metrics (like TAS) requiring student training. This criterion only depends on the teacher's output distribution on identified $\mathcal{S}_U$, allowing a priori selection via a single forward pass.

### Loss & Training
The AT objective is $\mathcal{L}_{AT}=\ell(yf_W(\tilde X))$, and the AD objective is $\mathcal{L}_{AD}=\sigma(yf_{W_T}(X))\ell(yf_W(\tilde X))+\sigma(-yf_{W_T}(X))\ell(-yf_W(\tilde X))$. Optimization uses full-batch gradient descent $W^{(t+1)}=W^{(t)}-\frac{\eta}{N}\sum\nabla_W\mathcal{L}$ for $T\ge\tilde\Omega(N/(\eta\sigma_0\sigma_n^3 d^{3/2}))$ steps to cover both signal learning and potential noise memorization. Theory holds with $1-\delta$ high probability.

## Key Experimental Results

### Main Results: Coupling of Unlearnable Set and Robust Overfitting
Statistics for $\mathcal{S}_U$ and $\mathcal{S}_L$ across architectures show that **robust unlearnability is a function of capacity**:

| Architecture | PGD-AT Unlearnable | TRADES Unlearnable | Intersection ($\mathcal{S}_U$) | Intersection ($\mathcal{S}_L$) |
| :--- | :--- | :--- | :--- | :--- |
| MobileNet-V2 | 13,898 | 12,261 | 8,979 | 19,385 |
| ResNet-18 | 8,360 | 10,217 | 5,217 | 21,899 |
| WRN-28-10 | 2,816 | 5,084 | 1,697 | 19,610 |
| WRN-34-10 | 2,608 | 4,511 | 1,559 | 16,397 |

The size of $\mathcal{S}_U$ decreases monotonically from ~9k to ~1.5k as capacity increases but remains non-zero.

### Ablation Study: Teacher Type vs. Student Overfitting
Comparison between two independently robust teachers in AD:

| Configuration | Student Peak robust acc | Student Final robust acc | Overfitting? | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| Standard PGD-AT | Moderate | Significant Decline | Yes | Residual gradients on $\mathcal{S}_U$ are unchecked |
| Self-Distill (Best) | High | Near Peak | No | Early teacher is uncertain on $\mathcal{S}_U$ $\to$ suppresses noise |
| Self-Distill (Last) | Moderate | Significant Decline | Yes | Overfitted teacher is confident on $\mathcal{S}_U$ $\to$ noise memorization |
| AD (Gowal teacher) | High | Maintained | No | High entropy on $\mathcal{S}_U$ $\to$ Good Teacher |
| AD (Chen teacher) | Moderate | Continuous Decline | Yes | Strong but low entropy on $\mathcal{S}_U$ $\to$ Bad Teacher |

### Key Findings
- **$\mathcal{S}_U$ drives overfitting, not AT itself**: Theorem 4.7 shows if $p_{un}=0$, noise responses remain suppressed and test error $\to 0$. In the sparse interval, $i\in\mathcal{S}_U$ inevitably pushes noise response to $\tilde\Omega(1)$.
- **Teacher strength is not sufficient**: Theorem 4.8 shows two equally robust teachers ($\Gamma$-margin) can lead to opposite AD outcomes based solely on their behavior on $\mathcal{S}_U$.
- **Entropy as an a priori metric**: Experiments verify high correlation between "teacher entropy on $\mathcal{S}_U$" and final robust accuracy.
- **Rationality of structural blindness**: The monotonic relationship between capacity and $|\mathcal{S}_U|$ explains why samples are unlearnable for small models but learnable for large ones.

## Highlights & Insights
- **Promoting "Hard Samples" to Theoretical Objects**: Using "cross-method prediction intersection" removes training stochasticity, defining unlearnability as a stable data-architecture property.
- **Ingenious Teacher Orthogonality Assumption**: Encoding "robust features invisible to the student" as hard orthogonal constraints allows analytical treatment of information asymmetry in AD.
- **Plug-and-play Entropy Metric**: Unlike TAS which requires training a student, calculating softmax entropy on $\mathcal{S}_U$ predicts AD success a priori, making it highly engineering-friendly.

## Limitations & Future Work
- The theory is built on a simplified two-layer cubic network with patch data; the transition to non-linear convolutional features in ResNet/WRN remains empirical.
- Identifying $\mathcal{S}_U$ requires training 60 models initially, posing a non-trivial front-end cost for new datasets.
- The teacher orthogonality assumption treats Good/Bad behaviors as binary (black and white), whereas real-world teachers exist on a continuum of confidence.

## Related Work & Insights
- **vs. Lee & Chung (2026, TAS)**: They used "TAS scarcity" as a signal for AD failure; this paper provides a mechanistic explanation showing TAS scarcity is equivalent to teacher over-confidence on $\mathcal{S}_U$.
- **vs. Li & Li (2025, AT feature learning)**: This work extends the feature learning framework to incorporate soft-label distillation and asymmetric teacher information.
- **vs. Goldblum et al. (2020, ARD)**: While ARD focused on transferring robustness, this paper solves the paradox of why stronger teachers can be worse.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<div class="related-papers" markdown="1">
<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Critique-Guided Distillation for Robust Reasoning via Refinement](critique-guided_distillation_for_robust_reasoning_via_refinement.md)
- [\[CVPR 2026\] Adversarial Concept Distillation for One-Step Diffusion Personalization](../../CVPR2026/model_compression/adversarial_concept_distillation_for_one-step_diffusion_personalization.md)
- [\[ICML 2026\] The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works](the_bridge-garden_dilemma_in_llm_distillation_why_mixing_hard_and_soft_labels_wo.md)
- [\[ICLR 2026\] Understanding Dataset Distillation via Spectral Filtering](../../ICLR2026/model_compression/understanding_dataset_distillation_via_spectral_filtering.md)
- [\[ICML 2026\] Detecting Fluent Optimization-Based Adversarial Prompts via Sequential Entropy Changes](detecting_fluent_optimization-based_adversarial_prompts_via_sequential_entropy_c.md)

</div>

<!-- RELATED:END -->
</div>

## Related Papers

- [\[ICML 2026\] Critique-Guided Distillation for Robust Reasoning via Refinement](critique-guided_distillation_for_robust_reasoning_via_refinement.md)
- [\[CVPR 2026\] Adversarial Concept Distillation for One-Step Diffusion Personalization](../../CVPR2026/model_compression/adversarial_concept_distillation_for_one-step_diffusion_personalization.md)
- [\[ICML 2026\] The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works](the_bridge-garden_dilemma_in_llm_distillation_why_mixing_hard_and_soft_labels_wo.md)
- [\[ACL 2025\] Who Taught You That? Tracing Teachers in Model Distillation](../../ACL2025/model_compression/who_taught_you_that_tracing_teachers_in_model_distillation.md)
- [\[ICLR 2026\] Understanding Dataset Distillation via Spectral Filtering](../../ICLR2026/model_compression/understanding_dataset_distillation_via_spectral_filtering.md)

</div>

<!-- RELATED:END -->
