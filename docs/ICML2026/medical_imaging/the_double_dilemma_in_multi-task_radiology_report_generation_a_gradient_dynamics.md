---
title: >-
  [Paper Note] CAME-Grad: The Double Dilemma in Multi-Task Radiology Report Generation — A Gradient Dynamics Analysis and Solution
description: >-
  [ICML 2026][Medical Imaging][Radiology Report Generation] This paper uses an SDE framework to analyze the "double dilemma" of gradient conflict between "report generation vs. clinical constraints" in Radiology Report Gen…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Radiology Report Generation"
  - "Multi-task Learning"
  - "Gradient Conflict"
  - "SDE Analysis"
  - "Plug-and-play Optimizer"
date: 2026-05-08
content_hash: b8123869bffaf257
---

# CAME-Grad: The Double Dilemma in Multi-Task Radiology Report Generation — A Gradient Dynamics Analysis and Solution

**Conference**: ICML 2026  
**arXiv**: [2605.22635](https://arxiv.org/abs/2605.22635)  
**Code**: https://github.com/vpsg-research/CAME-Grad  
**Area**: Medical Imaging / Radiology Report Generation / Multi-task Optimization  
**Keywords**: Radiology Report Generation, Multi-task Learning, Gradient Conflict, SDE Analysis, Plug-and-play Optimizer

## TL;DR
This paper uses an SDE framework to analyze the "double dilemma" of gradient conflict between "report generation vs. clinical constraints" in Radiology Report Generation (RRG) multi-task learning—specifically, drift term deviation from Pareto optimality and diffusion term decay preventing escape from local optima. The authors propose the CAME-Grad optimizer (direction rectification + energy injection + adaptive fusion) as a linear scaling plug-and-play alternative, achieving an average clinical efficacy gain of +2.3% / +1.9% across 8 RRG methods on MIMIC-CXR / IU X-Ray datasets.

## Background & Motivation

**Background**: RRG (automated generation of radiology reports) has evolved from single-task learning (NLL supervision only) to multi-task learning—simultaneously learning report generation $\mathcal{L}_{rg}$ (language modeling requiring a smooth semantic manifold) and auxiliary tasks (disease classification / image-text alignment / retrieval augmentation requiring discrete rigid structures). While architectural innovations abound, the **optimization side remains largely stuck with static linear weighting** $\mathcal{L}_{joint} = \sum_i \omega_i \mathcal{L}_i$.

**Limitations of Prior Work**: (1) The need for smooth generation and "hard" clinical constraints creates an inherent conflict—strengthening clinical supervision decreases report quality, and vice versa. (2) Linear scaling ignores gradient dynamics, leading tasks to "pull" against each other after weighted aggregation. (3) Existing multi-task optimization methods (CAGrad / PCGrad / MGDA) primarily focus on correcting direction but overlook insufficient exploration caused by magnitude collapse—if the magnitude is too low, the model fails to reach flat minima.

**Key Challenge**: From the perspective of the SDE framework, SGD optimization $d\Theta_t = -\bm g_{joint}(\Theta_t) dt + \sqrt{\eta \Sigma}d\bm W_t$ is composed of a drift term (first moment, determining convergence direction) and a diffusion term (second-order covariance, providing exploration to escape local optima). Gradient conflicts simultaneously lead to **drift deviation** (direction deviating from the Pareto front) and **diffusion decay** (energy collapse, failing to escape sharp minima). Correcting direction or increasing amplitude in isolation is insufficient; both must be addressed.

**Goal**: (1) Uncover the root causes of linear scaling failure in RRG through the SDE lens; (2) design a unified optimizer for simultaneous direction correction and amplitude enhancement; (3) implement this as a backbone-agnostic plug-and-play solution.

**Key Insight**: Observations reveal that the proportion of negative gradient inner products between RRG tasks is as high as 53.8% (Figure 3), confirming the "inherent conflict" hypothesis. Since both direction and amplitude are compromised, the approach integrates direction rectification (CAGrad-like) with energy injection (gradient magnitude amplification), followed by adaptive fusion to prevent total deviation from the original direction and loss of task-specific inductive bias.

**Core Idea**: A three-stage optimizer—Conflict-Averse Direction Rectification (maximizing worst-case improvement within a trust region) → Magnitude-Enhanced Energy Injection (amplifying gradient magnitude to escape sharp minima) → Adaptive Gradient Fusion (soft fusion between the rectified direction and the original direction to maintain task priors).

## Method

### Overall Architecture

At each step, CAME-Grad performs the following:
1. Calculates individual task gradients $\bm g_i$, the weighted joint gradient $\bm g_{joint}$, and the mean gradient $\bm \mu$.
2. **Stage 1**: Solves the dual problem to obtain the rectified direction $\bm u^*_{rect}$ (maximizing the worst-case improvement within a trust region).
3. **Stage 2**: Scales the amplitude of $\bm u^*_{rect}$ to $\kappa \|\bm g_{joint}\|$ to obtain the energy-injected gradient $\bm u_{en}$.
4. **Stage 3**: Computes $\bm g_{final} = (1-\nu)\bm u_{en} + \nu(\kappa \bm g_{joint})$, fusing it with the original direction.
5. Updates parameters via SGD: $\Theta \leftarrow \Theta - \eta \bm g_{final}$.

### Key Designs

1. **Conflict-Averse Direction Rectification (Stage 1)**:
    - **Function**: Find a direction that maximizes the "worst-case improvement" among all task gradients to ensure geometric validity.
    - **Mechanism**: Formulated as a trust region problem $\max_{\bm u} \min_i \bm g_i^\top \bm u$ s.t. $\|\bm u - \bm \mu\| \leq \rho \|\bm \mu\|$ (centered at the mean $\bm \mu$ with radius $\rho \|\bm \mu\|$); converted to the dual problem $\min_{\bm \alpha \in \Delta^{K+1}} \mathcal{F}(\bm \alpha) = \bm g_{\bm \alpha}^\top \bm \mu + \sqrt{\xi}\|\bm g_{\bm \alpha}\|$ where $\xi = \rho^2 \|\bm \mu\|^2$. The closed-form solution is $\bm u^*_{rect} = \bm \mu + \frac{\sqrt{\xi}}{\|\bm g_{\bm \alpha^*}\|} \bm g_{\bm \alpha^*}$.
    - **Design Motivation**: CAGrad-style but with an added trust region constraint to ensure convergence stability; the dual problem is solvable in low dimensions on the simplex (near-zero GPU overhead), avoiding high-dimensional primal problems while remaining Pareto-compatible.

2. **Magnitude-Enhanced Energy Injection (Stage 2)**:
    - **Function**: Compensate for diffusion decay, providing the model with sufficient exploration energy to escape sharp minima.
    - **Mechanism**: The target magnitude is set as $\tau_{mag} = \kappa \|\bm g_{joint}\|$ (where $\kappa > 1$ is the gain); the enhanced gradient is $\bm u_{en} = \bm u^*_{rect} \cdot \tau_{mag} / (\|\bm u^*_{rect}\| + \epsilon)$.
    - **Design Motivation**: Pure direction correction (like CAGrad) causes amplitude collapse during conflict, trapping the model in sharp minima. Amplifying the amplitude effectively recovers the diffusion coefficient in the SDE to normal levels, restoring the implicit regularization effect of SGD (favoring flat minima); $\kappa$ controls the degree of energy surplus.

3. **Adaptive Gradient Fusion (Stage 3)**:
    - **Function**: Perform soft fusion between the rectified direction and the original $\bm g_{joint}$ to avoid total deviation from the original direction and loss of task-specific inductive bias.
    - **Mechanism**: $\bm g_{final} = (1-\nu) \bm u_{en} + \nu (\kappa \bm g_{joint})$, where $\nu \in [0,1]$ adjusts the weight of task-specific priors.
    - **Design Motivation**: Purely rectified directions might lose task structure information (e.g., specific pre-trained knowledge); fusion preserves task bias. $\nu$ allows users to balance "full Pareto optimization" and "task prior preservation."

## Key Experimental Results

### Main Results on MIMIC-CXR (8 RRG methods + CAME-Grad)

| RRG Method | Baseline CE↑ | + CAME-Grad CE↑ | Gain |
|---------|---------|-------------|---|
| R2Gen | 35.7 | 38.4 | +2.7 |
| Multi-task R2Gen + DC | 38.1 | 40.6 | +2.5 |
| WCL | 39.0 | 41.5 | +2.5 |
| METransformer | 41.2 | 43.5 | +2.3 |
| KGAE | 40.8 | 42.9 | +2.1 |
| RGRG | 42.5 | 44.8 | +2.3 |
| PromptMRG | 43.7 | 46.0 | +2.3 |
| RECAP | 44.6 | 46.9 | +2.3 |
| **Average Gain** | – | – | **+2.3** |

All 8 methods benefited consistently (each +2.1 to +2.7 CE), proving plug-and-play versatility.

### IU X-Ray average gain of +1.9% (similar distribution)

### Ablation Study (MIMIC-CXR, PromptMRG Baseline)

| Configuration | CE | Gain |
|------|----|---|
| Linear Scaling Only (Baseline) | 43.7 | – |
| + Direction Rectification | 44.6 | +0.9 |
| + Magnitude Injection | 45.4 | +0.8 |
| + Adaptive Fusion (Complete) | **46.0** | +0.6 |

Each stage contributes +0.6 to +0.9 CE, indicating that all three are indispensable.

### Gradient Conflict Quantification (Figure 3)

Across multiple epochs, the inner product between $\bm g_0$ (generation) and $\bm g_k$ (clinical) was measured—**negative 53.8% of the time**, validating the "inherent conflict" hypothesis.

### Key Findings
- **Simultaneous treatment of drift + diffusion is effective**: Correcting direction (CAGrad) or increasing amplitude alone is suboptimal. The proposed method yields +2.3 CE, whereas direction only yields +0.9 and amplitude only yields +0.8—the synergy (+2.3) exceeds the sum (+1.7).
- **Plug-and-play universality**: 8 RRG methods with different architectures benefited consistently, suggesting the issue is optimization-level rather than architectural.
- **53.8% negative correlation time**: Quantitative evidence of the prevalence of conflict provides empirical justification for modifying RRG optimizers.
- **Consistency between MIMIC-CXR and IU X-Ray**: Performance gains across large and small datasets (+2.3 / +1.9) demonstrate stability.

## Highlights & Insights
- **SDE framework formalizes the "double dilemma"**: Whereas previous works treated multi-task conflict as either a "direction problem" or an "amplitude problem," this work uses SDE to derive drift deviation and diffusion decay as two sides of the same conflict that must be treated together. This framework can be extended to all multi-task/multi-objective RL/RLHF scenarios.
- **Trust region + closed-form solution + GPU-friendly**: The dual method of CAGrad in low-dimensional simplex space avoids $\mathcal{O}(d)$ high-dimensional operations. Adding trust region constraints ensures both stability and GPU efficiency.
- **Plug-and-play is highly practical**: No architecture changes or retraining is required—simply swap the optimizer. This is friendly for upgrading existing RRG systems.
- **Medical diagnostic significance of gradient conflict**: The conflict between generation and clinical constraints maps to the clinical reality of "fluent natural language vs. rare disease details." Resolving this conflict algorithmically reconciles the two clinical requirements of medical narratives.

## Limitations & Future Work
- $\rho, \kappa, \nu$ remain manual hyperparameters; adaptive scheduling based on current conflict intensity would be superior.
- Validated only on RRG; transferability to other multi-task medical applications (e.g., joint CT-report + segmentation) is untested.
- The center of the trust region is the mean gradient $\bm \mu$, which might be biased for highly imbalanced task numbers (e.g., 1 generation + 10 clinical tasks); weighted means could be considered.
- The SDE analysis is a continuous-time approximation; specific biases under discrete updates were not quantified.
- Training time overhead reports are sparse; while the dual solution is GPU-friendly, there is still an extra forward pass; costs on ultra-large models need assessment.

## Related Work & Insights
- **vs. CAGrad / PCGrad / MGDA**: Those methods only correct direction without addressing magnitude; CAME-Grad treats both.
- **vs. GradNorm / Uncertainty Weighting**: Those methods only adjust magnitude without addressing direction; similarly limited.
- **vs. Task Prioritization methods (Liu 2021 / Jeong 2024)**: Those assume static priorities; CAME is dynamically adaptive.
- **Insight**: Any training involving "multi-objective and structurally conflicting goals" (RLHF + KL, safety RL, multi-modal alignment) can be diagnosed using the SDE framework; the CAME-Grad template is directly transferable.

## Rating
- Novelty: ⭐⭐⭐⭐ The SDE "double dilemma" framing is a new perspective, and treating direction and amplitude together is systematically novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 2 datasets × 8 RRG methods × three-stage ablation × gradient conflict visualization yields consistent conclusions.
- Writing Quality: ⭐⭐⭐⭐⭐ The chain from SDE derivation → algorithm → experiments is clear; Figure 1 provides an intuitive explanation of the "double dilemma."
- Value: ⭐⭐⭐⭐ RRG is a high-value clinical NLP task; optimizer-level improvements benefit all RRG work; the theoretical framework is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](../../CVPR2026/medical_imaging/cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](../../ACL2026/medical_imaging/march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)
- [\[ACL 2026\] RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction](../../ACL2026/medical_imaging/ra-rrg_multimodal_retrieval-augmented_radiology_report_generation_with_key_phras.md)
- [\[CVPR 2026\] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](../../CVPR2026/medical_imaging/orapo_oracle-educated_reinforcement_learning_for_data-efficient_and_factual_radi.md)
- [\[ICML 2026\] SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment](synermedgen_synergizing_medical_multimodal_understanding_with_generation_via_tas.md)

</div>

<!-- RELATED:END -->
