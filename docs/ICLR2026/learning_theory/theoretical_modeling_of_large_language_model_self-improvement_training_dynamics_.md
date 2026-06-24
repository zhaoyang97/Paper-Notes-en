---
title: >-
  [Paper Note] Theoretical Modeling of Large Language Model Self-Improvement Training Dynamics Through Solver-Verifier Gap
description: >-
  [ICLR 2026][Learning Theory][Self-improvement] This paper models the training process of LLM "self-improvement" as a set of coupled differential equations inspired by physical potential energy. The gap between "solver capability" and "verifier capability" drives the exponential convergence of both throughout training iterations. This model can fit real training curves, quantify the upper bound of self-improvement, and further analyze the optimal allocation of external data in…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "LLM Self-Improvement"
  - "Training Dynamics"
  - "Self-improvement"
  - "Solver-verifier gap"
  - "Uncertainty"
  - "Cross-improvement"
date: 2026-05-08
content_hash: 3ec60b634f10ea77
---

# Theoretical Modeling of Large Language Model Self-Improvement Training Dynamics Through Solver-Verifier Gap

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Hh7x3c0cZl](https://openreview.net/forum?id=Hh7x3c0cZl)  
**Code**: To be confirmed  
**Area**: Learning Theory / LLM Self-Improvement / Training Dynamics  
**Keywords**: Self-improvement, Solver-verifier gap, Training dynamics, Uncertainty, Cross-improvement

## TL;DR
This paper models the training process of LLM "self-improvement" as a set of coupled differential equations inspired by physical potential energy. The gap between "solver capability" and "verifier capability" drives the exponential convergence of both throughout training iterations. This model can fit real training curves, quantify the upper bound of self-improvement, and further analyze the optimal allocation of external data in "cross-improvement" scenarios.

## Background & Motivation
**Background**: Self-improvement is a crucial method for enhancing LLM performance without relying on external data—using a pre-trained model to generate data and then fine-tuning itself on that data. Works like STaR, Self-Rewarding, and Constitutional AI have empirically demonstrated its effectiveness in tasks such as reasoning, alignment, and planning.

**Limitations of Prior Work**: While it is known that self-improvement "works," it remains unclear "why it improves and where it stops." Questions regarding the trajectory of model capability evolution during self-improvement, the existence of a predictable limit, and the factors determining that limit remain largely unanswered at the level of training dynamics. Without a theoretical model, practitioners often resort to trial-and-error training without knowing when to stop or if further breakthroughs are possible.

**Key Challenge**: The phenomenon of self-improvement is somewhat counter-intuitive—how can a model's performance increase by training on its own generated data without any net increase in information? Prior research (Song et al. 2025, Huang et al. 2025) provided a key clue: the "generation-validation gap" (or solver-verifier gap). A model's ability to **judge** the quality of an answer is often stronger than its ability to **directly generate** a high-quality answer. Self-improvement distills this "stronger verification" information back into the generation side. However, previous works only characterized this gap statically; they did not incorporate it into a dynamic model capable of describing the entire training trajectory.

**Goal**: To establish a theoretical framework characterizing the **full dynamics** of self-improvement, which can (i) explain how solver/verifier capabilities evolve over training iterations, (ii) quantify the upper bound of self-improvement and predict it using early-stage data, and (iii) generalize to "cross-improvement" using small amounts of external data to determine optimal data allocation.

**Key Insight**: The authors treat solver capability $U_s(t)$ and verifier capability $U_v(t)$ as two time-varying "energies." Borrowing the physical analogy of "potential energy driving a system toward equilibrium," they posit that a larger capability gap creates a stronger driving force, causing both capabilities to move toward each other and converge.

**Core Idea**: By using a set of coupled differential equations driven by the "capability gap," the self-improvement training dynamics are formulated as analytical, fittable, and extrapolatable exponential convergence laws.

## Method

### Overall Architecture
The objective is to determine whether a concise mathematical model can predict the capability curve and final limit of a given self-improvement process. The approach consists of three layers: first, defining "solver capability" and "verifier capability" computationally using **uncertainty**; second, assuming their evolution is driven by a "capability gap potential" to write a set of coupled differential equations; and finally, solving these equations to derive exponential convergence laws and extending this mechanism to cross-improvement scenarios involving external data.

A single loop of self-improvement involves: the model generating responses to prompts (solver output) → using Best-of-N to let the model select the optimal response (verifier output) → fine-tuning the model on the high-quality responses selected by the verifier. The paper does not modify this workflow but **abstracts it into dynamics**: training pulls the solver toward the verifier's level; the larger the gap, the faster the pull, with training saturating when the gap reaches a non-zero limit.

### Key Designs

**1. Defining Solver and Best-of-N Verifier Capabilities via Uncertainty**

To model dynamics, the abstract "capability" must be converted into a computable, differentiable quantity. Following Huang et al. (2025), the paper uses the **negative log-likelihood (uncertainty)** of the model's response to measure capability. For a response $\hat{y}$ and model $f$, capability is defined as $U_f(\hat{y}) = -\log \pi_f(\hat{y}\,|\,x)$, where lower uncertainty represents higher capability. Two types of capabilities are distinguished: the solver samples a response $\hat{y}_i \sim \pi_f(\cdot|x_i)$ for each prompt, with the solver uncertainty $U_s(t)$ being the average uncertainty. The verifier samples $N$ candidates, scores each one $s(\hat{y}_{i,j})\in[0,1]$, and selects the optimal response using a hybrid Best-of-N criterion:

$$\hat{y}^{\text{BoN}}_i = \arg\min_{\{\hat{y}_{i,j}:\, s(\hat{y}_{i,j})\ge \sigma\}} \frac{1}{L(\hat{y}_{i,j})} U_f(\hat{y}_{i,j}|x_i),$$

This discards low-score candidates via threshold $\sigma$ and selects the one with the lowest uncertainty among the remainder, penalized by length $1/L(\hat{y})$. The average uncertainty of these selected responses is the verifier uncertainty $U_v(t)$. The elegance of this BoN design is that it injects the model's discriminative power into the selected response. Since $U_v$ typically outperforms $U_s$, it serves as the information source for self-improvement.

**2. Characterizing Training Dynamics with Potential-Driven Coupled Differential Equations**

Given $U_s(t)$ and $U_v(t)$, the core question is how they evolve over iterations $t$. Since derivation from first principles of neural networks is intractable, the authors use **phenomenological modeling**. Defining the capability gap as $G(t) \triangleq U_s(t) - U_v(t)$, they assume the change in both capabilities is driven by a gap potential $E(t)=f(G(t))$ (where $f$ is monotonically increasing and $f(0)=0$):

$$\frac{dU_s(t)}{dt} = -\alpha E(t), \qquad \frac{dU_v(t)}{dt} = -\beta E(t),$$

where $\alpha>\beta\ge 0$, indicating the solver improves faster than the verifier—matching the intuition that training pulls the weaker solver toward the stronger verifier. Subtracting the equations yields $dG/dt = -(\alpha-\beta)E(t)$. By linearizing the potential $E(t)\approx kG(t)-b$ (supported by empirical data showing a strong linear correlation with $R^2 \approx 0.966$), the training process is reduced to a low-dimensional dynamical system.

**3. Exponential Convergence Laws and Quantifying Capability Limits**

The linearized equations yield a closed-form solution (Proposition 3.1): the gap, solver, and verifier all converge exponentially at the same rate to their respective limits:

$$G(t)\approx \delta e^{-k(\alpha-\beta)t} + G_\infty,\quad U_s(t)\approx \alpha' e^{-k(\alpha-\beta)t} + U_{s,\infty},\quad U_v(t)\approx \beta' e^{-k(\alpha-\beta)t} + U_{v,\infty},$$

where $\delta = U_{s,0}-U_{v,0}-b/k$, $\alpha'=\alpha\delta/(\alpha-\beta)$, $\beta'=\beta\delta/(\alpha-\beta)$, and the convergence limit $G_\infty=b/k$. Key conclusions include: (i) the partial derivative of the solver's final uncertainty with respect to the initial gap $G_0$ is a negative constant $\partial U_{s,\infty}/\partial G_0 = -\beta/(\alpha-\beta)$, theoretically confirming that **a larger initial gap leads to a stronger final solver**; (ii) the number of iterations required for a tolerance $\epsilon$ is $t>\ln(\delta/\epsilon)/[k(\alpha-\beta)]$; and (iii) the gap converges to a non-zero $G_\infty$, implying a ceiling for pure self-improvement.

**4. Extension to Cross-Improvement: External Data and Timing Independence**

To overcome the ceiling of self-improvement, the authors introduce **cross-improvement** using external data to enhance the verifier. A key insight is that external data only influences the framework through the verification step. Given $M$ external data points from a stronger model and a distribution ratio $\eta_t$ ($\sum_t \eta_t=1$) at iteration $t$, the verifier capability is modeled as $U_v^c(t)=(1+\gamma\eta_t)^{-1}U_v(t-1)$, where $\gamma>0$. Solving the modified dynamics (Proposition 5.1) yields a counter-intuitive but practical conclusion: **the final solver capability depends only on the total volume of external data ($\sum_t \eta_t$) and is independent of the specific allocation across iterations.**

### Loss & Training
The training objective is the verifier uncertainty: $\mathcal{L}_t(f)\triangleq U_v(t)=-\frac{1}{n}\sum_i \log \pi_f(\hat{y}^{\text{BoN}}_i(t)|x_i)$. Minimizing this is equivalent to increasing the likelihood of generating high-quality (BoN) responses. LoRA is used for efficient supervised fine-tuning. Evaluation methods include TrueFalse (TF) and Quality Evaluation (QE).

## Key Experimental Results

### Main Results: Verifying Exponential Laws
The exponential model was fitted to uncertainty curves of Phi (Phi-4-mini / 3.5-mini / 3-mini) and Llama (Llama-3.2-3B / 3.1-8B) over 10 iterations of self-improvement on Math and GSM8k.

| Setting (Phi-4-mini, train split) | Solver $R^2$ | Verifier $R^2$ | Gap $R^2$ |
|--------|------|------|------|
| Math (QE) | 0.998 | 0.981 | 0.981 |
| GSM8k (QE) | 0.996 | 0.966 | 0.948 |
| Math (TF) | 0.998 | 0.979 | 0.984 |
| GSM8k (TF) | 0.998 | 0.980 | 0.989 |

In all settings, $R^2 > 0.9$, validating the exponential convergence law from Proposition 3.1. Experiments also observed that accuracy and uncertainty for both solver and verifier improved synchronously, while the gap $G(t)$ gradually shrunk.

### Ablation Study: Impact of Training Phase (EvoLM, Llama-2 1B)
| Phase | Solver Initial Scale $\alpha'$ | Decay Rate | Solver Limit $U_{s,\infty}$ |
|------|------|------|------|
| Base-SFT | 146.3 | 0.244 | 45.6 (R²=0.999) |
| Mid-train | 83.1 | 0.182 | 20.3 (R²=1.000) |
| Post-train | 16.0 | 0.074 | 13.9 (R²=0.999) |

Base models show the highest plasticity (largest initial gap and fastest decay). Post-trained models exhibit "dynamic saturation," indicating that the gap has already been minimized, leaving little room for self-improvement.

### Data Allocation in Cross-Improvement (External Data from DeepSeek-V3)
| Strategy | Phi-4-mini Math(%) | Phi-4-mini GSM8k(%) | Llama-3.2-3B Math(%) |
|------|------|------|------|
| Initial | 30.31 | 73.42 | 36.02 |
| Baseline (Self-improvement) | 45.08 | 88.53 | 49.16 |
| Early (All in round 1) | 46.33 | 88.54 | 53.48 |
| Uniform (Distributed) | 46.56 | 88.44 | 52.74 |
| Late (All in round 8) | 45.83 | 88.90 | 51.31 |

Differences between allocation strategies were minimal (Max-Min of 0.73 on Phi-4-mini Math), verifying that timing does not matter. All strategies outperformed pure self-improvement.

### Key Findings
- The core driver of self-improvement is the "solver-verifier gap."
- Self-improvement has a non-zero capability ceiling ($G_\infty=b/k$); post-trained models are near saturation.
- Cross-improvement effectiveness depends on whether the external model's verification capability significantly exceeds that of the original model.

## Highlights & Insights
- **Modeling Training as a Physical System**: Using potential energy analogies to derive differential equations reduces the opaque process of self-improvement into a low-dimensional, analytical system.
- **Extrapolating Costs from Early Data**: Corollary 3.2 allows practitioners to calculate the required number of training rounds after just a few initial iterations.
- **Practical Data Scheduling**: The timing independence of external data allocation in cross-improvement provides engineering flexibility under API budget constraints.
- **Unified Framework**: Self-improvement and cross-improvement are unified within the same equations, differing only in the verifier's enhancement.

## Limitations & Future Work
- **Phenomenological Modeling**: The equations are based on physical analogies rather than first-principles derivations from neural network architectures.
- **Fit Parameters**: $\alpha, \beta$ are currently fitted rather than predicted; their relationship with model architecture remains unknown.
- **Time-Invariant $\gamma$**: The assumption that the relative advantage of an external verifier remains constant throughout training may not hold in reality.
- **Uncertainty as a Capability Metric**: While a reasonable approximation, the gap between uncertainty and actual task accuracy might introduce errors in some scenarios.

## Related Work & Insights
- **vs Song et al. (2025)**: While they proposed the "generation-validation gap," this work shifts the focus from static characterization to **dynamic modeling of the entire training trajectory**.
- **vs Huang et al. (2025)**: This work inherits their training setup but introduces a score threshold in BoN and focuses on deriving analytical dynamical equations.
- **vs Self-Distillation Theory**: Most traditional theories focus on linear models; this work specifically addresses the training dynamics of LLM self-improvement.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to formulate LLM self-improvement dynamics as analytical, physical-inspired differential equations.
- Experimental Thoroughness: ⭐⭐⭐⭐ High $R^2$ across multiple models, though tasks are mainly focused on mathematical reasoning and models are $\le$ 8B.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations with tight links between intuition and formulas.
- Value: ⭐⭐⭐⭐⭐ Provides tools to predict self-improvement limits and guide data allocation strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection](reshaping_reasoning_in_llms_a_theoretical_analysis_of_rl_training_dynamics_throu.md)
- [\[ICLR 2026\] A Theoretical Analysis of Mamba's Training Dynamics: Filtering Relevant Features for Generalization in State Space Models](a_theoretical_analysis_of_mambas_training_dynamics_filtering_relevant_features_f.md)
- [\[ICLR 2026\] Unveiling the Basin-like Loss Landscape in Large Language Models](unveiling_the_basin-like_loss_landscape_in_large_language_models.md)
- [\[ICLR 2026\] Theoretical Analysis of Contrastive Learning under Imbalanced Data: From Training Dynamics to a Pruning Solution](theoretical_analysis_of_contrastive_learning_under_imbalanced_data_from_training.md)
- [\[ICLR 2026\] Optimizing Data Augmentation through Bayesian Model Selection](optimizing_data_augmentation_through_bayesian_model_selection.md)

</div>

<!-- RELATED:END -->
