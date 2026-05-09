---
title: >-
  [Paper Note] Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models
description: >-
  [ICLR 2026][Self-Supervised Learning][behavioral foundation models] This paper proposes Regularized Latent Dynamics Prediction (RLDP), which augments a self-supervised latent next-state prediction objective with a simple orthogonality regularization to preserve feature diversity. RLDP matches or surpasses complex state-of-the-art representation learning methods in zero-shot RL, with particularly notable advantages in low-coverage settings.
tags:
  - ICLR 2026
  - Self-Supervised Learning
  - behavioral foundation models
  - zero-shot RL
  - latent dynamics prediction
  - orthogonality regularization
  - state feature learning
date: 2026-05-08
content_hash: 172458a9db6f44ee
---

# Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models

**Conference**: ICLR 2026
**arXiv**: [2603.15857](https://arxiv.org/abs/2603.15857)
**Code**: None
**Area**: Self-Supervised Learning
**Keywords**: behavioral foundation models, zero-shot RL, latent dynamics prediction, orthogonality regularization, state feature learning

## TL;DR
This paper proposes Regularized Latent Dynamics Prediction (RLDP), which augments a self-supervised latent next-state prediction objective with a simple orthogonality regularization to preserve feature diversity. RLDP matches or surpasses complex state-of-the-art representation learning methods in zero-shot RL, with particularly notable advantages in low-coverage settings.

## Background & Motivation
**Behavioral Foundation Models (BFMs)** aim to train agents capable of adapting to arbitrary unknown rewards or tasks. The core idea is to pre-train state feature representations on offline datasets so that, at test time, a near-optimal policy for a new reward function can be recovered in a zero-shot manner—without further interaction with the environment.

However, existing BFM methods face a fundamental limitation: they can only produce near-optimal policies for reward functions that lie within the **linear span** of some pre-existing state features. Consequently, the choice of state features is critical to the expressiveness of a BFM—features must be sufficiently diverse to cover as many reward functions as possible. To this end, prior methods design increasingly complex representation learning objectives (e.g., HILP's diversity objectives, Forward-Backward (FB) representations), which require adequate dataset coverage to learn useful spanning features.

This paper raises a key question: **Are these complex representation learning objectives truly necessary for zero-shot RL?** The authors find that simple self-supervised next-state prediction in the latent space can already learn useful features, but suffers from a collapse problem—this objective tends to make feature vectors converge to similar directions during training, reducing the dimensionality of the span. The solution turns out to be surprisingly simple: adding orthogonality regularization suffices.

## Method

### Overall Architecture
RLDP operates in two stages:
1. **Feature Pre-training**: An encoder and a latent dynamics model are trained on an offline dataset to learn a mapping from states to a latent space and to predict next-state dynamics, with orthogonality regularization enforcing feature diversity.
2. **Zero-Shot Policy Recovery**: Given a new reward function, the pre-trained feature representations are used to compute a policy directly via a linear combination of successor features, requiring no further training.

### Key Designs

1. **Latent Next-State Prediction**:
   Given a state $s$ and action $a$, an encoder $\phi$ maps the state to a latent vector $z = \phi(s)$, and a dynamics model $f$ predicts the next latent state $\hat{z}' = f(z, a)$. The training objective minimizes the prediction error:
   $$\mathcal{L}_{\text{dynamics}} = \|f(\phi(s), a) - \phi(s')\|_2^2$$
   Intuitively, this objective encourages the learning of features that are informative about environment dynamics. However, the authors identify a degeneracy: the model tends to collapse all state feature vectors toward similar directions, since encoding dynamics information via small magnitude variations is sufficient—but at the cost of drastically shrinking the feature span.

2. **Orthogonality Regularization**:
   To prevent feature collapse, RLDP imposes an orthogonality constraint on the feature matrix:
   $$\mathcal{L}_{\text{ortho}} = \|\Phi^T \Phi - I\|_F^2$$
   where $\Phi$ is the matrix of state features in a training batch. This regularization encourages features of different states to be as orthogonal as possible, preserving the richness and diversity of the feature space.

3. **Total Training Objective**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{dynamics}} + \lambda \mathcal{L}_{\text{ortho}}$$
   The overall objective comprises only latent dynamics prediction and orthogonality regularization—far simpler than existing BFM methods that require forward-backward models, contrastive learning, or diversity objectives.

4. **Zero-Shot Policy Recovery**: After pre-training, for a new reward function $r$, the successor features framework is applied: the learned features $\phi(s)$ enable reward approximation as $r(s) \approx w^T \phi(s)$. The weight vector $w$ is obtained via linear regression, and Q-values are computed as a linear combination of successor features, yielding a zero-shot policy.

### Loss & Training
- Pre-training is performed entirely offline, with no online interaction.
- Both the encoder and dynamics model are simple MLPs.
- The hyperparameter $\lambda$ controls the strength of the orthogonality regularization.

## Key Experimental Results

### Main Results
On standard zero-shot RL benchmarks (e.g., continuous control tasks on the ExORL dataset), RLDP is compared against complex state-of-the-art methods:

| Method | Complexity | Zero-Shot Performance | Notes |
|---|---|---|---|
| FB (Forward-Backward) | High | SOTA-level | Requires forward and backward models |
| HILP | High | SOTA-level | Requires hierarchical objectives |
| ICM (dynamics prediction only) | Low | Poor | Severe feature collapse |
| **RLDP** | **Lowest** | **Matches/Surpasses SOTA** | Dynamics prediction + orthogonality regularization only |

### Low-Coverage Experiments
RLDP's key advantage over SOTA methods is most pronounced in low-coverage scenarios:

| Coverage | RLDP | FB | HILP |
|---|---|---|---|
| Sufficient coverage | Matches SOTA | Good | Good |
| Low coverage | **Remains effective** | Performance degrades | Performance degrades |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Dynamics prediction only (no orthogonality regularization) | Poor | Feature collapse, reduced span |
| Orthogonality regularization only (no dynamics prediction) | Poor | Features lack dynamics semantics |
| Dynamics + Orthogonality (RLDP) | Optimal | Both components are necessary |
| Effect of $\lambda$ | Optimal range exists | Too large suppresses dynamics learning; too small fails to prevent collapse |

### Key Findings
- Simple self-supervised next-state prediction combined with orthogonality regularization is sufficient to match or surpass complex SOTA methods.
- The core failure mode of pure dynamics prediction is increased feature similarity leading to span reduction—orthogonality regularization addresses this precisely.
- The advantage is most pronounced in low-coverage settings: complex methods rely on data diversity to train features, while RLDP's orthogonality constraint provides additional structural guarantees.
- This finding challenges the prevailing assumption that zero-shot RL requires complex representation learning.

## Highlights & Insights
- **Minimal yet effective**: In a field increasingly dominated by complex objective functions, RLDP achieves SOTA performance with a minimal modification—a single orthogonality regularization term—demonstrating the value of strong baseline research.
- **Precise diagnostic insight**: The paper accurately identifies the degeneracy of latent dynamics prediction (increasing feature similarity → reduced span) and proposes a targeted remedy.
- **Strong practicality**: The method is simple enough that orthogonality regularization can be implemented in a single line of code ($\|\Phi^T\Phi - I\|$), requiring no additional network components or training procedures.
- **Low-coverage advantage**: Since offline datasets in practice often have limited coverage, RLDP's consistent performance in such realistic settings is practically significant.

## Limitations & Future Work
- The current linear successor features framework may underperform on tasks requiring nonlinear reward decoding.
- Orthogonality regularization may impose overly strong constraints when the feature dimensionality is very high.
- Evaluation is limited to continuous control environments such as MuJoCo; visual observations and high-dimensional inputs are not addressed.
- The hyperparameter $\lambda$ still requires tuning.
- The dynamics model uses a simple MLP; stronger architectures (e.g., Transformers) may yield further improvements.
- Although zero-shot performance is strong, it remains suboptimal; few-shot fine-tuning mechanisms warrant further exploration.

## Related Work & Insights
- **Relation to FB (Forward-Backward)**: FB requires training separate forward and backward models, whereas RLDP requires only a single-direction dynamics prediction.
- **Relation to HILP**: HILP requires hierarchical diversity objectives and intrinsic reward design; RLDP achieves the same effect via orthogonality regularization alone.
- **Relation to ICM/RND**: These classical exploration methods based on next-state prediction also employ dynamics objectives, but do not address the feature span reduction problem.
- **Insight**: In representation learning, maintaining feature diversity is often more important than designing complex learning objectives; a simple method with the right inductive bias can go a long way.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](../../NeurIPS2025/self_supervised/uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[ICLR 2026\] SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty](snap-uq_self-supervised_next-activation_prediction_for_single-pass_uncertainty_i.md)
- [\[AAAI 2026\] Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)
- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](../../CVPR2026/self_supervised/com_pt_chain_of_models_pretraining.md)
- [\[CVPR 2026\] Robustness of Vision Foundation Models to Common Perturbations](../../CVPR2026/self_supervised/robustness_of_vision_foundation_models_to_common_perturbations.md)

</div>

<!-- RELATED:END -->
