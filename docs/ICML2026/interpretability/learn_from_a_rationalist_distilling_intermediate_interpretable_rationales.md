---
title: >-
  [Paper Note] Learn from A Rationalist: Distilling Intermediate Interpretable Rationales
description: >-
  [ICML 2026][Interpretability][Rationale extraction] This paper proposes REKD, which introduces knowledge distillation into the "select-predict" rationale extraction framework. It enables a small student model to simultan…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Rationale extraction"
  - "Knowledge distillation"
  - "Gumbel-Softmax"
  - "Temperature annealing"
  - "Curriculum learning"
date: 2026-05-08
content_hash: f6185d0ce66f2168
---

# Learn from A Rationalist: Distilling Intermediate Interpretable Rationales

**Conference**: ICML 2026  
**arXiv**: [2601.22531](https://arxiv.org/abs/2601.22531)  
**Code**: https://github.com/JiayiDai/REKD (Available)  
**Area**: Interpretability / Knowledge Distillation / Rationale Extraction  
**Keywords**: Rationale extraction, Knowledge distillation, Gumbel-Softmax, Temperature annealing, Curriculum learning

## TL;DR
This paper proposes REKD, which introduces knowledge distillation into the "select-predict" rationale extraction framework. It enables a small student model to simultaneously mimic the teacher's feature selection distribution and the final prediction distribution. By coupling the distillation temperature with the Gumbel-Softmax annealing schedule, it implicitly forms a "soft-to-hard" curriculum, consistently improving the RE accuracy of ViT-Tiny on CIFAR-10 from 0.797 to 0.936.

## Background & Motivation

**Background**: There are two mainstream routes in Explainable AI (XAI). One consists of post-hoc methods like LIME, SHAP, Integrated Gradients, and Grad-CAM, which are easy to integrate but lack "faithfulness"—the highlighted features may not be the ones the model actually used for decision-making. The other is rationale extraction (RE) proposed by Lei et al. (2016): a generator first selects a small subset of features as the rationale, and a predictor makes predictions based only on this subset, structurally guaranteeing that "what is used is what is explained."

**Limitations of Prior Work**: The training of RE relies solely on remote supervision from the final task. The generator must select features based on predictor feedback, while the predictor only sees what the generator selects—a classic "chicken-and-egg" problem. This dilemma is severely magnified when the underlying network capacity is small (e.g., BERT-Mini, ViT-Tiny). In the authors' experiments, switching ViT-Tiny from pure classification (CLS, 0.968) to 15% rationale RE resulted in a drop to 0.797 (−0.171), whereas ViT-Base only dropped by 0.020.

**Key Challenge**: There is a bidirectionally coupled search problem between the generator and the predictor. Small models can neither withstand high-variance gradients nor effectively search for sparse feature subsets that enable accurate predictions. Simply increasing data or training time is ineffective for small models as they fail to explore properly.

**Goal**: To achieve prediction accuracy for small student RE models that is close to large teacher RE models, without relaxing the hard constraint of "faithful interpretability."

**Key Insight**: The authors draw an analogy to learning physics after Newton—once verifiable and interpretable intermediate representations (e.g., "mass and distance are key variables") are established, an average person can apply the laws accurately without reinventing them. The feature selection layer output by the generator in RE is an **architecture-agnostic** universal interface. As long as the teacher and student face the same feature space, the information of "which features are important" can be distilled from a large model to a small model, bypassing the difficulties of architectural alignment.

**Core Idea**: Add a distillation branch to the RE framework, allowing the student to simultaneously mimic the teacher's Gumbel-Softmax feature selection distribution and prediction distribution. By sharing the temperature of this distillation branch with the Gumbel-Softmax annealing schedule, the training process naturally forms a "broad-to-precise" curriculum.

## Method

### Overall Architecture
REKD consists of a "Teacher RE + Student RE + Shared Temperature Schedule" triplet. The input $\mathbf{X} \in \mathbb{R}^{L \times D}$ (L features/patches/tokens, each D-dimensional) passes through both teacher and student generator-predictor pipelines, respectively producing: (1) a Gumbel-Softmax soft distribution $\mathbf{S}$ and its STE-discretized binary mask $\mathbf{M}$ (indicating selected features), and (2) class logits $\mathbf{Q}$ obtained by the predictor from the rationale $\mathbf{R} = \mathbf{M} \odot \mathbf{X}$. The total loss for the student combines its original $\mathcal{L}_{\text{RE}}$ (task cross-entropy + length constraint) and the distillation loss $\mathcal{L}_{\text{KD}}$ (generator distillation + predictor distillation) via weight $\alpha$, following the same exponential annealing temperature $\tau_k = \tau_0 e^{-\gamma k}$ (decaying from $\tau_0 = 5$ to $\tau_K = 0.1$).

### Key Designs

1.  **Differentiable RE based on Straight-Through Gumbel-Softmax**:
    - **Function**: Allows the discrete event of "selecting the $l$-th feature" to backpropagate gradients, avoiding high-variance REINFORCE estimates as in Lei et al. (2016).
    - **Mechanism**: The generator outputs 2D logits for "select/not-select" at each feature position. A soft distribution is sampled via $S_{l,i} = \exp((Z_{l,i} + G_{l,i})/\tau) / \sum_j \exp((Z_{l,j}+G_{l,j})/\tau)$ and then discretized into a 0/1 mask $M_l = \arg\max_i S_{l,i}$ for the predictor. During backpropagation, the STE convention $\partial \mathbf{M}/\partial \mathbf{S} \approx 1$ is used. Length constraints use a rectifier-style squared loss $\mathcal{L}_{\text{select}} = (\sum_l M_l - L \cdot p_{\text{target}})^2$ to control sparsity near the target $p_{\text{target}}$ (15% for CIFAR, 10% for IMDB).
    - **Design Motivation**: The "faithfulness" of RE requires the predictor to see *only* selected features, necessitating actual discretization in the forward pass; gradients must pass through this discretization to train the generator. STE + Gumbel-Softmax is the cleanest differentiable solution under this constraint.

2.  **Generator and Predictor Dual-Path Distillation**:
    - **Function**: Simultaneously transfers two complementary pieces of information: "which features the teacher considers important" and "the teacher's prediction based on these features."
    - **Mechanism**: Generator distillation calculates the KL divergence between teacher and student Gumbel-Softmax distributions at each position, $\mathcal{L}_{\text{KD}}^{\text{R}} = \sum_l D_{\text{KL}}(\mathbf{S}^{(T)}_{\tau,l} \,\|\, \mathbf{S}^{(S)}_{\tau,l})$; predictor distillation follows classic Hinton-KD on temperature-scaled softmax outputs, $\mathcal{L}_{\text{KD}}^{\text{Y}} = D_{\text{KL}}(\hat{\mathbf{Y}}^{(T)}_\tau \,\|\, \hat{\mathbf{Y}}^{(S)}_\tau)$. The two are merged as $\mathcal{L}_{\text{KD}} = \lambda_R \mathcal{L}_{\text{KD}}^{\text{R}} + \tau^2 \mathcal{L}_{\text{KD}}^{\text{Y}}$ ($\tau^2$ compensates for gradient attenuation due to logit scaling). The final objective is $\mathcal{L}_{\text{REKD}} = \alpha \mathcal{L}_{\text{RE}} + (1-\alpha)\mathcal{L}_{\text{KD}}$.
    - **Design Motivation**: Distilling only the prediction treats the student as a black box and discards interpretable intermediate supervision. Distilling only the rationale loses downstream signals on how to use those features. Parallel paths mimic effective human learning—identifying key variables and demonstrating their application. Since the selection layer is a unified 2D distribution interface, distillation is naturally compatible with different hidden dimensions.

3.  **Shared Temperature Scheduling — Implicit Curriculum Learning**:
    - **Function**: Automatically adjusts the difficulty of knowledge transfer from easy to hard as training progresses.
    - **Mechanism**: Gumbel-Softmax requires $\tau$ to decay from large to small (high $\tau$ for low-variance exploration, low $\tau$ for discrete approximation). REKD binds the KD temperature to this same $\tau_k = \tau_0 e^{-\gamma k}$. In early training, large $\tau$ results in flat Gumbel-Softmax and soft outputs, allowing the student to learn coarse-grained knowledge. In late training, as $\tau$ reaches 0.1, distributions become sharp, forcing the student to match the teacher's high-confidence hard selections and predictions.
    - **Design Motivation**: While annealing KD (Jafari et al., 2021) uses soft-to-hard transitions to bridge capacity gaps, it requires manual scheduling. In REKD, temperature sharing is a structural requirement for Gumbel-Softmax, making the curriculum a "free" byproduct with zero extra design cost.

### Loss & Training
The final objective is $\mathcal{L}_{\text{REKD}} = \alpha(\mathcal{L}_{\text{pred}} + \lambda_{\text{select}}\mathcal{L}_{\text{select}}) + (1-\alpha)(\lambda_R \mathcal{L}_{\text{KD}}^{\text{R}} + \tau^2 \mathcal{L}_{\text{KD}}^{\text{Y}})$. Training lasts 35 epochs (20 for pure CLS), lr=1e-5, bs=32, $\tau_0 = 5$, $\tau_K = 0.1$, with $\tau$ updated every 100 steps; $\lambda_R = 0.5$. Sparsity $p_{\text{target}}$ is 15% on CIFAR and 10% on IMDB. Results are averaged over 10 seeds.

## Key Experimental Results

### Main Results

| Dataset | Student Model | CLS | RE | REKD | RE→REKD Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CIFAR 10 | ViT-Small | .981 | .889 | **.968** | +.079 |
| CIFAR 10 | ViT-Tiny | .968 | .797 | **.936** | +.139 |
| CIFAR 100 | ViT-Small | .944 | .779 | **.845** | +.066 |
| CIFAR 100 | ViT-Tiny | .903 | .645 | **.777** | +.132 |
| IMDB | BERT-Small | .889 | .881 | **.906** | +.025 |
| IMDB | BERT-Mini | .877 | .863 | **.892** | +.029 |

A ViT-Base teacher achieves 0.964 on CIFAR-10; the ViT-Small student via REKD reaches 0.968, **slightly exceeding the teacher average**.

### Ablation Study

| Configuration | Meaning | Conclusion |
| :--- | :--- | :--- |
| Full REKD | $\alpha \in (0,1)$, Dual R + Y distillation | Full model, best performance. |
| Pure KD (No RE, $\alpha=0$) | Equivalent to two-stage supervised distillation | Performance drop, but still better than pure RE. |
| Predictor Only | Removed generator distillation | Worse than Full → Rationale distillation is essential. |
| Generator Only | Removed predictor distillation | Worse than Full → Both paths are complementary. |

### Key Findings
- **The "chicken-and-egg" dilemma in small models is empirically confirmed**: The drop from CLS to RE scales monotonically with model capacity (ViT-Base drops 0.020 vs ViT-Tiny drops 0.171). REKD's recovery is correspondingly largest for the smallest models (Tiny gain +0.139 > Small +0.079).
- **Student outperforming teacher**: On CIFAR-10/100, ViT-Small students trained with REKD slightly outperform ViT-Base teachers trained with RE. The authors attribute this to REKD acting as a strong prior regularization, reducing variance (std from .019 to .006 over 10 seeds).
- **REKD > Student CLS**: BERT-Mini@REKD (0.892) exceeds BERT-Mini@CLS (0.877), suggesting that sparse rationales combined with teacher distillation extract "information-dense" features more conducive to classification than the full noisy input. This is a counter-intuitive "less is more" phenomenon.

## Highlights & Insights
- **The feature selection layer as an "architecture-agnostic interface" is the most elegant part of the methodology**: Traditional feature-based KD usually requires projection modules for dimension alignment, whereas RE squashes "important vs unimportant" into a decoupled 2D softmax. This reduces distillation to a simple KL divergence between binomial distributions.
- **The "necessary constraint" becoming a "curriculum" is a graceful byproduct**: Since Gumbel-Softmax requires annealing, the authors obtain a soft-to-hard curriculum for distillation at zero extra cost.
- **Critique of XAI evaluation**: In Section 3.4, the authors argue against the mainstream plausibility paradigm (aligning rationales with human labels), using the "hospital name predicting cancer" example to show that alignment can be a double-edged sword. They advocate for "prediction accuracy under sparse constraints" as a more objective metric.

## Limitations & Future Work
- Distillation is currently **verified only within the same architecture family** (ViT→ViT, BERT→BERT). Cross-architecture distilling (e.g., ViT→ResNet) remains the "last mile" for the architecture-agnostic claim.
- **"Covert communication channel" risk**: Cooperative RE is often criticized for the potential of generators/predictors learning non-semantic steganographic signals. While the authors suggest REKD regularizes this, explicit experimental evidence is lacking.
- **Strong assumption on teacher quality**: Experiments assume an already well-trained, strong teacher RE model, without exploring scenarios where the teacher is also small or biased.
- **Narrow task scope**: Verified on IMDB (binary) and CIFAR (coarse classes) only; more complex real-world benchmarks like ERASER or long-form QA are not yet covered.

## Related Work & Insights
- **vs Lei et al. (2016) Original RE**: Original RE uses REINFORCE with high variance. Ours uses STE + Gumbel-Softmax (modern standard) and adds KD to mitigate small-model optimization issues.
- **vs Jain et al. (2020) Two-stage RE**: Jain uses heuristics to get pseudo-rationales for independent training. REKD allows for autonomous student exploration by retaining $\mathcal{L}_{\text{RE}}$ and is more robust due to the curriculum.
- **vs Hinton et al. (2015) Classic KD**: REKD extends KD from final predictions to intermediate structures (feature selection).
- **Transferable Insight**: Any task using Gumbel-Softmax for discrete latent structures can adopt the "dual-path distillation + shared temperature" template to gain curriculum benefits with almost zero additional design cost.

## Rating
- Novelty: ⭐⭐⭐⭐ First to explore RE × KD thoroughly; the "implicit curriculum via temperature sharing" is a genuine insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-modal and cross-capacity with 10-seed averages.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear analogies and strong arguments regarding XAI evaluation.
- Value: ⭐⭐⭐⭐ Provides a practical, low-cost solution for on-device interpretable models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Pixel2Phys: Distilling Governing Laws from Visual Dynamics](../../CVPR2026/interpretability/pixel2phys_distilling_governing_laws_from_visual_dynamics.md)
- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ICML 2026\] Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](interpretable_self-supervised_learning_via_representer_landmarks_and_nyström_app.md)
- [\[ACL 2026\] A Systematic Comparison between Extractive Self-Explanations and Human Rationales in Text Classification](../../ACL2026/interpretability/a_systematic_comparison_between_extractive_self-explanations_and_human_rationale.md)
- [\[ICML 2026\] MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification](minimax_learning_of_interpretable_factored_stochastic_policies_from_conjoint_dat.md)

</div>

<!-- RELATED:END -->
