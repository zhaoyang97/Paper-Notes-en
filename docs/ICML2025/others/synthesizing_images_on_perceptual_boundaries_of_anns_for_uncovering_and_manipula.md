---
title: >-
  [Paper Note] Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability
description: >-
  [ICML 2025][Perceptual Variability] This paper proposes the BAM (Boundary Alignment & Manipulation) framework, which systematically uncovers, predicts, and manipulates perceptual variability among human individuals by sampling and generating image stimuli on the perceptual decision boundaries of ANNs.
tags:
  - "ICML 2025"
  - "Perceptual Variability"
  - "Decision Boundary Sampling"
  - "Diffusion Model Guidance"
  - "Human-AI Alignment"
  - "Modeling Individual Differences"
date: 2026-05-08
content_hash: 356700e4aedc10ec
---

# Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability

**Conference**: ICML 2025  
**arXiv**: [2505.03641](https://arxiv.org/abs/2505.03641)  
**Code**: None  
**Area**: Cognitive Science / Artificial Intelligence  
**Keywords**: Perceptual Variability, Decision Boundary Sampling, Diffusion Model Guidance, Human-AI Alignment, Modeling Individual Differences

## TL;DR

This paper proposes the BAM (Boundary Alignment & Manipulation) framework, which systematically uncovers, predicts, and manipulates perceptual variability among human individuals by sampling and generating image stimuli on the perceptual decision boundaries of ANNs.

## Background & Motivation

Humans exhibit significant decision variability in cognitive tasks, meaning that different individuals may have completely different perceptual experiences when facing the exact same physical stimuli. For instance, a handwritten digit image might be perceived as "3" by some and "5" by others. While such **perceptual variability** has been widely documented in complex cognitive tasks (e.g., aesthetic or moral judgments), individual differences in simple visual decision-making tasks (e.g., handwritten digit classification) have rarely been thoroughly investigated.

Existing studies show a strong correlation between the latent representations of ANNs and human mental representations. Inspired by this, the authors propose a core hypothesis: **the perceptual decision boundaries of ANNs are correlated with human individual perceptual variability**—images sampled and generated on these boundaries can evoke divergent perceptual experiences in human participants.

Prior related work suffers from the following limitations:

- **Controversial stimuli** (Golan et al., 2020) only focus on classification differences between models and do not extend to human perception
- **Adversarial perturbations** (Veerabadran et al., 2023; Gaziv et al., 2024) generate images with insufficient naturalness, making them difficult to effectively influence human cognition
- **Model metamers** (Feather et al., 2023) reveal inconsistencies between models and humans but do not explore individual discrepancies

## Method

### Overall Architecture

The BAM framework consists of three interconnected steps:

1. **Labeling**: Sampling and generating images on the perceptual boundaries of ANNs, collecting labels via large-scale human behavioral experiments, and constructing the high-perceptual-variability dataset variMNIST.
2. **Aligning**: Fine-tuning the ANN models using human behavioral data to build computational models of perceptual variability at both the group and individual levels.
3. **Manipulating**: Using individualized models as adversarial generators to synthesize controversial stimuli, thereby magnifying the perceptual differences between specific pairs of participants.

### Key Designs

#### 1. Perceptual Boundary Sampling Algorithm

A classifier-guided diffusion model is employed to generate images on the decision boundaries of ANNs. Specifically, two guidance strategies are used:

**Uncertainty Guidance**: Ensures that a single classifier's predictive probabilities for the generated image are equal between two classes, forcing the image to fall near the decision boundary.

$$\mathcal{L} = H(p_1(y|x), q_1(y))$$

where $H$ is the cross-entropy, and the target distribution $q_1(y)$ ensures equal probabilities for the two classes (e.g., "3" and "7").

**Controversial Guidance**: Encourages two classifiers to yield opposite, high-confidence predictions for the same image.

$$\mathcal{L} = H(p_1(y|x), q_1(y)) + H(p_2(y|x), q_2(y))$$

The target distribution directs Classifier 1 to predict Class A with high confidence, while directing Classifier 2 to predict Class B with high confidence, thereby generating images that lie within the intersecting region of the decision boundaries of the two models.

#### 2. Diffusion Prior for Enhanced Naturalness

The decision boundary regions of ANNs typically contain highly noisy images. This work utilizes a DiT (Diffusion Transformer) as the core for image generation, introducing the MNIST data distribution prior via DDPM sampling:

$$x_{t-1} = DDPM^{-}(x_t) - \gamma \nabla_{x_t} \ell(f_\phi(x_t), y)$$

where $\gamma$ controls the guidance strength. The DiT is configured with a patch size of 2×2, hidden dimension of 128, 4 Transformer layers, and 8 attention heads. The diffusion prior effectively addresses the issue of unnatural image generation inherent in prior methods.

#### 3. Digit Judgment Surrogate

To filter out sub-standard images, a surrogate model is trained to predict whether an image resembles a digit. This model is based on the SmallVGG architecture and trained using human judgment frequencies as the regression target. The Spearman rank correlation coefficient between the surrogate model's predicted score and human ratings reaches 0.8035, validating its effectiveness. The surrogate guidance loss is defined as:

$$\mathcal{L}_{surr} = \mathcal{L} + \max((1 - f_{surr}(x))^2, 0.5)$$

#### 4. Model Alignment and Personalized Fine-Tuning

- **Group Fine-Tuning (GroupNet)**: Fine-tuned on mixed MNIST + variMNIST (1:1 ratio) data.
- **Individual Fine-Tuning (IndivNet)**: Continues fine-tuning from GroupNet on variMNIST-i + variMNIST + MNIST (2:1:1 ratio).
- Trained using the AdamW optimizer (lr=1e-3), CrossEntropyLoss, a batch size of 128, for 16 epochs.

#### 5. Manipulation Experimental Paradigm

18 participants (6 groups × 3 people) were selected, pairwise-matched within each group to form 18 pairs. Two rounds of experiments were conducted:

- **Round 1**: Each pair completed ~500 trials, and behavioral data were collected to fine-tune individual models.
- **Round 2**: Two individualized models were utilized to generate ~180 stimuli via controversial guidance to validate the manipulation performance.

### Loss & Training

Hyperparameter configuration for guided generation: guidance strength $\gamma=0.1$, resampling steps = 5, inference steps = 50. For guidance with MSE constraints, the pixel-level constraint strength is $\alpha=50$.

Filtering strategy: Uncertainty sampling requires the top-2 probability value to be >0.4 and the surrogate score to be >0.5; controversial sampling requires the outputs of both classifiers to align with the guidance direction, the maximum probability to be >0.9, and the surrogate score to be >0.5.

## Key Experimental Results

### Main Results

Dataset variMNIST: 19,943 images, 246 participants, and 116,715 trials.

| Model | MNIST Accuracy | variMNIST Accuracy | variMNIST-i Accuracy |
|------|------------|----------------|-------------------|
| BaseNet | ~98% | ~60% | ~55% |
| GroupNet | ~98% | ~80% (+20%) | ~75% |
| IndivNet | ~98% | ~80% | ~80% (+5%) |

MNIST baseline accuracy of the five classifiers:

| Classifier | Architecture | MNIST Accuracy |
|--------|------|-------------|
| ViT | Vision Transformer | 97.2% |
| VGG | Small VGG | 98.2% |
| CORNet | CORnet-Z | 98.9% |
| MLP | Multi-Layer Perceptron | 98.3% |
| LRM | Logistic Regression | 92.7% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| VGG Model-Human Entropy Correlation (Before fine-tuning) | Spearman $\rho=0.08$ | Baseline model barely captures human variability |
| VGG Model-Human Entropy Correlation (After fine-tuning) | Spearman $\rho=0.74$ | Fine-tuning significantly improves alignment |
| Individual Fine-tuning Gain | Improved for 241/246 participants | Only 5 participants showed a slight decline |
| Controversial Guidance: CORNet/VGG | Success Rate $\sim 0.6$ | Optimal classifier combination |
| Controversial Guidance: LRM | Success Rate $\sim 0.2$ | Weakest classifier |
| Manipulation Success Rate (variMNIST $\rightarrow$ IndivNet) | $+3\%$ ($p<0.001$) | Personalized models improve guidance success rate |
| Manipulation Directional Ratio (variMNIST $\rightarrow$ IndivNet) | $+12\%$ ($p<0.001$) | Personalized models significantly improve guidance directionality |

### Key Findings

1. **ANN variability can induce human variability**: The human choices' entropy for more than half of the generated images is significantly greater than zero; the average of guidance success rate + bias rate is close to 80%.
2. **Individual fine-tuning is more effective on high-difficulty samples**: The improvement of IndivNet over GroupNet is primarily concentrated on high-entropy (high-difficulty) images.
3. **Participant clustering analysis** reveals 8 perceptual clusters; model predictions are more accurate for participants within the same cluster, confirming the existence of high-level perceptual differences.
4. **The most effective guidance target pairs** are (1,7), (1,2), and (4,9), with success rates $>0.35$; the most difficult guidance pairs are (1,8), (2,9), and (7,8), with success rates $<0.03$.
5. **ImageNet validation**: Consistent findings are replicated on natural images (9 classes of Restricted ImageNet), where group fine-tuning brings a $\sim 4\%$ improvement, and individual fine-tuning contributes an additional $\sim 2\%$.

## Highlights & Insights

- **Bridging Computational Models and Human Individual Differences**: This work is the first to systematically link ANN decision boundary sampling with human perceptual variability, establishing a new paradigm "from cross-model controversy to cross-human controversy."
- **Data-Driven Personalized Perceptual Models**: An effective individualized perceptual model can be established with only about 200 trials, representing an extremely low cost.
- **Ingenious Use of Diffusion Priors**: This solves the long-standing issue of severe image noise in decision boundary regions, making the generated images sufficiently natural while retaining boundary properties.
- **Large-Scale Human Experimental Verification**: The scale of 246 participants and 116,715 trials represents a state-of-the-art benchmark in similar lines of research.

## Limitations & Future Work

1. **Cultural Bias**: Training data are derived from a specific group and may not fully capture perceptual variability caused by cultural differences. Future work needs to recruit participants from diverse cultural backgrounds.
2. **Limited Task Scope**: The focus is solely on object recognition tasks, leaving out more complex visual phenomena such as similarity judgment, emotion recognition, and visual attention.
3. **Artificial-Human Alignment Gap Remains Significant**: The fine-tuned models' performance in predicting perceptual variability still falls heavily behind their performance in standard classification tasks.
4. **Limited Individual Experimental Trials**: The number of trials per participant in behavioral experiments is constrained, limiting the precision upper bound of the individual models.
5. **Future Directions**: Integrating Bayesian Optimal Experimental Design to generate stimuli that maximize information content utilizing individualized models, forming an "experiment $\rightarrow$ fine-tuning $\rightarrow$ generation $\rightarrow$ experiment" closed loop.

## Related Work & Insights

- **Controversial Stimuli** (Golan et al., 2020, 2023): Extending cross-model controversy to cross-human controversy.
- **Adversarial Perturbations Affecting Humans** (Veerabadran et al., 2023; Gaziv et al., 2024): Validating that ANN perturbations can affect human perception.
- **DreamSim** (Fu et al., 2023): Utilizing synthetic data to align perceptual metrics.
- **Counterfactual Explanations via Diffusion Models** (Jeanneret et al., 2023; Wei et al., 2024b): Providing a technological foundation for natural image generation.
- **Insights**: The proposed framework can be extended to broader AI-human alignment scenarios, such as user preference modeling in personalized recommendation systems and expert discrepancy analysis in medical image diagnosis.

## Rating

| Dimension | Rating | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐⭐ | First systematic study on the relationship between ANN decision boundaries and human perceptual variability. |
| Methodological Rigor | ⭐⭐⭐⭐⭐ | Large-scale human experiments + integrated "sampling $\rightarrow$ alignment $\rightarrow$ manipulation" closed-loop verification. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | 246 participants, 116K trials, covering 5 classifiers and two data domains. |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured, although some details require referencing the appendix. |
| Value | ⭐⭐⭐⭐ | Provides new tools for AI-human alignment and personalized perception research. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Manipulating Feature Visualizations with Gradient Slingshots](../../NeurIPS2025/others/manipulating_feature_visualizations_with_gradient_slingshots.md)
- [\[AAAI 2026\] On the Variability of Concept Activation Vectors](../../AAAI2026/others/on_the_variability_of_concept_activation_vectors.md)
- [\[ACL 2025\] From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding](../../ACL2025/others/from_real_to_synthetic_synthesizing_millions_of_diversified_and_complicated_user.md)
- [\[CVPR 2025\] Locally Orderless Images for Optimization in Differentiable Rendering](../../CVPR2025/others/locally_orderless_images_for_optimization_in_differentiable_rendering.md)
- [\[ACL 2025\] A Little Human Data Goes A Long Way](../../ACL2025/others/a_little_human_data_goes_a_long_way.md)

</div>

<!-- RELATED:END -->
