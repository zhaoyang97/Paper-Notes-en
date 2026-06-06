---
title: >-
  [Paper Note] GRADIEND: Feature Learning within Neural Networks Exemplified through Biases
description: >-
  [ICLR 2026][Social Computing][monosemantic feature learning] This paper proposes GRADIEND — a gradient-based encoder-decoder architecture that learns interpretable monosemantic features (exemplified by gender) from model…
tags:
  - "ICLR 2026"
  - "Social Computing"
  - "monosemantic feature learning"
  - "gender debiasing"
  - "gradient encoder-decoder"
  - "Transformer debiasing"
  - "interpretability"
date: 2026-05-08
content_hash: 89f937d93748b864
---

# GRADIEND: Feature Learning within Neural Networks Exemplified through Biases

**Conference**: ICLR 2026
**arXiv**: [2502.01406](https://arxiv.org/abs/2502.01406)  
**Code**: [https://github.com/aieng-lab/gradiend](https://github.com/aieng-lab/gradiend)  
**Area**: Social Computing
**Keywords**: monosemantic feature learning, gender debiasing, gradient encoder-decoder, Transformer debiasing, interpretability

## TL;DR
This paper proposes GRADIEND — a gradient-based encoder-decoder architecture that learns interpretable monosemantic features (exemplified by gender) from model gradients via a single bottleneck neuron. The framework not only identifies which weights encode a specific feature, but also directly modifies model weights through the decoder to mitigate bias. Combined with INLP, it achieves state-of-the-art debiasing results across all baseline models.

## Background & Motivation
AI systems frequently exhibit and amplify social biases (e.g., gender bias), causing harmful effects in critical domains such as law, healthcare, and recruitment. Amazon's AI recruiting tool favoring male candidates is a well-known example.

Existing Transformer debiasing methods include:
- **Counterfactual Data Augmentation (CDA)**: retraining after swapping gender-related words, at high cost
- **Dropout Augmentation**: increasing the Dropout rate during pretraining
- **INLP**: Iterative Nullspace Projection, repeatedly training linear classifiers and projecting to the nullspace
- **SentDebias / SelfDebias**: post-hoc methods adjusting embeddings or output distributions

**Key Challenge**: Existing unsupervised sparse autoencoder methods (e.g., Bricken et al., 2023) can extract interpretable features but require learning a large number of latent features before searching for meaningful explanations, with no guarantee that a desired feature (e.g., "gender") will emerge. Meanwhile, most debiasing methods are post-hoc and do not genuinely modify the internal representations of already-trained models.

**Key Insight**: The paper exploits the feature information encoded in model gradients — gradients naturally indicate which parameters need to be updated to change a given feature. By designing a minimal encoder-decoder structure, a monosemantic feature neuron with **desired semantics** can be learned from the difference between factual and counterfactual gradients.

## Method

### Overall Architecture
**Input**: a pretrained Transformer language model + template sentences containing names and pronouns. **Output**: (1) a scalar feature neuron $h$ encoding gender information; (2) a decoding vector indicating how to modify model weights to adjust the degree of gender bias.

### Key Designs

1. **Factual/Counterfactual Gradient Construction**: For a template sentence such as "Alice explained the vision as best [MASK] could", MLM gradients are computed with the correct pronoun ("she") and the counterfactual pronoun ("he") as targets, respectively:

    - Factual gradient $\nabla^+ W_m$: targeting the correct gendered pronoun
    - Counterfactual gradient $\nabla^- W_m$: targeting the opposite gendered pronoun
    - Gradient difference $\nabla^{\pm}W_m := \nabla^+ W_m - \nabla^- W_m$

   The gradient difference cancels shared update components unrelated to gender, retaining only gender-relevant directions.

2. **GRADIEND Encoder-Decoder**: A minimal architecture with a single hidden neuron as the bottleneck:

    $\text{enc}(\nabla^+ W_m) = \tanh(W_e^T \cdot \nabla^+ W_m + b_e) =: h \in \mathbb{R}$
    $\text{dec}(h) = h \cdot W_d + b_d \approx \nabla^{\pm} W_m$

   where $W_e, W_d, b_d \in \mathbb{R}^n$ and $b_e \in \mathbb{R}$, yielding only $3n+1$ total parameters. The encoder maps the factual gradient to a scalar $h$ (gender factor); the decoder reconstructs the gradient difference from $h$. The objective is MSE loss.

3. **Gender Debiasing Application**: Given a selected gender factor $h$ and learning rate $\alpha$, model weights are directly modified as:

    $\tilde{W}_m := W_m + \alpha \cdot \text{dec}(h)$

   When $h$ and $\alpha$ share the same sign, the model is steered toward male bias; opposite signs steer toward female bias. Values near $h=0$ correspond to the debiasing direction (the debiasing direction learned via the bias $b_e$).

4. **Three Composite Metrics**:

    - **BPI** (Balanced Prediction Index): measures debiasing degree while accounting for language modeling capability, gender prediction balance, and prediction plausibility
    - **FPI** (Female Prediction Index): measures degree of female bias
    - **MPI** (Male Prediction Index): measures degree of male bias

### Loss & Training
- Optimizer: Adam, learning rate 1e-5, weight decay 1e-2
- Batch size: 32, MSE loss
- Training steps: 23,653 (equal to the number of templates in the Genter training set)
- Evaluated every 250 steps using $\text{Cor}_{\text{Genter}}^{\text{val}}$; best model selected
- At each step, a gender is randomly selected and a name is sampled from NAMExact
- Custom initialization of decoder weights (using the same $n$ as the encoder as the initialization range)
- The prediction layer is excluded from GRADIEND parameters, ensuring debiasing acts on the language model itself

## Key Experimental Results

### Encoder Evaluation (H1: Learning Gender Features)

| Model | $\text{Acc}_{\text{Genter}}$ | $\text{Cor}_{\text{Genter}}$ | $\text{Acc}_{\text{Enc}}$ | $\text{Cor}_{\text{Enc}}$ |
|------|------|------|------|------|
| BERT-base | 1.000 | 0.957 | 0.612 | 0.669 |
| BERT-large | 1.000 | 0.908 | 0.578 | 0.616 |
| DistilBERT | 1.000 | 1.000 | 0.758 | 0.838 |
| RoBERTa | 1.000 | 1.000 | 0.909 | 0.935 |

All models achieve near-perfect separation of $\pm 1$ on gender-relevant data; gender-neutral inputs are mapped to values close to 0.

### Debiasing Comparison (H2: Modifying Gender Bias)

| Method | SS(%) | SEAT | CrowS(%) | LMS(%) | GLUE(%) |
|------|-------|------|----------|--------|---------|
| BERT-base | baseline | baseline | baseline | baseline | baseline |
| + GRADIEND-BPI | improved | — | — | maintained | maintained |
| + GRADIEND-BPI + INLP | **significantly improved** | improved | — | maintained | maintained |
| CDA / Dropout / INLP / SentDebias | partially improved | inconsistent | inconsistent | partially degraded | partially degraded |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Different base models | All 4 models succeed | BERT/DistilBERT/RoBERTa all learn gender features |
| Gender factor $h=0$ | BPI optimum approximated here | Decoder bias $b_e$ automatically learns debiasing direction |
| Overfitting analysis | No significant difference across train/val/test names | Generalizes to unseen names |
| Generalization to woman/man | he/she generalize to woman/man | Cross-lexical generalization of gender concept |

### Key Findings
- **GRADIEND-BPI + INLP is the only combined method that achieves significant improvement on the SS metric across all baseline models**, demonstrating strong robustness
- After introducing confidence intervals, the effectiveness of existing debiasing methods is far less conclusive than prior studies suggest
- RoBERTa unexpectedly exhibits a female bias ($\mathbb{P}(F) > \mathbb{P}(M)$), contrary to the commonly assumed male bias
- Steering toward a specific gender (FPI/MPI) is easier to achieve than debiasing (BPI)
- The effect of weight modification is approximately point-symmetric with respect to the signs of $h$ and $\alpha$

## Highlights & Insights
- **Learning features with desired semantics directly from gradients**: Unlike unsupervised sparse autoencoders (which learn many features and interpret them post-hoc), GRADIEND can learn "desired" interpretable features (e.g., gender), representing an important paradigm shift
- **Minimal yet elegant design**: Only a single scalar bottleneck neuron with $3n+1$ parameters effectively encodes the complex concept of gender. The architectural simplicity facilitates analysis and understanding
- **Introduction of bootstrap confidence intervals**: Reveals a long-overlooked issue in the field — prior debiasing method comparisons lack statistical rigor
- **Interesting finding regarding decoder bias**: Even at $h=0$ (no gender information), the decoder bias $b_d$ itself learns an effective debiasing direction

## Limitations & Future Work
- Only binary gender features are validated; generalization to continuous features (e.g., sentiment), multi-valued features (e.g., German articles der/die/das), or other types of bias (race, religion) requires further exploration
- Only tested on encoder-only models; applicability to generative Transformers (GPT-family) remains unverified
- Factual/counterfactual gradient construction relies on the MLM objective; adaptation for CLM tasks has yet to be established
- Debiasing trade-off: aggressive debiasing degrades language modeling capability, requiring careful selection within the $h$ and $\alpha$ search grid
- Gender is treated as binary, without accounting for non-binary gender identities

## Related Work & Insights
- **Monosemantic features / sparse autoencoders** (Bricken et al., 2023; Templeton et al., 2024): unsupervised methods that decompose interpretable features from high-dimensional feature spaces; gender-bias-sensitive features have been identified in Claude 3
- **INLP** (Ravfogel et al., 2020): iterative nullspace projection debiasing; most effective when combined with GRADIEND
- **Movement Pruning** (Joniak & Aizawa, 2022): reducing gender bias through pruning
- **Grad-CAM / Integrated Gradients**: pioneering work in gradient-based interpretation
- Insight: gradients can serve not only for explanation (attribution) but also for encoding and manipulating semantic features within models. This notion of "gradients as feature representations" may hold significant value for model editing and machine unlearning

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Learning desired semantic features from gradients is a highly novel paradigm)
- Experimental Thoroughness: ⭐⭐⭐⭐ (4 base models, multiple metrics, but only gender is evaluated)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rigorous formulations, comprehensive appendix)
- Value: ⭐⭐⭐⭐ (High proof-of-concept value, though practical scope awaits expansion)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](../../ACL2026/social_computing/investigating_counterfactual_unfairness_in_llms_towards_identities_through_humor.md)
- [\[ACL 2026\] Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation](../../ACL2026/social_computing/dynamics_of_cognitive_heterogeneity_investigating_behavioral_biases_in_multi-sta.md)
- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](../../ICCV2025/social_computing/learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[NeurIPS 2025\] Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE](../../NeurIPS2025/social_computing/noise-robustness_through_noise_a_framework_combining_asymmetric_lora_with_poison.md)
- [\[CVPR 2026\] Learning from Synthetic Data via Provenance-Based Input Gradient Guidance](../../CVPR2026/social_computing/learning_from_synthetic_data_via_provenance-based_input_gradient_guidance.md)

</div>

<!-- RELATED:END -->
