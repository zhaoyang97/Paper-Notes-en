---
title: >-
  [Paper Note] DynaDiff: Generative Adaptation of Dynamics to Environmental Shifts via Weight-space Diffusion
description: >-
  [ICML 2026][Image Generation][Weight-space Diffusion] DynaDiff reformulates the meta-learning problem of "training a predictor for a new environment" as a conditional sampling problem of "directly generating complete net…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Weight-space Diffusion"
  - "Dynamics Prediction"
  - "Cross-environment Generalization"
  - "Schrödinger-bridge Alternative"
  - "Meta-learning"
date: 2026-05-08
content_hash: 17db3d5018118027
---

# DynaDiff: Generative Adaptation of Dynamics to Environmental Shifts via Weight-space Diffusion

**Conference**: ICML 2026  
**arXiv**: [2505.13919](https://arxiv.org/abs/2505.13919)  
**Code**: https://github.com/tsinghua-fib-lab/DynaDiff (Available)  
**Area**: Diffusion Models / Scientific Machine Learning / Meta-learning  
**Keywords**: Weight-space Diffusion, Dynamics Prediction, Cross-environment Generalization, Schrödinger-bridge Alternative, Meta-learning

## TL;DR
DynaDiff reformulates the meta-learning problem of "training a predictor for a new environment" as a conditional sampling problem of "directly generating complete network weights using a diffusion model." By utilizing weight graphs, functional consistency loss, and a dynamics-aware prompter, it reduces the average RMSE by 10.78% over strong baselines across four PDE systems.

## Background & Motivation

**Background**: Data-driven dynamics prediction (e.g., FNO, Transformer neural operators) has increasingly replaced traditional numerical solvers in molecular dynamics, fluid mechanics, and meteorology. Two mainstream strategies for handling cross-environment shifts (e.g., different Reynolds numbers or external forces) are: (i) Meta-learning, which splits weights into environment-shared and environment-specific contexts, updating only the context for new environments; or (ii) Training a massive foundation model and fine-tuning a small subset of parameters.

**Limitations of Prior Work**: Both strategies only allow "adjustments within a small local subspace of the weight space," failing to represent the complete manifold of expert weights spanned across different environments. Furthermore, they rely on gradient optimization or massive backbones, making deployment difficult in data-scarce or hardware-constrained scenarios.

**Key Challenge**: A natural coupling exists between dynamics function space and environment space: $\frac{dx}{dt}=f(x,t,e)$. The optimal solution is an environment-dependent "complete set of weights," but existing paradigms are forced to optimize only a subset of weights, resulting in an expressivity ceiling.

**Goal**: Model the joint distribution $p(\theta\mid e)$ to generate complete expert weights $\theta_{new}$ in a single step given limited observations of a new environment, completely bypassing gradient backpropagation during inference.

**Key Insight**: Treat "model weights" as a data modality equivalent to images or text, and perform conditional generation in weight space using latent diffusion. However, weights present three unique challenges: topological structures cannot be simply flattened, the parameter space is high-dimensional with distorted MSE metrics, and new environments provide only short trajectories as conditions.

**Core Idea**: Use a triplet of "weight graph + functional consistency VAE + dynamics-aware prompter" to transform weight generation into a pure forward-sampling process.

## Method

### Overall Architecture
Offline stage: (1) Pre-train a base FNO on all visible environments, then rapidly fine-tune it for each environment to obtain experts, constructing a model zoo of size 100; (2) Organize each expert's weights as a "weight graph" and pass it through a node-attention VAE to compress it into a low-dimensional latent; (3) Train a conditional diffusion model $\epsilon_\theta(z_n,n,\text{prompt})$ in the latent space, where the prompt is extracted from short observation sequences $X_L$ by a dynamics-informed prompter.

Online stage: Receive $L=10$ frames of observations from a new environment → Prompter generates a condition vector → Diffusion model samples a latent → VAE decodes the latent back into a weight graph → Reorganize into FNO weights → Directly perform 100-step autoregressive prediction. The entire adaptation process requires no gradient updates.

### Key Designs

1.  **Weight Graph + Node-attention VAE**:
    - **Function**: Losslessly map network weights of arbitrary architectures into low-dimensional latents suitable for diffusion.
    - **Mechanism**: Treat output neurons or channels of each layer as graph nodes. Node features are the concatenation of all weights and biases flowing into that node (e.g., linear layer with $D_{out}$ nodes, each $D_{in}+1$ dimensional; convolutional layer with $C_{out}$ nodes, each $C_{in}\cdot h\cdot w+1$ dimensional). Skip connection weights are attached to merge nodes, and cross-layer dependencies use multi-head attention instead of GNNs to automatically adapt to various topologies.
    - **Design Motivation**: Flattening weights into tokens loses connection topology, while dense edge features are computationally expensive. Node aggregation provides a balance between computation and representation.

2.  **Functional Consistency Loss vs. MSE Reconstruction**:
    - **Function**: Ensure VAE-reconstructed weights are close to the original expert in "behavior" rather than just "numerical values."
    - **Mechanism**: Replace the naive $\|\hat{\mathbf{w}}-\mathbf{w}\|^2$ in the VAE objective with $L_{func}=\mathbb{E}_{x_i}\|f_{\hat{\mathbf{w}}}(x_i)-f_{\mathbf{w}}(x_i)\|_2^2$, which compares the outputs of two sets of weights given the same inputs.
    - **Design Motivation**: Neural network loss landscapes are highly multimodal, containing many function-equivalent solutions with different parameters. MSE treats these "behaviorally identical" weights as different samples, causing the latent space to be dominated by noise. Functional loss smooths the latent space semantically, allowing the diffusion model to learn effective conditional distributions (Appendix G provides a theoretical proof for generalization error bounds).

3.  **Dynamics-aware Prompter**:
    - **Function**: Distill a prompt vector representing the environment from $L$ short observation frames.
    - **Mechanism**: A dual-branch structure—the explicit branch uses temporal means and trends of physical statistics (1st/2nd moments, energy, enstrophy); the implicit branch uses FFT-derived real/imaginary spectral sequences processed through a GRU. Both branches are concatenated and injected into the diffusion transformer via adaLN. During training, observation lengths are randomly sampled within $[1,L]$ to ensure prompter robustness to varying frame counts.
    - **Design Motivation**: Real-world deployments lack environment labels (e.g., Reynolds number). Purely data-driven features may overfit sample noise, while purely physical quantities lack expressivity. The complementary branches ensure both physical interpretability and data-driven flexibility.

### Loss & Training
The VAE objective is $L_{VAE}=-\mathbb{E}[\log p(\mathbf{w}|\mathbf{z})]+\beta\,\mathrm{KL}+\lambda L_{func}$. The diffusion model uses the standard $\epsilon$-prediction objective $L_n=\mathbb{E}\|\epsilon_n-\epsilon_\theta(\sqrt{\bar\alpha_n}z_0+\sqrt{1-\bar\alpha_n}\epsilon_n,n,\text{prompt})\|^2$. Model zoo construction utilizes "domain-adaptive initialization": training a shared base first, then adding single-layer perturbations before fine-tuning each expert, which avoids non-stationary weight distributions caused by random initialization.

## Key Experimental Results

### Main Results
Evaluated on 4 PDE systems (Cylinder Flow / Lambda-Omega / Kolmogorov Flow / Navier-Stokes) and ERA5 real-world wind speed data. The generator has ~380M parameters, and the target FNO has ~1M parameters.

| System | Metric | DynaDiff | Strongest meta-learning (GEPS) | Strongest weight generation (D2NWG) | Gain vs SOTA |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Cylinder Flow (out-domain) | RMSE | **0.065** | 0.082 | 0.086 | −20.7% |
| Lambda-Omega (out-domain)  | RMSE | **0.091** | 0.092 | 0.105 | −1.1% |
| Kolmogorov Flow (out-domain) | RMSE | **0.079** | 0.086 | 0.090 | −8.1% |
| Navier-Stokes (out-domain) | RMSE | **0.064** | 0.099 | 0.089 | −35.4% |

DynaDiff even outperforms One-per-Env (the upper bound of training a separate FNO for each environment) in some cases. The authors attribute this to standard optimizers occasionally falling into local optima, while diffusion sampling is more stable on the weight manifold.

### Ablation Study

| Configuration | RMSE (Kolmogorov) | Description |
| :--- | :---: | :--- |
| Full DynaDiff | 0.079 | Full model |
| w/o Functional Loss | Significant ↑ | VAE degrades to pure MSE; sharp drop in OOD performance |
| w/o domain init | Substantial ↑ | Model zoo uses random initialization; weight distribution too scattered |
| Model zoo size 50 → 25 | Performance degrades | Stability requires zoo size ≥50 |
| Observation length $L$=1 vs 10 | Mild degradation | Variable-length training makes prompter robust |
| Neuron random permutation 10% | Almost unchanged | Global attention automatically adapts to equivalent permutations |

### Key Findings
- **Functional Loss is the Primary Contributor**: Its removal leads to the steepest decline in OOD performance, validating the core hypothesis that "weight similarity should be measured by functional similarity."
- **Linear Scalability**: Generator parameters scale linearly with predictor dimensions $L, d$ because attention is independent of node count.
- **Interpretability**: Encoder attention matrices exhibit block-diagonal structures aligning with FNO stages (lifting / fno\_blocks / projection), showing that DynaDiff identifies invariant functional partitions at the topological level.
- **Sampling Stability**: Standard deviation across 100 repeated samples in 10 OOD environments is extremely low; even the "worst sample" outperforms the strongest baseline.

## Highlights & Insights
- **Paradigm Shift**: Shifting from "gradient fine-tuning" to "forward generation." For deployment, adaptation to new environments changes from minutes to milliseconds and eliminates the memory overhead required for backpropagation.
- **Model-agnosticism**: The weight graph and attention mechanism proved stable when replacing FNO with WNO or UNO, suggesting a general-purpose interface for other operator families.
- **Transferable Trick**: The functional consistency loss—using output consistency instead of weight MSE—is a valuable technique for any task involving weights as a data modality (e.g., hypernetworks, LoRA generation, neural field generation).
- **Domain-adaptive Model Zoo**: The emphasis on "sample quality > diversity" contrasts with random initialization methods, suggesting that future work should prioritize distribution smoothness in training data.

## Limitations & Future Work
- Model zoo construction still requires training one expert per visible environment; offline costs scale linearly with the number of environments.
- Currently only covers small operators (~1M parameters); whether diffusion generation scales to 100M+ foundation model weights remains unverified.
- The prompter assumes $L=10$ frames are sufficient for environment identification; physical statistics and spectral features might fail in highly chaotic or noisy systems.
- Lack of discussion on weight safety; diffusion sampling could potentially produce "reasonable-looking but broken" weights. The paper only uses minimal reconstruction residuals for post-filtering.

## Related Work & Insights
- **vs CoDA / GEPS / CAMEL**: Meta-learning routes update environment contexts + shared weights via gradients; DynaDiff generates the entire weight set, offering a larger search space and higher expressivity.
- **vs Poseidon / DPOT / MPP**: While ~600M foundation models follow Empirical Risk Minimization (ERM), DynaDiff uses a 380M generator to "create" 1M experts. Both use similar total parameters but different divisions of labor—large models producing small models.
- **vs D2NWG / CVAE / HyperDiffusion**: Unlike D2NWG (flattened sequences) or CVAEs (lack of geometric constraints), DynaDiff's weight graph and functional loss result in significantly lower JSD and tighter distribution fitting.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First instance of framing environment adaptation as weight-space sampling with functional loss in SciML.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 PDEs + 1 real dataset, 14 baselines, and 6 ablation dimensions, including generalization error proofs.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative and ample illustrations; minor details (e.g., dual-branch prompter) require referring to the appendix.
- Value: ⭐⭐⭐⭐⭐ Provides a complete paradigm and reproducible code for "weight-as-modality" research, significantly advancing scientific computing for edge deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Compression as Adaptation: Implicit Visual Representation with Diffusion Foundation Models](compression_as_adaptation_implicit_visual_representation_with_diffusion_foundati.md)
- [\[ICML 2026\] Efficient Learning of Deep State Space Models via Importance Smoothing](efficient_learning_of_deep_state_space_models_via_importance_smoothing.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](../../ICLR2026/image_generation/generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[CVPR 2026\] Guiding a Diffusion Transformer with the Internal Dynamics of Itself](../../CVPR2026/image_generation/guiding_a_diffusion_transformer_with_the_internal_dynamics_of_itself.md)
- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](../../CVPR2026/image_generation/dip_taming_diffusion_models_in_pixel_space.md)

</div>

<!-- RELATED:END -->
