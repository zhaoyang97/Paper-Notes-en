---
title: >-
  [Paper Note] DynaDiff: Generative Adaptation of Dynamics to Environmental Shifts via Weight-space Diffusion
description: >-
  [ICML 2026][Weight-space diffusion] DynaDiff reframes the meta-learning problem of "training a predictor for a new environment" as a conditional sampling task of "directly generating the full network weights using a diff…
tags:
  - "ICML 2026"
  - "Weight-space diffusion"
  - "dynamics prediction"
  - "cross-environment generalization"
  - "Schrödinger-bridge alternative"
  - "meta-learning"
date: 2026-05-08
content_hash: ea5812399cccf9de
---

# DynaDiff: Generative Adaptation of Dynamics to Environmental Shifts via Weight-space Diffusion

**Conference**: ICML 2026  
**arXiv**: [2505.13919](https://arxiv.org/abs/2505.13919)  
**Code**: https://github.com/tsinghua-fib-lab/DynaDiff (available)  
**Area**: Diffusion Models / Scientific Machine Learning / Meta-learning  
**Keywords**: Weight-space diffusion, dynamics prediction, cross-environment generalization, Schrödinger-bridge alternative, meta-learning

## TL;DR
DynaDiff reframes the meta-learning problem of "training a predictor for a new environment" as a conditional sampling task of "directly generating the full network weights using a diffusion model." Leveraging weight graphs, function-consistency loss, and a dynamics-aware prompter, it achieves an average RMSE reduction of 10.78% over strong baselines across four PDE systems.

## Background & Motivation

**Background**: Data-driven dynamics prediction (e.g., FNO, Transformer-based neural operators) has replaced traditional numerical solvers in molecular dynamics, fluid mechanics, and meteorology. The two mainstream approaches for cross-environment adaptation (e.g., varying Reynolds numbers, external forces) are: (i) meta-learning that decomposes weights into environment-shared and environment-specific context, updating only the context for new environments; (ii) training a large foundation model and fine-tuning a small subset of parameters.

**Limitations of Prior Work**: Both approaches restrict adaptation to a small local subspace in weight space, failing to capture the full manifold spanned by expert weights across environments. They also rely on gradient optimization or massive backbones, making deployment in data-scarce or resource-constrained scenarios challenging.

**Key Challenge**: There is an inherent coupling between the function space of dynamics and the environment space, $\frac{dx}{dt}=f(x,t,e)$, where the optimal solution is a set of environment-dependent weights. Existing paradigms are forced to optimize only a subset of weights, resulting in limited expressiveness.

**Goal**: To model the joint distribution $p(\theta\mid e)$, enabling one-shot generation of the full expert weights $\theta_{new}$ for a new environment given a few observed frames, completely avoiding gradient backpropagation during inference.

**Key Insight**: Treat "model weights" as a data modality on par with images and text, and perform conditional generation in weight space using latent diffusion. However, weights present three unique challenges: non-flattenable topology, extremely high parameter dimensionality with distorted MSE metrics, and only short trajectories available for conditioning in new environments.

**Core Idea**: Employ a "weight graph + function-space-consistent VAE + dynamics-aware prompter" trio to make weight generation a purely forward sampling process.

## Method

### Overall Architecture
Offline phase: (1) Pretrain a base FNO on all visible environments, then quickly fine-tune for each environment to obtain experts, constructing a model zoo of size 100; (2) Organize each expert's weights into a "weight graph," pass it through a node-attention VAE to compress into a low-dimensional latent; (3) Train a conditional diffusion model $\epsilon_\theta(z_n,n,\text{prompt})$ on the latent, where the prompt is extracted from short observation sequences $X_L$ by a dynamics-informed prompter.

Online phase: Given $L=10$ frames from a new environment → prompter produces a conditioning vector → diffusion model samples a latent → VAE decodes back to a weight graph → reassemble as FNO weights → directly perform 100-step autoregressive prediction. The entire adaptation process requires no gradient updates.

### Key Designs

1. **Weight Graph + Node Attention VAE**:

    - **Function**: Losslessly maps network weights of arbitrary architectures to a low-dimensional latent suitable for diffusion.
    - **Mechanism**: Treats each output neuron/output channel of a layer as a graph node, with node features being the concatenation of all incoming weights and biases (linear layers: $D_{out}$ nodes, each $D_{in}+1$-dimensional; convolutional layers: $C_{out}$ nodes, each $C_{in}\cdot h\cdot w+1$-dimensional). Skip connection weights are attached to merge nodes; inter-layer dependencies are modeled with multi-head attention instead of GNNs, automatically adapting to various topologies.
    - **Design Motivation**: Flattening weights into tokens loses network connectivity; using dense edge features is too costly. Node aggregation balances computational efficiency and expressiveness.

2. **Function-consistency Loss as MSE Replacement**:

    - **Function**: Ensures that VAE-reconstructed weights are close to the original expert in "behavior" rather than "numerical value."
    - **Mechanism**: Replaces the naive $\|\hat{\mathbf{w}}-\mathbf{w}\|^2$ in the VAE objective with $L_{func}=\mathbb{E}_{x_i}\|f_{\hat{\mathbf{w}}}(x_i)-f_{\mathbf{w}}(x_i)\|_2^2$, i.e., compares outputs of the two weight sets on the same inputs.
    - **Design Motivation**: Neural network loss landscapes are highly multimodal; many parameter sets are functionally equivalent but numerically different. MSE treats these as different samples, causing the latent space to be dominated by noise. Function loss smooths the latent space in a functional sense, enabling the diffusion model to learn an effective conditional distribution (Appendix G provides a theoretical explanation for the generalization error bound).

3. **Dynamics-aware Prompter**:

    - **Function**: Distills a prompt vector from $L$ short observations that characterizes "which environment" is present.
    - **Mechanism**: Dual-branch structure—explicit branch computes temporal means and trends of physical statistics (first/second moments, energy, enstrophy); implicit branch passes real/imaginary FFT spectra through a GRU, taking the final state. The two branches are concatenated and injected into the diffusion transformer via adaLN. During training, observation length is randomly sampled from $[1,L]$ to make the prompter robust to varying frame counts at test time.
    - **Design Motivation**: In deployment, environment labels (e.g., Reynolds number) are unavailable and must be inferred from short trajectories. Purely data-driven features overfit to sample noise, while pure physical quantities lack expressiveness; the two branches complement each other, balancing physical interpretability and data-driven flexibility.

### Loss & Training

VAE objective: $L_{VAE}=-\mathbb{E}[\log p(\mathbf{w}|\mathbf{z})]+\beta\,\mathrm{KL}+\lambda L_{func}$; diffusion model uses standard $\epsilon$-prediction objective $L_n=\mathbb{E}\|\epsilon_n-\epsilon_\theta(\sqrt{\bar\alpha_n}z_0+\sqrt{1-\bar\alpha_n}\epsilon_n,n,\text{prompt})\|^2$. Model zoo construction uses "domain-adaptive initialization": first train a shared base, then fine-tune each expert with slight single-layer perturbations, avoiding the non-stationary weight distributions caused by random initialization.

## Key Experimental Results

### Main Results
Four PDE systems (Cylinder Flow / Lambda-Omega / Kolmogorov Flow / Navier-Stokes) plus ERA5 real wind speed. Generator has ~380M parameters, target FNO ~1M parameters.

| System | Metric | DynaDiff | Strongest meta-learning (GEPS) | Strongest weight generation (D2NWG) | Gain vs SOTA |
|--------|--------|---------:|-------------------------------:|-------------------------------------:|--------------|
| Cylinder Flow (out-domain) | RMSE | **0.065** | 0.082 | 0.086 | −20.7% |
| Lambda-Omega (out-domain)  | RMSE | **0.091** | 0.092 | 0.105 | −1.1% |
| Kolmogorov Flow (out-domain) | RMSE | **0.079** | 0.086 | 0.090 | −8.1% |
| Navier-Stokes (out-domain) | RMSE | **0.064** | 0.099 | 0.089 | −35.4% |

DynaDiff even surpasses One-per-Env (training a separate FNO for each environment, considered the "upper bound") in some cases. The authors attribute this to optimizers sometimes getting stuck in local minima, while diffusion sampling is more stable on the weight manifold.

### Ablation Study

| Configuration | RMSE (Kolmogorov) | Notes |
|---------------|------------------:|-------|
| Full DynaDiff | 0.079 | Full model |
| w/o function loss | ↑ significant | VAE degrades to pure MSE, severe OOD drop |
| w/o domain init | ↑ notable | Model zoo uses random init, weight distribution too scattered |
| Model zoo size 50 → 25 | Performance degrades | Zoo ≥50 is stable |
| Observation length $L$=1 vs 10 | Mild degradation | Variable-length training makes prompter robust |
| 10% neuron random permutation | Almost unchanged | Global attention auto-adapts to equivalent permutations |

### Key Findings
- **Function loss contributes most**: Removing it causes the sharpest OOD drop, validating the core hypothesis that "weight similarity should be measured by functional similarity."
- **Generator parameters grow linearly with predictor**: MLP module parameters are $O(Ld^2)$, but weight graph node count is $O(Ld)$ and feature dimension $O(d)$; attention is independent of node count, so generator scales linearly with $L,d$, not exponentially with total predictor parameters.
- **Interpretability**: Encoder attention matrices exhibit block-diagonal structure, with block boundaries aligning with FNO's lifting/fno_blocks/projection stages, and the structure remains stable across environments—DynaDiff identifies topological-level invariant functional partitions, not just numerical weights.
- **Sampling stability**: 100 repeated samplings across 10 OOD environments show extremely low variance; even the "worst sample" outperforms the strongest baseline.

## Highlights & Insights
- **Paradigm shift**: Replaces "gradient fine-tuning" with "forward generation." For deployment, adaptation to new environments drops from minutes to milliseconds, eliminating memory overhead for backpropagation.
- **Weight graph + attention is model-agnostic**: Experiments replacing FNO with WNO or UNO show stable performance, indicating this is a general interface reusable for other operator families.
- **Function loss is a transferable trick**: Any "weight-as-modality" generation task (hypernetwork, LoRA generation, neural field generation) can benefit from this "output consistency instead of weight MSE" technique.
- **Domain-adaptive model zoo is a hidden core**: The authors emphasize "sample quality > diversity," contrasting with D2NWG and other random-initialized weight generation methods, suggesting future work should not overlook the smoothness of the training data distribution.

## Limitations & Future Work
- Model zoo construction still requires training an expert for each visible environment; offline cost scales linearly with the number of environments, which remains heavy for large-scale scenarios.
- Currently only covers operators with ~1M parameters; whether diffusion generation can scale to 100M+ foundation model weights remains unverified.
- The prompter assumes $L=10$ frames are sufficient to distinguish environments; for highly chaotic systems or those with noisy observations, physical statistics and spectral features may fail.
- No discussion on the safety of generated weights—diffusion sampling may yield "seemingly reasonable but actually broken" weights outside the model zoo; the paper only uses "minimum reconstruction residual" as a post-filter.

## Related Work & Insights
- **vs CoDA / GEPS / CAMEL**: Meta-learning approaches only update environment context + shared weights, still requiring gradients for new environments; DynaDiff directly generates the full set of weights, offering a larger search space and expressiveness.
- **vs Poseidon / DPOT / MPP**: 600M-scale foundation models follow the ERM route; DynaDiff uses a 380M generator with 1M experts, similar total parameters but different division of labor—the large model "creates small models," and only the small model runs at inference.
- **vs D2NWG / CVAE / HyperDiffusion**: Also perform weight-space diffusion, but D2NWG flattens weights into sequences, CVAE lacks geometric constraints; DynaDiff's weight graph + function loss achieves significantly lower JSD and tighter distribution fitting.
- **Transferable insights**: Treating model weights as a data modality and defining similarity via functional equivalence rather than numerical equivalence are directly applicable to LoRA generation, neural field hypernetworks, continual learning, etc.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframes "training a network per environment" as "sampling a network"; the combination of weight graph and function loss is a first in scientific ML.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four PDEs + one real dataset, 14 baselines, six ablation dimensions (zoo size / environment count / observation length / permutation / function loss / domain init), plus generalization error bound proof.
- Writing Quality: ⭐⭐⭐⭐ Clear main line, ample figures and tables; some details (e.g., prompter dual-branch) require consulting the appendix.
- Value: ⭐⭐⭐⭐⭐ Provides a complete paradigm and reproducible code for "weight-as-modality" research, with practical impact for scientific computing at the edge.

## Related Papers

- [\[ICML 2026\] PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution](podiff_latent_diffusion_in_proper_orthogonal_decomposition_space_for_scientific_.md)
- [\[ICML 2025\] Hessian Geometry of Latent Space in Generative Models](../../ICML2025/image_generation/hessian_geometry_of_latent_space_in_generative_models.md)
- [\[CVPR 2026\] Guiding a Diffusion Transformer with the Internal Dynamics of Itself](../../CVPR2026/image_generation/guiding_a_diffusion_transformer_with_the_internal_dynamics_of_itself.md)
- [\[NeurIPS 2025\] PID-controlled Langevin Dynamics for Faster Sampling of Generative Models](../../NeurIPS2025/image_generation/pid-controlled_langevin_dynamics_for_faster_sampling_of_generative_models.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](../../ICLR2026/image_generation/generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)

</div>

<!-- RELATED:END -->
