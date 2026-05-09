---
title: >-
  [Paper Note] Scaling Can Lead to Compositional Generalization
description: >-
  [NeurIPS 2025][Image Generation][compositional generalization] Through theoretical proofs and large-scale experiments, this paper demonstrates that standard MLPs can achieve compositional generalization solely by scaling data and model size, without explicit modular architectural design. Moreover, when compositional generalization succeeds, task components can be linearly decoded from hidden-layer activations — a metric that correlates positively with compositional success rates in diffusion-based image generation.
tags:
  - NeurIPS 2025
  - Image Generation
  - compositional generalization
  - scaling
  - linear decodability
  - hyperteacher
  - MLP
date: 2026-05-08
content_hash: da7f3a48e6925c5a
---

# Scaling Can Lead to Compositional Generalization

**Conference**: NeurIPS 2025
**arXiv**: [2507.07207](https://arxiv.org/abs/2507.07207)
**Code**: [GitHub](https://github.com/smonsays/scale-compositionality)
**Area**: Theoretical Machine Learning / Compositional Generalization
**Keywords**: compositional generalization, scaling, linear decodability, hyperteacher, MLP

## TL;DR

Through theoretical proofs and large-scale experiments, this paper demonstrates that standard MLPs can achieve compositional generalization solely by scaling data and model size, without explicit modular architectural design. Moreover, when compositional generalization succeeds, task components can be linearly decoded from hidden-layer activations — a metric that correlates positively with compositional success rates in diffusion-based image generation.

## Background & Motivation

**State of the Field**: Compositional generalization — the ability to understand and produce novel combinations of known components — is considered a core capacity of intelligence. Since Fodor & Pylyshyn (1988), whether neural networks can genuinely achieve compositional generalization has remained a subject of debate. Large-scale models exhibit impressive compositional capabilities, yet even the strongest models still exhibit frequent compositional failures.

**Limitations of Prior Work**: Many works argue that networks must be endowed with explicit compositional structure (modular architectures, graph networks, etc.) to achieve compositionality. This position, however, conflicts with the empirical success of large models, which lack explicit modular design yet appear to possess some degree of compositional ability.

**Root Cause**: The number of tasks in a compositional task family grows exponentially as $\mathcal{O}(M^K)$, making exhaustive coverage infeasible. Memorizing all tasks requires exponential network capacity. If, however, a generalizing solution that exploits compositional structure exists with complexity far lower than the memorizing solution, can scaling automatically discover it?

**Starting Point**: This work studies the problem within a rigorous mathematical framework — formally defining compositional task families, designing the Hyperteacher as a general-purpose testbed, proving the existence and efficiency of generalizing solutions, and empirically verifying that scaling does lead to compositional generalization. The core idea is that "the algorithmic complexity of the generalizing solution scales linearly with the number of modules $M$, whereas the memorizing solution scales exponentially, so the generalizing solution becomes increasingly favored as scale grows."

## Method

### Overall Architecture

The research paradigm proceeds as follows: (1) formally define compositional task families (Definition 2.1) and compositional generalization (Definition 2.2); (2) instantiate a general compositional task family via the Hyperteacher — selecting $K$ from $M$ modules to produce $\mathcal{O}(M^K)$ tasks; (3) train standard MLPs on a subset of tasks and evaluate generalization to unseen compositional tasks; (4) vary data and model scale to observe the emergence of compositional generalization; (5) probe hidden layers with linear decoders to assess whether task components form linearly decodable representations.

### Key Designs

1. **Compositional Task Family & Hyperteacher (Experimental Testbed)**:
    - **Function**: Constructs a precisely controlled synthetic task family as an experimental platform for studying compositional generalization.
    - **Mechanism**: The Hyperteacher is defined as a linear hypernetwork: $C(\mathbf{z}) = \mathbf{x} \mapsto \mathbf{\Omega} \text{ReLU}(\sum_{k:z_k \neq 0} z_k \mathbf{\Theta}_k \mathbf{x})$, where $\{\mathbf{\Theta}_m\}_{m=1}^M$ are $M$ weight matrices and $\mathbf{\Omega}$ is a shared readout projection. Each task is specified by a composition vector $\mathbf{z}$ selecting $K$ modules; the task function $f_{\mathbf{z}}$ is a single-hidden-layer ReLU network. The composition operator $C$ satisfies injectivity (each combination yields a distinct task) and low complexity (the program length implementing $C$ is sub-exponential in $K$).
    - **Design Motivation**: The flexibility of neural networks allows the Hyperteacher to cover a broad range of functional behaviors while maintaining controllable compositional structure. Supplementary experiments replicate all findings on the Compositional Preference task family (grid world with compositional rewards).

2. **Theoretical Guarantee for Generalizing Solutions (Theorem 3.1)**:
    - **Function**: Proves that a ReLU MLP can approximate the Hyperteacher using a number of neurons linear in $M$, whereas memorizing all tasks requires exponential capacity.
    - **Mechanism**: For any $M, \varepsilon > 0$, on a compact input set, there exists a ReLU MLP with $\mathcal{O}(\frac{1}{\sqrt{\varepsilon}} + M)$ neurons that approximates the Hyperteacher to precision $\varepsilon$ in the $\|\cdot\|_\infty$ norm. Crucially, the generalizing solution scales **linearly with $M$**, while memorizing $\mathcal{O}(M^K)$ tasks requires exponential capacity — thus the efficiency advantage of the generalizing solution grows exponentially as $M$ increases.
    - **Design Motivation**: Provides theoretical support for the claim that "increased scale leads to discovery of the generalizing solution": SGD exhibits a simplicity bias toward low-complexity solutions, and the generalizing solution is far simpler than the memorizing one.

3. **Linear Decodability Metric (Theorem 4.1 + Empirical Validation)**:
    - **Function**: Proves that task components must be decodable from model representations when compositional generalization succeeds; experiments reveal the stronger finding that decoding from hidden-layer activations is **linear**.
    - **Mechanism**: Theorem 4.1 proves that if a model predicts correctly with probability $1-\delta$, there exists a decoder that recovers task components from task encodings with probability $1-\sqrt{\delta}$. Linear probe experiments find: (a) compositional generalization $R^2$ and linear decodability $R^2$ are highly correlated ($R^2 > 0.95$); (b) even when models are directly provided with task components (identity encoding), networks that fail to generalize compositionality lose this information in deeper layers — possessing disentangled representations does **not** imply compositional generalization.
    - **Design Motivation**: Provides an actionable diagnostic tool: the compositional generalization capacity of a model can be predicted by examining whether task components are linearly decodable from its hidden layers. Validation on diffusion models confirms that linear decodability correlates positively with compositional success in image generation.

### Loss & Training

Standard MLPs with ReLU activations are used, taking as input the concatenation of data point $\mathbf{x}$ and task encoding $\varphi(\mathbf{z}, r)$. Six task encodings are considered: Identity, Orthogonal, Language, Invertible NN, Interval Shuffle, and Few-shot. The training distribution must satisfy two conditions: "compositional support" (each module appears in at least some training tasks) and "connected support" (no subset of modules appears exclusively in isolation).

## Key Experimental Results

### Main Results

| Task Encoding | Task Decoder $R^2$ | Input Decoder $R^2$ | Comp. Gen. $R^2$ |
|---|---|---|---|
| Identity | 0.95 | 1.00 | 1.00 |
| Orthogonal | 0.96 | 1.00 | 1.00 |
| Language | 0.99 | 1.00 | 1.00 |
| Invertible NN | 0.94 | 0.56 | 0.95 |
| Interval Shuffle | 0.96 | 0.73 | 0.98 |
| Few-shot | 0.90 | -0.23 | 0.97 |

### Ablation Study

| Configuration | Effect on Comp. Gen. | Notes |
|---|---|---|
| Increase number of training tasks | Monotonic improvement | Required training tasks grow sub-exponentially w.r.t. total tasks |
| Increase model width | Monotonic improvement | All task encodings benefit |
| Increase model depth | Monotonic improvement | Deeper models perform better |
| Violate compositional support | Significant degradation | Absent module → cannot learn that module |
| Violate connected support | Significant degradation | Isolated module subsets → cannot generalize to new combinations |
| Few modules appear at low frequency | Noticeable degradation | Root cause is under-representation, not imbalance per se |
| Transformer architecture | Better data efficiency | Fewer training tasks needed to reach equivalent generalization |

### Key Findings

- The number of training tasks required for compositional generalization grows sub-exponentially with the total number of tasks, consistent with Definition 2.2.
- All six task encodings (including the highly nonlinear few-shot encoding) lead to compositional generalization given sufficiently large models.
- A strong positive correlation exists between linear decodability and compositional generalization, validated on image generation models (e.g., Stable Diffusion).
- Networks that fail to generalize compositionally under identity encoding lose component information in deeper layers — "seeing the components" does not imply "being able to compose them."
- Low-frequency occurrence of a minority of modules in the training distribution degrades generalization, whereas high-frequency occurrence of a minority does not — the key factor is avoiding under-representation.

## Highlights & Insights

- **Counterintuitive finding**: Standard MLPs can achieve compositional generalization without special modular design, challenging the long-standing view that "compositionality requires symbolic mechanisms." Theory and experiment align precisely — Theorem 3.1 establishes the existence of a linearly complex generalizing solution, and experiments confirm that scaling indeed discovers it.
- Linear decodability as a diagnostic tool for compositional generalization has direct practical value: it can identify which concept combinations a diffusion model will fail on (low decodability → low compositional success rate) without the need to generate large numbers of images.
- **Connection to biology**: Any information-preserving mapping (not only language) suffices for compositional generalization, potentially explaining how animals lacking complex language achieve compositional generalization — language is not a necessary condition for compositionality.

## Limitations & Future Work

- The theoretical conditions under which SGD is guaranteed to find the generalizing solution (rather than the memorizing one) remain unspecified; the argument relies on an empirical assumption of simplicity bias.
- Only the case where tasks are fully specified is considered; ambiguous or incomplete task descriptions are not addressed.
- Scaling experiments on synthetic data require knowledge of the generative process; how to design sufficiently covering training distributions on real data remains an open problem.
- Validation experiments in NLP are absent (only synthetic tasks and image generation are evaluated).

## Related Work & Insights

- **vs. Fodor & Pylyshyn (1988)**: The classical critique holds that connectionist models cannot achieve systematic compositionality — this paper demonstrates that they can, at sufficient scale.
- **vs. Boopathy et al. (2025)**: Modular architectures can break scaling laws — this paper complements that finding by showing that even without modularity, scaling alone can achieve compositional generalization, though architectural inductive biases improve data efficiency.
- **vs. image generation literature**: A new perspective on compositional failure in diffusion models is provided — the issue is not that "the model is too small" but that "the hidden layers have not formed linearly decodable component representations."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Addresses a fundamental question through the combination of theoretical proof and large-scale experiments.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage on synthetic tasks; image generation validation is convincing; NLP validation is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Definitions are rigorous, arguments are clear, and figures are well-crafted.
- Value: ⭐⭐⭐⭐ The linear decodability metric can be directly applied to diagnose compositional capacity in generative models.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Scaling Diffusion Transformers Efficiently via μP](scaling_diffusion_transformers_efficiently_via_μp.md)
- [\[NeurIPS 2025\] Graph Diffusion that can Insert and Delete](graph_diffusion_that_can_insert_and_delete.md)
- [\[NeurIPS 2025\] A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](a_closer_look_at_model_collapse_from_a_generalization-to-memorization_perspectiv.md)
- [\[NeurIPS 2025\] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)
- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)

<!-- RELATED:END -->
