---
title: >-
  [Paper Note] Geometric Priors for Generalizable World Models via Vector Symbolic Architecture
description: >-
  [NeurIPS 2025][Interpretability][World Models] This paper proposes introducing Fourier Holographic Reduced Representation (FHRR) from Vector Symbolic Architecture (VSA) as geometric priors into world models. By modeling state transitions via element-wise complex multiplication, it achieves an 87.5% zero-shot generalization accuracy on discrete GridWorld and a noise robustness four times greater than MLPs.
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "World Models"
  - "Vector Symbolic Architecture"
  - "Hyperdimensional Computing"
  - "FHRR"
  - "Geometric Deep Learning"
  - "Neurosymbolic AI"
date: 2026-05-08
content_hash: aee678abbf29b998
---

# Geometric Priors for Generalizable World Models via Vector Symbolic Architecture

**Conference**: NeurIPS 2025  
**arXiv**: [2602.21467](https://arxiv.org/abs/2602.21467)  
**Authors**: William Youngwoo Chung, Calvin Yeung, Hansen Jin Lillemark, Zhuowen Zou, Xiangjian Liu, Mohsen Imani (UC Irvine & UC San Diego)  
**Code**: Unreleased  
**Area**: Interpretability  
**Keywords**: World Models, Vector Symbolic Architecture, Hyperdimensional Computing, FHRR, Geometric Deep Learning, Neurosymbolic AI  

## TL;DR

This paper proposes introducing Fourier Holographic Reduced Representation (FHRR) from Vector Symbolic Architecture (VSA) as geometric priors into world models. By modeling state transitions via element-wise complex multiplication, it achieves an 87.5% zero-shot generalization accuracy on discrete GridWorld and a noise robustness four times greater than MLPs.

---

## Background & Motivation

### Problem Definition

World models learn the transition dynamics $T: S \times A \to S$ of an environment for planning and decision-making in reinforcement learning. Current mainstream approaches approximate the transition function using unstructured MLPs, which suffer from three core limitations:

**Low Sample Efficiency**: Black-box fitting requires extensive training data and fails to exploit inherent symmetries in the environment.

**Weak Generalization**: Models fail to extrapolate to unseen state-action combinations, yielding near-zero zero-shot accuracy.

**Long-Horizon Error Accumulation**: Errors grow exponentially during multi-step rollouts, as the latent space lacks an explicit geometric structure for error correction.

### Limitations of Prior Work

- **Model-based RL** (Ha & Schmidhuber 2018, Hafner 2019): Despite success in Atari and continuous control, transition functions remain unstructured mappings that lack interpretability.
- **Geometric Deep Learning** (Kipf 2019, Park 2022): While symmetry is introduced, the latent representations themselves lack algebraic composability, making direct vector-operation-based planning impossible.
- **VSA/Hyperdimensional Computing**: VSA has demonstrated potential in classification, time series, and graph reasoning, but remains unexplored in learnable transition modeling.

### Core Motivation

Biological systems leverage environmental symmetries and geometric structures to simplify learning (e.g., grid cells in the brain). If the latent space itself possesses a group structure, transitions can be executed via simple algebraic operations, inherently supporting composition, inversion, and error correction.

---

## Method

### 1. Group-Theoretic Formalization of Environmental Dynamics

The transition of a deterministic environment is modeled as a group action:

$$\cdot: G \times S \to S, \quad (g, s) \mapsto g \cdot s$$

Each action $a \in A$ corresponds to a generator $g_a$ of the group $G$, satisfying $T(s, a) = g_a \cdot s$. Group properties guarantee the existence of an identity element $e = g_a \circ g_a^{-1}$, and composition satisfies associativity $(g_1 \circ g_2) \cdot s = g_1 \cdot (g_2 \cdot s)$.

### 2. FHRR Encoder

A learnable FHRR encoder is used to map states and actions to a $D$-dimensional complex unit vector space $\mathcal{Z} = (S^1)^D$:

$$\phi_S(s) = [e^{i\theta_{j,s}^\top s}]_{j=1}^D, \quad \phi_A(a) = [e^{i\theta_{j,a}^\top a}]_{j=1}^D$$

where $\Theta_s \in \mathbb{C}^{D \times n_s}$ and $\Theta_a \in \mathbb{C}^{D \times n_a}$ are learnable projection parameters. Each component lies on the unit circle, and the overall representation forms a group $(\mathcal{Z}, \odot)$.

### 3. Latent Transition Model

State transitions are implemented via a binding operation (element-wise complex multiplication), rather than concatenation and feedforward operations of an MLP:

$$\phi_S(s_{t+1}) = \phi_S(s_t) \odot \phi_A(a_t)$$

In phase coordinates, this is equivalent to phase addition:

$$\Theta_s^\top s_{t+1} = \Theta_s^\top s_t + \Theta_a^\top a_t \pmod{2\pi}$$

Multi-step rollouts naturally extend to continuous binding (without recursive feedforward operations):

$$\phi_S(s_{t+k}) = \phi_S(s_t) \odot \prod_{j=1}^k \phi_A(a_{t+j-1})$$

Key difference from MLP: MLPs map concatenated state and action representations through non-linear layers, failing to disentangle their encodings. In contrast, FHRR preserves algebraic decomposability through element-wise multiplication.

### 4. Cleanup Error Correction Mechanism

A unique advantage of VSA is to correct accumulated error via nearest-neighbor search within a state codebook:

$$s^\star = \arg\max_{s \in \mathcal{S}} \operatorname{Re}\langle x, \Phi_s \rangle$$

The reliability is guaranteed by two geometric properties of the high-dimensional space:
- **Concentration of Self-Similarity**: The similarity between noisy and ground-truth embeddings is concentrated around 1, with a variance of $\mathcal{O}(1/D)$.
- **Concentration of Cross-Similarity**: Embeddings of different states are approximately orthogonal, with a separation margin of $\text{margin} \sim 1 - \mathcal{O}(1/\sqrt{D})$.

In practice, a state codebook matrix $\Phi \in \mathbb{C}^{|\mathcal{S}| \times D}$ is maintained, and the cleanup operation boils down to matrix-vector multiplication and argmax, incurring negligible overhead in discrete state spaces.

### 5. Training Objective

A weighted combination of three loss functions:

| Loss Term | Formula | Function |
|--------|------|------|
| Binding loss | $\mathcal{L}_{\text{bind}} = \|\phi_S(s_{t+1}) - \phi_S(s_t) \odot \phi_A(a_t)\|^2$ | Encourages transition equivariance |
| Invertibility loss | $\mathcal{L}_{\text{inv}} = \sum_{(a,a^{-1})} \|\phi_A(a) \odot \phi_A(a^{-1}) - \mathbf{1}\|^2$ | Encourages action representations to conform to a group structure |
| Orthogonality loss | $\mathcal{L}_{\text{ortho}} = \sum_{i \neq j} (\langle \phi_S(s_i), \phi_S(s_j) \rangle)^2$ | Encourages orthogonal separation between different state embeddings |

Total loss: $\mathcal{L} = \lambda_{\text{bind}} \mathcal{L}_{\text{bind}} + \lambda_{\text{inv}} \mathcal{L}_{\text{inv}} + \lambda_{\text{ortho}} \mathcal{L}_{\text{ortho}}$

Hyperparameters: $\lambda_{\text{bind}}=2, \lambda_{\text{inv}}=0.5, \lambda_{\text{ortho}}=0.05$, learning rate 0.007 (VSA) / 0.0005 (MLP), gradient clipping set to 1, trained for 500 epochs.

---

## Key Experimental Results

Experimental Setup: $10 \times 10$ GridWorld (100 discrete states, 4 deterministic actions: up, down, left, right), with an 80% training and 20% zero-shot split.

### Table 1: Comparison of Dynamics Modeling between VSA and MLP

| Metric | FHRR (Ours) | MLP-S | MLP-M | MLP-L |
|------|-------------|-------|-------|-------|
| 1-step Accuracy | **96.3%** | 80.0% | 80.0% | 80.25% |
| 1-step Zero-Shot | **87.5%** | 0.0% | 0.0% | 1.25% |
| Cosine Similarity | **83.0** | 79.5 | 79.9 | 80.6 |
| Zero-Shot Cosine Sim | **80.5** | 0.9 | 0.15 | 3.1 |
| 5-step Rollout | **74.6%** | 39.8% | 38.0% | 40.8% |
| 20-step Rollout | **34.6%** | 2.0% | 4.0% | 6.2% |
| 20-step Rollout + Cleanup | **61.4%** | 5.4% | 7.8% | 8.4% |
| 100-step Rollout | 1.8% | 0.8% | 1.8% | 2.0% |
| 100-step Rollout + Cleanup | **38.6%** | 2.8% | 4.0% | 3.2% |

### Table 2: Comparison of Parameter Count and Inference Speed

| Metric | VSA (HRR) | VSA (FHRR) | MLP-S | MLP-M | MLP-L |
|------|-----------|------------|-------|-------|-------|
| Parameter Count | 53,248 | 53,248 | 41,600 | 241,024 | 1,394,048 |
| Parameter Ratio | 1× | 1× | 0.8× | 4.5× | 26.2× |
| Inference Time (ms) | 0.2063 | 0.1528 | 0.1174 | 0.1715 | 0.3135 |
| Inference + Cleanup (ms) | 0.2632 | 0.2421 | 0.1743 | 0.2317 | 0.3761 |

---

## Highlights & Insights

1. **Dominant Zero-Shot Generalization over MLPs**: FHRR achieves 87.5% accuracy on unseen state-action pairs, whereas all three MLP scales yield near-zero performance. This indicates that scaling MLP parameter size (from 42K to 1.4M) cannot compensate for structural deficiencies.
2. **Cleanup Significantly Enhances Long-Horizon Prediction**: The 20-step rollout accuracy improves from 34.6% to 61.4% (+26.8%), which is 53 percentage points higher than the best-performing MLP + Cleanup baseline (8.4%).
3. **4x Noise Robustness Compared to MLPs**: Under Gaussian noise $\sigma \in [0, 5]$, FHRR maintains an 80%+ accuracy level while MLP-M degrades rapidly. Ablation studies show that increasing the dimension $D$ can further enhance robustness.
4. **Interpretable Latent Structure**: t-SNE visualizations display that FHRR learns an ordered embedding space that aligns with the grid's row-and-column structure, whereas the MLP's representations remain entirely unstructured.
5. **Extreme Parameter Efficiency**: FHRR uses only 53K parameters (comparable to MLP-S) and achieves an inference speed of 0.15 ms. Since all VSA operations are element-wise, they are highly conducive to hardware acceleration.
6. **Theoretical Connection to Random Fourier Features**: FHRR encoding is equivalent to RFF, satisfying $\langle \phi(x), \phi(y) \rangle / D \approx K(x - y)$, thereby unifying VSA with kernel methods.

---

## Limitations & Future Work

1. **Validation Limited to Small-Scale Discrete Environments**: The $10 \times 10$ GridWorld (100 states × 4 actions) is far removed from the complexity of real-world scenarios.
2. **Deterministic Transition Assumption**: Real-world environments are typically stochastic or partially observable, which cannot be modeled by the current framework.
3. **Cleanup Dependency on a Finite State Codebook**: Continuous state spaces require novel cleanup designs (such as approximate nearest neighbors), which are not addressed in this paper.
4. **Lack of RL/Planning Integration Experiments**: The paper does not demonstrate end-to-end performance in actual model-based RL tasks.
5. **Weak Baselines**: The framework is evaluated only against a vanilla MLP, lacking comparison with GNN-based, Attention-based, or other structured world models.
6. **High-Dimensional Observational Inputs Unverified**: Scalability under image-based or continuous perceptual inputs remains unknown.

---

## Related Work & Insights

- **World Models in Model-Based RL**: Ha & Schmidhuber (2018), Hafner et al. (2019/2020), Hansen et al. (2023, TD-MPC), etc., apply world models for reinforcement learning planning. However, their transition functions act as unstructured black boxes, suffering from rollout error accumulation and poor interpretability.
- **Geometric Deep Learning**: Kipf et al. (2019) and Park et al. (2022) introduce symmetry into object-centric world models. However, the latent representations lack algebraic composability, requiring full forward passes for planning.
- **Vector Symbolic Architecture (VSA)**: Kanerva (2009) and Kleyko et al. (2022) review applications of VSA/hyperdimensional computing in classification, time series, and graph reasoning; Yeung et al. (2025) construct cognitive maps with VSA; Ni et al. (2024) leverage VSA for RL classification. However, this work is the first to employ VSA for transition modeling in learnable world models.
- **FHRR and Kernel Methods**: Plate (2003) introduced the FHRR framework; Rahimi & Recht's Random Fourier Features established the theoretical links between FHRR encoding and kernel approximation.

---

## Rating

| Dimension | Rating | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | Using the algebraic structure of VSA/FHRR as a geometric prior for world models provides a novel perspective. |
| Theoretical Depth | ⭐⭐⭐⭐ | Comprehensive group-theoretic formulation, bridging group actions, equivariant representations, and kernel approximation. |
| Experimental Thoroughness | ⭐⭐ | Limited to a 10×10 GridWorld, using weak baselines and lacking evaluations on standard benchmarks. |
| Writing Quality | ⭐⭐⭐⭐ | Clear theoretical derivations, intuitive visualizations, and a highly cohesive structure. |
| Practical Value | ⭐⭐⭐ | The idea is promising, but validated scale is insufficient, leaving a gap before practical applicability. |
<!-- 由 src/gen_stubs.py 自动生成 -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)
- [\[AAAI 2026\] Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning](../../AAAI2026/interpretability/attention_as_binding_a_vector-symbolic_perspective_on_transformer_reasoning.md)
- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](towards_scaling_laws_for_symbolic_regression.md)
- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)
- [\[NeurIPS 2025\] Toward Real-world Text Image Forgery Localization: Structured and Interpretable Data Synthesis](toward_real-world_text_image_forgery_localization_structured_and_interpretable_d.md)

</div>

<!-- RELATED:END -->
