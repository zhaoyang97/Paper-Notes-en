---
title: >-
  [Paper Note] Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking
description: >-
  [NeurIPS 2025][LLM Pretraining][neural collapse] Using grokking (delayed generalization) as a causal probe, this paper demonstrates that **relative flatness is a (potentially) necessary condition for generalization**…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "neural collapse"
  - "relative flatness"
  - "generalization"
  - "grokking"
  - "loss landscape"
date: 2026-05-08
content_hash: 0b63efe128a5f90e
---

# Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking

**Conference**: NeurIPS 2025
**arXiv**: [2509.17738](https://arxiv.org/abs/2509.17738)  
**Code**: [GitHub](https://github.com/TrustworthyMachineLearning-Lab/grokking_flatness)  
**Authors**: Ting Han, Linara Adilova, Henning Petzka, Jens Kleesiek, Michael Kamp
**Affiliations**: TU Dortmund / Lamarr Institute / Ruhr University Bochum / UK Essen
**Area**: LLM Pretraining
**Keywords**: neural collapse, relative flatness, generalization, grokking, loss landscape

## TL;DR

Using grokking (delayed generalization) as a causal probe, this paper demonstrates that **relative flatness is a (potentially) necessary condition for generalization**, whereas neural collapse, despite frequently co-occurring with generalization, is not necessary — it is merely one pathway toward flatness.

---

## Background & Motivation

### Core Puzzle

Overparameterized deep networks can memorize arbitrary labels (Zhang et al., 2017), yet generalize well on natural data. Two widely discussed geometric signatures of generalization are:

**Neural Collapse (NC)**: In the terminal phase of training, penultimate-layer features collapse to class means (NC1), the means form a simplex ETF (NC2), classifier weights align with the means (NC3), and the classifier degenerates to a nearest-neighbor rule (NC4). This has been observed across numerous architectures and is regarded as a generalization indicator.

**Loss landscape flatness**: Insensitivity of the loss to parameter perturbations. Classical Hessian-based measures suffer from reparameterization sensitivity (Dinh et al., 2017); **relative flatness**, proposed by Petzka et al. (2021), addresses this issue and is invariant to reparameterization.

Both phenomena emerge in the terminal phase of training and correlate with generalization, but the core question remains: **are they causes or byproducts of generalization, and which is the more fundamental driver?**

### Limitations of Prior Work

- NC is contested on imbalanced data: Fang et al. (2021) find that minority collapse leads to classification failure; Hui et al. (2022) show that test-set NC hurts transfer learning performance; Hong & Ling (2024) demonstrate that a sufficiently high SNR is required for NC to correlate positively with generalization.
- Classical flatness measures (e.g., Hessian trace) are sensitive to reparameterization and computationally expensive. SAM-based methods have also been criticized (Andriushchenko et al., 2023).
- The causal relationship between the two has never been rigorously decoupled — prior work observes only co-occurrence and cannot establish necessity.

### Key Insight: Grokking as a Causal Probe

**Grokking** refers to the phenomenon where a network generalizes suddenly, long after it has fully memorized the training set (Power et al., 2022). This phenomenon **temporally separates memorization from generalization**, enabling causal inference by tracking when NC and flatness each emerge — this constitutes the paper's central methodological innovation.

---

## Method

### 1. Metric Definitions

#### NCC (Neural Collapse Clustering)

The paper adopts the simplified metric of Galanti et al. (2022), capturing within-class compactness and between-class separation:

$$\mathrm{NCC} = \sum_{c \neq c'} \frac{V_c + V_{c'}}{2\|\mu_c - \mu_{c'}\|^2}$$

where $\mu_c$ is the class mean of features for class $c$ and $V_c$ is the within-class variance. Low NCC indicates tight within-class clustering and well-separated class means, consistent with NC.

#### Relative Flatness (κ)

The reparameterization-invariant measure proposed by Petzka et al. (2021):

$$\kappa^{\phi}_{\mathrm{Tr}}(\mathbf{w}) = \sum_{s,s'=1}^{d} \langle \mathbf{w}_s, \mathbf{w}_{s'} \rangle \cdot \mathrm{Tr}(H_{s,s'}(\mathbf{w}, \phi(S)))$$

where $H_{s,s'}$ is the Hessian of the empirical loss with respect to rows of the weight matrix. This measure is invariant to neuron-wise rescaling and orthogonal transformations. Small κ indicates a flat solution and predicts good generalization. In practice, a closed-form upper bound derived by Walter et al. (2024) under cross-entropy loss is used, requiring only the penultimate layer and scaling independently of model size.

### 2. Grokking Experiments: Temporal Decoupling

A 2-layer Transformer is trained on a modular arithmetic task ($x + y \mod p$) with AdamW (lr=1e-4, wd=1.0) for $10^6$ steps and a 50/50 train/val split.

**Key observations** (Figure 1):
- **NCC** begins to decrease during the memorization phase — collapse signatures appear before the network generalizes.
- **Relative flatness** remains high during memorization and drops sharply only when generalization begins.
- Conclusion: both correlate with generalization, but NC **precedes generalization**, while flatness **emerges concurrently with generalization**.

### 3. NC is Not Necessary: Regularization-Based Suppression

On ResNet-18 / CIFAR-10, an NCC regularizer is applied:

$$\mathcal{L}_{\text{NC\_REG}} = \mathcal{L}_{\text{CE}} - \lambda \cdot \text{NCC}$$

By maximizing NCC, collapse is **actively suppressed** (shrinking inter-class mean distances and enlarging within-class variance).

**Results** (Figure 2):
- **Train/test accuracy is unaffected**: even as NCC continuously increases (collapse suppressed), generalization performance is unchanged.
- **Relative flatness is unaffected**: confirming that flatness does not depend on NC.
- Additionally, Figure 3 confirms that when NC is suppressed, cluster angles deviate from the optimal 10-simplex configuration, validating the NCC metric.

### 4. Flatness is Necessary: Inducing Delayed Generalization

A relative flatness regularizer is used to encourage sharp minima:

$$\mathcal{L}_{\text{RF\_REG}} = \mathcal{L}_{\text{CE}} - \lambda \cdot \kappa^{\phi}_{Tr}(\mathbf{w})$$

This is validated across the following architectures and datasets:
- **ResNet-18 / CIFAR-10**: validation accuracy is low before the regularizer is removed at epoch 200, after which it quickly recovers.
- **ViT / ImageNet-100**: generalization similarly recovers after removing the regularizer at epoch 150.
- **TinyBERT, DistilGPT2 / SST-5**: delayed generalization is also induced in pretrained language models.

**Core finding**: suppressing flatness **artificially induces grokking**, reproducing it in architectures and datasets where it does not naturally occur. This directly establishes the necessity of flatness for generalization.

### 5. Theoretical Contribution: NC Implies Flatness

**Proposition 5.3**: Under classical NC assumptions (NC1–NC4), relative flatness admits the upper bound:

$$\kappa_{\phi}(w) \leq \lambda^2 k^3 M^4 \cdot \frac{e^{-\lambda M^2 \cdot \frac{k}{k-1}}}{(1 + (k-1)e^{-\lambda M^2 \cdot \frac{k}{k-1}})^2}$$

For sufficiently large $\lambda$, $\kappa_{\phi}(w) \lesssim \lambda^2 k^3 M^4 e^{-\lambda M^2 \cdot k/(k-1)}$ — an **exponential decay**.

Proof sketch:
1. Under NC conditions, it is first shown that the bias plus the projection of class means onto the global mean is constant (Lemma A.1).
2. It is then proved that $\kappa_{\phi}(w) \leq \|w\|^2 \mathrm{Tr}(H(w))$ (Lemma A.2, using Hessian positive semi-definiteness and Cauchy-Schwarz).
3. Under the NC structure, the closed-form expression for the Hessian trace is expanded and the softmax margin $\delta = M^2 \cdot k/(k-1)$ is used to obtain the exponential decay bound.

**Significance**: NC is a sufficient pathway to flatness, explaining their empirical co-occurrence; however, flatness can also be achieved through other mechanisms.

### 6. The Role of Representativeness

Theoretical guarantees for flatness also rely on **representativeness** (the degree to which training features represent the true distribution). Appendix D confirms that representativeness improves only when generalization begins — implying that neither flatness nor NC can guarantee generalization in the absence of feature alignment.

---

## Key Experimental Results

### Table 1: Quantitative Results for Flatness Regularization-Induced Delayed Generalization

| Setting | κ Before Regularizer Removal | Final κ | Validation Accuracy Under Regularization | Best Validation Accuracy |
|---------|------------------------------|---------|------------------------------------------|--------------------------|
| ResNet-18 / CIFAR-10 (epoch 199) | 2209.94 | 1.91 | ~60% | ~78% |
| ViT / ImageNet-100 (epoch 149) | 22212.75 | 313.22 | ~38% | ~43% |

After removing the regularizer, κ drops substantially and validation accuracy recovers to baseline, directly demonstrating a causal link between flatness and generalization.

### Table 2: Effect of Different λ Values in the NC Suppression Experiment

| NCC Reg. λ | NCC Behavior | Effect on Validation Accuracy | Effect on Relative Flatness | Training Stability |
|-----------|--------------|------------------------------|-----------------------------|--------------------|
| 1×10⁻² | NCC increases | Severe drop (<60%) | Unstable | Training collapse (~epoch 70) |
| 1×10⁻³ | NCC continuously increases | **Unaffected** (~78%) | **Unaffected** | Stable |
| 1×10⁻⁴ | Almost no change | Unaffected | Unaffected | Stable |
| 1×10⁻⁵ | No change | Unaffected | Unaffected | Stable |

The setting λ=10⁻³ is the most critical: it successfully suppresses NC (increasing NCC) while leaving generalization and flatness completely unaffected.

---

## Highlights & Insights

1. **Methodological innovation**: Grokking is elevated from an "interesting anomaly" to a "causal analysis tool." Temporal separation is used to decouple correlated variables — this paradigm generalizes to studying other deep learning phenomena.
2. **Artificially induced grokking**: This is the first demonstration that delayed generalization can be induced via flatness regularization in standard settings such as ResNet/CIFAR-10, ViT/ImageNet, and pretrained LMs/SST-5, spanning vision and NLP modalities.
3. **Theoretical unification**: The exponential decay bound connecting NC to flatness integrates two independent lines of research into a single geometric framework. The proof exploits the specific structure of softmax-CE and does not rely on the unconstrained features model or layer-peeling approximations.
4. **Practical implications**: Relative flatness can serve as a single-layer diagnostic metric (computation independent of model size), offering a new tool for training monitoring and regularization design.

---

## Limitations & Future Work

1. **Restricted to classification**: All experiments involve classification networks and generative language models (CE loss); applicability to contrastive pretraining, regression tasks, or large-scale LLMs remains unclear.
2. **Representativeness assumption**: Theoretical guarantees for flatness rely on two assumptions — "locally constant labels" and "representative features" — which may not hold in structured prediction, high-noise regression, or complex scenarios.
3. **Fragility of the regularizer**: The flatness regularizer is in inherent tension with CE loss (CE reduces predictive uncertainty, while the regularizer is effective only under high uncertainty), making λ tuning sensitive, with excessively large values causing training instability.
4. **Induced grokking may differ mechanistically from natural grokking**: Whether artificially delayed generalization truly reproduces the internal mechanisms of natural grokking remains to be verified.
5. **Necessity of flatness is an empirical conclusion**: The paper uses the phrase "potentially necessary," and a rigorous mathematical proof of strict necessity has not been established.

---

## Related Work & Insights

- **Tension between NC and transfer learning**: Hui et al. (2022) find that test-set NC hurts downstream tasks, consistent with this paper's finding that NC is not necessary. This suggests that excessive NC may over-compress the representation space, discarding fine-grained information.
- **Limitations of SAM**: Although SAM improves performance on vision tasks, Andriushchenko & Flammarion (2022) find that it does not exclusively encourage flat minima. Relative flatness provides a more reliable measure.
- **Two-phase dynamics of grokking**: Kumar et al. (2024) interpret grokking as a transition from lazy to rich training; this paper provides a complementary geometric perspective — the rich regime corresponds to the emergence of flat solutions.
- **Implications for flatness-aware training**: Encouraging flat minima does not substantially improve generalization (SGD already has an implicit bias toward flatness), but suppressing flatness reliably delays generalization. This asymmetry suggests that flatness is better suited as a diagnostic tool than as an optimization target.

---

## Rating

| Dimension | Score (1–10) |
|-----------|-------------|
| Novelty | 8 — Using grokking as a causal probe is a novel methodological contribution to generalization theory |
| Theoretical Depth | 8 — Rigorous proof of NC→flatness and analysis of representativeness |
| Experimental Thoroughness | 8 — Comprehensive ablations across vision/NLP and multiple architectures |
| Writing Quality | 9 — Tight argumentation progressing seamlessly from observation to intervention to theory |
| Value | 6 — Relative flatness as a diagnostic metric is valuable, but its guidance for training optimization is limited |
| **Overall** | **7.8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Explaining Grokking and Information Bottleneck through Neural Collapse Emergence](../../ICLR2026/llm_pretraining/explaining_grokking_and_information_bottleneck_through_neural_collapse_emergence.md)
- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[NeurIPS 2025\] Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)
- [\[NeurIPS 2025\] Superposition Yields Robust Neural Scaling](superposition_yields_robust_neural_scaling.md)

</div>

<!-- RELATED:END -->
