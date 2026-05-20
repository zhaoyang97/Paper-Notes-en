---
title: >-
  [Paper Note] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models
description: >-
  [ICLR 2026][Interpretability][Truncated Polynomial Classifier] This paper proposes the Truncated Polynomial Classifier (TPC), which enables dynamic safety monitoring by training a polynomial over LLM activation spaces or…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Truncated Polynomial Classifier"
  - "Safety Monitoring"
  - "Dynamic Inference"
  - "Linear Probes"
  - "Activation Space"
date: 2026-05-08
content_hash: e565eca2dfed4c6e
---

# Beyond Linear Probes: Dynamic Safety Monitoring for Language Models

**Conference**: ICLR 2026
**arXiv**: [2509.26238](https://arxiv.org/abs/2509.26238)  
**Code**: [https://github.com/james-oldfield/tpc](https://github.com/james-oldfield/tpc)  
**Area**: Model Safety / Activation Space Monitoring / AI Safety
**Keywords**: Truncated Polynomial Classifier, Safety Monitoring, Dynamic Inference, Linear Probes, Activation Space

## TL;DR
This paper proposes the Truncated Polynomial Classifier (TPC), which enables dynamic safety monitoring by training a polynomial over LLM activation spaces order-by-order and evaluating via truncation at inference time. Low-order truncations (≈ linear probes) handle easy inputs quickly, while higher-order terms provide stronger protection for difficult inputs. TPC matches or outperforms MLP baselines on WildGuardMix and BeaverTails while offering built-in interpretability.

## Background & Motivation

**Background**: LLM safety monitoring is dominated by two paradigms — LLM-as-Judge natural language auditing (expensive but powerful) and activation-space linear probes (lightweight but static). The former incurs a fixed high cost per query; the latter provides only a minimal, static line of defense.

**Limitations of Prior Work**:
- Linear probes are static and cannot adjust protection strength based on input difficulty or available compute budget.
- LLM-as-Judge approaches are too costly to serve as always-on monitors.
- Recent cascading work (e.g., McKenzie et al., 2025) still requires additional LLM fine-tuning/prompting and extra inference calls.
- The linear representation hypothesis assumes high-level concepts are encoded in one-dimensional subspaces, yet growing evidence suggests not all features exhibit simple linear structure.

**Key Challenge**: Safety monitoring faces an inherent cost–accuracy trade-off. Most requests are benign and require no strong protection, yet a small fraction of ambiguous or adversarial requests demands greater discriminative power. Existing methods either process all inputs at maximum cost or at minimum accuracy.

**Goal**:
- How can a single safety monitor operate effectively across varying compute budgets?
- How can a monitor quickly pass easy inputs while deeply inspecting difficult ones?
- How can classification capability be improved while preserving interpretability (compared to black-box MLPs)?

**Key Insight**: Drawing inspiration from test-time compute scaling — compute should be allocated dynamically at inference time rather than fixed in advance. Polynomials possess a naturally additive, order-by-order structure that is well suited to progressive computation.

**Core Idea**: Generalize linear probes into truncatable polynomial classifiers. Order-by-order training produces a family of nested sub-models; at inference time, evaluation is truncated on demand — low orders recover the linear probe, while higher orders provide stronger protection.

## Method

### Overall Architecture
The input is a residual-stream representation $\bm{z} \in \mathbb{R}^D$ extracted from a chosen LLM layer (mean-pooled over tokens), and the output is a binary harmful/benign classification probability. The core model is an $N$-th order polynomial that can be truncated at inference time to any sub-model of order $n \leq N$:

$$P_{:n}^{[N]}(\bm{z}) = w^{[0]} + \bm{z}^\top \bm{w}^{[1]} + \sum_{k=2}^{n} \sum_{r=1}^{R} \lambda_r^{[k]} (\bm{z}^\top \bm{u}_r^{[k]})^k$$

### Key Designs

1. **Truncated Polynomial Classifier (TPC)**:

    - **Function**: Models high-order interactions among neurons in the activation space using an $N$-th order polynomial, replacing the linear probe.
    - **Mechanism**: At $n=1$, the model degrades to a standard linear probe $w^{[0]} + \bm{z}^\top \bm{w}^{[1]}$. Each additional order $k$ introduces multiplicative interaction terms among $k$ neurons. High-order weight tensors are parameterized via symmetric CP decomposition: $\mathcal{W}^{[k]} = \sum_{r=1}^{R} \lambda_r^{[k]} (\bm{u}_r^{[k]} \circ \cdots \circ \bm{u}_r^{[k]})$, requiring only $O(DR)$ parameters per order.
    - **Design Motivation**: The additive structure of polynomials means higher-order terms act as fine-grained corrections to lower-order logits, naturally supporting truncated evaluation. Symmetric decomposition eliminates redundant parameterization of equivalent monomials.

2. **Progressive Training**:

    - **Function**: Trains polynomial orders sequentially so that every truncated sub-model is itself a competent classifier.
    - **Mechanism**: The $k$-th order parameters $\bm{\theta}^{[k]}$ are learned by minimizing the BCE loss of the polynomial truncated at order $k$, while all parameters from orders $1$ through $k-1$ are frozen. First-order weights are initialized from a pre-trained sklearn linear probe.
    - **Design Motivation**: Directly training a full $N$-th order polynomial and truncating afterward yields unstable sub-model performance (confirmed experimentally). Progressive training guarantees that each truncation point is a valid classifier and that adding a new order does not degrade existing truncations.

3. **Cascading Defense**:

    - **Function**: Dynamically determines the number of orders to evaluate based on input difficulty — easy inputs exit early at low order, difficult inputs proceed to higher orders.
    - **Mechanism**: Evaluation begins at $n=1$ and proceeds order by order. At each order, the condition $\sigma(s) \in (\tau, 1-\tau)$ is checked (where $\tau$ is a confidence threshold). If the current prediction is sufficiently confident (probability falls outside the threshold band), the prediction is immediately output; otherwise evaluation continues to the next order. This is analogous to early-exit strategies in deep networks.
    - **Design Motivation**: The vast majority of requests are benign and can be classified with high confidence by a linear probe. Only a small fraction of ambiguous or adversarial inputs require the stronger discriminative power of higher-order terms. Experiments show that at medium-to-high $\tau$ values, cascade performance approaches that of the full polynomial while the net parameter count remains only slightly above that of a linear probe.

4. **Built-in Feature Attribution**:

    - **Function**: Leverages the explicit polynomial form to attribute classification decisions to individual neuron interactions.
    - **Mechanism**: The contribution of the second-order term can be decomposed as $c_{ij} = (w_{ij}^{[2]} + w_{ji}^{[2]}) z_i z_j$, directly quantifying the contribution of any neuron pair $(i,j)$ to the classification logit.
    - **Design Motivation**: MLPs are black boxes and cannot trace decisions back to specific neuron interactions. The polynomial form of TPC is inherently interpretable — one can precisely state, for example, that "the interaction between neuron 4830 and neuron 4916 increases the harmful classification logit by 0.005."

### Loss & Training
- Standard BCE loss is used to train each order, with all preceding orders frozen.
- First-order weights are initialized from a sklearn linear probe.
- Experiments use $N=5$, CP rank $R=64$, and 5 random seeds.
- Activations are extracted from intermediate layers (L32/L40 for gemma-3; L16/L20 for gpt-oss/llama).

## Key Experimental Results

### Main Results (WildGuardMix Static Evaluation, Test F1%)

| Method | gemma-3-27B | Qwen3-30B | gpt-oss-20b | Llama-3.2-3B |
|--------|------------|-----------|-------------|-------------|
| Linear probe | 88.03 | 85.53 | 86.70 | 83.24 |
| Bilinear probe | 88.79 | 84.87 | 87.13 | 84.78 |
| MLP | 88.49 | 85.48 | 87.86 | 83.77 |
| EE-MLP (5th exit) | 88.39 | 85.24 | 87.31 | 83.84 |
| **TPC (5th order)** | **88.86** | **85.57** | **88.05** | **84.48** |

### Cascading Defense Results (gemma-3-27B, L40)

| Configuration | Net Parameters | F1 | Notes |
|---------------|---------------|-----|-------|
| Linear probe only ($n=1$) | Baseline | ~88.0 | All inputs use linear probe |
| Full TPC ($n=5$, no cascade) | 5× | ~88.9 | All inputs use full polynomial |
| Cascade ($\tau$ = medium-high) | ~1.1× | ~88.8 | Most inputs exit at low order |
| Cascade ($\tau$ = high) | ~1.3× | ~88.9 | Approaches full polynomial performance |

### Key Findings
- **TPC outperforms all baselines on WildGuardMix across all models** (including parameter-matched MLPs); on BeaverTails it is on par with EE-MLP.
- On specific harmful categories, fixed-order TPC achieves up to 10% accuracy gain over linear probes and up to 6% over MLPs.
- **Cascading evaluation is the most notable contribution**: at medium-to-high $\tau$ values, performance approaches the full polynomial while net parameter count is only marginally above a linear probe — stronger protection obtained at nearly zero additional cost.
- Progressive training vs. direct training: directly training the full polynomial and truncating yields unstable performance at each truncation point; progressive training ensures every truncation point is a valid classifier.
- Neuron-pair attribution from the second-order TPC can explain classification decisions (e.g., the neuron 4830×2483 interaction increases the harmful logit for a nuclear-weapon prompt).

## Highlights & Insights
- **The "one model, multiple safety budgets" concept** is the central insight of this work — it transplants the idea of test-time compute scaling into safety monitoring and realizes it naturally through the truncation property of polynomials. This design principle is transferable to any classification task requiring flexible accuracy.
- **The progressive training scheme** elegantly resolves the train–evaluation inconsistency inherent to truncated polynomials. Analogous to greedy layer-wise training in deep networks, but applied along the polynomial order dimension — lower orders remain independently functional, while higher orders serve only as incremental refinements.
- **Symmetric CP decomposition** simultaneously addresses parameter explosion in high-order tensors and provides interpretable neuron-interaction attribution. Precise attribution of specific neuron-pair contributions to a decision — impossible with traditional MLPs — arises naturally in TPC.

## Limitations & Future Work
- Low-data regimes are not explored — high-order polynomials are prone to overfitting and may require stronger regularization.
- Although neuron-pair attribution is mechanistically faithful, it lacks human-readable semantic interpretation — "neuron 4830×4916 interaction" does not by itself explain *why* the decision is made.
- Performance does not increase monotonically with order, and all activation-based monitors require a manual search for the appropriate layer.
- Experiments are limited to prompt-level binary classification; generalization to finer-grained safety categorization (e.g., specific harmful category detection) or response-level monitoring has not been validated.
- **Future directions**: Applying polynomial expansion in the SAE feature space may simultaneously yield sparsity and interpretability; multi-layer probe ensembling could eliminate the need for manual layer selection.

## Related Work & Insights
- **vs. Linear Probes (Alain & Bengio, 2017)**: Linear probes are a special case of TPC at $n=1$. TPC retains all advantages of linear probes (lightweight, interpretable) while providing stronger classification capacity on demand via higher-order terms.
- **vs. McKenzie et al. (2025) cascading approach**: Their method cascades a linear probe with an external LLM, requiring additional LLM fine-tuning. TPC implements multi-level cascading within a single polynomial, requiring no external model and resulting in a significantly lighter system.
- **vs. MLP Probes**: MLPs may be more expressive but are black boxes. TPC achieves comparable or superior performance at matched parameter counts while providing built-in neuron-interaction attribution.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Polynomial probes are not new in isolation, but the combination of truncated evaluation + progressive training + cascading defense in the context of safety monitoring is a first, and the design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four models (up to 30B), two large-scale datasets, multi-layer sweeps, cascade ablations, progressive vs. direct training comparisons, and attribution visualizations — comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, rigorous notation, and an intuitive Figure 1; some notation is slightly redundant.
- **Value**: ⭐⭐⭐⭐ Provides a practical dynamic solution for LLM safety monitoring; cascading defense carries significant deployment value — achieving nonlinear-probe performance at approximately linear-probe cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GAVEL: Towards Rule-Based Safety through Activation Monitoring](gavel_towards_rule-based_safety_through_activation_monitoring.md)
- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text Alignment](dynamic_reflections_probing_video_representations_with_text_alignment.md)
- [\[NeurIPS 2025\] Emergence of Linear Truth Encodings in Language Models](../../NeurIPS2025/interpretability/emergence_of_linear_truth_encodings_in_language_models.md)
- [\[ICLR 2026\] Internal Planning in Language Models: Characterizing Horizon and Branch Awareness](internal_planning_in_language_models_characterizing_horizon_and_branch_awareness.md)
- [\[ICLR 2026\] Universal Properties of Activation Sparsity in Modern Large Language Models](universal_properties_of_activation_sparsity_in_modern_large_language_models.md)

</div>

<!-- RELATED:END -->
