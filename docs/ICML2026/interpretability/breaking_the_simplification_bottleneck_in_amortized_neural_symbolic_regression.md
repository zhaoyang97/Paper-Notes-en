---
title: >-
  [Paper Note] Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression
description: >-
  [ICML 2026][Interpretability][Symbolic Regression] The authors propose SimpliPy (a rule-based simplification engine 100x faster than SymPy) and Flash-ANSR (a Transformer-based amortized symbolic regression framework). Fl…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Symbolic Regression"
  - "Expression Simplification"
  - "Transformer"
  - "Amortized Inference"
  - "Scientific Discovery"
date: 2026-05-08
content_hash: 0813e284f7976096
---

# Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression

**Conference**: ICML 2026  
**arXiv**: [2602.08885](https://arxiv.org/abs/2602.08885)  
**Code**: https://github.com/psaegert/flash-ansr  
**Area**: Interpretability  
**Keywords**: Symbolic Regression, Expression Simplification, Transformer, Amortized Inference, Scientific Discovery  

## TL;DR

The authors propose SimpliPy (a rule-based simplification engine 100x faster than SymPy) and Flash-ANSR (a Transformer-based amortized symbolic regression framework). Flash-ANSR matches or exceeds the genetic programming method PySR on the FastSRB benchmark with ~58% recovery rate, while generating progressively more concise expressions as the inference budget increases.

## Background & Motivation

**Background**: Symbolic Regression (SR) aims to discover interpretable analytical expressions from observed data. Traditional methods primarily rely on Genetic Programming (GP) (e.g., PySR), but these search from scratch for every dataset and fail to transfer structural knowledge across tasks. Amortized SR learns the posterior $p(\bm{\tau}|\mathcal{D})$ by pre-training Transformers on massive synthetic data, shifting the computational burden to a one-time pre-training stage.

**Limitations of Prior Work**: Amortized SR faces a triple dilemma. First, static corpora schemes (e.g., NeSymReS) use SymPy offline to generate fixed datasets (~100M expressions), but the high costs of simplification limit coverage and dimensionality ($D \leq 3$). Second, some methods (e.g., E2E) forgo simplification and train on unnormalized expressions, causing the model to waste capacity learning syntactic redundancies (e.g., $x+0$, $1 \cdot x$). Third, embedding SymPy into the training loop (e.g., NSRwH) introduces severe bottlenecks, as SymPy's median simplification time is ~100ms per expression.

**Key Challenge**: A fundamental contradiction exists between the quality and speed of expression simplification—general Computer Algebra Systems (CAS) utilize object-oriented parsing and tree traversal mechanisms that are too heavy for SR training, yet omitting simplification leads to redundant training targets and low inference efficiency.

**Goal**: To design a fast and high-quality simplification engine that breaks the CAS bottleneck, enabling amortized SR to scale to larger and higher-dimensional training sets.

**Key Insight**: The authors observe that expressions encountered in SR training exhibit limited structural complexity. Thus, simplification itself can be "amortized"—equivalent rules for all short patterns can be discovered exhaustively offline, while runtime execution involves only fast table-lookup matching.

**Core Idea**: Replace general CAS with a pre-computed hash-indexed rule set to reduce symbolic simplification from $O(100\text{ms})$ to $O(1\text{ms})$, thereby enabling synchronous simplification of online-generated expressions within the training loop.

## Method

### Overall Architecture

The training pipeline of Flash-ANSR consists of four stages: (1) Skeleton Sampling—sampling the number of operators according to an exponential length prior and constructing prefix skeletons using the Lample & Charton algorithm; (2) SimpliPy Simplification—reducing redundant expressions to a canonical form; (3) De-contamination—filtering expressions equivalent to the test set through both symbolic and numerical detection; (4) Dataset Rendering—sampling constants and data points to generate $(X, y)$ pairs. During inference, the model uses softmax sampling to generate $K$ candidate skeletons, which are deduplicated by SimpliPy, followed by constant optimization using Levenberg-Marquardt. Finally, the best expression is selected according to fit quality and parsimony regularization.

### Key Designs

1.  **SimpliPy Simplification Engine**:
    - **Function**: Rapidly reduces algebraic expressions to their shortest canonical form, achieving a 100x speedup over SymPy.
    - **Mechanism**: Operates in two stages. **Offline stage**—exhaustively enumerates all expression patterns with up to $L_{\max}=7$ symbols across length tiers, discovering simplification rules $\bm{\tau} \to \bm{\tau}'$ via numerical equivalence tests. Rules must satisfy strict length reduction $|\bm{\tau}'| < |\bm{\tau}|$ and non-increasing variable count. **Online stage**—implements variable-free "ground rules" via $O(1)$ hash table lookups, while "pattern rules" with variables are stored in tree structures bucketed by operator and length for subtree matching. Runtime involves $K=5$ alternating rounds of rule application (ApplyRules) and term cancellation (CancelTerms), followed by sorting commutative operands and replacing merged constants.
    - **Design Motivation**: General CAS solve simplification from first principles, which is overkill for SR training. By amortizing simplification (investing ~100h offline to gain millisecond-level runtime performance), the CAS bottleneck in the training loop is eliminated.

2.  **Scalable Encoder-Decoder Architecture**:
    - **Function**: Encodes datasets into conditional information to autoregressively generate expression skeletons in prefix notation.
    - **Mechanism**: The encoder uses a Set Transformer to handle variable-length datasets, introducing masked RMSSetNorm instead of LayerNorm/SetNorm (halving parameters while correctly handling padding). Inputs use 32-bit IEEE-754 multi-hot encoding (covering $10^{-38}$ to $10^{38}$, far exceeding 16-bit ranges). The decoder utilizes Pre-RMSNorm + FlashAttention + RoPE positional embeddings. Inference employs softmax sampling instead of beam search to increase candidate diversity.
    - **Design Motivation**: Pre-Norm is more stable than Post-Norm (Post-Norm diverged in ablations); 32-bit encoding covers the true magnitude of physical domain data; softmax sampling at $c=4096$ produces only $1/70$ the syntactic rewrites of beam search with 9.4pp higher recovery.

3.  **Strict De-contamination and Evaluation Protocol**:
    - **Function**: Prevents leakage of training data and establishes reliable evaluation standards.
    - **Mechanism**: For de-contamination, skeletons are first pruned of constants, followed by simultaneous symbolic comparison (token equality) and numerical comparison (evaluation on a grid $X_{\text{check}} \in \mathbb{R}^{512 \times D}$ followed by four-decimal rounding and hashing). Evaluation uses a machine-precision recovery standard $\text{FVU} \leq 1.19 \times 10^{-7}$, analyzing the Pareto front of inference time vs. recovery rate.
    - **Design Motivations**: Most prior works lacked strict de-contamination, potentially overestimating performance; loose success thresholds (e.g., $R^2 > 0.9$) mask true failure cases.

### Loss & Training

The training objective is cross-entropy loss: $\hat{\theta} = \arg\min_{\theta} \mathbb{E}[-\sum_{t=1}^{L} \log p_{\theta}(\bar{\tau}_t^* | \bar{\tau}_{<t}^*, \mathcal{D})]$, with the encoder and decoder trained end-to-end. Four model sizes (3M / 20M / 120M / 1B parameters) were trained, with the largest model trained on 512M online-generated data-expression pairs. Inference re-ranking follows parsimony regularization: $\hat{\bm{\tau}}^{\star} = \arg\min \log_{10}\text{FVU}(\hat{\bm{\tau}}) + \gamma \cdot |\hat{\bm{\tau}}|$, defaulting to $\gamma = 0.05$.

## Key Experimental Results

### Main Results (FastSRB Benchmark, 115 Expressions)

| Method | Type | vNRR↑ (~10s) | vNRR↑ (Peak) | Length Ratio↓ | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| NeSymReS | Amortized SR | ~10% | ~10% | — | Saturated, fails to generalize |
| E2E | Amortized SR | <2.5% | <2.5% | — | Nearly total failure |
| PySR | Genetic Programming | ~45% | 50.0% | 0.94→1.85 | Complexity grows over time |
| Flash-ANSR 3M | Amortized SR | ~25% | ~35% | — | Lags behind PySR |
| Flash-ANSR 120M | Amortized SR | ~45% | **~58%** | 1.40→1.27 | Excels PySR, parsimony inversion |

### SimpliPy Efficiency Comparison

| Simplification Engine | Median Time | Simplification Ratio | Timeout Rate (>1s) | Length Growth |
| :--- | :--- | :--- | :--- | :--- |
| SymPy | ~100ms | Good | 9% | 38%-52% |
| SimpliPy ($L_{\max}=4$) | ~1ms | Near SymPy | 0% | 0% (Strict non-growth) |
| SimpliPy ($L_{\max} \geq 5$) | Few ms | **Exceeds SymPy** | 0% | 0% |

### Ablation Study

| Configuration | vNRR↑ | Length Ratio | Description |
| :--- | :--- | :--- | :--- |
| Full (SimpliPy, 100M) | Highest | Lowest | Complete model |
| A-U (No Simpl.) | Similar | +40-50% | Severe expression redundancy |
| B1 (Post-Norm) | Failed | — | Unstable gradients |
| B2 (16-bit encoding) | Sig. Drop | Sig. Increase | Insufficient numerical precision |
| Beam Search vs Softmax | -9.4pp | 70× more rewrites | Beam search mode collapse |

### Key Findings

-   **Parsimony Inversion**: While PySR expressions grow increasingly complex over time (length ratio 0.94→1.85), Flash-ANSR converges toward more concise forms (1.40→1.27). This is because increased sampling allows finding rare but concise "needle in a haystack" correct expressions.
-   **Three-stage Phase Transition in Data Sparsity**: A "complexity peak" occurs at $M \approx 8$ data points, similar to Deep Double Descent—too few points lead to concise high-bias approximations, while the critical point causes the model to interpolate with excessive constants before finally converging to the true expression with sufficient data.
-   **Insufficient Noise Robustness**: At noise levels $\eta \geq 10^{-2}$, PySR significantly outperforms Flash-ANSR because the model was trained only on noise-free data and misinterprets noise as high-frequency signals.

## Highlights & Insights

-   **Amortization of Simplification**: Viewing simplification as a pre-computable lookup table problem rather than an online solving problem. Trading "once-off heavy computation" for "runtime zero cost" is a concept transferable to any scenario requiring expensive symbolic operations in a training loop.
-   **Softmax Sampling vs. Beam Search**: Under multi-modal posteriors, the mode-seeking behavior of beam search leads to 70x redundant rewrites. Softmax sampling explores more functionally distinct hypotheses at lower cost—a finding applicable to all sequence generation tasks.
-   **Self-discovered Scaling Law**: The authors used Flash-ANSR itself to perform symbolic regression on its own scaling curve, finding that performance asymptotically follows $\text{vNRR} \propto \log\log T$, whereas PySR has an upper bound around 53%. Analyzing a tool's behavior with the tool itself is methodologically elegant.

## Limitations & Future Work

-   **Low Noise Robustness**: Trained only on noise-free data; noise represents an out-of-distribution shift. Future work should introduce noise augmentation during training.
-   **High Cost of Offline Discovery**: $L_{\max}=7$ requires ~100h (32 threads); costs for extending to longer patterns grow exponentially.
-   **Evaluation Limited to FastSRB**: 115 expressions is a limited scale; performance in more complex real-world scientific scenarios remains to be verified.
-   **Future Directions**: Incorporating noisy data, exploring wider generation distributions, and testing alternative encoding/decoding paradigms like diffusion models.

## Related Work & Insights

-   **NeSymReS / E2E**: Representative prior work in amortized SR, limited by static datasets and unsimplified training respectively; this paper addresses both bottlenecks.
-   **PySR**: The current GP SOTA, which Flash-ANSR matches or exceeds under moderate computational budgets.
-   **Insight**: Decoupling "simplification" as an independent amortizable component rather than a sub-problem that must be solved online is a valuable architectural pattern for other machine learning systems involving symbolic manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](../../NeurIPS2025/interpretability/towards_scaling_laws_for_symbolic_regression.md)
- [\[ICML 2026\] MAAT: Knowledge-Guided Kernel Regression for Heterogeneous Partially Observed State Reconstruction](knowledge-informed_kernel_state_reconstruction_from_heterogeneous_partial_observ.md)
- [\[ICML 2026\] Verified SHAP: Provable Bounds for Exact Shapley Values in Neural Networks](verified_shap_provable_bounds_for_exact_shapley_values_of_neural_networks.md)
- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[ICML 2026\] Neural Collapse by Design: Learning Class Prototypes on the Hypersphere](neural_collapse_by_design_learning_class_prototypes_on_the_hypersphere.md)

</div>

<!-- RELATED:END -->
