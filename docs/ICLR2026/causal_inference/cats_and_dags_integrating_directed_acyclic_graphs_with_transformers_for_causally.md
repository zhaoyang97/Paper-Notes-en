---
title: >-
  [Paper Note] CaTs and DAGs: Integrating Directed Acyclic Graphs with Transformers for Causally Constrained Predictions
description: >-
  [ICLR 2026][Causal Inference][Causal Transformer] This paper proposes the **Causal Transformer (CaT)**, which injects the adjacency matrix of a pre-specified Directed Acyclic Graph (DAG) as a mask into the transformer's cross-attention. This allows the network to strictly adhere to the causal structure while retaining strong functional approximation capabilities, resulting in improved robustness to covariate shift, better interpretability, and the ability to directly estimate…
tags:
  - "ICLR 2026"
  - "Causal Inference"
  - "Causal Transformer"
  - "DAG Constraints"
  - "Masked Attention"
  - "Covariate Shift Robustness"
  - "Intervention Effect Estimation"
date: 2026-05-08
content_hash: f5166171bd2f91c8
---

# CaTs and DAGs: Integrating Directed Acyclic Graphs with Transformers for Causally Constrained Predictions

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ZIQactmQxb](https://openreview.net/forum?id=ZIQactmQxb)  
**Code**: [https://github.com/matthewvowels1/Causal_Transformer](https://github.com/matthewvowels1/Causal_Transformer)  
**Area**: Causal Inference / Structural Inductive Bias  
**Keywords**: Causal Transformer, DAG Constraints, Masked Attention, Covariate Shift Robustness, Intervention Effect Estimation  

## TL;DR
This paper proposes the **Causal Transformer (CaT)**, which injects the adjacency matrix of a pre-specified Directed Acyclic Graph (DAG) as a mask into the transformer's cross-attention. This allows the network to strictly adhere to the causal structure while retaining strong functional approximation capabilities, resulting in improved robustness to covariate shift, better interpretability, and the ability to directly estimate intervention effects.

## Background & Motivation
- **Background**: Fully connected networks and transformers are highly flexible function approximators, yet they rely purely on statistical correlations and lack built-in prior knowledge about the Data Generating Process (DGP).
- **Limitations of Prior Work**: Models that do not respect the DGP tend to exploit non-causal associations in the training set (e.g., "desert background = camel"). These models fail significantly when spurious correlations change at test time—making them highly sensitive to **covariate shift** and difficult to interpret.
- **Key Challenge**: To perform unbiased causal effect estimation (such as ATE), structural constraints must be explicitly imposed to disentangle confounding. Purely statistical neural networks struggle with this. Meanwhile, existing neural causal methods (e.g., CEVAE, CFR, GANITE) often treat each variable as a **single scalar**, failing to handle high-dimensional embeddings (e.g., skeletal joints, speech/multimodal features, multi-item scales).
- **Goal**: Propose a **generic modeling framework**—not to achieve SOTA on a specific benchmark, but to cleanly integrate DAG structural inductive biases into popular neural architectures, enabling them to handle both scalar tabular data and multi-modal embeddings of arbitrary dimensions.
- **Core Idea**: **[Masking as Constraint]** Use the transposed adjacency matrix $A$ of a topologically sorted DAG as a Hadamard mask on the cross-attention scores. This ensures each node can only "attend" to its legitimate causal parents, enforcing the conditional independence implied by the DAG at the architectural level.

## Method

### Overall Architecture
CaT treats the input as a sequence of $d$-dimensional embeddings corresponding to DAG nodes. A learnable "blank" query vector $\gamma$ extracts information from the input embeddings through **causal masked cross-attention**. Information propagates through CBlocks, and prediction/reconstruction is provided by output heads at each node. The paper also introduces **CFCN** (a fully connected network extending MADE-style autoregressive masking to DAGs) as a baseline to prove that benefits stem from the structural mask itself rather than the transformer's attention mechanism.

```mermaid
flowchart LR
    X[Input X<br/>B×|Z|×C] -->|Independent Linear Layers| XE[Embedding XE<br/>B×|Z|×dE]
    DAG[DAG Input<br/>Adjacency Matrix A] --> Mask[Topological Sort + Mask]
    gamma[Learnable Query γ] --> CA
    XE -->|K, V| CA[Causal Cross-Attention<br/>softmax A⊤∘QKᵀ/√hs · V]
    Mask --> CA
    CA --> Block[CBlock: Multi-head + FF + Residuals]
    XE -. Re-fed at each layer .-> Block
    Block -->|Stack × Blocks| Out[Node Output Heads<br/>Reconstruct/Predict X̂]
```

### Key Designs

**1. DAG Masked Cross-Attention: Hardcoding Causal Constraints into Softmax**. This is the core modification of CaT. Input $X$ (shape $B\times|Z|\times C$) is first embedded into $X_E$ ($B\times|Z|\times d_E$, where $d_E\ge C$ and $d_E>1$) using $|Z|$ **independent** linear layers. Keys and values come from the input, while queries come from $\gamma$: $K=X_EW_K,\ Q=\gamma W_Q,\ V=X_EW_V$. The transposed adjacency matrix $A^\top$ is applied as an element-wise mask to the attention scores before the softmax:

$$O = \mathrm{softmax}\!\left(\frac{A^\top \circ QK^\top}{\sqrt{h_s}}\right)\cdot V.$$

This ensures each node (query position) only assigns attention weights to its causal parents. The conditional independence implied by the DAG is strictly preserved during forward propagation.

**2. Learnable Query $\gamma$ + Layer-wise Input Feeding: Matching "Current Estimate" against "Observation"**. Unlike standard cross-attention, CaT's query $\gamma$ is initially a random "blank" embedding ($|Z|\times d_E$), which is progressively filled. $X_E$ is **re-fed into every block** as keys/values, while $\gamma$ (and its successors $O^r_{\text{Block}}$) serves as the query. This allows the model to compare current estimates with actual observations to determine what additional information needs to be extracted under DAG constraints. CBlocks use residual connections and optional BatchNorm instead of LayerNorm, as the latter can rescale calibrated intervention values.

**3. CFCN Baseline: DAG Masked Fully Connected Network**. To isolate the value of the mask, the authors extend MADE to arbitrary DAGs. Masks $M_r$ are pre-assigned to each layer: $o_r=\sigma(o_{r-1}(W_r\circ M_r)+b_r)$. A critical technique is **adding an identity diagonal to the DAG starting from the second layer**: the first layer acts as a "barrier" to prevent variables from seeing themselves, while subsequent layers use the identity to propagate signals.

**4. Recursive Interventional Inference: Using g-formula for do-propagation**. To estimate ATE/CATE, the model sets the intervened variable to a specific value and **recursively updates all descendants** along the topological order. Effect estimation follows the g-formula, filtering out spurious paths through do-calculus. CaT uses **per-variable losses** (MSE/BCE/Softmax) without special regularization terms, maintaining simplicity.

## Key Experimental Results

### Main Results: Causal Inference Benchmarks (Twins / Jobs, lower is better)

| Model | Twins eATE (ws) | Twins eATE (os) | Jobs Risk (ws) | Jobs eATT (os) |
|---|---|---|---|---|
| **CaT** | .0110 | .0133 | .25 | .086 |
| **CFCN** | .0098 | .0102 | .25 | .084 |
| GANITE | .0058 | .0089 | .13 | .06 |
| CFR-wass | .0112 | .0284 | .17 | .09 |
| BART | .1206 | .1265 | .23 | — |
| C Forest | .0286 | .0335 | .19 | .07 |

CaT/CFCN achieve performance comparable to specialized causal inference methods (GANITE, CFR) while using a much weaker prior (assuming non-treatment/non-outcome variables are potential confounders) and **no hyperparameter tuning**.

### Key Findings
- **Correct DAG is essential for eATE**: Models using incorrect DAGs or non-causal architectures often show lower prediction MSE (higher fitting capacity) but exhibit significantly higher ATE estimation bias.
- **Robustness to Covariate Shift**: CaT/CFCN remains stable under distribution shifts, whereas Random Forest, MLP, and standard Transformers fluctuate due to reliance on spurious/confounder correlations.
- **Handling Multi-dimensional Input**: In a real-world psychological study (COVID-19 attachment vs. depression), CaT processed **entire questionnaire items** as vectors rather than aggregated scales, yielding results consistent with semi-parametric Targeted Learning.

## Highlights & Insights
- **Masked Attention as Structural Inductive Bias**: A simple Hadamard product hardcodes DAG conditional independence into the attention mechanism without needing auxiliary losses.
- **Causal Modeling for High-Dimensional Variables**: It breaks the "one-scalar-per-variable" limitation of prior neural causal methods, naturally supporting multi-modal data.
- **The "Low MSE Trap"**: The paper demonstrates that high predictive accuracy does not equate to causal correctness, emphasizing that causal models should be evaluated on eATE rather than fit.

## Limitations & Future Work
- **Error Accumulation**: In recursive inference, errors in predicting intermediate nodes on long causal chains can propagate and amplify.
- **Dependence on Prior DAG**: The framework assumes the DAG is known. Incorrect graph specifications lead to incorrect structural constraints.
- **Untestable Assumptions**: Causal identification still relies on standard assumptions like ignorability and positivity, which are generally untestable in practice.

## Related Work & Insights
- **Structural vs. Functional Bias**: This work focuses on structural bias (constraining variable interactions), distinct from functional biases like weight decay or convolutions.
- **Distinction from "Causal Masking" in GPT**: The authors clarify that the standard autoregressive "causal mask" in transformers is merely a weak Granger-style constraint and does not support true structural interventions or disentanglement.
- **Inspiration**: Injecting domain knowledge via adjacency matrices into attention is a low-cost "Knowledge-as-Architecture" paradigm applicable beyond causal inference.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Elegant integration of DAG masks into cross-attention; excellent support for high-dimensional embeddings.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers simulations, standard benchmarks, and real-world applications; however, avoids chasing SOTA through tuning.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic; the demonstration of "Low MSE $\neq$ Causal Correctness" is highly instructional.
- **Value**: ⭐⭐⭐⭐ A versatile causal modeling base for fields requiring robustness and interpretability (medicine, policy, psychology).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation](../../ACL2025/causal_inference/causalrag_integrating_causal_graphs_into_retrieval-augmented_generation.md)
- [\[ICLR 2026\] Beyond DAGs: A Latent Partial Causal Model for Multimodal Learning](beyond_dags_a_latent_partial_causal_model_for_multimodal_learning.md)
- [\[ICLR 2026\] Theoretical Guarantees for Causal Discovery on Large Random Graphs](theoretical_guarantees_for_causal_discovery_on_large_random_graphs.md)
- [\[AAAI 2026\] I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables](../../AAAI2026/causal_inference/i-cam-uv_integrating_causal_graphs_over_non-identical_variable_sets_using_causal.md)
- [\[ICLR 2026\] On the Identifiability of Causal Graphs with the Invariance Principle](on_the_identifiability_of_causal_graphs_with_the_invariance_principle.md)

</div>

<!-- RELATED:END -->
