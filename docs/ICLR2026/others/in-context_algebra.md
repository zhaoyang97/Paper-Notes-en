---
title: >-
  [Paper Note] In-Context Algebra
description: >-
  [ICLR2026][in-context learning] This paper introduces an **in-context algebra task**—where tokens serve as pure variables and each sequence randomly reassigns their meanings—and finds that Transformers in this setting no longer learn classical Fourier/geometric representations. Instead, three **symbolic reasoning mechanisms** emerge (commutative copying, identity element recognition, and closure-based cancellation), with these capabilities appearing sequentially as phase transitions during training.
tags:
  - ICLR2026
  - in-context learning
  - mechanistic interpretability
  - symbolic reasoning
  - finite groups
  - transformer mechanisms
date: 2026-05-08
content_hash: 464ce6ef5af14674
---

# In-Context Algebra

**Conference**: ICLR2026
**arXiv**: [2512.16902](https://arxiv.org/abs/2512.16902)
**Code**: [algebra.baulab.info](https://algebra.baulab.info)
**Area**: Other
**Keywords**: in-context learning, mechanistic interpretability, symbolic reasoning, finite groups, transformer mechanisms

## TL;DR

This paper introduces an **in-context algebra task**—where tokens serve as pure variables and each sequence randomly reassigns their meanings—and finds that Transformers in this setting no longer learn classical Fourier/geometric representations. Instead, three **symbolic reasoning mechanisms** emerge (commutative copying, identity element recognition, and closure-based cancellation), with these capabilities appearing sequentially as phase transitions during training.

## Background & Motivation

**Limitations of fixed embeddings**: Prior mechanistic interpretability studies (grokking, modular arithmetic) demonstrate that Transformers pre-encode task information in token embeddings (e.g., "108" encodes "divisible by 2"), learning periodic/Fourier-based geometric strategies.

**Genuine abstract reasoning**: A hallmark of abstract reasoning is the ability to handle **symbols whose meanings are not known in advance**. If tokens carry no fixed semantics, what strategies will the model learn?

**Pure-variable setting**: The authors propose a setting where tokens within each sequence serve solely as placeholder variables, with a random mapping $\varphi_s$ assigning finite group elements to vocabulary symbols differently for each sequence, forcing the model to reason purely from in-context relational structure.

**Connection to ICL**: This constitutes a deep investigation into the intrinsic mechanisms of in-context learning—the model must observe "facts" in context and infer algebraic structure, rather than relying on parametric memory.

**Interpretability methodology**: The authors design 5 target data distributions alongside causal intervention experiments, providing a rigorous methodological paradigm for mechanism verification.

**Staged learning**: Different capabilities emerge sequentially as phase transitions during training, revealing an intrinsic curriculum by which Transformers learn abstract operations.

## Method

### Task Framework

Given a collection of finite groups $\mathcal{G} = \{G_1, G_2, \ldots, G_m\}$, each training sequence $s$ is generated as follows:

1. **Sample groups**: Sample a subset $\mathcal{G}_s$ from $\mathcal{G}$, letting $H_s = \bigcup \mathcal{G}_s$ with $|H_s| \leq N$ (vocabulary size).
2. **Random mapping**: Construct a bijection $\varphi_s: H_s \to V$, randomly assigning group elements to variable tokens.
3. **Assemble sequence**: Sample multiplication facts $x \cdot y = z$ from the group, convert them to variable-token statements via $\varphi_s$, and concatenate.

Sequence format:

$$s = v_{x_1} v_{y_1} = v_{z_1},\; v_{x_2} v_{y_2} = v_{z_2},\; \cdots,\; v_{x_k} v_{y_k} = v_{z_k}$$

Each fact occupies 4 positions: **left slot** $v_{x_i}$, **right slot** $v_{y_i}$, **prediction token** "=", and **answer slot** $v_{z_i}$.

### Model Configuration

- **Architecture**: 4-layer autoregressive Transformer, 8 attention heads per layer, hidden dimension 1024
- **Training objective**: Standard next-token prediction
- **Sequence length**: $k = 200$ algebraic facts (~1000 tokens)
- **Training groups**: $\mathcal{G} = \{C_3, \ldots, C_{10}, D_3, D_4, D_5\}$ (cyclic and dihedral groups, order ≤ 10)
- **Vocabulary**: $N = 16$ variable tokens plus special tokens "=" and ","

### Five Hypothesized Mechanisms and Target Distributions

To disambiguate the algorithms the model may employ, the authors design 5 target data distributions:

| Distribution | Mechanism Tested | Construction |
|---|---|---|
| $\mathcal{D}_{\text{copy}}$ | Verbatim copying | Sequence contains a verbatim copy of the final fact |
| $\mathcal{D}_{\text{commute}}$ | Commutative copying | Contains the commuted fact $yx=z$, no verbatim copy |
| $\mathcal{D}_{\text{identity}}$ | Identity element recognition | Final fact involves identity $ey=y$; context exposes the identity |
| $\mathcal{D}_{\text{associate}}$ | Associative composition | Contains minimal facts derivable via associativity |
| $\mathcal{D}_{\text{cancel}}$ | Closure-based cancellation | Contains all facts sharing left/right slots; uses cancellation law |

### Causal Verification Method

**Indirect Effect (IE)** is used to quantify component importance:

$$\text{IE}(l,h) = P(v_{\text{target}} \mid a_{s_{\text{clean}}}^{(l,h)} \to s_{\text{corrupt}}) - P(v_{\text{target}} \mid s_{\text{corrupt}})$$

Key components are localized by **patching attention head activations** between clean/corrupted sequence pairs.

### Three Core Mechanisms

**1. Commutative Copying**: Implemented by a single attention head (Layer 3, Head 6). When a verbatim copy is present, this head attends to the answer slot and directly boosts the corresponding token's logit; when only the commuted fact $yx=z$ is present, it shifts attention to the answer slot of the commuted fact.

**2. Identity Element Recognition**: Achieved through two cooperative sub-mechanisms—**query boosting** and **identity suppression**. Head 3.1 boosts the logits of both variables in the query (query boosting), while Head 3.6 suppresses the logit of the identified identity token (identity suppression), leaving the non-identity variable as the correct answer. The first principal component of PCA on the final-layer attention output clearly separates identity from non-identity facts.

**3. Closure-Based Cancellation**: Computes $S_{\text{closure}} - S_{\text{cancel}}$. The closure sub-mechanism tracks all elements belonging to the same group as the query variable; the cancellation sub-mechanism exploits the cancellation law to eliminate elements already appearing as answers in facts sharing the same left or right slot. The authors train a 16-dimensional subspace $W$ and perform causal interventions on the closure set, achieving 99.8% intervention accuracy.

## Key Experimental Results

### Main Results: Algorithm Coverage and Model Performance

| Mechanism | Training Coverage (AUC) | Hold-out Coverage (AUC) |
|---|---|---|
| Verbatim copying | 67.9% | — |
| Commutative copying | +12.1% | — |
| Identity recognition | +4.2% | 28.7% |
| Closure cancellation | +2.7% | +39.1% |
| Associative composition | +3.6% | +16.9% |
| **Total coverage** | **90.4%** | **84.7%** |
| **Model actual accuracy** | **92.4%** | **87.3%** |

### Model Accuracy on Each Target Distribution

| Distribution | $k=50$ | $k=100$ |
|---|---|---|
| Verbatim copying $\mathcal{D}_{\text{copy}}$ | ~100% | 100.0% |
| Commutative copying $\mathcal{D}_{\text{commute}}$ | ~97% | 99.0% |
| Identity recognition $\mathcal{D}_{\text{identity}}$ | ~98% | 100.0% |
| Closure cancellation $\mathcal{D}_{\text{cancel}}$ | ~95% | 97.0% |
| Associative composition $\mathcal{D}_{\text{associate}}$ | ~55% | 60.2% |

### Generalization

- **Unseen group generalization**: Near-perfect accuracy on all order-8 groups not seen during training.
- **Semigroup generalization**: Non-trivial accuracy on non-group structures such as semigroups.
- **Quasigroups/Magmas**: Performance degrades on quasigroups and nearly fails on magmas, indicating the model relies on group-specific structural properties.

### Ablation Analysis: Phase Transitions in Staged Learning

Skill acquisition during training exhibits clear **five-stage phase transitions**:

| Stage | Skill Acquired | Training Step |
|---|---|---|
| ➀ | Structural token prediction ("=", ",") | Earliest |
| ➁➂ | Group closure + query boosting (identity 50% accuracy) | Second stage |
| ➃➄ | Verbatim copying + commutative copying | Sharp transition |
| ➅➆ | Closure cancellation + full identity recognition | Joint gradual improvement |
| ➇ | Associative composition | Last to emerge |

Key findings:
- **Copying is foundational**: Cancellation and identity recognition are built upon copying ability.
- **Joint emergence**: Identity suppression and the cancellation subspace perform analogous "suppression" functions, hence they are learned simultaneously.
- **Associativity is hardest**: The last skill to be acquired, reaching only ~60% accuracy.

### Causal Intervention Results

| Experiment | Key Head | AIE |
|---|---|---|
| Verbatim copying | Head 3.6 | **0.91** |
| Commutative copying | Head 3.6 | **0.48** |
| Highest AIE among other heads | — | < 0.08 |
| Closure subspace intervention accuracy | 16-dim $W$ | **99.8%** |

## Highlights & Insights

- **Minimalist yet profound experimental design**: The pure-variable algebra setting elegantly isolates "embedding priors" from "in-context reasoning," providing an exemplary paradigm for ICL mechanism research.
- **Complete mechanistic dissection**: The pipeline from hypothesis → target distribution design → coverage analysis → causal intervention → subspace probing forms a closed-loop verification framework.
- **Natural correspondence between phase transitions and curriculum learning**: Reveals spontaneous staged skill acquisition in Transformers, offering theoretical inspiration.
- **Symbolic vs. geometric strategy dependence**: Demonstrates that reasoning strategy depends on task structure—fixed tokens lead to geometric strategies, while pure variables induce symbolic strategies.
- **Code and data are open-sourced**, making experiments reproducible.

## Limitations & Future Work

1. **Limited model scale**: Verified only on a 4-layer small Transformer; applicability to large-scale pretrained LLMs remains unknown.
2. **Insufficient associative learning**: The model achieves only ~60% accuracy on associativity, indicating that multi-step reasoning remains challenging.
3. **Idealized task setting**: Finite group algebra is far from natural language reasoning; whether the findings transfer to more complex scenarios requires further validation.
4. **Vocabulary size constraint**: Only 16 variable tokens are used; whether the mechanisms persist with larger vocabularies is unexplored.
5. **Chain-of-thought not explored**: Incorporating chain-of-thought prompting may improve performance on complex reasoning tasks such as associative composition.

## Related Work & Insights

| Dimension | Ours | Prior Work (Nanda et al., Zhong et al.) |
|---|---|---|
| Token meaning | Pure variables, randomly reassigned per sequence | Fixed semantics (e.g., numbers) |
| Learned strategy | Symbolic reasoning (copying, cancellation) | Fourier basis / geometric representations |
| Grokking | Staged phase transitions, not classical grokking | Typical grokking |
| Generalization | Generalizes to unseen groups | Generalizes to in-distribution data |
| Analysis method | Causal intervention + subspace probing | Weight/embedding analysis |

This work complements the induction head / n-gram head analysis of Akyürek et al. (2024)—the copying head (3.6) identified here exhibits similar n-gram matching behavior, while additionally demonstrating higher-level functions such as commutative copying and identity suppression.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The pure-variable algebra setting is proposed for the first time, revealing reasoning mechanisms fundamentally different from those in traditional fixed-token settings.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 5 target distributions + causal intervention + subspace probing + phase transition analysis constitute an exceptionally thorough verification.
- Writing Quality: ⭐⭐⭐⭐⭐ — Figures are polished and logic is clear; visualizations in Figures 4/5/6 are particularly outstanding.
- Value: ⭐⭐⭐⭐ — Offers important insights for ICL mechanism research, though the bridge to practical LLM applications remains to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] LILAD: Learning In-context Lyapunov-stable Adaptive Dynamics Models](../../AAAI2026/others/lilad_learning_in-context_lyapunov-stable_adaptive_dynamics_models.md)
- [\[NeurIPS 2025\] ConTextTab: A Semantics-Aware Tabular In-Context Learner](../../NeurIPS2025/others/contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [\[AAAI 2026\] Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-editors](../../AAAI2026/others/verification-guided_context_optimization_for_tool_calling_via_hierarchical_llms-.md)
- [\[ICLR 2026\] Chart Deep Research in LVLMs via Parallel Relative Policy Optimization](chart_deep_research_in_lvlms_via_parallel_relative_policy_optimization.md)
- [\[ICLR 2026\] Condition Matters in Full-head 3D GANs](condition_matters_in_full-head_3d_gans.md)

</div>

<!-- RELATED:END -->
