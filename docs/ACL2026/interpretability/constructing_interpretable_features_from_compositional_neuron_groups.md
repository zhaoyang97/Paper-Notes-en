---
title: >-
  [Paper Note] Constructing Interpretable Features from Compositional Neuron Groups
description: >-
  [ACL 2026][Interpretability][SNMF] The authors use Semi-Nonnegative Matrix Factorization (SNMF) to directly decompose MLP activations into "sparse neuron groups $\times$ non-negative coefficients." This yields interpreta…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "SNMF"
  - "MLP decomposition"
  - "concept steering"
  - "Sparse Autoencoders"
  - "concept hierarchy"
date: 2026-05-08
content_hash: 448c2ce1c6f3f9ba
---

# Constructing Interpretable Features from Compositional Neuron Groups

**Conference**: ACL 2026  
**arXiv**: [2506.10920](https://arxiv.org/abs/2506.10920)  
**Code**: <https://github.com/ordavid-s/snmf-mlp-decomposition>  
**Area**: Interpretability / Mechanistic Interpretability / LLM Internal Representations  
**Keywords**: SNMF, MLP decomposition, concept steering, Sparse Autoencoders, concept hierarchy

## TL;DR
The authors use Semi-Nonnegative Matrix Factorization (SNMF) to directly decompose MLP activations into "sparse neuron groups $\times$ non-negative coefficients." This yields interpretable features that can be mapped back to activation contexts and combined across layers. In causal concept steering evaluations on Llama-3.1-8B, Gemma-2-2B, and GPT-2, this method comprehensively outperforms state-of-the-art SAEs (Llamascope / Gemmascope) and the strong supervised baseline DiffMeans.

## Background & Motivation

**Background**: The core question of mechanistic interpretability is "what units are used to explain LLMs." Early research focused on individual neurons, but they are known to be polysemantic. Recent consensus has shifted toward "directions in activation space," with the mainstream approach being Sparse Autoencoders (SAEs) learning a "feature dictionary" from the residual stream.

**Limitations of Prior Work**: SAEs frequently fail in causal evaluations—directly intervening on SAE features often fails to directionally change model behavior. Moreover, the directions learned by SAEs are not forcibly constrained to the original model's representation space nor linked to specific MLP computations, causing their "interpretability" to rely heavily on posterior natural language labeling.

**Key Challenge**: Learning a set of directions from scratch vs. discovering neuron combinations already existing within the model—the former is flexible but detached from mechanisms, while the latter naturally has mechanistic anchors but requires unsupervised methods to extract them.

**Goal**: Propose an unsupervised method where discovered "features" satisfy: (1) being sparse linear combinations of MLP neurons, (2) possessing a reverse mapping from features to activation contexts, and (3) enabling directional generation changes under causal intervention.

**Key Insight**: MLP output is inherently $\sum_i a_i \mathbf{v}_i$ (weighted neuron activation vectors). Similar inputs should activate similar neuron groups—if these "co-activation patterns" can be decomposed from the activation matrix $A$, features will naturally possess mechanistic anchors.

**Core Idea**: Use SNMF to decompose MLP activations $A \approx Z Y$, where $Z$ is a "feature matrix" (linear combination of neurons) allowing both positive and negative values, and $Y$ is a "coefficient matrix" constrained to be non-negative (indicating which tokens trigger which feature). The non-negativity of $Y$ encourages parts-based additive representations, while the sign-flexibility of $Z$ adapts to the bidirectional semantics of MLP activations.

## Method

### Overall Architecture
Given an MLP layer of a pre-trained LLM, forward a massive corpus to collect activation vectors $\mathbf{a} \in \mathbb{R}^{d_a}$ for each token, forming matrix $A \in \mathbb{R}^{d_a \times n}$. Solve $A \approx Z Y$ via SNMF to obtain $k$ MLP features $\mathbf{z}_i$ (each a $d_a$-dimensional weighted neuron direction). Each MLP feature is then projected back to the residual stream via $W_V$ to obtain $\mathbf{f}_i = W_V \mathbf{z}_i$, which is compared against SAE / DiffMeans in the same space. The coefficient matrix $Y$ directly indicates which tokens strongly activate which features—this provides the "activation context" automatic labeling absent in SAEs.

### Key Designs

1.  **SNMF Decomposition + Neuron Sparsity**:
    - **Function**: Decomposes the MLP activation matrix into a "feature basis" $Z$ and "non-negative usage coefficients" $Y$, constraining features to be linear combinations of a small number of neurons.
    - **Mechanism**: Use Multiplicative Updates (Ding et al.) to alternately update $Z$ (closed form: $Z \leftarrow A Y^\top (Y Y^\top + \lambda I)^{-1}$) and $Y$ (with sign-splitting multiplicative updates to maintain non-negativity). After each iteration, applying a hard winner-take-all (WTA) to each column of $Z$, keeping only the top $p\%=1\%$ neurons by absolute value and zeroing others to enforce neuron-level sparsity.
    - **Design Motivation**: SAE non-negative regularization encourages parts-based representation, but activations have both positive and negative aspects—SNMF relaxes the "non-negative feature" constraint to retain bidirectional directions while keeping "non-negative coefficients" for additive parts-based interpretability. Hard WTA follows the $\ell_0$ constraint scheme of Peharz & Pernkopf (2012); experiments show 1% is superior to 5% / 10%.

2.  **Automatic Concept Labeling based on $Y$**:
    - **Function**: Obtains "top activation contexts" directly for each MLP feature to generate descriptive labels, bypassing auxiliary pipelines based on raw activation ranking.
    - **Mechanism**: Take top-$m$ tokens from the $i$-th row of $Y$ for feature $\mathbf{z}_i$, feed contexts into GPT-4o-mini to summarize common semantic patterns; early layers use "act-in" descriptions, later layers use logit lens style "output token" descriptions (Gur-Arieh 2025).
    - **Design Motivation**: SAE descriptions rely on third-party pipelines like Neuronpedia. SNMF provides intrinsic token-to-feature attribution—this step increases interpretability and explains why SNMF often leads in concept detection (higher mean log-ratio $S_{CD} := \log \bar{a}_{\text{act}} / \bar{a}_{\text{neutral}}$).

3.  **Recursive SNMF to Reveal Concept Hierarchies**:
    - **Function**: Recursively decompose learned MLP features with decreasing $k$ to obtain a hierarchy tree from "specific concepts" to "abstract concepts."
    - **Mechanism**: Perform multi-stage SNMF with $k=[400, 200, 100, 50]$, then jointly fine-tune all levels via gradient descent to minimize $\mathcal{L} = \frac{1}{2}\|A - Z_L Y_L \cdots Y_1 Y_0\|_F^2$. This yields merged hierarchies like "Monday/Tuesday → Weekday/Weekend → Day of the week." Concept synonym neuron overlap is visualized by binarizing $Z$ to get $M = \bar{Z}\bar{Z}^\top$.
    - **Design Motivation**: This is the "inverse direction" of the SAE feature splitting phenomenon—SAECs split a feature into several when expanding the dictionary, whereas SNMF merges features into an abstract concept when shrinking $k$. By causal intervention on "weekday base neurons" vs. "Monday-only exclusive neurons," the authors prove the model indeed builds concepts using a "core + exclusive" composition.

### Loss & Training
SNMF minimizes Frobenius reconstruction error $\|A - ZY\|_F^2$ subject to non-negative constraints (on $Y$ only) and WTA sparsity (on $Z$ columns). Initialization: Random initialization with $Y \sim \mathcal{U}(0,1)$ and $Z \sim \mathcal{N}(0,1)$ performs comparably to SVD / K-Means but converges slower (3325 vs 1484 / 2474 iterations). All experiments use $k \in \{100, 200, 300, 400\}$, $p=1\%$ (5% for GPT-2).

## Key Experimental Results

### Main Results (Concept steering + fluency harmonic mean, higher is better)

| Model / Layer | SAE-out | SAE-act (Same Capacity) | DiffMeans (Supervised) | SNMF (Ours) |
| :--- | :--- | :--- | :--- | :--- |
| Llama-3.1-8B L23 | $\approx 0.35$ | $\approx 0.37$ | $\approx 0.40$ | **0.45** |
| Llama-3.1-8B L31 | $\approx 0.20$ | $\approx 0.25$ | $\approx 0.27$ | **0.31** |
| Gemma-2-2B L18 | Lower | Medium | Medium | **Highest** |
| GPT-2 Small | Similar trend | Similar trend | Similar trend | **Leading** |

> On Concept Detection, SNMF is roughly on par with SAE-out and significantly exceeds same-capacity SAE-act ($>75\%$ of features have $S_{CD} > 0$), but the decisive advantage of SNMF lies in Concept Steering.

### Ablation Study (Llama-3.1-8B, SNMF $k=100$)

| Configuration | Concept Detection ($L_0$) | Concept Steering+Fluency (L23) | Description |
| :--- | :--- | :--- | :--- |
| Random init | 2.99 $\pm$ 1.55 | 0.45 $\pm$ 0.32 | Default |
| SVD init | 2.76 $\pm$ 1.79 | 0.41 $\pm$ 0.31 | Comparable but faster convergence |
| K-Means init | 2.55 $\pm$ 1.51 | 0.47 $\pm$ 0.33 | 1484 iterations to converge |
| WTA = 1% | 2.99 / 1.67 / 0.81 | 0.45 $\pm$ 0.32 | Default, best |
| WTA = 5% | Slightly lower | 0.41 $\pm$ 0.30 | Decreased sparsity |
| WTA = 10% | Slightly lower | 0.34 $\pm$ 0.30 | Further degradation |

Hierarchy Experiment (GPT-2 Large Causal Intervention, selected from Table 2):

| Intervened Neuron Group | Monday logit | Tuesday | Friday | Sunday |
| :--- | :--- | :--- | :--- | :--- |
| Monday-exclusive | **+2.0** | -0.8 | -0.8 | -0.1 |
| Friday-exclusive | -2.9 | -2.8 | **+1.3** | -2.6 |
| Sunday-exclusive | -0.4 | -1.7 | -0.8 | **+2.6** |
| **Core Weekday (base)** | **+5.8** | **+5.7** | **+6.0** | **+4.7** |

### Key Findings
- SNMF consistently outperforms SAE-out and DiffMeans in concept steering, proving that "neuron combinations embedded in MLP weights" are the true intervenable units, whereas SAE-learned directions often fail to directionally manipulate behavior.
- The hierarchical structure exposed by recursive SNMF (specific days → weekend/weekday → day of week) and the "core base + exclusive neuron" mechanism provide a mechanistic explanation for the SAE "feature splitting" phenomenon—splitting is a reflection of the model using neuron groups to construct refined concepts, not an artifact of SAE training.
- 1% WTA sparsity + random initialization is the optimal trade-off; performance is robust to initialization strategies, with K-Means primarily accelerating convergence.
- The highest concept detection scores in shallow layers (layer 0/1) suggest that activations not yet mixed by attention are easier to deconstruct into monosemantic features.

## Highlights & Insights
- The SNMF choice of placing the non-negative constraint only on coefficients is correct: MLP activations are signed, representing concept promotion vs. inhibition. Forcing non-negative features (like NMF) loses half the information.
- The coefficient matrix $Y$ provides intrinsic token-to-feature mapping, saving the third-party description pipelines like Neuronpedia / autointerp, making the method a self-contained loop.
- Reinterpreting SAE "feature splitting" as the "inverse projection of the model's own feature merging" is a paradigm-level insight—it unifies "finding features" and "understanding how the model builds concepts with neurons."
- The causal evidence of "core base + exclusive neurons" (core activation pushes all related tokens, exclusive activation only pushes itself and inhibits siblings) is very elegant and can be transferred to studies of other linearly structured concepts like months, seasons, numbers, or grammatical attributes.

## Limitations & Future Work
- Experiments use $k < 500$; scalability to massive dictionaries (similar to SAEs with thousands to millions of features) is not yet verified. Multiplicative Update optimizers do not easily incorporate regularization; future work may need projected gradient descent.
- Some layers (Gemma-2-2B layer 12) show a drop in performance as $k$ increases, suggesting that the number of meaningful concepts is fewer than $k$, requiring automatic $k$ selection strategies.
- Evaluation relies heavily on a GPT-4o-mini judge (validated with human Spearman $\rho=0.8$), which may still be sensitive to prompts.
- Currently, fluency drops significantly in layer 0 interventions; the "propagation side effects" of shallow steering have not been systematically modeled.

## Related Work & Insights
- **vs SAE (Bricken et al. / Gemmascope / Llamascope)**: SAEs learn directions from the residual stream; they have scale but suffer from causal failure. SNMF decomposes MLP activations directly; it has smaller scale but strong causal grounding. They are complementary—SAE provides breadth, SNMF provides mechanistic anchors.
- **vs DiffMeans (Supervised Baseline)**: DiffMeans uses the difference between positive/negative averages to obtain directions and is heavily affected by noise from unrelated concepts; SNMF significantly outperforms it, especially in shallow layers.
- **vs Yun et al. 2021 (Residual Stream NMF)**: They performed NMF on the residual stream primarily for linguistic analysis. This work performs SNMF on MLP activations, anchors features to specific neuron combinations, and introduces causal evaluation.
- **vs Cao et al. 2025 (NeurFlow)**: Also concerns functional clustering of neuron groups; this work applies it to LLM MLPs under unsupervised causal scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of SNMF + WTA + recursive hierarchy + token-to-feature mapping is a fresh perspective in the LLM interpretability community.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three models $\times$ multiple layers $\times$ multiple $k$ $\times$ causal + detection axes, though $k$ is small and direct comparisons with large benchmarks like RAVEL / MIB are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear formulas, detailed appendices (initialization, sparsity, hierarchy datasets), and complete weekday case studies.
- Value: ⭐⭐⭐⭐⭐ Provides a "MLP-embedded feature" route for the MI community and open-sourced code, offering direct actionable value for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] DPN-LE: Dual Personality Neuron Localization and Editing for Large Language Models](dpn-le_dual_personality_neuron_localization_and_editing_for_large_language_model.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)
- [\[ACL 2026\] Interpretable Coreference Resolution Evaluation Using Explicit Semantics](interpretable_coreference_resolution_evaluation_using_explicit_semantics.md)
- [\[CVPR 2026\] CI-ICE: Intrinsic Concept Extraction Based on Compositional Interpretability](../../CVPR2026/interpretability/ciice_intrinsic_concept_extraction_compositional.md)

</div>

<!-- RELATED:END -->
