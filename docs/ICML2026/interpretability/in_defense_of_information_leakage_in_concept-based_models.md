---
title: >-
  [Paper Note] In Defense of Information Leakage in Concept-based Models
description: >-
  [ICML 2026][Interpretability][Concept Bottleneck Models] This is a position paper: the authors defend "information leakage" in concept models, an often-criticized phenomenon. They point out that when concept annotations are naturally incomplete in real-world scenarios, **moderate "benign leakage" is actually a necessary condition for building accurate and intervenable models**, and they provide a loss $\mathcal{L}_{\text{int}}$, requiring only one extra forward pass…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Concept Bottleneck Models"
  - "Information Leakage"
  - "Concept Intervention"
  - "Mutual Information"
  - "Simplicity Bias"
date: 2026-05-08
content_hash: 60bab3b206fbcf14
---

# In Defense of Information Leakage in Concept-based Models

**Conference**: ICML 2026  
**arXiv**: [2606.10669](https://arxiv.org/abs/2606.10669)  
**Code**: TBD  
**Area**: Explainability / Concept-based Models (XAI)  
**Keywords**: Concept Bottleneck Models, Information Leakage, Concept Intervention, Mutual Information, Simplicity Bias

## TL;DR
This is a position paper: the authors defend "information leakage" in concept models, an often-criticized phenomenon. They point out that when concept annotations are naturally incomplete in real-world scenarios, **moderate "benign leakage" is actually a necessary condition for building accurate and intervenable models**, and they provide a loss $\mathcal{L}_{\text{int}}$, requiring only one extra forward pass, to induce this benign leakage.

## Background & Motivation

**Background**: Concept-based Models (CMs) base predictions on human-understandable concepts (e.g., "stripes", "round"). The most common framework is the **Concept Bottleneck Model (CBM)**: an encoder $g$ maps input to $k$ concept representations $\hat{\bm{c}}$, a scoring function $s$ provides concept scores $\hat{\bm{p}} \approx \mathbb{P}(C_i=1 \mid \bm{x})$, and a label predictor $f$ predicts the task label $\hat{\bm{y}}$. Its selling points are explainability + support for **test-time intervention** (experts correct a concept value, and the model adjusts its prediction accordingly).

**Limitations of Prior Work**: Extensive research has found that information leakage occurs in CM concept representations—information not belonging to the concept's semantics is encoded. The dominant narrative insists that leakage is harmful and must be eliminated, arguing it makes models "unexplainable," breaks intervenability, and poses safety risks. Consequently, numerous "leakage prevention/measurement" methods have emerged.

**Key Challenge**: The authors argue that the "leakage is always harmful" view is **itself an ill-posed proposition**. On one hand, evidence that "leakage makes models less explainable" is often weak because explainability lacks a universally accepted, measurable definition. On the other hand, **incomplete concept sets are the norm** in the real world: labeling all relevant concepts for every sample is expensive, and it is impossible to know which concepts future tasks will require before labeling. In an incomplete setting, forcibly banning leakage prevents the label predictor from receiving the $I(Y;X \mid C)$ portion of task information (existing only outside the concept set), thereby sacrificing task accuracy.

**Goal**: To demonstrate that (1) some controlled leakage is a necessary condition for achieving high accuracy + intervenability in incomplete scenarios; (2) this leakage does not necessarily damage explainability or intervenability (as measured by current proxy metrics).

**Key Insight**: Distinguish between "malignant leakage" and **benign leakage**—not all leakage is bad; the key is whether the leaked information can be cleanly decomposed and localized.

**Core Idea**: Instead of eliminating leakage, one should **encourage and utilize benign leakage**. By minimizing the total intervention loss $\mathcal{L}_{\text{int}}$ (task loss after intervening on all concepts) during training, benign leakage can be induced without sacrificing accuracy or intervenability.

## Method

### Overall Architecture
This paper is argument-driven (position), but the core contribution is an operational conceptual framework + a loss function. The logic chain: first formalize "leakage" within the CBM framework (using mutual information), then provide two necessary and sufficient conditions for "benign leakage" (Sufficiency + Localization). Next, prove that **Sufficiency is equivalent to minimizing a total intervention loss $\mathcal{L}_{\text{int}}$**, and finally argue that **Localization emerges implicitly from the simplicity bias of deep networks**. Thus, by penalizing $\mathcal{L}_{\text{int}}$, benign leakage arises naturally, resulting in a model that is both accurate and intervenable despite "leakage."

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input x"] --> B["Concept Encoder g<br/>Output Representation ĉ"]
    B --> C["Benign Leakage Definition<br/>ĉ_i decomposed into C̄_i + R_i"]
    C -->|Sufficiency: R complements I(Y;X|C)| D["Total Intervention Loss L_int<br/>Loss after intervening all concepts"]
    C -->|Localization: Implicitly emerges via Simplicity Bias| D
    D --> E["Label Predictor f<br/>Accurate & Intervenable"]
```

### Key Designs

**1. Definition of Benign Leakage: Decomposing + Localizing Leaked Information**

The authors first clarify two types of leakage using mutual information: **Inter-concept leakage** $I(\hat{C}_i;C_j) > I(C_i;C_j)$ (information about concept $j$ mixed into representation $i$) and **Task leakage** $I(\hat{C}_i;Y) > I(C_i;Y)$ (unintended task information mixed in). They then define Benign Leakage (Definition 5.1): assuming each concept representation can be decomposed as $\hat{C}_i \equiv (\bar{C}_i, R_i)$, where $\bar{C}_i$ is the concept-aligned component (modified during intervention) and $R_i$ is an orthogonal residual component, requiring two conditions:
- **Sufficiency**: $I(Y;R \mid C) \approx I(Y;X \mid C)$, meaning the task information not covered by the concept set is fully retained in the residual $R$.
- **Localization**: $I(Y;\bar{C}_i \mid R_i, \hat{C}_{-i}) \approx I(Y;C_i \mid C_{-i})$, meaning task-relevant information about $C_i$ is locally fixed within $\bar{C}_i$.

Intuition: Sufficiency ensures "leaked information is useful" (accurate even when incomplete); Localization ensures "intervening on a concept only affects that specific part" (interventions are controllable). The authors emphasize that benign leakage is a **necessary condition** for $f$ to be intervenable—it is a property of $\hat{C}$, while intervenability also depends on how the predictor $f$ is trained (e.g., a Hybrid CBM may satisfy benign leakage but remain non-intervenable if trained only with joint loss).

**2. Total Intervention Loss $\mathcal{L}_{\text{int}}$: Turning "Sufficiency" into an Optimizable Objective**

Optimizing localization directly is nearly impossible (mutual information on high-dimensional $\hat{C}$ is hard to estimate). However, the authors prove that (under standard well-posed assumptions) **achieving Sufficiency is equivalent to minimizing the negative task likelihood after all concepts are intervened to ground-truth values**:
$$\mathcal{L}_{\text{int}}=\mathbb{E}_{(\bm{x},\bm{c},y)\sim\mathcal{D}}\big[\mathcal{L}_y\big(f(g(\bm{x}\mid C:=\bm{c})),\,y\big)\big].$$
In practice, this means replacing the concept components in the bottleneck with ground-truth $\bm{c}$ during the forward pass and calculating task loss. It can be seen as a **generalization of independent CBM training** and is architecture-agnostic (does not require explicit decomposition of variables), with minimal cost—just one extra forward pass of $f$.

**3. Simplicity Bias Hypothesis: Why Localization is "Free"**

Sufficiency can be optimized, but where does Localization come from? The core hypothesis is the **simplicity bias of deep networks**: when multiple hypotheses perform equally well, SGD prefers the simplest one. Specifically, when $\mathcal{L}_{\text{int}}$ is heavily penalized (every intervention on $C_i$ changes $\hat{\bm{c}}_i$), if the predictor can "extrapolate which variables the intervention changed," the encoder only needs to place $I(Y;X \mid C_i)$ in the other representations. This is **simpler** than a solution that "re-predicts $C_i$, moves its info elsewhere, and encodes remaining task info." Thus, SGD spontaneously converges to the simple hypothesis where concept information is **localized** and the predictor **reacts to intervened variables**. Localization and intervenability are byproducts of penalizing $\mathcal{L}_{\text{int}}$.

**4. Refuting the "Leakage is Unexplainable" Argument with Proxy Metrics**

The authors categorize anti-leakage arguments into three types (intervenability, explainability, safety) and test each using **existing proxy metrics** in the CM field (Concept ROC-AUC, Intervention Curves, Linear Predictor Weights). Core result: Hybrid CBMs with strong leakage capacity, when trained with $\mathcal{L}_{\text{int}}$, are not worse—and often better—on these metrics than non-leaky independent CBMs. If independent CBMs are considered "explainable," why should benignly leaking ones not be?

## Key Experimental Results

### Main Results: Benign Leakage Maintains Accuracy & Intervenability

| Model (Incomplete CUB, $k=22$) | Concept ROC-AUC | Task Accuracy | Intervenability |
|---------------------------|-------------|---------|---------|
| Independent CBM $M_I$ (No Leakage) | $93.90\pm0.11\%$ | $47.61\pm1.01\%$ | Rising curve but low accuracy ceiling |
| Hybrid CBM $M_H$ ($\lambda_{\text{int}}=5$, Leakage) | $93.65\pm0.22\%$ | $73.31\pm0.26\%$ | Monotonically rising curve, higher AUC |

In incomplete settings, the leaky model leads task accuracy by approximately **25.7 percentage points**, with almost no drop in concept fidelity.

### Ablation Study

| Configuration | Observation | Explanation |
|------|------|------|
| Leaky CM + RandInt / Implicit $\mathcal{L}_{\text{int}}$ (CEM, ProbCBM, etc.) | Intervenability $\ge$ non-leaky CBM | Leakage $\ne$ Non-intervenable |
| Leaky CM without Sufficiency optimization (e.g., CEM without RandInt) | Nearly zero intervenability (non-increasing curve) | Leakage **can** hurt intervenability |
| Strengthening $\mathcal{L}_{\text{int}}$ for the pathological model above | Becomes highly intervenable again | $\mathcal{L}_{\text{int}}$ is the switch |
| DNN without concept-alignment loss + $\mathcal{L}_{\text{int}}$ | Becomes intervenable; pseudo-concept ROC-AUC $80.79\%$ | Localization emerges implicitly |

### Key Findings
- **$\mathcal{L}_{\text{int}}$ is the watershed between "benign" and "malignant" leakage**: Given allowed leakage, models optimizing sufficiency are intervenable, while those that do not collapse.
- **RandInt implicitly minimizes $\mathcal{L}_{\text{int}}$**: Random interventions during training (as in CEM) are equivalent to an approximate total intervention loss.
- **Explainability arguments are non-falsifiable**: Without a universally accepted definition of explainability, assertions that "leakage makes models unexplainable" cannot be tested in incomplete settings and are thus ill-posed.

## Highlights & Insights
- **Turning philosophical debate into an optimizable loss**: Rather than arguing whether "leakage is good," the authors use $\mathcal{L}_{\text{int}}$ to make "benign leakage" something attainable via one extra forward pass.
- **Simplicity bias explains the emergence of localization**: Using "SGD's preference for simple hypotheses" to explain why optimizing sufficiency yields localization and intervenability is an elegant and verifiable argument.
- **Honest counter-examples**: The authors do not deny that leakage can be harmful, explicitly distinguishing between "benign vs malignant."

## Limitations & Future Work
- **Localization cannot be directly optimized**: It relies on simplicity bias, lacking a guarantee; mutual information estimation on high-dimensional residuals remains difficult.
- **Safety arguments not fully addressed**: The authors acknowledge concerns like shortcut learning but attribute them to general DNN representation learning problems rather than CM-specific ones—this defense is more of an evasion than a solution.
- **Proxy metrics for explainability**: The conclusion that "benign leakage is not worse" holds relative to current proxies; if a stricter definition of explainability were used, the conclusion might change.
- **Primary experiments are concentrated on CUB**, with limited coverage across diverse tasks in the main text.

## Related Work & Insights
- **vs. Mainstream anti-leakage routes (Marconato et al. 2022, Havasi et al. 2022)**: While they seek to eliminate/mitigate leakage, this paper argues it is a necessary condition for accuracy in incomplete settings and should be "tamed" rather than eradicated.
- **vs. CEM / RandInt (Espinosa Zarlenga et al. 2022)**: This paper notes that RandInt implicitly optimizes $\mathcal{L}_{\text{int}}$, providing a unified explanation for the success of certain "leaky yet intervenable" models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reverses the mainstream narrative with operational criteria.
- Experimental Thoroughness: ⭐⭐⭐⭐ Convincing controlled experiments, though largely focused on CUB.
- Writing Quality: ⭐⭐⭐⭐⭐ Step-by-step logic, dismantling opposing arguments while admitting unaddressed areas.
- Value: ⭐⭐⭐⭐ Likely to shift the concept model community's attitude toward leakage; $\mathcal{L}_{\text{int}}$ is plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)
- [\[ICML 2026\] Formal Concept Lattices are Good Semantic Scaffolds for Concept-Based Learning](formal_concept_lattices_are_good_semantic_scaffolds_for_concept-based_learning.md)
- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](../../CVPR2026/interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)

</div>

<!-- RELATED:END -->
