---
title: >-
  [Paper Note] Characterization and Learning of Causal Graphs with Latent Confounders and Post-treatment Selection from Interventional Data
description: >-
  [ICLR2026][Causal Inference][Post-treatment selection bias] This paper identifies a long-ignored challenge in interventional causal discovery: **post-treatment selection** (e.g., in single-cell experiments, only high-activity cells are retained according to quality control standards after intervention). This selection mimics causal responses, causing existing methods to misclassify the presence and absence of direct causal edges into the same equivalence class. The authors ex…
tags:
  - "ICLR2026"
  - "Causal Inference"
  - "Post-treatment selection bias"
  - "latent confounding"
  - "interventional causal discovery"
  - "Markov equivalence class"
  - "F-PAG"
date: 2026-05-08
content_hash: 3afe949839058277
---

# Characterization and Learning of Causal Graphs with Latent Confounders and Post-treatment Selection from Interventional Data

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=qclNnbjxNJ](https://openreview.net/forum?id=qclNnbjxNJ)  
**Code**: https://github.com/GongxuLuo/F-FCI  
**Area**: Causal Discovery / Interventional Causal Learning  
**Keywords**: Post-treatment selection bias, latent confounding, interventional causal discovery, Markov equivalence class, F-PAG  

## TL;DR
This paper identifies a long-ignored challenge in interventional causal discovery: **post-treatment selection** (e.g., in single-cell experiments, only high-activity cells are retained according to quality control standards after intervention). This selection mimics causal responses, causing existing methods to misclassify the presence and absence of direct causal edges into the same equivalence class. The authors explicitly model selection variables using augmented DAGs, propose **FI-Markov equivalence** (finer than traditional classes) and a new graph representation **F-PAG**, and provide the provably sound and complete **F-FCI** algorithm. This approach simultaneously identifies causal relationships, latent confounders, and post-treatment selection from observational and interventional data.

## Background & Motivation
**Background**: The mainstream approach in interventional causal discovery is using distribution changes brought by interventions to orient causal edges. The classic criterion involves a set of cross-interventional patterns: when the "cause" is intervened upon, the marginal distribution of the "effect" $p(\text{effect})$ changes, while the conditional distribution $p(\text{effect}\mid\text{cause})$ remains invariant. Conversely, intervening on the "effect" leaves $p(\text{cause})$ unchanged but alters $p(\text{cause}\mid\text{effect})$. Methods like GIES, IGSP, and FCI-with-intervention are built on this invariance analysis and have been extended to scenarios with latent confounders.

**Limitations of Prior Work**: The authors emphasize a pervasive issue in biological experiments ignored by current frameworks—**post-treatment selection**, where samples are selectively included in the dataset **after** intervention. Typical examples include genetic perturbation experiments where only cells passing quality control are sequenced, or clinical trials (per-protocol analysis) where only subjects completing over 80% of follow-ups are retained. The problem is that the statistical footprint of post-treatment selection—$p(\text{effect})$ changes while $p(\text{effect}\mid\text{cause})$ stays invariant—**is identical to that of a true causal relationship**.

**Key Challenge**: Due to identical footprints, existing frameworks place cases where "a direct causal edge exists between $X_1$ and $X_2$" and "the two are only linked by a shared selection variable without a direct edge" into the **same equivalence class** (Figure 1(a) vs. (b) in the paper). Consequently, they cannot distinguish causal relationships from post-treatment selection or detect where selection occurred. This represents a **representational gap**: the granularity of existing DAG/MAG/PAG equivalence classes is too coarse to express this distinction.

**Goal**: Under a general setting containing both latent confounders $L$ and selection $S$, the paper aims to: (1) establish a causal formalization that explicitly expresses post-treatment selection; (2) characterize its Markov properties and define a finer interventional equivalence class; and (3) provide a provably sound and complete algorithm to learn it from data.

**Key Insight**: The authors observe that while post-treatment selection and causal relationships are indistinguishable at the endpoints of a "cause-effect" pair, they exhibit different symmetries and interventional responses when considering **intermediate variables along the path**. For example, a hard intervention on an intermediate node $X_3$ can "open" a path blocked by the selection effect. Signals like $\psi_3 \not\perp\!\!\!\perp X_2$ can then determine whether a direct causal edge exists between $X_1$ and $X_2$.

**Core Idea**: By using "augmented DAGs + selection variables + extra hard interventions on intermediate induced nodes," the traditional equivalence class is refined into **FI-Markov equivalence**. A set of graph representations called **F-PAG** with new edge types and the **F-FCI** algorithm are introduced to bypass coarse equivalence classes and reach the true structure at the DAG level.

## Method

### Overall Architecture
The paper addresses the unidentifiability of post-treatment selection masquerading as causality. The approach progresses through three layers: **Modeling**—incorporating selection variables $S$ and intervention indicators $\psi$ into an augmented DAG to unify observational and interventional data; **Characterization**—analyzing Markov properties to define FI-Markov equivalence and the F-PAG representation; and **Learning**—providing the F-FCI algorithm to recover F-PAG from data with soundness and completeness guarantees.

Data generation is formulated as a factorization conditioned on $S=1$ (i.e., "selected"). The joint distribution under the $k$-th intervention is:

$$p^{(k)}_s(X) = \prod_{i\in I^{(k)}} p^{(k)}\!\big(X_i \mid \hat X_{\mathrm{pa}_G(i)}, S{=}1\big)\;\prod_{j\notin I^{(k)}} p^{(0)}\!\big(X_j \mid \hat X_{\mathrm{pa}_G(j)}, S{=}1\big),$$

where intervened variables $X_i\,(i\in I^{(k)})$ are replaced by post-intervention distributions, while others remain under the observational distribution, all conditioned on $S=1$. This reflects the fact that all samples (observational or interventional) pass through the selection filter.

### Key Designs

**1. Explicit Modeling of Post-treatment Selection via Augmented DAGs: Incorporating $S$ and $\psi$ into the graph**

Current frameworks fail to distinguish post-treatment selection from causality because "selection" is absent from their models. The paper first explicitly represents the action of changing intervention targets: beyond the original DAG $G$ (vertices $X$ and $L$), a set of exogenous binary indicators $\psi=\{\psi_{I^{(k)}}\}$ is added, each pointing to its intervened variable $X_{I^{(k)}}$. Thus, whether the $k$-th intervention changes the marginal distribution of $X_A$ is translated into the CI relationship $\psi_{I^{(k)}}\not\perp\!\!\!\perp X_A$, corresponding to d-separation $\psi_{I^{(k)}}\not\perp\!\!\!\perp_d X_A$ in the augmented DAG. Finally, selection variable $S$ (acting on at least two observed variables) is added, with all analysis conducted under $S=1$. This unification allows observational data $p(X\mid\psi{=}0,S{=}1)$ and interventional data $p(X\mid\psi{=}1,S{=}1)$ to coexist in one augmented DAG (Definition 1).

**2. Characterizing Markov Properties: Finding CI Fingerprints to Separate Selection, Confounding, and Causality**

The authors prove (Theorem 1) that under $S=1$, d-separation between $\psi\cup X$ in the augmented DAG precisely implies CI and invariance in the data. Specifically, "$\psi_{I^{(k)}}\perp\!\!\!\perp_d X_A\mid X_B$" holds in the graph if and only if "$p^{(k)}(X_A\mid X_B)=p^{(0)}(X_A\mid X_B)$." Three types of statistical signals are utilized: **interventional distribution changes**, **invariant relationships**, and **structural symmetries**. For instance, a symmetric selection structure yields $\psi_1\perp\!\!\!\perp X_2\mid X_1,\ \psi_2\perp\!\!\!\perp X_1\mid X_2,\ \psi_1\not\perp\!\!\!\perp X_2,\ \psi_2\not\perp\!\!\!\perp X_1$ (Figure 4(e)). This "tails at both ends" symmetric fingerprint distinguishes direct selection from causality. Lemma 1 notes that selection introduces additional dependencies, necessitating explicit modeling.

**3. FI-Markov Equivalence + F-PAG: Finer Equivalence Classes and New Graph Language**

Traditional methods only recover equivalence classes at the MAG/PAG level, using "circle" edges like $\circ\!\!-\!\!\circ$ that bundle different structures together. This paper defines **FI-Markov equivalence** (Definition 2): two augmented DAGs with the same intervention targets are equivalent if they share the same skeleton and v-structures on $X_{[N]\setminus I}$ and the same $\psi$–$X$ CI patterns. To uniquely represent this finer class, the authors propose **F-PAG** (Definition 5), adding a "square $\square$" endpoint (indicating it is at least one tail and at least one arrowhead) and new edge types like $\blacktriangleright\!\!\to$ and $\blacktriangleright\!\!-$. These describe induced paths that share CI patterns with $\to$ or $-$ but lack a direct causal edge or direct selection. The key to identifying these is the **Type I inducing node** (Definition 6).

**4. F-FCI Algorithm: Disambiguation via Intermediate Node Interventions**

**F-FCI** (Algorithm 1) operates in three steps. Step 1 recovers the skeleton from observational data using FCI-style constraints. Step 2 orients edges between intervened variables using observational-interventional CI patterns (outputting $\to, \leftrightarrow, \circ\!\!\to, -, -\square, \square\!-\!\square$). Step 2.3 is the **core of disambiguation**: for ambiguous edges, the algorithm identifies a Type I inducing node $X_n$ along the path and performs an **additional hard intervention** to test $\psi_n\perp\!\!\!\perp X_{I(i)}$. The intuition is that a hard intervention on $X_n$ "blocks" selection effects on latent confounders and opens the blocked path. If $\psi_n\perp\!\!\!\perp X_2\mid S$ holds, $X_1\to X_2$ is identified as an induced path rather than a direct edge, and the edge is updated to $\blacktriangleright\!\!-$/$\blacktriangleright\!\!\to$. The algorithm is proven **sound** (Theorem 3) and **complete** (Theorem 4).

### Loss & Training
This is a constraint-based causal discovery method and does not involve a differentiable objective or training process. The algorithm relies on the **faithfulness assumption** and oracle CI tests. In practice, statistical CI tests are used, and identification quality improves with sample size $n$.

## Key Experimental Results

### Main Results
Evaluated on synthetic data against 6 strong baselines (GIES, IGSP, UT-IGSP, JCI-GSP, FCI-interven, CDIS) using **DAG Precision (↑)** and **SHD (↓)** across various configurations (hard/soft interventions, $d\in\{10,\dots,25\}$, $n\in\{500,1500,2000\}$).

| Setting | Metric | F-FCI (Ours) | Baselines | Conclusion |
|------|------|------|----------|------|
| Hard/Soft Intv, $d$=10–25 | DAG Precision ↑ | Leads in most configs | GIES/IGSP/UT-IGSP/JCI-GSP/FCI-interven/CDIS | >5% higher on average |
| Same as above | SHD ↓ | Lower | Same as above | Smaller structural error |

Baselines trail because they misinterpret spurious dependencies induced by latent confounding and post-treatment selection as true causal edges.

### Key Findings
- **Identifiability depends on Type I inducing nodes**: The ability to distinguish direct causal edges/selection hinges on the presence of Type I inducing nodes for additional hard interventions.
- **Real-world data verification**: Tested on Norman single-cell gene perturbation data (HLEC). F-FCI reported both regulatory (causal) edges and spurious dependencies induced by post-treatment selection, validated using the Enrichr prior knowledge base.

## Highlights & Insights
- **Problem Statement as the Major Contribution**: Bringing "post-treatment selection" to light and showing it shares fingerprints with causality—representing a clear and previously unaddressed representational gap.
- **Clever Use of Intermediate Nodes**: Using hard interventions on intermediate Type I inducing nodes ($\psi_n\perp\!\!\!\perp X$ test) to separate indistinguishable endpoints. This expands the "where to intervene" scope beyond just causal endpoints.
- **Complete Graph Language Solution**: The square marker $\square$ and $\blacktriangleright$ edges provide a unique representation for FI-Markov equivalence, avoiding the ambiguity of standard PAGs.
- **Bridging Theory to Biological Application**: Post-treatment selection (e.g., quality control filtering) is ubiquitous in single-cell experiments, making the method highly relevant to gene regulatory network discovery.

## Limitations & Future Work
- **Dependency on Type I Nodes**: Identifiability relies on the presence of Type I inducing nodes. Handling induced paths consisting entirely of Type II inducing nodes remains an open problem.
- **Biological Constraints vs. Post-treatment Selection**: It remains difficult to separate biological constraints from selection variables.
- **Reliance on Strong Assumptions**: The method assumes faithfulness and reliable CI tests. Structural recognition is sensitive to CI test errors in finite samples.
- **Future Directions**: Generalizing disambiguation criteria to soft interventions or partially intervenable scenarios would widen the application scope.

## Related Work & Insights
- **vs. Traditional Interventional Equivalence (GIES / IGSP)**: These methods use "marginal change, conditional invariance" for orientation, which post-treatment selection mimics. F-FCI refines this to FI-Markov equivalence to overcome this barrier.
- **vs. FCI-interven**: While it handles latent confounders, it does not model post-treatment selection. F-FCI inherits its invariance rules but adds Type I node disambiguation and new edge types.
- **vs. CDIS**: CDIS focuses on **pre-treatment** selection; this paper addresses **post-treatment** selection, covering the other half of selection bias complexities.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic formalization of post-treatment selection with a new equivalence class (FI-Markov), graph language (F-PAG), and algorithm (F-FCI).
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparisons on synthetic data and validation on single-cell data; theoretically sound and complete.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation via examples, though dense in definitions and notation.
- Value: ⭐⭐⭐⭐⭐ Directly addresses quality control selection bias in single-cell and clinical fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Characterization and Learning of Causal Graphs from Hard Interventions](../../NeurIPS2025/causal_inference/characterization_and_learning_of_causal_graphs_from_hard_interventions.md)
- [\[ICLR 2026\] Beyond DAGs: A Latent Partial Causal Model for Multimodal Learning](beyond_dags_a_latent_partial_causal_model_for_multimodal_learning.md)
- [\[ICLR 2026\] Causal Structure Learning in Hawkes Processes with Complex Latent Confounder Networks](causal_structure_learning_in_hawkes_processes_with_complex_latent_confounder_net.md)
- [\[ICLR 2026\] On the Identifiability of Causal Graphs with the Invariance Principle](on_the_identifiability_of_causal_graphs_with_the_invariance_principle.md)
- [\[ICLR 2026\] Stochastic Neural Networks for Causal Inference with Missing Confounders](stochastic_neural_networks_for_causal_inference_with_missing_confounders.md)

</div>

<!-- RELATED:END -->
