---
title: >-
  [Paper Note] Dimensionality Controls When Modularity Helps in Continual Learning
description: >-
  [ICML2026][Interpretability][Continual Learning] This paper systematically compares "task-blocked modular recurrent networks" with "single networks" using an A→B→A sequential learning paradigm. It finds that modularity is not always beneficial—only when the initialization scale $\gamma$ compresses representations into the **low-dimensional "rich" regime** does modularity lead to lower interference and spontaneously organize a gradient geometry where "similar tasks overlap in…
tags:
  - "ICML2026"
  - "Interpretability"
  - "Continual Learning"
  - "Catastrophic Forgetting"
  - "Compositionality"
  - "Representation Dimensionality"
  - "Modularity"
  - "Stability-Plasticity"
date: 2026-05-08
content_hash: f4cb4e3c48da08fb
---

# Dimensionality Controls When Modularity Helps in Continual Learning

**Conference**: ICML2026  
**arXiv**: [2606.17889](https://arxiv.org/abs/2606.17889)  
**Code**: TBD  
**Area**: Interpretability / Representation Geometry / Continual Learning  
**Keywords**: Continual Learning, Catastrophic Forgetting, Compositionality, Representation Dimensionality, Modularity, Stability-Plasticity

## TL;DR
This paper systematically compares "task-blocked modular recurrent networks" with "single networks" using an A→B→A sequential learning paradigm. It finds that modularity is not always beneficial—only when the initialization scale $\gamma$ compresses representations into the **low-dimensional "rich" regime** does modularity lead to lower interference and spontaneously organize a gradient geometry where "similar tasks overlap in subspace while dissimilar tasks are orthogonal." In the high-dimensional "lazy" regime, the two architectures show negligible differences.

## Background & Motivation

**Background**: The core of continual learning is the stability–plasticity dilemma: learning new tasks requires reusing and rewriting old representations (plasticity), while preserving old tasks requires constraining these representations from being overwritten (stability). Compositionality is considered a hallmark of robust generalization—ideal systems should reuse components for similar tasks and remain separated for dissimilar ones. Methods to mitigate interference include replay, synaptic consolidation (e.g., EWC), and structural isolation through **modularity**.

**Limitations of Prior Work**: Whether modularity is actually beneficial remains inconsistent in the literature. On one hand, modular isolation reduces interference; on the other hand, **complete structural separation eliminates both transfer and interference**, potentially losing opportunities for compositional reuse. Consequently, it remains unclear "when modularity helps and when it hinders."

**Key Challenge**: The authors identify **representation dimensionality** as the critical variable. Prior work suggests that "rich" (low-dimensional, structured) and "lazy" (high-dimensional, close to initialization) learning regimes induce distinct representation geometries; modular organization often only emerges clearly when representations are low-dimensional and compressed. In other words, structural biases (like modularity) may only impact behavior **when the representation space is constrained tightly enough that geometry becomes a binding constraint**—architecture alone is insufficient; it depends on the representation regime it falls into.

**Goal**: To jointly investigate how "modular architecture $\times$ task similarity $\times$ representation dimensionality" shape compositional continual learning within a controlled transfer-interference paradigm, grounding the question of "when is modularity beneficial" in the adjustable knob of representation dimensionality.

**Key Insight**: Drawing on the A1→B→A2 paradigm from Holton et al. (2026), the initialization scale $\gamma$ is used as a control variable to vary effective representation dimensionality (large $\gamma$=lazy/high-dim, small $\gamma$=rich/low-dim). The study systematically sweeps across task similarity (same/near/far) and $\gamma$, measuring both behavioral metrics (accuracy, transfer, interference) and hidden state geometry (effective dimension, principal angles, 3D PCA trajectories).

**Core Idea**: The benefits of modularity are **conditional** rather than universal—representation dimensionality acts as a "gatekeeper." Only in the low-dimensional rich regime does modularity significantly reduce interference and induce similarity-dependent subspace geometry.

## Method

### Overall Architecture

This is a controlled comparative analysis study. It does not propose a new algorithm but uses an experimental design to isolate the interaction of "dimensionality $\times$ architecture $\times$ similarity." The workflow involves fitting trial sequences derived from original human experiments using recurrent networks under an A1→B→A2 three-stage protocol (A1 learns task A, B learns task B, A2 retests A). The task involves mapping six plant cues to angles on a disk; summer/winter seasons are related by a fixed angular offset (task rule). Task B uses new stimuli but the same formal structure, with Same/Near/Far levels controlling the offset of rule B relative to A. Two architectures (Single network vs. Task-blocked modular network) are tested. After initialization, all trainable weights are multiplied by a global factor $\gamma$ to create high/low-dimensional regimes. Behavioral and geometric metrics are then analyzed.

### Key Designs

**1. A1→B→A2 Transfer-Interference Paradigm + Same/Near/Far Similarity: Controlling Reuse vs. Interference**

To answer "when modularity helps," a controlled task is needed to expose both transfer opportunities and interference risks. The authors use the sequential protocol from Holton et al. (2026): task A is learned to ceiling in A1, task B is learned with new stimuli in stage B, and A is retested in A2 to quantify forgetting/interference. Task B shares the **same formal structure** as A (plant → angle, summer/winter linked by rule), varying only in the similarity of the rule value—Same (identical), Near (small offset), Far (large offset). Similarity thus becomes a clean knob: reuse should be beneficial when similar and harmful when dissimilar. Transfer is measured as "Stage B first six winter trials accuracy − Stage A1 last six winter trials accuracy," while interference is quantified in A2 using a von Mises mixture model to measure how much the response is pushed towards the task-B rule.

**2. Controlling Representation Dimensionality via $\gamma$: A Continuous Knob for Lazy/Rich Regimes**

This is the causal lever of the paper. After default PyTorch initialization, all trainable weights are multiplied by a global factor $\gamma \in \{0.001, 0.01, 0.1, 1.0, 2.0\}$. Large $\gamma$ scales up initial weights, leading to a lazy, high-dimensional regime (representations stay near initialization, high dimensionality, weak specialization). Small $\gamma$ scales down weights, leading to a rich, low-dimensional regime (structured, low-dimensional, specialized encoding). This manipulation is closely tied to the transition between lazy and rich learning regimes, which correlates with differences in effective representation dimensionality. The authors characterize $\gamma$ as a "practical probe for rich-vs-lazy regimes" rather than a pure dimensionality control, as it also affects optimization dynamics.

**3. Task-Blocked Modular Network vs. Single Network Baseline: Isolating Structural Separation**

To isolate the effect of "structural separation," the authors compare two recurrent architectures. The single network passes all inputs through one recurrent population $h_t = \tanh(W_{\text{ih}}x_t + W_{\text{hh}}h_{t-1})$ and a shared readout $y_t = W_{\text{out}}h_t$. The modular network contains two recurrent modules $M_A, M_B$. Inputs are routed by task identity (using binary masks $x_t^A = m_A \odot x_t$). While modules are independent in the recurrence (no inter-module communication in the main analysis), **the states of both modules are concatenated and passed through a shared readout**. The subtlety here is that "separation is structural but not absolute": due to the shared readout, the model must coordinate module activity at the output layer, creating "constrained specialization" rather than strict isolation. This explains the later observation of similarity-dependent overlapping representations. Parameter budgets are aligned for fair comparison.

**4. Representation Geometry Analysis: Quantifying "Geometry"**

To attribute behavioral differences to representation regimes, geometry must be measured. The authors take the hidden states of the last time step for every stimulus and stage to form a matrix for PCA. **Effective dimension** is defined as the number of PCs required to explain 99% of the variance. **Principal angles** are calculated between two-dimensional subspaces fitted to task A and task B hidden states to quantify if task subspaces are overlapping, partially aligned, or near-orthogonal. **3D PCA trajectories** project hidden states onto the first three PCs of the joint A1/B/A2 data for qualitative visualization.

### Loss & Training

All models use MSE loss targeting a four-dimensional cosine-sine encoding (summer cos/sin + winter cos/sin). Loss is calculated only on the probed feature component per trial to match the sequential protocol. Training uses SGD (LR 0.01) for 100 epochs per stage. Inputs are unrolled for two recurrent time steps, and hidden states are reset between trials. Results are reported across 305 participant training schedules derived from the original experiments.

## Key Experimental Results

### Main Results

The core comparison evaluates Modular vs. Single networks across different $\gamma$ (representation regimes) and task similarities, specifically focusing on stability in Stage A2 (retesting A):

| Regime / Similarity | Modular Network | Single Network |
|---------------------|-----------------|----------------|
| High-dim "Lazy" (Large $\gamma$) | High A2 accuracy, low interference | **Almost no difference** from modular |
| Low-dim "Rich" (Small $\gamma$) · Same | High A2, low interference | Equivalent performance |
| Low-dim "Rich" · Near / Far | Consistently high A2, **persistently low interference** | A2 **drops significantly** at smallest $\gamma$ in Far condition; interference rises |

Conclusion: Structural isolation **does not provide consistent benefits**—in high-dimensional regimes or when tasks are identical, single and modular networks perform similarly. Only in the low-dimensional regime with dissimilar tasks does modularity significantly suppress interference while preserving performance on both tasks.

### Ablation Study

| Analysis | High-dim Lazy Regime | Low-dim Rich Regime |
|----------|----------------------|----------------------|
| Effective Dimension (99% Var) | Both architectures high; weak similarity dependence | Dimensions collapse sharply as $\gamma$ decreases |
| Principal Angles (same/near/far) | Similar geometry for both; entangled subspaces | **Modular**: Gradient geometry (aligned for same, partial for near, orthogonal for far); **Single**: Weak similarity dependence, entangled subspaces |
| 3D PCA Trajectory | Large state space usage; diffuse structure; minimal architecture difference | Compact trajectories; **Modular** shows ordered similarity-based arrangement; **Single** shows more overlap |
| Architecture Ablations | — | Similarity-dependent geometry and selective modularity benefits **robustly preserved** | 

### Key Findings

- **Dimensionality is a Gatekeeper**: The behavioral benefit of modularity over the single network **co-occurs** with the transition from high to low effective dimensionality. In high dimensions, both architectures have sufficient degrees of freedom; in low dimensions, representation capacity becomes a binding constraint where tasks compete for directions, and architectural biases begin to dictate how capacity is allocated.
- **Modularity Induces Similarity-Dependent Geometry**: In the rich regime, modular networks align subspaces for similar tasks and orthogonalize them for dissimilar ones, a gradient structure lacking in single networks.
- **Separation is Structural but Not Absolute**: The shared readout forces modules to coordinate at the output, resulting in "similarity-dependent yet overlapping" representations rather than total isolation—which preserves compositional reuse.
- Ablations (module width, input routing, inter-module connections, initialization range, recurrence depth) all preserved these patterns, **ruling out pure capacity-based explanations**.

## Highlights & Insights
- **Reframing "Is Modularity Useful?" to "In Which Regime is Modularity Useful?"**: The study's most valuable shift is moving away from asking if an architecture IS modular to asking under what representation regime modularity changes behavior—this unifies conflicting literature.
- **$\gamma$ as a Dimensionality Knob**: A minimalist global scaling factor allows continuous sweeping across lazy $\leftrightarrow$ rich regimes. The authors' cautious labeling of it as a probe for regimes rather than a pure dimensionality variable is methodologically sound.
- **"The goal of continual learning is similarity-dependent geometry, not max separation"**: The authors propose viewing robustness as a problem of "adaptive subspace allocation"—overlapping when similar, partially reorganizing when medium, and separating when dissimilar.
- The "constrained specialization" design (shared readout + task routing) is transferable to multi-task systems requiring both reuse and isolation.

## Limitations & Future Work
- The authors acknowledge that PCA effective dimension is only a proxy for intrinsic dimensionality; $\gamma$ also changes optimization dynamics and architectural expressivity, not just dimensionality.
- The protocol is limited to short A1→B→A2 sequences and does not cover longer, more heterogeneous task streams. Whether dimensionality-dependent modularity effects scale to large-scale benchmarks and realistic task distributions remains an open question.
- **Ours** identified limitations: The task itself is a low-freedom angular regression toy task with small recurrent units (25–50); caution is needed when extrapolating to large models or real-world data.
- Future Directions: Testing how geometry evolves over repeated interference in longer sequences; using regularizers/bottlenecks to **directly control dimensionality** (rather than relying on initialization); and comparing modular designs with explicit inter-module communication.

## Related Work & Insights
- **vs. Holton et al. (2026)**: Uses their A1→B→A2 paradigm but shifts focus from human-network similarity to the "dimensionality $\times$ modularity" interaction.
- **vs. Johnston & Fusi (2024/2026)**: They noted that modular representations tend to emerge when inputs/hidden states are low-dimensional; this paper links that observation to behavioral consequences in continual learning.
- **vs. Flesch et al. (2021) / Flesch (2022)**: They established how rich/lazy regimes induce different geometries and transfer-interference tradeoffs; this paper reproduces that transition and bridges it to architectural choice.
- **vs. Classical Anti-Forgetting (EWC, Replay)**: Those methods constrain parameters or replay data at the algorithmic level; this paper argues that the stability-plasticity tradeoff is **partially architectural**—different architectures achieve different balances under fixed parameter budgets.

## Rating
- Novelty: ⭐⭐⭐⭐ Grounding modularity benefits in dimensionality as a gating variable provides a clear perspective for unifying literature.
- Experimental Thoroughness: ⭐⭐⭐ Controlled paradigm with systematic $\gamma \times$ similarity $\times$ architecture sweeps and five ablations is solid, but limited to toy-scale tasks.
- Writing Quality: ⭐⭐⭐⭐ The three-layered evidence (behavior-dimension-geometry) is logically tight; limitations are addressed honestly.
- Value: ⭐⭐⭐⭐ Provides actionable insights into when to use modularity and how to tune dimensionality for continual learning architecture design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prompt Optimization Is a Coin Flip: Diagnosing When It Helps in Compound AI Systems](prompt_optimization_is_a_coin_flip_diagnosing_when_it_helps_in_compound_ai_syste.md)
- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](../../CVPR2025/interpretability/language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)
- [\[ICLR 2026\] LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data](../../ICLR2026/interpretability/lore_jointly_learning_the_intrinsic_dimensionality_and_relative_similarity_struc.md)
- [\[ICLR 2026\] Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](../../ICLR2026/interpretability/towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)
- [\[ICLR 2026\] When Machine Learning Gets Personal: Evaluating Prediction and Explanation](../../ICLR2026/interpretability/when_machine_learning_gets_personal_evaluating_prediction_and_explanation.md)

</div>

<!-- RELATED:END -->
