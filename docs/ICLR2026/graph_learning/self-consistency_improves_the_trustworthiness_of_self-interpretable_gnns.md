---
title: >-
  [Paper Note] Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs
description: >-
  [ICLR 2026][Graph Learning][Fine-tuning] Self-interpretable GNNs (SI-GNNs) optimize cross-entropy and sparsity during training but are evaluated on faithfulness, creating a training-evaluation misalignment. This paper posits that faithfulness is essentially equivalent to "explanation self-consistency." By introducing a self-consistency (SC) loss that aligns t
tags:
  - ICLR 2026
  - Graph Learning
  - Fine-tuning
date: 2026-05-08
content_hash: d586bd911181c4bd
---
# Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=hxGdAUn3sB](https://openreview.net/forum?id=hxGdAUn3sB)  
**Code**: https://github.com/ICDM-UESTC/SelfConsistencyXGNN  
**Area**: Graph Learning / Interpretability  
**Keywords**: Self-interpretable GNNs, Explanation Faithfulness, Self-Consistency, Explanation Redundancy, Fine-tuning  

## TL;DR
Self-interpretable GNNs (SI-GNNs) optimize cross-entropy and sparsity during training but are evaluated on faithfulness, creating a training-evaluation misalignment. This paper posits that faithfulness is essentially equivalent to "explanation self-consistency." By introducing a self-consistency (SC) loss that aligns the original explanation with a secondary explanation generated after feeding the first back into the model, a model-agnostic fine-tuning approach is proposed to simultaneously improve explanation quality across consistency, accuracy, faithfulness, and informativeness.

## Background & Motivation

**Background**: GNNs provide strong predictions but act as black boxes, hindering their deployment in high-stakes or scientific scenarios. Self-interpretable GNNs (SI-GNNs) integrate an internal explainer $h_{G_s}$ to simultaneously learn predictions and explanations in an end-to-end fashion—assigning importance scores $\alpha_{ij}$ to each edge to select a subgraph $G_s \subseteq G$ as the explanation. Representative methods categorized by subset selection strategy include: attention-based (GAT), causal-based (CAL), size-constrained (SMGNN), and mutual information-constrained (GSAT).

**Limitations of Prior Work**: **Faithfulness** is commonly used to evaluate explanation quality—meaning if the explanatory subgraph $G_s$ is fed back into the model, the prediction remains unchanged (widely applicable as it doesn't rely on ground-truth labels). The problem is that SI-GNNs are trained using cross-entropy $L_{CE}$ plus a sparsity regularizer $R(G_s)$, with **no term explicitly optimizing faithfulness**. There is a disconnect between optimization and evaluation.

**Key Challenge**: The verification process of faithfulness (feeding the explanation back and checking prediction stability) implicitly requires the explainer to be stable and consistent across repeated extractions for the same instance. If the explainer truly captures the decisive structure, it should highlight the same subgraph a second time. In other words, **faithfulness inherently relies on "self-consistency"**: stable explanations lead to stable predictions, which satisfies faithfulness. Thus, faithfulness can be directly optimized via a loss that aligns consecutive explanations.

**Goal**: This work addresses two sub-problems: (i) Can the property of faithfulness be explicitly optimized during training? (ii) If so, does it truly improve explanation quality?

**Key Insight**: Empirical analysis (Figure 1) reveals that without self-consistency training, the first and second explanations of SI-GNNs differ significantly. This "self-inconsistency" primarily occurs on features where the ground-truth is **unimportant**, while important features remain stable. This aligns with "explanation redundancy" found in recent work (Tai et al., 2025): when sparsity constraints are insufficient, excessive budgets allow the explainer to irresponsibly assign high scores to unimportant edges. Since self-inconsistency is concentrated on unimportant edges, **fixing self-inconsistency $\approx$ mitigating redundancy $\approx$ improving quality**.

**Core Idea**: Add a self-consistency (SC) loss on top of the standard SI-GNN objective to minimize the difference between two consecutive explanations, implemented as a model-agnostic fine-tuning step.

## Method

### Overall Architecture
The method addresses the absence of faithfulness in training objectives by translating faithfulness into a differentiable self-consistency loss, injected into any SI-GNN via two-step fine-tuning. The process involves training the SI-GNN to convergence with standard objectives, freezing the GNN encoder (ensuring representation learning is not corrupted by subsequent losses while SC loss affects only the explainer), and then performing self-consistency fine-tuning. Given a graph $G$, the explainer generates the first explanation $G_s^{(1)}$, which is fed back into the model to generate the second explanation $G_s^{(2)}$, with an alignment loss forcing consistency between them. Only the explainer and classifier are updated during fine-tuning.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input Graph G"] --> B["Standard Training SI-GNN<br/>CE + Sparsity Regularization"]
    B --> C["Freeze GNN Encoder<br/>Tune Explainer/Classifier only"]
    C --> D["1st Explanation G_s^(1)<br/>Explainer Edge Scores α_ij"]
    D -->|Feed G_s^(1) back| E["2nd Explanation G_s^(2)"]
    E --> F["SC Loss Alignment<br/>L_SC = |G_s^(1) − G_s^(2)|"]
    F --> G["Output: More Trustworthy Explanations"]
```

### Key Designs

**1. Self-Consistency Fine-tuning: Translating "Faithfulness" into a Differentiable Dual-Process Alignment Loss**

Faithfulness ("prediction remains unchanged after feed-back") is a verification property that is non-differentiable and cannot serve as a training target. The key observation is that this verification implicitly requires the explainer to be self-consistent. By forcing consistency between two outputs, faithfulness is indirectly optimized. Specifically, the explainer generates $G_s^{(1)}$, which is fed back into the model (with frozen encoder) to obtain $G_s^{(2)}$. The SC loss is the L1 difference:

$$L_{SC} = |G_s^{(1)} - G_s^{(2)}|.$$

The final objective adds this to the standard SI-GNN loss: $L_{FT} = L_{GE} + \eta \cdot L_{SC}$, where $\eta$ is the weight. In the explainer, edge $e_{ij}$ importance is given by $w_{ij} = \text{MLP}([v_i; v_j])$ and $\alpha_{ij} = \sigma(w_{ij})$, using Gumbel–Sigmoid for differentiable sampling. This design is effective because it **does not modify architecture and only adds one loss term**, making it applicable to attention, causal, size, and MI-based SI-GNNs. The "pretrain-then-freeze-encoder" step ensures the SC loss reshapes explainer behavior without destabilizing learned representations.

**2. Near-fixed levels: Explaining why SC converges scores to stable states**

To understand what SC does to edge scores, define the mapping for the second pass as $T(\alpha) = \sigma(g(\alpha))$, where $g(\cdot)$ is the second-pass pre-activation. Enforcing self-consistency is equivalent to requiring $T(\alpha) \approx \alpha$. The authors characterize this using "near-fixed points": for a tolerance $\varepsilon$, $\alpha$ is an $\varepsilon$-near-fixed point iff $|T(\alpha) - \alpha| \le \varepsilon$, which means pre-activation falls into a logit window $g(a^*) \in [\text{logit}(a^*-\varepsilon),\ \text{logit}(a^*+\varepsilon)]$. The window width $\Delta g \approx \frac{2\varepsilon}{a^*(1-a^*)}$ is finite for internal points but degrades into a one-sided threshold as $a^* \to 0$ or $1$ (corresponding to sigmoid saturation zones). Consequently, **the extreme levels 0 and 1 are easier to reach than intermediate levels**. SC pushes scores toward a few stable near-fixed levels: important edges are pushed toward 1 by classification loss, while unimportant edges settle at a low, stable level.

**3. Interaction with Sparsity Regularization: CR strength determines convergence levels**

Design 2 explains stability, but the specific level is determined by the sparsity/complexity regularizer (CR). The gradient of the joint loss w.r.t. edge scores consists of three forces:

$$\frac{\partial L}{\partial \alpha_{ij}} \approx \underbrace{\frac{\partial L_{CE}}{\partial \alpha_{ij}}}_{\text{Classification}} + \beta \cdot \underbrace{\frac{\partial R(G_s)}{\partial \alpha_{ij}}}_{\text{Complexity}} + \eta \cdot \underbrace{\frac{\partial L_{SC}}{\partial \alpha_{ij}}}_{\text{Stability}}.$$

The classification term pushes important edges to 1. The complexity term depends on the CR form (SMGNN encourages 0; GSAT encourages 0.5 for independence). The stability term pulls scores towards near-fixed points. The game results in three regimes: when $\beta$ is too weak, CR is inactive and unimportant edges settle at arbitrary levels; when $\beta$ is moderate, CR actively suppresses unimportant edges (to 0 for SMGNN, 0.5 for GSAT) while important edges stay near 1—**here, CR and SC jointly push unimportant edges to low, stable scores, maximizing quality**; when $\beta$ is too strong, CR overrides classification, collapsing all edges (both toward 0 or 0.5), which destroys quality. This clarifies why SC alone is unstable for GAT/CAL (which lack CR) and requires the +CR+SC configuration.

### Loss & Training
A two-step strategy: Step 1 uses the standard objective $L_{GE}$ (specific to backbones) until convergence and freezes the encoder; Step 2 uses $L_{FT} = L_{GE} + \eta \cdot L_{SC}$ to fine-tune the explainer and classifier. $\eta$ controls the strength of self-consistency, and $\beta$ controls sparsity, requiring careful coordination (optimal in the moderate $\beta$ regime).

## Key Experimental Results

Datasets: Synthetic BA-2MOTIFS and three real-world molecular datasets: 3MR, BENZENE, and MUTAGENICITY. Metrics: SHD↓ (Consistency), AUC↑ (Accuracy), ACC↑ (Downstream Informativeness), FID↓ (Faithfulness). Baselines include original backbones and Explanation Ensemble (EE).

### Main Results (SMGNN / GSAT with CR)

| Method | BA-2MOTIFS SHD↓ | BA-2MOTIFS AUC↑ | BENZENE SHD↓ | BENZENE AUC↑ | MUTAG SHD↓ | MUTAG FID↓ |
|------|------|------|------|------|------|------|
| SMGNN | 10.44 | 99.32 | 16.06 | 84.38 | 12.65 | 1.72 |
| SMGNN+EE | 4.99 | 99.59 | 8.55 | 91.38 | 6.21 | – |
| SMGNN+SC | 3.48 | 99.87 | 7.19 | 90.07 | 2.51 | 0.61 |
| SMGNN+SC+EE | **1.52** | **99.90** | **4.14** | **92.20** | **1.44** | – |
| GSAT | 4.58 | 98.44 | 6.93 | 90.66 | 10.08 | 1.11 |
| GSAT+SC | 2.73 | 99.30 | 2.32 | 92.80 | 2.38 | 0.17 |
| GSAT+SC+EE | **1.19** | **99.35** | **1.14** | **93.53** | **1.06** | – |

SC provides comprehensive improvements across four dimensions. In most cases, SC outperforms EE while being ~5× faster and compatible with all standard metrics. SC and EE are complementary and yield further gains when combined.

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| GAT+SC (No CR) | MUTAG SHD 0.04 but AUC drops to 81.79 | SC is unstable without CR; important and unimportant edges collapse together. |
| GAT+CR+SC | AUC recovers to 99.87, SHD 3.48 | SC becomes stable and effective once CR is added. |
| CAL+SC (No CR) | BENZENE AUC 88.74, still below +CR | Causal backbones also require CR coordination. |
| CAL+CR+SC | AUC 89.87, SHD 6.25 | CR+SC joint configuration is the most stable. |
| Moderate vs Strong $\beta$ | Collapse when $\beta$ is too strong | Confirms the three-way gradient game; $\beta$ must fall in the moderate zone. |

Self-consistency is also directly verified: in Table 2, the cosine similarity between $G_s^{(1)}$ and $G_s^{(2)}$ for SMGNN rises from 99.68% to 99.98%, and L1 distance drops from 16.87% to 1.84% (BA-2MOTIFS). PCA (Figure 7) shows significantly shorter lines between consecutive representations.

### Key Findings
- SC improves faithfulness (FID) by directly minimizing differences between consecutive passes; consistent explanations lead to consistent representations and predictions, naturally lowering FID.
- Quality improvements stem almost entirely from pushing **unimportant edges** to low and stable levels (Figure 3/4). Important edges are shielded near 1 by classification loss, echoing the motivation that self-inconsistency is concentrated in unimportant features.
- SC effectiveness strongly depends on CR: without it, GAT/CAL with SC collapse, making important and unimportant edges indistinguishable.

## Highlights & Insights
- **Translating Non-differentiable Metrics to Differentiable Targets**: Faithfulness is traditionally a "check-and-see" property. By identifying self-consistency as an implicit requirement, the authors use a feed-back alignment L1 loss to make it optimizable—providing a template for bridging training and evaluation gaps in other XAI tasks.
- **Model-Agnostic and Plug-and-Play**: No architectural changes, just one loss and a fine-tuning step. Applicable to four major classes of SI-GNNs with minimal engineering overhead.
- **Theoretical Grounding**: Use of near-fixed points and gradient dynamics explains *why* SC converges scores to stable levels and how CR strength dictates that stability, moving beyond purely empirical tricks.

## Limitations & Future Work
- Validated only on global/instance-level edge importance; does not cover node features or other graph explanation formats.
- Dataset scale is relatively small (one synthetic + three molecular); needs verification on large-scale or more complex graph tasks.
- Sensitivity to CR strength $\beta$: the "moderate zone" is qualitative. Automatic selection of $\beta$ and $\eta$ remains an open problem for deployment.
- Self-consistency is a necessary but not sufficient condition for faithfulness—a stable but wrong explanation can still be self-consistent. While experiments mitigate this through CR and ground-truth labels, the "consistently wrong" possibility theoretically remains.

## Related Work & Insights
- **vs. Explanation Ensemble (EE, Tai et al., 2025)**: EE is a post-processing method using multiple ensembles to suppress redundancy; this work is a training-time single-pass fine-tuning for self-consistency. SC is generally higher quality, ~5× faster, and complementary to EE.
- **vs. Explanation Redundancy Analysis (Tai et al., 2025)**: That work focuses on inconsistency across models (different seeds) and attributes it to redundancy. This work reveals "self-inconsistency" within a single model and addresses it via explicit consistency constraints during training.
- **vs. Standard SI-GNNs (GAT/CAL/SMGNN/GSAT)**: These define various sparsity priors but ignore faithfulness. This work provides an orthogonal layer to align the training objective with faithfulness evaluation for any backbone.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative perspective equating faithfulness with self-consistency for differentiable optimization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across backbones, datasets, and metrics with theoretical visualization, though datasets are small.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical flow from motivation to theory and experiment is very clear.
- Value: ⭐⭐⭐⭐ High utility for trustworthy GNNs due to its model-agnostic and plug-and-play nature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HarmonyGNNs: Harmonizing Heterophily and Homophily in GNNs via Self-Supervised Node Encoding](harmonygnns_harmonizing_heterophily_and_homophily_in_gnns_via_self-supervised_no.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](../../AAAI2026/graph_learning/self-adaptive_graph_mixture_of_models.md)
- [\[AAAI 2026\] Self-Correction Distillation for Structured Data Question Answering](../../AAAI2026/graph_learning/self-correction_distillation_for_structured_data_question_answering.md)
- [\[ECCV 2024\] SENC: Handling Self-collision in Neural Cloth Simulation](../../ECCV2024/graph_learning/senc_handling_self-collision_in_neural_cloth_simulation.md)
- [\[ICLR 2026\] Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)

</div>

<!-- RELATED:END -->
