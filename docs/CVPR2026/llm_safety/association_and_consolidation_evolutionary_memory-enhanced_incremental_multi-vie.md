---
title: >-
  [Paper Note] Association and Consolidation: Evolutionary Memory-Enhanced Incremental Multi-View Clustering
description: >-
  [CVPR2026][LLM Safety][incremental multi-view clustering] This paper proposes EMIMC, a framework inspired by the hippocampus–prefrontal cortex collaborative memory mechanism in the brain. Three coordinated modules — a Rapid Associative Module (orthogonal mapping to ensure plasticity), a Cognitive Forgetting Module (power-law decay to simulate the forgetting curve), and a Knowledge Consolidation Module (temporal tensor low-rank decomposition to distill long-term memory) — jointly address the stability-plasticity dilemma in incremental multi-view clustering.
tags:
  - CVPR2026
  - LLM Safety
  - incremental multi-view clustering
  - stability-plasticity dilemma
  - memory consolidation
  - orthogonal association
  - tensor decomposition
  - ADMM
date: 2026-05-08
content_hash: 9e1e7b5b1f7b3bec
---

# Association and Consolidation: Evolutionary Memory-Enhanced Incremental Multi-View Clustering

**Conference**: CVPR2026
**arXiv**: [2509.14544](https://arxiv.org/abs/2509.14544)
**Code**: TBD
**Area**: LLM Safety
**Keywords**: incremental multi-view clustering, stability-plasticity dilemma, memory consolidation, orthogonal association, tensor decomposition, ADMM

## TL;DR

This paper proposes EMIMC, a framework inspired by the hippocampus–prefrontal cortex collaborative memory mechanism in the brain. Three coordinated modules — a Rapid Associative Module (orthogonal mapping to ensure plasticity), a Cognitive Forgetting Module (power-law decay to simulate the forgetting curve), and a Knowledge Consolidation Module (temporal tensor low-rank decomposition to distill long-term memory) — jointly address the stability-plasticity dilemma in incremental multi-view clustering.

## Background & Motivation

**Practical demand for multi-view clustering**: In real-world scenarios, data originates from different modalities (visual, textual, sensor, etc.). Multi-view clustering (MVC) must exploit complementary information across views to achieve more accurate clustering. However, conventional MVC assumes all views are simultaneously available, making it unsuitable for dynamic settings where views arrive incrementally.

**The inherent tension in incremental multi-view clustering (IMVC)**: Upon the arrival of a new view, the model must simultaneously (a) absorb new knowledge (plasticity) and (b) retain existing knowledge (stability) — the classic Stability-Plasticity Dilemma (SPD).

**Limitations of existing IMVC methods**:
- Methods such as CMVC, CAC, and LAIMVC lack effective mechanisms for preserving historical knowledge; as views arrive incrementally, information from earlier views is progressively lost.
- Simple concatenation or averaging fusion strategies cannot distinguish differences in importance across views at different time steps.
- There is no explicit modeling of the short-term memory → long-term memory transition.

**Neuroscientific inspiration**: The human brain rapidly encodes new experiences via the hippocampus (associative binding), while the prefrontal cortex consolidates short-term memories into stable long-term memories (knowledge consolidation), with natural decay following the forgetting curve in between. This biological mechanism closely mirrors the requirements of IMVC.

## Core Problem

In the incremental multi-view clustering setting, how can one simultaneously maintain rapid absorption of new view information (plasticity) and prevent catastrophic forgetting of knowledge from earlier views (stability), while achieving a balance between the two at a reasonable computational cost?

## Method

### Overall Architecture

EMIMC consists of three core modules that simulate the brain's memory process of "encoding–forgetting–consolidation":

1. **Rapid Associative Module (RAM)** — analogous to the hippocampus; responsible for rapidly associating new and old representations.
2. **Cognitive Forgetting Module (CFM)** — simulates the forgetting curve; fuses historical memory via time-weighted aggregation.
3. **Knowledge Consolidation Module (KCM)** — analogous to the prefrontal cortex; distills short-term memory into long-term memory.

### Rapid Associative Module (RAM)

- **Objective**: Upon arrival of new view $v_t$, establish a structured correspondence between the current consensus representation $Z_t$ and the previous representation $Z_{t-1}$.
- **Orthogonal mapping**: An orthogonal matrix $P_t \in \mathbb{R}^{m \times m}$ (satisfying $P_t^T P_t = I$) is introduced to align $Z_{t-1}$ into the space of $Z_t$.
- **Association loss**: $\mathcal{L}_{\text{associate}} = \|Z_t - Z_{t-1} P_t\|_F^2$
- **Closed-form solution**: The optimal $P_t$ under the orthogonal constraint is obtained by solving the Procrustes problem — performing SVD on $Z_{t-1}^T Z_t = U\Sigma V^T$, yielding $P_t = UV^T$.
- **Intuition**: The orthogonal constraint ensures that the mapping preserves the intrinsic structure of the representations (no compression or stretching), performing only rigid rotations/reflections, thereby retaining the semantic integrity of both old and new knowledge while associating them.

### Cognitive Forgetting Module (CFM)

- **Biological motivation**: The Ebbinghaus forgetting curve shows that memory decays over time following a power law, with more recent memories being clearer.
- **Power-law weights**: For the view at step $i$ ($i < t$), its weight at time step $t$ is:
  $$w_i^{(t)} = \frac{(t - i)^{-\lambda}}{\sum_{j=1}^{t-1}(t - j)^{-\lambda}}$$
  where $\lambda > 0$ controls the forgetting rate — larger $\lambda$ causes faster decay of distant views.
- **Historical memory**: $Z_{\text{hist}} = \sum_{i=1}^{t-1} w_i^{(t)} Z_i$
- **Key advantages**:
    - No need to store all historical representation matrices; only the weighted sum $Z_{\text{hist}}$ needs to be maintained (constant space).
    - Weight normalization ensures numerical stability.
    - $\lambda$ provides an adjustable stability-plasticity knob.

### Knowledge Consolidation Module (KCM)

- **Constructing the temporal tensor**: The historical memory $Z_{\text{hist}}$ and the current representation $Z_t$ are stacked along the temporal dimension to form a 3rd-order tensor $\mathcal{Z} \in \mathbb{R}^{n \times m \times 2}$.
- **ARMR low-rank constraint**: Augmented Multi-Rank Minimization with Relaxation is applied to $\mathcal{Z}$, constraining the rank of its mode-unfoldings to distill the shared low-rank structure between short-term memory and historical memory.
- **Consolidation loss**: $\mathcal{L}_{\text{consolidate}}$ enforces tensor approximation with low-rank regularization.
- **Intuition**: Low-rank decomposition forces the model to retain only the core patterns that are consistent across time, naturally filtering out noise and unstable short-term fluctuations — effectively "distilling" long-term memory.

### Overall Optimization

- **Total objective**: $\mathcal{L} = \mathcal{L}_{\text{recon}} + \alpha \cdot \mathcal{L}_{\text{associate}} + \beta \cdot \mathcal{L}_{\text{consolidate}}$
    - $\mathcal{L}_{\text{recon}}$: low-rank reconstruction loss for each view.
    - $\alpha, \beta$: hyperparameters balancing plasticity and stability.
- **ADMM optimization**: Auxiliary variables are introduced to decompose the problem into multiple subproblems, optimized alternately:
    - $Z_t$ update: solve a quadratic problem with other variables fixed.
    - $P_t$ update: Procrustes closed-form solution.
    - Tensor low-rank approximation: proximal operator based on the matrix nuclear norm.
- **Computational efficiency**: All orthogonal constraints admit closed-form solutions (Procrustes), eliminating the need for iterative projection; ADMM typically converges within 20–30 iterations.

## Key Experimental Results

### Datasets & Setup

- **Datasets**: Several classic multi-view datasets, including MSRC-v1, UCI-Digits, BBCSport, and NUS-WIDE-Object.
- **Incremental protocol**: Views arrive sequentially one at a time; each step processes only the current view and historical memory, without revisiting original data.
- **Evaluation metrics**: ACC (clustering accuracy), NMI (normalized mutual information), Purity.

### Main Results

- EMIMC outperforms all existing IMVC methods (CMVC, CAC, LAIMVC, EIMC, etc.) on all datasets, with an average improvement of 3–8 percentage points in ACC/NMI.
- As the number of views increases, EMIMC's performance improves steadily, whereas baseline methods exhibit performance fluctuations caused by knowledge forgetting.
- The advantage is more pronounced on large-scale datasets such as NUS-WIDE-Object, indicating robustness in high-dimensional settings.

### Ablation Study

| Module Configuration | ACC | NMI |
|---|---|---|
| $\mathcal{L}_{\text{recon}}$ only (no memory) | baseline | baseline |
| + RAM (associative module) | +2–4% | +2–3% |
| + RAM + CFM (with forgetting) | +4–6% | +3–5% |
| + RAM + CFM + KCM (full model) | **+6–8%** | **+5–7%** |

- RAM contributes the largest incremental gain, validating the effectiveness of orthogonal association.
- CFM's power-law decay outperforms uniform weighting by 1–2 percentage points.
- KCM further delivers stable additional gains.

### Key Findings

- **Forgetting rate $\lambda$**: Performance is robust for $\lambda \in [0.5, 1.5]$; both excessively small values (approaching uniform weights) and excessively large values (attending almost exclusively to the most recent view) underperform intermediate values.
- **Balancing parameters $\alpha, \beta$**: Grid search indicates that performance is relatively insensitive to $\alpha$ and $\beta$ within the range $10^{-2}$ to $10^{0}$, demonstrating good robustness.
- **ADMM convergence**: The objective function typically converges within 20 iterations, with manageable computational overhead.

## Highlights & Insights

- **Elegant mapping of brain memory mechanisms**: The three-stage hippocampus–prefrontal cortex collaborative memory process (rapid encoding → forgetting decay → long-term consolidation) is fully mapped to a mathematical framework, yielding strong biological interpretability.
- **All orthogonal constraints admit closed-form solutions**: The Procrustes solution avoids the computational overhead of iterative projection, making the entire optimization process efficient with guaranteed convergence.
- **Elegant design of power-law forgetting weights**: A single parameter $\lambda$ flexibly controls the stability-plasticity balance, and the weights can be updated online recursively without storing the full history.
- **3rd-order temporal tensor + low-rank constraint**: Short-term and long-term memory are jointly modeled as a tensor structure; low-rank decomposition naturally achieves memory distillation, avoiding heuristic fusion strategies.

## Limitations & Future Work

- Only the view-incremental scenario (new views arriving sequentially) is considered; sample-incremental (new samples arriving) and mixed-incremental scenarios are not addressed.
- The power-law forgetting model assumes that the importance of all historical views depends solely on temporal distance, ignoring differences in view quality and information content.
- For large numbers of views (e.g., $T > 50$), the computational cost of constructing and decomposing the 3rd-order tensor may increase significantly.
- Experimental datasets are relatively small-scale (hundreds to thousands of samples); scalability on datasets exceeding 100,000 samples remains to be validated.
- Robustness to missing views or partially corrupted views is not considered.
- Hyperparameters $\alpha$, $\beta$, and $\lambda$ still require tuning, despite experiments indicating reasonable robustness.

## Related Work & Insights

- **Traditional MVC**: Co-regularization, Co-training, MVSC — assume all views are simultaneously available; cannot handle incremental scenarios.
- **Incremental learning**: iCaRL, DER — primarily target single-view/classification incremental learning tasks; do not account for multi-view characteristics.
- **IMVC methods**:
    - CMVC — simple feature concatenation with no memory mechanism.
    - CAC — cross-view alignment but lacks retention of historical knowledge.
    - LAIMVC — anchor graph method; scalable but does not model forgetting.
    - EIMC — incorporates Elastic Weight Consolidation (EWC) but lacks global memory integration.
- **Tensor decomposition MVC**: t-SVD and Tucker decomposition for multi-view fusion, but all designed for static settings.

## Rating

- Novelty: ⭐⭐⭐⭐ — Systematically incorporating the neuroscientific memory mechanism into IMVC is original; the three-module design is complete and mutually coupled.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple datasets, complete ablation, and hyperparameter analysis; large-scale experiments are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Biological analogies are clear, mathematical derivations are rigorous, and motivation–method–experiment consistency is strong.
- Value: ⭐⭐⭐⭐ — Provides a new methodological perspective for IMVC; closed-form solutions ensure practical applicability.
- Value: TBD

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[ICLR 2026\] Lifelong Learning with Behavior Consolidation for Vehicle Routing](../../ICLR2026/llm_safety/lifelong_learning_with_behavior_consolidation_for_vehicle_routing.md)
- [\[ICCV 2025\] Adversarial Robust Memory-Based Continual Learner](../../ICCV2025/llm_safety/adversarial_robust_memory-based_continual_learner.md)
- [\[AAAI 2026\] GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs](../../AAAI2026/llm_safety/graphtextack_a_realistic_black-box_node_injection_attack_on_llm-enhanced_gnns.md)

<!-- RELATED:END -->
