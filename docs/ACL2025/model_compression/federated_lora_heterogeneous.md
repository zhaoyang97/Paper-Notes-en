---
title: >-
  [Paper Note] Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients
description: >-
  [ACL 2025 (Long Paper, pp. 416–429)][Model Compression][federated learning] Proposes LoRA-A² (Low Rank Adaptation with Alternating freeze and Adaptive rank selection), which addresses the aggregation discordance problem in federated LoRA by alternately freezing matrices A and B. Combined with an adaptive rank selection mechanism, it significantly compresses upload parameter volume (up to 99.8% reduction) while maintaining robustness, outperforming existing methods significant…
tags:
  - "ACL 2025 (Long Paper, pp. 416–429)"
  - "Model Compression"
  - "federated learning"
  - "LoRA"
  - "Aggregation Discordance"
  - "Alternating Freeze"
  - "Adaptive Rank Selection"
  - "Communication Efficiency"
date: 2026-05-08
content_hash: a94579f394d08ea3
---

# Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients

**Conference**: ACL 2025 (Long Paper, pp. 416–429)  
**arXiv**: [2410.22815](https://arxiv.org/abs/2410.22815)  
**Authors**: Jabin Koo, Minwoo Jang, Jungseul Ok (POSTECH, South Korea)
**Code**: Not publicly available  
**Area**: Federated Learning / Parameter-Efficient Fine-Tuning / Large Language Models  
**Keywords**: federated learning, LoRA, Aggregation Discordance, Alternating Freeze, Adaptive Rank Selection, Communication Efficiency

## TL;DR
Proposes LoRA-A² (Low Rank Adaptation with Alternating freeze and Adaptive rank selection), which addresses the aggregation discordance problem in federated LoRA by alternately freezing matrices A and B. Combined with an adaptive rank selection mechanism, it significantly compresses upload parameter volume (up to 99.8% reduction) while maintaining robustness, outperforming existing methods significantly, especially in low-rank and high data heterogeneity scenarios.

## Background & Motivation
Fine-tuning LLMs in Federated Learning (FL) faces enormous communication overheads. LoRA reduces trainable parameters through low-rank decomposition $\Delta W = BA$, but faces a key challenge in FL—**Aggregation Discordance**:

The multiplication of the weighted averages of B and A across clients does not equal the weighted average of $B_k A_k$ for each client:
$$\frac{1}{K}\sum(B_k + B_j) \cdot \frac{1}{K}\sum(A_k + A_j) \neq \frac{1}{K}\sum B_k A_k$$

The existing solution, FFA-LoRA, permanently freezes A and only trains B. Although this eliminates discordance, it limits the optimization space (A always retains its initial value), leading to severe performance degradation under **low-rank + high data heterogeneity** conditions.

## Core Problem
How to simultaneously address the following within the federated LoRA framework:
1. Aggregation discordance (ensuring mathematically correct aggregation)
2. Retaining the complete optimization parameter space (training both matrices A and B)
3. Maintaining robustness under low-rank and high-heterogeneity conditions
4. Further reducing communication costs

## Method

### Overall Architecture
LoRA-A² consists of two core components: **Alternating Freeze** and **Adaptive Rank Selection**. A global LoRA adapter with rank $r_G$ is maintained. In each round, A or B is trained alternately, and each client adaptively selects and uploads critical ranks based on local data.

### Key Designs

1. **Alternating Freeze**

    - Even rounds freeze A and train B; odd rounds freeze B and train A.
    - When A is frozen, all clients share the exact same A, and aggregation becomes:
    $\Delta W = \sum_k w_k B_k \cdot A = \sum_k w_k (B_k A_k) = \sum_k w_k \Delta W_k$
      The aggregation discordance is mathematically eliminated.
    - Compared to FFA-LoRA (which permanently freezes A), alternating freeze enables A to be trained, preserving the full optimization space.
    - Drawing inspiration from LoRA+, different learning rates are set for A and B to further enhance the optimization effect.

2. **Adaptive Rank Selection**

    - **Design Motivation**: Focus on upload communication (since uplink bandwidth is typically much slower than downlink) and allow different clients to select different ranks.
    - **Contribution Criterion**: Define the importance score of rank $i$ in module $m$:
    $S_{m,i}^{B_k} = \|\Delta B_k[:,i] \cdot A[i,:]\|_F$
      This criterion captures the contribution of each rank to the model update $\Delta W$ while accounting for the interaction between A and B (outperforming simple gradient magnitude criteria).
    - **Selection and Sparsification**: Top-$(r_i \times N)$ ranks are selected from the full model's $r_G \times N$ ranks (where $N$ is the number of target modules) to construct a binary mask $M_k$. Only $B_k \odot M_k$ (or $A_k \odot M_k$) is uploaded, achieving sparse communication.
    - **Two Major Benefits**: (1) Different clients can select different ranks, reducing client conflicts under heterogeneous data; (2) Rank resources are reallocated from less important modules to those requiring more fine-tuning.

3. **Theoretical Analysis**

    - Proven parameter space containment relationship:
    $\Omega_{\text{FFA-LoRA}} \subsetneq \Omega_{\text{FL+LoRA}} = \Omega_{\text{FlexLoRA}} \subset \Omega_{\text{LoRA-A}^2}$
    - LoRA-A² possesses the largest reachable parameter space while transmitting fewer parameters.

## Key Experimental Results

Experiments are evaluated on NLU tasks, using the Dirichlet distribution ($\alpha$) to control the degree of data heterogeneity, and testing performance under different ranks ($r$).

| Method | Aggregation Method | Low-Rank Robustness | High-Heterogeneity Robustness | Upload Parameters | Key Features |
|------|---------|-----------|------------|----------|---------|
| FL+LoRA | Aggregated separately for A, B | ❌ Severe degradation | ❌ Severe degradation | 100% (Baseline) | Suffers from aggregation discordance |
| FFA-LoRA | Permanently freezes A | ❌ Low-rank degradation | ❌ Heterogeneity degradation | ~50% | Limited optimization space |
| FlexLoRA | Full-size matrix + SVD | ✅ Good | ⚠️ Fair | High (Requires full matrix transmission) | High communication cost |
| **LoRA-A²** | Alternating freeze + Adaptive rank | ✅ Robust | ✅ Robust | **Lowest 0.2%** | Achieves both robustness and efficiency |

Key experimental findings:
- Under extreme conditions (low rank $r=1$ + high heterogeneity $\alpha=0.1$), LoRA-A² still maintains stable performance, while FFA-LoRA and FL+LoRA degrade significantly.
- Compared to full fine-tuning, the uploaded parameter volume is reduced by up to **99.8%** without performance loss.
- Alternating freeze alone yields significant improvements, and adding adaptive rank selection further compresses communication while maintaining or even improving performance.

## Ablation Study
- **Alternating Freeze vs. Permanent Freeze**: Alternating freeze consistently outperforms permanent freeze (FFA-LoRA) across various ranks and heterogeneity settings, validating the importance of preserving the complete optimization space.
- **Learning Rate Differentiation**: Setting different learning rates for A and B further enhances the effectiveness of alternating optimization.
- **Comparison of Contribution Criteria**: The proposed criterion based on $\|ΔB[:,i] \cdot A[i,:]\|_F$ outperforms simple gradient magnitude criteria ($\|ΔB[:,i]\|$ or $\|ΔA[i,:]\|$) because it explicitly models the interaction between A and B.
- **Effect of Rank Selection**: Adaptive rank selection allows different clients to select different important ranks, effectively reducing client conflicts in highly heterogeneous scenarios.

## Highlights & Insights
- **Simple and Elegant Design**: Alternating freeze is an extremely simple modification (shifting which matrix is frozen each round), yet it simultaneously solves both aggregation discordance and optimization space constraints.
- **Outstanding Robustness**: Remains stable in the "most difficult" scenario of extreme low-rank + extreme heterogeneity, where existing methods commonly fail.
- **Rigorous Theoretical Support**: Proves that the parameter space of LoRA-A² strictly contains that of other methods, providing a theoretical explanation for the method's advantages.
- **Cross-Module Reallocation via Adaptive Rank Selection**: Rank selection is not limited to individual modules; instead, ranks are ranked and selected globally across all modules in the model, enabling rank resources to flow from unimportant modules to critical ones.
- **Extremely High Communication Efficiency**: A 99.8% parameter reduction rate is highly practical in federated learning scenarios.

## Limitations & Future Work
- Alternating freeze trains only half the parameters (either A or B) in each round, which may slow down the convergence speed compared to training both simultaneously, requiring more communication rounds.
- Adaptive rank selection requires executing an extra epoch to compute contribution scores, increasing local computational overhead.
- The paper primarily focuses on NLU tasks, lacking validation on natural language generation (NLG) tasks and larger model scales (e.g., LLaMA-7B/13B).
- After server-side aggregation, sparse updates must be "added to the B (or A) from two rounds ago", which complicates implementation logic and requires maintaining historical states.
- Combining with other PEFT methods (e.g., Adapters, Prefix Tuning) in FL has not been explored.

## Related Work & Insights
- **vs. FFA-LoRA** (Sun et al., 2024): Permanently freezes A, leading to limited optimization space and degradation under low-rank/high-heterogeneity settings. LoRA-A² uses alternating freeze, and its parameter space strictly contains that of FFA-LoRA.
- **vs. FlexLoRA** (Bai et al., 2024): Re-decomposes after aggregating full-size matrices using SVD, resulting in high communication costs (requiring transmission of the full $d_1 \times d_2$ matrix). LoRA-A² only transmits sparse, low-rank updates, saving communication overhead.
- **vs. FL+LoRA** (FedAvg + LoRA): Separately aggregates A and B, which suffers from severe aggregation discordance and is highly sensitive to heterogeneity. LoRA-A² completely eliminates this discordance.
- **vs. RoLoRA** (Chen et al., 2024): Also adopts an alternating optimization idea, but LoRA-A² additionally introduces adaptive rank selection to further improve efficiency and robustness to heterogeneity.
- **vs. HETLORA** (Cho et al., 2024): Allows heterogeneous ranks but does not address aggregation discordance. LoRA-A² simultaneously addresses both challenges.

## Related Work & Insights
- The concept of "alternating optimization" can be generalized to other federated PEFT scenarios—any structure involving the product of two parameter groups could potentially benefit from alternating freeze.
- The idea of "global ranking across modules" in adaptive rank selection resembles the dynamic allocation of rank budgets in AdaLoRA, but holds greater significance in federated scenarios (where different clients may have different critical modules).
- The citation count of this paper has reached 19 (as of 2026.03), indicating that federated LoRA is a rapidly growing research direction.
- The 99.8% parameter compression rate can inspire LLM deployment strategies on edge devices.

## Rating
- **Novelty**: ⭐⭐⭐⭐ While the concept of alternating freeze is simple, it hits the mark directly. The contribution criterion design for adaptive rank selection shows high originality. The combination of both forms a highly effective framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thorough comparisons under various degrees of heterogeneity and rank settings. Ablation studies cover each component, though validation on NLG tasks and large-scale models is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition, concise theoretical analysis, and systematic method description. The logical integration of alternating freeze and adaptive rank selection is smooth.
- **Value**: ⭐⭐⭐⭐ Offers a clear analysis and a simple, elegant solution for the aggregation issue in federated LoRA, serving as a highly valuable reference for understanding the behavior of LoRA in distributed learning environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)
- [\[ACL 2025\] BeamLoRA: Beam-Constraint Low-Rank Adaptation](beamlora_beam_constraint_lora.md)
- [\[ACL 2025\] Low-Rank Interconnected Adaptation across Layers](low-rank_interconnected_adaptation_across_layers.md)
- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](../../NeurIPS2025/model_compression/robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)
- [\[ACL 2025\] TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition](teamlora_boosting_low-rank_adaptation_with_expert_collaboration_and_competition.md)

</div>

<!-- RELATED:END -->
