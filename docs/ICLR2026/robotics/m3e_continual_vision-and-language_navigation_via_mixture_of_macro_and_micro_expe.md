---
title: >-
  [Paper Note] M³E: Continual Vision-and-Language Navigation via Mixture of Macro and Micro Experts
description: >-
  [ICLR 2026][Robotics][VLN] M³E replaces the FFN layers of an LLM navigation agent with "Macro + Micro" dual-routed MoE-LoRA layers. The macro-router employs a GNN on a cognitive map for topology-aware scene-level expert selection, while the micro-router performs instruction-level expert selection based on token hidden states. Combined with a dynamic momentum update strategy that freezes or aggressively updates different experts, this approach achieves cross-environment contin…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "VLN"
  - "Continual Learning"
  - "Mixture-of-Experts"
  - "Replay-free"
  - "Catastrophic Forgetting"
  - "MoE-LoRA"
date: 2026-05-08
content_hash: 154db3f397d93807
---

# M³E: Continual Vision-and-Language Navigation via Mixture of Macro and Micro Experts

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=pFh5ygjN3V](https://openreview.net/forum?id=pFh5ygjN3V)  
**Code**: [https://yongliangliang.top/m3e](https://yongliangliang.top/m3e)  
**Area**: Vision-and-Language Navigation / Continual Learning / Embodied AI  
**Keywords**: VLN, Continual Learning, Mixture-of-Experts, Replay-free, Catastrophic Forgetting, MoE-LoRA  

## TL;DR
M³E replaces the FFN layers of an LLM navigation agent with "Macro + Micro" dual-routed MoE-LoRA layers. The macro-router employs a GNN on a cognitive map for topology-aware scene-level expert selection, while the micro-router performs instruction-level expert selection based on token hidden states. Combined with a dynamic momentum update strategy that freezes or aggressively updates different experts, this approach achieves cross-environment continual learning under a **replay-free** constraint, improving both navigation success rates and anti-forgetting capabilities on R2R and REVERIE.

## Background & Motivation
**Background**: Vision-and-Language Navigation (VLN) requires embodied agents to follow natural language instructions to reach goals in real indoor scenes, necessitating the tight integration of visual perception, language grounding, and sequential decision-making. Recent developments have progressed from cross-modal alignment and memory/topological maps to end-to-end fine-tuning using LLMs as policy cores (e.g., NaviLLM, NaVid), significantly enhancing generalization.

**Limitations of Prior Work**: Most VLN systems are trained on static datasets. Deploying them to new environments typically requires expensive full retraining and triggers **catastrophic forgetting**. Research on continual learning in VLN is scarce, and existing methods almost exclusively rely on **rehearsal buffers**, which store and repeatedly replay historical trajectories, leading to storage/computation overhead and privacy concerns. Furthermore, VLN is more challenging than classification due to sequential planning under partial observability and fine-grained instruction grounding, making replay-free methods from the classification domain (like CL-MoE for VQA) not directly applicable.

**Key Challenge**: Balancing anti-forgetting across environments vs. operating without historical data. The authors argue the key lies in **decoupling "high-level scene reasoning" from "low-level perception alignment."** Scene-level understanding (e.g., layout patterns of offices vs. residences) is transferable across domains, while token-level grounding (local decision cues) needs rapid adaptation based on context. Entangling these two leads to fragile policies and poor transfer.

**Goal**: To define the VLN Continual Learning (VLNCL) setting and evaluation protocol, and to propose the first replay-free MoE framework for this setting.

**Key Insight**: **[Dual-layer Routing Decoupling]** Use macro-routing for "global scene strategy" and micro-routing for "local token semantics," driving sparse MoE experts through their fusion. **[Graduated Momentum Consolidation]** Differentially update momentum based on expert contribution to the current task—important experts adapt aggressively while minor ones are preserved conservatively—to balance plasticity and stability without replay.

## Method

### Overall Architecture
M³E is built upon a trainable LLM-based navigation agent (ViT scene encoder + 7B Decoder-only LLM policy core). The core modification is replacing standard FFN layers in the LLM backbone with **M³E layers**. Each layer consists of a set of MoE-LoRA experts activated by the fusion of "Macro-router (scene-level) + Micro-router (token-level)." Knowledge is consolidated across the task stream during training via **dynamic MoE momentum updates**. The architecture comprises two main components: Macro–Micro MoE (§4.1) for specialized computation selection and Dynamic Momentum Update (§4.2) for cross-task updates.

```mermaid
flowchart TB
    subgraph Inputs[Inputs]
        I[Instruction: go to the kitchen]
        P[Panoramic 36 views]
    end
    P --> VIT[ViT + Multi-view Encoder] --> CM[Cognitive Map<br/>visited + frontier nodes]
    subgraph MacroR[Macro Routing Gma · Scene-level]
        CM --> ADJ[Sparse Adjacency Â + Node Features X]
        ADJ --> GNN[Topology-aware GNN Propagation]
        I --> ATT[Instruction-queried Attention Aggregation]
        GNN --> ATT --> SV[Scene Vector st] --> WMA[Macro Expert Weights w_ma]
    end
    subgraph MicroR[Micro Routing Gmi · Token-level]
        H[LLM token hidden state h] --> WMI[Micro Expert Weights w_mi]
    end
    WMA --> FUSE[Convex Fusion<br/>w = β·w_ma + (1-β)·w_mi]
    WMI --> FUSE
    FUSE --> MOE[MoE-LoRA Experts Top-K=2] --> ACT[Action Head scores candidates]
    MOE -.Cross-task.-> MOM[Dynamic Momentum Update<br/>Aggressive for Main / Conservative for Minor]
```

### Key Designs

**1. Macro Routing TATF: Understanding "where I am" before focusing on "what the task is."** The goal of macro-router $G_{ma}$ is to capture global environmental structural patterns and align them with high-level task intent, termed Topology-Aware, Task-Focused (TATF) routing. Rather than simple visual feature pooling, it proceeds in four steps: constructing a sparse adjacency matrix $\hat{A}_t$ from the current cognitive map (including visited and frontier nodes) via distance thresholding; initializing each node as a feature vector $x_v$ blending panoramic vision, spatial position, timestep, and navigation status to form $X\in\mathbb{R}^{N\times d}$; performing message passing via GNN to learn topology-aware representations $H_{gnn}=\mathrm{GNN}(\hat{A}_t,X)$; following this with an attention aggregation using instruction embeddings $\mathrm{Emb}_{Ins}$ as a query $\alpha_v=\mathrm{softmax}_v(h_v^\top \mathrm{Emb}_{Ins})$ and $s_t=\sum_v \alpha_v h_v$ to obtain a scene vector that understands structure and current task focus; finally, a routing head $w_{ma}=\mathrm{Softmax}(\mathrm{MLP}(s_t))\in\mathbb{R}^n$ produces scene-level expert weights. The cognitive map is built online from exploration history rather than relying on a predefined global map.

**2. Micro Routing: Individual token expert selection.** Unlike the "one vote per graph" macro approach, the micro-router $G_{mi}$ operates at the token granularity. For each navigation step, the hidden state $h$ of a token undergoes standard MoE gating $w_{mi}=\mathrm{Softmax}(\mathrm{MLP}(h))\in\mathbb{R}^n$. It captures fine-grained semantics within the instruction stream—for instance, the verb token "go" in "go to the kitchen" might favor action reasoning experts, while the noun token "kitchen" favors object/scene understanding experts—enabling context-sensitive specialization. This router is trained directly on the current task data $D_t$.

**3. Dual-routed Convex Fusion: Global priors × Local adaptation.** The weights from both paths are merged via convex interpolation: $w=\beta\,w_{ma}+(1-\beta)\,w_{mi}\in\mathbb{R}^n$, where $\beta$ (set to 0.3 in experiments) balances the "global/structural prior from macro" and "fine-grained token-level judgment from micro." The fused $w$ activates sparse experts with Top-K=2 in the MoE-LoRA layer, maintaining strategic awareness and fine-grained adaptation while remaining computationally efficient.

**4. Dynamic MoE Momentum Update: Graduated freezing by contribution.** This is the key to replay-free anti-forgetting. For each MoE layer, the fused routing weights for all tokens in the current task $D_t$ are accumulated into an expert workload $u=\sum_{x\in D_t} w(x)$. This is normalized into a contribution distribution $I_t(E_i)=u[i]/\sum_j u[j]$. The $K$ most significant experts $E^{imp}_t$ are identified. Let $\Theta_{t-1}$ be the consolidated historical parameters and $\Phi_t$ be the parameters obtained by fine-tuning (initialized from $\Theta_{t-1}$) on $D_t$. Momentum coefficients $\lambda_i=\gamma$ are assigned to important experts ($\gamma\in[0,0.5)$) and $1-\gamma$ to minor experts. Parameters are finally consolidated element-wise: $\Theta_t=\Lambda\odot\Theta_{t-1}+(1-\Lambda)\odot\Phi_t$. Since $\gamma<0.5$, important experts have a small $\lambda$ and lean toward the new task $\Phi_t$ (aggressive adaptation), while minor experts have a large $\lambda$ and lean toward old parameters $\Theta_{t-1}$ (conservative preservation). This allows for rapid adaptation and anti-forgetting without historical data.

## Key Experimental Results

### Main Results

Incremental Continual Learning on R2R (Domain-incremental; same training budget; Reg=Regularization / Reh=Rehearsal / RF=Replay-free):

| Method | Strategy | AvgSR%↑ | AvgSPL%↑ | AvgNE↓ | BWT↑ | FWT↑ |
|------|------|---------|----------|--------|------|------|
| Finetune | RF | 63.28 | 59.08 | 3.72 | -5.42 | -2.41 |
| L2 | Reg | 58.78 | 56.20 | 4.23 | -5.10 | -3.43 |
| EWC | Reg | 64.15 | 60.21 | 3.60 | -3.50 | -2.80 |
| ER | Reh | 66.35 | 62.10 | 3.45 | -1.50 | 0.50 |
| PerR | Reh | 67.05 | 62.93 | 3.38 | -1.35 | 0.62 |
| ESR | Reh | 68.12 | 63.88 | 3.25 | -1.10 | 0.85 |
| Dual-SR | Reg+Reh | 70.25 | 65.40 | 3.05 | -0.45 | 1.85 |
| **M³E (ours)** | **RF** | **71.92** | **66.96** | **2.95** | **0.04** | **2.15** |

Incremental on REVERIE (Goal-oriented, object-anchored, more difficult):

| Method | SR%↑ | SPL%↑ | BWT↑ | FWT↑ |
|------|------|-------|------|------|
| Finetune | 50.12 | 39.86 | -16.91 | -10.26 |
| **M³E (ours)** | **51.23** | **48.30** | **-5.91** | **-8.09** |

### Ablation Study

Full combinations of the three components (Micro / Macro / Momentum) on R2R (selected):

| Micro | Macro | Momentum | AvgSR%↑ | BWT↑ | FWT↑ |
|:---:|:---:|:---:|---------|------|------|
| × | × | × (Finetune) | 63.28 | -5.42 | -2.41 |
| × | × | ✓ (≈EMA) | 61.52 | -2.15 | — |
| ✓ | × | × | 65.51 | Severe | — |
| × | ✓ | × | — | Severe | +1.80 |
| × | ✓ | ✓ | — | — | +1.92 |
| ✓ | ✓ | × | 67.83 | -6.05 | — |
| **✓** | **✓** | **✓** | **71.92** | **≈0** | **2.15** |

### Key Findings
- **Replay-free outperforms rehearsal**: Despite storing no historical trajectories, M³E's AvgSPL is still +1.56% higher than the strongest rehearsal method, Dual-SR, with BWT≈0 (near-zero forgetting) and FWT=2.15 (strong forward transfer/zero-shot generalization).
- **Significant anti-forgetting on REVERIE**: Compared to Finetune, SPL increased by +8.44% (48.30 vs 39.86), and BWT improved from -16.91 to -5.91.
- **Robustness in bulk training**: When training continues directly on the full val-unseen set, NaviLLM drops -11.18 SR on REVERIE val-seen; M³E only drops -3.87 and even gains +2.15 SR on R2R val-seen.
- **Complementary components**: Momentum alone (≈EMA) prevents forgetting but sacrifices plasticity (SR drops to 61.52); dual-routing maximizes plasticity (67.83) but suffers the most forgetting (BWT -6.05); only the combination of "routing + momentum" achieves both 71.92 SR and BWT≈0.

## Highlights & Insights
- **Decoupling continual learning into "Specialized Routing + Momentum Consolidation"**: The ablation study clearly proves these two tasks are orthogonal yet complementary—a design philosophy more interpretable than mere MoE stacking.
- **Explicating scene context via Macro-routing**: By explicitly incorporating the cognitive map (topological structure) into expert selection, scene identity directly guides decision-making rather than being buried implicitly in LLM hidden states, driving cross-domain generalization (high FWT).
- **Privacy and storage benefits of replay-free**: This approach has practical significance for real-world deployments (e.g., home/office robots) where long-term storage of historical trajectories is inconvenient.
- The momentum graduation is essentially a differentiable approximation of "**learn fast for important experts, forget slow for minor ones**," requiring only an accumulation of routing weights with negligible overhead.

## Limitations & Future Work
- **Residual forgetting on REVERIE**: The BWT of -5.91 and FWT of -8.09 suggest that dual-routing and momentum are insufficient to fully eliminate forgetting in long-horizon, goal-oriented tasks with strong object grounding.
- The number of experts is fixed at 6, Top-K=2, and $\beta$/$\gamma$ are manually tuned hyperparameters; there is no mechanism to adaptively expand expert capacity, leaving its scalability over long task streams not fully validated.
- Evaluation is limited to Matterport3D simulations; **it does not cover real robots, continuous action spaces, or sim-to-real transfer**.
- Task splitting by scene ID with the removal of scenarios with ≤10 validation episodes results in relatively "clean" domain boundaries; robustness under more fragmented or long-tailed real-world domain streams remains to be tested.

## Related Work & Insights
- **VLN Backbones**: NaviLLM, EmbodiedGPT, and NaVid are end-to-end trainable LLM agents. M³E applies MoE-LoRA modifications to NaviLLM, serving as a plugin for continual learning.
- **Continual Learning in VLN**: Prior works like PerR/ESR and Dual-SR mostly rely on rehearsal. M³E is the first replay-free MoE framework for this setting, standardizing the VLNCL protocol and metrics (including BWT for the BaseAgent).
- **Cross-domain MoE Continual Learning**: CL-MoE applied MoE to continual VQA but did not address sequential nature, partial observability, or spatial reasoning in VLN; M³E’s "Macro-topological Routing" specifically fills this gap.
- **Insight**: The "sparse experts" of MoE are inherently suited for modular knowledge retention in continual learning. Assigning "expert selection" to routers and "expert updates" to momentum policies is a more elegant path for anti-forgetting than regularization or replay, transferable to other embodied/multimodal sequential decision tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The first replay-free MoE framework for VLNCL. The combination of "Macro-topological + Micro-token routing + Graduated Momentum" is a novel and interpretable design in the VLN context, though individual components (MoE-LoRA, EWC-style consolidation) are known.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation on R2R/REVERIE across three categories of baselines (regularization, rehearsal, replay-free), full ablation of eight combinations, and bulk training analysis; however, limited to simulation and lacks longer task streams.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic from motivation to method and experiments. VLNCL settings and metrics are well-defined, and formulas align well with architectural diagrams.
- **Value**: ⭐⭐⭐⭐ Provides a parameter-efficient, privacy-preserving strong baseline for continual adaptation of embodied agents. Establishes a new SOTA for replay-free VLNCL with practical real-world implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Abstracting Robot Manipulation Skills via Mixture-of-Experts Diffusion Policies](abstracting_robot_manipulation_skills_via_mixture-of-experts_diffusion_policies.md)
- [\[ACL 2025\] Task-aware MoILE: Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts for Embodied Continual Learning](../../ACL2025/robotics/hierarchical-task-aware_multi-modal_mixture_of_incremental_lora_experts_for_embo.md)
- [\[ACL 2026\] Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents](../../ACL2026/robotics/breaking_down_and_building_up_mixture_of_skill-based_vision-and-language_navigat.md)
- [\[ICLR 2026\] All-day Multi-scenes Lifelong Vision-and-Language Navigation with Tucker Adaptation](all-day_multi-scenes_lifelong_vision-and-language_navigation_with_tucker_adaptat.md)
- [\[ICLR 2026\] Uncertainty-Aware Gaussian Map for Vision-Language Navigation](uncertainty-aware_gaussian_map_for_vision-language_navigation.md)

</div>

<!-- RELATED:END -->
