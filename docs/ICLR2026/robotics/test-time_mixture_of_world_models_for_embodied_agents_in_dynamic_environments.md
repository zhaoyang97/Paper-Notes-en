---
title: >-
  [Paper Note] Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments
description: >-
  [ICLR 2026][Embodied AI] Proposes the TMoW framework, extending the MoE paradigm to World Models for Embodied AI agents. Through three mechanisms—multi-granularity prototype routing, test-time prototype refinement, and distillation mixture augmentation—it achieves dynamic environment adaptation without retraining (zero-shot SR +27.21%) and few-shot knowledge expansion (+25.66%).
tags:
  - ICLR 2026
  - Embodied AI
  - World Models
  - Mixture-of-Experts
  - Test-Time Adaptation
  - Task Planning
date: 2026-05-08
content_hash: 771335d4db2238dd
---
# Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments

**Conference**: ICLR 2026
**arXiv**: [2601.22647](https://arxiv.org/abs/2601.22647)  
**Area**: Robotics / Embodied Intelligence

## Rating

⭐⭐⭐⭐

This paper generalizes the static routing of MoE into **test-time trainable routing**—the conceptual transfer itself is valuable. The three technical components (multi-granularity prototype routing, test-time refinement, and distillation-based augmentation) form a complete closed loop of "instant adaptation → online optimization → long-term scaling," covering a spectrum of adaptation needs from similar domains to entirely novel ones encountered after deployment. Experiments span three simulation benchmarks plus a real Franka robot, with an average SR improvement of 27% in zero-shot scenarios, constituting strong empirical evidence. The main limitations are (1) reliance on structured graph observations, requiring additional perception modules for raw pixel inputs; and (2) the relatively small underlying LLMs (Llama-3.2-1B/3B), which may impose a lower performance ceiling.

---

## Background & Motivation

LLM-based embodied agents are rapidly being deployed in domestic, industrial, and virtual environments. Representative approaches include code-driven policies (Code as Policies), reward-guided methods (Language to Rewards), LLM with domain models (SayCanPay), and in-context learning (LLM-Planner). However, these methods face a **fundamental tension**:

> Real deployment environments change continuously over time and space, yet agent capabilities are frozen after training. The only pathways to adapt to new domains are full retraining (computationally and data-intensive) or in-context learning (which inflates the inference context window).

Although MoE architectures achieve structured modularity through expert modules, their routing functions are fixed after training and cannot reconfigure expert combinations post-deployment. The core insight of TMoW is: **make the routing function updatable at test time**—without modifying the expert weights themselves, but rather changing *who is selected and in what mixture ratio*.

---

## Method

### Overall Architecture

TMoW attaches $N$ domain adapters $\{m_j\}_{j=1}^N$ in LoRA form onto a base model $M$, where each adapter corresponds to a world model (state transitions + policy) for one domain. A **graph processor** (MPNN) extracts multi-granularity prototypes from structured observations and independently computes routing scores at each layer, selecting top-K experts for weighted combination. The framework comprises three mechanisms:

1. **Multi-granularity prototype routing** — leverages the hierarchical aggregation of graphs to achieve domain matching from object level to scene level
2. **Test-time prototype refinement** — dynamically adjusts routing online during inference via weighted interpolation among prototypes
3. **Distillation-based mixture augmentation** — constructs new world models from mixtures of existing models combined with few-shot data

### Key Design 1: Multi-Granularity Prototype Routing

The MPNN progressively aggregates neighbor information from observation graph $\mathcal{G}^{(o)}$ layer by layer. The prototype at layer $l$ is obtained by averaging over all observations in a mini-batch:

$$\boldsymbol{p}_j^{(l)} = \underset{(i,\vec{\tau})\in\mathcal{B}}{\mathbb{E}} \underset{(o,\cdot,\cdot)\in\vec{\tau}}{\mathbb{E}} \left[ f^{(l)}(\mathcal{G}^{(o)}, i) \right]$$

The technical contribution lies in introducing a **context-aware edge matrix** $\tilde{\boldsymbol{E}}^{(l)} = (\boldsymbol{A}+\boldsymbol{I}) \odot \boldsymbol{R} \odot f_\text{adj}^{(l)}(\boldsymbol{H}^{(l-1)}, i)$, where the adjustment function $f_\text{adj}$ dynamically controls neighbor aggregation via cross-attention (QK interaction between observation node features and instruction embeddings). This causes shallow-layer prototypes to encode local object reachability (high entropy, shared across experts) and deep-layer prototypes to encode global scene structure (low entropy, expert-specialized).

Routing scores are computed via cosine similarity between the current input embedding $\mathcal{E}^{(l)}$ and each prototype, followed by top-K sparsification and softmax normalization to weight-combine adapter outputs:

$$\bar{w}_j^{(l)} = \text{softmax}\left(\text{top}_K\left(\frac{\text{sim}(\mathcal{E}^{(l)}, \boldsymbol{p}_j^{(l)})}{\tau}\right)\right)$$

### Key Design 2: Test-Time Prototype Refinement

When facing unseen domains, prototypes are dynamically updated using embeddings $\mathcal{E}^{(l)}$ acquired during environment interaction:

$$\bar{\boldsymbol{p}}_j^{(l)} = (1 - \alpha \cdot s_j) \boldsymbol{p}_j^{(l)} + \alpha \cdot s_j \cdot \Delta \boldsymbol{p}_j^{(l)}, \quad s_j = \text{sim}(\mathcal{E}^{(l)}, \boldsymbol{p}_j^{(l)})$$

The refinement term $\Delta \boldsymbol{p}_j^{(l)} = \sum_{k=1}^N \bar{r}_{j,k}^{(l)} \boldsymbol{p}_k^{(l)}$ is a weighted interpolation of all prototypes, with weights determined by inter-prototype similarity. Intuitively, prototypes more similar to the new-domain embedding are adjusted more, in a direction guided by the "neighborhood consensus" of existing prototypes. The effect is to expand prototype space coverage and densify allocation in high-frequency regions, enabling the router to uncover **previously underutilized knowledge fragments** in existing world models. Refinement adds only approximately 43 ms of latency.

### Key Design 3: Distillation-Based Mixture Augmentation

When a new environment differs substantially from all seen domains, refinement alone is insufficient and a new world model must be constructed. The approach initializes by mixing existing adapters weighted by routing scores, then fine-tunes on few-shot demonstrations using a teacher-forcing loss:

$$m'^{(l)} = \sum_{j=1}^N \bar{w}_j^{(l)} m_j^{(l)} - \eta \nabla_{m'^{(l)}} \left[ \mathbb{E}_{(\cdot,\vec{\tau}')\in\mathcal{D}'} \mathcal{L}_\text{TF}(M \oplus m', \vec{\tau}') \right]$$

Prototypes for the new model are likewise extracted from few-shot data and incorporated into the routing system alongside existing prototypes, without altering the framework structure. Compared to training from scratch, distillation initialization leverages cross-domain shared knowledge, reducing data requirements by 40%.

---

## Key Experimental Results

### Experimental Setup

- **Environments**: VirtualHome (78 tasks × 20 scenes), ALFWorld (6 task categories × 4 scenes), RLBench (4 tasks × 6 scenes), real Franka Research 3 robot
- **Baselines**: ZSP (zero-shot prompting), LLM+FT (full fine-tuning), LLM-Planner (in-context learning), FLARE (environment-aware replanning), SayCanPay (heuristic cost planning)
- **Models**: TMoW uses Llama-3.2-1B; ZSP/LLM-Planner/FLARE/SayCanPay-Say use Llama-3.2-3B
- **Metrics**: SR (success rate ↑), PS (steps to completion ↓)

### Main Results: Zero-Shot Adaptation (Unseen Domains)

| Method | VirtualHome SR↑ | VirtualHome PS↓ | ALFWorld SR↑ | ALFWorld PS↓ | RLBench SR↑ | RLBench PS↓ |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| ZSP | 7.32% | 28.22 | 2.08% | 49.68 | 10.42% | 18.73 |
| LLM+FT | 44.24% | 21.00 | 39.61% | 41.24 | 15.63% | 17.44 |
| LLM-Planner | 36.05% | 22.93 | 8.46% | 43.54 | 19.79% | 17.19 |
| FLARE | 40.07% | 22.57 | 11.31% | 42.85 | 34.37% | 11.37 |
| SayCanPay | 49.53% | 18.55 | 42.04% | 40.64 | 38.54% | 10.76 |
| **TMoW** | **80.16%** | **13.20** | **68.83%** | **37.44** | **62.75%** | **8.95** |

Average SR improvement across three benchmarks: **27.21%** (vs. strongest baseline SayCanPay), with PS reduced by **14.81%**. On the real Franka robot, TMoW achieves an SR of 74.64% (vs. SayCanPay at 7.80% and FLARE at 36.04%), with an even larger margin. TMoW with a 1B model outperforms all baselines using 3B models, demonstrating that structured adaptation is more effective than simply scaling model size.

### Few-Shot Scaling (VirtualHome Unseen Domains)

| Method | 1-shot SR↑ | 1-shot PS↓ | 5-shot SR↑ | 5-shot PS↓ |
|------|:-:|:-:|:-:|:-:|
| LLM+FT | 50.46% | 19.51 | 54.36% | 18.55 |
| LLM-Planner | 40.97% | 22.07 | 43.61% | 21.06 |
| FLARE | 42.17% | 22.19 | 46.64% | 20.67 |
| SayCanPay | 54.98% | 17.77 | 58.88% | 16.92 |
| **TMoW** | **81.56%** | **13.20** | **83.61%** | **12.04** |

Distillation augmentation brings TMoW to an average SR of 82.59%, surpassing the strongest baseline by 25.66%. Notably, TMoW at 1-shot already outperforms all baselines at 5-shot, indicating that distillation initialization dramatically reduces data requirements.

### Ablation Study

| Variant | SR↑ | PS↓ | Analysis |
|------|:-:|:-:|------|
| TMoW-Object (local features only) | 65.25% | 16.72 | Loss of global scene structure causes ~15% SR drop |
| TMoW-Scene (global features only) | 8.74% | 27.38 | Without object-level matching, nearly equivalent to random |
| TMoW-NoRefine (no refinement) | 73.30% | 14.85 | Refinement contributes ~7% SR |
| **TMoW** | **80.74%** | **13.12** | Full framework |
| TMoW-Scratch (train from scratch) | 59.84% | 18.16 | Distillation init vs. random init |
| **TMoW (distillation augmented)** | **81.56%** | **13.20** | Distillation + fine-tuning significantly outperforms training from scratch |

**Top-K routing**: K=3 is optimal (80.16%); K=1 degrades to a single expert (65.43%); K=7 introduces noise (66.01%), forming an inverted-U curve.

**Layer-wise routing entropy**: The shallow-to-deep trend from high to low entropy confirms the design hypothesis that "shallow layers share object-level knowledge while deep layers specialize in scene structure." Post-refinement entropy increases across all layers, indicating the router learns to extract knowledge fragments from a broader set of world models.

**Continual scaling**: As new domains are incrementally added, performance on existing domains does not degrade but improves (positive knowledge transfer), with no catastrophic forgetting. This is attributed to prototype routing naturally isolating domain-specific adapters, allowing new and existing models to collaborate through expansion of the prototype space.

---

## Highlights & Insights

**Strengths**:

- Generalizing MoE's static routing to test-time trainable routing is conceptually concise yet far-reaching, opening a "third path" for post-deployment continual adaptation distinct from ICL and fine-tuning
- Multi-granularity prototypes naturally correspond to a local-to-global semantic gradient via MPNN's hierarchical aggregation, with a design highly aligned with the inductive bias of graph structures
- The three mechanisms complement each other across the adaptation spectrum: prototype routing handles in-distribution variation, refinement handles out-of-distribution similar domains, and distillation augmentation handles entirely novel domains
- Experiments cover both simulation and real robots; outperforming 3B baselines with a 1B model demonstrates the efficiency of structured adaptation
- Distillation initialization is particularly effective (1-shot TMoW > 5-shot SayCanPay), offering high practical deployment value

**Weaknesses**:

- Reliance on structured graph observations (object lists + relations) requires additional perception modules for purely visual inputs, limiting end-to-end applicability
- The underlying LLMs are only Llama-3.2-1B/3B, imposing a relatively low performance ceiling; the paper does not discuss combinations with larger models
- The refinement rate $\alpha$ must be set manually and is only effective when $\alpha \geq 0.5$, with no adaptive adjustment mechanism
- World model prediction accuracy may degrade rapidly in highly non-stationary environments such as multi-agent settings; the paper validates only single-agent scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](exopredicator_learning_abstract_models_of_dynamic_worlds_for_robot_planning.md)
- [\[ICCV 2025\] TesserAct: Learning 4D Embodied World Models](../../ICCV2025/robotics/learning_4d_embodied_world_models.md)
- [\[ICLR 2026\] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](rei-bench_can_embodied_agents_understand_vague_human_instructions_in_task_planni.md)
- [\[ICLR 2026\] Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization](embodied_agents_meet_personalization_investigating_challenges_and_solutions_thro.md)
- [\[CVPR 2026\] Test-time Ego-Exo-centric Adaptation for Action Anticipation via Multi-Label Prototype Growing and Dual-Clue Consistency](../../CVPR2026/robotics/test-time_ego-exo-centric_adaptation_for_action_anticipation_via_multi-label_pro.md)

</div>

<!-- RELATED:END -->
