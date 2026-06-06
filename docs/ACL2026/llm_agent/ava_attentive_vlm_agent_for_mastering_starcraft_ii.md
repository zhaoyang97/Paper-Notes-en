---
title: >-
  [Paper Note] AVA: Attentive VLM Agent for Mastering StarCraft II
description: >-
  [ACL 2026][LLM Agent][StarCraft II] This paper proposes AVACraft—the first StarCraft II multimodal benchmark supporting both MARL and VLM decision paradigms (21 scenarios / RGB+text+structured states)—and introduces the…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "StarCraft II"
  - "Multimodal RL"
  - "Zero-shot VLM"
  - "Cross-paradigm benchmark"
  - "Priority Inference"
date: 2026-05-08
content_hash: f8b3d58ba1469058
---

# AVA: Attentive VLM Agent for Mastering StarCraft II

**Conference**: ACL 2026  
**arXiv**: [2503.05383](https://arxiv.org/abs/2503.05383)  
**Code**: https://github.com/camel-ai/VLM-Play-StarCraft2  
**Area**: LLM Agent / Multimodal Game / VLM Decision Making  
**Keywords**: StarCraft II, Multimodal RL, Zero-shot VLM, Cross-paradigm benchmark, Priority Inference

## TL;DR
This paper proposes AVACraft—the first StarCraft II multimodal benchmark supporting both MARL and VLM decision paradigms (21 scenarios / RGB+text+structured states)—and introduces the VLM baseline AVA (Multimodal Priority Inference + RAG + Dynamic Role Assignment). Experiments show that MARL trained for 5M steps on the basic 3m scenario only achieves a 19–27% win rate, while VLM achieves 75–90% zero-shot.

## Background & Motivation
**Background**: StarCraft II is the gold standard benchmark for multi-agent decision-making. SMAC/SMACv2 have driven the development of MARL algorithms (QMIX, MAPPO, etc.) for years. Simultaneously, VLMs (GPT-4V, Qwen-VL) have emerged in zero-shot visual reasoning and are being applied to complex game decisions (LLM-PySC2, VS-Bench, etc.).

**Limitations of Prior Work**: (1) The SMAC series only supports **abstract feature vectors**, discarding RGB visual information, which prevents VLM integration and creates a lack of a common platform for fair comparison. (2) SMAC simplifies unit abilities, losing tactical depth. (3) Existing LLM game benchmarks either test only macro-strategy (LLM-PySC2) or use abstract multi-agent settings (VS-Bench); none focus on cross-paradigm comparisons of **fine-grained tactical micromanagement**.

**Key Challenge**: MARL training is expensive but precisely controllable, while zero-shot VLMs are fast, but their ability to handle high-frequency micromanagement is unknown. Previously, no one could fairly compare these two paradigms in the **same observation space**. Without a unified evaluation framework, the question of "whether VLMs can play SC2" remains unresolved.

**Goal**: (i) Build an SC2 environment natively supporting both MARL (RGB / scalar / hybrid) and VLM (RGB + natural language + structured metadata); (ii) Run complete baselines for both paradigms across 21 micromanagement/coordination/strategy scenarios; (iii) Provide a decent VLM agent baseline, AVA, to verify that VLMs can not only play but also provide human-like explanations.

**Key Insight**: Wrap the SC2 POMDP to support four coexisting observation modes—MARL receives RGB / scalar / hybrid, while VLM receives RGB + natural language text + structured unit information $\mathcal{U}_t = \{u_i = (\text{id}_i, \text{type}_i, \text{pos}_i, \text{hp}_i, \text{status}_i)\}$. All modes share the same action space and rewards, ensuring fair comparability.

**Core Idea**: Through an environment design featuring **multimodal unified observations + full unit abilities + adaptive enemy AI**, MARL and VLM are brought onto the same evaluation platform. A lightweight VLM baseline, AVA, demonstrates that zero-shot VLMs can outperform MARL trained for long durations in tactical micromanagement.

## Method

### Overall Architecture
AVACraft is formalized as a POMDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{O}, P, R, \gamma \rangle$, strictly adhering to the Fog of War (agents only see information within sight range). The observation space $\mathcal{O}$ provides four modes: RGB (screen 160×120 + minimap 32×32), SMAC-compatible scalar, Hybrid, and VLM-Optimized (RGB + natural language description + structured unit list). The action space consists of three categories: $\mathcal{A} = \mathcal{A}_{\text{atk}} \cup \mathcal{A}_{\text{mov}} \cup \mathcal{A}_{\text{abl}}$ (Attack/Move/Ability). Rewards are sparse: $R \in \{-1, 0, 1\}$. The 21 scenarios cover four difficulty levels (low/medium/high/extremely high), each paired with one built-in AI (VeryHard) + three LLM-synthesized scripted strategies + random selection to prevent exploitation of single strategies.

Baselines are divided into two sides:

- **MARL Side**: Six algorithms (IQL / QMIX / QTRAN / VDN / MAPPO / IPPO) using Swin-Tiny (27.5M parameters) as the visual backbone, trained up to 5M steps, with optional GTE-Base text embeddings for vision+text fusion.
- **VLM Side**: The AVA agent (see three key designs below), evaluating GPT-4o / GPT-4-Turbo / GPT-4o-mini / Qwen-VL-Plus / Qwen3-VL-30B / Qwen3-VL-8B, all zero-shot.

### Key Designs

1. **Multimodal Priority Inference (MPI)**:
    - **Function**: Synthesizes battlefield images + textual states + action history. First, the VLM Planner generates primary and secondary skill plans $S = \text{VLM}_{\text{plan}}(I, T, H) = \{s_{\text{primary}}, s_{\text{secondary}}\}$. Then, unit detection is performed $A = \text{VLM}_{\text{detect}}(I) = \{a_i = (p_i, c_i, b_i)\}$ (position + class + bbox). Finally, $U_{\text{priority}} = \text{VLM}_{\text{analyze}}(I, T, H, A, Q, S)$ uses a skill-aware prompt to prioritize targets for attack or protection.
    - **Mechanism**: Deconstructs SC2 tactical decisions into "define objective (skill plan) → observe entities (detect) → prioritize by plan (analyze)." Every step leverages native VLM visual and linguistic capabilities without fine-tuning. Compared to end-to-end action generation, this staged prompting significantly reduces "misidentified unit" or "forgotten objective" errors.
    - **Design Motivation**: The core of SC2 micromanagement is "focus fire on the right target"—selecting the wrong priority in one step can collapse the entire engagement. Extracting priority as a separate VLM sub-task focuses the model's attention on the most critical sub-task rather than overwhelming it with full-field information.

2. **RAG Knowledge Injection**:
    - **Function**: For the set of priority units $U_{\text{priority}}$ selected by MPI, a knowledge tuple $K(u) = \{s_u, m_u, t_u\}$ (unit specs, matchup data, tactical advice) is retrieved for each unit $u$ based on its class $c_u$. Then, $D = \text{VLM}_{\text{synthesize}}(I, T, H, U_{\text{priority}}, \{K(u)\})$ integrates this into final tactical commands.
    - **Mechanism**: SC2 tactics rely on "common sense"—e.g., Stalkers fear Marauder slow armor, Hydralisks fear Colossus AOE. This knowledge exists in VLM pre-training but is unstable to invoke. An external SC2 knowledge base provides hard injection, feeding unit/matchup awareness directly into the prompt to ensure the VLM avoids basic tactical errors.
    - **Design Motivation**: Zero-shot game understanding in VLMs is largely a synthesis of world knowledge and vision. Explicitly grounding domain knowledge in the prompt is more reliable than relying on "memory." Ablations confirm RAG provides significant standalone gains and synergistic benefits when combined with MPI.

3. **Dynamic Role Assignment**:
    - **Function**: Assigns $N$ agents from a role set $\mathcal{Z}$, defining a mapping $\phi: \mathcal{N} \to \mathcal{Z}$, with a utility function $U(\phi, s)$ evaluating the configuration. This is implemented as $z_i = \text{VLM}_{\text{role}}(I, T, C)$, where the VLM assigns roles like tank / DPS / scout based on image + text + context.
    - **Mechanism**: SC2 tactics often require division of labor—some Stalkers kite while others focus fire. Applying the same policy to all units leads to coordination failure. Making role assignment an independent VLM call explicitly models multi-agent coordination.
    - **Design Motivation**: Assigning roles before low-level actions acts as a skill prior for subsequent action generation, reducing the effective dimension of the action space. Ablations show win rates drop from 87% to 70% without Role assignment, proving coordination is scarcer than perception.

### Loss & Training
The VLM side is zero-shot with no training. The MARL side follows SMAC standards: 5M steps, 2Hz decision frequency, dual A100 40GB. Episode termination occurs upon total elimination, all-agent death, or a 300s timeout. Sparse rewards are used to avoid bias.

## Key Experimental Results

### Main Results (3m Basic Scenario)

| Paradigm | Method | Input Mode | Training Steps | Win Rate (%) |
|---|---|---|---|---|
| MARL | MAPPO | Vision+Text | 5M | 19.3 ± 3.2 |
| MARL | IPPO | Vision Only | 5M | 18.2 ± 2.8 |
| MARL | QMIX | Vision Only | 5M | **27.1 ± 4.1** |
| MARL | QTRAN | Vision Only | 5M | 2.0 ± 1.4 |
| MARL | IQL / VDN | Vision Only | 5M | 0.0 |
| VLM (Closed) | GPT-4o | VLM-Optimized | 0 | **81 ± 3.9** |
| VLM (Closed) | GPT-4-Turbo | VLM-Optimized | 0 | 79 ± 4.1 |
| VLM (Closed) | Qwen-VL-Plus | VLM-Optimized | 0 | 75 ± 4.3 |
| VLM (Open) | Qwen3-VL-30B | VLM-Optimized | 0 | 50 ± 5.0 |
| VLM (Open) | Qwen3-VL-8B | VLM-Optimized | 0 | 40 ± 4.9 |

The VLM paradigm significantly outperforms MARL trained for 5M steps. Notably, adding text to IPPO slightly degraded performance (16.6 vs 18.2), suggesting from-scratch MARL struggles to effectively fuse pre-trained text embeddings, whereas VLMs benefit naturally from language channels due to pre-training alignment.

### Ablation Study (AVA on mixed_units, GPT-4-Turbo)

| Role | MPI | RAG | Win Rate (%) | Meaning |
|---|---|---|---|---|
| ✓ | ✓ | ✓ | **87 ± 3.4** | Full AVA |
| ✓ | ✓ | – | 71 ± 4.5 | w/o RAG (-16) |
| ✓ | – | ✓ | 65 ± 4.8 | w/o MPI (-22) |
| – | ✓ | ✓ | 70 ± 4.6 | w/o Role (-17) |
| ✓ | – | – | 24 ± 4.3 | Role only, ineffective |
| – | ✓ | – | 50 ± 5.0 | MPI only |
| – | – | – | 20 ± 4.0 | Vanilla VLM |

### Key Findings
- **MPI is the most critical component of AVA**: Removing MPI caused a larger drop than removing Role or RAG (87→65), indicating that "identifying who to hit" is more vital than "assigning roles."
- **VLMs still have performance ceilings in high-complexity scenarios**: In `2c_vs_64zg` and `6r_vs_8z`, **all VLMs hit a 0% win rate**, including models and Qwen3-VL-30B which reached 90% elsewhere. This exposes VLM limitations in continuous kiting and high-frequency precision micro.
- **Cross-modal alignment capabilities differ significantly**: MARL performance dropped with text, while VLM performance surged, validating the cross-modal grounding advantages provided by VLM pre-training.
- **Training efficiency contrast is extreme**: MARL 5M steps ≈ 19% vs. VLM 0 steps ≈ 81%; however, once trained, MARL inference is controllable and cheap, whereas long-term VLM API costs remain high.

## Highlights & Insights
- **"Duel within the same observation space" is the key to a fair benchmark**: Previously, MARL used scalars and LLMs used strings, making conclusions difficult to generalize. This work unifies everything into POMDP + multi-mode observations for direct comparison.
- **VLM "Human Alignment" can be quantified**: Professional SC2 players performed blind evaluations, proving VLM decision interpretability is significantly higher than MARL (statistical significance)—a critical factor for future human-understandable AI decisions.
- **Failure cases map the VLM capability boundary**: The 0% win rate in extremely high-complexity scenarios is not an engineering bug but a real ceiling for "dense spatial reasoning + high-frequency temporal consistency," marking a clear frontier for future research.
- **Transferability of the AVA agent recipe**: The combination of MPI (prioritize critical targets) + Role (division of labor) + RAG (domain knowledge injection) can be applied to any "real-time multi-agent + VLM" scenario (e.g., autonomous vehicle fleets, robotic collaboration).

## Limitations & Future Work
- Comparisons focused mainly on 5M training steps on the basic 3m scenario; longer training or newer MARL algorithms (e.g., GRF, HASAC) might bridge the gap.
- Detailed VLM costs and latency were not listed; 2Hz decision frequency is already the upper limit for VLMs, and higher-frequency micro (kiting) remains impossible (0% win rate). Real-time deployment feasibility is limited.
- AVA is a proof-of-concept; the authors state the main contribution is the benchmark rather than a novel agent architecture.
- Evaluations utilized PvE and limited PvP; systematic ladder-level VS-human testing is needed to verify the "absolute meaning" of the win rates.

## Related Work & Insights
- **vs SMAC / SMACv2**: Addresses criticisms of abstract features and simplified abilities by providing an upgraded version with RGB, full abilities, and multi-mode observations with VLM interfaces.
- **vs LLM-PySC2**: While LLM-PySC2 focuses on macro-strategy (building, expansion), Ours focuses on micro-management (focus fire, ability timing), making them complementary.
- **vs VS-Bench**: Tests strategic reasoning across multiple games but with abstract multi-agent settings; AVACraft provides fine-grained evaluation for the gold standard SC2.
- **vs Voyager / LLM-Agent for Minecraft**: While both use VLM/LLM for gaming, Ours is the first to perform a systematic cross-paradigm evaluation in a real-time, high-frequency adversarial environment like SC2.

## Rating
- Novelty: ⭐⭐⭐⭐ Strong benchmark originality; AVA agent is an engineering combination
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 MARL × 6 VLM × 21 scenarios + component ablation + human evaluation
- Writing Quality: ⭐⭐⭐⭐ Clear environment formalization and intuitive cross-modal ablation
- Value: ⭐⭐⭐⭐⭐ Provides the first standardized arena for MARL ↔ VLM dialogue, which the community urgently needs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching](../../AAAI2026/llm_agent/pertouch_vlm-driven_agent_for_personalized_and_semantic_image_retouching.md)
- [\[ICCV 2025\] GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training](../../ICCV2025/llm_agent/gtr_guided_thought_reinforcement_prevents_thought_collapse_i.md)
- [\[ACL 2026\] Grounding Agent Memory in Contextual Intent](grounding_agent_memory_in_contextual_intent.md)
- [\[ACL 2026\] Mem^p: Exploring Agent Procedural Memory](memp_exploring_agent_procedural_memory.md)
- [\[ACL 2026\] GOAT: A Training Framework for Goal-Oriented Agent with Tools](goat_a_training_framework_for_goal-oriented_agent_with_tools.md)

</div>

<!-- RELATED:END -->
