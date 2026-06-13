---
title: >-
  [Paper Note] Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards
description: >-
  [ACL 2026][LLM/NLP][Floor plan generation] The authors converted 80,000 real apartment floor plans from RPLAN into a JSON polygon format and performed two-stage training on Llama-3.3-70B-Instruct (SFT + GRPO with verifia…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Floor plan generation"
  - "JSON structured output"
  - "GRPO"
  - "verifiable rewards"
  - "bubble diagram"
date: 2026-05-08
content_hash: 61dfc80f6da1cfb0
---

# Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards

**Conference**: ACL 2026 Findings  
**arXiv**: [2605.14117](https://arxiv.org/abs/2605.14117)  
**Code**: <https://github.com/ludolara/floor-plan-rlvr>  
**Area**: LLM Generation / RLVR / Architectural Design  
**Keywords**: Floor plan generation, JSON structured output, GRPO, verifiable rewards, bubble diagram

## TL;DR
The authors converted 80,000 real apartment floor plans from RPLAN into a JSON polygon format and performed two-stage training on Llama-3.3-70B-Instruct (SFT + GRPO with verifiable rewards: connectivity + total-area rewards, with overlap/parsing failures hard-zeroed). This allows the LLM to output CAD-ready floor plans that simultaneously satisfy bubble diagram topological constraints and numerical area constraints. On an 8-room task, Compatibility decreased by 94% (2.5 → 0.15) compared to HouseDiffusion.

## Background & Motivation
**Background**: Architectural floor plan generation is a classic task in AI-aided design. Mainstream approaches include: ① Visual GANs (House-GAN/House-GAN++) using bubble diagrams as graph conditions to output rasterized images; ② Diffusion-based models (HouseDiffusion) using 1-D polygonal loops for rooms with iterative denoising; ③ Early symbolic systems (Lopes et al.) using constraint satisfaction and manual rules.

**Limitations of Prior Work**: ① Mainstream models only accept connectivity (which rooms are adjacent) and cannot handle **numerical constraints** (e.g., a bedroom must be 12 m², total area 80 m²); ② Visual/diffusion outputs are rasterized or scale-invariant shapes, which are not directly compatible with CAD software; ③ Existing evaluations focus on Compatibility/Realism/Diversity but do not explicitly measure if "user-specified room areas are met" or basic geometric validity such as "room overlaps"; ④ Tell2Design attempted text-to-floor-plan but evaluated via image rendering, ignored structural errors, and handled spatial relations poorly.

**Key Challenge**: Professional architects require simultaneous control over topology (room adjacency) + geometric values (individual room size, polygon placement) + geometric validity (no-overlap, closed polygons). The first two requirements necessitate structured text output (avoiding rasterization), while the third requires the generator to learn "geometric feasibility." Default LLM free-form generation is prone to producing invalid JSON or overlapping polygons.

**Goal**: ① Use JSON as a unified representation for structural and numerical constraints; ② Use LLMs to learn the real distribution of RPLAN; ③ Implement automatic verifiable rewards using RL for "matching bubble diagrams" and "matching total area," with "overlap/parsing failure = 0 reward" as a hard feasibility condition; ④ Propose four new indicators (Room Area / Room ID / Overlap / % Overlap) to measure constraint adherence.

**Key Insight**: Reframe the problem from "image generation + post-processing" to a "structured text-to-text" seq2seq task. LLMs are proficient at generating JSON and following schemas. By using RLVR, rewards that can be programmatically verified are directly applied, avoiding the trap of using non-verifiable reward models to "learn taste."

**Core Idea**: Treat floor plan generation as a dual RL task of "program synthesis + geometric verification." Combining connectivity rewards, total-area rewards, and hard-zeroing for overlaps and invalid JSON enables LLMs to learn structured outputs that are both topologically and geometrically consistent.

## Method

### Overall Architecture
A two-stage training process is applied to Llama-3.3-70B-Instruct. The input is a JSON (containing room_count / total_area / spaces list / input_graph bubble diagram), and the output is also a JSON (each space contains id / room_type / area / floor_polygon vertex list in meters). **Stage 1 SFT**: LoRA supervised fine-tuning on RPLAN-converted JSON floor plans (rank 64, $\alpha=128$, 2 epochs) to learn the basic mapping from prompt to JSON. **Stage 2 GRPO**: After merging LoRA, RL is performed with verifiable rewards. For each prompt, $G=4$ candidates are sampled, and group-relative advantage is calculated based on connectivity and total-area rewards. **Inference**: Best-of-10 selection, choosing the final output based on "minimum overlap area → Compatibility tie-break."

### Key Designs

1.  **JSON-based Structured Representation + RPLAN → Polygon Pipeline**:
    - **Function**: Transitions floor plan generation from "image generation" to "structured text generation," leveraging inherent LLM capabilities.
    - **Mechanism**: Each space (room or door) is a JSON object including an id (e.g., `bedroom|0`), room_type, area (m²), and floor_polygon (absolute coordinate vertex list). The input includes a room_count, total_area, and input_graph adjacency list. A custom converter transforms RPLAN $256\times256\times4$ images into polygon JSON by reconstructing polygons from pixel scales to meters and deriving bubble diagrams from connectivity.
    - **Design Motivation**: JSON (1) unambiguously parses numerical constraints; (2) enforces schema consistency; (3) naturally represents hierarchical floor plans; (4) can be directly fed into CAD software.

2.  **GRPO + Dual Verifiable Reward + Hard Feasibility**:
    - **Function**: Aligns the model with programmatically verifiable constraints post-SFT and eliminates geometrically invalid outputs.
    - **Mechanism**: For a prompt $x$, $G=4$ candidates are sampled to calculate group-relative advantage $\hat{A}_i = (R(x, y_i) - \mu_x) / (\sigma_x + \epsilon)$ using a PPO-style surrogate objective $L^{\text{GRPO}}(\theta)=\mathbb{E}_x[\frac{1}{G}\sum_i \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}\hat{A}_i]$ without a critic network. Rewards include: (a) **Connectivity reward** $r_{\text{conn}}\in[0,1]$ by comparing the reconstructed graph to the input; (b) **Total area reward** $r_{\text{TA}}=\max(0, 1-\text{TAE})$, where $\text{TAE}=|A(y)-A^\star|/A^\star$. The critical component is the **Hard feasibility condition**: any JSON parsing failure or room polygon overlap results in a reward of 0.
    - **Design Motivation**: Architectural constraints are verifiable, making them ideal for RLVR. Hard feasibility prevents "reward hacking" (e.g., shrinking rooms to points to game the adjacency graph).

3.  **Best-of-10 Selection + Token-level Prompting**:
    - **Function**: Further refines geometrically valid and constraint-matching outputs during inference.
    - **Mechanism**: Ten candidates are sampled (temperature 0.7, top-p 0.9). The final output is selected by prioritizing "minimum overlap area" followed by Compatibility. System messages explicitly instruct that "no two room polygons ever overlap" and "every adjacency in the bubble diagram must be bridged by exactly one door."
    - **Design Motivation**: Single sampling occasionally results in overlaps. Best-of-10 reduces overlap from 0.26 (at n=1) to 0.13 (at n=10). This represents a direct trade-off between inference computation and constraint adherence.

### Loss & Training
SFT uses standard NLL: $L^{\text{SFT}}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}[-\sum_t \log \pi_\theta(y_t|y_{<t},x)]$, with LoRA rank 64, lr 1e-4, 2 epochs. GRPO uses AdamW, lr 1e-6, clip 0.2, KL coefficient 0.04, temperature 0.9, top-p 1.0. For each prompt, 4 generations are sampled with a total length of 4096 tokens. Optimal checkpoints often occur at step 100 (~2h). 80,788 floor plans from RPLAN are split by room count (5/6/7/8) for zero-shot cross-room-count generalization testing.

## Key Experimental Results

### Main Results (vs HouseDiffusion and other SOTA)

| Method | Comp(5)↓ | Comp(6)↓ | Comp(7)↓ | Comp(8)↓ | Realism↑ (8) | Div(8)↓ |
|------|----------|----------|----------|----------|--------------|---------|
| Johnson et al. 2018 | 7.7 | 6.5 | 10.2 | 11.3 | -1.00 | 186.0 |
| House-GAN | 2.5 | 2.4 | 3.2 | 5.3 | -0.95 | 66.4 |
| House-GAN++ | 1.9 | 2.2 | 2.4 | 3.9 | -0.52 | 32.9 |
| HouseDiffusion | 1.5 | 1.2 | 1.7 | 2.5 | -0.19 | 9.5 |
| **Ours (SFT+RLVR best-of-10)** | **0.01** | **0.02** | **0.10** | **0.15** | **0.03** | **7.0** |
| Gain vs HouseDiffusion | -99.3% | -98.3% | -94.1% | **-94.0%** | +0.22 | -26.3% |

Key Observation: In the 8-room task, Compatibility dropped from 2.5 (HouseDiffusion) to 0.15 (-94%). Realism is 0.03 (near 0, implying generated plans are indistinguishable from real ones), and Diversity (FID) is the lowest among all methods.

### Ablation Study (Training Stages + Inference Budget)

| Configuration | 8 Room Area↓ | Room ID↑ | Overlap↓ | % Overlap↓ | Compatibility↓ |
|------|------------------|----------|----------|------------|----------------|
| Few-shot (3-shot, best-of-10) | 0.12 | 0.91 | 0.51 | 0.04 | 6.89 |
| SFT only (best-of-10) | 0.08 | **1.00** | 0.37 | 0.01 | 0.41 |
| **SFT + RLVR (best-of-10)** | 0.10 | **1.00** | **0.13** | **0.00** | **0.15** |
| SFT + RLVR (best-of-1) | **0.09** | 1.00 | 0.26 | 0.01 | 1.89 |
| SFT + RLVR (best-of-100) | 0.09 | 1.00 | 0.10 | 0.00 | **0.02** |

Averaging 5-8 rooms, RLVR compared to SFT alone reduced Overlap by 65% and Compatibility by 56% without degrading Room Area or Room ID performance.

### Key Findings
- **The paradigm of structured text generation + RLVR significantly outperforms visual/diffusion SOTA in professional design tasks**: Compatibility improved by magnitudes, while geometric validity (Overlap 0.13), area precision (Room Area 10%), and label accuracy (Room ID 1.00) all passed benchmarks.
- **Hard feasibility is the key to preventing reward hacking in RLVR**: Without it, the model tends to shrink rooms to infinitesimal points to satisfy the adjacency graph.
- **70B models are necessary; 8B models are insufficient**: Llama-3.1-8B-Instruct, even with high-rank LoRA, often falls into repetitive loops and produces invalid geometry. Llama-3.3-70B demonstrated the stability required for such tasks.
- **Few-shot prompting with GPT-4o / o3 / QwQ-32B fails**: These models failed to produce closed polygons or consistent numerical values without SFT.
- **JSON is more reliable, but the model can handle natural language zero-shot**: Replacing JSON with natural language templates maintained similar metrics for Room Area/ID/Overlap, though Compatibility slightly decreased due to linguistic ambiguity.

## Highlights & Insights
- **Dual RL of "program synthesis + geometric verification" is a blueprint for verifiable rewards**: This model extends RLVR from math and code to 2D geometry and architecture, applicable to any task where constraints are programmatically verifiable (e.g., PCB layout, chip floor planning, UI mockups).
- **Hard feasibility + Soft rewards combination**: Using binary kill switches for essential constraints (non-overlap, valid JSON) and continuous rewards for optimization targets (connectivity, area) is more stable than a single composite reward.
- **RLVR is significantly shorter than SFT**: Optimal GRPO checkpoints often occur very early (step 100), suggesting that minimal RL steps can drastically enhance constraint adherence post-SFT.
- **New constraint adherence evaluation system**: The metrics Room Area / Room ID / Overlap / % Overlap fill the gaps left by previous metrics like Compatibility/Realism.
- **Transparent "behind the scenes" reporting**: The authors honestly shared failed experiments (e.g., ProcTHOR data augmentation failures), providing valuable engineering insights.

## Limitations & Future Work
- Only automatically verifiable constraints were modeled; architectural concerns like circulation, egress, daylight, and structural building codes were not included.
- The study was limited to the RPLAN dataset (Asian single-story residential); generalizability to commercial buildings or multi-story structures remains unknown.
- Inference requires a high budget (Best-of-N) to effectively suppress overlaps.
- Reward weights were heuristically set (50/50); the trade-offs between weights across different room scales were not explored.
- The RPLAN dataset carries restricted research-only licensing, limiting redistribution.

## Related Work & Insights
- **vs HouseDiffusion**: Their 1-D polygonal loop diffusion achieved peer-reviewed SOTA Compatibility (2.5), but lacked numerical constraints and overlap management. This work's 0.15 represents a massive improvement.
- **vs Tell2Design**: Similar text-to-floor-plan approach, but they evaluated via image rendering; this work performs geometric evaluation directly from JSON.
- **vs House-GAN / House-GAN++**: GAN approaches output scale-invariant images; this work outputs absolute coordinates in meters ready for CAD.
- **vs ProcTHOR**: Attempts to use ProcTHOR for data augmentation failed due to pathological topologies (e.g., bathrooms acting as hallways), demonstrating the risks of synthetic data distribution.
- **vs RLVR in Math/Code**: This work proves the RLVR paradigm can extend to professional fields like geometry and architecture, expanding verifiers from code execution to geometric detection.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First complete implementation of JSON representation + RLVR + hard feasibility for floor plan generation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison with 5 baselines across 4 room-count tasks and new metrics.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear methodology and honest discussion of failures.
- **Value**: ⭐⭐⭐⭐⭐ Provides a replicable template for structured generation in highly constrained professional domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Generative Interfaces for Language Models](generative_interfaces_for_language_models.md)
- [\[ACL 2026\] Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design](solver-independent_automated_problem_formulation_via_llms_for_high-cost_simulati.md)
- [\[ICLR 2026\] Rethinking Code Similarity for Automated Algorithm Design with LLMs](../../ICLR2026/llm_nlp/rethinking_code_similarity_for_automated_algorithm_design_with_llms.md)
- [\[ICML 2026\] T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning](../../ICML2026/llm_nlp/t2po_uncertainty-guided_exploration_control_for_stable_multi-turn_agentic_reinfo.md)
- [\[NeurIPS 2025\] Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options](../../NeurIPS2025/llm_nlp/preference-based_reinforcement_learning_beyond_pairwise_comparisons_benefits_of_.md)

</div>

<!-- RELATED:END -->
