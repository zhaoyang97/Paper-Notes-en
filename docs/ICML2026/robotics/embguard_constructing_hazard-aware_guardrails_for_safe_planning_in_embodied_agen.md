---
title: >-
  [Paper Note] EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents
description: >-
  [ICML 2026][Robotics][Embodied agent] EmbGuard decouples "physical safety judgment for embodied agents" from the policy into an independent small-model guardrail—taking (observation image…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Embodied agent"
  - "safety guardrail"
  - "action-conditioned risk"
  - "synthetic data"
  - "MLLM"
date: 2026-05-08
content_hash: 01407bf9e8f6044e
---

# EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents

**Conference**: ICML 2026  
**arXiv**: [2605.30924](https://arxiv.org/abs/2605.30924)  
**Code**: The paper promises public release (code/data/models)  
**Area**: Embodied Intelligence / AI Safety / Multimodal VLM  
**Keywords**: Embodied agent, safety guardrail, action-conditioned risk, synthetic data, MLLM

## TL;DR
EmbGuard decouples "physical safety judgment for embodied agents" from the policy into an independent small-model guardrail—taking (observation image, candidate action) as input and outputting (risk binary, risk category, hazard explanation). Models at 2B/4B scales match the performance of GPT-5.1/Gemini-2.5-Pro while suppressing the pervasive "false positive" issue commonly found in baselines.

## Background & Motivation

**Background**: MLLM-driven embodied agents (PaLM-E, RT, CogAct, GR00T, etc.) are capable of performing long-horizon physical tasks, but they typically delegate safety to the same policy model.

**Limitations of Prior Work**: Cramming safety and tasks into a single large policy results in sub-optimal performance for both: either the model focuses on the task and neglects risks (false negatives), or it becomes overly conservative and refuses tasks at the slightest hint of danger (false positives). IS-Bench data shows that strong models like Gemini-2.5-Pro misclassify 83.3% of benign scenarios as hazardous.

**Key Challenge**: (i) Physical risk stems neither from the "environment alone" nor the "action alone," but from their **interaction**—placing a potted plant above a power outlet is not inherently dangerous; "watering the plant" is. (ii) MLLM visual priors favor perceptually salient hazards (fire, electricity, sharp objects) but systematically under-report risks requiring causal or temporal reasoning (crushing, contamination, chemical exposure). (iii) Dumping safety reasoning into increasingly large policy models is costly and suffers from latency issues incompatible with real-time control.

**Goal**: (1) Decouple safety reasoning from the policy into an independent guardrail module; (2) achieve fine-grained judgment of "action-conditioned physical risk" (binary + category + natural language explanation); (3) maintain high precision while reducing false positives, and remain small enough for real-time deployment.

**Key Insight**: The authors model the task as a function $\mathcal{R}:(I,a)\to(r_{\text{bin}},r_{\text{type}},h)$ that outputs a risk binary $r_{\text{bin}}$, risk type $r_{\text{type}}$, and hazard description $h$ from an image $I$ and action $a$. By utilizing a three-stage synthesis (manual + GPT-5.1 + Gemini 3 Image), they generate large-scale (image, action) paired data, allowing small models to learn causal intuition for "action-triggered risks" through data diversity rather than parameter scale.

**Core Idea**: Use a scene graph for controllable hazardous structure representation. Extend variations via four categories (causal risky / selective risky / decoupled benign / absent benign) to generate 15.1K training samples, fine-tuning 2B/4B Qwen-3-VL models into specialized guardrails.

## Method

### Overall Architecture
EmbGuard consists of (i) a data generation pipeline, (ii) an SFT phase, and (iii) an inference-time guardrail module:

1.  **Data Pipeline**: Risk-driven scenario generation → compositional variant diversification → image generation and VQA verification. This produces the EmbHazard training set (15.1K (image, action) pairs / 8.7K images) and the EmbGuardTest test set (329 real-world scenarios with manual annotations).
2.  **Training**: SFT Qwen-3-VL-2B/4B on EmbHazard for 4 epochs, lr=1e-5, using 8×A6000; the vision encoder is frozen.
3.  **Inference**: Integrate the guardrail into the embodied agent's planning loop. At each step, query EmbGuard with (observation, candidate action) to provide (safe/unsafe, risk_type, hazard_description) feedback to the policy.

### Key Designs

1.  **Risk-Decoupled Task Formulation + 7-Category Risk Taxonomy**:
    *   **Function**: Explicitly decouples "safety vs. task" and constrains the guardrail to output judgments at three levels of granularity simultaneously.
    *   **Mechanism**: Defines $\mathcal{R}:(I,a)\to(r_{\text{bin}}\in\{0,1\},\ r_{\text{type}},\ h)$, where $r_{\text{type}}$ is restricted to 7 categories (Fire / Electrical / Slip-Trip-Fall / Cut-Sharp / Crush-Pinch / Contamination / Chemical-Toxic Exposure), derived from WHO ICD-11 and CPSC NEISS accident databases. Each category is refined into 24 risk-inducing patterns. $h$ uses free-text to describe the hazard configuration instead of closed-set labels to facilitate transfer to unseen object combinations. Evaluation is also hierarchical: Potential Risk Acc → Risk Type Acc → Hazard Acc, where the latter two are conditioned on samples where $r_{\text{bin}}$ is correct to prevent "correct guess with wrong reasoning."
    *   **Design Motivation**: The hierarchical task definition directly maps to "detect → classify → explain," allowing the policy to fallback even when only partial information is available. Functional verification in Figure 8 shows that mitigation alignment is 90.4% when both (risk_type + hazard) are correct, dropping to 28.4% when both are wrong.

2.  **Scene Graph-Based Compositional Variant Generation**:
    *   **Function**: Expands 2.4K seed scenarios into 15.1K training pairs, covering the four quadrants of "risky/benign × single/multiple risks × same hazard with different actions."
    *   **Mechanism**: Represents each hazard as a subgraph $\mathcal{H}\subseteq\mathcal{G}$ of a scene graph (e.g., (power_strip, beneath, plant_pot)). Four transformations are performed on the graph: (a) scene augmentation $f_{\text{scene}}:(\mathcal{G},\mathcal{H})\to(\mathcal{G}',\mathcal{H})$ adds irrelevant objects; (b) hazard addition $f_{\text{hazard}}^{+}$ introduces new hazards for Selective Risky; (c) action modification $f_{\text{action}}:a\to a'$ alters the action to break interaction with the hazard for Decoupled Benign; (d) hazard removal $f_{\text{hazard}}^{-}:(\mathcal{G},\mathcal{H})\to(\mathcal{G}',\emptyset)$ for Absent Benign. Operating at the graph level rather than the text level prevents accidental destruction of spatial relationships. Finally, GPT-5.1 converts variant graphs into scene descriptions for Gemini-3-pro-image-preview to generate images, which are then filtered by a VQA component (GPT-4o).
    *   **Design Motivation**: Covering these four quadrants provides *counterfactual* pairings (e.g., hazard exists but action is safe), which is critical for teaching MLLMs to distinguish "action-triggered risk" from merely "seeing fire and shouting danger." This directly addresses the over-conservative bias of baselines.

3.  **Frozen Vision Encoder SFT Recipe**:
    *   **Function**: Enables small models to learn risk detection and natural language explanation without degrading existing visual priors.
    *   **Mechanism**: Qwen-3-VL-2B/4B + LLaMA-Factory + lr=1e-5 + 4 epochs. The key trick is **freezing the vision encoder**. Preliminary experiments showed that while unfreezing the ViT improved binary risk detection, the quality of hazard explanations collapsed, likely because small model capacity is insufficient for simultaneous vision and reasoning adaptation.
    *   **Design Motivation**: Treating visual capabilities as a "pre-packaged sensor" and focusing the LLM head on risk causality is a common strategy for small models to avoid task interference.

### Loss & Training
Standard multi-task SFT using a single generative loss (outputting a JSON triplet). No additional auxiliary losses. GPT-4o-as-judge is used to evaluate free-text hazard descriptions (achieving human $\kappa=0.90$).

## Key Experimental Results

### Main Results
Comparison of 11 open-source MLLMs, 4 closed-source MLLMs, and EmbGuard-2B/4B on EmbGuardTest (329 real samples) and Held-out (563 synthetic samples). Metrics: (Potential Risk Acc / Risk Type Acc / Hazard Acc).

| Model | Scale | EmbGuardTest | Held-out | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| Qwen-3-VL-2B (Base) | 2B | 47.2 / 37.5 / 5.9 | 59.4 / 32.5 / 27.4 | Same-size baseline |
| EmbGuard-2B | 2B | 51.6 / 44.6 / 7.4 | 68.3 / 59.5 / 36.6 | Outperforms base after SFT |
| Qwen-3-VL-4B (Base) | 4B | 47.3 / 51.0 / 10.5 | 58.3 / 53.5 / 48.6 | Same-size baseline |
| EmbGuard-4B | 4B | 54.3 / 50.3 / 14.6 | 71.2 / 67.6 / 50.1 | Approaches GPT-5.1 |
| GPT-5.1 | Closed | 55.8 / 58.1 / 33.4 | 69.1 / 62.0 / 57.0 | Strongest commercial model |
| Gemini-2.5-Pro | Closed | 58.4 / 56.8 / 29.3 | 61.4 / 68.3 / 63.8 | High recall, high false positives |
| Qwen-3-VL-235B | 235B | 49.5 / 56.4 / 26.7 | 71.3 / 60.0 / 51.2 | 100× parameters |

Inference Latency: EmbGuard-2B 0.535s/sample, EmbGuard-4B 0.719s/sample (on RTX 6000 Ada), suitable for real-time embodied loops.

### Ablation Study

| Experiment | Key Metric | Description |
| :--- | :--- | :--- |
| Human vs MLLM (EmbGuardTest subset) | Human 85.6/90.9/63.6 vs GPT-5.1 55.5/42.0/31.9 | Massive headroom remains for models |
| IS-Bench Step Acc / Precision / Recall / F1 | EmbGuard-4B 63.1/25.7/71.7/38.3 vs Gemini-2.5-Pro 49.9/22.2/88.2/40.7 | Gemini has higher recall but poor precision; EmbGuard has highest step acc |
| Mitigation Alignment (Policy uses output) | Both Correct 90.4% → Risk Type Error 78.5% → Hazard Error 58.6% → Both Error 28.4% | Confirms fine-grained accuracy is required for correct mitigation |
| Over-conservative bias | Gemini-2.5-Pro misjudges 83.3% benign as risky | EmbGuard is significantly more balanced |

### Key Findings
- Small models can catch up to large closed-source models through data diversity: 2B/4B EmbGuard outperforms same-sized Qwen/InternVL/Gemma and rivals GPT-5.1/Gemini-2.5-Pro on EmbGuardTest.
- Baseline models are hypersensitive to **perceptually salient** risks (fire, electricity) but systematically miss risks requiring causal reasoning (crushing, contamination). EmbGuard flattens this bias via balanced 7-category training and counterfactual variants.
- On IS-Bench, increasing recall alone is insufficient: Gemini-2.5-Pro has 88.2% recall but only 49.9% step accuracy because it flags safe steps as unsafe, causing unnecessary interruptions. EmbGuard is more selective.
- Even the strongest closed-source MLLM only achieves 33.4% Hazard Acc (EmbGuardTest) vs. Human 63.6%, suggesting that causal explanations for "why it is dangerous" are far from saturated.

## Highlights & Insights
- The architectural choice to "decouple safety into a guardrail" is the core thesis, systematically migrating the LlamaGuard concept to physical embodied safety.
- Using scene graphs for structural counterfactual variants is a key technical contribution—avoiding the randomness of text-based prompts and using VQA filters for quality control.
- The "counter-intuitive" finding on freezing the vision encoder (where unfreezing degrades explanation capabilities in small models) is a valuable takeaway for multi-task SFT on small VLMs.
- The mitigation alignment experiment closes the loop on the guardrail's value: merely reporting "danger" is insufficient; reporting the correct risk type and hazard is necessary for the policy to select the right mitigation.

## Limitations & Future Work
- **Visual Sensor Assumptions**: The guardrail assumes the observation image contains all hazard information. It cannot detect hazards outside the FOV (e.g., an active stove behind the camera).
- **Incompatibility with Continuous Control Policies**: Currently, the guardrail accepts text-based action descriptions. VLA models that output continuous joint torques cannot be easily integrated; "safety reasoning under low-level control" is identified as future work.
- Lacks physical robot validation; all evaluations were conducted in OmniGibson/IS-Bench simulations and on static images.
- The 7-category taxonomy might miss niche risks (e.g., chemical reactions in factories, cross-contamination in medical settings).
- The pipeline depends heavily on GPT-5.1/Gemini 3 Image, which introduces cost and reproducibility issues due to potential API drift.

## Related Work & Insights
- **vs. IS-Bench (2025)**: IS-Bench evaluates whether agents plan safely; EmbGuard provides the implementation for the "proactive risk avoidance mechanism" IS-Bench calls for.
- **vs. LlamaGuard / ShieldAgent**: These are guardrails for text/video/digital agents. EmbGuard extends the architecture to the physical embodied domain with "action-conditioned + visual hazard" dimensions.
- **vs. Sermanet et al. (2025)**: They discuss embodied guardrails conceptually; EmbGuard realizes the concept with a trainable model and dataset.
- **vs. Risk-Aware Planning (Khan 2025)**: Rather than modifying the planner to be risk-aware, EmbGuard proposes a modular architecture where the planner remains unchanged while the guardrail is added as a plug-in.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First embodied physical safety guardrail; new task formulation and data pipeline.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive baselines and evaluations (EmbGuardTest, Held-out, IS-Bench), although missing real-world robot validation.
- **Writing Quality**: ⭐⭐⭐⭐ Very clear explanation of the four-quadrant scenario definition and hierarchical metrics.
- **Value**: ⭐⭐⭐⭐⭐ The dataset and open-source models provide immediately usable infrastructure for the embodied safety community.

## Rating
- **Novelty**: TBD
- **Experimental Thoroughness**: TBD
- **Writing Quality**: TBD
- **Value**: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](../../ICLR2026/robotics/rei-bench_can_embodied_agents_understand_vague_human_instructions_in_task_planni.md)
- [\[ICML 2026\] Embodied Task Planning via Graph-Informed Action Generation with Large Language Models](embodied_task_planning_via_graph-informed_action_generation_with_large_language_.md)
- [\[ICML 2026\] Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning](drift_is_a_sampling_error_snr-aware_power_distributions_for_long-horizon_robotic.md)
- [\[ICLR 2026\] OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning](../../ICLR2026/robotics/omnieva_embodied_versatile_planner_via_task-adaptive_3d-grounded_and_embodiment-.md)
- [\[ICLR 2026\] WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment](../../ICLR2026/robotics/weboperator_action-aware_tree_search_for_autonomous_agents_in_web_environment.md)

</div>

<!-- RELATED:END -->
