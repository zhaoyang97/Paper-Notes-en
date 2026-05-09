---
title: >-
  [Paper Note] Physics-Informed Autonomous LLM Agents for Explainable Power Electronics Modulation Design
description: >-
  [AAAI 2026][LLM Agent][physics-informed agent] This paper proposes PHIA, a system in which an LLM planner collects design requirements via a chat interface and autonomously coordinates a physics-informed neural network surrogate model (hierarchical PINN) with optimization algorithms to iteratively generate power converter modulation designs, achieving a 63.2% reduction in MAE, a 33× speedup in design time, and usability validated by 20 domain experts.
tags:
  - AAAI 2026
  - LLM Agent
  - physics-informed agent
  - power electronics
  - modulation design
  - PINN
  - explainability
date: 2026-05-08
content_hash: db3a53f5d60144ad
---

# Physics-Informed Autonomous LLM Agents for Explainable Power Electronics Modulation Design

**Conference**: AAAI 2026
**arXiv**: [2411.14214](https://arxiv.org/abs/2411.14214)
**Code**: None
**Area**: LLM Agent / Industrial Applications
**Keywords**: physics-informed agent, power electronics, modulation design, PINN, explainability

## TL;DR
This paper proposes PHIA, a system in which an LLM planner collects design requirements via a chat interface and autonomously coordinates a physics-informed neural network surrogate model (hierarchical PINN) with optimization algorithms to iteratively generate power converter modulation designs, achieving a 63.2% reduction in MAE, a 33× speedup in design time, and usability validated by 20 domain experts.

## Background & Motivation
**State of the Field**: As renewable energy systems scale up, modulation design for power converters in power electronics systems (PES) becomes increasingly complex. Existing AI-assisted approaches include XGBoost surrogate models combined with differential evolution optimization and offline Q-learning training.

**Limitations of Prior Work**: (a) Training is data-intensive, requiring large-scale simulation or hardware experimental data collection; (b) Computation is costly, with high energy consumption for large model training and inference; (c) Black-box opacity severely limits industrial adoption; (d) Methods are tailored to specific modulation strategies or preset objectives with poor scalability; (e) Significant manual involvement is required throughout the design process.

**Root Cause**: Industrial modulation design must simultaneously satisfy explainability, scalability, and low human intervention requirements, yet existing AI methods fall short on all three fronts.

**Paper Goals**: Given the operating conditions, modulation strategy, and performance objectives of a power converter, automatically generate optimal modulation parameters in a process that is explainable, easy to configure, and low in data requirements.

**Starting Point**: An LLM serves as the planner to handle natural-language requirements, while a physics-informed neural network acts as an efficient surrogate model to replace costly simulation or hardware experiments.

**Core Idea**: The LLM handles requirement understanding and workflow planning; the PINN handles physics modeling and performance prediction; the optimization algorithm handles parameter search — the three components collaborate in a division of labor to realize end-to-end automated design.

## Method

### Overall Architecture
The front end consists of a chat interface and a GPT-4 reasoning engine; the back end consists of an optimization algorithm and a PINN surrogate model. The workflow proceeds as follows: an engineer submits design requirements via chat → the LLM planner parses the requirements and generates design specifications → the planner invokes the back-end toolset → the surrogate model predicts performance metrics while the optimization algorithm searches for optimal parameters (iterative loop) → final results are presented with visualization.

### Key Designs

1. **Hierarchical Physics-Informed Surrogate Model (Hierarchical PINN)**:

    - **Function**: Predicts key waveforms and performance metrics of power converters with high accuracy using very few data samples.
    - **Mechanism**: A two-level hierarchical structure — ModNet (switch-level modeling) learns the non-ideal behavior of semiconductor switches (oscillations and overshoots caused by parasitic parameters) and outputs AC-side voltages $v_p, v_s$; CirNet (system-level modeling) learns circuit physical dynamics and outputs inductor current $i_L$ and capacitor voltages $v_{C1}, v_{C2}$. Both are built on LN-GRU sequential networks; the training loss combines a physics-informed loss $l_p$ (differential equation constraints embedding Kirchhoff's, Faraday's, and Gauss's laws, Equations 3–5) and an experimental data loss $l_d$, with total loss $l_{Cir} = \lambda_d l_d + \lambda_p l_p$.
    - **Design Motivation**: The hierarchical structure decouples complex switching behavior from circuit physics. ModNet captures high-frequency oscillations induced by non-ideal switches (which conventional networks struggle to learn), and CirNet builds upon this to predict system-level performance. The physical constraints enable the PINN to learn from as few as 10 samples.

2. **LLM Planner**:

    - **Function**: Collects design requirements through a natural language interface and coordinates back-end tool execution.
    - **Mechanism**: GPT-4 serves as the reasoning engine and gathers and validates design specifications (rated power, input/output voltage, modulation strategy selection, performance objective priorities) through multi-turn dialogue. Once confirmed, it autonomously invokes the back-end toolset.
    - **Design Motivation**: Traditional design requires engineers to manually compute or programmatically invoke simulations. The LLM planner enables non-expert users to complete the design through natural language while providing textual explanations and visualization outputs to enhance explainability.

3. **Optimization Engine**:

    - **Function**: Iteratively searches for optimal modulation parameters.
    - **Mechanism**: The optimization algorithm passes operating conditions and candidate modulation parameters to the PINN surrogate model to obtain performance evaluations, updates the search direction based on the results, and iterates until convergence.
    - **Design Motivation**: The surrogate model's inference speed far exceeds that of simulation or hardware experiments, making iterative optimization feasible; the entire toolset can be seamlessly extended with new surrogate models.

### Loss & Training
- ModNet training: physics-informed loss (switching synchronization constraints) + experimental data loss (waveforms acquired from oscilloscopes).
- CirNet training (Equation 7): physics-informed loss (embedded differential equation constraints) + experimental data loss (inductor current measurements), $l_{Cir} = \lambda_d l_d + \lambda_p l_p$.
- Hyperparameter search: CirNet optimal configuration is 2 layers with 32 hidden neurons (Figure 4, validation MAE = 0.235).

## Key Experimental Results

### Main Results (Low-Data Regime: Only 10 Training Samples)

| Model | Train MAE | Test MAE | p-value |
|-------|-----------|----------|---------|
| SVR | 1.984 | 2.286 | 1.16E-10* |
| LSTM | 1.164 | 1.560 | 7.23E-8* |
| GRU | 1.287 | 1.604 | 6.35E-10* |
| LN-GRU | 1.101 | 1.491 | — |
| CirNet (w/o ModNet) | — | ~0.8 | — |
| **PHIA (ModNet+CirNet)** | — | **Best** | — |

- PHIA's normalized MAE is 63.2% lower than the second-best benchmark in the low-data regime and 23.7% lower in the high-data regime, both statistically significant.
- User study: 20 domain experts validated that PHIA reduces design time by more than 33× compared to traditional methods.

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Full PHIA | ModNet + CirNet hierarchical structure; best performance |
| CirNet only | ModNet removed; fails to capture oscillations from non-ideal switching behavior |
| Other sequential models | LSTM / GRU / TCN / Transformer all significantly inferior to PHIA |

### Key Findings
- The hierarchical PINN structure is critical to performance: ModNet provides CirNet with inputs that account for non-ideal switching behavior, which is essential for accurate predictions under extreme operating conditions.
- Physical constraints enable effective surrogate model training with only 10 samples, whereas conventional ML methods require substantially more data.
- The sequential nature of LN-GRU aligns well with the modulation waveform prediction task, outperforming Transformer variants (TST/TSiTPlus).
- Hyperparameter insensitivity: CirNet performance is stable around 2 layers and 32 hidden units.

## Highlights & Insights
- The **collaborative architecture of PINN + LLM Agent** offers substantial industrial value: the LLM handles natural language interaction and workflow orchestration, while the PINN handles physics modeling and performance prediction, leveraging the strengths of each. This architecture is transferable to other engineering design automation scenarios, such as chip design and structural engineering.
- **Extremely low data requirements**: only 10 hardware experimental samples are needed to train the surrogate model, owing to the embedding of physical priors — a critical advantage in industrial settings where data acquisition is expensive.
- **Explainability by design**: textual explanations and visualization outputs delivered through the chat interface address industrial concerns over black-box systems.

## Limitations & Future Work
- Validation is limited to a single converter topology (Dual Active Bridge, DAB); applicability to more complex multi-topology systems remains to be verified.
- The LLM planner relies on the GPT-4 API, introducing latency and cost concerns; deployment of locally hosted smaller models is a potential remedy.
- The differential equations in the physics-informed loss must be manually derived by domain experts, limiting rapid adaptation to novel converter designs.
- The specific choice of optimization algorithm and convergence guarantees are not discussed in detail.
- System robustness to hardware faults (e.g., sensor failures, component aging) has not been evaluated.
- The reliability of the LLM planner in safety-critical real-time scenarios requires more rigorous validation — failures in power electronics systems can cause hardware damage.
- The user study sample size (20 participants) is relatively small, and the robustness of the statistical conclusions could be further strengthened.

## Related Work & Insights
- **vs. XGBoost + DE methods**: These require large amounts of simulation data for surrogate model training and lack explainability; PHIA reduces data requirements via PINN and enhances explainability via the LLM.
- **vs. Q-learning methods**: These require offline training and are tailored to specific modulation strategies; PHIA achieves flexible adaptation to multiple strategies through LLM-based planning.
- Insight: The collaborative paradigm of LLM Agent + physics-based simulator (PINN) is generalizable to other industrial control domains, such as chemical process control and robotic motion planning.

## Rating
- Novelty: ⭐⭐⭐⭐ First LLM agent system in the power electronics domain; the PINN + LLM collaborative architecture is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes a 20-expert user study and statistical significance testing.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with sufficient engineering detail.
- Value: ⭐⭐⭐⭐⭐ Highly practical for industrial deployment; the architecture is broadly transferable.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] AutoGLM: Autonomous Foundation Agents for GUIs](autoglm_autonomous_foundation_agents_for_guis.md)
- [\[AAAI 2026\] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search](agentswift_efficient_llm_agent_design_via_value-guided_hierarchical_search.md)
- [\[ICLR 2026\] PhyScensis: Physics-Augmented LLM Agents for Complex Physical Scene Arrangement](../../ICLR2026/llm_agent/physcensis_physics-augmented_llm_agents_for_complex_physical_scene_arrangement.md)
- [\[AAAI 2026\] Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)
- [\[ACL 2026\] Bayesian Social Deduction with Graph-Informed Language Models](../../ACL2026/llm_agent/bayesian_social_deduction_with_graph-informed_language_models.md)

<!-- RELATED:END -->
