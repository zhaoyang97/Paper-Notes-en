---
title: >-
  [Paper Note] AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling
description: >-
  [NeurIPS 2025][Robotics][Theory of Mind] AutoToM achieves fully automated model-based Theory of Mind inference—without requiring manual agent model specification—by automatically proposing Bayesian network structures and executing Bayesian inverse planning. Through uncertainty-driven iterative model refinement (adding mental variables or extending time steps), it achieves an average accuracy of 82.43% across 5 ToM benchmarks, surpassing SOTA models such as GPT-4o (63.39%) and o3-mini (73.94%).
tags:
  - NeurIPS 2025
  - Robotics
  - Theory of Mind
  - Bayesian inverse planning
  - automated agent modeling
  - mental inference
  - LLM
date: 2026-05-08
content_hash: c7fcf1a58221c2f0
---

# AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2502.15676](https://arxiv.org/abs/2502.15676)
**Code**: Available
**Area**: Theory of Mind / LLM Agent
**Keywords**: Theory of Mind, Bayesian inverse planning, automated agent modeling, mental inference, LLM

## TL;DR
AutoToM achieves fully automated model-based Theory of Mind inference—without requiring manual agent model specification—by automatically proposing Bayesian network structures and executing Bayesian inverse planning. Through uncertainty-driven iterative model refinement (adding mental variables or extending time steps), it achieves an average accuracy of 82.43% across 5 ToM benchmarks, surpassing SOTA models such as GPT-4o (63.39%) and o3-mini (73.94%).

## Background & Motivation

**Background**: Theory of Mind (ToM)—the ability to understand others' mental states (goals, beliefs, intentions)—is a cornerstone of social intelligence. Machine ToM research has two main paradigms: (a) directly prompting LLMs to perform inference (SimToM, SymbolicToM, etc.), which is flexible but prone to systematic errors in complex scenarios (especially long-context and multi-agent recursive reasoning); (b) model-based Bayesian inverse planning (BIP), which constructs generative models of agents and inversely infers mental states, offering robustness but requiring manually defined agent models (including the set of mental variables and causal structure), thus generalizing poorly.

**Limitations of Prior Work**: Pioneer works such as BIP-ALM and LIMP that integrate BIP with LLMs improve robustness, but still require manual specification of: (a) which mental variables are needed (goals, beliefs, observations, etc.); (b) causal relationships among variables (choice of MDP/POMDP/I-POMDP structure); and (c) which time steps to consider. These hand-crafted designs limit applicability to specific domains and preclude handling open-ended ToM problems.

**Key Challenge**: LLMs are flexible but not robust—even large reasoning models like o3-mini make systematic errors in complex ToM (long-context forgetting, recursive reasoning collapse); BIP is robust but not flexible—requiring human-designed agent models for each domain. **Core Idea**: Have LLMs automatically discover appropriate agent model structures, then perform automated Bayesian inference over those models—combining the flexibility of model discovery with the robustness of Bayesian inference to yield scalable, open-ended machine ToM.

## Method

### Overall Architecture
AutoToM comprises two core components forming a self-improvement loop: (1) Automated Bayesian Inverse Planning—executing Bayesian inference on a given agent model using an LLM as the computational backend; and (2) Automated Agent Model Discovery—automatically proposing and refining agent models based on inference uncertainty. The pipeline proceeds as: information extraction → initial model proposal → automated BIP → model utility evaluation → model refinement if utility is insufficient → re-inference → iteration until sufficient confidence is achieved.

### Key Designs

1. **Automated Bayesian Inverse Planning (Automated BIP)**:

    - **Function**: Executes complete inference over any given agent model (Bayesian network).
    - **Mechanism**: A two-stage approach—(a) Hypothesis sampling: an LLM generates a small set of high-quality candidate values for each latent mental variable (analogous to amortized inference), conditioned on the question and observable variables, followed by hypothesis reduction to eliminate implausible candidates (via local conditional probability evaluation); (b) Bayesian inference: an LLM estimates each local conditional probability $P(\text{child}|\text{parents})$ in the Bayesian network, then computes the posterior $P(q|X)$ over the target variable by explicit marginalization over the joint distribution.
    - **Design Motivation**: Unlike BIP-ALM and LIMP, which assume fixed model structures and manually defined variable representations, AutoToM's BIP applies to arbitrary graph structures and variable representations. It supports arbitrary-order recursive reasoning (via nested belief modeling in I-POMDPs) without requiring domain-specific implementations.

2. **Automated Agent Model Discovery**:

    - **Function**: Automatically constructs the most appropriate agent model for the current ToM problem, eliminating the bottleneck of manual model design.
    - **Mechanism**: A model $M = (V^{t_s:t}, X^{t_s:t})$ is uniquely defined by its set of mental variables and observable variables. Model utility is defined as $U(M, q) = R(M, q) - C(M)$, where $R = -H(P(q|X))$ (negative entropy of the inference result, i.e., confidence) and $C = \alpha|M|$ (complexity penalty). Three sub-modules: (a) Information extraction—using an LLM to extract observable variables (states, actions, utterances) from context and arrange them along a timeline; (b) Initial model proposal—proposing a minimal-complexity model containing only the variables strictly necessary to answer the question, starting from the last time step; (c) Iterative refinement—variable adjustment (introducing new mental variables such as belief/observation/interactive state) and time-step adjustment (extending further back in time), selecting the adjustment with the highest utility gain at each iteration.
    - **Design Motivation**: Uncertainty-driven design—model complexity is increased only when inference confidence is insufficient, avoiding the dual pitfalls of over-modeling (wasted computation) and under-modeling (insufficient accuracy). Constraining the search to MDP/POMDP/I-POMDP variable type spaces ensures that models can account for agent behavior.

3. **Unified Formalization and Scalable Design**:

    - **Function**: Provides a domain-agnostic framework for ToM inference.
    - **Mechanism**: BIP is uniformly formalized as inference over $P(V^{t_s:t}|X^{t_s:t})$, encompassing MDP (with goals, full observability), POMDP (partial observability + belief maintenance), and I-POMDP (multi-agent recursive reasoning) as different configurations of variable sets $V$ and $X$.
    - **Design Motivation**: Prior methods (BIP-ALM, LIMP) each provide customized implementations for specific model types, lacking cross-type generalization. The unified formalization enables a single inference engine to handle all model variants.

### Loss & Training
AutoToM requires no trainable parameters—it relies entirely on LLM in-context inference. Key hyperparameters include the model utility threshold $U_{\min}$ (determining when to stop model refinement) and the complexity weight $\alpha$.

## Key Experimental Results

### Main Results (Average Accuracy across 5 ToM Benchmarks)

| Method | ToMi | BigToM | MMToM-QA | MuMA-ToM | Hi-ToM | Avg. |
|--------|------|--------|----------|----------|--------|------|
| GPT-4o | 77.0 | 82.4 | 44.0 | 63.6 | 50.0 | 63.4 |
| o3-mini-high | 73.1 | 86.9 | 64.7 | 70.0 | 75.0 | 73.9 |
| Gemini 2.0 Flash Thinking | 78.0 | 82.8 | 54.0 | 82.6 | 73.5 | 74.2 |
| DeepSeek-R1 | 89.4 | 86.3 | 49.7 | 63.4 | 56.5 | 69.1 |
| BIP-ALM | 55.6 | 50.3 | 56.2 | 33.9 | 14.5 | 42.1 |
| LIMP | 44.6 | 61.7 | 55.3 | 76.6 | 6.5 | 48.9 |
| **AutoToM (GPT-4o)** | **88.3** | **86.9** | **83.0** | **81.4** | **72.5** | **82.4** |

### Ablation Study

| Configuration | Avg. Accuracy | Relative Computation | Notes |
|---------------|--------------|----------------------|-------|
| Full AutoToM | 82.4 | 1.0× | Best performance |
| w/o hypothesis reduction | ~80 | ~1.3× | Slight accuracy drop + increased computation |
| w/ fixed POMDP | ~78 | ~1.1× | Inflexibility causes over-modeling in some scenarios |
| w/o variable adjustment | ~76 | ~0.8× | Cannot adapt to scenarios requiring belief/observation |
| w/ last timestep only | ~74 | ~0.6× | Loss of historical context |
| w/ all timesteps | ~79 | ~1.5× | Unnecessary computational overhead |

### Key Findings
- AutoToM with GPT-4o as backbone (82.4%) substantially outperforms GPT-4o alone (63.4%)—structured inference > pure LLM reasoning.
- The largest gain is observed on the most challenging benchmark, MMToM-QA (long-context + multimodal): 83.0% vs. GPT-4o 44.0% and o3-mini 64.7%.
- AutoToM's advantage grows as context length, number of agents, and recursive reasoning depth increase (Figure 4), while large reasoning models exhibit significant performance fluctuations.
- Consistent improvements are observed across different LLM backends (Qwen3-235B, DeepSeek-V3, Gemini-2.5-Flash), each surpassing the corresponding base LLM, validating the backend-agnostic nature of the framework.
- Statistical reliability: on MMToM-QA, three runs yield a mean of 82.56% ± 0.45%.

## Highlights & Insights
- **Automated model discovery is the central contribution**—elevating model-based ToM from "requiring cognitive scientists to manually design models" to "a fully automated system," genuinely enabling open-ended ToM inference.
- The **uncertainty-driven model expansion** design is elegant—starting minimally and expanding on demand strikes an adaptive balance between efficiency and accuracy, essentially performing adaptive search over model complexity.
- **LLMs as probabilistic inference backends** rather than direct reasoners—this role assignment is a key insight. LLMs are ill-suited for systematic reasoning but excel at estimating local conditional probabilities (assessing the likelihood of a variable taking a specific value given a concrete scenario).
- The system produces human-like confidence estimates (not just answers, but degrees of certainty), which is crucial for downstream tasks such as embodied assistance.

## Limitations & Future Work
- The quality of model discovery remains constrained by the commonsense reasoning capabilities of the LLM backend—if the LLM fails to correctly identify the necessary mental variables, the resulting model may be inappropriate.
- The accuracy of hypothesis sampling and local conditional probability estimation depends on the LLM's contextual understanding, and may degrade for unconventional or counter-intuitive agent behaviors.
- Computational cost for multi-agent higher-order recursive reasoning grows exponentially with recursive depth.
- The current model discovery is restricted to MDP/POMDP/I-POMDP variable type spaces, potentially missing non-standard mental variables.

## Related Work & Insights
- **LLM + probabilistic inference paradigm**: AutoToM demonstrates the substantial potential of using LLMs as probabilistic inference backends (estimating likelihoods and generating hypotheses) rather than end-to-end reasoners, a paradigm generalizable to other domains requiring structured reasoning.
- **Automated modeling with LLMs**: This work echoes the approaches of Li et al. on automated statistical model construction and Wang et al. on hypothesis generation with programmatic verification, but is the first to apply such ideas to BIP/ToM.
- **Human-like AI**: The confidence estimates produced by AutoToM align with human behavioral experimental data, suggesting that its inference mechanism may capture certain computational characteristics of human ToM.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Fully automated model-based ToM is a breakthrough contribution; the combination of automated model discovery and automated BIP is unprecedented.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 benchmarks + cognitive experiments + embodied tasks + multiple LLM backends + ablations + statistical reliability tests.
- Writing Quality: ⭐⭐⭐⭐⭐ Unified formalization is clear, figures are excellent, and experimental analysis is thorough.
- Value: ⭐⭐⭐⭐⭐ Far-reaching impact on socially intelligent AI and human-computer interaction; the framework exhibits strong generality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)
- [\[AAAI 2026\] Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects](../../AAAI2026/robotics/causal_inference_under_threshold_manipulation_bayesian_mixtu.md)
- [\[NeurIPS 2025\] Can Agents Fix Agent Issues?](can_agents_fix_agent_issues.md)
- [\[NeurIPS 2025\] MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents](mip_against_agent_malicious_image_patches_hijacking_multimod.md)
- [\[NeurIPS 2025\] FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model](falcon_fine-grained_activation_manipulation_by_contrastive_orthogonal_unalignmen.md)

</div>

<!-- RELATED:END -->
