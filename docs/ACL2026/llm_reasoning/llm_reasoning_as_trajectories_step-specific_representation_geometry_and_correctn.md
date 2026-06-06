---
title: >-
  [Paper Note] LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals
description: >-
  [ACL 2026][LLM Reasoning][chain-of-thought] This paper conceptualizes LLM chain-of-thought reasoning as a geometric trajectory within the representation space. It discovers that (a) each reasoning step occupies a linearl…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "chain-of-thought"
  - "representation trajectories"
  - "linear separability"
  - "correctness prediction"
  - "activation steering"
date: 2026-05-08
content_hash: 33b61cd4964bd4c7
---

# LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals

**Conference**: ACL 2026  
**arXiv**: [2604.05655](https://arxiv.org/abs/2604.05655)  
**Code**: https://github.com/slhleosun/reasoning-trajectory  
**Area**: LLM Reasoning / Interpretability / Representation Geometry  
**Keywords**: chain-of-thought, representation trajectories, linear separability, correctness prediction, activation steering

## TL;DR
This paper conceptualizes LLM chain-of-thought reasoning as a geometric trajectory within the representation space. It discovers that (a) each reasoning step occupies a linearly separable subspace that becomes clearer in deeper layers, and (b) correct and incorrect trajectories overlap in early stages but diverge systematically later. Consequently, the final correctness can be predicted with an ROC-AUC of 0.87 before the answer is output. Based on this, "trajectory steering" is proposed for reasoning correction and length control.

## Background & Motivation
**Background**: LLM CoT reasoning is typically treated as a "text-sequential generation" black box. Interpretability research mostly resides at the level of identifying specific attention heads for certain functions or finding "concept directions," with few works analyzing the entire reasoning process as a continuous geometric object.

**Limitations of Prior Work**: (a) It is unclear what stage the model has "thought" through or why certain problems are solved incorrectly; (b) Existing inference-time interventions (test-time scaling via token injection, fixed steering vectors) are mostly triggered unconditionally, lacking criteria for "when to intervene," which leads to unstable effects; (c) It remains unclear what reasoning training (e.g., distillation from DeepSeek-R1) actually changes in the model.

**Key Challenge**: Reasoning is a **time-series** process, yet the vast majority of representation analysis methods perform probing only on single tokens or single layers, discarding the crucial signal of "how steps transition."

**Goal**: To model the internal states of the reasoning process as a trajectory $\mathbf{h}_{t(\text{Step }1)-1}^{(\ell)},\mathbf{h}_{t(\text{Step }2)-1}^{(\ell)},\dots,\mathbf{h}_{t(\text{term})-1}^{(\ell)}$, and answer: Do steps correspond to distinct subspaces? Are correct and incorrect trajectories distinguishable? Can midway intervention be performed based on this?

**Key Insight**: Under a fixed zero-shot CoT template, each step is naturally preceded by a "Step k:" marker. By extracting the hidden state immediately before this marker, one obtains an internal snapshot of "having completed step k and about to enter step k+1," avoiding contamination from surface formatting tokens.

**Core Idea**: Geometrize CoT using "pre-step activation" sequences, and use linear probing and distance analysis to reveal step separability and correctness divergence. Finally, perform low-rank steering correction based on an "ideal trajectory."

## Method

### Overall Architecture
The study is divided into three parts, corresponding to three discoveries:
1. **Geometric Structure Analysis**: Uses t-SNE visualization and linear probing (one-vs-rest classification) to measure whether "pre-step activations" occupy step-specific subspaces, comparing Base / Instruct / R1-Distill (Llama-3.1-8B variants) to analyze the impact of training paradigms.
2. **Correctness Signal Analysis**: Groups trajectories by final answer correctness and calculates Euclidean/cosine distances step-by-step to observe divergence points. A linear classifier is trained on "late-stage step features" to predict final correctness.
3. **Trajectory Steering Intervention**: Constructs an "ideal trajectory" (the mean path of correct samples). During runtime, if the current trajectory deviates beyond a threshold, a low-rank steering update is triggered to pull it back toward the ideal direction. The same approach is used to control reasoning length (pushing toward/away from termination subspaces).

Data consists of GSM8K (7,473 train / 1,319 test) and MATH-500, with prompts forcing each step to start with `Step k:` and answers to be marked with `####`.

### Key Designs

1. **Pre-step Activation Extraction and Linear Separability Measurement**:
    - **Function**: Extracts hidden states $\mathbf{h}_{t(\text{Step }k)-1}^{(\ell)}$ at all layers just before each `Step k:` marker, then uses one-vs-rest binary probes to test if each step occupies an independent linear subspace.
    - **Mechanism**: A linear classifier $\hat y = \mathrm{sign}(w^\top h)$ is trained for each step $k$, where positives are activations exactly at the completion of step $k$ and negatives are from other steps. Cross-step and cross-model variant transfers are performed to rule out "surface format" hypotheses.
    - **Design Motivation**: Taking activations before the marker captures the "accumulated reasoning state" rather than formatting info. Linear separability is a standard criterion in representation learning for "information directly readable by downstream components."

2. **Trajectory Distance Variance and Midway Correctness Prediction**:
    - **Function**: Calculates distances $d(\mathbf{h}_{t(a)-1}^{(L)}, \mathbf{h}_{t(b)-1}^{(L)})$ (Euclidean for magnitude, cosine for direction) between adjacent steps and uses $\Delta(\text{Incorrect}-\text{Correct})$ to identify when trajectories diverge.
    - **Mechanism**: Early steps are statistically indistinguishable (95% CI overlap), while systematic divergence begins in later stages (after approx. step 4). Based on this, a linear classifier predicts final correctness from late-stage activations and distance features, reaching a peak AUC of 0.87 at layer 29.
    - **Design Motivation**: Shifting the "correctness signal" from the final token to the midway stages allows interventions before the answer is output, saving test-time compute and preventing error propagation.

3. **Trajectory Steering: Adaptive Low-Rank Correction Based on Ideal Trajectories**:
    - **Function**: Real-time tracking of the deviation between the current trajectory and the "correct trajectory centroid." If it exceeds a threshold, a low-rank steering update $\mathbf{h}' = \mathbf{h} + \alpha U V^\top \mathbf{h}$ is applied to the current activation.
    - **Mechanism**: The ideal trajectory is the mean path of activations from correct samples. The steering matrix is a low-rank approximation derived from the Singular Value Decomposition (SVD) of the correct-incorrect difference vectors. Similarly, pushing activations toward/away from the termination subspace controls CoT length.
    - **Design Motivation**: Compared to unconditional test-time scaling (e.g., injecting "Wait" tokens), the gating and trajectory deviation provide objective criteria for "when and how much to intervene," ensuring perturbations occur only on incorrect trajectories.

### Loss & Training
The model itself is not trained. Only linear probes and classifiers (standard Logistic Regression / SVM) are trained. Steering matrices are obtained via closed-form SVD of difference vectors, requiring no gradients.

## Key Experimental Results

### Main Results (Linear probe step recognition accuracy, from Figure 1b / Table 1)

| Probe From | Eval On | Step 2 | Step 3 | Step 4 | Step 5 | Final Ans Marker |
|------------|---------|--------|--------|--------|--------|------------------|
| Instruct | R1-Distill | 0.99 (L18) | 0.93 (L19) | 0.87 (L12) | 0.91 (L18) | 0.87 (L23) |
| Instruct | Base | 1.00 (L12) | 0.97 (L08) | 0.95 (L08) | 0.93 (L18) | 0.97 (L21) |
| R1-Distill | Instruct | 1.00 (L03) | 0.92 (L08) | 0.88 (L07) | 0.93 (L27) | 0.98 (L19) |
| R1-Distill | Base | 1.00 (L06) | 0.94 (L04) | 0.89 (L30) | 0.91 (L08) | 0.96 (L19) |
| Base | Instruct | 1.00 (L21) | 0.98 (L18) | 0.97 (L18) | 0.97 (L23) | 1.00 (L31) |
| Base | R1-Distill | 0.99 (L12) | 0.91 (L18) | 0.90 (L18) | 0.92 (L17) | 0.94 (L02) |

Cross-model transfer scores are almost all >0.90, indicating that step-specific linear structures are shared across Base/Instruct/R1-Distill training paradigms. The main difference in R1-Distill is that the termination subspace forms at much shallower layers (0.99 at layer 0, compared to 0.80 for Instruct).

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| Full (Pre-step act. + Real labels) | Step 2 probe 1.00, AUC 0.87 | Full setup |
| Randomly shuffled step labels (Control) | Avg. 0.59 ± 0.04 | Near chance, proves geometry is not probe overfitting |
| Freeform prompt (No `Step k:` template) | Best-layer per step ≥ 0.84 | Model spontaneously uses `Step k:` in 64.5% of samples; others transferable at bounds |
| Predict with Step 1 features only | AUC ≈ 0.63 | Almost no correctness signal in early steps |
| Predict with last few steps features | AUC 0.83 / Peak 0.87 (L29) | Strong divergence in late steps |

### Key Findings
- Step-specific structures already exist in the Base model. Reasoning training merely highlights the termination subspace earlier; it does not "create new structures." This counter-intuitive finding suggests that features often attributed to SFT/RLHF are inherently present.
- Correct and incorrect trajectories are statistically identical in early steps; divergence occurs after step 4. This implies "early exit" is a poor strategy for reasoning tasks, as the first 4 steps are generally necessary.
- Under freeform prompts, the model spontaneously selects the `Step k:` format in 64.5% of cases. Probes from fixed templates transfer to freeform with >0.84 accuracy, proving the reasoning structure is real and not a prompt artifact.
- Gating interventions with a mid-reasoning predictor is more stable than unconditional token injection or indiscriminate steering on GSM8K and MATH-500.

## Highlights & Insights
- Reformulating CoT from a discrete token sequence to a "continuous trajectory in representation space" is a paradigm-level shift, providing a new unit of study (trajectory vs. token/direction) for the interpretability community.
- The finding that "training only changes when geometry emerges, not its form" prompts a re-evaluation of reasoning training—its primary benefit may be "termination calibration" rather than the acquisition of "new capabilities."
- The mid-stage correctness predictor (AUC 0.87) can be directly used as a reward proxy for RL, bypassing the sparsity issues of outcome-based rewards.
- The "ideal trajectory + low-rank steering" framework provides a clean geometric abstraction for inference-time intervention, transferable to code agents, planning, and other tasks requiring midway supervision.

## Limitations & Future Work
- Experiments were primarily conducted on Llama-3.1-8B and math tasks (GSM8K/MATH-500); applicability to other architectures (Qwen/Mistral) or domains (commonsense, code) remains to be confirmed.
- Defining "ideal trajectories" via centroids ignores the multi-modal nature of correct reasoning paths, which may limit effectiveness for open-ended problems.
- The actual accuracy improvement from midway prediction and steering was characterized as "modest but consistent," indicating loss during the transition from signal to action.
- Activation extraction currently relies on explicit `Step k:` markers. For tasks without clear step boundaries (NLG, dialogue), the marking strategy needs redesign, which remains a bottleneck for broader application.

## Related Work & Insights
- **vs. Lanham et al. (CoT faithfulness)**: While they measured faithfulness behaviorally (by deleting steps), this work provides a geometric explanation from the representation perspective.
- **vs. Park et al. (Linear Representation Hypothesis)**: This work extends "concepts are linear directions" to "reasoning stages are linear subspaces," broadening the scope of LRH.
- **vs. Turner et al. (Activation Steering)**: Traditional steering adds fixed vectors unconditionally; this work uses trajectory deviation as a gate for more precise, low-side-effect intervention.
- **vs. Muennighoff et al. (s1 / Test-time scaling)**: They artificially extend reasoning by injecting "Wait" tokens; this work achieves the same goal through termination subspace steering, which is continuous, controllable, and reversible.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Reasoning = Geometric trajectory" is a powerful new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid cross-paradigm and freeform controls, though limited to one model family and math tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with findings well-mapped to sections.
- Value: ⭐⭐⭐⭐ Opens new interfaces for interpretability, test-time intervention, and reward modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] On the Step Length Confounding in LLM Reasoning Data Selection](on_the_step_length_confounding_in_llm_reasoning_data_selection.md)
- [\[ACL 2026\] Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment](which_reasoning_trajectories_teach_students_to_reason_better_a_simple_metric_of_.md)
- [\[ACL 2026\] Reasoning Fails Where Step Flow Breaks](reasoning_fails_where_step_flow_breaks.md)
- [\[ICLR 2026\] The Path of Least Resistance: Guiding LLM Reasoning Trajectories with Prefix Consensus](../../ICLR2026/llm_reasoning/the_path_of_least_resistance_guiding_llm_reasoning_trajectories_with_prefix_cons.md)
- [\[ACL 2026\] Revisiting the Uniform Information Density Hypothesis in LLM Reasoning](revisiting_the_uniform_information_density_hypothesis_in_llm_reasoning.md)

</div>

<!-- RELATED:END -->
