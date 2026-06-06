---
title: >-
  [Paper Note] Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts
description: >-
  [ICML 2026][Multimodal VLM][Multimodal Representation] This paper proposes the S3 framework, which decomposes multimodal representations into concept-level experts via MoE (**Specialization**)…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Multimodal Representation"
  - "MoE"
  - "Task-Sufficiency"
  - "Information-Minimality"
  - "Inference-time Pruning"
date: 2026-05-08
content_hash: b832a16e2a016dc5
---

# Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts

**Conference**: ICML 2026  
**arXiv**: [2605.03348](https://arxiv.org/abs/2605.03348)  
**Code**: None  
**Area**: Multimodal VLM / MoE / Representation Learning  
**Keywords**: Multimodal Representation, MoE, Task-Sufficiency, Information-Minimality, Inference-time Pruning

## TL;DR
This paper proposes the S3 framework, which decomposes multimodal representations into concept-level experts via MoE (**Specialization**), activates relevant experts through task-based routing (**Selection**), and prunes low-contribution paths guided by routing scores during inference (**Sparsification**). Experiments on four MultiBench benchmarks reveal an "inverted U-shaped curve" where performance peaks at intermediate sparsity, offering a third multimodal representation paradigm beyond Contrastive Learning and InfoMax.

## Background & Motivation

**Background**: There are two mainstream paradigms in multimodal representation learning: Contrastive Learning (e.g., CLIP, AudioCLIP), which maps paired modalities to a shared space to maximize cross-modal mutual information; and InfoMax-style methods (e.g., FOCAL, DisentangledSSL, JointOpt), which aim to preserve both shared and modality-unique information. Both aim to learn a "fixed monolithic embedding."

**Limitations of Prior Work**: Contrastive learning has a theoretical upper bound—the mutual information of its optimal solution depends only on the entropy of the shared factor $H(X_S)$ (Proposition 2.3). Once a task relies on modality-unique factors $X_U^m$, contrastive representations cannot achieve Bayes optimality. While InfoMax can achieve task-sufficiency, it also maximizes $I(Z^m;X^m|Y)$, retaining a large amount of task-irrelevant information, which violates the InfoMin principle and hinders downstream classification.

**Key Challenge**: A single monolithic embedding simultaneously bears three conflicting demands: "alignment," "preserving differences," and "adapting to task variations." The combination of relevant factors for samples and tasks is highly variable, yet representations are fixed and cannot be selected on demand.

**Goal**: Construct multimodal representations that are both **Task-Sufficient** ($I(Z_Y^{1*},Z_Y^{2*};Y)=I(X^1,X^2;Y)$) and **Information-Minimal** ($I(Z_Y^{1*},Z_Y^{2*};X^1,X^2|Y)=0$), adjustable and controllable at the sample/task level.

**Key Insight**: Shift the focus from "optimizing objective functions" to "adding structural inductive biases." The representation space is explicitly decomposed into a set of concept subspaces $\mathcal{Z}=\bigoplus_{c\in\mathcal{C}}\mathcal{Z}_c$, where each subspace is implemented by an MoE expert. The same latent concept across different modalities should activate the same experts (Proposed Distributional Semantic Coherence), achieving "concept-level" rather than "instance-level" cross-modal alignment.

**Core Idea**: Reinterpret MoE as a tool for semantic specialization (rather than pure parameter scaling). A three-stage pipeline—Specialization $\to$ Selection $\to$ Sparsification—is used to address how to construct semantic expert spaces, how to activate relevant experts per task, and how to prune redundant paths during inference, achieving structurally controllable Task-Sufficient + Information-Minimal multimodal representations.

## Method

### Overall Architecture
MoE encoders $f^1, f^2$ are used for two modalities respectively. Each MoE layer contains $N_{\mathrm{expert}}=\chi \cdot \rho$ experts (granularity $\chi$ + expansion ratio $\rho$). The router $g$ uses top-$k$ softmax to determine which experts each token traverses: $g(\mathbf{x})=\mathrm{TOP}_k(\mathrm{softmax}(\mathbf{W}_g\mathbf{x}))$, outputting $\mathrm{MoE}(\mathbf{x})=\sum_i g(\mathbf{x})_i e_i(\mathbf{x})$. The process involves three sequential stages: Stage 1 involves SSL pre-training of the encoder and router; Stage 2 involves fine-tuning only the router; Stage 3 involves inference-time pruning.

### Key Designs

1.  **Specialization: Pre-training Concept-level Expert Spaces**:
    *   **Function**: Anchors each expert to a semantic concept and ensures the same concept across modalities is activated in aligned subspaces (DSC constraint).
    *   **Mechanism**: The objective is $\max_{f^1,f^2}[I(Z^1;X^1)+I(Z^2;X^2)]$ s.t. DSC (Proposition 3.4: For all shareable concepts $c$, $p(\pi_c(Z^1)|c\in C^1)=p(\pi_c(Z^2)|c\in C^2)$). Mutual information is approximated via the InfoNCE lower bound, and the loss consists of three parts: intra-modal $\mathcal{L}_{\mathrm{rep}}=\tfrac12(\mathcal{L}_{\mathrm{InfoNCE}}^{[1\to1]}+\mathcal{L}_{\mathrm{InfoNCE}}^{[2\to2]})$ for diversity; cross-modal $\mathcal{L}_{\mathrm{dsc}}=\tfrac12(\mathcal{L}_{\mathrm{InfoNCE}}^{[1\to2]}+\mathcal{L}_{\mathrm{InfoNCE}}^{[2\to1]})$ to implicitly align concept activation patterns; and an auxiliary routing loss $\mathcal{L}_{\mathrm{aux}}$ to prevent expert collapse and encourage balanced, confident activation.
    *   **Design Motivation**: Pure InfoNCE is instance-level, but its contrastive signal implicitly shapes the expert activation distribution, clustering synonymous concepts into the same experts. Explicit concept decomposition + DSC constraints ensure cross-modal alignment occurs at the "expert level" rather than the "feature vector level," naturally accommodating modality-unique parts (unique concepts utilize modality-exclusive experts).

2.  **Selection: Task Adaptation via Router-only Tuning**:
    *   **Function**: Freezes all experts and attention layers, fine-tuning only the router $g$ (a tiny fraction of total parameters) to activate task-relevant experts and suppress irrelevant ones.
    *   **Mechanism**: The goal $\max_g[I(Z_Y^1,Z_Y^2;Y)-\alpha\cdot I(Z_Y^1,Z_Y^2;X^1,X^2|Y)]$ simultaneously achieves Task-Sufficiency and Information-Minimality. The first term is approximated by SupCon loss—pulling samples with the same label closer (Proposition E.2 proves this is an effective lower bound for task-conditioned MI): $\mathcal{L}_{\mathrm{SupCon}}^{[m\to\bar m]}=-\mathbb{E}_{i,s\in\mathcal{S}_{y_i}}\log\frac{\exp(\langle z_i^m,z_s^{\bar m}\rangle/\tau)}{\sum_j\exp(\langle z_i^m,z_j^{\bar m}\rangle/\tau)}$. The second term $I(Z;X|Y)=\mathbb{E}_{p(x,y)}[D_{KL}(p(z|x)\|p(z|y))]$ is approximated using a vMF distribution (features are on a hypersphere after InfoNCE), simplifying to a compactness loss $\mathcal{L}_{\mathrm{Comp}}^{[m\to\bar m]}=-\mathbb{E}[\langle\mu_x^m,\hat\mu_y^{\bar m}\rangle]$, which pulls samples toward the spherical mean direction of their class.
    *   **Design Motivation**: Conventional fine-tuning updates the encoder, which disrupts the semantic expert structure learned in Stage 1. Tuning only the router strictly decouples "what is learned" from "what is used for the task"—the former is a fixed semantic basis, while the latter is a task-dependent selector. The effect is similar to prompt tuning but with a more structured goal.

3.  **Sparsification: Inference-time Pruning via Routing Scores**:
    *   **Function**: Without further training, top-$k$ routing pairs within each batch are sorted by score, retaining only a top-$p$ ratio of routing pairs while pruning the rest.
    *   **Mechanism**: After Stage 2, router scores themselves serve as "estimates of input-expert contribution to the task." Standard MoE uses fixed top-$k$ regardless of actual utility, activating unnecessary experts. The pruning process is expected to exhibit an inverted U-shaped curve: as $p$ decreases from 1, irrelevant paths are pruned first (performance increases or stabilizes) until a "sweet spot" of minimal sufficient representation (peak performance) is reached; as $p$ becomes too small, critical paths are mistakenly pruned (performance drops). Residual connections remain, so pruning a single routing path does not sever the information flow.
    *   **Design Motivation**: Extending "information minimization" from the training phase to the inference phase establishes representation compression as an inference-time knob. This allows for real-time efficiency-accuracy trade-offs based on downstream computational budgets without additional training, while providing a natural diagnostic for how many task-relevant routes actually exist.

### Loss & Training
- Stage 1: $\mathcal{L}_{\mathrm{special}}=\lambda_{\mathrm{rep}}\mathcal{L}_{\mathrm{rep}}+\lambda_{\mathrm{dsc}}\mathcal{L}_{\mathrm{dsc}}+\lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux}}$ (includes expert balance regularization).
- Stage 2: $\mathcal{L}_{\mathrm{select}}=\lambda_{\mathrm{suff}}\mathcal{L}_{\mathrm{suff}}+\lambda_{\mathrm{min}}\mathcal{L}_{\mathrm{min}}$ (excludes balance regularization, as the goal is unbalanced activation of relevant experts).
- Fair Comparison: Setting $k=\chi$ ensures the active parameters per token in the MoE are equivalent to a dense FFN (performance is not due to increased parameter count).

## Key Experimental Results

### Main Results
Linear probing accuracy on four MultiBench benchmarks (MOSEI / MOSI / UR-FUNNY / MUStARD). The table below shows the best results for S3 at different granularities $\chi$ compared to active-param equivalent baselines on MOSEI (best $p$ extracted from original paper tables):

| Dataset | Method | Best Accuracy (%) | Remarks |
| :--- | :--- | :--- | :--- |
| MOSEI | CLIP (Contrastive) | ~74.5 | shared-only |
| MOSEI | FactorCL / DisentSSL (InfoMax) | 74-76 | Preserves all information |
| MOSEI | **S3 ($\chi=8$, sweet spot)** | **77.95** | $p \approx 0.3$ |
| MOSI | InfoMax Baseline | ~63 | |
| MOSI | **S3 ($\chi=8$)** | **66.13** | $p \approx 0.6$ |
| UR-FUNNY | InfoMax Baseline | ~63 | |
| UR-FUNNY | **S3 ($\chi=4$)** | **64.74** | $p \approx 0.4$ |

S3 consistently outperforms contrastive and InfoMax baselines across all four benchmarks, with peaks occurring at medium sparsity rather than $p=1$.

### Ablation Study

| Configuration | MOSEI Accuracy (%) | Trend Shape |
| :--- | :--- | :--- |
| $\chi=2$ (Coarse) | 77.25 (Peak at $p=0.1$) | Delayed U-shape—routing ambiguity causes initial drop |
| $\chi=4$ (Medium) | 77.18 (Peak at $p=0.1$) | Smooth transition |
| $\chi=8$ (Fine) | **77.95** (Peak at $p=0.3$) | Classic inverted U-shape |
| $\chi=8, p=1.0$ (No Pruning) | 75.78 | 2 points lower than pruning peak |
| Specialization only (Skip Selection) | < 75 | Router not adapted to task |

### Key Findings
- **Granularity determines pruning curve shape**: At low granularity ($\chi=2$), each expert encapsulates multiple concepts, leading to routing ambiguity where initial pruning causes damage—improvements only appear after significant pruning to $p=0.1$ (Delayed U-shape). At high granularity ($\chi=8$), each expert specializes in one concept, router confidence is high, and pruning immediately benefits performance starting from $p=0.9$, with a sweet spot around $p=0.3$. This pattern is consistent across all benchmarks.
- **Inverted U-shape = Empirical validation of InfoMin**: Peak performance at medium sparsity provides strong evidence that task-irrelevant information hinders downstream tasks—this is a direct experimental verification of the InfoMin principle in multimodal scenarios.
- **Router accounts for < 5% of total parameters**, yet supports task adaptation—indicating that once a structural latent space is established, "which part to use" is more critical than "what to learn."
- **Robust across batch sizes**: The trend shape remains constant across batch sizes 64-512, suggesting pruning behavior is determined by structural properties rather than training details.

## Highlights & Insights
- **Paradigm shift from "loss tuning" to "structural bias"**: The authors point out that the failure of contrastive learning and InfoMax is not just due to suboptimal loss functions but the lack of structural inductive biases. This perspective shifts away from "objective function centricity" and is instructive for representation learning as a whole, including areas beyond SSL (e.g., few-shot, transfer learning).
- **Semantic Interpretation of MoE**: Reinterpreting MoE from a "parameter scaling tool" to "concept experts" is a novel, theoretically-backed perspective. Combined with the DSC concept, it provides a new mathematical language for cross-modal alignment ("expert activation distribution alignment" replacing traditional "feature alignment").
- **Inference-time Pruning Knob**: Turning Information-Minimality into a training-free inference-time hyperparameter, where the sweet spot is identified via the inverted U-curve, is highly valuable for engineering. The same model can dynamically switch efficiency/accuracy trade-offs based on downstream requirements.
- **Theoretic-Empirical Closed Loop**: The work first proves contrastive learning is strictly suboptimal when tasks depend on unique factors (Prop 2.5), then characterizes the limitations of InfoMax-determined task-irrelevant information decomposition (Eq 12), and finally echoes theoretical predictions with the inverted U-curve in S3 experiments—a complete and rare syllogistic structure.

## Limitations & Future Work
- Experiments are conducted on relatively small-scale, dual-modality (mostly text-audio/visual feature vectors) tasks like MultiBench. The stability, training cost, and convergence when scaling to large-scale vision-language (e.g., COCO/LAION) or scenarios with 3+ modalities (e.g., video-audio-text) are unknown.
- The DSC assumption that "shareable concepts activate the same expert in both modalities" might not find such natural alignment in actual multimodal representation spaces—e.g., do "timbre" in audio and "texture" in vision truly map to the same expert? There is a lack of interpretability analysis or visual evidence.
- The vMF approximation in Stage 2 requires features to be on a unit hypersphere (true after InfoNCE). However, if the backbone is changed to non-normalized outputs, the KL derivation fails.
- The "sweet spot" $p$ for Sparsification currently requires scanning on a validation set without theoretical guidance; there is no method to automatically determine $p$ across different tasks or data scales.
- Although fair comparison with FFN based on active parameters is reasonable, the extra total parameters in MoE still imply higher storage and loading costs, limiting deployment on edge devices.

## Related Work & Insights
- **vs. CLIP/ImageBind (Contrastive Learning)**: CLIP compresses all info into a single embedding, strictly limited by $H(X_S)$. S3 uses expert subspaces to carry shared and unique info separately, theoretically achieving task-sufficiency.
- **vs. FOCAL/JointOpt/DisentangledSSL (InfoMax)**: These methods use $Z_S+Z_U^m$ explicit splitting but remain fixed vectors that cannot be selected per task. S3 explicitly delegates "selection" to a trainable router and allows further pruning at inference.
- **vs. FactorCL (Enhanced Contrastive)**: FactorCL uses augmentation to indirectly expand shared factors but is still constrained by the nature of contrastive objectives. S3 structurally breaks the contrastive vs. InfoMax dichotomy.
- **vs. Prompt Tuning / LoRA**: Both are "lightweight fine-tuning for downstream adaptation," but prompt/LoRA scales the input or adds low-rank increments. S3 tunes the routing—the router act as a "controller" for experts, requiring no extra parameters besides the existing MoE router.
- **vs. Switch Transformer/MoE Works**: Traditional MoE focuses on computational scaling and load balancing. S3 conversely uses MoE to represent "concept decomposition" as a semantic structure, representing a fundamental difference in purpose—pointing MoE research toward a new "semantic specialization" direction.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The "structural inductive bias + MoE concept experts + inference-time pruning knob" trio is highly integrated and forms a clear contrast with contrastive/InfoMax paradigms; DSC formalization is also a new contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four benchmarks + three granularities + three batch sizes + complete pruning curves provide clear structural conclusions; however, the task scale is small and lacks large-scale vision-language verification.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations (Props 2.3/2.5, Def 3.1-3.4) are clean and self-consistent, with each theoretical proposition reflected in experiments, forming a very rigorous logical chain.
- **Value**: ⭐⭐⭐⭐ Points toward a third path for multimodal representation learning; the inference-time pruning knob has high engineering value; however, deployment costs and large-scale scalability still need further verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning](same_stabilized_mixture-of-experts_for_multimodal_continual_instruction_tuning.md)
- [\[ICML 2026\] VisionPulse: Dynamic Visual Sparsification in Multimodal Reasoning](visionpulse_dynamic_visual_sparsity_for_efficient_multimodal_reasoning.md)
- [\[ICLR 2026\] Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts](../../ICLR2026/multimodal_vlm/capacity-aware_inference_mitigating_the_straggler_effect_in_mixture_of_experts.md)
- [\[CVPR 2026\] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](../../CVPR2026/multimodal_vlm/modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)
- [\[AAAI 2026\] MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment](../../AAAI2026/multimodal_vlm/mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)

</div>

<!-- RELATED:END -->
