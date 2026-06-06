---
title: >-
  [Paper Note] Demystifying When Pruning Works via Representation Hierarchies
description: >-
  [ICML 2026][Model Compression][Network Pruning] Starting from the "embedding → logit → probability" representation hierarchy, this paper uses Taylor expansion theory to prove that pruning-induced perturbations in embeddi…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Network Pruning"
  - "Generative Task Degradation"
  - "Softmax Amplification"
  - "Representation Hierarchies"
  - "KL Divergence"
date: 2026-05-08
content_hash: f6d54625f70f9dea
---

# Demystifying When Pruning Works via Representation Hierarchies

**Conference**: ICML 2026  
**arXiv**: [2603.24652](https://arxiv.org/abs/2603.24652)  
**Code**: Mentioned as "available in the project repository" in the paper, but no public link is provided  
**Area**: LLM Compression / Network Pruning / Representation Analysis  
**Keywords**: Network Pruning, Generative Task Degradation, Softmax Amplification, Representation Hierarchies, KL Divergence

## TL;DR
Starting from the "embedding → logit → probability" representation hierarchy, this paper uses Taylor expansion theory to prove that pruning-induced perturbations in embedding and logit spaces are naturally small. However, the non-linear softmax step amplifies these perturbations into the probability space by $\mathrm{Var}_r(\Delta z)/(2T^2)$. Combined with step-wise accumulation in autoregressive decoding, this ultimately leads to generative task collapse. Conversely, non-generative tasks are naturally robust because they rely only on candidate token subspaces and are not exposed to this amplification loop—unifying the explanation for why pruning maintains performance on MMLU and retrieval but drops to zero on GSM8K and HumanEval.

## Background & Motivation
**Background**: As LLM scales expand, network pruning (Wanda, SparseGPT, ShortGPT, Attn/MLP Drop, etc.) has become a mainstream compression solution. Intra-layer approaches sparsify individual layers (unstructured / 2:4 / 4:8), while inter-layer approaches directly remove specific transformer blocks or attention/MLP sub-layers. These methods have been proven to preserve performance near-losslessly on "non-generative tasks" such as retrieval, multiple-choice QA, and text classification.

**Limitations of Prior Work**: A recurring anomaly has been observed in practical deployments—the same pruned model shows almost no degradation on MMLU but collapses to zero on GSM8K, HumanEval, and NarrativeQA (e.g., after removing 8 MLP layers, Mistral-7B-Instruct's GSM8K drops from 48.4 → 0.0). However, there is no theoretical explanation for where this "task-dependent vulnerability" originates, leaving the industry to rely on empirical trial and error.

**Key Challenge**: Existing explanations attribute the failure to "large output spaces in generative tasks (vocabulary $|\mathcal{V}|$ far exceeding embedding dimension $d$)" or "autoregressive accumulation." These are intuitive descriptions that lack quantitative predictive power. Most importantly, they fail to answer how small embedding perturbations transform into catastrophic probability shifts.

**Goal**: (1) Segment LLM inference into three representation spaces (embedding $h$, logit $z$, probability $p$) to quantify perturbations; (2) Provide closed-form formulas to analytically predict the impact of pruning on each space; (3) Explain the robustness of non-generative tasks vs. the fragility of generative tasks; (4) Provide practical guidance.

**Key Insight**: The authors focus on a specific detail—after pruning, the perturbation $\Delta z = W \Delta h$ in logit space is a linear transformation (rotation + stretching), but in the probability space, $\Delta p = \mathrm{softmax}(z + \Delta z)/T - \mathrm{softmax}(z)/T$ is significantly amplified by the non-linear exponential normalization. Autoregressive decoding further compounds small single-step errors into multi-step accumulation.

**Core Idea**: The "task-dependency" of pruning performance is attributed to **differences in perturbation propagation across representation hierarchies**. Linear layers (embedding → logit) mostly maintain similarity, the softmax non-linearity acts as the actual amplifier, and multi-step decoding functions as a "circular horn" for this amplifier. Non-generative tasks only care about logit order or small candidate subspaces, thus avoiding this amplification loop.

## Method

### Overall Architecture
The paper does not propose a new pruning algorithm; instead, it establishes an **analysis framework**: (1) Split LLM inference into three representation spaces: $e \to h^{(l)} \to z \to p$; (2) Independently apply pruning to each layer to obtain $\Delta h$, $\Delta z$, and $\Delta p$, quantifying perturbations using angular deviation and KL divergence; (3) Derive closed-form expressions for perturbations in each space using second-order Taylor expansion; (4) Extend single-step analysis to multi-step generation to analyze error accumulation; (5) Analyze local stability by isolating multiple-choice tasks to "candidate token subspaces." Representative pruning methods include Wanda/SparseGPT (intra-layer) and ShortGPT/Attn-Drop/MLP-Drop (inter-layer), using Qwen-2.5-7B-Instruct and Mistral-7B.

### Key Designs

1.  **Three-space Perturbation Measurement Protocol**:
    - Function: Segregate the impact of pruning on embedding, logit, and probability representations to avoid confounding perturbations from different spaces.
    - Mechanism: During the baseline model's forward pass, a single layer is replaced with its pruned version (other layers remain original) to obtain $\Delta h_l$. Angular deviation $1-\mathrm{CosineSim}(h_l, h_l+\Delta h_l)$ quantifies the embedding shift. This is projected to logit space $z^{(l)}=W h^{(l)}$ to measure $1-\mathrm{CosineSim}(z, z+\Delta z)$, followed by $p^{(l)}=\mathrm{softmax}(z^{(l)}/T)$ for the probability space shift. This is repeated for every layer and decoding step. Empirical findings: Cosine similarity in embedding and logit spaces remains near 1.0, while the probability space oscillates violently, identifying the exact stage of amplification.
    - Design Motivation: Previous works either only examined weight sparsity or final perplexity, obscuring internal propagation rules. This isolated design acts as a controlled probe to separate "layer-wise local perturbation" from "end-to-end accumulation."

2.  **Taylor Local Theory (Theorem 1-3)**:
    - Function: Provide a closed-form explanation for empirical observations, answering why logit space is stable while probability space is not.
    - Mechanism: Angular deviation in embedding/logit space is approximated via second-order Taylor expansion as $1-\mathrm{CosineSim}(h, h+\Delta h) \approx \|\Delta h_\perp\|^2 / (2\|h\|^2)$, depending only on the squared ratio of the orthogonal component to the original vector norm. Since $\|\Delta h\|$ is naturally much smaller than $\|h\|$, this ratio is small. **The critical amplification point is softmax**: $1-\mathrm{CosineSim}(p, p+\Delta p) \approx \mathrm{Var}_r(\Delta z)/(2T^2)$, where $r_i = p_i^2/\|p\|^2$. Distribution shift via KL divergence is $\mathrm{KL}(p\|q) \approx \mathrm{Var}_{i\sim p}(\Delta z_i)/(2T^2)$. Crucial is the **variance of $\Delta z$**, not its magnitude—even if $\Delta z$ is small, if its distribution across the vocabulary is non-uniform (high variance), softmax amplifies the "flat vs. peak" difference exponentially. Temperature $T$ is in the denominator—lower temperatures lead to more aggressive amplification.
    - Design Motivation: This theory provides the first computable benchmark for "softmax amplification of pruning errors." Fig. 6 confirms that theoretical estimates for angular deviation and KL divergence align closely with ground truth. This allows predicting generative failure directly from single-layer perturbation statistics without actual generation.

3.  **Subspace Mechanism (Multi-Scale Analysis)**:
    - Function: Explain why the probability space oscillates violently yet multiple-choice/retrieval tasks remain robust.
    - Mechanism: Generative tasks involve sampling from the full $|\mathcal{V}|$ at each step. Small single-step deviations are fed back into history via KV cache, causing the baseline and pruned models to condition on different token histories after step 1, resulting in explosive error accumulation (Fig. 7). Non-generative tasks only look at step 1 and the logit ranking or a candidate subset $\mathcal{C}\subset\{1,\dots,|\mathcal{V}|\}$ (e.g., A/B/C/D). Fig. 8 shows that candidate tokens are usually in the **tail** of the probability distribution, where relative perturbations are smaller than those of top tokens, leaving argmax largely unchanged. Retrieval tasks operate in the embedding space via cosine similarity, which is inherently stable.
    - Design Motivation: This links "macro task performance" to "micro representation geometry," proposing three practical takeaways for pruning feasibility: the representation layer used, the dimensionality of the task-relevant subspace, and temporal dependency.

### Loss & Training
This is a training-free analysis and does not involve training losses. All pruning methods (Wanda, SparseGPT, ShortGPT, Attn-Drop, MLP-Drop) are executed according to their original protocols. Experiments focus on forward measurements rather than fine-tuning.

## Key Experimental Results

### Main Results
Comparison of non-generative vs. generative tasks for Mistral-7B under inter-layer pruning (dropping 8 attention layers, Drop-8A, or 8 MLP layers, Drop-8M):

| Task Type | Task | Full (7.1B) | Drop-8A (6.8B) | Drop-8M (5.7B) |
|-----------|------|-------------|----------------|----------------|
| Retrieval (E5-Mistral) | Avg of 13 BEIR | 58.9 | 53.4 | 56.8 |
| Multi-choice | BoolQ | 85.9 | 86.0 | 78.2 |
| Multi-choice | MMLU | 62.1 | 62.0 | 59.1 |
| Multi-choice Avg | 5 Tasks | 69.3 | 69.8 | 64.3 |
| Generative | GSM8K | 48.4 | 36.2 | **0.0** |
| Generative | HumanEval | 4.9 | **0.0** | **0.0** |
| Generative | MBPP | 13.8 | 0.4 | **0.0** |
| Generative | NarrativeQA | 16.3 | 9.6 | 2.0 |
| Generative Avg | 5 Tasks | 22.3 | 13.2 | **0.8** |

Drop-8M loses only 5 points on multi-choice Avg but collapses from 22.3 to 0.8 (97% degradation) on generative Avg.

### Ablation Study
Consistency between theoretical estimates and actual measurements (Fig. 6, Qwen-2.5-7B Layer 14 attention pruning):

| Metric | Theory vs. Measured | Description |
|--------|---------------------|-------------|
| Angular deviation $\Delta p$ | Close Fit | $\mathrm{Var}_r(\Delta z)/(2T^2)$ formula is accurate |
| KL divergence $p\|q$ | Close Fit | $\mathrm{Var}_{i\sim p}(\Delta z_i)/(2T^2)$ formula is accurate |
| Embedding CosSim | ~1.0 | Single layer $\|\Delta h\| \ll \|h\|$ |
| Logit CosSim | ~1.0 | LM head further compresses relative orthogonal components |
| Probability CosSim | Large Fluctuation | Softmax non-linearly amplifies variance |

Deviation accumulation during generation (Fig. 7, Drop-8A on Qwen-2.5-7B):

| Decoding Step | Embedding/Logit Sim | Probability Sim | Note |
|---------------|---------------------|-----------------|------|
| 1 (Within prompt) | ~1.0 | Lower but manageable | Identical conditioning |
| 2-3 | ~0.95 | Sharp decline | Token history divergence begins |
| 10+ | < 0.5 | Near 0 | Total divergence, gibberish output |

### Key Findings
- **Softmax, not LM head, is the primary amplifier**: Intuition might suggest that $z = Wh$ amplifies perturbations due to vocabulary dimensionality, but logit cosine similarity matches the embedding space. The linear transformation actually compresses relative orthogonal components. Softmax is the real amplifier because $\mathrm{Var}_r(\Delta z)/(2T^2)$ explicitly depends on the variance of $\Delta z$ across the vocabulary and the inverse of the temperature.
- **Candidate token subspaces act as shields**: In multiple-choice tasks, answer tokens usually reside in the tail of the distribution, where absolute perturbation magnitudes are small. The argmax remains largely unaffected by probability oscillations at the top.
- **Autoregressive decoding is the "echo chamber"**: Even if single-step $\Delta z$ variance is moderate, autoregressive decoding amplifies KV cache state differences into sequence differences. The resulting gibberish (e.g., "ILUNNIE M ` <%=>t...") visualizes this feedback loop.
- **Temperature $T$ regulates pruning robustness**: As $T^2$ is in the denominator, lower temperatures (sharper outputs) make pruned models significantly more fragile—a major warning for low-temperature deployments.

## Highlights & Insights
- **Theoretical-Empirical-Performance "Closed Loop"**: The work progresses from controlled probing of representation spaces to Taylor-derived formulas, then validates predictions against task benchmarks.
- **Decomposition of Task Robustness into Engineering Variables**: Layer hierarchy + subspace dimensionality + temporal dependency—allows predicting pruning feasibility for new tasks more efficiently than "trial-and-error" perplexity checks.
- **Actionable `Var_r(Δz)/(2T²)` Formula**: Since it only requires single-layer perturbation statistics, it can be used for early stopping or adjusting pruning rates during the process, rather than requiring full generation for evaluation.
- **Unified Failure Mode for Pruning and Quantization**: The authors suggest in Appendix I that quantization-induced errors follow the same logic, providing a unified mathematical perspective on model compression.

## Limitations & Future Work
- The framework is training-free and does not discuss how post-training or fine-tuning might repair softmax amplification—crucial as most industrial models undergo SFT/distillation.
- Taylor expansions are local; for multi-layer joint pruning or extreme perturbations in the first/last layers, deviations between theory and measurement require finer boundaries.
- Experiments focus on dense LLMs (Qwen-2.5, Mistral); the behavior in MoE models (partial expert activation) or SSMs (Mamba) remains unexplored.
- The explanation that answer tokens are "in the tail" for multiple-choice tasks is an empirical observation; the boundary conditions for prompt engineering that might shift candidates to the head are not defined.
- Lacks an algorithmic tool for layer selection based on the $\mathrm{Var}_r(\Delta z)/(2T^2)$ metric to prevent generative collapse.

## Related Work & Insights
- **vs. ShortGPT / Attn-Drop / MLP-Drop**: These methods are the subjects of analysis. This paper acts as a diagnostic tool for their failure cases rather than a replacement.
- **vs. Wanda / SparseGPT**: The paper proves that the generative vs. non-generative split holds for both unstructured and structured intra-layer patterns.
- **vs. Gromov et al. 2024**: While previous work noted that deeper layers are easier to prune, this work upgrades that observation by explaining *why* based on the representation space and temporal dimensions used by specific tasks.
- **Insight**: This "controlled probe + Taylor expansion + task decomposition" paradigm can be extended to quantization, distillation, and early exit techniques. It also suggests that temperature and sampling length should be considered as co-variables for pruning feasibility.

## Rating
- Novelty: ⭐⭐⭐⭐ Does not propose a new algorithm but provides the first unified framework explaining task-dependency in pruning via hierarchy and Taylor expansion.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various pruning modes, multiple LLMs, and diverse task types (retrieval/MCQ/generative); lacks MoE and post-fine-tuning scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ Seamless integration of formulas and experiments, with clear visualization in Figures 4-8.
- Value: ⭐⭐⭐⭐ Directly informs deployment strategies (which tasks can be pruned, temperature sensitivity), although it lacks a realized layer-selection tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Multi-Adapter Representation Interventions via Energy Calibration](multi-adapter_representation_interventions_via_energy_calibration.md)
- [\[ICML 2026\] The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works](the_bridge-garden_dilemma_in_llm_distillation_why_mixing_hard_and_soft_labels_wo.md)
- [\[ICML 2026\] When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging](when_shared_knowledge_hurts_spectral_over-accumulation_in_model_merging.md)
- [\[ACL 2026\] Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics](../../ACL2026/model_compression/why_steering_works_toward_a_unified_view_of_language_model_parameter_dynamics.md)
- [\[ICML 2026\] Effective Model Pruning: Measure the Redundancy of Model Components](effective_model_pruning_measure_the_redundancy_of_model_components.md)

</div>

<!-- RELATED:END -->
