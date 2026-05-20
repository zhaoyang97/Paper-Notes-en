---
title: >-
  [Paper Note] Demystifying When Pruning Works via Representation Hierarchies
description: >-
  [ICML 2026][Model Compression][Network Pruning] Starting from the three-stage representation hierarchy "embedding → logit → probability…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Network Pruning"
  - "Generation Task Degradation"
  - "Softmax Amplification"
  - "Representation Hierarchies"
  - "KL Divergence"
date: 2026-05-08
content_hash: c1f053d32da22ca7
---

# Demystifying When Pruning Works via Representation Hierarchies

**Conference**: ICML 2026  
**arXiv**: [2603.24652](https://arxiv.org/abs/2603.24652)  
**Code**: Mentioned as "available in the project repository" in the paper, but no public link provided  
**Area**: LLM Model Compression / Network Pruning / Representation Analysis  
**Keywords**: Network Pruning, Generation Task Degradation, Softmax Amplification, Representation Hierarchies, KL Divergence

## TL;DR
Starting from the three-stage representation hierarchy "embedding → logit → probability," this paper uses Taylor local expansion theory to prove: pruning introduces inherently small perturbations in the embedding and logit spaces, but the nonlinear softmax step amplifies these perturbations into the probability space by a factor of $\mathrm{Var}_r(\Delta z)/(2T^2)$. Through stepwise accumulation in autoregressive decoding, this ultimately leads to catastrophic failure in generation tasks. In contrast, non-generation tasks are naturally robust to pruning since they only depend on a candidate token subspace—this unifies the explanation for why pruning is nearly lossless on MMLU and retrieval but drops to zero on GSM8K and HumanEval.

## Background & Motivation
**Background**: As LLMs scale up, network pruning (Wanda, SparseGPT, ShortGPT, Attn/MLP Drop, etc.) has become a mainstream compression approach. Intra-layer methods sparsify individual layers (unstructured / 2:4 / 4:8), while inter-layer methods directly remove certain transformer blocks or attention/MLP sublayers. These methods have been shown to preserve performance almost losslessly on "non-generation tasks" such as retrieval, multiple-choice QA, and text classification.

**Limitations of Prior Work**: However, a repeatedly observed anomaly in deployment is that the same pruned model loses almost no points on MMLU but collapses to zero on GSM8K / HumanEval / NarrativeQA (e.g., Mistral-7B-Instruct after pruning 8 MLP layers: GSM8K drops from 48.4 → 0.0, HumanEval from 4.9 → 0.0, MBPP from 13.8 → 0.0). There has been no theoretical explanation for this "task-dependent fragility," so industry relies on empirical avoidance.

**Key Challenge**: Existing explanations blame "large output space for generation tasks (vocabulary $|\mathcal{V}|$ far exceeds embedding dimension $d$ or candidate count $k$)" or "autoregressive accumulation," but these are intuitive and cannot quantitatively predict outcomes. More crucially, they do not answer "how small embedding perturbations become catastrophic probability shifts."

**Goal**: (1) Decompose LLM inference along the information flow into three representation spaces (embedding $h$ / logit $z$ / probability $p$) and quantify perturbations at each stage; (2) Provide closed-form formulas to analytically predict pruning impact on each space; (3) Explain why non-generation tasks are robust and generation tasks are fragile; (4) Offer practical guidance.

**Key Insight**: The authors focus on a specific detail—after the same pruning-induced $\Delta h$, the logit space perturbation $\Delta z = W \Delta h$ is a linear transformation (rotation + scaling), but in the probability space, $\Delta p = \mathrm{softmax}(z + \Delta z)/T - \mathrm{softmax}(z)/T$ is greatly amplified by nonlinear exponential normalization; autoregressive decoding further accumulates small per-step errors over multiple steps.

**Core Idea**: The "task dependence" of pruning performance is attributed to **differential propagation of perturbations across representation hierarchies**—linear layers (embedding → logit) largely preserve similarity, while the softmax nonlinear layer is the true amplifier, and multi-step decoding acts as a "feedback loop" for amplification. Non-generation tasks only care about logit order or a small candidate subspace and are never exposed to this amplification loop.

## Method

### Overall Architecture
The paper does not propose a new pruning algorithm but establishes an **analysis framework**: (1) Decompose LLM inference along $e \to h^{(l)} \to z \to p$ into three representation spaces; (2) Apply pruning to each layer independently to obtain $\Delta h$, $\Delta z$, $\Delta p$, and quantify perturbation magnitude in each space using cosine similarity and KL divergence; (3) Use second-order Taylor expansion to derive closed-form expressions for perturbations in each space; (4) Extend single-step analysis to multi-step generation to analyze error accumulation; (5) Further analyze multiple-choice tasks by focusing on the "candidate token subspace" for local stability. Representative pruning methods include Wanda / SparseGPT (intra-layer) and ShortGPT / Attn-Drop / MLP-Drop (inter-layer), with Qwen-2.5-7B-Instruct and Mistral-7B as representative models.

### Key Designs

1. **Three-Space Perturbation Measurement Protocol**:

    - **Function**: Separates the impact of pruning on the three internal representations—"embedding / logit / probability"—to avoid conflating perturbations across spaces.
    - **Mechanism**: During baseline model forward pass, replace the current layer with its pruned version (keeping other layers unchanged) to obtain perturbation $\Delta h_l$. Quantify embedding space shift using angular deviation $1-\mathrm{CosineSim}(h_l, h_l+\Delta h_l)$; project to logit space via LM head $z^{(l)}=W h^{(l)}$ and measure $1-\mathrm{CosineSim}(z, z+\Delta z)$; then compute probability space shift via $p^{(l)}=\mathrm{softmax}(z^{(l)}/T)$. Repeat for each layer and decoding step, plotting the three curves as in Figure 4. Empirically, cosine similarity in embedding and logit spaces is nearly 1 (except for the first and last layers), while probability space fluctuates dramatically; this directly pinpoints where amplification occurs.
    - **Design Motivation**: Previous work either focused only on weight sparsity or final perplexity, obscuring internal propagation patterns. This "replace only one layer at a time, others unchanged" isolation design acts as a controlled probe, cleanly separating "local layer perturbation" from "end-to-end accumulation."

2. **Taylor Local Theory (Theorem 1-3)**:

    - **Function**: Provides closed-form explanations for the empirical findings above, answering "why logit space is stable but probability space is not."
    - **Mechanism**: Cosine similarity in embedding/logit space can be approximated by second-order Taylor expansion as $1-\mathrm{CosineSim}(h, h+\Delta h) \approx \|\Delta h_\perp\|^2 / (2\|h\|^2)$, depending only on the squared ratio of the orthogonal component to the original vector norm; since $\|\Delta h\|$ from single-layer pruning is much smaller than $\|h\|$, this ratio is very small, and LM head projection further reduces the relative orthogonal component (confirmed in Fig. 5). **The key amplification occurs at softmax**: in probability space, $1-\mathrm{CosineSim}(p, p+\Delta p) \approx \mathrm{Var}_r(\Delta z)/(2T^2)$, where $r_i = p_i^2/\|p\|^2$; for distribution shift measured by KL divergence, $\mathrm{KL}(p\|q) \approx \mathrm{Var}_{i\sim p}(\Delta z_i)/(2T^2)$. The crucial factor is the **variance of $\Delta z$**, not its norm—even if $\Delta z$ is small overall, if it is unevenly distributed across vocab dimensions (high variance), softmax will exponentially amplify the "flat vs. peaked" difference. Temperature $T$ appears in the denominator—the lower the temperature, the stronger the amplification.
    - **Design Motivation**: This theory provides a computable, comparable metric for "softmax amplifies pruning error" for the first time; Fig. 6 shows that theoretical estimates of angular deviation and KL divergence closely match ground truth. This means one can predict whether a pruning operation will break generation tasks directly from single-layer perturbation statistics, without actual generation—valuable for engineering.

3. **Generation vs. Non-Generation Subspace Mechanism (Multi-Scale Analysis)**:

    - **Function**: Explains why probability space fluctuates wildly but multiple-choice/retrieval tasks remain robust.
    - **Mechanism**: Generation tasks sample from the full vocab $|\mathcal{V}|$ at each step and use autoregression; small per-step deviations are fed back via the KV cache, causing baseline and pruned models to condition on different token histories from the second step onward, leading to exponential error accumulation (Fig. 7: cosine similarity ~1 at step 1, drops to near 0 by step 10). Non-generation tasks only use the first step and focus on logit ranking/candidate token subset $\mathcal{C}\subset\{1,\dots,|\mathcal{V}|\}$ (e.g., A/B/C/D options). Fig. 8 shows candidate tokens are usually in the **tail** of the probability distribution, where relative perturbations are much smaller than for top tokens, so argmax is almost unchanged; retrieval tasks compute cosine in embedding space, which is inherently stable. Thus, "task robustness" is mechanically mapped to "which representation space is used + subspace size + number of steps."
    - **Design Motivation**: This step connects "macro task performance" with "micro representation geometry," proposing three practical takeaways: which representation space is used, whether the task-relevant subspace is low-dimensional, and whether there is temporal dependence—these three directly predict pruning feasibility.

### Loss & Training
This is a training-free analysis; no training loss is involved. All pruning methods (Wanda, SparseGPT, ShortGPT, Attn-Drop, MLP-Drop) are run according to their original protocols; experiments focus on forward measurements rather than fine-tuning.

## Key Experimental Results

### Main Results
Comparison of non-generation vs. generation tasks on Mistral-7B under inter-layer pruning (removing 8 attention layers Drop-8A or 8 MLP layers Drop-8M):

| Task Type | Task | Full (7.1B) | Drop-8A (6.8B) | Drop-8M (5.7B) |
|-----------|------|-------------|----------------|---------------|
| Retrieval (E5-Mistral) | Avg of 13 BEIR | 58.9 | 53.4 | 56.8 |
| Multiple Choice | BoolQ | 85.9 | 86.0 | 78.2 |
| Multiple Choice | MMLU | 62.1 | 62.0 | 59.1 |
| Multiple Choice Avg | 5 tasks | 69.3 | 69.8 | 64.3 |
| Generation | GSM8K | 48.4 | 36.2 | **0.0** |
| Generation | HumanEval | 4.9 | **0.0** | **0.0** |
| Generation | MBPP | 13.8 | 0.4 | **0.0** |
| Generation | NarrativeQA | 16.3 | 9.6 | 2.0 |
| Generation Avg | 5 tasks | 22.3 | 13.2 | **0.8** |

Drop-8M loses only 5 points on multiple-choice Avg, but generation Avg collapses from 22.3 to 0.8 (97% degradation).

### Ablation Study
Agreement between theoretical estimates and actual measurements (Fig. 6, Qwen-2.5-7B, 14th layer attention pruning):

| Metric | Theory vs. Measurement | Note |
|--------|-----------------------|------|
| Angular deviation $\Delta p$ | Matches closely | $\mathrm{Var}_r(\Delta z)/(2T^2)$ formula accurate |
| KL divergence $p\|q$ | Matches closely | $\mathrm{Var}_{i\sim p}(\Delta z_i)/(2T^2)$ formula accurate |
| Embedding cosine similarity | Nearly 1.0 | Single-layer $\|\Delta h\| \ll \|h\|$ |
| Logit cosine similarity | Nearly 1.0 | LM head further compresses relative orthogonal component |
| Probability cosine similarity | Large fluctuations | Softmax nonlinearity amplifies variance |

Error accumulation during generation (Fig. 7, Drop-8A on Qwen-2.5-7B):

| Decoding Step | Embedding/Logit Similarity | Probability Similarity | Note |
|---------------|---------------------------|-----------------------|------|
| 1 (within prompt) | ~1.0 | Low but controllable | Both models conditioned on same input |
| 2-3 | ~0.95 | Drops sharply | Token histories diverge |
| 10+ | < 0.5 | Near 0 | Complete divergence, output gibberish |

### Key Findings
- **The key amplifier is not the LM head but softmax**: Many intuitively believe $z = Wh$ (vocab dimension explosion) amplifies perturbations, but logit space cosine similarity is almost identical to embedding—linear transformation actually compresses the relative orthogonal component. The true amplifier is $\mathrm{softmax}(z/T)$, as $\mathrm{Var}_r(\Delta z)/(2T^2)$ explicitly depends on the variance of $\Delta z$ across vocab and the inverse temperature.
- **Candidate token subspace is a natural shield**: Multiple-choice answer tokens are usually in the distribution tail, where probability values and absolute perturbations are small, so argmax is barely affected by top-token probability fluctuations. This explains why MMLU still achieves 59.1 on the 5.7B model.
- **Autoregression is not the culprit, but it is the amplifier's echo chamber**: Even moderate variance in single-step $\Delta z$ is amplified by autoregression from single-step to multi-step, with KV cache state differences escalating into token sequence divergence, leading to complete generation collapse. Table 2's "ILUNNIE M ` <%=>t..." gibberish is a visualization of this echo.
- **Temperature $T$ not only affects generation diversity but directly modulates pruning robustness**: Since $T^2$ is in the denominator, lower temperature (sharper outputs) makes pruning more fragile; this is a red warning for combining "low temperature deployment + pruning."

## Highlights & Insights
- **Theory + empirical + task performance "triangular closure"**: Controlled probing exposes three-space perturbation differences, Taylor expansion provides formulas, and task-level benchmarks validate predictions—three layers of mutually reinforcing evidence, a rare level of rigor in pruning analysis.
- **Decomposes "task robustness" into three engineering-controllable variables**: Representation space (embedding / logit / probability) + task-relevant subspace dimension + temporal dependence—any pruning scheme can use these three to predict feasibility on new tasks, much more efficient than "try and see perplexity."
- **The formula `Var_r(Δz)/(2T²)` is actionable**: Since it only requires single-layer perturbation statistics, it can be used for early stopping or adjusting pruning rates during pruning; unlike typical approaches that require full generation to evaluate.
- **Unifies failure modes of pruning and quantization**: The appendix notes that quantization is also a compression-induced error and can be analyzed with the same theory. This "using more fundamental perturbation mathematics to unify adjacent problems" perspective is worth emulating.

## Limitations & Future Work
- The analysis framework is entirely training-free and does not discuss how post-training or pruning fine-tuning can repair softmax amplification—whereas in industry, almost all pruned models undergo SFT/distillation before deployment, so there is still a gap between theory and practice.
- Taylor first/second-order expansion only holds for "local, single-layer perturbations"; in scenarios with multi-layer joint pruning or severe perturbations in the first/last layers, the gap between theoretical estimates and actual measurements requires more precise boundaries.
- Experiments are mainly on Qwen-2.5-7B and Mistral-7B dense LLMs, not covering MoE models (activating only part of experts per step), state space models (Mamba), etc.; whether "softmax amplification" remains the main bottleneck in these architectures is unclear.
- The explanation that answer tokens are "in the distribution tail" for multiple-choice robustness is empirical; no boundary conditions are given for "what prompting/task format would push candidate tokens to the head"—this is actually a hidden assumption in MMLU-style prompt engineering.
- No algorithmic recommendations are provided for "how to select pruning layers to avoid generation collapse" (though the Discussion mentions takeaways); a layer ranking tool based on $\mathrm{Var}_r(\Delta z)/(2T^2)$ would be more practical.

## Related Work & Insights
- **vs ShortGPT / Attn-Drop / MLP-Drop**: These are the pruning methods analyzed in this paper; the paper is not a replacement but a diagnostic tool, providing closed-form explanations for their failure cases.
- **vs Wanda / SparseGPT**: Also analyzed as representative intra-layer methods. The paper shows that regardless of unstructured / 2:4 / 4:8 patterns, the generation vs. non-generation split holds.
- **vs Gromov et al. 2024 "Unreasonable Ineffectiveness of Deeper Layers"**: That work observed that pruning deeper layers has little effect; this paper upgrades the explanation to "why the effect depends on which representation space and time dimension the task uses."
- **Insights**: This "controlled probe + Taylor expansion + task decomposition" analysis paradigm can be extended to other compression techniques (quantization, distillation, early exit); it also suggests that when designing LLM deployment pipelines, temperature, sampling length, and task output space should be considered as co-variables for pruning feasibility, not just weight sparsity.

## Rating
- Novelty: ⭐⭐⭐⭐ Does not propose a new algorithm, but for the first time explains "task dependence of pruning" using a unified three-space representation + Taylor expansion framework—a high-quality contribution among analytical works.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers intra-/inter-layer pruning, multiple LLMs, embedding/multiple-choice/generation tasks, and theory vs. empirical comparison; lacks MoE / post-fine-tuning scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ Interleaves formulas and experiments, with Figs. 4-8 mapping each theoretical point to a figure, making for a very reader-friendly experience.
- Value: ⭐⭐⭐⭐ Directly guides practical deployment (which tasks can be pruned, which temperatures are sensitive, why to be cautious with generation), but lacks a practical tool for pruning layer selection.

## Related Papers

- [\[ACL 2025\] Disentangling the Roles of Representation and Selection in Data Pruning](../../ACL2025/model_compression/disentangling_the_roles_of_representation_and_selection_in_data_pruning.md)
- [\[ICML 2025\] From Logits to Hierarchies: Hierarchical Clustering made Simple](../../ICML2025/model_compression/from_logits_to_hierarchies_hierarchical_clustering_made_simple.md)
- [\[ICLR 2026\] Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models](../../ICLR2026/model_compression/scaling_reasoning_hop_exposes_weaknesses_demystifying_and_improving_hop_generali.md)
- [\[ICLR 2026\] What Layers When: Learning to Skip Compute in LLMs with Residual Gates](../../ICLR2026/model_compression/what_layers_when_learning_to_skip_compute_in_llms_with_residual_gates.md)
- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](../../CVPR2026/model_compression/generative_video_compression_with_one-dimensional_latent_representation.md)

</div>

<!-- RELATED:END -->
