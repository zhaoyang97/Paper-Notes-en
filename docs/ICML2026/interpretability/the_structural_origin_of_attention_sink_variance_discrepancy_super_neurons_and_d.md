---
title: >-
  [Paper Note] The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity
description: >-
  [ICML 2026][Interpretability][attention sink] This paper reveals the structural root of "attention sink to the first token" in LLMs—under causal masking, the first token lacks value aggregation…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "attention sink"
  - "variance discrepancy"
  - "super neurons"
  - "dimension collapse"
  - "head-wise RMSNorm"
date: 2026-05-08
content_hash: da4b6380be9427be
---

# The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity

**Conference**: ICML 2026  
**arXiv**: [2605.06611](https://arxiv.org/abs/2605.06611)  
**Code**: None  
**Area**: Interpretability / Transformer Mechanisms / LLM Optimization  
**Keywords**: attention sink, variance discrepancy, super neurons, dimension collapse, head-wise RMSNorm

## TL;DR
This paper reveals the structural root of "attention sink to the first token" in LLMs—under causal masking, the first token lacks value aggregation, leading to variance discrepancy, which is selectively amplified by super neurons in the FFN, resulting in extreme dimension disparity. This ultimately locks the QK projection, forcing the formation of an attention sink. Based on this, the authors propose head-wise RMSNorm during pretraining to fundamentally suppress the sink.

## Background & Motivation

**Background**: Attention sink (the first token in decoder-only Transformers inexplicably absorbing a large amount of attention scores) is a common phenomenon in GPT/LLaMA models. It is both exploited (e.g., KV cache compression in StreamingLLM) and criticized (causing activation outliers, representation collapse, and difficulties in low-bit quantization). Previous explanations include: Softmax "absorbing residual probability mass," positional encoding side effects, and spectral subspace issues (Xiao et al. 2023, Yan et al. 2024, Cancedda 2024).

**Limitations of Prior Work**: These explanations are either phenomenological (e.g., "Softmax needs a sink") or only cover partial cases (positional encoding theory cannot explain why the sink appears suddenly at layer 2 instead of layer 0). None can fully answer the three interlinked phenomena: **"Why specifically the first token, why at a particular layer, and why does the norm suddenly explode?"**

**Key Challenge**: The authors find that the onset of the attention sink is a **structural invariant**—on Llama-2, regardless of input, the sink consistently appears at layer 2, accompanied by a synchronous surge in the $\ell_2$-norm of the first token's representation. This indicates that the sink is not an emergent property but the inevitable trigger of a **deterministic causal chain** at a fixed layer. However, the nature of this chain and whether it can be intervened upon were previously unclear.

**Goal**: To fully delineate the three-stage causal chain from "token-level statistical discrepancy → amplification at the FFN neuron level → lock-in at the attention pattern level," and to validate causality at each stage with controlled experiments.

**Key Insight**: Starting from the **positional asymmetry** of value aggregation—under the causal mask, the first token $i=0$ can only attend to itself ($a_{0,0}=1$), while subsequent tokens aggregate convex combinations of $i+1$ vectors, causing variance to monotonically decay with position. Thus, the first token is naturally a variance outlier. This simple observation is the origin of the entire chain.

**Core Idea**: Attention sink = **variance discrepancy (from value aggregation) → selective activation of super neurons (FFN amplification) → dimension disparity (sparse down-projection channeling) → QK lock-in (RMSNorm projects the first token to a fixed direction)**. Once this chain is understood, head-wise RMSNorm can be used upstream to suppress variance discrepancy at its root.

## Method

### Overall Architecture
The authors first conduct "phenomenon diagnosis" (Sec 3): demonstrating layer 2 onset and synchronous norm surge; then "causal localization" (Sec 3.1-3.2): proving that value aggregation introduces positional variance discrepancy, and using two controlled interventions (mask intervention, variance amplification) to reproduce the sink at **any position**; then "propagation chain analysis" (Sec 4): tracking how variance discrepancy is preserved by $\mathbf{W}_O$ → triggers super neurons → forms dimension disparity via sparse $\mathbf{W}_{\text{down}}$ → degenerates into a single basis vector via RMSNorm → locks QK to form the sink; finally, "engineering intervention" (Sec 5): proposing head-wise RMSNorm to suppress variance discrepancy at its root, eliminating the sink and accelerating convergence during pretraining.

### Key Designs

1. **Diagnosis and Causal Validation of Positional Variance Discrepancy**:

    - **Function**: Pinpoints the root cause of "why the first token" to the unique operation of value aggregation.
    - **Mechanism**: Using fully random token sequences (excluding BOS bias), measures dimension-wise variance after value aggregation in Llama-2-7B layer 1, finding that position 0 has much higher variance than others, with variance monotonically decreasing with position. Two **controlled interventions** validate causality: (a) **Mask intervention**—modifying the attention mask of the $k$-th token to attend only to itself, simulating the unaggregated state of the first token, immediately turns $k$ into a new sink; (b) **Variance amplification**—directly amplifying the variance of any token via $\mathbf{o}_k'^{(l)}=\boldsymbol{\mu}^{(l)}+\lambda\cdot(\mathbf{o}_k^{(l)}-\boldsymbol{\mu}^{(l)})$ ($\lambda>1$) also creates a new sink. Key control: simply scaling the norm $\lambda\cdot \mathbf{o}_k$ **cannot** reproduce the sink, ruling out "large norm causes sink" confusion.
    - **Design Motivation**: This is the causal anchor of the paper—without these two interventions, the claim that "variance discrepancy is the root cause" is only correlational; with them, it is causal.

2. **Selective Activation of Super Neurons and Dimension Disparity**:

    - **Function**: Explains how variance discrepancy is exponentially amplified by a **few specific neurons** in the FFN into dimension collapse.
    - **Mechanism**: For SwiGLU FFN $\text{FFN}(\mathbf{x})=(\text{SiLU}(\mathbf{x}\mathbf{W}_{\text{gate}})\odot \mathbf{x}\mathbf{W}_{\text{up}})\mathbf{W}_{\text{down}}$, it is found that $\mathbf{W}_{\text{gate}}$/$\mathbf{W}_{\text{up}}$ contain a small number of columns with extremely large norms (super neurons, e.g., index 7890). Tracking these neurons' responses to the first token shows that cosine($\mathbf{x}_{\text{norm}}, \mathbf{w}_{\text{gate}}^{(7890)}$) is high for the first token and near zero elsewhere, and $\mathbf{W}_{\text{up}}^{(7890)}$ projects massive activation—i.e., super neurons **"open the gate" only for the first token**. The corresponding row $\mathbf{w}_{\text{down}}^{(7890)}$ in $\mathbf{W}_{\text{down}}$ is **heavy-tailed sparse**, with most values near zero and a few dimensions (e.g., dim 2533) extremely large, channeling the massive activation into those outlier dimensions. The Dominance Ratio $\text{DomRatio}(\mathbf{h}_0)=\max_j|\mathbf{h}_{0,j}|/(\frac{1}{d}\sum_k|\mathbf{h}_{0,k}|)$ quantifies this disparity, reaching 200+ in shallow Llama-2 layers.
    - **Design Motivation**: This step translates "statistical variance discrepancy" into "geometric collapse in parameter space," and explains why the sink appears at a fixed layer (layer 2)—because super neurons are a fixed structure learned during pretraining, requiring several layers to accumulate.

3. **RMSNorm Directional Collapse + QK Structural Lock-in**:

    - **Function**: Explains why dimension disparity **inevitably** translates into QK attention score lock-in for the first token.
    - **Mechanism**: When $\mathbf{x}_0$ has an overwhelmingly large value $\lambda$ in dimension $c$, the RMSNorm normalization constant is almost entirely determined by $\lambda$, and the output degenerates to $\text{RMSNorm}(\mathbf{x}_0)\approx \text{sgn}(\lambda)\sqrt{d}\gamma_c\cdot \mathbf{e}_c$ (a fixed basis vector direction). After key projection, $\mathbf{k}_0^{(h)}\approx \pm\sqrt{d}\cdot (\mathbf{W}_K^{(h)})_{c,:}$ (becomes the $c$-th row of $\mathbf{W}_K$). Using SVD, the cosine alignment between the principal direction $\mathbf{u}_1^{(h)}$ of the query matrix and $\mathbf{k}_0^{(h)}$ is measured; heads with high alignment have positive QK dot products for all tokens (positive ratio ~100%), meaning these heads' queries are structurally aligned with the sink key, forcing large attention scores.
    - **Design Motivation**: This step closes the causal chain—"high variance → super neurons → dimension disparity → fixed direction → high QK score → sink," with observable, quantifiable intermediate variables at each step and no unexplained leaps.

### Loss & Training

**Head-wise RMSNorm Intervention** (Sec 5.1): After value aggregation and before output projection $\mathbf{W}_O$, RMSNorm is applied to each head's output: $\hat{\mathbf{o}}_t^{(h)}=\frac{\mathbf{o}_t^{(h)}}{\text{RMS}(\mathbf{o}_t^{(h)})}\odot \boldsymbol{\lambda}$, where $\boldsymbol{\lambda}\in\mathbb{R}^{d_k}$ is a learnable scaling vector shared across heads. This ensures (i) variance normalization of aggregated vectors at all positions, (ii) balancing the contributions of low-entropy (high-variance) and high-entropy (low-variance) heads to $\mathbf{W}_O$, preventing any single head from overwhelmingly dominating the residual stream. Verified with 152M parameters / 20B tokens / OpenWebText from scratch pretraining.

## Key Experimental Results

### Main Results: Three Architecture Comparison (Llama-2 config, averaged over 4 random seeds)

| Metric | Baseline (Softmax) | Sigmoid Attention | **Ours (HeadNorm)** |
|--------|---------------------|-------------------|---------------------|
| Train Loss ↓ | 2.7483 ± 0.0118 | — | **2.7073 ± 0.0095** |
| Validation Loss ↓ | 2.7812 ± 0.0109 | (slow and high) | **2.7421 ± 0.0066** |
| Effective Rank ↑ (layer mean) | 343.71 ± 15.63 | high | **445.96 ± 5.37** |
| Dimension Disparity ↓ (layer mean) | 82.67 ± 8.09 | low | **33.74 ± 2.73** |
| Attention Sink Eliminated | No (appears from layer 5) | Yes | Yes |

### Ablation Study & Intervention Validation

| Experiment | Phenomenon | Conclusion |
|------------|------------|------------|
| Mask block at $k=10$ | $k$ immediately becomes sink | Variance discrepancy is the causal origin of sink |
| Variance amplify $\lambda\uparrow$ at $k=10$ | sink score increases monotonically | Magnitude control → strong causality |
| Scale norm $\lambda\cdot \mathbf{o}_k$ (control) | sink does not appear | Rules out "large norm causes sink" confusion |
| $\mathbf{W}_O$ Kendall $\tau$ vs $\boldsymbol{\sigma}_{in}$ | mean 0.32 (positive bias) | $\mathbf{W}_O$ structurally amplifies variance dimensions |
| Layer 2 outlier dim 2533 after RMSNorm | DomRatio 262.88× | Direction almost completely collapses to $\mathbf{e}_{2533}$ |

### Key Findings
- **Sigmoid attention can eliminate the sink but trains worse**: This validates that "variance discrepancy is the root cause," but also shows that simply changing the activation is not a free lunch—since $\sigma$ output magnitude scales with sequence length, introducing new training instability.
- **HeadNorm not only eliminates the sink but also accelerates convergence**: This is a theoretically explained empirical bonus—variance normalization improves the conditioning of the optimization landscape, allowing AdamW to descend on a flatter surface.
- **Effective rank increases from 343 → 446**: Indicates that the sink is not just an attention phenomenon, but is accompanied by manifold collapse of the hidden state; HeadNorm also rescues representation capacity.
- **Super neuron is not emergent but learned**: After pretraining, these neurons are fixed in position (e.g., index 7890), and the corresponding $\mathbf{W}_{\text{down}}$ row vectors are sparse—this also points to the root cause of outlier handling in low-bit quantization.

## Highlights & Insights
- **Three-stage causal chain + two controlled interventions turn attention sink from an "empirical phenomenon" into an "intervenable engineering problem"**: The mask and variance amplification experiments are particularly elegant, directly settling the long-standing causality question.
- **HeadNorm is an elegant and cheap engineering solution**: Just one line of RMSNorm plus a learnable $\boldsymbol{\lambda}$; it does not alter attention math or Softmax, and can be directly integrated into existing LLaMA pretraining pipelines—subsequent work can use it out of the box.
- **The super neuron + sparse down-projection perspective is highly generalizable**: It unifies a series of seemingly unrelated phenomena—attention sink, activation outliers, low-bit quantization difficulties, and representation collapse—under the same FFN structural explanation, providing an important lemma for model design and compression research.

## Limitations & Future Work
- Intervention experiments are mainly conducted on Llama-2-7B; although the appendix verifies multiple open-source LLMs, all are from the same architecture family (decoder-only + SwiGLU + RMSNorm). Generalization to GLU variants or different norm types (LayerNorm, DeepNorm) is not fully tested.
- HeadNorm's pretraining validation is only at 152M scale / 20B tokens; scaling law trends are not explored, and it is not guaranteed that the same convergence acceleration holds at industrial 7B+ scale.
- The impact of HeadNorm on downstream long-context performance (especially for methods like StreamingLLM that rely on the sink for KV compression) is not analyzed—if the sink is needed as an anchor in practice, this could be a drawback.
- Whether the learnable $\boldsymbol{\lambda}$ needs to be retuned for different tasks/data, and whether its learned values have interpretable patterns, is not discussed.
- Future directions: making "where and when to normalize" a dynamic decision (similar to Mixture-of-Norm), applying different treatments to different head behaviors.

## Related Work & Insights
- **vs Xiao et al. 2023 (StreamingLLM)**: They discovered the sink phenomenon and used it for KV compression; this paper digs three layers deeper to show the sink is a fully avoidable structural artifact. The two are complementary (one utilizes, one fixes).
- **vs Cancedda 2024 (spectral subspace)**: Cancedda explains the sink as a query/key spectral subspace issue; this paper further explains **why** the spectral subspace behaves this way—because RMSNorm projects the first token onto a specific row of $\mathbf{W}_K$.
- **vs Liu et al. 2024 (activation outliers)**: They focus on outliers in quantization scenarios; this paper proves that outliers and the sink share the same root (super neuron + sparse down-proj), so solving the sink also alleviates quantization difficulties.
- **vs Sigmoid attention (Ramapuram et al. 2024)**: Sigmoid is already known to eliminate the sink; this paper not only compares horizontally but also uses it to sanity-check its own causal hypothesis—since Sigmoid removes the sum-to-one constraint (thus eliminating variance discrepancy), the theory predicts it should also eliminate the sink, which is indeed observed.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The three-stage causal chain + super neuron perspective is a genuinely new explanation, not just another phenomenological conjecture.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two controlled interventions + multi-seed validation + multi-LLM replication (appendix), very solid.
- Writing Quality: ⭐⭐⭐⭐⭐ Progresses through phenomenon → hypothesis → causal validation → propagation chain → engineering fix, very well-structured; Schematic Figure 1 is highly illustrative.
- Value: ⭐⭐⭐⭐ HeadNorm is immediately usable in engineering, and the mechanism provides insights for quantization/long-context; slight deduction for not yet validating at large scale (>7B).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)
- [\[ACL 2026\] StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference](../../ACL2026/interpretability/structkv_preserving_the_structural_skeleton_for_scalable_long-context_inference.md)
- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](../../NeurIPS2025/interpretability/why_is_attention_sparse_in_particle_transformer.md)
- [\[ACL 2026\] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](../../ACL2026/interpretability/do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)
- [\[CVPR 2026\] Reallocating Attention Across Layers to Reduce Multimodal Hallucination](../../CVPR2026/interpretability/reallocating_attention_across_layers_to_reduce_multimodal_hallucination.md)

</div>

<!-- RELATED:END -->
