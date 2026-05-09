---
title: >-
  [Paper Note] Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba
description: >-
  [ICLR 2026][Model Compression][Mamba] This paper proposes Memba, a parameter-efficient fine-tuning method inspired by biological neuron membrane potentials. By introducing Leaky Integration Membrane (LIM) neurons into the gating branch of Mamba, Memba achieves temporal adaptability, combined with LoRA placement optimization and cross-layer membrane transfer. With minimal trainable parameters, Memba surpasses existing Mamba PEFT methods on both language and vision tasks.
tags:
  - ICLR 2026
  - Model Compression
  - Mamba
  - PEFT
  - membrane potential
  - leaky integration
  - state space models
date: 2026-05-08
content_hash: fe24d0cbfa41605a
---

# Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba

**Conference**: ICLR 2026
**arXiv**: [2506.18184](https://arxiv.org/abs/2506.18184)
**Code**: [GitHub](https://github.com/Intelligent-Computing-Lab-Panda/Memba)
**Area**: Model Compression
**Keywords**: Mamba, PEFT, membrane potential, leaky integration, state space models

## TL;DR
This paper proposes Memba, a parameter-efficient fine-tuning method inspired by biological neuron membrane potentials. By introducing Leaky Integration Membrane (LIM) neurons into the gating branch of Mamba, Memba achieves temporal adaptability, combined with LoRA placement optimization and cross-layer membrane transfer. With minimal trainable parameters, Memba surpasses existing Mamba PEFT methods on both language and vision tasks.

## Background & Motivation
State space models (SSMs) / Mamba replace the attention mechanism of Transformers with linear complexity. As model scale grows, PEFT becomes necessary. However, existing PEFT methods are directly transferred from Transformers to Mamba, overlooking the unique temporal processing dynamics of SSMs.

Key limitations:
1. Mamba's gating mechanism is a simple linear projection followed by SiLU, lacking the multi-gate temporal control capacity of LSTM/GRU.
2. Directly fine-tuning core SSM components (A, B, C, Δ in selective scan) leads to performance degradation, as verified by prior work.
3. How to introduce temporal adaptability without disrupting the pre-trained SSM's balanced dynamics?

**Core Idea**: Introduce biologically inspired LIM neurons into the gating branch (rather than the SSM branch) of Mamba. LIM neurons provide temporal selective memory through accumulation–leakage–reset dynamics of membrane potential, without introducing additional learnable parameters.

## Method

### Overall Architecture
Memba modifies the original Mamba architecture in three ways: ① LIM neurons are inserted into the gating branch to provide temporal processing capability; ② LoRA is applied only to the input and output projection layers (not SSM components); ③ the average membrane potential is transferred across layers. The SSM branch remains entirely unchanged.

### Key Designs
1. **Leaky Integration Membrane (LIM) Neurons**:

    - **Function**: Introduce temporal dynamics into the gating branch.
    - **Mechanism**: The input sequence is divided into $T$ equal-sized chunks and processed chunk by chunk: $\mathbf{u}[i+1]^l = r(\tau \mathbf{u}[i]^l + \mathbf{W}^l X[i])$, where $r(x) = 0$ if $x > V_{th}$, else $x$. $\tau \in (0,1]$ controls the leakage rate and $V_{th}$ is the reset threshold.
    - **Design Motivation**: LIM naturally realizes selective information retention — salient pathway features produce prominent membrane potential spikes, while baseline potentials gradually decay across chunks, mimicking the SSM's preference for recent tokens. No additional learnable parameters are introduced.

2. **LoRA Placement Optimization**:

    - **Function**: Determine which projection layers in Mamba should receive LoRA.
    - **Mechanism**: Ablation studies show that `in_proj` and `out_proj` are the most critical (removal degrades accuracy by 1.2% and 0.8%, respectively), while `dt_proj` and `x_proj` have minimal impact. Applying LoRA only to `in_proj` + `out_proj` already surpasses full fine-tuning.
    - **Design Motivation**: The input/output projections are the information bottlenecks of Mamba, whereas `dt`/`x` are internal SSM parameters that should not be modified.

3. **Cross-layer Membrane Potential Transfer**:

    - **Function**: Maintain temporal consistency across network depth.
    - **Mechanism**: After layer $l$ processes all chunks, the average membrane state $\bar{\mathbf{u}}^l = \frac{1}{T}\sum_{i=1}^T \mathbf{u}^l[i]$ is computed and used as the initial membrane potential for the first chunk of layer $l+1$: $\mathbf{u}^{l+1}[1] = \bar{\mathbf{u}}^l$.
    - **Design Motivation**: This prevents the loss of temporal context in deeper layers, and using the average rather than the final state avoids information loss.

### Theoretical Analysis
Theorem 1 shows that LIM has a dual effect: the mean membrane component provides temporal context integration via leaky dynamics, while the fluctuation component introduces bounded regularization $\mathcal{R}(\mathbf{y}_t, \bar{\mathbf{u}}_t) \leq \frac{\gamma}{2} \cdot \lambda_{\max} \cdot \epsilon^2$, smoothing the loss landscape.

## Key Experimental Results

### Main Results (Commonsense Reasoning, Mamba-130M)

| Method | #Params(%) | BoolQ | PIQA | SIQA | HellaS | WinoG | ARC-e | ARC-c | OBQA | Avg |
|--------|-----------|-------|------|------|--------|-------|-------|-------|------|-----|
| Full FT | 100 | 56.1 | 65.3 | 38.7 | 35.3 | 52.0 | 46.4 | 25.7 | 32.8 | 43.8 |
| SLL LoRA | 1.45 | 56.3 | 63.3 | 38.2 | 34.6 | 51.6 | 43.5 | 23.6 | 30.6 | 42.7 |
| LoRA (in_proj) | 2.23 | 53.5 | 62.9 | 38.2 | 33.8 | 53.1 | 46.4 | 23.7 | 30.8 | 42.8 |
| LoRAp (X) | 2.67 | 61.7 | 64.0 | 39.5 | 34.3 | 52.2 | 43.5 | 25.3 | 29.4 | 43.7 |
| **Memba (in+out)** | **5.20** | **58.8** | **65.8** | **40.1** | **34.7** | 51.6 | 47.7 | **24.7** | **31.2** | **44.3** |

### Ablation Study

| Configuration | Avg Acc (%) | Notes |
|---------------|------------|-------|
| All projectors LoRA | 43.9 | All projection layers |
| −dt_proj | 43.9 | Removing dt has negligible impact |
| −x_proj | 43.7 | Removing x has minor impact |
| −out_proj | 43.1 | Output projection is important |
| −in_proj | 42.7 | Input projection is most critical |
| Memba vs Full FT (790M) | Memba higher | PEFT outperforms full fine-tuning |
| Memba vs Full FT (1.4B) | Memba higher | Full fine-tuning prone to overfitting |

### Key Findings
- Memba surpasses full fine-tuning with only 5.2% of parameters across Mamba-130M/790M/1.4B; full fine-tuning is prone to overfitting.
- Membrane potential visualizations clearly show prominent spikes for salient features and gradual decay across chunks.
- `in_proj` and `out_proj` are the critical positions for Mamba PEFT; SSM components (`dt_proj`, `x_proj`) are unsuitable for fine-tuning.
- Cross-layer membrane transfer improves accuracy by ~0.5% over no transfer, with greater benefit for deeper networks.

## Highlights & Insights
- The biologically inspired LIM design is naturally complementary to Mamba's SSM — the SSM handles linear temporal processing while LIM (in the gating branch) provides nonlinear temporal selectivity.
- The design philosophy of "do not touch the SSM" is well-motivated: prior work has demonstrated that directly fine-tuning SSM components leads to degradation.
- The chunking strategy for membrane potentials elegantly addresses the efficiency challenge of processing long sequences token by token.
- The theoretical regularization analysis provides a principled explanation for the beneficial effect of membrane potential fluctuations.

## Limitations & Future Work
- Chunk size and chunk count $T$ are hyperparameters that require tuning.
- The sensitivity of the LIM leakage factor $\tau$ and threshold $V_{th}$ warrants further investigation.
- Validation on the more recent Mamba-2 architecture has not been performed.
- Vision task evaluation is limited to VTAB-1k; large-scale visual benchmarks are absent.

## Related Work & Insights
- **vs. SLL LoRA**: Memba achieves better temporal processing via LIM, with an average accuracy gain of 1.6%.
- **vs. Affix-tuning**: Memba achieves superior performance with 5.2% vs. 64.6% of parameters.
- **vs. Full Fine-tuning**: By avoiding overfitting, PEFT consistently outperforms full fine-tuning.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Combining biological membrane potentials with SSMs represents a genuinely novel direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-scale language and vision evaluations, though large-scale benchmarks are missing.
- **Writing Quality**: ⭐⭐⭐⭐ — Membrane potential visualizations are intuitive and the structure is clear.
- **Value**: ⭐⭐⭐⭐ — Opens a biologically inspired new avenue for PEFT in the Mamba era.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](../../ICCV2025/model_compression/generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)
- [\[ICLR 2026\] ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)
- [\[ICLR 2026\] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](fine-tuning_quantized_neural_networks_with_zeroth-order_optimization.md)
- [\[ICLR 2026\] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)
- [\[ICCV 2025\] TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning](../../ICCV2025/model_compression/tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)

</div>

<!-- RELATED:END -->
