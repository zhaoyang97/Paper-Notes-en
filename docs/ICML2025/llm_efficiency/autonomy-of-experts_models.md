---
title: >-
  [Paper Note] Autonomy-of-Experts Models (AoE)
description: >-
  [ICML 2025][LLM Efficiency][Mixture-of-Experts] AoE proposes allowing experts in an MoE to autonomously decide whether to process an input based on their own internal activation norms (rather than being determined by an external router). By reducing precomputation overhead through low-rank weight factorization, AoE outperforms traditional MoE in pre-training 700M-4B parameter language models.
tags:
  - "ICML 2025"
  - "LLM Efficiency"
  - "Mixture-of-Experts"
  - "Autonomous Expert Selection"
  - "Activation Norm"
  - "Low-Rank Decomposition"
  - "Large Language Models"
date: 2026-05-08
content_hash: fdf0f77ca16781cd
---

# Autonomy-of-Experts Models (AoE)

**Conference**: ICML 2025  
**arXiv**: [2501.13074](https://arxiv.org/abs/2501.13074)  
**Code**: [https://github.com/trestad/Autonomy-of-Experts](https://github.com/trestad/Autonomy-of-Experts)  
**Area**: LLM Efficiency  
**Keywords**: Mixture-of-Experts, Autonomous Expert Selection, Activation Norm, Low-Rank Decomposition, Large Language Models

## TL;DR

AoE proposes allowing experts in an MoE to autonomously decide whether to process an input based on their own internal activation norms (rather than being determined by an external router). By reducing precomputation overhead through low-rank weight factorization, AoE outperforms traditional MoE in pre-training 700M-4B parameter language models.

## Background & Motivation

**Background**: Mixture-of-Experts (MoE) is one of the core architectures of modern large language models (such as Mixtral, DeepSeek-MoE, and Qwen-MoE). MoE decomposes a large FFN into multiple smaller FFNs (experts) and uses a router to select Top-K experts for each token to process, achieving sparse activation to improve efficiency. The router is typically a simple MLP classifier that outputs expert selection probabilities based on the input hidden state.

**Limitations of Prior Work**: Traditional MoE suffers from a widely overlooked key issue—**the decoupling between router decisions and expert execution**. Specifically: (a) The router cannot directly evaluate the capability of an expert, making its selection essentially an "unlabeled prediction"; (b) If the router makes an incorrect prediction, the selected expert may struggle to effectively process the token, leading to an increased training loss; (c) Experts may be forced to adjust their parameters to handle tokens they are not suited for, conflicting with their original specialized domains; (d) The router can only learn better decisions through trial and error, wasting a large volume of training steps.

**Key Challenge**: In traditional MoE, the decision-making power of "who processes this token" lies in the hands of the router, but the router knows nothing about the actual capability of the experts. This decision-execution decoupling leads to **suboptimal expert selection** and **inefficient training**. Meanwhile, although auxiliary load-balancing losses in MoE mitigate the load imbalance issue among experts, they do not fundamentally address the selection quality problem.

**Goal**: (a) How to make expert selection more accurate—based on the expert's self-evaluation of its capability rather than an external router's guess? (b) How to eliminate the router while preserving computational efficiency?

**Key Insight**: Drawing from the FFN-as-key-value-memory perspective (Geva et al., 2021), the authors propose a key insight: **the internal activation norm of an expert reflects its capability to handle the input**. If an expert can effectively process a certain input, its "key" vector (internal activation) should be highly activated. Pilot experiments verify that removing the router and selecting Top-K experts solely based on the expert's internal activation norm on a pre-trained Mixtral 8×7B preserves up to 95% of the original performance without any parameter updates.

**Core Idea**: Remove the router, allowing each expert to first perform a low-rank precomputation to obtain its internal activation. Experts are then ranked by activation norms to select the Top-K for continued forward propagation, achieving autonomous expert selection.

## Method

### Overall Architecture

AoE modifies the workflow of the MoE layer:

- **Traditional MoE**: Input $\mathbf{x}$ → Router $R(\mathbf{x})$ outputs probabilities → Select Top-K experts → Selected experts perform complete FFN computation → Weighted sum
- **AoE**: Input $\mathbf{x}$ → All experts perform low-rank down-projection $\mathbf{x}\mathbf{W}_{down}^i$ → Cache activation and compute $L^2$ norm → Select Top-K based on norms → Selected experts continue forward computation from the cache → Weighted sum

The core difference is that there is no router. The selection of experts is entirely determined by their own internal activations. Unselected experts terminate their computation after the first step of low-rank projection.

### Key Designs

1. **Autonomous Expert Selection Mechanism**:

    - **Function**: Each expert performs precomputation on the input to generate internal activations, and chooses whether to continue processing based on the ranking of their $L^2$ norms.
    - **Mechanism**: The mathematical formulation of a traditional expert is $E_i(\mathbf{x}) = (\text{SiLU}(\mathbf{x}\mathbf{W}_g^i) \odot (\mathbf{x}\mathbf{W}_p^i))\mathbf{W}_o^i$. AoE's key observation is that the norm of $\mathbf{x}\mathbf{W}_g^i$ reflects the capability of expert $i$ to process $\mathbf{x}$. A large norm indicates the "key" of the expert is highly activated, meaning it is suited to handle this input; a small norm indicates a mismatch.
    - **Design Motivation**: FFNs can be interpreted as key-value memory networks. If an expert can effectively process an input, its corresponding "key" ($\mathbf{x}\mathbf{W}_g$) should be highly activated, thereby achieving effective knowledge retrieval via matching with the "value" ($\mathbf{W}_o$). This self-evaluation mechanism allows experts to make decisions based on their own "understanding" of the input, eliminating the information gap between router decisions and expert execution.

2. **Low-Rank Weight Factorization**:

    - **Function**: Decompose $\mathbf{W}_g$ into $\mathbf{W}_{down} \in \mathbb{R}^{d_{model} \times d_{low}}$ and $\mathbf{W}_{up} \in \mathbb{R}^{d_{low} \times d_{wide}}$, changing the AoE expert formula to $E_i(\mathbf{x}) = (\text{SiLU}(\mathbf{x}\mathbf{W}_{down}^i \mathbf{W}_{up}^i) \odot (\mathbf{x}\mathbf{W}_p^i))\mathbf{W}_o^i$.
    - **Mechanism**: This offers two critical benefits: (a) The $\mathbf{W}_{down}^i$ of all experts can be concatenated into a large matrix $\hat{\mathbf{W}}_{down} = [\mathbf{W}_{down}^1, \cdots, \mathbf{W}_{down}^n] \in \mathbb{R}^{d_{model} \times (n \cdot d_{low})}$, obtaining low-dimensional activation caches for all experts simultaneously via a single matrix multiplication $\mathbf{C} = \mathbf{x}\hat{\mathbf{W}}_{down}$; (b) Unselected experts only perform the low-dimensional projection ($d_{low} \ll d_{model}$), which incurs minimal computation and caching overhead.
    - **Design Motivation**: If factorization is not used, all experts would need to compute the full $\mathbf{x}\mathbf{W}_g^i \in \mathbb{R}^{d_{ffn}}$ (e.g., $d_{ffn}=14336$ in Mixtral), causing immense caching and computational overhead. Since LLM weights are inherently low-rank (as verified by works like LoRA), this decomposition does not harm expressivity. The optimal $d_{low}$ is approximately $d_{model}/3$.

3. **$d_{low}$ and $d_{wide}$ Parameter Budget Constraint**:

    - **Function**: Automatically calculate $d_{wide}$ based on $d_{low}$ under the premise that the total parameter count aligns with the traditional MoE.
    - **Mechanism**: $d_{wide} = \frac{3 \cdot d_{model} \cdot d_{ffn} - d_{low} \cdot d_{model}}{d_{low} + 2 \cdot d_{model}}$, ensuring that AoE and MoE are comparable in terms of total parameter count.
    - **Design Motivation**: Fair comparison requires maintaining a controlled parameter count. Decreasing $d_{low}$ increases $d_{wide}$, and vice versa.

4. **Optional Auxiliary Load-Balancing Loss**:

    - AoE is compatible with traditional MoE's $\mathcal{L}_{aux}$ by simply replacing the router output with the softmax of $L^2$-Norm($\mathbf{x}\mathbf{W}_{down}^i$). Experiments indicate that AoE is more balanced than traditional MoE even without $\mathcal{L}_{aux}$, though adding it yields better results.

### Loss & Training

- **Primary Loss**: Standard language model NLL loss.
- **Auxiliary Loss** (Optional): $\mathcal{L}_{aux} = \alpha_{aux} \cdot n \cdot \sum_{i=1}^{n} \mathbf{f}_i \cdot \mathbf{P}_i$, where $\alpha_{aux}=0.01$.
- **Optimizer**: AdamW, $(\beta_1, \beta_2) = (0.9, 0.95)$, weight decay 0.1.
- **Training Data**: RedPajama, 100B tokens (small models) / larger scale (large models).
- **Learning Rate**: $2 \times 10^{-4}$ (small models) / $3.2 \times 10^{-4}$ (large models), linear warmup + cosine decay.

## Key Experimental Results

### Main Results

732M parameter (247M active parameters) language model trained on 100B tokens. Average accuracy across 8 tasks:

| Model Configuration | ARC-E | PIQA | SIQA | WINO | HELLA | MNLI | QNLI | SST2 | AVG |
|---------|-------|------|------|------|-------|------|------|------|-----|
| Traditional MoE | 39.90 | 58.43 | 35.67 | 52.09 | 27.98 | 33.09 | 49.28 | 49.66 | 43.28 |
| MoE + $\mathcal{L}_{aux}$ | 40.74 | 58.49 | 36.13 | 51.30 | 28.11 | 32.67 | 50.23 | 51.83 | 43.68 |
| AoE ($d_{low}$=256) | 40.70 | **59.41** | 36.64 | 52.09 | 28.06 | 34.38 | 50.69 | 53.21 | **44.39** |
| AoE ($d_{low}$=256) + $\mathcal{L}_{aux}$ | 41.33 | 58.65 | 36.80 | 50.75 | 28.40 | 33.71 | 49.55 | 53.10 | 44.04 |

4B parameter (1.18B active parameters) large model comparison:

| Model | ARC-E | PIQA | SIQA | WINO | HELLA | MNLI | QNLI | SST2 | AVG |
|------|-------|------|------|------|-------|------|------|------|-----|
| Traditional MoE | 53.70 | 65.40 | 39.10 | 51.54 | 35.80 | 32.19 | 49.77 | 57.00 | 48.06 |
| Ours | **55.98** | **65.61** | **39.87** | **52.57** | **36.77** | **35.39** | **50.05** | **61.93** | **49.80** |

The advantage of AoE is even more prominent on the 4B model (+1.74 average accuracy), showing that AoE's benefits amplify as the model scale scales up.

### Ablation Study

| Configuration | AVG Accuracy | Description |
|------|-----------|------|
| MoE baseline | 43.28 | Traditional MoE without $\mathcal{L}_{aux}$ |
| MoE + factorized $\mathbf{W}_g$ | 43.70 | Factorize weights only, without changing the selection mechanism → almost no improvement |
| MoE + large router | 43.71 | Scaling up router parameter budget to the same level as AoE → no obvious improvement either |
| AoE ($d_{low}$=64) | 43.81 | Over-compression, large approximation loss |
| AoE ($d_{low}$=128) | 44.12 | Relatively good |
| AoE ($d_{low}$=256) | **44.39** | Optimal, approximately $d_{model}/3$ |
| AoE ($d_{low}$=512) | 44.12 | Too large, increased activation noise |

### Key Findings

- **AoE's improvement does not stem from weight factorization**: MoE + factorized $\mathbf{W}_g$ (Config 3) performs almost identically to the original MoE (Config 2), proving that the improvement stems from the autonomous selection mechanism itself.
- **AoE's improvement does not stem from more parameters participating in selection**: Even when increasing the router parameters of traditional MoE to match AoE (Config 4), it still underperforms compared to AoE.
- **$d_{low} \approx d_{model}/3$ is optimal**: Too small ($d_{low}=64$) leads to large low-rank approximation errors, while too large ($d_{low}=512$) introduces activation noise.
- **AoE is naturally more balanced**: Even without $\mathcal{L}_{aux}$, the expert load distribution entropy ($\text{Ent}_{load}$) of AoE is higher than that of traditional MoE + $\mathcal{L}_{aux}$.
- **AoE makes more confident selections**: The selection confidence entropy ($\text{Ent}_{conf}$) of AoE is significantly lower than that of MoE, and decreases from shallow to deep layers, complying with the intuition of "general processing in shallow layers, specialized tasks in deep layers".
- **AoE training is more efficient**: The NLL loss curve shows that AoE consistently stays below traditional MoE during training, indicating that experts learn more efficiently.
- **Efficiency overhead is acceptable**: AoE ($d_{low}$=256) achieves 97% of traditional MoE's throughput, requiring around 7GB of extra memory (57.32 vs 50.61 GB).
- **Compatible with multiple selection strategies**: AoE combined with Top-P and Expert-Choice strategies still outperforms their respective traditional MoE baselines.

## Highlights & Insights

- **The insight "Experts know what they are good at" is highly precise**: This is a clean, intuitive, and experimentally verified observation. Doing no training on Mixtral 8×7B and merely selecting experts using activation norms still retains 95% of performance. This pilot experiment is highly convincing.
- **The idea of eliminating routers has profound significance**: As a decoupled module from the experts, the router's decisions are essentially "blind selections". AoE internalizes the selection capability into the experts themselves, fundamentally resolving the decision-execution decoupling. This design philosophy can be generalized to broader modular/routed architectures.
- **Low-rank factorization kills two birds with one stone**: It simultaneously solves computational efficiency (by compressing precomputation dimensions) and provides a compact self-evaluation sign (the low-dimensional activation norm), making the approach technically elegant.
- **Spontaneous alignment of expert self-evaluation standards** (Figure 4): During training, experts in the same layer automatically align their activation norm scales without extra constraints, demonstrating strong self-organizing capabilities.

## Limitations & Future Work

- **Memory overhead scales with expert count and sequence length**: All experts must perform low-rank projection and cache results. When the expert count $n$ is massive or sequence length is very long, the overhead of caching $\mathbf{C} \in \mathbb{R}^{n \times d_{low}}$ per token is non-negligible.
- **Communication patterns in distributed scenarios**: When traditional MoEs are deployed in a distributed fashion, experts are distributed across different devices. AoE requires all experts to first compute the low-rank projection and then compare norms, which might alter the communication patterns. The paper does not delve into the practicalities of large-scale distributed training.
- **Only validated on Llama FFN architecture**: The $\mathbf{W}_g$ factorization scheme is designed for SiLU-gated FFNs and needs adaptation for other FFN architectures (such as GLU variants).
- **The choice of $d_{low}$ relies on empirical findings**: The optimal value of roughly $d_{model}/3$ is an empirical observation rather than a theoretical derivation.
- **Relatively limited experimental scale**: The largest model tested is 4B parameters with 100B tokens. Performance on truly large-scale models like DeepSeek-V3 (671B) or Mixtral (47B) has yet to be verified.
- **No comparison against alternative router distillation schemes**: For instance, using expert activation norms as training targets for the router (similar to the idea in Pham et al., 2024)—would this yield similar benefits while preserving the router?

## Related Work & Insights

- **vs Traditional MoE (Shazeer et al., Switch Transformer)**: Traditional methods use an independent router MLP for selection, whereas AoE internalizes selection capability into the experts. The fundamental difference is that AoE's selection signal stems from the expert's "understanding" of the input, rather than the router's external prediction.
- **vs CompeteSMoE (Pham et al., 2024)**: CompeteSMoE also utilizes the norm of expert outputs as a training label for the router, but still retains the router and requires all experts to perform full computations. AoE directly removes the router and avoids complete computations using low-rank decomposition.
- **vs Expert-Choice routing (Zhou et al., 2022)**: Expert-Choice routes along the token dimension, letting experts select tokens rather than tokens selecting experts. AoE is orthogonal and compatible with Expert-Choice—AoE changes the source of the selection signal (from router to activation norm), while Expert-Choice changes the selection dimension.
- **vs Mixture-of-Depths (Raposo et al., 2024)**: MoD dynamically decides whether a token needs computation in a given layer. AoE and MoD can complement each other—the former decides "which expert processes", while the latter decides "whether processing is needed".

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The paradigm shift of "autonomous expert selection" is highly innovative, with deep insights and strong experimental support.
- Experimental Thoroughness: ⭐⭐⭐⭐ Highly detailed ablation (8 research questions), but the model scale is limited to 4B, lacking extremely large-scale validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, progress driven by successive questions, intuitive toy experiments, and clean figures.
- Value: ⭐⭐⭐⭐⭐ Proposes an important direction of improvement for MoE architectures. The code is open-source, leaving a profound impact on future MoE research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Mixture of Lookup Experts](mixture_of_lookup_experts.md)
- [\[ACL 2025\] DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts](../../ACL2025/llm_efficiency/dive_moe_reconstruction.md)
- [\[ACL 2026\] The Illusion of Specialization: Revealing the "Standing Committee" in Mixture-of-Experts Models](../../ACL2026/llm_efficiency/the_illusion_of_specialization_unveiling_the_domain-invariant_34standing_committ.md)
- [\[ICLR 2026\] Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models](../../ICLR2026/llm_efficiency/towards_greater_leverage_scaling_laws_for_efficient_mixture-of-experts_language_.md)
- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)

</div>

<!-- RELATED:END -->
