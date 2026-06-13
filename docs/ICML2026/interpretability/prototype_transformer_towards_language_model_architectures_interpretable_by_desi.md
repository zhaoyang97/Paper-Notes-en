---
title: >-
  [Paper Note] Prototype Transformer: Towards Language Model Architectures Interpretable by Design
description: >-
  [ICML 2026][Interpretability][Prototype Networks] ProtoT replaces the $O(N^2)$ self-attention in Transformers with $R$ linear communication channels driven by learnable "prototype vectors" (employing write/read gates + E…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Prototype Networks"
  - "Autoregressive LM"
  - "Linear Attention Alternative"
  - "Concept Disentanglement"
  - "Behavioral Editing"
date: 2026-05-08
content_hash: 87eeb0b5f39b2933
---

# Prototype Transformer: Towards Language Model Architectures Interpretable by Design

**Conference**: ICML 2026  
**arXiv**: [2602.11852](https://arxiv.org/abs/2602.11852)  
**Code**: https://github.com/YDYordanov/prototype_transformer  
**Area**: Interpretability / Language Model Architectures  
**Keywords**: Prototype Networks, Autoregressive LM, Linear Attention Alternative, Concept Disentanglement, Behavioral Editing

## TL;DR
ProtoT replaces the $O(N^2)$ self-attention in Transformers with $R$ linear communication channels driven by learnable "prototype vectors" (employing write/read gates + EMA-based prefix mean). This forces each prototype to automatically bind to a nameable concept (e.g., "woman", "COVID", "New Zealand") during training, enabling surgical concept-level editing of model behavior while exceeding LLaMA of similar scale in text generation Elo.

## Background & Motivation
**Background**: Mainstream autoregressive LMs (such as GPT-4 and the LLaMA family) rely on $O(N^2)$ self-attention to model long-range dependencies. While highly capable, their internal reasoning processes remain opaque. Existing interpretability methods (e.g., attention visualization, probing, causal intervention, and SAE) are almost entirely *post-hoc*, attempting to extract explanations from architectures that were never designed for interpretability.

**Limitations of Prior Work**: Attention magnitude does not necessarily equate to causal importance (Jain & Wallace 2019). Furthermore, due to the phenomenon of superposition, a single neuron or attention head often encodes multiple concepts simultaneously. Methods like SAE require training additional auxiliary models to approximate disentanglement. Moreover, performing precise interventions—modifying a specific concept while preserving other capabilities—is extremely difficult, as side effects often propagate to global perplexity.

**Key Challenge**: The tension between the high expressivity of dense attention and the requirements for concept disentanglement or intervenability. Compressing all information into a shared KV space means any surgical edit inevitably affects the entire representation space.

**Goal**: (1) To design a mixer module with native support for concept binding; (2) To retain LLaMA-level generation quality; (3) To reduce inference complexity from $O(N^2)$ to $O(N)$.

**Key Insight**: Drawing from the "prototype = interpretable decision unit" paradigm in computer vision (e.g., ProtoPNet/ProtoViT), the authors integrate prototypes into *every mixer layer* and adapt them into a strictly causal, past-only autoregressive form. In this context, prototypes act not as classification anchors but as *filters for communication channels*.

**Core Idea**: The sequence is split into $R$ parallel channels using $R$ non-interacting prototypes. Each channel employs an EMA-based, time-discounted prefix mean. Under the pressure of channel-aligned softmax, prototypes are forced toward semantic specialization. Once each channel encodes a single concept, interpretability and surgical editing become natural properties of the architecture.

## Method
The ProtoT backbone is identical to LLaMA-3 (comprising $L$ layers of RMS-PreLN blocks, where each block consists of a mixer and a SwiGLU FFN with skip-connections). The sole modification is the replacement of the self-attention mixer with the **Prototype Mixer**. Other configurations (tokenizer, AdamW, cosine annealing, dropout=0.1, and shared weights between embeddings and the LM head) strictly follow the LLaMA setup.

### Overall Architecture
A single mixer maintains $R$ learnable prototypes $\mathbf{P}_1,\dots,\mathbf{P}_R \in \mathbb{R}^h$ (set to $R=32$ in experiments). At position $i$, the input $x_i$ and history $x_{<i}$ interact to produce output $y_i$ via two steps:

1. **Write gate**: Each historical token $x_j$ is written into channels based on its similarity to each prototype, using a softmax *along the prototype dimension*.
2. **Prefix mean**: Each channel independently performs an EMA-discounted causal cumulative average to obtain a context aggregation $\mathrm{PM}_k$ for each channel $k$.
3. **Read gate**: The current token $x_i$ reads information from the $R$ channels based on its similarity to the prototypes (softmax along the prototype dimension), summing and projecting the result to $y_i$.

The complete formulation ($U,V,W$ are linear maps; $\tau_w,\tau_r$ are learnable temperatures):

$y_i = U\!\left(\sum_{k=1}^R \mathrm{Softmax}_k\!\left(\tfrac{W(x_i)\cdot \mathbf{P}_k}{\tau_r}\right)\,\mathrm{PM}_k\right)$,
$\mathrm{PM}_k = \dfrac{\sum_{j<i}\beta_k^{i-j}\,\mathrm{Softmax}_k\!\left(\tfrac{x_j\cdot \mathbf{P}_k}{\tau_w}\right) V(x_j)}{\sum_{j<i}\beta_k^{i-j}\,\mathrm{Softmax}_k\!\left(\tfrac{x_j\cdot \mathbf{P}_k}{\tau_w}\right)}$

Where $\beta_k=\sigma(\gamma_k)\in(0,1)$ is the learnable EMA decay coefficient, representing the "temporal preference" of each prototype. This can be converted to a half-life $t_{1/2}^{(k)}=-\ln 2/\ln \beta_k$ for analysis. Since $\mathrm{PM}_k$ satisfies a recursive update ($x_i$ depends only on the state at $i-1$), autoregressive generation incurs $O(1)$ computation and memory per step, with a total cost of $O(N\cdot R\cdot h)$, which is linear with respect to sequence length.

### Key Designs

1. **Channel-aligned Prototype Gating (Write Gate × Read Gate)**:
    - **Function**: Explicitly decouples "where information is written" from "from where information is read," while ensuring the $R$ channels *do not interact* (unlike the latents in Perceiver, which utilize self-attention).
    - **Mechanism**: Both the write and read gates utilize similarity-based gating akin to cross-attention, but the softmax is applied across the *prototype dimension* rather than the *sequence dimension*. This is the fundamental difference between ProtoT and existing attention variants. It forces each token to perform "routing" among $R$ channels, creating competitive pressure. The $W$ matrix and independent temperature $\tau_r$ decouple reading from writing, allowing "predict-and-consolidate" behavior (where the reader anticipates at $t$ and the writer consolidates at $t+1$).
    - **Design Motivation**: Channel-aligned softmax is the core mechanism for "concept specialization." Attempting to force two semantics into one channel leads to forced averaging and increased loss; thus, training automatically pushes distinct concepts into distinct channels. The lack of interaction between channels ensures that surgical edits remain localized.

2. **Strictly Causal Prefix Mean with EMA Time Discounting**:
    - **Function**: Performs *past-only* time-discounted aggregation independently for each channel, fulfilling the role of both short- and long-term memory.
    - **Mechanism**: Cumulative write values for positions $j<i$ are multiplied by $\beta_k^{i-j}$ and normalized (mass normalization), which significantly reduces perplexity. The learnable $\beta_k$ assigns an independent time scale to each prototype; visualization shows that stop words/punctuation correspond to short half-lives, while narrative or thematic concepts correspond to long half-lives. Crucially, the sum is taken over $j<i$. While standard self-attention allows token $i$ to attend to itself, ProtoT deliberately severs this shortcut to force the write gate to anticipate the read gate's needs. Value streams utilize a low-rank projection ($h/2$) to reduce mixer computation by 50%. The first two layers include local convolutions ($kernel=5$) to compensate for short-range dependencies.
    - **Design Motivation**: EMA explicitly parameterizes "long-range dependency" into learnable time scales rather than relying on the attention mechanism to discover them. Severing the self-loop ensures channels act as "transit stations" rather than identity mappings. Low-rank projections and local convolutions mitigate fine-grained information loss in lower layers.

3. **Alpha Gate: A Natural Hierarchical Interpretability Probe**:
    - **Function**: Multiplies a scalar $\alpha$ (similar to ReZero) to each Prototype Mixer output before it is added to the residual stream. It serves as a zero-cost diagnostic for "layer contribution."
    - **Mechanism**: Unlike ReZero (initialized at 0), ProtoT initializes $\alpha$ to 1 (identity). If a layer's $\alpha$ value drops rapidly during training, it provides strong evidence that the mixer in that layer does not contribute to the final prediction. Experiments validated that sharing read/write routing in layer 0 with a sharper $\tau_r$ initialization (3.0 vs. 1.0) improves the utility of that layer.
    - **Design Motivation**: Traditional architectures require expensive ablation or probing to determine layer utility. ProtoT treats this as an emergent, observable scalar during training with negligible overhead, facilitating architecture hyperparameter tuning.

### Loss & Training
Standard next-token cross-entropy is used with AdamW, linear warmup (2% of training steps), and cosine annealing to 10% of the peak learning rate. All baselines share backbone hyperparameters with ProtoT ($h=256, L=6, FFN ratio \approx 2.7\times, dropout=0.1, BPE vocab=16k$), differing only in the mixer module to ensure fair comparison. The number of prototypes $R$ is set to 32, with 4 attention heads, and shared weights between word embeddings and the LM head.

## Key Experimental Results

Data: FineWeb-Edu subset, 250M tokens (default 18k documents $\times$ 10 epochs; large-scale 339k documents, $h=512, L=12, ctx=512$).

### Main Results

| Dataset / Metric | LLaMA | Mamba | DeltaNet | ProtoT | Notes |
|---|---|---|---|---|---|
| FineWeb perplexity (ctx=256) | **78.7** | 86.0 | 90.4 | 90.5 | Comparable to DeltaNet in default settings |
| FineWeb perplexity (Large-scale) | **25.8** | 26.5 | 31.5 | 29.5 | Outperforms DeltaNet, approaches Mamba at scale |
| Text Gen Elo (LLM-as-judge) | 975.2 | **1041.8** | 961.8 | 1021.2 | ProtoT > LLaMA, DeltaNet |
| GLUE Average (9 tasks) | **71.6** | 68.6 | 64.5 | 67.6 | Between Mamba and DeltaNet |
| Training throughput it/s (bsz=128, ctx=256) | **23.6** | 3.2 | 1.8 | 7.6 | Fastest among linear baselines |

### Interpretability + Intervention Experiments

| Method | Disentanglement ↑ | Coverage ↑ | Num. Themes ↓ |
|---|---|---|---|
| **ProtoT** | **6.52 ± 1.93** | **7.88 ± 2.25** | **3.86 ± 1.94** |
| LLaMA SAE (Top Variance) | 5.91 | 7.86 | 4.33 |
| LLaMA SAE (Top Frequency) | 5.52 | 7.47 | 4.68 |
| LLaMA Attention Heads | 5.02 | 6.69 | 5.02 |
| Null Model | 3.20 | 4.03 | 6.97 |

| Concept Intervention (WriteMask) | Max ΔProb | Mean ΔProb | Max ΔPPL | Mean ΔPPL |
|---|---|---|---|---|
| women | −16.60% | −3.13% | +0.29% | −0.08% |
| girls | −10.67% | −2.36% | +0.29% | −0.18% |
| COVID | −21.97% | −4.52% | +5.58% | +0.76% |
| New Zealand | −21.54% | −9.96% | +3.47% | +1.62% |
| mental | −2.20% | −0.73% | −0.04% | −1.20% |

### Key Findings
- **Concept disentanglement is guaranteed by the architecture**: Without training any auxiliary SAEs, ProtoT prototypes outperform LLaMA heads or LLaMA+SAE across disentanglement, coverage, and thematic purity metrics. This confirms that interpretability gains from channel-aligned softmax and independent PM channels are "free."
- **Surgical editing is truly "surgical"**: Disabling L9 P7 (female) reduces the emergence probability of "women" by 16.6%, while global perplexity fluctuates by only $\pm 1\%$. Disabling L9 P18 (male) actually increases the "women" probability by 16.95% (releasing complementary semantics). Neutral prototypes (e.g., L9 P2) show negligible impact. Such causal verification is difficult to achieve with post-hoc methods.
- **Predict-and-consolidate**: The read gate consistently activates one token earlier than the write gate, meaning the model automatically learns to "anticipate which channel the next concept will occupy before writing to it." This emerged after severing the self-loop via past-only PM.
- **Half-life $t_{1/2}$ correlates with semantics**: Low-half-life prototypes correspond to punctuation and stop words, while high-half-life prototypes correspond to narrative or thematic concepts, providing a readable metric for short- vs. long-range dependencies.
- **Long context is ProtoT's current weakness**: At a fixed $h=256$, increasing the context window from 256 to 2048 actually causes perplexity to rise. Increasing $h$ mitigates this, suggesting the bottleneck lies in the $h/2$ low-rank value projection and PM channel capacity.

## Highlights & Insights
- **Elevating "Interpretability" to an Architectural Inductive Bias**: The traditional Attention + SAE route is "train a black box, then train a second model to interpret it." ProtoT makes the mixer itself a set of "$R$ nameable concept slots," eliminating auxiliary training and achieving higher disentanglement—a promising scalable philosophy.
- **Channel-aligned Softmax as "Inverted Attention"**: While standard attention performs softmax along the token dimension to distribute weights among tokens, ProtoT performs it along the prototype dimension to distribute weights among channels. This axis swap transforms "information aggregation" into "information routing," serving as the pivot for emerging concepts.
- **Severing the input-output link (past-only PM) forces "Predict-and-Consolidate"**: This inductive bias of "deliberately removing shortcuts to force high-level behavior" is elegant and applicable to any sequence model requiring forward planning.
- **Alpha Gate as an Observable Metric**: This zero-overhead engineering trick allows developers to immediately identify "inactive" layers during hyperparameter tuning, which is far more efficient than post-hoc ablation studies.

## Limitations & Future Work
- **Weak Long-Context Scaling**: Perplexity increases with context size at fixed capacity. This is attributed to the $h/2$ low-rank projection and channel dimension bottlenecks; solving this is crucial for competing with SSMs in long-document scenarios.
- **Absolute Performance Lags Behind LLaMA**: Large-scale perplexity (29.5 vs. 25.8) and GLUE scores (67.6 vs. 71.6) show a remaining gap compared to dense attention. The authors acknowledge this as a first-generation design.
- **Throughput Constraints**: While algorithmic FLOPs are lower, PyTorch's self-attention is highly optimized (e.g., FlashAttention). ProtoT's EMA and per-channel softmax kernels have not yet been fully optimized.
- **Reliance on LLM-as-judge**: Concept naming and scoring (GPT-5.1) may involve subjective bias; future work should integrate human annotation and more robust probing tasks.
- **Empirical $R$ Selection**: $R=32$ is an empirical optimum; theoretical guidance for selecting $R$ across different concept densities or tasks is lacking.

## Related Work & Insights
- **vs LLaMA / Standard Self-attention**: LLaMA prioritizes expressivity and engineering (throughput, perplexity) but lacks architectural pressure for disentanglement. ProtoT trades some expressivity for native interpretability and surgical editing.
- **vs SAE**: SAE extracts disentangled features post-hoc. ProtoT embeds this into the training objective. Table 4 shows ProtoT prototypes surpass LLaMA+SAE in coverage/disentanglement without secondary model costs.
- **vs Slot Attention / Perceiver**: Slot Attention uses non-causal GRU iterations; Perceiver latents interact via self-attention ($O(R^2)$). ProtoT prototypes are non-interacting ($O(R)$) and use strictly autoregressive updates via past-only EMA.
- **vs Mamba / DeltaNet**: Performance is intermediate, but ProtoT adds interpretability and editability—features historically missing from the SSM/Linear Transformer route.
- **vs ProtoPNet / ProtoViT**: Earlier methods restricted prototypes to the final classification layer. ProtoT integrates them into *every mixer layer* and transitions successfully from discriminative vision models to autoregressive generative LMs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Channel-aligned softmax + past-only EMA prefix mean + per-layer prototyping represents the first systematic work embedding interpretability into autoregressive LM mixers.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers perplexity, text generation Elo, GLUE, robustness, interpretability metrics, causal intervention, and throughput, though it lacks 1B+ parameter verification.
- Writing Quality: ⭐⭐⭐⭐ Clear formulas, well-mapped motivation/ablation, and persuasive visualizations for predict-and-consolidate and half-life analysis.
- Value: ⭐⭐⭐⭐⭐ Provides a functional, scalable, and editable architectural template for the "interpretable by design" route, directly benefiting alignment and safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](../../ACL2026/interpretability/towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[ICML 2026\] DLLM-JEPA: Joint Embedding Predictive Architectures for Masked Diffusion Language Models](dllm-jepa_joint_embedding_predictive_architectures_for_masked_diffusion_language.md)
- [\[ICML 2026\] Discovering Implicit Large Language Model Alignment Objectives](discovering_implicit_large_language_model_alignment_objectives.md)
- [\[ICML 2026\] A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents](a_behavioural_and_representational_evaluation_of_goal-directedness_in_language_m.md)
- [\[ICLR 2026\] NIMO: a Nonlinear Interpretable MOdel](../../ICLR2026/interpretability/nimo_a_nonlinear_interpretable_model.md)

</div>

<!-- RELATED:END -->
