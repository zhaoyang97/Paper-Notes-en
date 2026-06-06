---
title: >-
  [Paper Note] Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing
description: >-
  [ICML 2026][LLM Efficiency][Multi-token prediction] This paper proposes ESP (Embedding-Space Probing): without modifying any weights or training auxiliary models…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Multi-token prediction"
  - "training-free"
  - "embedding-space probing"
  - "speculative decoding"
  - "dynamic draft tree"
date: 2026-05-08
content_hash: f646e206a91881b1
---

# Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing

**Conference**: ICML 2026  
**arXiv**: [2603.17942](https://arxiv.org/abs/2603.17942)  
**Code**: TBD  
**Area**: LLM Efficiency / Speculative Decoding / Multi-Token Prediction  
**Keywords**: Multi-token prediction, training-free, embedding-space probing, speculative decoding, dynamic draft tree  

## TL;DR
This paper proposes ESP (Embedding-Space Probing): without modifying any weights or training auxiliary models, it injects the "mean prompt embedding" as mask tokens into the input sequence of a frozen LLM. By probing multiple future tokens in a single forward pass and performing lossless speculative verification using the base model itself, ESP achieves 7–11% higher average acceptance lengths and 15–19% higher throughput than training-free baselines (LADE / STAND / PLD) on LLaMA3 and Qwen3.

## Background & Motivation

**Background**: Autoregressive decoding produces only one token per step, leading to significant GPU parallelism waste. Mainstream multi-token prediction (MTP) or speculative decoding schemes fall into two categories: (i) adding MTP heads to the main model and retraining (Medusa, Gloeckle et al.), or (ii) introducing an independent small draft model for speculation (Leviathan, Cai et al.). Both require dataset construction, architectural tuning, and expensive GPU training, adding ~400M extra parameters that are unfriendly to edge devices.

**Limitations of Prior Work**: Truly "training-free" baselines are scarce—PLD relies on n-gram copying from prompts, STAND uses adaptive n-gram caches, and LADE constructs drafts via Jacobi iteration. While effective on tasks with high n-gram repetition (e.g., coding, RAG), their acceptance rates drop significantly in open tasks like writing or math/reasoning, and they require maintaining online n-gram caches. Probing works like Future Lens observed that "future token information is latent within LLMs" but treated it as a diagnostic phenomenon rather than a decoding algorithm.

**Key Challenge**: To achieve "no retraining + no auxiliary models + losslessness," one must predict multiple future tokens simultaneously in a single forward pass using only the frozen model. However, since LLMs are trained for next-token prediction, how can they be "tricked" into outputting $k$ tokens at once?

**Goal**: (1) Identify a token representation that, when inserted into a sequence, allows the LLM to output the distribution of the "future $i$-th step" at that position; (2) Organize multiple candidates into a tree with budget-controlled expansion/pruning strategies; (3) Use the base model to verify candidates for guaranteed losslessness; (4) Provide a theoretical explanation for why this probe works.

**Key Insight**: The authors observe that during computation across decoder layers, the hidden states of "placeholder tokens" are **gradually pulled toward the hidden states of actual future tokens**. By using a "semantically neutral but prompt-aligned" vector as a mask token, deeper layers automatically align it with future token representations, allowing the LM head to naturally rank correct future tokens within the Top-K.

**Core Idea**: Use the "mean prompt embedding" as a soft mask token to probe future $k$ token logits directly in the embedding space; organize candidates via dynamic tree expansion; and perform parallel verification with the main model. The entire pipeline is completely training-free, requires no draft model, and is lossless.

## Method

### Overall Architecture
After a frozen LLM $f_\theta$ receives the prompt $x_{1:t}$, ESP does not perform direct next-token decoding. Instead: (1) It synthesizes $k$ mask tokens $m_1,\dots,m_k$ in the embedding space and appends them to the sequence; (2) It executes a single forward pass to obtain logits at all mask positions, sampling Top-K candidates via dynamic tree expansion to form a "draft token tree"; (3) It applies a simple pruning rule to remove redundant branches identical to parent nodes; (4) The entire draft tree is sent to the same $f_\theta$ for parallel verification (standard speculative decoding practice), where tokens are accepted if they match exactly and truncated otherwise; (5) Each accepted token triggers an update of the corresponding mask token (EMA-style fusion with the latest generated token embedding) for the next round. This process is completed in a single forward pass via a customized "tree attention mask + position indices."

### Key Designs

1. **Soft Mask Token Injection + EMA Online Update**:
    - **Function**: Constructs "placeholder vectors" capable of probing multiple future tokens from a frozen LLM and adapts them during the generation process.
    - **Mechanism**: Initializes all mask tokens with the mean prompt embedding $m_i = \frac{1}{t}\sum_{j=1}^t \mathbf{e}_j$. This ensures the placeholder follows the same statistical distribution as the current prompt, which is more stable than "hard initialization" (taking the last K embeddings) or sampling from the global embedding distribution. During generation, for each accepted token $x_{t+s}$, the mask token is updated via EMA: $m_i[s+1] = m_i[s] + \lambda(\mathbf{e}_{t+s} - m_i[s])$ ($\lambda = 0.1$), bleeding the latest context into the mask representation. Note: all future trajectories within the same tree share the same $m_i$ values; differentiation occurs via position IDs and tree-attention paths.
    - **Design Motivation**: Using Dolly-Databricks data, the authors observed that for accepted tokens, the cosine similarity between the mask token and the true future token hidden states rises steadily to ~0.45 starting from layer 15; rejected tokens stall at ~0.35. Lemma 3.1 proves: if $\cos(h_m, h_v) \geq \delta^*$, the true future token will fall within the Top-K logits of the mask token. Mean-prompt initialization maximizes this "inter-layer alignment," providing the theoretical foundation for this training-free method.

2. **Cumulative Probability-Based Dynamic Draft Tree Expansion (Algorithm 1)**:
    - **Function**: Adaptively determines tree width allocation under a fixed block complexity budget (total tokens per forward pass), avoiding manual Top-K grid tuning.
    - **Mechanism**: Employs Top-1 expansion—only the highest probability nodes at each layer are expanded further. Specifically, given a budget $B$ and $k$ mask tokens, for each layer $i$, it samples $B-i$ candidates for all current frontier nodes, updates cumulative probabilities $P(c) = P(n) \cdot P(t_j \mid l_n)$, and keeps the Top-$(B-i)$ for the next layer. Finally, it retains the $B-1$ trajectories with the highest cumulative probabilities. The closed-form expression for block complexity is $\text{Block Complexity} = (k+1)(1 + \sum_{i=1}^k K_i)$. Results show that the dynamic strategy matches or exceeds the best static $[K_1, K_2]$ configurations on LLaMA3 while saving offline search costs.
    - **Design Motivation**: The decision to expand "wide or deep" depends heavily on the prompt. Open tasks (writing/reasoning) favor "wide and shallow" exploration, while closed tasks (math/translation) favor "narrow and deep" exploitation. Fixed Top-K settings inevitably fail on certain tasks; using cumulative probability for budget allocation allows the model to decide where to spend its budget, with the $B-i$ decay naturally encouraging early branching and late-stage focus.

3. **GPU-Friendly Static Tree Attention and Position Index Implementation**:
    - **Function**: Converts "tree attention mask + position id" construction from serial node iteration to cacheable batch operations, eliminating hidden overheads of tree decoding.
    - **Mechanism**: Instead of regenerating masks per step, the attention mask is cached and **incrementally updated by appending columns**. Position IDs are reused via simple offsets. Combined with a layout where mask tokens are placed at the end of the sequence (Figure 3), a single forward pass covers the "last accepted token + all draft tree nodes + all mask tokens."
    - **Design Motivation**: Table 4 shows that with a naive implementation, LLaMA3.1-8B-Instruct at BC=60 achieves only 1.05–1.07× end-to-end speedup (overhead consumes most gains). The efficient implementation jumps to 1.35–1.38×, providing ~21% average gain and up to 30% at BC=60. This highlights that for training-free MTP, the throughput bottleneck is often attention mask construction rather than token acceptance rate.

### Loss & Training
**Completely training-free**. It introduces no trainable parameters and does not modify LLM weights. The only hyperparameters are the EMA coefficient $\lambda = 0.1$, the number of mask tokens $k$ (found to be optimal at $k=1, 2$; $k=3$ degrades performance as LLMs are only trained for next-token), and block complexity $B \in \{10, 30, 60\}$. Verification follows standard speculative decoding matching to ensure the generation distribution remains identical to the original autoregressive model (lossless).

## Key Experimental Results

### Main Results
Evaluated on SpecBench (writing, roleplay, coding, translation, summarization, math/reasoning, RAG) against PLD, STAND, and LADE. Reported metrics are average acceptance length $\tau$ and end-to-end wall-time speedup S/R.

| Model | BC | PLD $\tau$ / S/R | STAND $\tau$ / S/R | LADE $\tau$ / S/R | **ESP $\tau$ / S/R** |
|------|----|---|---|---|---|
| LLaMA3.1-8B-Instruct | 30 | 1.44 / 1.23× | 1.58 / 1.10× | 1.45 / 1.06× | **1.63 / 1.35×** |
| LLaMA3.1-8B-Instruct | 60 | 1.44 / 1.23× | 1.64 / 1.14× | 1.60 / 1.14× | **1.71 / 1.38×** |
| Qwen3-8B | 60 | 1.31 / 1.12× | 1.48 / 1.06× | 1.73 / 1.21× | **1.74 / 1.43×** |
| Qwen3-32B | 60 | 1.29 / 1.09× | 1.48 / 1.13× | 1.69 / 1.31× | **1.70 / 1.48×** |
| LLaMA3.2-3B-Instruct| 60 | 1.43 / 1.19× | 1.62 / 1.07× | 1.57 / 1.10× | **1.63 / 1.22×** |

ESP achieves the highest (or joint highest) $\tau$ and S/R across all models and BC settings. Compared to LADE, ESP's $\tau$ is 7–12% higher on LLaMA3 and 7–8% higher on Qwen3, with throughput gains of 15–19% over the strongest baselines. At BC=60, it reduces forward model calls by up to 42%.

### Ablation Study
| Configuration | LLaMA3.2-3B $\tau$ (BC=60) | LLaMA3.1-8B $\tau$ (BC=60) | Description |
|------|------|------|------|
| Mean (soft init) | **1.67** | **1.71** | Full method, mean-prompt initialization |
| Sample (embedding dist) | 1.65 | 1.69 | Sampled from $\mathcal{N}(\mu, \sigma^2 I)$ of embedding table |
| Last K (hard init) | 1.62 | 1.67 | Take embeddings of last K prompt tokens |
| 1 mask token $[29]$ | **1.65** | **1.73** | BC=60, single mask token |
| 2 mask tokens $[15,4]$ | 1.63 | 1.71 | Two mask tokens + dynamic branches |
| 3 mask tokens $[7,5,3]$ | 1.51 | 1.57 | Three mask tokens, **significant regression** |
| Efficient attention impl | 1.22× / 1.38× S/R | — | Gain over naive implementation |
| Naive attention impl | 0.96× / 1.07× S/R | — | Node-by-node mask construction |

### Key Findings
- **Mean-prompt soft init > other initializations**: Consistently higher by 0.02–0.05 $\tau$, validating Lemma 3.1 regarding "inter-layer cosine alignment." Placeholders following the prompt distribution are more easily pulled toward true future states.
- **More mask tokens are not necessarily better**: $k=1$ is often optimal; $k=3$ causes a drop of 0.1+ $\tau$. Deep probes fail to align because the base LLM is only trained for next-token prediction. Open tasks prefer $k=1$ (exploration), while closed tasks prefer $k=2$ (exploitation).
- **Dynamic tree matches/beats static trees**: At BC=60, dynamic (1.630) vs. best static $[15,4]$ (1.631). It removes the need for offline grid searches.
- **Engineering speedup ≈ Algorithmic speedup**: Efficient attention implementation alone contributes ~21% throughput, suggesting that speculative decoding research should not ignore construction costs.
- **Task correlation**: STAND (n-gram copy) slightly wins in coding/RAG/summarization due to high text repetition. ESP excels in math/reasoning where "models must actually generate" ($\tau=1.81$ on LLaMA3.1-8B).

## Highlights & Insights
- **The "Probe-as-Decoding" Paradigm Shift**: Previous probing works (e.g., Future Lens) treated "LLM internal encoding of future tokens" as an interpretability phenomenon. This work is the first to engineer it into a practical decoding algorithm without training.
- **Theory-Phenomenon Loop**: The paper observes mask tokens converging to future states, provides formal guarantees via Lemma 3.1, and finishes with mean-prompt experiment results—creating a compelling narrative that the method is not just ad-hoc tuning.
- **Block Complexity Abstraction**: Explicitly defining "tokens processed per forward pass" ($B$) allowed for fair comparisons $(k+1)(1 + \sum K_i)$, preventing "improvement by larger tree" biases.
- **Transferable EMA Updates**: The idea of using prompt statistics for placeholders and drifting them with EMA could be applied to prompt-tuning, continuous prompt optimization, or slot filling in retrieval-augmented decoding.

## Limitations & Future Work
- Underperforms STAND/PLD on high n-gram repetition tasks (coding/RAG); a hybrid strategy combining ESP with n-gram caches could be considered.
- Performance drops at $k=3$ because the base LLM lacks multi-token training. Beyond $k=2$, lightweight fine-tuning may be necessary, though it breaks the "training-free" premise.
- Evaluation was limited to single-A100/H100 with lengths of 100/256 tokens. Behavior in long-context (>1k), batch size > 1, or multi-GPU pipeline scenarios remains uninvestigated.
- Robustness of the mean-prompt initialization on extreme prompts (e.g., pure code or number strings) relative to Lemma 3.1 needs further study.

## Related Work & Insights
- **vs LADE (Lookahead Decoding)**: LADE uses Jacobi iterations for simultaneous guessing; ESP uses embedding-space mask probes. ESP achieves 7–11% higher $\tau$ on LLaMA3 and 7-8% on Qwen3 with no need for n-gram pools.
- **vs Medusa / Cai et al.**: These require training ~400M parameters. ESP uses zero extra parameters and memory, making it superior for edge deployment despite a lower acceptance ceiling.
- **vs PaSS / Future Lens**: PaSS introduces markers and requires fine-tuning; Future Lens is analysis-only. ESP requires no tuning or special vocabulary.
- **vs STAND / PLD**: These rely on "copying" and fail in reasoning/writing tasks where ESP thrives by leveraging the model's generative capacity.

## Rating
- Novelty: ⭐⭐⭐⭐ The shift to "embedding-space probing as decoding" is clean, with a solid theory-phenomenon loop.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various models, BC settings, and SpecBench tasks, with detailed four-dimensional ablations. (Minor deduction for lack of long-gen/high-batch data).
- Writing Quality: ⭐⭐⭐⭐ Clear progression from motivation to algorithm; Figure 1/3 effectively illustrates the injection and attention mechanisms.
- Value: ⭐⭐⭐⭐ A truly plug-and-play training-free MTP method with direct utility for edge LLM inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](../../NeurIPS2025/llm_efficiency/efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)
- [\[NeurIPS 2025\] L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models](../../NeurIPS2025/llm_efficiency/l-mtp_leap_multi-token_prediction_beyond_adjacent_context_for_large_language_mod.md)
- [\[ICML 2026\] Sparser Block-Sparse Attention via Token Permutation](sparser_block-sparse_attention_via_token_permutation.md)
- [\[ICML 2026\] PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](pipesd_an_efficient_cloud-edge_collaborative_pipeline_inference_framework_with_s.md)
- [\[CVPR 2026\] SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration](../../CVPR2026/llm_efficiency/sparvar_exploring_sparsity_in_visual_autoregressive_modeling_for_training-free_a.md)

</div>

<!-- RELATED:END -->
