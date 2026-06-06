---
title: >-
  [Paper Note] Locally Coherent Parallel Decoding in Diffusion Language Models
description: >-
  [ICML 2026][Code Intelligence][Diffusion Language Models] This paper proposes CoDiLA, which attaches a lightweight autoregressive (AR) model to a masked diffusion language model (DLM). By using "soft embeddings" to recei…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "Diffusion Language Models"
  - "Parallel Decoding"
  - "Local Autoregression"
  - "Soft-Conditioning"
  - "Code Generation"
date: 2026-05-08
content_hash: a7c06805276ea6eb
---

# Locally Coherent Parallel Decoding in Diffusion Language Models

**Conference**: ICML 2026  
**arXiv**: [2603.20216](https://arxiv.org/abs/2603.20216)  
**Code**: https://github.com/IBM/coherent-diffusion-local-autoregression (Available)  
**Area**: LLM Efficiency / Diffusion Language Models / Parallel Decoding  
**Keywords**: Diffusion Language Models, Parallel Decoding, Local Autoregression, Soft-Conditioning, Code Generation

## TL;DR
This paper proposes CoDiLA, which attaches a lightweight autoregressive (AR) model to a masked diffusion language model (DLM). By using "soft embeddings" to receive the marginal distributions from the DLM and performing local autoregressive decoding within small blocks, CoDiLA eliminates the local incoherence caused by parallel sampling while preserving the global bidirectional capabilities of DLM. It establishes a new Pareto frontier in code generation with a throughput gain of $\geq 2\times$.

## Background & Motivation

**Background**: Current State-of-the-Art (SoTA) discrete diffusion language models (Dream-Coder, LLaDA, DiffuCoder, etc.) learn reverse denoising via [MASK] absorbing states. Theoretically, they can predict multiple tokens in parallel, potentially breaking the linear latency bottleneck inherent in AR models. This bidirectional, fillable nature is particularly attractive for code generation.

**Limitations of Prior Work**: Standard DLMs use conditional **marginal** distributions to sample each mask token independently during the reverse step—this is the "Conditional Token Independence" assumption. While acceptable for distant tokens, this assumption causes "one-at-a-time correct, but syntactically nonsensical" outputs for adjacent tokens (e.g., multi-byte words, syntax blocks) decoded simultaneously (Figure 1a, predicting fragmented code like `merge_intervals problem):`). In practice, only a few tokens can be decoded per step to maintain accuracy, negating the sublinear latency advantage.

**Key Challenge**: Parallelism vs. Local Correlation. Parallelism requires the independence assumption, but strong correlations between adjacent tokens are fundamental to syntactic correctness.

**Goal**: Restore joint distribution modeling between simultaneously decoded adjacent tokens without undermining the global non-causal capabilities of DLM (e.g., infilling, bidirectional planning).

**Key Insight**: The authors elevate "diffusion" from the token level to the **block** level—maintaining joint modeling within blocks while keeping blocks independent according to the DLM framework. Theoretically, it is proven that the block independence bias strictly lowers the NELBO (Theorem 3.2). However, directly modeling the $|V|^B$-dimensional joint distribution leads to combinatorial explosion. Thus, the joint modeling within blocks is "outsourced" to a small AR model, which performs causal decoding only within a small window of $B$ tokens. The latency of a single AR call is limited by $B$ rather than the total sequence length $L$.

**Core Idea**: DLM acts as the global drafter, and AR acts as the local refiner. The marginal probability vectors from the DLM are fed into the AR model as **soft embeddings** to produce locally coherent blocks.

## Method

### Overall Architecture
The input is a sequence $x_t$ containing [MASK]. The DLM backbone (a bidirectional Transformer with parameters $\psi$, e.g., Dream-Coder-Instruct-7B) computes the marginal distribution $\pi^j_\psi(x_t) \in \Delta^{|V|-1}$ for each position in a single forward pass. The sequence is divided into contiguous blocks $b^i_t$ of length $B$. For each block to be decoded, its $B$ marginal distributions are used to query the embedding table of a small AR model (parameters $\phi$, e.g., Qwen3-0.6B) to obtain $B$ **soft embeddings**. This sequence of soft embeddings, wrapped between `<think>` and `<\think>` boundary tokens, is fed into the AR model to autoregressively decode the actual tokens for that block. The final joint probability is $p_\theta(b^i_0 \mid x_t) = p^{\text{AR}}_\phi(b^i_0 \mid \pi_\psi(x_t))$, with parameters $\theta = [\psi, \phi]$. During training, the DLM is frozen, and only the AR model is trained. During inference, blocks are selected for decoding based on a "lowest block entropy" confidence schedule.

### Key Designs

1.  **Block-level Diffusion and the NELBO Gap (Theoretical Foundation)**:
    - **Function**: Relaxes the token-level independence assumption to "block independence, intra-block jointness," providing a valid training objective for soft-conditioned AR.
    - **Mechanism**: Splits $x_0$ into $L/B$ blocks $b^i_0 \in W = V^B$ and factorizes the reverse process by blocks $p_\theta(x_0\mid x_t) = \prod_i p_\theta(b^i_0\mid x_t)$. The per-step loss in the NELBO retains the cross-entropy form of token-independent models $L_t = \mathbb{E}_{q(x_t\mid x_0)}\left[\sum_i -\delta_{x^i_t,[\text{MASK}]} \frac{\alpha_{t-1}-\alpha_t}{1-\alpha_t} \log p_\theta(x^i_0\mid x_t)\right]$. Theorem 3.2 proves that $B_1 - B_B = \sum_{t,i}\big(\sum_j H[x^{(i-1)B+j}_{t-1}\mid x_t] - H[b^i_{t-1}\mid x_t]\big) \geq 0$, indicating that the intra-block "total correlation" is exactly the irreducible error paid by token-level models.
    - **Design Motivation**: Previous works (Huang 2022, Liu 2025a, etc.) only discussed the specific case of $B=1$. This work quantifies the improvement for $B > 1$ for the first time, providing solid theoretical support for block-based decoding. Figure 3 empirically demonstrates that larger $B$ leads to lower training PPL without signs of saturation.

2.  **Soft-Conditioning Interface (The "Connector" of CoDiLA)**:
    - **Function**: Establishes a high-bandwidth channel between DLM marginals and the AR input space, allowing the AR to "hear" the full marginal signal from the DLM while reusing pretrained AR semantic embeddings.
    - **Mechanism**: Treats $\pi^j_t$ as the expected weights over the AR embedding table $E_\phi$, calculating a soft embedding $e^j_t = \sum_{v \in V} [\pi^j_t]_v \cdot E_\phi(v)$. The sequence $[E_\phi(\langle\text{think}\rangle), e^1_t, \ldots, e^B_t, E_\phi(\langle\backslash\text{think}\rangle)]$ is fed to the AR model. Theorem 3.3 proves: (i) coding on **complete** marginals allows a $\phi$ such that $p^{\text{AR}}_\phi(\cdot\mid\pi) = q(\cdot)$; (ii) using only top-$k$ truncation restricts the recoverable distribution to the Fréchet class; (iii) there exists a $q$ such that its global mode $b^* = \arg\max_b q(b)$ is permanently excluded by top-$k$, introducing irreducible bias.
    - **Design Motivation**: Works like APD or FlashDLLM use top-1/top-$k$ truncation as AR input. While simple, this theoretically excludes the "most coherent sequences that should have been chosen" from the solution space. Soft-conditioning ensures expressive sufficiency and compresses high-dimensional probability vectors into the AR's existing embedding space, avoiding pretraining from scratch. The `<think>` boundary tokens are a critical detail—removing them causes the NELBO to jump from 13.6 to 15.5 ($B=4$).

3.  **Three-Tier Generation Scheduling (Static / Dynamic / Verification)**:
    - **Function**: Converts the trained CoDiLA into various inference strategies to select points on the accuracy-throughput Pareto curve while maintaining DLM's bidirectional infilling capabilities.
    - **Mechanism**: Defines block entropy $h^i_t(k) = \frac{1}{k}\sum_j H[p_\theta(x^{(i-1)B+j}_0\mid x_t)]$. (a) **Static Parallel**: Decodes one full block per step—selecting the block with the lowest $h^i_t(B)$; a local window (10 blocks) prevents premature EOS. (b) **Dynamic Parallel**: Decodes the longest partial block ($k \leq B$) that satisfies $h^i_t(k) \leq \tau$. When $k=1$, it reverts to token-level DLM confidence sampling. (c) **AR Verification Mode**: AR only compares its top-1 against the DLM top-1; if they diverge, it penalizes the confidence of that position. This can be non-intrusively embedded into any confidence schedule.
    - **Design Motivation**: A single schedule cannot cover all scenarios. Static parallel is simple but accuracy drops as $B$ increases; dynamic parallel recovers accuracy by "parallelizing less where hard and more where easy"; verification mode suits global planning tasks (ParallelBench) where AR acts as a judge rather than a generator.

### Loss & Training
The model is trained end-to-end using the cross-entropy NELBO in Equation (2). The DLM backbone is frozen, and only the AR model is fine-tuned. Forward noise addition is changed from token-level to **block-level masking** (masking $B$ contiguous tokens at once), enabling the AR model to learn predicting entire blocks based on DLM soft embeddings. Training was conducted on Ling-Coder-SFT for 32k steps per $B$ on a single A100 80GB using PyTorch 2.7 and bf16.

## Key Experimental Results

### Main Results

The main experiment compared the baseline with ADJUST (Bansal & Sanghavi 2025) on Dream-Coder-Instruct-7B using static parallel decoding ($K=B$ per step). The following table shows the syntax error rate on HumanEval (percentage of scripts where code could not be extracted), illustrating the "local coherence" problem:

| Model | K=B=2 | K=B=4 | K=B=8 |
|-------|-------|-------|-------|
| Dream-Coder-Instruct-7B | 18 | 38 | 70 |
| **CoDiLA** (Ours) | **4** | **13** | **16** |
| Gain (pp) | −14 | −25 | **−54** |

In the highest parallelism setting ($B=8$), the syntax error rate dropped from 70% to 16%, a 54 percentage point improvement. Figure 4 reports Pass@1 vs. Throughput (tokens/sec, batch=1, A100) on HumanEval/+, MBPP/+, and BigCodeBench—CoDiLA occupies the outermost Pareto frontier across all six benchmarks. The authors clarify that the accuracy gain is not solely due to the AR model; Qwen3-0.6B alone achieves only 35% on HumanEval, whereas CoDiLA ($B \leq 4$) significantly exceeds the capabilities of the 0.6B model.

HumanEval-Infilling (Verification of bidirectional capability):

| Model | Pass@1 (%) | Tokens/step |
|-------|-----------|-------------|
| Deepseek-Coder-6.7B | 45.7 | 1 |
| Qwen2.5-Coder-7B | 58.7 | 1 |
| DreamOn (K=1, Seq.) | 62.5 | 1 |
| DreamOn (K=2, Par.) | 53.1 | 2 |
| DreamOn + CoDiLA ($\tau=0.2$) | **62.5** | 1.3 |
| DreamOn + CoDiLA ($\tau=0.5$) | 61.5 | **1.5** |

Under parallel decoding, accuracy is maintained while parallelism increases by 1.3–1.5×.

### Ablation Study

| Config | Key Finding | Description |
|--------|-------------|-------------|
| Soft- vs Top-K Conditioning | Top-K performs worse | Empirically supports Theorem 3.3: Information truncation introduces irreducible bias. |
| Removing `<think>` tags | NELBO 13.6 → 15.5 ($B=4$) | Boundary tokens are necessary to activate the pretrained AR's reasoning path. |
| AR size: 0.6B → 1.7B → 4B | No consistent gain | Performance does not rely on scaling AR parameters; 0.6B is sufficient. |
| Single $B=8$ vs Two $B=4$ blocks | Single block 8 pp higher | Intra-block jointness is the key to capturing parallel gains. |
| Candidate Range: 10 vs 50 blocks | Throughput diff < 15% | Local window selection is robust. |
| Generation Order Spearman | Decreases with $B$ | CoDiLA does not pull the global order back to left-to-right; DLM's arbitrary order advantage is preserved. |
| Batch size = 8 | AR overhead almost amortized | Small extra latency at bs=1 disappears at bs=8. |

### Key Findings
- **Larger blocks result in lower training loss without saturation** (Figure 3, $B \in \{2, 4, 8, 16, 32\}$ under 32-token continuous mask settings), providing the first empirical evidence for the directionality of the equality $B_1 - B_B$ in Theorem 3.2.
- **The root of accuracy improvement is local coherence, not AR capability**: The 0.6B AR model alone scores only 35% on HumanEval, but CoDiLA rescues the 7B DLM from a 70% error rate to 16%. The 0.6B model acts as a coherence judge and local refiner rather than a new generator.
- **Dynamic parallelism eliminates accuracy degradation**: A $B=4$ block with threshold $\tau$ scheduling matches sequential ($K=1$) accuracy while achieving $\geq 2\times$ speedup.
- **Bidirectional capabilities are preserved**: CoDiLA maintains or improves DLM performance on non-causal tasks like infilling and ParallelBench, as AR is causal only within blocks, while inter-block communication remains bidirectional via DLM.

## Highlights & Insights
- **Rare perfect loop between theory and engineering**: From the NELBO inequality (Theorem 3.2) to Fréchet class truncation bias (Theorem 3.3), training loss curves (Figure 3), and downstream accuracy—the logic holds together across all stages.
- **"Soft-embedding + `<think>` envelope" is an ingenious engineering trick**: Projecting DLM marginals into the AR's own embedding space allows the AR to understand the signal in its "own language." Using thinking tags "disguises" the input as internal thought, activating the AR's introspective decoding path.
- **The "DLM for drafting, AR for execution" division of labor** has significant transfer value for other tasks requiring "global planning + local precision" (e.g., SQL, HTML, JSON schema generation). AR handles short local segments, while DLM handles global editing and infilling.
- Compared to ADJUST, APD, and TiDAR, CoDiLA is the only solution that is **both fast and retains bidirectional capabilities**. Other solutions either sacrifice non-causality (reverting to left-to-right) or incur high training costs.

## Limitations & Future Work
- Context-dependent limitations: The block length $B$ is currently fixed. Future work could explore **semantic adaptive block lengths**. While large $B$ reduces loss, the serial AR latency eventually offsets parallel gains, requiring manual tuning of $B$.
- Observational limitations: (i) The "two-stage" setup assumes the DLM is already well-trained; effectiveness on small DLMs is unverified. (ii) Soft-conditioning requires identical tokenizers for AR and DLM. (iii) Experiments focus on code; whether local incoherence is equally problematic in natural language remains open. (iv) Batch=1 evaluations might overstate latency benefits, though amortization at batch=8 was noted.

## Related Work & Insights
- **vs ADJUST (Bansal & Sanghavi 2025)**: ADJUST uses a single-layer DLM as an auxiliary model, requiring pretraining from scratch and repeated full attention. CoDiLA uses a pretrained AR restricted to blocks, saving costs and achieving higher accuracy.
- **vs APD (Israel 2025) / FlashDLLM (Hu 2026) / TiDAR (Liu 2025b)**: These methods use AR for left-to-right verification, essentially turning the DLM into a quasi-AR model and discarding bidirectional/infilling capabilities. CoDiLA preserves these by keeping AR local.
- **vs Discrete Copula Diffusion (Liu 2025a)**: Shares the idea of merging DLM marginals with AR joints, but Copula requires multiple full-sequence passes. CoDiLA's soft-conditioning is a single-pass injection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "Global DLM + Local AR Soft-Conditioning" is the first end-to-end solution to integrate theory, interface, training, and scheduling.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage across benchmarks and tasks, but lacks high-concurrency throughput data for service scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical characterization and excellent motivation-to-method bridging.
- Value: ⭐⭐⭐⭐⭐ A must-read for DLM researchers seeking sublinear latency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Training Large Language Models To Reason In Parallel With Global Forking Tokens](../../ICLR2026/code_intelligence/training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)
- [\[ACL 2026\] Static Program Slicing Using Language Models With Dataflow-Aware Pretraining and Constrained Decoding](../../ACL2026/code_intelligence/static_program_slicing_using_language_models_with_dataflow-aware_pretraining_and.md)
- [\[ICML 2026\] Poison with Style: A Practical Poisoning Attack on Code Large Language Models](poison_with_style_a_practical_poisoning_attack_on_code_large_language_models.md)
- [\[ICML 2026\] Entropy-informed Decoding: Adaptive Information-Driven Branching](entropy-informed_decoding_adaptive_information-driven_branching.md)
- [\[ACL 2026\] Ro-SLM: Onboard Small Language Models for Robot Task Planning and Operation Code Generation](../../ACL2026/code_intelligence/ro-slm_onboard_small_language_models_for_robot_task_planning_and_operation_code_.md)

</div>

<!-- RELATED:END -->
