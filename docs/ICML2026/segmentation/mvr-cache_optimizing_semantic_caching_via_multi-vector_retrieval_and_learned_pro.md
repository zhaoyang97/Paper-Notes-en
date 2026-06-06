---
title: >-
  [Paper Note] MVR-cache: Optimizing Semantic Caching via Multi-Vector Retrieval and Learned Prompt Segmentation
description: >-
  [ICML 2026][Segmentation][Semantic Caching] MVR-cache upgrades the similarity metric for LLM semantic caching from "single-vector cosine similarity" to "multi-vector MaxSim after learned segmentation." It employs REINFOR…
tags:
  - "ICML 2026"
  - "Segmentation"
  - "Semantic Caching"
  - "Multi-Vector Retrieval"
  - "MaxSim"
  - "Prompt Segmentation"
  - "Reinforcement Learning"
  - "vCache"
date: 2026-05-08
content_hash: ce6dd79c1ec080ab
---

# MVR-cache: Optimizing Semantic Caching via Multi-Vector Retrieval and Learned Prompt Segmentation

**Conference**: ICML 2026  
**arXiv**: [2605.24914](https://arxiv.org/abs/2605.24914)  
**Code**: https://github.com/PKU-SDS-lab/MVR-Cache (Available)  
**Area**: LLM Efficiency / Semantic Caching / Multi-Vector Retrieval  
**Keywords**: Semantic Caching, Multi-Vector Retrieval, MaxSim, Prompt Segmentation, Reinforcement Learning, vCache

## TL;DR
MVR-cache upgrades the similarity metric for LLM semantic caching from "single-vector cosine similarity" to "multi-vector MaxSim after learned segmentation." It employs REINFORCE to train a lightweight segmentation model, increasing the cache hit rate by up to 37% while maintaining the same error rate upper bound $\delta$.

## Background & Motivation
**Background**: LLM inference is both expensive and slow. Semantic caching is a mainstream approach to cost reduction—historical prompts are embedded into a vector space, and new prompts are served directly from the cache if they are "sufficiently similar" to a cached entry. Production systems like Azure, LiteLLM, and GPTCache use this. The recent vCache (Schroeder et al., 2025) further learns adaptive thresholds for each prompt, providing a theoretical $(1-\delta)$ guarantee on the final error rate.

**Limitations of Prior Work**: Existing methods almost exclusively rely on the **cosine similarity of the entire prompt**. For complex prompts, a single global vector fails to capture the **critical sub-segments** that determine whether LLM responses will be identical. The counter-example in Figure 1 of the paper is intuitive: a positive movie review $x$ and a negative review $x_1$ may have high cosine similarity due to shared salient keywords like "crime drama," but their emotional polarities differ. Improperly labeling this as a cache hit would contaminate the output.

**Key Challenge**: Whether a cache should hit is fundamentally determined by "whether two prompts yield equivalent LLM responses," which often depends on **fine-grained matching of local segments**. Single-vector cosine similarity collapses the entire semantics into a single point, causing **fine-grained differences to be averaged out**. This leads to a dilemma: either raising the threshold (making the hit rate too low to be useful) or lowering it (violating the $\delta$ error bound).

**Goal**: Replace the similarity metric with a finer-grained version under the "adaptive threshold + error certificate" framework of vCache, aiming to **significantly improve hit rates at the same $\delta$**. The sub-problems include: (1) what similarity structure to use; (2) how to seamlessly integrate it with vCache's sigmoid calibration and threshold learning; and (3) how to train a **lightweight, variable-length output** segmentation model end-to-end without bloating online inference latency.

**Key Insight**: The Information Retrieval (IR) community has established that decomposing queries/documents into multiple vectors for MaxSim (ColBERT) is more accurate than single-vector cosine retrieval. however, token-level granularity from IR is inefficient and suboptimal for caching, and recent methods like POQD use LLMs for segmentation, which incurs excessive latency. The authors observe that the segmentation strategy can be **directly learned using "cache hit rate (under the $\delta$ constraint)" as a reward**, rather than relying on heuristic splits from IR.

**Core Idea**: A lightweight segmenter $\Theta$ (BERT+LSTM+Pointer Network) is trained to take a prompt and output a set of split positions. The similarity within vCache is replaced by segmented multi-vector MaxSim (SMaxSim). The "combinatorial + non-differentiable" training challenge is addressed via REINFORCE using a proxy BCE loss equivalent to "maximizing hit rate."

## Method

### Overall Architecture
Online Path: A new prompt $x$ arrives $\to$ Segmenter $\Theta$ selects split points from candidate positions (punctuation), dividing $x$ into $m$ segments $\to$ Shared encoder $\mathcal{E}$ embeds each segment into a vector, creating a multi-vector representation $\to$ Compute **symmetrized segmentation-aware MaxSim** (SMaxSim) with each cached prompt $\to$ Retrieve the nearest neighbor $nn_\Theta(x)$ and its score $s_\Theta(x)$ $\to$ Feed into the vCache sigmoid calibration module (Eq. 2-4) to obtain transition probability $\tau$ $\to$ Decide whether to reuse the cached response or fallback to the LLM.

Offline Training Path: The segmentation model is treated as an RL4CO policy $\pi_\Theta$, where the state is the prompt, the action is a "subset of split points," and the reward is the negative of the "similarity alignment BCE loss" calculated with the current segments. REINFORCE optimizes the expected reward, with the nearest neighbor mapping periodically refreshed.

### Key Designs

1.  **Segmentation-aware MaxSim (SMaxSim)**:
    - **Function**: Replaces cosine similarity as the prompt-pair similarity metric within vCache.
    - **Mechanism**: Original ColBERT MaxSim is asymmetric—$\text{MaxSim}(x,x_j)$ only ensures each segment of $x$ matches something in $x_j$, but not vice versa, leading to "short prompt parasitic hits" on long prompts. This paper introduces: (a) computing bidirectional MaxSim normalized by segment count; (b) taking the symmetric average $\text{SMaxSim}_\Theta(x_i,x_j)=0.5\cdot[\tfrac{1}{|x_i|}\text{MaxSim}(x_i,x_j)+\tfrac{1}{|x_j|}\text{MaxSim}(x_j,x_i)]$. Then $s_\Theta(x)=\text{SMaxSim}_\Theta(x,nn_\Theta(x))$ is fed into the vCache sigmoid $\Pr(c=1\mid s)=1/(1+e^{-\gamma(s-t)})$. Calibration parameters $(t,\gamma)$ are estimated via MLE, and the $1-\delta$ certificate is inherited.
    - **Design Motivation**: Retains the MVR advantage of "fine-grained local matching" while adding the semantic requirement that both prompts must match each other. Scale-invariance makes long and short prompts comparable, and the vCache interface remains unchanged.

2.  **Lightweight Variable-Length Pointer-Network Segmenter $\Theta$**:
    - **Function**: Segments a prompt online into a variable number of semantic segments with overhead much lower than the LLM.
    - **Mechanism**: Candidate split points $\mathcal{P}_x$ are restricted to punctuation to ensure splits follow natural semantic boundaries. The architecture uses a BERT encoder $\Theta_1$ for token embeddings $\to$ MLP $\Theta_2$ $\to$ single-layer LSTM $\Theta_3$ for context vector $d_1$ $\to$ Pointer-Network attention $\Theta_4$ to compute $u_{1j}=v^\top\tanh(W_1 h_j + W_2 d_1)$, with probabilities of non-candidate positions masked to zero. After selecting a split point, $d_1'$ is fed back to the LSTM for the next split until a `<stop>` token is generated.
    - **Design Motivation**: Pointer Networks naturally handle "input-dependent output length." The model consumes only 500-600 MB of GPU memory, making its overhead negligible compared to LLM calls.

3.  **RL4CO Training for Hit-Rate Optimization**:
    - **Function**: Enables the segmentation strategy $\Theta$ to learn splits that directly maximize the hit rate under the $\delta$ constraint.
    - **Mechanism**: Theoretically (Thm 3.3), minimizing the proxy BCE loss $\sum \ell_{\mathrm{BCE}}(\mathcal{L}(\text{SMaxSim}_\Theta(x_i,x_j);t_i,\gamma_i),c_j)$ is equivalent to maximizing the vCache hit rate. Since the output of $\Theta$ (discrete split sets) is non-differentiable, segmentation is treated as a policy $\pi_\Theta(\vec{p}\mid x)$. REWARD is the negative BCE loss, optimized via REINFORCE: $\max_\Theta \mathbb{E}_{\vec{p_{x_i}},\vec{p_{x_j}}\sim\pi_\Theta}[\text{Reward}]$. MLE for $(t_i,\gamma_i)$ is performed during training, and $nn_\Theta(\cdot)$ is refreshed every $K$ steps.
    - **Design Motivation**: Directly using hit rate as a reward lacks a gradient; the BCE proxy provides a theoretical guarantee and dense reward signal. REINFORCE plus periodic NN refreshes makes the combinatorial optimization problem tractable.

### Loss & Training
The proxy objective is $\sum_i\sum_{nn_\Theta(x_j)=x_i}\ell_{\mathrm{BCE}}(\mathcal{L}(\text{SMaxSim}_\Theta(x_i,x_j);t_i,\gamma_i),c_j)$, where $c_j$ is a binary label indicating if the LLM responses for $x_j$ and its nearest neighbor $x_i$ are identical (via exact string match). Only **3K** training samples per dataset are needed to train the model.

## Key Experimental Results

### Main Results
Four datasets: **SemCacheClassification (45K)**, **SemCacheSearchQueries (150K)**, **PromptBench (38K)**, and **QNLI (29K)**. Embedding: BGE; LLM: GPT-4o-mini; $\delta=0.01$. Baselines: vCache, ColBert+vCache, POQD+vCache.

| Dataset | Protocol | Hit Rate Gain vs vCache | Error Rate |
| :--- | :--- | :--- | :--- |
| SemCacheSearchQueries | always-cache | **+37%** (Max Gain) | $<\delta=0.01$ |
| SemCacheClassification | cache-on-miss | +9% (≈ 4.1K GPT-4 calls saved) | $<\delta$ |
| PromptBench | cache-on-miss | Leads all baselines | $<\delta$ |
| QNLI | cache-on-miss | Significantly higher than baselines | $<\delta$ |

End-to-end Latency (Table 1, minutes, parentheses show algorithm overhead excluding LLM):

| Method | SemCacheClassification | SemCacheSearchQueries | PromptBench | QNLI |
| :--- | :--- | :--- | :--- | :--- |
| vCache | 408.49 (23.21) | 6361.52 (69.77) | 1870.57 (19.58) | 1536.00 (14.10) |
| ColBert | 501.46 (25.84) | 6521.89 (130.00) | 2294.38 (150.32) | 1626.37 (39.28) |
| POQD | 971.51 (492.92) | 6990.08 (628.33) | 2945.20 (959.60) | 2648.80 (1048.48) |
| **MVR-cache** | **383.32 (34.14)** | **6345.61 (111.26)** | **1866.58 (27.49)** | **1504.43 (17.62)** |

MVR-cache's "algorithmic" overhead is slightly higher than vCache, but higher hit rates reduce total LLM calls, decreasing end-to-end latency by up to 6%. POQD's LLM segmentation overhead often exceeds the LLM inference itself.

### Ablation Study
| Configuration | Key Finding |
| :--- | :--- |
| MVR-cache (full) | Highest hit rate with segmentation model + SMaxSim + RL |
| w/ ColBERT token-level | Significantly lags behind MVR-cache; learned semantic splits are superior |
| w/ POQD LLM segmenter | Similar hit rate but latency explosion; unacceptable for online use |
| Unidirectional MaxSim | Increased false positives; validates necessary of SMaxSim normalization |
| Weak Supervision | Saves **80.4%** GPT-4 calls using GPT-4o-mini as a proxy labeler |
| Cross-distribution | Segmentation model generalizes well from PromptBench to QNLI |

### Key Findings
- Segmentation granularity is the decisive factor: token-level is too fine, whole-prompt is too coarse; learned semantic-level (punctuation-based) segmentation is the "sweet spot."
- The "adaptive threshold + error certificate" framework of vCache is a plug-and-play container; upgrading the similarity $s$ automatically upgrades the error guarantees.
- Minimal training data (3K) and weak supervision options mean low one-time human cost, quickly offset by saved LLM fees.

## Highlights & Insights
- **The equivalence theorem between "hit rate maximization" and "BCE minimization"** serves as a vital bridge between RL rewards and system metrics. This "theoretical proxy loss + RL" formulation is reusable for other non-differentiable system optimizations (e.g., cache eviction, scheduling).
- **Pointer Network + candidate masking** is a clean solution for "variable-length selection within constrained semantic boundaries," better fitting the inductive bias of segmentation than BIO tagging or binary classification.
- **Symmetric normalization of MaxSim** is a simple but effective modification that solves the "long-short prompt" false hit problem inherent in multi-vector retrieval.

## Limitations & Future Work
- Segmentation is currently limited to punctuation boundaries, which may lack flexibility for Chinese, code, or long unpunctuated instructions.
- Training depends on "exact string match" for LLM responses; open-ended generation tasks with high variance might introduce label noise, requiring LLM-as-judge or embedding-based equivalence.
- Periodic nearest-neighbor refreshes ($K$ steps) scale with cache size; incremental maintenance is needed for larger scales.
- Theoretical guarantees rely on the Gaussian assumption for $\Pr(s\mid c)$. While empirically justified, its validity for extreme non-Gaussian distributions remains an open question.

## Related Work & Insights
- **vs vCache** (Schroeder et al., 2025): vCache moves from global to per-prompt thresholds; MVR-cache keeps the framework but upgrades the internal similarity metric.
- **vs ColBERT** (Khattab & Zaharia, 2020): ColBERT uses token-level asymmetric MaxSim; MVR-cache uses learned semantic splits and symmetric MaxSim for the "bidirectional equivalence" required in caching.
- **vs POQD** (Liu et al., 2025): POQD uses LLMs for segmentation in offline retrieval; MVR-cache focuses on online low latency and "caching hit rate" rewards.
- **vs GPTCache / Wallmartcache**: These use static thresholds; MVR-cache + vCache represents the new generation of adaptive solutions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Love Me, Love My Label: Rethinking the Role of Labels in Prompt Retrieval for Visual In-Context Learning](../../CVPR2026/segmentation/love_me_love_my_label_rethinking_the_role_of_labels_in_prompt_retrieval_for_visu.md)
- [\[AAAI 2026\] Guideline-Consistent Segmentation via Multi-Agent Refinement](../../AAAI2026/segmentation/guideline-consistent_segmentation_via_multi-agent_refinement.md)
- [\[CVPR 2026\] MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator](../../CVPR2026/segmentation/matanyone_2_scaling_video_matting_via_a_learned_quality_evaluator.md)
- [\[CVPR 2026\] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth](../../CVPR2026/segmentation/geomprompt_rgbd_segmentation.md)
- [\[NeurIPS 2025\] OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation](../../NeurIPS2025/segmentation/omnisegmentor_a_flexible_multi-modal_learning_framework_for_semantic_segmentatio.md)

</div>

<!-- RELATED:END -->
