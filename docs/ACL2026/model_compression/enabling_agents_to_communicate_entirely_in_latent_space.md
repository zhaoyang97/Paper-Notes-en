---
title: >-
  [Paper Note] Enabling Agents to Communicate Entirely in Latent Space
description: >-
  [ACL 2026][Model Compression][latent space communication] This paper proposes Interlat, a framework enabling LLM agents to communicate entirely in latent space. The sender directly transmits the final-layer hidden states…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "latent space communication"
  - "multi-agent"
  - "hidden state transfer"
  - "information compression"
  - "inference acceleration"
date: 2026-05-08
content_hash: ab8fd0c118a00ae3
---

# Enabling Agents to Communicate Entirely in Latent Space

**Conference**: ACL 2026
**arXiv**: [2511.09149](https://arxiv.org/abs/2511.09149)  
**Code**: [GitHub](https://github.com/XiaoDu-flying/Interlat)  
**Area**: Model Compression
**Keywords**: latent space communication, multi-agent, hidden state transfer, information compression, inference acceleration

## TL;DR

This paper proposes Interlat, a framework enabling LLM agents to communicate entirely in latent space. The sender directly transmits the final-layer hidden states as a continuous representation of its "thoughts"; the receiver interprets these latent messages via a communication adapter and further compresses them to as few as 8 tokens through latent-space reasoning, achieving up to 24× communication speedup while maintaining competitive performance.

## Background & Motivation

**Background**: LLM-based multi-agent systems coordinate tasks through natural language communication. Despite its human readability, natural language constitutes a lossy communication medium—projecting high-dimensional internal states onto discrete tokens discards substantial information.

**Limitations of Prior Work**: (1) Natural language communication offers limited information bandwidth (~15 bits/token vs. ~40k bits/hidden-state), causing much of the reasoning trajectory and nuanced information to be discarded during tokenization; (2) a large portion of generated text serves linguistic coherence rather than task-relevant content, introducing redundancy; (3) the inherent ambiguity of language communication is a primary source of failure in multi-agent coordination; (4) existing hidden-state communication methods rely on single-pass activation grafting or are coupled with language trajectories, requiring specific layer selection.

**Key Challenge**: The majority of LLM computation occurs in a continuous latent space, and internal hidden states carry extraordinarily rich information—yet inter-agent communication forces this information to be compressed into discrete tokens, resulting in severe information loss.

**Goal**: To enable communication between agents to occur entirely in latent space—transmitting continuous hidden states directly rather than discrete tokens—and to achieve efficient communication through compression.

**Key Insight**: Analogous to "telepathy," the framework bypasses symbolic language and directly transmits internal representations. The final-layer hidden state sequences produced during LLM generation are exploited as continuous representations of "thoughts" for transmission.

**Core Idea**: Time-aligned final-layer hidden state sequences serve as latent communication messages. A conditional thought separation loss ensures that the receiver genuinely exploits rather than ignores the latent information. A latent-space reasoning model then compresses long sequences into extremely short latent messages.

## Method

### Overall Architecture

A sender–receiver two-agent setup: the reasoning agent (Sender) generates a plan along with its hidden states $H \in \mathbb{R}^{L \times d}$ → a communication adapter (lightweight self-attention + projection layers) processes the hidden states → the execution agent (Receiver) receives the hidden states and generates actions. After training, an additional compression model can be trained to compress $H_L$ into $H_K$ ($K \ll L$).

### Key Designs

1. **Latent Space Communication with Conditional Thought Separation**

    - **Function**: Ensures that the receiver genuinely leverages task-relevant information contained in latent messages.
    - **Mechanism**: The sender's time-aligned final-layer hidden state sequence $H = [h_1, \ldots, h_L]$ is transmitted, with special tokens `<bop>` and `<eop>` marking communication boundaries. During training, a conditional thought separation loss is minimized: the Jensen–Shannon divergence between the receiver's output distributions conditioned on matched latent states $H$ and mismatched latent states $\tilde{H}$ (from different tasks) is maximized, forcing the model to distinguish between the two.
    - **Design Motivation**: Naive SFT may cause the model to ignore latent inputs and rely solely on the prompt. The conditional separation loss explicitly encourages the model to exploit task-specific information encoded in the latent space.

2. **Plan Alignment Regularization**

    - **Function**: Prevents degenerate patterns during conditional separation training.
    - **Mechanism**: Maximizing separation may cause the model to shift probability mass toward tokens that increase divergence but harm task utility. The output distribution conditioned on the corresponding linguistic plan $P$ is used as a regularizer—a KL divergence constraint aligns the latent-conditioned output with the language-plan-conditioned output, supplemented by logit cosine similarity alignment.
    - **Design Motivation**: Ensures that latent space communication performs no worse than language communication—it should convey the same or more information.

3. **Latent Space Reasoning Compression**

    - **Function**: Compresses long latent messages into extremely short sequences.
    - **Mechanism**: An independent reasoning model $M_\phi$ is trained to autoregressively generate a compact message $H_K$ ($K \ll L$) in latent space by feeding its own hidden states back as the next input embeddings. During training, the receiver is frozen and three loss components are optimized: task loss (preserving downstream performance) + uncertainty-weighted consistency loss (aligning the distributions of the compressed and full messages at positions with informative latent content) + latent geometry alignment loss (preserving global semantic directions).
    - **Design Motivation**: A complete hidden state sequence may span hundreds of steps, introducing communication latency. Autoregressive latent-space reasoning distills information into a small number of steps.

### Loss & Training

Main training: $\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_S \mathcal{L}_{sep} + \lambda_A \mathcal{L}_{align}$, with a stochastic token–latent mixed curriculum to stabilize training. Compression training: $\mathcal{L}_{compress} = \lambda_{task}\mathcal{L}_{task} + \lambda_{pref}\mathcal{L}_{pref} + \lambda_{geom}\mathcal{L}_{geom}$, with the receiver frozen and only the compression model updated.

## Key Experimental Results

### Main Results

**Success Rate of Qwen2.5-7B on Seen/Unseen Tasks**

| Method | Seen Success Rate | Unseen Success Rate |
|--------|------------------|---------------------|
| No-Comm | 62.14 | 62.19 |
| Text (language communication + SFT) | 64.29 | 62.44 |
| CoT (full) | 67.14 | — |
| **Interlat (latent space communication)** | **70.48** | **65.42** |

### Ablation Study

**Communication Compression (Qwen2.5-7B, Seen Tasks)**

| Compression Token Count K | Success Rate | Speedup |
|--------------------------|-------------|---------|
| Full L | 70.48 | 1× |
| 64 | ~70 | ~4× |
| 32 | ~69 | ~8× |
| 16 | ~68 | ~16× |
| **8** | **~66** | **24×** |

**Cross-Model Heterogeneous Communication**

| Sender → Receiver | Latent Communication | Language Communication |
|-------------------|---------------------|----------------------|
| Qwen-7B → Qwen-0.5B | 61.19 | 54.52 |
| LLaMA-8B → LLaMA-8B | 70.71 | 62.86 |

### Key Findings

- Latent space communication (70.48%) substantially outperforms language communication (64.29%) and no-communication (62.14%), confirming that hidden states carry task-relevant information that language cannot express.
- The approach generalizes to cross-model heterogeneous settings (different architectures/sizes), suggesting that the information structure of final-layer hidden states exhibits a degree of cross-model universality.
- Compressing to 8 tokens incurs only ~4% performance degradation (~66% vs. 70.48%) while achieving 24× communication speedup.
- Analysis reveals that agents using latent communication exhibit more exploratory behavior, leveraging task-relevant information in latent space rather than surface-level pattern matching.
- The conditional separation loss is critical—without it, the model tends to ignore latent space inputs.

## Highlights & Insights

- The "telepathy" analogy, though evocative, captures the core insight accurately: communication between LLMs need not pass through human-readable intermediate representations.
- Latent space reasoning compression constitutes a novel form of "information distillation"—performing autoregressive reasoning in continuous space without decoding to tokens.
- A 24× communication speedup carries significant practical implications for the deployment of multi-agent systems.

## Limitations & Future Work

- Validation is limited to the two-agent sender–receiver setting and has not been extended to more complex multi-agent topologies.
- The communication adapter requires training, increasing deployment complexity.
- Latent space communication forgoes human interpretability, making it difficult to debug and audit inter-agent "conversations."
- Security implications of latent space communication remain unexplored.

## Related Work & Insights

- **vs. COCONUT/Thought-of-Thought**: These works perform latent-space reasoning within a single model; Interlat extends this paradigm to inter-agent communication.
- **vs. Ramesh & Li (2025)**: Their approach uses single-pass activation grafting, whereas Interlat transmits complete time-aligned hidden state sequences.
- **vs. Tang et al. (2025)**: Their latent space communication is coupled with language trajectories; Interlat operates entirely within latent space.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Fully latent communication combined with latent-space reasoning compression constitutes a genuinely new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-model, multi-task evaluation, though limited to two-agent scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation and methodology are clearly presented with complete mathematical formulations.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new direction for efficient communication in multi-agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ChemAmp: Amplified Chemistry Tools via Composable Agents](chemamp_amplified_chemistry_tools_via_composable_agents.md)
- [\[ACL 2026\] Latent-Condensed Transformer for Efficient Long Context Modeling](latent-condensed_transformer_for_efficient_long_context_modeling.md)
- [\[ACL 2026\] YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents](yield_a_large-scale_dataset_and_evaluation_framework_for_information_elicitation.md)
- [\[ACL 2026\] IMPACT: Importance-Aware Activation Space Reconstruction](impact_importance-aware_activation_space_reconstruction.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](selar_selective_latent_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->
