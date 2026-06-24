---
title: >-
  [Paper Note] AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs
description: >-
  [ICML 2025][Optimization][Adversarial prompting] Proposed AdvPrompter, which uses an LLM (AdvPrompter) to generate human-readable adversarial prompt suffixes for target LLMs in seconds. Trained via an alternating optimization algorithm, it achieves high attack success rates on AdvBench and HarmBench and transfers to closed-source black-box LLMs, while presenting a strategy for adversarial training using generated adversarial suffixes to enhance target LLM robustness.
tags:
  - "ICML 2025"
  - "Optimization"
  - "Adversarial prompting"
  - "LLM jailbreaking"
  - "Red-teaming"
  - "Adaptive attack"
  - "Adversarial training"
date: 2026-05-08
content_hash: e8f34af0b1997ed1
---

# AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs

**Conference**: ICML 2025  
**arXiv**: [2404.16873](https://arxiv.org/abs/2404.16873)  
**Code**: [https://github.com/facebookresearch/advprompter](https://github.com/facebookresearch/advprompter)  
**Area**: Optimization/AI Safety  
**Keywords**: Adversarial prompting, LLM jailbreaking, Red-teaming, Adaptive attack, Adversarial training

## TL;DR
Proposed AdvPrompter, which uses an LLM (AdvPrompter) to generate human-readable adversarial prompt suffixes for target LLMs in seconds. Trained via an alternating optimization algorithm, it achieves high attack success rates on AdvBench and HarmBench and transfers to closed-source black-box LLMs, while presenting a strategy for adversarial training using generated adversarial suffixes to enhance target LLM robustness.

## Background & Motivation

**Background**: LLMs remain vulnerable to jailbreaking attacks even after safety alignment (e.g., RLHF) — meticulously designed prompts can bypass safety mechanisms to elicit harmful content.

**Limitations of Prior Work**:
   - Manual red-teaming is time-consuming and labor-intensive, making it difficult to scale.
   - Automated methods like GCG (gradient token optimization) generate unreadable, nonsensical suffixes and require minutes of gradient search.
   - AutoDAN generates readable suffixes but is not adaptive to the input instructions.
   - LLM conversational-based methods like PAIR are slow, requiring multiple rounds of interaction.

**Key Challenge**: High attack success rate vs. generation speed vs. human readability vs. input adaptiveness — existing methods satisfy at most two of these requirements.

**Goal**: Achieve all four goals simultaneously — high success rate, generation in seconds, human readability, and input adaptiveness.

**Key Insight**: Train a dedicated "adversarial prompt generation LLM" (AdvPrompter) instead of searching at test time. Once trained, generation takes only a single forward pass (~1-2 seconds).

**Core Idea**: Formulate the adversarial prompt search problem as an LLM fine-tuning problem — AdvPrompter learns to automatically generate disguised suffixes for any harmful instruction.

## Method

### Overall Architecture
AdvPrompterTrain alternates between two steps:
1. **AdvPrompterOpt**: Generate target suffixes for the current harmful instructions using gradient-based optimization (offline, slow but high-quality).
2. **AdvPrompter Fine-tuning**: Fine-tune the AdvPrompter LLM on the generated (instruction, suffix) pairs (to learn fast generation of similar suffixes).
Once trained, AdvPrompter can generate adversarial suffixes for any new instruction in 1-2 seconds.

### Key Designs

1. **AdvPrompterOpt — Gradient-Guided Target Suffix Generation**:

    - **Function**: Find the optimal adversarial suffix for a given harmful instruction.
    - **Mechanism**:
        - Goal: Search for a suffix $s$ such that TargetLLM outputs an affirmative response to "instruction + suffix".
        - Loss function: $\mathcal{L} = -\log P_{\text{target}}(\text{"Sure, here is"}| \text{instruction} + s)$
        - Performs continuous relaxation and projected gradient descent in the token embedding space.
        - Extra constraint: The suffix should be human-readable (via a perplexity penalty) — $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{attack}} + \lambda \cdot \text{PPL}(s)$.
    - **Design Motivation**: GCG's discrete search is inefficient and yields unreadable suffixes. Continuous relaxation combined with perplexity constraints addresses both efficiency and readability.
    - Note: This step is only used during training and is completely bypassed at inference time.

2. **Alternating Training — Distillation from Search to Generation**:

    - **Function**: Distill the "search capability" of AdvPrompterOpt into the "generation capability" of AdvPrompter.
    - **Mechanism**:
        - Each round: (a) Run AdvPrompterOpt on a batch of harmful instructions to obtain target suffixes; (b) Fine-tune AdvPrompter using the (instruction, suffix) pairs.
        - Alternate the process until AdvPrompter learns to autonomously generate high-quality suffixes.
    - **Design Motivation**: Searching is slow but can be done offline, while generating is fast but needs to be learned. Alternating training transfers the quality of the former into the speed of the latter.

3. **Adaptive Generation at Inference**:

    - **Function**: Generate adversarial suffixes for any new harmful instruction in 1-2 seconds.
    - **Mechanism**: The harmful instruction is taken as input, and AdvPrompter autoregressively generates the suffix via a pure forward pass.
    - Key Design: Adaptive suffixes are generated for different instructions, rather than using a universal attack template.
    - **Design Motivation**: GCG requires a new search for every single instruction (taking minutes), whereas AdvPrompter completes it in a single forward pass.

4. **Adversarial Training for Enhanced Robustness**:

    - **Function**: Enhance target LLM safety using adversarial suffixes generated by AdvPrompter.
    - **Mechanism**: Mix adversarial examples generated by AdvPrompter into the safety fine-tuning data $\rightarrow$ TargetLLM learns to reject these attacks.
    - **Design Motivation**: Attack and defense are two sides of the same coin — the capability to generate fast attacks naturally serves defense training.

### Loss & Training
- AdvPrompterOpt: Attack loss + perplexity regularization
- AdvPrompter Fine-tuning: Standard language modeling loss (cross-entropy)
- Alternating Optimization: Search-then-fine-tune in each round
- AdvPrompter uses Llama-2-7B as its base architecture

## Key Experimental Results

### Main Results
AdvBench Attack Success Rate (open-source target LLMs):

| Method | Llama-2 ASR↑ | Vicuna ASR↑ | Speed | Readable | Adaptive |
|------|-------------|------------|------|------|--------|
| GCG | 56% | 98% | ~10min/item | ✗ | ✗ |
| AutoDAN | 63% | 95% | ~5min/item | ✓ | ✗ |
| PAIR | 12% | 54% | ~20 turns | ✓ | ✓ |
| **AdvPrompter** | **52%** | **96%** | **1-2sec** | **✓** | **✓** |

### Black-box Transfer Attacks

| Target LLM | GCG Transfer | AdvPrompter Transfer |
|---------|---------|----------------|
| GPT-3.5 | 8% | **21%** |
| GPT-4 | 3% | **11%** |
| Claude-2 | 5% | **15%** |

### HarmBench Extended Evaluation

| Method | Standard ASR | Functional ASR (Actual Harmfulness) |
|------|---------|---------------------|
| GCG | 47.6% | 32.1% |
| **AdvPrompter** | **49.8%** | **38.2%** |

### Defense Effectiveness of Adversarial Training

| Training Strategy | Original ASR | Post-Adversarial Training ASR | Normal Function Retained |
|---------|---------|-------------|-----------|
| No adversarial training | 96% | - | ✓ |
| GCG suffix training | 96%→82% | 14%↓ | ✓ |
| **AdvPrompter suffix training** | **96%→68%** | **28%↓** | **✓** |

### Ablation Study

| Configuration | ASR (Vicuna) | Speed | Description |
|------|-------------|------|------|
| No perplexity constraint | 98% | 1s | Unreadable suffix |
| Weak perplexity constraint | 96% | 1s | Semi-readable |
| **Strong perplexity constraint** | **92%** | **1-2s** | Fully readable |
| Alternating training 1 round | 71% | 1s | Under-trained |
| Alternating training 5 rounds | 92% | 1s | Converged |
| **Alternating training 10 rounds** | **96%** | **1-2s** | Optimal |

### Key Findings
- AdvPrompter is 300-600$\times$ faster than GCG in terms of speed while maintaining a competitive attack success rate.
- Human-readable suffixes exhibit higher transfer attack success rates because black-box LLMs are less guarded against "normal-looking text".
- Alternating training achieves excellent distillation effects — after 10 rounds, AdvPrompter's generation quality is close to the search quality of AdvPrompterOpt.
- Adversarial training using AdvPrompter suffixes yields better defense performance than using GCG suffixes because they are more diverse and represent "truer attacks."
- The strength of the perplexity constraint serves as a knob to trade off readability against attack strength.

## Highlights & Insights
- **A paradigm shift from searching to generating** — transforms adversarial prompting from "search every time" to "train once, generate forever", increasing efficiency hundreds of fold.
- The alternating optimization training framework elegantly resolves the conflict between "high search quality but slow" and "fast generation speed but requiring data."
- Human readability is not just an aesthetic demand — more readable adversarial prompts are more effective in black-box transfer scenarios! This is a counter-intuitive but crucial finding.
- Attack as defense — using AdvPrompter for both red-teaming and blue-teaming creates an iterative safety improvement loop.
- Possesses foundational tooling value for the LLM safety field — enabling continuous, fast, and diverse red-teaming.

## Limitations & Future Work
- Training AdvPrompter still requires target LLM gradients (for the AdvPrompterOpt step) — making it inapplicable to completely black-box target LLMs.
- As target LLM safety alignment improves, AdvPrompter may need to be retrained.
- There is a trade-off between readability and attack success rate — fully readable suffixes have slightly lower ASR.
- Evaluated only on English — multilingual jailbreaking remains to be explored.
- Ethical risks: This tool could be used maliciously — the paper discusses responsible use policies.

## Related Work & Insights
- **vs GCG**: Discrete token search $\rightarrow$ unreadable, slow; AdvPrompter continuous optimization + distillation $\rightarrow$ readable, fast.
- **vs AutoDAN**: Uses fixed template variants $\rightarrow$ not adaptive to inputs; AdvPrompter generates customized suffixes for each input.
- **vs PAIR**: Multi-turn LLM dialog $\rightarrow$ slow; AdvPrompter single forward pass $\rightarrow$ fast.
- **Insight**: A "generative" approach to adversarial attacks may be more practical than a "search-based" approach — similar to the paradigm shift from MCMC to normalizing flows.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The paradigm shift from search to generation is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on open/closed-source targets, AdvBench/HarmBench, adversarial training, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear and intuitive.
- Value: ⭐⭐⭐⭐⭐ Possesses foundational tool value for LLM safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Kernel Learning with Adversarial Features: Numerical Efficiency and Adaptive Regularization](../../NeurIPS2025/optimization/kernel_learning_with_adversarial_features_numerical_efficiency_and_adaptive_regu.md)
- [\[NeurIPS 2025\] Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints](../../NeurIPS2025/optimization/beyond_tildeosqrtt_constraint_violation_for_online_convex_optimization_with_adve.md)
- [\[ICLR 2026\] Trion: FFT-based Dynamic Subspace Selection for Low-Rank Adaptive Optimization of LLMs](../../ICLR2026/optimization/trion_fft-based_dynamic_subspace_selection_for_low-rank_adaptive_optimization_of.md)
- [\[ICLR 2026\] Fast Frank–Wolfe Algorithms with Adaptive Bregman Step-Size for Weakly Convex Functions](../../ICLR2026/optimization/fast_frankwolfe_algorithms_with_adaptive_bregman_step-size_for_weakly_convex_fun.md)
- [\[ICLR 2026\] Improving Feasibility via Fast Autoencoder-Based Projections](../../ICLR2026/optimization/improving_feasibility_via_fast_autoencoder-based_projections.md)

</div>

<!-- RELATED:END -->
