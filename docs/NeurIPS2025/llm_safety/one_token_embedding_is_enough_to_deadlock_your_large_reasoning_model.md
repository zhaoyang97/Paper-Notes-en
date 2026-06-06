---
title: >-
  [Paper Note] One Token Embedding Is Enough to Deadlock Your Large Reasoning Model
description: >-
  [NeurIPS 2025][LLM Safety][adversarial attack] This paper proposes the Deadlock Attack, which optimizes a single adversarial token embedding and implants it into a Large Reasoning Model (LRM) via a backdoor mechanism…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "adversarial attack"
  - "reasoning model security"
  - "resource exhaustion attack"
  - "backdoor attack"
  - "Chain-of-Thought"
date: 2026-05-08
content_hash: 32d994e5e258b532
---

# One Token Embedding Is Enough to Deadlock Your Large Reasoning Model

**Conference**: NeurIPS 2025
**arXiv**: [2510.15965](https://arxiv.org/abs/2510.15965)  
**Code**: [GitHub](https://github.com/UNITES-Lab/Deadlock-Attack)  
**Area**: LLM Reasoning / AI Safety
**Keywords**: adversarial attack, reasoning model security, resource exhaustion attack, backdoor attack, Chain-of-Thought

## TL;DR
This paper proposes the Deadlock Attack, which optimizes a single adversarial token embedding and implants it into a Large Reasoning Model (LRM) via a backdoor mechanism, causing the model to enter a permanent reasoning loop during inference (endlessly generating transition words such as "Wait" and "But"). The attack achieves a 100% attack success rate across 4 LRMs and 3 mathematical reasoning benchmarks, with negligible performance degradation on clean inputs.

## Background & Motivation
**Background**: Large Reasoning Models (LRMs) have achieved remarkable progress on complex tasks such as mathematics and programming through multi-step Chain-of-Thought (CoT) reasoning. However, this iterative thinking mechanism inherently exhibits an "overthinking" tendency, generating lengthy reasoning chains even for simple problems.

**Limitations of Prior Work**: Existing adversarial attacks against LRMs primarily target output correctness (inducing erroneous answers) or safety (jailbreaking), while the reasoning process itself as an attack surface has received little attention. The test-time compute scaling property of LRMs can be maliciously exploited as a resource exhaustion attack.

**Key Challenge**: The reasoning mechanism of LRMs is a double-edged sword — it endows models with powerful reasoning capabilities while simultaneously introducing a novel vulnerability: if the reasoning process is hijacked into an infinite loop, the model will exhaust all available computational resources without producing an answer, constituting a practical Denial-of-Service (DoS) attack.

**Goal**: (a) Can a single token embedding modification cause an LRM to enter a permanent reasoning loop? (b) How can adversarial embeddings in the continuous space be converted into practically deployable discrete triggers? (c) Can the attack simultaneously maintain stealthiness so that clean inputs are unaffected?

**Key Insight**: The authors observe that when an LRM is forced to generate transition words (e.g., "Wait", "But") following punctuation tokens (e.g., ".", "?") at the end of a reasoning step, the model is prevented from concluding its reasoning, thus falling into a permanent loop. This observation forms the design foundation of the attack.

**Core Idea**: An adversarial embedding is optimized such that the LRM generates transition words rather than conclusions after each reasoning step. This embedding is then implanted into the model's vocabulary via a backdoor mechanism, enabling a remotely activatable resource exhaustion attack triggered by a specific token.

## Method

### Overall Architecture
The Deadlock Attack proceeds in two phases. **Phase 1**: An adversarial prefix $\mathbf{e}_{adv} \in \mathbb{R}^{L \times d}$ is optimized in the continuous embedding space such that the model generates transition words with high probability following punctuation, rather than terminating its reasoning. **Phase 2**: The optimized embedding is injected into the model's embedding matrix as the representation of a predefined trigger token via a backdoor mechanism, and the poisoned model is released. An attacker need only prepend the trigger token to a query to remotely activate the attack.

### Key Designs

1. **Adversarial Embedding Optimization**:

    - Function: Learn a universal embedding prefix that, when prepended to arbitrary inputs, hijacks the model's reasoning process.
    - Mechanism: Define a transition word set $\mathcal{T}_{trans}$ = {"Wait", "But"} and a punctuation set $\mathcal{T}_{punct}$ = {".", "?"}. For each punctuation token position $a_j$ in the response, maximize the probability that the next token belongs to $\mathcal{T}_{trans}$: $\mathbb{P}_{trans}(a_j) = \frac{1}{|\mathcal{T}_{trans}|} \sum_{t \in \mathcal{T}_{trans}} p(t | \mathbf{z}_j)$.
    - Design Motivation: By repeatedly deferring conclusion generation, the model is prevented from ever entering the "output answer" stage, thereby exhausting the entire token budget.

2. **Discovery and Resolution of the Continuous-to-Discrete Projection Gap**:

    - Function: Reveal the critical challenge that attack effectiveness is nullified when mapping continuous adversarial embeddings to their nearest discrete token neighbors.
    - Mechanism: Linear Mode Connectivity (LMC) analysis demonstrates that projection errors typically exceed the perturbation tolerance of the adversarial embeddings. Remediation approaches including Gaussian noise injection during training and iterative projection steps both fail to fully bridge the gap.
    - Design Motivation: This finding explains why direct adversarial prompt attacks are ineffective and motivates the backdoor-based solution.

3. **Backdoor Implantation Strategy**:

    - Function: Directly write the optimized adversarial embedding into the model's embedding matrix, replacing the representation of a predefined trigger token.
    - Mechanism: A rarely used token is selected as the trigger, and its embedding is replaced with $\mathbf{e}_{adv}$. Since only the embedding of the specific token is modified rather than any other model parameters, the model behaves completely normally when the trigger token is absent.
    - Design Motivation: This approach perfectly circumvents the continuous-to-discrete projection gap — no projection is required, as the continuous embedding is directly implanted into the model.

### Loss & Training
- Convergence is achievable with as few as 1 training sample, though approximately 20 samples are required for generalization to unseen inputs.
- A prefix length of $L=1$ (a single token embedding) is sufficient to achieve a 100% attack success rate.

## Key Experimental Results

### Main Results (4 LRMs × 3 Benchmarks, max generation 4000 tokens)

| Model | Dataset | Clean ASR | Attack ASR | Clean Avg Tokens | Attack Avg Tokens |
|------|--------|---------|---------|---------------|---------------|
| Phi-RM | GSM8K | 0.0% | **100%** | 867 | 4000 |
| Nemotron-Nano | GSM8K | 4.0% | **100%** | 955 | 4000 |
| R1-Qwen | MATH500 | 6.98% | **100%** | 917 | 4000 |
| R1-Llama | MMLU-Pro | 4.0% | **100%** | 1219 | 4000 |

### Ablation Study — Stealthiness Evaluation (Accuracy without trigger: Clean vs. Backdoored Model)

| Model | GSM8K (Clean→DA) | MATH500 (Clean→DA) | MMLU-Pro (Clean→DA) |
|------|---|---|---|
| Phi-RM | 94.0→96.0 | 88.4→90.7 | 86.0→76.0 |
| R1-Qwen | 80.0→82.0 | 90.7→93.0 | 90.0→82.0 |
| R1-Llama | 80.0→76.0 | 93.0→83.7 | 82.0→80.0 |

### Key Findings
- **100% Attack Success Rate**: All 4 LRMs achieve 100% ASR across all 3 benchmarks, with models consistently generating up to the maximum token limit.
- **Existing Defenses Are Ineffective**: Test-time efficiency optimization strategies such as CoD, CCoT, and NoThinking all fail to defend against the attack, with ASR remaining at 100%.
- **High Stealthiness**: Model accuracy without the trigger remains largely consistent with the original model, making detection via standard evaluation infeasible.
- **Single Token Suffices**: $L=1$ is sufficient, and only 20 training samples are required for generalization.

## Highlights & Insights
- **Novel Attack Surface**: This work is the first to target the reasoning mechanism of LRMs itself, rather than inducing incorrect outputs or harmful content. By causing the model to "think indefinitely," it poses a serious DoS threat in real-world deployments.
- **Discovery of the Continuous-to-Discrete Projection Gap**: Through systematic LMC analysis, the paper explains why continuous-space adversarial perturbations are difficult to map to effective discrete prompts — a finding with broad implications for the adversarial NLP community.
- **Elegant Backdoor Implantation**: By modifying only the embedding vector of a single token in the embedding matrix without altering any other model parameters, the attack achieves a "precise surgical" implantation with high stealthiness.

## Limitations & Future Work
- **Strong White-Box Assumption**: Full access to model parameters and the ability to modify the embedding matrix are required, limiting applicability to closed-source models. The authors acknowledge the need to explore zeroth-order optimization for black-box settings.
- **Trigger Token Detectability**: If deployers perform anomaly detection on the embedding matrix (e.g., checking per-token embedding drift from pre-training), the tampered token could in principle be identified.
- **Limited Evaluation Scale**: The main experiments use only 50 samples per benchmark (43 for MATH500). While ASR reaches 100%, larger-scale evaluation would strengthen the statistical significance.
- **Single Attack Objective**: The current work only explores the "infinite loop" attack mode. Whether one can precisely control thinking duration (e.g., consuming exactly $N$ times the normal compute) or implant erroneous reasoning steps within the loop warrants further investigation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Introduces a fundamentally new attack surface by targeting the reasoning process itself as a security vulnerability; the discovery of the continuous-to-discrete projection gap also carries theoretical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Full coverage across 4 models and 3 benchmarks with sufficient ablations, though per-benchmark sample counts are small.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, the threat model is rigorously defined, and the narrative is coherent.
- Value: ⭐⭐⭐⭐⭐ — Raises important security warnings for LRM deployment; the backdoor implantation scheme constitutes a credible practical threat.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Attention! Your Vision Language Model Could Be Maliciously Manipulated](attention_your_vision_language_model_could_be_maliciously_manipulated.md)
- [\[NeurIPS 2025\] Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)
- [\[NeurIPS 2025\] PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)
- [\[NeurIPS 2025\] AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text](agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)
- [\[NeurIPS 2025\] HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning](holollm_multisensory_foundation_model_for_language-grounded_human_sensing_and_re.md)

</div>

<!-- RELATED:END -->
