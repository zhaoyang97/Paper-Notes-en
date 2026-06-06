---
title: >-
  [Paper Note] FoeGlass: Simple In-Context Learning Is Enough for Red Teaming Audio Deepfake Detectors
description: >-
  [ICML 2026][LLM Safety][Audio Deepfake Detection] FoeGlass applies the "using LLM to red-team LLM" approach to audio deepfake detection (ADD): without fine-tuning the LLM…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Audio Deepfake Detection"
  - "Red Teaming"
  - "In-Context Learning"
  - "TTS Attacks"
  - "Diversity Feedback"
date: 2026-05-08
content_hash: 1a0c7d0d26a6e1a0
---

# FoeGlass: Simple In-Context Learning Is Enough for Red Teaming Audio Deepfake Detectors

**Conference**: ICML 2026  
**arXiv**: [2606.05101](https://arxiv.org/abs/2606.05101)  
**Code**: To be confirmed  
**Area**: AI Safety / Audio Deepfake Detection / Automated Red Teaming  
**Keywords**: Audio Deepfake Detection, Red Teaming, In-Context Learning, TTS Attacks, Diversity Feedback

## TL;DR
FoeGlass applies the "using LLM to red-team LLM" approach to audio deepfake detection (ADD): without fine-tuning the LLM, it uses in-context learning combined with realism and diversity feedback to let a black-box reasoning LLM automatically generate TTS prompts that deceive ADD. Starting from a cold start, it can increase the False Negative Rate (FNR) of existing ADDs from 0% to a maximum of 96%, with attacks demonstrating high transferability across eight different ADD models.

## Background & Motivation

**Background**: Audio Deepfake Detection (ADD) serves as the primary defense against Text-to-Speech (TTS) abuse. Mainstream evaluations rely on manually curated spoofing datasets like ASVspoof5 and VoxCelebSpoof, which cover various spoofing techniques, acoustic conditions, and adversarial perturbations.

**Limitations of Prior Work**: (i) High cost of manual data collection; (ii) Significant under-representation of "challenging outputs" that a single TTS model can generate, failing to identify blind spots in ADD; (iii) Existing automated attacks focus on low-norm perturbations centered around a reference audio, which are local and fail to sample "natural adversarial examples" from the generative model itself.

**Key Challenge**: To realistically evaluate ADD, one must directly sample natural adversarial examples from the TTS output distribution that can naturally deceive ADD. However, the TTS input space suffers from combinatorial explosion, making manual prompt engineering non-scalable. Directly porting the "attacker LLM fine-tuned to red-team target LLM" paradigm from the LLM community to ADD faces a triple challenge: scarcity of FN samples (preventing the construction of a fine-tuning set), potential mode collapse in RL fine-tuning toward a single deterministic strategy, and the requirement for LLM weights (limiting the use of top-tier closed-source LLMs).

**Goal**: Automatically, efficiently, and diversely sample TTS outputs that cause ADD misclassification while maintaining black-box access only to the reasoning LLM, TTS model, and ADD system.

**Key Insight**: The authors observe that the in-context learning capability of reasoning LLMs is already sufficiently powerful. By feeding "past successful/failed TTS prompts + CoT + scores + diversity feedback" into the context, the LLM can iteratively drive TTS prompts toward the blind spots of ADD without any parameter updates.

**Core Idea**: Formulate red teaming as "black-box in-context optimization": the LLM writes TTS inputs $\to$ TTS synthesizes audio $\to$ ADD provides realness scores + WavLM calculates min-cosine diversity scores $\to$ feedback is inserted back into the context for the next round, utilizing a carefully designed context template to suppress mode collapse.

## Method

### Overall Architecture
Formalization: Define the TTS as $G:\mathcal{U}\to\mathcal{X}$ (mapping from prompt/parameter space to audio space) and the ADD as a binary classifier $f:\mathcal{X}\to[0,1]$ with threshold $\tau$. The goal of red teaming is to sample from the subset of TTS inputs $F^{-1}((\tau,1])$ that are classified as "real."

FoeGlass solves this sampling problem via an in-context loop: in each round $t$, the attacker LLM $L$ reads the current context $c$ and outputs a TTS input $u_t$ with its Chain-of-Thought (CoT); the TTS synthesizes $x_t=G(u_t)$; the ADD provides a realness score $r_t=f(x_t)$; diversity is calculated using WavLM embeddings $w$ as $d_t = 1 - \max_{z\in w(X_\text{hist})}\langle w(x_t), z\rangle_{\cos}$; and finally $(u_t, \text{CoT}_t, r_t, d_t)$ is added to the history buffer for the DesignContext module to construct the next round's context. The entire pipeline requires no weight updates for the LLM, TTS, or ADD.

### Key Designs

1.  **Structured In-Context Template (instruction + failures + successes + CoT)**:
    - **Function**: Compresses the entire experience of the red-teaming process into a single context, allowing the LLM to "learn" in-context which TTS prompts deceive the ADD.
    - **Mechanism**: The context consists of three segments: (a) an instruction prompt describing the task and the semantics of TTS parameters (transcript, speed, temperature, style, voice), mandating JSON output; (b) the most recent $\ell/2$ failed attacks with their CoT, scores, and diversity feedback; (c) the $\ell/2$ successful attacks with the highest historical realness scores. The paper uses DeepSeek-R1-Distill-Llama-3.1-8B as the attacker LLM with $\ell=40$. The inclusion of CoT allows the LLM to continue its previous reasoning threads; ablation studies in Appendix B prove that CoT significantly aids performance.
    - **Design Motivation**: Unconditional sampling of TTS prompts by the LLM (baseline) yields very low success rates (often < 10% FNR). By providing "success/failure + reasoning + diversity status," the LLM's in-context learning stably converges to the weaknesses of the ADD. The balanced structure of successes and failures prevents the model from collapsing into a single prompt template more effectively than using only successful examples.

2.  **Realness + Diversity Dual Feedback (min-cosine over avg-cosine)**:
    - **Function**: Provides the LLM with two scalar signals—attack success (realness) and repetition (diversity)—to drive the explore/exploit trade-off.
    - **Mechanism**: Realness is taken directly from $f(x_t)$, with success defined as exceeding threshold $\tau$. Diversity is often measured by average cosine distance $d_\text{avg}$, but this can be diluted by "distant samples." The paper instead uses the minimum cosine distance $d(x';X)=1-\max_{z\in w(X)}\langle w(x'),z\rangle_{\cos}$, enforcing that new samples must be "sufficiently far" from all individual historical samples ($d>\tau_d$, where $\tau_d=0.01$ using WavLM embeddings). If this is not met, the feedback explicitly tells the LLM: "Output is too similar, please modify the transcript to increase diversity."
    - **Design Motivation**: The most common failure mode in red-teaming is mode collapse—where the LLM discovers one prompt that fools the ADD and repeatedly re-generates variants of it. Min-cosine turns diversity into a hard constraint, eliminating the dilution effect of average distance. Providing diversity as feedback rather than a rigid constraint preserves the LLM's flexibility to balance exploration and exploitation.

3.  **Cold-start / Warm-start Modes + Cross-ADD Transferability**:
    - **Function**: Operates even without known FN samples (cold start); gains significant performance with just 3 samples (warm start); and allows samples generated for one ADD to directly attack seven other ADD models.
    - **Mechanism**: In cold start, the instruction prompt contains no examples and the history starts empty. In warm start, 2 known FN samples and 1 TP sample are embedded into the instruction. Transferability arises because in-context exploration discovers regions in the TTS output space that are "broadly ignored by multiple ADDs" rather than local vulnerabilities of a specific model.
    - **Design Motivation**: Traditional fine-tuned attackers require large FN datasets, which are scarce in ADD. The in-context approach is naturally suited for low-data scenarios. The performance gain from just three examples suggests the LLM learns "how to reason" rather than just "memorizing successful prompts."

### Loss & Training
**No training involved**. FoeGlass is entirely an inference-time pipeline. The three main hyperparameters are context length $\ell=40$, diversity threshold $\tau_d=0.01$, and the number of iterations $T$ (generating 500 samples per run). Results are averaged across 5 seeds.

## Key Experimental Results

### Main Results: Significant FNR Increase (8 ADDs × 3 TTS)

| TTS | ADD | Uncond. Sampling FNR(%) | FoeGlass Cold FNR(%) | FoeGlass Warm FNR(%) |
|---|---|---|---|---|
| VITS | VIT-VoxCeleb-ConstantQ | 42.02 | 94.04 | **96.15** |
| VITS | VIT-VoxCeleb-MFCC | 32.57 | 95.28 | **98.08** |
| Kokoro-82M | VIT-VoxCeleb-MelSpec | 0.00 | 7.52 | **39.72** |
| xTTS-v2 | VIT-VoxCeleb-ConstantQ | 2.24 | 80.72 | **96.29** |
| xTTS-v2 | VIT-VoxCeleb-MFCC | 9.16 | 71.60 | **93.13** |
| xTTS-v2 | AST-VoxCeleb | 9.68 | 48.43 | **63.30** |

The maximum absolute gain is approximately +94 percentage points (xTTS-v2 $\to$ VIT-VoxCeleb-ConstantQ: 2.24% to 96.29%). Even for ADDs trained on ASVspoof5 (which has seen VITS data), cold start still achieves a 74.2% FNR, indicating that ASVspoof5 does not cover the complete output space of VITS.

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Cold vs Warm | Warm gains +2~+30 FNR | Only requires 2 FN + 1 TP samples with no extra compute |
| Fine-tuning RawNetLite with Ours | 49.6% $\to$ **8.2%** acc (-41) | Higher robustness on held-out VITS set |
| Fine-tuning AASIST (Black-box Transfer) | 15.2% $\to$ **0.2%** acc (-15) | Data generated from attacking RawNetLite without querying AASIST |
| vs ASVspoof5 TTS Subset (VIT-CQ) | 0.35% $\to$ 81.34% FNR | FoeGlass data is significantly harder than the original ASVspoof5 |
| No Diversity Feedback | FNR decrease, fewer clusters | Min-cosine feedback is critical to resisting mode collapse |

### Key Findings
- **In-Context is Sufficient**: With a purely black-box approach using only prompt engineering, FNR is pulled from single digits to 90%+. This shows that reasoning LLM's in-context exploration can replace fine-tuned attackers for narrow tasks.
- **min-cosine > avg-cosine**: PCA + WavLM + k-means visualization shows adversarial samples forming multiple semantic clusters (e.g., "social plans," "self-reflection"), proving min-cosine feedback pushes the LLM to sample across semantic boundaries.
- **High Transferability**: Samples generated against a single ADD consistently outperform unconditional baselines on 7 other ADDs, implying FoeGlass discovers shared blind spots rather than model-specific artifacts.
- **Kokoro-82M is the hardest target**: On VoxCeleb-trained ADDs, cold start FNR is near 0%, reaching only 39.72% with warm start. This suggests training distribution mismatches between TTS and ADD still limit attack success, highlighting a defense strategy.

## Highlights & Insights
- **In-context Learning as a Black-box Optimizer**: The CoT + history of a reasoning LLM forms an implicit optimizer that requires neither gradients nor weight access, making it a plug-and-play paradigm for red-teaming any black-box classifier.
- **Min-cosine over Avg-cosine Trick**: This subtle change should be considered in any generative red-teaming or RLHF diversity reward system. Average distance is easily masked by outliers; minimum distance is the true measure of repetition.
- **CoT in Context for Self-Improvement**: Feeding the LLM's own CoT back into its context allows for "fine-tuning-free self-improvement," which could extend to jailbreaking, CV red-teaming, and prompt recovery.
- **Adversarial Training Loop**: Fine-tuning ADD with FoeGlass data drops error rates on held-out sets twice as much as unconditional data, proving these "hard samples" are the necessary signals for advancing ADD.

## Limitations & Future Work
- **Hyperparameter Sensitivity**: Context length $\ell$, $\tau_d$, and the choice of attacker LLM significantly impact success and diversity; the exploration/exploitation trade-off still requires manual tuning.
- **Dependence on WavLM**: Min-cosine is calculated in WavLM space. If the embedding is insensitive to certain spoofing modes (e.g., prosody details), diversity feedback might fail. Multi-embedding voting is suggested.
- **Evaluated only on Open-source ADDs**: Performance against commercial systems (e.g., Pindrop) remains unknown.
- **Dual-use Risks**: The authors warn of potential misuse and suggest defenses such as LLM output watermarking and TTS input anomaly detection, though end-to-end defense experiments are limited.

## Related Work & Insights
- **vs Low-norm Adversarial Perturbation**: Unlike methods that add $\ell_p$ noise to specific audio, FoeGlass samples "natural spoofs" from the TTS distribution without needing reference audio.
- **vs Diffusion-based Natural Adversarial**: Those methods require white-box access to latent spaces; FoeGlass performs black-box searches in the prompt space.
- **vs Zhu et al. 2024b (GA for T2I Prompts)**: While similar in spirit, that work used genetic algorithms on fixed vocabularies. FoeGlass uses the open semantic space of LLMs, offering much higher flexibility.

## Rating
- Novelty: ⭐⭐⭐⭐ First automated black-box red-teaming for ADD. The min-cosine feedback and CoT-in-context are highly reusable tricks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of 8 ADDs and 3 TTS models, transfer matrices, and fine-tuning defenses is comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation and intuitive pipeline diagrams.
- Value: ⭐⭐⭐⭐ Provides an immediate toolchain for augmenting ADD datasets and identifies blind spots in standard benchmarks like ASVspoof5.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Understanding Generalization and Forgetting in In-Context Continual Learning](understanding_generalization_and_forgetting_in_in-context_continual_learning.md)
- [\[ACL 2026\] STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](../../ACL2026/llm_safety/star-teaming_a_strategy-response_multiplex_network_approach_to_automated_llm_red.md)
- [\[ICML 2026\] Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance](stable-gflownet_toward_diverse_and_robust_llm_red-teaming_via_contrastive_trajec.md)
- [\[ICML 2026\] OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents](otora_a_unified_red_teaming_framework_for_reasoning-level_denial-of-service_in_l.md)
- [\[ICML 2026\] STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack](stare_step-wise_temporal_alignment_and_red-teaming_engine_for_multi-modal_toxici.md)

</div>

<!-- RELATED:END -->
