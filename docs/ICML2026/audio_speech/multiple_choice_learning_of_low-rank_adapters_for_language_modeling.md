---
title: >-
  [Paper Note] Multiple Choice Learning of Low-Rank Adapters for Language Modeling
description: >-
  [ICML 2026][Audio & Speech][LoRA] This paper proposes LoRA-MCL, which introduces the "Winner-Takes-All" training paradigm of Multiple Choice Learning into LoRA fine-tuning. By treating $K$ sets of low-rank adapters as $K…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "LoRA"
  - "Multiple Choice Learning"
  - "Winner-Takes-All"
  - "Diverse Generation"
  - "Mixture Distributions"
date: 2026-05-08
content_hash: 9c024bed9bfd56f1
---

# Multiple Choice Learning of Low-Rank Adapters for Language Modeling

**Conference**: ICML 2026  
**arXiv**: [2507.10419](https://arxiv.org/abs/2507.10419)  
**Code**: https://github.com/Victorletzelter/LoRA-MCL (Available)  
**Area**: LLM Efficiency / PEFT / Diverse Decoding  
**Keywords**: LoRA, Multiple Choice Learning, Winner-Takes-All, Diverse Generation, Mixture Distributions

## TL;DR
This paper proposes LoRA-MCL, which introduces the "Winner-Takes-All" training paradigm of Multiple Choice Learning into LoRA fine-tuning. By treating $K$ sets of low-rank adapters as $K$ competing hypotheses and updating only the most suitable adapter for each training sample, a single base model can generate multiple diverse and plausible texts covering different modes of the conditional distribution in a single forward pass. It refreshes the quality-diversity Pareto frontier in audio/image captioning and machine translation.

## Background & Motivation

**Background**: In "one-to-many" tasks such as audio/image captioning and machine translation, the target distribution $p(x\mid c)$ for a given context $c$ is typically multimodal (e.g., the same image can have descriptions in English or French; the same audio clip can have different event labels). Current large models almost exclusively use Maximum Likelihood Estimation (MLE / teacher forcing) for next-token training, relying on decoding strategies like Beam Search, Diverse Beam Search (DBS), or nucleus sampling to "artificially" create diversity during inference.

**Limitations of Prior Work**: Minimizing MLE for a mixture distribution $p(x)=\sum_k p(z_k)p(x\mid z_k)$ collapses to a weighted average rather than recovering individual modes. Inference-side DBS requires manual tuning of the diversity penalty $\lambda$ and often forces a trade-off between diversity and quality. Techniques like TTA or temperature sampling are either unstable or compromise readability.

**Key Challenge**: The training objective itself lacks the concept of "modes," meaning all diversity patches are post-hoc remedies at inference time, addressing symptoms rather than the root cause. To enable models to "naturally" output multiple reasonable candidates, diversity must be embedded into the training objective.

**Goal**: (1) Adapt the Multiple Choice Learning (MCL) multi-hypothesis training paradigm to next-token language modeling; (2) Solve two major bottlenecks of MCL in LLMs: parameter explosion from multiple heads and training collapse into a single hypothesis; (3) Theoretically prove that this training recovers modes of mixture distributions instead of collapsing to an average; (4) Validate the quality-diversity trade-off on real large-scale models.

**Key Insight**: The authors observe that while the classical MCL approach involves a "shared backbone + multiple output heads," copying the LLM's `lm_head` (e.g., ~640M parameters for Qwen2-Audio) $K$ times is impractical. LoRA, however, provides the capability to "cheaply replicate a model" by simply adding a set of rank $r$ adapters $A_k, B_k$ per layer, with all hypotheses sharing the frozen base.

**Core Idea**: Replace $K$ output heads with $K$ sets of LoRA adapters, combined with a relaxed Winner-Takes-All loss, allowing each set of adapters to automatically specialize in one mode of the target distribution.

## Method

### Overall Architecture
The formalization of LoRA-MCL is remarkably clean: at each LoRA-enabled layer $\ell$, $K$ sets of adapters $\{(A_\ell^k, B_\ell^k)\}_{k=1}^K$ are prepared, while base parameters $\theta$ are frozen. The parameter set for the $k$-th "hypothesis model" is $\theta_k = \theta \cup \{(A_\ell^k, B_\ell^k)\}_\ell$. Given context $c$ and target sequence $x$, the likelihood $p(x\mid c;\theta_k)$ is calculated in parallel for all $K$ hypotheses, and the Winner-Takes-All (WTA) loss backpropagates only to the best hypothesis. During inference, WTA is not used; instead, each hypothesis independently decodes a candidate, producing $K$ different descriptions in a single forward pass. The training logic resembles a hard-EM algorithm: the E-step selects the winner $k^\star=\arg\max_k p(x\mid c;\theta_k)$, and the M-step updates only $\theta_{k^\star}$.

### Key Designs

1. **LoRA-MCL: Instantiating MCL Hypotheses with K Sets of LoRA Adapters**:

    - **Function**: Enables a single LLM to carry $K$ competing "hypothesis models" simultaneously without replicating the base or replacing the `lm_head`. The additional parameters for each hypothesis are merely a pair of rank $r$ matrices $(A_\ell^k, B_\ell^k)$, which is negligible compared to the base $|\theta|$.
    - **Mechanism**: The MCL WTA loss $\mathcal{L}^{\mathrm{WTA}} = -\mathbb{E}_{c,x}[\max_{k}\log p(x\mid c;\theta_k)]$ is applied directly to next-token modeling, where $\log p(x\mid c;\theta_k)=\sum_t \log p(x_t\mid x_{<t},c;\theta_k)$. Theoretically (Prop. 1), the authors prove that when data originates from a finite mixture and the model is sufficiently expressive, LoRA-MCL is equivalent to conditional hard-EM. The optimal loss is $\mathcal{H}(x\mid c,z)$ (conditional entropy given the latent topic $z$), which is strictly no greater than the MLE optimum. They also provide a lower bound $\min \mathcal{L}(\theta) - \log K \le \min \mathcal{L}^{\mathrm{WTA}}(\theta)$, characterizing the precise range of information gain from multiple hypotheses.
    - **Design Motivation**: Classical MCL implementations either replicate the entire head (infeasible for LMs) or train new heads from scratch (destroying pre-trained knowledge). LoRA’s low-rank residual form is naturally suited for "lightweight cloning"—preserving base semantics while adapters provide directions for modal specialization, keeping both parameter and computational overhead controllable.

2. **Relaxed WTA Loss: Solving Collapse with Relaxed-WTA and Annealed-MCL**:

    - **Function**: The biggest engineering pitfall of MCL is "initial random bias toward one hypothesis, where the winner keeps winning and others never receive gradients." This design prevents collapse by softening the WTA, leaving a residual gradient for all hypotheses to maintain competitiveness.
    - **Mechanism**: The $\max$ operator is replaced with a weighted sum $\mathcal{L}^{\mathrm{WTA}} = -\mathbb{E}_{c,x}[\sum_k q_k \log p(x\mid c;\theta_k)]$, where $\{q_k\}$ are normalized coefficients. Two instantiations are proposed: (i) **Relaxed-WTA**: the winner receives $q_{k^\star}=1-\varepsilon$, and others receive $\varepsilon/(K-1)$, where $\varepsilon$ is a small constant (typically 0.05–0.1 in experiments); (ii) **Annealed-MCL**: $q_k(x,c;\uptau)=p(x\mid c;\theta_k)^{1/\uptau}/Z$, with the temperature $\uptau(t)=\uptau(0)\rho^t$ ($\rho<1$) annealing from high to low. High temperatures allow nearly uniform updates to avoid collapse, while low temperatures smoothly converge to hard WTA.
    - **Design Motivation**: Relaxed-WTA is simple and stable, but an excessively large $\varepsilon$ might force hypotheses to converge (losing diversity). Annealed-MCL automatically transitions from "exploration" to "exploitation," theoretically achieving purer specialization, though it introduces a temperature schedule hyperparameter. Both schemes show advantages depending on the task/$K$.

3. **Grouped Convolutional Parallelism: Compressing K Forward Passes into a Single Batched Pass**:

    - **Function**: A naive implementation would sequentially run $K$ hypotheses, linearly increasing training time. This design leverages PyTorch’s grouped 1D convolutions to fuse all LoRA computations into a single batched operation, allowing $K$ hypotheses to run in parallel almost "for free."
    - **Mechanism**: The input is replicated $K$ times along the batch dimension, with each replica using its own set of LoRA weights for the residual. Specifically, stacking $K$ sets of $(A_\ell^k, B_\ell^k)$ into one tensor is equivalent to running a `nn.Conv1d` grouped variant (groups=$K$) on the LoRA path, ensuring each group only multiplies with its corresponding input. The frozen base forward pass is naturally shared. Since $r \ll d$, the extra memory is primarily due to activations doubling $K$ times, while the parameter increase is minimal.
    - **Design Motivation**: Aligning the training cost of LoRA-MCL with LoRA-MLE rather than $K \times$ LoRA-MLE is the critical engineering foundation for scaling this paradigm to 7B–8B models; a $K$-fold slowdown would render the theory impractical.

### Loss & Training
The final training objective is Relaxed WTA: $\mathcal{L}^{\mathrm{WTA}}(\theta_1,\dots,\theta_K)=-\mathbb{E}_{c,x}\big[\sum_{k=1}^{K} q_k \log p(x\mid c;\theta_k)\big]$. For LoRA configuration, adapters are injected into $Q, K, V$, and FFN up/down-projection matrices across all Transformer layers, with rank $r=8$, scaling $\alpha=8$ (vision version $\alpha=32$), and $K \in \{3, 5, 7\}$. Training spans 1 epoch (AudioCaps) or 10 epochs (Clotho) on Qwen2-Audio, and 1 epoch on LLaVA-1.6. During inference, WTA is absent; each hypothesis decodes independently. For MAP decoding (greedy/Beam Search/DBS), to ensure computational fairness, LoRA-MCL uses a beam size of $B/K$ for each hypothesis when LoRA-MLE uses $B$.

## Key Experimental Results

### Main Results

| Dataset | Metric | LoRA-MLE Best | LoRA-MoE Best | LoRA-MCL ($K=3$, BS=1) | Gain |
|---------|--------|---------------|---------------|------------------------|------|
| TextCaps (Image Captioning) | SPIDEr ↑ | 0.915 (DBS λ=1.0) | 0.926 (DBS λ=1.0) | **0.955** | +0.029 |
| TextCaps | mBLEU-4 ↓ | 0.416 | 0.421 | 0.520 | Higher (Worse) than DBS |
| AudioCaps (Audio Captioning) | Test loss ↓ | 2.181 ($r=8{\times}5$) | – | **1.999** ($K=5$) | −0.18 |
| Clotho | Test loss ↓ | 2.812 ($r=8$) | – | **2.612** ($K=7$) | −0.20 |
| Synthetic Bilingual Captioning (FR) | SPIDEr ↑ | 0.411 | – | **0.464** | +0.053 |
| Synthetic Bilingual Captioning (FR) | mBLEU-4 ↓ | 0.138 | – | **0.027** | −0.111 |

Note: In image captioning, LoRA-MCL exhibits slightly lower diversity (higher mBLEU-4) than DBS, but offers a significant advantage in SPIDEr/CIDEr. The authors note that DBS can be stacked with LoRA-MCL for combined benefits.

### Ablation Study (K-Scanning)

| $K$ (LoRA-MCL, $\varepsilon=0.05$) | AudioCaps Test loss ↓ | Clotho Test loss ↓ |
|---|---|---|
| LoRA-MLE Single ($r=8$) | 2.203 | 2.812 |
| LoRA-MLE ($r=8\times3$, equal params) | 2.195 | 2.868 |
| LoRA-MLE ($r=8\times7$, equal params) | 2.182 | 2.935 |
| LoRA-MCL, $K=3$ | 2.063 | 2.663 |
| LoRA-MCL, $K=5$ | 1.999 | 2.643 |
| LoRA-MCL, $K=7$ | **1.932** | **2.612** |

### Key Findings
- **Gains are not from parameter count**: Expanding the rank of LoRA-MLE to $r=8K$ yields minimal changes in AudioCaps test loss (2.18→2.18), whereas LoRA-MCL significantly reduces it by over 0.2 with the same parameters, proving the gain stems from "multi-hypothesis specialization" rather than model capacity.
- **Monotonic loss decrease with larger $K$**: Prop. 1 establishes a lower bound of $\min\mathcal{L}-\log K$. Experimentally, the loss decreases monotonically up to $K=7$ without saturating, consistent with the theory.
- **Strong specialization in bilingual experiments**: In a bilingual setup (50% French) with $K=2$, the winner for French samples falls on head 1 ~89% of the time, and for English on head 2 ~97% of the time. Conversely, MLE collapses to a weighted "average strategy," even entering repetitive loops for French. This visualizes the theoretical recovery of mixture modes from Prop. 1.
- **Toy Markov chain experiment**: Under a mixture of two Markov chains, MLE converges to a weighted average transition matrix $\bar P$, while LoRA-MCL’s two hypotheses recover the original transition matrices, validating the collapse formula in equation (9).

## Highlights & Insights
- **Repurposing LoRA as a "cheap MCL head" is an elegant paradigm shift**: MCL has struggled with head explosion for a decade. LoRA’s low-rank residual channels provide a natural way to create "lightweight clones," bypassing the `lm_head` bottleneck while retaining pre-trained knowledge.
- **Strong coupling between theory and experiments**: Prop. 1 is rigorous, providing a conditional entropy lower bound and equivalence to hard-EM. The transition from toy Markov chains (quantifying MLE collapse) to bilingual captioning in LLMs provides a standard "theory-to-reality" validation.
- **Grouped convolutional parallelism is key for deployment**: By addressing the $K \times$ training speed issue using grouped Conv1d, the authors ensure LoRA-MCL is as efficient as standard LoRA, making it viable for 7B–8B models.
- **High transferability**: This "K LoRA + Relaxed WTA" paradigm can be applied to any PEFT-friendly base model for language specialization in translation, preference learning in alignment, or style specialization in code generation.

## Limitations & Future Work
- Hyperparameters like $\varepsilon$ in Relaxed-WTA and the Annealed-MCL schedule are currently fixed; future work could adapt them based on data distribution.
- LoRA-MCL’s diversity (mBLEU-4) in image captioning still lags behind DBS+LoRA-MLE, suggesting training-side diversity hasn't fully "exhausted" the potential.
- The number of hypotheses $K$ is pre-defined. A data-driven mechanism to select $K$ is needed to avoid redundant or insufficient heads.
- Experiments are restricted to fine-tuning. Scaling to pre-training could potentially address fundamental collapse modes in current LLMs.
- Combinability with other LoRA variants (e.g., LoRA+, DoRA, AdaLoRA) remains an open question for future systematic investigation.

## Related Work & Insights
- **vs. Classical MCL**: Traditional MCL uses shared backbones + multiple heads + hard WTA. This work adapts it to LLMs using shared base + multiple LoRA + relaxed WTA, solving both the head replication and training time issues.
- **vs. LoRA-MoE**: LoRA-MoE uses multiple adapters but remains under the MLE objective, focusing on computational sparsity via routing. Experts often exhibit redundancy and low diversity. LoRA-MCL explicitly encourages modal specialization through the WTA objective, outperforming LoRA-MoE in experiments (SPIDEr 0.955 vs 0.926).
- **vs. Diverse Beam Search / TTA**: These are inference-side "patches" decoupled from training. LoRA-MCL shifts the source of diversity to the multi-modal distribution of the training data, resulting in a superior quality-diversity Pareto frontier.
- **Insight for PEFT**: Beyond "parameter-efficient adaptation," this work highlights LoRA’s utility for "parameter-efficient multi-hypothesis representation," applicable to multi-task, multi-style, and multi-reward scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Marrying MCL with LoRA is intuitive but previously unexplored; the combination is simple yet effective, and the theoretical equivalence is solid.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers toy models, audio/image captioning, machine translation, and synthetic data across multiple $K$ and relaxation strategies. Lacks combination with recent LoRA variants like DoRA.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, strong alignment between Prop. 1 and toy experiments, and reproducible engineering details.
- **Value**: ⭐⭐⭐⭐ Provides a "diversity-by-training" paradigm for LLM fine-tuning, which is more fundamental than inference-side tricks. Highly applicable to any task requiring multi-candidate outputs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](../../ACL2026/audio_speech/multimodal_in-context_learning_for_asr_of_low-resource_languages.md)
- [\[ICML 2026\] Algorithmic Recourse of In-Context Learning for Tabular Data](algorithmic_recourse_of_in-context_learning_for_tabular_data.md)
- [\[ICML 2026\] The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning](the_silent_thought_modeling_internal_cognition_in_full-duplex_spoken_dialogue_mo.md)
- [\[NeurIPS 2025\] A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings](../../NeurIPS2025/audio_speech/a_multitask_benchmark_for_abusive_language_detection_in_lowr.md)
- [\[ICML 2026\] Attend to Anything: Foundation Model for Unified Human Attention Modeling](attend_to_anything_foundation_model_for_unified_human_attention_modeling.md)

</div>

<!-- RELATED:END -->
