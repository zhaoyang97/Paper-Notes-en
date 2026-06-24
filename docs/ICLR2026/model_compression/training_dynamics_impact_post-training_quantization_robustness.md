---
title: >-
  [Paper Note] Training Dynamics Impact Post-Training Quantization Robustness
description: >-
  [ICLR 2026][Model Compression][Post-training quantization] The authors systematically measured GPTQ post-training quantization (PTQ) errors across open-source large model training trajectories (up to 32B parameters and 15T tokens). They found that the surge in quantization error is driven by training dynamics, such as learning rate decay, rather than increased training data volume. Accordingly, they propose two types of interventions—maintaining a larger learning rate and per…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Post-training quantization"
  - "learning rate scheduling"
  - "training dynamics"
  - "weight averaging"
  - "flat minima"
date: 2026-05-08
content_hash: 043bbb97e301363c
---

# Training Dynamics Impact Post-Training Quantization Robustness

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ZXr3Xx7Z1O](https://openreview.net/forum?id=ZXr3Xx7Z1O)  
**Area**: Model Compression  
**Keywords**: Post-training quantization, learning rate scheduling, training dynamics, weight averaging, flat minima

## TL;DR
The authors systematically measured GPTQ post-training quantization (PTQ) errors across open-source large model training trajectories (up to 32B parameters and 15T tokens). They found that the surge in quantization error is driven by training dynamics, such as learning rate decay, rather than increased training data volume. Accordingly, they propose two types of interventions—maintaining a larger learning rate and performing weight averaging along the trajectory—which significantly improve quantization robustness without sacrificing precision. These findings are unified through an explanation based on loss surface flatness (curvature/Hessian).

## Background & Motivation
**Background**: Post-training quantization (PTQ) is a core technology for efficient LLM deployment. Methods like GPTQ, AWQ, and BitsAndBytes, which compress 16/32-bit models to 3/4-bit, have become standard for model release and serving. However, a mechanistic understanding of what makes a trained model easy to quantize and the magnitude of resulting errors is still lacking.

**Limitations of Prior Work**: Recent studies by Kumar et al. (2024) and Ouyang et al. (2024) proposed scaling laws for quantization error, concluding that **more training tokens lead to more severe quantization degradation**. This suggests that as datasets grow, PTQ will become increasingly unfeasible, questioning whether future models can be quantized at all. This pessimistic conclusion conflicts with the trend of training stronger models with more data (overtraining).

**Key Challenge**: Prior works attributed the worsening of quantization to **data volume** but overlooked a critical confounding factor: **training dynamics**, particularly learning rate (LR) scheduling. In their experiments, checkpoints with more tokens typically coincided with phases of lower learning rate decay, confounding data volume with low learning rate.

**Goal**: (1) Characterize the relationship between PTQ error, training stages, and LR scheduling on real-world large-scale training trajectories; (2) Decouple data volume and learning rate through controlled experiments; (3) Identify training hyperparameter interventions to actively regulate quantization robustness; (4) Provide a unified geometric explanation.

**Key Insight**: Conventional open-source models often release only a **single final checkpoint**, obscuring the evolution of quantization error. New open-source projects like OLMo, OLMo2, SmolLM3, and Apertus provide **hundreds of intermediate checkpoints** and full training configurations, offering an unprecedented window to align training trajectories with quantization error.

**Core Idea**: Quantization difficulty is not determined by the amount of training data but by training dynamics (LR decay, weight averaging). Thus, models can be made more quantization-friendly by actively tuning training hyperparameters.

## Method

### Overall Architecture
The paper does not propose a new quantization algorithm but rather presents a **systematic empirical analysis, causal clarification, and actionable interventions**. The study proceeds in four steps.

First is **in-the-wild observation**: Selecting six modern open-source LLM training projects (OLMo 1B/7B, OLMo2 1B/7B/13B/32B, SmolLM3 3B, Apertus 8B, Open-science 1.3B, Amber 7B). Hundreds of intermediate checkpoints were quantized using GPTQ to 3-bit and 4-bit. Quantization error was measured using relative cross-entropy $\big(\tfrac{\mathrm{CE}(\hat W)}{\mathrm{CE}(W)}\big)-1$ alongside accuracy degradation on 12 benchmarks. A key observation under the Warmup–Stable–Decay (WSD) schedule was that quantization error remained nearly constant during the stable phase (even after 11T tokens) and **only surged sharply when the learning rate began to decay**, while verification loss continued to decrease.

Second is **controlled experiments to decouple confounders**: Pre-training small models (70M–160M) from scratch while varying one variable at a time (token budget, peak LR, schedule shape, weight decay). The results showed that models with different token budgets (10B–100B) reached **similar quantization errors after decay**, indicating that the error spike is tied to training dynamics rather than token count. Replicating Kumar et al.’s experiments showed that the "error increase with data" slows or disappears under WSD schedules, proving that **prior conclusions were dominated by the confounding factor of the LR schedule**.

Third is **proactive intervention**: Since LR decay is the primary cause, (a) maintaining a higher peak LR or (b) using weight averaging as an alternative to LR decay can improve quantization. Fourth is **geometric explanation**: By visualizing 2D slices of the loss surface and estimating the Hessian trace (Hutchinson) and maximum eigenvalue (sharpness via power iteration), the authors found that these interventions push the model toward **flatter minima**, making it more robust to weight perturbations introduced by quantization.

### Key Designs

**1. Decoupling LR and Data Volume: Training dynamics, not token count, drive quantization error**

This directly addresses the primary misattribution in prior scaling laws. Observation of trajectories in SmolLM3, OpenSci, and OLMo2 revealed a recurring pattern: during the stable phase of WSD, quantization error remains nearly constant for long periods (e.g., SmolLM3 at 11T tokens). **Once the linear LR decay phase begins, quantization error spikes sharply, exceeding any previous phase.** To rule out tokens as the cause, controlled experiments with different token budgets (12B to 100B) triggered cooldowns at different points. All runs converged to **nearly identical quantization errors** post-decay, regardless of training duration. This refutes the conclusion of Kumar et al. (2024), showing that the trend of "more data equals harder quantization" was driven by the uncontrolled LR schedule variable.

**2. Intervention 1: Higher peak LR achieves better low-bit quantization at equivalent precision**

The authors swept peak LR (3e-4 to 1e-2) while keeping other recipes fixed. Quantization error curves followed the **inverse order of LR magnitude**—higher LR resulted in lower quantization error. When comparing LR=1e-3 and 3e-3 at **similar full-precision validation loss**, the larger LR yielded significantly better low-bit quantization performance without compromising full-precision accuracy. This was replicated in OLMo2-7B experiments on 300B tokens. A related discovery is that cosine schedules, which drop LR to near-zero at the end, cause 3-bit quantization error to spike much more severely than WSD, which maintains better LR control. While larger weight decay $\lambda$ also helps (consistent with Ahmadian et al. 2023), its impact is significantly smaller than that of the LR.

**3. Intervention 2: Weight averaging along the trajectory can substitute for LR decay to improve robustness**

This intervention stems from an counter-intuitive observation: the final weights of OLMo2 and SmolLM3 are often **averages of multiple checkpoints** (model soup/linear merging), and these averaged models exhibit **lower quantization degradation than any single constituent model**. The authors proposed using Latest Weight Averaging (LAWA) during the stable phase to achieve the noise reduction effect of LR decay without its quantization-damaging side effects. Results on 160M models showed that while LAWA is inferior to LR decay in full precision, **LAWA-derived checkpoints match or exceed the performance of LR-decay models in 3-bit quantization**. This provides a path to high-quality checkpoints that bypass decay-induced quantization degradation.

**4. Unified Mechanism: Flat Minima explain the effectiveness of interventions**

The authors unify these findings using loss geometry. 2D loss surface visualization (spanning the final point $\Theta_K$, the previous step $\Theta_{K-1}$, and the quantized point $\hat\Theta_K$) shows that smaller LRs lead to **sharper** basins. Although the geometric distance between $\Theta_K$ and $\hat\Theta_K$ is smaller with low LRs, the sharpness causes the quantized model to land at much higher loss levels. Quantitatively, the evolution of the Hessian trace—estimated via Hutchinson's method—**closely mirrors the quantization error curve**, with sharpness increasing sharply during LR decay. In summary, LR decay pushes models into sharper minima sensitive to quantization noise, while higher LRs and weight averaging preserve **flatness**, enhancing quantization robustness.

## Key Experimental Results

### Main Results
Trajectory analysis and quantization settings (GPTQ, 3/4-bit):

| Model Family | Param Scale | Training Tokens | Key Observation |
|--------------|-------------|-----------------|-----------------|
| SmolLM3 | 3B | 11T (WSD) | Error constant in stable phase; spikes in decay phase |
| OLMo2 | 1B–32B | 4–6T | Gradual increase in cosine; spike in linear annealing; soup < individual |
| OpenSci | 1.3B | 1T | Quantization error surges as LR decreases across all budgets |
| Small Models | 70M/160M | 10B–100B | Post-decay quantization error is similar across different budgets |

Learning Rate Intervention (160M, fixed recipe sweeping peak LR):

| Peak LR | Relative Quantization Error Trend | Note |
|---------|-----------------------------------|------|
| 3e-4 (Min) | Highest | Error curves ordered inversely by LR |
| 1e-3 | High | Inferior to larger LR at same validation loss |
| 3e-3 | Low | Better quantization at same loss; no precision cost |
| 1e-2 (Max) | Lowest | Lowest quantization error |

### Ablation Study

| Config | Quantization | Key Finding |
|--------|--------------|-------------|
| WSD Stable Phase | 3/4-bit | Error remains constant even after 11T tokens |
| WSD Decay Phase | 3/4-bit | Error spikes sharply as validation loss decreases |
| Cosine vs WSD | 3/4-bit | WSD degradation grows slower, refuting "data volume" as the cause |
| LAWA (Constant LR) | 3-bit | Matches/exceeds LR decay quantization performance |
| LAWA | full-precision | Inferior to LR decay models |
| High Weight Decay $\lambda$ | 3/4-bit | Higher $\lambda$ gives lower error at same loss, but less impact than LR |

### Key Findings
- **LR decay is the trigger for quantization error surges**: In the stable phase, error remains constant; once decay starts, error spikes. This is the core consistent phenomenon.
- **Data volume is not the cause**: Runs from 10B to 100B tokens converge to similar errors after decay, refuting the pessimistic scaling laws.
- **Full-precision vs. Quantization trade-offs diverge**: LAWA is worse than LR decay in full precision but better in 3-bit, meaning interventions must be evaluated per precision level.
- **Curvature and quantization error are isomorphic**: The Hessian trace evolution almost perfectly replicates the quantization error curve, linking learning rate, flatness, and robustness.

## Highlights & Insights
- **The paradigm of clarifying confounding factors**: Decoupling "data volume" from "learning rate scheduling" proves that prior pessimistic scaling laws were dominated by an uncontrolled variable.
- **Unified geometric mechanism**: Higher learning rates, weight averaging, and larger weight decay are all attributed to "remaining in flatter minima." The isomorphic relationship between the Hessian trace and quantization error is highly convincing.
- **Actionable advice**: Quantization robustness should be an additional evaluation dimension during hyperparameter selection. Choosing the smaller of two LRs that yield equivalent loss may create a "time bomb" for downstream quantization.
- **Weight averaging as a free lunch**: LAWA is compatible with existing pipelines at zero cost, improves PTQ, and can even be applied retroactively to open-source models with multiple checkpoints.

## Limitations & Future Work
- **Quantization methods limited to GPTQ**: While trends were verified for AWQ and BNB in the appendix, more complex codebook or rotation-based methods were not fully explored.
- **Scale of controlled experiments**: Intervention experiments (LR, weight averaging) were conducted on 70M–160M models. Direct large-scale validation of interventions at 32B scale is lacking.
- **Associative interpretation**: The flat minima explanation relies on visualizations and Hessian estimates, providing strong correlation rather than absolute causal proof.
- **Future Work**: Integration of quantization robustness into LR schedule design; derivation of predictive formulas linking curvature, LR, and bit-width to replace pure data-count scaling laws.

## Related Work & Insights
- **vs. Kumar et al. (2024) / Ouyang et al. (2024)**: They established scaling laws where quantization error grows with tokens. This paper identifies the lack of control for LR schedules and shows that under WSD, the trend disappears, shifting the cause to training dynamics.
- **vs. Ahmadian et al. (2023)**: They used weight decay to suppress outliers for PTQ. This paper replicates this but proves LR is a much stronger lever and unifies it under the flat minima framework.
- **vs. Model Soup (Wortsman et al. 2022) / Weight Averaging (Izmailov et al. 2018)**: These works used averaging to boost full-precision performance; this paper reveals its benefits for **quantization robustness**.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Corrects a widely accepted scaling law misinterpretation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid in-the-wild observation (32B/15T), but interventions were validated on smaller models.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain: observation → decoupling → intervention → mechanism.
- Value: ⭐⭐⭐⭐⭐ Highly practical guidance for pre-training hyperparameters in models destined for quantization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Post-Training Quantization for Video Matting](post-training_quantization_for_video_matting.md)
- [\[ICLR 2026\] SliderQuant: Accurate Post-Training Quantization for LLMs](sliderquant_accurate_post-training_quantization_for_llms.md)
- [\[ICLR 2026\] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)
- [\[ICLR 2026\] LogART: Pushing the Limit of Efficient Logarithmic Post-Training Quantization](logart_pushing_the_limit_of_efficient_logarithmic_post-training_quantization.md)
- [\[ICLR 2026\] Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization](qronos_correcting_the_past_by_shaping_the_future_in_post-training_quantization.md)

</div>

<!-- RELATED:END -->
