# Diffusion vs flow matching in VLA action heads — fact sheet (source notes for the literature matrix)

All values read from the papers' arXiv PDF/HTML (URLs per section). "Not reported" = not found in the text read.

## Diffusion Policy (Chi et al.; RSS 2023 conference version; arXiv 2303.04137, extended version read)
Source: https://arxiv.org/pdf/2303.04137
- Objective: DDPM noise prediction. Sample denoising iteration k and noise ε, minimise MSE(ε, ε_θ(O_t, A_t^0 + ε_k, k)) conditioned on observations O_t (Eq. 5, paraphrased; the PDF equation text is garbled).
- Steps: simulation uses "the same 100 denoising diffusion iterations for both training and inference" (App. A.4). Real world: "DDIM with 100 training iterations and 10 inference iterations enables 0.1s inference latency on a Nvidia 3080 GPU" (Sec. 3.4). App. A.4 instead says DDIM reduces inference iterations "to 16", so the paper is inconsistent.
- Horizons (sim CNN, Table 7): observation To = 2, prediction Tp = 16, action execution Ta = 8 (Ta = 8 best for most tasks). Receding-horizon control: execute Ta actions, then replan, warm-starting from the previous prediction.
- Size: CNN denoiser ~22M (45M Transport); transformer ~9M (sim) / 80M (Kitchen, real Push-T); ResNet-18 encoder ~22M.
- Limitations (Sec. 9): more compute and latency than LSTM-GMM; may not suffice for high-rate control.

## CogACT (Li et al., arXiv 2411.19650 v1; venue not stated)
Source: https://arxiv.org/pdf/2411.19650
- Diffusion action module: a DiT conditioned on the cognition feature, which is the output of a learnable cognition token appended to the visual+language tokens of a 7B VLM (DINOv2+SigLIP, LLaMA-2). The cognition feature and noisy action tokens are inputs to the DiT blocks. The diffusion step is added to the cognition feature via sinusoidal encoding.
- Objective: MSE between predicted and ground-truth noise. 8 diffusion steps sampled per training sample.
- Inference: DDIM, 10 steps, classifier-free guidance scale 1.5.
- Chunk: current action + 15 future actions (16 total). Adaptive Action Ensemble (α = 0.1, C = 0.2).
- Action-module sizes: DiT-S 13M, DiT-B 89M (default), DiT-L 308M. Total 7.6B (abstract: 7B).
- Latency: not reported in the paper. Third-party measurements: 54.29 ms on RTX 4090 (VLA-Cache, SIMPLER VM) and 0.2342 s/step on A40 (EfficientVLA).

## π0 (Black et al., arXiv 2410.24164 v4)
Source: https://arxiv.org/html/2410.24164 , https://arxiv.org/pdf/2410.24164
- Loss (conditional flow matching): L^τ(θ) = E ‖v_θ(A_t^τ, o_t) − u(A_t^τ | A_t)‖², with A_t^τ = τA_t + (1−τ)ε and u = A_t − ε, τ ∈ [0,1]. Note: in π0, τ = 0 is noise and τ = 1 is data.
- Time sampling: Beta distribution emphasising low (noisier) timesteps, p(τ) = Beta((s−τ)/s; 1.5, 1), with s = 0.999 (App. B).
- Integration: 10 forward Euler steps, δ = 0.1, from noise at τ = 0. Prefix (observation) KV is reused across the steps; timing lists "x10 action forward pass (flow) 27 ms".
- Action expert: ~300M (width 1024, mlp dim 4096), initialised from scratch; total 3.3B with PaliGemma 3B.
- Chunk: H = 50. Re-inference after 16 actions (20 Hz robots, every 0.8 s) or 25 actions (50 Hz robots, every 0.5 s).
- Latency on RTX 4090: 73 ms on-board, 86 ms off-board.

## SmolVLA (Shukor et al., arXiv 2506.01844 v1)
Source: https://arxiv.org/html/2506.01844 , https://arxiv.org/pdf/2506.01844
- Action expert: conditional flow-matching transformer with interleaved cross-attention and causal self-attention, attending to VLM features from layer N = L/2 of SmolVLM-2. Expert hidden size is 0.75 × d (VLM hidden size); the expert's parameter count is not reported. Total model 0.45B.
- Flow steps: "During inference, the flow matching is fixed to 10 steps."
- Chunk: n = 50 by default. In the chunk-size ablation, n = 10 gives a higher LIBERO avg (84.0 vs 80.3 for n = 50).
- Async inference: a RobotClient streams observations to a PolicyServer, and an action-queue threshold g triggers the next chunk request before the queue empties. On SO-100: completion time 13.75 s → 9.70 s (~30% faster), 19 vs 9 cubes in a fixed-time episode, success 78.3% (sync) vs 73.3% (async).
- Other efficiency: 64 visual tokens/frame; ~40% faster to train and 6× less memory than π0 (authors' claim).

## Diffusion vs flow matching across these papers
- Diffusion Policy and CogACT predict noise (ε-prediction) and sample with DDIM (10 steps at deployment). π0 and SmolVLA regress a velocity field (u = A − ε) and integrate 10 Euler steps.
- At deployment all four use about 10 steps: DP with real-world DDIM (100 in DP simulation), CogACT DDIM, and π0/SmolVLA Euler. (reviewer inference) The per-step cost therefore depends mainly on what each step re-runs, here a small expert/DiT on a cached VLM prefix or feature (π0, SmolVLA, CogACT). FAST (arXiv 2501.09747) reports ~750 ms per chunk for autoregressive π0-FAST vs ~100 ms for flow π0 on a 4090.
- Chunk/horizon: DP Tp = 16 (execute 8); CogACT 16; π0 H = 50 (execute 16 or 25); SmolVLA n = 50 (n = 10 better on LIBERO in its own ablation).
- Related: RTC (arXiv 2506.07339) adds inference-time inpainting for flow policies (n = 5 steps, 76 → 97 ms on RTX 4090 for π0.5). A2C2 (arXiv 2509.23224) adds a ~32M correction head on SmolVLA (~4.7 ms/step vs ~101 ms SmolVLA inference on an RTX 5080 laptop GPU).
