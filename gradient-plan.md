## Gradient blending and seamless mouth overlay – project plan

### Objectives
- Deliver a natural, invisible seam between AI mouth and target face across varied poses, skin tones, and motion.
- Eliminate overspill of original mouth features and remove sharp edges/halos.
- Maintain temporal stability with minimal flicker.

### Success criteria
- Quantitative
  - Seam discontinuity index (gradient mismatch across a narrow seam ring) ≤ 0.12 on 95% frames per clip.
  - SSIM across seam ring ≥ 0.92; temporal flicker metric (per-pixel alpha/std in seam ring) ≤ 8% of mean alpha.
  - Overspill rate (visible original lip pixels under AI mouth, measured by lip-parsing intersection) ≈ 0%.
- Qualitative
  - Human raters (n≥5) judge seam as “unnoticeable” in ≥ 90% of clips.

### Baseline
- Config: `configs/inference/test.yaml`
  - `mask_shape: ultra_wide_ellipse`, `ellipse_padding_factor: 0.02`, `upper_boundary_ratio: 0.22`, `blur_kernel_ratio: 0.06`, `mouth_scale_factor: 1.0`.
- Current method: face-parse mask ∩ shape mask → upper-boundary crop → Gaussian feather → composite.

### Dataset for evaluation
- Internal: `data/video/*` and paired audios, plus at least 8 additional clips covering:
  - Poses: frontal, 20–35° yaw, nods.
  - Lighting: soft daylight, mixed indoor, backlight.
  - Skin tones: Fitzpatrick I–VI.
  - Occlusion: hand/mic partial occlusions.
  - Motion: still, moderate head motion, fast speech.
- For each clip, enable debug exports in `debug_mouth_masks/<run_name>/`.

### Metrics and tooling
- Seam ring definition: 6–16 px band around mask boundary (normalized to face width).
- Compute per-frame:
  - Gradient mismatch: mean |∇(target)−∇(composite)| over seam ring.
  - SSIM in seam ring (grayscale and chroma).
  - Alpha statistics (mean/std) in seam ring; flicker = temporal std of mean alpha.
  - Overspill: area of original-lip parsing inside final alpha > 0.5 (should be ~0).
- Save artifacts: mask, overlay, seam ring visualization, per-clip CSV metrics.

---

## Prioritized roadmap

### P0 – Quick wins (largest quality gains, minimal complexity)
1) Mask rotation alignment (mouth angle)
- Category: Geometry/angle
- Idea: Estimate mouth angle from mouth-corner landmarks; align ellipse/superellipse to this angle.
- Expected impact: Strong reduction of upper-lip/philtrum flattening; fewer halos at corners.
- Tests: Ablate on 10 clips; measure seam index, overspill; human rating.

2) Morphological erosion before feathering
- Category: Mask smoothing/blending
- Idea: Erode the mask by a small normalized radius (e.g., 0.5–1.5% of face width) prior to blur to avoid including original lip edge; then feather outward.
- Expected impact: Removes original-lip bleed-through; sharper but still soft seam.
- Tests: Grid search erosion radius; track overspill → target 0%.

3) Adaptive blur cap and floor
- Category: Mask smoothing/blending
- Idea: `blur_kernel_ratio` auto-tuned by face size with min/max caps (e.g., [0.02, 0.08]) to avoid over-/under-feathering across scales.
- Expected impact: Consistent seam softness across different crop sizes.
- Tests: Multi-resolution clips; report seam index variance.

4) Temporal smoothing of mask parameters
- Category: Temporal stability
- Idea: EMA on mask center/size/angle (and vertical offset) with small momentum (β≈0.8–0.9). Optional tighter lock via existing YOLO primary face lock.
- Expected impact: Reduced seam jitter/flicker; more stable overlays.
- Tests: Compare flicker metric; plot alpha mean/std over time.

5) Per-channel mean/std color matching (inner mouth → target neighborhood)
- Category: Color/illumination
- Idea: Match mean and std of RGB or YCbCr inside a stable inner region to surrounding skin ring.
- Expected impact: Removes subtle color casts across seam.
- Tests: SSIM/chroma error in seam ring; human A/B.

6) Debug/QA overlays and metrics
- Category: Diagnostics
- Idea: Standardize seam ring visualization, CSV metrics, and per-run summary reports.
- Expected impact: Faster iteration; early detection of regressions.

### P1 – Medium complexity, high payoff
7) Distance-transform based alpha with gamma shaping
- Category: Mask smoothing/blending
- Idea: Build alpha from normalized distance-to-boundary; shape falloff with piecewise/gamma curve (faster ramp inside, slower near seam).
- Expected impact: Natural, halo-free transitions; better control over seam width.
- Tests: Compare against Gaussian feather on seam index and overspill.

8) Edge-aware feather (guided/bilateral modulation)
- Category: Mask smoothing/blending
- Idea: Modulate alpha falloff by local gradient magnitude; preserve true facial edges (lip line, nostrils) while keeping skin smooth.
- Expected impact: Avoids washing out crisp edges; reduces visible halos near strong edges.
- Tests: Edge-preservation score (gradient correlation across seam); human eval.

9) Multi-band (Laplacian pyramid) seam blend
- Category: Mask smoothing/blending
- Idea: Blend low frequencies over wider band and high frequencies narrowly within seam ring; apply only near boundary for efficiency.
- Expected impact: Texture continuity across seam; improved realism.
- Tests: Frequency-domain SSIM; human eval focusing on texture.

10) Temporal smoothing for color/illumination parameters
- Category: Color/illumination, temporal
- Idea: EMA over per-frame color-correction parameters to prevent hue/brightness pumping.
- Expected impact: Stable color across frames.
- Tests: Temporal variance of color stats in seam ring.

### P2 – Advanced/optional (bigger engineering effort)
11) Gradient-domain seam (Poisson/seamless cloning) in a thin ring
- Category: Mask smoothing/blending
- Idea: Solve gradient-domain blend constrained to a ring; equalize illumination while preserving interior texture.
- Expected impact: Best-in-class seam invisibility in challenging lighting.
- Tests: Compare to multi-band; profile performance.

12) Landmark-based local warp (TPS) of AI mouth to target geometry
- Category: Geometry/angle
- Idea: Thin-plate spline warp using nose/philtrum/corner landmarks for micro-alignment.
- Expected impact: Reduced stretching/edge stress; better corner fit.
- Tests: Corner alignment error; seam index near corners.

13) Motion-aware feathering
- Category: Temporal stability
- Idea: Slightly widen feather along motion direction during fast movement; optionally flow-warp previous alpha and average.
- Expected impact: Fewer motion halos during blurs.
- Tests: High-motion clips; flicker and seam indices.

14) Occlusion-aware masking
- Category: Occlusion handling
- Idea: Use parsing to cut mask where hands/hair/mic overlap; reduce alpha under occluders.
- Expected impact: Correct depth ordering; fewer artifacts.
- Tests: Occlusion clips; occlusion error rate.

15) High-res blending + subpixel resampling
- Category: Resolution/subpixel
- Idea: Perform mask and composite at 1.5–2× and downsample; ensure subpixel alignment on paste.
- Expected impact: Cleaner seams, fewer stair-steps on diagonals.
- Tests: Visual inspection; seam index at diagonal edges.

---

## Implementation and testing plan

### Phase 0 – Baseline and harness (1–2 days)
- Create evaluation harness:
  - Batch runner with fixed seed and identical decode settings.
  - Exports: mask, overlay, seam ring visualizations; metric CSV per clip.
  - Config toggles for all planned features; defaults preserve current behavior.
- Lock baseline metrics on the evaluation set.

### Phase 1 – P0 quick wins (3–5 days)
1) Rotation alignment
  - Compute mouth angle from mouth-corner landmarks.
  - Add angle parameter to ellipse/superellipse; ensure bounds/clipping + intersection with face-parse mask.
  - Grid search angle smoothing (EMA) factor.
2) Erosion before feather
  - Normalize erosion radius by face width; sweep [0.3%, 1.5%].
3) Blur cap/floor
  - Auto-compute blur from face size; clamp to [0.02, 0.08].
4) Temporal smoothing of mask parameters
  - EMA on center/scale/angle/offset; sweep β in [0.7, 0.95].
5) Color matching
  - Mean/std match in YCbCr; clamp gains; optional per-channel gamma tweak.
6) QA/metrics
  - Generate per-clip summary with rankings and deltas vs baseline.

Exit criteria: ≥ 30% reduction in seam discontinuity, zero overspill, ≤ 40% reduction in flicker.

### Phase 2 – P1 methods (4–7 days)
7) Distance-transform alpha
  - Build signed distance; normalize by seam width; apply gamma shaping; compare to Gaussian feather.
8) Edge-aware feather
  - Guided filter or gradient-weighted falloff; guardrails for high-contrast regions.
9) Multi-band seam blend
  - 3–4 level Laplacian within seam ring; profile and add fast path.
10) Temporal color smoothing
  - EMA over per-frame color params; add hysteresis.

Exit criteria: Additional ≥ 15% seam improvement; texture continuity improved in human eval.

### Phase 3 – P2 advanced (timeboxed 1–2 weeks)
11) Gradient-domain ring blend
  - Deploy Poisson on thin band with Dirichlet boundary; fallback to P1 method on failure.
12) TPS micro-warp
  - Use stable landmarks; regularize warp; add temporal smoothing.
13) Motion-aware feather; 14) Occlusion-aware; 15) High-res/subpixel
  - Implement incrementally; measure performance and gains.

Exit criteria: Edge cases resolved without regressions; runtime overhead acceptable.

---

## Configuration design
- New YAML toggles (proposed; default to current behavior):
  - Geometry: `enable_mask_rotation`, `mask_rotation_ema`.
  - Smoothing: `enable_pre_erosion`, `erosion_ratio`, `blur_ratio_min`, `blur_ratio_max`, `enable_distance_alpha`, `distance_alpha_gamma`.
  - Edge-aware: `enable_edge_aware_feather`, `edge_preserve_strength`.
  - Multi-band: `enable_multiband_seam`, `multiband_levels`.
  - Color: `enable_color_match`, `color_space`, `color_gain_clamp`, `enable_color_ema`, `color_ema_beta`.
  - Temporal: `mask_param_ema_beta`, `enable_flow_alpha_avg`.
  - Resolution: `enable_highres_blend`, `blend_scale`.
  - Occlusion: `enable_occlusion_cutout`.
- Debug: `export_seam_ring`, `export_metrics_csv`, `run_tag` to name output directory.

---

## Testing protocol
- For each feature:
  - A/B vs baseline on evaluation set; sweep 2–4 plausible parameter values.
  - Record metrics and render small GIFs for side-by-side human eval.
  - Track runtime and memory delta; aim ≤ 15% overhead for P0/P1.

## Risk management
- Over-smoothing causing plastic look → cap blur, prefer distance alpha with gamma.
- Landmark noise → EMA, primary face lock, reject outliers by temporal median.
- Performance regression → ring-limited operations; early-out to baseline path.
- Color shifts → clamp gains; revert switch per-clip if skin mismatch spikes.

## Rollout
- Ship P0 as default-on if metrics pass; gate P1/P2 behind flags.
- Keep `ultra_wide_ellipse` as default mask_shape; rotation and erosion improve it without overspill.
- Document parameter presets for “natural”, “maximum concealment”, and “high-motion”.

## Acceptance checklist
- [ ] Seam discontinuity meets threshold on 95% frames across eval set.
- [ ] Overspill measured ≈ 0%.
- [ ] Temporal flicker ≤ target.
- [ ] Human eval: ≥ 90% “unnoticeable seam”.
- [ ] Runtime overhead within budget.


