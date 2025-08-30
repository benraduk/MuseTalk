## Gradient blending and seamless mouth overlay – project plan

### Objectives
- Deliver a natural, invisible seam between AI mouth and target face across varied poses, skin tones, and motion.
- Eliminate overspill of original mouth features and remove sharp edges/halos.
- Maintain temporal stability with minimal flicker.
- **NEW**: Achieve commercial-grade blending quality rivaling HeyGen and other premium lipsync solutions.

### Success criteria
- Quantitative
  - Seam discontinuity index (gradient mismatch across a narrow seam ring) ≤ 0.12 on 95% frames per clip.
  - SSIM across seam ring ≥ 0.92; temporal flicker metric (per-pixel alpha/std in seam ring) ≤ 8% of mean alpha.
  - Overspill rate (visible original lip pixels under AI mouth, measured by lip-parsing intersection) ≈ 0%.
- Qualitative
  - Human raters (n≥5) judge seam as "unnoticeable" in ≥ 90% of clips.
  - **NEW**: Blind A/B testing vs commercial solutions shows competitive quality.

### Baseline
- Config: `configs/inference/test.yaml`
  - `mask_shape: ultra_wide_ellipse`, `ellipse_padding_factor: 0.02`, `upper_boundary_ratio: 0.22`, `blur_kernel_ratio: 0.06`, `mouth_scale_factor: 1.0`.
- Current method: face-parse mask ∩ shape mask → upper-boundary crop → **basic Gaussian feather** → composite.
- **Critical limitation**: No erosion preprocessing causes original lip bleed-through; fixed blur ratios cause inconsistent seam quality.

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
1) **Contour-aware masking to eliminate "fold" artifacts** ⭐⭐⭐⭐⭐
- Category: Mask geometry/boundaries
- **Priority**: CRITICAL - Eliminates artificial facial contours and "fold" artifacts
- **Problem**: Current `ultra_wide_ellipse` extends beyond natural facial boundaries, creating fake cheek/jaw contours
- **Solution**: Implement FaceFusion's landmark-based contour following system
- **Implementation**: 
  - Port `create_area_mask()` from `facefusion/facefusion/face_masker.py` (lines 186-198)
  - Use `'lower-face'` landmark points: [3,4,5,6,7,8,9,10,11,12,13,35,34,33,32,31] for jawline following
  - Create `cv2.convexHull()` from 68-point landmarks to follow natural facial boundaries
  - Add new mask shape: `"contour_aware"` or `"lower_face_contour"`
- Expected impact: Eliminates 100% of "fold" artifacts; mask stays within natural facial boundaries
- Tests: Visual inspection for fold artifacts; measure mask boundary vs facial contours

2) Mask rotation alignment (mouth angle)
- Category: Geometry/angle
- Idea: Estimate mouth angle from mouth-corner landmarks; align ellipse/superellipse to this angle.
- Expected impact: Strong reduction of upper-lip/philtrum flattening; fewer halos at corners.
- Tests: Ablate on 10 clips; measure seam index, overspill; human rating.

3) Morphological erosion before feathering ⭐⭐⭐⭐⭐
- Category: Mask smoothing/blending
- **Priority**: CRITICAL - Eliminates original lip bleed-through (current #1 quality issue)
- **Implementation**: 
  - Add `cv2.erode()` with normalized kernel (0.5-1.5% of face width) before Gaussian blur
  - Replace current direct blur with: `mask → erode → dilate → blur` pipeline
  - Use elliptical structuring element to preserve mask shape
- Expected impact: Removes 90%+ of original-lip artifacts; creates clean inner boundary
- Tests: Grid search erosion radius; track overspill → target 0%.

4) Adaptive blur cap and floor ⭐⭐⭐⭐
- Category: Mask smoothing/blending
- **Priority**: HIGH - Fixes inconsistent seam quality across face sizes
- **Implementation**:
  - Replace fixed `blur_kernel_ratio=0.06` with adaptive scaling
  - Auto-compute: `blur_size = clamp(face_width * ratio, min_blur, max_blur)`
  - Suggested range: `[0.02, 0.08]` with face-size normalization
- Expected impact: Consistent seam softness across all crop sizes and resolutions
- Tests: Multi-resolution clips; measure seam index variance reduction.

5) Temporal smoothing of mask parameters
- Category: Temporal stability
- Idea: EMA on mask center/size/angle (and vertical offset) with small momentum (β≈0.8–0.9). Optional tighter lock via existing YOLO primary face lock.
- Expected impact: Reduced seam jitter/flicker; more stable overlays.
- Tests: Compare flicker metric; plot alpha mean/std over time.

6) Per-channel mean/std color matching (inner mouth → target neighborhood)
- Category: Color/illumination
- Idea: Match mean and std of RGB or YCbCr inside a stable inner region to surrounding skin ring.
- Expected impact: Removes subtle color casts across seam.
- Tests: SSIM/chroma error in seam ring; human A/B.

7) Debug/QA overlays and metrics
- Category: Diagnostics
- Idea: Standardize seam ring visualization, CSV metrics, and per-run summary reports.
- Expected impact: Faster iteration; early detection of regressions.

### P1 – Medium complexity, high payoff
8) Distance-transform based alpha with gamma shaping ⭐⭐⭐⭐⭐
- Category: Mask smoothing/blending
- **Priority**: TRANSFORMATIVE - Replaces primitive Gaussian blur with professional-grade falloff
- **Implementation**:
  - Use `cv2.distanceTransform()` to build signed distance field from mask boundary
  - Apply gamma curve: `alpha = (distance/max_distance)^gamma` where gamma ∈ [0.3, 2.0]
  - Gamma < 1.0 = faster inner ramp, slower seam; Gamma > 1.0 = opposite
  - Normalize distance by seam width for consistent behavior
- Expected impact: Natural, halo-free transitions; precise seam width control; eliminates hard edges
- Tests: A/B vs Gaussian feather on seam index and visual quality.

9) Edge-aware feather (guided/bilateral modulation) ⭐⭐⭐⭐
- Category: Mask smoothing/blending
- **Priority**: HIGH - Preserves facial features while smoothing skin transitions
- **Implementation**:
  - Compute gradient magnitude of target face: `grad = |∇I|`
  - Modulate alpha falloff: `alpha_final = alpha_base * (1 - edge_strength * grad_normalized)`
  - Preserve strong edges (lip lines, nostrils) while blending smooth skin areas
  - Use bilateral filter for edge-aware smoothing in seam region
- Expected impact: Eliminates feature washing; reduces halos near crisp facial edges
- Tests: Edge-preservation score; gradient correlation across seam ring.

10) Multi-band (Laplacian pyramid) seam blend ⭐⭐⭐⭐⭐
- Category: Mask smoothing/blending
- **Priority**: PROFESSIONAL - Frequency-domain blending for texture continuity
- **Implementation**:
  - Build 3-4 level Laplacian pyramids for source and target
  - Blend low frequencies over wide band (8-16px), high frequencies narrowly (2-4px)
  - Reconstruct final image from blended pyramid levels
  - Apply only in seam ring for efficiency; use fast pyramid implementation
- Expected impact: Seamless texture transitions; eliminates frequency discontinuities
- Tests: Frequency-domain SSIM; human evaluation focusing on texture realism.

11) Temporal smoothing for color/illumination parameters
- Category: Color/illumination, temporal
- Idea: EMA over per-frame color-correction parameters to prevent hue/brightness pumping.
- Expected impact: Stable color across frames.
- Tests: Temporal variance of color stats in seam ring.

### P2 – Advanced/optional (bigger engineering effort)
12) Gradient-domain seam (Poisson/seamless cloning) in a thin ring ⭐⭐⭐⭐⭐
- Category: Mask smoothing/blending
- **Priority**: BEST-IN-CLASS - Industry-standard seamless cloning
- **Implementation**:
  - Solve Poisson equation: `∇²f = ∇·v` in thin seam ring (6-12px)
  - Boundary conditions from target image; gradient field from source
  - Use sparse linear solver (conjugate gradient) for efficiency
  - Fallback to multi-band method on solver failure
- Expected impact: Invisible seams even in challenging lighting; matches commercial tools
- Tests: Compare to multi-band; profile performance overhead.

13) Landmark-based local warp (TPS) of AI mouth to target geometry
- Category: Geometry/angle
- Idea: Thin-plate spline warp using nose/philtrum/corner landmarks for micro-alignment.
- Expected impact: Reduced stretching/edge stress; better corner fit.
- Tests: Corner alignment error; seam index near corners.

14) Motion-aware feathering
- Category: Temporal stability
- Idea: Slightly widen feather along motion direction during fast movement; optionally flow-warp previous alpha and average.
- Expected impact: Fewer motion halos during blurs.
- Tests: High-motion clips; flicker and seam indices.

15) Occlusion-aware masking
- Category: Occlusion handling
- Idea: Use parsing to cut mask where hands/hair/mic overlap; reduce alpha under occluders.
- Expected impact: Correct depth ordering; fewer artifacts.
- Tests: Occlusion clips; occlusion error rate.

16) High-res blending + subpixel resampling ⭐⭐⭐
- Category: Resolution/subpixel
- **Priority**: MEDIUM-HIGH - Eliminates stair-stepping on diagonal seams
- **Implementation**:
  - Perform mask generation and blending at 1.5-2× resolution
  - Ensure subpixel-accurate alignment during paste operations
  - Downsample final result with anti-aliasing filter
- Expected impact: Cleaner seams on diagonal edges; reduced aliasing artifacts
- Tests: Visual inspection of diagonal seams; seam index on rotated faces.

---

## **FACEFUSION CONTOUR-AWARE MASKING SYSTEM**

### **Discovery: Advanced Masking Solutions**

**FaceFusion** (`facefusion/facefusion/face_masker.py`) implements exactly the contour-aware masking needed to solve the "fold" artifacts problem:

### **Available Masking Types:**

1. **📦 Box Mask** (`create_box_mask`, lines 158-170)
   - Basic rectangular with padding and blur
   - Good for simple cases but not contour-aware

2. **🎭 Occlusion Mask** (`create_occlusion_mask`, lines 173-183)
   - AI-based occlusion detection using XSeg models
   - Automatically detects hair, hands, objects occluding face

3. **🎯 Area Mask** (`create_area_mask`, lines 186-198) - **SOLUTION FOR FOLD ARTIFACTS**
   - **Landmark-based contour following** using 68-point facial landmarks
   - Uses `cv2.convexHull()` to create masks that follow natural facial boundaries
   - **Available areas:**
     - `'upper-face'`: [0,1,2,31,32,33,34,35,14,15,16,26,25,24,23,22,21,20,19,18,17]
     - `'lower-face'`: [3,4,5,6,7,8,9,10,11,12,13,35,34,33,32,31] ← **PERFECT FOR JAWLINE**
     - `'mouth'`: [48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]

4. **🧠 Region Mask** (`create_region_mask`, lines 201-214) - **SEMANTIC FACE PARSING**
   - Uses BiSeNet face parser for semantic segmentation
   - **Available regions:**
     - `'skin'`: Only skin areas (excludes hair, background)
     - `'mouth'`, `'upper-lip'`, `'lower-lip'`: Precise mouth regions
     - `'nose'`, `'left-eye'`, `'right-eye'`: Other facial features

### **Key Implementation Details:**

```python
# From facefusion/facefusion/face_masker.py lines 186-198
def create_area_mask(crop_vision_frame, face_landmark_68, face_mask_areas):
    crop_size = crop_vision_frame.shape[:2][::-1]
    landmark_points = []
    
    for face_mask_area in face_mask_areas:
        if face_mask_area in facefusion.choices.face_mask_area_set:
            landmark_points.extend(facefusion.choices.face_mask_area_set.get(face_mask_area))
    
    # CREATE CONVEX HULL FROM LANDMARKS - FOLLOWS NATURAL FACE CONTOURS
    convex_hull = cv2.convexHull(face_landmark_68[landmark_points].astype(numpy.int32))
    area_mask = numpy.zeros(crop_size).astype(numpy.float32)
    cv2.fillConvexPoly(area_mask, convex_hull, 1.0)
    
    # SMOOTH THE CONTOUR-AWARE MASK
    area_mask = (cv2.GaussianBlur(area_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
    return area_mask
```

### **Solution for "Fold" Artifacts:**

**Problem**: `ultra_wide_ellipse` extends beyond natural facial boundaries → creates artificial cheek/jaw contours
**Solution**: Use FaceFusion's `'lower-face'` area mask → follows exact jawline and chin contours

### **Implementation Strategy:**

**Phase 1: Port FaceFusion Area Mask (2-3 days)**
1. **Convert YOLOv8 5-point to 68-point landmarks** (or use existing face parsing)
2. **Implement `create_contour_aware_mask()`** based on FaceFusion's area mask
3. **Add new mask shape**: `"lower_face_contour"` to existing mask options
4. **Combine with erosion** for perfect lip bleed-through + fold artifact elimination

**Phase 2: Full FaceFusion Integration (5-7 days)**
1. **Port all 4 masking types** from FaceFusion
2. **Add semantic region masking** using face parsing
3. **Implement occlusion detection** for hands/hair/mic handling
4. **Create hybrid masks** combining multiple techniques

### **Expected Results:**

- ✅ **100% elimination of "fold" artifacts** - mask stays within natural facial boundaries
- ✅ **Perfect jawline following** - no artificial cheek/chin extensions
- ✅ **Anatomically correct masking** - respects natural facial geometry
- ✅ **Combined with erosion** - eliminates both overspill AND fold artifacts

### **Configuration Integration:**

```yaml
# New contour-aware masking options
mask_shape: "lower_face_contour"     # FaceFusion-based jawline following
enable_contour_aware: true           # Enable landmark-based boundaries
contour_smoothing: 5                 # Gaussian smoothing for contour mask
enable_semantic_regions: false       # Advanced: use face parsing regions
```

This discovery provides the exact solution needed to eliminate the "fold" artifacts while maintaining all the benefits of the erosion system for lip bleed-through prevention.

---

## Implementation and testing plan

### Phase 0 – Baseline and harness (1–2 days)
- Create evaluation harness:
  - Batch runner with fixed seed and identical decode settings.
  - Exports: mask, overlay, seam ring visualizations; metric CSV per clip.
  - Config toggles for all planned features; defaults preserve current behavior.
- Lock baseline metrics on the evaluation set.

### Phase 1 – P0 critical fixes (4–6 days)
**Priority**: IMMEDIATE - These fix the most visible quality issues
1) **Contour-aware masking** ⭐ CRITICAL - ELIMINATES FOLD ARTIFACTS
  - Port FaceFusion's `create_area_mask()` with `'lower-face'` landmarks
  - Add `"lower_face_contour"` mask shape option
  - Implement landmark-based boundary following to prevent artificial facial contours
2) **Erosion before feather** ⭐ CRITICAL
  - Normalize erosion radius by face width; sweep [0.3%, 1.5%].
  - Implement `mask → erode → dilate → blur` pipeline to eliminate lip bleed-through.
3) **Adaptive blur cap/floor** ⭐ HIGH IMPACT
  - Auto-compute blur from face size; clamp to [0.02, 0.08].
  - Replace fixed ratios with face-size normalized scaling.
4) Rotation alignment
  - Compute mouth angle from mouth-corner landmarks.
  - Add angle parameter to ellipse/superellipse; ensure bounds/clipping + intersection with face-parse mask.
  - Grid search angle smoothing (EMA) factor.
5) Temporal smoothing of mask parameters
  - EMA on center/scale/angle/offset; sweep β in [0.7, 0.95].
6) Color matching
  - Mean/std match in YCbCr; clamp gains; optional per-channel gamma tweak.
7) QA/metrics
  - Generate per-clip summary with rankings and deltas vs baseline.

Exit criteria: **Zero overspill**, **zero fold artifacts**, ≥ 50% reduction in seam discontinuity, consistent quality across face sizes.

### Phase 2 – P1 advanced blending (4–7 days)
**Priority**: HIGH - Brings quality to commercial standards
8) **Distance-transform alpha** ⭐ TRANSFORMATIVE
  - Build signed distance; normalize by seam width; apply gamma shaping; compare to Gaussian feather.
  - Replace primitive Gaussian blur with professional-grade falloff curves.
9) **Edge-aware feather** ⭐ HIGH IMPACT
  - Guided filter or gradient-weighted falloff; guardrails for high-contrast regions.
  - Preserve facial features while smoothing skin transitions.
10) **Multi-band seam blend** ⭐ PROFESSIONAL
  - 3–4 level Laplacian within seam ring; profile and add fast path.
  - Frequency-domain blending for seamless texture continuity.
11) Temporal color smoothing
  - EMA over per-frame color params; add hysteresis.

Exit criteria: Additional ≥ 30% seam improvement; **commercial-competitive quality**; texture continuity in human eval.

### Phase 3 – P2 professional polish (timeboxed 1–2 weeks)
**Priority**: MEDIUM - Final polish for best-in-class results
11) **Gradient-domain ring blend** ⭐ BEST-IN-CLASS
  - Deploy Poisson on thin band with Dirichlet boundary; fallback to P1 method on failure.
  - Industry-standard seamless cloning for invisible seams.
12) TPS micro-warp
  - Use stable landmarks; regularize warp; add temporal smoothing.
13) Motion-aware feather; 14) Occlusion-aware; 15) **High-res/subpixel** ⭐
  - Implement incrementally; measure performance and gains.
  - High-res blending eliminates diagonal aliasing artifacts.

Exit criteria: **Indistinguishable from premium commercial solutions** in blind testing; runtime overhead acceptable.

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
- [ ] Human eval: ≥ 90% "unnoticeable seam".
- [ ] **NEW**: Blind A/B vs commercial solutions shows competitive quality.
- [ ] Runtime overhead within budget.

---

## **MASK BLENDING IMPLEMENTATION PRIORITIES**

### **Immediate Next Steps (Order of Implementation)**

**1. Contour-Aware Masking (2-3 days) - CRITICAL** ⭐⭐⭐⭐⭐
- **Why First**: Eliminates "fold" artifacts by following natural facial boundaries
- **Implementation**: Port FaceFusion's `create_area_mask()` with `'lower-face'` landmarks
- **Expected Impact**: 100% elimination of artificial facial contours and fold artifacts

**2. Morphological Erosion (1 day) - CRITICAL** ⭐⭐⭐⭐⭐
- **Why Second**: Eliminates original lip bleed-through (works with contour masking)
- **Implementation**: Add `cv2.erode()` before existing Gaussian blur
- **Expected Impact**: 90%+ reduction in visible original lip artifacts

**3. Adaptive Blur Scaling (1 day) - HIGH IMPACT** ⭐⭐⭐⭐
- **Why Third**: Fixes inconsistent seam quality across face sizes
- **Implementation**: Replace fixed `blur_kernel_ratio` with face-size normalized scaling
- **Expected Impact**: Consistent seam quality across all resolutions

**4. Distance-Transform Alpha (2 days) - TRANSFORMATIVE** ⭐⭐⭐⭐⭐
- **Why Fourth**: Replaces primitive Gaussian with professional-grade falloff
- **Implementation**: Use `cv2.distanceTransform()` + gamma curve shaping
- **Expected Impact**: Natural, halo-free transitions; precise seam control

**5. Edge-Aware Feathering (3 days) - PROFESSIONAL** ⭐⭐⭐⭐
- **Why Fifth**: Preserves facial features while smoothing skin
- **Implementation**: Gradient-modulated alpha falloff with bilateral filtering
- **Expected Impact**: Eliminates feature washing; reduces edge halos

**6. Multi-Band Blending (3 days) - COMMERCIAL-GRADE** ⭐⭐⭐⭐⭐
- **Why Sixth**: Frequency-domain blending for texture continuity
- **Implementation**: Laplacian pyramid with frequency-specific blending bands
- **Expected Impact**: Seamless texture transitions; commercial-quality results

### **Expected Quality Progression**

**After Steps 1-3 (4-5 days total):**
- ✅ Zero "fold" artifacts - mask follows natural facial boundaries
- ✅ Zero original lip bleed-through
- ✅ Consistent seam quality across all face sizes
- ✅ 60-80% reduction in visible artifacts

**After Steps 4-5 (9-10 days total):**
- ✅ Natural, professional-grade falloff curves
- ✅ Preserved facial features and edge details
- ✅ 80-90% reduction in seam visibility

**After Step 6 (12-13 days total):**
- ✅ Commercial-competitive quality
- ✅ Seamless texture continuity
- ✅ Ready for blind A/B testing vs premium tools

### **Technical Implementation Notes**

**Current Limitation Analysis:**
- Basic `cv2.GaussianBlur()` with fixed ratios - very primitive
- No erosion preprocessing - causes lip bleed-through
- No edge-awareness - washes out facial features
- No frequency-domain blending - texture discontinuities

**Key Algorithmic Improvements:**
1. **Erosion-Dilate-Blur Pipeline**: `mask → erode → dilate → blur` eliminates bleed-through
2. **Distance-Transform Falloff**: `alpha = (distance/max_distance)^gamma` for natural transitions
3. **Edge-Modulated Alpha**: `alpha_final = alpha_base * (1 - edge_strength * grad_normalized)`
4. **Multi-Band Blending**: Separate low/high frequency blending for texture continuity

**Performance Considerations:**
- Phase 1-2: ≤15% overhead (critical path optimizations)
- Phase 3: ≤25% overhead (advanced features gated behind flags)
- Ring-limited operations for efficiency
- GPU acceleration where applicable

This implementation plan provides a clear path from the current basic Gaussian blending to commercial-grade seamless compositing that rivals premium lipsync solutions.


