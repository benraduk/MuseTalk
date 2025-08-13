# BraivTalk Enhancement Plan: Multi-Angle Face Processing

## **Project Goal**
Improve mouth alignment accuracy for non-frontal faces by implementing angle-aware processing inspired by FaceFusion's approach.

## **Current Problem**
- Non-frontal faces (side angles, tilted heads) get AI mouths misaligned
- Mouth appears on cheek or wrong position
- Current pipeline treats all faces identically regardless of orientation

## **Success Metrics**
- [ ] Eliminate "mouth on cheek" artifacts for angled faces
- [ ] Maintain or improve processing reliability 
- [ ] Preserve existing cutaway handling functionality
- [ ] Measurable quality improvement on test videos

---

## **Phase 1: Face Angle Detection & Diagnostics** 
**Priority**: HIGH | **Complexity**: LOW | **Timeline**: Week 1

### **Goal**
Understand face orientations in our videos and correlate with alignment issues.

### **Tasks**
- [x] **1.1** Add face angle estimation function to `musetalk/utils/preprocessing.py`
  - Use FaceFusion's exact jaw line method (landmarks 0 and 16)
  - Discretize to 5 angles: 0°, 90°, 180°, 270°, 360°
  - Classify as: Frontal (±30°), Left Profile, Right Profile, Angled
- [x] **1.2** Log angle detection results during preprocessing
  - Output angle for each detected face
  - Create angle distribution report
- [x] **1.3** Test on problematic video
  - Run enhanced preprocessing on test video with misalignment issues
  - Identify which frames/angles cause problems
- [x] **1.4** Create angle visualization tool ✅ **COMPLETED**
  - Generate debug images showing detected angles
  - Overlay angle information on frames
  - Create angle distribution charts
  - **Result**: `scripts/visualize_face_angles.py` tool created with full debugging capabilities

### **Expected Outcomes**
- Clear understanding of face angle distribution in test videos
- Correlation between face angle and mouth misalignment
- Baseline data for measuring improvements

### **Files to Modify**
- `musetalk/utils/preprocessing.py` - Add angle detection
- `scripts/inference.py` - Add angle logging
- Create `debug/angle_analysis.py` - Visualization tool

### **Success Criteria**
- [ ] Angle detection working on all test videos
- [ ] Clear correlation identified between angle >30° and misalignment
- [ ] Debug visualizations showing angle classifications

---

## **Phase 2: Enhanced Face Detection & Quality Scoring** 
**Priority**: HIGH | **Complexity**: LOW | **Timeline**: Week 2

### **Goal**
Upgrade face detection to FaceFusion-style multi-angle detection with quality scoring.

### **Tasks**
- [ ] **2.1** Integrate FaceFusion face detection models
  - Add SCRFD/YOLO_Face detection alongside existing FaceAlignment
  - Implement multi-angle detection (0°, 90°, 180°, 270°)
  - Select best detection across all angles
- [ ] **2.2** Implement face quality scoring system
  - Add detector confidence scores
  - Add landmark quality scoring  
  - Combine scores for overall face quality metric
- [ ] **2.3** Smart filtering with quality + angle thresholds
  - Filter faces below quality threshold OR above angle threshold
  - Use passthrough for filtered faces (maintain existing cutaway logic)
  - Make thresholds configurable
- [ ] **2.4** Performance and quality validation
  - Compare new detection vs. original on test videos
  - Measure improvement in detection accuracy
  - Ensure no regression in processing speed

### **Expected Outcomes**
- Better face detection for angled faces
- Quality-based filtering reduces artifacts
- Foundation for rotation normalization in Phase 3
- Maintained video continuity with smart passthrough

### **Files to Modify**
- `musetalk/utils/preprocessing.py` - Integrate FaceFusion detection
- `musetalk/utils/utils.py` - Update datagen for quality scores
- `configs/inference/` - Add detection model and threshold parameters
- `requirements.txt` - Add FaceFusion dependencies

### **Success Criteria**
- [ ] Multi-angle face detection working
- [ ] Quality scoring system operational
- [ ] Improved detection on angled faces (measurable)
- [ ] No performance regression
- [ ] Configurable thresholds working

---

## **Phase 3: Rotation Normalization Pipeline** 
**Priority**: HIGH | **Complexity**: MEDIUM | **Timeline**: Week 3-4

### **Goal**
Implement the core rotation normalization: rotate angled faces to frontal → MuseTalk → rotate back.

### **Tasks**
- [ ] **3.1** Implement rotation normalization functions
  - Create `normalize_face_rotation()` - rotate face to frontal (0°)
  - Create `restore_face_rotation()` - rotate result back to original angle
  - Use precise landmark-based angle calculation from Phase 1
- [ ] **3.2** Integrate normalization into preprocessing pipeline
  - Detect face angle → normalize to frontal → resize to 256×256
  - Store rotation matrix and original size for restoration
  - Handle edge cases (already frontal faces, extreme angles)
- [ ] **3.3** Implement angle restoration in blending
  - Restore original angle after MuseTalk inference
  - Resize back to original face crop dimensions
  - Maintain face quality through transformations
- [ ] **3.4** Add comprehensive testing and fallback
  - Test on faces from 0° to ±45° angles
  - Fallback to passthrough if normalization fails
  - Compare quality: normalized vs. original vs. passthrough

### **Expected Outcomes**
- Angled faces (±45°) processed with frontal-quality lip sync
- Dramatically improved mouth alignment for non-frontal faces
- Robust fallback system maintains reliability
- Clear quality improvement measurable on test videos

### **Files to Modify**
- `musetalk/utils/preprocessing.py` - Add rotation normalization functions
- `musetalk/utils/blending.py` - Add angle restoration to `get_image()`
- `musetalk/utils/utils.py` - Update datagen to handle angle metadata
- `scripts/inference.py` - Integrate full normalization pipeline

### **Success Criteria**
- [ ] Faces at ±45° angles processed successfully
- [ ] Mouth alignment significantly improved (measurable)
- [ ] No quality regression on frontal faces
- [ ] Processing time increase <30%
- [ ] Robust fallback prevents crashes

---

## **Phase 4: Advanced FaceFusion Integration** 
**Priority**: MEDIUM | **Complexity**: HIGH | **Timeline**: Week 5-6

### **Goal**
Integrate FaceFusion's advanced face alignment and blending techniques for extreme angles.

### **Tasks**
- [ ] **4.1** Implement FaceFusion warp templates
  - Add FFHQ_512 and other warp templates from FaceFusion
  - Implement `warp_face_by_face_landmark_5()` function
  - Use 5-point landmarks for precise face alignment
- [ ] **4.2** Advanced landmark processing
  - Integrate FaceFusion's 68-point landmark detection
  - Implement landmark quality scoring and selection
  - Handle landmark-based angle estimation refinements
- [ ] **4.3** FaceFusion-style blending system
  - Implement `paste_back()` function with affine transformations
  - Add advanced masking and feathering
  - Integrate face parsing for better mouth region isolation
- [ ] **4.4** Extreme angle handling (±90°)
  - Extend rotation normalization to handle profile views
  - Implement perspective correction for severe angles
  - Add quality validation for extreme angle processing

### **Expected Outcomes**
- Process faces at extreme angles (up to ±90°)
- Professional-grade blending quality matching FaceFusion
- Significant quality improvement for all face orientations
- Robust handling of challenging face poses

### **Files to Modify**
- Create `musetalk/utils/face_alignment.py` - FaceFusion alignment functions
- `musetalk/utils/blending.py` - Advanced blending with paste_back
- `musetalk/utils/preprocessing.py` - Enhanced landmark processing
- `musetalk/utils/face_parsing.py` - Extend for better mouth isolation

### **Success Criteria**
- [ ] Successfully process faces at ±90° angles
- [ ] Blending quality matches FaceFusion standards
- [ ] No artifacts in extreme angle scenarios
- [ ] Processing time increase <50% vs Phase 3

---

## **Phase 5: Performance Optimization & Production Readiness** 
**Priority**: MEDIUM | **Complexity**: MEDIUM | **Timeline**: Week 7-8

### **Goal**
Optimize the complete pipeline for production use with performance improvements.

### **Tasks**
- [ ] **5.1** Pipeline performance optimization
  - Implement rotation matrix caching for common angles
  - Batch processing for multiple faces in single frame
  - GPU memory optimization for larger batch sizes
- [ ] **5.2** Quality validation and metrics
  - Implement automated quality scoring system
  - Create comprehensive test suite with diverse face angles
  - Benchmark against original MuseTalk on quality metrics
- [ ] **5.3** Configuration and user experience
  - Add comprehensive configuration system for all parameters
  - Implement progressive fallback (normalization → filtering → passthrough)
  - Create user-friendly quality/speed trade-off settings
- [ ] **5.4** Production deployment features
  - Add detailed logging and monitoring
  - Implement error recovery and graceful degradation
  - Create performance profiling and debugging tools

### **Expected Outcomes**
- Production-ready pipeline with optimized performance
- Comprehensive quality validation and testing
- User-friendly configuration and deployment
- Robust error handling and monitoring

### **Files to Modify**
- `musetalk/utils/optimization.py` - Performance optimization functions
- `configs/inference/` - Comprehensive configuration system
- `scripts/benchmark.py` - Quality and performance testing
- `scripts/inference.py` - Production-ready error handling

### **Success Criteria**
- [ ] Performance within 20% of original MuseTalk speed
- [ ] Comprehensive quality improvement documented
- [ ] Zero crashes on diverse test videos
- [ ] Production deployment ready

---

## **Phase 6: Advanced Model Enhancement (Optional)** 
**Priority**: LOW | **Complexity**: HIGH | **Timeline**: Week 9-12

### **Goal**
Optional advanced enhancements for specialized use cases and maximum quality.

### **Tasks**
- [ ] **6.1** Angle-conditioned model training
  - Collect multi-angle training dataset
  - Train MuseTalk variant with angle embeddings
  - Compare against rotation normalization approach
- [ ] **6.2** 3D-aware face processing
  - Implement pitch/yaw correction alongside roll
  - Add depth estimation for perspective correction
  - Handle extreme profile views (±90°+ angles)
- [ ] **6.3** Real-time processing optimization
  - Model quantization and pruning
  - TensorRT/ONNX optimization
  - Multi-GPU scaling for batch processing
- [ ] **6.4** Advanced quality metrics
  - Implement perceptual quality scoring
  - Add temporal consistency validation
  - Create benchmark against commercial solutions

### **Expected Outcomes**
- State-of-the-art quality for all face orientations
- Real-time processing capability
- Research-grade evaluation and benchmarking
- Optional deployment for maximum quality scenarios

### **Files to Modify**
- `train.py` - Angle-conditioned training pipeline
- `musetalk/models/unet.py` - Enhanced model architecture
- `musetalk/utils/advanced_processing.py` - 3D-aware processing
- `scripts/benchmark_advanced.py` - Research-grade evaluation

### **Success Criteria**
- [ ] Custom models outperform rotation normalization
- [ ] Real-time processing on high-end hardware
- [ ] Published benchmark results
- [ ] Research paper quality evaluation

---

## **Testing & Validation Protocol**

### **Test Videos**
1. **Primary Test**: Current problematic video with angled faces
2. **Frontal Test**: Video with primarily frontal faces (baseline)
3. **Multi-Angle Test**: Video with diverse face orientations
4. **Cutaway Test**: Video with face transitions (regression testing)

### **Quality Metrics**
- [ ] Mouth alignment accuracy (manual scoring 1-10)
- [ ] Artifact count (misplaced mouths, distortions)
- [ ] Processing coverage (% faces successfully processed)
- [ ] Video continuity (smooth playback, audio sync)

### **Performance Metrics**
- [ ] Processing time per frame
- [ ] GPU memory usage
- [ ] CPU utilization
- [ ] Overall pipeline throughput

---

## **Risk Mitigation**

### **Technical Risks**
- **Risk**: Angle detection inaccuracy
  - **Mitigation**: Multiple detection methods, manual validation
- **Risk**: Performance degradation
  - **Mitigation**: Incremental testing, optimization focus
- **Risk**: Quality regression on frontal faces
  - **Mitigation**: Comprehensive regression testing

### **Project Risks**
- **Risk**: Complexity creep
  - **Mitigation**: Strict phase boundaries, MVP approach
- **Risk**: Integration issues
  - **Mitigation**: Maintain backward compatibility, thorough testing

---

## **Progress Tracking**

### **Phase 1 Status**: ✅ **COMPLETE** 🎉
- [x] Angle detection implemented
- [x] Diagnostic tools created  
- [x] Test video analysis complete
- [x] Findings documented

### **Phase 2 Status**: ⏳ Not Started  
- [ ] FaceFusion face detection integrated
- [ ] Multi-angle detection working
- [ ] Quality scoring system operational
- [ ] Performance validated

### **Phase 3 Status**: ⏳ Not Started
- [ ] Rotation normalization functions implemented
- [ ] Pipeline integration complete
- [ ] Angle restoration in blending working
- [ ] Quality improvement measured

### **Phase 4 Status**: ⏳ Not Started
- [ ] FaceFusion warp templates integrated
- [ ] Advanced blending system implemented
- [ ] Extreme angle handling working
- [ ] Professional-grade quality achieved

### **Phase 5 Status**: ⏳ Not Started
- [ ] Performance optimization complete
- [ ] Quality validation system working
- [ ] Configuration system implemented
- [ ] Production deployment ready

### **Phase 6 Status**: ⏳ Optional
- [ ] Advanced model training evaluated
- [ ] 3D-aware processing implemented
- [ ] Real-time optimization complete
- [ ] Research-grade benchmarking done

---

## **Key Decisions Log**

### **Decision 1**: Start with filtering approach
- **Rationale**: Reliability over coverage, quick wins
- **Date**: [To be filled]
- **Impact**: Eliminates artifacts immediately

### **Decision 2**: Use existing landmark system first
- **Rationale**: Leverage current infrastructure
- **Date**: [To be filled] 
- **Impact**: Faster implementation, lower risk

### **Decision 3**: Prioritize rotation normalization over filtering-only approach
- **Rationale**: Rotation normalization processes more faces with better quality than filtering alone
- **Date**: [Current]
- **Impact**: Dramatically improved coverage and quality for angled faces

### **Decision 4**: Implement FaceFusion detection models in Phase 2
- **Rationale**: Better angle detection foundation essential for rotation normalization
- **Date**: [Current]
- **Impact**: Improved face detection accuracy enables better normalization results

---

## **Resources & References**

### **FaceFusion Key Files**
- `face_analyser.py` - Multi-angle detection and quality scoring
- `face_helper.py` - Warp templates, alignment, and angle estimation
- `face_detector.py` - Rotated face detection
- `face_selector.py` - Quality-based face filtering
- `types.py` - Face data structures with angle and quality scores

### **BraivTalk Key Files**
- `musetalk/utils/preprocessing.py` - Current face processing
- `musetalk/utils/blending.py` - Current mouth integration
- `scripts/inference.py` - Main processing pipeline

### **Technical References**
- Face alignment research papers
- OpenCV transformation documentation
- MuseTalk original paper methodology

---

*Last Updated: January 2025*  
*Next Review: After Phase 1 completion*

## **Implementation Notes**

### **Rotation Normalization Implementation Strategy**
- **Phase 1**: ✅ **COMPLETE** - FaceFusion's `estimate_face_angle()` logic implemented
- **Phase 2**: Integrate FaceFusion's multi-angle face detection and quality scoring
- **Phase 3**: **CORE** - Implement rotation normalization pipeline (rotate → MuseTalk → rotate back)
- **Phase 4**: Advanced FaceFusion integration (warp templates, advanced blending)
- **Phase 5**: Production optimization and deployment readiness
- **Phase 6**: Optional advanced model enhancements

### **Key Dependencies for FaceFusion Integration**
- `onnx>=1.17.0` - For FaceFusion model compatibility
- `onnxruntime>=1.22.0` - For FaceFusion inference engines  
- `scipy>=1.15.0` - For advanced mathematical operations
- `psutil>=7.0.0` - For system monitoring and optimization
- `opencv-python>=4.8.0` - For advanced rotation and affine transformations

### **Critical Implementation Functions to Create**
```python
# Phase 2: Enhanced Detection
def integrate_facefusion_detection(frame) -> (bbox, landmarks, angle, quality)
def select_best_multi_angle_detection(detections) -> best_detection

# Phase 3: Core Rotation Normalization  
def normalize_face_rotation(face_crop, angle) -> (normalized_face, rotation_matrix)
def restore_face_rotation(processed_face, rotation_matrix) -> restored_face
def enhanced_musetalk_pipeline(frame, audio, models) -> result_frame

# Phase 4: Advanced Integration
def warp_face_by_face_landmark_5(frame, landmarks, template) -> (warped, matrix)
def paste_back_with_affine(frame, face, mask, matrix) -> blended_frame
```
