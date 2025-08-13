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
- [ ] **1.1** Add face angle estimation function to `musetalk/utils/preprocessing.py`
  - Use FaceFusion's exact jaw line method (landmarks 0 and 16)
  - Discretize to 5 angles: 0°, 90°, 180°, 270°, 360°
  - Classify as: Frontal (±30°), Left Profile, Right Profile, Angled
- [ ] **1.2** Log angle detection results during preprocessing
  - Output angle for each detected face
  - Create angle distribution report
- [ ] **1.3** Test on problematic video
  - Run enhanced preprocessing on test video with misalignment issues
  - Identify which frames/angles cause problems
- [ ] **1.4** Create angle visualization tool
  - Generate debug images showing detected angles
  - Overlay angle information on frames

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

## **Phase 2: Hybrid Quality + Angle Filtering** 
**Priority**: HIGH | **Complexity**: LOW | **Timeline**: Week 2

### **Goal**
Eliminate misaligned mouths using FaceFusion-inspired quality + angle filtering.

### **Tasks**
- [ ] **2.1** Implement hybrid face filtering (FaceFusion approach)
  - Add face quality scoring (detector + landmarker scores)
  - Combine quality thresholds with angle limits
  - Use passthrough for low-quality OR highly-angled faces
- [ ] **2.2** Add configurable filtering thresholds
  - Make quality and angle limits adjustable via config
  - Test different threshold combinations
- [ ] **2.3** Enhance passthrough logic
  - Extend existing cutaway passthrough to handle filtered faces
  - Ensure smooth video continuity
- [ ] **2.4** Performance testing
  - Compare pure angle vs. hybrid filtering
  - Measure quality improvement vs. processing coverage

### **Expected Outcomes**
- Zero "mouth on cheek" artifacts (better than pure angle filtering)
- Optimal balance between quality and coverage
- Maintained video continuity and audio sync

### **Files to Modify**
- `musetalk/utils/preprocessing.py` - Add quality scoring and hybrid filtering
- `musetalk/utils/utils.py` - Extend passthrough in datagen
- `configs/inference/` - Add quality + angle threshold parameters

### **Success Criteria**
- [ ] No misaligned mouths in test videos
- [ ] Higher processing coverage than pure angle filtering
- [ ] Configurable quality + angle thresholds working
- [ ] Quality improvement documented with metrics

---

## **Phase 3: Basic Face Normalization** 
**Priority**: MEDIUM | **Complexity**: MEDIUM | **Timeline**: Week 3-4

### **Goal**
Process more faces by normalizing orientation before lip-sync generation.

### **Tasks**
- [ ] **3.1** Implement face rotation correction
  - Detect face angle and rotate to frontal orientation
  - Use OpenCV affine transformations
- [ ] **3.2** Normalize face processing pipeline
  - Rotate face → Generate mouth → Rotate back
  - Maintain original perspective in final output
- [ ] **3.3** Quality validation system
  - Compare normalized vs. original processing
  - Implement quality scoring for processed faces
- [ ] **3.4** Fallback mechanism
  - If normalization fails, use passthrough
  - Ensure reliability over quality

### **Expected Outcomes**
- More faces processed successfully
- Better mouth alignment for moderately angled faces
- Maintained reliability with fallback system

### **Files to Modify**
- `musetalk/utils/preprocessing.py` - Add normalization functions
- `musetalk/utils/blending.py` - Handle rotated face blending
- `scripts/inference.py` - Integrate normalization pipeline

### **Success Criteria**
- [ ] Successfully process faces at ±45° angles
- [ ] Quality improvement measurable
- [ ] Fallback system prevents crashes
- [ ] Processing time increase <50%

---

## **Phase 4: Advanced Multi-Template Alignment** 
**Priority**: LOW | **Complexity**: HIGH | **Timeline**: Week 5-8

### **Goal**
Implement FaceFusion-style warp templates for optimal face alignment.

### **Tasks**
- [ ] **4.1** Study FaceFusion warp templates
  - Analyze different template types (arcface, ffhq, etc.)
  - Understand template selection criteria
- [ ] **4.2** Implement template-based face warping
  - Add warp template system to preprocessing
  - Use angle-appropriate templates
- [ ] **4.3** Multi-angle face detection
  - Implement rotated face detection (0°, 90°, 180°, 270°)
  - Use best detection result for each frame
- [ ] **4.4** Advanced landmark alignment
  - Implement 5-point and 68-point landmark systems
  - Use landmark-specific processing pipelines

### **Expected Outcomes**
- Professional-grade face alignment
- Processing capability for extreme angles
- Significant quality improvement across all orientations

### **Files to Modify**
- Create `musetalk/utils/face_alignment.py` - Template system
- `musetalk/utils/preprocessing.py` - Multi-angle detection
- `musetalk/models/` - Template-aware model loading

### **Success Criteria**
- [ ] Process faces at any reasonable angle
- [ ] Quality comparable to FaceFusion
- [ ] Template system fully configurable
- [ ] Comprehensive testing on diverse videos

---

## **Phase 5: Model Fine-Tuning & Optimization** 
**Priority**: LOW | **Complexity**: HIGH | **Timeline**: Week 9-12

### **Goal**
Optimize models specifically for multi-angle face processing.

### **Tasks**
- [ ] **5.1** Collect multi-angle training data
  - Curate dataset with diverse face orientations
  - Include angle labels and quality annotations
- [ ] **5.2** Fine-tune UNet for angle awareness
  - Train angle-conditioned lip-sync model
  - Implement angle embeddings in model
- [ ] **5.3** Advanced blending techniques
  - Implement perspective-aware blending
  - Use depth estimation for better integration
- [ ] **5.4** Real-time optimization
  - Optimize pipeline for real-time processing
  - Implement model quantization and acceleration

### **Expected Outcomes**
- Custom models optimized for multi-angle processing
- Real-time processing capability
- State-of-the-art quality results

### **Files to Modify**
- `train.py` - Add angle-aware training
- `musetalk/models/unet.py` - Angle conditioning
- Create `musetalk/utils/advanced_blending.py`

### **Success Criteria**
- [ ] Custom models outperform base MuseTalk
- [ ] Real-time processing achieved
- [ ] Comprehensive evaluation on benchmark datasets

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

### **Phase 1 Status**: ⏳ Not Started
- [ ] Angle detection implemented
- [ ] Diagnostic tools created
- [ ] Test video analysis complete
- [ ] Findings documented

### **Phase 2 Status**: ⏳ Not Started  
- [ ] Filtering system implemented
- [ ] Quality improvement measured
- [ ] Configuration system working
- [ ] Documentation updated

### **Phase 3 Status**: ⏳ Not Started
- [ ] Normalization pipeline working
- [ ] Quality validation complete
- [ ] Performance benchmarks met
- [ ] Integration testing passed

### **Phase 4 Status**: ⏳ Not Started
- [ ] Template system implemented
- [ ] Multi-angle detection working
- [ ] Advanced alignment complete
- [ ] Comprehensive testing done

### **Phase 5 Status**: ⏳ Not Started
- [ ] Training pipeline established
- [ ] Models fine-tuned
- [ ] Optimization complete
- [ ] Final evaluation done

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

### **Decision 3**: Implement FaceFusion-style hybrid filtering
- **Rationale**: Quality scores + angle limits better than angle-only filtering
- **Date**: [Current]
- **Impact**: Higher processing coverage while maintaining quality

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

### **FaceFusion Integration Approach**
- **Phase 1**: Use FaceFusion's exact `estimate_face_angle()` logic
- **Phase 2**: Implement FaceFusion's quality scoring system alongside angle filtering
- **Phase 3+**: Gradually adopt more FaceFusion techniques (warp templates, multi-angle detection)

### **Dependencies Added**
- `onnx>=1.17.0` - For FaceFusion model compatibility
- `onnxruntime>=1.22.0` - For FaceFusion inference engines
- `scipy>=1.15.0` - For advanced mathematical operations
- `psutil>=7.0.0` - For system monitoring and optimization
