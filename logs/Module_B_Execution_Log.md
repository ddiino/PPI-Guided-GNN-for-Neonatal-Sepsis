# Module B Execution Log

## Task B.1: ID Mapping

### Phase 1: Implementation
- Loaded GEO datasets in SOFT format via GEOparse
- Mapped probes to genes using platform annotations (GPL570, GPL20292, GPL13667)
- Fixed column detection for 'symbol' (GPL20292)

### Phase 2: Verification (CoVe)
- **GSE25504:** 22,880 genes x 170 samples ✓
- **GSE69686:** 20,151 genes x 149 samples ✓
- **GSE26440:** 22,880 genes x 130 samples ✓
- **Common genes (GSE25504 ∩ GSE69686):** 14,921 ✓

### Phase 3: Technical Documentation
- Error: Initial series_matrix files had no expression data embedded
- Fix: Downloaded full SOFT format files via GEOparse
- Error: 'symbol' column not detected for GPL20292
- Fix: Added 'symbol' to column detection list

---

## Task B.2: Merging & ComBat Batch Correction

### Phase 1: Implementation
- Merged GSE25504 + GSE69686 on 14,921 common genes
- Applied ComBat batch correction using pycombat
- Generated PCA plots (before/after)

### Phase 2: Verification (CoVe)
- **Combined training samples:** 319 ✓ (170 + 149)
- **Final features:** 14,921 genes ✓
- **No NaN values:** Verified ✓
- **PCA plots:** Saved to figures/
- **Condition Distribution:** Control=186, Sepsis=133 ✓

### Phase 3: Technical Documentation
- **BUG FIX (2026-01-27):** Original parser only checked `characteristics` field
- **Root Cause:** GSM1404208+ samples had labels in `title` field (Con/Inf prefix)
- **Fix Applied:** Updated `parse_conditions()` to check title prefix first:
  - `Con` → Control
  - `Inf` → Sepsis
  - `Sus` → Control (suspected but not confirmed)
  - `NEC`/`Vir` → Sepsis (infection-related)
- **Result:** All 319 samples now correctly labeled (was 212 usable before)

---

**Module B Status:** ✅ COMPLETE (Fixed 2026-01-27)

---

## Phase 1.5: Strategic Review Fix (2026-01-28)

### Critical Issue Identified
- **Problem:** GSE25504 contained 170 samples mixed from two platforms (Affymetrix GPL570, Illumina GPL6947).
- **Impact:** Previous ComBat run treated them as a single batch, failing to correct intra-dataset platform effects.

### Fix Implementation
- **Platform Splitting:** Modified `02_merge_combat.py` to parse GSM IDs:
  - `GSM627xxx` → `GSE25504_Affy` (63 samples)
  - `GSM1404xxx` → `GSE25504_Illu` (107 samples)
- **ComBat Re-run:** Applied correction with 3 batches (`Affy`, `Illu`, `GSE69686`).
- **Result:** 319 samples successfully processed with correct platform adjustments.
