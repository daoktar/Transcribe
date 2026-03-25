# Design Critique: Media Transcriber (Live Build)

**Date:** 2026-03-25
**Test files:** `test-files/test_video1.mp4`, `test-files/test_video2.mp4`
**Models tested:** tiny, small, medium
**Viewport tested:** Desktop (1280x800), Mobile (375x812)

---

## States Captured

| # | State | Tab | Description |
|---|-------|-----|-------------|
| 1 | Upload — Empty | UPLOAD | Initial state, drop zone with "Drop files here / or click to browse", default settings (large-v3, no language, no diarize) |
| 2 | Upload — Files Selected | UPLOAD | 2 files selected, checkmark icon in drop zone, "Click to change or drop new files" hint |
| 3 | Upload — Configured | UPLOAD | Model: tiny, Language: ru, Detect Speakers: checked, HF token filled |
| 4 | Process — Loading Model | PROCESS | "Loading model 'small'..." message, 0% progress, Job Queue: file1=PROCESSING, file2=PENDING, Cancel button visible |
| 5 | Process — Extracting Audio | PROCESS | "Extracting audio for speech detection..." message, 2% progress bar (blue fill) |
| 6 | Process — Between Files | PROCESS | "Done in 5s. (5 speech regions transcribed)", 100% teal bar, file1=DONE, file2=PROCESSING |
| 7 | Review — Transcript | REVIEW | Full transcript text in monospace, SUCCESS status, Language ru, Segments 27, Speakers N/A |
| 8 | Review — Diarize Warning | REVIEW | WARNING status, Speakers 0, "Retry Speaker Detection" button visible (yellow) |
| 9 | Mobile — Upload | UPLOAD | Stacked layout: drop zone -> format tags -> settings panel -> Transcribe button |
| 10 | Mobile — Review | REVIEW | Stacked stats, wrapped buttons, transcript text with proper line wrapping |

---

## Overall Impression

The dark navy theme is cohesive and professional. The 3-tab workflow (Upload -> Process -> Review) is intuitive and maps well to the user's mental model. The biggest opportunities are: **restructuring the Review tab** for better information density, **fixing broken Copy functionality**, and **improving low-contrast elements** for accessibility.

---

## Usability

| Finding | Severity | Recommendation |
|---------|----------|----------------|
| **File selector dropdown on Review tab is nearly invisible** — tiny native `<select>` element, file names shown as a plain unstyled list below it. Users cannot easily tell which file is selected or switch between files. The dropdown blends into the background and the file list has no interactive affordance (no hover, no active state, no icons distinguishing selected vs unselected) | **Critical** | Replace with a prominent custom file switcher: either (a) styled tab-pills with the filename + status icon per file, or (b) a sidebar card list where the active file is visually highlighted with accent border + background. The current file should be immediately obvious at a glance. On mobile, use a full-width styled select with larger touch target (min 44px height) |
| Drop zone "or click to browse" text is low-contrast — not immediately clear it's clickable | Moderate | Make it a visible link color or underline it |
| Job Queue cards lack left-border color coding for instant scanning | Moderate | Add 3px left border: blue=processing, gray=pending, green=done |
| Review tab is vertically stacked — summary stats, file list, buttons, subtabs, AND transcript all compete in one column | Moderate | Move file list to a compact sidebar or collapsible section; give transcript max space |
| Copy Text button fails silently (Clipboard API denied without HTTPS) — no fallback | Moderate | Add `document.execCommand('copy')` fallback with a textarea workaround |
| "Retry Speaker Detection" button appears even when diarization wasn't requested | Minor | Only show when `diarize: true` was in settings |
| Progress bar shows "0%" at both left AND center when at 0% — redundant | Minor | Hide center percentage when it equals 0% or 100% |
| "Cancel Remaining" button has no confirmation dialog | Minor | Add a brief "Are you sure?" or require double-click |

---

## Visual Hierarchy

### Upload Tab
- **Drop zone and Transcribe button** are correctly the focal points. Settings panel is clearly secondary.
- Format tags (AAC, AVI, FLAC...) at ~10px are extremely small — almost decorative noise. Either make them useful (clickable filters) or remove them.

### Process Tab
- **Progress bar** dominates attention — correct.
- "CURRENT ACTIVITY" label + message text is clear and informative.
- Job Queue on the right is well-positioned in two-column layout.

### Review Tab
- **"Review Results" H1** is the largest element but it's decorative, not actionable.
- Actual content (transcript text) is buried below 4 layers: summary -> file list -> buttons -> subtabs -> transcript.
- The transcript — the thing users care most about — needs significantly more breathing room.

---

## Consistency

| Element | Issue | Recommendation |
|---------|-------|----------------|
| Badge styles | PROCESSING (blue outline), PENDING (gray), DONE (green) — good color coding but inconsistent letter casing | Standardize to all UPPERCASE |
| Button hierarchy | "Copy Text" (outline), "Download ALL" (gradient fill), "Retry Speaker Detection" (yellow outline) — 3 different styles at same level | Make Download primary, Copy secondary, Retry tertiary |
| Card backgrounds | Process tab cards use slightly different shade than Review file list cards | Unify to one card background token |
| Progress bar color | Yellow during loading -> cyan/teal at 100%. Good semantic shift but the yellow is harsh against the navy | Consider blue -> green gradient instead, matching the accent palette |
| Border radius | Mix of values across components | Standardize to 2-3 values (e.g., 8px small, 12px medium, 16px large) |

---

## Accessibility

| Check | Status | Details |
|-------|--------|---------|
| Body text contrast (light on dark navy) | PASS | Adequate ratio |
| Subdued text ("Identify unique voices", "Required for speaker detection") | BORDERLINE | ~4.2:1 — passes AA for large text but may fail for small text |
| Format tags (AAC, AVI, FLAC) | FAIL | Very low contrast, nearly invisible |
| Progress percentage labels (0%, 100%) | PASS | Good contrast |
| Touch targets on mobile | ISSUES | "Copy Text", "Download ALL", "Retry" buttons wrap to 2 lines at 375px — tap targets overlap |
| Subtab buttons (Transcript/Speakers) | PASS | Adequate size and contrast |
| Keyboard navigation | NOT TESTED | Tabs should be navigable via arrow keys |

---

## Mobile Responsiveness (375px)

| Finding | Severity |
|---------|----------|
| Upload tab stacks correctly — drop zone -> settings vertically | Good |
| Review tab summary stats overflow — "WARNING", "Language ru", "Segments 23", "Speakers 0" each on separate line, taking massive vertical space | Moderate |
| Action buttons wrap ugly — "Copy Text" / "Download ALL (.zip)" / "Retry Speaker Detection" become 2-line buttons side-by-side | Moderate |
| File list cards take full width | Good |
| Transcript text is readable with proper wrapping | Good |

---

## What Works Well

1. **Dark navy palette** — Consistent, premium feel. The blue accent (#3e90ff) is distinctive and recognizable.
2. **Progress feedback** — "Loading model 'medium'...", "Extracting audio for speech detection...", "Done in 5s. (5 speech regions transcribed)" — excellent contextual status messages that keep users informed at every step.
3. **Job Queue** — Visual batch progress with per-file status badges is clear and functional. The two-column desktop layout (progress + queue) is logical.
4. **Progress bar animation** — Color shift from yellow -> teal on completion provides satisfying visual feedback.
5. **Responsive stacking** — Upload tab handles mobile gracefully with natural column stacking.

---

## Priority Recommendations

### 1. Redesign file selector on Review tab
**Severity:** Critical (unusable)

The current file selector is a tiny native `<select>` dropdown that's nearly invisible against the dark background. Below it, filenames are listed as plain text with no interactive affordance — no hover states, no active highlighting, no way to tell which file is currently displayed. For a multi-file transcription tool, **this is the primary navigation element on the most important screen** and it's almost unfindable.

**Fix:** Replace with styled file-switcher cards or tab-pills:
- Each file gets a card/pill with: filename, status icon (checkmark/warning/error), and clear active state (accent border + lighter background)
- Active file should be immediately obvious at a glance
- On mobile: full-width styled select with min 44px height, or vertically stacked file cards
- Consider showing a brief preview (first line of transcript, word count) in each card

### 2. Fix Copy Text — add clipboard fallback
**Severity:** Critical (broken feature)

The `navigator.clipboard.writeText()` fails without HTTPS or proper user-gesture context. Add `document.execCommand('copy')` fallback with a hidden textarea. This is a core user workflow that is completely non-functional.

### 3. Restructure Review tab layout
**Severity:** Moderate

The summary stats (Language, Segments, Speakers) should be a compact inline bar, not stacked vertically. The file list should be collapsible or shown as compact tabs. Give 70%+ of vertical space to the transcript itself — that's what users are here for.

### 4. Add left-border color coding to Job Queue cards
**Severity:** Moderate

Processing=blue (`#3e90ff`), Pending=gray (`#414755`), Done=green (`#22c55e`), Error=red (`#ef4444`). The text badges work but a colored left strip enables instant visual scanning of batch status.

### 5. Fix mobile button wrapping
**Severity:** Moderate

At 375px, "Download ALL (.zip)" and "Retry Speaker Detection" text breaks onto 2 lines creating cramped tap targets. Solutions:
- Use icon-only buttons on mobile with tooltips
- Stack buttons vertically on narrow viewports
- Abbreviate labels ("Download All", "Retry Speakers")

### 6. Improve format tags or remove them
**Severity:** Minor

At current contrast (~0.3-0.4 opacity) they're nearly invisible. Options:
- Boost to 60% opacity with better contrast
- Add a tooltip on "+10 More" showing all supported formats
- Remove entirely if they don't serve user needs

---

## Bugs Found During Testing

1. **Copy Text** — `navigator.clipboard.writeText()` throws `NotAllowedError: Write permission denied` in non-HTTPS context. Spams console with errors on every attempt.
2. **Speakers tab** — Clicking "Speakers" subtab shows identical content to "Transcript" tab when diarization returns 0 speakers (no speaker labels in text). Should show an informative empty state like "No speaker data available. Try Retry Speaker Detection."
3. **File selector** — `<select>` dropdown keeps value "test_video1.mp4" label but changing to file index 1 doesn't always update transcript text in view (possible race condition with async fetch).
4. **Model select on mobile** — After page state manipulation, the select element appears empty (no visible selected value). May be a rendering artifact but worth verifying with real user flow.
