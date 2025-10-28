# Stream Clipper Project Tasks

## Current Refactoring: Feature-Based Architecture

### Refactoring Goals
- Split 1501-line monolithic `clipper.py` into manageable feature modules
- Organize by feature (transcription/, analysis/, extraction/, etc.)
- Separate API keys for Gemini LLM vs Gemini transcription
- Extract CLI logic to separate module with command pattern
- Maintain backwards compatibility with existing usage

## Completed Refactoring Tasks

- [x] Create directory structure (src/, transcription/, analysis/, extraction/, utils/, cli/)
- [x] Extract configuration module with separate Gemini API keys
- [x] Extract utils module (gpu.py, dependencies.py, file_io.py)
- [x] Extract extraction module (audio_converter.py, video_clipper.py)
- [x] Extract transcription module with base interface and implementations
- [x] Extract analysis/llm module with base interface and providers
- [x] Extract analysis tools (clip_selector, timestamp_generator, title_generator)

## Remaining Refactoring Tasks

- [x] Update import statements in extracted modules
- [x] Update clipper.py to use new modules instead of duplicate definitions
- [x] Update all `__init__.py` exports for cleaner imports
- [x] Fix import errors and test basic functionality
- [ ] Test all functionality works (transcription, LLM, extraction) - in progress
- [ ] Fix circular dependencies if any (test needed)
- [ ] Extract CLI module (parser, commands, workflows) - Major task, see REFACTOR_STATUS.md
- [ ] Update documentation to reflect new structure

See `REFACTOR_STATUS.md` for detailed status.

---

## Previous Features

### Video Timestamps and Clip Titles Implementation

Implementation of two new features for the video clipper:
1. Video section timestamps with Arabic titles (max 10 sections)
2. Suggested Arabic and English titles for each clip

## Completed Tasks

- [x] Analyze existing codebase structure
- [x] Understand current workflow and file formats
- [x] Design timestamp generation system
- [x] Design clip title generation system
- [x] Implement timestamp generation functionality
- [x] Implement clip title generation functionality
- [x] Add command line flags for new features
- [x] Update prompts for new functionality
- [x] Integrate with existing workflow

## In Progress Tasks

- [x] Test the new features
- [x] Fix any integration issues
- [x] Update timestamp format to YouTube description format

## Future Tasks

- [ ] Add validation for timestamp limits
- [ ] Add error handling for title generation

## Recently Completed Features

### Open Router API Support ✅

Added support for Open Router as an LLM provider option, allowing users to access multiple AI models through a single API.

**Implementation:**
- Created `OpenRouterLLM` class for API integration
- Added environment variables: `OPEN_ROUTER_API_KEY` and `OPEN_ROUTER_MODEL`
- Added command-line arguments: `--openrouter-api-key` and `--openrouter-model`
- Updated LLM initialization logic to support Open Router
- Updated documentation with Open Router usage examples

**Usage:**
```bash
# Set in .env file
OPEN_ROUTER_API_KEY=your_key
OPEN_ROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free

# Use with clipper
python clipper.py video.mp4 --llm-provider openrouter
```

**Supported Models:**
- All models available on Open Router (Claude, GPT-4, Gemini, Mistral, etc.)
- Default: `meta-llama/llama-3.3-70b-instruct:free`
- See available models: https://openrouter.ai/models

## Implementation Summary

### ✅ Completed Features

**1. Video Section Timestamps**
- Generates up to 10 logical sections with Arabic titles
- Default behavior: Created after transcript generation
- `--create-timestamps` flag: Generate timestamps for existing transcript only
- `--skip-timestamps` flag: Skip timestamp generation entirely
- Saves to `{video_name}_timestamps.txt` in YouTube description format
- Format: `HH:MM:SS - Arabic Title` (one per line)
- Ready to copy-paste into YouTube video descriptions
- Validates section continuity and coverage

**2. Clip Suggested Titles**
- Automatically generates Arabic and English titles for each clip
- Arabic titles in informal style (عامية)
- English titles SEO-friendly and descriptive
- Added to clip metadata in JSON files
- Integrated into existing clip workflow

### 🔧 Technical Implementation

**New Classes:**
- `TimestampGenerator`: Handles video section timestamp creation
- `ClipTitleGenerator`: Generates suggested titles for clips

**New Prompts:**
- `get_timestamp_generation_prompt()`: Creates section timestamps
- `get_clip_title_generation_prompt()`: Generates clip titles

**Command Line Flags:**
- `--create-timestamps`: Generate timestamps only
- `--skip-timestamps`: Skip timestamp generation

**File Formats:**
- Timestamps: `{video_name}_timestamps.txt` (YouTube format)
- Updated clips: Include `suggested_titles` field with Arabic/English titles
- [ ] Update documentation

## Implementation Plan

### Feature 1: Video Section Timestamps

**Purpose**: Create up to 10 timestamped sections with Arabic titles for the full video

**Workflow Integration**:
1. Default behavior: Generate timestamps after transcript creation
2. `--create-timestamps` flag: Generate timestamps for existing transcript
3. `--skip-timestamps` flag: Skip timestamp generation entirely

**Technical Details**:
- Use LLM to analyze transcript and create logical sections
- Each section gets: start_time, end_time, arabic_title
- Save to `{video_name}_timestamps.json`
- Maximum 10 sections, evenly distributed across video duration
- Sections should represent natural content breaks

**File Format**:
```json
[
  {
    "section": 1,
    "start_time": 0.0,
    "end_time": 120.5,
    "arabic_title": "المقدمة والتعريف بالموضوع"
  },
  {
    "section": 2,
    "start_time": 120.5,
    "end_time": 245.8,
    "arabic_title": "المناقشة الرئيسية"
  }
]
```

### Feature 2: Clip Suggested Titles

**Purpose**: Generate Arabic and English suggested titles for each clip

**Workflow Integration**:
- Always generated as part of clip metadata
- No flags needed - automatic feature
- Added to existing clips JSON structure

**Technical Details**:
- Use LLM to analyze each clip's content and reason
- Generate both Arabic and English titles
- Titles should be descriptive and engaging
- Arabic titles should be in informal style (عامية)
- English titles should be clear and SEO-friendly

**Updated Clip JSON Format**:
```json
[
  {
    "clip_number": 1,
    "start": 10.5,
    "end": 635.2,
    "duration": 624.7,
    "reason": "مناقشة كاملة حول الموضوع",
    "source_video": "video.mp4",
    "suggested_titles": {
      "arabic": "كيف تبدأ في مجال الكتابة أونلاين؟",
      "english": "How to Start Online Writing Career"
    }
  }
]
```

### Command Line Flags

**New Flags**:
- `--create-timestamps`: Generate timestamps for existing transcript
- `--skip-timestamps`: Skip timestamp generation (default behavior)

**Flag Logic**:
- If `--create-timestamps` used: Only generate timestamps, skip other steps
- If `--skip-timestamps` used: Skip timestamp generation in normal flow
- Default: Generate timestamps after transcript creation

### Relevant Files

- clipper.py - Main implementation file
- prompts.py - LLM prompts for new features
- TASKS.md - This task tracking file

### Implementation Steps

1. **Add new prompts to prompts.py**:
   - Timestamp generation prompt
   - Clip title generation prompt

2. **Create new classes in clipper.py**:
   - `TimestampGenerator` class
   - `ClipTitleGenerator` class

3. **Update argument parser**:
   - Add `--create-timestamps` flag
   - Add `--skip-timestamps` flag

4. **Update main workflow**:
   - Integrate timestamp generation after transcript
   - Integrate clip title generation after clip analysis
   - Handle new flags appropriately

5. **Update file handling**:
   - Save timestamps to JSON file
   - Update clips JSON with suggested titles

6. **Add validation**:
   - Ensure max 10 timestamps
   - Validate title generation success
   - Handle LLM failures gracefully
