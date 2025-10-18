"""
Prompts for video content analysis and clip selection
Users can edit these prompts to customize the AI behavior
"""

# Transcription prompt for Gemini
GEMINI_TRANSCRIPTION_PROMPT = """Transcribe this Arabic video with precise timestamps. The content is in informal Arabic (عامية) with occasional English words mixed in.

For each segment of speech, provide:
1. Start time in seconds (decimal format)
2. End time in seconds (decimal format)  
3. The exact text spoken (preserve informal Arabic and any English words)

Format your response as CSV with headers:
start_time,end_time,text
0.0,5.2,النص العربي هنا
5.5,12.3,المزيد من النص مع كلمات إنجليزية
...

IMPORTANT:
- Use exact timestamps in seconds (e.g., 125.5 not "2:05")
- Transcribe EXACTLY as spoken - informal Arabic (عامية), not formal (فصحى)
- Preserve English words when they appear in the speech
- Include ALL spoken content, even if informal, colloquial, or mixed language
- Each segment should be a natural speech unit (phrase or sentence)
- Return ONLY the CSV data, no other text
- Use proper CSV escaping for text containing commas or quotes
- Do NOT translate or formalize the language - keep it as spoken"""

# Clip selection prompts
def get_clip_selection_prompt(transcript_text: str, min_duration: float, max_duration: float) -> str:
    """Generate clip selection prompt based on duration requirements"""
    
    # Determine prompt based on duration requirements
    if max_duration >= 300:  # 5+ minutes = long-form content
        content_type = "long-form video segments"
        instructions = """Look for:
- Complete topics or chapters (full discussions, not snippets)
- Self-contained story arcs or narratives
- Deep dives into specific subjects
- Full explanations or tutorials
- Extended conversations or interviews
- Complete arguments or debates
- Thematic sections that form a coherent whole

Each segment should:
- Start and end with the topic (complete topic coverage)
- Include necessary context if needed for understanding
- Focus on a SINGLE topic, regardless of size
- Ignore strict length requirements if the topic needs more or less time
- Stay approximately around the assigned duration when possible
- Be a COMPLETE, STANDALONE piece of content that covers an entire topic from start to finish

DO NOT create short clips - focus on longer, substantial segments that tell complete stories.
Think "video" not "short" - capture the full context and development of ideas."""
    else:  # Short-form content
        content_type = "short clips"
        instructions = """Look for:
- Funny moments or jokes
- Interesting stories or anecdotes
- Surprising facts or revelations
- Emotional or impactful moments
- Complete thoughts or ideas
- Controversial or debate-worthy statements"""
    
    return f"""You are analyzing Arabic video transcript (informal Arabic with occasional English words) to find interesting {content_type}.

{instructions}

Transcript with timestamps (in seconds):
{transcript_text}

Respond with ONLY a JSON array of clips in this exact format:
[
  {{"start": 10.5, "end": 635.2, "reason": "مناقشة كاملة حول [الموضوع] - تشمل المقدمة والنقاط الرئيسية والخلاصة"}},
  {{"start": 645.0, "end": 1278.5, "reason": "قصة كاملة حول [الموضوع] من البداية للنهاية"}}
]

IMPORTANT: 
- Each clip must be between {min_duration} and {max_duration} seconds
- For long-form content, prefer LONGER segments that tell a complete story
- The content is in informal Arabic (عامية) with some English words - this is normal
- Focus on the meaning and content quality, not the language formality
- Write ALL reasons in Arabic (العربية) - do not use English for the reason field
- Return ONLY the JSON array, no other text"""
