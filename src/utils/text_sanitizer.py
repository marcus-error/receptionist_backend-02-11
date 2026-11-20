import re

def clean_markdown_for_voice(text: str) -> str:
    """
    Sanitizes LLM responses for voice synthesis (TTS) and UI display by removing all
    Markdown formatting syntax (**bold**, *italic*, headers, bullet dashes, double dashes --,
    asterisks, brackets, and extra spaces) to ensure continuous, natural human-like speech.
    """
    if not text:
        return ""

    # 1. Remove markdown headers (#, ##, ###)
    text = re.sub(r'#+\s*', '', text)

    # 2. Remove bold and italics formatting (**bold**, *italic*, __bold__, _italic_)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # 3. Remove parenthetical markers and brackets e.g. "(e.g: ...)", "(Doctor/Staff)"
    text = re.sub(r'[\(\[\{]\s*(?:e\.g\.?:?|i\.e\.?:?|ex\.?:?)?\s*([^()\[\]{}]+)[\)\]\}]', r'\1', text)
    text = re.sub(r'[()\[\]{}]', '', text)

    # 4. Replace double dashes (--), em-dashes (—), en-dashes (–), and bullet markers with a single space
    text = re.sub(r'-{2,}', ' ', text)
    text = re.sub(r'[—–]', ' ', text)
    text = re.sub(r'(?:^|\n)\s*[-•*]\s*', ' ', text)
    text = re.sub(r'\s+[-•*]\s+', ' ', text)

    # 5. Convert numbered list items e.g. "1. First ... 2. Second" into conversational pauses
    text = re.sub(r'(?:^|\n)\s*\d+[\.\)]\s*', ', ', text)

    # 6. Strip all remaining asterisks, tildes, backticks, and stray markdown chars
    text = text.replace("**", "").replace("*", "").replace("`", "").replace("~", "").replace("\\", "")

    # 7. Convert introductory colons (e.g. "Services: Facial, Massage") to conversational flow
    text = re.sub(r'([a-zA-Z0-9])\s*:\s+', r'\1, ', text)

    # 8. Clean up colon, comma, or punctuation artifacts e.g. "Chennai: , Dr. Sanjiv" -> "Chennai, Dr. Sanjiv"
    text = re.sub(r':\s*,\s*', ', ', text)
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r'^\s*,\s*', '', text)

    # 9. Collapse all multiple spaces, tab stops, and newlines into a single clean space for smooth continuous speech
    text = re.sub(r'\s+', ' ', text).strip()

    # 10. Deduplicate repeated sentences or phrases within the same message
    text = deduplicate_repeated_sentences(text)

    return text

def deduplicate_repeated_sentences(text: str) -> str:
    """
    Removes duplicated sentences, repeated paragraphs, or multi-word phrases within a single AI response,
    guaranteeing clean, non-repetitive conversational speech for both TTS and UI display.
    """
    if not text or not text.strip():
        return ""

    clean_text = text.strip()

    # 1. Regex to collapse adjacent identical phrases (>= 12 chars)
    collapse_pattern = r'(\b[A-Za-z0-9][^\.!\?]{12,300}?[\.!\?]?)(?:\s+\1)+'
    prev = None
    while prev != clean_text:
        prev = clean_text
        clean_text = re.sub(collapse_pattern, r'\1', clean_text, flags=re.IGNORECASE)

    # 2. Multi-sentence block collapse: look for identical blocks of 20+ chars
    clean_len = len(clean_text)
    for length in range(clean_len // 2, 20, -1):
        for start in range(clean_len - 2 * length + 1):
            block1 = clean_text[start:start + length].strip()
            block2 = clean_text[start + length:start + 2 * length].strip()
            if len(block1) >= 20 and block1.lower() == block2.lower():
                clean_text = clean_text[:start + length] + clean_text[start + 2 * length:]
                clean_len = len(clean_text)
                break

    # 3. Sentence-level deduplication across the whole response with substring containment
    sentence_matches = list(re.finditer(r'([^\.\?!]+[\.\?!]+(?:\s+|$)|[^\.\?!]+$)', clean_text))

    seen_normalized = []
    deduped_sentences = []

    for match in sentence_matches:
        sent_raw = match.group(1).strip()
        norm = re.sub(r'[^\w\s]', '', sent_raw.lower()).strip()
        norm = re.sub(r'\s+', ' ', norm)

        if not norm:
            continue

        # Do not deduplicate tiny words like 'yes.', 'okay.', etc.
        if len(norm.split()) <= 2 and len(norm) < 10:
            deduped_sentences.append(sent_raw)
            continue

        # Check for duplication or substring containment with seen sentences
        is_dup = False
        replace_idx = None
        for idx, seen in enumerate(seen_normalized):
            if norm == seen:
                is_dup = True
                break
            if len(norm) >= 15 and len(seen) >= 15:
                if norm in seen:
                    is_dup = True
                    break
                elif seen in norm:
                    # Check if seen was just a small incomplete fragment (e.g. "I assist you today" vs "How else may I assist you today")
                    if len(seen.split()) <= 5 and len(norm.split()) > len(seen.split()):
                        replace_idx = idx
                        break
                    else:
                        is_dup = True
                        break

        if is_dup:
            continue

        if replace_idx is not None:
            seen_normalized[replace_idx] = norm
            deduped_sentences[replace_idx] = sent_raw
            continue

        seen_normalized.append(norm)
        deduped_sentences.append(sent_raw)

    result = " ".join(deduped_sentences).strip()
    result = re.sub(r'\s+([.!?])', r'\1', result)
    result = re.sub(r'\s+', ' ', result)
    return result

def is_whisper_hallucination(text: str) -> bool:
    """
    Detects repetitive Whisper hallucinations caused by static/silence/no-speech input or acoustic echo.
    E.g. "வடுவடுவடுவடு", "Thank you for watching", "Subtitles by", "See ya", "Und dann...".
    """
    if not text or not text.strip():
        return True

    clean = text.strip()
    lower = clean.lower()

    # 1. Reject non-English script characters during English-only phase (e.g. Tamil, Devanagari, Arabic, Cyrillic, CJK)
    if re.search(r'[\u0b80-\u0bff\u0900-\u097f\u0600-\u06ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uaac0-\uaafe\u0400-\u04ff]', clean):
        return True

    # 2. Known Whisper hallucination phrases on silence/static/echo
    hallucination_phrases = [
        "thank you for watching", "subtitles by", "amara.org", "thanks for watching",
        "like and subscribe", "copyright", "all rights reserved", "bye bye", "mbc",
        "gonna let you go", "see ya", "und dann", "thanks gonna", "thank you. bye",
        "you're watching", "channel", "video"
    ]
    if any(phrase in lower for phrase in hallucination_phrases):
        return True

    # 3. Detect 2-6 character repetitive loops (e.g., "abcabcabcabc", "123123123123")
    if len(clean) >= 8:
        for sub_len in range(2, 7):
            pattern = rf'(.{{{sub_len}}})\1{{3,}}'
            if re.search(pattern, clean):
                return True

    # 4. Reject single punctuation or stray noise characters
    if len(clean) <= 2 and not clean.isalnum():
        return True

    return False
