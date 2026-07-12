"""
Layer 1: Structural Extractor
Zero-cost, zero-latency, always runs.
Extracts deterministic features from raw prompt text.
"""
import re
import ast
from pipeline.prompt_models import StructuralFeatures, QuestionType

class StructuralExtractor:
    _MATH_PATTERN = re.compile(
        r'(\d+\s*[\+\-\*\/\^]\s*\d+|calculate|compute|how many|total|average)',
        re.IGNORECASE
    )
    _CODE_PATTERN = re.compile(
        r'(```|def |class |import |#include|function\s*\(|write.*code|debug|fix.*bug)',
        re.IGNORECASE
    )
    _NER_PATTERN = re.compile(
        r'(extract|entities|named entity|who is|find all|list all people|locations|organizations)',
        re.IGNORECASE
    )
    _SENTIMENT_PATTERN = re.compile(
        r'(sentiment|positive.{0,10}negative|classify.*review|classify.*tweet|tone of this)',
        re.IGNORECASE
    )
    _SUMMARY_PATTERN = re.compile(
        r'(summar|condense|tl;?dr|in (exactly )?\w+ (sentence|bullet|word)s?)',
        re.IGNORECASE
    )
    _LOGIC_OPS = re.compile(
        r'\b(if|not|and|or|only if|therefore|implies|unless|neither|either)\b',
        re.IGNORECASE
    )
    _QUESTION_STARTERS = {
        QuestionType.WHAT: re.compile(r'^(what|what\'s)\b', re.IGNORECASE),
        QuestionType.HOW: re.compile(r'^(how|how\'s)\b', re.IGNORECASE),
        QuestionType.WHICH: re.compile(r'^(which|whose)\b', re.IGNORECASE),
        QuestionType.IMPERATIVE: re.compile(r'^(write|create|generate|fix|debug|summarize|extract|find|list)\b', re.IGNORECASE),
    }

    def extract(self, prompt: str) -> StructuralFeatures:
        words = prompt.split()
        word_count = len(words)
        token_estimate = max(1, len(prompt) // 4)

        has_code_block = bool(self._CODE_PATTERN.search(prompt))
        has_math = bool(self._MATH_PATTERN.search(prompt))
        has_ner = bool(self._NER_PATTERN.search(prompt))
        has_sentiment = bool(self._SENTIMENT_PATTERN.search(prompt))
        has_summary = bool(self._SUMMARY_PATTERN.search(prompt))
        logic_count = len(self._LOGIC_OPS.findall(prompt))

        # Attempt AST parse for code detection
        if not has_code_block:
            try:
                ast.parse(prompt)
                # Only mark as code if it actually compiles as Python
                has_code_block = "def " in prompt or "class " in prompt
            except SyntaxError:
                pass

        # Determine question type
        question_type = QuestionType.OPEN
        first_word = words[0].lower() if words else ""
        for qt, pattern in self._QUESTION_STARTERS.items():
            if pattern.match(prompt.strip()):
                question_type = qt
                break

        # Compute structural confidence
        # High confidence = strong, unambiguous single-domain signal
        signals = sum([has_code_block, has_math, has_ner, has_sentiment, has_summary])
        if signals == 1:
            structural_confidence = 0.95  # Single clear signal
        elif signals == 0:
            structural_confidence = 0.30  # No structural signals — ambiguous
        else:
            structural_confidence = 0.55  # Multiple competing signals — ambiguous

        return StructuralFeatures(
            word_count=word_count,
            token_estimate=token_estimate,
            has_code_block=has_code_block,
            has_math_expression=has_math,
            has_ner_signal=has_ner,
            has_sentiment_signal=has_sentiment,
            has_summary_signal=has_summary,
            logical_operator_count=logic_count,
            question_type=question_type,
            structural_confidence=structural_confidence,
        )
