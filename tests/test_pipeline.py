import pytest
from models.schemas import TaskContext, TaskRequest
from models.enums import TaskCategory, ComplexityLevel
from pipeline.analyzer import PromptAnalyzer
from pipeline.complexity import ComplexityEstimator

def test_analyzer_math():
    analyzer = PromptAnalyzer()
    prompt = "Calculate 2 + 2"
    profile = analyzer.analyze(prompt)
    assert profile.has_math is True
    assert profile.word_count == 4

def test_classifier_math():
    analyzer = PromptAnalyzer()
    prompt = "What is 5 * 5?"
    profile = analyzer.analyze(prompt)
    assert profile.primary_category == TaskCategory.MATH

def test_complexity_easy():
    analyzer = PromptAnalyzer()
    prompt = "What is the capital of France?"
    req = TaskRequest(task_id="2", prompt=prompt)
    profile = analyzer.analyze(prompt)
    ctx = TaskContext(request=req, profile=profile, category=TaskCategory.FACTUAL)
    
    comp = ComplexityEstimator.estimate(ctx)
    assert comp == ComplexityLevel.EASY
