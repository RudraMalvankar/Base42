import pytest
from models.schemas import TaskContext, TaskRequest, AnalyzerMetadata
from models.enums import TaskCategory, ComplexityLevel
from pipeline.analyzer import PromptAnalyzer
from pipeline.classifier import TaskClassifier
from pipeline.complexity import ComplexityEstimator

def test_analyzer_math():
    prompt = "Calculate 2 + 2"
    meta = PromptAnalyzer.analyze(prompt)
    assert meta.has_math is True
    assert meta.word_count == 4

def test_classifier_math():
    prompt = "What is 5 * 5?"
    req = TaskRequest(task_id="1", prompt=prompt)
    meta = PromptAnalyzer.analyze(prompt)
    ctx = TaskContext(request=req, metadata=meta)
    
    category = TaskClassifier.classify(ctx)
    assert category == TaskCategory.MATH

def test_complexity_easy():
    prompt = "What is the capital of France?"
    req = TaskRequest(task_id="2", prompt=prompt)
    meta = PromptAnalyzer.analyze(prompt)
    ctx = TaskContext(request=req, metadata=meta, category=TaskCategory.FACTUAL)
    
    comp = ComplexityEstimator.estimate(ctx)
    assert comp == ComplexityLevel.EASY
