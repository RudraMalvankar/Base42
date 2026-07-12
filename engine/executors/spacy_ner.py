import json
import logging
from typing import List, Dict, Any
from models.schemas import TaskContext, ExecutionResult
from models.enums import ExecutionRoute

logger = logging.getLogger("spacy_ner")

class SpacyNERExecutor:
    def __init__(self):
        self.nlp = None
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_md")
                logger.info("Loaded en_core_web_md for spaCy NER.")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded en_core_web_sm for spaCy NER.")
                except OSError:
                    logger.error("Failed to load any spaCy model.")
        except ImportError:
            logger.error("spaCy not installed.")

    async def execute(self, context: TaskContext) -> ExecutionResult:
        result = ExecutionResult(
            task_id=context.request.task_id,
            answer="",
            route_taken=ExecutionRoute.PYTHON,  # Route taken could be local conceptually, but it saves api tokens.
            fallback_triggered=True,
            fireworks_tokens=0
        )

        if not self.nlp:
            logger.warning("spaCy NLP model is not loaded. Triggering fallback.")
            return result

        try:
            doc = self.nlp(context.request.prompt)
            entities = []
            seen = set()

            mapping = {
                "PERSON": "PERSON",
                "ORG": "ORGANIZATION",
                "GPE": "LOCATION",
                "LOC": "LOCATION",
                "DATE": "DATE"
            }

            for ent in doc.ents:
                mapped_label = mapping.get(ent.label_)
                if mapped_label:
                    # Normalize whitespace
                    text = " ".join(ent.text.split())
                    key = (text, mapped_label)
                    if key not in seen:
                        seen.add(key)
                        entities.append({
                            "entity": text,
                            "label": mapped_label
                        })

            if entities:
                result.answer = json.dumps(entities)
                result.fallback_triggered = False
            else:
                # If no entities found, it might be a failure to extract. Trigger fallback.
                result.fallback_triggered = True

        except Exception as e:
            logger.error(f"spaCy NER failed: {e}")
            result.fallback_triggered = True

        return result
