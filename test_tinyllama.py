import asyncio
from engine.executors.local_llm import LocalLLMExecutor
from models.schemas import TaskContext, PromptProfile, TaskRequest
from models.enums import TaskCategory

async def run_test():
    executor = LocalLLMExecutor(model_path="/model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    if not executor.workers:
        print("No workers initialized")
        return
        
    print("\n=== ISOLATED TINYLLAMA BENCHMARK ===\n")
    
    # Test 1: Sentiment
    prompt1 = "Classify the sentiment of this customer review as Positive, Negative, or Neutral and give a one-sentence reason: 'The product arrived two days late and the packaging was damaged, but the item worked perfectly and customer support resolved my complaint within an hour.' "
    ctx1 = TaskContext(
        request=TaskRequest(task_id="T03", prompt=prompt1), category=TaskCategory.SENTIMENT,
        profile=PromptProfile(word_count=10, estimated_output_tokens=60, complexity_score=0.5)
    )
    print("Testing Sentiment...")
    ans1 = await executor.execute(ctx1)
    print(f"Sentiment Output:\n{ans1.answer}\n")

    # Test 2: NER
    prompt2 = "Extract all named entities from the following text and label each as PERSON, ORGANIZATION, LOCATION, or DATE:\n'On March 15 2023, Sundar Pichai announced that Google would open a new AI research lab in Zurich, partnering with ETH Zurich to focus on large language model safety.' "
    ctx2 = TaskContext(
        request=TaskRequest(task_id="T05", prompt=prompt2), category=TaskCategory.NER,
        profile=PromptProfile(word_count=10, estimated_output_tokens=250, complexity_score=0.5)
    )
    print("Testing NER (Tokens: max(256))...")
    ans2 = await executor.execute(ctx2)
    print(f"NER Output:\n{ans2.answer}\n")

if __name__ == "__main__":
    asyncio.run(run_test())
