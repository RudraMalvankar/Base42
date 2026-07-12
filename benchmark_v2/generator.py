import json
import random

def generate_tasks():
    tasks = []
    
    # 1. Factual
    capitals = [("France", "Paris"), ("Japan", "Tokyo"), ("Brazil", "Brasilia"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("South Korea", "Seoul"), ("India", "New Delhi")]
    planets = [("Mars", "4th"), ("Jupiter", "5th"), ("Saturn", "6th"), ("Venus", "2nd"), ("Neptune", "8th"), ("Uranus", "7th"), ("Earth", "3rd"), ("Mercury", "1st")]
    inventors = [("Telephone", "Alexander Graham Bell"), ("Lightbulb", "Thomas Edison"), ("Airplane", "Wright Brothers"), ("Penicillin", "Alexander Fleming"), ("World Wide Web", "Tim Berners-Lee")]
    
    for i in range(10):
        c, a = capitals[i % len(capitals)]
        tasks.append({"task_id": f"factual-easy-{i+1}", "category": "factual", "difficulty": "easy", "prompt": f"What is the capital city of {c}?", "expected_output": a, "grading_rubric": "exact_match"})
        
    for i in range(10):
        p, a = planets[i % len(planets)]
        tasks.append({"task_id": f"factual-medium-{i+1}", "category": "factual", "difficulty": "medium", "prompt": f"Which planet from the sun is {p}?", "expected_output": a, "grading_rubric": "exact_match"})
        
    for i in range(10):
        inv, a = inventors[i % len(inventors)]
        tasks.append({"task_id": f"factual-hard-{i+1}", "category": "factual", "difficulty": "hard", "prompt": f"Who is credited with the invention of the {inv}?", "expected_output": a, "grading_rubric": f"contains: {a.split()[-1]}"})

    # 2. Math
    for i in range(10):
        a, b = random.randint(10, 50), random.randint(10, 50)
        tasks.append({"task_id": f"math-easy-{i+1}", "category": "math", "difficulty": "easy", "prompt": f"What is {a} + {b}?", "expected_output": str(a+b), "grading_rubric": "exact_match"})
        
    for i in range(10):
        a, b = random.randint(100, 500), random.randint(5, 20)
        tasks.append({"task_id": f"math-medium-{i+1}", "category": "math", "difficulty": "medium", "prompt": f"Calculate {a} * {b}.", "expected_output": str(a*b), "grading_rubric": "exact_match"})
        
    for i in range(10):
        a, b, c = random.randint(10, 50), random.randint(10, 50), random.randint(2, 5)
        tasks.append({"task_id": f"math-hard-{i+1}", "category": "math", "difficulty": "hard", "prompt": f"A store has {a} apples and {b} oranges. They divide all fruit equally into {c} baskets. How many total fruits are in each basket?", "expected_output": str((a+b)//c) if (a+b)%c==0 else str((a+b)/c), "grading_rubric": "exact_match"})

    # 3. Sentiment
    pos = ["I love this product, it works perfectly!", "Absolutely fantastic experience.", "Highly recommend to everyone.", "Best purchase ever.", "Amazing quality and fast shipping."]
    neg = ["Terrible service, completely broken.", "Do not buy this, it is a scam.", "I hate it, worst thing I've ever used.", "Awful quality.", "Disappointed with the results."]
    neu = ["The item arrived on Tuesday.", "It is a standard tool.", "The color is blue.", "It functions as described.", "I have no strong feelings about it."]
    
    for i in range(10):
        tasks.append({"task_id": f"sentiment-easy-{i+1}", "category": "sentiment", "difficulty": "easy", "prompt": f"Classify the sentiment: '{pos[i%len(pos)]}'", "expected_output": "positive", "grading_rubric": "contains: positive"})
    for i in range(10):
        tasks.append({"task_id": f"sentiment-medium-{i+1}", "category": "sentiment", "difficulty": "medium", "prompt": f"Classify the sentiment: '{neg[i%len(neg)]}'", "expected_output": "negative", "grading_rubric": "contains: negative"})
    for i in range(10):
        tasks.append({"task_id": f"sentiment-hard-{i+1}", "category": "sentiment", "difficulty": "hard", "prompt": f"Classify the sentiment: '{neu[i%len(neu)]}'", "expected_output": "neutral", "grading_rubric": "contains: neutral"})

    # 4. Summarization
    for i in range(10):
        tasks.append({"task_id": f"summarization-easy-{i+1}", "category": "summarization", "difficulty": "easy", "prompt": "Summarize in 3 words: The quick brown fox jumps over the lazy dog.", "expected_output": "fox jumps dog", "grading_rubric": "semantic_similarity"})
    for i in range(10):
        tasks.append({"task_id": f"summarization-medium-{i+1}", "category": "summarization", "difficulty": "medium", "prompt": "Summarize this: AI is transforming industries by automating tasks, improving data analysis, and enhancing customer experiences.", "expected_output": "AI automates tasks and improves industries.", "grading_rubric": "semantic_similarity"})
    for i in range(10):
        tasks.append({"task_id": f"summarization-hard-{i+1}", "category": "summarization", "difficulty": "hard", "prompt": "Provide a concise summary: Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight and turn it into chemical energy.", "expected_output": "Plants convert sunlight into chemical energy.", "grading_rubric": "semantic_similarity"})

    # 5. NER
    for i in range(10):
        tasks.append({"task_id": f"ner-easy-{i+1}", "category": "ner", "difficulty": "easy", "prompt": "Extract the person name: John went to the store.", "expected_output": "John", "grading_rubric": "contains: John"})
    for i in range(10):
        tasks.append({"task_id": f"ner-medium-{i+1}", "category": "ner", "difficulty": "medium", "prompt": "Extract the organization: Google announced a new product today.", "expected_output": "Google", "grading_rubric": "contains: Google"})
    for i in range(10):
        tasks.append({"task_id": f"ner-hard-{i+1}", "category": "ner", "difficulty": "hard", "prompt": "Extract the location: The conference will be held in San Francisco, California next year.", "expected_output": "San Francisco, California", "grading_rubric": "contains: San Francisco"})

    # 6. Logic
    for i in range(10):
        tasks.append({"task_id": f"logic-easy-{i+1}", "category": "logic", "difficulty": "easy", "prompt": "If A > B and B > C, is A > C? Answer Yes or No.", "expected_output": "Yes", "grading_rubric": "contains: Yes"})
    for i in range(10):
        tasks.append({"task_id": f"logic-medium-{i+1}", "category": "logic", "difficulty": "medium", "prompt": "All blips are blops. Some blops are bloops. Are all blips bloops? Answer Yes, No, or Unknown.", "expected_output": "Unknown", "grading_rubric": "contains: Unknown"})
    for i in range(10):
        tasks.append({"task_id": f"logic-hard-{i+1}", "category": "logic", "difficulty": "hard", "prompt": "There are 3 doors. Behind one is a car, behind two are goats. You pick Door 1. The host opens Door 3 to reveal a goat. Should you switch to Door 2? Answer Yes or No.", "expected_output": "Yes", "grading_rubric": "contains: Yes"})

    # 7. Code Generation
    for i in range(10):
        tasks.append({"task_id": f"code_generation-easy-{i+1}", "category": "code_generation", "difficulty": "easy", "prompt": "Write a Python function to add two numbers a and b.", "expected_output": "def add(a, b): return a + b", "grading_rubric": "contains: return a + b"})
    for i in range(10):
        tasks.append({"task_id": f"code_generation-medium-{i+1}", "category": "code_generation", "difficulty": "medium", "prompt": "Write a Python function to reverse a string s.", "expected_output": "return s[::-1]", "grading_rubric": "contains: [::-1]"})
    for i in range(10):
        tasks.append({"task_id": f"code_generation-hard-{i+1}", "category": "code_generation", "difficulty": "hard", "prompt": "Write a Python function to check if a number n is prime.", "expected_output": "n % i == 0", "grading_rubric": "contains: %"})

    # 8. Code Debugging
    for i in range(10):
        tasks.append({"task_id": f"code_debugging-easy-{i+1}", "category": "code_debugging", "difficulty": "easy", "prompt": "Fix this code: def add(a, b) return a + b", "expected_output": "def add(a, b):", "grading_rubric": "contains: :"})
    for i in range(10):
        tasks.append({"task_id": f"code_debugging-medium-{i+1}", "category": "code_debugging", "difficulty": "medium", "prompt": "Fix the loop bounds: for i in range(len(arr) + 1): print(arr[i])", "expected_output": "range(len(arr))", "grading_rubric": "contains: range(len(arr))"})
    for i in range(10):
        tasks.append({"task_id": f"code_debugging-hard-{i+1}", "category": "code_debugging", "difficulty": "hard", "prompt": "Fix the recursion limit issue: def fact(n): return n * fact(n-1)", "expected_output": "if n == 0", "grading_rubric": "contains: if"})

    with open("tasks.json", "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=4)
        
    print(f"Generated {len(tasks)} benchmark tasks.")

if __name__ == "__main__":
    generate_tasks()
