import json
import random

categories = {
    "Greetings": ["Hello, how are you?", "Good morning, AI.", "Hi there, can you help me?", "Hey, are you online?", "Greetings! What can you do?"],
    "Factual QA": ["What is the capital of France?", "Who wrote Hamlet?", "What is the speed of light?", "When did the Titanic sink?", "What is photosynthesis?"],
    "Math": ["What is 15% of 850?", "Solve for x: 2x + 5 = 15", "Calculate the square root of 144", "What is the derivative of x^2?", "If I travel 60 mph for 2.5 hours, how far do I go?"],
    "Logical Reasoning": ["If all A are B, and some B are C, are some A definitely C?", "Solve this logic puzzle: Three friends live in three houses...", "If tomorrow is two days after Monday, what day is today?", "A bat and ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball?", "Which word does not belong: Apple, Banana, Carrot, Date?"],
    "Named Entity Recognition": ["Extract entities from: Tim Cook announced the new Apple iPhone in Cupertino on September 12th.", "Find organizations in: The UN met with WHO officials in Geneva.", "Identify people in: Elon Musk and Mark Zuckerberg might fight.", "Extract locations: I traveled from New York to Tokyo via London.", "List dates in: The contract was signed on 2023-10-01 and expires next Friday."],
    "Sentiment Analysis": ["I absolutely loved the movie, it was fantastic!", "The service was terrible and the food was cold.", "It was okay, nothing special but not bad either.", "I am so frustrated with this buggy software!", "What a delightful experience from start to finish."],
    "Summarization": ["Summarize this article about climate change in one sentence...", "Give me a bulleted summary of the 2008 financial crisis.", "Condense this legal document into plain English.", "TLDR for the plot of The Matrix.", "Summarize the key takeaways from the Agile Manifesto."],
    "Python": ["Write a Python script to reverse a string.", "Create a Python decorator that measures execution time.", "Write a Python script to parse a CSV and filter rows.", "How do I use list comprehensions in Python?", "Implement a singleton pattern in Python."],
    "JavaScript": ["Write a JS function to debounce an API call.", "How does event delegation work in the DOM?", "Implement a Promise from scratch in JS.", "Write a React component that fetches data on mount.", "Explain the Node.js event loop."],
    "SQL": ["Write an SQL query to find the second highest salary.", "Explain the difference between INNER and LEFT JOIN.", "Optimize this slow SQL query...", "How do you prevent SQL injection?", "Write a query to group users by month of registration."],
    "Debugging": ["Find the memory leak in this C++ code...", "Why is this React component rendering infinitely?", "Debug this Python race condition in asyncio.", "Fix this NullPointerException in Java.", "Why is my CSS flexbox layout breaking on mobile?"],
    "System Design": ["Design a URL shortener like Bitly.", "Design a chat application like WhatsApp.", "How would you design Netflix's video streaming backend?", "Design a distributed cache system.", "Design a rate limiter for a public API."],
    "Distributed Systems": ["Explain the Paxos consensus algorithm.", "How does Kafka handle partition replication?", "Discuss CAP theorem in the context of Cassandra.", "What is split-brain in a distributed cluster and how do you fix it?", "Explain eventual consistency vs strong consistency."],
    "Cloud Computing": ["How do you set up a VPC peering connection in AWS?", "Explain the difference between Kubernetes NodePort and LoadBalancer.", "Write an IAM policy that restricts S3 access to a specific IP.", "How do you autoscale a stateless microservice in GCP?", "Explain cold starts in AWS Lambda and how to mitigate them."],
    "Security": ["Explain how a CSRF attack works and how to prevent it.", "How do you securely store passwords in a database?", "What is a buffer overflow vulnerability?", "Explain the OAuth 2.0 authorization code flow.", "How do you prevent XSS in a React application?"],
    "Extreme Reasoning": [
        "Design a distributed rate limiter capable of handling 100 million requests per minute across multiple data centers. Compare Token Bucket, Leaky Bucket, Sliding Window, and Fixed Window approaches. Recommend the best architecture, discuss CAP theorem implications, failure scenarios, consistency trade-offs, and provide high-level pseudocode.",
        "You are given a legacy Python microservice with race conditions, deadlocks, memory leaks, and intermittent failures under high concurrency. Explain every issue you expect, propose a refactored architecture using async programming, provide corrected production-ready code, and explain why each change improves scalability.",
        "Design a globally distributed lock manager for a multi-region database. Handle network partitions, clock drift (Spanner/TrueTime constraints), and node crashes. Provide the algorithm for lock acquisition and release.",
        "A Kubernetes cluster is experiencing cascading failures due to OOMKills, DNS resolution timeouts, and persistent volume detach errors. Outline your step-by-step forensic debugging process, the kubectl commands you would run, and the architecture changes to prevent this.",
        "Implement a custom memory allocator in C/C++ that prevents fragmentation and includes a basic garbage collection mechanism. Explain the pointer arithmetic, metadata overhead, and thread-safety considerations."
    ]
}

tasks = []
task_id_counter = 1

# Generate variations to reach 1000 tasks
for i in range(1000):
    # Pick a random category, weighted towards harder categories later
    if i > 900:
        cat_name = "Extreme Reasoning"
    elif i > 700:
        cat_name = random.choice(["System Design", "Distributed Systems", "Cloud Computing", "Security", "Debugging"])
    else:
        cat_name = random.choice(list(categories.keys()))
        
    base_prompt = random.choice(categories[cat_name])
    
    # Add slight variations to ensure uniqueness
    variation = ""
    if i % 2 == 0:
        variation = " Provide examples."
    elif i % 3 == 0:
        variation = " Explain step-by-step."
    elif i % 5 == 0:
        variation = " Focus on performance."
        
    prompt = f"{base_prompt}{variation} (Variant {i})"
    
    tasks.append({
        "task_id": f"{cat_name.lower().replace(' ', '-')}-{task_id_counter}",
        "prompt": prompt
    })
    task_id_counter += 1

with open("C:/Users/mypc/Desktop/amd/input/stress_tasks.json", "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Generated {len(tasks)} tasks to stress_tasks.json")
