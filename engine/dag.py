from models.schemas import TaskContext, ExecutionResult
import asyncio
from typing import List, Callable, Awaitable

class DAGNode:
    def __init__(self, task_id: str, executable: Callable[[TaskContext], Awaitable[ExecutionResult]], context: TaskContext):
        self.task_id = task_id
        self.executable = executable
        self.context = context
        self.dependencies: List['DAGNode'] = []
        self.result: ExecutionResult = None
        
    def add_dependency(self, node: 'DAGNode'):
        self.dependencies.append(node)

class DAGEngine:
    def __init__(self):
        self.nodes = []
        
    def add_node(self, node: DAGNode):
        self.nodes.append(node)
        
    async def execute_graph(self):
        """
        Executes the DAG using an asynchronous topological sort.
        """
        pending = set(self.nodes)
        completed = set()
        running_tasks = {}
        
        while pending or running_tasks:
            # Find nodes ready to run (all dependencies met)
            ready_nodes = [
                n for n in pending 
                if all(d in completed for d in n.dependencies)
            ]
            
            for node in ready_nodes:
                pending.remove(node)
                task = asyncio.create_task(node.executable(node.context))
                running_tasks[task] = node
                
            if not running_tasks:
                if pending:
                    raise RuntimeError("DAG Deadlock detected!")
                break
                
            # Wait for at least one task to finish
            done, _ = await asyncio.wait(
                running_tasks.keys(), 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                node = running_tasks.pop(task)
                node.result = task.result()
                completed.add(node)
                
        return [n.result for n in self.nodes]
