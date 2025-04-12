from langgraph.checkpoint.memory import MemorySaver
from langgraph_swarm import create_swarm

from swarm_agent.evaluator import evaluator
from swarm_agent.tester import tester

builder = create_swarm(
    [evaluator, tester],
    default_active_agent="evaluator")

graph = builder
graph.name = "Swarm Agent"

def test_full_graph():
    config = {"thread_id": "1", 'recursion_limit': 25}
    compiled = graph.compile(checkpointer=MemorySaver())
    compiled.invoke(config=config, input={})

def test_chart():
    graph.compile().get_graph(xray=2).draw_mermaid_png(
        output_file_path=f'agent_chart.png')
