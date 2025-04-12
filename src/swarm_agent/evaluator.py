import subprocess

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph_swarm import create_handoff_tool

from swarm_agent.filenames import bank_account, tests_filename
from swarm_agent.filenames import target_dir, coverage_json
from swarm_agent.models import llm_large
from swarm_agent.utils import cat_file

evaluator_prompt = """
You are the Evaluator.
Given the tools and the class and the tests, 
your task is to evaluate the test coverage of the class.
-----------------------------
Code to evaluate:
{code}
-----------------------------
Existing tests:
{tests}
"""

@tool
def evaluate_coverage():
    """Evaluate the test coverage of the target class."""
    commands = [
        ['coverage', 'erase'],
        ['coverage', 'run', 'tests.py'],
        ['coverage', 'report'],
        ['coverage', 'json'],
    ]

    for cmd in commands:
        subprocess.run(cmd, cwd=target_dir, check=True)

    result = subprocess.run(
        ['cat', coverage_json],
        cwd=target_dir,
        capture_output=True,
        check=True
    )

    return result.stdout.decode('utf-8')

transfer_to_tester_tool = create_handoff_tool(
    agent_name="tester",
    description="Transfer user to the tester to create tests")

def run_evaluator(_):
    prompt = ChatPromptTemplate([('system', evaluator_prompt)])
    messages = prompt.format_messages(
        code=cat_file(bank_account), tests=cat_file(tests_filename))
    agent = create_react_agent(
        llm_large, [evaluate_coverage], checkpointer=MemorySaver())
    result = agent.invoke({'messages': messages})['messages']
    return {'messages': result}

def transfer_to_tester(state):
    messages = (state.get('messages') +
        [('human', 'Summarize the convo and transfer to tester if needed.'
         'If no tests are needed (coverage > 80%), do not call any tools.')])
    result = llm_large.bind_tools([transfer_to_tester_tool]).invoke(messages)
    return {'messages': [result]}

def route_after_transfer(state):
    last_message = state['messages'][-1]
    if len(last_message.tool_calls) > 0:
        return 'evaluator_tools'
    return END

evaluator_graph_builder = StateGraph(MessagesState)
evaluator_graph_builder.add_node('run_evaluator', run_evaluator)
evaluator_graph_builder.add_node('transfer_to_tester', transfer_to_tester)
evaluator_graph_builder.add_node(
    'evaluator_tools', ToolNode([transfer_to_tester_tool]))

evaluator_graph_builder.set_entry_point('run_evaluator')
evaluator_graph_builder.add_edge('run_evaluator', 'transfer_to_tester')
evaluator_graph_builder.add_conditional_edges(
    'transfer_to_tester', route_after_transfer, ['evaluator_tools', END])

evaluator = evaluator_graph_builder.compile()
evaluator.name = "evaluator"

def test_evaluate_coverage():
    evaluate_coverage.invoke({})

def test_evaluator():
    evaluator.invoke({})
