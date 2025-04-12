from typing import Annotated

from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send, END
from langgraph.graph import add_messages, StateGraph
from pydantic import BaseModel

from swarm_agent.filenames import tests_filename
from swarm_agent.models import llm_large
from swarm_agent.unit_tester import unit_tester_agent_builder
from swarm_agent.utils import cat_file

class TesterState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = []
    requested_unit_tests: list[str] = []
    produced_unit_tests: Annotated[list[AnyMessage], add_messages] = []

def fork_tester(state: TesterState):
    class UnitTestsRequired(BaseModel):
        unit_tests: list[str]
    messages = state.messages + [
        ('human', 'Summarize and split the above into individual cases (no code).')]
    result = llm_large.with_structured_output(UnitTestsRequired).invoke(messages)
    return {'requested_unit_tests': result.unit_tests}

def route_after_tester(state: TesterState):
    return [Send("run_unit_test", {"test": t, "messages": state.messages})
            for t in state.requested_unit_tests]

def run_unit_test(state: TesterState):
    test_case = state.get('test')
    prompt = ChatPromptTemplate([
        ('human', 'Write a unit test for the following case:\n{test_case}')])
    messages = prompt.format_messages(test_case=test_case)
    agent = unit_tester_agent_builder.compile(checkpointer=MemorySaver())
    result = agent.invoke({'messages': messages})
    return {'produced_unit_tests': [result['messages'][-1]]}

def finalize_tests(state: TesterState):
    prompt = ChatPromptTemplate([
        ('human', 'Add unit cases to the existing class.\n'
         'Return code only (plain text, no formatting): to save to a file.\n'
         f'Existing file:\n{cat_file(tests_filename)}'),
        MessagesPlaceholder(variable_name="produced"),
    ])
    result = llm_large.invoke(
        prompt.format_messages(produced=state.produced_unit_tests))

    with open(tests_filename, 'w') as f:
        f.write(result.content)

    return {'messages': [result]}

tester_graph_builder = StateGraph(TesterState)

tester_graph_builder.add_node('fork_tester', fork_tester)
tester_graph_builder.add_node('run_unit_test', run_unit_test)
tester_graph_builder.add_node('finalize_tests', finalize_tests)

tester_graph_builder.set_entry_point('fork_tester')
tester_graph_builder.add_conditional_edges(
    'fork_tester', route_after_tester, ['run_unit_test'])
tester_graph_builder.add_edge('run_unit_test', 'finalize_tests')
tester_graph_builder.add_edge('finalize_tests', END)

tester = tester_graph_builder.compile()
tester.name = "tester"
