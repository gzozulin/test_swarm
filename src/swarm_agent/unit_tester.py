import builtins
import contextlib
import io

from langgraph_codeact import create_codeact, create_default_prompt
from typing_extensions import Any

from swarm_agent.filenames import bank_account, tests_filename
from swarm_agent.models import llm_large
from swarm_agent.utils import cat_file

unit_tester_tools = []  # Essentially it is empty, the agent has eval tool only

unit_tester_agent_prompt = """
You are the Unit Test Agent. 
Your task is to produce a single unit test for the test case provided.
After the unit test is produced, you need to run it to check if it passes.

1. To run and iterate on the test case, output the test as code,
2. As a FINAL answer, you need to return ONLY the test case produced,
IN PLAIN TEXT (without any code block):
-----------------------------
Existing code:
{code}
-----------------------------
Existing tests:
{tests}
"""

def eval_fn(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            exec(code, builtins.__dict__, _locals)
        result = stdout.getvalue()
        if not result:
            result = "<code ran, no output printed to stdout>"
    except Exception as e:
        result = f"Error during execution: {repr(e)}"
    return result, {}

unit_tester_agent_prompt = (
    create_default_prompt(unit_tester_tools, unit_tester_agent_prompt))

unit_tester_agent_prompt = (
    unit_tester_agent_prompt.format(code=cat_file(bank_account), tests=cat_file(tests_filename)))

unit_tester_agent_builder = create_codeact(
    llm_large, prompt=unit_tester_agent_prompt, tools=unit_tester_tools, eval_fn=eval_fn)
