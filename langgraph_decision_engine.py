# langgraph_decision_engine.py

# ... (imports remain the same) ...
import os
import json
from typing import Dict, TypedDict, Optional, Literal, Callable, Any, List
from functools import partial
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field # Use Pydantic v2 directly
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from dotenv import load_dotenv
import operator
import warnings

from IPython.core.debugger import set_trace

# --- Debug Configuration ---
# Set LANGGRAPH_DEBUG=True in your environment or .env file to enable debug prints

DEBUG_MODE = os.getenv("LANGGRAPH_DEBUG", "False").lower() == "true"
#DEBUG_MODE = True

# --- Configuration ---
load_dotenv()
LLM_API_TYPE = os.getenv("LLM_API_TYPE", "OPENAI").upper()
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
TGI_API_BASE_URL = os.getenv("TGI_API_BASE_URL")
TGI_API_KEY = os.getenv("TGI_API_KEY", "no-key-needed")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- State Definition ---
# ... (no changes) ...
class DecisionState(TypedDict):
    input_data: Dict[str, Any]
    decision_result: Optional[Literal["yes", "no"]]
    intermediate_results: Dict[str, Any]
    error: Optional[str]
    current_node: Optional[str]

# --- Helper Functions ---
def _get_llm_client() -> Optional[ChatOpenAI]:
    """Initializes the ChatOpenAI client based on environment variables."""
    if DEBUG_MODE: print(f"--- Configuring LLM Client (API Type: {LLM_API_TYPE}) ---")
    # ... (rest of the function logic is the same, just add DEBUG_MODE checks for prints) ...
    if not LLM_MODEL_NAME: warnings.warn("LLM_MODEL_NAME environment variable is not set.", stacklevel=2); return None
    client = None
    try:
        if LLM_API_TYPE == "OPENAI":
            if not OPENAI_API_KEY: warnings.warn("LLM_API_TYPE is 'OPENAI' but OPENAI_API_KEY is not set.", stacklevel=2); return None
            if DEBUG_MODE: print(f"Using OpenAI API with model: {LLM_MODEL_NAME}")
            client = ChatOpenAI(model=LLM_MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.0);
            if DEBUG_MODE: print("OpenAI client configured.")
        elif LLM_API_TYPE == "TGI":
            if not TGI_API_BASE_URL: warnings.warn("LLM_API_TYPE is 'TGI' but TGI_API_BASE_URL is not set.", stacklevel=2); return None
            if DEBUG_MODE: print(f"Using TGI/OpenAI-compatible API at: {TGI_API_BASE_URL} with model: {LLM_MODEL_NAME}")
            client = ChatOpenAI(model=LLM_MODEL_NAME, openai_api_base=TGI_API_BASE_URL, openai_api_key=TGI_API_KEY, temperature=0.0);
            if DEBUG_MODE: print("TGI client configured.")
        else: warnings.warn(f"Unknown LLM_API_TYPE: '{LLM_API_TYPE}'. Set to 'OPENAI' or 'TGI'.", stacklevel=2); return None
        return client
    except Exception as e: warnings.warn(f"Error initializing LLM client for type {LLM_API_TYPE}: {e}", stacklevel=2); return None


def _get_input_results(state: DecisionState, input_keys: List[str], node_name: str) -> Optional[List[bool]]:
    """Helper to safely retrieve boolean results for logic gates."""
    results = []
    if DEBUG_MODE:
        print(f"DEBUG [_get_input_results for {node_name}]: Expecting keys {input_keys}")
        print(f"DEBUG [_get_input_results for {node_name}]: Full intermediate state: {state.get('intermediate_results')}")

    for key in input_keys:
        if DEBUG_MODE: print(f"DEBUG [_get_input_results for {node_name}]: Checking key '{key}'")
        value = state['intermediate_results'].get(key)
        if DEBUG_MODE: print(f"DEBUG [_get_input_results for {node_name}]: Value for key '{key}' is '{value}' (Type: {type(value)})")

        if isinstance(value, bool):
            results.append(value)
            if DEBUG_MODE: print(f"DEBUG [_get_input_results for {node_name}]: Appended {value}. Current results: {results}")
        else:
            msg = f"Input key '{key}' not found or not a boolean in intermediate_results (value was '{value}')."
            # Keep error print, maybe make warning? Let's keep it for now.
            print(f"Error in {node_name} (via _get_input_results): {msg}")
            state['error'] = f"Missing or invalid input '{key}' for logic gate {node_name}"
            state['decision_result'] = None
            return None
    if DEBUG_MODE: print(f"DEBUG [_get_input_results for {node_name}]: Successfully retrieved all keys. Returning: {results}")
    return results

# --- Tool Definition ---
# ... (Keep DecisionArgs) ...
class DecisionArgs(BaseModel):
    decision: bool = Field(description="The boolean decision based on the query, True for yes, False for no.")

# In langgraph_decision_engine.py

def simple_conditional_node(state: DecisionState, condition_func: Callable[[Dict], bool], node_name: str) -> DecisionState:
    """BASE NODE: Executes a simple boolean condition, stores boolean result, sets decision_result."""
    # Set current node and clear previous error
    state['current_node'] = node_name
    state['error'] = None
    result_key = f"{node_name}_result"

    # Unconditional print to track node execution
    print(f"--- Running Node (Simple): {node_name} ---")

    try:
        # --- Execute condition function ---
        result_bool: bool = condition_func(state['input_data'])

        # --- Determine 'yes'/'no' string for routing ---
        decision: Literal["yes", "no"] = "yes" if result_bool else "no"

        # --- Store the ACTUAL boolean result ---
        state['intermediate_results'][result_key] = result_bool

        # --- Store the routing decision string ---
        state['decision_result'] = decision

        # --- Original debug print (now conditional) ---
        if DEBUG_MODE:
            print(f"Condition Result (Debug): {decision} ({result_bool})")

    except Exception as e:
        # Keep error prints visible
        print(f"Error in {node_name}: {e}")
        state['error'] = str(e)
        # Always store False on error
        state['intermediate_results'][result_key] = False
        state['decision_result'] = "no"

    return state
    
def llm_decision_node(state: DecisionState, prompt_template_str: str, node_name: str) -> DecisionState:
    """BASE NODE: Makes a decision via LLM using Tool Calling for structured output."""
    if DEBUG_MODE: print(f"--- Running Node: {node_name} (using Tool Calling) ---")
    state['current_node'] = node_name; state['error'] = None; state['decision_result'] = None; result_key = f"{node_name}_result"; raw_response_key = f"{node_name}_raw_response"; result_bool = False
    llm_client = _get_llm_client()
    if not llm_client: state['error'] = "LLM client initialization failed."; state['decision_result'] = "no"; state['intermediate_results'][result_key] = False; return state
    decision_tool = convert_to_openai_tool(DecisionArgs); llm_with_tool = llm_client.bind_tools([decision_tool], tool_choice={"type": "function", "function": {"name": decision_tool["function"]["name"]}})
    try:
        prompt_template = ChatPromptTemplate.from_template(prompt_template_str); prompt_input_dict = {k: v for k, v in state['input_data'].items() if f"{{{k}}}" in prompt_template_str}; prompt_value = prompt_template.format_prompt(**prompt_input_dict)
        if DEBUG_MODE: print(f"Formatting prompt with data: {prompt_input_dict}")
    except KeyError as e: print(f"Error: Missing key '{e}' in input_data for prompt formatting in {node_name}."); state['error'] = f"Missing data for prompt: {e}"; state['decision_result'] = "no"; state['intermediate_results'][result_key] = False; return state
    except Exception as e: print(f"Error formatting prompt in {node_name}: {e}"); state['error'] = f"Prompt formatting error: {e}"; state['decision_result'] = "no"; state['intermediate_results'][result_key] = False; return state
    try:
        if DEBUG_MODE: print(f"Sending request to LLM (Type: {LLM_API_TYPE}) expecting tool call...")
        ai_message = llm_with_tool.invoke(prompt_value)
        if DEBUG_MODE: print(f"LLM Raw Response Message: {ai_message}")
        # Storing raw message removed previously
        tool_call = next(iter(ai_message.tool_calls), None) if ai_message.tool_calls else None
        if tool_call and tool_call.get("name") == decision_tool["function"]["name"]:
            tool_args = tool_call.get("args")
            if tool_args:
                try: 
                    parsed_args = DecisionArgs.parse_obj(tool_args); result_bool = parsed_args.decision
                    if DEBUG_MODE: print(f"LLM Tool Call Parsed Decision: {result_bool}")
                except Exception as pydantic_error: print(f"Error parsing tool arguments in {node_name}: {pydantic_error}"); state['error'] = f"LLM tool arg parsing error: {pydantic_error}"; result_bool = False
            else: print(f"Error: Tool call for '{decision_tool['function']['name']}' has no arguments."); state['error'] = "LLM tool call missing arguments."; result_bool = False
        else: print(f"Error: Expected tool call '{decision_tool['function']['name']}' not found in LLM response."); state['error'] = "LLM did not make the expected tool call."; print(f"LLM actual content: {ai_message.content}"); result_bool = False
    except Exception as e: print(f"Error during LLM tool call in {node_name} (Type: {LLM_API_TYPE}): {e}"); state['error'] = f"LLM API/Tool Call Error: {e}"; result_bool = False
    decision_str: Literal["yes", "no"] = "yes" if result_bool else "no"; state['intermediate_results'][result_key] = result_bool; state['decision_result'] = decision_str
    if DEBUG_MODE: print(f"LLM Final Decision: {decision_str} ({result_bool})")
    return state

def and_gate_node(state: DecisionState, input_keys: List[str], node_name: str) -> DecisionState:
    """BASE NODE: Performs a logical AND on results stored in intermediate_results."""
    if DEBUG_MODE: print(f"--- Running Node: {node_name} (Inputs Required: {input_keys}) ---")
    state['current_node'] = node_name; state['error'] = None; result_key = f"{node_name}_result"
    input_bools = _get_input_results(state, input_keys, node_name)
    if DEBUG_MODE: print(f"DEBUG [{node_name}]: Retrieved input_bools: {input_bools} (Type: {type(input_bools)})")
    if input_bools is None:
        if DEBUG_MODE: print(f"DEBUG [{node_name}]: input_bools is None, setting result to False due to retrieval error.")
        state['intermediate_results'][result_key] = False; state['decision_result'] = "no"; return state
    try:
        if DEBUG_MODE: print(f"DEBUG [{node_name}]: Inputs to all() function: {input_bools}")
        all_result = all(input_bools)
        if DEBUG_MODE: print(f"DEBUG [{node_name}]: Result of all({input_bools}) = {all_result} (Type: {type(all_result)})")
        final_result_bool = all_result
    except Exception as e_all: print(f"DEBUG [{node_name}]: ERROR during all() call: {e_all}"); state['error'] = f"Error during AND logic execution: {e_all}"; state['intermediate_results'][result_key] = False; state['decision_result'] = "no"; return state
    final_decision: Literal["yes", "no"] = "yes" if final_result_bool else "no"; state['decision_result'] = final_decision; state['intermediate_results'][result_key] = final_result_bool
    if DEBUG_MODE: print(f"AND Gate Result: {final_decision} ({final_result_bool})")
    return state

# --- Add DEBUG_MODE checks similarly to or_gate_node, not_gate_node, terminal_node ---
def or_gate_node(state: DecisionState, input_keys: List[str], node_name: str) -> DecisionState:
    if DEBUG_MODE: print(f"--- Running Node: {node_name} (Inputs Required: {input_keys}) ---")
    state['current_node'] = node_name; state['error'] = None; result_key = f"{node_name}_result"; input_bools = _get_input_results(state, input_keys, node_name)
    if DEBUG_MODE: print(f"DEBUG [{node_name}]: Retrieved input_bools: {input_bools} (Type: {type(input_bools)})")
    if input_bools is None: state['intermediate_results'][result_key] = False; state['decision_result'] = "no"; return state
    try: final_result_bool = any(input_bools)
    except Exception as e_any: print(f"DEBUG [{node_name}]: ERROR during any() call: {e_any}"); state['error'] = f"Error during OR logic execution: {e_any}"; state['intermediate_results'][result_key] = False; state['decision_result'] = "no"; return state
    final_decision: Literal["yes", "no"] = "yes" if final_result_bool else "no"; state['decision_result'] = final_decision; state['intermediate_results'][result_key] = final_result_bool
    if DEBUG_MODE: print(f"OR Gate Result: {final_decision} ({final_result_bool})"); return state

def not_gate_node(state: DecisionState, input_key: str, node_name: str) -> DecisionState:
    if DEBUG_MODE: print(f"--- Running Node: {node_name} (Input Required: {input_key}) ---")
    state['current_node'] = node_name; state['error'] = None; result_key = f"{node_name}_result"; input_bools = _get_input_results(state, [input_key], node_name)
    if DEBUG_MODE: print(f"DEBUG [{node_name}]: Retrieved input_bools: {input_bools} (Type: {type(input_bools)})")
    if input_bools is None: state['intermediate_results'][result_key] = False; state['decision_result'] = "no"; return state
    try: final_result_bool = not input_bools[0]
    except Exception as e_not: print(f"DEBUG [{node_name}]: ERROR during not call: {e_not}"); state['error'] = f"Error during NOT logic execution: {e_not}"; state['intermediate_results'][result_key] = False; state['decision_result'] = "no"; return state
    final_decision: Literal["yes", "no"] = "yes" if final_result_bool else "no"; state['decision_result'] = final_decision; state['intermediate_results'][result_key] = final_result_bool
    if DEBUG_MODE: print(f"NOT Gate Result: {final_decision} ({final_result_bool})"); return state

def terminal_node(state: DecisionState, outcome: str, node_name: str) -> DecisionState:
    """BASE NODE: Represents a final outcome."""
    # Keep this print as it signifies reaching the end
    print(f"--- Reached Terminal Node: {node_name} ({outcome}) ---")
    state['current_node'] = node_name; state['intermediate_results']['final_outcome'] = outcome; state['decision_result'] = None; state['error'] = None; return state

# --- Routing Logic ---
# In langgraph_decision_engine.py

def route_binary_decision(state: DecisionState) -> Literal["yes", "no", "__error__"]:
    """Determines the next node based on 'decision_result', handles errors."""

    if DEBUG_MODE:
        print(f"--- Routing Decision from node '{state.get('current_node', 'N/A')}' ---")

    # Check for errors first
    if state.get('error'):
        # Keep error prints visible
        print(f"Routing to error handler due to error: {state['error']}")
        # Return on its own line
        return "__error__"

    # Get the decision from the previous node
    decision = state.get('decision_result')

    if DEBUG_MODE:
        print(f"Decision to route: {decision}")

    # Route based on the decision string
    if decision == "yes":
        # Return on its own line
        return "yes"
    elif decision == "no":
        # Return on its own line
        return "no"
    else:
        # Handle unexpected/None decision results
        # Keep warning visible
        warnings.warn(f"Decision result is '{decision}' from node '{state.get('current_node', 'N/A')}'. Routing to '__error__'.", stacklevel=2)
        # Return on its own line
        return "__error__"

# End of module langgraph_decision_engine.py