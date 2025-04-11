# example_graphs.py

"""
Contains specific example graph builder functions and their
helper condition logic, using the core components from
langgraph_decision_engine.
"""

import re
import os # Import os to read env var
from typing import Dict, List, Callable
from functools import partial
from langgraph.graph import StateGraph, END

# Import necessary components from the core engine module
from langgraph_decision_engine import (
    DEBUG_MODE, # Use debug flag from core module
    DecisionState,
    simple_conditional_node,
    llm_decision_node,
    and_gate_node,
    or_gate_node,
    not_gate_node,
    terminal_node,
    route_binary_decision,
)

# --- Example 1: Simple Sequential Graph ---

def build_simple_sequential_graph():
    """Builds and returns a compiled LangGraph with sequential decisions."""
    # --- ADD THIS LINE ---
    workflow = StateGraph(DecisionState)
    # --- END ADDITION ---

    node_check_value = "seq_ex_check_value_gt_50" # Prefixed for uniqueness
    node_check_sentiment = "seq_ex_llm_sentiment"
    node_pos_outcome = "seq_ex_positive_outcome"
    node_neg_outcome = "seq_ex_negative_outcome"
    node_error = "seq_ex_error_handler"

    # Define node functions
    def check_value_condition(data: Dict) -> bool: return data.get('value', 0) > 50
    check_value_node_func = partial(simple_conditional_node, condition_func=check_value_condition, node_name=node_check_value)

    sentiment_prompt = "Is the sentiment of the text positive? Provide your decision using the 'DecisionArgs' tool.\n\nText: {user_text}"
    llm_sentiment_node_func = partial(llm_decision_node, prompt_template_str=sentiment_prompt, node_name=node_check_sentiment)

    pos_outcome_func = partial(terminal_node, outcome="Positive Path Outcome", node_name=node_pos_outcome)
    neg_outcome_func = partial(terminal_node, outcome="Negative Path Outcome", node_name=node_neg_outcome)
    error_handler_func = partial(terminal_node, outcome="Process Failed", node_name=node_error)

    # Add nodes
    workflow.add_node(node_check_value, check_value_node_func)
    workflow.add_node(node_check_sentiment, llm_sentiment_node_func)
    workflow.add_node(node_pos_outcome, pos_outcome_func)
    workflow.add_node(node_neg_outcome, neg_outcome_func)
    workflow.add_node(node_error, error_handler_func)

    # Define connections
    workflow.set_entry_point(node_check_value)
    workflow.add_conditional_edges(node_check_value, route_binary_decision, {"yes": node_check_sentiment, "no": node_neg_outcome, "__error__": node_error})
    workflow.add_conditional_edges(node_check_sentiment, route_binary_decision, {"yes": node_pos_outcome, "no": node_neg_outcome, "__error__": node_error})

    workflow.add_edge(node_pos_outcome, END)
    workflow.add_edge(node_neg_outcome, END)
    workflow.add_edge(node_error, END)

    # Compile
    app = workflow.compile()
    # Keep this print as it's useful feedback
    print(f"\nCompiled graph from example module: '{build_simple_sequential_graph.__name__}'")
    return app


# --- Example 2: Simple AND Gate Graph ---

def build_and_gate_example_graph():
    """Builds and returns a compiled graph demonstrating a simple AND gate."""
    # --- ADD THIS LINE ---
    workflow = StateGraph(DecisionState)
    # --- END ADDITION ---

    # --- Define Specific Nodes ---
    node_check_value = "and_ex_check_value_gt_50"
    node_check_sentiment = "and_ex_llm_sentiment_check"
    node_and_gate = "and_ex_value_and_sentiment_check"
    node_success = "and_ex_success_outcome"
    node_failure = "and_ex_failure_outcome"
    node_error = "and_ex_error_handler"

    # Decision A: Check if 'value' > 50
    def check_value_condition(data: Dict) -> bool: return data.get('value', 0) > 50
    check_value_node_func = partial(simple_conditional_node, condition_func=check_value_condition, node_name=node_check_value)

    # Decision B: LLM check for positive sentiment (using tool calling)
    sentiment_prompt = ("...") # Keep prompt definition
    llm_sentiment_node_func = partial(llm_decision_node, prompt_template_str=sentiment_prompt, node_name=node_check_sentiment)
    # Re-add prompt
    sentiment_prompt = (
        "Is the sentiment of the following text positive? Provide your decision using the 'DecisionArgs' tool.\n\n"
        "Text: {user_text}"
    )
    llm_sentiment_node_func = partial(llm_decision_node, prompt_template_str=sentiment_prompt, node_name=node_check_sentiment)


    # Logic Gate: AND Gate
    combined_check_and_gate_func = partial(and_gate_node, input_keys=[f"{node_check_value}_result", f"{node_check_sentiment}_result"], node_name=node_and_gate)

    # Terminal Nodes
    success_outcome_func = partial(terminal_node, outcome="Success: Value > 50 AND Sentiment Positive", node_name=node_success)
    failure_outcome_func = partial(terminal_node, outcome="Failure: Condition (Value > 50 AND Sentiment Positive) not met", node_name=node_failure)
    error_handler_func = partial(terminal_node, outcome="Process Failed (AND Example)", node_name=node_error)

    # --- Add Nodes ---
    workflow.add_node(node_check_value, check_value_node_func); workflow.add_node(node_check_sentiment, llm_sentiment_node_func); workflow.add_node(node_and_gate, combined_check_and_gate_func); workflow.add_node(node_success, success_outcome_func); workflow.add_node(node_failure, failure_outcome_func); workflow.add_node(node_error, error_handler_func)

    # --- Define Connections ---
    workflow.set_entry_point(node_check_value); workflow.add_conditional_edges(node_check_value, route_binary_decision, {"yes": node_check_sentiment, "no": node_check_sentiment, "__error__": node_error}); workflow.add_conditional_edges(node_check_sentiment, route_binary_decision, {"yes": node_and_gate, "no": node_and_gate, "__error__": node_error}); workflow.add_conditional_edges(node_and_gate, route_binary_decision, {"yes": node_success, "no": node_failure, "__error__": node_error})
    workflow.add_edge(node_success, END); workflow.add_edge(node_failure, END); workflow.add_edge(node_error, END)

    # Compile
    app = workflow.compile()
    print(f"\nCompiled graph from example module: '{build_and_gate_example_graph.__name__}'")
    return app


# --- Example 3: Word/Poetry Graph ---

# Helper conditions specific to this example
def check_number_condition(input_data: Dict) -> bool:
    paragraph_text = input_data.get('paragraph_text', ''); has_number = bool(re.search(r'\d', paragraph_text));
    if DEBUG_MODE: print(f"DEBUG [check_number_condition]: Result = {has_number}")
    return has_number

def check_is_english_condition(input_data: Dict) -> bool:
    paragraph_language = input_data.get('paragraph_language', '').lower(); is_english = paragraph_language == 'en';
    if DEBUG_MODE: print(f"DEBUG [check_is_english_condition]: Result = {is_english}")
    return is_english

def build_word_poetry_graph(seed_words: List[str]):
    """Builds the Word/Poetry graph example."""
    # --- ADD THIS LINE ---
    workflow = StateGraph(DecisionState)
    # --- END ADDITION ---

    # --- Define Node Names ---
    node_is_english = "wp_check_is_english"; node_word_in_para = "wp_llm_check_word_in_para"; node_number_in_para = "wp_check_number_in_para"; node_or_word_num = "wp_gate_word_or_number"; node_is_poem = "wp_llm_check_is_poem"; node_and_or_poem = "wp_gate_or_and_poem"; node_meets_condition = "wp_terminal_meets_condition"; node_does_not_meet_condition = "wp_terminal_does_not_meet_condition"; node_error = "wp_error_handler_complex"

    # --- Prepare Node Functions ---
    is_english_node_func = partial(simple_conditional_node, condition_func=check_is_english_condition, node_name=node_is_english)
    word_prompt = ("...") # Keep prompt definition
    word_in_para_node_func = partial(llm_decision_node, prompt_template_str=word_prompt, node_name=node_word_in_para)
    number_in_para_node_func = partial(simple_conditional_node, condition_func=check_number_condition, node_name=node_number_in_para)
    or_word_num_gate_func = partial(or_gate_node, input_keys=[f"{node_word_in_para}_result", f"{node_number_in_para}_result"], node_name=node_or_word_num)
    poem_prompt = ("...") # Keep prompt definition
    is_poem_node_func = partial(llm_decision_node, prompt_template_str=poem_prompt, node_name=node_is_poem)
    and_or_poem_gate_func = partial(and_gate_node, input_keys=[f"{node_or_word_num}_result", f"{node_is_poem}_result"], node_name=node_and_or_poem)
    meets_condition_func = partial(terminal_node, outcome="Meets Condition", node_name=node_meets_condition)
    does_not_meet_condition_func = partial(terminal_node, outcome="Does NOT Meet Condition", node_name=node_does_not_meet_condition)
    error_handler_func = partial(terminal_node, outcome="Process Failed Complex", node_name=node_error)
    # Re-paste the full prompts here
    word_prompt = (
        "You are given a list of English seed words and a paragraph written in a specific language.\n"
        "Seed Words: {seed_words_str}\n"
        "Paragraph Language: {paragraph_language}\n"
        "Paragraph Text:\n```\n{paragraph_text}\n```\n\n"
        "Does the Paragraph Text contain any of the Seed Words, potentially translated into the Paragraph Language? "
        "Provide your boolean decision using the 'DecisionArgs' tool."
    )
    word_in_para_node_func = partial(llm_decision_node, prompt_template_str=word_prompt, node_name=node_word_in_para)
    poem_prompt = (
        "Analyze the following text.\n"
        "Text:\n```\n{paragraph_text}\n```\n\n"
        "Is this text written in a poetic style (e.g., verse, meter, rhyme, distinct structure)? "
        "Provide your boolean decision using the 'DecisionArgs' tool."
    )
    is_poem_node_func = partial(llm_decision_node, prompt_template_str=poem_prompt, node_name=node_is_poem)

    # --- Add Nodes to Workflow ---
    workflow.add_node(node_is_english, is_english_node_func); workflow.add_node(node_word_in_para, word_in_para_node_func); workflow.add_node(node_number_in_para, number_in_para_node_func); workflow.add_node(node_or_word_num, or_word_num_gate_func); workflow.add_node(node_is_poem, is_poem_node_func); workflow.add_node(node_and_or_poem, and_or_poem_gate_func); workflow.add_node(node_meets_condition, meets_condition_func); workflow.add_node(node_does_not_meet_condition, does_not_meet_condition_func); workflow.add_node(node_error, error_handler_func)

    # --- Define Connections (Optimized Flow) ---
    # (Paste the corrected connection block from response #41 here)
    workflow.set_entry_point(node_is_english)
    workflow.add_conditional_edges(node_is_english, route_binary_decision, {"yes": node_does_not_meet_condition, "no": node_word_in_para, "__error__": node_error})
    workflow.add_conditional_edges(node_word_in_para, route_binary_decision, {"yes": node_number_in_para, "no": node_number_in_para, "__error__": node_error})
    workflow.add_conditional_edges(node_number_in_para, route_binary_decision, {"yes": node_or_word_num, "no": node_or_word_num, "__error__": node_error})
    workflow.add_conditional_edges(node_or_word_num, route_binary_decision, {"yes": node_is_poem, "no": node_is_poem, "__error__": node_error})
    workflow.add_conditional_edges(node_is_poem, route_binary_decision, {"yes": node_and_or_poem, "no": node_and_or_poem, "__error__": node_error})
    workflow.add_conditional_edges(node_and_or_poem, route_binary_decision, {"yes": node_meets_condition, "no": node_does_not_meet_condition, "__error__": node_error})
    workflow.add_edge(node_meets_condition, END); workflow.add_edge(node_does_not_meet_condition, END); workflow.add_edge(node_error, END)


    # Compile
    app = workflow.compile()
    print(f"\nCompiled graph from example module: '{build_word_poetry_graph.__name__}'")
    return app


# End of module example_graphs.py