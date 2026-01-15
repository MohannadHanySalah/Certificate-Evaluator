import os
import sys
import operator
import json
import base64
import mimetypes
import threading
import time
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

try:
    import msvcrt
    WINDOWS = True
except ImportError:
    WINDOWS = False
    import select


# Load environment variables

if "GOOGLE_API_KEY" not in os.environ:
    # load API KEY from .env file
    from dotenv import load_dotenv
    load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    print("CRITICAL ERROR: GOOGLE_API_KEY environment variable not set.")
    print("Please set it via 'set GOOGLE_API_KEY=...' or in a .env file.")
    sys.exit(1)
 

# --- State Definition ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    extracted_data: Dict[str, Any]  # Includes 'confidence' per field if provided
    evaluation_criteria: Dict[str, Any]  # Includes 'weights' per criterion if provided
    evaluation_result: Dict[str, Any]
    conversation_context: Dict[str, Any]  # Tracks user intent, corrections, scope changes

# --- LLM Tools ---

@tool
def record_extracted_data(data: Dict[str, Any], source_file: str = "unknown") -> str:
    """
    Records the extracted certificate data into the agent's state.
    YOU (the LLM) must extract data visually from the certificate image/PDF and call this to persist it.
    
    Parameters:
    - data: Dictionary containing extracted fields. Recommended fields:
        - student_name: Full name of the certificate holder
        - gpa: Grade Point Average (numeric string like "3.490")
        - degree: Degree title (e.g., "Bachelor of Petroleum Engineering")
        - university: University name
        - graduation_date: When they graduated
        - honors/appreciation: Any honors or grade classification
        - confidence: Optional dict with confidence levels per field (e.g., {"gpa": "high", "degree": "medium"})
    - source_file: The filename this data was extracted from
    """
    data["source_file"] = source_file
    return f"Data recorded successfully: {json.dumps(data)}"

@tool
def update_evaluation_criteria(criteria_updates: Dict[str, Any]) -> str:
    """
    Updates or sets the evaluation criteria. 
    The criteria should be a dictionary of rules, e.g., {"min_gpa": "3.5", "required_degree": "Bachelor"}.
    """
    return f"Criteria updated with: {json.dumps(criteria_updates)}"

@tool
def calculate_score(gpa: float, min_gpa: float, degree_matches: bool, degree_reason: str) -> Dict[str, Any]:
    """
    Records the evaluation result. YOU (the LLM) must determine if criteria are met.
    
    Parameters:
    - gpa: The student's GPA (numeric)
    - min_gpa: The minimum required GPA (numeric)
    - degree_matches: True if YOU determine the degree meets the requirement (use your reasoning for abbreviations, synonyms, etc.)
    - degree_reason: Your explanation of why the degree matches or doesn't match
    
    Returns the final PASS/FAIL result.
    """
    reasons = []
    passed = True
    
    # GPA check
    if gpa >= min_gpa:
        reasons.append(f"GPA {gpa} meets minimum {min_gpa}")
    else:
        passed = False
        reasons.append(f"GPA {gpa} is below minimum {min_gpa}")
    
    # Degree check (LLM already decided)
    if degree_matches:
        reasons.append(f"Degree requirement met: {degree_reason}")
    else:
        passed = False
        reasons.append(f"Degree requirement not met: {degree_reason}")

    return {
        "pass_fail": "PASS" if passed else "FAIL",
        "reasons": reasons
    }

# --- User Intervention Tools ---


@tool
def correct_extracted_data(field_name: str, old_value: str, new_value: str, reason: str) -> Dict[str, Any]:
    """
    Corrects a specific field in the extracted data based on user intervention.
    Use this when:
    - The user identifies an error in extracted data
    - The user provides a correction or override
    - You re-read the document and find a discrepancy
    
    Parameters:
    - field_name: The name of the field to correct (e.g., 'gpa', 'degree', 'student_id')
    - old_value: The previous (incorrect) value
    - new_value: The corrected value provided by user or re-extraction
    - reason: Why this correction is being made
    """
    return {
        "corrected_field": field_name,
        "old_value": old_value,
        "new_value": new_value,
        "correction_reason": reason,
        field_name: new_value  # Include the corrected field for easy state update
    }

@tool
def override_evaluation_result(new_result: str, justification: str) -> Dict[str, Any]:
    """
    Allows user to override the final evaluation result (PASS/FAIL).
    Use this when the user explicitly requests to override the system's decision.
    
    Parameters:
    - new_result: Must be either 'PASS' or 'FAIL'
    - justification: The user's reason for the override
    """
    if new_result.upper() not in ['PASS', 'FAIL']:
        return {"error": "Invalid result. Must be 'PASS' or 'FAIL'."}
    
    return {
        "pass_fail": new_result.upper(),
        "reasons": [f"USER OVERRIDE: {justification}"],
        "was_overridden": True
    }

# --- Nodes ---

def agent_node(state: AgentState):
    """
    The main reasoning node. It decides what to do next based on the state.
    """
    system_prompt = """You are the Certificate Evaluation Agent. 
Do NOT mention "Gemini" or "Google" in any of your responses. 
Your goal is to evaluate university certificates based on user-defined criteria.
    
Current Context:
- Extracted Data: {extracted}
- Evaluation Criteria: {criteria}
- Evaluation Result: {result}

You have access to the certificate images or text provided. Each is labeled with its filename.
    
## Core Decision Logic
You must decide the next step dynamically based on context:
- If multiple certificates are provided, specify which one you are evaluating in your response.
- If you have the certificate but no extracted data:
  1. READ the certificate image/PDF visually (you have multimodal vision)
  2. EXTRACT all relevant fields yourself (name, GPA, degree, university, graduation date, etc.)
  3. Call 'record_extracted_data' to PERSIST the data you extracted into state
  4. Include confidence levels for uncertain fields (e.g., {{"confidence": {{"gpa": "high", "name": "medium"}}}})
- If you have data but no criteria, ask the user for criteria.
- If you have data and criteria but no result, proceed directly with 'calculate_score'.
- If the user changes criteria, you must re-evaluate.

## User Intervention Support
The user can interrupt you at any time by typing. When you see a message like "[USER INTERRUPTED with: ...]":
- STOP your current action immediately
- Read the user's new instruction carefully
- Respond to their new request instead of continuing your previous task

Supported interventions:
1. **Data Corrections**: If the user says "that's wrong", "the GPA is actually X", etc.:
   - Use 'correct_extracted_data' to record the correction
   - Acknowledge the change

2. **Result Overrides**: If the user says "override to PASS" or "I want this to FAIL":
   - Use 'override_evaluation_result' to apply the user's decision

3. **Change of Direction**: If the user gives new instructions mid-process:
   - Follow their new instructions
   - You can skip, reorder, or restart any steps as needed

## Response Guidelines
- State CLEARLY which certificate (filename) you are currently processing
- Explain your reasoning for each step you take
- Accept user corrections gracefully and update accordingly
- Do NOT ask for confirmation - proceed directly unless interrupted
- For simple greetings (hi, hello, hey), respond briefly and naturally - don't dump information
- Match the user's tone and intent - if they want a quick answer, give a quick answer
"""
    extracted = json.dumps(state.get("extracted_data", {}))
    criteria = json.dumps(state.get("evaluation_criteria", {}))
    result = json.dumps(state.get("evaluation_result", {}))
    
    prompt = system_prompt.format(
        extracted=extracted,
        criteria=criteria,
        result=result
    )
    
    messages = [SystemMessage(content=prompt)] + state['messages']
    
    # Bind LLM tools to the LLM (no external tools)
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    
    #core tools + user intervention tools (using LLM) (no external tools)
    all_tools = [
        record_extracted_data, 
        update_evaluation_criteria, 
        calculate_score,
        correct_extracted_data,
        override_evaluation_result
    ]
    llm_with_tools = llm.bind_tools(all_tools)
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node_func(state: AgentState):
    """
    Executes the tools requested by the agent and updates the state.
    """
    last_message = state['messages'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
    
    # all tools: core tools + user intervention tools (using the LLM) (no external tools)
    tool_map = {
        "record_extracted_data": record_extracted_data,
        "update_evaluation_criteria": update_evaluation_criteria,
        "calculate_score": calculate_score,
        "correct_extracted_data": correct_extracted_data,
        "override_evaluation_result": override_evaluation_result
    }
    
    outputs = []
    extracted_data_update = {}
    criteria_update = {}
    result_update = {}

    for tool_call in last_message.tool_calls:
        t_name = tool_call['name']
        t_args = tool_call['args']
        tool_func = tool_map.get(t_name)
        
        if tool_func:
            # Execute tool
            try:
                res = tool_func.invoke(t_args) 
                
                # Update specific state keys based on tool purpose
                if t_name == "record_extracted_data":
                    extracted_data_update = t_args.get('data', {})
                    extracted_data_update['source_file'] = t_args.get('source_file', 'unknown')
                    outputs.append(ToolMessage(tool_call_id=tool_call['id'], content=str(res)))
                elif t_name == "update_evaluation_criteria":
                    criteria_update = t_args.get('criteria_updates', {})
                    outputs.append(ToolMessage(tool_call_id=tool_call['id'], content=str(res)))
                elif t_name == "calculate_score":
                    result_update = res
                    outputs.append(ToolMessage(tool_call_id=tool_call['id'], content=f"Score: {json.dumps(res)}"))
                
                # --- User Intervention Tool Handlers ---
                elif t_name == "correct_extracted_data":
                    # Update the extracted data with the correction
                    if isinstance(res, dict) and 'corrected_field' in res:
                        field_name = res['corrected_field']
                        new_value = res['new_value']
                        # Get current extracted data and update the field
                        current_extracted = state.get("extracted_data", {}).copy()
                        current_extracted[field_name] = new_value
                        # Add correction metadata
                        if 'corrections_history' not in current_extracted:
                            current_extracted['corrections_history'] = []
                        current_extracted['corrections_history'].append({
                            "field": field_name,
                            "old_value": res.get('old_value'),
                            "new_value": new_value,
                            "reason": res.get('correction_reason')
                        })
                        extracted_data_update = current_extracted
                    outputs.append(ToolMessage(tool_call_id=tool_call['id'], content=f"Correction applied: {json.dumps(res)}"))
                
                elif t_name == "override_evaluation_result":
                    # Override the evaluation result
                    if isinstance(res, dict) and 'error' not in res:
                        result_update = res
                        outputs.append(ToolMessage(tool_call_id=tool_call['id'], content=f"Override applied: {json.dumps(res)}"))
                    else:
                        outputs.append(ToolMessage(tool_call_id=tool_call['id'], content=f"Override failed: {json.dumps(res)}"))
                    
            except Exception as e:
                outputs.append(ToolMessage(tool_call_id=tool_call['id'], content=f"Error: {str(e)}"))
        else:
             outputs.append(ToolMessage(tool_call_id=tool_call['id'], content="Tool not found"))

    # Return the updates
    update = {"messages": outputs}
    if extracted_data_update:
        update["extracted_data"] = extracted_data_update
    if criteria_update:
        current = state.get("evaluation_criteria") or {}
        current.update(criteria_update)
        update["evaluation_criteria"] = current
    if result_update:
        update["evaluation_result"] = result_update
        
    return update

# --- Graph ---

def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node_func)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# --- Main Execution ---

if __name__ == "__main__":
    # --- Interrupt Handling ---
    class InterruptibleAgent:
        """Wrapper to make agent execution interruptible by user input."""
        
        def __init__(self, app):
            self.app = app
            self.interrupted = False
            self.interrupt_message = None
            self.result = None
            self.error = None
            self.is_running = False
            self._lock = threading.Lock()
        
        def _run_agent(self, state):
            """Run the agent in a thread."""
            try:
                self.result = self.app.invoke(state)
            except Exception as e:
                self.error = e
            finally:
                with self._lock:
                    self.is_running = False
        
        def invoke_with_interrupt(self, state, check_interval=0.1):
            """
            Run the agent while checking for user keyboard input.
            If user starts typing, interrupt and return the partial result.
            
            Returns: (result, was_interrupted, interrupt_input)
            """
            self.interrupted = False
            self.interrupt_message = None
            self.result = None
            self.error = None
            self.is_running = True
            
            # Start agent in background thread
            agent_thread = threading.Thread(target=self._run_agent, args=(state,))
            agent_thread.daemon = True
            agent_thread.start()
            
            print("Agent: [Thinking...] (Type to interrupt)", flush=True)
            
            interrupt_buffer = []
            
            # Monitor for keyboard input while agent is running
            while True:
                with self._lock:
                    if not self.is_running:
                        break
                
                # Check for keyboard input
                if WINDOWS:
                    if msvcrt.kbhit():
                        char = msvcrt.getwch()
                        if char == '\r':  
                            if interrupt_buffer:
                                self.interrupted = True
                                self.interrupt_message = ''.join(interrupt_buffer)
                                print()  
                                break
                        elif char == '\x03':  
                            self.interrupted = True
                            self.interrupt_message = None
                            break
                        elif char == '\x08':  
                            if interrupt_buffer:
                                interrupt_buffer.pop()
                                # Visual feedback
                                sys.stdout.write('\r' + ' ' * (len(interrupt_buffer) + 20) + '\r')
                                if interrupt_buffer:
                                    sys.stdout.write('Interrupt: ' + ''.join(interrupt_buffer))
                                sys.stdout.flush()
                        else:
                            interrupt_buffer.append(char)
                            # Show typing status
                            if len(interrupt_buffer) == 1:
                                sys.stdout.write('\r' + ' ' * 50 + '\r')  
                            sys.stdout.write('\rInterrupt: ' + ''.join(interrupt_buffer))
                            sys.stdout.flush()
                else:
                    # Use select
                    if select.select([sys.stdin], [], [], 0)[0]:
                        line = sys.stdin.readline().strip()
                        if line:
                            self.interrupted = True
                            self.interrupt_message = line
                            break
                
                time.sleep(check_interval)
            
            # Wait for thread to complete if not interrupted
            agent_thread.join(timeout=0.5 if self.interrupted else None)
            
            if self.error:
                raise self.error
            
            return (self.result, self.interrupted, self.interrupt_message)
    
    # helper for loading data
    def load_document_message(path):
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            if path.lower().endswith('.pdf'):
                mime_type = "application/pdf"
            else:
                mime_type = "image/jpeg" # default
            
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
        return HumanMessage(
            content=[
                {"type": "text", "text": f"Here is the certificate file ({os.path.basename(path)}):"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{data}"}
                }
            ]
        )

    # Load certificate
    available_files = []
    for filename in os.listdir("."):
        low_file = filename.lower()
        if low_file.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.txt')) and \
           filename not in ['requirements.txt', 'transcript.txt', 'test_output.txt', 'README.md']:
            available_files.append(filename)

    print("=" * 60)
    print("   AGENTIC CERTIFICATE EVALUATOR (with Real-time Interrupt)")
    print("=" * 60)
    print("\n[!] You can interrupt the agent at any time by typing!")
    print("[!] Just start typing while the agent is thinking.\n")
    
    if not available_files:
        print("No certificate files found in the current directory.")
        sys.exit(0)

    print("Available files:")
    for i, f in enumerate(available_files):
        print(f"  [{i}] {f}")
    
    choice = input("\nEnter the index of the certificate to evaluate (or 'all' for all): ").strip().lower()
    
    selected_files = []
    if choice == 'all':
        selected_files = available_files
    else:
        try:
            idx = int(choice)
            selected_files = [available_files[idx]]
        except (ValueError, IndexError):
            print("Invalid selection. Loading all files by default.")
            selected_files = available_files

    initial_messages = []
    for filename in selected_files:
        low_file = filename.lower()
        if low_file.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            print(f"Loading document: {filename}...")
            initial_messages.append(load_document_message(filename))
        elif low_file.endswith('.txt'):
            print(f"Loading text file: {filename}...")
            with open(filename, "r") as f:
                content = f.read()
                initial_messages.append(HumanMessage(content=f"Certificate Text from {filename}:\n{content}"))

    # Initialize state
    current_state = {
        "messages": initial_messages,
        "extracted_data": {},
        "evaluation_criteria": {},
        "evaluation_result": {},
        "conversation_context": {}
    }
    
    print("\n" + "-" * 60)
    print("Certificate Loaded. Type your message (or 'q' to quit).")
    print("You can interrupt the agent anytime by typing while it thinks!")
    print("-" * 60 + "\n")
    
    app = build_graph()
    interruptible_agent = InterruptibleAgent(app)
    
    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            current_state["messages"].append(HumanMessage(content=user_input))
            
            # Run agent with interrupt
            result, was_interrupted, interrupt_msg = interruptible_agent.invoke_with_interrupt(current_state)
            
            if was_interrupted:
                # Agent was stopped by user typing
                print(f"\n{'='*50}")
                
                if interrupt_msg:
                    # Use LLM to classify if this is a correction or new instruction
                    print("[CLASSIFYING] - Detecting intent...")
                    
                    # Get the last user message before the interruption
                    last_user_task = None
                    for msg in reversed(current_state["messages"]):
                        if isinstance(msg, HumanMessage) and not msg.content.startswith("[USER"):
                            last_user_task = msg.content
                            break
                    
                    # Quick LLM classification
                    classifier_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
                    classification_prompt = f"""Classify this user interrupt. Reply with ONLY "CORRECTION" or "NEW_INSTRUCTION".

Previous task: "{last_user_task or 'None'}"
User interrupt: "{interrupt_msg}"

- CORRECTION: User is modifying/amending the previous task (e.g., changing a value, adjusting criteria)
- NEW_INSTRUCTION: User wants something completely different (e.g., greeting, new question, different topic)

Reply with one word only: CORRECTION or NEW_INSTRUCTION"""
                    
                    try:
                        classification = classifier_llm.invoke(classification_prompt).content.strip().upper()
                        is_correction = "CORRECTION" in classification
                    except Exception:
                        # Treat as new instruction if classification fails
                        is_correction = False
                    
                    if is_correction:
                        print("[AGENT PAUSED] - Applying your correction...")
                        print(f"{'='*50}")
                        # Continue previous task with applying the correction
                        current_state["messages"].append(
                            HumanMessage(content=f"[USER CORRECTION: Apply this change and continue with the previous task: {interrupt_msg}]")
                        )
                        print(f"\nCorrection applied: '{interrupt_msg}'")
                        print("Continuing previous task with correction...\n")
                    else:
                        print("[AGENT STOPPED] - Processing your new input...")
                        print(f"{'='*50}")
                        # If the user wants to do something completely different
                        current_state["messages"].append(
                            HumanMessage(content=f"[USER INTERRUPTED: Stop the previous task and respond to this instead: {interrupt_msg}]")
                        )
                        print(f"\nNew instruction: '{interrupt_msg}'")
                        print("Switching to your new request...\n")
                    
                    # Run agent with the new input
                    result, was_interrupted_again, new_interrupt = interruptible_agent.invoke_with_interrupt(current_state)
                    
                    # Nested Interrupt
                    while was_interrupted_again and new_interrupt:
                        print(f"\n{'='*50}")
                        print("[CLASSIFYING] - Detecting intent...")
                        
                        # Classify nested interrupt with LLM
                        nested_prompt = f"""Classify this user interrupt. Reply with ONLY "CORRECTION" or "NEW_INSTRUCTION".

Previous task: "{last_user_task or 'None'}"
User interrupt: "{new_interrupt}"

- CORRECTION: User is modifying/amending the previous task
- NEW_INSTRUCTION: User wants something completely different

Reply with one word only: CORRECTION or NEW_INSTRUCTION"""
                        
                        try:
                            nested_class = classifier_llm.invoke(nested_prompt).content.strip().upper()
                            nested_is_correction = "CORRECTION" in nested_class
                        except Exception:
                            nested_is_correction = False
                        
                        if nested_is_correction:
                            print(f"[ANOTHER CORRECTION] - Applying: '{new_interrupt}'")
                            current_state["messages"].append(
                                HumanMessage(content=f"[USER CORRECTION: Apply this change and continue: {new_interrupt}]")
                            )
                        else:
                            print(f"[NEW INSTRUCTION] - Switching to: '{new_interrupt}'")
                            current_state["messages"].append(
                                HumanMessage(content=f"[USER INTERRUPTED: Stop and respond to this instead: {new_interrupt}]")
                            )
                        print(f"{'='*50}\n")
                        result, was_interrupted_again, new_interrupt = interruptible_agent.invoke_with_interrupt(current_state)
                    
                    if not result and was_interrupted_again:
                        print("Operation cancelled.")
                        continue
                else:
                    # If the user pressed Ctrl+C or started typing without finish
                    print("\nOperation stopped. Waiting for your input...")
                    new_input = input("User: ").strip()
                    
                    if new_input.lower() in ['q', 'quit', 'exit']:
                        print("\nGoodbye!")
                        break
                    
                    if new_input:
                        current_state["messages"].append(
                            HumanMessage(content=f"[USER INTERRUPTED: User stopped the previous operation]")
                        )
                        current_state["messages"].append(HumanMessage(content=new_input))
                        
                        # To run the agent with the new input in considration
                        result, _, _ = interruptible_agent.invoke_with_interrupt(current_state)
                    else:
                        print("No input provided. Ready for next command.")
                        continue
            
            if result:
                # Extract the last message info
                last_msg = result['messages'][-1]
                content = last_msg.content
                if isinstance(content, list):
                    text_parts = [c.get('text', '') for c in content if c.get('type') == 'text']
                    print(f"Agent: {' '.join(text_parts)}")
                else:
                    print(f"Agent: {content}")
                
                current_state = result
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by Ctrl+C. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue