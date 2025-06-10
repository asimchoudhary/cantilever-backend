from dotenv import load_dotenv
import os
import uuid
load_dotenv()
from langgraph.graph import START, StateGraph
from langgraph.graph import StateGraph, MessagesState, START, END

from langgraph.graph import START, StateGraph
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage,ToolMessage
from langchain_together import ChatTogether
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import tools_condition, ToolNode

azure_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
azure_container_name = os.getenv("AZURE_CONTAINER_NAME")

from azure.storage.blob import BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)

def upload_content(content:str, content_type:str)-> str:
    
    content_id = str(uuid.uuid4())
    blob_name = f'{content_type}/{content_id}.md'


    blob_client = blob_service_client.get_blob_client(container=azure_container_name, blob=blob_name)
    blob_client.upload_blob(content, overwrite=True)
    return blob_client.url

from langchain_google_genai import ChatGoogleGenerativeAI

agent = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0)

from langgraph.graph import StateGraph, MessagesState
class State(MessagesState):
    pre_class_content_generated:bool = False
    in_class_lessons_generated:bool = False
    post_class_quiz_content_generated:bool = False
    pre_class_content_url:str = "None"
    in_class_lessons_url:str = "None"
    post_class_quiz_content_url:str = "None"
    error_occured: bool = False
    error_message: str = "None"

from langchain_core.messages import SystemMessage, HumanMessage,ToolMessage
def pre_class_generator(user_query:str)->dict:
    """Generate pre-class notes based on user query.
    Args:
        user_query (str): The user's query or topic for which pre-class notes are to be generated.
    Returns:
        dict: A dictionary containing the URL of the generated pre-class notes.
    """

    sys_prompt = '''
    You are an expert educator tasked with generating detailed pre-class notes for students based on the user's query.
    Your notes should include:
    - The generated content will be shared with the students, it should only contain the required content.
    - A clear and concise introduction to the topic.
    - Key concepts and definitions that students need to understand.
    - Relevant background information and context.
    - Important formulas, diagrams, or frameworks (if applicable).
    - Real-world examples or applications to illustrate the topic.
    - A summary of what students should focus on before class.
    - The content generated should be in markdown format for easy readability.
    Ensure the notes are well-structured, easy to follow, and suitable for students preparing for an in-class session.
    '''
    response = agent.invoke([SystemMessage(content=sys_prompt),HumanMessage(content=user_query)])
    content = response.content
    try:
        content_url = upload_content(content, "pre_class_notes")
    except Exception as e:
        return {"error": "Failed to upload content."}
    return {
        "pre_class_notes_url":content_url
    }

def in_class_generator(user_query:str)->dict:
    """Generate in-class lesson plan based on user query.
    Args:
        user_query (str): The user's query or topic for which an in-class lesson plan is to be generated.
    Returns:
        dict: A dictionary containing the URL of the generated in-class lesson plan.
    """

    sys_prompt = '''
    You are an expert educator tasked with creating a structured 1-hour lesson plan for a mentor (lecturer) who will be teaching a class of students, based on the user's query.
    Your lesson plan should include:
    - The generated content will be shared with the mentor, it should only contain the required content.
    - The content generated should be in correct markdown format for easy readability and should be rendered easily through any markdown renderer.
    - Learning objectives for the class.
    - A detailed outline of the class structure, including time allocations for each section.
    - Key concepts to be covered and how they will be taught.
    - Examples and case studies to illustrate the concepts.
    - Activities or discussions to engage students.
    - Assessment methods to evaluate student understanding.
    Ensure the plan is comprehensive, easy to follow, and suitable for a mentor (lecturer) guiding a class session for students.
    '''
    response = agent.invoke([SystemMessage(content=sys_prompt),HumanMessage(content=user_query)])
    content = response.content
    try:
        content_url = upload_content(content, "in_class_lessons")
        return {"in_class_lessons_url": content_url}
    except Exception as e:
        return {"error": "Failed to upload content."}
    
def post_class_generator(user_query:str)->dict:
    """Generate post-class quiz and summary based on user query.
    Args:
        user_query (str): The user's query or topic for which post-class quiz and summary are to be generated.
    Returns:
        dict: A dictionary containing the quiz and summary document tailored to the user's query.
    """
    sys_prompt = '''
    You are an expert educator tasked with generating a short quiz and summary document for students based on the user's query.
    Your output should include:
    - The generated content will be shared with the students, it should only contain the required content.
    - A brief summary of key takeaways from the class.
    - A set of 5-10 quiz questions that assess students' understanding of the topic.
    - Answers and explanations for each quiz question.
    - The content generated should be in markdown format for easy readability.
    Ensure the content is clear, concise, and suitable for reinforcing students' learning after class.
    '''
    response = agent.invoke([SystemMessage(content=sys_prompt),HumanMessage(content=user_query)])
    content = response.content
    try:
        content_url = upload_content(content, "post_class_quiz")
        return {"post_class_quiz_content_url": content_url}
    except Exception as e:
        return {"error": "Failed to upload content."}

tools = [pre_class_generator, in_class_generator, post_class_generator]
agentT = agent.bind_tools(tools)

def update_state(state:State)->State:
    """Update the state with output from the last tool executed.
    Args:
        state (MessagesState): The current state of the conversation.
    Returns:
        state: The updated state with the latest tool output.
    """
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            try:
                parsed_message = json.loads(message.content)
                if "pre_class_notes_url" in parsed_message:
                    state.pre_class_content_generated = True
                    state.pre_class_content_url = parsed_message["pre_class_notes_url"]
                elif "in_class_lessons_url" in parsed_message:
                    state.in_class_content_generated = True
                    state.in_class_content_url = parsed_message["in_class_lessons_url"]
                elif "post_class_quiz_content_url" in parsed_message:
                    state["post_class_quiz_content_generated"] = True
                    state["post_class_quiz_content_url"] = parsed_message["post_class_quiz_content_url"]
                elif "error" in parsed_message:
                    state.error_occured = True
                    state.error_message = parsed_message["error"]
                break
            except json.JSONDecodeError as e:
                try:
                    import ast
                    parsed_message = ast.literal_eval(message.content)
                    if "pre_class_notes_url" in parsed_message:
                        state.pre_class_content_generated = True
                        state.pre_class_content_url = parsed_message["pre_class_notes_url"]
                    elif "in_class_lessons_url" in parsed_message:
                        state.in_class_content_generated = True
                        state.in_class_content_url = parsed_message["in_class_lessons_url"]
                    elif "post_class_quiz_content_url" in parsed_message:
                        state["post_class_quiz_content_generated"] = True
                        state["post_class_quiz_content_url"] = parsed_message["post_class_quiz_content_url"]
                    elif "error" in parsed_message:
                        state.error_occured = True
                        state.error_message = parsed_message["error"]
                    break
                except Exception as e:
                    state.error_occured = True
                    state.error_message = f"Error parsing tool message content: {str(e)}"
                    break
    return state


def agent_node(state:State):
    '''Main orchestrator node of the graph'''

    if state.get("error_occured", False):
        return {
            "messages": [
                AIMessage(content=f"An error occurred: Please try again.")
            ]
        }

    state_summary = []
    if state.get("pre_class_content_generated", False):
        state_summary.append(f"PRE class notes generated successfully, URL is: {state.pre_class_content_url}")
    if state.get("in_class_content_generated", False):
        state_summary.append(f"IN class lesson plan generated successfully, URL is: {state.in_class_content_url}")
    if state.get("post_class_quiz_content_generated", False):
        state_summary.append(f"POST class quiz generated successfully")



    sys_msg = '''
    You are a specialized  agent within an educational content generation system. Your sole function is to analyze a user's request, determine the most appropriate tool for the task, and format the output for system execution. You do not generate content or answer questions yourself.

    OBJECTIVE: Parse the user's request, select one of the available tools, and extract the required parameters. If the request is invalid or cannot be handled by any tool, you must reject it.

    IMPORTANT: For every educational content request, you must determine the intended level of the content: "introductory/beginner", "intermediate", or "advanced". 
    - If the user query specifies the level, extract and use it.
    - If the user query does NOT specify the level, ask the user to clarify the desired level before proceeding.

    When invoking any tool, always pass both the topic and the level as parameters.

    AVAILABLE TOOLS:

    pre_class_generator

    Purpose: Creates a 1-2 page preparatory document for students.

    Triggers: User asks for pre-class notes, background material, an overview, key concepts to study beforehand, or a "cheat sheet".

    Required Parameters: 
      - topic (string)

    in_class_generator

    Purpose: Creates a structured 1-hour lesson plan for a mentor.

    Triggers: User asks for a lesson plan, a teaching script, lecture notes, class structure, examples to use in class, or a mentor's guide.

    Required Parameters: 
      - topic (string)

    post_class_generator

    Purpose: Creates a short quiz and summary document for students.

    Triggers: User asks for a quiz, assessment, practice questions, a summary of key takeaways, or reinforcement material.

    Required Parameters: 
      - topic (string)

    TOOL OUTPUTS:
    
    Each tool will return a URL of the uploaded content if everything works well. The agent must share each URL with the user, clearly indicating which URL corresponds to which content (pre-class notes, in-class lesson plan, or post-class quiz/summary).
    Share the URLs in a structured format.

    When user passes a relevant query, you need to generate all three content materials: pre-class notes, in-class notes, and post-class assessments using the relevant tools, and always specify the level.
    You are not allowed to respond to or entertain any queries  that are not related to educational content generation.

    
    '''
    if state_summary:
        state_context = SystemMessage(content=f"Current State : {' , '.join(state_summary)}")
        return {"messages": [agentT.invoke([SystemMessage(content=sys_msg),state_context] +[message for message in state["messages"] if not isinstance(message,SystemMessage)])]}
       

    return {
        "messages":[agentT.invoke([SystemMessage(content=sys_msg)]+ [message for message in state["messages"] if not isinstance(message,SystemMessage)])]
    }

workflow = StateGraph(State)
from langgraph.prebuilt import tools_condition, ToolNode


workflow.add_node("agent",agent_node)
workflow.add_node("tools",ToolNode(tools))
workflow.add_node("update_state", update_state)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "update_state")
workflow.add_edge("update_state", "agent")

graph = workflow.compile()