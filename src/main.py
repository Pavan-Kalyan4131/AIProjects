import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.tools import Tool
from pyvis.network import Network
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
import tempfile
from bs4 import BeautifulSoup
import configparser
import os
import rdflib
from graphviz import Digraph

# Load configuration
def load_config():
    config = configparser.ConfigParser()
    config.read('config/properties.cfg')
    return config

# Azure API Configuration (Set your Azure credentials)
config = load_config()
os.environ["OPENAI_API_VERSION"] = config['LLM']['version']
os.environ["AZURE_OPENAI_ENDPOINT"] = config['LLM']['endpoint']
os.environ["AZURE_OPENAI_API_KEY"] = config['LLM']['api_key']
os.environ["GRAPHVIZ_DOT"] = config['graphviz']['executable']
os.environ["PATH"] += os.pathsep + config['graphviz']['path']
llm_model = config['LLM']['model']
# Initialize the AzureChatOpenAI model from langchain
model = AzureChatOpenAI(deployment_name=llm_model, temperature=0)
checkpointer = InMemorySaver()
store = InMemoryStore()


def simplify_uri(uri):
    """
    Simplify URIs by removing the base part and keeping the local part for better readability.
    """
    # You can replace this with the base URI used in your RDF data.
    base_uri = "http://example.org/"
    schema_base_uri = "http://www.w3.org/2000/01/rdf-schema#"
    property_base_uri = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    
    # If the URI contains the base URI, replace it with a simpler version
    if uri.startswith(base_uri):
        return uri.replace(base_uri, "ex:")
    else:
        if uri.startswith(schema_base_uri):
            return uri.replace(schema_base_uri, "")
        if uri.startswith(property_base_uri):
            return uri.replace(property_base_uri, "")

def generate_graph_image(skipped_messages, output_format="png"):
    # Create the 'agenticflow' folder if it doesn't exist

    folder_name = config['workflow']['agenticflow_folder']
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Define the output path with the folder name
    image_path = os.path.join(folder_name, "agent_interaction_flow")

    # Create the graph
    dot = Digraph(comment="Agent Interaction Flow", format=output_format)
    # dot.attr(rankdir='TB', nodesep='1.0', ranksep='1.2')
    dot.attr(rankdir='LR', nodesep='1.0', ranksep='1.2')
    dot.attr('node', style='filled', fontname="Helvetica")

    dot.node("User", "🧍 User Input", shape="ellipse", fillcolor="#a3d3f5", color="#3399cc")
    dot.node("Supervisor", "🧑‍💼 Supervisor", shape="box", fillcolor="#e0e0e0", color="#888888")
    dot.edge("User", "Supervisor", label="Start")

    i = 0
    step = 1
    while i < len(skipped_messages):
        item = skipped_messages[i]
        if item['name'].startswith("transfer_to_"):
            expert = item['name'].replace("transfer_to_", "")
            expert_label = f"Step {step}: {expert.replace('_', ' ').title()}"
            expert_node_id = f"{expert}_{step}"

            dot.node(
                expert_node_id,
                f"🧠 {expert_label}",
                shape="box",
                fillcolor="#d3f9d8",
                color="#34a853",
                style="rounded"
            )

            dot.edge("Supervisor", expert_node_id, label=f"Step {step} →", color="#555555")
            dot.edge(expert_node_id, "Supervisor", label="↩ Return", style="dashed", color="#aaaaaa")

            i += 3
            step += 1
        else:
            i += 1

    dot.node(
        "Output",
        "🏁 Final Output",
        shape="ellipse",
        fillcolor="#ffd699",
        color="#ff9900",
        style="filled"
    )
    dot.edge("Supervisor", "Output", label="Finish", color="#ff9900", fontcolor="#ff9900")

    # Render to the specified folder and format
    image_path = dot.render(filename=image_path, cleanup=True)

    return {
        "image_path": image_path
    }



def extract_skipped_messages(messages, skip_phrases=None):
    if skip_phrases is None:
        skip_phrases = [
            "successfully transferred",
            "transferring back",
            "task is done",
            "Transferring back",
            "Task is done"
        ]

    skipped_messages_log = []

    for msg in messages:
        assistant_response = msg.content.strip()

        if not assistant_response:
            continue

        lower_content = assistant_response.lower()

        if any(phrase in lower_content for phrase in skip_phrases):
            skipped_entry = {
                "name": getattr(msg, 'name', 'N/A'),
                "type": type(msg).__name__,
                "content": assistant_response
            }
            skipped_messages_log.append(skipped_entry)

    return skipped_messages_log


def turtle_to_graph_tool(turtle_text:str):
    """
    Generate a knowledge graph visualization from Turtle ontology.
    Args:
        turtle_text (str): The Turtle ontology text.
    Returns:
        str: Path to the HTML file containing the knowledge graph visualization.
    """
    # Initialize the pyvis network for visualizing the RDF graph
    print("Generating graph")
    #print("turtle text----",turtle_text)
    net = Network(directed=True, height='750px', width='100%', notebook=False, font_color='white')
    
    # Parse the Turtle file to extract RDF triples
    g = rdflib.Graph()
    try:
        g.parse(data=turtle_text, format="turtle")
    except Exception as e:
        print(f"Error parsing Turtle file: {e}")
        return

    # Add nodes and edges to the graph
    for subj, pred, obj in g:
        subj_label = simplify_uri(str(subj))
        obj_label = simplify_uri(str(obj))
        pred_label = simplify_uri(str(pred))

        # Add nodes for the subject and object (if they are not already added)
        if subj_label not in net.nodes:
            if "Class" in str(g.value(subj, rdflib.RDF.type)):  # Check if it's a class
                net.add_node(subj_label, label=subj_label, title=f"Class: {subj_label}", color='green', size=10)
            else:  # Otherwise, it's an instance (object)
                net.add_node(subj_label, label=subj_label, title=f"Object: {subj_label}", color='orange', size=8)
        
        if obj_label not in net.nodes:
            if "Class" in str(g.value(obj, rdflib.RDF.type)):  # Check if it's a class
                net.add_node(obj_label, label=obj_label, title=f"Class: {obj_label}", color='green', size=10)
            else:  # Otherwise, it's an instance (object)
                net.add_node(obj_label, label=obj_label, title=f"Object: {obj_label}", color='orange', size=8)

        # Add edges for the properties (predicates)
        net.add_edge(subj_label, obj_label, title=pred_label, color="white")

    # Add legend HTML (this part remains similar to your lineage example)
    legend_html = """
        <div style="position: absolute; top: 10px; right: 10px; width: 120px; background-color: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 5px;">
            <ul style="list-style: none; padding: 0; font-size: 12px;">
                <li><span style="height: 10px; width: 10px; background-color: green; display: inline-block;"></span> Subject</li>
                <li><span style="height: 10px; width: 10px; background-color: #ff9933; display: inline-block;"></span> Object</li>
            </ul>
        </div>
    """
    
    # Save the pyvis graph to a temporary HTML file
    #with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
    net.save_graph("graph.html")

    # Read the saved HTML content
    with open("graph.html", 'r') as file:
        html_content = file.read()

    # Modify the HTML content to add the legend and styles
    soup = BeautifulSoup(html_content, 'html.parser')
    for style_tag in soup.find_all("style"):
        style_tag.decompose()

    # Add custom styles for the network and the body
    new_style_tag = soup.new_tag("style")
    new_style_tag.string = """
    .vis-tooltip {
      font-size: 12px !important;
    }
    body, html {
        height: 100%;
        width: 100%;
        margin: 0;
        padding: 0;
        background-color: #144352 !important; /* Ensure background color is applied to the whole document */
    }
    #mynetwork {
        height: 100%;
        width: 100%;
        border: none;
    }
    .card {
        height: 100%;
        background-color: #144352;
        border: none;
    }
    """
    soup.body.append(BeautifulSoup(legend_html, 'html.parser'))
    soup.head.append(new_style_tag)

    # Write the modified content back to the file
    image_path = "graph.html"
    with open(image_path, "w") as file:
        file.write(str(soup))

    return image_path
def rdf_to_turtle_tool(rdf_text:str):
    """
    Convert RDF text into Turtle format.
    Args:
        rdf_text (str): The input rdf text to be converted.

    Returns:
        str: The converted Turtle format.
    """
    print("Entered turtle tool")
    g = rdflib.Graph()

    # Clean the input text - remove code block markers if present
    cleaned_rdf_text = rdf_text.strip()
    if cleaned_rdf_text.startswith('```'):
        lines = cleaned_rdf_text.split('\n')
        # Remove first line (```turtle or similar) and last line (```)
        cleaned_rdf_text = '\n'.join(lines[1:-1])
    
    #Parse the RDF triples into the graph - try multiple formats
    try:
        g.parse(data=cleaned_rdf_text, format="turtle")
        print(f"Successfully parsed as turtle format")
        ontology_turtle = g.serialize(format="turtle")
        
        # Save to file for potential download later
        with open("generated_ontology.ttl", "w") as f:
            f.write(ontology_turtle)
        
        return ontology_turtle
    except Exception as e:
        print(f"Failed to parse as turtle: {e}")
        error_msg = f"Failed to parse RDF text in any known format. Input text: {cleaned_rdf_text[:200]}..."
        print(error_msg)
        return f"Error: {error_msg}"


def txt_to_rdf_tool(text:str):
    """
    Convert natural language text into RDF triples.
    Args:
        text (str): The input text to be converted.

    Returns:
        str: The RDF triples format.
    """
    print("Entered the tool")
    rdf_prompt = PromptTemplate.from_template(
        """
       You are an ontology builder.  
        Given the following text, identify the key entities (instances), classes (types), and relationships (predicates).  
        Represent them as RDF triples in valid Turtle syntax.  

        Requirements:
        - Use clear URIs or prefixes (e.g., ex:Patient, ex:Doctor).  
        - Define classes with `rdf:type` and `rdfs:Class`.  
        - Define relationships with meaningful predicates (e.g., ex:diagnoses, ex:treats).  
        - Ensure the output is syntactically valid Turtle.  
        - Do not include explanations, commentary, or natural language text.  
        - Output only the RDF ontology.  

        Text: {text}  

        RDF Format:
        """
    )
    rdf_chain = rdf_prompt | model | StrOutputParser()
    result = rdf_chain.invoke({"text": text})
    return result
# Define the tools
rdf_conversion_tool = Tool(
    name="rdf_conversion_agent",
    func=txt_to_rdf_tool,
    description="Convert the text to rdf triple format."
)

turtle_conversion_tool = Tool(
    name="rdf_to_turtle_agent",
    func=rdf_to_turtle_tool,
    description="Convert RDF triples to Turtle syntax."
)

graph_conversion_tool = Tool(
    name="graph_generation_agent",
    func=turtle_to_graph_tool,
    description="Generate a knowledge graph visualization from Turtle ontology and return the HTML content."
)

rdf_conversion_agent = create_react_agent(
    model=model,
    tools=[txt_to_rdf_tool],
    name="rdf_conversion_agent",
    store=store
)

turtle_conversion_agent = create_react_agent(
    model=model,
    tools=[rdf_to_turtle_tool],
    name="rdf_to_turtle_agent",
    store=store
)

graph_generation_agent = create_react_agent(
    model=model,
    tools=[turtle_to_graph_tool],
    name="graph_generation_agent",
    store=store
)

def supervisor(user_input: str):
    workflow = create_supervisor(
        [rdf_conversion_agent, turtle_conversion_agent, graph_generation_agent],
        model=model,
        prompt="""
        You are a supervisor agent. Your ONLY job is to coordinate the three agents and return like turtle path or final graph HTML path based on user input.
        
        Process:
        1. Call rdf_conversion_agent to convert text to RDF triples
        2. Call rdf_to_turtle_agent with the RDF output to get Turtle format
        3. Call graph_generation_agent with the Turtle content to get HTML visualization path

        IMPORTANT: If user query requests only ontology generation, skip step 3 and return ONLY the Turtle content from the turtle_conversion_agent. 
        If user query requests knowledge graph generation, follow all three steps and return ONLY the HTML graph path from the graph_generation_agent.
        If user query doesn't specify any specific request, then also follow all three steps and return ONLY the HTML graph path from the graph_generation_agent.
        
        CRITICAL: Return ONLY the output from the agent. 
        Do NOT add any explanatory text, comments, or status messages.

        Return False if any step fails.
        """
    )
    result = workflow.compile(store=store).invoke({
        "messages": [{"role": "user", "content": user_input}]
    })
    skipped_messages = extract_skipped_messages(result["messages"])
    graph_output = generate_graph_image(skipped_messages)
    
    # Store workflow image path in session state
    if 'image_path' in graph_output:
        print("Entered image path storage")
        st.session_state.workflow_image_path = graph_output['image_path']
    
    return result['messages'][-1].content

# Initialize session state
if 'graph_generated' not in st.session_state:
    st.session_state.graph_generated = False
if 'ontology_content' not in st.session_state:
    st.session_state.ontology_content = None
if 'workflow_image_path' not in st.session_state:
    st.session_state.workflow_image_path = None

def reset_session_state():
    st.session_state.graph_generated = False
    st.session_state.ontology_content = None
    st.session_state.workflow_image_path = None

# Streamlit app structure
st.title('Ontology-Driven Knowledge Representation Using Agentic AI')

# Sidebar for Agentic Workflow
st.sidebar.header("Agentic Interaction Workflow")

# Input from the user
user_input = st.text_area("Enter the text:", height=100)

if st.button('Submit'):
    reset_session_state()
    if user_input.strip() != "":
        with st.spinner("Processing the input..."):
            # Get the output from the supervisor agent
            output = supervisor(user_input)
            print("supervisor output----",output)
            if "False" in output:
                st.error("An error occurred during processing.")
            elif "html" in output.lower():
                st.session_state.graph_generated = True
                # Store ontology content in session state
                if os.path.exists("generated_ontology.ttl"):
                    with open("generated_ontology.ttl", "r") as f:
                        st.session_state.ontology_content = f.read()
            else:
                # Store ontology content in session state
                if os.path.exists("generated_ontology.ttl"):
                    with open("generated_ontology.ttl", "r") as f:
                        st.session_state.ontology_content = f.read()
    else:
        st.warning("Please enter some text to convert to RDF!")

# Display results if they exist in session state
if st.session_state.ontology_content:
    st.subheader("Generated Ontology")
    st.text_area("Ontology in Turtle format", st.session_state.ontology_content, height=300)
    st.download_button(
        label="Download Ontology (Turtle Format)",
        data=st.session_state.ontology_content,
        file_name="generated_ontology.ttl",
        mime="text/plain"
    )

if st.session_state.graph_generated and os.path.exists("graph.html"):
    st.header("Knowledge Graph Visualization")
    # Read HTML content from the existing graph.html file
    with open("graph.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Display the HTML content
    st.components.v1.html(html_content, height=600, width=800)
    
    # Add download button for the graph
    st.download_button(
        label="Download Knowledge Graph (HTML)",
        data=html_content,
        file_name="knowledge_graph.html",
        mime="text/html"
    )
if st.session_state.workflow_image_path and os.path.exists(st.session_state.workflow_image_path):
    st.sidebar.image(st.session_state.workflow_image_path, use_container_width=True)
    st.sidebar.download_button(
        label="Download Workflow Graph",
        data=open(st.session_state.workflow_image_path, 'rb').read(),
        file_name="agent_workflow.png",
        mime="image/png"
    )
else:
    st.sidebar.info("Workflow graph will appear here after processing.")