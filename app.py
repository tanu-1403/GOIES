import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import requests
import json

# Import the local extraction engine
try:
    from extractor import extract_intelligence
except ImportError:
    st.error("Critical Error: extractor.py not found in the same directory.")

# --- 1. SETUP & STATE MANAGEMENT ---
st.set_page_config(layout="wide", page_title="Global Ontology Engine")

# Initialize the graph in session state
if "kg" not in st.session_state:
    st.session_state.kg = nx.DiGraph()


# --- 2. HELPER FUNCTIONS ---
def update_graph(extractions):
    """Parses extractions and updates the NetworkX graph."""
    for ext in extractions:
        node_class = ext.extraction_class.lower()

        # Add Nodes
        if node_class in ["country", "technology", "event"]:
            st.session_state.kg.add_node(
                ext.extraction_text, title=str(ext.attributes), group=node_class
            )

        # Add Edges
        elif node_class == "relationship":
            source = ext.attributes.get("source")
            target = ext.attributes.get("target")
            if source and target:
                if not st.session_state.kg.has_node(source):
                    st.session_state.kg.add_node(source, group="unknown")
                if not st.session_state.kg.has_node(target):
                    st.session_state.kg.add_node(target, group="unknown")

                st.session_state.kg.add_edge(source, target, label=ext.extraction_text)


def generate_graph_html():
    """Converts the NetworkX graph to an Obsidian-style PyVis HTML file."""
    net = Network(
        height="650px",
        width="100%",
        bgcolor="#0d1117",
        font_color="#c9d1d9",
        directed=True,
    )
    net.from_nx(st.session_state.kg)

    # Apply styling
    for node in net.nodes:
        node["size"] = 12
        node["borderWidth"] = 0
        node["font"] = {"size": 14, "color": "#ffffff", "strokeWidth": 0}

        # Group Colors
        group = node.get("group", "unknown")
        if group == "country":
            node["color"] = "#ff7b72"  # Soft Red
        elif group == "technology":
            node["color"] = "#79c0ff"  # Soft Blue
        elif group == "event":
            node["color"] = "#7ee787"  # Soft Green
        else:
            node["color"] = "#8b949e"  # Gray default

    # Advanced Physics and Styling Options
    options = """
    var options = {
      "edges": {
        "color": {"color": "#30363d", "highlight": "#58a6ff", "hover": "#58a6ff", "inherit": false},
        "smooth": {"type": "continuous", "forceDirection": "none"},
        "width": 1.5
      },
      "interaction": {"hover": true, "navigationButtons": true, "tooltipDelay": 150},
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -4000,
          "centralGravity": 0.1,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.1,
          "avoidOverlap": 0.1
        },
        "minVelocity": 0.75
      }
    }
    """
    net.set_options(options)

    html_file = "graph_render.html"
    net.save_graph(html_file)
    return html_file


# --- 3. DASHBOARD UI ---
st.title("🌐 AI-Powered Global Ontology Engine")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📥 Data Ingestion")
    user_input = st.text_area(
        "Raw Intelligence Feed",
        height=200,
        placeholder="Paste news or geopolitical reports here...",
    )

    if st.button("Extract & Map", type="primary"):
        if user_input:
            with st.spinner("Processing via Local LLM..."):
                try:
                    results = extract_intelligence(user_input)
                    update_graph(results)
                    st.success("Ontology Updated!")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
        else:
            st.warning("Please enter text to process.")

    if st.button("Clear Ontology"):
        st.session_state.kg.clear()
        st.rerun()

with col2:
    st.header("🕸️ Intelligence Graph")
    if len(st.session_state.kg.nodes) > 0:
        html_path = generate_graph_html()
        with open(html_path, "r", encoding="utf-8") as f:
            source_code = f.read()
        components.html(source_code, height=670, scrolling=False)
    else:
        st.info("The graph is empty. Ingest data to build the ontology.")

# --- 4. LOCAL GraphRAG ENGINE ---
st.divider()
st.header("💬 Strategic AI Analyst (GraphRAG)")


def retrieve_graph_context(query, graph):
    """Finds edges in the graph related to the query."""
    if len(graph.nodes) == 0:
        return "The graph is currently empty."

    query_words = query.lower().split()
    relevant_context = []

    # Check if nodes match the query
    for u, v, data in graph.edges(data=True):
        if any(word in u.lower() or word in v.lower() for word in query_words):
            relationship = data.get("label", "is connected to")
            relevant_context.append(f"- {u} {relationship} {v}")

    if not relevant_context:
        # Fallback to general context if no direct match
        edges = list(graph.edges(data=True))[:10]
        return "\n".join(
            [f"- {u} {d.get('label', 'connects to')} {v}" for u, v, d in edges]
        )

    return "\n".join(relevant_context)


chat_input = st.text_input(
    "Query the Intelligence Graph (e.g., 'How are the US and China connected?')..."
)

if chat_input:
    with st.spinner("Traversing Knowledge Graph & Generating Insight..."):
        graph_context = retrieve_graph_context(chat_input, st.session_state.kg)

        rag_prompt = f"""
        You are a geopolitical intelligence analyst. Answer the user's question using ONLY the provided Knowledge Graph Context.
        If the context does not contain the answer, say "I do not have enough data in the current intelligence graph."

        Knowledge Graph Context:
        {graph_context}

        User Question: {chat_input}

        Provide a concise, strategic answer:
        """

        try:
            # Query local Ollama API directly
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2", "prompt": rag_prompt, "stream": False},
            )

            result_text = response.json().get("response", "No response generated.")

            st.markdown("### 🧠 Analyst Insight")
            st.write(result_text)

            with st.expander("🔍 View Traversed Graph Context (Transparency)"):
                st.code(graph_context)

        except Exception as e:
            st.error(
                f"Failed to connect to Local Ollama. Is `ollama run llama3.2` running? Error: {e}"
            )
