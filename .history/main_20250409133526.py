import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_ollama import OllamaLLM

# Load environment variables from .env file
load_dotenv()

# Connect to Neo4j Aura
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Connect to your local Ollama model
llm = OllamaLLM(model="mistral")

# Create the chain
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True  # Required to run Cypher queries
)

# Simple CLI to ask questions
def ask_question():
    print("\n🧪 Ask a question about your drug knowledge graph (type 'exit' to quit):")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        try:
            result = chain.invoke({"query": user_input})
            print("\n🧠 Answer:", result, "\n")
        except Exception as e:
            print("⚠️ Error:", e)

if __name__ == "__main__":
    ask_question()
