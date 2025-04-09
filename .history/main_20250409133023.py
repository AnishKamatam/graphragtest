from langchain_neo4j.graph.cypher_qa import Neo4jCypherQAChain
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

llm = OllamaLLM(model="mistral")

# ✅ Use the modern LangChain Neo4j Cypher QA chain
chain = Neo4jCypherQAChain.from_llm(llm=llm, graph=graph, verbose=True)

def ask_question():
    print("\nAsk me anything about the drug graph (type 'exit' to quit):")
    while True:
        query = input(">> ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            result = chain.invoke({"query": query})
            print("🧠 Answer:", result)
        except Exception as e:
            print("⚠️ Error:", e)

if __name__ == "__main__":
    ask_question()
