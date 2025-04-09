import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_ollama import OllamaLLM

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

llm = OllamaLLM(model="mistral")

chain = GraphCypherQAChain.from_llm(llm=llm, graph=graph, verbose=True)

def ask():
    print("\nAsk a question about the drug graph (type 'exit' to quit):")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        try:
            result = chain.invoke({"query": user_input})
            print("🧠", result)
        except Exception as e:
            print("⚠️", e)

if __name__ == "__main__":
    ask()

