import os
from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaLLM

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

llm = OllamaLLM(model="mistral")  # This connects to http://localhost:11434

chain = GraphCypherQAChain.from_llm(llm=llm, graph=graph, verbose=True)

def ask_question():
    print("\nAsk me anything about the drug graph (type 'exit' to quit):")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        try:
            result = chain.invoke({"query": user_input})
            print("🧠 Answer:", result)
        except Exception as e:
            print("⚠️ Error:", e)

if __name__ == "__main__":
    ask_question()
