import os
from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph  # ✅ community version
from langchain_ollama import OllamaLLM             # ✅ new Ollama class

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

llm = OllamaLLM(model="mistral")  # works with LangChain 0.3+

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
