import os
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

chain = GraphCypherQAChain.from_llm(
    llm=llm, 
    graph=graph, 
    verbose=True,
    allow_dangerous_requests=True
)

def ask_question():
    print("\nAsk me anything about the drug graph (type 'exit' to quit):")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        try:
            result = chain.run(user_input)
            print("🧠 Answer:", result)
        except Exception as e:
            print("⚠️ Error:", e)

if __name__ == "__main__":
    ask_question()