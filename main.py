from rag_pipeline import create_chain

qa_chain = create_chain()

def run_console():
    while True:
        query = input("❓ Your question: ")
        if query.lower() == "exit":
            break
        response = qa_chain.invoke({"query": query})
        print("\n💡 Answer:", response["result"], "\n")

if __name__ == "__main__":
    run_console()
