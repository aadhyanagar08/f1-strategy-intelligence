from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")
response = llm.invoke("In one sentence, what is DRS in Formula 1?")
print(response.content)