from flask import Flask, render_template, request
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
app = Flask(__name__)

@app.route("/")
@app.route("/about")
def about():
    return render_template("index.html", title="Brian Frechette - About")

@app.route("/experience")
def experience():
    return render_template("experience.html", title="Brian Frechette - Experience")

@app.route("/projects")
def projects():
    return render_template("projects.html", title="Brian Frechette - Projects")

@app.route("/brian_ai")
def brian_ai():
    return render_template("brian_ai.html", title="Brian.AI - ChatBot")

@app.route("/get-chatbot-response", methods=["POST"])
def get_chatbot_response():

    # Connect to vector database for context retrieval
    vector_store = PineconeVectorStore(
        index_name="digital-twin",
        embedding=OpenAIEmbeddings()
    )

    # Use pinecone vector db as retriver, chat gpt as LLM, give basic prompt
    retriever = vector_store.as_retriever()  # Your retriever
    llm = ChatOpenAI()
    system_prompt = (
        "You are Brian's AI personal assistant."
        "Your purpose is to speak on Brian's behalf and answer questions about his hobbies, interests, professional experience, education, and career goals."
        "Use the given context to answer the question. "
        "If you don't know the answer, politely say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "If any questions are asked that are inappropriate or irrelevant to Brian, let the user know your only purpose is to answer questions related to Brian's hobbies, interests, and career.."
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    # Pass in user message, return answer
    try:
        response = chain.invoke({"input": request.form["user_message"]})
    except:
        return "I am sorry, the chatbot is currently undergoing maintenance at this time."
    return response["answer"]

if __name__ == "__main__":
    app.run(debug=True)