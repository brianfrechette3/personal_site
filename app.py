import os
import random
from flask import Flask, render_template, request, session
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

app = Flask(__name__)
session_id = os.urandom(24)
# session_id = random.randint(1, 9**9)
app.secret_key = session_id

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
    return render_template("brian_ai.html", title="Brian's AI Assistant")

@app.route("/get-chatbot-response", methods=["POST"])
def get_chatbot_response():

    # Connect to vector database for context retrieval
    vector_store = PineconeVectorStore(
        index_name="digital-twin",
        embedding=OpenAIEmbeddings()
    )

    # Use pinecone vector db as retriver, chat gpt as LLM, give basic prompt
    retriever = vector_store.as_retriever()  # Your retriever
    llm = ChatOpenAI(temperature=0.9)

    # Contextualize question
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("memory"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Prompt LLM with clear instructions and updated context
    system_prompt = (
        "You are Brian's AI personal assistant."
        "Your purpose is to speak on Brian's behalf and answer questions about his hobbies, interests, professional experience, education, and career goals."
        "Use the given context to answer the question. "
        "If you don't know the answer, politely say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "If any questions are asked that are inappropriate or irrelevant to Brian, let the user know your only purpose is to answer questions related to Brian's hobbies, interests, and career.."
        "Context: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("memory"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Pass in session memory to keep current conversation context, save it out after response
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_memory,
        input_messages_key="input",
        history_messages_key="memory",
        output_messages_key="answer",
    )
    
    # Pass in user message, return answer
    try:
        response = conversational_rag_chain.invoke(
            {"input": request.form["user_message"]},
            # config={
            #     "configurable": {"session_id": session_id}
            # },
        )["answer"]
        store_message_in_session(request.form["user_message"], response)
    except Exception as e:
        print(f"Exception querying chain: {e}")
        return "I am sorry, the chatbot is currently undergoing maintenance at this time."
    return response

def get_session_memory():
    """Return memory for current session"""
    
    if "memory" not in session:
        session["memory"] = []
    
    # TODO: Update process when figuring out better solution
    # for memory - this doesn't scale well
    chat_hist = ChatMessageHistory()
    for queries in session["memory"]:
        chat_hist.add_user_message(queries["user"])
        chat_hist.add_ai_message(queries["bot"])
    return chat_hist

def store_message_in_session(user_message, bot_response):
    """Store the user message and bot response in session memory"""

    if "memory" not in session:
        session["memory"] = []
        
    # Append user message and bot response to the session memory
    session["memory"] = session["memory"] + [{"user": user_message, "bot": bot_response}]
    if len(session["memory"]) > 10:  # Keep the last 10 messages
        print("removing oldest message from memory")
        session["memory"].pop(0)  # Remove the oldest message


if __name__ == "__main__":
    app.run(debug=True)