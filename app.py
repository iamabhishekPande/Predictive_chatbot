from flask import Flask, request, render_template, jsonify
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate


app = Flask(__name__)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Define the folder where you want to save the uploaded CSV files
UPLOAD_FOLDER = "uploads"

# Create the folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    context = {"title": "Prediction Model", "message": "Hello, World!"}
    return render_template("index.html", context=context)

@app.route("/upload", methods=["POST"])
def upload_csv():
    uploaded_file_paths = []

    files = request.files.getlist("files")

    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save the uploaded file to the specified folder
        file.save(file_path)

        uploaded_file_paths.append(file_path)

    return jsonify({"message": f"{len(files)} CSV file(s) uploaded successfully", "file_paths": uploaded_file_paths})

@app.route("/chat", methods=["POST"])
def conversational_chat():
    try:
        question_data = request.json
        # Extract the question from the JSON input
        question = question_data["question"]

        # Initialize Langchain components
        embeddings = OpenAIEmbeddings()

        loader = CSVLoader(file_path="uploads/Final_Data_Updated_1.csv", encoding="cp1252", csv_args={'delimiter': ','})
        data = loader.load()
        
        langchain_prompt_template = """
        # Chat roles
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"
        
                  
        system_message_chat_conversation = Assistant helps the company employees with their project plan questions, and questions about the projects. Be brief in your answers.
            Answer ONLY with the facts which are in the excel. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
            For tabular information return it as an html table. Do not return markdown format. If the question is not in English, answer in the language used in the question.
            "Don't provide different answers for the same question"


        system_chat_prompt = \
            "You are an intelligent assistant helping employees with their Project and Project details questions. " + \
            "Use 'you' to refer to the individual asking the questions even if they ask with 'I'. " + \
            "Answer the following question using only the data provided in the excel. " + \
            "When asked in Tabular format then only provide, Don't provide it yourself"
            "For tabular information return it as an html table. Do not return markdown format. "  + \
            "If you don't know the answer then don't try to make the answers by yourself,Just say I Don't know."
            "All questions must be answered from the results from search or look up actions, only facts resulting from those can be used in an answer. "
            "Answer questions as truthfully as possible, and ONLY answer the questions using the information from observations, do not speculate or your own knowledge."
            "Don't provide different answers for the same question"
            "Conclusion:Conclude the conversation and offer further assistance."

    
        user_prompt=You are a helpful Project Management Assistant.
            "Don't provide different answers for the same question"
            Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a excel about Project details.
            If the question is not in English, translate the question to English before generating the search query.
            You must need to follow some steps before answering the user question:

            1.Introduction and Context: Greet the user and establish the context.
            2.You have data regarding the projects so answer user question after getting full understanding of the projects.
            3.You need to undesratnd the user question then only you need to answer the question, if you don't understand the question then ask the user to elaborate the questionb.
            
        
        {context}
       
            USER : 'What is the likelihood of a project with Amber status to turn Red?'
            ASSISTANT : "Here is the list of projects which are turning Red from Amber status are CORP - WEC Returns Automation For Home Depot: 8.45%,GTS- NA Millcreek WMS Conversion Legacy to JDA: 7.88%,2023 - EA - Tech Debt Management Optimization: 19.57%,GTS - 2023 - CA – ServiceNet:7.64%, KOFAX Upgrade:15.21%" 
            USER : 'What is the likelihood of a project with Green status to turn Amber?'
            ASSISTANT : "Here is the list of projects which are turning Amber from Green status are CORP - WEC Returns Automation For Home Depot: 15.71%,GTS- NA Millcreek WMS Conversion Legacy to JDA: 22.80%,2023 - EA - Tech Debt Management Optimization: 19.85%,GTS - 2023 - CA – ServiceNet:8.05%,KOFAX Upgrade:22.80% " 
            USER : 'Which project is likely be have an escalation in the coming week/month?' 
            ASSISTANT : "Here is the list of projects which can cause escalation in upcoming week/month are CORP - WEC Returns Automation For Home Depot, Sphinx – Security Divestiture – Project Sphinx – Sub Project Create SAP Contract (RMR) from SES SFDC  , Tax/Stat - Colorado Delivery Fee" 

        # Question: {question}

      
            
        """
        prompt = PromptTemplate(input_variables=["question","context"],template=langchain_prompt_template)
        vectorstore = FAISS.from_documents(data, embeddings)
        retriever = vectorstore.as_retriever()
        chain = RetrievalQA.from_chain_type(OpenAI(), chain_type="stuff", retriever=retriever,chain_type_kwargs={"prompt": prompt} )
        result = chain(question)
        

        return jsonify({"answer": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8000,debug=True)
