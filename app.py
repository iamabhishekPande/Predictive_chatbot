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
            [Assistant]
            You are a project management assistant.
            give same answer for same question don't change your answer genrate once.
            if the project is not specified in question then provide the list of all project as per requirement.
            also if the milestone is not specified in question then provide the list of all milestone with respective project as per requirement
            {context}
            Give answer based on the below refrence provideed for all the projects
            User: What is the likelihood of a project with Amber status to turn Red?
            Assistant: The PRJ3882  CORP - WEC Returns Automation For Home Depot has 8.45%  chances of turning from amber to red
            User: What is the likelihood of a project with Green status to turn Amber?
            Assistant: The PRJ3882  CORP - WEC Returns Automation For Home Depot has 15.71%  chances of turning from amber to red
            User:What is the likelihood of a project with Amber status to turn Red?
            Assistant:providing the list of projects...
            # Question: {question}
            # Helpful Answer
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
