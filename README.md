Imports and Environment Setup
python
Copy code
import os
import openai
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
Imports:
os and dotenv: Used to manage and load environment variables securely.
openai: Enables interaction with OpenAI APIs.
LangChain Modules: Various components are imported to create the AI-powered system, such as chat models, document loaders, text splitters, embeddings, and vector stores.
python
Copy code
# Load environment variables from .env file
load_dotenv()
# Set OpenAI API key from environment variable
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
Environment Variables:
Loads the .env file to securely access the OPENAI_API_KEY.
Sets the key in the environment for OpenAI interaction.
LLM Initialization
python
Copy code
# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
Initializes a ChatGPT model (gpt-3.5-turbo) for generating responses.
Sets temperature to 0 to ensure deterministic, precise outputs.
PDF Document Loading
python
Copy code
# Load the PDF document
loader = PyPDFLoader(r"C:\Users\ISHAN\Music\Coronavirus.pdf")
docs = loader.load()
Uses PyPDFLoader to load the content of a PDF file specified by its path.
docs contains the extracted content in a format ready for processing.
Document Splitting
python
Copy code
# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
# Split the documents into chunks
chunks = text_splitter.split_documents(docs)
Text Splitting:
Splits the document into manageable "chunks" of 400 characters each, with a 50-character overlap between chunks.
Overlap ensures continuity for context across splits.
Embedding Initialization and Document Embedding
python
Copy code
# Initialize embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
# Embed document chunks
chunk_embeddings = embedding_model.embed_documents(
    [chunk.page_content for chunk in chunks])
Embeddings:
text-embedding-ada-002: OpenAI's model is used to create numerical representations of text chunks.
The embed_documents method converts each chunk into an embedding vector, facilitating similarity-based retrieval.
Vector Store and Retriever Setup
python
Copy code
# Initialize the vector database
vector_db = FAISS.from_documents(chunks, embedding_model)
# Create the retriever
retriever = vector_db.as_retriever()
Vector Store:
Uses FAISS (Facebook AI Similarity Search) for indexing and querying document embeddings.
Retriever:
Allows retrieval of the most relevant chunks based on query similarity.
Prompt Template Setup
python
Copy code
# Define the system prompt
system_prompt = (
    "You are an intelligent chatbot that answers only using the information provided in the loaded data."
    " If the loaded data does not contain any relevant information for the question, respond first 'no idea,' followed by a general knowledge answer."
    " Do not blend the loaded data with general knowledge when answering."
    "\n\n"
    "responses need to short for only 20 words"
    "Context: {context}"
    "\n"
    "Note: Provide a general knowledge answer if the loaded data does not contain the required information, but clearly separate the two responses."
)

# Create the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
Prompt Configuration:
Creates a chatbot with specific behavior:
Answer only using the loaded data.
If data is insufficient, respond with "no idea" and provide a general knowledge answer, separated clearly.
Response constraint: Limits responses to 20 words.
Includes a context field to supply the retrieved chunks dynamically.
ChatPromptTemplate creates a structured prompt for the model.
History and QA Function
python
Copy code
history = []

def qa(question):
    # Retrieve the most relevant chunks from the vector store
    relevant_docs = retriever.invoke(question)

    # Format the context using the retrieved documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Ensure context is not empty
    if not context:
        context = "No relevant information found."

    # Generate the prompt text
    prompt_text = prompt_template.format(input=question, context=context)

    print(f"Prompt Text: {prompt_text}")  # Debugging line

    # Generate the response from the model
    response = llm.invoke(prompt_text)  # Call to the language model

    # Extract the content from the AIMessage object
    response_text = response.content if hasattr(
        response, 'content') else str(response)

    # Check if the response contains 'no idea'
    if "no idea" in response_text.lower():
        final_answer = response_text.split("Human: ")[-1].strip()
    else:
        final_answer = response_text.strip()

    # Update history with the new question and extracted answer
    history.append({"question": question, "answer": final_answer})

    return final_answer
Function Purpose:
Accepts a question.
Retrieves relevant chunks using the retriever.
Prepares a context string by concatenating the retrieved chunks.
Sends the context and question to the LLM for a response.
Handles responses and appends the question-answer pair to the history.
Wrapper Function
python
Copy code
def load_pdf_and_answer(question):
    return qa(question)
Wrapper to answer questions using the QA pipeline.
Optional LLMChain Creation
python
Copy code
# Create the LLMChain (optional, though this is not directly interacting with the retriever here)
chain = LLMChain(llm=llm, prompt=prompt_template)
Creates a LLMChain to manage prompt and LLM integration. This isn't used in the qa() function directly.
