#1. Import Required Libraries
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
os: Provides access to environment variables and file paths.
openai: Enables interaction with OpenAI's API for embeddings or language models.
dotenv: Manages environment variables from a .env file.
langchain: Offers tools for building applications using language models. Key components here:
ChatOpenAI: Wraps OpenAI's GPT models for use in chat applications.
PyPDFLoader: Loads content from PDF files.
RecursiveCharacterTextSplitter: Splits long documents into manageable chunks for processing.
OpenAIEmbeddings: Generates vector embeddings from text using OpenAI models.
FAISS: A vector database to store and retrieve embeddings for similarity search.
ChatPromptTemplate: Creates structured prompts for chat-based LLMs.
LLMChain: Chains prompts and responses for querying the LLM.
#2. Environment Variables Setup
python
Copy code
# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
Loads the .env file to fetch sensitive configuration values like the OpenAI API key.
Sets the API key in the environment so the OpenAI library can authenticate requests.
3. Initialize Chat Model
python
Copy code
# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
ChatOpenAI is initialized with the model gpt-3.5-turbo for chatbot functionality.
temperature=0 ensures deterministic responses, favoring precision over randomness.
4. Load and Preprocess the PDF Document
python
Copy code
# Load the PDF document
loader = PyPDFLoader(r"C:\Users\ISHAN\Music\Coronavirus.pdf")
docs = loader.load()
Loads the PDF from the given path.
Converts the PDF content into a list of documents (docs) for further processing.
python
Copy code
# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=50)

# Split the documents into chunks
chunks = text_splitter.split_documents(docs)
RecursiveCharacterTextSplitter breaks long text into smaller chunks (400 characters each with 50-character overlap) to ensure better embedding and retrieval performance.
Splits the loaded docs into smaller chunks (chunks).
5. Generate Embeddings for Chunks
python
Copy code
# Initialize embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Embed document chunks
chunk_embeddings = embedding_model.embed_documents(
    [chunk.page_content for chunk in chunks])
OpenAIEmbeddings creates embeddings using OpenAI's text-embedding-ada-002 model.
embed_documents generates vector representations for each chunk's text content.
6. Initialize FAISS Vector Database
python
Copy code
# Initialize the vector database
vector_db = FAISS.from_documents(chunks, embedding_model)

# Create the retriever
retriever = vector_db.as_retriever()
FAISS: A fast, in-memory vector database for similarity search.
Stores the embeddings along with their associated text chunks.
Converts the database into a retriever, allowing relevant chunks to be fetched based on query similarity.
7. Create Prompt Template
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
system_prompt: Provides instructions to the chatbot:
Use only the provided data for answers.
Include a "no idea" response if relevant information isn’t found, followed by general knowledge as a fallback.
Limit responses to 20 words.
ChatPromptTemplate: Constructs a structured prompt, combining system instructions and user input.
8. Define QA Functionality
python
Copy code
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
Retrieve Chunks: Fetches relevant text chunks from the vector database.
Create Context: Joins the retrieved chunks into a single context string.
Format Prompt: Combines the system instructions, context, and question into a formatted prompt.
Invoke LLM: Passes the prompt to the language model for generating a response.
Post-Process Response:
If the response includes "no idea," separate the fallback general knowledge.
Otherwise, use the response as-is.
Track History: Logs each question and its answer for future reference.
9. Helper Function and Chain
python
Copy code
def load_pdf_and_answer(question):
    return qa(question)
Wrapper function to answer questions using the loaded PDF.
python
Copy code
# Create the LLMChain (optional, though this is not directly interacting with the retriever here)
chain = LLMChain(llm=llm, prompt=prompt_template)
LLMChain: Creates a reusable pipeline with the LLM and prompt, although it’s not integrated with the retriever here.
