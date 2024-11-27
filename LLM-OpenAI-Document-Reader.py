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

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Load the PDF document
loader = PyPDFLoader(r"C:\Users\ISHAN\Music\Coronavirus.pdf")
docs = loader.load()

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=50)

# Split the documents into chunks
chunks = text_splitter.split_documents(docs)

# Initialize embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Embed document chunks
chunk_embeddings = embedding_model.embed_documents(
    [chunk.page_content for chunk in chunks])

# Initialize the vector database
vector_db = FAISS.from_documents(chunks, embedding_model)

# Create the retriever
retriever = vector_db.as_retriever()

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

# history
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


def load_pdf_and_answer(question):
    return qa(question)


# Create the LLMChain (optional, though this is not directly interacting with the retriever here)
chain = LLMChain(llm=llm, prompt=prompt_template)
