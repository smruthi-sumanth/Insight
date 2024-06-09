from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import git
import os
import deeplake
from queue import Queue
import requests

from langchain.vectorstores import DeepLake
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

API_KEY = os.getenv("API_KEY")

# Define constants
model_id = "mistral"
model_kwargs = {"device": "cpu"}
allowed_extensions = ['.py', '.ipynb', '.md']

class Embedder:
    def __init__(self, git_link) -> None:
        self.git_link = git_link
        last_name = self.git_link.split('/')[-1]
        self.clone_path = last_name.split('.')[0]
        self.deeplake_path = f"hub://smruthisumanthrao/{self.clone_path}"
        self.model = ChatOllama(model="mistral")
        self.ol = OllamaEmbeddings(model=model_id)
        self.MyQueue = Queue(maxsize=2)

    def add_to_queue(self, value):
        if self.MyQueue.full():
            self.MyQueue.get()
        self.MyQueue.put(value)

    def clone_repo(self):
        if not os.path.exists(self.clone_path):
            # Clone the repository
            git.Repo.clone_from(self.git_link, self.clone_path)

    def extract_all_files(self):
        root_dir = self.clone_path
        self.docs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in allowed_extensions:
                    try:
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        self.docs.extend(loader.load_and_split())
                    except Exception as e:
                        pass

    def chunk_files(self):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = text_splitter.split_documents(self.docs)
        self.num_texts = len(self.texts)

    def embed_deeplake(self):
        db = DeepLake(dataset_path=self.deeplake_path,token=API_KEY, embedding_function=self.ol)
        db.add_documents(self.texts)
        return db

    def delete_directory(self, path):
        if os.path.exists(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            os.rmdir(path)

    def load_db(self):
        exists = deeplake.exists(path=self.deeplake_path, token=API_KEY, org_id="smruthisumanthrao")
        if exists:
            # Just load the DB
            self.db = DeepLake(self.deeplake_path,
                                       token=API_KEY,
                                       read_only=True,  
                                       embedding_function=self.ol,  
            )
        else:
            # Create and load
            self.extract_all_files()
            self.chunk_files()
            self.db = self.embed_deeplake()

        self.retriever = self.db.as_retriever()
        self.retriever.search_kwargs['distance_metric'] = 'cos'
        self.retriever.search_kwargs['fetch_k'] = 100
        self.retriever.search_kwargs['k'] = 3

    def retrieve_results(self, query):
        chat_history = list(self.MyQueue.queue)
        qa = ConversationalRetrievalChain.from_llm(
            self.model,
            chain_type="stuff",
            retriever = self.retriever,
            condense_question_llm=ChatOllama(temperature=0, model="llama2")
        )
        result = qa({"question": query, "chat_history": chat_history})
        self.add_to_queue((query, result["answer"]))
        return result['answer']
    
        
