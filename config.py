from dataclasses import dataclass
from box import box_from_file
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os


@dataclass
class Config:

    def __post_init__(self):
        self.__set_vars_from_yaml("configs/app_config.yml")
        self.__set_vars_from_yaml("configs/metadata_filters.yml")


    @property
    def LLM_MODEL(self):
        
        if self.provider.llm.lower() == "groq":
            llm = ChatGroq(**self.LLM.groq)
        elif self.provider.llm.lower() in ["google", "gemini"]:
            llm = ChatGoogleGenerativeAI(**self.LLM.google)
        elif self.provider.llm.lower() in ["openai"]:
            llm = ChatOpenAI(**self.LLM.openai)
        elif self.provider.llm.lower() in ["openrouter"]:
            llm = ChatOpenAI(**self.LLM.openrouter)
        else:
            raise ValueError("Invalid/Unsupported key provided for llm_provider. Choose one of groq/gemini")
        return llm

    @property
    def EMBEDDINGS_MODEL(self):

        if self.provider.embeddings.lower() == "groq":
            emb_model = HuggingFaceEmbeddings(**self.EMBEDDINGS.groq)
        elif self.provider.embeddings.lower() in ["google", "gemini"]:
            emb_model = GoogleGenerativeAIEmbeddings(**self.EMBEDDINGS.google)
        elif self.provider.embeddings.lower() in ["openai"]:
            emb_model = OpenAIEmbeddings(**self.EMBEDDINGS.openai)
        else:
            raise ValueError("Invalid/Unsupported key provided for embeddings_provider. Choose one of groq/gemini")
        return emb_model

    def __set_vars_from_yaml(self, file_path):
        file = box_from_file(file_path)
        for key, value in file.items():
            setattr(self, key, value)
