from langchain_community.document_loaders import TextLoader
# Import the CharacterTextSplitter class from langchain_text_splitters to split texts into chunks based on character count.
from langchain_text_splitters import CharacterTextSplitter

import openai
from langchain.llms import OpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
# Import ChatOpenAI from langchain_openai module.
# This class is designed to interface with OpenAI's models for generating responses.
from langchain_openai import ChatOpenAI

# Import OpenAI class from langchain_openai.
# This is an additional import from the same module, used for other interactions with OpenAI's API that aren't handled by ChatOpenAI.
from langchain_openai import OpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

import getpass
import os

#logger.init_logging()

#1 Load text data
text = r"""Sarah is an employee at prismaticAI, a leading technology company based in Westside Valley. She has been working there for the past three years as a software engineer.
Michael is also an employee at prismaticAI, where he works as a data scientist. He joined the company two years ago after completing his graduate studies.
prismaticAI is a well-known technology company that specializes in developing cutting-edge software solutions and artificial intelligence applications. The company has a diverse workforce of talented individuals from various backgrounds.
Both Sarah and Michael are highly skilled professionals who contribute significantly to prismaticAI's success. They work closely with their respective teams to develop innovative products and services that meet the evolving needs of the company's clients."""

text2 = r"""Ashok Kumar "A.K" fixes bombs in three places and informs the cops about them. Before the cops arrive, he sets off the bombs. 
Ashok and his lady love, Maya, threaten her old college mate hacker, Arjun, forcing him to hack each system simultaneously. 
Ashok threatens Sriram Raghavan by trying to kill his child so that he can extract the truth about black money, where he murders Sriram. 
Arjun gets irritated by the things happening around him and complains about Ashok's misdeeds to Inspector Prakash. When Ashok gets closer to achieving his goals, Arjun gives him away to the cops, and both are arrested. 
Upon discovering his true identity, the police release Arjun. Later, Maya, who had escaped from the scene, tracks Arjun and his girlfriend Anitha and tells them about Ashok's original motive."""
#loader = TextLoader(text)
#documents = loader.load()


def read_api_key(key_file):
    os.environ['OPENAI_API_KEY'] = open(key_file, 'r').read().strip()
    openai.api_key = os.getenv('OPENAI_API_KEY')


def init_llm():
    read_api_key(r"C:\Users\gopal\OneDrive\Documents\Data\Projects\key.txt")
    openai_apiKey = openai.api_key

    # Initialize LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo", openai_api_key=openai_apiKey)
    return llm


def get_graph_documents(llm, text):
    documents = [text]
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = text_splitter.create_documents(documents)
    # Extract Knowledge Graph
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(texts)
    return graph_documents


def upload_to_graph(graph_docs):
    graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="r2bnb.ai")
    graph.add_graph_documents(graph_docs)

    graph.refresh_schema()
    return graph


def implement_graph_chain(llm, graph):
    chain = GraphCypherQAChain.from_llm(
        cypher_llm=ChatOpenAI(temperature=0, model_name="gpt-4-turbo"),
        qa_llm=ChatOpenAI(temperature=0, model_name="gpt-4-turbo"),
        graph=graph,
        verbose=True,
    )
    return chain


def execute_graph_chain(queries, chain):
    for query in queries:
        print(query)
        print(chain.invoke(query)['result'])


def construct_queries1():
    queries = []
    queries.append("Where does Sarah work?")
    queries.append("Who are the people working for prismaticAI based on what you see in the database?")
    queries.append("Does Michael work for the same company as Sarah?")
    return queries


def main():
    llm = init_llm()
    graph_docs = get_graph_documents(llm, text)
    graph = upload_to_graph(graph_docs)
    chain = implement_graph_chain(llm, graph)
    queries = construct_queries1()
    execute_graph_chain(queries, chain)


if __name__ == '__main__':
    main()