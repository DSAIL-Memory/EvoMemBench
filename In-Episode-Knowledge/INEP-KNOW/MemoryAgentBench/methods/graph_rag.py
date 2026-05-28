import networkx as nx
from langchain_community.vectorstores import FAISS

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate
try:
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
except ImportError:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
try:
    from langchain_community.callbacks import get_openai_callback
except ImportError:
    from langchain.callbacks import get_openai_callback
from langchain_openai import OpenAIEmbeddings

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.callbacks import BaseCallbackHandler
from volcenginesdkarkruntime import Ark
from typing import Any, List, Tuple, Dict, Optional
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import spacy
import heapq
import argparse
import tiktoken
from pydantic import BaseModel, Field, PrivateAttr


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from spacy.cli import download
from spacy.lang.en import English

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path sicnce we work with notebooks


def _ark_role(msg: BaseMessage) -> str:
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, AIMessage):
        return "assistant"
    return "user"


class TokenCountingCallback(BaseCallbackHandler):
    """Accumulates prompt/completion tokens across all LangChain LLM calls in a chain."""

    def __init__(self):
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def on_llm_end(self, response, **kwargs):
        # OpenAI-shape (ChatOpenAI)
        if response.llm_output and "token_usage" in (response.llm_output or {}):
            u = response.llm_output["token_usage"]
            self.prompt_tokens += u.get("prompt_tokens", 0)
            self.completion_tokens += u.get("completion_tokens", 0)
            return
        # usage_metadata shape: populated by ChatArkBatch, modern ChatOpenAI, ChatGoogleGenerativeAI
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                um = getattr(msg, "usage_metadata", None) if msg else None
                if um:
                    self.prompt_tokens += um.get("input_tokens", 0)
                    self.completion_tokens += um.get("output_tokens", 0)

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def snapshot(self) -> tuple:
        return self.prompt_tokens, self.completion_tokens


class ChatArkBatch(BaseChatModel):
    """LangChain-compatible chat model wrapper backed by Ark batch inference API (火山引擎)."""

    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: str = ""

    _client: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(
            self, "_client",
            Ark(api_key=kwargs.get("api_key") or os.getenv("DEEPSEEK_API_KEY", ""))
        )

    @property
    def _llm_type(self) -> str:
        return "ark-batch"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs,
    ) -> ChatResult:
        ark_messages = [{"role": _ark_role(m), "content": m.content} for m in messages]
        resp = None
        for attempt in range(10):
            try:
                resp = self._client.batch.chat.completions.create(
                    model=self.model,
                    messages=ark_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                break
            except Exception as e:
                if attempt == 9:
                    raise
                wait = 5 * (2 ** min(attempt, 5))
                import logging
                logging.getLogger(__name__).warning(
                    f"Ark batch API call failed (attempt {attempt+1}/10): {e}. Retrying in {wait}s..."
                )
                time.sleep(wait)
        text = resp.choices[0].message.content or ""
        usage_meta = {}
        if resp.usage:
            usage_meta = {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        msg = AIMessage(content=text, usage_metadata=usage_meta)
        return ChatResult(generations=[ChatGeneration(message=msg)])

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
# Set the Google API key environment variable for Gemini models
if os.getenv('GOOGLE_API_KEY'):
    os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


# Define the document processor class
# Define the DocumentProcessor class
class DocumentProcessor:
    def __init__(self):
        """
        Initializes the DocumentProcessor with a text splitter and OpenAI embeddings.

        Attributes:
        - text_splitter: An instance of RecursiveCharacterTextSplitter with specified chunk size and overlap.
        - embeddings: An instance of OpenAIEmbeddings used for embedding documents.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=200)
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL")
        if dashscope_api_key and dashscope_base_url:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-v4",
                openai_api_key=dashscope_api_key,
                openai_api_base=dashscope_base_url,
                dimensions=1024,
                check_embedding_ctx_length=False,
                chunk_size=10,
            )
        else:
            self.embeddings = OpenAIEmbeddings()

    def process_documents(self, documents):
        """
        Processes a list of documents by splitting them into smaller chunks and creating a vector store.

        Args:
        - documents (list of str): A list of documents to be processed.

        Returns:
        - tuple: A tuple containing:
          - splits (list of str): The list of split document chunks.
          - vector_store (FAISS): A FAISS vector store created from the split document chunks and their embeddings.
        """
        splits = self.text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(splits, self.embeddings)
        return splits, vector_store

    def create_embeddings_batch(self, texts, batch_size=32):
        """
        Creates embeddings for a list of texts in batches.

        Args:
        - texts (list of str): A list of texts to be embedded.
        - batch_size (int, optional): The number of texts to process in each batch. Default is 32.

        Returns:
        - numpy.ndarray: An array of embeddings for the input texts.
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def compute_similarity_matrix(self, embeddings):
        """
        Computes a cosine similarity matrix for a given set of embeddings.

        Args:
        - embeddings (numpy.ndarray): An array of embeddings.

        Returns:
        - numpy.ndarray: A cosine similarity matrix for the input embeddings.
        """
        return cosine_similarity(embeddings)


# Define the knowledge graph class
# Define the Concepts class
class Concepts(BaseModel):
    concepts_list: List[str] = Field(description="List of concepts")


# Define the KnowledgeGraph class
class KnowledgeGraph:
    def __init__(self):
        """
        Initializes the KnowledgeGraph with a graph, lemmatizer, and NLP model.

        Attributes:
        - graph: An instance of a networkx Graph.
        - lemmatizer: An instance of WordNetLemmatizer.
        - concept_cache: A dictionary to cache extracted concepts.
        - nlp: An instance of a spaCy NLP model.
        - edges_threshold: A float value that sets the threshold for adding edges based on similarity.
        """
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}
        self.nlp = self._load_spacy_model()
        self.edges_threshold = 0.8

    def build_graph(self, splits, llm, embedding_model):
        """
        Builds the knowledge graph by adding nodes, creating embeddings, extracting concepts, and adding edges.

        Args:
        - splits (list): A list of document splits.
        - llm: An instance of a large language model.
        - embedding_model: An instance of an embedding model.

        Returns:
        - None
        """
        self._add_nodes(splits)
        embeddings = self._create_embeddings(splits, embedding_model)
        self._extract_concepts(splits, llm)
        self._add_edges(embeddings)

    def _add_nodes(self, splits):
        """
        Adds nodes to the graph from the document splits.

        Args:
        - splits (list): A list of document splits.

        Returns:
        - None
        """
        for i, split in enumerate(splits):
            self.graph.add_node(i, content=split.page_content)

    def _create_embeddings(self, splits, embedding_model):
        """
        Creates embeddings for the document splits using the embedding model.

        Args:
        - splits (list): A list of document splits.
        - embedding_model: An instance of an embedding model.

        Returns:
        - numpy.ndarray: An array of embeddings for the document splits.
        """
        texts = [split.page_content for split in splits]
        return embedding_model.embed_documents(texts)

    def _compute_similarities(self, embeddings):
        """
        Computes the cosine similarity matrix for the embeddings.

        Args:
        - embeddings (numpy.ndarray): An array of embeddings.

        Returns:
        - numpy.ndarray: A cosine similarity matrix for the embeddings.
        """
        return cosine_similarity(embeddings)

    def _load_spacy_model(self):
        """
        Loads the spaCy NLP model, downloading it if necessary.

        Args:
        - None

        Returns:
        - spacy.Language: An instance of a spaCy NLP model.
        """
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def _extract_concepts_and_entities(self, content, llm):
        """
        Extracts concepts and named entities from the content using spaCy and a large language model.

        Args:
        - content (str): The content from which to extract concepts and entities.
        - llm: An instance of a large language model.

        Returns:
        - list: A list of extracted concepts and entities.
        """
        if content in self.concept_cache:
            return self.concept_cache[content]

        # Extract named entities using spaCy
        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]

        # Extract general concepts using LLM
        concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract key concepts (excluding named entities) from the following text:\n\n{text}\n\nRespond in JSON format with a single field 'concepts_list' containing a list of strings."
        )
        try:
            import re as _re, json as _json
            concept_chain = concept_extraction_prompt | llm
            _response = concept_chain.invoke({"text": content})
            _response_text = _response.content if hasattr(_response, 'content') else str(_response)
            _json_match = _re.search(r'\{.*\}', _response_text, _re.DOTALL)
            if _json_match:
                _data = _json.loads(_json_match.group())
                general_concepts = _data.get("concepts_list", [])
            else:
                general_concepts = []
        except Exception as e:
            print(f"Error extracting concepts: {e}")
            general_concepts = []
            
        # Combine named entities and general concepts
        all_concepts = list(set(named_entities + general_concepts))

        self.concept_cache[content] = all_concepts
        return all_concepts

    def _extract_concepts(self, splits, llm):
        """
        Extracts concepts for all document splits using multi-threading.

        Args:
        - splits (list): A list of document splits.
        - llm: An instance of a large language model.

        Returns:
        - None
        """
        #tok = tiktoken.encoding_for_model("gpt-4o-mini")
        #for i, split in enumerate(splits):
        #    print("spilit tokens", len(tok.encode(split.page_content)))
            
        with ThreadPoolExecutor() as executor:
            future_to_node = {executor.submit(self._extract_concepts_and_entities, split.page_content, llm): i
                              for i, split in enumerate(splits)}

            for future in tqdm(as_completed(future_to_node), total=len(splits),
                               desc="Extracting concepts and entities"):
                node = future_to_node[future]
                concepts = future.result()
                self.graph.nodes[node]['concepts'] = concepts


    def _add_edges(self, embeddings):
        """
        Adds edges to the graph based on the similarity of embeddings and shared concepts.

        Args:
        - embeddings (numpy.ndarray): An array of embeddings for the document splits.

        Returns:
        - None
        """
        similarity_matrix = self._compute_similarities(embeddings)
        num_nodes = len(self.graph.nodes)

        for node1 in tqdm(range(num_nodes), desc="Adding edges"):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]
                if similarity_score > self.edges_threshold:
                    shared_concepts = set(self.graph.nodes[node1]['concepts']) & set(
                        self.graph.nodes[node2]['concepts'])
                    edge_weight = self._calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    self.graph.add_edge(node1, node2, weight=edge_weight,
                                        similarity=similarity_score,
                                        shared_concepts=list(shared_concepts))

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha=0.7, beta=0.3):
        """
        Calculates the weight of an edge based on similarity score and shared concepts.

        Args:
        - node1 (int): The first node.
        - node2 (int): The second node.
        - similarity_score (float): The similarity score between the nodes.
        - shared_concepts (set): The set of shared concepts between the nodes.
        - alpha (float, optional): The weight of the similarity score. Default is 0.7.
        - beta (float, optional): The weight of the shared concepts. Default is 0.3.

        Returns:
        - float: The calculated weight of the edge.
        """
        max_possible_shared = min(len(self.graph.nodes[node1]['concepts']), len(self.graph.nodes[node2]['concepts']))
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        return alpha * similarity_score + beta * normalized_shared_concepts

    def _lemmatize_concept(self, concept):
        """
        Lemmatizes a given concept.

        Args:
        - concept (str): The concept to be lemmatized.

        Returns:
        - str: The lemmatized concept.
        """
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])


# Define the Query Engine class
# Define the AnswerCheck class
class AnswerCheck(BaseModel):
    is_complete: bool = Field(description="Whether the current context provides a complete answer to the query")
    answer: str = Field(description="The current answer based on the context, if any")


# Define the QueryEngine class
class QueryEngine:
    def __init__(self, vector_store, knowledge_graph, llm, retrieve_num=5):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        #self.max_context_length = 4000
        self.retrieve_num = retrieve_num
        self.answer_check_chain = self._create_answer_check_chain()

    def _create_answer_check_chain(self):
        """
        Creates a chain to check if the context provides a complete answer to the query.

        Args:
        - None

        Returns:
        - Chain: A chain to check if the context provides a complete answer.
        """
        answer_check_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Given the query: '{query}'\n\nAnd the current context:\n{context}\n\nDoes this context provide a complete answer to the query? If yes, provide the answer. If no, state that the answer is incomplete.\n\nIs complete answer (Yes/No):\nAnswer (if complete):"
        )
        return answer_check_prompt | self.llm

    def _check_answer(self, query: str, context: str) -> Tuple[bool, str]:
        """
        Checks if the current context provides a complete answer to the query.

        Args:
        - query (str): The query to be answered.
        - context (str): The current context.

        Returns:
        - tuple: A tuple containing:
          - is_complete (bool): Whether the context provides a complete answer.
          - answer (str): The answer based on the context, if complete.
        """
        try:
            response = self.answer_check_chain.invoke({"query": query, "context": context})
            text = response.content if hasattr(response, 'content') else str(response)
            _after_label = text.lower().split("is complete answer")[-1] if "is complete answer" in text.lower() else ""
            is_complete = "yes" in _after_label[:10]
            answer = text.split("Answer (if complete):")[-1].strip() if "Answer (if complete):" in text else ""
        except Exception as e:
            print(f"Error checking answer: {e}")
            is_complete = False
            answer = ""

        return is_complete, answer

    def _expand_context(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:
        """
        Expands the context by traversing the knowledge graph using a Dijkstra-like approach.

        This method implements a modified version of Dijkstra's algorithm to explore the knowledge graph,
        prioritizing the most relevant and strongly connected information. The algorithm works as follows:

        1. Initialize:
           - Start with nodes corresponding to the most relevant documents.
           - Use a priority queue to manage the traversal order, where priority is based on connection strength.
           - Maintain a dictionary of best known "distances" (inverse of connection strengths) to each node.

        2. Traverse:
           - Always explore the node with the highest priority (strongest connection) next.
           - For each node, check if we've found a complete answer.
           - Explore the node's neighbors, updating their priorities if a stronger connection is found.

        3. Concept Handling:
           - Track visited concepts to guide the exploration towards new, relevant information.
           - Expand to neighbors only if they introduce new concepts.

        4. Termination:
           - Stop if a complete answer is found.
           - Continue until the priority queue is empty (all reachable nodes explored).

        This approach ensures that:
        - We prioritize the most relevant and strongly connected information.
        - We explore new concepts systematically.
        - We find the most relevant answer by following the strongest connections in the knowledge graph.

        Args:
        - query (str): The query to be answered.
        - relevant_docs (List[Document]): A list of relevant documents to start the traversal.

        Returns:
        - tuple: A tuple containing:
          - expanded_context (str): The accumulated context from traversed nodes.
          - traversal_path (List[int]): The sequence of node indices visited.
          - filtered_content (Dict[int, str]): A mapping of node indices to their content.
          - final_answer (str): The final answer found, if any.
        """
        # Initialize variables
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""

        priority_queue = []
        distances = {}  # Stores the best known "distance" (inverse of connection strength) to each node

        print("\nTraversing the knowledge graph:")

        # Initialize priority queue with closest nodes from relevant docs
        for doc in relevant_docs:
            # Find the most similar node in the knowledge graph for each relevant document
            closest_nodes = self.vector_store.similarity_search_with_score(doc.page_content, k=1)
            closest_node_content, similarity_score = closest_nodes[0]

            # Get the corresponding node in our knowledge graph
            closest_node = next(n for n in self.knowledge_graph.graph.nodes if
                                self.knowledge_graph.graph.nodes[n]['content'] == closest_node_content.page_content)

            # Initialize priority (inverse of similarity score for min-heap behavior)
            priority = 1 / similarity_score
            heapq.heappush(priority_queue, (priority, closest_node))
            distances[closest_node] = priority

        step = 0
        while priority_queue:
            # Get the node with the highest priority (lowest distance value)
            current_priority, current_node = heapq.heappop(priority_queue)

            # Skip if we've already found a better path to this node
            if current_priority > distances.get(current_node, float('inf')):
                continue

            if current_node not in traversal_path:
                step += 1
                traversal_path.append(current_node)
                node_content = self.knowledge_graph.graph.nodes[current_node]['content']
                node_concepts = self.knowledge_graph.graph.nodes[current_node]['concepts']

                # Add node content to our accumulated context
                filtered_content[current_node] = node_content
                expanded_context += "\n" + node_content if expanded_context else node_content

                # Log the current step for debugging and visualization
                print(f"\nStep {step} - Node {current_node}:")
                print(f"Content: {node_content[:100]}...")
                print(f"Concepts: {', '.join(node_concepts)}")
                print("-" * 50)

                # Check if we have a complete answer with the current context
                is_complete, answer = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    break

                # Process the concepts of the current node
                node_concepts_set = set(self.knowledge_graph._lemmatize_concept(c) for c in node_concepts)
                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)

                    # Explore neighbors
                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]
                        edge_weight = edge_data['weight']

                        # Calculate new distance (priority) to the neighbor
                        # Note: We use 1 / edge_weight because higher weights mean stronger connections
                        distance = current_priority + (1 / edge_weight)

                        # If we've found a stronger connection to the neighbor, update its distance
                        if distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))

                            # Process the neighbor node if it's not already in our traversal path
                            if neighbor not in traversal_path:
                                step += 1
                                traversal_path.append(neighbor)
                                neighbor_content = self.knowledge_graph.graph.nodes[neighbor]['content']
                                neighbor_concepts = self.knowledge_graph.graph.nodes[neighbor]['concepts']

                                filtered_content[neighbor] = neighbor_content
                                expanded_context += "\n" + neighbor_content if expanded_context else neighbor_content

                                # Log the neighbor node information
                                print(f"\nStep {step} - Node {neighbor} (neighbor of {current_node}):")
                                print(f"Content: {neighbor_content[:100]}...")
                                print(f"Concepts: {', '.join(neighbor_concepts)}")
                                print("-" * 50)

                                # Check if we have a complete answer after adding the neighbor's content
                                is_complete, answer = self._check_answer(query, expanded_context)
                                if is_complete:
                                    final_answer = answer
                                    break

                                # Process the neighbor's concepts
                                neighbor_concepts_set = set(
                                    self.knowledge_graph._lemmatize_concept(c) for c in neighbor_concepts)
                                if not neighbor_concepts_set.issubset(visited_concepts):
                                    visited_concepts.update(neighbor_concepts_set)

                # If we found a final answer, break out of the main loop
                if final_answer:
                    break

        # If we haven't found a complete answer, generate one using the LLM
        if not final_answer:
            print("\nGenerating final answer...")
            response_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="Based on the following context, please answer the query very concisely.\n\nContext: {context}\n\nQuery: {query}"
            )
            response_chain = response_prompt | self.llm
            
            ### Not exceed 128K
            tokenzier = tiktoken.encoding_for_model("gpt-4o-mini")
            llm_length_limit = 127000
            if len(tokenzier.encode(expanded_context)) > llm_length_limit:
                print("\n\n\n(545) The context is too long, we need to truncate it.\n\n\n")
                encoded = tokenzier.encode(expanded_context)
                expanded_context = tokenzier.decode(encoded[llm_length_limit:])
            ###
            input_data = {"query": query, "context": expanded_context}   
            final_answer = response_chain.invoke(input_data)

        return expanded_context, traversal_path, filtered_content, final_answer

    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:
        """
        Processes a query by retrieving relevant documents, expanding the context, and generating the final answer.

        Args:
        - query (str): The query to be answered.

        Returns:
        - tuple: A tuple containing:
          - final_answer (str): The final answer to the query.
          - traversal_path (list): The traversal path of nodes in the knowledge graph.
          - filtered_content (dict): The filtered content of nodes.
        """
        ## find the "Question" in the message
        import re
        match = re.search(r"Now Answer the Question:\s*(.*)", query, re.DOTALL)
        if match:
            retrieval_query =  ''.join(match.groups())
        else:
            match = re.search(r"Here is the conversation:\s*(.*)", query, re.DOTALL)
            if match:
                retrieval_query =  ''.join(match.groups())
            else:
                retrieval_query = query
        ## End of newly added code
        print(f"\nRetrieval query: {retrieval_query}")
        relevant_docs = self._retrieve_relevant_documents(retrieval_query)
        expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(query,
                                                                                                relevant_docs)

        if not final_answer:
            print("\nGenerating final answer...")
            response_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="Based on the following context, please answer the query very concisely.\n\nContext: {context}\n\nQuery: {query}"
            )

            response_chain = response_prompt | self.llm
            ### Not exceed 128K
            tokenzier = tiktoken.encoding_for_model("gpt-4o-mini")
            llm_length_limit = 127000
            if len(tokenzier.encode(expanded_context)) > llm_length_limit:
                print("\n\n\n(592) The context is too long, we need to truncate it.\n\n\n")
                encoded = tokenzier.encode(expanded_context)
                expanded_context = tokenzier.decode(encoded[llm_length_limit:])
            ###
            input_data = {"query": query, "context": expanded_context}
            response = response_chain.invoke(input_data)
            final_answer = response
        else:
            print("\nComplete answer found during traversal.")

        print(f"\nFinal Answer: {final_answer}")

        # add expanded_context here
        return final_answer, expanded_context, traversal_path, filtered_content

    def _retrieve_relevant_documents(self, query: str):
        """
        Retrieves relevant documents based on the query using the vector store.

        Args:
        - query (str): The query to be answered.

        Returns:
        - list: A list of relevant documents.
        """
        print("\nRetrieving relevant documents...")
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": self.retrieve_num})
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        
        return compression_retriever.invoke(query)


# Import necessary libraries
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Define the Visualizer class
class Visualizer:
    @staticmethod
    def visualize_traversal(graph, traversal_path):
        """
        Visualizes the traversal path on the knowledge graph with nodes, edges, and traversal path highlighted.

        Args:
        - graph (networkx.Graph): The knowledge graph containing nodes and edges.
        - traversal_path (list of int): The list of node indices representing the traversal path.

        Returns:
        - None
        """
        traversal_graph = nx.DiGraph()

        # Add nodes and edges from the original graph
        for node in graph.nodes():
            traversal_graph.add_node(node)
        for u, v, data in graph.edges(data=True):
            traversal_graph.add_edge(u, v, **data)

        fig, ax = plt.subplots(figsize=(16, 12))

        # Generate positions for all nodes
        pos = nx.spring_layout(traversal_graph, k=1, iterations=50)

        # Draw regular edges with color based on weight
        edges = traversal_graph.edges()
        edge_weights = [traversal_graph[u][v].get('weight', 0.5) for u, v in edges]
        nx.draw_networkx_edges(traversal_graph, pos,
                               edgelist=edges,
                               edge_color=edge_weights,
                               edge_cmap=plt.cm.Blues,
                               width=2,
                               ax=ax)

        # Draw nodes
        nx.draw_networkx_nodes(traversal_graph, pos,
                               node_color='lightblue',
                               node_size=3000,
                               ax=ax)

        # Draw traversal path with curved arrows
        edge_offset = 0.1
        for i in range(len(traversal_path) - 1):
            start = traversal_path[i]
            end = traversal_path[i + 1]
            start_pos = pos[start]
            end_pos = pos[end]

            # Calculate control point for curve
            mid_point = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
            control_point = (mid_point[0] + edge_offset, mid_point[1] + edge_offset)

            # Draw curved arrow
            arrow = patches.FancyArrowPatch(start_pos, end_pos,
                                            connectionstyle=f"arc3,rad={0.3}",
                                            color='red',
                                            arrowstyle="->",
                                            mutation_scale=20,
                                            linestyle='--',
                                            linewidth=2,
                                            zorder=4)
            ax.add_patch(arrow)

        # Prepare labels for the nodes
        labels = {}
        for i, node in enumerate(traversal_path):
            concepts = graph.nodes[node].get('concepts', [])
            label = f"{i + 1}. {concepts[0] if concepts else ''}"
            labels[node] = label

        for node in traversal_graph.nodes():
            if node not in labels:
                concepts = graph.nodes[node].get('concepts', [])
                labels[node] = concepts[0] if concepts else ''

        # Draw labels
        nx.draw_networkx_labels(traversal_graph, pos, labels, font_size=8, font_weight="bold", ax=ax)

        # Highlight start and end nodes
        start_node = traversal_path[0]
        end_node = traversal_path[-1]

        nx.draw_networkx_nodes(traversal_graph, pos,
                               nodelist=[start_node],
                               node_color='lightgreen',
                               node_size=3000,
                               ax=ax)

        nx.draw_networkx_nodes(traversal_graph, pos,
                               nodelist=[end_node],
                               node_color='lightcoral',
                               node_size=3000,
                               ax=ax)

        ax.set_title("Graph Traversal Flow")
        ax.axis('off')

        # Add colorbar for edge weights
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                                   norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Edge Weight', rotation=270, labelpad=15)

        # Add legend
        regular_line = plt.Line2D([0], [0], color='blue', linewidth=2, label='Regular Edge')
        traversal_line = plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Traversal Path')
        start_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15,
                                 label='Start Node')
        end_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=15,
                               label='End Node')
        legend = plt.legend(handles=[regular_line, traversal_line, start_point, end_point], loc='upper left',
                            bbox_to_anchor=(0, 1), ncol=2)
        legend.get_frame().set_alpha(0.8)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_filtered_content(traversal_path, filtered_content):
        """
        Prints the filtered content of visited nodes in the order of traversal.

        Args:
        - traversal_path (list of int): The list of node indices representing the traversal path.
        - filtered_content (dict of int: str): A dictionary mapping node indices to their filtered content.

        Returns:
        - None
        """
        print("\nFiltered content of visited nodes in order of traversal:")
        for i, node in enumerate(traversal_path):
            print(f"\nStep {i + 1} - Node {node}:")
            print(
                f"Filtered Content: {filtered_content.get(node, 'No filtered content available')[:200]}...")  # Print first 200 characters
            print("-" * 50)


# Define the graph RAG class
class GraphRAG:
    def __init__(self, temperature=0.7, model_name="gpt-4o-mini", retrieve_num=5, max_tokens=100):
        """
        Initializes the GraphRAG system with components for document processing, knowledge graph construction,
        querying, and visualization.

        Args:
        - temperature (float): The temperature value for the large language model (default is 0.7).
        - model_name (str): The name of the large language model to use. 
                           OpenAI models: "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", etc.
                           Gemini models: "gemini-pro", "gemini-2.0-flash", "gemini-1.5-pro", etc.
        - retrieve_num (int): Number of documents to retrieve (default is 5).
        - max_tokens (int): The maximum number of tokens for the large language model (default is 100).

        Attributes:
        - llm: An instance of a large language model (LLM) for generating responses.
        - embedding_model: An instance of an embedding model for document embeddings.
        - document_processor: An instance of the DocumentProcessor class for processing documents.
        - knowledge_graph: An instance of the KnowledgeGraph class for building and managing the knowledge graph.
        - query_engine: An instance of the QueryEngine class for handling queries (initialized as None).
        - visualizer: An instance of the Visualizer class for visualizing the knowledge graph traversal.
        """
        self.retrieve_num = retrieve_num
        
        # Auto-detect provider based on model name
        if self._is_gemini_model(model_name):
            self.llm = ChatGoogleGenerativeAI(
                temperature=temperature,
                model=model_name,
                max_tokens=max_tokens
            )
        elif "deepseek" in model_name.lower():
            # Use Ark batch inference (火山引擎) — same as mem0 deepseek path
            self.llm = ChatArkBatch(
                model=os.getenv("BATCH_MODEL") or model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            )
        elif self._is_openai_model(model_name):
            self.llm = ChatOpenAI(
                temperature=temperature,
                model_name=model_name,
                max_tokens=max_tokens
            )
        else:
            # Default to OpenAI for unknown models
            print(f"Warning: Unknown model '{model_name}'. Defaulting to OpenAI provider.")
            self.llm = ChatOpenAI(
                temperature=temperature,
                model_name=model_name,
                max_tokens=max_tokens
            )

        # Attach token-counting callback so every LLM call in any chain is tracked
        self.token_callback = TokenCountingCallback()
        self.llm = self.llm.with_config({"callbacks": [self.token_callback]})

        # Token accumulators (reset per context / per query by process_documents / query)
        self.construction_prompt_tokens: int = 0
        self.construction_completion_tokens: int = 0
        self.query_prompt_tokens: int = 0
        self.query_completion_tokens: int = 0

        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL")
        if dashscope_api_key and dashscope_base_url:
            self.embedding_model = OpenAIEmbeddings(
                model="text-embedding-v4",
                openai_api_key=dashscope_api_key,
                openai_api_base=dashscope_base_url,
                dimensions=1024,
                check_embedding_ctx_length=False,
                chunk_size=10,
            )
        else:
            self.embedding_model = OpenAIEmbeddings()
        self.document_processor = DocumentProcessor()
        self.knowledge_graph = KnowledgeGraph()
        self.query_engine = None
        self.visualizer = Visualizer()
        #self.process_documents(documents)

    def _is_gemini_model(self, model_name: str) -> bool:
        """
        Check if the model name corresponds to a Gemini model.
        
        Args:
        - model_name (str): The model name to check.
        
        Returns:
        - bool: True if it's a Gemini model, False otherwise.
        """
        gemini_patterns = [
            'gemini',
            'models/gemini'  # Full model path format
        ]
        model_name_lower = model_name.lower()
        return any(pattern in model_name_lower for pattern in gemini_patterns)
    
    def _is_openai_model(self, model_name: str) -> bool:
        """
        Check if the model name corresponds to an OpenAI model.
        
        Args:
        - model_name (str): The model name to check.
        
        Returns:
        - bool: True if it's an OpenAI model, False otherwise.
        """
        openai_patterns = [
            'gpt',
            'text-davinci',
            'text-curie',
            'text-babbage',
            'text-ada',
            'o1',  # OpenAI o1 models
            'claude'  # In case someone uses this with Claude through OpenAI-compatible APIs
        ]
        model_name_lower = model_name.lower()
        return any(pattern in model_name_lower for pattern in openai_patterns)

    def process_documents(self, documents):
        """
        Processes a list of documents by splitting them into chunks, embedding them, and building a knowledge graph.

        Args:
        - documents (list of str): A list of documents to be processed.

        Returns:
        - None
        """
        # Reset construction counters so each new context starts clean
        self.construction_prompt_tokens = 0
        self.construction_completion_tokens = 0
        self.token_callback.reset()

        splits, vector_store = self.document_processor.process_documents(documents)
        self.knowledge_graph.build_graph(splits, self.llm, self.embedding_model)

        # Snapshot tokens consumed during graph construction (concept extraction per chunk)
        p, c = self.token_callback.snapshot()
        self.construction_prompt_tokens += p
        self.construction_completion_tokens += c

        self.query_engine = QueryEngine(vector_store, self.knowledge_graph, self.llm, retrieve_num=self.retrieve_num)

    def query(self, query: str):
        """
        Handles a query by retrieving relevant information from the knowledge graph and visualizing the traversal path.

        Args:
        - query (str): The query to be answered.

        Returns:
        - str: The response to the query.
        """
        # Reset per-query so token snapshot reflects only this query's LLM calls
        self.token_callback.reset()

        final_answer, expanded_context, traversal_path, filtered_content = self.query_engine.query(query)

        p, c = self.token_callback.snapshot()
        self.query_prompt_tokens = p
        self.query_completion_tokens = c

        if traversal_path and False:
            self.visualizer.visualize_traversal(self.knowledge_graph.graph, traversal_path)
        else:
            print("No traversal path to visualize.")

        return final_answer, expanded_context


if __name__ == '__main__':
    pass 
    # Load the documents
    #documents = " "

    # Create a graph RAG instance
    #graph_rag = GraphRAG(documents)

    # Process the documents and create the graph
    #graph_rag.process_documents(documents)

    # Input a query and get the retrieved information from the graph RAG
    #response = graph_rag.query(args.query)