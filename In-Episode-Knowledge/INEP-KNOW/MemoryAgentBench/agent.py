import os
import json
import torch
import tiktoken
import httpx
from openai import OpenAI
from utils.templates import get_template
from utils.eval_data_utils import (
    format_chat,
)
import re
import time
import asyncio
import threading

from langchain_core.documents import Document
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig


class AgentWrapper:
    """
    A wrapper class for different types of memory agents including:
    - Long context agents (GPT, Claude, Gemini)
    - Letta agents
    - Mem0 agents  
    - Cognee agents
    - RAG agents (various implementations)
    """

    def __init__(self, agent_config, dataset_config, load_agent_from):
        """
        Initialize the agent wrapper with specified configuration.
        
        Args:
            agent_config: Configuration dictionary for the agent
            dataset_config: Configuration dictionary for the dataset
            load_agent_from: Optional path to load existing agent state from
        """
        # Basic agent configuration
        self.agent_name = agent_config['agent_name']
        self.sub_dataset = dataset_config['sub_dataset']
        self.context_max_length = dataset_config['context_max_length']
        self.dataset = dataset_config['dataset']
        
        # Output and storage configuration
        self.output_dir = agent_config['output_dir']
        self.agent_save_to_folder = load_agent_from
        
        # Context and token limits
        max_tokens = agent_config.get('generation_max_length') or dataset_config['generation_max_length']
        self.input_length_limit = (agent_config['input_length_limit'] -
                                 agent_config['buffer_length'] -
                                 max_tokens)

        # Model configuration
        self.model = agent_config['model']
        self.max_tokens = max_tokens
        self.temperature = agent_config.get('temperature', 0.0)
        
        # Initialize tokenizer (default to gpt-4o-mini for non-gpt models)
        model_for_tokenizer = self.model if "gpt-4o" in self.model else "gpt-4o-mini"
        self.tokenizer = tiktoken.encoding_for_model(model_for_tokenizer)
        
        # Per-agent memorization-token counters (cumulative across run).
        self._memoryos_memorization_input_tokens = 0
        self._memoryos_memorization_output_tokens = 0
        self._amem_memorization_input_tokens = 0
        self._amem_memorization_output_tokens = 0

        # Initialize agent based on type
        self._initialize_agent_by_type(agent_config, dataset_config)

    def _initialize_agent_by_type(self, agent_config, dataset_config):
        """Initialize the specific agent type based on agent name."""
        
        if 'Long_context_agent' in self.agent_name:
            self._initialize_long_context_agent()
        elif self._is_agent_type("letta"):
            self._initialize_letta_agent(agent_config, dataset_config)
        elif self._is_agent_type("mem0"):
            self._initialize_mem0_agent(agent_config, dataset_config)
        elif self._is_agent_type("cognee"):
            self._initialize_cognee_agent(agent_config, dataset_config)
        elif self._is_agent_type("zep"):
            self._initialize_zep_agent(agent_config)
        elif self._is_agent_type("memagent"):
            self._initialize_memagent_agent(agent_config, dataset_config)
        elif self._is_agent_type("memobrain"):
            self._initialize_memobrain_agent(agent_config, dataset_config)
        elif self._is_agent_type("memos"):
            self._initialize_memos_agent(agent_config, dataset_config)
        elif self._is_agent_type("memoryos"):
            self._initialize_memoryos_agent(agent_config, dataset_config)
        elif self._is_agent_type("amem"):
            self._initialize_amem_agent(agent_config, dataset_config)
        elif self._is_agent_type("rag"):
            self._initialize_rag_agent(agent_config, dataset_config)
        else:
            raise NotImplementedError(f"Agent type not supported: {self.agent_name}")

    def _is_agent_type(self, agent_type):
        """Check if the current agent is of a specific type."""
        return agent_type in self.agent_name

    def _create_oai_client(self):
        """Create an OpenAI-compatible client. Uses Azure OpenAI if env vars are set.

        Environment variables for Azure:
          - AZURE_OPENAI_ENDPOINT
          - AZURE_OPENAI_API_VERSION (optional; default provided by SDK or pinned elsewhere)
          - AZURE_OPENAI_API_KEY

        When using Azure, ensure self.model is the deployment name.
        """
        _timeout = httpx.Timeout(1200.0, connect=30.0)
        if "deepseek" in self.model:
            return OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url=os.environ.get("DEEPSEEK_BASE_URL"),
                timeout=_timeout,
                max_retries=20,
            )
        try:
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            if azure_endpoint:
                # Lazy import to avoid requiring Azure class when not used
                from openai import AzureOpenAI
                return AzureOpenAI(
                    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=azure_endpoint,
                    timeout=_timeout,
                    max_retries=20,
                )
        except Exception:
            pass
        return OpenAI(timeout=_timeout, max_retries=20)

    def _is_deepseek(self):
        return "deepseek" in self.model

    def _get_ark_client(self):
        if not hasattr(self, "_ark_client") or self._ark_client is None:
            from volcenginesdkarkruntime import Ark
            self._ark_client = Ark(api_key=os.environ.get("DEEPSEEK_API_KEY"))
        return self._ark_client

    def _chat_complete(self, messages, *, temperature=None, max_tokens=None, fallback_client=None, response_format=None):
        """Route to Ark batch when deepseek model; otherwise use fallback_client (real-time)."""
        if self._is_deepseek():
            kwargs = {"model": os.environ.get("BATCH_MODEL"), "messages": messages}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if response_format is not None:
                kwargs["response_format"] = response_format
            return self._get_ark_client().batch.chat.completions.create(**kwargs)
        client = fallback_client if fallback_client is not None else self._create_oai_client()
        kwargs = {"model": self.model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        return client.chat.completions.create(**kwargs)

    def _create_standard_response(self, output, input_tokens, output_tokens, memory_time, query_time):
        """Create standardized response dictionary."""
        return {
            "output": output,
            "input_len": input_tokens,
            "output_len": output_tokens,
            "memory_construction_time": memory_time,
            "query_time_len": query_time,
        }

    def _initialize_long_context_agent(self):
        """Initialize long context agent with appropriate client."""
        self.context = ''
        
        if "gpt" in self.model or "o4" in self.model:
            self.client = self._create_oai_client()
        elif "claude" in self.model:
            self.client = self._create_oai_client()
        elif "gemini" in self.model:
            self.client = self._create_oai_client()
        elif "deepseek" in self.model:
            self.client = None  # Ark batch client constructed lazily via _chat_complete
        else:
            raise NotImplementedError(f"Model not supported for long context agent: {self.model}")

    def _initialize_letta_agent(self, agent_config, dataset_config):
        """Initialize Letta agent with proper configuration."""
        if "api" not in agent_config['agent_name']:
            from letta import create_client, LLMConfig, EmbeddingConfig, BasicBlockMemory

            self.chunk_size = agent_config['agent_chunk_size']
            self.letta_mode = agent_config['letta_mode']
            
            self.client = create_client()
            self.client.set_default_llm_config(LLMConfig.default_config(agent_config['model']))             
            self.agent_start_time = time.time()
            
            # Configure embedding
            if agent_config['text_embedding'] == 'text-embedding-3-small':
                self.client.set_default_embedding_config(EmbeddingConfig(
                    embedding_model="text-embedding-3-small",
                    embedding_endpoint_type="openai",
                    embedding_endpoint="https://api.openai.com/v1",
                    embedding_dim=1536,
                    embedding_chunk_size=self.chunk_size * 2,
                ))
            else:
                self.client.set_default_embedding_config(
                    EmbeddingConfig.default_config(agent_config['text_embedding'])
                )

            # Load system prompt
            system_path = agent_config['system_path']
            with open(system_path, 'r') as f:
                self.system = f.read()

            # Load or create agent
            if os.path.exists(self.agent_save_to_folder):
                self.load_agent()
            else:
                human_block = self.client.create_block(
                    label='human', 
                    value='User is sharing the contents they are reading recently.', 
                    limit=2000000
                )
                persona_block = self.client.create_block(
                    label='persona', 
                    value='You are a helpful assistant that can help memorize details in the conversation.', 
                    limit=2000000
                )
                memory = BasicBlockMemory(blocks=[human_block, persona_block])
                self.agent_state = self.client.create_agent(
                    name='mm_agent',
                    memory=memory,
                    system=self.system
                )
        ## use the letta api to create the agent
        else:
            from letta_client import Letta, CreateBlock
            
            self.chunk_size = agent_config['agent_chunk_size']
            self.letta_mode = agent_config['letta_mode']
            self.agent_start_time = time.time()
            
            
            self.client = Letta(token=os.environ.get('Letta_API_KEY'))
            self.agent_state = self.client.agents.create(
            memory_blocks=[
                CreateBlock(
                    label="human",
                    limit=2000000,
                    value="User is sharing the contents they are reading recently."
                ),
                CreateBlock(
                    label="persona",
                    limit=2000000,
                    value="You are a helpful assistant that can help memorize details in the conversation."
                )
            ],
            model=f"openai/{agent_config['model']}",
            embedding=f"openai/{agent_config['text_embedding']}"
        )

            
            
    def _initialize_mem0_agent(self, agent_config, dataset_config):
        """Initialize Mem0 agent with retrieval configuration."""
        from mem0.memory.main import Memory

        self.retrieve_num = agent_config['retrieve_num']
        self.context = ''
        self.agent_start_time = time.time()
        self._mem0_memorization_input_tokens = 0
        self._mem0_memorization_output_tokens = 0

        if "deepseek" in self.model:
            from mem0.configs.base import MemoryConfig
            from mem0.llms.configs import LlmConfig
            from mem0.embeddings.configs import EmbedderConfig
            from volcenginesdkarkruntime import Ark

            self.batch_model = os.environ.get("BATCH_MODEL")
            self.client = Ark(api_key=os.environ.get("DEEPSEEK_API_KEY"))
            from mem0.vector_stores.configs import VectorStoreConfig

            mem_config = MemoryConfig(
                llm=LlmConfig(
                    provider="ark",
                    config={"max_tokens": 32768},
                ),
                embedder=EmbedderConfig(
                    provider="openai",
                    config={
                        "model": "text-embedding-v4",
                        "api_key": os.environ.get("DASHSCOPE_API_KEY"),
                        "openai_base_url": os.environ.get("DASHSCOPE_BASE_URL"),
                        "embedding_dims": 1024,
                    },
                ),
                vector_store=VectorStoreConfig(
                    provider="qdrant",
                    config={"embedding_model_dims": 1024, "path": os.environ.get(
                        "QDRANT_PATH",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", ".qdrant"),
                    )},
                ),
                history_db_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "mem0_history.db"),
            )
            self.memory = Memory(config=mem_config)
        else:
            self.client = self._create_oai_client()
            self.memory = Memory()

    def _initialize_cognee_agent(self, agent_config, dataset_config):
        """Initialize Cognee agent with knowledge graph configuration."""
        self.context = ''
        self.chunks = []
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.agent_start_time = time.time()
        self.cognee_dir = './cognee/.cognee_system/databases/cognee.lancedb'
    
    def _initialize_zep_agent(self, agent_config):
        # from zep_cloud.client import AsyncZep
        # self.client = AsyncZep(api_key=os.getenv("ZEP_API_KEY"), base_url="https://api.development.getzep.com/api/v2")
        from zep_cloud import Zep
        from methods.zep import OpenAIAgent
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.context_id = -1

        self.client = Zep(api_key=os.getenv("ZEP_API_KEY"))
        self.oai_client = OpenAIAgent(model=self.model, source="azure", api_dict={"endpoint":os.environ.get("AZURE_OPENAI_ENDPOINT"), "api_version":os.environ.get("AZURE_OPENAI_API_VERSION"), "api_key":os.environ.get("AZURE_OPENAI_API_KEY")}, temperature=self.temperature)
        self.agent_start_time = time.time()

    def _initialize_memagent_agent(self, agent_config, dataset_config):
        """Initialize memagent agent: recurrent memory compression via memagent-14b, answering via deepseek-chat."""
        # Client for memagent-14b (vLLM server) used during memorization
        memagent_url = agent_config.get('memagent_url', 'http://localhost:8000/v1')
        memagent_api_key = agent_config.get('memagent_api_key', 'EMPTY')
        self.memagent_client = OpenAI(
            base_url=memagent_url,
            api_key=memagent_api_key,
            timeout=httpx.Timeout(1200.0, connect=30.0),
        )
        self.memagent_model = agent_config['memagent_model']

        # Client for answer step; deepseek uses Ark batch via _chat_complete
        self.answer_client = None if self._is_deepseek() else self._create_oai_client()

        self.chunk_size = agent_config['agent_chunk_size']
        self.recurrent_max_new = agent_config.get('recurrent_max_new', 1024)

        # Recurrent memory state (shared across all questions for one context)
        self.memagent_memory = "No previous memory"

        # Token counters for memorization phase
        self._memagent_memorization_input_tokens = 0
        self._memagent_memorization_output_tokens = 0

        self.agent_start_time = time.time()

        # Generic prompt used during memorization (replaces per-question prompt from Mem-alpha)
        self._memagent_generic_prompt = (
            "Extract and remember all important information, facts, events, "
            "and details from this section."
        )

        # Template for recurrent memory update (from Mem-alpha long_context_eval.py:102-117)
        self._memagent_memorize_template = (
            "You are presented with a problem, a section of an article that may contain "
            "the answer to the problem, and a previous memory. Please read the provided "
            "section carefully and update the memory with the new information that helps "
            "to answer the problem. Be sure to retain all relevant details from the "
            "previous memory while adding any new, useful information.\n\n"
            "<problem>\n{prompt}\n</problem>\n\n"
            "<memory>\n{memory}\n</memory>\n\n"
            "<section>\n{chunk}\n</section>\n\n"
            "Updated memory:\n"
        )

    def _handle_memagent_agent(self, message, memorizing, query_id, context_id):
        """Handle memagent agent: memorize via recurrent memagent-14b, answer via deepseek-chat."""
        if memorizing:
            # Recurrently update memory using memagent-14b
            formatted_msg = self._memagent_memorize_template.format(
                prompt=self._memagent_generic_prompt,
                memory=self.memagent_memory,
                chunk=message,
            )
            response = self.memagent_client.chat.completions.create(
                model=self.memagent_model,
                messages=[{"role": "user", "content": formatted_msg}],
                temperature=0.1,
                top_p=0.95,
                max_tokens=self.recurrent_max_new,
            )
            self.memagent_memory = response.choices[0].message.content
            self._memagent_memorization_input_tokens += response.usage.prompt_tokens
            self._memagent_memorization_output_tokens += response.usage.completion_tokens
            return "Memorized"
        else:
            # Answer using deepseek-chat with accumulated memory as context
            memory_construction_time = time.time() - self.agent_start_time

            system_message = get_template(self.sub_dataset, 'system', self.agent_name)
            system_with_memory = system_message + "\n\nRelevant memory:\n" + self.memagent_memory

            formatted_message = format_chat(message=message, system_message=system_with_memory)

            start_query = time.time()
            response = self._chat_complete(
                formatted_message,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                fallback_client=self.answer_client,
            )
            query_time_len = time.time() - start_query

            output = self._create_standard_response(
                response.choices[0].message.content,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                memory_construction_time,
                query_time_len,
            )
            output["memorization_input_len"] = self._memagent_memorization_input_tokens
            output["memorization_output_len"] = self._memagent_memorization_output_tokens
            self.agent_start_time = time.time()
            return output

    def _initialize_memos_agent(self, agent_config, dataset_config):
        """Initialize MemOS: in-memory Qdrant + DASHSCOPE embedding + deepseek-chat answering."""
        import uuid
        from memos.configs.memory import MemoryConfigFactory
        from memos.memories.factory import MemoryFactory

        llm_api_key  = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY")
        llm_base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        embed_api_key  = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY")
        embed_base_url = os.environ.get("DASHSCOPE_BASE_URL",
                                        "https://dashscope.aliyuncs.com/compatible-mode/v1")

        self.retrieve_num = agent_config['retrieve_num']
        self._mem0_memorization_input_tokens  = 0
        self._mem0_memorization_output_tokens = 0
        self.agent_start_time = time.time()

        config = MemoryConfigFactory(
            backend="general_text",
            config={
                "extractor_llm": {
                    "backend": "openai",
                    "config": {
                        "model_name_or_path": agent_config.get("memos_model", "deepseek-chat"),
                        "api_key": llm_api_key,
                        "api_base": llm_base_url,
                    },
                },
                "vector_db": {
                    "backend": "qdrant",
                    "config": {
                        "collection_name": f"memos_{uuid.uuid4().hex}",
                        "distance_metric": "cosine",
                        "vector_dimension": int(agent_config.get("vector_dimension", 1024)),
                        "path": ":memory:",
                    },
                },
                "embedder": {
                    "backend": "universal_api",
                    "config": {
                        "provider": "openai",
                        "model_name_or_path": agent_config.get("embed_model", "text-embedding-v4"),
                        "api_key": embed_api_key,
                        "base_url": embed_base_url,
                    },
                },
            },
        )

        self.memos = MemoryFactory.from_config(config)
        self.client = self._create_oai_client()

    def _initialize_memoryos_agent(self, agent_config, dataset_config):
        """Initialize MemoryOS: per-context instance created lazily, DashScope embedding, Ark LLM."""
        self.retrieve_num = agent_config['retrieve_num']
        self._memoryos_short_term_capacity = int(agent_config.get('short_term_capacity', 10))
        self._mem0_memorization_input_tokens  = 0
        self._mem0_memorization_output_tokens = 0
        self.agent_start_time = time.time()
        self.memoryos = None
        self._memoryos_current_context_id = None
        self._memoryos_storage_path = None
        self.client = self._create_oai_client()

    def _initialize_amem_agent(self, agent_config, dataset_config):
        """Initialize A-MEM: per-context instance, DashScope embedding, Ark LLM."""
        self.retrieve_num = agent_config['retrieve_num']
        self._amem_evo_threshold = int(agent_config.get('evo_threshold', 9999))
        self._amem_embed_model   = agent_config.get('embed_model', 'text-embedding-v4')
        self._mem0_memorization_input_tokens  = 0
        self._mem0_memorization_output_tokens = 0
        self.agent_start_time = time.time()
        self.amem = None
        self._amem_current_context_id = None
        self.client = self._create_oai_client()

    def _initialize_memobrain_agent(self, agent_config, dataset_config):
        """Initialize MemoBrain agent: reasoning-graph memory via cloud API, answering via deepseek-chat."""
        from memobrain import MemoBrain

        # API config: agent_config → DEEPSEEK_* env → OPENAI_* env
        memobrain_api_key = (agent_config.get('memobrain_api_key')
                             or os.environ.get("DEEPSEEK_API_KEY")
                             or os.environ.get("OPENAI_API_KEY", "EMPTY"))
        memobrain_url     = (agent_config.get('memobrain_url')
                             or os.environ.get("DEEPSEEK_BASE_URL")
                             or os.environ.get("OPENAI_BASE_URL"))
        self.memobrain_model  = agent_config.get('memobrain_model', 'deepseek-v3-2-251201')
        self._token_budget    = int(agent_config.get('token_budget', 20000))
        self._memobrain_api_key = memobrain_api_key
        self._memobrain_url     = memobrain_url

        self.memobrain = MemoBrain(
            api_key=memobrain_api_key,
            base_url=memobrain_url,
            model_name=self.memobrain_model,
        )
        self.memobrain.init_memory(
            "Extract and remember all important information, facts, events, and details."
        )

        # Monkey-patch _create_completion to count tokens and route through Ark batch.
        # MemoBrain._create_completion is async; we bridge into the synchronous _chat_complete
        # via asyncio.to_thread so the event loop is never blocked.
        agent_ref = self
        async def _tracked_create(messages, stream=False):
            resp = await asyncio.to_thread(agent_ref._chat_complete, messages)
            if hasattr(resp, 'usage') and resp.usage:
                agent_ref._memobrain_memorization_input_tokens  += getattr(resp.usage, 'prompt_tokens',     0) or 0
                agent_ref._memobrain_memorization_output_tokens += getattr(resp.usage, 'completion_tokens', 0) or 0
            return resp
        self.memobrain._create_completion = _tracked_create

        # Daemon thread with persistent event loop (avoids asyncio.run() multi-call conflict)
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

        # Client for deepseek-chat (answering)
        self.answer_client = self._create_oai_client()

        self.chunk_size = agent_config['agent_chunk_size']

        # Recall cache: populated during memorize phase or on first query
        self._memobrain_recall_cache = None

        # Token counters (memorization phase only)
        self._memobrain_memorization_input_tokens  = 0
        self._memobrain_memorization_output_tokens = 0

        self.agent_start_time = time.time()

    def _memobrain_run_async(self, coro, timeout=120):
        """Bridge async MemoBrain calls into synchronous benchmark code."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    def _format_memobrain_recall(self, organized):
        """Format organized recall messages into a plain-text memory string."""
        parts = []
        for msg in organized:
            content = msg.get("content") or ""
            role    = msg.get("role", "")
            if content:
                prefix = "[Content]" if role == "user" else "[Summary]"
                parts.append(f"{prefix}: {content}")
        return "\n".join(parts)

    def _handle_memobrain_agent(self, message, memorizing, query_id, context_id):
        """Handle MemoBrain agent: memorize via MemoBrain model, answer via deepseek-chat."""
        if memorizing:
            # Convert raw chunk to user/assistant pair for MemoBrain's memorize()
            pair = [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": "I've processed and remembered this information."},
            ]
            try:
                self._memobrain_run_async(self.memobrain.memorize(pair))
            except Exception as e:
                print(f"[MemoBrain] memorize failed for chunk, skipping: {e}")

            # Progressive recall: compress if total chars exceed token budget
            total_chars = sum(len(m.get("content") or "") for m in self.memobrain.messages)
            if total_chars // 4 > self._token_budget:
                try:
                    organized = self._memobrain_run_async(self.memobrain.recall())
                    self._memobrain_recall_cache = self._format_memobrain_recall(organized)
                except Exception as e:
                    print(f"[MemoBrain] progressive recall failed: {e}")
            return "Memorized"
        else:
            memory_construction_time = time.time() - self.agent_start_time

            # If recall() was never triggered during memorization (context under budget),
            # use the raw accumulated messages directly — no compression needed.
            if self._memobrain_recall_cache is None:
                self._memobrain_recall_cache = self._format_memobrain_recall(
                    self.memobrain.messages
                )

            system_message     = get_template(self.sub_dataset, 'system', self.agent_name)
            system_with_memory = system_message + "\n\nRelevant memory:\n" + self._memobrain_recall_cache

            formatted_message = format_chat(message=message, system_message=system_with_memory)

            start_query = time.time()
            response = self._chat_complete(
                formatted_message,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                fallback_client=self.answer_client,
            )
            query_time_len = time.time() - start_query

            output = self._create_standard_response(
                response.choices[0].message.content,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                memory_construction_time,
                query_time_len,
            )
            output["memorization_input_len"]  = self._memobrain_memorization_input_tokens
            output["memorization_output_len"] = self._memobrain_memorization_output_tokens
            self.agent_start_time = time.time()
            return output

    def _handle_memos_agent(self, message, memorizing, query_id, context_id):
        """Handle MemOS: direct-embed chunks during memorize, semantic search + generate at query.

        Note: MemOS general_text memory's add() is embedding-only (no LLM call); the
        extractor_llm wired in _initialize_memos_agent is dead code in this harness
        because we pre-build TextualMemoryItem dicts and bypass extract(). Therefore
        memorization_input/output_len are genuinely 0 for MemOS — not a tracking gap.
        """
        if memorizing:
            try:
                self.memos.add([{
                    "memory": message,
                    "metadata": {"source": "conversation"},
                }])
            except Exception as e:
                print(f"[MemOS] memorize chunk failed, skipping: {e}")
            return "Memorized"
        else:
            memory_construction_time = time.time() - self.agent_start_time

            try:
                retrieval_query = self._extract_retrieval_query(message)
                search_results  = self.memos.search(retrieval_query, top_k=self.retrieve_num)
                memories_str    = "\n".join(f"- {r.memory}" for r in search_results)
            except Exception as e:
                print(f"[MemOS] search failed: {e}")
                memories_str = ""

            system_prompt = (get_template(self.sub_dataset, 'system', self.agent_name)
                             + "\n\nRelevant context:\n" + memories_str)
            llm_messages = format_chat(message=message, system_message=system_prompt)

            response = self._chat_complete(
                llm_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                fallback_client=self.client,
            )
            query_time_len = time.time() - self.agent_start_time - memory_construction_time

            output = self._create_standard_response(
                response.choices[0].message.content,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                memory_construction_time,
                query_time_len,
            )
            output["memorization_input_len"]  = 0
            output["memorization_output_len"] = 0
            self.agent_start_time = time.time()
            return output

    def _install_memoryos_token_tracker(self):
        """Wrap MemoryOS's three internal OpenAIClient instances to count memorization tokens.

        MemoryOS creates a fresh openai.OpenAI inside _create_client() on every call for
        thread safety, so we cannot wrap a persistent .client attribute. Instead we replace
        chat_completion on each of the three long-lived OpenAIClient holders — memoryos.client,
        mid_term_memory.client, and updater.client — with a re-implementation that captures
        response.usage before discarding it. retriever.retrieve_context is embedding-only
        and is not touched.

        All LLM calls are routed through self._chat_complete (Ark batch for DeepSeek).
        """
        from openai import OpenAI as _OpenAI
        try:
            from memoryos.utils import clean_reasoning_model_output
        except ImportError:
            import re as _re2
            def clean_reasoning_model_output(text):
                return _re2.sub(r'<think>.*?</think>', '', text, flags=_re2.DOTALL).strip()
        parent = self

        def make_tracked(orig_client):
            fallback = _OpenAI(api_key=orig_client.api_key, base_url=orig_client.base_url)
            def tracked_chat_completion(model, messages, temperature=0.7, max_tokens=2000):
                try:
                    response = parent._chat_complete(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        fallback_client=fallback,
                    )
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        parent._memoryos_memorization_input_tokens  += getattr(usage, "prompt_tokens", 0) or 0
                        parent._memoryos_memorization_output_tokens += getattr(usage, "completion_tokens", 0) or 0
                    raw = response.choices[0].message.content.strip()
                    return clean_reasoning_model_output(raw)
                except Exception as e:
                    print(f"[MemoryOS] tracked chat_completion error: {e}")
                    return "Error: Could not get response from LLM."
            return tracked_chat_completion

        seen = set()
        for attr_path in ["client", "mid_term_memory.client", "updater.client"]:
            obj = self.memoryos
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                if id(obj) in seen:
                    continue
                seen.add(id(obj))
                obj.chat_completion = make_tracked(obj)
            except AttributeError:
                print(f"[MemoryOS] tracker: missing attr {attr_path}")
            except Exception as e:
                print(f"[MemoryOS] tracker install failed on {attr_path}: {e}")

    def _install_amem_token_tracker(self):
        """Wrap A-MEM's single LLM chokepoint to count memorization tokens.

        A-MEM's OpenAIController.get_completion calls self.client.chat.completions.create
        then discards response.usage. We replace get_completion with a re-implementation
        that captures usage before returning .content. This covers both the evolution prompt
        in process_memory (every add_note) and analyze_content (if ever invoked).

        All LLM calls are routed through self._chat_complete (Ark batch for DeepSeek).
        """
        import re as _re3

        def _strip_markdown_json(text):
            text = text.strip()
            text = _re3.sub(r'^```(?:json)?\s*', '', text)
            text = _re3.sub(r'\s*```$', '', text)
            return text.strip()

        llm = self.amem.llm_controller.llm
        _client = getattr(llm, '_client', None) or getattr(llm, 'client', None)
        parent = self

        def tracked_get_completion(prompt, response_format=None, temperature=0.7):
            try:
                messages = [
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt},
                ]
                resp = parent._chat_complete(
                    messages,
                    temperature=temperature,
                    max_tokens=4096,
                    response_format=response_format,
                    fallback_client=_client,
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    parent._amem_memorization_input_tokens  += getattr(usage, "prompt_tokens", 0) or 0
                    parent._amem_memorization_output_tokens += getattr(usage, "completion_tokens", 0) or 0
                return _strip_markdown_json(resp.choices[0].message.content)
            except Exception as e:
                print(f"[A-MEM] tracked get_completion failed: {e}")
                raise

        llm.get_completion = tracked_get_completion

    def _handle_memoryos_agent(self, message, memorizing, query_id, context_id):
        """Handle MemoryOS: add_memory for chunks, retriever.retrieve_context + own LLM for queries."""
        import uuid, shutil

        if context_id != self._memoryos_current_context_id:
            if self._memoryos_storage_path and os.path.exists(self._memoryos_storage_path):
                shutil.rmtree(self._memoryos_storage_path, ignore_errors=True)
            self._memoryos_storage_path = f"./agents/memoryos_{uuid.uuid4().hex}"
            self._memoryos_current_context_id = context_id

            from memoryos import Memoryos
            self.memoryos = Memoryos(
                user_id=f"context_{context_id}",
                openai_api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                openai_base_url=os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
                data_storage_path=self._memoryos_storage_path,
                llm_model=self.model,
                embedding_model_name="text-embedding-v4",
                short_term_capacity=self._memoryos_short_term_capacity,
            )
            self._install_memoryos_token_tracker()
            self.agent_start_time = time.time()

        if memorizing:
            try:
                self.memoryos.add_memory(user_input=message, agent_response="Stored.")
            except Exception as e:
                print(f"[MemoryOS] memorize chunk failed, skipping: {e}")
            return "Memorized"
        else:
            memory_construction_time = time.time() - self.agent_start_time
            memories_str = ""
            try:
                retrieval_query = self._extract_retrieval_query(message)
                results = self.memoryos.retriever.retrieve_context(
                    user_query=retrieval_query,
                    user_id=self.memoryos.user_id,
                    top_k_sessions=self.retrieve_num,
                    top_k_knowledge=self.retrieve_num,
                )
                parts = []
                for page in results.get("retrieved_pages", []):
                    text = page.get("user_input", "") if isinstance(page, dict) else str(page)
                    if text:
                        parts.append(f"- {text}")
                for item in results.get("retrieved_user_knowledge", []):
                    text = item.get("knowledge", item.get("content", str(item))) if isinstance(item, dict) else str(item)
                    if text:
                        parts.append(f"- {text}")
                memories_str = "\n".join(parts)
            except Exception as e:
                print(f"[MemoryOS] retrieve failed: {e}")

            system_prompt = (get_template(self.sub_dataset, 'system', self.agent_name)
                             + "\n\nRelevant context:\n" + memories_str)
            llm_messages = format_chat(message=message, system_message=system_prompt)

            response = self._chat_complete(
                llm_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                fallback_client=self.client,
            )
            query_time_len = time.time() - self.agent_start_time - memory_construction_time

            output = self._create_standard_response(
                response.choices[0].message.content,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                memory_construction_time,
                query_time_len,
            )
            output["memorization_input_len"]  = self._memoryos_memorization_input_tokens
            output["memorization_output_len"] = self._memoryos_memorization_output_tokens
            self.agent_start_time = time.time()
            return output

    def _handle_amem_agent(self, message, memorizing, query_id, context_id):
        """Handle A-MEM: add_note per chunk, search_agentic + own LLM for queries."""
        if context_id != self._amem_current_context_id:
            self._amem_current_context_id = context_id

            from agentic_memory.memory_system import AgenticMemorySystem
            from openai import OpenAI as _OpenAI

            embed_api_key  = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY")
            embed_base_url = os.environ.get("DASHSCOPE_BASE_URL",
                                            "https://dashscope.aliyuncs.com/compatible-mode/v1")
            _embed_client = _OpenAI(api_key=embed_api_key, base_url=embed_base_url,
                                    timeout=httpx.Timeout(120.0, connect=30.0))
            _embed_model  = self._amem_embed_model

            class _DashScopeEmbFn:
                def name(self):
                    return "dashscope_text_embedding_v4"
                def __call__(self, input):
                    resp = _embed_client.embeddings.create(model=_embed_model, input=input)
                    return [e.embedding for e in resp.data]
                def embed_query(self, input):
                    # ChromaDB 1.5.8 calls embed_query() during collection.query()
                    return self.__call__(input)
                def get_config(self):
                    return {"model": _embed_model}
                @staticmethod
                def build_from_config(config):
                    raise NotImplementedError("rebuild not supported for in-process use")

            self.amem = AgenticMemorySystem(
                llm_backend="openai",
                llm_model=self.model,
                api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
                evo_threshold=self._amem_evo_threshold,
                embedding_function=_DashScopeEmbFn(),
            )
            self._install_amem_token_tracker()
            self.agent_start_time = time.time()

        if memorizing:
            try:
                self.amem.add_note(content=message)
            except Exception as e:
                print(f"[A-MEM] memorize chunk failed, skipping: {e}")
            return "Memorized"
        else:
            memory_construction_time = time.time() - self.agent_start_time
            memories_str = ""
            try:
                retrieval_query = self._extract_retrieval_query(message)
                results = self.amem.search_agentic(retrieval_query, k=self.retrieve_num)
                memories_str = "\n".join(f"- {r['content']}" for r in results)
            except Exception as e:
                print(f"[A-MEM] search failed: {e}")

            system_prompt = (get_template(self.sub_dataset, 'system', self.agent_name)
                             + "\n\nRelevant context:\n" + memories_str)
            llm_messages = format_chat(message=message, system_message=system_prompt)

            response = self._chat_complete(
                llm_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                fallback_client=self.client,
            )
            query_time_len = time.time() - self.agent_start_time - memory_construction_time

            output = self._create_standard_response(
                response.choices[0].message.content,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                memory_construction_time,
                query_time_len,
            )
            output["memorization_input_len"]  = self._amem_memorization_input_tokens
            output["memorization_output_len"] = self._amem_memorization_output_tokens
            self.agent_start_time = time.time()
            return output

    def _initialize_rag_agent(self, agent_config, dataset_config):
        """Initialize RAG agent with retrieval configuration."""
        self.context = ''
        self.chunks = []
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = dataset_config['chunk_size']
        self.context_len = 0
        self.context_id = -1

    def send_message(self, message, memorizing=False, query_id=None, context_id=None):
        """
        Send a message to the agent for either memorization or querying.
        
        Args:
            message: The message content (context for memorization, query for answering)
            memorizing: Whether to memorize the message (True) or answer it (False)
            query_id: Unique identifier for the query
            context_id: Unique identifier for the context
            
        Returns:
            dict or str: Agent response with metadata (for queries) or confirmation (for memorization)
        """
        # Route to appropriate agent handler based on agent type
        if 'Long_context_agent' in self.agent_name:
            return self._handle_long_context_agent(message, memorizing)
        elif any(self._is_agent_type(agent_type) for agent_type in ["letta", "cognee", "mem0", "zep"]):
            return self._handle_memory_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("memagent"):
            return self._handle_memagent_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("memobrain"):
            return self._handle_memobrain_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("memos"):
            return self._handle_memos_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("memoryos"):
            return self._handle_memoryos_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("amem"):
            return self._handle_amem_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("rag"):
            return self._handle_rag_agent(message, memorizing, query_id, context_id)
        else:
            raise NotImplementedError(f"Agent type not supported: {self.agent_name}")

    def _handle_long_context_agent(self, message, memorizing):
        """Handle message processing for long context agents."""
        if memorizing:
            # Add message to context memory
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            return "Memorized"
        else:
            # Process query with context
            return self._query_long_context_agent(message)

    def _query_long_context_agent(self, message):
        """Process a query for long context agents."""
        # Get appropriate tokenizer
        try:
            tokenizer = tiktoken.encoding_for_model(self.model)
        except:
            tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        
        # Handle context truncation for non-long context models
        buffer_length = 50000
        if self.input_length_limit <= self.context_max_length + buffer_length:
            self._truncate_context_if_needed(tokenizer)
                
        # Format message with context and system prompt
        full_message = self.context + "\n" + message
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        formatted_message = format_chat(message=full_message, system_message=system_message)
        
        # Query the model
        start_time = time.time()
        
        if "o4" in self.model or "gpt-5" in self.model:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_message,
            )
            return self._format_openai_response(response, start_time)

        elif "deepseek" in self.model:
            response = self._chat_complete(
                formatted_message,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return self._format_openai_response(response, start_time)

        elif "gpt" in self.model:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_message,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return self._format_openai_response(response, start_time)
            
        elif "claude" in self.model:
            return self._query_claude(full_message, system_message, start_time)
            
        elif "gemini" in self.model:
            return self._query_gemini(formatted_message, start_time)
            
        else:
            raise NotImplementedError(f"Model not supported: {self.model}")

    def _truncate_context_if_needed(self, tokenizer):
        """Truncate context if it exceeds limits."""
        # Truncate context if it exceeds the context_max_length
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.context_max_length:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.context_max_length:])
        
        # Truncate if context exceeds the input_length_limit
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.input_length_limit:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.input_length_limit:])

    def _format_openai_response(self, response, start_time):
        """Format OpenAI API response into standard output format."""
        msg = response.choices[0].message
        content = msg.content or msg.refusal or ""
        return self._create_standard_response(
            content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            0,
            time.time() - start_time
        )

    def _query_claude(self, message, system_message, start_time):
        """Query Claude model via OpenAI-compatible proxy."""
        formatted_message = format_chat(message=message, system_message=system_message)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return self._format_openai_response(response, start_time)

    def _query_gemini(self, formatted_message, start_time):
        """Query Gemini model via OpenAI-compatible proxy."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return self._format_openai_response(response, start_time)
        
    def _handle_memory_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for memory-based agents (Letta, Cognee, Mem0)."""
        if self._is_agent_type("letta"):
            return self._handle_letta_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("cognee"):
            return self._handle_cognee_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("mem0"):
            return self._handle_mem0_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("zep"):
            return self._handle_zep_agent(message, memorizing, query_id, context_id)
        else:
            raise NotImplementedError(f"Memory agent type not supported: {self.agent_name}")

    def _handle_letta_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Letta agents."""
        # Format message based on context
        if memorizing:
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
        else:
            formatted_message = message
        
        # Handle memory construction time for queries
        memory_construction_time = 0 if memorizing else time.time() - self.agent_start_time
        
        # Reload agent for queries
        if not memorizing:
            if os.path.exists(self.agent_save_to_folder):
                self.load_agent()
            else:
                print(f"\n\nAgent {self.agent_name} not found in {self.agent_save_to_folder}\n\n")
        
        # Process based on Letta mode
        response = self._process_letta_message(formatted_message, memorizing, query_id, context_id)
        
        if memorizing:
            return "Memorized"
        
        # Create response for queries
        tokenizer = self.tokenizer
        query_time_len = time.time() - self.agent_start_time - memory_construction_time
        output = self._create_standard_response(
            response,
            len(tokenizer.encode(message, disallowed_special=())),
            len(tokenizer.encode(response, disallowed_special=())),
            memory_construction_time,
            query_time_len
        )
        self.agent_start_time = time.time()  # Reset time
        return output
    
    def _process_letta_message(self, formatted_message, memorizing, query_id, context_id):
        """Process message with Letta client based on mode."""
        from letta_client import Letta, MessageCreate
        
        try:
            if self.letta_mode == 'insert':
                if memorizing:
                    self.client.server.passage_manager.insert_passage(
                        agent_state=self.agent_state,
                        agent_id=self.agent_state.id,
                        text=formatted_message,
                        actor=self.client.user,
                    )
                    # import ipdb; ipdb.set_trace()
                    return "Memorized"
                else:
                    response = self.client.send_message(
                        agent_id=self.agent_state.id,
                        message=formatted_message,
                        role='user')
                    ## save response.messages to a file / for debugging as JSON     
                    return json.loads(response.messages[-3].tool_call.arguments)['message']
            
            elif self.letta_mode == 'chat':
                response = self.client.send_message(
                    agent_id=self.agent_state.id,
                    message=formatted_message,
                    role='user')
                
                if memorizing:
                    return "Memorized"
                else:
                    ## save response.messages to a file / for debugging as JSON    
                    return json.loads(response.messages[-3].tool_call.arguments)['message']
            elif self.letta_mode == 'api':
                response = self.client.agents.messages.create(
                    agent_id=self.agent_state.id,
                    messages=[
                        MessageCreate(
                            role="user",
                            content=formatted_message,
                        ),
                    ],
                )
                print(f"\n\n\nresponse: {response}\n\n\n")
                return response.messages[-1].content
        except Exception as e:
            print(f"\n\n\nerror: {e}\n\n\n")
            return "Error"

    def _handle_cognee_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Cognee agents."""
        import cognee
        import asyncio
        dataset_name = f'default_dataset_{self.sub_dataset}_context_{context_id}'
        
        if memorizing:
            # Add context to Cognee knowledge base
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            
            # Add text to cognee and generate knowledge graph
            asyncio.run(cognee.add(formatted_message, dataset_name=dataset_name))
            asyncio.run(cognee.cognify(datasets=[dataset_name], chunk_size=self.chunk_size))

            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            return "Memorized"
        else:                    
            # Query the knowledge graph
            memory_construction_time = time.time() - self.agent_start_time
            searched_results = asyncio.run(cognee.search(
                query_text=message, 
                top_k=self.retrieve_num, 
                datasets=[dataset_name]
            ))
                    
            # Format results
            total_results = ("".join([f"{result}\n" for result in searched_results]) 
                           if searched_results else "No results found.")
            
            # Return formatted output
            tokenizer = self.tokenizer
            query_time_len = time.time() - self.agent_start_time - memory_construction_time
            output = self._create_standard_response(
                total_results,
                len(tokenizer.encode(self.context, disallowed_special=())),
                len(tokenizer.encode(total_results, disallowed_special=())),
                memory_construction_time,
                query_time_len
            )
            self.agent_start_time = time.time()  # Reset time
            return output

    def _handle_mem0_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Mem0 agents."""
        user_id = f'context_{context_id}_{self.sub_dataset}'
        if memorizing:
            system_message = get_template(self.sub_dataset, 'system', self.agent_name)
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            
            # Generate Assistant response
            # memory_messages = [{"role": "system", "content": system_message}, {"role": "user", "content": formatted_message}]
            # response = OpenAI().chat.completions.create(
            #             model=self.model,
            #             messages=memory_messages,
            #             max_tokens=1000,
            #         )
            # memory_messages = [
            #     {"role": "system", "content": system_message}, 
            #     {"role": "user", "content": formatted_message},
            #     {"role": "assistant", "content": response.choices[0].message.content}
            # ]
            memory_messages = [
                {"role": "system", "content": system_message}, 
                {"role": "user", "content": formatted_message},
                {"role": "assistant", "content": "I'll make sure to add the content into the memory."}
            ]
            
            vector_results = self.memory.add(memory_messages, user_id=user_id)
            print(f"\n\n\nvector_results: {vector_results}\n\n\n")
            mem_tokens = self.memory.get_and_reset_memorization_tokens()
            self._mem0_memorization_input_tokens += mem_tokens["prompt_tokens"]
            self._mem0_memorization_output_tokens += mem_tokens["completion_tokens"]
            return "Memorized"
        else:
            # Retrieve relevant memories and generate response
            memory_construction_time = time.time() - self.agent_start_time
            relevant_memories = self.memory.search(query=message, filters={"user_id": user_id}, top_k=self.retrieve_num)
            print(f"\n\n\nrelevant_memories: {relevant_memories}\n\n\n")

            memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])

            # Truncate memories if they exceed token budget
            system_prefix = "You are a helpful AI. Answer the question based on query and memories.\n"
            user_content = message + "\n\nCurrent Time: " + time.strftime("%Y-%m-%d %H:%M:%S")
            non_memory_tokens = len(self.tokenizer.encode(system_prefix + user_content, disallowed_special=())) + self.max_tokens
            memory_budget = self.input_length_limit - non_memory_tokens
            encoded_memories = self.tokenizer.encode(memories_str, disallowed_special=())
            if len(encoded_memories) > memory_budget:
                memories_str = self.tokenizer.decode(encoded_memories[:memory_budget])
                print(f"\nMemories truncated: {len(encoded_memories)} -> {memory_budget} tokens\n")

            # Generate assistant response
            system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\n{memories_str}\n"
            llm_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message + "\n\nCurrent Time: " + time.strftime("%Y-%m-%d %H:%M:%S")}
            ]
            response = self.client.batch.chat.completions.create(
                model=self.batch_model,
                messages=llm_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            memory_retrieval_length = len(self.tokenizer.encode(memories_str, disallowed_special=()))
            query_time_len = time.time() - self.agent_start_time - memory_construction_time
            print(f"\nmemory_length: {memory_retrieval_length}\n")

            output = self._create_standard_response(
                response.choices[0].message.content,
                response.usage.prompt_tokens + memory_retrieval_length,
                response.usage.completion_tokens,
                memory_construction_time,
                query_time_len
            )
            output["memorization_input_len"] = self._mem0_memorization_input_tokens
            output["memorization_output_len"] = self._mem0_memorization_output_tokens
            self.agent_start_time = time.time()  # Reset time
            return output
    
    # Zep
    def _handle_zep_agent(self, message, memorizing, query_id, context_id):
        """Handle Zep processing."""
        import inspect
        from zep_cloud import Message
        from methods.zep import compose_search_context, llm_response, get_retrieval_query, construct_messages
        
        # user id / session id / oai client
        user_id = f'user_{context_id}_{self.sub_dataset}'
        graph_id = f'graph_{context_id}_{self.sub_dataset}'
        thread_id = f'thread_{context_id}_{self.sub_dataset}'
                
        # check the context id for user and session creation
        if self.context_id != context_id and memorizing:
            # User creation
            self.client.user.add(user_id=user_id)
            
            # Thread creation
            self.client.thread.create(thread_id=thread_id, user_id=user_id)
                    
            # Graph creation
            self.client.graph.create(graph_id=graph_id)
            self.context_id = context_id
        else:
            pass
            
        if memorizing:
            # graph add
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            content = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            self.client.graph.add(
                graph_id=graph_id, 
                type="text",
                data=content[:9998]
            )

            # # thread add
            messages = construct_messages(content, user_id)
            self.client.thread.add_messages(thread_id=thread_id, messages=messages)
            return "Memorized"
        else:
            memory_construction_time = time.time() - self.agent_start_time
            
            # graph search
            retrieval_query = get_retrieval_query(message)
            print(f"\n\n\nretrieval_query: {retrieval_query}\n\n\n")

            edges_results = self.client.graph.search(graph_id=graph_id, query=retrieval_query[:399], scope='edges', limit=self.retrieve_num).edges
            node_results = self.client.graph.search(graph_id=graph_id, query=retrieval_query[:399], scope='nodes', limit=self.retrieve_num).nodes
            episode_results = self.client.graph.search(graph_id=graph_id, query=retrieval_query[:399], scope='episodes', limit=self.retrieve_num).episodes
            
            # print(f"\n\n\nepisode_results: {episode_results}\n\n\n")
            # print(f"\n\n\nedges_results: {edges_results}\n\n\n")
            # print(f"\n\n\nnode_results: {node_results}\n\n\n")
                        
            # thread search / currently we do not use the thread info
            memory = self.client.thread.get_user_context(thread_id=thread_id)
            context_block = memory.context

            # Prompt an LLM with relevant context
            retrieved_context = compose_search_context(edges_results, node_results, context_block, episode_results)
            import asyncio
            response = asyncio.run(llm_response(self.oai_client, retrieved_context, message))
            query_time_len = time.time() - self.agent_start_time - memory_construction_time

            output = self._create_standard_response(
                response,
                len(self.tokenizer.encode(retrieved_context, disallowed_special=())),
                len(self.tokenizer.encode(response, disallowed_special=())),
                memory_construction_time,
                query_time_len
            )
            self.agent_start_time = time.time()  # Reset time
            
            # save the context
            save_dir = f"./outputs/rag_retrieved/{self.agent_name}/k_{self.retrieve_num}/{self.sub_dataset}/chunksize_{self.chunk_size}/query_{query_id}_context_{context_id}.json"
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            with open(save_dir, "w") as f:
                paragraphs = [p for p in retrieved_context.replace("\r\n", "\n").split("\n") if p.strip()]
                json.dump({"retrieved_context_paragraphs": paragraphs, "response": response}, f, ensure_ascii=False, indent=2)
            
            return output
    
    def _handle_rag_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for RAG agents."""
        if memorizing:
            # Add message to chunks and context
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message, **({'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S")} if '{time_stamp}' in memorize_template else {}))
            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            self.chunks.append(formatted_message)
            self.context_len = self.context_len + self.chunk_size
            
            # Truncate context if it exceeds limits
            if self.context_len > self.input_length_limit:
                self.chunks = self.chunks[1:]
                self.context_len = self.context_len - self.chunk_size
            return ''
        else:
            # Handle query processing for different RAG types
            return self._process_rag_query(message, query_id, context_id)

    def _process_rag_query(self, message, query_id, context_id):
        """Process query for RAG agents with different retrieval strategies."""
                
        # Truncate context if needed
        tokenizer = self.tokenizer
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.input_length_limit:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.input_length_limit:])
        if self.context_len > self.input_length_limit:
            self.chunks = self.chunks[1:]
            self.context_len = self.context_len - self.chunk_size
        
        # Route to specific RAG implementation and get result
        rag_handlers = {
            "graph_rag": lambda: self._handle_graph_rag(message, context_id, tokenizer),
            "hippo_rag_v2_nv": lambda: self._handle_hippo_rag(message, context_id, tokenizer),
            "hippo_rag_v2_openai": lambda: self._handle_hippo_rag(message, context_id, tokenizer),
            "rag_bm25": lambda: self._handle_bm25_rag(message, context_id, tokenizer),
            "rag_contriever": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_text_embedding_3_large": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_text_embedding_3_small": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_qwen3_embedding_4b": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_text_embedding_v4": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_raptor": lambda: self._handle_raptor_rag(message, context_id, tokenizer),
            "self_rag": lambda: self._handle_self_rag(message, context_id, tokenizer),
            "memo_rag": lambda: self._handle_memorag(message, context_id, tokenizer),
        }
        
        # Find matching handler
        handler = next((handler for agent_type, handler in rag_handlers.items() if self._is_agent_type(agent_type)), None)
        if not handler:
            raise NotImplementedError(f"RAG agent type not supported: {self.agent_name}")
        
        output = handler()

        # Save the retrieved context as JSON (if the method provides it)
        if output.get("retrieval_context"):
            save_dir = f"./outputs/rag_retrieved/{self.agent_name}/k_{self.retrieve_num}/{self.sub_dataset}/chunksize_{self.chunk_size}/query_{query_id}_context_{context_id}.json"
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            with open(save_dir, "w") as f:
                json.dump(output["retrieval_context"], f)
            
            # drop the retrieval_context       
            output.pop("retrieval_context")
        
        return output

    def _handle_graph_rag(self, message, context_id, tokenizer):
        """Handle Graph RAG processing."""
        start_time = time.time()

        # Build vectorstore if context changed
        memory_construction_time = 0
        if self.context_id != context_id:
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            try:
                from methods.graph_rag import GraphRAG
                self.graph_rag = GraphRAG(temperature=self.temperature, model_name=self.model, retrieve_num=self.retrieve_num, max_tokens=self.max_tokens)
                self.graph_rag.process_documents(docs)
                memory_construction_time = time.time() - start_time
            except Exception as e:
                import traceback
                print(f"\n\n\n\nError: {e}\n\n\n\n")
                traceback.print_exc()
            print(f"\n\nGraph RAG build vectorstore finished...\n\n")
            print(f"\n\nContext {context_id} already processed, skipping Graph RAG build vectorstore...\n\n")

        # Process query
        if not hasattr(self, 'graph_rag') or self.graph_rag is None or self.graph_rag.query_engine is None:
            response = "GraphRAG initialization failed"
            retrieval_context = "ERROR"
        else:
            try:
                response, retrieval_context = self.graph_rag.query(query=message)
            except Exception as e:
                response = f"{e}"
                retrieval_context = "ERROR"
                print(f"\n\n\n\nError: {e}\n\n\n\n")
        
        self.context_id = context_id
        
        print(f"\n\n\n\nResponse: {response}\n\n\n\n")
        if isinstance(response, str):
            response = response
        else:
            response = response.content
        query_time_len = time.time() - start_time - memory_construction_time

        # Read token counts from GraphRAG's attached callback (covers ALL LLM calls)
        graph_rag_instance = getattr(self, "graph_rag", None)
        query_input_tokens = getattr(graph_rag_instance, "query_prompt_tokens", 0) if graph_rag_instance else 0
        query_output_tokens = getattr(graph_rag_instance, "query_completion_tokens", 0) if graph_rag_instance else 0
        memorization_input_tokens = getattr(graph_rag_instance, "construction_prompt_tokens", 0) if graph_rag_instance else 0
        memorization_output_tokens = getattr(graph_rag_instance, "construction_completion_tokens", 0) if graph_rag_instance else 0

        return {
            "output": response,
            "input_len": query_input_tokens,
            "output_len": query_output_tokens,
            "memorization_input_len": memorization_input_tokens,
            "memorization_output_len": memorization_output_tokens,
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    def _handle_hippo_rag(self, message, context_id, tokenizer):
        """Handle HippoRAG processing."""
        start_time = time.time()
        
        if self.context_id != context_id:
            docs = self.chunks
            from methods.hipporag import HippoRAG
            if any(agent_name in self.agent_name for agent_name in ["hippo_rag_v2_nv"]):
                save_dir = os.path.join(f"./outputs/rag_retrieved/NV-Embed-v2", self.sub_dataset, f'chunksize_{self.chunk_size}', f'context_id_{context_id}')
                embedding_model_name = 'nvidia/NV-Embed-v2'
            elif any(agent_name in self.agent_name for agent_name in ["hippo_rag_v2_openai"]):
                save_dir = os.path.join(f"./outputs/rag_retrieved/OpenAIEmbedding", self.sub_dataset, f'chunksize_{self.chunk_size}', f'context_id_{context_id}') 
                embedding_model_name = 'text-embedding-ada-002'
            
            self.hipporag = HippoRAG(save_dir=save_dir,
                                llm_model_name=self.model,
                                embedding_model_name=embedding_model_name) 
            self.hipporag.index(docs=docs)
            memory_construction_time = time.time() - start_time
            print(f"\n\nHippoRAG build vectorstore finished...\n\n")
        else:
            memory_construction_time = 0
            print(f"\n\nContext {context_id} already processed, skipping HippoRAG build vectorstore...\n\n")
            
        # Retrieve and answer
        queries = [message]
        retrieval_results, top_k_docs = self.hipporag.retrieve(queries=queries, num_to_retrieve=self.retrieve_num)
        
        qa_results = self.hipporag.rag_qa(retrieval_results)
        response = qa_results[0][0].answer
        
        retrieval_context = "\n\n".join([f"Passage {i+1}:\n{text}" for i, text in enumerate(top_k_docs)])
        query_time_len = time.time() - start_time - memory_construction_time
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    # RAG implementation methods
    def _handle_bm25_rag(self, message, context_id, tokenizer):
        """Handle BM25 RAG processing."""
        start_time = time.time()
        
        # Extract retrieval query from message
        retrieval_query = self._extract_retrieval_query(message)
        print(f"\n\n\n\nretrieval_query: {retrieval_query}\n\n\n\n")
        
        # Build vectorstore if context changed
        if self.context_id != context_id:
            from langchain_community.retrievers import BM25Retriever
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            print(f"\n\nBM25 build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping BM25 build vectorstore...\n\n")
        
        # Retrieve documents
        self.bm25_retriever.k = self.retrieve_num
        bm25_documents = self.bm25_retriever.invoke(retrieval_query)
        retrieval_context = [f"{doc.page_content}\n" for doc in bm25_documents] 
        memory_construction_time = time.time() - start_time
        
        # Answer the query
        retrieval_memory_string = "\n".join([f"Memory {i+1}:\n{text}" for i, text in enumerate(retrieval_context)])
        
        # Format the message
        ask_llm_message = retrieval_memory_string + "\n" + message
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        format_message = format_chat(message=ask_llm_message, system_message=system_message)
        
        # Generate response
        response = self._chat_complete(
            format_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens if "gpt-4" in self.model else None,
        )
        
        query_time_len = time.time() - start_time - memory_construction_time
        self.context_id = context_id
        
        return {
            "output": response.choices[0].message.content,
            "input_len": response.usage.prompt_tokens,
            "output_len": response.usage.completion_tokens,
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }
    
    def _extract_retrieval_query(self, message):
        """Extract retrieval query from message using regex patterns."""
        patterns = [
            r"Now Answer the Question:\s*(.*)",
            r"Here is the conversation:\s*(.*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.DOTALL)
            if match:
                return ''.join(match.groups())
        
        return message
        
    def _handle_embedding_rag(self, message, context_id, tokenizer):
        """Handle embedding-based RAG processing (Contriever, Text-embedding models)."""
        from methods.embedding_retriever import TextRetriever, RAGSystem
        
        # Determine embedding model
        if any(agent_name in self.agent_name for agent_name in ["rag_contriever"]):
            embedding_model_name = "facebook/contriever"
        elif any(agent_name in self.agent_name for agent_name in ["rag_text_embedding_3_large"]):
            embedding_model_name = "text-embedding-3-large"
        elif any(agent_name in self.agent_name for agent_name in ["rag_text_embedding_3_small"]):
            embedding_model_name = "text-embedding-3-small"
        elif any(agent_name in self.agent_name for agent_name in ["rag_qwen3_embedding_4b"]):
            embedding_model_name = "Qwen/Qwen3-Embedding-4B"
        elif any(agent_name in self.agent_name for agent_name in ["rag_text_embedding_v4"]):
            embedding_model_name = "text-embedding-v4"
        else:
            raise NotImplementedError

        # Build vectorstore if context changed
        if self.context_id != context_id:
            self.retriever = TextRetriever(
                embedding_model_name=embedding_model_name,
                dashscope_base_url=os.environ.get("DASHSCOPE_BASE_URL") if embedding_model_name == "text-embedding-v4" else None,
                dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY") if embedding_model_name == "text-embedding-v4" else None,
            )
            self.retriever.build_vectorstore(self.chunks)
            print(f"\n\n{embedding_model_name} build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping {embedding_model_name} build vectorstore...\n\n")

        # Retrieve relevant passages and answer the query
        use_deepseek = "deepseek" in self.model
        rag_system = RAGSystem(
            self.retriever, self.model, self.temperature, self.max_tokens,
            use_azure=os.environ.get("AZURE_OPENAI_ENDPOINT") is not None and not use_deepseek,
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            azure_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            use_deepseek=use_deepseek,
            deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL"),
            deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY"),
            use_ark_batch=use_deepseek,
            ark_api_key=os.environ.get("DEEPSEEK_API_KEY"),
            batch_model=os.environ.get("BATCH_MODEL"),
        )
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        result = rag_system.answer_query(
            query=message, 
            top_k=self.retrieve_num, 
            system_message=system_message
        )
        retrieval_context = result['context_used']
        
        self.context_id = context_id
        
        return {
            "output": result["answer"],
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(result["answer"], disallowed_special=())),
            "memory_construction_time": result.get("memory_construction_time", result.get("memory_construction_time", 0)),
            "query_time_len": result["query_time_len"],
            "retrieval_context": retrieval_context,
        }
        
    def _handle_raptor_rag(self, message, context_id, tokenizer):
        """Handle RAPTOR RAG processing."""
        # Build vectorstore if context changed
        if self.context_id != context_id:
            texts = self.chunks
            from methods.raptor import RAPTORMethod
            self.raptor_method = RAPTORMethod(texts, max_levels=3)
            print(f"\n\nRaptor build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping Raptor build vectorstore...\n\n")
        
        # Retrieve relevant passages and answer the query
        result = self.raptor_method.run(query=message, k=self.retrieve_num)
        response = result['answer']
        retrieval_context = result['context_used']
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": result.get("memory_construction_time", result.get("memory_construction_time", 0)),
            "query_time_len": result["query_time_len"],
            "retrieval_context": retrieval_context,
        }
        
    def _handle_self_rag(self, message, context_id, tokenizer):
        """Handle Self-RAG processing."""
        from methods.self_rag import SelfRAG
        start_time = time.time()
        
        # Build vectorstore if context changed
        if self.context_id != context_id:
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            self.self_rag = SelfRAG(documents=docs, temperature=self.temperature, top_k=self.retrieve_num)
            print(f"\n\nSelf-RAG build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping Self-RAG build vectorstore...\n\n")
        
        # Process query
        try:
            response, retrieval_context_list, memory_construction_time, query_time_len = self.self_rag.run(query=message)
        except Exception as e:
            response = f"{e}"
            retrieval_context_list = ["ERROR"]
            memory_construction_time = 0
            query_time_len = 0
            print(f"\n\n\n\nError: {e}\n\n\n\n")
        
        # Prepare the context
        retrieval_context = "\n\n".join([f"Passage {i+1}:\n{text}" 
                                        for i, text in enumerate(retrieval_context_list)])
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    # memorag
    def _handle_memorag(self, message, context_id, tokenizer):
        """Handle MemoRAG processing."""
        from methods.memorag import Agent, MemoRAG
        start_time = time.time()
        memory_construction_time = 0
        cache_context_save_dir=f"./outputs/rag_retrieved/MemoRAG/{self.sub_dataset}/chunksize_{self.chunk_size}/context_id_{context_id}"
        
        # build rag agent
        if self.context_id != context_id:
            # API configuration
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
            api_key=os.environ.get("AZURE_OPENAI_API_KEY")
            gen_model = Agent(model=self.model, source="azure", temperature=self.temperature, api_dict={"endpoint":endpoint, "api_version":api_version, "api_key":api_key})
            self.MemoRAG = MemoRAG(
                mem_model_name_or_path="TommyChien/memorag-qwen2-7b-inst",
                ret_model_name_or_path="BAAI/bge-m3",   
                customized_gen_model=gen_model,
                ret_hit=self.retrieve_num, 
                retrieval_chunk_size=self.chunk_size
            )
            # Use the loaded context / memorize the context for question answering
            context = " ".join(self.chunks)
            ## load the context from the cache
            if os.path.exists(f'{cache_context_save_dir}/memory.bin'):
                self.MemoRAG.load(cache_context_save_dir, print_stats=True)
            else:
                self.MemoRAG.memorize(context, save_dir=None, print_stats=True)
            memory_construction_time = time.time() - start_time
            print(f"Finish memorizing, time cost {memory_construction_time}")
        else:
            print(f"\n\nContext {context_id} already processed, skipping MemoRAG build vectorstore...\n\n")
            
        # Retrieve and answer
        response, retrieval_context = self.MemoRAG(query=message, task_type="memorag", max_new_tokens=self.max_tokens)
        
        query_time_len = time.time() - start_time - memory_construction_time
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(str(retrieval_context) + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }
        
    def save_agent(self):
        """Save agent state to disk for persistence."""
        if self._is_agent_type("memagent"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(f"{self.agent_save_to_folder}/memagent_memory.txt", "w") as f:
                f.write(self.memagent_memory)
            print("\n\n memagent memory saved...\n\n")
            return

        if self._is_agent_type("memobrain"):
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            self.memobrain.save_memory(f"{self.agent_save_to_folder}/memobrain_graph.json")
            print("\n\n memobrain memory saved...\n\n")
            return

        # Currently only implemented for Letta agents
        if not self._is_agent_type("letta") and not self._is_agent_type("zep"):
            print("\n\n Agent not saved (not implemented for this agent type) \n\n")
            return
        
        if self._is_agent_type("letta") and "api" not in self.agent_name:
            agent_save_folder = self.agent_save_to_folder
            os.makedirs(agent_save_folder, exist_ok=True)
            
            import shutil
            # Copy the SQLite database file to the target folder
            source_db_path = os.path.expanduser("~/.letta/sqlite.db")
            target_db_path = f"{agent_save_folder}/sqlite.db"
            shutil.copyfile(source_db_path, target_db_path)
            
            # Save the agent ID for future loading
            with open(f"{agent_save_folder}/agent_id.txt", "w") as f:
                f.write(self.agent_state.id)
        elif self._is_agent_type("zep"):
            # save the message that agent has processed
            messages = "agent finished memorization"
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(f"{self.agent_save_to_folder}/messages.txt", "w") as f:
                f.write(messages)
                
        print("\n\n Agent saved...\n\n")

    def load_agent(self):
        """Load agent state from disk."""
        agent_save_folder = self.agent_save_to_folder
        assert os.path.exists(agent_save_folder), f"Folder {agent_save_folder} does not exist."

        if self._is_agent_type("memagent"):
            with open(f"{agent_save_folder}/memagent_memory.txt", "r") as f:
                self.memagent_memory = f.read()
            print("\n\n memagent memory loaded...\n\n")
            return

        if self._is_agent_type("memobrain"):
            self.memobrain.load_memory(f"{agent_save_folder}/memobrain_graph.json")
            self._memobrain_recall_cache = None
            print("\n\n memobrain memory loaded...\n\n")
            return

        if not self._is_agent_type("letta") and not self._is_agent_type("zep"):
            print("\n\nAgent loading not implemented for this agent type\n\n")
            return None

        if self._is_agent_type("letta") and "api" not in self.agent_name:
            import shutil
            # Copy the database file back to the Letta directory
            source_db_path = f"{agent_save_folder}/sqlite.db"
            target_db_path = os.path.expanduser("~/.letta/sqlite.db")
            shutil.copyfile(source_db_path, target_db_path)

            # Load agent ID and find the corresponding agent state
            with open(f"{agent_save_folder}/agent_id.txt", "r") as f:
                agent_id = f.read()

            # Find the agent state with the matching ID
            for agent_state in self.client.list_agents():
                if agent_state.id == agent_id:
                    self.agent_state = agent_state
                    break
        elif self._is_agent_type("zep"):
            # load the message that agent has processed
            os.makedirs(self.agent_save_to_folder, exist_ok=True)
            with open(f"{self.agent_save_to_folder}/messages.txt", "r") as f:
                messages = f.read()
        
        print("\n\n Agent loaded successfully...\n\n")
        
