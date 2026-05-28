import os
import time
import json
from typing import Optional, List, Dict, Callable, Any
import functools

import torch
from transformers import PreTrainedTokenizer, set_seed
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def format_chat(
    message: str,
    system_message: Optional[str]=None,
) -> List[Dict[str, str]]:
    """
    Format the message into a list of dictionaries with role and content keys.
    This is useful for the chat-based models without tokenizer that does this.
    """
    if system_message is not None:
        chat = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]
    else:
        chat = [{"role": "user", "content": message}]
    return chat


def call_api(func:Callable, limit: int=5, pause: int=10):
    """
    Call the API function with retries and rate limit handling.
    TODO: more error handling?
    """
    count = 0
    while True:
        try:
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            msg = str(e).lower()
            if "rate limit" in msg or "rate_limit" in msg or "quota" in msg or "429" in msg:
                logger.info(f"Rate limit exceeded, waiting {pause} secs and retrying...")
                time.sleep(pause)
            elif count < limit:
                logger.info(f"Encountered error {e}, retrying...")
                count += 1
            else:
                logger.info("Skipping generation due to unknown error")
                output = None
                break
    return output


class LLM:
    """
    Base class for generative models.
    """
    def __init__(
        self,
        model_name: str,
        temperature: float=0.9,
        top_p: float=0.9,
        max_length: int=32768,
        generation_max_length: int=2048,
        generation_min_length: int=0,
        do_sample: bool=True,
        stop_newline: bool=False,
        use_chat_template: bool=False,
        system_message: Optional[str]="You are a helpful assistant.",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.generation_max_length = generation_max_length
        self.generation_min_length = generation_min_length
        self.do_sample = do_sample
        self.use_chat_template = use_chat_template
        self.system_message = system_message
        self.stops = None
        if stop_newline:
            self.stops = ["\n", "\n\n"]

    """
    Prepare the data for input to the llm

    test_item: dict[str, any]
        the test item to be used for the generation, this dictionary is from the data preprocessing step and are used for further formatting to specific models, such as tokenization and/or chat formatting
    data: dict[str, any]
        the data dictionary that contains the template for the user message and system

    Returns the prepared input (type is model-specific)
    """
    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Any:
        raise NotImplementedError("prepare_inputs not implemented for LLM")

    """
    Generate the output from the model

    The inputs have been prepared, the prompt is only the user message as a string that needs to be pre-processed.
    kwargs contains any additional parameters.
    This function should be implemented by the children class.

    The output should be a dictionary with the following:
     - "output" (str): the generated output
     - "input_len" (int): the length of the input tokens
     - "output_len" (int): the length of the output tokens
     - "input_text" (str or List[Dict[str, str]]): the input text or the chat format
    There may be additional keys depending on the model.
    This function may also return None in case of errors (e.g., denied by the API provider).

    """
    def generate(self, inputs: Optional[Any]=None, prompt: Optional[str]=None, **kwargs) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("generate not implemented for LLM")

    """
    Generate the output from the model for a list of inputs or prompts.
    This is similar to to the generate function but everything is in a list.

    The children classes may override this function for optimization.
    """
    def generate_batch(self, inputs: Optional[List[Any]]=None, prompt: Optional[List[str]]=None, **kwargs) -> List[Optional[Dict[str, Any]]]:
        outputs = []
        if inputs is None:
            for p in tqdm(prompt):
                outputs.append(self.generate(prompt=p, **kwargs))
        else:
            for i in tqdm(inputs):
                outputs.append(self.generate(inputs=i, **kwargs))
        return outputs


class OpenAIModel(LLM):
    def __init__(
        self,
        model_name,
        temperature=0.9,
        top_p=0.9,
        max_length=32768,
        generation_max_length=2048,
        generation_min_length=0,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        system_message=None,
        seed=42,
        **kwargs,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
            system_message=system_message,
        )
        import openai
        import tiktoken
        if "azure" in model_name:
            # env var: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION
            self.model = openai.AzureOpenAI()
            model_name = model_name[model_name.index("/")+1:]
        else:
            # make sure to set the OPENAI_API_KEY environment variable
            self.model = openai.OpenAI()
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.seed = seed
        self.API_MAX_LENGTH = 128000 # this is defined by the OPENAI API


    def prepare_inputs(self, test_item, data):
        buffer = 100
        # we don't include system message to stay consistent with other models, which defaults to None
        prompt = format_chat(data["user_template"].format(**test_item), system_message=self.system_message)
        inputs = "\n".join([f"Role: {x['role']}\nContent: {x['content']}" for x in prompt])
        tokens = self.tokenizer.encode(inputs)
        input_len = len(tokens)

        if self.max_length > self.API_MAX_LENGTH:
            logger.warning(f"max_length {self.max_length} is greater than {self.API_MAX_LENGTH}, setting to {self.API_MAX_LENGTH}")
            self.max_length = self.API_MAX_LENGTH

        if input_len > self.max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (self.max_length - self.generation_max_length - buffer)
            new_context = self.tokenizer.decode(self.tokenizer.encode(test_item["context"])[:-truncate_length])
            test_item["context"] = new_context
            prompt = format_chat(data["user_template"].format(**test_item), system_message=self.system_message)
        return prompt


    def generate(self, inputs=None, prompt=None, **kwargs):
        if inputs is None:
            # for system_message, set the self.system_message attribute
            inputs = format_chat(prompt, system_message=self.system_message)

        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
        func = functools.partial(
            self.model.chat.completions.create,
            model=self.model_name,
            messages=inputs,
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            stop=self.stops,
            seed=self.seed,
            **kwargs,
        )
        output = call_api(func)
        if output is not None:
            if output.choices[0].message.content is None:
                # sometimes the model output can get filtered but still return a message
                return None
            return {
                "output": output.choices[0].message.content,
                "input_len": output.usage.prompt_tokens,
                "output_len": output.usage.completion_tokens,
                "input_text": inputs,
                "system_fingerprint": output.system_fingerprint,
            }
        return None

    def batch_api(self, inputs, batch_file, **kwargs):
        with open(batch_file, "w") as f:
            for idx, p in enumerate(inputs):
                f.write(json.dumps({
                    "custom_id": f"{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model_name, 
                        "messages": p,
                        "max_tokens": self.generation_max_length,
                        "temperature": self.temperature if self.do_sample else 0.0,
                        "top_p": self.top_p,
                        "stop": self.stops,
                        "seed": self.seed,
                        **kwargs,
                    }
                }) + "\n")
        upload_file = self.model.files.create(file=open(batch_file, "rb"), purpose="batch")
        batch_job = self.model.batches.create(input_file_id=upload_file.id, endpoint="/v1/chat/completions", completion_window='24h')
        logger.info(f"Starting batch job: {batch_job.id}")

        while batch_job.status != "completed":
            if batch_job.status in ['failed', 'expired', 'cancelled']:
                logger.error(f"Batch job failed: {batch_job.status}")
                raise Exception(f"Batch job {batch_job.id} failed: {batch_job.status}")
            time.sleep(5)
            batch_job = self.model.batches.retrieve(batch_job.id)
            logger.info(batch_job)

        result_file_id = batch_job.output_file_id
        result = self.model.files.content(result_file_id).content
        outputs = [None for _ in inputs]
        # save a copy just in case but there may be name collision so we don't read from this file
        with open(batch_file+".result", "wb") as f:
            f.write(result)

        for line in result.decode("utf-8").strip().split("\n"):
            output = json.loads(line)
            task_id = int(output["custom_id"])
            res = output["response"]['body']
            if res["choices"][0]["message"]["content"] is not None:
                outputs[task_id] = {
                    "output": res["choices"][0]["message"]["content"],
                    "input_len": res["usage"]["prompt_tokens"],
                    "output_len": res["usage"]["completion_tokens"],
                    "input_text": inputs[task_id],
                    "system_fingerprint": res["system_fingerprint"],
                }

        return outputs


    def generate_batch(self, inputs=None, prompt=None, **kwargs):
        """
        Generate for a batch of inputs.
        There are two methods:
        1. Use the batch API provided by OpenAI, which involves uploading all requests in a file and getting an output file. This is cheaper and should be faster than just calling the API for each request. To use this, set batch_file to a file path.
        2. Use the normal API call for each request with multiple threads for some speedup.
        """
        # https://cookbook.openai.com/examples/batch_processing
        # https://platform.openai.com/docs/api-reference/batch/create
        batch_file = kwargs.pop("batch_file", None)
        if batch_file:
            # use the batch api, which only supports upto 50k requests/lines and 200MB in size
            logger.info(f"Using {batch_file} for batch generation")
            if inputs is None:
                inputs = [format_chat(p, system_message=self.system_message) for p in prompt]

            try:
                outputs = self.batch_api(inputs, batch_file, **kwargs)
            except Exception as e:
                # one possible error is that the file is too large, so we need to split it
                batch_size = 100
                logger.info(f"Error in batch generation: {e} with size {len(inputs)}, re-running with batch size {batch_size}, you may want to change the batch size if this fails...")
                outputs = []
                for i in range(0, len(inputs), batch_size):
                    outputs.extend(self.batch_api(inputs[i:i+batch_size], batch_file, **kwargs))

        else:
            if inputs is None:
                inputs = [None for _ in prompt]
            else:
                prompt = [None for _ in inputs]

            # we don't support kwargs here for now
            if len(kwargs) > 0:
                logger.warning("kwargs are not supported for batch generation")
            # use thread_map instead of process_map since the bottleneck is the api call
            outputs = thread_map(self.generate, inputs, prompt, max_workers=32)

        return outputs







