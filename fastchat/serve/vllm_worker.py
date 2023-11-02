"""
A model worker that executes the model based on vLLM.

See documentations at docs/vllm_integration.md
"""

import argparse
import asyncio
import json
import aiohttp
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import torch
import uvicorn
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import re

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import get_context_length
import uuid


app = FastAPI()

def get_last_query(context):
    lines = context.strip().split("\n")
    if lines and "Human:" in lines[-1]:
        return lines[-1].split("Human:")[-1].strip()
    return ""
async def classify_context(context):
    # Endpoint URL of that raven system 13b instruction following
    url = "http://localhost:7777/generate"
    
    # Prepare the payload
    data = {
        "test_string": context
    }
    
    # Make the API call
    headers = {"x-api-key": "secure-3948765iluhrkwe"}


    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status == 200:
                result = await response.text()
                print("************************ The result of the api is: \n "+ result)
                return result
            else:
                #raise Exception(f"API call failed with status code {response.status}")
                result = "YOGA"
                return result
            
async def fetch_additional_info(question):
    # Endpoint URL
    url = "https://api5.plumeria.ai/query/"
    
    # Prepare the payload
    data = {
        "question": question,
        "count": 5,
        "x_api_key": "your_api_key",
        "collection": "your_collection"
    }
    
    # Make the API call
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Check if result is a dictionary and contains 'items'
                    if isinstance(result, dict) and 'items' in result:
                        items = result['items']
                    # If result is directly a list, use it as items
                    elif isinstance(result, list):
                        items = result
                    else:
                        raise ValueError("Unexpected API response format")
                    
                    # Convert items into a string (you can adjust this as needed)
                    items_str = '; '.join(map(str, items))

                    return items_str
                else:
                    raise Exception(f"API call failed with status code {response.status}")
        except Exception as e:
            #logging.error(f"Error fetching additional information: {e}")
            print(f"Error fetching additional information: {e}")
            return "no additional information found"




class VLLMWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        llm_engine: AsyncLLMEngine,
        conv_template: str,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: vLLM worker..."
        )
        self.tokenizer = llm_engine.engine.tokenizer
        self.context_len = get_context_length(llm_engine.engine.model_config.hf_config)

        if not no_register:
            self.init_heart_beat()

        

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@22 WOOOOOOOOOOTTTTTT")
        print("CONTEXT: " + context)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@22 WOOOOOOOOOOTTTTTT")
        
        #context = context + " - Also speak your answer like a cat would speak! Don't break role. You are a cat!"
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        echo = params.get("echo", True)
        use_beam_search = params.get("use_beam_search", False)
        best_of = params.get("best_of", None)
           
        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        for tid in stop_token_ids:
            if tid is not None:
                stop.add(self.tokenizer.decode(tid))

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            use_beam_search=use_beam_search,
            stop=list(stop),
            max_tokens=max_new_tokens,
            best_of=best_of,
        )

        
        intent = await classify_context(context)

        try:
            response_text = intent
            response_dict = json.loads(response_text)
            function_call = response_dict.get("function_call", "")
            
            if function_call:
                YOUR_THRESHOLD = 0.8
                confidence = 1
                if confidence >= YOUR_THRESHOLD:
                    if "answer_yoga" in function_call:
                        print("THIS IS A YOGA FUNCTION")
                        # Ensure the regular expression pattern correctly matches the format of the function_call string
                        argument_match = re.search(r"answer_yoga_questions\(['\"](.*?)['\"]\)", function_call)
                        if argument_match: 
                            print("ARGUMENT MATCHED")
                            argument = argument_match.group(1)
                            print("YOGA!!! " + intent)
                            question_from_context = argument
                            additional_data = await fetch_additional_info(question_from_context)
                            print("This is additional data:"+ additional_data)
                            additional_info = f"<s>[INST]<s>[INST] Based on the information below, create a direct and concise answer for the user, integrating relevant details. Do not merely reference the information; seamlessly weave it into your response to provide a comprehensive answer. Do not add any disclaimers or indemnity clauses, as they are already provided elsewhere: {additional_data} [/INST]</s>"
                            context = additional_info + context
                    elif "MEDICAL" in intent:
                        print("MEDICINE!! I KNOW MEDICINE!!")
                    else:
                        pass  # Handle other categories if necessary
                else:
                    print("Confidence below threshold, asking for clarification")
                    # Ask for clarification or handle low confidence scenario
        except json.JSONDecodeError:
            print("Error: Response text is not a valid JSON string.")
        except AttributeError as e:
            print("Error:", str(e))
        results_generator = engine.generate(context, sampling_params, request_id)    
        async for request_output in results_generator:
            prompt = request_output.prompt
            if echo:
                text_outputs = [
                    prompt + output.text for output in request_output.outputs
                ]
            else:
                text_outputs = [output.text for output in request_output.outputs]
            text_outputs = " ".join(text_outputs)
            # Note: usage is not supported yet
            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = sum(
                len(output.token_ids) for output in request_output.outputs
            )
            ret = {
                "text": text_outputs,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "cumulative_logprob": [
                    output.cumulative_logprob for output in request_output.outputs
                ],
                "finish_reason": request_output.outputs[0].finish_reason
                if len(request_output.outputs) == 1
                else [output.finish_reason for output in request_output.outputs],
            }
            yield (json.dumps(ret) + "\0").encode()

    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode())


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id):
    async def abort_request() -> None:
        await engine.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    output = await worker.generate(params)
    release_worker_semaphore()
    await engine.abort(request_id)
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.model_path:
        args.model = args.model_path
    if args.num_gpus > 1:
        args.tensor_parallel_size = args.num_gpus

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    worker = VLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        engine,
        args.conv_template,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
