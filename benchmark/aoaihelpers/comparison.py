import os
import time
import json
import asyncio
import logging
from datetime import datetime, timezone
from tabulate import tabulate
from openai import AsyncAzureOpenAI, AzureOpenAI
from utils.ml_logging import get_logger

# Set up logger
logger = get_logger()

class AzureOpenAIStreamingBenchmark:
    def __init__(self, api_key, azure_endpoint, api_version="2024-02-15-preview"):
        """
        Initialize the AzureOpenAILatencyBenchmark with the API key, API version, and endpoint.
        """
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.results = {}

    async def make_call(self, deployment_name, max_tokens):
        """
        Make a chat completion call and print the time taken for the call.
        """
        start_time = time.time()
        logger.info(f"Starting call to model {deployment_name} with max tokens {max_tokens} at (Local time): {datetime.now()}, (GMT): {datetime.now(timezone.utc)}")

        response = await self.client.chat.completions.create(
            model=deployment_name,
            temperature=0.7,
            max_tokens=max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Give me history of New York"}
            ]
        )

        try:
            async for chunk in response:
                chunk_json = json.loads(chunk.model_dump_json(indent=2))
                if chunk_json["choices"]:
                    content = chunk_json["choices"][0]["delta"]["content"]
                    print(content, end='')

        except Exception as e:
            error_type = type(e).__name__
            error_code = e.response.status_code if hasattr(e, 'response') else None  # Get the API error code
            if "errors" not in self.results[deployment_name]:
                self.results[deployment_name]["errors"] = {"count": 0, "codes": []}
            self.results[deployment_name]["errors"]["count"] += 1
            if error_code:
                self.results[deployment_name]["errors"]["codes"].append(error_code)

        end_time = time.time()
        time_taken = end_time - start_time

        print()
        logger.info(f"Finished call to model {deployment_name}. Time taken for chat: {round(time_taken)} seconds or {round(time_taken * 1000)} milliseconds.")
        
        if deployment_name not in self.results:
            self.results[deployment_name] = {"times": [], "tokens": []}
        self.results[deployment_name]["times"].append(time_taken)
        self.results[deployment_name]["tokens"].append(max_tokens)

    async def run_tests(self, deployment_names, max_tokens_list, iterations=1, same_model_interval=1, different_model_interval=5):
        """
        Run the tests for each deployment name and max tokens.
        """
        for deployment_name in deployment_names:
            for max_tokens in max_tokens_list:
                for _ in range(iterations):
                    await self.make_call(deployment_name, max_tokens)
                    await asyncio.sleep(same_model_interval)

            await asyncio.sleep(different_model_interval)

    def save_results(self, location=None, show_table=True):
        """
        Save the results to a JSON file.
        """
        stats = {}
        for key, data in self.results.items():
            max_time = max(data["times"])
            min_time = min(data["times"])
            avg_time = sum(data["times"]) / len(data["times"])
            avg_tokens = sum(data["tokens"]) / len(data["tokens"])
            iterations = len(data["times"])  # Count the number of times
            errors = data.get("errors", {"count": 0, "codes": []})
            stats[key] = {"max_time": max_time, "min_time": min_time, "avg_time": avg_time, "tokens": avg_tokens, "iterations": iterations, "errors": errors}
    
        if location:
            with open(location, 'w') as f:
                json.dump(stats, f, indent=2)  # Add the indent parameter here
    
        if show_table:
            headers = ["Model_maxtokens", "Max Time (Seconds)", "Min Time (Seconds)", "Avg Time (Seconds)", "max_tokens", "Iterations", "Error Count", "Error Codes"]
            table = []
            for key, data in stats.items():
                row = [key, data["max_time"], data["min_time"], data["avg_time"], data["tokens"], data["iterations"], data["errors"]["count"], ", ".join(map(str, data["errors"]["codes"]))]
                table.append(row)
            print(tabulate(table, headers, tablefmt="pretty"))


class AzureOpenAINonStreamingBenchmark:
    def __init__(self, api_key, azure_endpoint, api_version="2024-02-15-preview"):
        """
        Initialize the AzureOpenAINonStreamingBenchmark with the API key, API version, and endpoint.
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.results = {}

    async def make_call(self, deployment_name, max_tokens):
        """
        Make a chat completion call and print the time taken for the call.
        """
        start_time = time.time()
        print(f"Call sent at (Local time): {datetime.now()}, (GMT): {datetime.now(timezone.utc)}, Model: {deployment_name}, Max Tokens: {max_tokens}")

        try:
            response = self.client.chat.completions.create(
                model=deployment_name,
                temperature=0.7,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Give me history of Seattle"}
                ]
            )

            end_time = time.time()
            time_taken = end_time - start_time

            logger.info(response.choices[0].message.content)
            logger.info(f"Time taken for chat: {round(time_taken)} seconds or {round(time_taken * 1000)} milliseconds.")
            
            key = f"{deployment_name}_{max_tokens}"
            if key not in self.results:
                self.results[key] = {"times": [], "tokens": [], "errors": {"count": 0, "codes": []}}
            self.results[key]["times"].append(time_taken)
            self.results[key]["tokens"].append(max_tokens)

        except Exception as e:
            error_type = type(e).__name__
            error_code = e.response.status_code if hasattr(e, 'response') else None  # Get the API error code
            self.results[key]["errors"]["count"] += 1
            if error_code:
                self.results[key]["errors"]["codes"].append(error_code)

    async def run_tests(self, deployment_names, max_tokens_list, iterations=1, same_model_interval=5, different_model_interval=15):
        """
        Run the tests for each deployment name and max tokens.
        """
        for deployment_name in deployment_names:
            for max_tokens in max_tokens_list:
                for _ in range(iterations):
                    await self.make_call(deployment_name, max_tokens)
                    await asyncio.sleep(same_model_interval)
    
            await asyncio.sleep(different_model_interval)


    def save_results(self, location=None, show_table=True):
        """
        Save the results to a JSON file.
        """
        stats = {}
        for key, data in self.results.items():
            max_time = max(data["times"])
            min_time = min(data["times"])
            avg_time = sum(data["times"]) / len(data["times"])
            avg_tokens = sum(data["tokens"]) / len(data["tokens"])
            iterations = len(data["times"])  # Count the number of times
            errors = data.get("errors", {"count": 0, "codes": []})
            stats[key] = {"max_time": max_time, "min_time": min_time, "avg_time": avg_time, "tokens": avg_tokens, "iterations": iterations, "errors": errors}
    
        if location:
            with open(location, 'w') as f:
                json.dump(stats, f, indent=2)  # Add the indent parameter here
    
        if show_table:
            headers = ["Model_maxtokens", "Max Time (Seconds)", "Min Time (Seconds)", "Avg Time (Seconds)", "max_tokens", "Iterations", "Error Count", "Error Codes"]
            table = []
            for key, data in stats.items():
                row = [key, data["max_time"], data["min_time"], data["avg_time"], data["tokens"], data["iterations"], data["errors"]["count"], ", ".join(map(str, data["errors"]["codes"]))]
                table.append(row)
            print(tabulate(table, headers, tablefmt="pretty"))