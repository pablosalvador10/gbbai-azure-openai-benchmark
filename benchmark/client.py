import datetime
import subprocess
import logging
import os
from typing import Optional, Literal
from utils.ml_logging import get_logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logger
logger = get_logger()


class BenchmarkingTool:
    """
    A tool for benchmarking Azure OpenAI.

    This class provides a tool for running load generation tests on Azure OpenAI. It allows for specification of 
    parameters such as model, region, and endpoint.

    :param model: Specifies the name of the Azure OpenAI model to be used for benchmarking. 
                  This should be a string representing the model name, such as 'gpt-3'.
    :param region: Specifies the region where the Azure OpenAI model is deployed. 
                   This should be a string representing the region, such as 'westus'.
    :param endpoint: Specifies the base endpoint URL of the Azure OpenAI deployment. 
                     This should be a string representing the URL, such as 'https://api.openai.com'.
    """
    def __init__(self, model: str, region: str, endpoint: str):
        self.model = model
        self.region = region
        self.endpoint = endpoint

    def set_region(self, region: str) -> None:
        """
        Set a new region for the Azure OpenAI model.

        :param region: A string representing the new region, such as 'eastus'.
        """
        self.region = region

    def set_model(self, model: str) -> None:
        """
        Set a new model for benchmarking.

        :param model: A string representing the new model name, such as 'gpt-4'.
        """
        self.model = model

    def run_tests(self,
                deployment: str,
                api_base_endpoint: Optional[int] = None,
                api_version: str = "2023-05-15",
                api_key_env: str = "OPENAI_API_KEY",
                clients: int = 20,
                requests: Optional[int] = None,
                duration: Optional[int] = None,
                run_end_condition_mode: Literal["and", "or"] = "or",
                rate: Optional[float] = None,
                aggregation_window: float = 60,
                context_generation_method: Literal["generate", "replay"] = "generate",
                replay_path: Optional[str] = None,
                shape_profile: Literal["balanced", "context", "generation", "custom"] = "balanced",
                context_tokens: Optional[int] = None,
                max_tokens: Optional[int] = None,
                prevent_server_caching: bool = True,
                completions: int = 1,
                frequency_penalty: Optional[float] = None,
                presence_penalty: Optional[float] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                output_format: Literal["jsonl", "human"] = "human",
                log_save_dir: Optional[str] = "logs/",
                retry: Literal["none", "exponential"] = "none") -> None:
        """
        Run load generation tests using Azure OpenAI.

        Retrieve and process data from the specified data repository.

        This function fetches and processes data from a given source in the specified data repository project. It allows
        for optional specification of deployment parameters such as API version, API key environment, number of clients,
        requests, duration, run end condition mode, rate, aggregation window, context generation method, replay path,
        shape profile, context tokens, max tokens, prevent server caching, completions, frequency penalty, presence
        penalty, temperature, top p, output format, log save directory, and retry strategy.

        :param deployment: Azure OpenAI deployment name.
        :param api_base_endpoint: Azure OpenAI deployment base endpoint.
        :param api_version: Set OpenAI API version. Defaults to "2023-05-15".
        :param api_key_env: Environment variable that contains the API KEY. Defaults to "OPENAI_API_KEY".
        :param clients: Set number of parallel clients to use for load generation. Defaults to 20.
        :param requests: Number of requests for the load run (whether successful or not). Default to 'until killed'.
        :param duration: Duration of load in seconds. Defaults to 'until killed'.
        :param run_end_condition_mode: Determines whether both the `requests` and `duration` args must be reached before
                                        ending the run ('and'), or whether to end the run when either arg is reached ('or').
                                        If only one arg is set, the run will end when it is reached. Defaults to 'or'.
        :param rate: Rate of request generation in Requests Per Minute (RPM). Default to as fast as possible.
        :param aggregation_window: Statistics aggregation sliding window duration in seconds. Defaults to 60.
        :param context_generation_method: Source of context messages to be used during testing. Defaults to "generate".
        :param replay_path: Path to JSON file containing messages for replay when using --context-message-source=replay.
        :param shape_profile: Shape profile of requests. Defaults to "balanced".
        :param context_tokens: Number of context tokens to use when --shape-profile=custom.
        :param max_tokens: Number of requested max_tokens when --shape-profile=custom. Defaults to unset.
        :param prevent_server_caching: Adds a random prefixes to all requests in order to prevent server-side caching.
                                        Defaults to True.
        :param completions: Number of completion for each request. Defaults to 1.
        :param frequency_penalty: Request frequency_penalty.
        :param presence_penalty: Request presence_penalty.
        :param temperature: Request temperature.
        :param top_p: Request top_p.
        :param output_format: Output format. Defaults to "human".
        :param log_save_dir: If provided, will save stdout to this directory. Filename will include important run parameters.
        :param retry: Request retry strategy. See README for details. Defaults to "none".
        :raises: subprocess.CalledProcessError: If an error occurs while running the command.
        :raises: Exception: If an unexpected error occurs.
        """
        process = None
        try:
            if api_base_endpoint is None and self.endpoint is None:
                raise ValueError("Either 'api_base_endpoint' must be provided as an argument, or 'endpoint' must be set in the constructor.")
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            date_str, time_str = now_str.split('_')
            log_file_path = os.path.join(log_save_dir, f"{self.model}/{self.region}/{date_str}/{time_str.replace('_', ' ')}.log")
            
            command = [
                "python", "-m", "benchmark.bench", "load",
                "--api-version", api_version,
                "--api-key-env", api_key_env,
                "--clients", str(clients),
            ]

            if requests is not None:
                command.extend(["--requests", str(requests)])

            if duration is not None:
                command.extend(["--duration", str(duration)])

            command.extend([
                "--run-end-condition-mode", run_end_condition_mode,
            ])

            if rate is not None:
                command.extend(["--rate", str(rate)])

            command.extend([
                "--aggregation-window", str(aggregation_window),
                "--context-generation-method", context_generation_method,
            ])

            if replay_path is not None:
                command.extend(["--replay-path", replay_path])

            command.extend([
                "--shape-profile", shape_profile,
            ])

            if context_tokens is not None:
                command.extend(["--context-tokens", str(context_tokens)])

            if max_tokens is not None:
                command.extend(["--max-tokens", str(max_tokens)])

            command.extend([
                "--prevent-server-caching", str(prevent_server_caching),
                "--completions", str(completions),
            ])

            if frequency_penalty is not None:
                command.extend(["--frequency-penalty", str(frequency_penalty)])

            if presence_penalty is not None:
                command.extend(["--presence-penalty", str(presence_penalty)])

            if temperature is not None:
                command.extend(["--temperature", str(temperature)])

            if top_p is not None:
                command.extend(["--top-p", str(top_p)])

            command.extend([
                "--output-format", output_format,
            ])

            if log_save_dir is not None:
                command.extend(["--log-save-dir", log_save_dir])

            command.extend([
                "--retry", retry,
                "--deployment", deployment,
                api_base_endpoint or self.endpoint
            ])

            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            logger.info(f"Initiating load generation tests. Log output will be directed to: {log_file_path}")
            logger.info(f"Executing command: {' '.join(command)}")

            with open(log_file_path, 'w') as log_file:
                process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)

                # Wait for the process to terminate
                process.communicate()

            logger.info(f"Load generation tests have completed. Please refer to {log_file_path} for the detailed logs.")

        except subprocess.CalledProcessError as e:
            logger.error(f"An error occurred while executing the command {command}: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
            raise
        finally:
            if process and process.poll() is None:
                process.kill()