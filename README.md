# 🛠 Azure OpenAI GGB AI Benchmarking Tool

## 📊 Performance Benchmarking Fundamentals

To thoroughly evaluate the capabilities and enhancements of the model, we categorize our assessments into two primary areas: model quality and performance metrics.

This repo provides benchmarking tools to assist customers in evaluating provisioned-throughput deployments, which offer a fixed amount of model compute. However, actual performance can vary based on several factors, including prompt size, generation size, and call rate. This tool simplifies the process of running test traffic on your deployment to validate throughput for specific workloads, providing key performance statistics such as average and 95th percentile latencies and deployment utilization. It also provides abstraction layers in the form of a Pythonic clients that facilitates running tests from your notebooks, especially easy to run with the Jupyter VS Code integration.

Utilize this tool to experiment with total throughput at 100% utilization across various traffic patterns for a `Provisioned-Managed` deployment type, enabling you to fine-tune your solution design by adjusting prompt size, generation size, and deployed PTUs.

### 📈 Performance Metrics

These metrics are crucial for assessing the operational efficiency of the model, focusing on its responsiveness and capacity to handle requests effectively.

- **Latency:** This measures the time taken to receive a response for a request, which is critical for understanding the model's responsiveness.
- **Throughput:** This evaluates the volume of requests the model can process within a specific timeframe, indicating its capacity to handle workload.

### 📈 Quality Metrics
#TODO

## 📝 Pre-requisites
1. An Azure OpenAI Service resource with a model model deployed with a provisioned deployment (either ```Provisioned``` or ```Provisioned-Managed```) deployment type. For more information, see the [resource deployment guide](https://learn.microsoft.com/azure/ai-services/openai/how-to/create-resource?pivots=web-portal).
2. A detailed guide on how to set up your environment and get ready to run all the notebooks and code in this repository can be found in the [SETTINGS.md](SETTINGS.md) file. Please follow the instructions there to ensure a smooth experience.

## Latency Test 

Dive into the intricacies of latency testing with our streamlined guide. Starting with initializing the testing class tailored to your deployment type, we guide you through executing comprehensive tests using the `run_latency_benchmark_bulk` method. Our approach meticulously evaluates performance across various metrics, including Median Time, Percentile Times, and Coefficient of Variation, to pinpoint potential bottlenecks and optimize model behavior under diverse token limits.

### Run your Latency Benchmarking in Just a Few Lines of Code

```python 
# Import the necessary class for benchmarking
from benchmarking.azure_openai_benchmark import AzureOpenAIBenchmarkNonStreaming

# Define your Azure OpenAI credentials and parameters
api_key = "YOUR_AZURE_OPENAI_API_KEY"
azure_endpoint = "YOUR_AZURE_OPENAI_ENDPOINT"
api_version = "YOUR_AZURE_OPENAI_API_VERSION"

# Initialize the benchmarking class with your credentials
benchmark_client = AzureOpenAIBenchmarkNonStreaming(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version
)

# Define the deployments and token configurations for the tests
deployment_names = ["deployment1", "deployment2"]  # Example deployment names
max_tokens_list = [100, 500, 1000]  # Example token counts to test
num_iterations = 5  # Number of iterations per test

# Execute the benchmark tests asynchronously
await benchmark_client.run_latency_benchmark_bulk(
    deployment_names=deployment_names,
    max_tokens_list=max_tokens_list,
    iterations=num_iterations,
    context_tokens=1000,
    multiregion=False
)
```

For a deeper understanding of these processes and to master interpreting the nuanced results, our detailed [HOWTO-Latency.md](benchmarks/HOWTO-LATENCY.md) guide is your go-to resource. 

## Throughput Test 

You can run these tests in a script, docker container or Jupyter Notebook. For a deeper understanding of these processes and to master interpreting the nuanced results, our detailed [HOWTO-Throughput.md](benchmarks/HOWTO-LATENCY.md) guide is your go-to resource.  

### Common Scenarios:
The table below provides an example prompt & generation size we have seen with some customers. Actual sizes will vary significantly based on your overall architecture For example,the amount of data grounding you pull into the prompt as part of a chat session can increase the prompt size significantly.

| Scenario | Prompt Size | Completion Size | Calls per minute | Provisioned throughput units (PTU) required |
| -- | -- | -- | -- | -- |
| Chat | 1000 | 200 | 45 | 200 |
| Summarization | 7000 | 150 | 7 | 100 |
| Classification | 7000 | 1 | 24 | 300|

Or see the [pre-configured shape-profiles below](#shape-profiles).

### Run samples 

During a run, statistics are output every second to `stdout` while logs are output to `stderr`. Some metrics may not show up immediately due to lack of data. 

**Run load test at 60 RPM with exponential retry back-off**

```
$ python -m src.performance.bench load \
    --deployment gpt-4 \
    --rate 60 \
    --retry exponential \
    https://myaccount.openai.azure.com

2023-10-19 18:21:06 INFO     using shape profile balanced: context tokens: 500, max tokens: 500
2023-10-19 18:21:06 INFO     warming up prompt cache
2023-10-19 18:21:06 INFO     starting load...
2023-10-19 18:21:06 rpm: 1.0   requests: 1     failures: 0    throttled: 0    ctx tpm: 501.0  gen tpm: 103.0  ttft avg: 0.736  ttft 95th: n/a    tbt avg: 0.088  tbt 95th: n/a    e2e avg: 1.845  e2e 95th: n/a    util avg: 0.0%   util 95th: n/a   
2023-10-19 18:21:07 rpm: 5.0   requests: 5     failures: 0    throttled: 0    ctx tpm: 2505.0 gen tpm: 515.0  ttft avg: 0.937  ttft 95th: 1.321  tbt avg: 0.042  tbt 95th: 0.043  e2e avg: 1.223 e2e 95th: 1.658 util avg: 0.8%   util 95th: 1.6%  
2023-10-19 18:21:08 rpm: 8.0   requests: 8     failures: 0    throttled: 0    ctx tpm: 4008.0 gen tpm: 824.0  ttft avg: 0.913  ttft 95th: 1.304  tbt avg: 0.042  tbt 95th: 0.043  e2e avg: 1.241 e2e 95th: 1.663 util avg: 1.3%   util 95th: 2.6% 
```

**Load test with custom messages being loaded from file and used in all requests**

```
$ python -m src.performance.bench load \
    --deployment gpt-4 \
    --rate 1 \
    --context-generation-method replay
    --replay-path replay_messages.json
    --max-tokens 500 \
    https://myaccount.openai.azure.com
```

**Load test with custom request shape**

```
$ python -m src.performance.bench load \
    --deployment gpt-4 \
    --rate 1 \
    --shape custom \
    --context-tokens 1000 \
    --max-tokens 500 \
    https://myaccount.openai.azure.com
```

**Obtain number of tokens for input context**

`tokenize` subcommand can be used to count number of tokens for a given input.
It supports both text and json chat messages input.

```
$ python -m src.performance.bench tokenize \
    --model gpt-4 \
    "this is my context"
tokens: 4
```

Alternatively you can send your text via stdin:
```
$ cat mychatcontext.json | python -m src.performance.bench tokenize \
    --model gpt-4
tokens: 65
```

### Running Tests with LoadTestBenchmarking Client

To run these tests using a Python client, you can follow the steps below:

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Now you can access the environment variables using os.getenv
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_ID = os.getenv("AZURE_AOAI_DEPLOYMENT_NAME")
DEPLOYMENT_VERSION = os.getenv("AZURE_AOAI_API_VERSION")

from src.performance.client import LoadTestBenchmarking

# Create a client
benchmarking_client = LoadTestBenchmarking(
    model="gpt-4-1106-pagyo",
    region="swedencentral",
    endpoint=AZURE_OPENAI_ENDPOINT,
)

benchmarking_client.run_tests(
    deployment="gpt-4-1106-pagyo",
    rate=5,
    duration=180,
    shape_profile="custom",
    clients=10,
    context_tokens=1000,
    max_tokens=500,
    prevent_server_caching=True,
    log_save_dir="logs/",
)
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
