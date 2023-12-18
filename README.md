# Document QA

## Introduction

Welcome to my DocQA repository! llmware is a powerful natural language processing library that leverages state-of-the-art language models for various applications. This README will guide you through the setup process and provide details on how to run the included sample script.

## Setup Instructions

Before running the script, please follow these setup instructions:

1. **Install Anaconda:** If you haven't installed Anaconda, you can download it [here](https://www.anaconda.com/products/distribution).

2. **Install Docker:** Ensure Docker is installed on your machine. You can find installation instructions [here](https://docs.docker.com/get-docker/).

3. **Create a New Virtual Environment:** Open a terminal and run the following commands:

   ```bash
   conda create -n llmware python=3.10
   conda activate llmware
   ```

4. **Pull Docker Files:** Download the Docker Compose file by running:

   ```bash
   curl -o docker-compose.yaml https://raw.githubusercontent.com/llmware-ai/llmware/main/docker-compose.yaml
   ```

5. **Initiate Docker Instance:** Start the Docker containers with:

   ```bash
   docker compose up -d
   ```

6. **Install llmware:** Install the llmware library using:

   ```bash
   pip install llmware
   ```

7. **API Key Configuration:** Add your API key locally. Note: **Do not push the API key to version control.**

8. **Set Library Documents Path:** Ensure the path to your library documents is correctly configured in the script.

## Sample Script

The provided Python script demonstrates how to use llmware to interact with language models. The script performs the following tasks:

- Loads a specified language model.
- Creates a library and adds documents from a specified path.
- Indexes documents for prompt generation.
- Prompts the user for a query.
- Generates responses using the language model.
- Prints the model response.

Here's the sample script:

```python
import os
import time
from llmware.configs import LLMWareConfig
from llmware.library import Library
from llmware.prompts import HumanInTheLoop, Prompt
import re
import logging

# ... (script content)

if __name__ == "__main__":
    model_list = ["llmware/bling-1b-0.1", "gpt-4"]
    custom(model_list[1])
```

Feel free to customize the script and explore different language models.

## Notes

- Ensure your API key is kept secure and is not shared or pushed to version control.
- Review and modify the script according to your specific use case.
