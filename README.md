# autogen-llm-agents
A repository for developers to build LLM Agents using Microsoft AutoGen framework

## Getting Started

Step 1: Install Python packages from [requirements.txt](./requirements.txt)
```commandline
pip install -r requirement.txt
```

Step 2: update OpenAI API Key, Endpoint, Deployment Name and API Version in the [.env](./.env) file. Similarly update Bing API Key as well.
```commandline
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
OPENAI_ENDPOINT=https://<YOUR_OPENAI_ENDPOINT>.openai.azure.com
OPENAI_DEPLOYMENT_NAME=<YOUR_OPENAI_DEPLOYMENT_NAME>
OPENAI_API_VERSION=<API_VERSION>
BING_API_KEY=<YOUR_BING_API_KEY>
```

## List of LLM Agents
- [Trip Planner Agent](#trip-planner-agent)

## Trip Planner Agent
The Trip Planner Agent is an implementation of an AutoGen based LLM Agent whose objective is to create an itinerary for the input travel related query.
This implementation leverages a ReAct based planning and reasoning engine that creates a plan of execution to serve the query. 
It support implementation for few tools as follows:
- Tools to fetch weather data.
- Tools to search the web using Bing search
- Tools to scrape web content
- Tools to retrieve relevant sections from a document using Retrieval Augmented Generation (RAG).


