import os
from autogen.agentchat.contrib.web_surfer import WebSurferAgent
from autogen.coding.func_with_reqs import with_requirements
import requests
import chromadb
from geopy.geocoders import Nominatim
from pathlib import Path
from bs4 import BeautifulSoup
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen import AssistantAgent, UserProxyAgent
from autogen import register_function
from autogen.cache import Cache
from autogen.coding import LocalCommandLineCodeExecutor, CodeBlock
from typing import Annotated, List
import typing
import logging
import autogen
from dotenv import load_dotenv, find_dotenv
import tempfile

# Set logging Configuration
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)
# Set a log level for the logger
logger.setLevel(logging.INFO)

# load environment variables from .env file
load_dotenv(find_dotenv())

# load parameters and configurations
config_list = [{
    "model": os.environ.get("OPENAI_DEPLOYMENT_NAME"),
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "base_url": os.environ.get("OPENAI_ENDPOINT"),
    "api_version": os.environ.get("OPENAI_API_VERSION"),
    "api_type": "azure"
}]
llm_config = {
    "seed": 42,
    "config_list": config_list,
    "temperature": 0.5
}
bing_api_key = os.environ.get("BING_API_KEY")

# Get the current project directory
current_dir = os.getcwd()
# Create a temporary file in the current directory with .txt extension
temp_file = tempfile.NamedTemporaryFile(dir=current_dir, suffix=".txt", delete=False)
# Store the file path in a variable
temp_file_path = temp_file.name
print(f"Temporary file path: {temp_file_path}")

# Define Tools
@with_requirements(python_packages=["typing", "requests", "autogen", "chromadb"], global_imports=["typing", "requests", "autogen", "chromadb"])
def rag_on_document(query: typing.Annotated[str, "The query to search in the index."], document: Annotated[Path, "Path to the document"]) -> str:
    logger.info(f"************  RAG on document is executed with query: {query} ************")
    default_doc = temp_file_path
    doc_path = default_doc if document is None or document == "" else document
    ragproxyagent = autogen.agentchat.contrib.retrieve_user_proxy_agent.RetrieveUserProxyAgent(
        "ragproxyagent",
        human_input_mode="NEVER",
        retrieve_config={
            "task": "qa",
            "docs_path": doc_path,
            "chunk_token_size": 1000,
            "model": config_list[0]["model"],
            "client": chromadb.PersistentClient(path="./tmp/chromadb"),
            "collection_name": "tourist_places",
            "get_or_create": True,
            "overwrite": False
        },
        code_execution_config={"use_docker": False}
    )
    res = ragproxyagent.initiate_chat(planner_agent, message=ragproxyagent.message_generator, problem = query, n_results = 2, silent=True)
    return str(res.chat_history[-1]['content'])


@with_requirements(python_packages=["typing", "requests", "autogen", "chromadb"], global_imports=["typing", "requests", "autogen", "chromadb"])
def bing_search(query: typing.Annotated[str, "The input query to search"]) -> Annotated[str, "The search results"]:
    web_surfer = WebSurferAgent(
        "bing_search",
        system_message="You are a Bing Web surfer Agent whose objective is to find relevant website urls for travel planning.",
        llm_config= llm_config,
        summarizer_llm_config=llm_config,
        browser_config={"viewport_size": 4096, "bing_api_key": bing_api_key}
    )
    register_function(
        visit_website,
        caller=web_surfer,
        executor=user_proxy,
        name="visit_website",
        description="This tool is to scrape content of website using a list of urls and store the website content into a text file that can be used for rag_on_document"
    )
    search_result = user_proxy.initiate_chat(web_surfer, message=query, summary_method="reflection_with_llm", max_turns=2)
    return str(search_result.summary)

def get_lat_lon(location):
    geolocator = Nominatim(user_agent="my_geocoder")
    location = geolocator.geocode(location)

    if location:
        return location.latitude, location.longitude
    else:
        return None, None

@with_requirements(python_packages=["typing", "requests", "autogen", "chromadb"], global_imports=["typing", "requests", "autogen", "chromadb"])
def get_weather_info(destination: typing.Annotated[str, "The place of which weather information to retrieve"], start_date: typing.Annotated[str, "The date of the trip to retrieve weather data"]) -> typing.Annotated[str, "The weather data for given location"]:
    # Use Open-Meteo API to get weather forecast for the destination
    # Assuming the location of the destination is hardcoded for simplicity; in practice, use a geo-location API
    logger.info(f"************  Get weather API is executed for {destination}, {start_date} ************")
    coordinates = {"Grand Canyon": {"lat": 36.1069, "lon": -112.1129},
                   "Philadelphia": {"lat": 39.9526, "lon": -75.1652},
                   "Niagara Falls": {"lat": 43.0962, "lon": -79.0377},
                   "Goa": {"lat": 15.2993, "lon": 74.1240}}

    destination_coordinates = coordinates[destination]

    lat, lon = destination_coordinates["lat"], destination_coordinates["lon"] if destination in coordinates else get_lat_lon(destination)
    forecast_api_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum&start={start_date}&timezone=auto"

    weather_response = requests.get(forecast_api_url)
    weather_data = weather_response.json()
    return str(weather_data)

@with_requirements(python_packages=["typing", "requests", "autogen", "chromadb"], global_imports=["typing", "requests", "autogen", "chromadb"])
def visit_website(urls: Annotated[List[str], "The list of url to scrape"], output_file: Annotated[str, "The path to the destination file to store scraped text"] = temp_file_path) -> Annotated[None, "This tools appends the scraped website into a text file"]:
    """
    Scrapes content from a list of URLs and saves the content into a .txt file.

    Parameters:
    - urls (list): List of URLs to scrape.
    - output_file (str): Name of the output file where scraped content is stored.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        for url in urls:
            try:
                print(f"Scraping: {url}")
                response = requests.get(url)
                response.raise_for_status()  # Ensure the request was successful

                # Parse the HTML content with BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract paragraphs from the HTML
                paragraphs = [p.get_text() for p in soup.find_all('p')]
                content = "\n".join(paragraphs)

                # Write the content into the file with a separator
                file.write(f"URL: {url}\n{'=' * 80}\n")
                file.write(content + "\n\n")

            except requests.exceptions.RequestException as e:
                print(f"Failed to scrape {url}. Error: {str(e)}")
                file.write(f"URL: {url}\nError: {str(e)}\n\n")

    print(f"Scraping completed. Content saved to '{output_file}'")


planner_agent = AssistantAgent(
    "Planner_Agent",
    system_message="You are a trip planner assistant whose objective is to plan itineraries of the trip to a destination. "
                   "Use tools to fetch weather, search web using bing_search, "
                   "scrape web context for search urls using visit_website tool and "
                   "do RAG on scraped documents to find relevant section of web context to find out accommodation, "
                   "transport, outfits, adventure activities and bookings need. "
                   "Use only the tools provided, and reply TERMINATE when done. "
                   "While executing tools, print outputs and reflect exception if failed to execute a tool. "
                   "If web scraping tool is required, create a temp txt file to store scraped website contents "
                   "and use the same file for rag_on_document as input.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Setting up the code executor (optional for running generated code snippets)
work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
# Test Code execution configuration
print(
    code_executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code="print('Hello, World!');"),
        ]
    )
)

# Define the UserProxyAgent
user_proxy = UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"executor": code_executor},
)

# Register tools with the LLM Agent
register_function(
    get_weather_info,
    caller=planner_agent,
    executor=user_proxy,
    name = "get_weather_info",
    description = "This tool fetch weather data from open source api"
)

register_function(
    rag_on_document,
    caller=planner_agent,
    executor=user_proxy,
    name = "rag_on_document",
    description = "This tool fetch relevant information from a document"
)

register_function(
    bing_search,
    caller=planner_agent,
    executor=user_proxy,
    name = "bing_search",
    description = "This tool to search a query in web and get results."
)

register_function(
    visit_website,
    caller=planner_agent,
    executor=user_proxy,
    name = "visit_website",
    description = "This tool is to scrape content of website using a list of urls and store the website content into a text file that can be used for rag_on_document"
)

# The input Question the User want to ask the LLM Agent
question = "Plan a trip to Goa next month on 16 Nov 2024, I will stay for 5 nights"

ReAct_prompt = """
You are a Trip Planning expert tasked with helping users making a trip itinerary.
You can analyse the query, figure out the travel destination, dates and assess the need of checking weather forecast, search accomodation, recommend outfits and suggest adventure activities like hiking, trekking opportunity and need for advance booking.
Use the following format:

Question: the input question or request
Thought: you should always think about what to do to respond to the question
Action: the action to take (if any)
Action Input: the input to the action (e.g., search query, location for weather, query for rag, url for web scraping)
Observation: the result of the action
... (this process can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question or request

Begin!
Question: {input}
"""

# Define the ReAct prompt message
def react_prompt_message(sender, recipient, context):
    return ReAct_prompt.format(input=context["question"])

with Cache.disk(cache_seed=43) as cache:
    planner_result = user_proxy.initiate_chat(
        planner_agent,
        message=react_prompt_message,
        question=question,
        cache=cache,
        summary_method="reflection_with_llm",
        max_turns=10
    )

print(f"Summary of the Trip Planning")
print(planner_result.chat_history)
print(planner_result.summary)