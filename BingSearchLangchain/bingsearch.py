#  this code sets up an environment for interacting with an AI agent powered by Azure OpenAI. The agent is capable of processing user input, utilizing Bing search if necessary, and generating responses based on the conversation context and its built-in knowledge.
import os
from langchain.tools import StructuredTool
from langchain_community.utilities import BingSearchAPIWrapper
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import AzureOpenAI,AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

os.environ["BING_SUBSCRIPTION_KEY"] = os.getenv("BING_SUBSCRIPTION_KEY")
os.environ["BING_SEARCH_URL"] = os.getenv("BING_SEARCH_URL")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_API_TYPE"] = os.getenv("AZURE_OPENAI_API_TYPE")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")

# Create Bing search tool
search_api = BingSearchAPIWrapper()

def bing_search(query: str, max_results: int = 3):
    return search_api.results(query, max_results)

# Define the tools to be used
search = StructuredTool.from_function(
    func=bing_search,
    name="bing_search",
    description="Search the web using Bing search engine"
)
tools = [search]

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI agent designed to answer user queries. Your primary goal is to generate responses using your built-in knowledge and capabilities. If you can confidently provide an answer, do so directly without calling external sources. However, if you're unsure or unable to answer the user's query, you may resort to Bing search for additional information. Use your judgment to determine when to call Bing search based on the clarity and complexity of the user's request. Your role is to assist the user by providing accurate and relevant information."""

        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
# Initialize the AzureOpenAI instance
llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo", 
    openai_api_version="2024-02-15-preview"
)

# Create an agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an agent executor to handle the invocation of agent and tool based on i/p
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Take input from the user
user_input = input("Please enter your question: ")

# Invoke the agent with user input
response = agent_executor.invoke({"input": user_input})

print(response)
