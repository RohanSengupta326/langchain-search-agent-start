from dotenv import load_dotenv
import os

from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI  # Still using LangChain's OpenAI wrapper

from langchain_ollama import ChatOllama

from langchain_core.output_parsers import StrOutputParser

# use the linkedin scrape that we have implemented ourselves.
from third_parties.linkedin import scrape_linkedin_profile

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent


def ice_break_with(name: str):
    print("Hello LangChain with OpenRouter")

    # print(os.environ['OPENAI_API_KEY'])

    # with os module we can access the env variables
    # print(os.environ['COOL_API_KEY'])

    # Prompt template
    summary_template = """
    Given the LinkedIn information {information} about a person, create:
    1. A short summary
    2. Two interesting facts about them
    """

    # langchain's prompt template to create prompt with
    # input variable & base prompt.
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # ChatOpenAI is a langchain wrapper which wraps the process of using
    # api keys of openAI
    # similarly it has other wrappers to wrap the usage of other orgs api keys
    # anthropic, deepseek, google etc.

    # Using OpenRouter instead of OpenAI
    # hence had to update the OPENAI_API_BASE
    # in the .env file to redirect to openrouter and not openAI
    # as langchain doesn't support openRouter directly. so
    # had to bypass the base url.
    llm = ChatOpenAI(
        temperature=0, model="cognitivecomputations/dolphin3.0-r1-mistral-24b:free"
    )

    # USAGE OF OLLAMA LLM model. with langchain.
    # downloaded latest version of ollama locally llama3.2 first.
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
        # other params...
    )

    # langchain is supported with the llm providers like
    # ollama :  let us use llm models locally
    # openAI : let us use llm models on cloud with api.

    # pull mistral from cmdline first
    # ollama pull mistral
    llm = ChatOllama(
        model="mistral",
        temperature=0,
        # other params...
    )

    # Create a LangChain pipeline (| operator passes input from prompt to model)
    # chain = summary_prompt_template | llm

    # to parse the output from LLM to desired output type.
    # in this case the output is coming like: content=""
    # so we extract it and show
    chain = summary_prompt_template | llm | StrOutputParser()

    # Example LinkedIn data (commented out actual scraping)
    linkedin_data = """Rohan Sengupta
    (He/Him)
    SDE-1 @ Mantis Pro Gaming
    """

    # agentic search to find linkedin profile url
    linkedin_username = linkedin_lookup_agent(name=name)
    # send the url to scrape linkedin to get the data.
    # linkedin_data = scrape_linkedin_profile(
    #     linkedin_profile_url="https://www.linkedin.com/in/eden-marco/", mock=True
    # )
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_username, mock=True
    )

    # actually invoke the api call to get the information from the linkedin data.
    res = chain.invoke(input={"information": linkedin_data})

    print(res)


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    print("Ice Breaker Enter")
    ice_break_with(name="Eden Marco")
