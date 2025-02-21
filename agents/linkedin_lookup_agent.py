from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from tools.tools import get_profile_url_tavily

from langchain_ollama import ChatOllama


# -> str : hints that this method will return str type
# doesn't enforce the method to return that though.
def lookup(name: str) -> str:
    llm = ChatOllama(
        model="mistral",
        temperature=0,
        # other params...
    )

    template = """given the full name {name_of_person} I want you to get me a link to their Linkedin profile page.
                              Your answer should contain only 1 URL strictly."""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    # list of tools for the agent to use
    tools_for_agent = [
        # from the langchain tool package
        # tools provided to the llm to actually perform something
        # on the internet or something.
        Tool(
            name="Search Google for linkedin profile page",
            # which method to run when the tool is used.
            func=get_profile_url_tavily,
            # desc: to specify for the agent when to run the tool
            description="useful for when you need get the Linkedin Page URL",
        )
    ]

    #  from langchain hub package
    # react : a prompt sends to agent including tools and stuff
    # explained later on ReAct paper uses chain of thoughts.
    react_prompt = hub.pull("hwchase17/react")

    # from langchain agents package
    # creates the agent by providing it with all it needs.
    # this is actually the recipe of how to perform actions.
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    # from langchain agents pacakge
    # runtime for the agent.
    # this actually is responsible for actually invoking the required python methods that the agent needs
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    # print(result)

    # get the output from the response
    linked_profile_url = result["output"]

    # print(linked_profile_url)

    return linked_profile_url


if __name__ == "__main__":
    print(lookup(name="Rohan Sengupta Mantis Pro Gaming"))
