from langchain.utilities import DuckDuckGoSearchAPIWrapper
#from langchain_community.tools import DuckDuckGoSearchResults
ddg = DuckDuckGoSearchAPIWrapper()
results = ddg.run("Who is Current President of United States?")
print(results)

