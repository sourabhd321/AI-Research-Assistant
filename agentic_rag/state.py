from typing_extensions import TypedDict
from pydantic import Field

class MyState(TypedDict):
    query: str
    document: str
    score: str
    response: str
    count: str
    retry_count: str
    refined_query: str
