from typing import List, Dict, Any

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# extends BaseModel class
class Summary(BaseModel):
    # these 'Field's are from pydantic and used when converting to pydantic Object.
    # meaning the normal string data will be converted to
    # a dict like object with summary key with string summary
    # and facts key : which consists of List of string .
    # means, it will simply parse the normal string data into this pydantic object.
    # similar to a json.
    summary: str = Field(description="summary")
    facts: List[str] = Field(description="interesting facts about them")

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "facts": self.facts}


summary_parser = PydanticOutputParser(pydantic_object=Summary)
