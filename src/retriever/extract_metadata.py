import re
import json
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ExtractMetadatafromQuery:
    query: str
    CONFIG: Any


    def execute(self,) -> Dict[str, Any]:
        prompt = self.__extraction_prompt()
        response = self.CONFIG.LLM_MODEL.invoke(prompt).content
        response = re.search(r"\{.*\}", response, re.DOTALL).group().strip()
        json_data = json.loads(response)
        final_res = self.restructure_metadata(json_data)
        return final_res
    
    def restructure_metadata(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        output = {}
        for key, value in json_data.items():

            if key in self.CONFIG.metadata_filters.allowed_keys:
                if isinstance(value, list):
                    value = [str(i).strip() for i in value]
                    output[key] = {"$in": value}
                else:
                    output[key] = str(value).strip()
        return output
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        for key in response.keys():
            if key not in self.CONFIG.metadata_filters.allowed_keys:
                return False
            self.__validate_allowed_values(key, response[key])
        return True
    
    def __validate_allowed_values(self, key: str, value: Any) -> bool:
        if key == "section_name":
            allowed_section_names = self.CONFIG.metadata_filters.section_name.to_list()
            if isinstance(value, list):
                return all(isinstance(i, str) and i in allowed_section_names for i in value)
            return isinstance(value, str) and value in allowed_section_names
        return False

    def __extraction_prompt(self):
        return f"""
    You are a librarian. Read the following text chunk and generate metadata in JSON format.
    
    Extract:
    1. "topic_num": It is an integer value present in query. It can have value from 1 to 15. If multiple topics are present then return a list containing all values
    2. "section_name": It is the sub-section title of text in the document. It should be one of  values from below:
        {"\n".join([f"- {i}" for i in self.CONFIG.metadata_filters.section_name.to_list()])}
        If none of the section_name value mentioned above is applicable, skip the key in response.

    Text Chunk:
    {self.query}
    Instructions:
        - Return ONLY the JSON string.
        - Your responses will be validated against te provided values and hence do not make any error.
        - The keys of json should be the one mentioned i.e. ["topic_num", "section_name"]
        - No need to provide any explanation about the response
        - Output should always be json
    """