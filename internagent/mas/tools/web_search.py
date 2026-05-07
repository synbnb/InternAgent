from typing import List, Dict, Any
import os
import requests

class WebSearch:
    def __init__(self, config: Dict[str, Any]):
        self.serper_api_key = config.get("serper_api_key", os.getenv("SERPER_API_KEY"))

    def search_serper(self, query: str, num_result_pages: int = 5) -> List[Dict[str, Any]]:
        import requests

        responses = []
        
        # Check if API key is available
        if not self.serper_api_key:
            responses.append({"error": "SERPER_API_KEY not set"})
            return responses
        
        # Fetch the results given the URL
        try:
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json",
            }
            payload = {"q": query, "num": num_result_pages}

            result = requests.post(url, headers=headers, json=payload, timeout=30)
            data = result.json()

            if "organic" in data:
                search_items = data["organic"]
                for i, search_item in enumerate(search_items, start=1):
                    title = search_item.get("title", "N/A")
                    snippet = search_item.get("snippet", "N/A")
                    link = search_item.get("link", "N/A")

                    response = {
                        "result_id": i,
                        "title": title,
                        "description": snippet,
                        "long_description": search_item.get("content", snippet),
                        "url": link,
                        "raw": search_item,
                    }
                    
                    # Skip filtered results
                    if "huggingface" in link.lower() and "gaia" in link.lower():
                        continue
                    if "2311.12983" in link.lower():
                        continue
                    if "gaia" in snippet.lower() and "benchmark" in snippet.lower():
                        continue
                    
                    responses.append(response)
            else:
                responses.append({"error": "No organic results"})
                
        except requests.RequestException:
            responses.append({"error": "Request failed"})
        except Exception:
            responses.append({"error": "Search failed"})
        
        return responses

    def search_wiki(self, entity: str) -> str:
        r"""Search the entity in WikiPedia and return the summary of the
            required page, containing factual information about
            the given entity.

        Args:
            entity (str): The entity to be searched.

        Returns:
            str: The search result. If the page corresponding to the entity
                exists, return the summary of this entity in a string.
        """
        import wikipedia

        result: str

        try:
            result = wikipedia.summary(entity, sentences=5, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            result = wikipedia.summary(
                e.options[0], sentences=5, auto_suggest=False
            )
        except wikipedia.exceptions.PageError:
            result = (
                "There is no page in Wikipedia corresponding to entity "
                f"{entity}, please specify another word to describe the"
                " entity to be searched."
            )
        except wikipedia.exceptions.WikipediaException as e:
            result = f"An exception occurred during the search: {e}"

        return result
