import aiohttp
import os
from urllib.parse import quote_plus

# Use DuckDuckGo API directly (no external service needed)
DUCKDUCKGO_URL = "https://api.duckduckgo.com/"

async def search(query: str):
    """Search using DuckDuckGo instant answer API"""
    try:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(DUCKDUCKGO_URL, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    # Abstract (main answer)
                    if data.get("Abstract"):
                        results.append({
                            "title": data.get("Heading", "Answer"),
                            "content": data.get("Abstract"),
                            "url": data.get("AbstractURL", "")
                        })
                    
                    # Related topics
                    for topic in data.get("RelatedTopics", [])[:5]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append({
                                "title": topic.get("Text", "")[:100],
                                "content": topic.get("Text", ""),
                                "url": topic.get("FirstURL", "")
                            })
                    
                    # If no results from instant answer, return empty
                    if not results:
                        results.append({
                            "title": "No instant results",
                            "content": f"No direct answers found for: {query}. Try rephrasing your question.",
                            "url": f"https://duckduckgo.com/?q={quote_plus(query)}"
                        })
                    
                    return {"results": results}
                else:
                    return {"results": [], "error": f"Search failed with status {response.status}"}
                    
    except Exception as e:
        return {"results": [], "error": str(e)}


async def _search(query: str):
    """Legacy function for compatibility"""
    return await search(query)

