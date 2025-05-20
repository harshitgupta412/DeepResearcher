import requests
import json
import http.client
import time
from lotus.web_search import web_search as lotus_web_search, WebSearchCorpus
import pandas as pd
import os 

def web_search(query, config):
    if not query:
        raise ValueError("Search query cannot be empty")
    if config['search_engine'] == 'google':
        return serper_google_search(
            query=query,
            serper_api_key=config['serper_api_key'],
            top_k=config['search_top_k'],
            region=config['search_region'],
            lang=config['search_lang']
        )
    elif config['search_engine'] == 'bing':
        return azure_bing_search(
            query=query,
            subscription_key=config['azure_bing_search_subscription_key'],
            mkt=config['azure_bing_search_mkt'],
            top_k=config['search_top_k']
        )
    elif config['search_engine'] == 'lotus':
        corpuses = [
            WebSearchCorpus(corpus)
            for corpus in config['lotus_corpus']
        ]
        results = []
        for corpus in corpuses:
            df: pd.DataFrame = lotus_web_search(
                query=query,
                corpus=corpus,
                K=config['search_top_k']
            )
            if corpus == WebSearchCorpus.ARXIV:
                df.rename(
                    columns={"abstract": "snippet", "link": "url", "published": "date"},
                    inplace=True,
                )
                df["date"] = df["date"].astype(str)
            elif (
                corpus == WebSearchCorpus.GOOGLE or corpus == WebSearchCorpus.GOOGLE_SCHOLAR
            ):
                df.rename(columns={"link": "url"}, inplace=True)
            elif corpus == WebSearchCorpus.BING:
                df.rename(columns={"name": "title"}, inplace=True)
            elif corpus == WebSearchCorpus.TAVILY:
                df.rename(columns={"content": "snippet"}, inplace=True)
            results.extend([{
                "title": row['title'],
                "link": row['url'],
                "snippet": row.get('snippet', ''),
            } for _, row in df.iterrows()])
        return results


def azure_bing_search(query, subscription_key, mkt, top_k, depth=0):
    params = {'q': query, 'mkt': mkt, 'count': top_k}
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    results = []

    try:
        response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params)
        json_response = response.json()
        for e in json_response['webPages']['value']:
            results.append({
                "title": e['name'],
                "link": e['url'],
                "snippet": e['snippet']
            })
    except Exception as e:
        print(f"Bing search API error: {e}")
        if depth < 1024:
            time.sleep(1)
            return azure_bing_search(query, subscription_key, mkt, top_k, depth+1)
    return results


def serper_google_search(
        query, 
        serper_api_key,
        top_k,
        region,
        lang,
        depth=0
    ):
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
                "q": query,
                "num": top_k,
                "gl": region,
                "hl": lang,
            })
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))

        if not data:
            raise Exception("The google search API is temporarily unavailable, please try again later.")

        if "organic" not in data:
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        else:
            results = data["organic"]
            print("search success")
            return results
    except Exception as e:
        # print(f"Serper search API error: {e}")
        if depth < 512:
            time.sleep(1)
            return serper_google_search(query, serper_api_key, top_k, region, lang, depth=depth+1)
    print("search failed")
    return []


if __name__ == "__main__":
    print(serper_google_search("test", "your serper key",1,"us","en"))