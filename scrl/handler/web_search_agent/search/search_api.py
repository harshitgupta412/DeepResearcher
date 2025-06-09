import requests
import json
import http.client
import time
from lotus.web_search import web_search as lotus_web_search, WebSearchCorpus
import pandas as pd
import os 
from datetime import datetime, timedelta

end_dates=(
"03 Jun 2025",
"03 Jun 2025",
"01 Jun 2025",
"01 Jun 2025",
"31 May 2025",
"30 May 2025",
"30 May 2025",
"30 May 2025",
"29 May 2025",
"29 May 2025",
"28 May 2025",
"03 Jun 2025",
"03 Jun 2025",
"03 Jun 2025",
"02 Jun 2025",
"02 Jun 2025",
"01 Jun 2025",
"01 Jun 2025",
"31 May 2025",
"31 May 2025",
"30 May 2025",
"30 May 2025",
"29 May 2025",
"29 May 2025",
"28 May 2025",
"28 May 2025",
"28 May 2025",
"28 May 2025",
"28 May 2025",
"30 Apr 2025",
"30 Apr 2025",
"24 Apr 2025",
"21 Apr 2025",
"09 Apr 2025",
"02 Jun 2025",
"31 May 2025",
"30 May 2025",
"30 May 2025",
"30 May 2025",
"03 Jun 2025",
"28 May 2025",
"15 May 2025",
"12 May 2025",
"08 Apr 2025",
"27 Apr 2025",
"15 Apr 2025",
"12 Apr 2025",
"03 Jun 2025",
"29 May 2025",
"27 May 2025",
"26 May 2025",
"25 May 2025",
"25 May 2025",
"23 May 2025",
"19 May 2025",
"12 May 2025",
"06 May 2025",
"01 May 2025",
"29 Apr 2025",
"25 Apr 2025",
"24 Apr 2025",
"22 Apr 2025",
"21 Apr 2025",
"19 Apr 2025",
"17 Apr 2025",
"14 Apr 2025"
)

def web_search(query, config, query_id):
    end_date = end_dates[query_id % len(end_dates)] if query_id is not None else None
    print(query_id, end_date)
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
            while True:
                try:
                    df: pd.DataFrame = lotus_web_search(
                        query=query,
                        corpus=corpus,
                        K=config['search_top_k'],
                        sort_by_date=False
                    )
                    break
                except Exception as e:
                    print(f"Error in lotus_web_search: {e}")
                    time.sleep(1)
                    continue
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
            try:
                print(df.columns)
                if end_date and "date" in df.columns:
                    print("Initial length: ", len(df))
                    _end_date = datetime.strptime(end_date, "%d %b %Y")
                    df["_date"] = pd.to_datetime(df["date"], errors='coerce')
                    _end_date = _end_date - timedelta(days=1)
                    df = df[df["_date"].dt.date <= _end_date.date()]
                    df.drop(columns=["_date"], inplace=True)
                    print("Final length: ", len(df))
            except Exception as e:
                print(f"Error processing date: {e}")
                
            if len(df) > config['search_top_k']:
                df = df.head(config['search_top_k'])
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