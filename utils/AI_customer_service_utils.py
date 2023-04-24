from opencc import OpenCC
import requests
import json

def translation_stw(text):
    cc = OpenCC('likr-s2twp')
    return cc.convert(text)

def fetch_url_response(url, retry=3):
    while retry:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"URL: {url} 連線失敗，原因: {e}")
        retry -= 1
    return None

def shorten_url(auth, token, name, url):
    response = requests.post(
        f"https://likr.io/api/short_url/create/{token}",
        headers={'Authorization': auth},
        data={'name': name, 'url': url}
    )
    response = json.loads(response.text)

    if response.get('code') == 200 and type(response.get('message')) == dict:
        return response.get('message').get('short_url')
    else:
        return None