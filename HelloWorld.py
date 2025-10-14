import requests

x = requests.get("https://www.baidu.com")
print(x.text)