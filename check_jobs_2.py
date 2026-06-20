import urllib.request
import json
import ssl
import time

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = 'https://api.github.com/repos/soumyashiv/asft/actions/runs/27880223843'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

res = urllib.request.urlopen(req, context=ctx)
data = json.loads(res.read().decode())
print(f"Status: {data['status']}, Conclusion: {data['conclusion']}")

jobs_url = f"https://api.github.com/repos/soumyashiv/asft/actions/runs/27880223843/jobs"
jobs_req = urllib.request.Request(jobs_url, headers={'User-Agent': 'Mozilla/5.0'})
jobs_res = urllib.request.urlopen(jobs_req, context=ctx)
jobs_data = json.loads(jobs_res.read().decode())

for j in jobs_data['jobs']:
    print(f"Job: {j['name']} - Conclusion: {j['conclusion']}")
