import urllib.request
import json
import ssl
import gzip
import io

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

jobs_url = 'https://api.github.com/repos/soumyashiv/asft/actions/runs/27879782140/jobs'
req = urllib.request.Request(jobs_url, headers={'User-Agent': 'Mozilla/5.0'})
res = urllib.request.urlopen(req, context=ctx)
jobs_data = json.loads(res.read().decode())
for j in jobs_data['jobs']:
    print(f"Job: {j['name']} - ID: {j['id']} - Conclusion: {j['conclusion']}")
    if j['conclusion'] == 'failure':
        # Get logs
        log_url = f"https://api.github.com/repos/soumyashiv/asft/actions/jobs/{j['id']}/logs"
        try:
            log_req = urllib.request.Request(log_url, headers={'User-Agent': 'Mozilla/5.0'})
            log_res = urllib.request.urlopen(log_req, context=ctx)
            logs = log_res.read().decode('utf-8')
            print("LOGS:")
            # Print the last 50 lines
            lines = logs.split('\n')
            for line in lines[-50:]:
                print(line)
            print("="*40)
        except Exception as e:
            print("Could not get logs:", e)
