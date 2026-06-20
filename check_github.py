import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

try:
    url = 'https://api.github.com/repos/soumyashiv/asft/actions/runs?per_page=1'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    res = urllib.request.urlopen(req, context=ctx)
    data = json.loads(res.read().decode())
    
    run_id = data['workflow_runs'][0]['id']
    print("Latest Run ID:", run_id)
    print("Run Status:", data['workflow_runs'][0]['status'])
    print("Run Conclusion:", data['workflow_runs'][0]['conclusion'])
    
    jobs_url = f"https://api.github.com/repos/soumyashiv/asft/actions/runs/{run_id}/jobs"
    jobs_req = urllib.request.Request(jobs_url, headers={'User-Agent': 'Mozilla/5.0'})
    jobs_res = urllib.request.urlopen(jobs_req, context=ctx)
    jobs_data = json.loads(jobs_res.read().decode())
    
    print("\nJobs:")
    for j in jobs_data['jobs']:
        print(f"Job: {j['name']} - Status: {j['status']} - Conclusion: {j['conclusion']}")
        if j['conclusion'] == 'failure':
            print("Failed steps:")
            for s in j['steps']:
                if s['conclusion'] == 'failure':
                    print(f"  - {s['name']}")
except Exception as e:
    print(e)
