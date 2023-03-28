import subprocess,shlex
for tmp in ['one','two']:
    subprocess.run('{0} {1}'.format('echo', tmp),shell=True).



