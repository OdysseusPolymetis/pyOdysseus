from setuptools import setup, find_packages

from setuptools import setup, find_packages
from pathlib import Path

def read_requirements(filename="requirements.txt"):
    print("Installation des modules")
    req_path = Path(__file__).parent / filename
    if not req_path.exists():
        return []
    reqs = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # ignore options pip (ex: -f, --extra-index-url) si tu en as
        if line.startswith("-"):
            continue
        reqs.append(line)
    return reqs

setup(
    name='Bertalign',
    version='0.1.0',
    url='https://github.com/bfsujason/bertalign',
    description='An automatic mulitlingual sentence aligner.',
    packages=find_packages(),    
    install_requires=read_requirements(),
)
