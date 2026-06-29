import os

import wget


path = os.environ.get("DATA_ROOT", "data")
filename = "evel23.tar.gz"
url = "https://zenodo.org/record/7874573/files/eval23.tar.gz?download=1"

response = wget.download(url, os.path.join(path, filename))