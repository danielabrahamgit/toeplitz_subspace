from pathlib import Path
import subprocess

import gdown

def download():
    folder_url = 'https://drive.google.com/drive/folders/12IiApk8485EnmA0oe9doif9dg1oOxFOL'
    datapath = Path('./data')
    if not datapath.is_dir():
        # Download and extract
        gdown.download_folder(folder_url, quiet=False, use_cookies=False)
        subprocess.Popen('unzip mrf_2d_data/data.zip', shell=True).wait()
        subprocess.Popen('rm -rf mrf_2d_data', shell=True).wait()
    else:
        print(f'{datapath} already exists')

if __name__ == '__main__':
    download()
