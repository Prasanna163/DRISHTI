import urllib.request
import re
import os

url = 'https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html'
html = urllib.request.urlopen(url).read().decode('utf-8')

# Search for any sector 3 or 4 files
links = re.findall(r'href="(/(?:missions|tess)/[^"]+-s000[34][^"]+\.csv)"', html)
print(f"Found {len(links)} matching links:")
print(links)

base_url = 'https://archive.stsci.edu'
ref_dir = r'e:\Prasanna\ISRO\data\Ref'
os.makedirs(ref_dir, exist_ok=True)

for link in links:
    full_url = base_url + link
    filename = link.split('/')[-1]
    dest = os.path.join(ref_dir, filename)
    if os.path.exists(dest):
        print(f"Skipping {filename}, already exists.")
        continue
    print(f'Downloading {filename}...')
    try:
        urllib.request.urlretrieve(full_url, dest)
        print(f'Saved to {dest}')
    except Exception as e:
        print(f'Error downloading {filename}: {e}')
