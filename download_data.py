import os
import tarfile

import gdown


def main():
    url = ("https://drive.google.com/u/0/uc?"
    "id=1ZcKZ1is0VEkY9kNfPxIG19qEIqHE5LIO&export=download"
    )
    path = "data"

    archive_path = path + ".tar.bz2"

    os.makedirs(path, exist_ok=True)
    gdown.cached_download(url, archive_path)

    with tarfile.open(archive_path, "r:bz2") as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, os.path.dirname(archive_path))

    os.remove(archive_path)

if __name__ == "__main__":
    main()
