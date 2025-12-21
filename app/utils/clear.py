import shutil
from config import *


def clear_static_folder():
    """
    Deletes all files in static folders except for '.gitkeep'.
    Subdirectories and their contents inside the folders are also removed.
    The main directories themselves are preserved.
    """
    folders = [RESULT_DIR, UPLOAD_DIR, LABELS_DIR, RESULT_JSON_DIR]

    for folder in folders:
        for item in folder.iterdir():
            if item.is_file() and item.name != ".gitkeep":
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
