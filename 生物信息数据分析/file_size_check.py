import os
import sys
def get_file_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:    
                total_size += os.path.getsize(filepath)
            except PermissionError:
                pass
    return total_size

c_drive_path = sys.argv[1]
file_size = get_file_size(c_drive_path)

print(f"The total size of files in C drive is: {file_size} bytes.")

