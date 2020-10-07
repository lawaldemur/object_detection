import os, glob

list_of_files = glob.glob(os.getcwd() + '/detections/' + str(29) + '/*.jpeg')
print(*sorted(list_of_files), sep='\n')
latest_file = max(list_of_files, key=os.path.getctime)
