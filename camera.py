import time
import os, glob


class Camera(object):
    def get_frame(self, id):
    	list_of_files = sorted(glob.glob(os.getcwd() + '/detections/' + str(id) + '/*.jpeg'))
    	if len(list_of_files) < 2:
    		return open(list_of_files[-0], 'rb').read()
    	else:
        	return open(list_of_files[-2], 'rb').read()
