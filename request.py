import time

query = [
	"""curl --header "Content-Type: application/json" --request POST --data '{"access":"0","start_time":""",
	  ""","endtime":""",
	  ""","place":"0","controlplace":0,"zone":"0","activezone":0,"videostream":"0","videostreamid":"0","regulationid":"0","objective":"0","bodyguard":"","active":1}' http://0.0.0.0:5000/api"""
]

print(query[0], int(time.time() + 100), query[1], int(time.time() + 200), query[2], sep='')
