import time

query = [
	"""curl --header "Content-Type: application/json" --request POST --data '{"access":"0","start_time":""",
	  ""","endtime":""",
	  ""","place":"0","controlplace":0,"zone":"Холл в центре завода","activezone":0,"videostream":"0","videostreamid":"00000000-f41c-11ea-a41e-4cedfb43b7af","regulationid":"00000000-0c81-11eb-8133-00155d3c2b05","objective":"00000000-f41c-11ea-a41e-4cedfb43b7af","bodyguard":["123","825"],"active":1,"pointx":0,"pointy":0,"width":0,"height":0}' http://0.0.0.0:5001/api"""
]

print(query[0], int(time.time() + 100), query[1], int(time.time() + 200), query[2], sep='')
