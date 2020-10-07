import time

query = [
	"""curl --header "Content-Type: application/json" --request POST --data '{"access":"0","start_time":""",
	  ""","endtime":""",
	  ""","place":"0","zone":"0","videostream":"0","objective":"0","active":1}' http://0.0.0.0:5000/api"""
]

print(query[0], int(time.time() + 100), query[1], int(time.time() + 200), query[2], sep='')


"""
curl --header "Content-Type: application/json" --request POST --data '{"access":"0","start_time":1601389057,"endtime":1601389157,"place":"0","zone":"0","videostream":"0","objective":"0","active":1}' http://0.0.0.0:5000/api
"""