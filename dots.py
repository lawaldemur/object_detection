import requests


# send updated zone to 1C
url="https://db.1c-ksu.ru/VA_Prombez2/ws/ExchangeVideoserverPoints/ExchangeVideoserverPoints.1cws"
#headers = {'content-type': 'application/soap+xml'}
headers = {'content-type': 'text/xml'}
body = """
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" xmlns:c="http://www.1c.exchange-videoserver-points.serv.org" xmlns:c1="http://www.1c.exchange-videoserver-points.org">
   <soap:Header/>
   <soap:Body>
      <c:addVSpoints>
         <c:metadata>
            <!--Optional:-->
            <c1:idRequest>idRequest_1</c1:idRequest>
         </c:metadata>
         <c:data>
            <c1:pointX>123</c1:pointX>
            <c1:pointY>412</c1:pointY>
            <c1:width>250</c1:width>
            <c1:height>250</c1:height>
            <c1:idVideostream>fad1d66d-0c81-11eb-8133-00155d3c2b05</c1:idVideostream>
            <c1:idPlace>73f8413e-1918-11eb-ab52-00155d3c2f56</c1:idPlace>
            <!--Optional:-->
            <c1:zone>Холл</c1:zone>
         </c:data>
      </c:addVSpoints>
   </soap:Body>
</soap:Envelope>"""
body = body.encode(encoding='utf-8')

response = requests.post(url=url, data=body, headers=headers, auth=('WebServerVideo', 'Videoanalitika2020'))
print('sending zone coords result: ' + str(response.text))