import requests

url = 'http://localhost:5000/data-science-intern-api'
r = requests.post(url,json={'Python (out of 3)':2, 'R Programming (out of 3)':1, 'Data Science (out of 3)':3, 'type_of_degree_PG':0,'type_of_degree_UG':1,'ML':1,'DL':1,'NLP':1,'SM':0,'AWS':0,'SQL':0,'NoSQL':0,'Excel':0,'year':2})
print(r.json())