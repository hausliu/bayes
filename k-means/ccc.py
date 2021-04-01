dic = {"deep": {"a": 1, 'b': 2}, "low": 2}
a = dic.copy()
dic["deep"]['b'] = 2333
print(dic)
print(a)
