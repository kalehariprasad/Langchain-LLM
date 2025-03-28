from typing import TypedDict


class person(TypedDict):
    name : str
    age  : int


new_person : person ={'name':'hari prasad' , 'age': '26 '}

print(new_person)
print(type(new_person))
