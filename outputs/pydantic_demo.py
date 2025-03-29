from pydantic import BaseModel , Field
from typing import Optional


class student (BaseModel):
    name  : str = 'hari prasad'
    age   :  Optional[int] = None
    cgpa  : float =Field(gt=0,lt=10,description='this is of valus og student attained marks ')  # can apply constrains and default values  and also can add decription
    


new_student ={'age':'26','cgpa':9.9 }

student =student(**new_student)

student_dict =  (dict(student))
print(student_dict['age'])