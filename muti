class Student:
    def __init__(self, name, age, major):
        self.name = name
        self.age = age
        self.major = major

    def __str__(self):
        return f"Student Name: {self.name}, Age: {self.age}, Major: {self.major}"

    def get_name(self):
        return self.name

    def get_teacher(self):
        return Teacher()

class Teacher:
    def __init__(self, name, subject, years_of_experience):
        self.name = name
        self.subject = subject
        self.years_of_experience = years_of_experience

    def __str__(self):
        return f"Teacher Name: {self.name}, Subject: {self.subject}, Years of Experience: {self.years_of_experience}"

    def get_school(self):
        return School()

    def get_name(self):
        return self.name

class School:
    def __init__(self, name):
        self.name = name
        self.students = []
        self.teachers = []

    def add_student(self, student):
        self.students.append(student)

    def add_teacher(self, teacher):
        self.teachers.append(teacher)

    def get_name(self):
        return self.name

    def __str__(self):
        students_info = "\n".join(str(student) for student in self.students)
        teachers_info = "\n".join(str(teacher) for teacher in self.teachers)
        return f"School Name: {self.name}\nStudents:\n{students_info}\nTeachers:\n{teachers_info}"


school = School()
school.get_name()
