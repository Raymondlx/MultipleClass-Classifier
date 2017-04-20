
students = ['mike','raymond','saki']
courses = [1.,2.,3.]

tmp = zip(students, courses)
print tmp
c= dict((y, x) for x, y in tmp)
print c

d = {courses[i]:students[i] for i in range(len(courses))}
print d