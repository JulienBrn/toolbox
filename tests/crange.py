

from toolbox import crange

# for i in crange(0, 5):
x = crange(0, 5)
print(len(list(x)), len(x), list(x))

x = crange(0, -5, 5)
print(len(list(x)), len(x), list(x))

x = crange(0, -5, 5, step =2)
print(len(list(x)), len(x), list(x))

x = crange(0, -3, 5)
print(len(list(x)), len(x), list(x))

x = crange(0, -5, 3)
print(len(list(x)), len(x), list(x))

x = crange(2, 5)
print(len(list(x)), len(x), list(x))

x = crange(2, -6, 5)
print(len(list(x)), len(x), list(x))


