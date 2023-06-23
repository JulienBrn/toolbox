

from toolbox import crange

x = crange(0, 5)
print(len(list(x)), len(x), list(x))

x = crange(0, 5, direction_right=False)
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


