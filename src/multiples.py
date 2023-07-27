# Find the multiples of 14 and 8 upto 1000
multiples = []
for i in range(1, 1000):
    if i % 14 == 0 and i % 8 == 0:
        multiples.append(i)
print(multiples)
