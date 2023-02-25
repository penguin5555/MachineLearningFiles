w = open("weights.txt", "r+")
w.truncate(0)

weights1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
weights2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
w.write(str(weights1))
w.write("\n")
w.write(str(weights2))
w.close()

w = open("weights.txt", "r+")
read = w.readlines()
x = []
for i in range(len(list(read[0]))):
    y = list(read[0])[i]
    print(y)
    if y != "' '":
        if y != "''":
            if y != "'":
                if y != ",":
                    x.append(list(read[0])[i])

print(x)
