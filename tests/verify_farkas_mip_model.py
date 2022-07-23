y = [0.0]
y.append(0.0)
y.append(0.0)
y.append(0.0)

y.append(9.99999450000275e-06)
y.append(6.499996700001652e-05)
y.append(9.99999450000275e-06)
y.append(9.99999450000275e-06)
y.append(5.499997250001376e-05)

y.append(0.0)
y.append(0.0)
y.append(0.0)

z = [1]
z.append(1)
z.append(0)

alpha = [7.0000021999989]
alpha.append(7.0)

print(- alpha[0] + 100000 * z[0])
print(alpha[0] + 100000 * z[0])
print(- alpha[1] + 100000 * z[0])
print(alpha[1] + 100000 * z[0])
print(- alpha[0] + 100000* z[1])
print(alpha[0] + 100000 * z[1])
print(- alpha[1] + 100000 * z[1])
print(alpha[1] + 100000 * z[1])
print(- alpha[0] + 100000 * z[2])
print(alpha[0] + 100000 * z[2])
print(- alpha[1] + 100000 * z[2])
print(alpha[1] + 100000 * z[2])
print(- y[0] + y[1] - y[4] + y[5] - y[8] + y[9])
print(- y[2] + y[3] - y[6] + y[7] - y[10] + y[11])
print(- 2 * y[0] + 10 * y[1] - 2 * y[2] + 10 * y[3] - 4 * y[4] + 9 * y[5] - 4 * y[6] + 9 * y[7] - 11 * y[8] + 12 * y[9] + 8 * y[11])
print(- 100000 * z[1] + y[4])