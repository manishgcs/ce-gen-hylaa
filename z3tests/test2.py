from z3 import *

x = Int('x')
y = Int('y')

s = Solver()
print s

s.add(x > 10, y == x + 2)
print s
print "Solving constraints in the solver s ..."
print s.check()

print "Create a new scope..."
s.push()
s.add(y < 11)
print s
print "Solving updated set of constraints..."
print s.check()

print "Restoring state..."
s.pop()
print s
print "Solving restored set of constraints..."
print s.check()

x = Real('x')
s = Solver()
s.add(2**x == 3)
print s.check()

x = Real('x')
y = Real('y')
s = Solver()
s.add(x > 1, y > 1, Or(x + y > 3, x - y < 2))
print "asserted constraints..."
for c in s.assertions():
    print c

print s.check()
print "statistics for the last check method..."
#print s.statistics()
# Traversing statistics
#for k, v in s.statistics():
#    print "%s : %s" % (k, v)

x, y, z = Reals('x y z')
s = Solver()
s.add(x > 1, y > 1, x + y > 3, z - x < 10)
print s.check()

m = s.model()
print "x = %s" % m[x]

print "traversing model..."
for d in m.decls():
    print "%s = %s" % (d.name(), m[d])

x = Real('x')
y = Int('y')
a, b, c = Reals('a b c')
s, r = Ints('s r')
print x + y + 1 + (a + s)
print ToReal(y) + c

x, y = Reals('x y')
# Put expression in sum-of-monomials form
t = simplify((x + y)**3, som=True)
print t
# Use power operator
t = simplify(t, mul_to_power=True)
print t

x = Int('x')
y = Int('y')
f = Function('f', IntSort(), IntSort())
s = Solver()
s.add(f(f(x)) == x, f(x) == y, x != y)
print s.check()
m = s.model()
print m
print "f(f(x)) =", m.evaluate(f(f(x)))
print "f(x)    =", m.evaluate(f(x))

p, q = Bools('p q')
demorgan = And(p, q) == Not(Or(Not(p), Not(q)))
print demorgan

def prove(f):
    s = Solver()
    s.add(Not(f))
    if s.check() == unsat:
        print "proved"
    else:
        print "failed to prove"

print "Proving demorgan..."
prove(demorgan)