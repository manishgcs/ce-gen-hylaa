from z3 import *
x = Int('x')
y = Int('y')
solve(x > 2, y < 10, x + 2*y == 7)

print simplify(x + y + 2*x + 3)
print simplify(x < y + x + 2)
print simplify(And(x + 1 >= 3, x**2 + x**2 + y**2 + 2 >= 5))
print x**2 + y**2 >= 1
set_option(html_mode=False)
print x**2 + y**2 >= 1
n1 = x + y >= 3
n2 = x + 2*y == 7
n = And(n1, n2)
print "num args: ", n.num_args()
print "children: ", n.children()
print "1st child:", n.arg(0)
print "2nd child:", n.arg(1)
print "operator: ", n.decl()
print "op name:  ", n.decl().name()

x = Real('x')
y = Real('y')
solve(x**2 + y**2 > 3, x**3 + y < 5)
#set_option(precision=30)
print "Solving, and displaying result with 30 decimal places"
solve(x**2 + y**2 == 3, x**3 == 2)

print 1/3
print RealVal(1)/3
print Q(1, 3)

x = Real('x')
print x + 1/3
print x + Q(1,3)
print x + "1/3"
print x + RealVal(1)/3
set_option(rational_to_decimal=True)
solve(x + 0.25 == 3, x < 0)

p = Bool('p')
q = Bool('q')
r = Bool('r')
solve(Implies(p, q), r == Not(q), Or(Not(p), r))

p = Bool('p')
q = Bool('q')
print And(p, q, True)
print simplify(And(p, q, True))
print simplify(And(p, False))
print simplify(And(p, Or(q, False), True))

p = Bool('p')
x = Real('x')
solve(Or(x > 10, x < 6), Or(p, x**2 == 2), Not(p))