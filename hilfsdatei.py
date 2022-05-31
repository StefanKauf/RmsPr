#%%
import sympy as sym
import numpy as np


# Aufgabe 1.4 Liniarisierung


epsilon, p_epsilon, p_alpha, J1, J2, J3, J4, J5,c1,c2, d1,d2,s1,s2, u1, u2, Vmax = sym.symbols('epsilon, p_epsilon, p_alpha, J1, J2, J3, J4, J5, c1,c2, d1,d2,s1,s2, u1, u2, Vmax')

# %%
# Eingangsgleichung
ceps = sym.cos(epsilon)
seps = sym.sin(epsilon)

dot_alpha = (p_alpha-J4*ceps*u2)/(J1+(J4+J2)*np.square(ceps))
dot_epsilon= (p_epsilon-J5*u1)/(J3+J5)

f_alpha = s1*u1**2      
d_alpha = c1*dot_alpha

f_epsilon = s2*u2**2
d_epsilon = c2*dot_epsilon
        
        
dot_epsilon= dot_epsilon 
dot_p_alpha= d1*ceps*f_alpha - d_alpha
dot_p_epsilon= -Vmax*ceps- 0.5*(J2+J4)*sym.sin(2*epsilon)*dot_alpha**2 - J4*u2*seps*dot_alpha + d2*f_epsilon - c2*(p_epsilon-J5*u1)/(J3+J5)

dx=sym.Matrix([dot_epsilon,dot_p_alpha,dot_p_epsilon])

# Ableitungen nach den Zuständen

a = sym.diff(dx, epsilon)
b = sym.diff(dx, p_alpha)
c = sym.diff(dx, p_epsilon)
d = sym.diff(dx, u1)
e = sym.diff(dx, u2)

 #%%
a = sym.simplify(a)
b = sym.simplify(b)
c = sym.simplify(c)
d = sym.simplify(d)
e = sym.simplify(e)

A = [a,b,c]
B = [d,e]


#%%
# Ausgangsgleichung

y1 = epsilon
y2  = dot_alpha

y = sym.Matrix([y1,y2])


a = sym.diff(y, epsilon)
b = sym.diff(y, p_alpha)
c = sym.diff(y, p_epsilon)
d = sym.diff(y, u1)
e = sym.diff(y, u2)

a = sym.simplify(a)
b = sym.simplify(b)
c = sym.simplify(c)
d = sym.simplify(d)
e = sym.simplify(e)

C = [a,b,c]
D = [d,e]










# %%
