"""
Author:   Gruppe 11
Member:   Paul Leonard Zens, Stefan Kaufmann
Date:     14.09.2022
Project:  Programmierübung Helirack

"""


# %%


import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

from libmimo import mimo_rnf, mimo_acker, poly_transition
from scipy.linalg import solve_continuous_are
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

# %%
class LinearizedSystem:
    def __init__(self,A,B,C,D,x_equi,u_equi,y_equi):
        self.A=A
        self.B=B
        self.C=C
        self.D=D
        self.x_equi=x_equi
        self.u_equi=u_equi
        self.y_equi=y_equi

    #Ackermannformel
    def acker(self,eigs):
        return mimo_acker(self.A,self.B,eigs)

    #Berechnung des Ausgangs
    def output(self,t,x,controller):
        #Regler auswerten (wichtig falls Durchgriff existiert)
        u = controller(t,x)        
        if x.ndim==1:
            y = self.C@(x-self.x_equi) + self.y_equi + self.D@(u-self.u_equi)
        else:
            x_equi=self.x_equi.reshape((self.x_equi.shape[0],1))
            u_equi=self.u_equi.reshape((self.u_equi.shape[0],1))
            y_equi=self.y_equi.reshape((self.y_equi.shape[0],1))
            y = self.C@(x-x_equi)+y_equi+self.D@(u-u_equi)
        return y

class DiscreteLinearizedSystem(LinearizedSystem):
    def __init__(self,A,B,C,D,x_equi,u_equi,y_equi,Ta):
        super().__init__(A,B,C,D,x_equi,u_equi,y_equi)
        
        self.A, self.B, self.C, self.D, self.Ta = cont2discrete((A,B,C,D),Ta)
        
    #Quadratisch optimaler Regler
    def lqr(self,Q,R,S):
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Reglerverstärkungen berechnet werden
        K_lqr=np.zeros((R.shape[0],Q.shape[0]))

        P = solve_discrete_are(self.A, self.B,Q, R , None, S)
        
        K_lqr = sla.inv(R)@self.B.transpose()@P
        
        return K_lqr
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

    def rest_to_rest_trajectory(self,ya,yb,N,kronecker,maxderi=None):
        return DiscreteFlatnessBasedTrajectory(self,ya,yb,N,kronecker,maxderi)

class ContinuousLinearizedSystem(LinearizedSystem):
    def __init__(self,A,B,C,D,x_equi,u_equi,y_equi):
        super().__init__(A,B,C,D,x_equi,u_equi,y_equi)
        
    def discretize(self,Ta):
        assert Ta>0
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        ##bitte anpassen
        return DiscreteLinearizedSystem(self.A,self.B,self.C,self.D,self.x_equi,self.u_equi,self.y_equi,Ta)
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

    #Quadratisch optimaler Regler
    def lqr(self,Q,R,S):
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Reglerverstärkungen berechnet werden
        K_lqr=np.zeros((R.shape[0],Q.shape[0]))

        P = solve_continuous_are(self.A, self.B,Q, R , None, S)
        
        K_lqr = sla.inv(R)@self.B.transpose()@P
        

        return K_lqr
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

    def rest_to_rest_trajectory(self,ya,yb,T,kronecker,maxderi=None):
        return ContinuousFlatnessBasedTrajectory(self,ya,yb,T,kronecker,maxderi)
        
    #linearisiertes Zustandsraumodell des Systems
    def model(self,t,x,controller): 
        # t: Zeit
        # Zustand
        # x[0]: Elevations-Winkel epsilon
        # x[1]: Azimutwinkelwinkelgeschwindigkeit \dot{\alpha}
        # x[2]: Winkelgeschwindigkeit \dot{\epsilon}
        # controller(t,x): Solldrehzahlen werden als Funktion übergeben

        dim_u=self.u_equi.shape[0]
        dim_x=self.x_equi.shape[0]
        dim_y=self.y_equi.shape[0]
        
        #Auslenkung aus Ruhelage
        x_rel=np.array(x).flatten()-self.x_equi

        #Eingang auswerten
        u=controller(t,x)
        u=np.array(u).flatten()
        
        #Auslenkung aus Ruhelage
        u_rel=u-self.u_equi
        x_rel=x_rel.flatten()

        #Zustand auspacken
        dx=(self.A@x_rel+self.B@u_rel).flatten()


        #Ableitungen zurückgeben
        return dx


class Heli:
    """Modell des Heliracks"""
    

    def __init__(self):
        self.Vmax=0.024
        self.J1=0.002 
        self.J2=0.012 
        self.J3=0.013 
        self.J4=1e-05
        self.J5=1e-05  
        self.c1=0.001       
        self.c2=0.001
        self.s1=8e-7
        self.s2=8e-7
        self.d1=0.272
        self.d2=0.272
        self.Ta=0.1

    def equilibrium(self,y):
        """  Berechnung des nichtlinearen Systemausgang  
        Params
         --------
        self:          alles Konstanten              
        y[0]:          Elevations-Winkel epsilon
        y[1]:          alpha_punkt,  Rotationsgeschwindigkeit des Auslegers um die  Vertikale
       
                
        Returns
        --------
        x_quer:          Zustände bei gegebener Ruhelage
        u_quer:          Eingänge bei gegebener Ruhelage                           
                
        """

        ## Elevationwinkel
        epsilon=y[0]

        ## Rotationsgeschwindigkeit des Auslegers um die  Vertikale
        dalpha=y[1]

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier die sollten die korrekten Ruhelagen in Abhängigkeit des zugehörigen Ausgangs berechnet werden
        # %%
        x=np.zeros((3,))
        u=np.zeros((2,))
        
        # %%
        """
        Implementierung der Gleichungen für die Ruhelagen (siehe Bericht Aufgabe 1b.)
        p,q ....  Hilfsgrößen für quadratische Kösungsformel --> es gibt mehr als eine richtige Lösung. Deshalb genügt es auch wenn wir weiters nur eine der beiden Lösungen betrachten.
        """
        q = -(self.Vmax*np.cos(epsilon)+0.5*(self.J2+self.J4)*np.sin(2*epsilon)*np.square(dalpha))/(self.d2*self.s2)
        p = -self.J4*np.sin(epsilon)*dalpha/(self.d2*self.s2)

        u[0] = np.sqrt(self.c1*dalpha/(self.d1*np.cos(epsilon)*self.s1)) 
        u[1] = np.abs(-p/2 + np.sqrt(np.square(p/2)-q))

        x[0] = epsilon
        x[1] = (self.J1+(self.J2+self.J4)*np.square(np.cos(epsilon)))*dalpha + self.J4*np.cos(epsilon)*u[1]
        x[2] = self.J5*u[0]


        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        return x,u
        
    def linearize(self,y_equi,debug=False):
        #Berechnung der Systemmatrizen des linearen Systems
        
        ##Ruhelage berechnen

        # Aufruf der Funktion equilibrium welche die Ruhelagen (Zustände und Eingänge) zurückgibt
        x_equi,u_equi=self.equilibrium(y_equi)
        

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier die sollten die korrekten Matrizen angegeben werden
        # %%
        A=np.zeros((3,3))
        B=np.zeros((3,2))
        C=np.zeros((2,3))
        D=np.zeros((2,2))
        # %%
        # Händisches Einsetzten der symbolischen Ableitung aus der "Hilfsdatei.py"
        epsilon = y_equi[0]
        dalpha  = y_equi[1]       

        c2eps = self.J1+(self.J2+self.J4)*np.square(np.cos(epsilon))    
  
        A[1,0] = (-self.J4*self.c1*u_equi[1]*(c2eps) + 2*self.c1*(self.J2+self.J4)*(self.J4*u_equi[1]*np.cos(epsilon) - x_equi[1])*np.cos(epsilon) - self.d1*self.s1*u_equi[0]**2*(self.J1 + (self.J2+self.J4)*np.cos(epsilon)**2)**2)*np.sin(epsilon)/(self.J1 + (self.J2+self.J4)*np.cos(epsilon)**2)**2
        A[2,0] = (self.J4*u_equi[1]*(c2eps)**2*(-self.J4*u_equi[1]*np.sin(epsilon)**2 + self.J4*u_equi[1] - x_equi[1]*np.cos(epsilon)) + self.Vmax*(c2eps)**3*np.sin(epsilon) - 0.5*(1 - np.cos(4*epsilon))*(self.J2 + self.J4)**2*(self.J4*u_equi[1]*np.cos(epsilon) - x_equi[1])**2 + 2*(c2eps)*(self.J2 + self.J4)*(self.J4*u_equi[1]*np.cos(epsilon) - x_equi[1])*(-3.0*self.J4*u_equi[1]*np.cos(epsilon)**3 + 2.5*self.J4*u_equi[1]*np.cos(epsilon) + x_equi[1]*np.cos(epsilon)**2 - 0.5*x_equi[1]))/(c2eps)**3
        A[1,1] = -self.c1/c2eps
        A[2,1] = (-self.J4*u_equi[1]*(c2eps)*np.sin(epsilon) +(self.J2 + self.J4)*(self.J4*u_equi[1]*np.cos(epsilon) - x_equi[1])*np.sin(2*epsilon))/(c2eps)**2
        A[0,2] = 1/(self.J3+self.J5)
        A[2,2] = -self.c2/(self.J3+self.J5)

        B[0,0] = -self.J5/(self.J3+self.J5)
        B[1,0] = 2*self.d1*self.s1*np.cos(epsilon)*u_equi[0]         
        B[2,0] = (self.J5*self.c2)/(self.J3 + self.J5) 
        B[1,1] = self.J4*self.c1*np.cos(epsilon)/c2eps
        B[2,1] = (self.J4*(c2eps)*(2*self.J4*u_equi[1]*np.cos(epsilon) - x_equi[1])*np.sin(epsilon) - self.J4*(self.J2 + self.J4)*(self.J4*u_equi[1]*np.cos(epsilon) - x_equi[1])*np.sin(2*epsilon)*np.cos(epsilon) + 2*self.d2*self.s2*u_equi[1]*(c2eps)**2)/(c2eps)**2



        C[0,0] = 1
        C[1,0] = (self.J4*u_equi[1]*c2eps - 2*(self.J2 + self.J4)*(self.J4*u_equi[1]*np.cos(epsilon) - x_equi[1])*np.cos(epsilon))*np.sin(epsilon)/np.square(c2eps)
        C[1,1] = 1/c2eps         

        D[1,1] = -self.J4*np.cos(epsilon)/c2eps
       
       

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        
        return ContinuousLinearizedSystem(A,B,C,D,x_equi,u_equi,y_equi)
    

    def verify_linearization(self,linear_model,eps_x=1e-6,eps_u=1e-6):
        """Compare linear state space model with its approximate Taylor linearization of non-linear model
        Parameters
        ----------
        linear_model : object of type LinearizedModel

        Returns
        -------
        nothing

        Notes
        -----
        Tay

        """
        A_approx=np.zeros_like(linear_model.A)
        B_approx=np.zeros_like(linear_model.B)
        C_approx=np.zeros_like(linear_model.C)
        D_approx=np.zeros_like(linear_model.D)
        x_equi=linear_model.x_equi
        u_equi=linear_model.u_equi
        dim_x=3
        dim_u=2
        dim_y=2

        if np.isscalar(eps_x):
            eps_x=np.ones((dim_x,))*eps_x
        if np.isscalar(eps_u):
            eps_u=np.ones((dim_u,))*eps_u

        for jj in range(dim_x):
            _fnu=lambda t,x:u_equi
            x_equi1=np.array(x_equi)
            x_equi2=np.array(x_equi)
            x_equi2[jj]+=eps_x[jj]
            x_equi1[jj]-=eps_x[jj]
            #print("Ruhelage:",x_equi)
            dx=(self.model(0,x_equi2,_fnu)-self.model(0,x_equi1,_fnu))/2/eps_x[jj]
            A_approx[:,jj]=dx
            dy=(self.output(0,x_equi2,_fnu)-self.output(0,x_equi1,_fnu))/2/eps_x[jj]
            C_approx[:,jj]=dy
            
        error_A=np.abs(linear_model.A-A_approx)
        idx= np.unravel_index(np.argmax(error_A, axis=None), error_A.shape)
        print("Maximaler absoluter Fehler in Matrix A Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_A[idx[0],idx[1]])+".")
        scale_A=np.hstack([np.where(abs(A_approx[:,jj:jj+1]) > eps_x[jj], abs(A_approx[:,jj:jj+1]), eps_x[jj]) for jj in range(dim_x)])
        error_rel_A=error_A/scale_A
        idx= np.unravel_index(np.argmax(error_rel_A, axis=None), error_rel_A.shape)
        print("Maximaler relativer Fehler in Matrix A Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_rel_A[idx[0],idx[1]])+".")

        error_C=np.abs(linear_model.C-C_approx)
        idx= np.unravel_index(np.argmax(error_C, axis=None), error_C.shape)
        print("Maximaler absoluter Fehler in Matrix C Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_C[idx[0],idx[1]])+".")
        scale_C=np.hstack([np.where(abs(C_approx[:,jj:jj+1]) > eps_x[jj], abs(C_approx[:,jj:jj+1]), eps_x[jj]) for jj in range(dim_x)])
        error_rel_C=error_C/scale_C
        idx= np.unravel_index(np.argmax(error_rel_C, axis=None), error_rel_C.shape)
        print("Maximaler relativer Fehler in Matrix C Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_rel_C[idx[0],idx[1]])+".")
        
        for jj in range(dim_u):
            url1=np.array(u_equi)
            url2=np.array(u_equi)
            eps_u[jj]=1e-6
            url1[jj]-=eps_u[jj]
            url2[jj]+=eps_u[jj]
            _fnu1=lambda t,x:url1
            _fnu2=lambda t,x:url2
            dx=(self.model(0,x_equi,_fnu2)-self.model(0,x_equi,_fnu1))/2/eps_u[jj]
            B_approx[:,jj]=dx
            dy=(self.output(0,x_equi,_fnu2)-self.output(0,x_equi,_fnu1))/2/eps_x[jj]
            D_approx[:,jj]=dy
        error_B=np.abs(linear_model.B-B_approx)
        idx= np.unravel_index(np.argmax(error_B, axis=None), error_B.shape)
        print("Maximaler absoluter Fehler in Matrix B Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_B[idx[0],idx[1]])+".")
        scale_B=np.hstack([np.where(abs(B_approx[:,jj:jj+1]) > eps_x[jj], abs(B_approx[:,jj:jj+1]), eps_x[jj]) for jj in range(dim_u)])
        error_rel_B=error_B/scale_B
        idx= np.unravel_index(np.argmax(error_rel_B, axis=None), error_rel_B.shape)
        print("Maximaler relativer Fehler in Matrix B Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_rel_B[idx[0],idx[1]])+".")

        error_D=np.abs(linear_model.D-D_approx)
        idx= np.unravel_index(np.argmax(error_D, axis=None), error_D.shape)
        print("Maximaler absoluter Fehler in Matrix D Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_D[idx[0],idx[1]])+".")
        scale_D=np.hstack([np.where(abs(D_approx[:,jj:jj+1]) > eps_x[jj], abs(D_approx[:,jj:jj+1]), eps_x[jj]) for jj in range(dim_u)])
        error_rel_D=error_D/scale_D
        idx= np.unravel_index(np.argmax(error_rel_D, axis=None), error_rel_D.shape)
        print("Maximaler relativer Fehler in Matrix D Zeile "+str(idx[0]+1)+", Spalte "+str(idx[1]+1)+" beträgt",str(error_rel_D[idx[0],idx[1]])+".")
        return A_approx, B_approx, C_approx

    #Umrechnung zwischen Winkelgeschwindigkeiten und Impulsen
    def velocities_to_momentums(self,q,dq,u):
        p=np.zeros_like(dq)
        bJ2=self.J2+self.J4
        bJ3=self.J3+self.J5
        if dq.ndim==1:
            p[0]=self.J5*u[1]+bJ3*dq[0]    
            p[1]=self.J4*np.cos(q[0])*u[0]+(self.J1+bJ2*np.cos(q[0])**2)*dq[1]
        else:
            p[0,:]=self.J5*u[1,:]+bJ3*dq[0,:]
            p[1,:]=self.J4*np.cos(q[0,:])*u[0,:]+(self.J1+bJ2*np.cos(q[0,:])**2)*dq[1,:]
        return p
    
    #Nichtlineares Zustandsraumodell des Heli-Racks
    def model(self,t,x,controller): 
        """  Berechnung der nichtlinearen Zustandsgleichung  
        Params
         --------
        self:            alles Konstanten
        x:               der Zustandsvektor     
        x[0]:            Elevations-Winkel epsilon
        x[1]:            Drehimpuls p_alpha
        x[2]:            Drehimpuls p_epsilon
        t:               Zeit
        controller(t,x): Solldrehzalen werden als Funktion übergeben
                
        Returns
        --------
        y_[0]:          Epsilon
        y_[1]:          Alpha_Punkt                                 
                
        """
        # t: Zeit
        # Zustand
        # x[0]:  Elevations-Winkel epsilon
        # x[1]:  Drehimpuls p_alpha
        # x[2]:  Drehimpuls p_epsilon
        # controller(t,x): Solldrehzalen werden als Funktion übergeben
      
        #Eingang auswerten
        u=controller(t,x).flatten()   
        assert np.shape(u)==(2,)

        #Zustand auspacken
        epsilon=x[0]
        p_alpha=x[1]
        p_epsilon=x[2]

        #Winkelfunktionen vorausberechnen
        ceps=np.cos(epsilon)
        seps=np.sin(epsilon)
    
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Ableitungen berechnet und zurückgegebenn werden

        # Hier ist die nichtlineare Zustandsgleichung implementiert, welche händisch (siehe Dokumentation) ermittelt wurde
        # Der Übersicht halber wurden Hilfsgrößen definiert
        dot_alpha = (p_alpha-self.J4*ceps*u[1])/(self.J1+(self.J4+self.J2)*np.square(ceps))
        dot_epsilon= (p_epsilon-self.J5*u[0])/(self.J3+self.J5)

        f_alpha = self.s1*np.abs(u[0])*u[0]       # Rotorkraft Alpha
        d_alpha = self.c1*dot_alpha               # Durch Reibung verursachtes Drehmoment in Alpha

        f_epsilon = self.s2*np.abs(u[1])*u[1]     # Rotorkraft Epsilon
        d_epsilon = self.c2*dot_epsilon           #  Durch Reibung verursachtes Drehmoment in Epsilon
        
        # Nicht lineare Zustandsdifferntialgleichung
        dot_epsilon= dot_epsilon 
        dot_p_alpha= self.d1*ceps*f_alpha - d_alpha
        dot_p_epsilon= -self.Vmax*ceps- 0.5*(self.J2+self.J4)*np.sin(2*epsilon)*np.square(dot_alpha) - self.J4*u[1]*seps*dot_alpha + self.d2*f_epsilon - d_epsilon

        dx=np.array([dot_epsilon,dot_p_alpha,dot_p_epsilon])

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return dx

    def output(self,t,x,controller):

        """  Berechnung des nichtlinearen Systemausgang  
        Params
         --------
        self:          alles Konstanten
        x:             der Zustandsvektor          
        x[0]:          Elevations-Winkel epsilon
        x[1]:          Drehimpuls p_alpha
        x[2]:          Drehimpuls p_epsilon
                
        Returns
        --------
        y_[0]:          Epsilon
        y_[1]:          Alpha_Punkt                            
                
        """

        #Berechnung des Systemausgangs
        #Eingang auswerten
        u=controller(t,x)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Ausgänge berechnet werden
        # Basis für die Ausgangsgleichung ist die in der Dokumentation ermittelten Ausgangsgleichung

        if np.isscalar(t): 
            y=np.zeros((2,))
            y[0] = x[0]
            y[1] = (x[1]-self.J4*np.cos(x[0])*u[1])/(self.J1+(self.J2+self.J4)*np.square(np.cos(x[0])))

        else:
            y=np.zeros((2,t.shape[0]))
            y[0,:] = x[0,:]
            y[1,:] = (x[1]-self.J4*np.cos(x[0])*u[1])/(self.J1+(self.J2+self.J4)*np.square(np.cos(x[0])))
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return y

    def generate_model_test_data(self,filename,count):
        import pickle
        x=np.random.rand(3,count)
        u=np.random.rand(2,count)
        dx=np.zeros_like(x)
        y=np.zeros_like(u)
        for ii in range(count):
            dx[:,ii]=self.model(0,x[:,ii],lambda t,x: u[:,ii])
            y[:,ii]=self.output(0,x[:,ii],lambda t,x: u[:,ii])
            
        with open(filename, 'wb') as f: 
            pickle.dump([x, u, dx, y], f)
        f.close()

    def verify_model(self,filename):
        import pickle
        with open(filename,'rb') as f:
            x, u, dx_load, y_load = pickle.load(f)
        f.close()
        dx=np.zeros_like(x)
        y=np.zeros_like(u)
        count=dx.shape[1]
        for ii in range(count):
            dx[:,ii]=self.model(0,x[:,ii],lambda t,x: u[:,ii])
            y[:,ii]=self.output(0,x[:,ii],lambda t,x: u[:,ii])
        error_dx_abs_max=np.max(np.linalg.norm(dx-dx_load,2,axis=0))
        error_dx_rel_max=np.max(np.linalg.norm(dx-dx_load,2,axis=0)/np.linalg.norm(dx_load,2,axis=0))
        error_y_abs_max=np.max(np.linalg.norm(y-y_load,2,axis=0))
        error_y_rel_max=np.max(np.linalg.norm(y-y_load,2,axis=0)/np.linalg.norm(y_load,2,axis=0))
        dx_load_max=np.max(np.linalg.norm(dx_load,2,axis=0))
        y_load_max=np.max(np.linalg.norm(y_load,2,axis=0))
        print("Maximaler absoluter Fehler in Modellgleichung (euklidische Norm):",error_dx_abs_max)
        print("Maximaler relativer Fehler in Modellgleichung (euklidische Norm):",error_dx_rel_max)
        print("Maximaler absoluter Fehler in Ausgangsgleichung (euklidische Norm):",error_y_abs_max)
        print("Maximaler relativer Fehler in Ausgangsgleichung (euklidische Norm):",error_y_rel_max)
        
    

class ContinuousFlatnessBasedTrajectory:
    """Zeitkontinuierliche flachheitsbasierte Trajektorien-Planung zum Arbeitspunktwechsel 
    für das lineare zeitkontinuierliche Modelle, die aus der Linearisierung im Arbeitspunkt abgeleitete worden sind.

    Args:
       ya, ye (numpy.array):  Anfangs- und Endwerte den Ausgang (absolut)
       T (float): Überführungszeit
       linearized_system: Entwurfsmodel
       kronecker: zu verwendende Steuerbarkeitsindizes
       maxderi: maximal Differenzierbarkeitsanforderungen für flachen Ausgang (None entspricht maxderi=kronecker)
    """
    def __init__(self,linearized_system,ya,yb,T,kronecker,maxderi=None):
        self.linearized_system=linearized_system
        self.T=T
        ya_rel=np.array(ya)-linearized_system.y_equi      
        yb_rel=np.array(yb)-linearized_system.y_equi
        self.kronecker=np.array(kronecker,dtype=int)
        if maxderi==None:
            self.maxderi=self.kronecker
        else:
            self.maxderi=self.maxderi
            
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier bitte benötigte Zeilen wieder "dekommentieren" und Rest löschen
        self.A_rnf, self.Brnf, self.Crnf, self.M, self.Q, self.S = mimo_rnf(linearized_system.A, linearized_system.B, linearized_system.C, kronecker)


        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        #Umrechnung stationäre Werte zwischen Ausgang und flachem Ausgang
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Anfangs und Endwerte für den flachen Ausgang berechnet werden
        #Achtung: Hier sollten alle werte relativ zum Arbeitspunkt angegeben werden

        """
        Für die stationären Werte werden lediglich eta 1 und 2, nicht jedoch deren Ableitungen betrachtet. Deshalb reduziert sich die Ausgansmatrix C auf zwei Spalten.
        y = C*eta + D,    D = 0
        diese wird auf eta aufgelöst

        eta = inv(C)*y    --> eta entspricht dabei einem Spaltenvektor mit zwei stationären flachen Ausgängen  [eta_1 , eta_2] für die jeweiligen Teilssysteme     

        """
        self.eta_a=np.zeros_like(ya_rel)
        self.eta_b=np.zeros_like(yb_rel)

        C_inv = np.linalg.inv(self.Crnf[:,0:2])

        self.eta_a = np.dot(C_inv,ya_rel)         
        self.eta_b = np.dot(C_inv,yb_rel) 

       

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########


    # Trajektorie des flachen Ausgangs
    def flat_output(self,t,index,derivative):
        tau = t / self.T
        if derivative==0:
            return self.eta_a[index] + (self.eta_b[index] - self.eta_a[index]) * poly_transition(tau,0,self.maxderi[index])
        else:
            return (self.eta_b[index] - self.eta_a[index]) * poly_transition(tau,derivative,self.maxderi[index])/self.T**derivative 

    #Zustandstrajektorie 
    def state(self,t):
        tv=np.atleast_1d(t)
        dim_u=np.size(self.linearized_system.u_equi)
        dim_x=np.size(self.linearized_system.x_equi)
        dim_t=np.size(tv)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
   
        state=np.zeros((dim_x,dim_t))
        # Calc eta
        """   
        der flache Ausgang besteht aus dem jetzigen Wert eta + der Änderung. Die Änderung hängt nun von der Ordnung der Ableitung von eta ab.
        im zweiten Schritt xrnf = eta wir eta als Spaltenvektor dargestellt. In der RNF ist der erste Zustand = dem flachen Ausgang, da das Teilsystem somit nur mehr von eta, dessen Ableitungen und dem Eingang abhängt.
        """
        eta=list()
        for index in range(dim_u):
            eta=eta+[self.flat_output(tv,index,deri) for deri in range(self.kronecker[index])]
        xrnf=np.vstack(eta)
        """   
        nun wird eta in die Orginalkoordinaten Transformiert und die Ruhelage dazugerechnet, da wir bis jetzt nur die Differenz von eta nicht jedoch die Absolutwerte betrachtet haben.
        Die Transformationsforschrift kann dem Skriptum S.79   x_quer = Q*x   entnommen werden
        """
        # Transform from RNF to original Koordinates
        state=(np.linalg.inv(self.Q)@xrnf)+self.linearized_system.x_equi.reshape((dim_x,1))
        if (np.isscalar(t)):
            state=state[:,0]
        

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return state

    #Ausgangstrajektorie
    def output(self,t):
        tv=np.atleast_1d(t)
        dim_u=np.size(self.linearized_system.u_equi)
        dim_y=np.size(self.linearized_system.y_equi)
        dim_x=np.size(self.linearized_system.x_equi)

        x_abs=self.state(tv)
        u_abs=self.input(tv)
        x_rel=x_abs-self.linearized_system.x_equi.reshape((dim_x,1))
        u_rel=u_abs-self.linearized_system.u_equi.reshape((dim_u,1))
        y_rel=self.linearized_system.C@x_rel+self.linearized_system.D@u_rel
        y_abs=y_rel+self.linearized_system.y_equi.reshape((dim_y,1))
        if (np.isscalar(t)):
            y_abs=y_abs[:,0]
        return y_abs

    #Eingangstrajektorie
    def input(self,t):
        tv=np.atleast_1d(t)
        dim_u=np.size(self.linearized_system.u_equi)
        eta=list()
        for index in range(dim_u):
            eta=eta+[self.flat_output(tv,index,deri) for deri in range(self.kronecker[index])]
        xrnf=np.vstack(eta)
        v=-self.A_rnf[self.kronecker.cumsum()-1,:]@xrnf
        for jj in range(self.kronecker.shape[0]):
            v[jj,:]+=self.flat_output(tv,jj,self.kronecker[jj])
        result=(np.linalg.inv(self.M)@v)+self.linearized_system.u_equi.reshape((dim_u,1))
        if (np.isscalar(t)):
            result=result[:,0]
        return result

class DiscreteFlatnessBasedTrajectory:
    """Zeitdiskrete flachheitsbasierte Trajektorien-Planung zum Arbeitspunktwechsel 
    für lineare Modelle, die aus der Linearisierung im Arbeitspunkt abgeleitet worden sind.

    Args:
       ya, ye (numpy.array):  Anfangs- und Endwerte den Ausgang (absolut)
       T (float): Überführungszeit
       linearized_system: zeitdiskretes Entwurfsmodel
       kronecker: zu verwendende Steuerbarkeitsindizes
       maxderi: maximale Differenzierbarkeitsanforderungen für die Komponenten des flachen Ausgangs (bei None werden die Kronecker-Indizes gewählt; maxderi=kronecker)
    """
    def __init__(self,linearized_system,ya,yb,N,kronecker,maxderi=None):
        self.linearized_system=linearized_system
        self.N=N

        dim_u=np.size(linearized_system.u_equi)
        dim_x=np.size(linearized_system.x_equi)

        #Abstand von der Ruhelage berechnen
        ya_rel=np.array(ya)-linearized_system.y_equi
        yb_rel=np.array(yb)-linearized_system.y_equi
        self.kronecker=np.array(kronecker,dtype=int)

        #Glattheitsanforderungen an Trajektorie
        if maxderi==None:
            self.maxderi=self.kronecker
        else:
            self.maxderi=self.maxderi
            
        #Matrizen der Regelungsnormalform holen
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier bitte benötigte Zeilen wieder "einkommentieren" und Rest löschen
        self.A_rnf, self.Brnf, self.Crnf, self.M, self.Q, S = mimo_rnf(linearized_system.A, linearized_system.B, linearized_system.C, self.kronecker)

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        #Umrechnung stationäre Werte zwischen Ausgang und flachem Ausgang

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Anfangs und Endwerte für den flachen Ausgang berechnet werden
        #Achtung: Hier sollten alle Werte relativ zum Arbeitspunkt angegeben werden


        """
        Für die stationären Werte müssen eta_1 eta_2 und um 1 verschobenen eta_2 betrachtet werden.  Die Ausgngsmatrix kann auf zwei Splalten reduziert werden, wenn man die letzten zwei Spalten
        Komponentenweise addiert. Beide sind eta_2. Damit ergibt sich die Form wie im Skriptum S.56 
        y = C*eta + D,    D = 0
        diese wird auf eta aufgelöst

        eta = inv(C)*y    --> eta entspricht dabei einem Spaltenvektor mit zwei stationären flachen Ausgängen  [eta_1 , eta_2] für die jeweiligen Teilssysteme     

        """

        self.eta_a=np.zeros_like(ya_rel)
        self.eta_b=np.zeros_like(yb_rel)

        C_discrete = self.Crnf[:,0:2] + 0
        C_discrete[:,1] = self.Crnf[:,1] + self.Crnf[:,2] 

        C_inv = np.linalg.inv(C_discrete)            

        self.eta_a =  np.dot(C_inv,ya_rel)        
        self.eta_b =  np.dot(C_inv,yb_rel) 

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        
    def flat_output(self,k,index,shift=0):
        """Berechnet zeitdiskret die Trajektorie des flachen Ausgangs

        Parameter:
        ----------
        # k: diskrete Zeitpunkte als Vektor oder Skalar
        # shift: Linksverschiebung der Trajektorie
        # index: Komponente des flachen Ausgangs"""

        ######-------!!!!!!Aufgabe!!!!!!-------------########
        #Hier sollten die korrekten Werte für die um "shift" nach links verschobene Trajektorie des flachen Ausgangs zurückgegeben werden          
        # 
        """
        Die Implementierung von eta kann dem Skriptum Seite 56-58 entnommen werden. 
        Zur Implementierung des Zeitverlaufs ist es nötig für eta_2 bei k+1 eine linksverschiebung zu bewirken. Da im diskreten Fall die Ableitung des flachen Ausgangs einer Linksverschiebung der Funktion gleichkommt. 
        Dies wird in tau folgendermaßen implementiert:  tau = (k - maxderi[i] + shift)/ (N-maxder[i])     k entspricht hierbei dem aktuellen Zeitpunkt, maxderi[i] der Maximalen Verschiebung für jeden flachen Ausgang seperat und shift die Verschiebung selbst.         
        """    
        tau = (k-self.maxderi[index]+shift)/(self.N-self.maxderi[index])     # jede Ableitung von der Trajektorie ist um eine Zeiteinheit nach links Verschoben  --> Skriptum S.56-58  
        
        eta= np.zeros_like(k)
        
        eta = self.eta_a[index] + (self.eta_b[index] - self.eta_a[index]) * poly_transition(tau,0,self.maxderi[index])
        
        return eta

        ######-------!!!!!!Aufgabe Ende!!!!!!-------########

        

    #Zustandstrajektorie 
    def state(self,k):
        kv=np.atleast_1d(k)
        dim_u=np.size(self.linearized_system.u_equi)
        dim_x=np.size(self.linearized_system.x_equi)
        dim_k=np.size(k)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        state=np.zeros((dim_x,dim_k))
        eta=list()
        """   
        der flache Ausgang besteht aus dem jetzigen Wert eta + der Änderung. Die Änderung hängt nun von der Linksverschiebung ab.
        im zweiten Schritt xrnf = eta wir eta als Spaltenvektor dargestellt. In der RNF ist der erste Zustand = dem flachen Ausgang, da das Teilsystem somit nur mehr von eta, dessen Ableitungen und dem Eingang abhängt.
        """
        for index in range(dim_u):
            eta=eta+[self.flat_output(kv,index,shift) for shift in range(self.kronecker[index])]
        xrnf=np.vstack(eta)
        # Transform from RNF to original Koordinates
        """   
        nun wird eta in die Orginalkoordinaten Transformiert und die Ruhelage dazugerechnet, da wir bis jetzt nur die Differenz von eta nicht jedoch die Absolutwerte betrachtet haben.
        Die Transformationsforschrift kann dem Skriptum S.79   x_quer = Q*x   entnommen werden
        """
        state=(np.linalg.inv(self.Q)@xrnf)+self.linearized_system.x_equi.reshape((dim_x,1))
        if (np.isscalar(k)):
            state=state[:,0]
        
        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        return state

    #Ausgangstrajektorie
    def output(self,k):
        kv=np.atleast_1d(k)
        dim_y=np.size(self.linearized_system.y_equi)
        dim_x=np.size(self.linearized_system.x_equi)
        dim_u=np.size(self.linearized_system.u_equi)

        x_abs=self.state(kv)
        u_abs=self.input(kv)
        x_rel=x_abs-self.linearized_system.x_equi.reshape((dim_x,1))
        u_rel=u_abs-self.linearized_system.u_equi.reshape((dim_u,1))
        y_rel=self.linearized_system.C@x_rel+self.linearized_system.D@u_rel
        y_abs=y_rel+self.linearized_system.y_equi.reshape((dim_y,1))
        if (np.isscalar(k)):
            y_abs=y_abs[:,0]
        return y_abs

    #Eingangstrajektorie
    def input(self,k):
        kv=np.atleast_1d(k)
        dim_u=np.size(self.linearized_system.u_equi)
        dim_k=np.size(kv)
        ######-------!!!!!!Aufgabe!!!!!!-------------########
        result=np.zeros((dim_u,dim_k))
        """   
        der flache Ausgang besteht aus dem jetzigen Wert eta + der Änderung. Die Änderung hängt nun von der Linksverschiebung ab.
        im zweiten Schritt xrnf = eta wir eta als Spaltenvektor dargestellt. In der RNF ist der erste Zustand = dem flachen Ausgang, da das Teilsystem somit nur mehr von eta, dessen Ableitungen und dem Eingang abhängt.
        """
        eta=list()
        for index in range(dim_u):
            eta=eta+[self.flat_output(kv,index,shift) for shift in range(self.kronecker[index])]
        xrnf=np.vstack(eta)
        """   
        Anschließend wir der Hilfseingang v als Funktion von eta definiert --> siehe Skriptum S. 79
        die Spalten von v werden in einem zweiten Schritt zusammengezählt. Alternativ kann mann dies auch als Matrix Vektor Multiplikation geschrieben werden.
        im letzten Schritt wird noch die Ruhelage addiert.
        """
        v=-self.A_rnf[self.kronecker.cumsum()-1,:]@xrnf
        for jj in range(self.kronecker.shape[0]):
            v[jj,:]+=self.flat_output(kv,jj,self.kronecker[jj])
        
        result=  (np.linalg.inv(self.M)@v)  + self.linearized_system.u_equi.reshape((dim_u,1))



        ######-------!!!!!!Aufgabe Ende!!!!!!-------########
        if (np.isscalar(k)):
            result=result[:,0]
        return result


def plot_results(t,x,u,y):
    plt.figure(figsize=(15,7))
    plt.subplot(2,2,1,ylabel="Winkel $\\epsilon$ in Grad")
    plt.grid()
    leg=["Soll","Ist"]
    for v in y:
        plt.plot(t,v[0,:]/np.pi*180)
    plt.legend(leg)
    plt.subplot(2,2,2,ylabel="Winkelgeschwindigkeit $\\dot{\\alpha}$")
    plt.grid()
    for v in y:
        plt.plot(t,v[1,:]/np.pi*180)
    leg=["Soll","Ist"]
    plt.subplot(2,2,3,ylabel="Eingang 1 1/s")
    plt.grid()
    for v in u:
        plt.plot(t,v[0,:])
    plt.legend(leg)
    plt.subplot(2,2,4,ylabel="Eingang 2 1/s")
    plt.grid()
    for v in u:
        plt.plot(t,v[1,:])
    plt.legend(leg)





