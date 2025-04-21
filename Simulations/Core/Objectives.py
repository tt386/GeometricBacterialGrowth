import numpy as np

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.optimize import minimize

def MakeObjective_ClosedOpen_Saureus(Model_Closed,Model_Open,datas,time):
    """
    Creates the objective function for least-squares fitting of parameters for
    S.aureus, finding relavant parameters at the same time for opena and 
    closed systems

    ARGS:
    Model_Closed: a function from Models.py
        Model for the closed data
    Model_Open: a function from Models.py
        Model for the open dtaa
    datas: list of lists
        The Closed, Open Monoculture and Open Coculture data
    time: list of lists:
        The time points of the data lists in datas

    RETURNS:
    objective:
        A function to be minimised with least-squares
    """


    def objective(params):
        """
        The objective function to be used in a least-squares algorithm to 
        fit the params supplied.

        RETURNS:
            The residuals of the log of the models outputs and log of datas
        """

        alpha, b,I_closed,I_openMono,bI0,    g1_mono,g1_co,g2, T, X0_Closed,X0_OpenMono,X0_OpenCo = params

        #ICs are Clsed X(0), Open X(0), and Open Q(0

        #Closed, Monoculture
        ICs = [10**X0_Closed]
        params = (alpha,b,g1_mono,g2,T,I_closed)
        result_Closed = solve_ivp(Model_Closed,[0,time[0][-1]],ICs,args=tuple(params),
                t_eval=time[0])


        # Open, Monoculute
        params = (alpha,b+I_openMono,g1_mono)
        ICs = [10**X0_OpenMono]

        result_OpenMono = solve_ivp(Model_Open,[0,time[1][-1]],ICs,args=tuple(params),
                t_eval=time[1])




        #Open. from coculture
        params = (alpha,bI0,g1_co)
        ICs = [10**X0_OpenCo]

        result_OpenCo = solve_ivp(Model_Open,[0,time[2][-1]],ICs,args=tuple(params),
                t_eval=time[2])

        #Concatenate the output for easier comparison with the data
        output = np.concatenate((result_Closed.y[0],result_OpenMono.y[0],result_OpenCo.y[0]))
        output[output<0] = 1e-10
        
        #Compare log of the result due to the scales involved
        return np.log10(output) - np.log10(np.concatenate((datas[0],datas[1],datas[2])))

    return objective



def MakeObjective_ClosedOpen_Paeruginosa(Model_Closed,Model_Open,datas,time):

    def objective(params):
        alpha,b,IOpen,IClosed,g1,g2,T,Ex,Ec,X0_Closed,X0_Open = params


        #ICs are Clsed X(0), Open X(0), and Open Q(0)
        ICs = [10**X0_Closed]
        params= (alpha,b,g1,g2,T,IClosed)

        result_Closed = solve_ivp(Model_Closed,[0,time[0][-1]],ICs,args=tuple(params),
                t_eval=time[0])

        params = (alpha,b+IOpen,g1,Ex,Ec)
        ICs = [10**X0_Open,0]

        result_Open = solve_ivp(Model_Open,[0,time[1][-1]],ICs,args=tuple(params),
                t_eval=time[1])


        output = np.concatenate((result_Closed.y[0],result_Open.y[0]))
        output[output<0] = 1e-10

        return np.log10(output) - np.log10(np.concatenate((datas[0],datas[1])))

    return objective

