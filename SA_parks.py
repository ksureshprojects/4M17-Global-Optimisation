import numpy as np
import matplotlib.pyplot as plt

class SA_parks:
    def __init__(self, obj, x0, chi=0.8, alpha=0.1, seed=1, 
                 omega=2.1, d_low=0.0000001, d_max=5):
        """
        Class attribute initialisations.
        
        Parameters:
        obj - Reference to objective function
        x0 - Initial input value
        chi - Target average probability, used to calculate T0
        alpha - Step size controller
        seed - Randomizer seed.
        Updates:
        self.r_uni - Random vector generator with given seed
        self.T - Current annealing temperature
        self.a - Alpha parameter
        self.o - Omega parameter
        self.obj - Objective function reference
        self.x0 - Initial input
        self.avg_p - Target initial increase acceptance probability
        self.d_low - Minimium value of any element in update matrix D
        self.d_max - Maximum value of any element in update matrix D
        self.i - Counter to record number of objective evaluations
        self.p - Variable to record current acceptance probability
        """
        self.r_uni = np.random.RandomState(seed=seed).uniform
        self.T = None
        self.a = alpha
        self.o = omega
        self.obj = obj
        self.x0 = x0
        self.avg_p = chi
        self.d_low = d_low
        self.d_max = d_max
        self.i = 0
        self.p = None
        
    def eval_inc(self, df, d):
        """
        Function to evaluate whether objective increase
        should be accepted.
        
        Parameters:
        df - Change in objective function.
        T - Current temperature.
        d - Actual step size
        Returns:
        True is increase accepted, False if rejected
        """
        # Check temperature valid
        if self.T <= 0 or self.T is None:
            raise ValueError("T must be positive")
            
        # Calculate probability
        if df >= 0 and self.T * d > 0:
            self.p = np.exp(-df/(self.T * d))
        else:
            print(self.a)
            print(df)
            print(self.T * d)
            raise ValueError('All parameters not evaluated \
                              successfully')
        
        # Accept if rand no. less than p
        if self.r_uni() < self.p:
            return True
        else:
            return False
        
    def run(self, maxit=10000, temp_dec=0.95, L=100, 
            N=30, c_tol=0.1):
        """
        Run simulated annealing algorithm.

        Parameters:
        maxit - Maximum number of objective function
                evaluations
        temp_dec - Temperature decrement factor
        L - Maximum length of annealing sequence at
            a given temperature
        N - Length of short term memory
        c_tol - Defines stopping condition on 
                average of short term memory
        Returns:
        x_opt - Optimal input
        o_min - Minimum objective function value
        results_loc - Intermediate inputs
        results_obj - Intermediate objective evaluations
        """
        # Initialise min acceptances limit
        n_min = 0.6 * L
        # Initialise temperature
        x, o_min, D, D_min, D_max = self.initialise_T()
        # Dimension of input vectors
        d = x.shape[0]
        # Current optimal input found
        x_opt = x.copy()
        # Current optimal objective value
        o_c = o_min.copy()
        # Counter to record chain length 
        # at current temperature
        L_count = 0
        # Counter to record acceptances in
        # chain at current temperature
        acc_count = 0
        # Record all intermediate inputs and
        # objective evaluations.
        results_loc = x.copy()
        results_obj = o_c.copy()
        # Short term memory record of absolute 
        # values for last N changes
        df_rec = np.array([])
        # Average Change
        d_a = None
        while self.i < maxit:
            # Generate random vector u, calculate
            # update vector r.
            u = self.r_uni(low=-1, high=1, size=(d,1))
            r = np.dot(D, u)
            # Update x
            x_ = x + r
            # Evaluate objective
            o_n = self.obj(x_)
            # Check feasibility
            if o_n is not None:
                # Calculate change in objective
                df =  o_n - o_c
                # Actual step size
                d_e = np.sum(np.square(r)) ** 0.5
                # Check if update to be accepted
                if (df < 0).all() or self.eval_inc(df[0], d_e):
                    # Update short term memory
                    if df_rec.shape[0] < N:
                        df_rec = np.block([df_rec, df])
                    else:
                        df_rec = np.block([df_rec[1:], df])
                    # Average change
                    d_a = np.mean(np.abs(df_rec))
                    # Check if best seen
                    if o_n < o_min:
                        x_opt = x_.copy()
                        o_min = o_n.copy()
                    # Accept decrease, 
                    # sometimes accept increase
                    x = x_.copy()
                    o_c = o_n.copy()
                    # Store intermeadiate results
                    results_loc = np.block([[results_loc, x]])
                    results_obj = np.block([[results_obj, o_c]])
                    # Increment acceptances in chain
                    acc_count += 1
                    # Update D
                    D = ((1 - self.a) * D + 
                         self.a * self.o * np.diag(np.abs(r.flatten())))
                    D = np.maximum(D,D_min)
                    D = np.minimum(D, D_max)
                # Increment chain length count
                L_count += 1
                # Reset chain, decrment temperature
                if L_count == L or acc_count > n_min:
                    L_count = 0
                    acc_count = 0
                    self.T = self.T * temp_dec
            self.i += 1
            if d_a is not None and d_a < c_tol:
                break

        
        return x_opt, o_min, results_loc, results_obj, self.i
                
            
        
    def initialise_T(self, runs=20):
        """
        Initilise T value by evaluating average increase
        over set number of updates.
        
        Parameters:
        runs - Number of increases over which to calculate
               average increase in obj function.
        Returns:
        x_opt - Optimal input
        o_min - Minimum objective function value
        D - Current update matrix
        D_min - Update matrix lower limit
        D_max - Update matrix upper limit
        """
        # Store initial value
        x = self.x0.copy()
        # Record vector dimension
        d = x.shape[0]
        # Initialise update matrix and 
        # limiting matrices
        D = np.eye(x.shape[0])
        D_min = D * self.d_low
        D_max = D * self.d_max
        # Intialise variables to count
        # increases and total objective increase
        inc_count = 0
        df_tot = 0
        # Variables to record current objective,
        # min objective found so far and corresponding
        # input vector
        o_c = self.obj(x)
        o_min = o_c.copy()
        x_opt = x.copy()
        # Calculate results of series of updates
        while inc_count < runs:
            # Generate random vector u, calculate
            # update vector r.
            u = self.r_uni(low=-1, high=1, size=(d,1))
            r = np.dot(D, u)
            # Update x
            x_ = x + r
            # Evaluate objective
            o_n = self.obj(x_)
            if o_n is not None:
                # Calculate change in objective
                df =  o_n - o_c
                # Record objective fn increases
                if (df > 0).all():
                    inc_count += 1
                    df_tot += df
                else:
                    # Update D only for decreases
                    D = ((1 - self.a) * D + 
                         self.a * self.o * np.diag(np.abs(r.flatten())))
                    D = np.maximum(D,D_min)
                    D = np.minimum(D, D_max)
                # Update optimal solution
                if (o_n < o_min).all():
                    x_opt = x_.copy()
                    o_min = o_n.copy()
                x = x_.copy()
                o_c = o_n.copy()
            self.i += 1
                
        
        # Calculate and store initial temp
        if 0 < self.avg_p < 1:
            self.T = -(df_tot/inc_count)/np.log(self.avg_p)
        else:
            raise ValueError("Target p0 parameter invalid.")
        
        return x_opt, o_min, D, D_min, D_max