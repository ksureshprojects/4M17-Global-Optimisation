import numpy as np
import matplotlib.pyplot as plt

class TS:

    def __init__(self, obj, x0, intensify=10, diversify=15,
                 red=25, stm_l=7, mtm_l=4, step=1, 
                 step_red=0.5, seed=1, tol=0.0001):
        """
        Class attribute initialisations.
        
        Parameters:
        obj - Reference to objective function
        x0 - Initial input value
        intensify - Steps without decrease before search 
                    intensification
        diversify - Steps without decrease before search 
                    diversification
        red - Steps without decrease before step 
              reduction
        stm_l - Size of short term memory
        mtm_l - Size of long term memory
        step - Initial step size
        step_red - Factor for step size reduction
        seed - Random number generator seed
        tol - Stopping condition on step size
        """
        self.r_uni = np.random.RandomState(seed=seed).uniform
        self.x0 = x0
        self.obj = obj
        # Search stage parameters
        self.I = intensify
        self.D = diversify
        self.R = red
        # Search stage counters
        self.curr_i = 0
        self.curr_d = 0
        self.curr_r = 0
        # Memory references
        self.STM = None
        self.MTM = None
        self.LTM = None
        # Length of memory
        self.stm_l = stm_l
        self.mtm_l = mtm_l
        # Number of block regions
        # per control variable
        self.ltm_s = 2
        # Step and step decrement factor
        self.step = step
        self.step_red = step_red
        # Objective function evaluation counter
        self.its = 0
        # Current search stage
        self.stage = 0
        # Stopping condition
        self.tol = tol
        # Reference for array with all visted points
        self.bases = None
        

    def obj_wrap(self, x):
        """
        Increments objective evaluations counter, after
        objective evaluation.

        Parameters:
        x - Objective input.
        Updates:
        self.its - Class attribute that records no. 
                   of evaluations.
        Returns:
        Objective fn result at x.
        """
        self.its += 1
        return self.obj(x)

    def search_intensification(self):
        """
        Changes search stage and updates base.

        Updates:
        self.stage - Search stage
        Returns:
        base - Average of best results found so far.
        """
        self.stage = 1
        return np.mean(self.MTM[:,:-1], axis=0, 
                       keepdims=True).T

    def search_diversification(self):
        """
        Changes search stage and updates base.

        Updates:
        self.stage - Search stage
        Returns:
        base - Random in area that has been least explored.
        """
        self.stage = 2
        # Find index of least explored region,
        # express in binary
        b = bin(np.argmin(self.LTM))[2:]
        # Pad binary number
        if len(b) < self.x0.shape[0]:
            pad = self.x0.shape[0] - len(b)
            b = ''.join(['0' for i in range(pad)]) + b
        # Store digits in column vector
        d = len(b)
        b = np.array(list(b), dtype='int').reshape((d,1))
        # Generate random positive vector
        base = self.r_uni(low=0, high=2, size=(d,1))
        # Transform vector into appropriate region
        return base * (b * 2 - 1)

    def check_status(self, base):
        """
        Checks whether search stage needs to be changed.

        Parameters:
        base - Last accepted objective input.
        Returns:
        base - Same if no change in stage otherwise updated.
        """
        change = False
        # Trigger intensification
        if self.curr_i == self.I:
            self.curr_i = 0
            base = self.search_intensification()
            change = True
        # Trigger diversification
        elif self.curr_d == self.D:
            self.curr_d = 0
            base = self.search_diversification()
            change = True
        # Trigger step reduction
        elif self.curr_r == self.R:
            self.curr_r = 0
            # Start from best point found so far
            base = self.MTM[[-1], :-1].T
            self.update_STM(base)
            self.update_LTM(base)
            self.step = self.step_red * self.step

        if change:
            curr_obj = self.obj_wrap(base)
            self.update_MTM(base, curr_obj)
            self.update_STM(base)
            self.update_LTM(base)
            self.bases = np.block([[self.bases],[base.T, curr_obj]])

        return base

    def best_allowed(self, base):
        """
        Evaluates best possible update.

        Parameters:
        x - Last objective input.
        Returns:
        Updated x or None if no update possible.
        """
        x = base.copy()
        var_opt = None
        dim = x.shape[0]
        for i in range(dim):
            # Plus increment
            x[i] += self.step
            curr_obj = self.obj_wrap(x)
            # Check update feasible, obj improved
            # new point in STM, before accepting
            if (curr_obj and 
                not np.isclose(x.T, self.STM).all(axis=1).any()):
                if var_opt is None:
                    var_opt = (i, self.step, curr_obj)
                elif var_opt[2] > curr_obj:
                    var_opt = (i, self.step, curr_obj)

            
            # Minus increment
            x[i] -= 2 * self.step
            curr_obj = self.obj_wrap(x)
            # Check update feasible, obj improved
            # new point in STM, before accepting
            if (curr_obj and 
                not np.isclose(x.T, self.STM).all(axis=1).any()):
                if var_opt is None:
                    var_opt = (i, -self.step, curr_obj)
                elif var_opt[2] > curr_obj:
                    var_opt = (i, -self.step, curr_obj)
            
            # Restore to original value
            x[i] += self.step
        
        if var_opt:
            x[var_opt[0]] += var_opt[1]
            return x, var_opt[2]
        else:
            return None

    def update_base(self, base, x, curr_obj):
        """
        Evaluates a pattern move.

        Parameters:
        base - Last base point.
        x - Proposed new point
        curr_obj - objective at new point
        Updates:
        base - Current stage of search (local,
                     intensified, diversified).
        obj - Counters recording propensity to
                     change stage.
        """
        pattern = x - base + x
        new_obj = self.obj_wrap(pattern)
        if new_obj is not None and new_obj < curr_obj:
            return pattern, new_obj
        else:
            return x, curr_obj

    def update_status(self, action):
        """
        Updates status.

        Parameters:
        action - 0 if new best found, 1 otherwise.
        Updates:
        self.stage - Current stage of search (local,
                     intensified, diversified).
        self.curr_ - Counters recording propensity to
                     change stage.
        """
        if action:
            if self.stage == 0:
                self.curr_i += 1
            elif self.stage == 1:
                self.curr_d += 1
            elif self.stage == 2:
                self.curr_r += 1
        else:
            self.curr_i = self.curr_d = self.curr_r = 0
            self.stage = 0

    def update_MTM(self, x, curr_obj):
        """
        Updates MTM.

        Parameters:
        x - Last objective input.
        curr_obj - Objective value for last input.
        Updates:
        self.MTM - Matrix with best seen locations and 
                   corresponding objectives.
        """
        # Check if row exists in MTM
        m = np.block([[x.T, curr_obj]])
        if np.isclose(self.MTM, m).all(axis=1).any():
            self.update_status(1)
            return None
        
        i = 0
        rows = self.MTM.shape[0]
        # Find index of first row that current
        # point is not better than
        while (i < rows and
               curr_obj < self.MTM[i, -1]):
               i += 1
        # If currently MTM is not full
        if rows < self.mtm_l:
            self.MTM = np.block([[self.MTM[:i, :]],
                                 [x.T, curr_obj],
                                 [self.MTM[i:, :]]])
            if i == rows:
                # New best found
                self.update_status(0)
            else:
                self.update_status(1)
        # If currently MTM is full
        elif i == 0:
            self.update_status(1)
        else:
            # Drop off first row to add current point
            self.MTM = np.block([[self.MTM[1:i, :]],
                                 [x.T, curr_obj],
                                 [self.MTM[i:, :]]])
            if i == rows:
                # New best found
                self.update_status(0)
            else:
                self.update_status(1)
        
        return None
        
    def update_STM(self, x):
        """
        Updates STM.

        Parameters:
        x - Last objective input.
        curr_obj - Objective value for last input.
        Updates:
        self.STM - Matrix with most recently visited locations.
        """
        if self.STM.shape[0] < self.stm_l:
            self.STM = np.block([[self.STM],[x.T]])
        else:
            self.STM = np.block([[self.STM[1:,:]],[x.T]])


    def update_LTM(self, base):
        """
        Updates long term memory.

        Parameters:
        base - Last objective input.
        Updates:
        self.LTM - Array where each element represents
                   'd' digit binary number which 
                   corresponds to +/- areas of each
                   control variable.
        """
        # Generate binary number corresponding to sign
        # of each input variable.
        b = ((base > 0).astype(int)).astype(str).flatten()
        # Update frequency of relevant area.
        self.LTM[int(''.join(b), 2)] += 1

    def advance_stage(self):
        """
        Advances search stage if local search
        finds no acceptable location.
        """
        if self.stage == 0:
            self.curr_i = self.I
        elif self.stage == 1:
            self.curr_d = self.D
        elif self.stage == 2:
            self.curr_r == self.R

    def run(self, maxit=10000):
        """
        Initiates tabu search.

        Returns:
        best location, minimum objective value,
        all visited places + objective results.
        """
        base = self.x0.copy()
        dim = base.shape[0]
        curr_obj = self.obj_wrap(base)
        self.its = 0
        self.STM = np.block([[base.T]])
        self.MTM = np.block([[base.T, curr_obj]])
        self.LTM = np.zeros(self.ltm_s ** dim)
        self.bases = np.block([[base.T, curr_obj]])
        while self.its < maxit:
            base = self.check_status(base)
            result = self.best_allowed(base)
            if result is None:
                self.advance_stage()
            else:
                x, curr_obj = result
                base, curr_obj = self.update_base(base, x, curr_obj)
                self.update_MTM(base, curr_obj)
                self.update_STM(base)
                self.update_LTM(base)
                self.bases = np.block([[self.bases],[base.T, curr_obj]])

            if self.step < self.tol:
                break

        return (self.MTM[[-1], :-1], self.MTM[-1, -1], self.bases, 
               None, self.its)


        
        


        




