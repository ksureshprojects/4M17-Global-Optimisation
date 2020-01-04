import numpy as np
import matplotlib.pyplot as plt

class Schubert:
    def __init__(self, var):
        self.v = var
        
    def eval_obj(self, x):
        """
        Method that evaluates specific version of Schubert's
        function.
        
        Parameters:
        x - Input at which to evaluate function. Vector of dim
            (self.v, n), where n > 0.
            
        Returns:
        Value of objective function at input if input valid. 
        Returns None otherwise.
        """
        # Check input dimensions valid
        if len(x.shape) != 2:
            raise ValueError('Given vector does not have \
                             the right dimensions.')
        
        # Initialise variable for objective function value
        obj = np.zeros(x.shape[1])
        for i in range(x.shape[0]):
            
            # Check input in feasible region
            if (x[i] < -2).all() or (x[i] > 2).all():
                return None
            
            # Update objective function variable
            for j in range(1,6):
                obj += j * np.sin((j + 1) * x[i] + j)
        
        return obj
    
    def plot(self, opt=None, title=None, save=None):
        """
        Create contour plot of 2D schubert function.
        Parameters:
        opt - Matrix of points visited. x coordinates
              in first row, y coordinates in second
        title - Title string for plot
        save - Filename to save plot
        """
        if self.v != 2:
            raise ValueError("Can't plot in more \
                             than 2 dimensions")
        
        # Create contour plot
        x_coord = np.linspace(-2, 2, 300)
        y_coord = np.linspace(-2, 2, 300)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        XY = np.block([[X_coord.flatten()], 
                       [Y_coord.flatten()]])
        Z_coord = self.eval_obj(XY)
        plt.figure()
        
        plt.contourf(X_coord, Y_coord, 
                     Z_coord.reshape((300,300)), levels=10)
        plt.colorbar()
        # Add optimisation intermediate points.
        if opt is not None and opt.shape[0] == 2:
            plt.scatter(opt[0], opt[1], marker='x',color='r')
            
        plt.title(title)
        plt.xlabel('x1')
        plt.ylabel('x2')
        if save is not None:
            plt.savefig(save)
        plt.show()
