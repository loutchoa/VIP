# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 17:00:15 2012

@author: Francois Lauze
"""

import sys
sys.path.append('../ImageTools')


from Snakes import Snake
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import interpolate
from resample import resample
import math

class SnakeAnimator(object):
    """ Plot handles the Snake graphics. 

        The initial snake is created by clicking on some points in the  figure.
        left click to add a point, right click to finish. Extra-points will be
        inserted as specified the valued used in constructig the snake object.
        
        Then pressing a key other than ('r','R', 'q', 'Q') will iterate the
        snake.

        Pressing 'q' or 'Q' exits the snake program.
        Pressing 'r' or 'R' restarts the snake program.          
    """
    def __init__(self, snake, iters):
        self.snake = snake
        self.nx, self.ny = self.snake.im.shape
        self.fig, self.ax = plt.subplots()
        plt.gray()
        self.ShowNormalized([0,self.ny-1,0,self.nx-1])
        self.fig.canvas.mpl_connect('button_press_event', self.OnClick)
        self.fig.canvas.mpl_connect('key_press_event', self.OnKey)
        self.doInitCurve = True   
        self.iters = iters
        self.v = []   
        # start the event loop
        plt.show()
    # end method __init__()
       
       
    def ShowNormalized(self, extent):
        immin = self.snake.im.min()
        immax = self.snake.im.max()
        im = 255.0*(self.snake.im-immin)/(immax-immin)
        im = (im.round()).astype(np.uint8)
        plt.imshow(im, extent=extent)
        plt.axis('off')
        
       
    def OnClick(self, event):
        if self.doInitCurve:
            if event.button == 1:
                self.v.append((event.xdata, event.ydata))
                self.DrawRedCross([event.xdata, event.ydata])
            elif event.button == 3:
                self.doInitCurve = False
                self.InitCurve()
    # end method on_click()
                
            
    def OnKey(self, event):
        """ The key callback: only active after the initial curve has been produced.
        The command provided are 
        'q' 'Q': quit the program.
        'r', 'R': run again with a new initialization.
        any other key: compute the next iteration of the snake.
        """
        if not self.doInitCurve:
            if event.key in ['q', 'Q']:
                # I terminate the event loop by closing the figure
                print "Bye."
                plt.close()
            elif not event.key in ['r', 'R']:
                print "I will run next iteration"
                try:
                    v, veloc  = self.snakeEvol.next()
                    v = self.Mat2PlotCurve(v)
                    veloc = self.Mat2PlotVField(veloc)
                    self.DrawCurve(v)
                    self.DrawVeloc(v, veloc)
                except StopIteration:
                    print "I have done all my iterations"
                    # at the end of the iteration process
                except:
                    print "Something went wrong, check the traceback."
                    raise       
            else:
                # if restart and we are not selecting the 
                # initial curve, so clean up and restart.
                if not self.doInitCurve:
                    self.doInitCurve = True
                    self.ax.lines = []
                    self.v = []
                    self.fig.canvas.draw()
    # end method on_key()                
                
        
    def InitCurve(self):
        #print "Entering initcurve()"
        """ Compute the initial curve from the clicked points. I add an extra 
        point to close the curve, this is for spline interpolation. I will 
        remove it from the interpolant when done. Then  I create the generator
        object that will produce the snake iterations. """
        self.v = np.array(self.v)
        vx = self.v[:,0]
        vx = np.append(vx, vx[0])
        vy = self.v[:,1]
        vy = np.append(vy, vy[0])
        
        nbr_vertices = len(vx)
        t = np.linspace(0,nbr_vertices,nbr_vertices)
        sx = interpolate.splrep(t, vx, s= 0)
        sy = interpolate.splrep(t, vy, s= 0)
        tnew = np.linspace(0, nbr_vertices, self.snake.n + 1)
    
        vxnew = interpolate.splev(tnew, sx, der=0)
        vynew = interpolate.splev(tnew, sy, der=0)

        # I don't want the last point        
        vxnew = np.matrix(vxnew[:-1]).T
        vynew = np.matrix(vynew[:-1]).T
        self.v = np.hstack((vxnew, vynew))
        # resample for better equally spaced points
        self.v = resample(self.v)
        self.DrawCurve(self.v)
        self.snakeEvol = snake.CreateGenerator(self.Plot2MatCurve(self.v), self.iters)
    
    # end method initcurve()


    def DrawRedCross(self, pos):
        """ Draw a red cross at position (x,y) on figure axes. """
        x = pos[0]
        y = pos[1]
        self.ax.plot([x-3, x+3],[y, y], 'r-')
        self.ax.plot([x ,x],[y-3, y+3], 'r-')
        self.fig.canvas.draw()

    # end method draw_red_cross()
        

    def DrawCurve(self, v):
        """ Draw the curve represented by v. """
        # first clear the previous curve if any.        
        self.ax.lines = []
        x = np.array(v[:,0])
        y = np.array(v[:,1])

        cx = np.append(x, x[0])
        cy = np.append(y, y[0])
        self.ax.plot(cx, cy, 'r-', linewidth=2)
        self.fig.canvas.draw()

    # end method draw_curve()



    def DrawVeloc(self, v, veloc):
        """ Draw a vector field along v, reprersented by arrows. """
        l = 16.0
        c = -math.sqrt(3)/2.0
        s = 0.5   
        d = 0.8
        dx = []
        dy = []
        for i in range(len(veloc)):
            x0 = v[i,0]
            y0 = v[i,1]
            x1 = x0 + l*veloc[i,0]
            y1 = y0 + l*veloc[i,1]

            # The body of the arrow              
            dx += [x0, x1, np.nan]
            dy += [y0, y1, np.nan]
            
            # the arrow endpoint >             
            xs = (1-d)*x0 + d*x1
            ys = (1-d)*y0 + d*y1
            px = c*(x1-xs)  - s*(y1-ys) + x1;
            py = s*(x1-xs)  + c*(y1-ys) + y1
            qx = c*(x1-xs)  + s*(y1-ys) + x1
            qy = -s*(x1-xs) + c*(y1-ys) + y1
            dx += [x1, px, np.nan]
            dy += [y1, py, np.nan]
            dx += [x1, qx, np.nan]
            dy += [y1, qy, np.nan]
        self.ax.plot(dx, dy, 'g-')
        self.fig.canvas.draw()


    # conversions from plot to matrix coordinates
    def Plot2MatPoint(self, x, y):
        return self.nx-1-y, x
        
    def Mat2PlotPoint(self, i, j):
        return j, self.nx-1-i
        
    def Plot2MatCurve(self, v):
        w = np.array(v)
        matc = []
        for pt in w:
            matc.append(self.Plot2MatPoint(pt[0], pt[1]))
        return np.matrix(matc)
        
    def Mat2PlotCurve(self, v):
        w = np.array(v)
        plotc = []
        for pt in w:
            plotc.append(self.Mat2PlotPoint(pt[0], pt[1]))
        return np.matrix(plotc) 

    def Plot2MatVector(self, x, y):
        return -y,x


    def Mat2PlotVector(self,x,y):
        return y,-x
        
    def Plot2MatVField(self, v):
        w = np.array(v)
        plotc = []
        for vect in w:
            plotc.append(self.Plot2MatVField(vect[0],vect[1]))
        return np.matrix(plotc)
        
    def Mat2PlotVField(self, v):
        w = np.array(v)
        matc = []
        for vect in w:
            matc.append(self.Mat2PlotVector(vect[0],vect[1]))
        return np.matrix(matc)
        
    

# end class SnakeAnimator

            
if __name__ == "__main__":
    #im = np.array(Image.open('football.jpg').convert('L')).astype(np.float64)
    #im = np.array(Image.open('AT3_1m4_01.tif').convert('L')).astype(np.float64)
    im = np.array(Image.open('coins.png').convert('L')).astype(np.float64)
    
    # Filling-in gaps
    #im = np.zeros((200,200))
    #im[40:120,90:160] = 1.0
    #im[75:85,85:95] = 0.0
   
    snake = Snake(im, alpha=12.5, beta=0, gamma=10000, delta=2.5, tau=0.1, n=80, resample=20)
    #snake.set_extforces_Canny(sigma=3.0, interp='bilinear')
    SnakeAnimator(snake, 2000)
    
        
        
        
        
