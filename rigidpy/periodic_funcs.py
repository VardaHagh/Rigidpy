from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from .periodic_framework import Periodic_Framework
from .periodic_configuration import Periodic_Configuration
import time

class Point(object):
    """
    class for representing a single point in high dimension.

    Parameters:
    ----------
    coordinates: sequence of d floats
        Point coordinates in d dimension.
    edgelist: tuple or array
        Edge list
    pins: array
        pins list

    Returns
    -------
    The class does not return anything.
    """

    def __init__(self, coordinates, edges, a1, a2, L, k=1, optimization=False):

        # Ctreate an empty dictionary
        self.info = d = {}

        coordinates = np.array(coordinates)
        N,dimension = coordinates.shape
        C=Periodic_Configuration(dim=dimension)
        '''optimize structure'''
        if optimization:
            coordinates = C.energy_minimize(coordinates, edges, a1, a2, L, k=1)
            d['report'] = C.report
            d['energy'] = C.finalenergy
        else:
            d['energy'] = C.initialenergy
        d['coordinates'] = coordinates

        '''create a framework'''
        xhat = np.tile([1,0],N)/np.sqrt(N)
        yhat = np.tile([0,1],N)/np.sqrt(N)
        PF = Periodic_Framework(coordinates, edges, a1, a2, dimension)
        evalues, evectors = PF.Eigenspace()

        e1 = evectors[0]
        beta = np.dot(e1,xhat)
        gamma = np.dot(e1,yhat)

        e = e1 - beta*xhat - gamma*yhat
        e = e / np.linalg.norm(e)


        d['eigenvalue']=evalues[0]
        d['eigenvector']= e.reshape(-1,dimension)
        d['direction'] = e.reshape(-1,dimension)

def TotalDisplacement(coordinates1, coordinates2):
    '''
    Compute sum of displacement for all sites
    '''
    if coordinates1.shape == coordinates2.shape:
        dists = np.linalg.norm(coordinates1-coordinates2,axis=1)
        return np.sum(dists)
    else:
        print ('The size of two arrays do not match!')

def NextPoint(coordinates, edges, a1, a2, direction, stepsize, L, k=1, threshold=0.99, optimization=False):
    '''
    This function determines the optimal step size,
    so the trackor remains on the path.

    Parameters
    ----------
    coordinates: array
        coordinates
    direction: array
        direction of motion, a normalized vector
    threshold:
        Minimum value of dot product between two sunsequent steps.


    Returns
    -------
    stepsize: float
    optimal step size
    '''
    # find if next point is suitable
    scale = stepsize
    nextp = coordinates + scale * direction

    nextpoint = Point(coordinates=nextp, edges=edges, a1 = a1, a2 = a2, L=L, k=1, optimization=optimization)
    test = np.abs(np.vdot(direction,nextpoint.info['direction']))
    '''If dot product is less than threshold, decrease
    the step size until it is larger.'''
    while test < threshold:
        scale *= 0.5
        if scale < 10**(-10): break
        nextp = coordinates + scale * direction
        nextpoint = Point(coordinates=nextp, edges=edges, a1 = a1, a2 = a2, L=L, k=1, optimization=optimization)
        test = np.abs(np.vdot(direction,nextpoint.info['direction']))
    return nextpoint

def CircuitFollower(coordinates, edges, a1, a2, bondnumber,
                     L, stepsize = 10**-3, iteration =1000000, radius=0.1,
                     relax_step=5, report=True, optimization=False,
                     lazy_search=False,kick=1,cent=None):
    '''
    circuit_follower removes the requested
    bond from the network and follows the
    direction of the eigenvector with the
    lowest eigenvalue (=zero).

    Parameters
    ----------
    coordinates:
    edges:
    pins:
    bondnumber: int, which bond to cut
    L: rest length of sprins, size should match the length of edges
    stepsize:
    iteration:
    report: boolean
    optimization: boolean, optional
        whther system being optimized, default is True.

    a : array_like
        Input array
    n : int, optional
        The number of times values are differenced.
    axis : int, optional
        The axis along which the difference is taken, default is the last axis.
    lazy_search: boolean
    If True, stop after finding first solution, and don't complete the circle.
    Returns
    -------
    diff : ndarray
        The `n` order differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`.

    To-Do list:
    ---------
    1) Make it possible to provide two ends of a bond to cut,
    instead of passing the index of bond.
    '''
    time0 = time.time()

    '''
    check whether bondnumber is
    given or the bond itself
    '''
    if isinstance(bondnumber, tuple):
        pass
    elif isinstance(bondnumber, int):
        pass
    else:
        pass

    # two ends of the removed bond
    a,b = edges[bondnumber]
    N,dimension = coordinates.shape # N: number of sites
    coordinates = np.array(coordinates,dtype=np.float64)
    p = np.copy(coordinates) #working copy

    '''
    make a working copy from edges,
    mask the requested element
    '''
    edges = np.array(edges,int)
    edgelist = np.copy(edges,int)
    mask=np.ones(len(edgelist),dtype=bool)
    mask[bondnumber]= False
    edgelist =  edgelist[mask]

    '''rest length of springs'''
    mask_lens = np.ones(len(edges),dtype=bool)
    mask_lens[bondnumber] = False
    L = L[mask_lens]

    '''center of mass of system'''
    if cent.all() == None:
        center = np.mean(coordinates,axis = 0)
    else:
        center = cent

    '''containers to collect data'''
    data={}
    data['datas']=stepsize
    datax=[] # removed edge length
    datay=[] # total displacement
    dataw=[] # distance from center of mass
    datap=[] # coordinates of all sites
    datae=[] # energy at each step after relaxation
    data['nsteps'] = iteration # number of steps to take

    for i in range(0,iteration):
        print(i)
        # find current direction
        point = Point(coordinates=p, edges=edgelist, a1=a1, a2=a2, L=L, k=1)

        translation_vec = point.info['direction']


        # make sure we move forward
        if i==0:
            if kick==1:
                translation_vec_old = translation_vec
            else:
                translation_vec_old = -translation_vec
        projection = np.vdot(translation_vec,translation_vec_old)
        direction = np.sign(projection) * translation_vec
        translation_vec_old = direction


        # find the length of cut edge
        #length = point.info['lengths'][bondnumber]
        delta = np.abs(p[a]-p[b])
        box = np.array([a1[0],a2[1]])
        delta = np.where(delta > 0.5 * box, delta - box, delta)
        length = np.sqrt((delta ** 2).sum(axis=-1))

        # add to dic, to return later
        datap.append(p)
        datax.append(length)
        datay.append(TotalDisplacement(coordinates,p))
        w = np.mean(np.linalg.norm(p - center,axis =1))
        dataw.append(w)

        if i==0:
            nextp=NextPoint(p, edgelist, a1, a2, direction, stepsize, L=L, k=1, threshold=0.99, optimization=False)

        else:
            if i%relax_step==0:
                nextp=NextPoint(p, edgelist, a1, a2, direction, stepsize, L=L, k=1, threshold=0.99, optimization=True)
            else:
                nextp=NextPoint(p, edgelist, a1, a2, direction, stepsize, L=L, k=1, threshold=0.99, optimization=False)


        #nextp=NextPoint(p, edgelist, pins, direction, stepsize, L=L0, k=1, threshold=0.90, optimization=optimization)
        datae.append(nextp.info['energy'])

        '''if path wants to repeat itself, break the loop.The
        idea is to check when previous step gets closer to initial
        point but next step gets far. This does happen only close
        near starting point. To in addition we check if the mid point
        is closer than a cutoff to initial point. The cutoff is chosen
        as 5 times of stepsize.
        '''
        dist1 = np.linalg.norm(datap[i-1]-coordinates)
        dist2 = np.linalg.norm(p-coordinates)
        dist3 = np.linalg.norm(nextp.info['coordinates']-coordinates)
        turning = np.vdot(dist2-dist1,dist3-dist2)
        if (i!=0) and turning<0 and dist2<radius:
            print ('Tracking stopped. The last point was closer than radius from the starting point!')
            break
        p = nextp.info['coordinates']
        if lazy_search:
            passage = DetectPassagePoints(datax)
            if len(passage)>1: break

    data['nsteps']=i+1
    time1=time.time()
    # report and collect
    if report:
        print ('Removed bond connects node {:<5d} to node {:<5d}'.format(a+1,b+1))
        print ('Bond number supplied: {:<5d}'.format(bondnumber))
        print ('Total time={:<.4f} seconds'.format(time1-time0))

    data['datax']=np.array(datax)
    data['datap']=np.array(datap)
    data['dataw']=dataw
    data['datae']=np.array(datae)
    data['bond']=(a,b)
    data['datay'] = np.array(datay)

    return data

def DetectPassagePoints(datax):
    '''
    This function determines if the length
    of the cut edge returns to its original
    length.
    '''
    lens = datax
    arr  = lens - lens[0]
    # find when sign flips
    mask = np.diff(np.sign(arr))!=0
    return np.nonzero(mask)[0]

def PlotRealization(results,save=False):
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    ax.ticklabel_format(useOffset=False)
    ax.set_ylabel('Average Distance from Center of Mass')
    ax.set_xlabel('Length of the removed edge')

    datax,datay=results['datax'],results['dataw']
    ax.plot(datax,datay,'-',color='k',zorder=1)
    ax.scatter(datax[0],datay[0],marker='*',color='r',zorder=2,s=200)
    conjugate = DetectPassagePoints(results['datax'])
    for item in conjugate:
        ax.scatter(datax[item],datay[item],marker='*',color='r',zorder=2,s=200)

    #now = datetime.datetime.now()
    #variables = map(str,[results['bond'],np.mean(lave),results['nsteps']])
    #vars_strings = "_".join(variables)
    #name = 'trihex_circuit_'+vars_strings+'.pdf'
    #plt.title(name)
    plt.tight_layout()
    if save:
        plt.savefig('./results_circuit/'+name,dpi=100)
        plt.close()
    else:
        return plt.show(fig)

def DotProduct(results,save=False):
    nsteps = results['nsteps']
    m,n = results['datap'][0].shape
    P = np.array(results['datap']).reshape(-1,m*n)
    T_diff = P[1:] - P[:-1]
    T_diff = T_diff/np.linalg.norm(T_diff,axis=1)[:,np.newaxis]

    delta = np.diag(np.dot(T_diff[1:,:],T_diff[:-1,:].T))
    mag = np.linalg.norm(P,axis=1)

    fig, (ax1,ax3) = plt.subplots(1,2,figsize=(12,6))
    ax1.plot(np.arange(1,nsteps-1),delta,'-r')
    ax1.set_xlabel('Iteration')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Dot product', color='r')
    ax1.tick_params('y', colors='r')
    ax1.ticklabel_format(useOffset=False)
    #ax1.set_ylim(0.99,1)

    ax2 = ax1.twinx()
    ax2.plot(np.arange(nsteps),mag,'-ob',markersize=0.1)
    ax2.plot(np.arange(nsteps),np.ones(nsteps)*mag[0],'--k')
    ax2.set_ylabel('Distance from origin', color='b')
    ax2.tick_params('y', colors='b')
    ax2.ticklabel_format(useOffset=False)

    ax3.plot(np.arange(nsteps),results['datax'],'-ok',markersize=0.1)
    ax3.plot(np.arange(nsteps),np.ones(nsteps)*results['datax'][0],'--k')
    ax3.set_ylabel('Length of cut bond', color='k')
    ax3.set_xlabel('Iteration')

    ax4 = ax3.twinx()
    dists = np.linalg.norm(P-P[0],axis=1)
    ax4.plot(np.arange(nsteps),dists,'-og',markersize=0.1)
    ax4.set_ylabel('Distance from starting point', color='g')
    ax4.tick_params('y', colors='g')
    ax4.ticklabel_format(useOffset=False)

    fig.tight_layout()
    #now = datetime.datetime.now()
    #variables = map(str,[results['bond'],np.mean(lave),nsteps])
    #vars_strings = "_".join(variables)
    #name = 'trihex_lengths_'+vars_strings+'.pdf'
    #plt.title(name)
    plt.tight_layout()
    if save:
        plt.savefig('./results_trihex/'+name,dpi=100)
        plt.close()
    else:
        plt.show(fig)
