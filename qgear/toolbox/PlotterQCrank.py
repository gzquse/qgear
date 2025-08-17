__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from qgear.toolbox.PlotterBackbone import PlotterBackbone
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec

from matplotlib.colors import LinearSegmentedColormap

#...!...!....................
def create_jet_white_colormap():
    """
    Creates a custom colormap similar to 'jet' but with neon green/cyan replaced by white.

    Returns:
    LinearSegmentedColormap: The custom colormap.
    """
    jet_white_colors = [
        (0, 0, 0.5),  # Dark blue
        (0, 0, 1),    # Blue
        (0, 1, 1),    # Cyan
        (1, 1, 1),    # White, was green
        #(0, 1, 0),    # Green
        (1, 1, 0),    # Yellow
        (1, 0, 0),    # Red
        (0.5, 0, 0)   # Dark red
    ]

    # Create and return the colormap
    return LinearSegmentedColormap.from_list("jet_white_map", jet_white_colors)



#...!...!....................
def compute_correlation_and_draw_line(ax, x_data, y_data,xLR=[]):
    """Compute correlation and draw a line at the angle of correlation."""
    correlation = np.corrcoef(x_data, y_data)[0, 1]

    # Line representing correlation - slope based on correlation
    # y = mx + c, where m is the correlation coefficient
    # We pass through the mean of the points for the line of best fit
    mean_x, mean_y = np.mean(x_data), np.mean(y_data)
    ax.plot(mean_x,mean_y,'Dr',ms=5)
    m = correlation * np.std(y_data) / np.std(x_data)
    c = mean_y - m * mean_x
    
    # Points for the line
    x12 = np.array([min(x_data), max(x_data)])
    y12 = m * x12 + c
    ax.plot(x12, y12, 'r--',lw=1.0)

    th=np.arctan(m)
    txt='correl: %.2f, theta %.0f deg'%(correlation,th/np.pi*180)
    # ax.text(0.05,0.92,txt,transform=ax.transAxes,color='r')    
    
    ax.grid(True)
    return 

        
#...!...!....................
def plot_histogram(ax, res_data):
    """Plot histogram of the difference and annotate mean and std."""
    
    ax.hist(res_data, bins=25, color='salmon', alpha=0.7)
    mean = np.mean(res_data)
    std = np.std(res_data)
    # assuming normal distribution, compute std error of std estimator
    # SE_s=std/sqrt(2(n-1)), where n is number of samples
    N=res_data.shape[0]
    se_s=std/np.sqrt(2*(N-1))
    ax.axvline(mean, color='r', linestyle='dashed', linewidth=1)
    txt='Mean: %.3f\nStd: %0.3f +/- %0.3f'%(mean,std,se_s)
    #ax.annotate(txt, xy=(0.05, 0.85),c='r', xycoords='axes fraction')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))


#...!...!....................
def sum_column(md):
    pmd=md['payload']
    smd=md['submit']
    txt=md['short_name']
    tmd=md['transpile']
    txt+='\nback: %s'%smd['backend']
    txt+='\nshots/addr : %d'%(smd['num_shots']/pmd['seq_len'])
    txt+='\nimages: %d '%(pmd['num_sample'])
    txt+='\nshots/img : %d k'%(smd['num_shots']/1000)
    txt+='\nseq len: %d'%pmd['seq_len']
    txt+='\nqubits: %d'%pmd['num_qubit']
    txt+='\nn2q gates: %d'%tmd['2q_gate_count']
    txt+='\n2q gates depth: %d'%tmd['2q_gate_depth']
 
    return txt
  
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)
        
#...!...!..................
    def qcrank_accuracy(self,bigD,md,figId=1):
        #pprint(md)
        pmd=md['payload']
        smd=md['submit']
        tmd=md['transpile']
        xrL,xrR=-1.05, 1.05

       
        figId=self.smart_append(figId)        
        nrow,ncol=1,3
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,3*nrow))
                
        rec_udata=bigD['rec_udata']
        inp_udata=bigD['inp_udata']  # im,add,dat
           
        topTit=[ 'job: '+md['short_name'], 'Residua ',smd['backend']+','+smd['date'] ]

        resMX=md['plot']['resid_max_range']
        #....... plot data .....
        rdata=rec_udata.flatten()
        tdata=inp_udata.flatten()

        xLab=' true input'
        #....  left column ....
        ax = self.plt.subplot(nrow,ncol,1)
           
        ax.scatter(tdata,rdata,alpha=0.6,s=12)
        ax.set(xlabel=xLab,ylabel='reco')
        compute_correlation_and_draw_line(ax, tdata, rdata)
           
        ax.set_aspect(1.)
        ax.set_xlim(xrL,xrR);ax.set_ylim(xrL,xrR)
        x12 = np.array([min(tdata), max(tdata)])
        ax.plot(x12,x12,ls='--',c='k',lw=0.7)           
        # ax.set_title(topTit[0]) 
        
        
        #..... right column ....
        ax = self.plt.subplot(nrow,ncol,3)
        res_data = rdata - tdata
        h = ax.hist2d(tdata, res_data, bins=25, cmap='Blues',cmin=0.1)
        self.plt.colorbar(h[3], ax=ax)

        #print('tt',tdata[:40])
        #print('re',res_data[:40])
        
        compute_correlation_and_draw_line(ax, tdata , res_data) 
        ax.axhline(0.,ls='--',c='k',lw=1.0)

        ax.set_ylabel('reco-true')
        ax.set(xlabel=xLab,ylabel='reco-true')
        # ax.set_title(topTit[1])
        
        ax.set_xlim(xrL,xrR); ax.set_ylim(-resMX,resMX)
        ax.grid()
        
        #..... middle column ....
        ax = self.plt.subplot(nrow,ncol,2) 
        plot_histogram(ax,  res_data)
        # ax.set_title(topTit[2])
        xLab= 'reco-true'
        ax.set(xlabel=xLab,ylabel='samples')
        ax.axvline(0.,ls='--',c='k',lw=1.0)
        ax.set_xlim(-resMX,resMX)
        return
            
        # .... decorations ....
        # Overlay the text on top of the plots
        txt=sum_column(md)
        ax.text(0.88, 0.90, txt, fontsize=10, color='m', ha='left', va='top',transform=ax.transAxes)

#...!...!..................
    def dynamic_range(self,bigD,md,figId=3):
        #pprint(md)
        pmd=md['payload']
        smd=md['submit']

        figId=self.smart_append(figId)        
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,4))

        tudata=bigD['inp_udata'].flatten()
        rudata=bigD['rec_udata'].flatten()
        
        ax = self.plt.subplot(nrow,ncol,1)        
        plot_histogram(ax, tudata)
        ax.set(xlabel='inp_udata  values')

        ax = self.plt.subplot(nrow,ncol,2)        
        plot_histogram(ax, rudata)
        ax.set(xlabel='rec_udata values')
        
#...!...!..................
    def canned_image(self,bigD,md,figId=1):
        figId=self.smart_append(figId)        
        nrow,ncol=2,3
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,6.2))
        # Define the GridSpec layout: 60% for the first row and 40% for the second row
        gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, height_ratios=[4, 2])

        pmd=md['payload']
        smd=md['submit']
        cad=md['canned']
        resMX=md['plot']['resid_max_range']
        reco_img= bigD['rec_norm_image']
        true_img=bigD['norm_image']
        # Plot 2D images

        imgL=[true_img, reco_img,true_img- reco_img]
        titL=['Input: %d k pixels, %s'%(cad['image_pixels']/1000,md['short_name']),
              'Reco , %s'%(smd['backend']),
              'Residual ']
        jet_white = create_jet_white_colormap()
        cmapL=[ 'gray','grey',jet_white] #,'jet']

        #..... top row images:  input-gray, meas-grad, true-grad
        for j in range(3):
            ax=fig.add_subplot(gs[0,j])
            v1,v2=None,None
            if j>1: v1,v2=-0.11,0.11  # restrict z-range for gradient
            zBar = ax.imshow(imgL[j], cmap=cmapL[j],vmin=v1,vmax=v2)            
            ax.set_title(titL[j])

            #if  imgOp not in ['sqrGradX','sqrGrdM']:
            fig.colorbar(zBar, ax=ax, orientation='horizontal', shrink=0.75, pad=0.1)
            if j==0: ax.set_ylabel(cad['image_name'])
            
        
        #..... bottom row QA plots
        rdata=reco_img.flatten()
        tdata=true_img.flatten()
        res_data = rdata - tdata
        if 'truth_rangeLR' in md:
            xrL,xrR=md['truth_rangeLR']
        else:
            xrL,xrR=-1.05, 1.05

        xLab='true value'

        # ... fig A
        ax=fig.add_subplot(gs[1,0])
        ax.scatter(tdata,rdata,alpha=0.6,s=8)
        ax.set(xlabel=xLab,ylabel='reco')
        compute_correlation_and_draw_line(ax, tdata, rdata)
        ax.set_aspect(1.)
        ax.plot([0,1],[0,1],ls='--',c='k',lw=0.7)     
        ax.set_xlim(xrL,xrR);ax.set_ylim(xrL,xrR)
         
        # ... fig C
        ax=fig.add_subplot(gs[1,2])
        h = ax.hist2d(tdata, res_data, bins=25, cmap='Blues') #,cmin=0.1)
        self.plt.colorbar(h[3], ax=ax) #, orientation='horizontal', shrink=0.75, pad=0.1)
        ax.set_ylabel('reco-true')
        ax.set(xlabel=xLab,ylabel='reco-true')
        compute_correlation_and_draw_line(ax, tdata , res_data)
        ax.axhline(0.,ls='--',c='k',lw=1.0)

        # ... fig B        
        ax=fig.add_subplot(gs[1,1])
        plot_histogram(ax,  res_data)
        ax.set(xlabel='reco-true',ylabel='num pixels')
        ax.axvline(0.,ls='--',c='k',lw=1.0)
        ax.set_xlim(-resMX,resMX)

        txt=sum_column(md)
        #txt+='\n\n'+
        ax.text(0.65, 0.95, txt,transform=ax.transAxes, fontsize=10, color='m', ha='left', va='top')

        print('plotted ',cad['image_name'])

