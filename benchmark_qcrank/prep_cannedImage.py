#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Create canned input image for QCrank

'''
import copy
from pprint import pprint
import numpy as np
from PIL import Image
import os,sys
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
#from toolbox.Util_EscherHands import ehandsInput_to_qcrankInput

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--basePath",default=None,help="head path for set of experiments, or 'env'")
    parser.add_argument("--outPath",default='out/',help="output from ppacking image")

    parser.add_argument("-t","--tag",  default='b2',help="hardcoded configuration defining type of the image ")
    parser.add_argument('-q','--nqAddr', default=9, type=int, help='paralelizm of QCrank encoding ')

    args = parser.parse_args()
    # make arguments  more flexible
    if args.basePath=='env': args.basePath= os.environ['Cudaq_dataVault']
    if args.basePath!=None:
        args.outPath=os.path.join(args.basePath,'jobs')
    
    args.inpPath='../input_gray_images'
 
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    return args

#...!...!....................
def importImage(args):
    tag=args.tag
    inpN=None
    if tag=='a1':  inpN='alphabet_x32_y1'
    if tag=='b1':  inpN='alphabet_x64_y16' 
    if tag=='b2':  inpN='high-heels_x32_y32' 
    if tag=='b3':  inpN='high-heels_x64_y64' 
    if tag=='b4':  inpN='high-heels_x128_y128' 
    if tag=='c1':  inpN='xray-finger-8y_x64_y80' 
    if tag=='d1':  inpN='igb_facade_x192_y128' 
    if tag=='d2':  inpN='zebra_x192_y128' 
    if tag=='d3':  inpN='xray-hand-4y_x128_y192' 
    if tag=='d4':  inpN='xray-foot-12y_x192_y128' 
    if tag=='d5':  inpN='e_coli_bacteria_x192_y128' 
    if tag=='e1':  inpN='zebra_x384_y256' 
    if tag=='e2':  inpN='cameraman_x256_y384' 
    if tag=='e3':  inpN='micro_germ_x384_y256' 
    if tag=='x':  inpN=1 ;  nqAddr=1
    if tag=='x':  inpN=1 ;  nqAddr=1

    if inpN==None: print('undefined tag, abort'); exit(99)
    inpFF=os.path.join(args.inpPath,inpN+'.png')
    img = Image.open(inpFF)
    imgA=np.array(img)
    print('imported imgA:',inpN,imgA.shape,imgA.dtype)
    args.image_name=inpN
    bigD={'phys_image':imgA}

    
    return bigD


#...!...!....................
def buildMeta_CannedImage(args,bigD):
    pd={}  # payload
    pd['num_sample']=1
    pixY,pixX=bigD['phys_image'].shape
    
    cad={} # canned MD
    cad['image_name']=args.image_name
    cad['image_shape_xy']=[pixX,pixY]
    cad['image_pixels']=pixX*pixY
        
    pd['nq_addr']=args.nqAddr
    pd['seq_len']=1<<pd['nq_addr']
    assert cad['image_pixels']% pd['seq_len']==0
    pd['nq_fdata']= cad['image_pixels']// pd['seq_len']
    pd['num_clbit']=pd['nq_fdata']+pd['nq_addr']  # number of measured qubits
    assert pd['num_clbit'] <=34  # needs GPU's to run, Martin change it to a higher value
    
    pd['qcrank_max_fval']=np.pi # default range for QCrank
    
    cad['canned_type']='gray_image'
    md={ 'payload':pd,'canned':cad}
    
    md['short_name']='canImg_%s_%d_%d'%(args.tag,pixX,pixY)

    print('input phys image:',cad)
    if args.verb>1:  print('\nBMD:');pprint(md)            
    return md

#...!...!....................
def ehandsInput_to_qcrankInput(udata):  # shape: n_img,  nq_data, n_addr,
    fdata=np.arccos(udata) # encode user data for qcrank
    # QCrank wants indexs order  to be: n_addr, nq_data, n_img --> (2,1,0)
    fdata=np.transpose(fdata, ( 2,1, 0))
    # QCrank also reverses the order of data qubits
    # Reverse the order of values along axis 1
    fdata = fdata[:, ::-1, :]
    return fdata  # shape : n_addr, nq_data, n_img

#...!...!....................
def prepImageQCrankInput(md,bigD):
    pmd=md['payload']
    cad=md['canned']
    
    n_addr=pmd['seq_len']
    nq_data=pmd['nq_fdata']
    n_img=pmd['num_sample']
    #imgOp=pmd['math_op']
    #nEHbl=pmd['ehands_blocks']
    
    #.... Renormalize the array  values to range [-1,1] needed by Escherhands
    imgA=bigD['phys_image']
    min_val = imgA.min()
    max_val = imgA.max()
    imgAN = 2 * ((imgA - min_val) / (max_val - min_val)) - 1
    #print('imgAN:',imgAN.dtype,imgAN.shape,imgAN[:])
    print('img min/max:',imgAN.min(),imgAN.max())    
    bigD['norm_image']=imgAN.astype(np.float32)

    # Flatten along x-axis (row-wise)
    inpA=imgAN.flatten()        # Default is C-style order

    print('PQCT: inpA:',inpA.shape, inpA.reshape(-1,n_addr).shape)
    
    #.... final format for QCrank
    inp_udata = np.zeros((n_img, nq_data, n_addr),dtype=np.float32)
    assert n_img==1; im=0
    inp_udata[im] =inpA.reshape(-1,n_addr)
    print('inp_udata:',inp_udata.shape)
    
    bigD['inp_udata']=inp_udata  # shape: n_img, nq_data, n_addr
    bigD['inp_fdata']=ehandsInput_to_qcrankInput(inp_udata)  # shape : n_addr, nq_data, n_img
    

    
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    bigD=importImage(args)   
    MD=buildMeta_CannedImage(args,bigD)
    pprint(MD)
    
    prepImageQCrankInput(MD,bigD)
    
      
    print('M:inp bigD:',sorted(bigD))
    
    outF=MD['short_name']+'.qcrank_inp.h5'
    fullN=os.path.join(args.outPath,outF)
    write4_data_hdf5(bigD,fullN,metaD=MD)

    pprint(MD)

    print('local sim for cpu:\n time  ./run_aer_job.py --cannedExp   %s   -n 300   -E \n'%(MD['short_name'] ))
    print('cloud sim for two nodes 8 gpus:\n   ./run_cudaq.sh (remember to quit shifter)\n')
    print('M:done')
   
