import numpy as np
from qpixl import qcrank
from bitstring import BitArray
from pprint import pprint
# This class is an extension of QCrank encoding NEQR-equivalent image representation

# Jan: I did not know how to access this method from ParametrizedQCRANK, so I copy/past it here - this need to be clened up

from ._util import  convert_shots_to_pdf

def yields_to_pdf_jan(yields, nqubits, normalize=False): # Daaan, please remove it
        '''The yields are  expected as a list of dictionaries in the standard
        qiskit format. The number of qubits is the length of each bitstring.
        The normalize flag indicates if sum to 1 or not.'''
        if isinstance(yields, dict):
            yields = [yields]
        pdfs = np.zeros([2**nqubits, len(yields)])
        for i, y in enumerate(yields):
            pdfs[:, i] = convert_shots_to_pdf(y, normalize=normalize)
        return pdfs


#............................
#............................
#............................
class ParametrizedQBART(qcrank.ParametrizedQCRANK):

#...!...!....................
    def __init__(self, nq_addr, nq_data,
                 measure: bool = True,
                 statevec: bool = False,
                 barrier: bool =True,
                 subopt: bool =False ):
        
        qcrank.ParametrizedQCRANK.__init__(self,
            nq_addr, nq_data,
            qcrank.QKAtan2DecoderQCRANK,
            measure=measure, statevec=statevec,
            keep_last_cx=True,  # I want only one mapping of the bits 
            reverse_bits=True,  # I'm not sure if I want that
            barrier=barrier,
            parallel=not subopt
        )
        #print('QBart cstr')


#...!...!....................
    def bind_data(self, data):
        '''Enables binding the QBArt circuit to data

        Args:
            data:
                Numerical data to bind to the parametrized QBArt circuit:
                  * numpy array of size (2**nq_addr, k)
         '''
        # verify data dimensionality and dynamic range
        assert data.ndim==2, 'reshape input'
        num_addr=1<<self.nq_addr
        max_val=1<<self.nq_data
        assert num_addr==data.shape[1], 'mismatch address lines'
        assert np.min(data)>=0
        assert np.max(data)<max_val, 'increase number of data lines'
        
        # break the data in to bits for UCRy
        num_img=data.shape[0]  # Jan: input image index is the first because it is natural to process output for each circuit, sequentially. Here I transpose the data, so data_bits have image index as the last one for   consistency w/ QCrank. (I'd be more happy of QCrank also uses image index as the 1st)
        data_bits=np.zeros((num_addr,self.nq_data,num_img),dtype=np.uint8)

        ''' Jan: This code is not vectorized.
        I prefere simpler code over speed during developement.
        But I do not mind to vecotrize it in the future. It will matter for mega-pixel images '''

        for k in range(num_img):
            for  i, a in enumerate(data[k]):
                val=BitArray(uint=int(a),length=self.nq_data) 
                for j in range(self.nq_data):
                    data_bits[i,j,k]=val[j]            
                
        # bind the binary data
        qcrank.ParametrizedQCRANK.bind_data(self,data_bits, max_val=1)
        # Daan: The max_val should be 1: you want 1 bits to be mapped to pi, not pi/2
        
#...!...!....................
    def decode_meas(self,yields, is_numpy=False):
        ''' sorts measured bitstring by address, selects MPV data for each address 
        '''

        if is_numpy:
            assert yields.ndim == 2  # expected shape:  [bitstrings,images]
            pdfs = yields.T  # Dann assumes images index is the last one, Jan assumes the opposite
        else:  # yields is a Qiskit dictionary
            pdfs = yields_to_pdf_jan(yields, self.nq_addr + self.nq_data)
        #print('DNE: pdfs=',pdfs.shape,'sum=',np.sum(pdfs))
        
        #Jan: the code below is not vectorized - it is too early to decide what is the relevant output from NEQR

        
        #========== STEP 1 : sort all observed bit strings by address
        ''' output of this step has variable lenght, it is hard to store it permanently,
        assume it is a transient object  '''
        # raw output container : [images][addr][ var_len_list[ (val,prob) ] ]
        num_addr=1<<self.nq_addr
        num_img=pdfs.shape[1]
        num_qubits=self.nq_addr+self.nq_data
        #print('eee',self.nq_addr,self.nq_data)
        
        reco_raw={ k:{ iadr:[] for iadr in range(num_addr) } for k in range(num_img)}
        
        assert pdfs.shape[1]==num_img
        assert pdfs.shape[0]==1<<num_qubits

        # decode all NEQR-style measurements
        for k in range(num_img):
            one_raw=reco_raw[k]
            one_pdf=pdfs[:,k]  # Dann assumes images index is the last one
            shots=np.sum(one_pdf) # I like to see probability for final results
            hits=np.argwhere(one_pdf>0)[:,0]  # find non-zero entries
            #print('img=',k,'hits=',hits,num_qubits,hits.dtype,hits.shape)

            for ibits in hits:
                Bmeas=BitArray(uint=int(ibits),length=num_qubits)
                Badr=Bmeas[:self.nq_addr]
                Bval=Bmeas[self.nq_addr:]
                prob=one_pdf[ibits]/shots
                iadr=Badr.uint
                #if self.nq_addr<3 : print('adr=%d  val=%2d  meas=%s ->%s:%s mprob=%.2f'%(iadr,val.uint,meas.bin,adr.bin,val.bin,prob))

                assert iadr in one_raw ,'serious logical issue'
               
                one_raw[iadr].append([Bval,prob])

                
        #========== STEP 2 :  find most probable values and some info about alternatives
        ''' output of this step is fixed size numpy array - it is easy to store for post processing '''
        # most probable output container:  [images][addr][ [ valMpv, probMpv, probAny, numAny] ]
        #    meaning: mpv=Most probably value, any=anu observed data-value bit string at this address
        KV=4 # number of retianed observables
        reco_mpv=np.zeros((num_img,num_addr,KV))        

        for k in range(num_img):
            one_raw=reco_raw[k]
            one_mpv=reco_mpv[k]
            for iadr in range(num_addr):
                rec=one_raw[iadr] # this is list of all observed values+probs
                numAny=len(rec)
                pbest=-0.1; rval=None; psum=0; Brbest=BitArray(uint=0,length=1)
                for ir in range(numAny): # find MPV
                    Brval,prob=rec[ir]
                    psum+=prob
                    if pbest>prob: continue
                    pbest=prob; Brbest=Brval
                one_mpv[iadr]=np.array([Brbest.uint,pbest,psum,numAny])

        print('MPV measurement, sample:\n[addr] [ valMpv,   probMpv,  probAny,  numAny ]')
        pprint(reco_mpv[:2,:10])
        return reco_raw,reco_mpv

    
#...!...!....................
    def eval_meas(self,true_data,reco_mpv,trueSigned=False):
        #========== STEP 3 :  compare most probable values with ground truth, and some info about alternatives
        ''' output of this step is fixed size numpy array - it is easy to store for post processing '''
        KV=5 # number of observables  [images][addr][ valMpv, probMpv, probAny, numAny, hammMpv]
        #    meaning:  hammMpv= Hamming distance between MV value and the truth
        #              probTrue= prob. of the correct value
        #  sum_eval [images] [success, numCorrect]  contains summary for every image
        
        num_img,num_addr,_=reco_mpv.shape
        reco_eval=np.zeros((num_img,num_addr,KV))        
        sum_eval=np.zeros((num_img,2)) 

        #Jan: the code below is not vectorized
        totOk=0 
        for k in range(num_img):
            nok=0
            one_mpv =reco_mpv[k]
            one_eval=reco_eval[k]
            one_data=true_data[k]
            for iadr in range(num_addr):
                #print('ggg',iadr,one_data[iadr],self.nq_data)
                rec=one_mpv[iadr] # contains: [ valMpv, probMpv, probAny, numAny]
                if trueSigned:
                    tval=BitArray(int=one_data[iadr],length=self.nq_data)
                else:
                    tval=BitArray(uint=int(one_data[iadr]),length=self.nq_data)
                rval=BitArray(uint=int(rec[0]),length=self.nq_data)
                hamm=(tval^rval).count(1) # Hamming distance between MPV and truth
                if hamm==0: nok+=1
                out=rec.tolist()+[hamm ]
                one_eval[iadr]=np.array(out)
                #print('qq',iadr,tval.bin,rval.bin,(tval^rval).bin,ham)
                good= (nok==num_addr)
            sum_eval[k]=[good,nok]
            totOk+=good
        allGood=totOk==num_img
        test='---PASS---' if allGood else '****FAILD****  for %d images'%(num_img -totOk)
        print('MPV evaluation, sample \n [addr] [ valMpv,  probMpv,  probAny,  numAny,   hammMpv ]')
        pprint(reco_eval[:2,:10])

        print('eval_NEQR_meas: num_img=%d  test: %s'%(num_img,test))
        return reco_eval,sum_eval,totOk,allGood
    
#...!...!....................
    def find_truth(self,true_data,reco_raw):
        #========== STEP 4 :  (optional) find probability of the true bit string
        num_img,num_addr=true_data.shape
        KV=1 # number of observables  [images][addr][ probTrue]
        #Jan: the code below is not vectorized
        for k in range(num_img):
            one_raw =reco_raw[k]
            one_data=true_data[k]
            for iadr in range(num_addr):
                rec=one_raw[iadr] # contains: [ var_list[ (val,prob) ] ]
       
        UNFINISHED
