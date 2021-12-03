import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write

class ica:
    
    def __init__(self,file_1,file_2):
        self.file_1=file_1
        self.file_2=file_2
    
    def load_data(self):
        fs1,s1=wavfile.read(self.file_1)
        fs2,s2=wavfile.read(self.file_2)
        return fs1,fs2,s1,s2
        
    def compute_ica(self):
        from sklearn.decomposition import FastICA
        fs1,fs2,s1,s2=ica.load_data(self)
        ica_1=FastICA(n_components=2)
        S_=ica_1.fit_transform(np.c_[s1,s2])
        return S_
        
    def show_result(self):
        import matplotlib.pyplot as plt
        fs1,fs2,s1,s2=ica.load_data(self)
        result=ica.compute_ica(self)
        s1_result=result[:,0]
        s2_result=result[:,1]
        plt.figure(figsize=(12,8))
        plt.subplot(4,1,1);plt.plot(s1);plt.title("Mixing Sound1"); plt.xlabel('time')
        plt.subplot(4,1,2);plt.plot(s2);plt.title("Mixing Sound2"); plt.xlabel('time')
        plt.subplot(4,1,3);plt.plot(s1_result);plt.title("Sound Source1"); plt.xlabel('time')
        plt.subplot(4,1,4);plt.plot(s2_result);plt.title("Sound Source2");plt.xlabel('time') 
        plt.tight_layout();plt.show()
        return s1_result,s2_result,fs1,fs2
        
    def save_data(self):
        s1_result,s2_result,fs1,fs2=ica.show_result(self)
        scaled = np.int16(s1_result/np.max(np.abs(s1_result)) * 32767)
        write('C:/Users/user/Desktop/파일/디지털 생체신호처리/11주차-ICA/result_1.wav', fs1, scaled)
        ##
        scaled = np.int16(s2_result/np.max(np.abs(s2_result)) * 32767)
        write('C:/Users/user/Desktop/파일/디지털 생체신호처리/11주차-ICA/result_2.wav', fs2, scaled)

if __name__=="__main__":
    file_1="mix1.wav"
    file_2="mix2.wav"
    ica(file_1,file_2).save_data()