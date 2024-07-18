import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft
t = np.linspace(0, 4, 10 * 1024)
N = 4*1024
f = np.linspace(0 , 512 , int(N/2))
C3=130.81
D3=146.83
E3=164.81
F3=174.61
G3=196
A3=220
B3=246.93
C4=261.63
D4=293.66
E4=329.63
F4=349.23
G4=392
A4=440
B4=493.88
F1=[A3,B3,C3,A3]
f1=[A4,B4,C4,A4]
ti=[0.6,1.2,1.8,2.4]
T=0.5
x=0
n=4

while (n>0):
    z= ((np.sin(2*np.pi*F1[n-1]*t)+(np.sin(2*np.pi*f1[n-1]*t)))* (np.heaviside(t - ti[n-1], 0) - np.heaviside(t - ti[n-1] - T, 0)))
    n= n - 1
    x= x + z
x_f = fft(x)
X_f = 2/N * np.abs(x_f [0:np.int(N/2)])
fn1,fn2=np.random.randint(0,512,2)
n_t=np.sin(2*fn1*np.pi*t)+np.sin(2*fn2*np.pi*t)
xn_t=x+n_t
xn_f = fft(xn_t)
Xn_f= 2/N * np.abs (xn_f[0:np.int(N/2)])

def highest_indices(x, y):
    highest_val = max(y)
    highest_two_indices = sorted(range(len(x)), key=lambda i: x[i], reverse=True)[:2]
    if x[highest_two_indices[0]] > highest_val and x[highest_two_indices[1]] > highest_val:
        return highest_two_indices
    else:
        return highest_indices(x,y)
a=highest_indices(Xn_f,X_f)[0]
b=highest_indices(Xn_f,X_f)[1]
fn1d=round(f[a])
fn2d=round(f[b])
xFiltered=xn_t-(np.sin(2*np.pi*fn1d*t)+np.sin(2*np.pi*fn2d*t))

xFilteredF1=fft(xFiltered)
xFilteredF2 = 2/N * np.abs(xFilteredF1[0:np.int(N/2)])
sd.play(xFiltered, 4* 1024)
plt.figure()
plt.xlabel('time')
plt.ylabel('Signal')
plt.title('Signal in time domain (with noise) ')
plt.plot(t,xn_t)
plt.grid(True)

plt.figure()
plt.xlabel('frequency')
plt.ylabel('Signal')
plt.title('Signal in frequency domain (with noise) ')
plt.plot(f,Xn_f)
plt.grid(True)

plt.figure()
plt.xlabel('time')
plt.ylabel('Signal')
plt.title('Signal in time domain (after noise cancellation) ')
plt.plot(t,xFiltered)
plt.grid(True)

plt.figure()
plt.xlabel('frequency')
plt.ylabel('Signal')
plt.title('Signal in frequency domain (after noise cancellation) ')
plt.plot(f,xFilteredF2)
plt.grid(True)