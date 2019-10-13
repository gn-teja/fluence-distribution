import math as math
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def program(P, d0, t1, dp1, dh1, t2, dp2, dh2):

    # Contour Plot for aaa
    # plot1 = compute_plot_aaa(aaa,dx);

    return None


def fluence_energy_dist_over_larger_distance(P, d0, t1, dp1, dh1, t2, dp2, dh2):

    w0 = d0/2 #um
    n = 10
    dx = 0.5

    F0 =  (8*P*t1/1*10**6/math.pi/(w0/1*10**4)**2)*1*10**4
    F01 = (8*P*t2/1*10**6/math.pi/(w0/1*10**4)**2)*1*10**4

    F = P*t1/dp1/dh1*100
    F1 = P*t2/dp2/dh2*100

    if (dp1 < dp2):
        dp = dp2
    else:
        dp = dp1

    if (dh1 < dh2):
        dh = dh2
    else:
        dh = dh1

    m = round(n*dp/dh)

    # have to define range from 0 to n*dp+dx with dx as differnce, to set the last element as n*dp
    s_x = np.arange(0,n*dp+dx,dx, dtype=float)
    s_y = np.arange(0,n*dh+dx,dx, dtype=float)
    x = np.arange(-2*w0, n*dp+dx,dx, dtype=float)
    y = np.arange(-2*w0, n*dp+dx,dx, dtype=float)

    Fp = [[0 for _ in range(len(x))] for j in range(len(y))]
    for ii in range(len(x)):
        for jj in range(len(y)):
            Fp[ii][jj] = F0* np.exp(-8*((x[ii]**2 + y[jj]**2) / w0**2))

    Fp1 = [[0 for _ in range(len(x))] for j in range(len(y))]
    for ii in range(len(x)):
        for jj in range(len(y)):
            Fp1[ii][jj] = F01* np.exp(-8*((x[ii]**2 + y[jj]**2) / w0**2))

    Fp = np.array(Fp)
    Fp1 = np.array(Fp1)
    Ftot = np.zeros(Fp.shape)

    # values of ll is from 0 to 9 and kk from is 0 to 8. so np.roll varies from kk and ll instead of kk-1 and ll-1
    for ll in range(0,int(n)):
        for kk in range(0,int(m)):
            Fp1CircY = np.roll(Fp1,int((ll)*dp2/dx), axis = 1)
            FpCircY = np.roll(Fp,int((ll)*dp1/dx), axis = 1)
            Ftot = Ftot + np.roll(Fp1CircY,int((kk)*dh2/dx),axis = 0) + np.roll(FpCircY,int((kk)*dh1/dx),axis = 0)

    Flineminx = min(Ftot[int(2*w0/dx),int(2*w0/dx): int(n*dp/dx)])
    Flinemaxx = max(Ftot[int(2*w0/dx),int(2*w0/dx): int(n*dp/dx)])
    Flineminy = min(Ftot[int(2*w0/dx): int(m*dh/dx),int(2*w0/dx)])
    Flinemaxy = max(Ftot[int(2*w0/dx): int(m*dh/dx),int(2*w0/dx)])

    aaa=Ftot[int((2*w0+2*dh)/dx):int((2*w0+3*dh)/dx)+1,int((2*w0+2*dp)/dx):int((2*w0+3*dp)/dx)+1]

    Fmean = np.mean(aaa)

    # Contour Plot for aaa
    plot1 = compute_plot_aaa(aaa,dx)
    plot2 = compute_plot_fp(P,t1,t2,d0,dp,dh,x,y,Fp,w0)
    plot3 = compute_plot_ftot(dp,dh,x,y,Ftot,w0)

    FFF = [F0, F01, F, F1,Flineminx, Flineminy,Flinemaxx, Flinemaxy, Fmean]
    print(FFF)
    print(plot1)
    return FFF, plot1, plot2, plot3;

def compute_plot_aaa(aaa,dx):
    bytes_image = io.BytesIO()
    m1, n1 = aaa.shape
    m2 = np.arange(0,np.floor(m1*dx)+dx,dx)
    n2 = np.arange(0,np.floor(n1*dx)+dx,dx)
    fig, ax = plt.subplots()
    X,Y = np.meshgrid(n2,m2)
    plt.rcParams['lines.linewidth'] = 1
    ax.contour(X,Y,aaa)
    ax.set_xlabel('Scan direction x (\mum)')
    ax.set_ylabel('Line overlap direction y (\mum)')
    # plt.show()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    plot_url = base64.b64encode(bytes_image.getvalue()).decode()
    return  'data:image/png;base64,{}'.format(plot_url)

def compute_plot_fp(P,t1,t2,d0,dp,dh,x,y,Fp,w0):
    bytes_image = io.BytesIO()
    fig, ax1 = plt.subplots()
    plt.rcParams['lines.linewidth'] = 1
    ax1.contour(x,y,Fp,20)
    ttl = ax1.set_title('P = {} W;  t1 = {} \mus;  t2= {} \mus;  d_0 = {} \mum; dp = {} \mum; dh = {} \mum '.format(P,t1,t2,d0,dp,dh))
    ttl.set_position([.5, 1.05])
    ax1.axis([-w0, w0, -w0, w0])
    ax1.set_xlabel('r (\mum)')
    ax1.set_ylabel('r (\mum)')
    # plt.show()
    # plt.show()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    plot_url = base64.b64encode(bytes_image.getvalue()).decode()
    return  'data:image/png;base64,{}'.format(plot_url)

def compute_plot_ftot(dp,dh,x,y,Ftot,w0):
    bytes_image = io.BytesIO()
    fig, ax2 = plt.subplots()
    plt.rcParams['lines.linewidth'] = 1
    CS = ax2.contour(x,y,Ftot,20)
    # plt.imshow(Ftot)
    # plt.colorbar();
    fig.colorbar(CS, ax=ax2)
    ax2.axis([-dp, max(x), -dp, max(y)])
    ax2.set_xlabel('d_p (\mum)')
    ax2.set_ylabel('d_h (\mum)')
    # plt.show()
    # plt.show()
    # plt.show()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    plot_url = base64.b64encode(bytes_image.getvalue()).decode()
    return  'data:image/png;base64,{}'.format(plot_url)

def convert_byte_image():
    return None

# print(F0)
# print(F01)
# print(F)
# print(F1)
# print (dp, dh)
# print(m)
# print('[Peak fluence of the single pulse frist pass {} J/cm^2]'.format(F0) )
# print('[Peak fluence of the single pulse second pass {} J/cm^2]'.format(F01) )
# print('[Areal fluence of frist pass {} J/cm^2]'.format(F) )
# print('[Areal fluence of second pass {} J/cm^2]'.format(F1) )
# print('[Max accumulated fluence over the scan line  {} J/cm^2]'.format(Flinemaxx))
# print('[Max accumulated fluence between the lines  {} J/cm^2]'.format(Flinemaxy))
# print('[Min accumulated fluence over the scan line  {} J/cm^2]'.format(Flineminx))
# print('[Min accumulated fluence between the lines   {} J/cm^2]'.format(Flineminy))
# print('[Average accumulated fluence  {} J/cm^2]'.format(Fmean))
# print(FFF)