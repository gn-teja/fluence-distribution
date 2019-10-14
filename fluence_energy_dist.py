import math as math
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import threading

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

    FFF = [F0, F01, F, F1,Flineminx, Flineminy,Flinemaxx, Flinemaxy, Fmean]

    plot1 = compute_plot_aaa(aaa,dx)
    plot2 = compute_plot_fp(P,t1,t2,d0,dp,dh,x,y,Fp,w0)
    plot3 = compute_plot_ftot(dp,dh,x,y,Ftot,w0)
    plot4 = compute_plot_ftot_x_axis(x,y,Ftot,w0,dx,F0,dp,n,Flineminx,Flinemaxx,Fmean)
    plot5 = compute_plot_ftot_y_axis(x,y,Ftot,w0,dx,F0,dh,m,Flineminy,Fmean)

    # # Plots executing in different threads
    # plot1 = threading.Thread(target=compute_plot_aaa, args=(aaa,dx))
    # plot2 = threading.Thread(target=compute_plot_fp, args=(P,t1,t2,d0,dp,dh,x,y,Fp,w0))
    # plot3 = threading.Thread(target=compute_plot_ftot, args=(dp,dh,x,y,Ftot,w0))
    # plot4 = threading.Thread(target=compute_plot_ftot_x_axis, args=(x,y,Ftot,w0,dx,F0,dp,n,Flineminx,Flinemaxx,Fmean))
    # plot5 = threading.Thread(target=compute_plot_ftot_y_axis, args=(x,y,Ftot,w0,dx,F0,dh,m,Flineminy,Fmean))

    # plot1.start()
    # plot2.start()
    # plot3.start()
    # plot4.start()
    # plot5.start()

    # plot1 = plot1.join()
    # plot2 = plot2.join()
    # plot3 = plot3.join()
    # plot4 = plot4.join()
    # plot5 = plot5.join()
    return FFF, plot1, plot2, plot3, plot4, plot5;

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
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    plot_url = base64.b64encode(bytes_image.getvalue()).decode()
    return  'data:image/png;base64,{}'.format(plot_url)

def compute_plot_ftot(dp,dh,x,y,Ftot,w0):
    bytes_image = io.BytesIO()
    fig, ax2 = plt.subplots()
    plt.rcParams['lines.linewidth'] = 1
    CS = ax2.contour(x,y,Ftot,20)
    fig.colorbar(CS, ax=ax2)
    ax2.axis([-dp, max(x), -dp, max(y)])
    ax2.set_xlabel('d_p (\mum)')
    ax2.set_ylabel('d_h (\mum)')
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    plot_url = base64.b64encode(bytes_image.getvalue()).decode()
    return  'data:image/png;base64,{}'.format(plot_url)

def compute_plot_ftot_x_axis(x,y,Ftot,w0,dx,F0,dp,n,Flineminx,Flinemaxx,Fmean):
    bytes_image = io.BytesIO()
    fig, ax3 = plt.subplots()
    ax3.plot(x,Ftot[int(2*w0/dx),:])
    ax3.axis([-w0, dp*n, 0, 5750])
    ax3.axhline(y = F0, color='b', linestyle='--')
    ax3.axhline(y = Flineminx, color='g', linestyle='--')
    ax3.axhline(y = Flinemaxx, color='r', linestyle='--')
    ax3.axhline(y = Fmean, color='k', linestyle='--')
    ax3.set_xlabel('Scan direction x (\mum)')
    ax3.set_ylabel('F_a_c_c (J/cm^2)')
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    plot_url = base64.b64encode(bytes_image.getvalue()).decode()
    return  'data:image/png;base64,{}'.format(plot_url)

def compute_plot_ftot_y_axis(x,y,Ftot,w0,dx,F0,dh,m,Flineminy,Fmean):
    bytes_image = io.BytesIO()
    fig, ax4 = plt.subplots()
    ax4.plot(x,Ftot[:,int(2*w0/dx)])
    ax4.axhline(y = F0, color='b', linestyle='--')
    ax4.axhline(y = Flineminy, color='g', linestyle='--')
    ax4.axhline(y = Fmean, color='b', linestyle='--')
    ax4.axis([-w0, dh*m, 0, 5750])
    ax4.set_xlabel('Line overlap direction y (\mum)')
    ax4.set_ylabel('F_a_c_c (J/cm^2)')
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