#----------------------------------------------------#
# Rxy vs. 2theta interactive fit plot v0.2           #
# author: tatsunootoshigo, 7475un00705hi90@gmail.com #
#----------------------------------------------------#

# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter

# script version
version = '0.1a'
version_name = 'mr_fit_slider_plot_v' + version + '.py'

# parameter adjustment step
valstep_R0 = 1e-2

sample_mdate = '1111019'
sample_id = '000000'
model_symbol = 'A'
plot_title = 'model' + model_symbol 

sample_R0 = np.array([877.0, 632.0, 388.0, 174.0, 139.0, 83.115])
sample_rho = np.array([88.0, 105.0, 129.0, 174.0, 231.0, 194.0])
FM_thickness = np.array([3, 5, 10, 30, 50, 70])
MR_ratio = ([0.82, 0.609, 0.371, 0.183, 0.109, 0.118],[0.026, 0.035, 0.053, 0.016, 0.138, 0.038],[0.793, 0.584, 0.377, 0.176, 0.238, 0.142])
Delta_R_R0 = ([0.456, 0.235, 0.092, 0.020, 0.01, 0.006],[0.015, 0.014, 0.013, 0.002, 0.012, 0.02],[0.442, 0.235, 0.084, 0.02, 0.021, 0.08]) 
print(MR_ratio[0])

# adjust axis min/max and number fromat precision
xmin = 0
xmax = 500

ymin = 0
ymax = 1.0

xprec = 0
yprec = 2

desc_x = 0.1
desc_y = 0.1

# define string for plot axis labels
axis_label_theta = r'$\theta\, / \, \circ$'
axis_label_nm = r'$d_F\;/\;nm$'
axis_label_ohm = r'$R\;/\;\Omega$'
axis_label_delta_Ryz = r'$\Delta R_{SMR}\;/\;\%$'

# data series for the plot
x1 =  FM_thickness
y1 = MR_ratio[2]

# initial paremeter values
theta_SH = 0.20
theta_AH = 0.9
theta_F = 0.9
d_N = 3.0

lambda_N = 0.64
lambda_F = 1.0
rho_N = 1.56
rho_F = 1.92

Constant = 0.11

Gr = 1e+20
P = 0.4

# plot label defs
label_xy = r'$xy$'
label_xy_fit = r'$fit$'

# plot the fitting function using initial parameter values
xdf = np.arange(0.1, 1.25*xmax, 0.1)
d_F = xdf

dn_lambdan = (d_N / lambda_N)
df_lambdaf = (d_F / lambda_F)
dn_2lambdan = (d_N / 2.0*lambda_N)
df_2lambdaf = (d_F / 2.0*lambda_F)
lambdan_dn = (lambda_N / d_N)
lambdaf2_df = (2.0*lambda_F / d_F)

#x_N_mcd = (rho_N*xdf) / (rho_N*xdf + rho_F*d_N)
#x_N_mab = (rho_N*d_F / rho_F*d_N)

# fitting model A (gf = 0)
#gr_mab = 2*rho_N*lambda_N*Gr
#gf_mab = (( 1-(P**2))*rho_N*lambda_N ) / ( rho_F*lambda_F*(1 / tanh_df_lambdaf) )

#Delta_R_R0_ma = -1.0*(theta_SH**2)*lambdan_dn*( (tanh_dn_2lambdan**2) / (1 + x_N_mab) )*( gr_mab / (1 + gr_mab*(1 / tanh_dn_lambdan)) )
Delta_R_R0_ma = -1.0*(theta_SH**2)*lambdan_dn*( ((np.sinh(d_N / 2.0*lambda_N) / np.cosh(d_N / 2.0*lambda_N))**2) / (1 + (rho_N*d_F / rho_F*d_N)) )*( 2*rho_N*lambda_N*Gr / (1 + (2*rho_N*lambda_N*Gr)*(1 / (np.sinh(d_N / lambda_N) / np.cos(d_N / lambda_N)) )) ) + Constant

# fitting Model B
#Delta_R_R0_mb = -1.0*(theta_SH**2)*lambdan_dn*( (tanh_dn_2lambdan**2) / (1 + x_N_mab) )*( gr_mab / (1 + gr_mab*(1 / tanh_dn_lambdan)) - (gf_mab / (1 + gf_mab*(1/tanh_dn_lambdan)) ) )

# fitting Model C
#Delta_Ryz_mc = -1.0*x_N_mcd*(theta_SH**2)*(d_N/(2.0*lambda_N))*(np.sinh(d_N/lambda_N)/np.cosh(d_N/lambda_N))*(1-(1/(1-np.cosh(d_N/lambda_N)))) + Constant

# fitting Model D
#gs_cd = rho_N*lambda_N*Gr
#gf_cd = ((1-P)**2) * ((rho_N*lambda_N)/(rho_F*lambda_F)) * tanh_df_lambdaf
#dzeta_mcd = (theta_F*lambda_F*tanh_df_2lambdaf) / (theta_SH*lambda_N*tanh_dn_2lambdan)

#Delta_Ryz_md = -(1 - x_N_mcd)*x_N_mcd*(theta_AH**2) - (1 - x_N_mcd)*(1-P**2)*(theta_F**2)*(2*lambda_F / d_F)*(tanh_df_2lambdaf**2) - x_N_mcd*(theta_SH**2)*(lambda_N / d_N)*(tanh_dn_2lambdan**2)*(gs_cd / ( 1 + (1 / tanh_dn_lambdan) ) - ( ((1+dzeta_mcd)**2) ) / ( 1 + gf_cd*(1 / tanh_dn_lambdan) ) ) 

# create the fig and plot
fig, ax = plt.subplots(figsize=(9, 16))
fig.tight_layout(pad=4.0, w_pad=6.0, h_pad=2.0)
fig.canvas.set_window_title('mr_fit_model' + model_symbol) 
plt.subplots_adjust(left=0.15, bottom=0.5, wspace=0.0, hspace=0.0)
plt.figtext(0.80, 0.99, version_name, size=8)

# formatting the plot axes
def custom_axis_formater(custom_title, custom_x_label, custom_y_label, xmin, xmax, ymin, ymax, xprec, yprec):
	
	# get axes and tick from plot 
	ax = plt.gca()
	
	# set the number of major and minor bins for x,y axes
	# prune='lower' --> remove lowest tick label from x axis
	xmajorLocator = MaxNLocator(12, prune='lower') 
	xmajorFormatter = FormatStrFormatter('%.'+ np.str(xprec) + 'f')
	xminorLocator = MaxNLocator(12) 
	
	ymajorLocator = MaxNLocator(10) 
	ymajorFormatter = FormatStrFormatter('%.'+ np.str(yprec) + 'f')
	yminorLocator = MaxNLocator(20)
	
	# format major and minor ticks width, length, direction 
	ax.tick_params(which='both', width=1, direction='in', labelsize=14)
	ax.tick_params(which='major', length=6)
	ax.tick_params(which='minor', length=4)

	# set axes thickness
	ax.spines['top'].set_linewidth(1.5)
	ax.spines['bottom'].set_linewidth(1.5)
	ax.spines['right'].set_linewidth(1.5)
	ax.spines['left'].set_linewidth(1.5)

	ax.xaxis.set_major_locator(xmajorLocator)
	ax.yaxis.set_major_locator(ymajorLocator)

	ax.xaxis.set_major_formatter(xmajorFormatter)
	ax.yaxis.set_major_formatter(ymajorFormatter)

	# for the minor ticks, use no labels; default NullFormatter
	ax.xaxis.set_minor_locator(xminorLocator)
	ax.yaxis.set_minor_locator(yminorLocator)

	# grid and axes are drawn below the data plot
	ax.set_axisbelow(True)

	# convert x axis units to radians
	#ax.convert_xunits(radians)

	# add x,y grids to plot area
	ax.xaxis.grid(True, zorder=0, color='lightgray', linestyle='-', linewidth=1)
	ax.yaxis.grid(True, zorder=0, color='lightgray', linestyle='-', linewidth=1)

	# set axis labels
	ax.set_xlabel(custom_x_label, fontsize=16)
	ax.set_ylabel(custom_y_label, fontsize=16)

	# set plot title
	#ax.set_title(custom_title, loc='right', fontsize=12)

	return;

custom_axis_formater(plot_title, axis_label_nm, axis_label_delta_Ryz, xmin, xmax, ymin, ymax, xprec, yprec)

# plot data and fit
tx1, = plt.plot(x1, y1, 'ko', mfc='red', markersize=6, label=label_xy)
tx2, = plt.plot(xdf, Delta_R_R0_ma, 'r-', lw=1, markersize=6, label=label_xy_fit)
plt.legend([tx1, tx2], [label_xy, label_xy_fit], loc='lower left', frameon=False)
plt.axis([xmin, xmax, ymin, ymax])

axcolor = 'white'
# updating plot rc params to increase font size
plt.rc('font', size=16)
# slider position and size
slider_width = 0.65
slider_height = 0.01
slider_pos_x = 0.15
slider_pos_y = 0.33 
slider_sep = 0.015

#slider_theta_AH =  plt.axes([slider_pos_x, slider_pos_y, slider_width, slider_height], frameon=True, facecolor=axcolor)
slider_theta_SH =  plt.axes([slider_pos_x, slider_pos_y - 1.0*(slider_sep), slider_width, slider_height], frameon=True, facecolor=axcolor)
#slider_theta_F =  plt.axes([slider_pos_x, slider_pos_y - 2.0*(slider_sep), slider_width, slider_height], frameon=True, facecolor=axcolor)
#slider_P =  plt.axes([slider_pos_x, slider_pos_y - 3.0*(slider_sep), slider_width, slider_height], frameon=True, facecolor=axcolor)
slider_Gr =  plt.axes([slider_pos_x, slider_pos_y - 4.0*(slider_sep), slider_width, slider_height], frameon=True, facecolor=axcolor)
#slider_d_N = plt.axes([slider_pos_x, slider_pos_y - 5.0*(slider_sep), slider_width, slider_height], facecolor=axcolor)
#slider_d_F = plt.axes([slider_pos_x, slider_pos_y - 2.0*(slider_sep), slider_width, slider_height], facecolor=axcolor)
slider_lambda_N = plt.axes([slider_pos_x, slider_pos_y - 6.0*(slider_sep), slider_width, slider_height], facecolor=axcolor)
#slider_lambda_F = plt.axes([slider_pos_x, slider_pos_y - 7.0*(slider_sep), slider_width, slider_height], facecolor=axcolor)
slider_rho_N = plt.axes([slider_pos_x, slider_pos_y - 8.0*(slider_sep), slider_width, slider_height], facecolor=axcolor)
slider_rho_F = plt.axes([slider_pos_x, slider_pos_y - 9.0*(slider_sep), slider_width, slider_height], facecolor=axcolor)
slider_Constant = plt.axes([slider_pos_x, slider_pos_y - 10.0*(slider_sep), slider_width, slider_height], facecolor=axcolor)

# min max settin values for the fit parameters
#set_R0 = Slider(slider_R0, 'R0', -10.0, 10.0, valinit=R0, valstep=valstep_R0)
#set_theta_AH = Slider(slider_theta_AH, r'$\theta_{AH}$', -10*theta_AH, 10*theta_AH, valfmt='%1.5f', valinit=theta_AH)
set_theta_SH = Slider(slider_theta_SH, r'$\theta_{SH}$', -10*theta_SH, 10*theta_SH, valfmt='%1.5f', valinit=theta_SH, color='lightgray')
#set_theta_F = Slider(slider_theta_F, r'$\theta_{F}$', -10*theta_F, 10*theta_F, valfmt='%1.5f', valinit=theta_F)
#set_P = Slider(slider_P, r'$P$', -10*P, 10*P, valfmt='%1.5f', valinit=P)
set_Gr = Slider(slider_Gr, r'$Gr$', -1e+6*Gr, 1e+6*Gr, valfmt='%1.5e', valinit=Gr, color='lightgray')
#set_d_N = Slider(slider_d_N, r'$d_N$', -10.0*d_N, 10.0*d_N, valfmt='%1.5f', valinit=d_N)
#set_d_F = Slider(slider_d_F, r'$d_F$', -10.0*d_F, 10.0*d_F, valfmt='%1.5f', valinit=d_F)
set_lambda_N = Slider(slider_lambda_N, r'$\lambda_N$', 0.0*lambda_N, 2.0*lambda_N, valfmt='%1.5f', valinit=lambda_N, color='lightgray')
#set_lambda_F = Slider(slider_lambda_F, r'$\lambda_F$', -10.0*lambda_F, 10.0*lambda_F, valfmt='%1.5f', valinit=lambda_F, color='lightgray')
set_rho_N = Slider(slider_rho_N, r'$\rho_N$', -10.0*rho_N, 10.0*rho_N, valfmt='%1.5f', valinit=rho_N, color='lightgray')
set_rho_F = Slider(slider_rho_F, r'$\rho_F$', -10.0*rho_F, 10.0*rho_F, valfmt='%1.5f', valinit=rho_F, color='lightgray')
set_Constant = Slider(slider_Constant, r'$Const$', -10.0*Constant, 10.0*Constant, valfmt='%1.5f', valinit=Constant, color='lightgray')

resetax = plt.axes([slider_pos_x, slider_pos_y - 11.5*(slider_sep), slider_width, slider_height + 0.005])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

fig1, ax = plt.subplots(figsize=(9, 9))
fig1.tight_layout(pad=4.0, w_pad=6.0, h_pad=2.0)
fig1.canvas.set_window_title('mr_fit_model' + model_symbol) 
plt.subplots_adjust(left=0.15, bottom=0.15, wspace=0.0, hspace=0.0)
#plt.figtext(0.80, 0.99, version_name, size=8)

# updating plot plot parameters with sliders
def update(val):

#	theta_AH = set_theta_AH.val
	theta_SH = set_theta_SH.val
#	theta_F = set_theta_F.val
#	P = set_P.val
	Gr = set_Gr.val
#	d_N = set_d_N.val
	lambda_N = set_lambda_N.val 
#	lambda_F = set_lambda_F.val
	rho_N = set_rho_N.val 
	rho_F = set_rho_F.val
	Constant = set_Constant.val

	tx2.set_ydata(-1.0*(theta_SH**2)*lambdan_dn*( ((np.sinh(d_N / 2.0*lambda_N) / np.cosh(d_N / 2.0*lambda_N))**2) / (1 + (rho_N*d_F / rho_F*d_N)) )*( 2*rho_N*lambda_N*Gr / (1 + (2*rho_N*lambda_N*Gr)*(1 / (np.sinh(d_N / lambda_N) / np.cos(d_N / lambda_N)) )) ) + Constant)
		#- ((( 1-(P**2))*rho_N*lambda_N ) / ( rho_F*lambda_F*(1 / (np.sinh(d_F/lambda_F) / np.cosh(d_F/lambda_F))) ) / (1 + ((( 1-(P**2))*rho_N*lambda_N ) / ( rho_F*lambda_F*(1 / tanh_df_lambdaf) ))*(1 / (np.sinh(d_N / lambda_N) / np.cos(d_N / lambda_N)) )) )
	tx4.set_ydata(-1.0*(theta_SH**2)*lambdan_dn*( ((np.sinh(d_N / 2.0*lambda_N) / np.cosh(d_N / 2.0*lambda_N))**2) / (1 + (rho_N*d_F / rho_F*d_N)) )*( 2*rho_N*lambda_N*Gr / (1 + (2*rho_N*lambda_N*Gr)*(1 / (np.sinh(d_N / lambda_N) / np.cos(d_N / lambda_N)) )) ) + Constant)
	tx5.set_ydata(Constant)
	
	fig.canvas.draw_idle()
	fig1.canvas.draw_idle()

# setting the slider values
#set_theta_AH.on_changed(update)
set_theta_SH.on_changed(update)
#set_theta_F.on_changed(update)
#set_P.on_changed(update)
set_Gr.on_changed(update) 
#set_d_N.on_changed(update)  
#set_d_F.on_changed(update) 
set_lambda_N.on_changed(update) 
#set_lambda_F.on_changed(update)
set_rho_N.on_changed(update) 
set_rho_F.on_changed(update)
set_Constant.on_changed(update)

# reseting the plot to its initial values
def reset(event):

	#set_theta_AH.reset()
	set_theta_SH.reset()
	#set_theta_F.reset()
	#set_P.reset()
	set_Gr.reset() 
	#set_d_N.reset()  
#	set_d_F.reset() 
	set_lambda_N.reset() 
	#set_lambda_F.reset()
	set_rho_N.reset() 
	set_rho_F.reset()
	set_Constant.reset()
  
button.on_clicked(reset)


custom_axis_formater(plot_title, axis_label_nm, axis_label_delta_Ryz, xmin, xmax, ymin, ymax, xprec, yprec)
# plot data and fit
tx3, = plt.plot(x1, y1, 'ro', mfc='lavenderblush', markersize=6, label=label_xy)
tx4, = plt.plot(xdf, Delta_R_R0_ma, 'r-', lw=1, markersize=6, label=label_xy_fit)
plt.legend([tx3, tx4], ['expt', 'fit'], loc='upper right', frameon=True, fontsize=9)
tx5, = plt.plot(xdf, np.full_like(xdf, Constant), 'r-.', lw=1, markersize=6, label=label_xy_fit)
plt.axis([xmin, xmax, ymin, ymax])

#plt.figtext(desc_x, desc_y, r'$R_{y-z} = -x_N \Theta^2 \tanh \frac{d_N}{2 \lamda_N}(1 - \frac{1}{\cosh \frac{d_N}{\lambda_N}})$')

plt.show()
