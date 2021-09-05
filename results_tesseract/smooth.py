
importnumpyas_np

defestimate_val_at_x(xt,x_data,y_data,window_width,polynomial_degree,weights=None):
non
tparamxt:1ldarrayofsizeN.thetargetpointsatwhichthesmoothsignalistobeestimated
tparamx_data:1darrayofsizeM.noisyxdata
:paramy_data:1ldarrayofsizeM.noisyydata
:paramwindow_width:scalaror1darrayofsizeNorM,or2darrayofsize(N,M).widthofgaussianwindow(2std'sofgaussian)
s:parampolynomial_degree:degreeofpolynomial
:paramweights:weightsofnoisydatapoints,fornormaldistributions,useweight_of_point=1/sigma_of_point

:return:
non

ifweightsisNone:

weights=[1/len(y_data)]*len(y_data)
iftype(window_width)isnp.ndarray:
iflen(xt)==len(x_data)andwindow_width.shape!= (len(xt),len(x_data)):

raiseValueError("window_widthisgivenasaldarray,whilextandx_datahavethesamelength."
"thisisnotallowedbecauseitcannotbeinferredwhichofthetwoisused."
"pleaseinsertwindow_widthin2darrayformat,(orchangethesizesofx_dataorxttobedifferent)")

ifwindow_width.shape==(len(xt),):
window_width=np.tile(window_width, (len(x_data),1)).T
elifwindow_width.shape==(len(x_data),):

window_width =np.tile(window_width,(len(xt),1))
weights=np.array(weights)
yt=np.full(xt.shape,np.nan,np.float)
x_data_mat=np.tile(x_data,(len(xt),1))
xt_mat_=np.tile(xt,(len(x_data),1)).T
slidingweight_matrix=np.exp(-(x_data_mat- xt_mat)**2/(2*(window_width/2)**2))

foriinrange(sliding_weight_matrix.shape[0]):                        aps     ;
ai=np.polyfit(x_data,y_data,polynomial_degree, w=sliding_weight_matrix[i,:]*weights)
yt[i]_=sum([xt[i]**n*ai[len(ai)-n-1]forninrange(len(ai))])

returnyt

if7name-="oomaina:

importmatplotlib.pyplotasplt

xr_=np.linspace(-2,2,300)

np.random.seed(1)

X=xr+np.random.uniform(-0.5,0.5,xr.shape)

y=xr**2+np.random.uniform(-0.1,0.1,xr.shape)

plt.figure()

plt.plot(x,y,"o")

v=[(0.3,1),(1,1),(0.3,3),(1,3)]

forpl,p2inv:
yr=estimate_val_at_x(xr,x,y,pl,p2,weights=1/(np.abs(xr-x)+0.1))
plt.plot(xr,yr,label=f"{p1},{p2}")

plt.legend()
plt.grid(ls="--")
