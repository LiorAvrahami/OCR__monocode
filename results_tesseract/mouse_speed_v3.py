
importmatplotlib.pyplotasplt
importnumpyasnp

fromtimeimporttime,ctime
importwin32api,win32con
fromscipy.ndimageimportgaussian_filter
fromColor_DistributerimportColor_Distributer
importos,itertools

Exyare=[]
txy_np_arr=np.zeros((0,3))
new_point_btn_down=False
delete_point_btn_down=False

Load=True

ifLoad:
trigger_times=np.array([])
for1initertools.count():
name=f"v3\\mouse_speed_res{i}.npz"
ifnotos.path.exists(name):
break
f=np.load(name)
new_txy=f["arr_0"]
new_trigger_times=f["arr_1"]
iflen(txy_np_arr)_!=0:
toadd=np.max(txy_np_arr[:,0])
new_txy[:,0]+=toadd
new_trigger_times+=toadd
txy_np_arr=np.concatenate([txy_np_arr,new_txy])
trigger_times=np.concatenate([trigger_times,new_trigger_times])

else:
trigger_times=[]
po=(1000,600)
to=time()

win32api.SetCursorPos(p0)

defprint_state():
iflen(trigger_times)%2
iflen(trigger_times)

 
print(trigger_times[-1]- trigger_times[-2])
print("waitingforentrance")
else:
print("waitingforexit")

last_xy=p0
whilewin32api.GetAsyncKeyState(win32con.VK_ESCAPE)==
t=time()-t0
xy=win32api.GetCursorPos()
txy_arr.append((t,xy[0]-last_xy[0],xy[1]-last_xy[1]))

ifxKy[0]<100o~%xy[1]~100orxy[0]>1820or-xyil])>1100:
win32api.SetCursorPos(p0)
last_xy=po

else:
last_xy=xy

ifwin32api.GetAsyncKeyState(win32con.VK_INSERT)!=0:;
ifnotnew_point_btn_down:
trigger_times.append(t)
print_state()
new_point_btn_down=True
else:
new_point_btn_down=False
ifwin32api.GetAsyncKeyState(win32con.VK_END)!=0:
ifnotdelete_point_btn_down:
iflen(trigger_times)%2==
trigger_times.pop()
else:
trigger_times.pop()
trigger_times.pop()
print_state()
delete_point_btn_down=True
else:
delete_point_btn_down=False

txy_np_arr=np.array(txy_arr)

indexes_to_keep=np.convolve(txy_np_arr[:

txy_np_arr=txy_np_arr[indexes_to_keep,:]

trigger_times=np.array(trigger_times)

#save

foriinitertools.count():
name=f"v3\\mouse_speed_res{i}.npz"
ifnotos.path.exists(name):

np.savez(name,txy_np_arr,trigger_times)

 

1],[1,1,1],mode="same")!=0

break
t=txy_np_arr[:,0]
X=np.cumsum(txy_np_arr[:,1])
v= np.cumsum(txy_np_arr[:, 2])
#plotraw

plt.figure()

plE.plot(&;x,;"-0",_Label="x")

plt.plot(t,y,label="y")

plt.ylim(*plt.ylim())

#plottriggertimes
pit.vlines(trigger_times,*plt.ylim(),colors=(1,0.1,0.1,0.8),linestyles="--")

#-extracttest
trigger_indexes=
test_x_graphs=[]
test_std=[]
foriinrange(0,len(trigger_times),2):
t_test=np.concatenate([[trigger_times[i]l],tltrigger_indexes[i]:trigger_indexes[i+1]],[trigger_times[i+1]]])

x_test=np.concatenate([[np.interp(trigger_times[i],t,x)],x[trigger_indexes[i]:trigger_indexes[i+1]],[np.interp(trigger_times[i+1]t,x)]))


test_x_graphs.append((t_test,x_test)
test_std.append(np.std(np.diff(x_test)[np.diff(x_test)!=0]))

np.interp(trigger_times,t,range(len(t))).astype(int)

test_colors= Color_Distributer(len(test_x_graphs), b_most_diffrent_first=False).colors_in_order
plt.figure()
plt.xlabel("t");plt.ylabel("x");plt.title("extractedtests")
fori,ginenumerate(test_x_graphs):
plt.plot(*g,_color=test_colors[i]
plt.figure()
bins=np.linspace(0,np.max(np.abs(np.diff(x))),100)
plt.xlabel("f(dx)");plt.ylabel("distribution");plt.title("distributionoff(dx)")
fori,ginenumerate(test_x_graphs):
np.abs(np.diff(x_test)[np.diff(x_test)!=0])
y,_=np-histogram(np.abs(np.diff(g{1])),bins)
plt.plot(bins[1:-1]+np.diff(bins[1:])/2,y[1:],color=test_colors[i],alpha=0.75)
plt.figure()
plt.xlabel("dt");plt.ylabel("f(dx)");plt.title("dtf(dx)correlationplot")
fori,ginenumerate(test_x_graphs):
plt.plot(np.diff(g[0]),np.abs(np.diff(g[1])),"o",alpha=0.3,color=test_colors[i])

#meananalysis
f_mean_with_error=[]

g_mean_with_error=[]
fortest,stdinzip(test_x_graphs,test_std):
N=np.sum(np.diff(test[0])!=0)#todofuckesupwith g_mean_with_errorforsomereason

total_time=test[0][-1]-test[0][0]
tot_length_rel=25
total_time_error=0.05

tot_length_screen=np.abs(test[1][-1]-test[1][0]
f_mean_with_error.append((tot_length_rel/N,tot_length_screen/N,std))

mean_vel_rel=tot_length_rel/total_time
mean_vel_screen=tot_length_screen/total_time

g_mean_with_error.append((mean_vel_rel, tot_length_screen /N,std,mean_vel_rel*total_time_error/total_time))

plt.figure()
plt.xlabel("V_mouse[cm/s]");plt.ylabel("f(dx_mouse)_[pixels]");plt.title("meananalysisresultsofV_sreenvsV_mouse")
plt.plot([0],[0],"x",color="k")

plt.errorbar(*zip(*g_mean_with_error),ecolor=test_colors,fmt="none",alpha=0.5)

plt.scatter(*((*zip(*g mean_with error),){0:2]), c=np.array(test_colors))

plt.grid(ls="--")     a    7    7

 

plt.figure()
plt.xlabel("dx_mouse");plt.ylabel("f(dx_mouse)");plt.title("meananalysisresultsoff(dx_mouse)vsdx_mouse")
ple-pist4(O1,(01,"e",calor="k")

plt.errorbar(*zip(*f mean_with_error), ecolor=test_colors, fmt="none",alpha=0.5)
plt.scatter(*((*zip(*f_mean_with_error),)[0:2]), c=np.array(test_colors))
plt-.grid(ls="--")

#smoothoutandsave

fromsmoothimportestimate_val_at_x
f_mean_with_error=np.array(f_mean_with_error+[(0,0,0.00001)])
weights=1/f_mean_with_error[:,2]

xt=np.linspace(0,max(f_mean_with_error[:,0]),1000

width=np.linspace(0.1,1.3,len(xt))
yt=estimate_val_at_x(xt,f_mean_with_error[:,0],f_mean_with_error[:,1],width,3,weights)
plt.plot(xt,yt)

importscipy.ioasspio
spio.savemat("v3\\smoothed_out_signal",{"f(dx)_vs_dx":(xt,yt),"time":ctime()})
