
importmatplotlib.pyplotasplt
importnumpyasnp

fromtimeimporttime

importwin32api,win32con
fromscipy.ndimageimportgaussian_filter
fromColor_DistributerimportColor_Distributer
importos,itertools

file_namel="v3/mouse_speed_res5"
file_name2="v3/mouse_speed_res5_new"

defload_draw_f(file_name):
#load
trigger_times=np.array([])
f=np.load(file_name+".npz")
txy_np_arr=f["arr_0"]
trigger_times=f["arr_1"]

#find_outliers

t=txy_np_arr[:,0]

xX=np.cumsum(txy_np_arr[:,1])

y=np.cumsum(txy_nparr[:,_2])
trigger_indexes=np.interp(trigger_times,t,range(len(t))).astype(int)
test_x_graphs=[]

test_std=[]

foriinrange(0,len(trigger_times),2):

t_test=np.concatenate([[trigger_times[i]],t[trigger_indexes[i]:trigger_indexes[i+1]],[trigger_times[i+1]]])

x_test=np.concatenate([[np.interp(trigger_times[i],t,x)],x[trigger_indexes[i]:trigger_indexes[i+1]],
[np.interp(trigger_times[i+1],t,x)]])

test_x_graphs.append((t_test,x_test)

test_std.append(np.std(np.diff(x_test)[np.diff(x_test)!=0]))

test_colors=Color_Distributer(len(test_x_graphs),b_most_diffrent_first=False).colors_in_order

plt.figure(

plt.plot(t,x,"-o",label="x")

plt.plot(t,y,label="y")

plt.ylim(*plt.ylim())

#plottriggertimes
plt.vlines(trigger_times,*plt.ylim(),colors=(1,0.1,0.1,0.8),linestyles="--"

plt.xlabel("t");plt.ylabel("x");plt.title("extractedtests")
fori,ginenumerate(test_x_graphs):
plt.plot(*g,"-o",color=test_colors[i])

load_draw_f(file_namel)
load_draw_f(file_name2)

plt.show()
