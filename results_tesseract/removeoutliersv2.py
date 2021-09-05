
importmatplotlib.pyplotasplt
importnumpyasnp

fromtimeimporttime

importwin32api,_win32con
fromscipy.ndimageimportgaussian_filter
fromColor_DistributerimportColor_Distributer
importos,itertools

file_name

"v3/mouse_speed_res1"

#load
trigger_times=np.array([])
f= np.load(file_name Menp2)
txy_np_arr=f["arr_0"]
trigger_times=f["arr_1"]

#find_outliers

==txynpvarri[:;,0]

X=np.cumsum(txy_np_arr[:,1])

y=np.cumsum(txy_np_arr[:,2])

trigger_indexes=np.interp(trigger_times,t,range(len(t))).astype(int)

test_x_graphs=[]

test_std=[]

foriinrange(0,len(trigger_times),2):
t_test=np.concatenate([[trigger_times[i]],t[trigger_indexes[i]:trigger_indexes[i+1]],[trigger_times[i+1]]])
x_test=np.concatenate([[np.interp(trigger_times[i],t,x)],x[trigger_indexes[i]:trigger_indexes[i+1]], [np.interp(trigger_times[i+1],

t,x)]1)
test_x_graphs.append((t_test,x_test))

test_std.append(np.std(np.diff(x_test)[np.diff(x_test)!=0]))
out_liers_indexes=[]
plt.pause(0.1
plt.ion()

fori,testinenumerate(test_x_graphs):

plt.gcf().clf()         i

plt.plot(*test)

plt.gcf().canvas.draw_idle()

plt.gcf().canvas.start_event_loop(0.3)

plt.draw()

whileTrue:
print("selectisoutlier:\nyforyes\nnoremptyforno"
r=input()

ifr=="y":
out_liers_indexes.append(i)
break
LE=="n"orrS=s
break

iflen(out_liers_indexes)==0:
print("nooutliersselected")
quit()

#modify

out_liers_indexes=[a%(len(trigger_times)/2)forainout_liers_indexes]
out_liers_indexes=[a*2forainout_liers_indexes]+[a*2+1forainout_liers_indexes]
trigger_times=list(trigger_times)
trigger_times=[trigger_times[n]forninrange(len(trigger_times))ifnnotinout_liers_indexes]
trigger_times=np.array(trigger_times)

#save
np.savez(file_name+"_new.npz",txy_np_arr,trigger_times)
