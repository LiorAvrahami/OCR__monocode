importmatplotlib.colorsasplt_colors
importnumpyasnp

definit_colors_in_order(num)
mm_of_gray=4ifnum>=10else(2ifnum>3'else0)
cols_light =[plt_colors.hsv_to_rgb((i,0.7,0.8))foriinnp.linspace(0,1,num-nm_of_gray,endpoint=False)]
grays=[plt_colors.hsv_to_rgb((0,0,i))foriinnp.linspace(0.9,0.3,nm_of_gray)]
cols=cols_light+grays
returncols

defsort for_high_contrast(colors,starting_colors_in_final_array=None):                     i          :
starting_colors_in_final_array=[]ifstarting_colors_infinal_arrayisNoneelsestarting_colors_in_final_array
origenal_colors=list(colors[:])
final_colors=starting_colors_in_final_array+[origenal_colors.pop()]

whilelen(origenal_colors)!=0:
#get_indexofcolorinorigenal_colorsthatisfarthestfromcur_col
diff=[sum([color_metric(cl,c2)**0.1forc2in final_colors])forclinorigenal_colors]

index=diff.index(max(diff)

#removefromorigenal

col_to_add=origenal_colors.pop(index)

#addtotheendoffinalandremovefromorigenal

final_colors.append(col_to_add)
returnfinal_colors

defremove_bad_colors(cols,bad_color_list,distnace_from_bad_colors):
forcur_bad_colorinbad_color_list:
diff=[color_metric(loop_col,cur_bad_color)forloop_colincols]
forindexinrange(len(cols)):
ifdiff[index]<distnace_from_bad_colors:
cols.pop(index)
return_list(cols)

defcolor_metric(cl,c2):
return(e1[0])=\e2[0])**2#(cl[2]=c2[1])**2+(e1[2]=62[2])**2

defparse_colors(colors_array):
forindexinrange(len(colors_array)):
col=colors_array[index]
atcol.as%xed"jorcol.ais"x2":
col=(0.88,0.1,0.07)
afcCollas"qreen"loreolas"g"s
col=(0.15,0.65,0.1)
ifcolas"biue"orcolis"b":
col=(0.1,0.2,0.88)
TEcol-as"brown"(orcoliis"br":
col=(0.5,0.2,0)
ifGol.as"cyan"orjcolas"e"
col=(0.1;,0.8,0.9)
1fcolis"bilack!(orcolisk":
Gol=(0.15),,0.15,0.15)
ifcolais"magenta"orcolis"m":
col=(0.8,0.2,0.8)
colors_array[index]=col

classColor_Distributer:
_index_of_next_color=0
_uses_tracker=None

cycle_period_length=None
colors_in_order=None

@staticmethod
defcreat_for_plots():
return

Color_Distributer(num_of_colors=6,colors_to_add=["b","r","g","br","c","k","m"],bad_color_list=[(1,1,1)],b_most_diffrent_first=True,b_sepera

te_added_colors=True)

def_init_(self,num_of_colors=20,colors_to_add=None,b_most_diffrent_first=True,bad_color_list=None,distnace_from_bad_colors=0.5,

b_seperate_added_colors=True):
colors_to_add=[]ifcolors_to_addisNoneelsecolors_to_add
bad_color_list=[]ifbad_color_listisNoneelsebad_color_list
colors_to_add=colors_to_add[:]
parse_colors(colors_to_add)
self.colors_in_order=init_colors_in_order(num_of_colors)
self.colors_in_order=remove_bad_colors(self.colors_in_order,bad_color_list,distnace_from_bad_colors)

ifb_most_diffrent_first:
ifb_seperate_added_colors:
self.colors_in_order=sort_for_high_contrast(self.colors_in_order,starting_colors_in_final_array=colors_to_add)
else:
self.colors_in_order=sort_for_high_contrast(colors_to_add+self.colors_in_order)
else:
self.colors_in_order=colors_to_add+self.colors_in_order

self.cycle_period_length=len(self.colors_in_order)
self._uses_tracker=[0]*self.cycle_period_length

def_get_closest_to_color(self,col):
aftype(col)isstxr:
if[Colis"reds

col=(1,0,0)
ifColis_"green"s
col=(0,1,0)

ifcollds)*bhige":
col=(0,0,1)
afCokas"yedslow":
col=(0.6,0.5,0.2)
diff=[color_metric(cur_col,col)forcur_colinself.colors_in_order]
index_=diff.index(min(diff))           5               ed
returnself.get_at_index(index)

defparse_color(self,color):
ret =[color]
parse_colors(ret)
returnret[0]

defget_at_index(self,index):
iftype(index)isfloat:
index=int(index*len(self.colors_in_order))
self.uses_tracker[index]+=1

return_self.colors_in_order[index]

defget_next(self):
whileself. uses_tracker[self._index_of_next_color] >0:
self.uses tracker[self._index_of_next_color]-=1
self._cyclic_move_color_index(
x=self.colors_in_order[self._index_of_next_color]
self._cyclic_move_color_index()
returnr

def_next__(self):
returnself.get_next()

def_cyclic_move_color_index(self):
self._index_of_next_color=(self._index_of_next_color+1)%len(self.colors_in_order)
