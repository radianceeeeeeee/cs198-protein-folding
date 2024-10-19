from numpy import shape,tile,r_,array,append,concatenate,diff,floor,hstack,vstack,unique,intersect1d,ones,where,sort,argsort


def e2d(y):
# Local extrema search for 2D data

  ymax,imax,ymin,imin = [],[],[],[]

  leny = len(y)
  indy = array(xrange(0,leny))

  # Difference between subsequent elements:
  dy = diff(y,n=1)


  # Flat peaks? Put the middle element:
  ind = array(where(dy != 0)).ravel()              # Indices where y changes
  nc = array(where(diff(ind,n=1) != 1)).ravel()    # Indices where ind do not change
  inc = array([ind[i] if i in nc else 0 for i in xrange(len(ind))]) # a(nc)
  inc1 = array([ind[i-1] if i in nc else 0 for i in xrange(len(ind))]) # a(nc-1)
  d2 = array(floor((inc-inc1)/2),dtype=int)   # Number of elements in the flat peak 
  ind = append(ind,leny-1)


  # Peaks?
  ya  = array([y[i] for i in ind]).ravel()    # Series without flat peaks

  b = array((diff(ya,n=1) > 0),dtype=int)     # 1  =>  positive slopes (minima begin)  
                                              # 0  =>  negative slopes (maxima begin)

  yb  = diff(b,n=1)                           # -1 =>  maxima indexes (but one) 
                                              # +1 =>  minima indexes (but one)

  imax = array(where(yb == -1)).ravel()  # maxima indices
  imax = array([ind[i] for i in imax+1])
  imin = array(where(yb ==  1)).ravel()  # minima indices
  imin = array([ind[i] for i in imin+1])
 
  nmaxi = len(imax)
  nmini = len(imin)        

  
  if (nmaxi == 0) and (nmini == 0): # Maximum or minimum on a flat peak at the ends?
    if y[0] > y[leny-1]:
      imax = [indy[0]]
      imin = [indy[leny-1]]
    elif y[0] < y[leny-1]:
      imax = [indy[leny-1]]
      imin = [indy[0]]
  
  elif (nmaxi == 0) and (nmini > 0): # Maximum at the ends?
    imax = array([0,leny-1],dtype=int) 

  elif (nmaxi > 0) and (nmini == 0): # Maximum at the ends?
    imin = array([0,leny-1],dtype=int) 
    
  else:
    if imax[0] < imin[0]:
      imin = hstack(([0],imin))
    else:
      imax = hstack(([0],imax))

    if imax[len(imax)-1] > imin[len(imin)-1]:
      imin = append(imin,[leny-1])
    else:
      imax = append(imax,[leny-1])


  ymax = [y[i] for i in imax]
  ymin = [y[i] for i in imin]

  # Descending order
  nmax = argsort(ymax)
  nmax[:]=nmax[::-1]
  ymax = [ymax[i] for i in nmax]
  imax = [imax[i] for i in nmax]

  nmin = argsort(ymin)
  ymin = [ymin[i] for i in nmin]
  imin = [imin[i] for i in nmin]


  return ymax, imax, ymin, imin

# ---------------------------------------------------------------------------------------------

def e3d(G):
# Local extrema search for 3D data

	R,C = shape(G)

	# Search peaks through columns:
	smaxcol,smincol = extremes(G)

	# Search peaks through rows, on columns with extrema points:
	im = unique(r_[smaxcol[:,0],smincol[:,0]]) # Rows with column extrema
	Gim = array(G[im,:]).T
	smaxrow,sminrow = extremes(Gim)


	# # Conversion from 2 to 1 index:
	smaxcol = array(smaxcol[:,0] + ((smaxcol[:,1])*shape(G)[0]))
	smincol = array(smincol[:,0] + ((smincol[:,1])*shape(G)[0]))
	smaxrow = array(smaxrow[:,1] + (im[smaxrow[:,0]]*shape(G)[0]))
	sminrow = array(sminrow[:,1] + (im[sminrow[:,0]]*shape(G)[0]))


	# Peaks in rows and in columns:
	smax = intersect1d(smaxcol,smaxrow)
	smin = intersect1d(smincol,sminrow)


	# Check peaks on down-up diagonal:
	idx = unique(r_[smax,smin])
	iext = idx % R
	jext = (idx - iext)/R
	Gemax,Gemin = extremes_diag(iext,jext,G,1) 


		# Check peaks on up-down diagonal:
	smax = intersect1d(smax,r_[R-1, R*C-R-1, Gemax])
	smin = intersect1d(smin,r_[R-1, R*C-R-1, Gemin])


	# Peaks on up-down diagonals:
	idx = unique(r_[smax,smin])
	iext = idx % R
	jext = (idx - iext)/R
	Gemax,Gemin = extremes_diag(iext,jext,G,-1)


	# Peaks on columns, rows and diagonals:
	smax = intersect1d(smax,r_[0,R*C-1,Gemax])
	smin = intersect1d(smin,r_[0,R*C-1,Gemin])

	zmax = array(G[smax/C, smax % C])
	zmin = array(G[smin/C, smin % C])

	nmax = argsort(zmax)
	nmax[:]=nmax[::-1]
	imax = smax[nmax]
	zmax = zmax[nmax]

	nmin = argsort(zmin)
	imin = smin[nmin]
	zmin = zmin[nmin]

	return zmax, imax, zmin, imin

# ---------------------------------------------------------------------------------------------

def extremes(G): 
# Peaks through columns or rows.

	a,b,c,d = [],[],[],[]
	R,C = shape(G)

	for i in xrange(R):
		arr = [G[i,j] for j in xrange(C)]

		tmp,imaxrow,tmp1,iminrow = e2d(arr)
		imaxrow = array(imaxrow,dtype=int)
		iminrow = array(iminrow,dtype=int)

		if imaxrow.any():     # Maxima indexes
			a = array(concatenate((a,imaxrow),axis=0))
			imaxcol = tile(i,(len(imaxrow),1)).ravel()
			b = array(concatenate((b,imaxcol),axis=0),dtype=int)
		
		if iminrow.any():     # Minima indexes
			c = array(concatenate([c,iminrow]))
			imincol = tile(i,(len(iminrow),1)).ravel()
			d = array(concatenate([d,imincol]),dtype=int)

	smax = array(vstack((a,b)).T,dtype=int)
	smin = array(vstack((c,d)).T,dtype=int)

	return smax, smin

# ---------------------------------------------------------------------------------------------

def extremes_diag(rext,cext,G,arg): 
# Peaks through diagonals (down-up, a = -1)

	R,C = shape(G)

	if arg == -1:
		rext = R - rext - 1

	rini,cini = cross(rext,cext,0,0)

	idx = unique(rini + (cini*shape(G)[0]))
	rini = idx % R 
	cini = (idx-rini)/R  
	rfin,cfin = cross(rini,cini,R-1,C-1)


	extmax,extmin = [],[]
	for i in xrange(len(rini)): 
		rses = array(xrange(rini[i],rfin[i]+1))
		cses = array(xrange(cini[i],cfin[i]+1))

		if arg == -1:
			rses = R - rses - 1

		s = rses + (cses*shape(G)[0])
		Gs = array(G[cses,rses])
		tmp,imax,tmp,imin = e2d(Gs) 

		extmax.extend(s[imax])
		extmin.extend(s[imin])

	return array(extmax), array(extmin)

# ---------------------------------------------------------------------------------------------

def cross(r0,c0,R,C):
# Indexes where the diagonal of the element r0,c0 crosses the left/superior
# (R=0,C=0) or right/inferior (R=M,C=N) side of an MxN matrix. 
	an = 2*(R*C == 0) - 1
	si = array(where(an*(c0-C) > an*(r0-R)),dtype=int).ravel()
	m = array([1 if i in si else 0 for i in xrange(len(r0))])

	r = (R-C-r0+c0)*m + C + r0 - c0 
	c = (R-C-r0+c0)*m + C

	return r,c

