import cv2
import numpy as np
import os

class FLANN:
	def __init__(self,feature_dir="",distance=0.75,min_score=10,RANSAC_threshold=5.0,FLANN_INDEX_KDTREE=1,nKDtrees=5,nLeafChecks=50):
		'''
		distance         : matching distance [0,1]
		min_score (>=4)  : number of match pointing points should be found
		RANSAC_threshold : noise threshold
		'''
		self.sift=cv2.SIFT.create()
		indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=nKDtrees)
		searchParams=dict(checks=nLeafChecks)
		self.flann=cv2.FlannBasedMatcher(indexParams,searchParams)
		self.min_score=min_score
		self.distance=distance
		self.RANSAC_threshold=RANSAC_threshold
		self.kp=[]
		self.desc=[]
		self.feature_sz=[]
		if(not feature_dir or not os.path.exists(feature_dir)): return
		for e in os.listdir(feature_dir):
			gray=cv2.imread(os.path.join(feature_dir,e),cv2.IMREAD_GRAYSCALE)
			self.feature_sz.append(gray.shape)
			kp, desc=self.sift.detectAndCompute(gray,None)
			if(len(kp)<2): continue
			self.kp.append(kp)
			self.desc.append(desc)

	def __call__(self,gray):
		if(gray is None or gray.size == 0): return 0
		gray=cv2.cvtColor(gray,cv2.CV_8U)
		kp, desc=self.sift.detectAndCompute(gray,None)
		if(len(kp)<2): return
		best=(-1,[])  # id, pts
		for i,e in enumerate(self.desc):
			matches=self.flann.knnMatch(e,desc,k=2)
			pts=[]
			for m,n in matches:
				if(m.distance<self.distance*n.distance):
					pts.append(m)
			if(len(pts)>len(best[1])): best=(i,pts)
		if(best[0]==-1 or len(best[1])<self.min_score): return []

		query_pts=np.float32([self.kp[best[0]][m.queryIdx].pt for m in best[1]]).reshape(-1,1,2)
		train_pts=np.float32([kp[m.trainIdx].pt for m in best[1]]).reshape(-1,1,2)
		
		mtx, msk=cv2.findHomography(query_pts,train_pts,cv2.RANSAC,self.RANSAC_threshold)
		# matches_msk=msk.ravel().tolist()
		if(mtx is None): return []

		h,w=self.feature_sz[best[0]]
		return [np.int32(cv2.perspectiveTransform(np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2),mtx))]
	
	def getScore(self,gray):
		if(gray is None or gray.size == 0): return 0
		gray=cv2.cvtColor(gray,cv2.CV_8U)
		kp, desc=self.sift.detectAndCompute(gray,None)
		if(len(kp)<2): return 0
		best=0
		for e in self.desc:
			matches=self.flann.knnMatch(e,desc,k=2)
			cur=0
			for m,n in matches:
				if(m.distance<self.distance*n.distance):
					cur=cur+1
			if(cur>best): best=cur
		return best

	def test(self,feature_gray,gray):
		kp1, desc1=self.sift.detectAndCompute(feature_gray,None)
		kp2, desc2=self.sift.detectAndCompute(gray,None)
		matches=self.flann.knnMatch(desc1,desc2,k=2)
		pts=[]
		for m,n in matches:
			if(m.distance<self.distance*n.distance):
				pts.append(m)
		return cv2.drawMatches(feature_gray,kp1,gray,kp2,pts,None)
