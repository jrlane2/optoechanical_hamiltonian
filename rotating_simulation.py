import matplotlib as mpl
import numpy as np
import matplotlib.pylab as pl
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy.optimize import root
from scipy.integrate import solve_ivp
from datetime import datetime
import math

hbar = 1.054571817*1e-34
c = 299792458
Lambda = 1549.961*1e-9
Omega_L = 2*np.pi*c/Lambda




'''
Useful functions
'''


def continuity_sort(list1, list2):
	# Sorts 2 lists of numbers such that the the ith element of each list is closest
	# to the i-1th element of the corresponding list
	# Usefull when, say, calculating eigenvalues that cross branch cuts
	sortedlist1, sortedlist2 = np.zeros(len(list1))+ 1j, np.zeros(len(list2))+ 1j
	firstelements = np.array([list1[0], list2[0]]) 
	# sort first element by imaginary value
	sortedlist1[0], sortedlist2[0] = firstelements[np.argsort(np.imag(firstelements))]
	for i in range(1, len(list1)):
		prev = np.array([sortedlist1[i-1], sortedlist2[i-1]])
		current = np.array([list1[i], list2[i]])
		list1element = np.argmin(np.abs(current[0] - prev))
		sortedlist1[i] = current[list1element]
		sortedlist2[i] = current[(list1element + 1)%2]
	return sortedlist1, sortedlist2

def remove_duplicates(arr, threshold):
	unique_rows = [arr[0]]
	for i in range(1, len(arr)):
		row = arr[i]
		is_duplicate = False
		for u in unique_rows:
			if abs(u[0] + u[1] - row[0] - row[1]) <= threshold:
				is_duplicate = True
				break
		if not is_duplicate:
			unique_rows.append(row)
	return np.array(unique_rows)

def removeinf(arr):
	return np.where(arr == -np.inf, 0, arr)

def gain_mode_selector(M):
	e1, e2 = np.linalg.eigvals(M)
	if np.sign(np.real(e1)) == np.sign(np.imag(e1)):
		return 1 # in gain mode
	else:
		return -1 # in loss mode

def get_gain_leftright(M):
	[er1, er2], [vr1, vr2] = np.linalg.eig(M)
	[el1, el2], [vl1, vl2] = np.linalg.eig(M.T)
	index = np.argsort(np.imag(np.array([er1, er2])))[1] # positive imaginary eigenvalue
	egain, rgain, lgain = [er1, er2][index], [vr1, vr2][index], [vl1, vl2][index]
	norm = np.dot(lgain, rgain)
	return egain, rgain, lgain/norm



'''
Hamiltonian Functions
'''


def Hbare(omega1, omega2, gamma1, gamma2):
	return np.array([[omega1 - 1j*gamma1/2, 0], [0, omega2 - 1j*gamma2/2]])

def chi_c(omega, kappa_0):
	return 1 / (kappa_0 / 2 - 1j * omega)

def ncav(omega, kappa_0, kappa_ext, P, Omega_L):
	return abs(P) / (hbar * Omega_L) * kappa_ext / (omega ** 2 + (kappa_0 / 2) ** 2)

def alpha_n(omega, kappa_0, kappa_ext, P, Omega_L):
	return np.sqrt(abs(P) / (hbar * Omega_L)) * np.sqrt(kappa_ext) / (kappa_0 / 2 - 1j * omega)

def alpha_ncc(omega, kappa_0, kappa_ext, P, Omega_L):
	return np.sqrt(abs(P) / (hbar * Omega_L)) * np.sqrt(kappa_ext) / (kappa_0 / 2 + 1j * omega)

def sigma_jk(Pj, Delta_j, Pk, Delta_k, omega_m, kappa_0, kappa_ext, Omega_L):
	alpha_ncc_j = alpha_ncc(Delta_j, kappa_0, kappa_ext, Pj, Omega_L)
	alpha_n_k = alpha_n(Delta_k, kappa_0, kappa_ext, Pk, Omega_L)
	chi_mj = chi_c(omega_m + Delta_j, kappa_0)
	chi_mk = chi_c(omega_m - Delta_k, kappa_0)
	return alpha_ncc_j * alpha_n_k * (chi_mj - chi_mk)

def H_sigma(P1, Delta1, P2, Delta2, phi12, omega1, omega2, Gamma1, Gamma2, kappa, kappa_ext, g1, g2):
	sigma_111 = sigma_jk(P1, Delta1, P1, Delta1, omega1, kappa, kappa_ext, Omega_L)
	sigma_221 = sigma_jk(P2, Delta2, P2, Delta2, omega1, kappa, kappa_ext, Omega_L)
	sigma_112 = sigma_jk(P1, Delta1, P1, Delta1, omega2, kappa, kappa_ext, Omega_L)
	sigma_222 = sigma_jk(P2, Delta2, P2, Delta2, omega2, kappa, kappa_ext, Omega_L)
	sigma_12 = sigma_jk(P1, Delta1, P2, Delta2, omega1, kappa, kappa_ext, Omega_L)
	sigma_21 = sigma_jk(P2, Delta2, P1, Delta1, omega2, kappa, kappa_ext, Omega_L)
	return -1j * np.array([[g1**2 * (sigma_111 + sigma_221), g1 * g2 * sigma_12 * np.exp(1j * phi12)],
						   [g1 * g2 * sigma_21 * np.exp(-1j * phi12), g2**2 * (sigma_112 + sigma_222)]])

def H_photothermal(P1, Delta1, P2, Delta2, kappa, kappa_ext, A1, A2):
	ncav1 = ncav(Delta1, kappa, kappa_ext, P1, Omega_L)
	ncav2 = ncav(Delta2, kappa, kappa_ext, P2, Omega_L)
	return np.array([[-A1*(ncav1 + ncav2), 0], [0, -A2*(ncav1 + ncav2)]])




class Dynamical_Hamiltonian():
	'''
	Class for *dynamical* Optomechanical Hamiltonian in the rotating frame
	Initializing assumes 2 tones that are spaced to provide coupling between 2 modes
	Initial inputs: Hamiltonian parametes, 2 x Powers, 2 x frequencies, and the beatnote phase
	Powers, frequencies, and beatnote phases are assumed to be *functions* of 3 variables:
	t (time), Tend (the total time), and ccw (1 or -1, telling me whether to run time forwards or bakwards)
	'''
	def __init__(self, P1, P2, delta, eta, Phi12, Hparams):
		# Initialize an instance of the object
		self.omega1, self.omega2, self.gamma1, self.gamma2, self.Kappa, self.Kappaext, self.g1, self.g2, self.A1, self.A2 = Hparams
		self.P1 = lambda t, Tend, ccw: P1(t, Tend, ccw)
		self.P2 = lambda t, Tend, ccw: P2(t, Tend, ccw)
		self.delta = lambda t, Tend, ccw: delta(t, Tend, ccw)
		self.eta = lambda t, Tend, ccw: eta(t, Tend, ccw)
		self.phi12 = lambda t, Tend, ccw: Phi12(t, Tend, ccw)
		self.Delta1 = lambda t, Tend, ccw: -self.omega1 + delta(t, Tend, ccw)
		self.Delta2 = lambda t, Tend, ccw: -self.omega2 + delta(t, Tend, ccw) + eta(t, Tend, ccw)
		self.Hbare = Hbare(self.omega1, self.omega2, self.gamma1, self.gamma2)

		HSigma = lambda t, Tend, ccw: H_sigma(P1(t, Tend, ccw), self.Delta1(t, Tend, ccw), P2(t, Tend, ccw), self.Delta2(t, Tend, ccw),
					Phi12(t, Tend, ccw), self.omega1, self.omega2, self.gamma1, self.gamma2, self.Kappa, self.Kappaext, self.g1, self.g2)

		HPhotoThermal = lambda t, Tend, ccw: H_photothermal(P1(t, Tend, ccw), self.Delta1(t, Tend, ccw), P2(t, Tend, ccw), self.Delta2(t, Tend, ccw), 
					self.Kappa, self.Kappaext, self.A1, self.A2)

		PauliZ = np.array([[1, 0], [0, -1]])

		self.Htot = lambda t, Tend, ccw: (self.Hbare + HSigma(t, Tend, ccw) + HPhotoThermal(t, Tend, ccw) + 
					((self.Delta1(t, Tend, ccw)- self.Delta2(t, Tend, ccw)) / 2) * PauliZ)

		return

	def add_3rd_tone(self, P3, Delta3):
		# Add a third tone at some incomensurate frequency 
		# such that it effectively does single tone optomechanics
		self.P3 = lambda t, Tend, ccw: P3(t, Tend, ccw)
		self.Delta3 = lambda t, Tend, ccw: Delta3(t, Tend, ccw)

		sigma_331 = lambda t, Tend, ccw: sigma_jk(P3(t, Tend, ccw), Delta3(t, Tend, ccw), P3(t, Tend, ccw), Delta3(t, Tend, ccw),
					self.omega1, self.kappa, self.kappa_ext, Omega_L)

		sigma_332 = lambda t, Tend, ccw: sigma_jk(P3(t, Tend, ccw), Delta3(t, Tend, ccw), P3(t, Tend, ccw), Delta3(t, Tend, ccw),
					self.omega2, self.kappa, self.kappa_ext, Omega_L)

		self.Htot = lambda t, Tend, ccw: self.Htot(t, Tend, ccw) + np.array([[sigma_331(t, Tend, ccw), 0],[0, sigma_332(t, Tend, ccw)]])

		return


	def Htraceless(self, t, Tend, ccw):
		return self.Htot(t, Tend, ccw) - np.trace(self.Htot(t, Tend, ccw))/2 * np.identity(2)

	def HRetraceless(self, t, Tend, ccw):
		return self.Htot(t, Tend, ccw) - np.real(np.trace(self.Htot(t, Tend, ccw)))/2 * np.identity(2)

	def HRotating(self, t, Tend, ccw):
		return self.Htot(t, Tend, ccw) - (self.omega1 + self.omega2)/2*np.identity(2)

	def AJDiscriminant(self, t, Tend, ccw):
		temp = self.Htot(t, Tend, ccw)
		return np.sqrt(np.trace(temp)**2 - 4 * np.linalg.det(temp))

	def HamX(self, t, Tend, ccw):
		temp = self.Htraceless(t, Tend, ccw)
		return (temp[1][0] + temp[0][1]) / 2

	def HamY(self, t, Tend, ccw):
		temp = self.Htraceless(t, Tend, ccw)
		return -1j * (temp[1][0] - temp[0][1]) / 2

	def HamZ(self, t, Tend, ccw):
		temp = self.Htraceless(t, Tend, ccw)
		return temp[0][0]

	def AsimuthalPhi(self, t, Tend, ccw):
		X = self.HamX(t, Tend, ccw)
		Y = self.HamY(t, Tend, ccw)
		return np.arctan(X/Y)

	def polar_theta(self, t, Tend, ccw):
		X = self.HamX(t, Tend, ccw)
		Y = self.HamY(t, Tend, ccw)
		Z = self.HamZ(t, Tend, ccw)
		rho = np.sqrt(X**2 + Y**2)
		rho = rho*np.sign(np.imag(rho)) # temporary bandage for branch cut.
		return 2*np.arctan(rho/(np.sqrt(Z**2 + rho**2) + Z))

	def cos_theta(self, t, Tend, ccw):
		Z = self.HamZ(t, Tend, ccw)
		R = np.sort(np.linalg.eigvals(self.Htraceless(t, Tend, ccw)))[1]
		return Z/R

	def Berry_phase_discrete(self, t, Tend, ccw, steps = 5000):
		rights = np.zeros([steps, 2])+1j
		lefts = np.zeros([steps, 2])+1j
		inners = np.zeros(steps)+1j
		
		for i in range(steps):
			Ham = self.Htraceless(t, Tend, ccw)
			_, rights[i,:], lefts[i,:] = get_gain_leftright(Ham)
		for i in range(steps):
			if i == steps-1:
				inners[i] = np.dot(lefts[0,:], rights[i,:])
			else:
				inners[i] = np.dot(lefts[i+1,:], rights[i,:])

		return -1j*np.log(np.prod(inners))




class Static_Hamiltonian():
	'''
	Class for *static* Optomechanical Hamiltonian in the rotating frame
	Usefull for doing things like making plots over parameter space or finding EPs
	Initializing assumes 2 tones that are spaced to provide coupling between 2 modes
	Initial inputs: Hamiltonian parametes, 2 x Powers, 2 x frequencies, and the beatnote phase
	Powers, frequencies, and beatnote phases are all numbers.
	'''
	def __init__(self, P1, P2, delta, eta, Phi12, Hparams):

		self.omega1, self.omega2, self.gamma1, self.gamma2, self.Kappa, self.Kappaext, self.g1, self.g2, self.A1, self.A2 = Hparams
		self.P1 = P1
		self.P2 = P2
		self.delta = delta
		self.eta = eta
		self.phi12 = Phi12
		self.Delta1 =  -self.omega1 + delta
		self.Delta2 = -self.omega2 + delta + eta
		self.Hbare = Hbare(self.omega1, self.omega2, self.gamma1, self.gamma2)

		HSigma = H_sigma(P1, self.Delta1, P2, self.Delta2, Phi12, self.omega1, self.omega2, 
			self.gamma1, self.gamma2, self.Kappa, self.Kappaext, self.g1, self.g2)

		HPhotoThermal = H_photothermal(P1, self.Delta1, P2, self.Delta2, self.Kappa, self.Kappaext, self.A1, self.A2)

		PauliZ = np.array([[1, 0], [0, -1]])

		self.Htot = self.Hbare + HSigma + HPhotoThermal + ((self.Delta1- self.Delta2) / 2) * PauliZ

		return

	def add_3rd_tone(self, P3, Delta3):
		# Add a third tone at some incomensurate frequency 
		# such that it effectively does single tone optomechanics
		self.P3 = P3
		self.Delta3 = Delta3

		sigma_331 = sigma_jk(P3, Delta3, P3, Delta1, self.omega1, self.kappa, self.kappa_ext, Omega_L)

		sigma_332 = sigma_jk(P3, Delta3, P3, Delta1, self.omega2, self.kappa, self.kappa_ext, Omega_L)

		self.Htot = self.Htot + np.array([[sigma_331, 0],[0, sigma_332]])

		return


	def Htraceless(self):
		return self.Htot - np.trace(self.Htot/2) * np.identity(2)

	def HRetraceless(self):
		return self.Htot - np.real(np.trace(self.Htot))/2 * np.identity(2)

	def HRotating(self):
		return self.Htot - (self.omega1 + self.omega2)/2*np.identity(2)

	def AJDiscriminant(self):
		temp = self.Htot
		return np.sqrt(np.trace(temp)**2 - 4 * np.linalg.det(temp))

	def HamX(self):
		temp = self.Htraceless
		return (temp[1][0] + temp[0][1]) / 2

	def HamY(self):
		temp = self.Htraceless
		return -1j * (temp[1][0] - temp[0][1]) / 2

	def HamZ(self):
		temp = self.Htraceless
		return temp[0][0]

	def AsimuthalPhi(self):
		X = self.HamX
		Y = self.HamY
		return np.arctan(X/Y)

	def polar_theta(self):
		X = self.HamX
		Y = self.HamY
		Z = self.HamZ
		rho = np.sqrt(X**2 + Y**2)
		rho = rho*np.sign(np.imag(rho)) # temporary bandage for branch cut.
		return 2*np.arctan(rho/(np.sqrt(Z**2 + rho**2) + Z))

	def cos_theta(self):
		Z = self.HamZ
		R = np.sort(np.linal.eigvals(Htraceless))[1]
		return Z/R

	def classic_Berry_phase(self):
		# For classic phi loops only
		g = gain_mode_selector(self.Htot)
		costh = self.cos_theta
		return -np.pi*(1-costh)*g


class Manual_Hamiltonian():
	'''
	Class for just plugging in a dynamical matrix by hand
	This one takes in time dependence simply by rotating the off-diagonal terms by 2pi
	'''
	def __init__(self, Ham):

		def phidep(t, Tend, ccw):
				return np.array([[1, np.exp(-1j*2*np.pi*t*ccw/Tend)],[np.exp(1j*2*np.pi*t*ccw/Tend), 1]])

		self.Htot = lambda t, Tend, ccw: Ham*phidep(t, Tend, ccw)
		return

	def Htraceless(self, t, Tend, ccw):
		return self.Htot(t, Tend, ccw) - np.trace(self.Htot(t, Tend, ccw))/2 * np.identity(2)

	def HRetraceless(self, t, Tend, ccw):
		return self.Htot(t, Tend, ccw) - np.real(np.trace(self.Htot(t, Tend, ccw)))/2 * np.identity(2)

	def HRotating(self, t, Tend, ccw):
		return self.Htot(t, Tend, ccw) - (self.omega1 + self.omega2)/2*np.identity(2)

	def AJDiscriminant(self, t, Tend, ccw):
		temp = self.Htot(t, Tend, ccw)
		return np.sqrt(np.trace(temp)**2 - 4 * np.linalg.det(temp))

	def HamX(self, t, Tend, ccw):
		temp = self.Htraceless(t, Tend, ccw)
		return (temp[1][0] + temp[0][1]) / 2

	def HamY(self, t, Tend, ccw):
		temp = self.Htraceless(t, Tend, ccw)
		return -1j * (temp[1][0] - temp[0][1]) / 2

	def HamZ(self, t, Tend, ccw):
		temp = self.Htraceless(t, Tend, ccw)
		return temp[0][0]

	def AsimuthalPhi(self, t, Tend, ccw):
		X = self.HamX(t, Tend, ccw)
		Y = self.HamY(t, Tend, ccw)
		return np.arctan(X/Y)

	def polar_theta(self, t, Tend, ccw):
		X = self.HamX(t, Tend, ccw)
		Y = self.HamY(t, Tend, ccw)
		Z = self.HamZ(t, Tend, ccw)
		rho = np.sqrt(X**2 + Y**2)
		rho = rho*np.sign(np.imag(rho)) # temporary bandage for branch cut.
		return 2*np.arctan(rho/(np.sqrt(Z**2 + rho**2) + Z))

	def cos_theta(self, t, Tend, ccw):
		Z = self.HamZ(t, Tend, ccw)
		R = np.sort(np.linalg.eigvals(self.Htraceless(t, Tend, ccw)))[1]
		return Z/R













def Ham_at_EP_get_Rphi(EpHamiltonian):
    R = EpHamiltonian[0,0]
    HEPnorm = EpHamiltonian/R
    phi = -1j/2*np.log(EpHamiltonian[1,0]/EpHamiltonian[0,1])
    return R, phi











'''
EP2 finders
'''

class EP2_finder():
	def __init__(self, Hparams):
		self.Hparams = Hparams
		return



	def EP2finderPgeoDelta(self, eta):
		Parr = np.linspace(0, 50, 8)
		Darr = np.linspace(-2*2*np.pi,2*2*np.pi,8)
		P, D = np.meshgrid(Parr, Darr)
		Pguess, Dguess = np.ravel(P), np.ravel(D)
		Pfound, Dfound, resfound = np.zeros(len(Pguess)), np.zeros(len(Pguess)), np.zeros(len(Pguess))

		def minfun(z):
			P, delta = z
			Ham = Static_Hamiltonian(np.abs(P)*1e-6, np.abs(P)*1e-6, delta*2*np.pi*1e6, 
								 2*np.pi*eta, 0, self.Hparams)
			AJX = Ham.AJDiscriminant()
			return np.log(np.real(AJX)**2 + np.imag(AJX)**2)

		for i in range(len(Pguess)):
			res = op.minimize(minfun, [Pguess[i], Dguess[i]], method = "Powell")
			Pfound[i] = np.abs(res['x'][0])
			Dfound[i] = res['x'][1]
			resfound[i] = minfun(res['x'])
		resfound = removeinf(resfound)
		sort = np.argsort(resfound)
		ressort = resfound[sort]
		args = sort[np.where(ressort < np.min(ressort)+4)]
		return remove_duplicates(np.array([Pfound[args], Dfound[args]]).T, 1)



	def EP2finderP2Delta(self, P1, eta):
		Parr = np.linspace(0, 50, 8)
		Darr = np.linspace(-2*2*np.pi,2*2*np.pi,8)
		P, D = np.meshgrid(Parr, Darr)
		Pguess, Dguess = np.ravel(P), np.ravel(D)
		Pfound, Dfound, resfound = np.zeros(len(Pguess)), np.zeros(len(Pguess)), np.zeros(len(Pguess))

		def minfun(z):
			P, delta = z
			Ham = Static_Hamiltonian(np.abs(P1)*1e-6, np.abs(P)*1e-6, delta*2*np.pi*1e6, 
								 2*np.pi*eta, 0, self.Hparams)
			AJX = Ham.AJDiscriminant()
			return np.log(np.real(AJX)**2 + np.imag(AJX)**2)

		for i in range(len(Pguess)):
			res = op.minimize(minfun, [Pguess[i], Dguess[i]], method = "Powell")
			Pfound[i] = np.abs(res['x'][0])
			Dfound[i] = res['x'][1]
			resfound[i] = minfun(res['x'])
		resfound = removeinf(resfound)
		sort = np.argsort(resfound)
		ressort = resfound[sort]
		args = sort[np.where(ressort < np.min(ressort)+4)]
		return remove_duplicates(np.array([Pfound[args], Dfound[args]]).T, 1)



'''
Simulation class
'''

def lowpass(w, BW):
	# assume 1st order filter function
	# both frequency and bandwidth in rad/s
	return 1/(1+1j*w/BW)

def fix_S(S):
	# S is a matrix whos columns are eigenvectors
	# I want to fix S such that (a) it's colulmns are normalized
	# and (b) the first component of each eigenvector is real
	S1, S2 = S[:,0], S[:,1]
	S1 = S1/np.linalg.norm(S1)/(S1[0]/np.abs(S1[0]))
	S2 = S2/np.linalg.norm(S2)/(S2[0]/np.abs(S2[0]))
	return np.array([S1, S2]).T





class dynamics_sim_EP2():
	def __init__(self, Hparams, manual_Ham = None):
		if type(manual_Ham) != type(None):

			self.Ham = Manual_Hamiltonian(manual_Ham)
			self.H0 = self.Ham.HRetraceless(0,1,1)

			self.get_H0_stuff()
			return
		self.Hparams = Hparams
		return
	
	def load_path(self, P1, P2, delta, eta, phi12):

		self.P1 = lambda t, Tend, ccw:  1e-6*P1(t, Tend, ccw)
		self.P2 = lambda t, Tend, ccw:  1e-6*P2(t, Tend, ccw)
		self.delta = lambda t, Tend, ccw:  1e6*2*np.pi*delta(t, Tend, ccw)
		self.eta = lambda t, Tend, ccw: 2*np.pi*eta(t, Tend, ccw)
		self.eta0 = self.eta(0,1,1)
		self.phi12 = phi12 

		self.Ham = Dynamical_Hamiltonian(self.P1, self.P2, self.delta, self.eta, self.phi12, self.Hparams)
		
		self.H0 = self.Ham.HRetraceless(0,1,1)
		
		self.get_H0_stuff()

	
	def get_H0_stuff(self):
		# gets and stores a bunch of useful parameters at
		# the beginning of the cicuit
		
		evals, S = np.linalg.eig(self.H0)
		sort_indicesR = np.argsort(np.real(evals)) # sort by *real* component
		self.e1 = evals[sort_indicesR[0]]
		self.e2 = evals[sort_indicesR[1]]
		S = fix_S(S) # unfuck S
		self.S = S #C.O.B. operator to diagonalize Hamiltonian
		self.Sinv = np.linalg.inv(S)
		self.psi1 = S[:,0]
		self.psi2 = S[:,1]
		
		# Arnold Jordan stuff
		# tr = np.trace(self.H0) # use traceless eigenvalues
		# H0trless = self.H0-tr
		# R, _ Ham_at_EP_get_Rphi(H0trless):
		self.cos = self.Ham.cos_theta(0,1,1)

		self.T = np.array([[1, self.cos],[1, -self.cos]])
		self.Tinv = np.linalg.inv(self.T)

		self.Gamma = np.array([[1,1],[1j,-1j]])
		self.Gammainv = np.linalg.inv(self.Gamma)

		# self.Tinv = 0.5*np.array([[1,1],[self.e2 - tr/2, self.e1 - tr/2]])
		# self.T = np.linalg.inv(self.Tinv)
		
		# determine if gain mode is mode 1 or mode 2
		sort_indicesI = np.argsort(np.imag(evals))
		if sort_indicesI[0] == sort_indicesR[0]: 
			self.gainmode = 1
		else:
			self.gainmode = 0
		return
	
	def get_EP(self, eta = None):
		self.EP2_finder = EP2_finder(self.Hparams)
		if type(eta) == type(None):
			self.EP2_finder.EP2_list = self.EP2_finder.EP2finderPgeoDelta(self.eta0/(2*np.pi))
		else:
			self.EP2_finder.EP2_list = self.EP2_finder.EP2finderPgeoDelta(eta)
		self.EP = self.EP2_finder.EP2_list[0]
		return


	def set_initial_state(self, c0):
		# institute gauge choice here
		cnorm = c0/np.linalg.norm(c0)
		self.c0 = cnorm*np.exp(-1j*np.angle(c0[0]))
		return
	
	def run_ringdown(self, traceless = False, Tend = None):
		# simple time independent simulation
		if traceless == False:
			def H(t):
				return self.Ham.HRetraceless(0,1,1)
		else:
			def H(t):
				return self.Ham.Htraceless(0,1,1)
			
		def f(t, y):
			return -1j * H(t) @ y

		if self.gainmode == 0:
			min_gamma = -np.imag(self.e1)/np.pi
		else:
			min_gamma = -np.imag(self.e2)/np.pi
		if type(Tend) == type(None):
			Tend = 3/min_gamma
		self.sol1 = solve_ivp(f, [0, Tend], self.c0, method='RK45', max_step=Tend/1000)
		self.realtime = np.linspace(0, Tend, len(self.sol1.y[0,:]))
		return

	def run_loop(self, Tend, ccw, traceless = False):
		# simple time independent simulation
		if traceless == False:
			def H(t):
				return self.Ham.HRetraceless(t, Tend, ccw)
		else:
			def H(t):
				return self.Ham.Htraceless(t, Tend, ccw)
			
		def f(t, y):
			return -1j * H(t) @ y

		self.sol1 = solve_ivp(f, [0, Tend], self.c0, method='RK45', max_step=Tend/1000)
		self.realtime = np.linspace(0, Tend, len(self.sol1.y[0,:]))
		return
	
	def project_solution(self, projection = 'Diagonal'):
		if projection == 'Diagonal':
			sol1proj = self.Sinv @ self.sol1.y
			self.c1 = sol1proj[0,:]
			self.c2 = sol1proj[1,:]
		elif projection == "Rotating":
			sol1proj = self.sol1.y
			self.c1 = sol1proj[0,:]
			self.c2 = sol1proj[1,:]
		elif projection == "AJ":
			sol1proj = self.Tinv @ self.Sinv @ self.sol1.y
			self.c1 = sol1proj[0,:]
			self.c2 = sol1proj[1,:]
		elif projection == "Jordan":
			sol1proj = self.Gammainv @ self.sol1.y
			self.c1 = sol1proj[0,:]
			self.c2 = sol1proj[1,:]
		else:
			print("Choose a valid basis dummy")
		return

	def propagator(self, Tend, ccw, projection = 'Diagonal', traceless = False):
		# Constructs propagator, written in chosen basis
		# Assumes the loop is closed.

		# Do the first simulation
		if projection == "Diagonal":
			self.c0 = self.S @ np.array([1,0])
		elif projection == "Rotating":
			self.c0 =  np.array([1,0]) + 0j
		elif projection == "AJ":
			self.c0 = self.S @ self.T @ np.array([1,0])
		elif projection == "Jordan":
			self.c0 = self.Gamma @ np.array([1,0])
		else:
			print("Choose a valid basis dummy")
			return

		self.run_loop(Tend, ccw, traceless)
		self.project_solution(projection)
		a = self.c1[-1]
		c = self.c2[-1]

		# Do the second simulation
		if projection == 'Diagonal':
			self.c0 = self.S @ np.array([0,1])
		elif projection == "Rotating":
			self.c0 =  np.array([0,1]) + 0j
		elif projection == "AJ":
			self.c0 = self.S @ self.T @ np.array([0,1])
		elif projection == "Jordan":
			self.c0 = self.Gamma @ np.array([0,1])

		self.run_loop(Tend, ccw, traceless)
		self.project_solution(projection)
		b = self.c1[-1]
		d = self.c2[-1]

		return np.array([[a, b], [c, d]])

	def calc_prop_list(self, Tlist, Projection, traceless = False):

		self.tstamps = Tlist
		self.steps = len(Tlist)

		a, b, c, d = np.zeros(self.steps)+1j, np.zeros(self.steps)+1j, np.zeros(self.steps)+1j, np.zeros(self.steps)+1j
		for i, t in enumerate(self.tstamps):
			proptemp = self.propagator(t, 1, Projection, traceless)
			a[i], b[i], c[i], d[i] = proptemp[0,0], proptemp[0,1], proptemp[1,0], proptemp[1,1]
		self.current_prop_f = np.vstack([a, b, c, d])
		
		a, b, c, d = np.zeros(self.steps)+1j, np.zeros(self.steps)+1j, np.zeros(self.steps)+1j, np.zeros(self.steps)+1j
		for i, t in enumerate(self.tstamps):
			proptemp = self.propagator(t, -1, Projection, traceless)
			a[i], b[i], c[i], d[i] = proptemp[0,0], proptemp[0,1], proptemp[1,0], proptemp[1,1]
		self.current_prop_b = np.vstack([a, b, c, d])

		return

	def create_dir(self, dirname):
		isExist = os.path.exists(dirname)
		if not isExist:
		   # Create a new directory because it does not exist
		   os.makedirs(dirname)
		self.current_path = dirname
		return

	def calc_and_save_prop(self, Tlist, Projection, filename_f, filename_b, traceless = False, plot = False):

		self.calc_prop_list(Tlist, Projection, traceless)
		filename_f = self.current_path + '/' + filename_f
		filename_b = self.current_path + '/' + filename_b

		savef = np.vstack([self.tstamps, self.current_prop_f]).T
		np.savetxt(filename_f, savef)
		saveb = np.vstack([self.tstamps, self.current_prop_b]).T
		np.savetxt(filename_b, saveb)
		if plot == True:
			self.plot_diags()
		return


	def plot_diags(self):
		a_f = self.current_prop_f[0]
		d_f = self.current_prop_f[3]
		a_b = self.current_prop_b[0]
		d_b = self.current_prop_b[3]
		EPoffsetP = self.P1(0,1,1)*1e6 - self.EP[0]
		EPoffsetD = self.delta(0,1,1)/(2*np.pi*1e6) - self.EP[1]

		fig, [ax1, ax2] = plt.subplots(nrows = 1, ncols = 2, figsize = (13,6))
		ax1.plot(self.tstamps*1e3, np.log(np.abs(a_f)), label = '$U_{11}$ forwards')
		ax1.plot(self.tstamps*1e3, np.log(np.abs(d_f)), label = '$U_{22}$ forwards')
		ax1.plot(self.tstamps*1e3, np.log(np.abs(a_b)), label = '$U_{11}$ backwards', linestyle = '--')
		ax1.plot(self.tstamps*1e3, np.log(np.abs(d_b)), label = '$U_{22}$ backwards', linestyle = '--')
		ax1.legend()
		ax1.set_title("$\delta$ = {:.3f} MHz from EP".format(EPoffsetD))
		ax1.set_xlabel('loop time (ms)')
		ax1.set_ylabel('log[Abs[prop coeff]]')

		ax2.plot(self.tstamps*1e3, np.unwrap(np.angle(a_f)), label = '$U_{11}$ forwards')
		ax2.plot(self.tstamps*1e3, np.unwrap(np.angle(d_f)), label = '$U_{22}$ forwards')
		ax2.plot(self.tstamps*1e3, np.unwrap(np.angle(a_b)), label = '$U_{11}$ backwards', linestyle = '--')
		ax2.plot(self.tstamps*1e3, np.unwrap(np.angle(d_b)), label = '$U_{22}$ backwards', linestyle = '--')
		ax2.set_xlabel('loop time (ms)')
		ax2.set_ylabel('Phase[prop coeff]')
		ax2.set_title("$P_1$ = {:.3f} $\mu$W from EP".format(EPoffsetP))
		ax2.legend()
		filename = self.current_path + "/diag_prop_plot_poff_{:.3f}_doff_{:.3f}.png".format(EPoffsetP, EPoffsetD)
		plt.savefig(filename, bbox_inches = 'tight')
		plt.close()
		return






	def lockin_signal(self, BW, fdemod1, fdemod2):
		# signal in the diagonal basis
		# only going to work for solutions to the real traceless hamiltonian
		w1 = fdemod1*2*np.pi
		w2 = fdemod2*2*np.pi
		wm1 = self.Hparams[0]
		wm2 = self.Hparams[1]
		g01 = self.Hparams[6]
		g02 = self.Hparams[7]
		chi1 = chi_c(w1, self.Hparams[4])
		chi2 = chi_c(w2, self.Hparams[4])
		
		# calculate state vector diagonal basis
		sol1proj = self.Sinv @ self.sol1.y
		c1 = sol1proj[0,:]
		c2 = sol1proj[1,:]
		# these guys are already ringing at e_1 and e_2 respectively
		# but the all get filtered differently so this is a pain in the ass
		[[u11, u12], [u21, u22]] = self.S
		W11 = lowpass(np.real(self.e1) + wm1 + self.eta0/2 - w1, BW*2*np.pi)
		W12 = lowpass(np.real(self.e2) + wm1 + self.eta0/2 - w1, BW*2*np.pi)
		W21 = lowpass(np.real(self.e1) + wm2 - self.eta0/2 - w2, BW*2*np.pi)
		W22 = lowpass(np.real(self.e2) + wm2 - self.eta0/2 - w2, BW*2*np.pi)
		v1lab = chi1*g01*(
			W11*u11*c1*np.exp(-1j*(wm1 - w1 + self.eta0/2)*self.realtime) +
			W12*u12*c2*np.exp(-1j*(wm1 - w1 + self.eta0/2)*self.realtime)
					)
		v2lab = chi2*g02*(
			W21*u21*c1*np.exp(-1j*(wm2 - w2 + self.eta0/2)*self.realtime) +
			W22*u22*c2*np.exp(-1j*(wm2 - w2 + self.eta0/2)*self.realtime)
					)		
		self.v1lab = v1lab*1e6
		self.v2lab = v2lab*1e6 # arbitrary scaling
		return

