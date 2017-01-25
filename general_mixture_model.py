# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats.distributions import norm, expon
import matplotlib.pylab as plt
from scipy.stats import genpareto

#import sys
#from os.path import join, dirname

#sys.path.append(join(dirname(__file__), "..", "..", "resources", "pyfak"))
#sys.path.append(join(dirname(__file__), "..", "..", "resources", "fractional_octave_filterbank"))

#from pyfak.dsp.common import wavread, spectrogram, get_regions, spectrogram2sound, wavwrite
#from fractional_octave_filterbank import fractional_octave_filterbank
    
def genpareto_ll(k, sigma, x):
    # translated from the MATLAB function gpfit.m
    n = 1#len(x);
    z = x/sigma
    lnsigma = np.log(sigma)

    if abs(k) > np.finfo('float').resolution:
        if k > 0 or max(z) < -1/k:
            sumlnu = np.log1p(k*z) # sum(log(1+k.*z)
            nll = n*lnsigma + (1+1/k) * sumlnu
        else:
            # The support of the GP when k<0 is 0 < x < abs(sigma/k).
            nll = np.inf
    else: # limiting exponential dist'n as k->0
        sumz = sum(z)
        nll = n*lnsigma + sumz
        
    return -nll

def get_log_likelihood(x, type_dist, **kwargs):
    if type_dist == 'normal':
        LL = -1/2 * np.log(2*np.pi) - 1/2 * np.log(kwargs['sigma_2']) - 1/(2*kwargs['sigma_2']) * (x - kwargs['mu'])**2 # only for one sample
    elif type_dist == 'exp':
        # LL = -1 * np.log(2*kwargs['sigma_2']) - 1/(2*kwargs['sigma_2']) * x
        LL = -1 * np.log(2*kwargs['sigma_2']) - 1 / (2*kwargs['sigma_2']) * x # TODO: this change is related to the mu <-> (2*)sigma_2 confusion
    elif type_dist == "genpareto":
#        LL = np.array([genpareto_ll(kwargs['k'], kwargs['sigma'], _) for _ in x])
        LL =genpareto_ll(kwargs['k'], kwargs['sigma'], x)
        
    return LL

def get_likelihood(x, type_dist, **kwargs):
    LL = get_log_likelihood(x, type_dist, **kwargs)
    
    L = np.exp(LL)
    
    return L
    
def ml_parameter_estimation(x, weights, type_dist):
    sum_weights = np.sum(weights)
    if type_dist == 'normal':
        mu = np.sum(weights * x) / sum_weights
        sigma_2 = np.sum(weights * (x - mu)**2) / sum_weights
        
        params = {'mu': mu, 'sigma_2': sigma_2}
    elif type_dist == 'exp':
        sigma_2 = 1 / 2  * np.sum(weights * x) / sum_weights # TODO: is this correct?
        params = {'sigma_2': sigma_2, 'mu': 2 * sigma_2} # TODO: Also: The nomenclature is confusing. Sigma_2 is actually the mean of the sample values. I use it because it represents the power in each frequency bin... (-> probably not a good idea)
    elif type_dist == 'genpareto':
        fit_result = genpareto.fit(x)
        k, sigma = [fit_result[_] for _ in [0,2]]
        params = {'k': k, 'sigma': sigma}
        
    return params
    
class EM:
    def __init__(self, data):
        self.components = []
#        self.components.append({'type': 'normal', 'params': {'mu': 100, 'sigma_2': 1}})
        
#        self.components.append({'type': 'normal', 'params': {'mu': 1, 'sigma_2': 1}})
#        self.components.append({'type': 'normal', 'params': {'mu': 2, 'sigma_2': 1}})
#        self.components.append({'type': 'exp', 'params': {'sigma_2': 0.01}})
#        self.components.append({'type': 'normal', 'params': {'mu': 0, 'sigma_2': 0.01}})
#        self.components.append({'type': 'exp', 'params': {'sigma_2': 0.02}})
#        self.components.append({'type': 'normal', 'params': {'mu': 10, 'sigma_2': 1}})
        
        self.data = data
        self.n = len(data)
        
        self.delta_evidence = 1e-9
        
        
        
    def add_component(self, component):
        self.components.append(component)
        self.component_probabilities = 1/len(self.components) * np.ones((self.n, len(self.components)))
        
    def E_step(self):
        # determine component probabilities
        P_comp = np.array([get_likelihood(self.data, component['type'], **component['params']) for component in self.components]).T
        
        for idx, P_comp_i in enumerate(P_comp.T):
            self.component_probabilities[:,idx] = P_comp_i / (np.sum(P_comp, axis=1)) # TODO: +1e-9 ?
        
    def M_step(self):
        # update the distribution parameters
        # (weighted ml estimates)
        for idx, component in enumerate(self.components):
            component['params'] = ml_parameter_estimation(self.data, self.component_probabilities[:,idx], component['type'])
            
    def model_evidence(self, idx_component=None):
        if idx_component is None:
            idx_component = range(len(self.components))

        ME = 0
        for idx, component in enumerate([self.components[_] for _ in idx_component]):
            ME += np.sum(self.component_probabilities[:, idx] * get_log_likelihood(self.data, component['type'], **component['params'])) # TODO: should this be mean() to make it independent of the signal length?
            
        return ME

    def get_pdf(self, x, idx_component=None):
        if idx_component is None:
            idx_component = range(len(self.components))

        f = np.zeros(len(x))
        for idx, component in enumerate([self.components[_] for _ in idx_component]):
            f += get_likelihood(x, component['type'], **component['params']) # TODO: weigh with component probability?

        return f

    def get_cdf(self, x, idx_component=None):
        if idx_component is None:
            idx_component = range(len(self.components)) # TODO: not supported yet
            # raise ValueError("Selecting more than one components is not supported yet.")

        # component = self.components[idx_component]

        F = np.zeros(1)
        for idx, component in enumerate([self.components[_] for _ in idx_component]):
            if component['type'] == 'exp':
                F += np.mean(self.component_probabilities[:, idx]) * expon.cdf(x, loc=0, scale=2*component['params']['sigma_2'])
            if component['type'] == 'norm':
                F += np.mean(self.component_probabilities[:, idx]) * norm.cdf(x, loc=component['params']['mu'], scale=component['params']['sigma_2'])
            if component['type'] == 'genpareto':
                F += np.mean(self.component_probabilities[:, idx]) * genpareto.cdf(x, component['params']['k'], loc=0, scale=component['params']['sigma'])


        return F[0] # TODO: find a better way to return a list

            
    def estimate(self, N_iter=10000):
        evidence_last = -np.inf
        for _ in range(N_iter):
            self.E_step()
            self.M_step()
            print("iteration {} -- evidence: {}".format(_, self.model_evidence()))
            evidence = self.model_evidence()
            if np.isnan(evidence):
                raise RuntimeError("EM likelihood maximization failed.")
            if np.abs(evidence - evidence_last) < self.delta_evidence: # why is this abs required? shouldn't the evidence increase monotonically?
                break
            evidence_last = evidence

if __name__ == '__main__':
    
    if False:
        N = 100000
        mu = 100
        sigma_2 = 2
        x = np.random.normal(loc=mu, scale=np.sqrt(sigma_2), size=N)
    #    x[1000:2000] = np.random.normal(loc=2*mu, scale=np.sqrt(sigma_2), size=1000)
        x[5000:16500] =  np.random.exponential(scale=10 * sigma_2, size=11500)
        
    else:
        filename_input = "/media/matthias/testsignale/project_related/noise_region_detection/disturbed/000005.wav"
        x, fs, N_bits = wavread(filename_input)
        
#        x = x[:10*fs,:]
#        x = x[:,0]
        
        L_block = 2048
        L_feed = L_block // 2
        
        idx_bin = 10
        
        print("frequency: {} Hz".format(idx_bin/L_block * fs))
        
        spec_out = spectrogram(x[:,0], L_block, L_feed, L_block, fs, window_type='sqrthann')
        
        # determine true powers
        spec_out_noise = spectrogram(x[:,2], L_block, L_feed, L_block, fs, window_type='sqrthann')
        sigma_2_true = 1/2 * np.mean(spec_out_noise.P[idx_bin, :])
#        x = x[:,0]
        
        X = spec_out.P
#        x = spec_out.P[idx_bin, :]
        
    
    plt.close('all')
    if False:
        plt.figure(1)
        plt.plot(spec_out.vec_t, 10*np.log10(x))
    
#    L = get_likelihood(x, type_dist='normal', mu=10, sigma_2 = 1)
    
#    print(L)
    if False:
        # a single dft band
        em = EM(x)
    
        em.estimate()
        
        print(em.components)
    #    print(em.component_probabilities)
        
        
        plt.figure(2)
        for idx, P_comp in enumerate(em.component_probabilities.T):
            color = ['r', 'b', 'g', 'y'][idx]
            idx_this_comp = np.where(P_comp > 0.5)[0]
            for region in get_regions(idx_this_comp):
                indices = np.arange(region[0], region[1])
                plt.plot(spec_out.vec_t[indices], 10*np.log10(x[indices]),color)
    #        print(P_comp)
    #        break
    #    plt.plot(x)
        plt.show()
        
        plt.figure(3)
        _, bins, _ = plt.hist(x, bins=500, normed=True)
        
        # plot the estimation result on top
        for idx, component in enumerate(em.components):
            weight = np.sum(em.component_probabilities[:,idx])/np.sum(em.component_probabilities)
            if component['type'] == 'normal':
                f_theo = weight * norm.pdf(bins, component['params']['mu'], np.sqrt(component['params']['sigma_2']))
            elif component['type'] == 'exp':
                f_theo = weight * expon.pdf(bins, loc=0, scale=2 * component['params']['sigma_2'])
            plt.plot(bins, f_theo)
            
        print(em.model_evidence())
    elif False:
        # all dft bands
        vec_params = [None] * X.shape[0]
        
        vec_evidence = [None] * X.shape[0]

        # create masks for noise model
        B_noise = np.zeros(X.shape)
        for k in range(X.shape[0]):
            em = EM(X[k,:])
            em.estimate()
            
            vec_params[k] = em.components
            B_noise[k, :] = em.component_probabilities[:,0]
            
            B_noise[k, np.isnan(B_noise[k, :])] = 0
            
            vec_evidence[k] = em.model_evidence([0])
            
            print(k)
            
        # collect estimates
        vec_sigma_2_hat = np.array([params[0]['params']['sigma_2'] for params in vec_params])
        
        vec_sigma_2_hat[np.isnan(vec_sigma_2_hat)] = 0
        
            
        # determine true values
        vec_sigma_2_true = 1/2 * np.mean(spec_out_noise.P, axis=1)
        
        plt.figure()
        plt.plot(spec_out.vec_f, 10*np.log10(np.hstack((vec_sigma_2_true[:,np.newaxis], vec_sigma_2_hat[:, np.newaxis]))))
        
        plt.figure()
        plt.imshow(10*np.log10(spec_out.P), extent=(spec_out.vec_t[0], spec_out.vec_t[-1], spec_out.vec_f[0], spec_out.vec_f[-1]), aspect='auto')
        
        plt.figure()
        plt.imshow(B_noise, extent=(spec_out.vec_t[0], spec_out.vec_t[-1], spec_out.vec_f[0], spec_out.vec_f[-1]), aspect='auto')
        plt.colorbar()
        
        plt.figure()
        plt.plot(spec_out.vec_f, vec_evidence)
        plt.xlabel('f')
        plt.title('model evidence')
        
        plt.show()
        
        # create a time domain signal from the noise estimate
        n_hat = spectrogram2sound(spec_out.X, spec_out.P * (B_noise), fs)
        
        wavwrite(n_hat, fs, 16, 'out.wav')
        wavwrite(x[:,0], fs, 16, 'in.wav')
        wavwrite(x[:n_hat.shape[0],0]-n_hat.flatten(), fs, 16, 'diff.wav')
        
    else:
        # octave bands
        L_block = 512
        L_feed = L_block // 2
        L_DFT = L_block
        overlap_factor = 0
        Pxx_oct = fractional_octave_filterbank(x[:,0], fs, L_DFT, L_block, L_feed, overlap_factor)
        Pxx_oct_noise = fractional_octave_filterbank(x[:,2], fs, L_DFT, L_block, L_feed, overlap_factor)
        
        X = Pxx_oct[0]
        vec_f = Pxx_oct[1]
        vec_t = Pxx_oct[2]
        
        # all dft bands
        vec_params = [None] * X.shape[0]
        
        vec_evidence = [None] * X.shape[0]
        
        vec_sigma_2_hat = np.zeros(Pxx_oct[0].shape[0])

        # create masks for noise model
        B_noise = np.zeros(X.shape)
        for k in range(X.shape[0]):
            em = EM(X[k,:])
            em.estimate()
            
            vec_params[k] = em.components
            
            # sort components with increasing mu
            idx_sort = [_[0] for _ in sorted(zip(range(len(em.components)), [comp['params']['mu'] for comp in em.components]), key=lambda x: x[1])]
            idx_min_comp = idx_sort[0]
            B_noise[k, :] = em.component_probabilities[:,idx_min_comp]
            
            B_noise[k, np.isnan(B_noise[k, :])] = 0
            
            vec_evidence[k] = em.model_evidence([idx_min_comp]) /X.shape[1]
            
            print(k)
            
            # collect estimates
            vec_sigma_2_hat[k] = vec_params[k][idx_min_comp]['params']['mu']
        
        vec_sigma_2_hat[np.isnan(vec_sigma_2_hat)] = 0
        
        # determine true values
        vec_sigma_2_true = np.mean(Pxx_oct_noise[0], axis=1)
        
        plt.figure()
        plt.plot(vec_f, 10*np.log10(np.hstack((vec_sigma_2_true[:,np.newaxis], vec_sigma_2_hat[:, np.newaxis]))))
        
        plt.figure()
        plt.imshow(10*np.log10(Pxx_oct[0]), extent=(vec_t[0], vec_t[-1], vec_f[0], vec_f[-1]), aspect='auto', interpolation='none')
        
        plt.figure()
        plt.imshow(B_noise, extent=(vec_t[0], vec_t[-1], vec_f[0], vec_f[-1]), aspect='auto', interpolation='none')
        plt.colorbar()
    
        plt.figure()
        plt.plot(vec_f, vec_evidence)
        plt.xlabel('f')
        plt.title('model evidence')
        
        plt.figure()
        plt.plot(vec_t, 10*np.log10(Pxx_oct[0][-1,:]))
        
        plt.figure()
        _, bins, _ = plt.hist(Pxx_oct[0][-1], 500, normed=True)
        
        # plot the estimation result on top
        for idx, component in enumerate(em.components):
            weight = np.sum(em.component_probabilities[:,idx])/np.sum(em.component_probabilities)
            if component['type'] == 'normal':
                f_theo = weight * norm.pdf(bins, component['params']['mu'], np.sqrt(component['params']['sigma_2']))
            elif component['type'] == 'exp':
                f_theo = weight * expon.pdf(bins, loc=0, scale=2 * component['params']['sigma_2'])
            plt.plot(bins, f_theo)
        
        plt.figure()
        for idx, P_comp in enumerate(em.component_probabilities.T):
            color = ['r', 'b', 'g', 'y'][idx]
            idx_this_comp = np.where(P_comp > 0.5)[0]
            for region in get_regions(idx_this_comp):
                indices = np.arange(region[0], region[1])
                plt.plot(vec_t[indices], 10*np.log10(Pxx_oct[0][-1][indices]),color)
            
        print(em.model_evidence())
        
        plt.show()