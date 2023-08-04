import torch
import torch.nn as nn
import numpy as np

from model.embedder import get_embedder

TINY_NUMBER = 1e-6

class SphericalHarmonicMix:

    # initialize parameters for a spherical harmonic of degree l and order m
    def __init__(self, lmax):
        self.l = torch.repeat_interleave(torch.arange(lmax+1), 2*torch.arange(lmax+1) + 1)
        self.m = torch.cat([torch.arange(-l, l+1) for l in range(lmax+1)])
        self.weights = torch.sqrt( ((2*self.l + 1) * self.factorial(self.l - torch.abs(self.m))) / (4 * torch.pi * self.factorial(self.l + torch.abs(self.m))) )
        # self.weights = torch.zeros_like(self.l, dtype=torch.float)

        # mask = self.m == 0
        # self.weights[mask] = torch.sqrt( (2*self.l[mask] + 1) / (4*torch.pi) )
        # self.weights[~mask] = torch.sqrt( ((2*self.l[~mask] + 1) / (2*torch.pi)) * (self.factorial(self.l[~mask] - torch.abs(self.m[~mask])) / self.factorial(self.l[~mask] + torch.abs(self.m[~mask]))) )

    def __len__(self):
        return len(self.weights)

    def factorial(self, n):
        mask = n == 0
    
        n[mask] =  1
        if any(~mask):
            n[~mask] = n[~mask] * self.factorial(n[~mask]-1)
        return n
    
    def legendre(self, x, n):
        if n == 0:
            P = torch.ones_like(x)
        
        elif n == 1:
            P = x
        
        else:
            P = (1/n) * ((2*n - 1) * x * self.legendre(x, n-1) - (n - 1) * self.legendre(x, n-2))

        return P

    # generate a Legendre polynomial of degree l and order m at point x, using recursion formulas 
    def assoc_legendre(self, x, l, m):
        # if l < 0 or torch.abs(m) > l:
        #     return torch.zeros_like(x)
        # if m < 0:
        #     return - self.assoc_legendre(x, l, -m) * self.factorial(l+m) / self.factorial(l-m)
        # if m == 0 and l == 0:
        #     return torch.ones_like(x)
        # if m == 0 and l == 1:
        #     return x
        # if m == 1 and l == 1:
        #     return - torch.sqrt(1 - x*x)
        # if m == 1 or m == 0:
        #     a = (2*(l-1) + 1) * x * self.assoc_legendre(x, l-1, m)
        #     b = - (l+m-1) * self.assoc_legendre(x, l-2, m)
        #     return (a + b) / (l - m)
        # x = x.clamp(min=-1+TINY_NUMBER,max=1-TINY_NUMBER)
        # a = - 2 * (m-1) * x * self.assoc_legendre(x, l, m-1) / torch.sqrt(1 - x*x)
        # b = - (l+m-1) * (l-m+2) * self.assoc_legendre(x, l, m-2)
        # return (a + b)#.clamp(min=0)
        try:
            if m == 0:
                P = self.legendre(x, l)
        
            elif m > 0:
                P = (1 / torch.sqrt((1 - x**2).clamp(min=TINY_NUMBER))) * ((l - m + 1) * x * self.assoc_legendre(x, l, m-1) - (l + m - 1) * self.assoc_legendre(x, l-1, m-1)) 
        
            elif m < 0:
                m = torch.abs(m)
                P = ((-1) ** m) * (self.factorial(l - m) / self.factorial(l + m)) * self.assoc_legendre(x, l, m)
        
        except ZeroDivisionError:
            P = torch.zeros_like(x)
        
        return P

    def single(self, theta, phi, l, m, weight):
        # if m >= 0:
        #     return ((-1)**(-m)) * weight * torch.cos(m * theta) * self.assoc_legendre(torch.cos(phi), l, m)
        # else:
        #     return ((-1)**(-m)) * weight * torch.sin(-m * theta) * self.assoc_legendre(torch.cos(phi), l, -m)
        if m > 0:

            Y = 1.4142 * weight * torch.cos(m * theta) * self.assoc_legendre(torch.cos(phi), l, m)

        elif m < 0:

            Y = 1.4142 * weight * torch.sin(torch.abs(m) * theta) * self.assoc_legendre(torch.cos(phi), l, torch.abs(m))
        
        elif m == 0:

            Y = weight * self.legendre(torch.cos(phi), l)

        return Y

    def basis(self, theta, phi):
        num = len(self)
        height, width = theta.shape

        ret = torch.zeros(num, height, width)
        for i in range(num):
            ret[i,...] = self.single(theta, phi, self.l[i], self.m[i], self.weights[i])
            # import matplotlib.pyplot as plt
            # plt.imshow(ret[i,...].T)
            # plt.show()
                
        return ret

# theta, phi = torch.meshgrid(torch.linspace(0, 2*torch.pi, 200), torch.linspace(0, torch.pi, 100))
# SphericalHarmonicMix(9).basis(theta.cuda(), phi.cuda())

# eucalyptusgrove =  torch.tensor([[ 0.38,  0.43,  0.45],
#                                  [-0.29, -0.36, -0.41],
#                                  [ 0.04,  0.03,  0.01],
#                                  [ 0.10,  0.10,  0.09],
#                                  [-0.06, -0.06, -0.04],
#                                  [-0.01,  0.01,  0.05],
#                                  [-0.09, -0.13, -0.15],
#                                  [ 0.06,  0.05,  0.04],
#                                  [ 0.02, -0.00, -0.05]]) 

class SHEnvmapMaterialNetwork(nn.Module):
    def __init__(self, multires=0, dims=[256, 256, 256],
                 white_specular=False,
                 white_light=False,
                 max_sh_order=2,
                 fix_specular_albedo=False,
                 specular_albedo=[-1.,-1.,-1.]):
        super().__init__()

        input_dim = 3
        self.embed_fn = None
        if multires > 0:
            self.embed_fn, input_dim = get_embedder(multires)
        
        self.actv_fn = nn.ELU()
        ############## spatially-varying diffuse albedo############
        print('Diffuse albedo network size: ', dims)
        diffuse_albedo_layers = []
        dim = input_dim
        for i in range(len(dims)):
            diffuse_albedo_layers.append(nn.Linear(dim, dims[i]))
            diffuse_albedo_layers.append(self.actv_fn)
            dim = dims[i]
        diffuse_albedo_layers.append(nn.Linear(dim, 3))

        self.diffuse_albedo_layers = nn.Sequential(*diffuse_albedo_layers)

        ##################### specular rgb ########################
        self.max_sh_order = max_sh_order
        self.sh_mix = SphericalHarmonicMix(self.max_sh_order)

        phi, theta = torch.meshgrid(torch.linspace(0, torch.pi, 256), torch.linspace(-torch.pi, torch.pi, 512), indexing='ij')
        self.sh_basis = self.sh_mix.basis(theta, phi).cuda()

        self.numSHs = len(self.sh_mix)
        print('Max order of SH: ', self.max_sh_order)
        print('So, number of SH used: ', self.numSHs)
        self.white_light = white_light
        
        testcoeffs = torch.zeros((self.numSHs, 3))
        testcoeffs[0,:] = 1.0

        if self.white_light: 
            print('Using white light!')
            self.rgb_coeffs = nn.Parameter(torch.randn((self.numSHs, 1)), requires_grad=True)
        else:
            self.rgb_coeffs = nn.Parameter(testcoeffs, requires_grad=True)
        
        self.white_specular = white_specular
        self.fix_specular_albedo = fix_specular_albedo
        if self.fix_specular_albedo:
            print('Fixing specular albedo: ', specular_albedo)
            specular_albedo = np.array(specular_albedo).astype(np.float32)
            assert(np.all(np.logical_and(specular_albedo > 0., specular_albedo < 1.)))
            self.specular_reflectance = nn.Parameter(torch.from_numpy(specular_albedo).reshape((1, 3)),
                                                     requires_grad=False)
        else:
            if self.white_specular:
                print('Using white specular reflectance')
                self.specular_reflectance = nn.Parameter(torch.ones(1, 1),
                                                         requires_grad=True)
            else:
                self.specular_reflectance = nn.Parameter(torch.ones(1, 3),
                                                         requires_grad=True)
            self.specular_reflectance.data = torch.abs(self.specular_reflectance.data)

        # optimize
        roughness = [np.random.uniform(1.5, 2.0)]
        roughness = np.array(roughness).astype(dtype=np.float32).reshape((1, 1))
        print('init roughness: ', roughness)
        self.roughness = nn.Parameter(torch.from_numpy(roughness),
                                      requires_grad=True)

        # blending weights
        self.blending_weights_layers = []
    
    def get_light(self):
        rgb_coeffs = self.rgb_coeffs.clone().detach()
        sh_basis = self.sh_basis
        return rgb_coeffs, sh_basis
    
    def get_base_materials(self):
        roughness = self.roughness.clone().detach()
        shininess = torch.zeros_like(roughness)
        bandwidth_parameter = torch.zeros_like(roughness)
        if self.fix_specular_albedo:
            specular_reflectance = self.specular_reflectance
        else:
            specular_reflectance = torch.sigmoid(self.specular_reflectance.clone().detach())
            if self.white_specular:
                specular_reflectance = specular_reflectance.expand((-1, 3))
        return roughness, shininess, specular_reflectance, bandwidth_parameter

    def forward(self, points):
        if points is None:
            diffuse_albedo = None
            blending_weights = None
        else:
            if self.embed_fn is not None:
                points = self.embed_fn(points)
            diffuse_albedo = torch.sigmoid(self.diffuse_albedo_layers(points))

            blending_weights = None
        
        if self.fix_specular_albedo:
            specular_reflectance = self.specular_reflectance
        else:
            specular_reflectance = torch.sigmoid(self.specular_reflectance)
            if self.white_specular:
                specular_reflectance = specular_reflectance.expand((-1, 3))
        
        roughness = torch.sigmoid(self.roughness)

        ret = dict([
            ('sh_rgb_coeffs', self.rgb_coeffs),
            ('sh_basis', self.sh_basis),
            ('sh_roughness', roughness),
            ('sh_specular_reflectance', specular_reflectance),
            ('sh_diffuse_albedo', diffuse_albedo),
            ('sh_blending_weights', blending_weights)
        ])
        return ret

