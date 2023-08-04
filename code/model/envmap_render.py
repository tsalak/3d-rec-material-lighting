import torch
import torch.nn.functional as F

# compute envmap from SH
def compute_sh_envmap(rgb_coeffs, sh_basis, width=512, height=256):
    '''
    :param rgb_coeffs: [(max_SH_order+1)^2, 3]
    :param sh_basis: [(max_SH_order+1)^2, height, width]; 
    :param width: scalar; in pixels
    :param height: scalar; in pixels
    :returns [height, width, 3]; in equirectangular format
    '''
    # envmap = torch.zeros((height, width, 3)).cuda()
    # theta = torch.linspace(-torch.pi/2, 3 * torch.pi/2, width)  # azimuthal angle
    # phi = torch.linspace(0, torch.pi, height)                   # polar angle

    # THETA, PHI = torch.meshgrid(theta, phi, indexing='ij')

    # sh_basis = sh_basis(THETA, PHI).cuda()
    # for rgb_coeff, sh in zip(rgb_coeffs, sh_basis):
    #     envmap += rgb_coeff.view(1, 1, 3) * sh
    
    envmap = torch.sum(rgb_coeffs.unsqueeze(1).unsqueeze(1) * sh_basis.unsqueeze(-1), dim=0)

    return envmap.clamp(min=0., max=1.)

# compute envmap from SW
def compute_sw_envmap(rgb_coeffs, sw_basis, width=512, height=256):
    '''
    :param rgb_coeffs: [t_design_order + 1, 3]
    :param sw_basis: [t_design_order, height, width]; 
    :param width: scalar; in pixels
    :param height: scalar; in pixels
    :returns [height, width, 3]; in equirectangular format
    '''
    # envmap = torch.zeros((3, height, width)).cuda()
    envmap = rgb_coeffs[0,:].view(3, 1, 1)*0.2821*torch.ones((3, height, width)).cuda()
    # theta = 2 * torch.pi * torch.arange(width) / width  # azimuthal angle
    # theta = torch.linspace(-torch.pi/2, 3 * torch.pi/2, width)
    # phi = torch.pi * torch.arange(height) / height      # polar angle

    # PHI, THETA = torch.meshgrid(phi, theta, indexing='ij')

    # sw_basis = sw_mix.basis(THETA, PHI).cuda()
    for rgb_coeff, sw in zip(rgb_coeffs[1:], sw_basis):
        envmap += rgb_coeff.view(3, 1, 1) * sw.T.cuda()

    return envmap.clamp(min=0.0, max=1.0).permute((1, 2, 0))

def sample_envmap(envmap, directions):
    envmap = envmap.permute(2,0,1).unsqueeze(0)

    # Convert direction to latitude and longitude
    lat = torch.arcsin(directions[:,1])
    lon = torch.arctan2(directions[:,2], directions[:,0])

    # Map latitude and longitude to coordinates in the environment map
    lon[lon<-torch.pi/2] += 2*torch.pi
    # lon += torch.pi/2
    u = (lon + torch.pi/2) / (2 * torch.pi)
    v = (torch.pi/2 - lat) / (torch.pi)

    points = torch.stack((u,v), dim=1)

    # reshape points for use in grid_sample
    points = points.unsqueeze(0).unsqueeze(0)
    
    # normalize to [-1, 1]
    points = points * 2 - 1

    # sample the environment map at the given points using linear interpolation
    samples = F.grid_sample(envmap, points, mode="bilinear", align_corners=True)

    # reshape samples to match the shape of points
    samples = samples.squeeze(0).squeeze(1).T 

    return samples

# sample the envmap in the given directions and return the resulting color
def OLD_sample_envmap(envmap, directions):
    # print(directions)
    # Assume that the envmap is in equirectangular projection format
    # Convert direction to latitude and longitude
    lat = torch.arcsin(directions[:, 1])
    lon = torch.atan2(directions[:, 2], directions[:, 0])
    # print(lon)

    # Map latitude and longitude to coordinates in the environment map
    lon[lon < -torch.pi/2] += 2*torch.pi
    # lon += torch.pi / 2
    u = (lon + torch.pi/2) / (2 * torch.pi)
    v = (torch.pi/2 - lat) / torch.pi

    # Convert point coordinates to pixel coordinates
    h, w = envmap.shape[:2]
    u = u * (w-1)
    v = v * (h-1)

    # Extract neighboring pixels for each point
    umin = u.long()
    vmin = v.long()
    
    tlcolor = envmap[vmin, umin, :]
    trcolor = envmap[vmin, (umin+1)%w, :]
    blcolor = envmap[(vmin+1)%h, umin, :]
    brcolor = envmap[(vmin+1)%h, (umin+1)%w, :]

    # Compute fractional distances along each axis
    uf = (u - umin.float()).unsqueeze(-1)
    vf = (v - vmin.float()).unsqueeze(-1)

    color = tlcolor*(1-uf)*(1-vf) + trcolor*uf*(1-vf) + blcolor*(1-uf)*vf + brcolor*uf*vf

    return color

# import pandas as pd
# df = pd.read_hdf('icosphere_data.h5', '2562')
# phi = torch.deg2rad(torch.tensor(df['theta'].values))
# theta = torch.deg2rad(torch.tensor(df['phi'].values))

# N = 2500

# h = -1 + ((2 * (torch.arange(1,N))) / (N - 1))

# phi = torch.arccos(h)
# theta = torch.zeros_like(phi)
# for k in torch.arange(1,N-1):
#     theta[k] = ( theta[k-1] + (3.6 / torch.sqrt(torch.tensor(N))) * (1 / torch.sqrt(1 - h[k-1]**2)) )  % (2 * torch.pi)

# x = torch.sin(phi) * torch.cos(theta)
# y = torch.cos(phi)
# z = torch.sin(phi) * torch.sin(theta)
# light_dirs = torch.stack((x.view(-1,1), y.view(-1, 1), z.view(-1, 1)), dim=1).squeeze(-1).cuda().type(torch.float32)

def envmap_irradiance(envmap, normals):
    normals = normals.float()

    up = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).cuda()
    right = F.normalize(torch.cross(up, normals))
    up = F.normalize(torch.cross(normals, right))
    
    sample_delta = 0.05
    theta = torch.arange(0.0, 2.0 * torch.pi, sample_delta).cuda()
    phi = torch.arange(0.0, 0.5 * torch.pi, sample_delta).cuda()
    nr_samples = len(theta) * len(phi)
    THETA, PHI = torch.meshgrid(theta, phi, indexing='ij')
    theta, phi = THETA.flatten(), PHI.flatten()

    # spherical to cartesian (in tangent space)
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    # tangent space to world
    sample_vec = x.unsqueeze(1).unsqueeze(-1) * right.unsqueeze(0) + y.unsqueeze(1).unsqueeze(-1) * up.unsqueeze(0) + z.unsqueeze(1).unsqueeze(-1) *  normals.unsqueeze(0)

    texture = sample_envmap(envmap, sample_vec.view(-1, 3)).view(sample_vec.shape[0], sample_vec.shape[1], sample_vec.shape[2])

    irradiance = torch.sum(texture * torch.cos(phi).unsqueeze(-1).unsqueeze(-1) * torch.sin(phi).unsqueeze(-1).unsqueeze(-1), dim=0) * torch.pi / nr_samples

    return irradiance

def hammersley(i, N):
    return torch.tensor([float(i)/float(N), van_der_corput(i, 2)]) 

def van_der_corput(n, base):
    inv_base = 1.0 / float(base)
    denom = 1.0
    result = 0.0

    for _ in torch.arange(32): # maybe different range
        if n > 0:
            denom = n % 2
            result += denom * inv_base
            inv_base = inv_base / 2.0
            n = int(float(n) / 2.0)
    
    return result

def importance_sample_GGX(Xi, roughness, N):
    a = roughness * roughness

    theta = ( 2 * torch.pi * Xi[..., 0] ).unsqueeze(-1)
    cos_phi = torch.sqrt( (1- Xi[..., 1]) / ( 1 + (a*a - 1) * Xi[..., 1] + 1e-6 ) ).T
    sin_phi = torch.sqrt( 1 - cos_phi * cos_phi )

    Hx = sin_phi * torch.cos(theta)
    Hy = sin_phi * torch.sin(theta)
    Hz = cos_phi

    ## POSSIBLE MISTAKE
    # up = torch.tensor([0., 0., 1.]) if torch.abs(Hz) < 0.999 else torch.tensor([1., 0., 0.])
    up = torch.where(torch.abs(cos_phi) < 0.999, torch.tensor([0., 0., 1.]).cuda(), torch.tensor([1., 0., 0.]).cuda())
    # up = up.cuda()
    tangent_x = F.normalize(torch.cross(up.unsqueeze(1), N.unsqueeze(0)), dim=2)
    tangent_y = torch.cross(N.unsqueeze(0), tangent_x)

    # tangent to world space
    return Hx.unsqueeze(1) * tangent_x + Hy.unsqueeze(1) * tangent_y + Hz.unsqueeze(1) * N.unsqueeze(0)

def G_Schlick_GGX(n_dot_v, roughness):
    a = roughness
    k = (a * a) / 2.0

    nom = n_dot_v
    denom = n_dot_v * (1.0 - k) + k

    return nom / (denom + 1e-6)

def G_Smith(roughness, n_dot_v, n_dot_l):
    ggx2 = G_Schlick_GGX(n_dot_v, roughness)
    ggx1 = G_Schlick_GGX(n_dot_l, roughness)
    
    return ggx1 * ggx2

num_samples = 1024
X = torch.zeros((num_samples, 2)).cuda()
for i in torch.arange(num_samples):
    X[i, :] = hammersley( i, num_samples)

def render_with_envmap(points, normals, view_dirs, diffuse_albedo, specular_reflectance, roughness, envmap):
    '''
    :param points: [..., 3]
    :param light_dirs: [H, W, 3]; same shape as envmap
    :param normals: [..., 3]; must have unit norm
    :param view_dirs: [..., 3]; must have unit norm
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param specular_reflectance: [1, 3];
    :param shininess: scalar
    :param envmap: [H, W, 3]; must be in equirectangular format
    :return [..., 3]
    '''
    TINY_NUMBER = 1e-6
    '''
    light_dirs =  torch.tensor([[ 0.893,  0.413, -0.182],
                                    [ 0.413, -0.182,  0.893],
                                    [-0.893, -0.413, -0.182],
                                    [-0.182,  0.893,  0.413],
                                    [-0.413, -0.182, -0.893],
                                    [-0.182, -0.893, -0.413],
                                    [-0.413,  0.182,  0.893],
                                    [ 0.182, -0.893,  0.413],
                                    [ 0.413,  0.182, -0.893],
                                    [ 0.182,  0.893, -0.413],
                                    [ 0.893, -0.413,  0.182],
                                    [-0.893,  0.413,  0.182],
                                    [-0.292, -0.296,  0.910],
                                    [-0.296,  0.910, -0.292],
                                    [ 0.292,  0.296,  0.910],
                                    [ 0.910, -0.292, -0.296],
                                    [ 0.296,  0.910,  0.292],
                                    [ 0.910,  0.292,  0.296],
                                    [ 0.296, -0.910, -0.292],
                                    [-0.910,  0.292, -0.296],
                                    [-0.296, -0.910,  0.292],
                                    [-0.910, -0.292,  0.296],
                                    [-0.292,  0.296, -0.910],
                                    [ 0.292, -0.296, -0.910],
                                    [-0.575,  0.024,  0.818],
                                    [ 0.024,  0.818, -0.575],
                                    [ 0.575, -0.024,  0.818],
                                    [ 0.818, -0.575,  0.024],
                                    [-0.024,  0.818,  0.575],
                                    [ 0.818,  0.575, -0.024],
                                    [-0.024, -0.818, -0.575],
                                    [-0.818,  0.575,  0.024],
                                    [ 0.024, -0.818,  0.575],
                                    [-0.818, -0.575, -0.024],
                                    [-0.575, -0.024, -0.818],
                                    [ 0.575,  0.024, -0.818],
                                    [-0.129,  0.052,  0.990],
                                    [ 0.052,  0.990, -0.129],
                                    [ 0.129, -0.052,  0.990],
                                    [ 0.990, -0.129,  0.052],
                                    [-0.052,  0.990,  0.129],
                                    [ 0.990,  0.129, -0.052],
                                    [-0.052, -0.990, -0.129],
                                    [-0.990,  0.129,  0.052],
                                    [ 0.052, -0.990,  0.129],
                                    [-0.990, -0.129, -0.052],
                                    [-0.129, -0.052, -0.990],
                                    [ 0.129,  0.052, -0.990],
                                    [ 0.718,  0.657, -0.229],
                                    [ 0.657, -0.229,  0.718],
                                    [-0.718, -0.657, -0.229],
                                    [-0.229,  0.718,  0.657],
                                    [-0.657, -0.229, -0.718],
                                    [-0.229, -0.718, -0.657],
                                    [-0.657,  0.229,  0.718],
                                    [ 0.229, -0.718,  0.657],
                                    [ 0.657,  0.229, -0.718],
                                    [ 0.229,  0.718, -0.657],
                                    [ 0.718, -0.657,  0.229],
                                    [-0.718,  0.657,  0.229],
                                    [ 0.863,  0.468,  0.189],
                                    [ 0.468,  0.189,  0.863],
                                    [-0.863, -0.468,  0.189],
                                    [ 0.189,  0.863,  0.468],
                                    [-0.468,  0.189, -0.863],
                                    [ 0.189, -0.863, -0.468],
                                    [-0.468, -0.189,  0.863],
                                    [-0.189, -0.863,  0.468],
                                    [ 0.468, -0.189, -0.863],
                                    [-0.189,  0.863, -0.468],
                                    [ 0.863, -0.468, -0.189],
                                    [-0.863,  0.468, -0.189],
                                    [ 0.773, -0.517,  0.368],
                                    [-0.517,  0.368,  0.773],
                                    [-0.773,  0.517,  0.368],
                                    [ 0.368,  0.773, -0.517],
                                    [ 0.517,  0.368, -0.773],
                                    [ 0.368, -0.773,  0.517],
                                    [ 0.517, -0.368,  0.773],
                                    [-0.368, -0.773, -0.517],
                                    [-0.517, -0.368, -0.773],
                                    [-0.368,  0.773,  0.517],
                                    [ 0.773,  0.517, -0.368],
                                    [-0.773, -0.517, -0.368],
                                    [-0.848, -0.066, -0.526],
                                    [-0.066, -0.526, -0.848],
                                    [ 0.848,  0.066, -0.526],
                                    [-0.526, -0.848, -0.066],
                                    [ 0.066, -0.526,  0.848],
                                    [-0.526,  0.848,  0.066],
                                    [ 0.066,  0.526, -0.848],
                                    [ 0.526,  0.848, -0.066],
                                    [-0.066,  0.526,  0.848],
                                    [ 0.526, -0.848,  0.066],
                                    [-0.848,  0.066,  0.526],
                                    [ 0.848, -0.066,  0.526],
                                    [ 0.010,  0.943,  0.333],
                                    [ 0.943,  0.333,  0.010],
                                    [-0.010, -0.943,  0.333],
                                    [ 0.333,  0.010,  0.943],
                                    [-0.943,  0.333, -0.010],
                                    [ 0.333, -0.010, -0.943],
                                    [-0.943, -0.333,  0.010],
                                    [-0.333, -0.010,  0.943],
                                    [ 0.943, -0.333, -0.010],
                                    [-0.333,  0.010, -0.943],
                                    [ 0.010, -0.943, -0.333],
                                    [-0.010,  0.943, -0.333],
                                    [ 0.786, -0.405, -0.468],
                                    [-0.405, -0.468,  0.786],
                                    [-0.786,  0.405, -0.468],
                                    [-0.468,  0.786, -0.405],
                                    [ 0.405, -0.468, -0.786],
                                    [-0.468, -0.786,  0.405],
                                    [ 0.405,  0.468,  0.786],
                                    [ 0.468, -0.786, -0.405],
                                    [-0.405,  0.468, -0.786],
                                    [ 0.468,  0.786,  0.405],
                                    [ 0.786,  0.405,  0.468],
                                    [-0.786, -0.405,  0.468],
                                    [-0.737,  0.621, -0.266],
                                    [ 0.621, -0.266, -0.737],
                                    [ 0.737, -0.621, -0.266],
                                    [-0.266, -0.737,  0.621],
                                    [-0.621, -0.266,  0.737],
                                    [-0.266,  0.737, -0.621],
                                    [-0.621,  0.266, -0.737],
                                    [ 0.266,  0.737,  0.621],
                                    [ 0.621,  0.266,  0.737],
                                    [ 0.266, -0.737, -0.621],
                                    [-0.737, -0.621,  0.266],
                                    [ 0.737,  0.621,  0.266],
                                    [ 0.727, -0.027, -0.686],
                                    [-0.027, -0.686,  0.727],
                                    [-0.727,  0.027, -0.686],
                                    [-0.686,  0.727, -0.027],
                                    [ 0.027, -0.686, -0.727],
                                    [-0.686, -0.727,  0.027],
                                    [ 0.027,  0.686,  0.727],
                                    [ 0.686, -0.727, -0.027],
                                    [-0.027,  0.686, -0.727],
                                    [ 0.686,  0.727,  0.027],
                                    [ 0.727,  0.027,  0.686],
                                    [-0.727, -0.027,  0.686],
                                    [ 0.665,  0.581,  0.469],
                                    [ 0.581,  0.469,  0.665],
                                    [-0.665, -0.581,  0.469],
                                    [ 0.469,  0.665,  0.581],
                                    [-0.581,  0.469, -0.665],
                                    [ 0.469, -0.665, -0.581],
                                    [-0.581, -0.469,  0.665],
                                    [-0.469, -0.665,  0.581],
                                    [ 0.581, -0.469, -0.665],
                                    [-0.469,  0.665, -0.581],
                                    [ 0.665, -0.581, -0.469],
                                    [-0.665,  0.581, -0.469],
                                    [-0.580, -0.779,  0.238],
                                    [-0.779,  0.238, -0.580],
                                    [ 0.580,  0.779,  0.238],
                                    [ 0.238, -0.580, -0.779],
                                    [ 0.779,  0.238,  0.580],
                                    [ 0.238,  0.580,  0.779],
                                    [ 0.779, -0.238, -0.580],
                                    [-0.238,  0.580, -0.779],
                                    [-0.779, -0.238,  0.580],
                                    [-0.238, -0.580,  0.779],
                                    [-0.580,  0.779, -0.238],
                                    [ 0.580, -0.779, -0.238],
                                    [ 0.959,  0.101, -0.266],
                                    [ 0.101, -0.266,  0.959],
                                    [-0.959, -0.101, -0.266],
                                    [-0.266,  0.959,  0.101],
                                    [-0.101, -0.266, -0.959],
                                    [-0.266, -0.959, -0.101],
                                    [-0.101,  0.266,  0.959],
                                    [ 0.266, -0.959,  0.101],
                                    [ 0.101,  0.266, -0.959],
                                    [ 0.266,  0.959, -0.101],
                                    [ 0.959, -0.101,  0.266],
                                    [-0.959,  0.101,  0.266],
                                    [-0.784,  0.284,  0.551],
                                    [ 0.284,  0.551, -0.784],
                                    [ 0.784, -0.284,  0.551],
                                    [ 0.551, -0.784,  0.284],
                                    [-0.284,  0.551,  0.784],
                                    [ 0.551,  0.784, -0.284],
                                    [-0.284, -0.551, -0.784],
                                    [-0.551,  0.784,  0.284],
                                    [ 0.284, -0.551,  0.784],
                                    [-0.551, -0.784, -0.284],
                                    [-0.784, -0.284, -0.551],
                                    [ 0.784,  0.284, -0.551],
                                    [ 0.167,  0.979,  0.113],
                                    [ 0.979,  0.113,  0.167],
                                    [-0.167, -0.979,  0.113],
                                    [ 0.113,  0.167,  0.979],
                                    [-0.979,  0.113, -0.167],
                                    [ 0.113, -0.167, -0.979],
                                    [-0.979, -0.113,  0.167],
                                    [-0.113, -0.167,  0.979],
                                    [ 0.979, -0.113, -0.167],
                                    [-0.113,  0.167, -0.979],
                                    [ 0.167, -0.979, -0.113],
                                    [-0.167,  0.979, -0.113],
                                    [ 0.904,  0.099,  0.417],
                                    [ 0.099,  0.417,  0.904],
                                    [-0.904, -0.099,  0.417],
                                    [ 0.417,  0.904,  0.099],
                                    [-0.099,  0.417, -0.904],
                                    [ 0.417, -0.904, -0.099],
                                    [-0.099, -0.417,  0.904],
                                    [-0.417, -0.904,  0.099],
                                    [ 0.099, -0.417, -0.904],
                                    [-0.417,  0.904, -0.099],
                                    [ 0.904, -0.099, -0.417],
                                    [-0.904,  0.099, -0.417],
                                    [ 0.279,  0.349, -0.895],
                                    [ 0.349, -0.895,  0.279],
                                    [-0.279, -0.349, -0.895],
                                    [-0.895,  0.279,  0.349],
                                    [-0.349, -0.895, -0.279],
                                    [-0.895, -0.279, -0.349],
                                    [-0.349,  0.895,  0.279],
                                    [ 0.895, -0.279,  0.349],
                                    [ 0.349,  0.895, -0.279],
                                    [ 0.895,  0.279, -0.349],
                                    [ 0.279, -0.349,  0.895],
                                    [-0.279,  0.349,  0.895],
                                    [ 0.556, -0.677,  0.483],
                                    [-0.677,  0.483,  0.556],
                                    [-0.556,  0.677,  0.483],
                                    [ 0.483,  0.556, -0.677],
                                    [ 0.677,  0.483, -0.556],
                                    [ 0.483, -0.556,  0.677],
                                    [ 0.677, -0.483,  0.556],
                                    [-0.483, -0.556, -0.677],
                                    [-0.677, -0.483, -0.556],
                                    [-0.483,  0.556,  0.677],
                                    [ 0.556,  0.677, -0.483],
                                    [-0.556, -0.677, -0.483]]).cuda()
    '''  
    # light_dirs = normals
    # light_colors = sample_envmap(envmap, light_dirs)

    # define the center of the scene
    # scene_center = torch.tensor([0.0, 0.0, 0.0]).cuda()

    # Compute the direction from the surface point to the center of the scene
    # light_dirs = normals #- scene_center
    # light_dirs = light_dirs / (light_dirs.norm(dim=-1, keepdim=True)+TINY_NUMBER)

    # Sample the environment map in the light directions
    # light_colors = sample_envmap(envmap, light_dirs)

    # Compute diffuse component
    # n_dot_l = torch.sum(normals * light_dirs, dim=-1, keepdim=True).clamp_(min=0.)
    irradiance = envmap_irradiance(envmap, normals)
    diffuse_rgb = irradiance * diffuse_albedo 
    # diffuse_rgb = (diffuse_albedo * torch.sum((n_dot_l * light_colors), dim=0) * 4 / light_dirs.shape[0]).clamp_(min=0., max=1.).squeeze(0)
    
    # Compute specular component
    specular_lighting = torch.zeros_like(normals)

    num_samples = 1024
    # for i in torch.arange(num_samples):
    # Xi = X[i, :]
    H = importance_sample_GGX(X, roughness, normals)
    L = 2 * torch.sum(H * view_dirs.unsqueeze(0), dim=-1, keepdim=True) * H - view_dirs.unsqueeze(0)

    n_dot_l = torch.sum(normals.unsqueeze(0) * L, dim=-1, keepdim=True).clamp(min=0., max=1.)
    n_dot_v = torch.sum(normals * view_dirs, dim=-1, keepdim=True).clamp(min=0., max=1.)
    v_dot_h = torch.sum(view_dirs.unsqueeze(0) * H, dim=-1, keepdim=True).clamp(min=0., max=1.)
    n_dot_h = torch.sum(normals.unsqueeze(0) * H, dim=-1, keepdim=True).clamp(min=0., max=1.)

    sample_color = sample_envmap(envmap, L.view(-1, 3)).view(H.shape[0], H.shape[1], H.shape[2])
    
    mask = (n_dot_l > 0)

    G = G_Smith( roughness, n_dot_v.unsqueeze(0), n_dot_l )#.T
    Fc = torch.pow( 1 - v_dot_h, 5 )#.unsqueeze(-1)
    F = (1 - Fc) * specular_reflectance + Fc

    specular_lighting = torch.sum(mask * sample_color * F * G * v_dot_h / (n_dot_h * n_dot_v.unsqueeze(0) + TINY_NUMBER), dim=0)
    
    specular_rgb = specular_lighting / num_samples
    # half_dirs = light_dirs.unsqueeze(1) + view_dirs.unsqueeze(0)
    # half_dirs = half_dirs / (half_dirs.norm(dim=-1, keepdim=True)+TINY_NUMBER)
    # h_dot_n = torch.sum(half_dirs * normals.unsqueeze(0), dim=-1, keepdim=True)
    # v_dot_h = torch.sum(view_dirs.unsqueeze(0) * half_dirs, dim=-1, keepdim=True)
    # D = torch.clamp(h_dot_n, min=0.) * (roughness ** 2) / (torch.pi * ((h_dot_n ** 2) * (roughness ** 2) + (1 - (h_dot_n ** 2))) ** 2)
    # F = specular_reflectance + (1. - specular_reflectance) * torch.pow(1 - v_dot_h, 5)
    # D = torch.exp(-(torch.arccos(h_dot_n)**2)/(roughness**2 + TINY_NUMBER))/ (torch.pi * roughness* roughness + TINY_NUMBER)
    # F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

    # v_dot_n = torch.sum(view_dirs.unsqueeze(0) * normals.unsqueeze(0), dim=-1, keepdim=True)  # equals <o, n>
    # chi = torch.clamp(v_dot_h, min=0.) / (torch.clamp(v_dot_n, min=0.) + TINY_NUMBER)
    # tan2 = (1 - (v_dot_h.clamp(min=0.) ** 2)) / (v_dot_h.clamp(min=0.) ** 2 + TINY_NUMBER)
    # G = (chi.clamp(min=0.) * 2) / (1 + torch.sqrt(1 + (roughness ** 2) * tan2))
    # specular_rgb = (torch.sum(F*G*D * light_colors / (v_dot_n.clamp(min=0.) + TINY_NUMBER), dim=0) * torch.pi / light_dirs.shape[0]).clamp_(min=0., max=1.)
    # n_dot_v = torch.sum(normals * view_dirs, dim=-1, keepdim=True)
    # light_dirs = 2 * n_dot_v * normals - view_dirs
    # light_colors = sample_envmap(envmap, light_dirs)
    # n_dot_l = torch.sum(normals * light_dirs, dim=-1, keepdim=True)
    # n_dot_v = torch.sum(normals * view_dirs, dim=-1, keepdim=True)
    # reflection_dirs = 2 * n_dot_v * normals - view_dirs
    # light_colors = sample_envmap(envmap, reflection_dirs)
    # r_dot_v = torch.sum(view_dirs * reflection_dirs, dim=-1, keepdim=True)
    # specular_intensity = r_dot_v.clamp(min=0.0) ** roughness
    # specular_rgb = specular_reflectance.unsqueeze(0)  * light_colors

    # combine diffuse and specular rgb, then return
    rgb = specular_rgb + diffuse_rgb
    ret = {'our_rgb': rgb,
           'our_diffuse_rgb': diffuse_rgb,
           'our_specular_rgb': specular_rgb,
           'our_diffuse_albedo': diffuse_albedo}
    return ret


# from sh_envmap_material import SphericalHarmonicMix

# testcoeffs =       torch.tensor([[ 2.586760,  2.730808,  3.152812],
#                                  [-0.431493, -0.665128, -0.969124],
#                                  [-0.353886,  0.048348,  0.672755],
#                                  [-0.604269, -0.886230, -1.298684],
#                                  [ 0.320121,  0.422942,  0.541783],
#                                  [-0.137435, -0.168666, -0.229637],
#                                  [-0.052101, -0.149999, -0.232127],
#                                  [-0.117312, -0.167151, -0.265015],
#                                  [-0.090028, -0.021071,  0.089560]])  
# shmix = SphericalHarmonicMix(lmax=2)

# phi, theta = torch.meshgrid(torch.linspace(0, torch.pi, 256), torch.linspace(-torch.pi, torch.pi, 512), indexing='ij')

# import matplotlib.pyplot as plt
# plt.imshow(compute_sh_envmap(testcoeffs/torch.pi, shmix.basis(theta, phi)))
# plt.show()

# x = torch.sin(phi) * torch.cos(theta)
# y = torch.sin(phi) * torch.sin(theta)
# z = torch.cos(phi)

# import matplotlib.pyplot as plt
# plt.imshow(shmix.basis(theta, phi)[8])
# plt.show()
# plt.imshow(0.546274*(x*x - y*y))
# plt.show()

