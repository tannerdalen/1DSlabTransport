from SlabTransportSolver import Region, Model
import numpy as np

### General idea:
###     - Create Regions (think of cells in MCNP)
###     - Create a Mesh object for each Region
###     - Create a Model composed of Regions
###     - Combine Region Mesh objects into large Model object
###     - Solve combined Mesh for the Model properties

def UniformInfiniteMedium():
    
    width = 1
    sigma_t = 1
    sigma_s = 0.8
    source  = 1
    num_cells = 100
    num_angs = 8
    
    r = Region(width, sigma_t, sigma_s, source, "inf_medium", "reflecting", "reflecting")
    r.applyMesh(num_cells, width/num_cells, num_angs)
    
    model = Model(r)
    
    # Source Iteration
    _ = model.doTransport(itermethod="SI", LOUD=False)
    model.solution_UIM()
    model.plotModel(show_AnalySol=True)
    model.plotSpectralRadius()
    model.plotIterations()
    
    # Quasi-Diffusion
    _ = model.doTransport(itermethod="QD", LOUD=False)
    model.solution_UIM()
    model.plotModel(show_AnalySol=True)
    model.plotSpectralRadius()
    model.plotIterations()
    
    return 

# UniformInfiniteMedium()

def TwoRegionReflecting():
    
    width = 5
    sigma_t = 10
    sigma_s = 8
    source  = 10
    num_cells = 200
    num_angs = 4
    
    r1 = Region(width, sigma_t, sigma_s, source, "1", "reflecting", "vacuum")
    r2 = Region(width, sigma_t, sigma_s, 5, "2", "vacuum", "reflecting")
    
    r1.applyMesh(num_cells, width/num_cells, num_angs)
    r2.applyMesh(num_cells, width/num_cells, num_angs)
    
    model = Model(r1,r2)#, angfluxLBC=10, angfluxRBC=12)
    
    _ = model.doTransport(itermethod="QD",LOUD=False)
    # model.solution_UIM()
    model.plotModel(show_AnalySol=False, show_current=True) # Shows current
    # model.plotOptics()
    
    return

# TwoRegionReflecting()

def ApplicationProblem():
    
    r1 = Region(1, 1, 0, 1, "1", "vacuum", "vacuum")
    r2 = Region(1, 1, 0.1, 1, "2", "vacuum", "vacuum")
    r3 = Region(1, 1, 0.2, 10, "3", "vacuum", "vacuum")
    r4 = Region(1, 1, 0.3, 1, "4", "vacuum", "vacuum")
    r5 = Region(1, 1, 0.4, 1, "5", "vacuum", "vacuum")
    r6 = Region(1, 1, 0.5, 1, "6", "vacuum", "vacuum")
    r7 = Region(1, 1, 0.6, 1, "7", "vacuum", "vacuum")
    r8 = Region(1, 1, 0.7, 1, "8", "vacuum", "vacuum")
    r9 = Region(1, 1, 0.8, 1, "9", "vacuum", "vacuum")
    r10 = Region(1, 1, 0.999, 1, "10", "vacuum", "vacuum")
    
    cells_per_region = 100
    width = 1/cells_per_region
    angdegree = 8
    
    r1.applyMesh(cells_per_region, width, angdegree)
    r2.applyMesh(cells_per_region, width, angdegree)
    r3.applyMesh(cells_per_region, width, angdegree)
    r4.applyMesh(cells_per_region, width, angdegree)
    r5.applyMesh(cells_per_region, width, angdegree)
    r6.applyMesh(cells_per_region, width, angdegree)
    r7.applyMesh(cells_per_region, width, angdegree)
    r8.applyMesh(cells_per_region, width, angdegree)
    r9.applyMesh(cells_per_region, width, angdegree)
    r10.applyMesh(cells_per_region, width, angdegree)
    
    model = Model(r1,r2,r3,r4,r5,r6,r7,r8,r9,r10)
    
    _ = model.doTransport(itermethod="QD")
    
    model.plotModel(show_current=False)
    model.plotSpectralRadius()
    model.plotIterations()
    
    print(f"Est. spectral radius = {model.spectral_radii[-1]}")
    
    return

# ApplicationProblem()

def SrcFreePureAbsorber():
    
    width = 1
    sigma_t = 5
    sigma_s = 0
    source  = 0
    num_cells = 100
    angdegs = 10
    
    r = Region(width, sigma_t, sigma_s, source, "inf_medium", "vacuum", "vacuum")
    r.applyMesh(num_cells, width/num_cells, angdegs)
    
    model = Model(r, angfluxLBC=10, angfluxRBC=0)
    
    _ = model.doTransport(itermethod="QD",LOUD=False)
    model.solution_SFPA()
    model.plotModel(show_AnalySol=True) # Shows current
    # model.plotModel(show_AnalySol=True)  # Doesn't show current
    # model.plotOptics()
    
    return 

# SrcFreePureAbsorber()

def SrcFreeHalfSpace():
    
    num_cells = 400
    angdeg = 4
    
    r1 = Region(8, 1, 0.9, 0, "half1", "vacuum", "vacuum")
    r2 = Region(12, 3, 1.5, 0, "half2", "vacuum", "vacuum")
    
    r1.applyMesh(int(8/20*num_cells), 20/num_cells, angdeg)
    r2.applyMesh(int(12/20*num_cells), 20/num_cells, angdeg)
    
    model = Model(r1,r2, angfluxLBC=10, angfluxRBC=8)
    
    _ = model.doTransport(itermethod="QD",LOUD=False)
    # model.plotModel(show_AnalySol=False)
    # model.plotOptics()
    
    # Compare to excel sheet
    
    scalflux_analytical = [
        15.1924105,
        12.42069136,
        10.32110292,
        8.638627101,
        7.252849891,
        6.096886441,
        5.127034092,
        4.31109668,
        3.623631504,
        3.043805081,
        2.554266969,
        2.14044928,
        1.79006645,
        1.492725088,
        1.239603678,
        1.023179075,
        0.836978405,
        0.675321336,
        0.532972682,
        0.404498315,
        0.28276821,
        0.28276821,
        0.169866755,
        0.105691491,
        0.066770412,
        0.008001988,
        0.000965989,
        0.000116625,
        1.40817E-05,
        1.71072E-06,
        2.94484E-07,
        7.63996E-07,
        6.1258E-06,
        5.07146E-05,
        0.000420058,
        0.003479268,
        0.028818295,
        0.238751308,
        2.00845306,
        3.216496044,
        5.311347819,
        9.372583002,
        ]
    
    x_centers_analytical = [
        0,
        0.4,
        0.8,
        1.2,
        1.6,
        2,
        2.4,
        2.8,
        3.2,
        3.6,
        4,
        4.4,
        4.8,
        5.2,
        5.6,
        6,
        6.4,
        6.8,
        7.2,
        7.6,
        8,
        8,
        8.05,
        8.1,
        8.15,
        8.985714286,
        9.821428571,
        10.65714286,
        11.49285714,
        12.32857143,
        13.16428571,
        14,
        14.83571429,
        15.67142857,
        16.50714286,
        17.34285714,
        18.17857143,
        19.01428571,
        19.85,
        19.9,
        19.95,
        20
        ]
    
    model.plotModel(show_AnalySol=True, sol_x=x_centers_analytical, sol_y=scalflux_analytical)
    
    return

# SrcFreeHalfSpace()

def Problem1(num_cells):
    
    r1 = Region(10,100,99.5,5,"thick","vacuum","reflecting")
    r1.applyMesh(num_cells, 10/num_cells, 4)
    
    model = Model(r1, angfluxLBC=np.array([0,10]))
    
    scalarfluxes = model.doTransport(itermethod="QD", LOUD=False)
    # print(scalarfluxes[:10])
    model.plotModel()
    model.plotIterations()
    # model.plotOptics()
    
    return 

# Problem1(10)
# Problem1(1000)

def ReedsProblemUniform():
    
    r1_width, r1_sigma = 2, 1
    r2_width, r2_sigma = 2, 1
    r3_width, r3_sigma = 1, 0
    r4_width, r4_sigma = 1, 5
    r5_width, r5_sigma = 2, 50
    tot_width = 8
    
    num_cells = 400
    angdeg = 8
    
    r1 = Region(r1_width,r1_sigma,0.9,0,"ScatNoSrc","vacuum","vacuum")
    r2 = Region(r2_width,r2_sigma,0.9,1,"ScatWSrc", "vacuum","vacuum")
    r3 = Region(r3_width,r3_sigma,0,  0,"Void",     "vacuum","vacuum")
    r4 = Region(r4_width,r4_sigma,0,  0,"Absorber", "vacuum","vacuum")
    r5 = Region(r5_width,r5_sigma,0, 50,"BigSrc",   "vacuum","reflecting")
    
    r1.applyMesh(int(r1_width/tot_width*num_cells), tot_width/num_cells, angdeg)
    r2.applyMesh(int(r2_width/tot_width*num_cells), tot_width/num_cells, angdeg)
    r3.applyMesh(int(r3_width/tot_width*num_cells), tot_width/num_cells, angdeg) # Vacuum
    r4.applyMesh(int(r4_width/tot_width*num_cells), tot_width/num_cells, angdeg)
    r5.applyMesh(int(r5_width/tot_width*num_cells), tot_width/num_cells, angdeg)
    
    model = Model(r1,r2,r3,r4,r5, angfluxLBC=0, angfluxRBC=0)
    
    _ = model.doTransport(itermethod="QD",LOUD=False)
    model.plotModel(show_current=True)
    # model.plotOptics()
    
    return 

# ReedsProblemUniform()

    
