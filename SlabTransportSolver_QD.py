### Tanner Heatherly
### NSE 654 - Computational Particle Transport
### June 13th, 2024
### Final Project
 #
### Descritization method: Diamond Difference
 #
### Final Project addition: Quasi-Diffusion

from dataclasses import dataclass, field
from typing import Any
import itertools
import numpy as np
import matplotlib.pyplot as plt

BOUNDARY_CONDITIONS = ["vacuum", "reflecting", None]
ITERATION_METHODS = ["SI","QD"]

##########################################

class Mesh:
    
    """ Mesh to lay onto a Region. """
    
    def __init__(self, cellbins=1, cell_widths=1, angdegree=2) -> None:
        """  
        The methodology goes like this...
        
        "cellbins" is used to define the number of spatial cells in
        the mesh. This is always an integer. Positive integers are
        not enforced, because the user is expected to know better.
        
        "cell_widths" is used to define the width of each spatial cell.
        There are two options for inputs:
            1. Integer - the mesh is made such that all cells have a 
                         uniform width == cell_width
            2. List    - Each element of cell_widths defines the width
                         for the "ith" cell. An error is thrown if
                         len(cel_widths) != cellbins.
                         
        "angdegree" is used to define the angular quadrature in each cell.
        Must be an even integer.
        """
        
        if not angdegree % 2 == 0:
            raise TransportError("Angular mesh degree must be even")
        
        self.cellbins = cellbins
        self.angdegree = angdegree
        self.angbins = int(self.angdegree/2)
        
        # Update cell widths if uniform
        if not isinstance(cell_widths, list): # Uniform cell widths desired, convert to list
            self.cell_widths = [cell_widths]*self.cellbins
        else:
            self.cell_widths = cell_widths
        
        # Initialize the angular flux array. This will remain zeros until transport is performed.
        # Top down (first idx) is spatial elements, left to right (second idx) is angular.
        # There are half as many angbins as their are mu/wgt pairs. 
        # For example if self.cells = array([[1,2,3], then self.angflux[1,2] == 6 
        #                                    [4,5,6],
        #                                    [7,8,9]) 
        #
        # The angular flux array only represents the angular fluxes in a particular direction
        
        self.angfluxes = np.zeros([self.cellbins, self.angbins])
        self.scalarfluxes = np.zeros([self.cellbins])
        self.netCurrent = np.zeros([self.cellbins])
        self.analyticalsol = np.zeros([self.cellbins])
        
        # Create separate array for edges of a given cell
        self.scalarfluxes_edges = np.zeros([self.cellbins+1])
        self.angfluxes_edges = np.zeros([self.cellbins+1, self.angbins])
        
        # Create mus and wgts for desired degree
        self.mus, self.wgts = np.polynomial.legendre.leggauss(angdegree)
        
        # Create cell bin center and edge locations
        self.xEdgelocs_left = []
        self.xEdgelocs_right = []
        self.xCenters  = []
        
        for i, width in enumerate(self.cell_widths):
            if i == 0:
                self.xEdgelocs_left.append(0)
                self.xEdgelocs_right.append(width)
                self.xCenters.append(width/2)
            else:
                # Avoid float point errors with round()
                self.xEdgelocs_left.append(round(self.xEdgelocs_left[i-1] + width, 10))
                self.xEdgelocs_right.append(round(self.xEdgelocs_right[i-1] + width, 10))
                
                # Cell widths are not guarenteed to be uniform, so this assigns centers accordingly
                self.xCenters.append(round(self.xCenters[i-1] + (self.cell_widths[i]/2 + self.cell_widths[i-1]/2), 10))
            
        # Checks
        if not len(list(self.cell_widths)) == self.cellbins:
            raise TransportError("Number of cell widths must equal number of cells")
            
        return
    
    def getPositiveOrdinates(self) -> (np.array, np.array):
        """ Gets only the positive mus and their corresponding weights """
        mus = np.array([mu for mu in self.mus if mu>0])
        wgts = self.wgts[0:len(self.mus)][::-1]
        
        return mus, wgts

@dataclass
class Region:
    
    """ Model segments """
    
    width: float = field(default=0)    # Region thickness
    sigma: float = field(default=0)    # Total cross section
    sigma_s: float = field(default=0)  # Scattering cross section
    source : float = field(default=0)  # Source "S"
    label: str = field(default="r1")   # Region name
    LBC: Any = field(default=None)     # Left boundary condition
    RBC: Any = field(default=None)     # Right boundary condition
    ALB: float = field(default=0)      # Absolute left boundary location 
    ARB: float = field(default=0)      # Absolute right boundary location

    def __post_init__(self) -> None:
        
        # Checks
        if not self.LBC in BOUNDARY_CONDITIONS or not self.RBC in BOUNDARY_CONDITIONS:
            raise TransportError(f"Invalid boundary conditions: {self.LBC},{self.RBC}")
            
    def applyMesh(self, cellbins=1, cell_widths=1, angdegree=2) -> None:
        """ Binds a Mesh onto a Region """
        
        self.mesh = Mesh(cellbins, cell_widths, angdegree)
        
        # Mesh must span the entire Region
        if not self.width == self.mesh.xEdgelocs_right[-1]:
            raise TransportError(f"Cells must span entire model {self.ALB,self.ARB}, not {self.mesh.xEdgelocs_right[-1],self.width}")
               
        return

class Model:
    
    """
    1-D model composed of Regions. Out of bounds
    considered vacuum (void).
    """
    
    def __init__(self, *regions: Region, angfluxLBC = 0, angfluxRBC = 0) -> None:
        
        # Definitions
        self.regions = list(regions)
        self._setAbsoluteLocations()
        self.angfluxLBC = angfluxLBC
        self.angfluxRBC = angfluxRBC
        self.mesh = None
        
        # Checks
        if len(set([r.label for r in regions])) != len([r.label for r in regions]):
            raise TransportError("Cannot have duplicate region labels")
        if "reflecting" in [r.LBC for r in regions][1:-1] or "reflecting" in [r.RBC for r in regions][1:-1]:
            raise TransportError("Cannot have a reflective boundary in the middle of the model")
        
        return
        
    def _setAbsoluteLocations(self) -> None:
        """ 
        Determines absolute location, starting 
        from zero, of the left and right boundary 
        of each region.
        """
        for i in range(len(self.regions)):
            # Leftmost region
            if i == 0:
                self.regions[i].ALB = 0
                self.regions[i].ARB = self.regions[i].width
            else:
                self.regions[i].ALB = self.regions[i-1].ARB
                self.regions[i].ARB = round(self.regions[i].ALB + self.regions[i].width, 10) # Avoid floating-point errors
        
        return
    
    def getMaterialProperties(self, x: float) -> tuple:
        """ 
        Gets material properties of some region 
        at location 'x' 
            
        Args:
            x - location within model
        Returns:
            sigma_s, sigma_t, source, scatt_ratio at location 'x' 
        """
        for r in self.regions:
            if r.ALB <= x <= r.ARB:
                
                # Get scattering ratio during rightward sweep
                try:
                    scatt_ratio = r.sigma_s/r.sigma
                except ZeroDivisionError: # i.e., vacuum region
                    scatt_ratio = 0
                
                # Sure, I could return here instead. But I prefer 
                # to return the other way instead. Keeps things readable.
                break
                    
        return r.sigma_s, r.sigma, r.source, scatt_ratio
    
    def _convertBC(self, BC: float) -> np.array:
        """ Converts non-array boundary condition (BC) to np.array. 
            Assumes all incident flux[m] == BC. """
        return np.array([BC]*self.mesh.angbins)
    
    def combineMesh(self) -> None:
        """ Combines each Region Mesh into
            one large Mesh object """
        
        # Checks
        for r in self.regions:
            if not r.mesh.angdegree == self.regions[0].mesh.angdegree:
                raise TransportError("Region Mesh objects must have same angle degree but have degrees: " + 
                                     str([r.mesh.angdegree for r in self.regions]))
        
        cellbins_tot = sum(r.mesh.cellbins for r in self.regions)
        cell_widths_tot = list(itertools.chain(*[r.mesh.cell_widths for r in self.regions])) # [1,2],[3],[4,5,6] -> [1,2,3,4,5,6]
        
        self.mesh = Mesh(cellbins_tot, cell_widths_tot, self.regions[0].mesh.angdegree)
        
        # Change BCs to match mesh angular shape if necessary
        if not isinstance(self.angfluxLBC, np.ndarray):
            self.angfluxLBC = self._convertBC(self.angfluxLBC)
        if not isinstance(self.angfluxRBC, np.ndarray):
            self.angfluxRBC = self._convertBC(self.angfluxRBC)
        
        return
    
    def _convertAng2Scal(self, angfluxes: np.array) -> np.array:
        """ Convert angular to scalar flux. Only accurate if
            angfluxes is increasing in angle from -1 to 1 for
            each spatial element """
        
        phis = []
        for i in range(len(angfluxes)):
            phi = sum(self.mesh.wgts[m]*angfluxes[i][m] for m in range(len(self.mesh.wgts)))
            phis.append(phi)
            
        return np.array(phis)
    
    def _convertEdge2Center(self, left_edge_array: iter, right_edge_array: iter) -> np.array:
        """ Convert  flux at edges to flux in center bins.
            This is the definition of Diamond Difference. """
        
        return np.array([1/2*(left_edge_array[i] + right_edge_array[i]) for i in range(len(left_edge_array))])
    
    def _convertAng2NetCurrent(self, rightward_ang_centers: np.array, leftward_ang_centers: np.array) -> np.array:
        """ Converts partial angular flux to net current.
            These are messy but honestly it's just funny
            to leave them as nested list comprehensions.
            
            This should probably be changed. However it
            is not vital at the moment. """
        
        mus, wgts = self.mesh.getPositiveOrdinates()
        
        partial_rightward = np.array([sum(mus[m]*wgts[m]*rightward_ang_centers[i][m] for m in range(self.mesh.angbins)) \
                                      for i in range(len(rightward_ang_centers))])
        partial_leftward = np.array([sum(-mus[m]*wgts[m]*leftward_ang_centers[i][m] for m in range(self.mesh.angbins)) \
                                     for i in range(len(leftward_ang_centers))])
        
        return partial_rightward + partial_leftward
    
    def _combineAngularFlux(self, leftward: np.array, rightward: np.array) -> np.array:
        """ Combine fluxes together. NOTE that the mu values for
            both arrays are increasing in MAGNITUDE, so to account
            for the negative mus in the leftward flux, the arrays
            are combined such that the leftward mus (columns) are
            flipped for each spatial element "i":
            
                 left    right    combined
              i1 [1 2] + [5 6] -> [2 1 5 6]
              i2 [3 4]   [7 8]    [4 3 7 8] """
              
        return np.hstack((np.fliplr(leftward),rightward))
    
    def _sourceIteration(self, phis_latest: np.array) -> np.array:
        """ Determines the new phis based on the old phis
            using source iteration. Unsurprisely redundant! """
        return phis_latest
    
    def _quasiDiffusion(self, angfluxes_leftward_edges: np.array, angfluxes_rightward_edges: np.array) -> np.array:
        """ Determines the new phis using Quasi-Diffusion (QD).
            See 'The Quasi-Diffusion Method for Solving Transport
            Problems in Planar and Spherical Geometry' by Miften
            and Larson. 
            
            Remember! 'rightward' means it swept starting from the LEFT
            and vice versa!
            """
        
        def Bfactor(angfluxes_i: np.array) -> float:
            """ Boundary Eddington factor. Angular fluxes should
                encapsulate both left and rightward directions
                (in that order) of a single spatial cell 'i'. """
            top = sum([abs(self.mesh.mus[m])*angfluxes_i[m]*self.mesh.wgts[m] for m in range(len(self.mesh.wgts))])
            bottom = sum([angfluxes_i[m]*self.mesh.wgts[m] for m in range(len(self.mesh.wgts))])
            return top/bottom
        
        def Efactor(angfluxes_i: np.array) -> float:
            """ Intermediary (normal) Eddington factor. Angular 
                fluxes should encapsulate both left and 
                rightward directions (in that order) of a single
                spatial cell 'i'."""
            top = sum([(self.mesh.mus[m]**2)*angfluxes_i[m]*self.mesh.wgts[m] for m in range(len(self.mesh.wgts))])
            bottom = sum([angfluxes_i[m]*self.mesh.wgts[m] for m in range(len(self.mesh.wgts))])
            return top/bottom
        
        # Initialize arrays
        A = np.zeros((len(angfluxes_leftward_edges),len(angfluxes_leftward_edges)))
        b = np.zeros(len(angfluxes_leftward_edges))
        angfluxes_combined = self._combineAngularFlux(angfluxes_leftward_edges, angfluxes_rightward_edges)
        mus, wgts = self.mesh.getPositiveOrdinates()
        
        # This is for visualization only...
        A_eddingtons = np.zeros((len(angfluxes_leftward_edges),len(angfluxes_leftward_edges)))
        
        # Setup matrix
        for i in range(len(angfluxes_leftward_edges)):
            
            # Leftmost boundary...
            if i == 0:
                
                sigma_s, sigma, source, _, = self.getMaterialProperties(self.mesh.xCenters[0])
                sigma_a = sigma - sigma_s
                dx = self.mesh.cell_widths[0]
                
                # "E_12" or E_32" represents "E_1/2" and "E_3/2"
                E_12 = Efactor(angfluxes_combined[0])
                E_32 = Efactor(angfluxes_combined[1])
                B_12 = Bfactor(angfluxes_combined[0])
                # E_12 = 1/3
                # E_32 = 1/3
                # B_12 = 1/2
                
                if self.regions[0].LBC == "reflecting":
                    A[0][0] = E_12/(sigma*dx) + sigma_a*dx/2 # phi_1/2
                    A[0][1] = -E_32/(sigma*dx) # phi_3/2
                    b[0] = source*dx/2
                    
                elif self.regions[0].LBC == "vacuum":
                    A[0][0] = B_12 + E_12/(sigma*dx) + sigma_a*dx/2 # phi_1/2
                    A[0][1] = -E_32/(sigma*dx) # phi_3/2
                    b[0] = source*dx/2 + 2*sum([mus[m]*angfluxes_rightward_edges[0][m]*wgts[m] for m in range((len(mus)))])
            
                A_eddingtons[0][0] = E_12
                A_eddingtons[0][1] = E_32
            
            # Rightmost boundary...
            elif i == len(angfluxes_leftward_edges)-1:
                
                sigma_s, sigma, source, _, = self.getMaterialProperties(self.mesh.xCenters[-1])
                sigma_a = sigma - sigma_s
                dx = self.mesh.cell_widths[-1]
                
                E_Jminhalf = Efactor(angfluxes_combined[-2])
                E_Jplushalf = Efactor(angfluxes_combined[-1])
                B_Jplushalf = Bfactor(angfluxes_combined[-1])
                # E_Jminhalf = 1/3
                # E_Jplushalf = 1/3
                # B_plushalf = 1/2
                
                if self.regions[-1].RBC == "reflecting":
                    A[-1][-2] = -E_Jminhalf/(sigma*dx) # phi_J-1/2
                    A[-1][-1] = E_Jplushalf/(sigma*dx) + sigma_a*dx/2 # phi_J+1/2
                    b[-1] = source*dx/2
                    
                elif self.regions[-1].RBC == "vacuum":
                    A[-1][-2] = -E_Jminhalf/(sigma*dx) # phi_J-1/2
                    A[-1][-1] = B_Jplushalf + E_Jplushalf/(sigma*dx) + sigma_a*dx/2 
                    b[-1] = source*dx/2 + 2*sum([mus[m]*angfluxes_leftward_edges[-1][m]*wgts[m] for m in range(len(mus))])
                
                A_eddingtons[-1][-2] = E_Jminhalf
                A_eddingtons[-1][-1] = E_Jplushalf
                
            # In the middle...
            else:
                
                # jth cell properties
                sigma_si, sigma_i, source_i, _, = self.getMaterialProperties(self.mesh.xCenters[i-1])
                sigma_ai = sigma_i - sigma_si
                dx_i = self.mesh.cell_widths[i-1]
                
                # jth + 1 cell properties
                sigma_siplus, sigma_iplus, source_iplus, _, = self.getMaterialProperties(self.mesh.xCenters[i])
                sigma_aiplus = sigma_iplus - sigma_siplus
                dx_iplus = self.mesh.cell_widths[i]
                
                E_Jminhalf = Efactor(angfluxes_combined[i-1])
                E_Jplushalf = Efactor(angfluxes_combined[i])
                E_Jplusthreehalf = Efactor(angfluxes_combined[i+1])
                # E_Jminhalf = 1/3
                # E_Jplushalf = 1/3
                # E_Jplusthreehalf = 1/3
                
                A[i][i-1] = -E_Jminhalf/(sigma_i*dx_i)
                A[i][i+1] = -E_Jplusthreehalf/(sigma_iplus*dx_iplus)
                A[i][i] = E_Jplushalf/(sigma_iplus*dx_iplus) + E_Jplushalf/(sigma_i*dx_i) + (sigma_ai*dx_i + sigma_aiplus*dx_iplus)/2
                b[i] = (source_i*dx_i + source_iplus*dx_iplus)/2
                
                A_eddingtons[i][i-1] = E_Jminhalf
                A_eddingtons[i][i+1] = E_Jplusthreehalf
                A_eddingtons[i][i] = E_Jplushalf
                
        print(f"{A=} \n{b=}")
        # print(f"{A_eddingtons = }")
        # print(f"{B_12 = }, {B_Jplushalf = }")
        
        # Solve the matrix
        phi_new_edges = np.linalg.solve(A,b)
        
        # Split the new phi edges into 'left' and 'right' edges, then approx. centers
        phi_new_centers = self._convertEdge2Center(phi_new_edges[:-1], phi_new_edges[1:])
        
        # Old centers, just for clerical sanity
        # phi_old_centers = self._convertAng2Scal(self.mesh.angfluxes) # defined in the current iteration
        # print(f"{phi_old_centers = }")
        print(f"{phi_new_centers = }")
                
        # Checks
        if np.any(phi_new_centers < 0):
            raise TransportError("Negative flux occurred during QD iteration")
        if np.any(np.isnan(phi_new_centers)):
            raise TransportError("NaN values appeared during QD iteration. Is there a vacuum region?")
        
        return phi_new_centers
    
    def doDiamondDiffV3(self, eps=1e-6, itermethod="SI", LOUD=False) -> tuple:
        """ 
        Args:
            eps - Allowable error between iterations
            itermethod - Iteration method used to determine
                         scalar flux for the next iteration
            LOUD - Enable/disable print statements for debugging
            
        Returns:
            Scalar fluxes, Angular fluxes, net current, scattering ratios, max optical thicknesses, iteration number
        """
        
        # Initialize arrays
        phis_old = self.mesh.scalarfluxes
        angfluxes_rightward_centers = np.zeros_like(self.mesh.angfluxes)
        angfluxes_leftward_centers = np.zeros_like(self.mesh.angfluxes)
        angfluxes_left_edge = np.zeros_like(self.mesh.angfluxes)
        angfluxes_right_edge = np.zeros_like(self.mesh.angfluxes) 
        self.scatt_ratios = np.zeros_like(self.mesh.scalarfluxes)
        self.maxOptThicknesses = np.zeros_like(self.mesh.scalarfluxes)
        
        mus, wgts = self.mesh.getPositiveOrdinates()
        
        # Prepare for convergence loop
        isConverged = False
        iterNum = 1
        max_diff_old = np.inf
        
        # Start convergence loop...
        while not isConverged:    
            
            if iterNum == 10_000:
                raise TransportError(f"Could not converge in {iterNum} iterations")
            
            # Set left boundary flux
            if self.regions[0].LBC == "reflecting":
                angfluxes_left_edge[0] = angfluxes_leftward_centers[0]#[::-1]
            else:
                angfluxes_left_edge[0] = self.angfluxLBC
                
            if LOUD: print(f"Initial left psi_inc = {angfluxes_left_edge[0]}")
                
            # Sweep from left --> right
            for i in range(len(angfluxes_rightward_centers)):
                
                # Region properties
                sigma_si, sigma_i, source_i, scatt_ratio = self.getMaterialProperties(self.mesh.xCenters[i])
                        
                # Organize cell properties
                dx = self.mesh.cell_widths[i]
                self.scatt_ratios[i] = scatt_ratio
                self.maxOptThicknesses[i] = sigma_i*dx/min(mus)
                
                # Iterate over angle
                for m in range(self.mesh.angbins):
                    angfluxes_right_edge[i][m] = (source_i/2 + 
                                                  sigma_si*phis_old[i]/2 + 
                                                  mus[m]/dx*angfluxes_left_edge[i][m] -
                                                  sigma_i/2*angfluxes_left_edge[i][m]) / \
                                                  (mus[m]/dx + sigma_i/2)
                                                  
                    # "set-to-zero" any negative fluxes
                    if angfluxes_right_edge[i][m] < 0:
                        angfluxes_right_edge[i][m] = 0
                    
                    # Set edges
                    if not i+1 == len(self.mesh.angfluxes):
                        angfluxes_left_edge[i+1][m] = angfluxes_right_edge[i][m]
                        
                if LOUD: print(f"angfluxes_left_edge[{i}] = {angfluxes_left_edge[i]}")
            
            # Convert left and right edges into separate arrays of centers and edges (in case you want to use either)
            angfluxes_rightward_edges = np.append(angfluxes_left_edge, [angfluxes_right_edge[-1]], axis=0) # [[1,2],[3,4]] + [5,6] -> [[1,2],[3,4],[5,6]]
            angfluxes_rightward_centers = self._convertEdge2Center(angfluxes_left_edge, angfluxes_right_edge)
            
            if LOUD: 
                print("left edges --- right edges (rightward)")
                for l,r in zip(angfluxes_left_edge, angfluxes_right_edge):
                    print(" ",l," ", r)
            
            # Clear edge fluxes
            angfluxes_left_edge = np.zeros_like(self.mesh.angfluxes)
            angfluxes_right_edge = np.zeros_like(self.mesh.angfluxes)
            
            # Set right boundary flux
            if self.regions[-1].RBC == "reflecting":
                angfluxes_right_edge[-1] = angfluxes_rightward_centers[-1]#[::-1]
            else:
                angfluxes_right_edge[-1] = self.angfluxRBC
                
            if LOUD: print(f"Initial right psi_inc = {angfluxes_right_edge[-1]}")
            
            # Sweep from right --> left
            for i in range(len(angfluxes_leftward_centers)-1, -1 ,-1):
                
                # Region properties
                sigma_si, sigma_i, source_i, scatt_ratio = self.getMaterialProperties(self.mesh.xCenters[i])
                        
                # Organize cell properties
                dx = self.mesh.cell_widths[i]
                
                # Iterate over angle
                for m in range(self.mesh.angbins):
                    angfluxes_left_edge[i][m] = (angfluxes_right_edge[i][m]*(-mus[m]/dx + sigma_i/2) -
                                                 source_i/2 - sigma_si*phis_old[i]/2) / (-mus[m]/dx - sigma_i/2)
                    
                    # "set-to-zero" any negative fluxes
                    if angfluxes_left_edge[i][m] < 0:
                        angfluxes_left_edge[i][m] = 0
                    
                    # Set edges
                    if i != 0 != len(self.mesh.angfluxes):
                        angfluxes_right_edge[i-1][m] = angfluxes_left_edge[i][m]
                
                if LOUD: print(f"angfluxes_right_edge[{i}] = {angfluxes_right_edge[i]}")
                
            if LOUD: 
                print("left edges --- right edges (leftward)")
                for l,r in zip(angfluxes_left_edge, angfluxes_right_edge):
                    print(" ",l," ", r)
            
            # Convert left and right edges into separate arrays of centers and edges
            angfluxes_leftward_edges = np.append(angfluxes_left_edge, [angfluxes_right_edge[-1]], axis=0) # [[1,2],[3,4]] + [5,6] -> [[1,2],[3,4],[5,6]]
            angfluxes_leftward_centers = self._convertEdge2Center(angfluxes_left_edge, angfluxes_right_edge)
            
            # Combine fluxes together. NOTE that the mu values for
            # both arrays are increasing in MAGNITUDE, so to account
            # for the negative mus in the leftward flux, the arrays
            # are combined such that the leftward mus (columns) are
            # flipped for each spatial element "i":
            #
            #     left    right    combined
            #  i1 [1 2] + [5 6] -> [2 1 5 6]
            #  i2 [3 4]   [7 8]    [4 3 7 8]
            
            angfluxes_centers_combined = self._combineAngularFlux(angfluxes_leftward_centers, angfluxes_rightward_centers)
            angfluxes_edges_combined = self._combineAngularFlux(angfluxes_leftward_edges, angfluxes_rightward_edges)
            
            # Convert to scalar flux
            phis_new = self._convertAng2Scal(angfluxes_centers_combined)
            phis_new_edges = self._convertAng2Scal(angfluxes_edges_combined)
            
            # Check convergence. If converged, sigh in relief. If not, cope.
            max_diff = np.max(np.abs(phis_new - phis_old))
            # if max_diff in max_diffs_seen:
            #     # raise TransportError("Cyclical fluxes encountered.")
            #     self.mesh.scalarfluxes = phis_new
            #     self.mesh.netCurrent = np.round(self._convertAng2NetCurrent(angfluxes_rightward_centers, angfluxes_leftward_centers), decimals=5)
            #     print("Duplicate fluxes encountered.")
            #     print(f"Finished in {iterNum} iterations.")
            #     break
            # else:
            #     max_diffs_seen[max_diff] = True
            if max_diff >= max_diff_old:
                print("Warning. Maximum difference remained stationary or increased between iterations.")
                print(f"{max_diff = }")
                self.mesh.scalarfluxes = phis_new
                self.mesh.netCurrent = np.round(self._convertAng2NetCurrent(angfluxes_rightward_centers, angfluxes_leftward_centers), decimals=5)
                print(f"Finished in {iterNum} iterations.")
                break
            else:
                max_diff_old = max_diff
            
            if max_diff < eps:
                
                # Converged!
                isConverged = True
                # self.mesh.scalarfluxes = np.round(phis_new ,decimals=5) # Avoid graphical errors
                self.mesh.scalarfluxes = phis_new
                self.mesh.netCurrent = np.round(self._convertAng2NetCurrent(angfluxes_rightward_centers, angfluxes_leftward_centers), decimals=5)
                print(f"Finished in {iterNum} iterations. Max difference = {max_diff}")
                break
                
            else: # Not converged...
                
                print(f"======= ITERATION {iterNum} ========")
                # print(f"{phis_new = }")
                # print(f"{phis_old = }")
                # print(f"{np.abs(phis_new - phis_old) = }")
            
                if itermethod == "SI":
                    phis_old = self._sourceIteration(phis_new)
                    
                elif itermethod == "QD":
                    phis_old = self._quasiDiffusion(angfluxes_leftward_edges, angfluxes_rightward_edges)
                    
                else:
                    raise TransportError(f"Unknown iteration method '{itermethod}'")
                
                iterNum +=1
                
        return self.mesh.scalarfluxes, self.mesh.angfluxes, self.mesh.netCurrent, self.scatt_ratios, self.maxOptThicknesses, iterNum
            
    
    def solution_UIM(self) -> np.array:
        """ Calculates uniform infinite medium solution """
        
        self.analytical_solution = []
        
        for _ in self.mesh.angfluxes:
            
            # Region properties
            sigma_si = self.regions[0].sigma_s
            sigma_i  = self.regions[0].sigma
            source_i = self.regions[0].source
                    
            self.analytical_solution.append(source_i/(sigma_i - sigma_si))
        
        return np.array(self.analytical_solution)
    
    def solution_SFPA(self) -> np.array:
        """ Calculates the source-free pure absorber solution """
        
        self.analytical_solution = np.zeros_like(self.mesh.angfluxes)
        mus, wgts = self.mesh.getPositiveOrdinates()
        
        # Create first solution that is only dx/2 away from left border
        self.analytical_solution[0] = self.angfluxLBC * [np.exp(-self.regions[0].sigma*self.mesh.cell_widths[0]/(2*abs(mus[m]))) \
                                                          for m in range(self.mesh.angbins)]
        
        for i in range(1, len(self.analytical_solution)):
            
            # Region properties
            sigma_si, sigma_i, source_i, scatt_ratio = self.getMaterialProperties(self.mesh.xCenters[i])
            dx = self.mesh.cell_widths[i]
            
            # Get analytical angular fluxes at boundaries
            for m in range(self.mesh.angbins):
                optThickness = sigma_i*dx/abs(mus[m])
                self.analytical_solution[i][m] = self.analytical_solution[i-1][m]*np.exp(-optThickness)
            
        # Convert angular flux to scalar flux
        self.analytical_solution = self._convertAng2Scal(self.analytical_solution)
            
        return self.analytical_solution
    
    def plotOptics(self) -> None:
        """ Plots max optical thickness in each cell of the model """
        
        fig, ax1 = plt.subplots()
        
        for region in self.regions:
            
            # Create vertical lines based on region width and BC
            if region.LBC == "reflecting":
                LBC_linestyle = "--"
            else:
                LBC_linestyle = "-"
            
            if region.RBC == "reflecting":
                RBC_linestyle = "--"
            else:
                RBC_linestyle = "-"
            
            ax1.axvline(x=region.ALB, color="black", linestyle=LBC_linestyle)
            ax1.axvline(x=region.ARB, color="black", linestyle=RBC_linestyle)
            
            # Color regions by type
            colormap = plt.get_cmap("gist_rainbow")
            color_tot = (region.sigma_s+region.sigma+region.source)
            if region.source + region.sigma == 0: # Void
                color = "white"
            elif region.sigma_s >= region.sigma: # Scattering dominant
                color = colormap(region.sigma_s/color_tot)
            elif region.source >= region.sigma: # Source dominant
                color = colormap(region.source/color_tot)
            elif region.sigma > region.sigma_s: # Absorber dominant
                color = colormap(region.sigma/color_tot)
            ax1.axvspan(region.ALB, region.ARB, color=color, alpha=0.25)
            
        # plot "grid lines" for each cell boundary in mesh
        for x in self.mesh.xEdgelocs_right:
            ax1.axvline(x=x, color="black", linestyle="dotted", alpha=0.1)
        
        plt.plot(self.mesh.xCenters, self.maxOptThicknesses)
        plt.xlabel("x (cm)")
        plt.ylabel("Max Optical Thickness")
        
        plt.show()
        
        return
    
    def plotModel(self, show_AnalySol=False, sol_x=[], sol_y=[]) -> None:
        """ Plots Regions, Numerical, and (optional) analytical solutions """
        
        fig, ax1 = plt.subplots()
        plt.grid(False)
        
        for region in self.regions:
            
            # Create vertical lines based on region width and BC
            if region.LBC == "reflecting":
                LBC_linestyle = "--"
            else:
                LBC_linestyle = "-"
            
            if region.RBC == "reflecting":
                RBC_linestyle = "--"
            else:
                RBC_linestyle = "-"
            
            ax1.axvline(x=region.ALB, color="black", linestyle=LBC_linestyle)
            ax1.axvline(x=region.ARB, color="black", linestyle=RBC_linestyle)
            
            # Color regions by type
            colormap = plt.get_cmap("gist_rainbow")
            color_tot = (region.sigma_s+region.sigma+region.source)
            if region.source + region.sigma == 0: # Void
                color = "white"
            elif region.sigma_s >= region.sigma: # Scattering dominant
                color = colormap(region.sigma_s/color_tot)
            elif region.source >= region.sigma: # Source dominant
                color = colormap(region.source/color_tot)
            elif region.sigma > region.sigma_s: # Absorber dominant
                color = colormap(region.sigma/color_tot)
            ax1.axvspan(region.ALB, region.ARB, color=color, alpha=0.25)
            
        line1, = ax1.plot(self.mesh.xCenters, list(np.round(self.mesh.scalarfluxes, decimals=5)), label="Scalar Flux")
        
        ax1.set_ylabel("ϕ (particle per $cm^2$-sec)")
        ax1.set_xlabel("x (cm)")
        
        # plot "grid lines" for each cell boundary in mesh
        for x in self.mesh.xEdgelocs_right:
            ax1.axvline(x=x, color="black", linestyle="dotted", alpha=0.1)
        
        # Insert new axis and plot net current if no analytical solution
        if not show_AnalySol:
            ax2 = ax1.twinx()
            
            line2, = ax2.plot(self.mesh.xCenters, self.mesh.netCurrent, color="red", label="Net Current")
            ax2.set_ylabel("J (particle per $cm^2$-sec)")
            
            lines = [line1, line2]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels)
            plt.show()
            
            return
        
        # Analytical solution is calculated automatically, no current shown
        elif show_AnalySol and sol_x == sol_y == []:
            try:
                ax1.plot(self.mesh.xCenters, self.analytical_solution, label="Exact Solution")
            except ValueError: # You forgot to define self.analytical_solution...
                raise TransportError("Must either call analytical solution method or explicitly define sol_x and sol_y")
            
            ax1.legend()
            plt.show()
        
        # Analytical solution is provided explicitly, no current shown
        elif show_AnalySol and sol_x != sol_y != []:
            ax1.plot(sol_x, sol_y, label="Exact Solution")
            ax1.legend()
            plt.show()
        
        return
    
###############################################

class TransportError(Exception):
    """ Class for catching transport errors """

############## Test Cases ##############

### Uniform infinite medium, source iteration
def test1():
    
    r_width = 1
    num_regions = 1 # Don't change
    num_cells = 10
    num_mus = 4
    
    sigma, sigma_s, source = 10, 0, 10

    medium1 = Region(r_width,sigma,sigma_s,source,"inf_medium","reflecting","reflecting")
    medium1.applyMesh(num_cells, num_regions/num_cells*r_width, num_mus)
    
    infModel1 = Model(medium1, angfluxLBC = 0, angfluxRBC = 0)
    infModel1.combineMesh()
    infModel1.doDiamondDiffV3(itermethod = "SI", LOUD=False)
    
    infModel1.plotModel(show_AnalySol=False)
    infModel1.plotOptics()
    
    return

### Uniform infinite medium, Quasi-diffusion
def test2():
    
    r_width = 1
    num_regions = 1 # Don't change
    num_cells = 3
    num_mus = 4
    
    sigma, sigma_s, source = 10, 0, 10

    medium1 = Region(r_width,sigma,sigma_s,source,"inf_medium","reflecting","reflecting")
    medium1.applyMesh(num_cells, num_regions/num_cells*r_width, num_mus)
    
    infModel1 = Model(medium1, angfluxLBC = 0, angfluxRBC = 0)
    infModel1.combineMesh()
    infModel1.doDiamondDiffV3(itermethod = "QD", LOUD=False)
    
    infModel1.plotModel(show_AnalySol=False)
    infModel1.plotOptics()
    
    return

def main():
    
    test1()
    print("Test 1 (UIM w/ SI) success!")
    
    test2()
    print("Test 2 (UIM w/ QD) success!")

if __name__ == "__main__":
    main()

