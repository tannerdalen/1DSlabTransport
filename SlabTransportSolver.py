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
from collections import Counter
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
        
        self.xEdgelocs = np.append(self.xEdgelocs_left, [self.xEdgelocs_right[-1]], axis=0)
        
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
            sigma_s, sigma_t, source at location 'x' 
        """
        for r in self.regions:
            if r.ALB <= x <= r.ARB:  
                return r.sigma_s, r.sigma, r.source
    
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
            try:
                phi = sum(self.mesh.wgts[m]*angfluxes[i][m] for m in range(len(self.mesh.wgts)))
                phis.append(phi)
            except IndexError:
                raise TransportError("Too few angular flux directions. Did you combine directions?")
            
        return np.array(phis)
    
    def _convertEdge2Center(self, left_edge_array: iter, right_edge_array: iter) -> np.array:
        """ Convert flux at edges to flux in center bins.
            This is the definition of Diamond Difference. """
        
        return np.array([1/2*(left_edge_array[i] + right_edge_array[i]) for i in range(len(left_edge_array))])
    
    def _convertAng2NetCurrent(self, angfluxes_combined_edges: np.array) -> np.array:
        """ Converts angular flux to net current. """
        
        return np.array([sum(self.mesh.mus[m]*self.mesh.wgts[m]*angfluxes_combined_edges[i][m]
                         for m in range(len(self.mesh.mus))) \
                         for i in range(len(angfluxes_combined_edges))])
    
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
    
    def _quasiDiffusion(self, angfluxes_combined_edges: np.array) -> np.array:
        """ Determines the new phis using Quasi-Diffusion (QD).
            See 'The Quasi-Diffusion Method for Solving Transport
            Problems in Planar and Spherical Geometry' by Miften
            and Larson.
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
        A = np.zeros((len(angfluxes_combined_edges),len(angfluxes_combined_edges)))
        b = np.zeros(len(angfluxes_combined_edges))
        
        # This is for visualization only...
        A_eddingtons = np.zeros([len(angfluxes_combined_edges),3])
        
        ### Setup matrix...
        
        ### Leftmost boundary...    
        sigma_s, sigma, source = self.getMaterialProperties(self.mesh.xCenters[0])
        sigma_a = sigma - sigma_s
        dx = self.mesh.cell_widths[0]
        
        # "E_12" or E_32" represents "E_1/2" and "E_3/2"
        E_12 = Efactor(angfluxes_combined_edges[0])
        E_32 = Efactor(angfluxes_combined_edges[1])
        B_12 = Bfactor(angfluxes_combined_edges[0])
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
            b[0] = source*dx/2 + 2*sum([self.mesh.mus[m]*angfluxes_combined_edges[0][m]*self.mesh.wgts[m] \
                                        for m in range((len(self.mesh.mus))) if self.mesh.mus[m] > 0])
    
        A_eddingtons[0][0] = E_12
        A_eddingtons[0][1] = E_32
        A_eddingtons[0][2] = 0
        
        ### Rightmost boundary...   
        sigma_s, sigma, source = self.getMaterialProperties(self.mesh.xCenters[-1])
        sigma_a = sigma - sigma_s
        dx = self.mesh.cell_widths[-1]
        
        E_Jminhalf = Efactor(angfluxes_combined_edges[-2])
        E_Jplushalf = Efactor(angfluxes_combined_edges[-1])
        B_Jplushalf = Bfactor(angfluxes_combined_edges[-1])
        # E_Jminhalf = 1/3
        # E_Jplushalf = 1/3
        # B_Jplushalf = 1/2
        
        if self.regions[-1].RBC == "reflecting":
            A[-1][-2] = -E_Jminhalf/(sigma*dx) # phi_J-1/2
            A[-1][-1] = E_Jplushalf/(sigma*dx) + sigma_a*dx/2 # phi_J+1/2
            b[-1] = source*dx/2
            
        elif self.regions[-1].RBC == "vacuum":
            A[-1][-2] = -E_Jminhalf/(sigma*dx) # phi_J-1/2
            A[-1][-1] = B_Jplushalf + E_Jplushalf/(sigma*dx) + sigma_a*dx/2 
            b[-1] = source*dx/2 + 2*sum([abs(self.mesh.mus[m])*angfluxes_combined_edges[-1][m]*self.mesh.wgts[m] \
                                         for m in range(len(self.mesh.mus)) if self.mesh.mus[m] < 0])
        
        A_eddingtons[-1][-3] = 0
        A_eddingtons[-1][-2] = E_Jminhalf
        A_eddingtons[-1][-1] = E_Jplushalf
        
        ### Everything else in between...
        for i in range(1,len(angfluxes_combined_edges)-1):
            
            # jth cell properties
            sigma_si, sigma_i, source_i = self.getMaterialProperties(self.mesh.xCenters[i-1])
            sigma_ai = sigma_i - sigma_si
            dx_i = self.mesh.cell_widths[i-1]
            
            # jth + 1 cell properties
            sigma_siplus, sigma_iplus, source_iplus = self.getMaterialProperties(self.mesh.xCenters[i])
            sigma_aiplus = sigma_iplus - sigma_siplus
            dx_iplus = self.mesh.cell_widths[i]
            
            E_Jminhalf = Efactor(angfluxes_combined_edges[i-1])
            E_Jplushalf = Efactor(angfluxes_combined_edges[i])
            E_Jplusthreehalf = Efactor(angfluxes_combined_edges[i+1])
            # E_Jminhalf = 1/3
            # E_Jplushalf = 1/3
            # E_Jplusthreehalf = 1/3
            
            A[i][i-1] = -E_Jminhalf/(sigma_i*dx_i)
            A[i][i+1] = -E_Jplusthreehalf/(sigma_iplus*dx_iplus)
            A[i][i] = E_Jplushalf/(sigma_iplus*dx_iplus) + E_Jplushalf/(sigma_i*dx_i) + (sigma_ai*dx_i + sigma_aiplus*dx_iplus)/2
            b[i] = (source_i*dx_i + source_iplus*dx_iplus)/2
            
            A_eddingtons[i][-1] = E_Jminhalf
            A_eddingtons[i][1] = E_Jplusthreehalf
            A_eddingtons[i][0] = E_Jplushalf
                
        # print(f"{A=} \n{b=}")
        # print(f"{A_eddingtons = }")
        # print(f"{B_12 = }, {B_Jplushalf = }")
        
        # Solve the matrix
        phi_new_edges = np.linalg.solve(A,b)
                
        # Checks
        if np.any(phi_new_edges < 0):
            raise TransportError("Negative flux occurred during QD iteration")
        if np.any(np.isnan(phi_new_edges)):
            raise TransportError("NaN values appeared during QD iteration. Is there a vacuum region?")
        
        return phi_new_edges
    
    def doDiamondDiffV4(self, eps=1e-6, itermax=10_000, itermethod="SI", LOUD=False) -> np.array:
        """ 
        Args:
            eps - Allowable error between iterations
            itermax - maximum allowable iterations 
            itermethod - Iteration method used to determine
                         scalar flux for the next iteration
            LOUD - Enable/disable print statements for debugging
            
        Returns:
            scalar fluxes at the edges
        """
        
        # Checks
        if not itermethod in ITERATION_METHODS:
            raise TransportError(f"Unknown itermethod '{itermethod}'")
        
        # Initialize arrays
        phi_old_edges = np.zeros_like(self.mesh.scalarfluxes_edges)
        phi_new_edges = np.zeros_like(self.mesh.scalarfluxes_edges)
        angfluxes_leftward_edges = np.zeros_like(self.mesh.angfluxes_edges)
        angfluxes_rightward_edges = np.zeros_like(self.mesh.angfluxes_edges)
        
        # Grab only the positive mus
        mus, _ = self.mesh.getPositiveOrdinates()
        
        # Set convergence properties
        iterNum = 1
        max_diff = np.inf
        max_diffs = []
        all_diffs = []
        isConverged = False
        
        # Begin loop
        while not isConverged:
            
            print(f"Iteration {iterNum}...")
            
            if iterNum+1 == itermax:
                
                plt.plot([x for x in range(2,iterNum)], max_diffs[1:])
                plt.xlabel("Iteration Number")
                plt.ylabel("Max Difference between Iterations")
                plt.yscale("log")
                plt.show()
                
                plt.plot(self.mesh.xEdgelocs, all_diffs[-1])
                plt.xlabel("x (cm)")
                plt.ylabel("Differences at last Iteration")
                plt.yscale("log")
                plt.show()
                
                self.scalarfluxes_edges = phi_new_edges
                self.plotModel()
                
                raise TransportError(f"Could not converge within {itermax} iterations. "+
                                     f"Last maximum difference = {max_diff}. "+
                                     "Attempted to plot latest scalar flux...")
                
            ### Sweep left --> right, starting with boundary conditions...
            if self.regions[0].LBC == "reflecting":
                angfluxes_rightward_edges[0] = angfluxes_leftward_edges[0]
            if self.regions[0].LBC == "vacuum":
                angfluxes_rightward_edges[0] = self.angfluxLBC
                
            ### Begin the rightward sweep...
            for i in range(1,len(angfluxes_rightward_edges)):
                
                # Get model properties
                sigma_s, sigma, source = self.getMaterialProperties(self.mesh.xCenters[i-1])
                dx = self.mesh.cell_widths[i-1]
                
                for m in range(len(mus)):
                    
                    angfluxes_rightward_edges[i][m] = (sigma_s/4*(phi_old_edges[i]+phi_old_edges[i-1]) + 
                                                       source/2 - 
                                                       angfluxes_rightward_edges[i-1][m]*(-mus[m]/dx + sigma/2)) / \
                                                       (mus[m]/dx + sigma/2)
                                                       
            ### Sweep right --> left, starting with boundary conditions...
            if self.regions[-1].RBC == "reflecting":
                angfluxes_leftward_edges[-1] = angfluxes_rightward_edges[-1]
            if self.regions[-1].RBC == "vacuum":
                angfluxes_leftward_edges[-1] = self.angfluxRBC
                
            ### Begin the leftward sweep...
            for i in range(len(angfluxes_leftward_edges) - 2, -1, -1):
                
                # Get model properties
                sigma_s, sigma, source = self.getMaterialProperties(self.mesh.xCenters[i])
                dx = self.mesh.cell_widths[i]
                
                for m in range(len(mus)):
                    
                    angfluxes_leftward_edges[i][m] = (sigma_s/4*(phi_old_edges[i]+phi_old_edges[i+1]) + 
                                                      source/2 -
                                                      angfluxes_leftward_edges[i+1][m]*(-mus[m]/dx + sigma/2)) / \
                                                      (mus[m]/dx + sigma/2)
                                                      
            # Combine fluxes together. NOTE that the mu values for
            # both arrays are increasing in MAGNITUDE, so to account
            # for the negative mus in the leftward flux, the arrays
            # are combined such that the leftward mus (columns) are
            # flipped for each spatial element "i":
            #
            #     left    right    combined
            #  i1 [1 2] + [5 6] -> [2 1 5 6]
            #  i2 [3 4]   [7 8]    [4 3 7 8]
            angfluxes_combined_edges = self._combineAngularFlux(angfluxes_leftward_edges, angfluxes_rightward_edges)
            
            # Calculate scalar flux for the next iteration
            if itermethod == "SI":
                phi_new_edges = self._sourceIteration(self._convertAng2Scal(angfluxes_combined_edges))
            elif itermethod == "QD":
                phi_new_edges = self._quasiDiffusion(angfluxes_combined_edges)
            
            # Check if maximum difference between iterations is within
            # convergence criteria
            diff = np.abs(phi_new_edges - phi_old_edges)
            all_diffs.append(diff)
            max_diff = np.max(diff)
            max_diffs.append(max_diff)
            
            if max_diff <= eps:
                
                isConverged = True
                print(f"-=- Converged in {iterNum} iterations -=-")
                
                self.scalarfluxes_edges = phi_new_edges
                self.netCurrent_edges = self._convertAng2NetCurrent(angfluxes_combined_edges)
            
            else:
                
                iterNum +=1
                if itermethod == "SI":
                    phi_old_edges = self._sourceIteration(phi_new_edges)
                elif itermethod == "QD":
                    phi_old_edges = self._quasiDiffusion(angfluxes_combined_edges)
                    
        return phi_new_edges
    
    def solution_UIM(self) -> np.array:
        """ Calculates uniform infinite medium solution 
            for all edges """
        
        r = self.regions[0]
        self.analytical_solution = np.array([r.source/(r.sigma-r.sigma_s)]*len(self.mesh.cell_widths))
        return self.analytical_solution
    
    def solution_SFPA(self) -> np.array:
        """ Calculates the source-free pure absorber
            soolution for all edges """
        
        self.analytical_solution = np.zeros_like(self.mesh.angfluxes_edges)
        
        # Set leftmost incident flux
        self.analytical_solution[0] = np.append(np.zeros_like(self.angfluxLBC), self.angfluxLBC, axis=0)
        
        for i in range(1,len(self.mesh.angfluxes_edges)):
            
            _, sigma, _ = self.getMaterialProperties(self.mesh.xCenters[i-1])
            dx = self.mesh.cell_widths[i-1]
            
            for m in range(len(self.mesh.mus)):
                self.analytical_solution[i][m] = self.analytical_solution[i-1][m]*np.exp(sigma*dx/abs(self.mesh.mus[m]))
            
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
        
        # Get the max optical thicknesses of each cell
        maxOptThicknesses = np.zeros_like(self.mesh.xCenters)
        for i, x in enumerate(self.mesh.xCenters):
            _, sigma, _ = self.getMaterialProperties(x)
            dx = self.mesh.cell_widths[i]
            maxOptThicknesses[i] = sigma*dx/min(np.abs(self.mesh.mus))
        
        plt.plot(self.mesh.xCenters, maxOptThicknesses)
        plt.xlabel("x (cm)")
        plt.ylabel("Max Optical Thickness")
        
        plt.show()
        
        return
    
    def plotModel(self, show_NumericalSol=True, show_current=False, show_AnalySol=False, sol_x=[], sol_y=[]) -> None:
        """ Plots Regions, Numerical, and (optional) analytical solutions """
        
        fig, ax1 = plt.subplots()
        lines = []
        
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
            
        # Plot numerical solution
        if show_NumericalSol:
            line, = ax1.plot(self.mesh.xEdgelocs, np.round(self.scalarfluxes_edges, decimals=5), label="Scalar Flux (Numerical)")
            ax1.set_ylabel("ϕ (particle per $cm^2$-sec)")
            ax1.set_xlabel("x (cm)")
            lines.append(line)
        
        # Plot computed analytical solution
        if show_AnalySol and sol_x == sol_y == []:
            try:
                line, = ax1.plot(self.mesh.xCenters, self.analytical_solution, label="Scalar Flux (Analytical)")
                lines.append(line)
            except AttributeError: # You forgot to define self.analytical_solution...
                raise TransportError("Must either call analytical solution method or explicitly define sol_x and sol_y")
        
        # Plot provided analytical solution
        elif show_AnalySol and sol_x != sol_y != []:
            line, = ax1.plot(sol_x, sol_y, label="Scalar Flux (Analytical)")
            lines.append(line)
        
        # Plot current on new axis
        if show_current:
            ax2 = ax1.twinx()
            
            line2, = ax2.plot(self.mesh.xEdgelocs, np.round(self.netCurrent_edges, decimals=5), color="red", label="Net Current")
            ax2.set_ylabel("J (particle per $cm^2$-sec)")
            lines.append(line2)
        
        # Show plot
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels)
        
        return
    
###############################################

class TransportError(Exception):
    """ Class for catching transport errors """

################ Test Cases ###################

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
    infModel1.doDiamondDiffV4(itermethod="SI", LOUD=False)
    infModel1.solution_UIM()
    infModel1.plotModel(show_AnalySol=True, show_current=True)
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
    infModel1.doDiamondDiffV4(itermethod="QD", LOUD=False)
    
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

