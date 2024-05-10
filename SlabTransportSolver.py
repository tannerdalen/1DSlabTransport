### Tanner Heatherly
### NSE 654 - Computational Particle Transport
### May 10th, 2024
### Homework #3
 #
### Descritization method: Diamond Difference
 #
### Code requirements:
###     - Heterogenetic cross sections (sigma, sigma_s)
###     - Customizable boundary conditions (iso/anisotropic angular flux, reflecting)
###     - Allow for unique mesh cell sizes
 #
### There are several deliverables for this assignment, including:
###     - Demonstrate correct solution for Uniform Infinite Medium
###     - Demonstrate correct solution for Source-free pure absorber
###     - Demonstrate correct solution for Source-free half-space
###     - Record Source Iteration rate of convergence as a function
###       of max scattering ratio, max optical thickness, and BCs.
###     - Solve custom Problem #1
###     - Solve Reed's Problem
###
###           Prepare SCALAR FLUX and CURRENT for all problems!
###

from dataclasses import dataclass, field
from typing import Any
import itertools
import numpy as np
import matplotlib.pyplot as plt

BOUNDARY_CONDITIONS = ["vacuum", "reflecting", None]

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
        # For example if self.cells = array([[1,2,3], then self.angflux[1,2] == 6 
        #                                    [4,5,6],
        #                                    [7,8,9]) 
        
        self.angfluxes = np.zeros([self.cellbins, self.angbins])
        self.scalarfluxes = np.zeros([self.cellbins])
        self.netCurrent = np.zeros([self.cellbins])
        self.analyticalsol = np.zeros([self.cellbins])
        
        # Create separate arrays for left edge and right edge of a given cell
        
        self.angfluxes_left_edge = np.zeros([self.cellbins, self.angbins])
        self.angfluxes_right_edge = np.zeros([self.cellbins, self.angbins])
        
        # Create mus and wgts for desired degree
        # Note: only taking the positive values!
        self.mus, self.wgts = np.polynomial.legendre.leggauss(angdegree)
        self.mus = np.array([mu for mu in self.mus if mu>0])
        self.wgts = self.wgts[0:len(self.mus)]
        
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
    
    def _convertBC(self, BC, angbins) -> np.array:
        """ Converts non-array boundary condition (BC) to np.array. 
            Assumes all incident flux[m] == BC. """
        return np.array([BC]*angbins)
    
    def combineMesh(self) -> None:
        """ Combines each Region Mesh into
            one large Mesh object """
        
        # Checks
        for r in self.regions:
            if not r.mesh.angdegree == self.regions[0].mesh.angdegree:
                raise TransportError("Region Mesh objects must have same angle degree but have degrees: " + 
                                     str([r.mesh.angdegree for r in self.regions]))
        
        cellbins_tot = sum(r.mesh.cellbins for r in self.regions)
        cell_widths_tot = list(itertools.chain(*[r.mesh.cell_widths for r in self.regions]))
        
        self.mesh = Mesh(cellbins_tot, cell_widths_tot, self.regions[0].mesh.angdegree)
        
        # Change BCs to match mesh angular shape if necessary
        if not isinstance(self.angfluxLBC, np.ndarray):
            self.angfluxLBC = self._convertBC(self.angfluxLBC, self.mesh.angbins)
        if not isinstance(self.angfluxRBC, np.ndarray):
            self.angfluxRBC = self._convertBC(self.angfluxRBC, self.mesh.angbins)
        
        return
    
    def _convertAng2Scal(self, ang_centers_array: np.array) -> np.array:
        """ Convert angular to scalar flux """
        
        phis = []
        for i in range(len(ang_centers_array)):
            phi = sum(self.mesh.wgts[m]*ang_centers_array[i][m] for m in range(self.mesh.angbins))
            phis.append(phi)
            
        return np.array(phis)
    
    def _convertEdge2Center(self, left_edge_array: iter, right_edge_array: iter) -> np.array:
        """ Convert angular flux at edges to angular flux in center bins.
            This is the definition of Diamond Difference. """
        
        return np.array([1/2*(left_edge_array[i] + right_edge_array[i]) for i in range(len(self.mesh.angfluxes))])
    
    def _convertAng2NetCurrent(self, rightward_ang_centers: np.array, leftward_ang_centers: np.array) -> np.array:
        """ Converts partial angular flux to net current.
            These are messy but honestly it's just funny
            to leave them as nested list comprehensions."""
        
        partial_rightward = np.array([sum(self.mesh.mus[m]*self.mesh.wgts[m]*rightward_ang_centers[i][m] for m in range(self.mesh.angbins)) \
                                 for i in range(len(rightward_ang_centers))])
        partial_leftward = np.array([sum(-self.mesh.mus[m]*self.mesh.wgts[m]*leftward_ang_centers[i][m] for m in range(self.mesh.angbins)) \
                                 for i in range(len(leftward_ang_centers))])
        
        return partial_rightward + partial_leftward
    
    def doDiamondDiffV3(self, eps=1e-9, LOUD=False) -> tuple():
        """ 9:03pm, May 9th. I am not happy with my code, even
            after two weeks of working on it.
            
            5:42am, May 10th. I have done my best. May
            judgement be swift.
            
            1:05pm, May 10th. I found the bug. The code
            works. My spirits are high with hopes my grade
            is higher!
            
            Args:
                eps - Allowable error between iterations
                LOUD - Enable/disable print statements for debugging
                
            Returns:
                Scalar fluxes, Angular fluxes, net current, scattering ratios, max optical thicknesses, iteration number """
        
        # Initialize arrays
        phis_old = self.mesh.scalarfluxes
        angfluxes_rightward = np.zeros_like(self.mesh.angfluxes)
        angfluxes_leftward = np.zeros_like(self.mesh.angfluxes)
        angfluxes_left_edge = np.zeros_like(self.mesh.angfluxes)
        angfluxes_right_edge = np.zeros_like(self.mesh.angfluxes) 
        self.scatt_ratios = np.zeros_like(self.mesh.scalarfluxes)
        self.maxOptThicknesses = np.zeros_like(self.mesh.scalarfluxes)
        
        mus = self.mesh.mus
        wgts = self.mesh.wgts
        
        # Prepare for convergence loop
        isConverged = False
        iterNum = 1
        
        # Start convergence loop...
        while not isConverged:    
            
            # Set left boundary flux
            if self.regions[0].LBC == "reflecting":
                angfluxes_left_edge[0] = angfluxes_leftward[0][::-1]
            else:
                angfluxes_left_edge[0] = self.angfluxLBC
                
            if LOUD: print(f"Initial left psi_inc = {angfluxes_left_edge[0]}")
                
            # Sweep from left --> right
            for i in range(len(angfluxes_rightward)):
                
                # Region properties
                for r in self.regions:
                    if r.ALB <= self.mesh.xCenters[i] <= r.ARB:
                        sigma_si = r.sigma_s
                        sigma_i  = r.sigma
                        source_i = r.source
                        
                        # Get scattering ratio during rightward sweep
                        try:
                            scatt_ratio = sigma_si/sigma_i
                        except ZeroDivisionError: # i.e., vacuum region
                            scatt_ratio = 0
                        
                # Organize cell properties
                dx = self.mesh.cell_widths[i]
                self.scatt_ratios[i] = scatt_ratio
                self.maxOptThicknesses[i] = sigma_i*dx/mus[0]
                
                # if LOUD: print(f"cell     = {i}\n"
                #                f"{sigma_i  = }\n"+
                #                f"{sigma_si = }\n"+
                #                f"{source_i = }\n"+
                #                f"{dx       = }\n"+
                #                f"{self.mesh.xCenters[i] = }\n")
                
                # Iterate over angle
                for m in range(self.mesh.angbins):
                    angfluxes_right_edge[i][m] = (source_i/2 + 
                                                  sigma_si*phis_old[i]/2 + 
                                                  mus[m]/dx*angfluxes_left_edge[i][m] -
                                                  sigma_i/2*angfluxes_left_edge[i][m]) / \
                                                  (mus[m]/dx + sigma_i/2)
                    # Set edges
                    if not i+1 == len(self.mesh.angfluxes):
                        angfluxes_left_edge[i+1][m] = angfluxes_right_edge[i][m]
                        
                if LOUD: print(f"angfluxes_left_edge[{i}] = {angfluxes_left_edge[i]}")
            
            # Convert edges back to centers and append
            angfluxes_rightward = self._convertEdge2Center(angfluxes_left_edge, angfluxes_right_edge)
            
            if LOUD: 
                print("left edges --- right edges")
                for l,r in zip(angfluxes_left_edge, angfluxes_right_edge):
                    print(" ",l," ", r)
            
            # Clear edge fluxes
            angfluxes_left_edge = np.zeros_like(self.mesh.angfluxes)
            angfluxes_right_edge = np.zeros_like(self.mesh.angfluxes)
            
            # Set right boundary flux
            if self.regions[-1].RBC == "reflecting":
                angfluxes_right_edge[-1] = angfluxes_rightward[-1][::-1]
            else:
                angfluxes_right_edge[-1] = self.angfluxRBC
                
            if LOUD: print(f"Initial right psi_inc = {angfluxes_right_edge[-1]}")
            
            # Sweep from right --> left
            for i in range(len(angfluxes_leftward)-1, -1 ,-1):
                
                # Region properties
                for r in self.regions:
                    if r.ALB < self.mesh.xCenters[i] < r.ARB:
                        sigma_si = r.sigma_s
                        sigma_i  = r.sigma
                        source_i = r.source
                        
                # Organize cell properties
                dx = self.mesh.cell_widths[i]
                
                # Iterate over angle
                for m in range(self.mesh.angbins):
                    angfluxes_left_edge[i][m] = (angfluxes_right_edge[i][m]*(-mus[m]/dx + sigma_i/2) -
                                                 source_i/2 - sigma_si*phis_old[i]/2) / (-mus[m]/dx - sigma_i/2)
                    # Set edges
                    if i != 0 != len(self.mesh.angfluxes):
                        angfluxes_right_edge[i-1][m] = angfluxes_left_edge[i][m]
                
                if LOUD: print(f"angfluxes_right_edge[{i}] = {angfluxes_right_edge[i]}")
                
            if LOUD: 
                print("left edges --- right edges")
                for l,r in zip(angfluxes_left_edge, angfluxes_right_edge):
                    print(" ",l," ", r)
            
            # Convert edges back to centers and append
            angfluxes_leftward = self._convertEdge2Center(angfluxes_left_edge, angfluxes_right_edge)
            
            # Add fluxes together
            self.mesh.angfluxes = angfluxes_leftward + angfluxes_rightward
            
            # Convert to scalar flux
            phis_new = self._convertAng2Scal(self.mesh.angfluxes)
            
            # if LOUD: print(f"{phis_new = }")
            
            # Check convergence. If converged, sigh in relief. If not, cope.
            if np.all(np.abs(phis_new - phis_old)) < eps:
                
                # Converged!
                isConverged = True
                self.mesh.scalarfluxes = np.round(phis_new, decimals=5) # Avoid graphical errors
                self.mesh.netCurrent = np.round(self._convertAng2NetCurrent(angfluxes_rightward, angfluxes_leftward), decimals=5)
                print(f"Finished in {iterNum} iterations.")
                break
                
            else:
                # Not converged...
                phis_old = phis_new.copy()
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
        
        # Create first solution that is only dx/2 away from left border
        self.analytical_solution[0] = self.angfluxLBC * [np.exp(-self.regions[0].sigma*self.mesh.cell_widths[0]/(2*abs(self.mesh.mus[m]))) \
                                                         for m in range(self.mesh.angbins)]
        
        for i in range(1, len(self.analytical_solution)):
            
            # Region properties
            for r in self.regions:
                # print(r.ALB, self.mesh.xEdgelocs[prev], r.ARB)
                if r.ALB <= self.mesh.xCenters[i] <= r.ARB:
                    sigma_si = r.sigma_s
                    sigma_i  = r.sigma
                    source_i = r.source
                    
            # Organize cell properties
            mus = self.mesh.mus
            wgts = self.mesh.wgts
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
            
        line1, = ax1.plot(self.mesh.xCenters, list(self.mesh.scalarfluxes), label="Scalar Flux")
        
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

############## Test Case ##############

if __name__ == "__main__":
    
    r_width = 1
    num_regions = 1 # Don't change
    num_cells = 20
    num_mus = 4
    
    sigma, sigma_s, source = 10, 0, 10

    medium1 = Region(r_width,sigma,sigma_s,source,"inf_medium","reflecting","reflecting")
    medium1.applyMesh(num_cells, num_regions/num_cells*r_width, num_mus)
    
    infModel1 = Model(medium1, angfluxLBC = 0, angfluxRBC = 0)
    infModel1.combineMesh()
    infModel1.doDiamondDiffV3(LOUD=True)
    
    infModel1.plotModel(show_AnalySol=False)
    infModel1.plotOptics()

###############################################

class TransportError(Exception):
    """ Class for catching transport errors """

