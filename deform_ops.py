import numpy as np
import igl
from scipy.sparse.linalg import inv, spsolve


class Deformer:
    """ Class to deform a mesh (M) 
        The class is initialized with M's vertices, faces, and handle indices and contains functions to 
            (1) Smooth M (or a transformation of M) by solving a Laplacian minimization problem
            (2) Compute detail encodings for M by defining an orthogonal reference frame for each of M's vertices
            (3) Apply detail encodings to a transformation of M using the orthogonal reference frames
    """

    def __init__(self, v_in: np.ndarray, f_in: np.ndarray, handle_indices: np.ndarray):
        self.v_in = v_in
        self.f_in = f_in
        self.handle_indices = handle_indices

        # Compute A for each vertex using the Laplacian: 
        M = igl.massmatrix(v_in, f_in, igl.MASSMATRIX_TYPE_VORONOI)
        M_inv = inv(M)
        S = igl.cotmatrix(v_in, f_in)
        self.A = 2 * S @ M_inv @ S  # A = 2L_w M^-1 L_w 
        
        # Compute b (the differential coordinates)
        L = M_inv @ S
        delta = L @ v_in 
        b_big = 2 * S @ delta # b = 2L_w * d -- these are the constraints on b
        
        # b is 0 if vertex is not a handle, and b is the delta if it is a handle
        self.b = np.zeros_like(self.v_in)
        self.b[self.handle_indices] = b_big[self.handle_indices]
        
        # Separate the constrained and unconstrained vertices
        constraint_mask = np.zeros(v_in.shape[0], dtype=bool) 
        constraint_mask[handle_indices] = True
        self.unconstrained_v = np.where(~constraint_mask)[0] # array of indices of unconstrained verts 
        self.constrained_v = np.where(constraint_mask)[0] # indices of constraint vertices

    def smooth_mesh(self, v_new: np.ndarray) -> np.ndarray:
        """ Function to smooth the mesh by solivng a bilaplacian minimization w.r.t old positions of handle.
            v_new is the desired position of the vertices after the transformation 
                (i.e. v if just the first smoothing step, or the transformed position of the handle if the deform smoothing step)
        """

        A_ff = self.A[self.unconstrained_v][:, self.unconstrained_v] # unconstrained rows x unconstrained columns
        A_fc = self.A[self.unconstrained_v][:, self.constrained_v] # unconstrained rows x constrained columns
        x_c = v_new[self.constrained_v] # constrained vertices have fixed positions (either the old pos of handle, v, or the transformed position)

        # solve A_ff x_f = b - A_fc x_c
        rhs = self.b[self.unconstrained_v] - A_fc @ x_c # b - A_fc vc
        x_f = spsolve(A_ff, rhs) # Aff*vf = rhs

        # now stack the unconstrained and constrained vertices together
        x = np.zeros_like(self.v_in)
        x[self.unconstrained_v] = x_f
        x[self.constrained_v] = x_c # constrained vertices are fixed

        return x

    def compute_detail_encoding(self, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Function to encode the details of B, where d = d_i^x(x_i) + d_i^y(y_i) + d_i^n(n_i)
        Returns coeffs of details (d_i^{x,y,x}), and the x_is 
        """
        B_norms = igl.per_vertex_normals(B, self.f_in)
        adjacent_vs = igl.adjacency_list(self.f_in)
        dets = self.v_in - B # details in a general reference frame
        coeffs = np.zeros_like(dets)
        tangent_vectors = np.zeros_like(B)

        for idx, vertex in enumerate(B):
            di = dets[idx] # detail at the vertex
            max_tangent_norm = 0
            best_tangent = np.zeros(3)
            # find the edge with the longest projection onto the tangent plane
            for vert_idx in adjacent_vs[idx]:
                norm = B_norms[idx]
                edge = B[vert_idx] - vertex
                proj_normal = np.dot(edge, norm) * norm
                proj_tangent = edge - proj_normal
                tangent_norm = np.linalg.norm(proj_tangent)
                if tangent_norm > max_tangent_norm:
                    max_tangent_norm = tangent_norm
                    best_tangent = proj_tangent
            # save that tangent vector
            xi = best_tangent / np.linalg.norm(best_tangent)
            tangent_vectors[idx] = xi
            norm = B_norms[idx]
            # compute the third vector in the basis (tangent * normal)
            yi = np.cross(norm, xi)
            #save the coefficients -- the coeff is the projection of the detail (di) onto each of the basis vectors
            coeffs[idx] = [np.dot(di, xi), np.dot(di, yi), np.dot(di, norm)]
        return coeffs, tangent_vectors

    def apply_detail_encoding(self, B_prime: np.ndarray, coeffs: np.ndarray, tangent_vectors: np.ndarray) -> np.ndarray:
        """
        Function to apply the detail encoding from base mesh B to the transformed mesh B_prime
        """
        B_prime_norms = igl.per_vertex_normals(B_prime, self.f_in)
        dets = np.zeros_like(B_prime) # will hold the details of B' in the reference frame
        for idx, _ in enumerate(B_prime):
            c = coeffs[idx]
            xi = tangent_vectors[idx]
            norm = B_prime_norms[idx]
            yi = np.cross(norm, xi)
            # apply coefficients to the tangent vectors to get the details in the new reference frame
            dets[idx] = c[0] * xi + c[1] * yi + c[2] * norm
        return dets
