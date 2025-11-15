import numpy as np
import igl
from scipy.sparse.linalg import inv, spsolve
from sksparse.cholmod import cholesky

class Deformer:
    """ Class to deform a mesh (M) - OPTIMIZED VERSION
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

        A_ff = self.A[self.unconstrained_v][:, self.unconstrained_v] # unconstrained rows x unconstrained columns
        self.A_fc = self.A[self.unconstrained_v][:, self.constrained_v] # unconstrained rows x constrained columns

        # Factor A_ff because it is constant 
        self.factor = cholesky(A_ff) # A_ff = L L^T, so we can use cholesky to factor it

    def smooth_mesh(self, v_new: np.ndarray) -> np.ndarray:
        """ Function to smooth the mesh by solivng a bilaplacian minimization w.r.t old positions of handle.
            v_new is the desired position of the vertices after the transformation 
                (i.e. v if just the first smoothing step, or the transformed position of the handle if the deform smoothing step)
        """
        x_c = v_new[self.constrained_v] # constrained vertices have fixed positions (either the old pos of handle, v, or the transformed position)

        # solve A_ff x_f = b - A_fc x_c
        rhs = self.b[self.unconstrained_v] - self.A_fc @ x_c 
        x_f = self.factor(rhs) # Aff*vf = rhs

        # now stack the unconstrained and constrained vertices together
        x = np.zeros_like(self.v_in)
        x[self.unconstrained_v] = x_f
        x[self.constrained_v] = x_c # constrained vertices are fixed

        return x

    def compute_detail_encoding(self, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Function to encode the details of B, where each detail, d = d_i^x(x_i) + d_i^y(y_i) + d_i^n(n_i)
        Returns coeffs of details (d_i^{x,y,x}), and the x_is 
        """
        B_norms = igl.per_vertex_normals(B, self.f_in) # num_vertices, 3
        num_vertices = B.shape[0]
        adjacent_vs = igl.adjacency_list(self.f_in) # num_adj
        dets = self.v_in - B # details in a general reference frame -- num_vertices, 3
        coeffs = np.zeros_like(dets)

        best_tangents = np.zeros((num_vertices, 3))

        # for each vertex, find the edge with the longest projection onto the tangent plane
        for idx in range(num_vertices):
            vertex = B[idx]
            vertex = vertex.reshape(1, 3) # broadcast vertex
            norm = B_norms[idx]
            norm = norm.reshape((1, 3))
            adj_idx = adjacent_vs[idx]
            adjacent_vertices = B[adj_idx] # has dimension num_adj, 3
            edges = adjacent_vertices - vertex  # has dimension num_adj, 3
            proj_normals = np.sum(edges * norm, axis=1, keepdims=True) * norm #np.sum does adds each element of the element-wise manipulation of each (edge*norm )
            # edges * norm gives us a num_adj * 3 vector of all of the element-wise products of each edge & norm -> np.sum() gives us in num_adj * 1 list of the dot products -> sum*norm gives us the projections -> num_adj * 3
            proj_tangents = edges - proj_normals
            tangent_norms = np.linalg.norm(proj_tangents, axis=1)
            max_tangent_idx = np.argmax(tangent_norms)
            best_tangents[idx] = proj_tangents[max_tangent_idx]

        # normalize all the tangent vectors
        xi = best_tangents / np.linalg.norm(best_tangents, axis=1, keepdims=True)
        # get the yi component by crossing all the normals with the normalized tangents
        yi = np.cross(B_norms, xi)
        coeffs = np.stack([np.sum(dets * xi, axis=1), # sum of element wise multiplication = dot product! each row is a vertex ->  axis=1 
                           np.sum(dets * yi, axis=1),
                           np.sum(dets * B_norms, axis=1)], axis=1) # we store each coeff as a row -> axis = 1 because we go over columns
        return coeffs, xi

    def apply_detail_encoding(self, B_prime: np.ndarray, coeffs: np.ndarray, tangent_vectors: np.ndarray) -> np.ndarray:
        """
        Function to apply the detail encoding from base mesh B to the transformed mesh B_prime
        """
        B_prime_norms = igl.per_vertex_normals(B_prime, self.f_in)
        yi = np.cross(B_prime_norms, tangent_vectors)

        # the coeff is always the column -- i.e. coeff for x is :, 0; y is :,1, n is :,2
        dets = (coeffs[:, 0, np.newaxis] * tangent_vectors + coeffs[:, 1, np.newaxis] * yi + coeffs[:, 2, np.newaxis] * B_prime_norms)

        return dets
    